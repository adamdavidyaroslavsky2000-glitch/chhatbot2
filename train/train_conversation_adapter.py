"""
Fine‑tune a conversation adapter on top of an existing checkpoint.

This script reads a JSONL file of conversation examples where each pair of lines
represents a user prompt followed by an assistant response. It encodes the
examples using a SentencePiece tokenizer and trains a simple decoder‑only
transformer to produce the assistant responses conditioned on the user input.

The loss is masked so that the model does not learn to reproduce the user
portion of the conversation (prompts), focusing only on generating the
assistant portion. This helps prevent the model from copying the input and
encourages it to learn appropriate responses.

Usage:
  python -m train.train_conversation_adapter \
    --ckpt_in checkpoints/usage_adapter.pt \
    --ckpt_out checkpoints/conversation_adapter.pt \
    --spm_model tokenizer/spm.model \
    --data_path data/conversation_sft.jsonl \
    --batch_size 4 --steps 2000 --device mps
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM


class ConversationDataset(Dataset):
    """Dataset for conversation fine‑tuning.

    Each example in the input file is a JSON object with `role` and `content`.
    The file alternates between a user line and an assistant line. This class
    groups consecutive pairs into a single training example. The resulting
    sequence has the form:

        User: <user content>\nAssistant: <assistant content>

    A BOS token is prepended and an EOS token appended. The loss is masked so
    that only tokens in the assistant portion contribute to the loss.
    """

    def __init__(self, data_path: str, tokenizer: SPMTokenizer, max_seq_len: int = 1024):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        # Load the JSONL file
        with open(data_path, "r", encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        # Check that lines alternate role user/assistant
        self.examples = []
        i = 0
        while i < len(lines) - 1:
            user_obj = lines[i]
            assistant_obj = lines[i + 1]
            if user_obj.get("role") == "user" and assistant_obj.get("role") == "assistant":
                user_text = user_obj.get("content", "").strip()
                assistant_text = assistant_obj.get("content", "").strip()
                full = f"User: {user_text}\nAssistant: {assistant_text}"
                ids = self.tok.encode(full, add_bos=True, add_eos=True)
                # Clip to max length
                if len(ids) > self.max_seq_len:
                    ids = ids[: self.max_seq_len]
                self.examples.append((full, ids))
                i += 2
            else:
                # If misaligned, skip one line
                i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        full, ids = self.examples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # Mask prompt tokens: compute where "Assistant:" starts
        assist_idx = full.find("Assistant:")
        prefix = full[:assist_idx]
        # Encode prefix without BOS/EOS to find number of tokens to ignore
        prefix_ids = self.tok.encode(prefix, add_bos=True, add_eos=False)
        ignore_token = self.tok.PAD
        labels = y.clone()
        # Mask tokens before assistant; subtract 1 because we are shifted by one
        labels[: max(1, len(prefix_ids)) - 1] = ignore_token
        # Pad if necessary to max_seq_len - 1
        pad_len = (self.max_seq_len - 1) - x.numel()
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), ignore_token, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), ignore_token, dtype=torch.long)])
        return x, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_in", type=str, required=True, help="Path to base checkpoint to load")
    parser.add_argument("--ckpt_out", type=str, default="checkpoints/conversation_adapter.pt", help="Where to save the fine‑tuned checkpoint")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="SentencePiece model path")
    parser.add_argument("--data_path", type=str, default="data/conversation_sft.jsonl", help="Path to conversation JSONL file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--steps", type=int, default=2000, help="Number of optimization steps")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, mps, cuda)")
    args = parser.parse_args()

    # Load tokenizer and model
    tok = SPMTokenizer(args.spm_model)
    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    cfg = ckpt["config"]
    cfg["vocab_size"] = tok.VOCAB_SIZE
    model = TinyTransformerLM(**cfg)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model.to(device)
    model.train()

    # Dataset and DataLoader
    ds = ConversationDataset(args.data_path, tokenizer=tok, max_seq_len=cfg["max_seq_len"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)
    vocab_size = tok.VOCAB_SIZE
    for step, (x, labels) in enumerate(dl):
        if step >= args.steps:
            break
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type != "cpu" else torch.float32, enabled=(device.type != "cpu")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=tok.PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[conversation-adapter] step {step}/{args.steps} loss {loss.item():.3f}")
    # Save checkpoint
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "spm_model": args.spm_model}, args.ckpt_out)
    print(f"✅ Saved conversation adapter to {args.ckpt_out}")


if __name__ == "__main__":
    main()