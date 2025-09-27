"""Train a usage adapter that teaches the model how to apply a source text to a task.

This script fine‑tunes a base language model on a set of meta‑usage examples.
Each example contains a `prompt` describing the source and requested task and
a `target` containing guidance and an example answer.  The goal of the adapter
is to learn how to interpret instructions and produce structured outputs
without memorising the prompt template.  To achieve this we mask the loss
over the prompt tokens: only the assistant portion contributes to the loss.

Usage:
    python -m train.train_usage_adapter \
        --ckpt_in checkpoints/base.pt \
        --ckpt_out checkpoints/usage_adapter.pt \
        --spm_model tokenizer/spm.model \
        --meta_glob "data_use/meta_use_*.jsonl" \
        --steps 2000 \
        --batch_size 4 \
        --device mps

The script loads the base checkpoint, initialises the tokenizer and dataset,
performs a few thousand optimisation steps and saves the resulting adapter.

Note: this script assumes the SentencePiece tokenizer supports BOS/EOS/PAD
tokens and that the base model was trained with the same vocab size.
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM
from .datasets_usage import UsagePairDataset  # use external dataset loader




def main() -> None:
    parser = argparse.ArgumentParser(description="Fine‑tune a usage adapter on meta examples")
    parser.add_argument("--ckpt_in", type=str, required=True, help="Path to the base checkpoint to start from")
    parser.add_argument("--ckpt_out", type=str, required=True, help="Path to save the trained adapter checkpoint")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="Path to the SentencePiece model")
    parser.add_argument("--meta_glob", type=str, default="data_use/meta_use_*.jsonl", help="Glob pattern for meta usage data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--steps", type=int, default=3000, help="Number of optimisation steps to run")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (e.g. cpu, mps, cuda)")
    parser.add_argument("--limit_per_file", type=int, default=None, help="Optional limit of examples per file")
    args = parser.parse_args()

    # Load tokenizer
    tok = SPMTokenizer(args.spm_model)

    # Load base model checkpoint
    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    config = ckpt["config"]
    config["vocab_size"] = tok.VOCAB_SIZE
    model = TinyTransformerLM(**config)
    model.load_state_dict(ckpt["model_state"])

    # Set device
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()

    # Build dataset and dataloader using the external UsagePairDataset
    dataset = UsagePairDataset(
        args.meta_glob,
        tok,
        max_seq_len=config["max_seq_len"],
        limit_per_file=args.limit_per_file,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)
    vocab_size = tok.VOCAB_SIZE

    step = 0
    while step < args.steps:
        for x, labels in dataloader:
            if step >= args.steps:
                break
            x = x.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type != "cpu" else torch.float32, enabled=(device.type != "cpu")):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=tok.PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"[usage adapter] step {step}/{args.steps} loss {loss.item():.4f}")

    # Save adapter
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "spm_model": args.spm_model}, args.ckpt_out)
    print(f"Saved usage adapter to {args.ckpt_out}")


if __name__ == "__main__":
    main()