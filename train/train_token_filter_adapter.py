"""
Train the token filter adapter on pre‑generated examples.

This script fine tunes a base language model checkpoint to learn how
to classify and reason about token importance.  It reads JSONL
examples produced by the seed and expansion scripts, constructs a
prompt/target pair, encodes them with a SentencePiece tokenizer, and
performs supervised training.  The adapter learns to output
explanations for why certain tokens are important, which can later be
used to guide token selection during generation.

Example invocation::

    python -m train.train_token_filter_adapter \
        --ckpt_in checkpoints/usage_adapter.pt \
        --ckpt_out checkpoints/token_filter_adapter.pt \
        --spm_model tokenizer/spm.model \
        --filter_glob 'token_filter_data/filter_data_*.jsonl' \
        --steps 6000 --batch_size 4 --device mps
"""

import argparse
import glob
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM
from train.datasets_token_filter import TokenFilterDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_in", type=str, required=True, help="Path to the base model checkpoint")
    ap.add_argument("--ckpt_out", type=str, required=True, help="Where to save the tuned checkpoint")
    ap.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="Path to the SentencePiece model")
    ap.add_argument("--filter_glob", type=str, default="token_filter_data/filter_data_*.jsonl", help="Glob pattern for token filter datasets")
    ap.add_argument("--steps", type=int, default=6000, help="Number of training steps")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    ap.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu, cuda, mps)")
    ap.add_argument("--limit_per_file", type=int, default=None, help="Optional limit of examples per file (for quick runs)")
    args = ap.parse_args()

    # Load tokenizer
    tok = SPMTokenizer(args.spm_model)

    # Load base checkpoint
    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    cfg = ckpt["config"]
    # Ensure vocab size matches tokenizer
    cfg["vocab_size"] = tok.VOCAB_SIZE
    model = TinyTransformerLM(**cfg)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()

    # Build dataset
    paths = sorted(glob.glob(args.filter_glob))
    if not paths:
        raise RuntimeError(f"No dataset files matched pattern {args.filter_glob}")
    dataset = TokenFilterDataset(paths, tokenizer=tok, max_seq_len=cfg["max_seq_len"], limit_per_file=args.limit_per_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.05)
    step_count = 0
    for epoch in range(1000000):  # effectively infinite, break by steps
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type != "cpu" else torch.float32, enabled=(device.type != "cpu")):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=tok.PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_count += 1
            if step_count % 50 == 0:
                print(f"[token-filter] step {step_count} loss {loss.item():.3f}")
            if step_count >= args.steps:
                break
        if step_count >= args.steps:
            break

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "spm_model": args.spm_model}, args.ckpt_out)
    print(f"✅ Saved token filter adapter checkpoint to {args.ckpt_out}")


if __name__ == "__main__":
    main()