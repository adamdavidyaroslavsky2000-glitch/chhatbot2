#!/usr/bin/env python3
"""
Train a social pressure adapter on a set of examples.

This script fine‑tunes a TinyTransformerLM model to better align its outputs
with high‑quality responses using the social pressure dataset.  Each example
consists of a user input and a corresponding good response.  The model
learns to map prompts to appropriate outputs, forming an adapter layer
specialized in producing higher quality and more user‑friendly answers.

Usage example:

    python -m train.train_social_pressure_adapter \
      --ckpt_in checkpoints/rules_adapter.pt \
      --ckpt_out checkpoints/social_pressure_adapter.pt \
      --spm_model tokenizer/spm.model \
      --data_glob "social_pressure_data/social_pressure_data_*.jsonl" \
      --steps 5000 --batch_size 4 --device mps

The adapter is trained using a simple cross‑entropy loss on next‑token
prediction.  It reuses the same model architecture as the underlying
language model and shares the embedding weights.
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM
from train.datasets_social_pressure import SocialPressureDataset


def main():
    parser = argparse.ArgumentParser(description="Train a social pressure adapter")
    parser.add_argument("--ckpt_in", type=str, required=True, help="Path to base model checkpoint (.pt)")
    parser.add_argument("--ckpt_out", type=str, required=True, help="Path to save the adapted checkpoint")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="Path to SentencePiece model")
    parser.add_argument("--data_glob", type=str, default="social_pressure_data/social_pressure_data_*.jsonl", help="Glob pattern for training data")
    parser.add_argument("--steps", type=int, default=5000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu, mps, cuda)")
    parser.add_argument("--limit_per_file", type=int, default=None, help="Limit number of records per file for quick experiments")
    args = parser.parse_args()

    # load tokenizer
    tok = SPMTokenizer(args.spm_model)
    # load base checkpoint
    base = torch.load(args.ckpt_in, map_location="cpu")
    cfg = dict(base["config"])
    cfg["vocab_size"] = tok.VOCAB_SIZE
    # instantiate model
    model = TinyTransformerLM(**cfg)
    model.load_state_dict(base["model_state"])
    device = torch.device(args.device if (args.device == "cpu" or args.device == "cuda" or args.device == "mps") else "cpu")
    model.to(device)
    model.train()
    # dataset
    dataset = SocialPressureDataset(args.data_glob, tokenizer=tok, max_seq_len=cfg.get("max_seq_len", 1024), limit_per_file=args.limit_per_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.05)
    step = 0
    # training loop
    for x, y in dataloader:
        if step >= args.steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type if device.type != "cpu" else "cpu", dtype=torch.float16 if device.type != "cpu" else torch.float32, enabled=(device.type != "cpu")):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=tok.PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[social_pressure] step {step}/{args.steps} loss {loss.item():.3f}")
        step += 1
    # save adapted checkpoint
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "spm_model": args.spm_model}, args.ckpt_out)
    print(f"✅ Saved social pressure adapter to {args.ckpt_out}")


if __name__ == "__main__":
    main()