#!/usr/bin/env python
"""Train a language model from scratch on a text corpus.

This script performs self‑supervised training of a decoder‑only
transformer on a plain text file.  It uses SentencePiece for
tokenisation, packs the data into fixed‑length sequences and trains
using cross‑entropy loss with optional padding masking.  The script
supports mixed precision (autocast) on Apple silicon via MPS and can
save checkpoints for later fine‑tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM


@dataclass
class PackedDataset(Dataset):
    """Pack a stream of token IDs into fixed‑length training examples."""

    token_stream: list[int]
    ctx: int
    pad_id: int

    def __len__(self) -> int:
        return (len(self.token_stream) - 1) // self.ctx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # compute start of segment
        start = idx * self.ctx
        x_ids = self.token_stream[start : start + self.ctx]
        y_ids = self.token_stream[start + 1 : start + self.ctx + 1]
        # pad if necessary
        if len(x_ids) < self.ctx:
            pad_len = self.ctx - len(x_ids)
            x_ids = x_ids + [self.pad_id] * pad_len
            y_ids = y_ids + [self.pad_id] * pad_len
        return (
            torch.tensor(x_ids, dtype=torch.long),
            torch.tensor(y_ids, dtype=torch.long),
        )


def load_token_stream(path: str, tokenizer: SPMTokenizer, ctx: int) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    # Always pad so that (len(ids)-1) is divisible by ctx
    remainder = (len(ids) - 1) % ctx
    if remainder:
        pad_len = ctx - remainder
        ids += [tokenizer.PAD] * pad_len
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer LM from scratch")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps (batches)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, or mps")
    parser.add_argument("--ckpt_out", type=str, required=True, help="Output checkpoint path")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="SentencePiece model path")
    args = parser.parse_args()

    # Load config and tokenizer
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tok = SPMTokenizer(args.spm_model)
    cfg["vocab_size"] = tok.VOCAB_SIZE

    # Build model
    model = TinyTransformerLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        mlp_mult=cfg["mlp_mult"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg.get("dropout", 0.0),
    )
    device = torch.device(args.device)
    model.to(device)

    # Prepare dataset and dataloader
    ids = load_token_stream(args.data, tok, cfg["max_seq_len"])
    dataset = PackedDataset(ids, cfg["max_seq_len"], tok.PAD)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Optimiser
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    steps = 0
    for epoch in range(1000000):  # loop until we reach the required number of steps
        for x, y in loader:
            if steps >= args.steps:
                break
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type != "cpu" else torch.float32, enabled=(device.type != "cpu")):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, tok.VOCAB_SIZE), y.view(-1), ignore_index=tok.PAD
                )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            steps += 1
            if steps % 50 == 0:
                print(f"step {steps}/{args.steps} | loss={loss.item():.4f}")
        if steps >= args.steps:
            break

    # Save checkpoint
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "spm_model": args.spm_model}, args.ckpt_out)
    print(f"Saved checkpoint to {args.ckpt_out}")


if __name__ == "__main__":
    main()