"""
Dataset utilities for the token filter adapter.

The token filter adapter aims to teach the model to distinguish
important tokens from less important ones given a user input.  The
dataset format is JSONL, where each line is a JSON object with the
following keys:

    - input: string describing the user's query or sentence
    - important_tokens: list of strings representing key tokens
    - less_tokens: list of strings representing less important tokens
    - explanation: short natural language explanation of the
                   classification

This module exposes a ``TokenFilterDataset`` class compatible with
PyTorch's DataLoader.  Each sample is converted into an input‐target
pair suitable for supervised fine tuning: the input contains the
source text and token lists, and the target contains the
explanation.  This encourages the model to output structured
reasoning about token importance.

Example usage::

    from train.datasets_token_filter import TokenFilterDataset
    ds = TokenFilterDataset(["token_filter_data/filter_data_01.jsonl"], tokenizer, max_seq_len=1024)
    for x, y in ds:
        ...  # x and y are tensors of token ids
"""

import json
import random
from pathlib import Path
from typing import Iterator, List

import torch
from torch.utils.data import Dataset


class TokenFilterDataset(Dataset):
    """A dataset for training the token filter adapter."""

    def __init__(self, paths: List[str], tokenizer, max_seq_len: int = 1024, limit_per_file: int = None):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.samples: List[List[int]] = []
        for p in paths:
            for ex in self._load_file(Path(p), limit_per_file):
                ids = ex
                # Only use sequences with at least one token
                if ids:
                    self.samples.append(ids)
        random.shuffle(self.samples)

    def _load_file(self, path: Path, limit: int = None) -> Iterator[List[int]]:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                inp = obj["input"]
                important = obj.get("important_tokens", [])
                less = obj.get("less_tokens", [])
                explanation = obj.get("explanation", "")
                # Construct a prompt for the model
                prompt_parts = [
                    "Source text:", inp,
                    "Important tokens:", ", ".join(important) if important else "None",
                    "Less important tokens:", ", ".join(less) if less else "None",
                    "Explain why these tokens are classified this way."
                ]
                prompt = "\n".join(prompt_parts)
                target = explanation
                # Encode with BOS at start and EOS at end
                input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
                target_ids = self.tokenizer.encode(target, add_bos=False, add_eos=True)
                # Concatenate prompt and target with a separator (newline)
                ids = input_ids + [self.tokenizer.PAD] + target_ids
                # Trim/pad to max sequence length
                if len(ids) > self.max_seq_len:
                    ids = ids[: self.max_seq_len]
                else:
                    ids = ids + [self.tokenizer.PAD] * (self.max_seq_len - len(ids))
                yield ids
                count += 1
                if limit is not None and count >= limit:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        # For language modelling: input is all but last token, target is all but first token
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y