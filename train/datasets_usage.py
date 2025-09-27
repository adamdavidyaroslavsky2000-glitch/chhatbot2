"""
Dataset utilities for training the usage adapter.

This module defines a UsageMetaStream class to load meta usage data
from JSONL files and a UsagePairDataset class that constructs input
and target token sequences for fine‑tuning a language model.

The dataset loader masks the prompt portion of the text so that
only the assistant response contributes to the training loss. This
prevents the model from learning to emit the system/user prompt
verbatim, improving generalisation and avoiding meta leakage.

All code in this file is written in English to maintain consistency
across the project.
"""

from __future__ import annotations

import json
import random
from glob import glob
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class UsageMetaStream:
    """A streaming reader for usage meta data stored as JSONL.

    Each record in the JSONL file is expected to contain at least a
    ``prompt`` and ``target`` field. Additional metadata may be present
    but is ignored by this loader. The stream can optionally limit
    the number of records read per file and shuffle the combined
    records from multiple files.

    Args:
        paths: A list of file paths to JSONL files.
        max_records_per_file: Optional maximum number of records to read
            from each file. If ``None`` (default) all records are read.
        shuffle: Whether to shuffle the order of records across all
            loaded files. Defaults to ``True``.
    """

    def __init__(
        self,
        paths: List[str],
        max_records_per_file: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        self.records: List[Dict[str, str]] = []
        for path in paths:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_records_per_file is not None and count >= max_records_per_file:
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
                    # Only keep records with the required fields
                    if "prompt" in obj and "target" in obj:
                        self.records.append(obj)
                        count += 1
        if shuffle:
            random.shuffle(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.records)


class UsagePairDataset(Dataset):
    """Dataset of prompt/target pairs for usage adapter fine‑tuning.

    For each record loaded from the UsageMetaStream, the prompt and
    target text are concatenated with an "Assistant:" marker. The
    resulting string is tokenised and returned as two sequences:
    ``x`` (the input tokens) and ``y`` (the target tokens). All tokens
    corresponding to the prompt portion of the text are masked out in
    ``y`` so that the loss is computed only on the assistant's answer.

    Args:
        meta_glob: A glob pattern matching JSONL files containing meta
            usage data.
        tokenizer: An object providing ``encode`` and ``decode`` methods
            compatible with SentencePiece. It should also define
            ``PAD`` for the padding token id.
        max_seq_len: Maximum sequence length (in tokens) for each
            example including BOS/EOS markers. Sequences longer than
            this value will be truncated. Defaults to 1024.
        limit_per_file: Optional limit on the number of records to read
            from each JSONL file. Useful for debugging.
    """

    def __init__(
        self,
        meta_glob: str,
        tokenizer,
        max_seq_len: int = 1024,
        limit_per_file: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        # Discover all files matching the glob pattern
        file_paths = sorted(glob(meta_glob))
        # Load the meta data
        stream = UsageMetaStream(file_paths, max_records_per_file=limit_per_file, shuffle=True)
        self.samples: List[Tuple[str, List[int]]] = []
        for record in stream:
            prompt: str = record["prompt"]
            target: str = record["target"]
            # Construct full text with assistant marker
            full_text = f"{prompt}\n\nAssistant: {target}"
            # Tokenise and truncate
            ids = self.tok.encode(full_text, add_bos=True, add_eos=True)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            self.samples.append((full_text, ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        full_text, ids = self.samples[index]
        # Construct input and target sequences
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # Determine the position of the assistant marker to mask the prompt
        assist_pos = full_text.find("Assistant:")
        # Tokenise the text up to the assistant marker
        assist_prefix = full_text[:assist_pos]
        assist_ids = self.tok.encode(assist_prefix, add_bos=True, add_eos=False)
        # Mask all prompt tokens in the target sequence
        labels = y.clone()
        ignore_index = self.tok.PAD
        # All tokens before the assistant marker (minus one) are masked
        mask_len = max(1, len(assist_ids)) - 1
        labels[:mask_len] = ignore_index
        # Pad to fixed length
        pad_len = (self.max_seq_len - 1) - x.numel()
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), ignore_index, dtype=torch.long)
            x = torch.cat([x, pad_tensor])
            labels = torch.cat([labels, pad_tensor])
        else:
            # Trim to desired length
            x = x[: self.max_seq_len - 1]
            labels = labels[: self.max_seq_len - 1]
        return x, labels