"""
Dataset loader for social pressure training.

This dataset reads a set of JSONL files where each record contains
an `input` field (the user's question) and a `good_sentence` field
representing an appropriate response.  The dataset yields tokenized
pairs (x, y) where x is the input sequence and y is the target
sequence.  A SentencePiece tokenizer is used for encoding.

We ignore the `bad_sentence`, `reason`, and `synonyms` fields during
training; those fields are utilized during inference to adjust
generation heuristics.
"""
import json
from glob import glob
from typing import List, Iterator, Dict
import torch
from torch.utils.data import Dataset


class SocialPressureDataset(Dataset):
    def __init__(self, pattern: str, tokenizer, max_seq_len: int = 1024, limit_per_file: int = None):
        """Load and tokenize social pressure examples.

        Args:
            pattern: Glob pattern for input JSONL files.
            tokenizer: An instance of SPMTokenizer for encoding text.
            max_seq_len: Maximum sequence length.  Longer sequences are truncated.
            limit_per_file: Optional cap on the number of samples per file.
        """
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        paths = sorted(glob(pattern))
        for path in paths:
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if limit_per_file and count >= limit_per_file:
                        break
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    prompt = obj.get("input", "").strip()
                    target = obj.get("good_sentence", "").strip()
                    # build a single sequence: BOS + prompt + newline + target + EOS
                    txt = prompt + "\n\n" + target
                    ids = self.tok.encode(txt, add_bos=True, add_eos=True)
                    if len(ids) > max_seq_len:
                        ids = ids[:max_seq_len]
                    # x: all tokens except last; y: all tokens except first
                    x = torch.tensor(ids[:-1], dtype=torch.long)
                    y = torch.tensor(ids[1:], dtype=torch.long)
                    # pad to max_seq_len-1
                    pad_len = (max_seq_len - 1) - len(x)
                    if pad_len > 0:
                        x = torch.cat([x, torch.full((pad_len,), self.tok.PAD, dtype=torch.long)])
                        y = torch.cat([y, torch.full((pad_len,), self.tok.PAD, dtype=torch.long)])
                    self.samples.append({"x": x, "y": y})
                    count += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        return sample["x"], sample["y"]