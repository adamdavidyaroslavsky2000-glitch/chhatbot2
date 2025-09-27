#!/usr/bin/env python3
"""
expand_social_pressure_data.py
--------------------------------

This script expands small seed JSONL datasets containing social pressure examples
into much larger files suitable for training.  Each input JSONL file should
contain a list of JSON objects with at least the following fields:
  - input:  the user input prompt
  - good_sentence: a well‑aligned response
  - bad_sentence: an inappropriate or unhelpful response
  - reason: rationale explaining why the good sentence is appropriate
  - synonyms: a mapping of terms to lists of synonyms

The script randomly samples from these seed examples until the desired
target length is reached.  For diversity, it optionally performs simple
paraphrasing of the `input` field by replacing a random word with one
of its synonyms.  This does not guarantee perfect grammatical output
but introduces variation into the dataset.

Usage:

    python expand_social_pressure_data.py --glob "social_pressure_data/social_pressure_data_*.jsonl" --target 120000

The expanded files will overwrite the originals.  A backup of the original
file will be saved with the suffix `.seed.bak`.
"""
import argparse
import json
import os
import random
import glob
from typing import List, Dict


def paraphrase_input(example: Dict[str, any]) -> str:
    """Create a simple paraphrased version of the input by replacing a random
    key in the synonyms map with one of its synonyms.  If no synonyms
    are present, return the original input.
    """
    inp = example.get("input", "")
    synonyms = example.get("synonyms", {}) or {}
    if not synonyms:
        return inp
    # choose a word in the input that appears in the synonyms mapping
    words = inp.split()
    keys_in_input = [w for w in synonyms.keys() if w.lower().strip('.,!?') in [t.lower().strip('.,!?') for t in words]]
    if not keys_in_input:
        return inp
    key = random.choice(keys_in_input)
    syns = synonyms.get(key, [])
    if not syns:
        return inp
    replacement = random.choice(syns)
    # replace only the first occurrence (case‑insensitive)
    new_words: List[str] = []
    replaced = False
    for w in words:
        stripped = w.lower().strip('.,!?')
        if not replaced and stripped == key.lower().strip('.,!?'):
            # preserve punctuation
            prefix = w[:-len(stripped)] if len(stripped) != len(w) else ""
            suffix = w[len(stripped):] if len(stripped) != len(w) else ""
            new_words.append(prefix + replacement + suffix)
            replaced = True
        else:
            new_words.append(w)
    return " ".join(new_words)


def expand_file(path: str, target: int) -> None:
    """Expand a single JSONL file to have at least `target` entries.  The original
    file is backed up with a `.seed.bak` suffix before being overwritten.
    """
    with open(path, "r", encoding="utf-8") as f:
        seed_records = [json.loads(line) for line in f if line.strip()]
    if not seed_records:
        print(f"No seed data found in {path}; skipping")
        return
    # backup the original seed file
    backup_path = path + ".seed.bak"
    if not os.path.exists(backup_path):
        os.replace(path, backup_path)
    else:
        # remove current file contents if backup already exists
        os.remove(path)
    # generate records
    with open(path, "w", encoding="utf-8") as out_file:
        count = 0
        while count < target:
            ex = random.choice(seed_records)
            # create a shallow copy to avoid mutating seed
            new_ex = dict(ex)
            # occasionally paraphrase the input to add variation
            if random.random() < 0.5:
                new_ex["input"] = paraphrase_input(ex)
            out_file.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
            count += 1


def main():
    parser = argparse.ArgumentParser(description="Expand social pressure dataset seeds to a larger size")
    parser.add_argument("--glob", type=str, required=True, help="Glob pattern for JSONL files to expand (e.g. 'social_pressure_data/social_pressure_data_*.jsonl')")
    parser.add_argument("--target", type=int, default=120000, help="Desired number of records per file")
    args = parser.parse_args()
    files = glob.glob(args.glob)
    if not files:
        print(f"No files matched pattern {args.glob}")
        return
    for file_path in files:
        expand_file(file_path, args.target)
        print(f"Expanded {file_path} to {args.target} lines")


if __name__ == "__main__":
    main()