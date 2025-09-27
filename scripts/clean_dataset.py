"""
Clean and deduplicate a text dataset.

This script reads a text file containing one example per line,
applies simple cleaning heuristics and removes duplicate entries. The
cleaned data is written to a new file. Cleaning steps include:

  * Stripping leading/trailing whitespace and normalising internal
    whitespace to a single space.
  * Filtering out lines that are too short (e.g. fewer than two words).
  * Filtering out lines with a high proportion of non‑Latin characters
    (to ensure English‑only content as required for the project).
  * Removing exact duplicate lines.

The script is idempotent: running it multiple times produces the same
output. It is intended to help improve data quality prior to
training, as recommended in several sources. Removing noisy or
duplicate data can improve model accuracy and reduce overfitting【670547615243133†L83-L103】【670547615243133†L111-L139】.

Usage::
    python -m scripts.clean_dataset --input data/train.txt --output data/train_clean.txt

Note: This script is intentionally conservative. It does not attempt
to lemmatise, stem or remove stop words; those operations may be
performed at training time if desired. The goal is simply to reduce
obvious noise and duplication.
"""

from __future__ import annotations

import argparse
import re


def is_mostly_non_latin(text: str, threshold: float = 0.3) -> bool:
    """Return True if the proportion of non‑Latin characters exceeds threshold.

    Latin characters are defined as the ranges A–Z, a–z, digits and basic
    punctuation. If more than `threshold` fraction of characters fall
    outside this set, the line is considered non‑English and filtered.
    """
    if not text:
        return True
    total = len(text)
    non_latin = sum(1 for ch in text if not re.match(r"[A-Za-z0-9 .,;:!?'\-]", ch))
    return (non_latin / total) > threshold


def clean_dataset(input_path: str, output_path: str, min_words: int = 2) -> None:
    """Read a dataset file, clean and deduplicate, and write the result.

    Args:
        input_path: Path to the source text file (one example per line).
        output_path: Path to write the cleaned file.
        min_words: Minimum number of whitespace‑separated words required for
            a line to be kept. Defaults to 2.
    """
    seen = set()
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            # Normalize whitespace
            stripped = " ".join(line.strip().split())
            if not stripped:
                continue
            # Filter by word count
            if len(stripped.split()) < min_words:
                continue
            # Filter by character content
            if is_mostly_non_latin(stripped):
                continue
            # Deduplicate
            if stripped in seen:
                continue
            seen.add(stripped)
            fout.write(stripped + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and deduplicate a text dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to output the cleaned file")
    parser.add_argument("--min_words", type=int, default=2, help="Minimum number of words per line to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_dataset(args.input, args.output, min_words=args.min_words)
    print(f"Cleaned dataset written to {args.output}")


if __name__ == "__main__":
    main()