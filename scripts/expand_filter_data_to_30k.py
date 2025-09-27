"""
Expand token filter seed datasets to a larger size.

This script reads seed JSONL files under ``rule_data`` or another directory
and writes out expanded versions with a target number of entries.  It
works by replicating the existing examples and lightly mutating the
explanations to introduce lexical diversity.  The original lists of
important and less‑important tokens are preserved to avoid changing
the semantics.

The expanded files overwrite the originals.  Because the expansion
occurs locally on your machine, you can produce very large datasets
without needing to transmit them over the network.

Example usage::

    python scripts/expand_filter_data_to_30k.py --glob 'token_filter_data/filter_data_*.jsonl' --target 30000

This will read all matching seed files and expand each of them to at
least 30000 lines.
"""

import argparse
import json
import glob
import random
import os
from pathlib import Path


# Synonym lists for explanation variation
MEANING_SYNS = ["core meaning", "key meaning", "central idea", "main idea"]
FILLER_SYNS = ["filler words", "stop words", "nonessential words", "connective words"]
EXAMPLE_SYNS = ["For example", "As an illustration", "For instance"]


def mutate_explanation(base_explanation: str) -> str:
    """Return a mutated explanation string with synonyms inserted.

    The function replaces occurrences of particular phrases in the
    explanation to introduce variation.  If the base explanation
    doesn't match the simple pattern used in seed generation, it
    returns it unchanged.
    """
    # Look for the pattern 'Tokens such as ... carry the core meaning of the sentence, while words like ... are filler or stop words.'
    if "carry the" in base_explanation and "are filler" in base_explanation:
        parts = base_explanation.split("carry the")
        if len(parts) >= 2:
            prefix, rest = parts[0], parts[1]
            # Choose random synonyms
            meaning = random.choice(MEANING_SYNS)
            filler = random.choice(FILLER_SYNS)
            # Replace phrases
            mutated = prefix + f"carry the {meaning} of the sentence, while words like" + rest.split("are filler")[0] + f"are {filler}."
            return mutated
    # Default: add a random example phrase at the beginning
    return random.choice(EXAMPLE_SYNS) + ": " + base_explanation


def expand_file(path: Path, target_count: int) -> None:
    """Read a seed JSONL file and expand it to ``target_count`` entries.

    The expanded file overwrites the original.  Duplicate entries are
    created by reusing the important/less token lists and the input
    sentence, while mutating the explanation to produce some
    variability.
    """
    with open(path, "r", encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f if line.strip()]
    if not seeds:
        print(f"No examples in {path}")
        return
    expanded = []
    # Continue until we reach the target size
    i = 0
    while len(expanded) < target_count:
        ex = seeds[i % len(seeds)]
        new_ex = ex.copy()
        # mutate explanation
        new_ex["explanation"] = mutate_explanation(ex["explanation"])
        expanded.append(new_ex)
        i += 1
    # Write back to file
    with open(path, "w", encoding="utf-8") as f:
        for ex in expanded:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Expanded {path} to {len(expanded)} entries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        required=True,
        help="Glob pattern to match JSONL seed files (e.g. 'token_filter_data/filter_data_*.jsonl')",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=30000,
        help="Target number of entries per file.",
    )
    args = parser.parse_args()
    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("No files matched the given glob.")
        return
    for p in paths:
        expand_file(Path(p), args.target)


if __name__ == "__main__":
    main()