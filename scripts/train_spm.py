#!/usr/bin/env python
"""Train a SentencePiece model from a text corpus.

This script wraps the SentencePieceTrainer API to create a subword
tokeniser.  It accepts common parameters such as the input file,
vocabulary size and model type.  See the README for usage examples.
"""

import argparse
import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokeniser")
    parser.add_argument("--input", type=str, required=True, help="Path to the training text file")
    parser.add_argument("--model_prefix", type=str, default="tokenizer/spm", help="Prefix for the output model files")
    parser.add_argument("--vocab_size", type=int, default=16000, help="Size of the vocabulary")
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "char", "word"], help="Type of model to train")
    parser.add_argument("--character_coverage", type=float, default=1.0, help="Character coverage (for languages with large character sets)")
    parser.add_argument("--input_sentence_size", type=int, default=0, help="Limit on the number of sentences to use for training (0 = no limit)")
    args = parser.parse_args()

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )
    print(f"Trained SentencePiece model: {args.model_prefix}.model")


if __name__ == "__main__":
    main()