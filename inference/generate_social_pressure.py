#!/usr/bin/env python3
"""
generate_social_pressure.py
---------------------------

Generate text from a TinyTransformerLM model using a social pressure mechanism.
This script loads a fine‑tuned model (potentially with rule and usage adapters) and
applies additional heuristics based on a synonyms dictionary to encourage
responses that are closer to preferred wording and avoid undesirable terms.

The synonyms file (`social_pressure_synonyms.json`) should define two arrays:
`good_synonyms` and `bad_synonyms`.  Tokens corresponding to words in
`good_synonyms` are boosted while those in `bad_synonyms` are penalized.

Usage example:

    python -m inference.generate_social_pressure \
      --ckpt checkpoints/social_pressure_adapter.pt \
      --spm_model tokenizer/spm.model \
      --synonyms_file social_pressure_synonyms.json \
      --prompt "Describe machine learning." \
      --max_new_tokens 100 --temperature 0.7 --top_p 0.9 --device mps

"""
import argparse
import json
import string
import unicodedata
import torch
import math
from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM


PRINTABLE = set(string.printable) | {"\n", " "}
STOP_STRINGS = ["\nSystem:", "\nSource:", "\nUser:"]


def is_printable_english(s: str) -> bool:
    """Return True if the string contains only printable ASCII characters (plus newline and space) and no control codes."""
    if not s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "\uFFFD" or cat.startswith("C"):
            return False
        if ch not in PRINTABLE:
            return False
    return True


def build_bad_token_ids(tok: SPMTokenizer) -> set:
    bad = []
    for tid in range(tok.VOCAB_SIZE):
        piece = tok.decode([tid])
        if not is_printable_english(piece):
            bad.append(tid)
    return set(bad)


def block_repeated_trigrams(x_ids: torch.Tensor, logits: torch.Tensor) -> None:
    """Block generation of any trigram that has already appeared in the sequence."""
    if x_ids.size(1) < 3:
        return
    hist = x_ids[0].tolist()
    tail = tuple(hist[-2:])
    for i in range(len(hist) - 2):
        tri = tuple(hist[i:i+3])
        if tri[:-1] == tail:
            logits[0, tri[-1]] = -1e9


def load_synonym_ids(synonyms_file: str, tok: SPMTokenizer) -> (set, set):
    """Load synonyms JSON and return sets of token ids for good and bad words."""
    with open(synonyms_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    good_words = data.get("good_synonyms", []) or []
    bad_words = data.get("bad_synonyms", []) or []
    good_ids = set()
    bad_ids = set()
    for word in good_words:
        ids = tok.encode(word, add_bos=False, add_eos=False)
        # only take the first piece (some words may split)
        if ids:
            good_ids.add(ids[0])
    for word in bad_words:
        ids = tok.encode(word, add_bos=False, add_eos=False)
        if ids:
            bad_ids.add(ids[0])
    return good_ids, bad_ids


@torch.no_grad()
def sample_social(model: TinyTransformerLM, tok: SPMTokenizer, prompt: str,
                  good_ids: set, bad_ids: set,
                  temperature: float = 0.7, top_p: float = 0.9,
                  good_boost: float = 0.3, bad_penalty: float = 0.8,
                  freq_penalty: float = 0.8, pres_penalty: float = 0.4,
                  max_new_tokens: int = 200, min_tokens: int = 40,
                  device: str = "cpu") -> str:
    """Generate a response using social pressure heuristics."""
    device = torch.device(device)
    # encode the prompt
    x = torch.tensor(tok.encode(prompt, add_bos=True, add_eos=False), dtype=torch.long, device=device).unsqueeze(0)
    bad_token_ids = build_bad_token_ids(tok)
    # track frequencies for penalties
    freq = {}
    for tid in x[0].tolist():
        freq[tid] = freq.get(tid, 0) + 1
    out_started = False
    seq_generated = []
    for step in range(max_new_tokens):
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len:]
        logits = model(x)[:, -1, :]  # shape (1, vocab_size)
        # frequency and presence penalties
        for tid, cnt in freq.items():
            logits[0, tid] -= freq_penalty * cnt
        for tid in set(x[0].tolist()):
            logits[0, tid] -= pres_penalty
        # apply good/bad synonym adjustments
        if good_ids:
            logits[0, list(good_ids)] += good_boost
        if bad_ids:
            logits[0, list(bad_ids)] -= bad_penalty
        # block non‑printable tokens
        logits[0, list(bad_token_ids)] = -1e9
        # block repeated trigrams
        block_repeated_trigrams(x, logits)
        # apply temperature
        logits = logits / max(1e-6, temperature)
        # nucleus sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(sorted_probs, num_samples=1)
        token = sorted_idx.gather(-1, idx_next)
        tok_id = token.item()
        x = torch.cat([x, token], dim=1)
        freq[tok_id] = freq.get(tok_id, 0) + 1
        seq_generated.append(tok_id)
        # decode partial and check stop conditions
        text = tok.decode(x.squeeze(0).tolist())
        for stop_str in STOP_STRINGS:
            pos = text.find(stop_str)
            if pos != -1 and len(seq_generated) >= min_tokens:
                return text[:pos].rstrip()
    return tok.decode(x.squeeze(0).tolist()).rstrip()


def main():
    parser = argparse.ArgumentParser(description="Generate text using social pressure heuristics")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="Path to SentencePiece model")
    parser.add_argument("--synonyms_file", type=str, default="social_pressure_synonyms.json", help="Path to synonyms JSON file")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--good_boost", type=float, default=0.3, help="Logit boost for good synonyms")
    parser.add_argument("--bad_penalty", type=float, default=0.8, help="Logit penalty for bad synonyms")
    parser.add_argument("--max_new_tokens", type=int, default=180, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, mps, cuda)")
    args = parser.parse_args()
    # load tokenizer and synonyms
    tok = SPMTokenizer(args.spm_model)
    good_ids, bad_ids = load_synonym_ids(args.synonyms_file, tok)
    # load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    model = TinyTransformerLM(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()
    text = sample_social(model, tok, args.prompt, good_ids, bad_ids,
                         temperature=args.temperature, top_p=args.top_p,
                         good_boost=args.good_boost, bad_penalty=args.bad_penalty,
                         max_new_tokens=args.max_new_tokens, device=args.device)
    print(text)


if __name__ == "__main__":
    main()