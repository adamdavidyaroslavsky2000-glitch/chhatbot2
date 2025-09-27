"""
Token‑filtered generation script.

This generator extends the rule‑guided generator by incorporating a
simple token filter heuristic.  It boosts the probability of
important tokens derived from the input and penalizes known filler
tokens (e.g. stop words).  The goal is to encourage the model to
select semantically meaningful tokens at each step, improving
generation quality when the base model occasionally outputs
gibberish or irrelevant text.

The filtering mechanism is not a learned classifier; instead, it
uses the same simple heuristic employed when generating the token
filter datasets.  For a richer solution you can train the
token‑filter adapter via ``train/train_token_filter_adapter.py`` and
integrate its outputs here.

Usage::

    python -m inference.generate_with_filter \
        --ckpt checkpoints/rules_adapter.pt \
        --spm_model tokenizer/spm.model \
        --prompt "Explain dropout for beginners." \
        --topic machine_learning --format paragraph \
        --controls_json '{"temperature":0.7, "top_p":0.9}'
"""

import argparse
import json
import re
import math
import string
import unicodedata
from typing import List, Set

import torch

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM


STOPWORDS = {
    "the", "is", "and", "or", "to", "of", "in", "a", "an", "on", "for",
    "with", "at", "from", "by", "about", "as", "into", "like", "through",
    "after", "over", "between", "out", "against", "during", "without",
    "before", "under", "around", "among", "up", "down", "off", "than",
    "so", "too", "are", "am", "be", "been", "being", "do", "does", "did",
    "but", "yet", "if", "then", "else", "when", "while", "where", "why",
    "how", "what", "which", "that", "this", "it", "its", "it's"
}

PRINTABLE = set(string.printable) | {"\n", " "}
STOP_STRINGS = ["\nSystem:", "\nSource:", "\nUser:"]


def tokenize_simple(text: str) -> List[str]:
    """Tokenize a string into lowercase words, removing punctuation."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return [t.lower() for t in cleaned.split() if t.strip()]


def classify_input_tokens(text: str) -> (Set[str], Set[str]):
    """Return sets of important and less important tokens based on heuristics."""
    important = set()
    less = set()
    for t in tokenize_simple(text):
        if t.isdigit() or (t not in STOPWORDS and len(t) > 2):
            important.add(t)
        else:
            less.add(t)
    return important, less


def is_printable_english(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "\uFFFD" or cat.startswith("C"):
            return False
        if ch not in PRINTABLE:
            return False
    return True


def build_bad_token_ids(tok: SPMTokenizer) -> Set[int]:
    """Return set of token IDs that should never be produced (non‑printable/gibberish)."""
    bad = set()
    for tid in range(tok.VOCAB_SIZE):
        piece = tok.decode([tid])
        if not is_printable_english(piece):
            bad.add(tid)
    return bad


def block_repeated_trigrams(x_ids: torch.Tensor, logits: torch.Tensor) -> None:
    """Apply in‑place penalty to logits for repeating trigrams."""
    if x_ids.size(1) < 3:
        return
    hist = x_ids[0].tolist()
    tail = tuple(hist[-2:])
    for i in range(len(hist) - 2):
        tri = tuple(hist[i:i+3])
        if tri[:-1] == tail:
            logits[0, tri[-1]] = -1e9


@torch.no_grad()
def sample_filtered(model: TinyTransformerLM, tok: SPMTokenizer, prompt: str,
                    controls: dict, topic: str, fmt: str, device: torch.device) -> str:
    """Generate a response using token filtering heuristics."""
    temperature = float(controls.get("temperature", 0.7))
    top_p     = float(controls.get("top_p", 0.9))
    max_len   = int(controls.get("max_tokens", 180))
    min_len   = int(controls.get("min_tokens", 50))
    freq_pen  = float(controls.get("frequency_penalty", 0.8))
    pres_pen  = float(controls.get("presence_penalty", 0.5))

    important, less = classify_input_tokens(prompt)
    bad_token_ids = build_bad_token_ids(tok)
    input_ids = torch.tensor(tok.encode(prompt, add_bos=True, add_eos=False), dtype=torch.long, device=device).unsqueeze(0)
    freq = {}
    for t in input_ids[0].tolist():
        freq[t] = freq.get(t, 0) + 1

    for step in range(max_len):
        if input_ids.size(1) > model.max_seq_len:
            input_ids = input_ids[:, -model.max_seq_len:]
        logits = model(input_ids)[:, -1, :]
        # repetition penalties
        for tid, cnt in freq.items():
            logits[0, tid] -= freq_pen * cnt
        for tid in set(input_ids[0].tolist()):
            logits[0, tid] -= pres_pen
        # penalise bad tokens
        if bad_token_ids:
            logits[0, list(bad_token_ids)] = -1e9
        # simple token importance adjustment
        for tid in range(tok.VOCAB_SIZE):
            piece = tok.decode([tid])
            if not piece:
                continue
            text_piece = piece.lower().strip()
            # If a piece corresponds exactly to an important token, boost
            if text_piece in important:
                logits[0, tid] += 0.3
            elif text_piece in less:
                logits[0, tid] -= 0.3
        # trigram blocking
        block_repeated_trigrams(input_ids, logits)
        # sampling
        probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, idx_next)
        t_id = next_token.item()
        input_ids = torch.cat([input_ids, next_token], dim=1)
        freq[t_id] = freq.get(t_id, 0) + 1
        # stop if we have met stop conditions
        if step >= min_len:
            text_out = tok.decode(input_ids.squeeze(0).tolist())
            for s in STOP_STRINGS:
                pos = text_out.find(s)
                if pos != -1:
                    return text_out[:pos].rstrip()
        # if EOS appears
        if t_id == tok.EOS:
            break
    return tok.decode(input_ids.squeeze(0).tolist()).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint to use")
    ap.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="Path to SentencePiece model")
    ap.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model")
    ap.add_argument("--topic", type=str, default="general", help="Topic or domain (not used but kept for compatibility)")
    ap.add_argument("--format", type=str, default="paragraph", help="Desired format (paragraph, bullets, etc.)")
    ap.add_argument("--controls_json", type=str, default="{}", help="JSON string of control parameters (temperature, top_p, etc.)")
    ap.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
    args = ap.parse_args()
    controls = json.loads(args.controls_json)
    # load tokenizer and model
    tok = SPMTokenizer(args.spm_model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    model = TinyTransformerLM(**cfg)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    output = sample_filtered(model, tok, args.prompt, controls, args.topic, args.format, device)
    print(output)


if __name__ == "__main__":
    main()