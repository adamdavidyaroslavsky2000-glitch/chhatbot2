"""Generate text using a rule‑guided decoder.

This script loads a fine‑tuned model (potentially with a usage adapter and a
token filter adapter) and produces outputs guided by user‑specified controls.
It enforces cleanliness by blocking non‑printable tokens, penalising
repetitions, preventing meta prompts ("System:", "Source:", etc.) from
appearing in the output, and dynamically adjusting the temperature over time.

Example usage:
    python -m inference.generate_rule_guided \
        --ckpt checkpoints/rules_adapter.pt \
        --spm_model tokenizer/spm.model \
        --prompt "Explain what lifelong learning means in 3 bullet points." \
        --topic "education" \
        --format "bullets" \
        --controls_json '{"temperature":0.6,"top_p":0.88,"frequency_penalty":0.9,"presence_penalty":0.5}'

The script prints the generated text to stdout.
"""

from __future__ import annotations

import argparse
import json
import string
import unicodedata

import torch

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM


PRINTABLE = set(string.printable) | {"\n", " "}
STOP_STRINGS = ["\nSystem:", "\nSource:", "\nUser:"]


def is_printable_english(s: str) -> bool:
    """Return True if all characters in s are printable English letters/punctuation."""
    if not s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "\uFFFD" or cat.startswith("C"):
            return False
        if ch not in PRINTABLE:
            return False
    return True


def build_bad_token_ids(tok: SPMTokenizer) -> set[int]:
    """Compute a set of token ids whose decoded forms contain non‑printable chars."""
    bad = []
    for tid in range(tok.VOCAB_SIZE):
        piece = tok.decode([tid])
        if not is_printable_english(piece):
            bad.append(tid)
    return set(bad)


def block_repeated_trigrams(x_ids: torch.Tensor, logits: torch.Tensor) -> None:
    """Block tokens that would repeat a trigram in the generated sequence."""
    if x_ids.size(1) < 3:
        return
    hist = x_ids[0].tolist()
    tail = tuple(hist[-2:])
    for i in range(len(hist) - 2):
        tri = tuple(hist[i : i + 3])
        if tri[:-1] == tail:
            logits[0, tri[-1]] = -1e9


@torch.no_grad()
def sample_rule_guided(
    model: TinyTransformerLM,
    tok: SPMTokenizer,
    prompt: str,
    controls: dict,
    topic: str,
    fmt: str,
    max_new_tokens: int = 180,
    device: str = "cpu",
) -> str:
    """Generate a completion given a prompt and optional control parameters."""
    # Base controls
    temperature = float(controls.get("temperature", 0.7))
    top_p = float(controls.get("top_p", 0.9))
    freq_penalty = float(controls.get("frequency_penalty", 0.8))
    pres_penalty = float(controls.get("presence_penalty", 0.5))

    # Topic and format guidance
    topic_boost = float(controls.get("topic_anchor_boost", 0.8))
    drift_penalty = float(controls.get("topic_drift_penalty", 1.2))
    fmt_penalty = float(controls.get("format_violation_penalty", 0.0))
    temp_floor = float(controls.get("temp_floor", 0.5))
    temp_ceiling = float(controls.get("temp_ceiling", temperature))
    min_tokens = int(controls.get("min_tokens", 60))
    hard_max = int(controls.get("max_tokens", max_new_tokens))

    # Encode the initial prompt
    x = torch.tensor(tok.encode(prompt, add_bos=True, add_eos=False), dtype=torch.long, device=device).unsqueeze(0)
    freq = {}
    topic_ids = set(tok.encode(topic, add_bos=False, add_eos=False)) if topic else set()
    bad_token_ids = build_bad_token_ids(tok)

    for i in range(hard_max):
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len:]
        logits = model(x)[:, -1, :]
        # Repetition penalties
        for tid, cnt in freq.items():
            logits[0, tid] -= freq_penalty * cnt
        for tid in set(x[0].tolist()):
            logits[0, tid] -= pres_penalty
        # Topic guidance: nudge tokens within the topic
        for tt in topic_ids:
            logits[0, tt] += topic_boost
        # Format guidance: encourage appropriate bullet/number/json tokens
        if fmt == "bullets":
            for sym in tok.encode("- ", add_bos=False, add_eos=False):
                logits[0, sym] += 0.1 - fmt_penalty
        elif fmt == "numbered":
            for sym in ["1", "2", "3", "4", ")"]:
                for b in tok.encode(sym, add_bos=False, add_eos=False):
                    logits[0, b] += 0.05
        elif fmt in ("json", "yaml"):
            for sym in ["{", "}", "\"", ":", ","]:
                for b in tok.encode(sym, add_bos=False, add_eos=False):
                    logits[0, b] += 0.05
        # Block non‑printable tokens
        if bad_token_ids:
            logits[0, list(bad_token_ids)] = -1e9
        # Block repeated trigrams
        block_repeated_trigrams(x, logits)
        # Dynamic temperature schedule
        progress = i / max(1, hard_max - 1)
        dyn_temp = max(temp_floor, min(temp_ceiling, temperature * (1.0 - 0.2 * progress)))
        logits = logits / max(1e-6, dyn_temp)
        # Top‑p sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, idx_next)
        tid = next_token.item()
        x = torch.cat([x, next_token], dim=1)
        freq[tid] = freq.get(tid, 0) + 1
        # Convert to text and check for stop sequences
        text = tok.decode(x.squeeze(0).tolist())
        for s in STOP_STRINGS:
            pos = text.find(s)
            if pos != -1 and i >= min_tokens:
                return text[:pos].rstrip()
    return tok.decode(x.squeeze(0).tolist()).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with rule‑guided decoding")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="SentencePiece model")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--topic", type=str, default="", help="Topic for guidance")
    parser.add_argument("--format", type=str, default="paragraph", help="Desired format: paragraph, bullets, numbered, json")
    parser.add_argument("--controls_json", type=str, default="{}", help="JSON string with control parameters")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, mps or cuda")
    args = parser.parse_args()

    tok = SPMTokenizer(args.spm_model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    model = TinyTransformerLM(**config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    controls = json.loads(args.controls_json)
    output = sample_rule_guided(model, tok, args.prompt, controls, args.topic, args.format, device=device)
    print(output)


if __name__ == "__main__":
    main()