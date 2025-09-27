"""
General purpose text generation script for the tiny transformer model.

This script loads a model checkpoint and a SentencePiece tokenizer and
generates a continuation for a given prompt.  It incorporates a number of
techniques to produce higher quality responses:

* **Stop sequences**: if any of the stop strings (e.g. "\nSystem:" or
  "\nSource:") appear in the generated text, generation stops.  This
  prevents the model from leaking meta information or repeating prompt
  structure in the output.
* **Printable token filtering**: tokens that decode to non‑printable or
  invalid Unicode characters are prevented from being sampled.  This
  eliminates gibberish artifacts in the output.
* **N‑gram blocking**: repeated trigrams (3‑token sequences) are blocked to
  discourage the model from repeating phrases verbatim and to reduce
  degenerative loops.
* **Repetition penalties**: frequency and presence penalties are applied
  based on the history of generated tokens, encouraging the model to use
  diverse vocabulary and avoid looping.
* **Dynamic temperature scheduling**: temperature starts at a user supplied
  value and gradually decays towards a lower bound as generation
  progresses.  This makes early tokens more diverse while later tokens
  become more deterministic.
* **Top‑p sampling**: nucleus sampling restricts sampling to the most
  probable subset of tokens whose cumulative probability exceeds a
  threshold.  This provides a principled way of balancing diversity and
  coherence and is recommended over top‑k when controlling randomness
 【924795240632236†L220-L249】.

Example usage:

```bash
python -m inference.generate \
  --ckpt checkpoints/usage_adapter.pt \
  --spm_model tokenizer/spm.model \
  --prompt "Explain what machine learning is in two sentences." \
  --max_new_tokens 160 \
  --temperature 0.7 \
  --top_p 0.9 \
  --device mps
```

This will produce a concise description without leaking any internal
instructions or emitting non‑printable characters.  See the README for
guidelines on tuning the temperature and top‑p parameters.
"""

import argparse
import json
import math
import string
import unicodedata
from typing import Iterable, List

import torch

from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM

# Define a set of stop sequences that should not appear in outputs.
STOP_STRINGS: List[str] = ["\nSystem:", "\nSource:", "\nUser:"]

# The set of printable characters plus whitespace.  Any decoded token that
# contains only these characters is considered safe.
PRINTABLE = set(string.printable) | {"\n", " "}

def is_printable(s: str) -> bool:
    """Return True if the string contains only printable ASCII characters.

    Non‑printable or control characters (including the Unicode replacement
    character U+FFFD) are considered unsafe and will cause the token to be
    banned from sampling.
    """
    if not s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "\uFFFD" or cat.startswith("C"):
            return False
        if ch not in PRINTABLE:
            return False
    return True

def build_bad_token_ids(tokenizer: SPMTokenizer) -> set:
    """Return a set of token IDs that decode to non‑printable sequences."""
    bad: List[int] = []
    for tid in range(tokenizer.VOCAB_SIZE):
        piece = tokenizer.decode([tid])
        if not is_printable(piece):
            bad.append(tid)
    return set(bad)

def block_repeated_trigrams(x: torch.Tensor, logits: torch.Tensor) -> None:
    """Block tokens that would create repeated 3‑grams.

    Args:
        x: tensor of shape (1, T) containing the current sequence of token IDs.
        logits: tensor of shape (1, V) containing the logits for the next token.
    """
    if x.size(1) < 3:
        return
    hist = x[0].tolist()
    tail = tuple(hist[-2:])
    for i in range(len(hist) - 2):
        tri = tuple(hist[i:i + 3])
        if tri[:-1] == tail:
            # Mask the token that would complete this trigram
            logits[0, tri[-1]] = -1e9

@torch.no_grad()
def generate_sequence(
    model: TinyTransformerLM,
    tokenizer: SPMTokenizer,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
    frequency_penalty: float = 0.9,
    presence_penalty: float = 0.5,
    min_tokens: int = 40,
    temp_floor: float = 0.5,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt using the provided model and tokenizer.

    Args:
        model: Loaded Transformer model.
        tokenizer: SentencePiece tokenizer.
        prompt: Input prompt (string).  BOS is automatically prepended.
        max_new_tokens: Maximum number of tokens to generate.  The total
            sequence length (prompt + generated) will be at most
            model.max_seq_len.
        temperature: Initial temperature for sampling.  A higher value
            results in more diverse output.  See EntryPoint AI’s tuning
            guidelines【819446651303748†L104-L111】.
        top_p: Nucleus sampling threshold (between 0 and 1).  Only the most
            probable tokens whose cumulative probability sums to `top_p` will
            be considered【924795240632236†L220-L249】.
        frequency_penalty: Penalty applied once per occurrence of a token in
            the generated sequence.  Higher values discourage repeated words
            and reduce looping【924795240632236†L220-L249】.
        presence_penalty: Penalty applied if a token has appeared at least
            once in the sequence.  Lower values encourage diversity but may
            still discourage reuse of tokens.
        min_tokens: Minimum number of tokens to generate before checking for
            stop conditions.  Prevents premature truncation.
        temp_floor: Lower bound on the temperature schedule.  Temperature
            decays linearly from the initial value to this floor.
        device: Device to run inference on (e.g. "cpu", "mps", "cuda").
    Returns:
        Generated text including the prompt and new tokens up to a stop
        condition.
    """
    # Encode the prompt.  Add a BOS token and omit EOS.
    x = torch.tensor(
        tokenizer.encode(prompt, add_bos=True, add_eos=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    # Track token frequencies for repetition penalties
    freq: dict = {}
    for t in x[0].tolist():
        freq[t] = freq.get(t, 0) + 1
    # Precompute non‑printable token set
    bad_token_ids = build_bad_token_ids(tokenizer)
    total_steps = max_new_tokens
    for step in range(max_new_tokens):
        # Maintain sequence length within model limits
        if x.size(1) > model.max_seq_len:
            x = x[:, -model.max_seq_len:]
        # Forward pass.  Only the last logits are needed.
        logits = model(x)[:, -1, :]
        # Apply repetition penalties
        for tid, count in freq.items():
            logits[0, tid] -= frequency_penalty * count
        for tid in freq.keys():
            logits[0, tid] -= presence_penalty
        # Mask bad tokens
        if bad_token_ids:
            logits[0, list(bad_token_ids)] = -1e9
        # Block repeated 3‑grams
        block_repeated_trigrams(x, logits)
        # Linearly decay the temperature
        progress = step / max(1, total_steps - 1)
        dyn_temp = max(temp_floor, temperature * (1.0 - 0.5 * progress))
        # Rescale logits
        logits = logits / max(1e-6, dyn_temp)
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        # Nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumprobs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(sorted_probs, num_samples=1)
        token = sorted_idx.gather(-1, idx_next)
        tok_id = token.item()
        # Append token and update frequency
        x = torch.cat([x, token], dim=1)
        freq[tok_id] = freq.get(tok_id, 0) + 1
        # Decode current text
        if x.size(1) >= min_tokens:
            text = tokenizer.decode(x.squeeze(0).tolist())
            for s in STOP_STRINGS:
                pos = text.find(s)
                if pos != -1:
                    return text[:pos].rstrip()
    return tokenizer.decode(x.squeeze(0).tolist()).rstrip()

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the TinyTransformerLM.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--spm_model", type=str, default="tokenizer/spm.model", help="Path to SentencePiece model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Initial sampling temperature (0 < T <= 1)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold (0 < p <= 1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu/mps/cuda)")
    args = parser.parse_args()
    # Load tokenizer and model
    tokenizer = SPMTokenizer(args.spm_model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    model = TinyTransformerLM(**config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(
        args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()
    # Generate and print
    output = generate_sequence(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(output)

if __name__ == "__main__":
    main()