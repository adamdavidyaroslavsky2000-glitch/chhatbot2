"""Core neural network components for the Transformer model.

The layers defined here are intentionally minimal: they avoid
dependencies on external libraries beyond PyTorch and implement only
what is necessary for a decoder‑only language model.  You can extend
these layers with rotary positional embeddings or other techniques as
needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square normalisation.

    RMSNorm scales inputs based on their root mean square to stabilise
    training.  It behaves similarly to LayerNorm but is computationally
    cheaper and often yields comparable performance.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.weight * (x / (rms + self.eps))


class MultiHeadSelfAttention(nn.Module):
    """Multi‑head self‑attention module without bias.

    This implementation uses a single linear layer to compute queries,
    keys and values (QKV) and then splits them into heads.  It
    supports an optional rotary position embedding function passed via
    the constructor.
    """

    def __init__(self, d_model: int, n_heads: int, rope_fn=None) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope = rope_fn
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each has shape (B, T, n_heads, d_head)
        if self.rope:
            q, k = self.rope(q), self.rope(k)
        # Compute scaled dot‑product attention
        scores = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores += attn_mask  # mask should already be additive
        w = scores.softmax(dim=-1)
        out = torch.einsum('bhtT,bThd->bthd', w, v)
        out = out.contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """Position‑wise feed‑forward network with SiLU activation."""

    def __init__(self, d_model: int, mult: float = 4.0) -> None:
        super().__init__()
        hidden = int(d_model * mult)
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))