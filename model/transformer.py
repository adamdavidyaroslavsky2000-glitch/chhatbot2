"""A minimal decoder‑only Transformer language model.

This module defines a simple transformer architecture suitable for
training on small datasets and fine‑tuning on consumer hardware.  It
uses RMSNorm, multi‑head self‑attention and a feed‑forward network.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from .layers import RMSNorm, MultiHeadSelfAttention, MLP


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create an additive causal mask for self‑attention.

    The mask has shape (1, 1, seq_len, seq_len) and contains 0s on
    and below the diagonal and −inf above the diagonal.  Adding this
    mask to the attention scores prevents attending to future tokens.
    """
    mask = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TransformerBlock(nn.Module):
    """A single decoder transformer block."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: float = 4.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformerLM(nn.Module):
    """A minimal decoder‑only transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        mlp_mult: float = 4.0,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp_mult = mlp_mult
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_mult) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying between embedding and lm_head
        self.lm_head.weight = self.token_embed.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for each position in the input.

        Args:
            idx: Tensor of shape (batch, seq_len) of token IDs.

        Returns:
            Tensor of shape (batch, seq_len, vocab_size) of unnormalised logits.
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, "Sequence length exceeds maximum length"
        device = idx.device
        # Input embeddings + learned positional embeddings
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        attn_mask = build_causal_mask(T, device)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.lm_head(x)
        return logits