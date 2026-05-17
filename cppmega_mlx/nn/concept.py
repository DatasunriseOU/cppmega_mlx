"""MLX port of nanochat's CBlock concept-retrieval cross-attention.

Each token cross-attends from its hidden state into a learned bank of K concept
vectors. The bank is read-only at forward time; it is updated only through
gradient descent. ``out_proj`` is zero-initialized so the block is the identity
at initialization, which makes it safe to insert at any position in the stack.

Reference: nanochat/nanochat/concepts.py:CBlock
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class ConceptBlockConfig:
    """Static-shape config for the MLX concept-retrieval block."""

    hidden_size: int
    num_concepts: int = 64
    concept_dim: int | None = None
    num_heads: int = 4
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_concepts <= 0:
            raise ValueError(f"num_concepts must be positive, got {self.num_concepts}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads "
                f"{self.num_heads}"
            )
        if self.concept_dim is not None and self.concept_dim <= 0:
            raise ValueError(f"concept_dim must be positive, got {self.concept_dim}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    @property
    def effective_concept_dim(self) -> int:
        return self.concept_dim if self.concept_dim is not None else self.hidden_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _rms_norm_last(x: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


class ConceptBlock(nn.Module):
    """Cross-attention from token hidden states into a learned concept bank.

    Shape contract: input ``(B, S, D)`` → output ``(B, S, D)``. No causal mask
    is applied: concepts are global prototypes, order-invariant by construction.
    """

    def __init__(self, config: ConceptBlockConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        cd = config.effective_concept_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.eps = config.eps

        self.concept_bank = nn.Embedding(config.num_concepts, cd)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(cd, d, bias=False)
        self.v_proj = nn.Linear(cd, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.out_proj.weight = mx.zeros_like(self.out_proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 3:
            raise ValueError(f"x must be shaped (B, S, D), got {x.shape}")
        if x.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"x last dim must be {self.config.hidden_size}, got {x.shape[-1]}"
            )

        b, s, d = x.shape
        h = self.num_heads
        dh = self.head_dim
        k = self.config.num_concepts

        x_norm = _rms_norm_last(x, self.eps)

        # Q: [B, S, D] -> [B, H, S, Dh]
        q = self.q_proj(x_norm).reshape(b, s, h, dh)
        q = mx.transpose(q, (0, 2, 1, 3))

        concepts = self.concept_bank.weight  # [K, concept_dim]
        # Projected to D, reshaped to [H, K, Dh] and broadcast over batch.
        k_c = self.k_proj(concepts).reshape(k, h, dh)
        v_c = self.v_proj(concepts).reshape(k, h, dh)
        k_c = mx.transpose(k_c, (1, 0, 2))[None, :, :, :]
        v_c = mx.transpose(v_c, (1, 0, 2))[None, :, :, :]

        # Scores: [B, H, S, Dh] x [1, H, Dh, K] -> [B, H, S, K]
        scores = mx.matmul(q, mx.transpose(k_c, (0, 1, 3, 2))) * self.scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)

        # Retrieve: [B, H, S, K] x [1, H, K, Dh] -> [B, H, S, Dh]
        out = mx.matmul(weights, v_c)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, s, d)
        return self.out_proj(out)


__all__ = ["ConceptBlock", "ConceptBlockConfig"]
