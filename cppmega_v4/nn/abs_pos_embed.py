"""Learned absolute positional embedding block — GPT-2 style.

Gallery entry #1 (GPT-2 XL). Adds a learnable position-table lookup to
the residual stream. Drop-in passthrough that only changes the input by
adding ``position_table[arange(S)]`` once at the start of the network.

Categorised as ``norm_or_proj`` in the fusion compatibility table —
trivially fuses with anything next to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AbsPosEmbedConfig:
    hidden_size: int
    max_position_embeddings: int = 4096


class AbsPosEmbedBlock(nn.Module):
    """Learned absolute positional embedding (GPT-2 style)."""

    def __init__(self, cfg: AbsPosEmbedConfig):
        super().__init__()
        self.cfg = cfg
        self.pos_table = nn.Embedding(
            cfg.max_position_embeddings, cfg.hidden_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        if S > self.cfg.max_position_embeddings:
            raise ValueError(
                f"AbsPosEmbedBlock: seq_len {S} exceeds "
                f"max_position_embeddings {self.cfg.max_position_embeddings}"
            )
        positions = mx.arange(S)
        return x + self.pos_table(positions)


__all__ = ["AbsPosEmbedBlock", "AbsPosEmbedConfig"]
