"""Per-layer scaled embedding block — Gemma 4 E2B/E4B style.

Gallery entries #57/#58. Adds a per-layer learnable scale + bias to the
residual stream (one scale vector per token-position). Conceptually a
"sidechannel embedding lookup that's offset-dependent".

Categorised as ``norm_or_proj``. Drop-in shape-preserving block.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class PerLayerEmbedConfig:
    hidden_size: int
    layer_index: int = 0
    num_layers: int = 32


class PerLayerEmbedBlock(nn.Module):
    """Per-layer learnable scale + bias modulation of the residual stream."""

    def __init__(self, cfg: PerLayerEmbedConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        # Scale starts at 1.0, bias at 0 (identity at init).
        self.scale = mx.ones((H,))
        self.bias = mx.zeros((H,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale + self.bias


__all__ = ["PerLayerEmbedBlock", "PerLayerEmbedConfig"]
