"""Gated Attention — direct re-export of mlx-lm's Qwen3NextAttention.

mlx-lm already ships the production Gated Attention used by Qwen3-Next /
Qwen3.5 / Qwen3.6 (the 25% softmax slot in their hybrid 3:1 GDN+Attn
stack). It implements all four features that distinguish Gated Attention
from vanilla SDPA:

  - asymmetric GQA (num_attention_heads != num_key_value_heads)
  - per-head RMSNorm on Q and K (post-projection)
  - partial RoPE (partial_rotary_factor controls the fraction of head_dim)
  - sigmoid output gate: ``self.o_proj(SDPA(q, k, v) * sigmoid(gate))``
    where ``gate`` is split from a 2x-sized q_proj output

Under the hood it calls ``mx.fast.scaled_dot_product_attention`` (the
Apple MPS fused softmax SDPA kernel) — there is no faster Metal path on
M-series, so wrapping it ourselves would add nothing. We import the
upstream module verbatim and just expose it via our V4 block-builder
interface.

Upstream reference:
    mlx_lm/models/qwen3_next.py::Qwen3NextAttention
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

# Direct upstream import — no vendoring, no re-implementation. The mlx-lm
# Qwen3-Next module is part of the installed package; we use it as-is.
from mlx_lm.models.qwen3_next import (
    ModelArgs as _Qwen3NextModelArgs,
    Qwen3NextAttention as _UpstreamGatedAttention,
)


@dataclass(frozen=True)
class GatedAttentionConfig:
    """Configuration for the Gated Attention block.

    Defaults match the Qwen3-Next / Qwen3.6 attention slot:
        num_attention_heads=16, num_key_value_heads=2 (8:1 GQA),
        head_dim=128, partial_rotary_factor=0.25.
    """

    hidden_size: int
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262_144
    attention_bias: bool = False
    rope_scaling: Optional[dict] = None


def _build_upstream_args(cfg: GatedAttentionConfig) -> _Qwen3NextModelArgs:
    """Build the minimal mlx-lm ModelArgs that Qwen3NextAttention reads.

    Qwen3NextAttention.__init__ only touches the attention-related fields
    of ModelArgs; the linear-attn / MoE / norm-vocab fields are unused at
    instantiation time but required by the @dataclass schema. Fill them
    with neutral defaults so the dataclass constructor accepts.
    """
    return _Qwen3NextModelArgs(
        model_type="qwen3_next",
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        intermediate_size=cfg.hidden_size * 4,
        num_attention_heads=cfg.num_attention_heads,
        # Linear-attn fields (unused by Qwen3NextAttention)
        linear_num_value_heads=cfg.num_attention_heads,
        linear_num_key_heads=cfg.num_attention_heads,
        linear_key_head_dim=cfg.head_dim,
        linear_value_head_dim=cfg.head_dim,
        linear_conv_kernel_dim=4,
        # MoE fields (unused by Qwen3NextAttention)
        num_experts=1,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=cfg.hidden_size,
        mlp_only_layers=[],
        moe_intermediate_size=cfg.hidden_size,
        # Norm / vocab (unused for this block but required by schema)
        rms_norm_eps=cfg.rms_norm_eps,
        vocab_size=1,
        # Attention-specific
        num_key_value_heads=cfg.num_key_value_heads,
        rope_theta=cfg.rope_theta,
        partial_rotary_factor=cfg.partial_rotary_factor,
        max_position_embeddings=cfg.max_position_embeddings,
        head_dim=cfg.head_dim,
        attention_bias=cfg.attention_bias,
        rope_scaling=cfg.rope_scaling,
    )


class GatedAttentionBlock(nn.Module):
    """Thin V4-block wrapper around mlx-lm's Qwen3NextAttention.

    Exposes ``self.attn`` so weight loaders that walk ``Qwen3NextAttention``
    sub-parameter names (q_proj/k_proj/v_proj/o_proj/q_norm/k_norm) find
    them at the expected paths.
    """

    def __init__(self, cfg: GatedAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = _UpstreamGatedAttention(_build_upstream_args(cfg))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return self.attn(x, mask=mask, cache=cache)


__all__ = [
    "GatedAttentionBlock",
    "GatedAttentionConfig",
]
