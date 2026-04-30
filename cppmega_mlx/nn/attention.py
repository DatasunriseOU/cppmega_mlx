"""MLX causal attention references for cppmega NAM56R A-layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

AttentionMode = Literal["mla", "dsa"]


@dataclass(frozen=True)
class AttentionRouteInfo:
    """Runtime marker for cppmega A-layer routing.

    ``dsa`` currently uses the same dense causal reference path as ``mla``.
    The marker keeps NAM56R layer intent visible until sparse DSA/MLA Metal
    kernels are wired in.
    """

    mode: AttentionMode
    backend: str
    sparse_reference: bool = False


@dataclass(frozen=True)
class AttentionConfig:
    """Small local config for MLX attention smoke models and tests."""

    d_model: int
    num_q_heads: int
    num_kv_heads: int | None = None
    head_dim: int | None = None
    mode: AttentionMode = "mla"
    use_rope: bool = True
    rope_theta: float = 10000.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("mla", "dsa"):
            raise ValueError(f"mode must be 'mla' or 'dsa', got {self.mode!r}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_q_heads <= 0:
            raise ValueError(f"num_q_heads must be positive, got {self.num_q_heads}")
        num_kv_heads = self.num_q_heads if self.num_kv_heads is None else self.num_kv_heads
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if self.num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads {self.num_q_heads} must be divisible by "
                f"num_kv_heads {num_kv_heads}"
            )
        head_dim = self.d_model // self.num_q_heads if self.head_dim is None else self.head_dim
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if self.head_dim is None and self.d_model % self.num_q_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by num_q_heads "
                f"{self.num_q_heads} when head_dim is omitted"
            )
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")

    @property
    def kv_heads(self) -> int:
        return self.num_q_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def q_head_dim(self) -> int:
        return self.d_model // self.num_q_heads if self.head_dim is None else self.head_dim

    @property
    def q_proj_dim(self) -> int:
        return self.num_q_heads * self.q_head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.kv_heads * self.q_head_dim

    @property
    def is_gqa(self) -> bool:
        return self.kv_heads != self.num_q_heads


def _causal_additive_mask(seq_length: int, dtype: mx.Dtype) -> mx.array:
    return nn.MultiHeadAttention.create_additive_causal_mask(seq_length, dtype=dtype)


class CausalSelfAttention(nn.Module):
    """Correctness-first MLX causal self-attention for cppmega A-layers.

    The module uses MLX fast SDPA with tensors in ``(B, heads, S, D)`` form.
    ``mode='dsa'`` intentionally remains a dense causal placeholder/reference:
    it preserves the layer route marker but does not implement sparse DSA top-k
    indexing or production MLA absorbed projections.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.q_proj_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.kv_proj_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.kv_proj_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.q_proj_dim, config.d_model, bias=config.bias)
        self.rope = nn.RoPE(config.q_head_dim, base=config.rope_theta) if config.use_rope else None
        self.route_info = AttentionRouteInfo(mode=config.mode, backend="mlx.fast.sdpa")

    def _project_qkv(self, hidden_states: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        batch, seq, _ = hidden_states.shape
        cfg = self.config
        q = self.q_proj(hidden_states).reshape(batch, seq, cfg.num_q_heads, cfg.q_head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, cfg.kv_heads, cfg.q_head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, cfg.kv_heads, cfg.q_head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        return q, k, v

    def __call__(self, hidden_states: mx.array, mask: mx.array | Literal["causal"] | None = None) -> mx.array:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be shaped (B,S,D), got {hidden_states.shape}")
        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states last dim must be {self.config.d_model}, got {hidden_states.shape[-1]}"
            )

        q, k, v = self._project_qkv(hidden_states)
        if mask is None:
            mask = _causal_additive_mask(hidden_states.shape[1], hidden_states.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.config.q_head_dim**-0.5,
            mask=mask,
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.config.q_proj_dim,
        )
        return self.out_proj(out)


__all__ = [
    "AttentionConfig",
    "AttentionMode",
    "AttentionRouteInfo",
    "CausalSelfAttention",
]
