"""MLX causal attention references for cppmega NAM56R A-layers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import QuantizedKVCache

from cppmega_mlx.inference.engine import ContiguousKVCache

AttentionMode = Literal["mla", "dsa"]
RopeType = Literal["standard", "llama3", "yarn"]


@dataclass(frozen=True)
class AttentionRouteInfo:
    """Runtime marker for cppmega A-layer routing.

    dsa currently uses the same dense causal reference path as mla.
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
    rope_type: RopeType = "standard"
    rope_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position: int = 8192
    rope_scaling_factor: float = 40.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    attn_softcap: float = 0.0
    bias: bool = False
    sliding_window: int | None = None

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
        if self.rope_type not in ("standard", "llama3", "yarn"):
            raise ValueError(
                f"rope_type must be one of 'standard', 'llama3', or 'yarn', "
                f"got {self.rope_type!r}"
            )
        if self.use_rope and head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even when use_rope=True, got {head_dim}")
        if self.rope_factor <= 0:
            raise ValueError(f"rope_factor must be positive, got {self.rope_factor}")
        if self.rope_low_freq_factor <= 0:
            raise ValueError(
                f"rope_low_freq_factor must be positive, got {self.rope_low_freq_factor}"
            )
        if self.rope_high_freq_factor <= self.rope_low_freq_factor:
            raise ValueError(
                "rope_high_freq_factor must be greater than rope_low_freq_factor, "
                f"got {self.rope_high_freq_factor} <= {self.rope_low_freq_factor}"
            )
        if self.rope_original_max_position <= 0:
            raise ValueError(
                "rope_original_max_position must be positive, "
                f"got {self.rope_original_max_position}"
            )
        if self.rope_scaling_factor <= 0:
            raise ValueError(
                f"rope_scaling_factor must be positive, got {self.rope_scaling_factor}"
            )
        if self.rope_beta_fast <= 0 or self.rope_beta_slow <= 0:
            raise ValueError(
                "rope_beta_fast and rope_beta_slow must be positive, "
                f"got {self.rope_beta_fast} and {self.rope_beta_slow}"
            )
        if self.rope_beta_fast <= self.rope_beta_slow:
            raise ValueError(
                "rope_beta_fast must be greater than rope_beta_slow, "
                f"got {self.rope_beta_fast} <= {self.rope_beta_slow}"
            )
        if self.attn_softcap < 0:
            raise ValueError(f"attn_softcap must be non-negative, got {self.attn_softcap}")
        if self.attn_softcap > 0:
            raise NotImplementedError(
                "attn_softcap is not supported by the MLX fast SDPA path yet"
            )
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")

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


def causal_sdpa_mask(
    seq_length: int,
    *,
    sliding_window: int | None = None,
    expand_heads: bool = False,
    query_offset: int = 0,
    key_length: int | None = None,
) -> mx.array:
    """Return a boolean causal mask suitable for MLX fast SDPA.

    ``sliding_window`` counts the current token, so ``sliding_window=2`` allows
    each query to see itself and the immediately previous key. ``query_offset``
    and ``key_length`` cover cached decode where a short query attends over a
    longer contiguous key cache.
    """

    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    if query_offset < 0:
        raise ValueError(f"query_offset must be non-negative, got {query_offset}")
    key_length = seq_length if key_length is None else key_length
    if key_length <= 0:
        raise ValueError(f"key_length must be positive, got {key_length}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")

    query_positions = (query_offset + mx.arange(seq_length))[:, None]
    key_positions = mx.arange(key_length)[None, :]
    mask = query_positions >= key_positions
    if sliding_window is not None:
        mask = mask & (query_positions < key_positions + sliding_window)
    mask = mask.astype(mx.bool_)
    if expand_heads:
        return mask[None, None, :, :]
    return mask


def _validate_attention_sinks(sinks: mx.array | None, num_q_heads: int) -> mx.array | None:
    if sinks is None:
        return None
    if sinks.ndim != 1:
        raise ValueError(f"attention sinks must be 1D, got shape {sinks.shape}")
    if sinks.shape[0] != num_q_heads:
        raise ValueError(
            f"attention sinks must have one value per query head ({num_q_heads}), "
            f"got shape {sinks.shape}"
        )
    return sinks


def yarn_attention_factor(scaling_factor: float) -> float:
    """Return nanochat's YaRN attention-temperature factor."""

    if scaling_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scaling_factor) + 1.0


def _base_inv_freq(head_dim: int, theta: float) -> mx.array:
    channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
    return mx.exp(-(channel_range / float(head_dim)) * math.log(theta))


def _llama3_inv_freq(config: AttentionConfig) -> mx.array:
    inv_freq = _base_inv_freq(config.q_head_dim, config.rope_theta)
    wavelen = (2.0 * math.pi) / inv_freq
    low_freq_wavelen = float(config.rope_original_max_position) / config.rope_low_freq_factor
    high_freq_wavelen = float(config.rope_original_max_position) / config.rope_high_freq_factor
    smooth = (
        float(config.rope_original_max_position) / wavelen - config.rope_low_freq_factor
    ) / (config.rope_high_freq_factor - config.rope_low_freq_factor)
    mid_freq = (1.0 - smooth) * inv_freq / config.rope_factor + smooth * inv_freq
    return mx.where(
        wavelen < high_freq_wavelen,
        inv_freq,
        mx.where(wavelen > low_freq_wavelen, inv_freq / config.rope_factor, mid_freq),
    )


def _yarn_correction_dim(
    num_rotations: float,
    head_dim: int,
    theta: float,
    original_max_position: int,
) -> float:
    return (
        head_dim
        * math.log(original_max_position / (num_rotations * 2.0 * math.pi))
        / (2.0 * math.log(theta))
    )


def _yarn_linear_ramp(low: float, high: float, dim: int) -> mx.array:
    if low == high:
        high = low + 0.001
    ramp = (mx.arange(dim, dtype=mx.float32) - low) / (high - low)
    return mx.clip(ramp, 0.0, 1.0)


def _yarn_inv_freq(config: AttentionConfig) -> mx.array:
    half_dim = config.q_head_dim // 2
    inv_freq = _base_inv_freq(config.q_head_dim, config.rope_theta)
    low = _yarn_correction_dim(
        config.rope_beta_fast,
        config.q_head_dim,
        config.rope_theta,
        config.rope_original_max_position,
    )
    high = _yarn_correction_dim(
        config.rope_beta_slow,
        config.q_head_dim,
        config.rope_theta,
        config.rope_original_max_position,
    )
    low = max(math.floor(low), 0)
    high = min(math.ceil(high), half_dim - 1)
    smooth = _yarn_linear_ramp(low, high, half_dim)
    return inv_freq / config.rope_scaling_factor * (1.0 - smooth) + inv_freq * smooth


def rotary_inv_freq(config: AttentionConfig) -> mx.array:
    """Return nanochat-compatible inverse frequencies for the configured RoPE."""

    if config.rope_type == "standard":
        return _base_inv_freq(config.q_head_dim, config.rope_theta)
    if config.rope_type == "llama3":
        return _llama3_inv_freq(config)
    if config.rope_type == "yarn":
        return _yarn_inv_freq(config)
    raise ValueError(f"unsupported rope_type {config.rope_type!r}")


def precompute_rotary_embeddings(
    seq_len: int,
    head_dim: int,
    *,
    theta: float = 10000.0,
    rope_type: RopeType = "standard",
    offset: int = 0,
    factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_position: int = 8192,
    scaling_factor: float = 40.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Precompute split-half RoPE tables for tensors shaped ``(B,H,S,D)``."""

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    config = AttentionConfig(
        d_model=head_dim,
        num_q_heads=1,
        head_dim=head_dim,
        rope_theta=theta,
        rope_type=rope_type,
        rope_factor=factor,
        rope_low_freq_factor=low_freq_factor,
        rope_high_freq_factor=high_freq_factor,
        rope_original_max_position=original_max_position,
        rope_scaling_factor=scaling_factor,
        rope_beta_fast=beta_fast,
        rope_beta_slow=beta_slow,
    )
    inv_freq = rotary_inv_freq(config)
    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    freqs = mx.outer(positions, inv_freq)
    return mx.cos(freqs)[None, None, :, :], mx.sin(freqs)[None, None, :, :]


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply nanochat split-half RoPE to ``x`` shaped ``(B,H,S,D)``."""

    if x.ndim != 4:
        raise ValueError(f"expected a 4-D attention tensor, got shape {x.shape}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")
    half = x.shape[-1] // 2
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


class CausalSelfAttention(nn.Module):
    """Correctness-first MLX causal self-attention for cppmega A-layers.

    The module uses MLX fast SDPA with tensors in (B, heads, S, D) form.
    mode='dsa' intentionally remains a dense causal placeholder/reference:
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
        self.rope_inv_freq = rotary_inv_freq(config) if config.use_rope else None
        self.rope_attention_factor = (
            yarn_attention_factor(config.rope_scaling_factor)
            if config.use_rope and config.rope_type == "yarn"
            else 1.0
        )
        self.route_info = AttentionRouteInfo(mode=config.mode, backend="mlx.fast.sdpa")

    def _rotary_tables(self, seq_len: int, offset: int) -> tuple[mx.array, mx.array]:
        if self.rope_inv_freq is None:
            raise ValueError("RoPE tables requested when use_rope=False")
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, self.rope_inv_freq)
        return mx.cos(freqs)[None, None, :, :], mx.sin(freqs)[None, None, :, :]

    def _project_qkv(
        self,
        hidden_states: mx.array,
        *,
        rope_offset: int = 0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch, seq, _ = hidden_states.shape
        cfg = self.config
        q = self.q_proj(hidden_states).reshape(batch, seq, cfg.num_q_heads, cfg.q_head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq, cfg.kv_heads, cfg.q_head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq, cfg.kv_heads, cfg.q_head_dim)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        if self.rope_inv_freq is not None:
            cos, sin = self._rotary_tables(seq, rope_offset)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        return q, k, v

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | Literal["causal"] | None = None,
        *,
        sinks: mx.array | None = None,
        kv_cache: ContiguousKVCache | None = None,
        layer_idx: int | None = None,
    ) -> mx.array:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must be shaped (B,S,D), got {hidden_states.shape}")
        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states last dim must be {self.config.d_model}, got {hidden_states.shape[-1]}"
            )

        cache_position = 0
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            if layer_idx < 0 or layer_idx >= len(kv_cache.layers):
                raise IndexError("layer_idx out of range")
            layer_cache = kv_cache.layers[layer_idx]
            if isinstance(layer_cache, QuantizedKVCache):
                raise NotImplementedError(
                    "quantized KV cache is not integrated with MLX SDPA attention yet"
                )
            cache_position = int(layer_cache.offset)

        q, k, v = self._project_qkv(hidden_states, rope_offset=cache_position)
        if kv_cache is not None:
            updated_k, updated_v = kv_cache.update_and_fetch(layer_idx, k, v)
            if not isinstance(updated_k, mx.array) or not isinstance(updated_v, mx.array):
                raise NotImplementedError(
                    "quantized KV cache is not integrated with MLX SDPA attention yet"
                )
            k = updated_k
            v = updated_v

        if mask is None or (isinstance(mask, str) and mask == "causal"):
            mask = causal_sdpa_mask(
                hidden_states.shape[1],
                sliding_window=self.config.sliding_window,
                query_offset=cache_position,
                key_length=k.shape[2],
            )
        sinks = _validate_attention_sinks(sinks, self.config.num_q_heads)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=(self.config.q_head_dim**-0.5) / self.rope_attention_factor,
            mask=mask,
            sinks=sinks,
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
    "RopeType",
    "apply_rotary_emb",
    "causal_sdpa_mask",
    "precompute_rotary_embeddings",
    "rotary_inv_freq",
    "yarn_attention_factor",
]
