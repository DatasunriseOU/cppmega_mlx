"""MLX causal attention references for cppmega NAM56R A-layers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx._mlx_lm_imports import scaled_dot_product_attention
from cppmega_mlx.inference.engine import ContiguousKVCache
from cppmega_mlx.runtime.kernel_policy import KernelPath, record_dispatch, selected_path

AttentionMode = Literal["mla", "dsa"]
RopeType = Literal["standard", "llama3", "yarn"]
SPARSE_MLA_FP8_PRODUCER_OWNER = (
    "cppmega_mlx.nn.attention.CausalSelfAttention.prepare_sparse_mla_fp8"
)
SPARSE_MLA_FP8_PRODUCER_STAGE = "attention_qkv_projection"
SPARSE_MLA_FP8_PREPARED_BUFFER_NAMES = (
    "q_fp8",
    "q_scale",
    "kv_fp8",
    "kv_scale",
)


@dataclass(frozen=True)
class AttentionRouteInfo:
    """Runtime marker for cppmega A-layer routing.

    ``mode='dsa'`` defaults to the dense causal SDPA reference unless
    ``CPPMEGA_KERNEL_PATH__SPARSE_MLA=path_c`` selects the prepared FP8
    Sparse-MLA route.
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
    sparse_topk: int = 16

    def __post_init__(self) -> None:
        if self.mode not in ("mla", "dsa"):
            raise ValueError(f"mode must be 'mla' or 'dsa', got {self.mode!r}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_q_heads <= 0:
            raise ValueError(f"num_q_heads must be positive, got {self.num_q_heads}")
        num_kv_heads = (
            self.num_q_heads if self.num_kv_heads is None else self.num_kv_heads
        )
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if self.num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads {self.num_q_heads} must be divisible by "
                f"num_kv_heads {num_kv_heads}"
            )
        head_dim = (
            self.d_model // self.num_q_heads if self.head_dim is None else self.head_dim
        )
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
            raise ValueError(
                f"head_dim must be even when use_rope=True, got {head_dim}"
            )
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
            raise ValueError(
                f"attn_softcap must be non-negative, got {self.attn_softcap}"
            )
        if self.attn_softcap > 0:
            raise NotImplementedError(
                "attn_softcap is not supported by the MLX fast SDPA path yet"
            )
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be positive, got {self.sliding_window}"
            )
        if self.sparse_topk <= 0:
            raise ValueError(f"sparse_topk must be positive, got {self.sparse_topk}")

    @property
    def kv_heads(self) -> int:
        return self.num_q_heads if self.num_kv_heads is None else self.num_kv_heads

    @property
    def q_head_dim(self) -> int:
        return (
            self.d_model // self.num_q_heads if self.head_dim is None else self.head_dim
        )

    @property
    def q_proj_dim(self) -> int:
        return self.num_q_heads * self.q_head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.kv_heads * self.q_head_dim

    @property
    def is_gqa(self) -> bool:
        return self.kv_heads != self.num_q_heads


@dataclass(frozen=True)
class SparseMLAFp8Prepared:
    """First-class DSA/Sparse-MLA FP8 carrier produced by the attention layer."""

    q_fp8: mx.array
    q_scale: mx.array
    kv_fp8: mx.array
    kv_scale: mx.array
    indices: mx.array
    sm_scale: float
    d_v: int
    q: mx.array | None = None
    kv: mx.array | None = None
    # Historical field name retained for the Path C VJP ABI.  True means the
    # backward can scatter directly into full-window owner buffers and skip the
    # dkv_partial materialization; the sparse indices may come from causal or
    # explicit document masks.
    causal: bool = False
    producer_owner: str = SPARSE_MLA_FP8_PRODUCER_OWNER
    producer_stage: str = SPARSE_MLA_FP8_PRODUCER_STAGE
    prepared_buffer_names: tuple[str, ...] = SPARSE_MLA_FP8_PREPARED_BUFFER_NAMES
    hidden_wrapper_quantization_allowed: bool = False


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


def causal_sparse_indices(
    batch_size: int,
    seq_length: int,
    kv_group: int,
    topk: int,
    *,
    query_offset: int = 0,
    key_length: int | None = None,
) -> mx.array:
    """Return token-level causal sparse indices shaped ``[B, S, G, topk]``."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    if kv_group <= 0:
        raise ValueError(f"kv_group must be positive, got {kv_group}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if query_offset < 0:
        raise ValueError(f"query_offset must be non-negative, got {query_offset}")
    key_length = seq_length if key_length is None else key_length
    if key_length <= 0:
        raise ValueError(f"key_length must be positive, got {key_length}")

    query_positions = (query_offset + mx.arange(seq_length, dtype=mx.int32))[:, None]
    offsets = mx.arange(topk, dtype=mx.int32)[None, :]
    indices = query_positions - offsets
    valid = cast(mx.array, (indices >= 0) & (indices < key_length))
    indices = mx.where(valid, indices, mx.zeros_like(indices) - 1)
    indices = indices.reshape(1, seq_length, 1, topk)
    return mx.broadcast_to(indices, (batch_size, seq_length, kv_group, topk)).astype(
        mx.int32
    )


def sparse_indices_from_attention_mask(
    mask: mx.array,
    *,
    batch_size: int,
    seq_length: int,
    kv_group: int,
    topk: int,
    key_length: int,
) -> mx.array:
    """Convert an explicit attention mask to Path C sparse indices.

    Path C consumes token indices rather than a dense mask.  This keeps
    document-boundary masks in metadata form and selects the newest valid keys
    per query/group without materializing K/V copies.
    """

    if not isinstance(mask, mx.array):
        raise TypeError("attention mask must be an mlx.core.array")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    if kv_group <= 0:
        raise ValueError(f"kv_group must be positive, got {kv_group}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if key_length <= 0:
        raise ValueError(f"key_length must be positive, got {key_length}")

    if mask.ndim == 2:
        valid = mask[None, None, :, :]
    elif mask.ndim == 3:
        valid = mask[:, None, :, :]
    elif mask.ndim == 4:
        valid = mask
    else:
        raise ValueError(
            "attention mask must have shape (S,K), (B,S,K), or (B,H,S,K), "
            f"got {mask.shape}"
        )

    if valid.shape[-2:] != (seq_length, key_length):
        raise ValueError(
            f"attention mask trailing dims must be ({seq_length}, {key_length}), "
            f"got {valid.shape[-2:]}"
        )
    if valid.shape[0] not in (1, batch_size):
        raise ValueError(
            f"attention mask batch dimension must be 1 or {batch_size}, got {valid.shape[0]}"
        )
    if valid.shape[1] not in (1, kv_group):
        raise ValueError(
            f"Path C sparse masks must be shared across heads or have {kv_group} kv groups, "
            f"got head/group dimension {valid.shape[1]}"
        )

    if valid.dtype == mx.bool_:
        valid_mask = valid
    else:
        valid_mask = mx.isfinite(valid.astype(mx.float32))
    valid_mask = mx.broadcast_to(
        valid_mask,
        (batch_size, kv_group, seq_length, key_length),
    )
    valid_mask = mx.transpose(valid_mask, (0, 2, 1, 3))

    key_positions = mx.arange(key_length, dtype=mx.int32)
    scores = mx.where(valid_mask, key_positions, mx.zeros_like(key_positions) - 1)
    ordered = mx.argsort(scores, axis=-1)
    selected = ordered[..., -min(topk, key_length) :].astype(mx.int32)
    selected_scores = mx.take_along_axis(scores, selected, axis=-1)
    return mx.where(selected_scores >= 0, selected, mx.zeros_like(selected) - 1).astype(
        mx.int32
    )


def _to_fp8_with_per_token_scale(x: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize a producer tensor to e4m3 with one fp32 scale per final-dim row."""

    from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
        _to_fp8_with_per_token_scale as _path_c_to_fp8_with_per_token_scale,
    )

    return _path_c_to_fp8_with_per_token_scale(x)


def _validate_attention_sinks(
    sinks: mx.array | None, num_q_heads: int
) -> mx.array | None:
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
    low_freq_wavelen = (
        float(config.rope_original_max_position) / config.rope_low_freq_factor
    )
    high_freq_wavelen = (
        float(config.rope_original_max_position) / config.rope_high_freq_factor
    )
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
    ``mode='dsa'`` can additionally produce first-class FP8 Sparse-MLA buffers
    and call the prepared Path C kernel when the sparse_mla policy selects
    ``path_c``.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.q_proj_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.kv_proj_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.kv_proj_dim, bias=config.bias)
        self.sparse_kv_proj = (
            nn.Linear(config.d_model, config.kv_proj_dim, bias=config.bias)
            if config.mode == "dsa"
            else None
        )
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
        q = self.q_proj(hidden_states).reshape(
            batch, seq, cfg.num_q_heads, cfg.q_head_dim
        )
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

    def _project_qkv_bshd(
        self,
        hidden_states: mx.array,
        *,
        rope_offset: int = 0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        q, k, v = self._project_qkv(hidden_states, rope_offset=rope_offset)
        return (
            mx.transpose(q, (0, 2, 1, 3)),
            mx.transpose(k, (0, 2, 1, 3)),
            mx.transpose(v, (0, 2, 1, 3)),
        )

    def _project_sparse_mla_qkv_bshd(
        self,
        hidden_states: mx.array,
        *,
        rope_offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        batch, seq, _ = hidden_states.shape
        cfg = self.config
        q = self.q_proj(hidden_states).reshape(
            batch, seq, cfg.num_q_heads, cfg.q_head_dim
        )
        if self.sparse_kv_proj is None:
            kv = self.k_proj(hidden_states).reshape(
                batch, seq, cfg.kv_heads, cfg.q_head_dim
            )
        else:
            kv = self.sparse_kv_proj(hidden_states).reshape(
                batch,
                seq,
                cfg.kv_heads,
                cfg.q_head_dim,
            )
        q = mx.transpose(q, (0, 2, 1, 3))
        kv = mx.transpose(kv, (0, 2, 1, 3))
        if self.rope_inv_freq is not None:
            cos, sin = self._rotary_tables(seq, rope_offset)
            q = apply_rotary_emb(q, cos, sin)
            kv = apply_rotary_emb(kv, cos, sin)
        return mx.transpose(q, (0, 2, 1, 3)), mx.transpose(kv, (0, 2, 1, 3))

    def prepare_sparse_mla_fp8(
        self,
        hidden_states: mx.array,
        *,
        rope_offset: int = 0,
        key_length: int | None = None,
        mask: mx.array | Literal["causal"] | None = None,
        kv_cache: ContiguousKVCache | None = None,
        layer_idx: int | None = None,
    ) -> SparseMLAFp8Prepared:
        """Produce first-class FP8 buffers consumed by Sparse-MLA Path C."""

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be shaped (B,S,D), got {hidden_states.shape}"
            )
        cfg = self.config
        batch, seq, _ = hidden_states.shape

        q, kv = self._project_sparse_mla_qkv_bshd(
            hidden_states, rope_offset=rope_offset
        )
        q_fp8, q_scale = _to_fp8_with_per_token_scale(q)
        kv_fp8, kv_scale = _to_fp8_with_per_token_scale(kv)
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            kv_fp8, kv_scale = kv_cache.update_and_fetch_sparse_fp8(
                layer_idx,
                kv_fp8,
                kv_scale,
            )
            key_length = int(kv_fp8.shape[1])
        sparse_window = key_length if key_length is not None else seq
        if cfg.sliding_window is not None:
            sparse_window = min(sparse_window, cfg.sliding_window)
        effective_topk = min(cfg.sparse_topk, sparse_window)
        is_causal_sparse = mask is None or (isinstance(mask, str) and mask == "causal")
        if is_causal_sparse:
            indices = causal_sparse_indices(
                batch,
                seq,
                cfg.kv_heads,
                effective_topk,
                query_offset=rope_offset,
                key_length=key_length,
            )
        elif isinstance(mask, str):
            raise ValueError(f"unsupported attention mask sentinel {mask!r}")
        else:
            indices = sparse_indices_from_attention_mask(
                mask,
                batch_size=batch,
                seq_length=seq,
                kv_group=cfg.kv_heads,
                topk=effective_topk,
                key_length=key_length if key_length is not None else seq,
            )
        full_window_owner_buffers = (
            kv_cache is None
            and rope_offset == 0
            and (key_length is None or key_length == seq)
        )
        return SparseMLAFp8Prepared(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            kv_scale=kv_scale,
            indices=indices,
            sm_scale=(cfg.q_head_dim**-0.5) / self.rope_attention_factor,
            d_v=cfg.q_head_dim,
            q=q,
            kv=kv,
            causal=full_window_owner_buffers,
        )

    def _apply_sparse_mla_fp8_path_c_prepared(
        self,
        prepared: SparseMLAFp8Prepared,
        *,
        output_shape: tuple[int, int, int],
        sinks: mx.array | None = None,
    ) -> mx.array:
        from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
            sparse_mla_fp8_path_c_apply,
            sparse_mla_fp8_path_c_apply_prepared_float,
        )

        sinks = _validate_attention_sinks(sinks, self.config.num_q_heads)
        if prepared.hidden_wrapper_quantization_allowed:
            raise RuntimeError(
                "Sparse-MLA FP8 Path C requires producer-owned prepared buffers; "
                "wrapper quantization is not allowed"
            )
        if prepared.q is not None and prepared.kv is not None:
            out = sparse_mla_fp8_path_c_apply_prepared_float(
                prepared.q,
                prepared.kv,
                prepared.q_fp8,
                prepared.q_scale,
                prepared.kv_fp8,
                prepared.kv_scale,
                prepared.indices,
                sm_scale=prepared.sm_scale,
                d_v=prepared.d_v,
                sinks=sinks,
                force_path_c=True,
                causal=prepared.causal,
                output_dtype=prepared.q.dtype,
            )
        else:
            out = sparse_mla_fp8_path_c_apply(
                prepared.q_fp8,
                prepared.q_scale,
                prepared.kv_fp8,
                prepared.kv_scale,
                prepared.indices,
                sm_scale=prepared.sm_scale,
                d_v=prepared.d_v,
                sinks=sinks,
                force_path_c=True,
            )
        if out is None:
            raise RuntimeError(
                "sparse_mla_fp8_path_c_apply returned None under forced Path C"
            )
        record_dispatch(
            "sparse_mla", KernelPath.PATH_C, "tilelang_fp8_prepared_path_c_fwd"
        )
        # out_proj consumes the Path C output directly; do not stage a
        # full-tensor dtype cast at this boundary.
        out = out.reshape(output_shape)
        return self.out_proj(out)

    def _prepare_sparse_mla_float_baseline(
        self,
        hidden_states: mx.array,
        *,
        rope_offset: int = 0,
        key_length: int | None = None,
        mask: mx.array | Literal["causal"] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, float, int]:
        cfg = self.config
        batch, seq, _ = hidden_states.shape
        q, kv = self._project_sparse_mla_qkv_bshd(
            hidden_states, rope_offset=rope_offset
        )
        sparse_window = key_length if key_length is not None else seq
        if cfg.sliding_window is not None:
            sparse_window = min(sparse_window, cfg.sliding_window)
        effective_topk = min(cfg.sparse_topk, sparse_window)
        is_causal_sparse = mask is None or (isinstance(mask, str) and mask == "causal")
        if is_causal_sparse:
            indices = causal_sparse_indices(
                batch,
                seq,
                cfg.kv_heads,
                effective_topk,
                query_offset=rope_offset,
                key_length=key_length,
            )
        elif isinstance(mask, str):
            raise ValueError(f"unsupported attention mask sentinel {mask!r}")
        else:
            indices = sparse_indices_from_attention_mask(
                mask,
                batch_size=batch,
                seq_length=seq,
                kv_group=cfg.kv_heads,
                topk=effective_topk,
                key_length=key_length if key_length is not None else seq,
            )
        return (
            q,
            kv,
            indices,
            (cfg.q_head_dim**-0.5) / self.rope_attention_factor,
            cfg.q_head_dim,
        )

    def _apply_sparse_mla_fp8_path_b_baseline(
        self,
        q: mx.array,
        kv: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int,
        output_shape: tuple[int, int, int],
        sinks: mx.array | None = None,
    ) -> mx.array:
        from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import sparse_mla_fp8_apply

        if sinks is not None:
            raise RuntimeError("Sparse-MLA FP8 Path B baseline does not support sinks")
        out = sparse_mla_fp8_apply(
            q,
            kv,
            indices,
            sm_scale=sm_scale,
            d_v=d_v,
            force_metal=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        record_dispatch(
            "sparse_mla", KernelPath.PATH_B, "sparse_mla_fp8_reference_path_b"
        )
        out = out.reshape(output_shape)
        return self.out_proj(out)

    def _use_sparse_mla_fp8_path_c(
        self,
        mask: mx.array | Literal["causal"] | None,
        *,
        sinks: mx.array | None,
        kv_cache: ContiguousKVCache | None,
    ) -> bool:
        del sinks, kv_cache
        return (
            self.config.mode == "dsa"
            and selected_path("sparse_mla") is KernelPath.PATH_C
            and (not isinstance(mask, str) or mask == "causal")
        )

    def _use_sparse_mla_fp8_path_b_baseline(
        self,
        mask: mx.array | Literal["causal"] | None,
        *,
        sinks: mx.array | None,
        kv_cache: ContiguousKVCache | None,
    ) -> bool:
        del sinks
        return (
            self.config.mode == "dsa"
            and kv_cache is None
            and selected_path("sparse_mla") is KernelPath.PATH_B
            and (not isinstance(mask, str) or mask == "causal")
        )

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
            raise ValueError(
                f"hidden_states must be shaped (B,S,D), got {hidden_states.shape}"
            )
        if hidden_states.shape[-1] != self.config.d_model:
            raise ValueError(
                f"hidden_states last dim must be {self.config.d_model}, got {hidden_states.shape[-1]}"
            )
        cache_position = 0
        cache_layer_idx: int | None = None
        if kv_cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            if layer_idx < 0 or layer_idx >= len(kv_cache.layers):
                raise IndexError("layer_idx out of range")
            cache_layer_idx = layer_idx
            cache_position = kv_cache.layer_position(cache_layer_idx)

        if self._use_sparse_mla_fp8_path_b_baseline(
            mask, sinks=sinks, kv_cache=kv_cache
        ):
            q, kv, indices, sm_scale, d_v = self._prepare_sparse_mla_float_baseline(
                hidden_states,
                rope_offset=cache_position,
                key_length=None,
                mask=mask,
            )
            return self._apply_sparse_mla_fp8_path_b_baseline(
                q,
                kv,
                indices,
                sm_scale=sm_scale,
                d_v=d_v,
                output_shape=(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self.config.q_proj_dim,
                ),
                sinks=sinks,
            )

        if self._use_sparse_mla_fp8_path_c(mask, sinks=sinks, kv_cache=kv_cache):
            prepared = self.prepare_sparse_mla_fp8(
                hidden_states,
                rope_offset=cache_position,
                key_length=None,
                mask=mask,
                kv_cache=kv_cache,
                layer_idx=cache_layer_idx,
            )
            return self._apply_sparse_mla_fp8_path_c_prepared(
                prepared,
                output_shape=(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self.config.q_proj_dim,
                ),
                sinks=sinks,
            )

        q, k, v = self._project_qkv(hidden_states, rope_offset=cache_position)
        if kv_cache is not None:
            if cache_layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            updated_k, updated_v = kv_cache.update_and_fetch(cache_layer_idx, k, v)
            k = updated_k
            v = updated_v

        key_length = k[0].shape[-2] if isinstance(k, tuple) else k.shape[2]
        if mask is None or (isinstance(mask, str) and mask == "causal"):
            mask = causal_sdpa_mask(
                hidden_states.shape[1],
                sliding_window=self.config.sliding_window,
                query_offset=cache_position,
                key_length=key_length,
            )
        sinks = _validate_attention_sinks(sinks, self.config.num_q_heads)
        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=kv_cache.layers[cache_layer_idx]
            if cache_layer_idx is not None
            else None,
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
    "SparseMLAFp8Prepared",
    "SPARSE_MLA_FP8_PREPARED_BUFFER_NAMES",
    "SPARSE_MLA_FP8_PRODUCER_OWNER",
    "SPARSE_MLA_FP8_PRODUCER_STAGE",
    "apply_rotary_emb",
    "causal_sparse_indices",
    "causal_sdpa_mask",
    "precompute_rotary_embeddings",
    "rotary_inv_freq",
    "sparse_indices_from_attention_mask",
    "yarn_attention_factor",
]
