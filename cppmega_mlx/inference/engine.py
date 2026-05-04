"""Contiguous KV-cache helpers for the Mac-local MLX inference path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from cppmega_mlx.inference.quantization import make_quantized_kv_cache

_LayerCache: TypeAlias = KVCache | QuantizedKVCache
_StateTree: TypeAlias = mx.array | tuple["_StateTree", ...]


@dataclass(frozen=True)
class ContiguousKVCacheConfig:
    """Shape contract for a small MLX-LM contiguous KV-cache stack.

    MLX-LM stores cache tensors as ``(batch, kv_heads, sequence, head_dim)``.
    This helper intentionally covers only nanochat Track A semantics: contiguous
    append/trim/prefill. Paged serving and integrated model decode stay separate.
    """

    num_layers: int
    batch_size: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int | None = None
    dtype: mx.Dtype | None = None
    quantized: bool = False
    kv_bits: int = 4
    kv_group_size: int = 64

    def __post_init__(self) -> None:
        _validate_positive_int("num_layers", self.num_layers)
        _validate_positive_int("batch_size", self.batch_size)
        _validate_positive_int("num_kv_heads", self.num_kv_heads)
        _validate_positive_int("head_dim", self.head_dim)
        if self.max_seq_len is not None:
            _validate_positive_int("max_seq_len", self.max_seq_len)
        if self.quantized:
            if self.kv_bits not in (4, 8):
                raise ValueError("kv_bits must be one of 4 or 8")
            if self.kv_group_size not in (32, 64, 128):
                raise ValueError("kv_group_size must be one of 32, 64, or 128")
            if self.head_dim % self.kv_group_size != 0:
                raise ValueError("head_dim must be divisible by kv_group_size")


class ContiguousKVCache:
    """A thin validated wrapper around one MLX-LM KV cache per layer."""

    def __init__(self, config: ContiguousKVCacheConfig) -> None:
        self.config = config
        self.layers: list[_LayerCache] = [
            _make_layer_cache(config) for _ in range(config.num_layers)
        ]

    def update_and_fetch(
        self,
        layer_idx: int,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array] | tuple[tuple[mx.array, ...], tuple[mx.array, ...]]:
        """Append ``keys``/``values`` to one layer and return that layer's cache."""

        layer = self._layer(layer_idx)
        _validate_kv_update(self.config, layer, keys, values)
        return layer.update_and_fetch(keys, values)

    def position(self) -> int:
        """Return the aligned decode position for all cache layers."""

        return kv_cache_position(self)

    def trim(self, num_tokens: int) -> int:
        """Trim up to ``num_tokens`` from all layers, matching MLX-LM semantics."""

        return trim_contiguous_kv_cache(self, num_tokens)

    def rollback(self, num_tokens: int) -> int:
        """Strict rollback used by speculative decode style flows."""

        return rollback_contiguous_kv_cache(self, num_tokens)

    def prefill_from(self, source: "ContiguousKVCache") -> None:
        """Copy a prefilled cache into this empty cache."""

        prefill_contiguous_kv_cache(self, source)

    def _layer(self, layer_idx: int) -> _LayerCache:
        if not isinstance(layer_idx, int):
            raise TypeError("layer_idx must be an int")
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError("layer_idx out of range")
        return self.layers[layer_idx]


def make_contiguous_kv_cache(
    config: ContiguousKVCacheConfig | None = None,
    *,
    num_layers: int | None = None,
    batch_size: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    max_seq_len: int | None = None,
    dtype: mx.Dtype | None = None,
    quantized: bool = False,
    kv_bits: int = 4,
    kv_group_size: int = 64,
) -> ContiguousKVCache:
    """Create a validated contiguous KV-cache stack."""

    if config is not None:
        if any(
            value is not None
            for value in (num_layers, batch_size, num_kv_heads, head_dim)
        ):
            raise ValueError("pass either config or shape kwargs, not both")
        return ContiguousKVCache(config)
    if (
        num_layers is None
        or batch_size is None
        or num_kv_heads is None
        or head_dim is None
    ):
        raise ValueError("num_layers, batch_size, num_kv_heads, and head_dim are required")
    return ContiguousKVCache(
        ContiguousKVCacheConfig(
            num_layers=num_layers,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            quantized=quantized,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
    )


def kv_cache_position(cache: ContiguousKVCache) -> int:
    """Return the aligned offset for all layers, failing closed on drift."""

    _validate_cache(cache)
    offsets = [int(layer.offset) for layer in cache.layers]
    first = offsets[0]
    if any(offset != first for offset in offsets):
        raise RuntimeError("contiguous KV cache layer offsets are not aligned")
    return first


def trim_contiguous_kv_cache(cache: ContiguousKVCache, num_tokens: int) -> int:
    """Trim up to ``num_tokens`` tokens from every layer."""

    _validate_cache(cache)
    _validate_non_negative_int("num_tokens", num_tokens)
    trimmed = [int(layer.trim(num_tokens)) for layer in cache.layers]
    first = trimmed[0]
    if any(count != first for count in trimmed):
        raise RuntimeError("contiguous KV cache layers trimmed different token counts")
    return first


def rollback_contiguous_kv_cache(cache: ContiguousKVCache, num_tokens: int) -> int:
    """Roll back exactly ``num_tokens`` tokens from every layer."""

    _validate_non_negative_int("num_tokens", num_tokens)
    position = kv_cache_position(cache)
    if num_tokens > position:
        raise RuntimeError("contiguous KV cache rollback would go below position 0")
    return trim_contiguous_kv_cache(cache, num_tokens)


def prefill_contiguous_kv_cache(
    destination: ContiguousKVCache,
    source: ContiguousKVCache,
) -> None:
    """Clone ``source`` cache state into empty ``destination`` cache."""

    _validate_cache(destination)
    _validate_cache(source)
    if kv_cache_position(destination) != 0:
        raise RuntimeError("cannot prefill a non-empty contiguous KV cache")
    _validate_prefill_configs(destination.config, source.config)
    source_position = kv_cache_position(source)
    if destination.config.max_seq_len is not None and (
        source_position > destination.config.max_seq_len
    ):
        raise RuntimeError("prefill source exceeds destination max_seq_len")

    for dst_layer, src_layer in zip(destination.layers, source.layers, strict=True):
        if src_layer.empty():
            if source_position != 0:
                raise RuntimeError("source cache has an empty layer at non-zero position")
            continue
        state = src_layer.state
        if state is None:
            raise RuntimeError("source cache layer has no state")
        copied_state = _copy_state_for_batch(
            cast(_StateTree, state),
            destination.config.batch_size,
        )
        dst_layer.state = copied_state
        if isinstance(dst_layer, QuantizedKVCache):
            dst_layer.meta_state = src_layer.meta_state


def _make_layer_cache(config: ContiguousKVCacheConfig) -> _LayerCache:
    if config.quantized:
        return make_quantized_kv_cache(bits=config.kv_bits, group_size=config.kv_group_size)
    return KVCache()


def _validate_cache(cache: ContiguousKVCache) -> None:
    if not isinstance(cache, ContiguousKVCache):
        raise TypeError("expected a ContiguousKVCache")


def _validate_prefill_configs(
    destination: ContiguousKVCacheConfig,
    source: ContiguousKVCacheConfig,
) -> None:
    if destination.num_layers != source.num_layers:
        raise ValueError("prefill source and destination must have the same num_layers")
    if destination.num_kv_heads != source.num_kv_heads:
        raise ValueError("prefill source and destination must have the same num_kv_heads")
    if destination.head_dim != source.head_dim:
        raise ValueError("prefill source and destination must have the same head_dim")
    if destination.quantized != source.quantized:
        raise ValueError("prefill source and destination must have the same quantized mode")
    if destination.kv_bits != source.kv_bits:
        raise ValueError("prefill source and destination must have the same kv_bits")
    if destination.kv_group_size != source.kv_group_size:
        raise ValueError("prefill source and destination must have the same kv_group_size")
    if source.batch_size not in (1, destination.batch_size):
        raise ValueError("prefill source batch_size must be 1 or match destination")


def _validate_kv_update(
    config: ContiguousKVCacheConfig,
    layer: _LayerCache,
    keys: mx.array,
    values: mx.array,
) -> None:
    if not isinstance(keys, mx.array) or not isinstance(values, mx.array):
        raise TypeError("keys and values must be mlx.core.array instances")
    if len(keys.shape) != 4 or len(values.shape) != 4:
        raise ValueError("keys and values must have shape (batch, kv_heads, sequence, head_dim)")
    if keys.shape != values.shape:
        raise ValueError("keys and values must have matching shapes")
    if keys.dtype != values.dtype:
        raise ValueError("keys and values must have matching dtype")
    if config.dtype is not None and keys.dtype != config.dtype:
        raise ValueError("keys and values dtype must match cache config dtype")

    batch, num_kv_heads, sequence, head_dim = (int(dim) for dim in keys.shape)
    if batch != config.batch_size:
        raise ValueError("keys batch dimension must match cache config batch_size")
    if num_kv_heads != config.num_kv_heads:
        raise ValueError("keys kv_heads dimension must match cache config num_kv_heads")
    _validate_positive_int("sequence", sequence)
    if head_dim != config.head_dim:
        raise ValueError("keys head_dim dimension must match cache config head_dim")
    if config.max_seq_len is not None and layer.offset + sequence > config.max_seq_len:
        raise RuntimeError("contiguous KV cache update would exceed max_seq_len")


def _copy_state_for_batch(state: _StateTree, batch_size: int) -> _StateTree:
    if isinstance(state, mx.array):
        if len(state.shape) == 0:
            raise ValueError("cache state arrays must have a batch dimension")
        if state.shape[0] == batch_size:
            return mx.array(state)
        if state.shape[0] == 1:
            return mx.array(mx.broadcast_to(state, (batch_size, *state.shape[1:])))
        raise ValueError("cache state batch dimension cannot be expanded")
    return tuple(_copy_state_for_batch(item, batch_size) for item in state)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


__all__ = [
    "ContiguousKVCache",
    "ContiguousKVCacheConfig",
    "kv_cache_position",
    "make_contiguous_kv_cache",
    "prefill_contiguous_kv_cache",
    "rollback_contiguous_kv_cache",
    "trim_contiguous_kv_cache",
]
