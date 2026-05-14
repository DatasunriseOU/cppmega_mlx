"""Inference-only quantization helpers for the Mac-local MLX path."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import mlx.nn as nn

from cppmega_mlx._mlx_lm_imports import KVCache, QuantizedKVCache
_SUPPORTED_BITS = frozenset({4, 8})
_SUPPORTED_GROUP_SIZES = frozenset({32, 64, 128})
_SUPPORTED_MODES = frozenset({"affine"})
_DEFAULT_MODULE_QUANTIZATION_SKIP_NAMES = frozenset(
    {"embed", "embedding", "embeddings", "embed_tokens", "lm_head"}
)

ModuleSkipPredicate = Callable[[str, nn.Module], bool]


@dataclass(frozen=True)
class InferenceQuantizationConfig:
    """Default q4/q4-KV inference policy.

    This config only describes the local inference helper path. It does not
    imply training quantization, paged serving, long-context quality closure, or
    post-training PPL acceptance.
    """

    bits: int = 4
    group_size: int = 64
    mode: str = "affine"
    kv_bits: int = 4
    kv_group_size: int = 64
    quantized_kv_start: int = 256

    def __post_init__(self) -> None:
        _validate_quant_args(bits=self.bits, group_size=self.group_size, mode=self.mode)
        _validate_quant_args(
            bits=self.kv_bits,
            group_size=self.kv_group_size,
            mode=self.mode,
        )
        if self.quantized_kv_start < 0:
            raise ValueError("quantized_kv_start must be non-negative")


def quantize_linear_for_inference(
    linear: nn.Linear,
    *,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> nn.QuantizedLinear:
    """Convert an ``nn.Linear`` layer to ``nn.QuantizedLinear`` for inference."""

    _validate_quant_args(bits=bits, group_size=group_size, mode=mode)
    if not isinstance(linear, nn.Linear):
        raise TypeError("quantize_linear_for_inference expects an mlx.nn.Linear layer")

    input_dims = int(linear.weight.shape[-1])
    if input_dims % group_size != 0:
        raise ValueError("linear input dimension must be divisible by group_size")

    return nn.QuantizedLinear.from_linear(
        linear,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def quantize_module_for_inference(
    module: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
    skip_module_names: Iterable[str] = _DEFAULT_MODULE_QUANTIZATION_SKIP_NAMES,
    skip_predicate: ModuleSkipPredicate | None = None,
) -> nn.Module:
    """Replace eligible child ``nn.Linear`` layers with ``nn.QuantizedLinear``.

    The conversion is intentionally narrow for Mac-local inference: embeddings
    are skipped by type, common embedding/output-head names are skipped by
    default, and linears whose input dimension is incompatible with the group
    size are left unchanged.
    """

    _validate_quant_args(bits=bits, group_size=group_size, mode=mode)
    if not isinstance(module, nn.Module):
        raise TypeError("quantize_module_for_inference expects an mlx.nn.Module")

    skip_names = frozenset(skip_module_names)
    if _should_skip_module("", module, skip_names, skip_predicate):
        return module
    if isinstance(module, nn.Linear) and _can_quantize_linear(module, group_size):
        return quantize_linear_for_inference(
            module,
            bits=bits,
            group_size=group_size,
            mode=mode,
        )

    def class_predicate(path: str, child: nn.Module) -> bool:
        return (
            isinstance(child, nn.Linear)
            and _can_quantize_linear(child, group_size)
            and not _should_skip_module(path, child, skip_names, skip_predicate)
        )

    nn.quantize(
        module,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=class_predicate,
    )
    return module


def make_quantized_kv_cache(
    *,
    bits: int = 4,
    group_size: int = 64,
) -> QuantizedKVCache:
    """Create an mlx-lm ``QuantizedKVCache`` with q4 defaults."""

    _validate_quant_args(bits=bits, group_size=group_size, mode="affine")
    return QuantizedKVCache(group_size=group_size, bits=bits)


def quantize_kv_cache(
    cache: KVCache | QuantizedKVCache,
    *,
    bits: int = 4,
    group_size: int = 64,
) -> QuantizedKVCache:
    """Convert an existing mlx-lm ``KVCache`` to ``QuantizedKVCache``."""

    _validate_quant_args(bits=bits, group_size=group_size, mode="affine")
    if isinstance(cache, QuantizedKVCache):
        if cache.bits != bits or cache.group_size != group_size:
            raise ValueError(
                "existing QuantizedKVCache has incompatible bits/group_size"
            )
        return cache
    if not isinstance(cache, KVCache):
        raise TypeError("quantize_kv_cache expects an mlx_lm.models.cache.KVCache")
    return cache.to_quantized(group_size=group_size, bits=bits)


def validate_kv_head_dim(head_dim: int, *, group_size: int = 64) -> None:
    """Fail closed before feeding incompatible head dims to MLX quantized KV."""

    _validate_group_size(group_size)
    if head_dim <= 0:
        raise ValueError("KV head_dim must be positive")
    if head_dim % group_size != 0:
        raise ValueError("KV head_dim must be divisible by group_size")


def should_start_kv_quantization(
    position: int,
    *,
    quantized_kv_start: int = 256,
) -> bool:
    """Return whether a decode position is beyond the KV-q4 start threshold."""

    if position < 0:
        raise ValueError("position must be non-negative")
    if quantized_kv_start < 0:
        raise ValueError("quantized_kv_start must be non-negative")
    return position >= quantized_kv_start


def _validate_quant_args(*, bits: int, group_size: int, mode: str) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError("bits must be one of 4 or 8 for inference quantization")
    _validate_group_size(group_size)
    if mode not in _SUPPORTED_MODES:
        raise ValueError("mode must be 'affine'")


def _validate_group_size(group_size: int) -> None:
    if group_size not in _SUPPORTED_GROUP_SIZES:
        raise ValueError("group_size must be one of 32, 64, or 128")


def _can_quantize_linear(linear: nn.Linear, group_size: int) -> bool:
    return int(linear.weight.shape[-1]) % group_size == 0


def _should_skip_module(
    name: str,
    module: nn.Module,
    skip_names: frozenset[str],
    skip_predicate: ModuleSkipPredicate | None,
) -> bool:
    if isinstance(module, nn.Embedding):
        return True
    if name and any(part in skip_names for part in name.split(".")):
        return True
    if skip_predicate is not None and skip_predicate(name, module):
        return True
    return False


__all__ = [
    "InferenceQuantizationConfig",
    "make_quantized_kv_cache",
    "quantize_module_for_inference",
    "quantize_kv_cache",
    "quantize_linear_for_inference",
    "should_start_kv_quantization",
    "validate_kv_head_dim",
]
