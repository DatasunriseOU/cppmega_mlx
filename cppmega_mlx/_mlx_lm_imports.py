"""Narrow MLX-LM import boundary for cppmega.

MLX-LM imports its tokenizer stack from package ``__init__``.  On Python 3.13
the transitive ``sentencepiece`` SWIG extension emits deprecation warnings while
its C types are registered.  Keep that third-party warning local to this boundary
instead of leaking it into every cppmega import or test collection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
import sys
from typing import Any
import warnings

import mlx.core as mx


_SENTENCEPIECE_SWIG_WARNING_MESSAGES = (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
)


def _install_sentencepiece_swig_filters() -> None:
    for message in _SENTENCEPIECE_SWIG_WARNING_MESSAGES:
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=DeprecationWarning,
        )


@contextmanager
def suppress_sentencepiece_swig_warnings() -> Iterator[None]:
    """Suppress only the known Python 3.13 SWIG warnings from sentencepiece."""

    with warnings.catch_warnings():
        _install_sentencepiece_swig_filters()
        yield


_install_sentencepiece_swig_filters()


def _load_mlx_lm_scaled_dot_product_attention() -> object:
    if "tvm" in sys.modules and "mlx_lm.models.base" not in sys.modules:
        raise RuntimeError(
            "MLX-LM quantized attention cannot be imported after TVM/LLVM in "
            "the same process; run inference and TileLang compile in separate "
            "processes or load MLX-LM before TVM"
        )
    with suppress_sentencepiece_swig_warnings():
        from mlx_lm.models.base import scaled_dot_product_attention

    return scaled_dot_product_attention


def scaled_dot_product_attention(
    queries: mx.array,
    keys: mx.array | tuple[mx.array, ...],
    values: mx.array | tuple[mx.array, ...],
    cache: Any,
    scale: float,
    mask: mx.array | None,
    sinks: mx.array | None = None,
) -> mx.array:
    """MLX-LM-compatible SDPA without importing its Torch/Triton stack by default."""

    needs_mlx_lm_quantized_path = hasattr(cache, "bits") or (
        isinstance(keys, tuple)
        and cache is not None
        and cache.__class__.__name__ == "TurboQuantKVCache"
    )
    if needs_mlx_lm_quantized_path:
        sdpa = _load_mlx_lm_scaled_dot_product_attention()
        return sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )


@lru_cache(maxsize=1)
def _load_kv_cache_classes() -> tuple[type[object], type[object]]:
    with suppress_sentencepiece_swig_warnings():
        from mlx_lm.models.cache import KVCache, QuantizedKVCache

    return KVCache, QuantizedKVCache


def __getattr__(name: str) -> object:
    if name == "KVCache":
        return _load_kv_cache_classes()[0]
    if name == "QuantizedKVCache":
        return _load_kv_cache_classes()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KVCache",
    "QuantizedKVCache",
    "scaled_dot_product_attention",
    "suppress_sentencepiece_swig_warnings",
]
