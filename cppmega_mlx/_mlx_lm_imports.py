"""Narrow MLX-LM import boundary for cppmega.

MLX-LM imports its tokenizer stack from package ``__init__``.  On Python 3.13
the transitive ``sentencepiece`` SWIG extension emits deprecation warnings while
its C types are registered.  Keep that third-party warning local to this boundary
instead of leaking it into every cppmega import or test collection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import warnings


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
with suppress_sentencepiece_swig_warnings():
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache, QuantizedKVCache


__all__ = [
    "KVCache",
    "QuantizedKVCache",
    "scaled_dot_product_attention",
    "suppress_sentencepiece_swig_warnings",
]
