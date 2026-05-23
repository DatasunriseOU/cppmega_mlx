"""V7-D25 (za1.2): canonical import path for the topk_selector Metal kernel.

Thin re-export of the existing implementation in
cppmega_mlx.nn._tilelang.topk_selector — the audit asked for a stable
``cppmega_mlx.kernels.topk_selector_metal`` module so downstream code
has a path that matches the per-kernel naming used in tests.

The underlying Metal acceleration goes through mlx.core's
``mx.argpartition`` (already a Metal kernel on Apple Silicon) plus a
shape-validation wrapper.
"""

from __future__ import annotations

from cppmega_mlx.nn._tilelang.topk_selector import (
    topk_selector,
    topk_selector_metal,
    topk_selector_reference,
    topk_selector_tilelang,
)


def topk(scores, k, *, starts=None, ends=None):
    """Stable alias matching the audit spec."""
    return topk_selector(scores, k, starts=starts, ends=ends)


__all__ = [
    "topk",
    "topk_selector",
    "topk_selector_metal",
    "topk_selector_reference",
    "topk_selector_tilelang",
]
