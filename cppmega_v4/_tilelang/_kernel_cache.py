"""Shared Metal kernel cache for v4 Path B implementations.

Mirrors ``mlx-recurrence/mlx_recurrence/_utils.py`` (MIT, D-CSIL): one
compiled ``mx.fast.metal_kernel`` per shape config keyed by name.
"""

from __future__ import annotations

import mlx.core as mx

_KERNEL_CACHE: dict[str, object] = {}


def get_or_build_kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
):
    if name not in _KERNEL_CACHE:
        _KERNEL_CACHE[name] = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
        )
    return _KERNEL_CACHE[name]


__all__ = ["get_or_build_kernel"]
