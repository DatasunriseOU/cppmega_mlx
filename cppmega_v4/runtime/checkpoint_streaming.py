"""V7-C04: streaming safetensors loader for large checkpoints.

safetensors.mlx.load_file mmaps/reads the full blob into one host
buffer. For multi-GB models this doubles peak memory at resume.
This module iterates per-tensor via safe_open, loads, hands to the
caller, and drops the host reference before the next tensor.

Usage:
    for key, tensor in streaming_load(path):
        live_param_tree[key] = tensor
        # implicit drop of `tensor` reference at next iter step.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Iterator, Iterable

import mlx.core as mx


def streaming_load(path: str | pathlib.Path) -> Iterator[
        tuple[str, mx.array]]:
    """Yield (key, mx.array) per tensor in the safetensors file."""
    from safetensors import safe_open
    with safe_open(str(path), framework="mlx") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)


def streaming_load_all(path: str | pathlib.Path,
                        *,
                        progress_cb: Callable[[int, int], None] | None
                        = None) -> dict[str, mx.array]:
    """Eagerly collect all tensors. Same as load_file but goes
    through the per-tensor iterator; useful when callers want a
    progress callback (loaded, total) every 5%.
    """
    from safetensors import safe_open
    out: dict[str, mx.array] = {}
    with safe_open(str(path), framework="mlx") as f:
        keys = list(f.keys())
        total = len(keys)
        last_pct = -1
        for i, k in enumerate(keys):
            out[k] = f.get_tensor(k)
            if progress_cb:
                pct = int((i + 1) / max(1, total) * 100)
                if pct // 5 != last_pct // 5:
                    progress_cb(i + 1, total)
                    last_pct = pct
    return out


def streaming_load_sharded(paths: Iterable[str | pathlib.Path]
                            ) -> Iterator[tuple[str, mx.array]]:
    """V7-C04 + V7-C02 composition: walk multiple shard files,
    yielding (key, tensor) across all of them."""
    for p in paths:
        yield from streaming_load(p)


__all__ = [
    "streaming_load", "streaming_load_all", "streaming_load_sharded",
]
