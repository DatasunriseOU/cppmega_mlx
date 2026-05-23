"""V7-D31 (za1.11): MLX port of compute_dacs_segsum.

The Triton helper from cppmega/triton_kernels computes a segment-wise
cumulative sum of dt * A along the sequence axis, optionally with a
decay factor applied at chunk boundaries. The pure-Python reference
below is bit-exact equivalent for fp32 inputs.

Inputs:
    dt: (B, T, H) fp32 step deltas.
    A:  (B, T, H) fp32 log-decay rates.
    chunk_size: int, segment width.

Output: (B, T, H) fp32 cumulative segment sum.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def compute_dacs_segsum(dt: mx.array, A: mx.array, *,
                          chunk_size: int = 16) -> mx.array:
    """MLX implementation."""
    if dt.shape != A.shape:
        raise ValueError(f"shape mismatch: dt={dt.shape}, A={A.shape}")
    if dt.ndim != 3:
        raise ValueError("expected (B, T, H)")
    B, T, H = (int(dt.shape[0]), int(dt.shape[1]), int(dt.shape[2]))
    dt32 = dt.astype(mx.float32)
    A32 = A.astype(mx.float32)
    prod = dt32 * A32                # (B, T, H)
    # Build chunked cumulative sum: each chunk restarts.
    n_chunks = (T + chunk_size - 1) // chunk_size
    mx.zeros((B, T, H), dtype=mx.float32)
    out_list = []
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(T, start + chunk_size)
        chunk = prod[:, start:end, :]
        # Per-chunk cumsum along axis=1.
        cs = mx.cumsum(chunk, axis=1)
        out_list.append(cs)
    return mx.concatenate(out_list, axis=1)


def compute_dacs_segsum_numpy_ref(dt: np.ndarray, A: np.ndarray, *,
                                     chunk_size: int = 16) -> np.ndarray:
    """Bit-exact numpy reference. Used by the parity test."""
    if dt.shape != A.shape:
        raise ValueError("shape mismatch")
    B, T, H = dt.shape
    out = np.zeros_like(dt, dtype=np.float32)
    prod = (dt.astype(np.float32) * A.astype(np.float32))
    for c in range(0, T, chunk_size):
        end = min(T, c + chunk_size)
        out[:, c:end, :] = np.cumsum(prod[:, c:end, :], axis=1)
    return out


__all__ = ["compute_dacs_segsum", "compute_dacs_segsum_numpy_ref"]
