"""V7-C06: per-tensor symmetric int8 quantisation + fp16 cast helpers.

quantize_int8(t) → (q_int8, scale_fp32)
dequantize_int8(q, scale) → fp32 tensor

cast_fp16(t) → fp16; uncast_to_fp32 just calls .astype(mx.float32).
"""

from __future__ import annotations

import mlx.core as mx


def quantize_int8(t: mx.array) -> tuple[mx.array, float]:
    """Symmetric int8: scale = max(abs(t)) / 127, q = round(t / scale)."""
    if t.size == 0:
        return mx.zeros(t.shape, dtype=mx.int8), 1.0
    absmax = float(mx.max(mx.abs(t.astype(mx.float32))).item())
    scale = max(absmax / 127.0, 1e-9)
    q = mx.round(t.astype(mx.float32) / scale)
    q = mx.clip(q, -127, 127).astype(mx.int8)
    return q, float(scale)


def dequantize_int8(q: mx.array, scale: float) -> mx.array:
    return q.astype(mx.float32) * scale


def cast_fp16(t: mx.array) -> mx.array:
    return t.astype(mx.float16)


def uncast_to_fp32(t: mx.array) -> mx.array:
    return t.astype(mx.float32)


__all__ = [
    "quantize_int8", "dequantize_int8",
    "cast_fp16", "uncast_to_fp32",
]
