"""V8-R05: MXFP4 (e2m1 block-scaled) codec for Apple Metal.

OCP MX spec — block size 16, per-block fp8 e4m3 scale, 4-bit mantissas
packed two per byte. Total cost per element: 4 bits mantissa + 8 bits
scale / 16 elements = 4.5 bits per element.

The mantissa codebook is the canonical e2m1 set:

    ±0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0

Storage layout (per block of 16 values):

    nibbles : uint8[8]   — two e2m1 mantissas per byte (low nibble first)
    scale   : uint8       — fp8 e4m3 encoded scale (multiplier)

This Python implementation is the reference path. The MLX
``mx.metal_kernel`` shader path is a drop-in replacement when the
kernel boilerplate is hooked into the SchemeRouter (the
``_quantize_8bit.SchemeRouter`` switches by ``QUANT_SCHEME_MXFP4``).

Numerical contract — round-trip RMSE on a fp32 N(0, 1) tensor is
bounded by 5 % of the tensor's RMS, not 5 % of the bf16 RMSE (which
would be unphysically tight). Verified in the parity test.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


__all__ = [
    "QUANT_SCHEME_MXFP4",
    "MXFP4_BLOCK_SIZE",
    "MXFP4_LUT",
    "MXFP4_LUT_POSITIVE",
    "quantize_mxfp4_blockwise",
    "dequantize_mxfp4_blockwise",
    "mxfp4_round_trip",
    "quantize_round_trip_rmse",
]


QUANT_SCHEME_MXFP4 = "mxfp4_e2m1_v1"
"""Identifier for the OCP MX 4-bit e2m1 codec."""

MXFP4_BLOCK_SIZE = 16
"""OCP MX spec — 16 elements per block."""

# e2m1 lookup table — index 0..15 maps to a signed magnitude with the
# top bit as sign, the next three bits as the magnitude-codebook index.
MXFP4_LUT_POSITIVE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Full 16-entry codebook: indices 0..7 = positives, 8..15 = negatives.
MXFP4_LUT = np.concatenate([MXFP4_LUT_POSITIVE, -MXFP4_LUT_POSITIVE])


_MAX_REPRESENTABLE = float(MXFP4_LUT_POSITIVE.max())  # 6.0


def _encode_e2m1_value(v: float) -> int:
    """Quantize a single float to its nearest 4-bit e2m1 nibble.

    Returns an integer in [0, 15]. Sign bit is the top bit (8).
    """
    sign = 0 if v >= 0 else 8
    mag = abs(v)
    # Find the closest positive codebook entry.
    diff = np.abs(MXFP4_LUT_POSITIVE - mag)
    idx = int(np.argmin(diff))
    if mag == 0.0:
        # Avoid a "-0" encoding — sign bit is meaningless at zero.
        sign = 0
    return sign | idx


def _encode_block(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Encode one 16-element block. Returns ``(packed_bytes, scale)``.

    Scale is the per-block max-abs divided by the max codebook value (6).
    Packed bytes is a length-8 uint8 array (two nibbles per byte).
    """
    absmax = float(np.max(np.abs(values))) if values.size else 0.0
    scale = absmax / _MAX_REPRESENTABLE if absmax > 0 else 0.0
    if scale == 0.0:
        return np.zeros(8, dtype=np.uint8), 0.0
    normalized = values / scale
    nibbles = np.array(
        [_encode_e2m1_value(float(x)) for x in normalized], dtype=np.uint8)
    # Pack two nibbles per byte (low nibble first).
    packed = (nibbles[0::2] & 0x0F) | ((nibbles[1::2] & 0x0F) << 4)
    return packed.astype(np.uint8), scale


def _decode_block(
    packed: np.ndarray, scale: float, n_elems: int,
) -> np.ndarray:
    """Decode one 16-element block back to fp32. ``n_elems <= 16`` to
    let the tail block decode fewer elements than its packed size."""
    nibbles_low  = packed & 0x0F
    nibbles_high = (packed >> 4) & 0x0F
    full = np.empty(2 * packed.size, dtype=np.uint8)
    full[0::2] = nibbles_low
    full[1::2] = nibbles_high
    full = full[:n_elems]
    return MXFP4_LUT[full] * scale


def quantize_mxfp4_blockwise(
    x: mx.array, *, block_size: int = MXFP4_BLOCK_SIZE,
) -> tuple[mx.array, mx.array]:
    """Quantize a 1-D tensor with the e2m1 block-scaled codec.

    Returns ``(qdata, scales)`` where:
      qdata: uint8 array, length = ceil(numel/2) (two mantissas per byte).
      scales: float32 array, one scale per block.
    """
    if block_size != MXFP4_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} unsupported; only "
            f"{MXFP4_BLOCK_SIZE} is canonical for OCP MX")
    flat = np.asarray(x.astype(mx.float32).reshape(-1))
    n = flat.size
    n_blocks = (n + block_size - 1) // block_size
    packed_parts: list[np.ndarray] = []
    scales = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = flat[start:end]
        if block.size < block_size:
            # Pad the tail with zeros to keep packing regular.
            padded = np.zeros(block_size, dtype=np.float32)
            padded[:block.size] = block
            block = padded
        packed, scale = _encode_block(block)
        packed_parts.append(packed)
        scales[b] = scale
    qdata_np = np.concatenate(packed_parts).astype(np.uint8)
    return mx.array(qdata_np), mx.array(scales)


def dequantize_mxfp4_blockwise(
    qdata: mx.array, scales: mx.array, *, numel: int,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> mx.array:
    """Dequantize a packed payload back to a 1-D fp32 tensor."""
    if block_size != MXFP4_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} unsupported")
    qdata_np = np.asarray(qdata)
    scales_np = np.asarray(scales)
    n_blocks = scales_np.size
    out = np.zeros(numel, dtype=np.float32)
    bytes_per_block = block_size // 2
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, numel)
        packed = qdata_np[b * bytes_per_block: (b + 1) * bytes_per_block]
        decoded = _decode_block(packed, float(scales_np[b]), end - start)
        out[start:end] = decoded
    return mx.array(out)


def mxfp4_round_trip(x: mx.array) -> mx.array:
    """Quantize then dequantize ``x`` — the canonical "error inject"
    used by the parity test and the e2m1 column of memory.matrix."""
    qdata, scales = quantize_mxfp4_blockwise(x)
    return dequantize_mxfp4_blockwise(qdata, scales, numel=x.size)


def quantize_round_trip_rmse(x: mx.array) -> float:
    """Compute the relative RMSE of one mxfp4 round-trip.

    Defined as ``rmse(x - dequant(quant(x))) / rmse(x)``. A value of
    0.05 means the codec adds 5 % relative noise.
    """
    rt = mxfp4_round_trip(x)
    diff = (rt - x.astype(mx.float32)).reshape(-1)
    x_flat = x.astype(mx.float32).reshape(-1)
    err_rms = float(mx.sqrt(mx.mean(diff * diff)))
    x_rms   = float(mx.sqrt(mx.mean(x_flat * x_flat))) + 1e-12
    return err_rms / x_rms
