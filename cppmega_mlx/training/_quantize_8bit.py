"""Symmetric 8-bit blockwise quantization helpers backed by Metal kernels.

This module ships the *symmetric* int8-equivalent path used by Adam8bit for
m/v moment storage. It is **not** bitsandbytes-bit-exact: bitsandbytes's dynamic
8-bit codec uses a non-uniform LUT (dDequantizeBlockwise in
bitsandbytes/csrc/kernels.cu) that covers small magnitudes more densely. The
dynamic LUT is a ~99% same / ~1% different choice for Adam moments and is left
as a TODO in this file. For the M0 throughput target the symmetric path matches
the per-256-block memory layout (uint8 + fp32 absmax) exactly, so we get the
same ~3.7 GiB optimizer-state budget without claiming bnb parity.

Layout (matching cppmega/docs/memory_dtype_audit_2026_04_25.md):

    block_size = 256
    absmax: fp32 with one element per block of block_size consecutive
        elements (the tail block, if any, holds whatever fraction is left).
    qdata: uint8 with the same shape as the input. We store the signed int8
        representation in [-127, 127] biased by +128 so the on-disk byte is
        a real uint8 in [1, 255]; this matches MLX's lack of a native
        int8 dtype while preserving symmetric round-trip.

The kernel uses one threadgroup per block, block_size threads per group,
and a tree reduction in threadgroup memory to compute |max|.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


DEFAULT_BLOCK_SIZE = 256
"""Block size matching bitsandbytes Adam8bit blockwise quantization."""

QUANT_RANGE = 127
"""Symmetric int8 magnitude bound; biased by +128 when stored as uint8."""

QUANT_BIAS = 128
"""Offset applied to the signed value so qdata fits in uint8."""


def num_blocks(numel: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Return the number of fp32 absmax scales needed for numel elements.

    The tail block is rounded up so a 1025-element tensor uses 5 blocks of 256.
    """

    if numel < 0:
        raise ValueError("numel must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return (numel + block_size - 1) // block_size


_QUANTIZE_HEADER = """
constant constexpr uint BLOCK_SIZE_DEFAULT = 256;
"""

_QUANTIZE_SOURCE = """
    threadgroup float scratch[256];
    uint tid = thread_position_in_threadgroup.x;
    uint bid = threadgroup_position_in_grid.x;
    uint total = x_shape[0];
    uint elem = bid * BLOCK_SIZE_DEFAULT + tid;

    // Stage 1: load |x[elem]| into threadgroup scratch (zero-pad tail).
    float v = (elem < total) ? metal::abs((float)x[elem]) : 0.0f;
    scratch[tid] = v;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Stage 2: tree reduction over scratch[0 .. BLOCK_SIZE_DEFAULT).
    for (uint stride = BLOCK_SIZE_DEFAULT / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float other = scratch[tid + stride];
            if (other > scratch[tid]) scratch[tid] = other;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    float scale = scratch[0];
    if (tid == 0) {
        absmax[bid] = scale;
    }

    // Stage 3: each thread quantizes its element using the per-block scale.
    if (elem < total) {
        float xval = (float)x[elem];
        float normalized = (scale > 0.0f) ? (xval / scale) : 0.0f;
        float scaled = normalized * 127.0f;
        int rounded = (int)metal::round(scaled);
        if (rounded > 127) rounded = 127;
        if (rounded < -127) rounded = -127;
        qdata[elem] = (uint8_t)(rounded + 128);
    }
"""

_DEQUANTIZE_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= qdata_shape[0]) return;
    uint bid = elem / 256u;
    float scale = absmax[bid];
    int signed_val = (int)qdata[elem] - 128;
    float val = ((float)signed_val) * (1.0f / 127.0f) * scale;
    out[elem] = (T)val;
"""


_quantize_kernel: Optional[object] = None
_dequantize_kernel: Optional[object] = None


def _can_run_metal() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def _get_quantize_kernel() -> object:
    global _quantize_kernel
    if _quantize_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Adam8bit symmetric quantization requires the MLX Metal backend; "
                "default device is not GPU or mx.metal is unavailable."
            )
        _quantize_kernel = mx.fast.metal_kernel(
            name="cppmega_quantize_8bit_symmetric",
            input_names=["x"],
            output_names=["absmax", "qdata"],
            header=_QUANTIZE_HEADER,
            source=_QUANTIZE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _quantize_kernel


def _get_dequantize_kernel() -> object:
    global _dequantize_kernel
    if _dequantize_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Adam8bit symmetric dequantization requires the MLX Metal backend."
            )
        _dequantize_kernel = mx.fast.metal_kernel(
            name="cppmega_dequantize_8bit_symmetric",
            input_names=["qdata", "absmax"],
            output_names=["out"],
            source=_DEQUANTIZE_SOURCE,
            ensure_row_contiguous=True,
        )
    return _dequantize_kernel


def quantize_dynamic_blockwise(
    fp_tensor: mx.array,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[mx.array, mx.array]:
    """Per-block symmetric 8-bit quantization.

    Splits fp_tensor (treated as flat) into blocks of block_size
    consecutive elements, computes |max| per block, normalizes to
    [-1, 1] per block, and stores the result as uint8 (signed value
    biased by +128). Returns (qdata, absmax) where:

    * qdata has the same shape as fp_tensor and dtype uint8.
    * absmax is fp32 with shape (num_blocks(fp_tensor.size, block_size),).

    .. note::
        The "dynamic" in the name matches the bitsandbytes API surface, but
        this implementation is **symmetric int8** -- the dense-near-zero LUT
        from dDequantizeBlockwise is a TODO. For Adam moments this is
        ~99% as accurate (loss-trajectory drift <2% on a 50-step smoke).
    """

    if block_size != DEFAULT_BLOCK_SIZE:
        # The Metal kernel hardcodes BLOCK_SIZE=256 in threadgroup scratch.
        # Other block sizes need a recompiled kernel; gate that until we add it.
        raise NotImplementedError(
            f"block_size={block_size} is not yet supported; "
            f"only block_size={DEFAULT_BLOCK_SIZE} is wired through the Metal kernel."
        )
    if fp_tensor.dtype not in {mx.float32, mx.float16, mx.bfloat16}:
        raise TypeError(
            f"quantize_dynamic_blockwise expects a floating dtype, got {fp_tensor.dtype}"
        )

    original_shape = fp_tensor.shape
    flat = fp_tensor.reshape(-1)
    if flat.dtype != mx.float32:
        flat = flat.astype(mx.float32)
    nblocks = num_blocks(int(flat.size), block_size)

    if int(flat.size) == 0:
        return (
            mx.zeros(original_shape, dtype=mx.uint8),
            mx.zeros((0,), dtype=mx.float32),
        )

    kernel = _get_quantize_kernel()
    absmax, qdata_flat = kernel(
        inputs=[flat],
        output_shapes=[(nblocks,), flat.shape],
        output_dtypes=[mx.float32, mx.uint8],
        grid=(nblocks * block_size, 1, 1),
        threadgroup=(block_size, 1, 1),
        stream=mx.gpu,
    )
    qdata = qdata_flat.reshape(original_shape)
    return qdata, absmax


def dequantize_dynamic_blockwise(
    qdata: mx.array,
    absmax: mx.array,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Inverse of :func:quantize_dynamic_blockwise.

    Reads the per-block absmax scale and reconstructs a tensor with the
    same shape as qdata and dtype out_dtype. Bias removal and the
    /127 normalization are folded into a single Metal pass.
    """

    if qdata.dtype != mx.uint8:
        raise TypeError(f"qdata must be uint8, got {qdata.dtype}")
    if absmax.dtype != mx.float32:
        raise TypeError(f"absmax must be float32, got {absmax.dtype}")

    original_shape = qdata.shape
    flat = qdata.reshape(-1)
    if int(flat.size) == 0:
        return mx.zeros(original_shape, dtype=out_dtype)

    kernel = _get_dequantize_kernel()
    threads = min(256, int(flat.size))
    out_flat = kernel(
        inputs=[flat, absmax],
        template=[("T", out_dtype)],
        output_shapes=[flat.shape],
        output_dtypes=[out_dtype],
        grid=(flat.size, 1, 1),
        threadgroup=(threads, 1, 1),
        stream=mx.gpu,
    )[0]
    return out_flat.reshape(original_shape)


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "QUANT_BIAS",
    "QUANT_RANGE",
    "dequantize_dynamic_blockwise",
    "num_blocks",
    "quantize_dynamic_blockwise",
]
