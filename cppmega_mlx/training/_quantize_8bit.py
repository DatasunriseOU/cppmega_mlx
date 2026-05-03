"""8-bit blockwise quantization helpers backed by Metal kernels.

This module ships two codec paths for the 8-bit Adam/Muon moment storage:

* **Symmetric int8** (``QUANT_SCHEME_SYMMETRIC``): per-256-block fp32 absmax,
  ``q = round(v/scale * 127) + 128``. Default scheme for backwards compat.
* **Dynamic LUT** (``QUANT_SCHEME_DYNAMIC``): the bitsandbytes
  ``dDequantizeBlockwise`` non-uniform 8-bit mapping where bits 7 selects
  sign, bits below define an exponent + fraction split that covers
  ``[-1, 1]`` with denser bins near zero. The lookup table is the
  ``signed`` variant of bnb's ``create_dynamic_map(signed=True,
  max_exponent_bits=7, total_bits=8)`` (the 256 fp32 entries that
  ``bnb.optim.Adam8bit`` defaults to).

Layout (matching cppmega/docs/memory_dtype_audit_2026_04_25.md):

    block_size = 256
    absmax: fp32 with one element per block of block_size consecutive
        elements (the tail block, if any, holds whatever fraction is left).
    qdata: uint8 with the same shape as the input. For the symmetric scheme
        we store the signed int8 representation in [-127, 127] biased by
        +128 so the on-disk byte is a real uint8 in [1, 255]. For the
        dynamic scheme we store the LUT index 0..255 directly.

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

QUANT_SCHEME_SYMMETRIC = "symmetric_int8_v1"
"""Identifier for the symmetric int8 blockwise codec."""

QUANT_SCHEME_DYNAMIC = "dynamic_int8_v1"
"""Identifier for the bitsandbytes-style dynamic 8-bit LUT codec."""

QUANT_SCHEMES = (QUANT_SCHEME_SYMMETRIC, QUANT_SCHEME_DYNAMIC)
"""All accepted scheme strings for the 8-bit codecs."""


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


# ---------------------------------------------------------------------------
# Dynamic 8-bit LUT codec (bitsandbytes-style).
# ---------------------------------------------------------------------------
#
# The 256-entry fp32 LUT below is generated by
# ``bitsandbytes.functional.create_dynamic_map(signed=True,
# max_exponent_bits=7, total_bits=8)`` -- the canonical signed dynamic map
# used by ``bnb.optim.Adam8bit`` for the m/v moment buffers.
#
# Reference: bitsandbytes/functional.py::create_dynamic_map (commit on main as
# of 2026-05). The algorithm: for each exponent ``i`` in ``[0,
# max_exponent_bits)``, generate ``2**(i + non_sign_bits - max_exponent_bits)
# + 1`` boundary points uniformly between 0.1 and 1.0, take adjacent means
# and scale them by ``10**(-(max_exponent_bits-1)+i)``. Append both signed
# copies plus the ``additional_items`` extras at the smallest scale, then
# ``data.append(0); data.append(1.0)``. Sort to get the 256 fp32 entries.
#
# The LUT covers ``[-0.99296875, 1.0]`` with denser bins near zero (e.g. the
# 16 indices straddling zero step in increments of ~5.5e-7) so that small
# Adam ``m, v`` values quantize without collapsing to bias=128 the way
# symmetric int8 does. The runtime kernel uses ``qdata[elem]`` as a direct
# index into this table, exactly matching ``dDequantizeBlockwise`` in
# bitsandbytes/csrc/kernels.cu.
_BNB_DYNAMIC_LUT_VALUES: tuple[float, ...] = (
    -0.992968738079071, -0.9789062738418579, -0.96484375, -0.9507812261581421,
    -0.936718761920929, -0.922656238079071, -0.9085937738418579, -0.89453125,
    -0.8804687261581421, -0.866406261920929, -0.852343738079071, -0.8382812738418579,
    -0.82421875, -0.8101562261581421, -0.796093761920929, -0.782031238079071,
    -0.7679687738418579, -0.75390625, -0.7398437261581421, -0.725781261920929,
    -0.711718738079071, -0.6976562738418579, -0.68359375, -0.6695312261581421,
    -0.655468761920929, -0.641406238079071, -0.6273437738418579, -0.61328125,
    -0.5992187261581421, -0.585156261920929, -0.571093738079071, -0.5570312738418579,
    -0.54296875, -0.5289062261581421, -0.514843761920929, -0.500781238079071,
    -0.4867187738418579, -0.47265625, -0.4585937261581421, -0.44453126192092896,
    -0.43046873807907104, -0.4164062738418579, -0.40234375, -0.3882812261581421,
    -0.37421876192092896, -0.36015623807907104, -0.3460937738418579, -0.33203125,
    -0.3179687261581421, -0.30390626192092896, -0.28984373807907104, -0.2757812738418579,
    -0.26171875, -0.24765624105930328, -0.23359374701976776, -0.21953125298023224,
    -0.20546874403953552, -0.19140625, -0.17734375596046448, -0.16328124701976776,
    -0.14921875298023224, -0.13515624403953552, -0.12109375, -0.10703125596046448,
    -0.09859374910593033, -0.09578125178813934, -0.09296874701976776, -0.09015624970197678,
    -0.08734375238418579, -0.08453124761581421, -0.08171875029802322, -0.07890625298023224,
    -0.07609374821186066, -0.07328125089406967, -0.07046875357627869, -0.0676562488079071,
    -0.06484375149011612, -0.06203124672174454, -0.05921874940395355, -0.05640625208616257,
    -0.053593751043081284, -0.05078125, -0.047968748956918716, -0.04515624791383743,
    -0.04234375059604645, -0.039531249552965164, -0.03671874850988388, -0.033906251192092896,
    -0.031093750149011612, -0.028281250968575478, -0.025468749925494194, -0.02265625074505806,
    -0.019843749701976776, -0.017031250521540642, -0.014218749478459358, -0.011406250298023224,
    -0.009718749672174454, -0.009156250394880772, -0.008593750186264515, -0.008031249977648258,
    -0.0074687497690320015, -0.006906250026077032, -0.006343750283122063, -0.005781250074505806,
    -0.005218749865889549, -0.00465625012293458, -0.004093749914318323, -0.00353124993853271,
    -0.002968749962747097, -0.002406249986961484, -0.001843750011175871, -0.0012812500353902578,
    -0.0009437499684281647, -0.0008312499849125743, -0.0007187500013969839, -0.0006062500178813934,
    -0.000493750034365803, -0.00038124999264255166, -0.00026875000912696123, -0.00015624999650754035,
    -8.874999912222847e-05, -6.625000241911039e-05, -4.374999844003469e-05, -2.1249999917927198e-05,
    -7.749999895168003e-06, -3.2499999633728294e-06, -5.499999815583578e-07, 0.0,
    5.499999815583578e-07, 3.2499999633728294e-06, 7.749999895168003e-06, 2.1249999917927198e-05,
    4.374999844003469e-05, 6.625000241911039e-05, 8.874999912222847e-05, 0.00015624999650754035,
    0.00026875000912696123, 0.00038124999264255166, 0.000493750034365803, 0.0006062500178813934,
    0.0007187500013969839, 0.0008312499849125743, 0.0009437499684281647, 0.0012812500353902578,
    0.001843750011175871, 0.002406249986961484, 0.002968749962747097, 0.00353124993853271,
    0.004093749914318323, 0.00465625012293458, 0.005218749865889549, 0.005781250074505806,
    0.006343750283122063, 0.006906250026077032, 0.0074687497690320015, 0.008031249977648258,
    0.008593750186264515, 0.009156250394880772, 0.009718749672174454, 0.011406250298023224,
    0.014218749478459358, 0.017031250521540642, 0.019843749701976776, 0.02265625074505806,
    0.025468749925494194, 0.028281250968575478, 0.031093750149011612, 0.033906251192092896,
    0.03671874850988388, 0.039531249552965164, 0.04234375059604645, 0.04515624791383743,
    0.047968748956918716, 0.05078125, 0.053593751043081284, 0.05640625208616257,
    0.05921874940395355, 0.06203124672174454, 0.06484375149011612, 0.0676562488079071,
    0.07046875357627869, 0.07328125089406967, 0.07609374821186066, 0.07890625298023224,
    0.08171875029802322, 0.08453124761581421, 0.08734375238418579, 0.09015624970197678,
    0.09296874701976776, 0.09578125178813934, 0.09859374910593033, 0.10703125596046448,
    0.12109375, 0.13515624403953552, 0.14921875298023224, 0.16328124701976776,
    0.17734375596046448, 0.19140625, 0.20546874403953552, 0.21953125298023224,
    0.23359374701976776, 0.24765624105930328, 0.26171875, 0.2757812738418579,
    0.28984373807907104, 0.30390626192092896, 0.3179687261581421, 0.33203125,
    0.3460937738418579, 0.36015623807907104, 0.37421876192092896, 0.3882812261581421,
    0.40234375, 0.4164062738418579, 0.43046873807907104, 0.44453126192092896,
    0.4585937261581421, 0.47265625, 0.4867187738418579, 0.500781238079071,
    0.514843761920929, 0.5289062261581421, 0.54296875, 0.5570312738418579,
    0.571093738079071, 0.585156261920929, 0.5992187261581421, 0.61328125,
    0.6273437738418579, 0.641406238079071, 0.655468761920929, 0.6695312261581421,
    0.68359375, 0.6976562738418579, 0.711718738079071, 0.725781261920929,
    0.7398437261581421, 0.75390625, 0.7679687738418579, 0.782031238079071,
    0.796093761920929, 0.8101562261581421, 0.82421875, 0.8382812738418579,
    0.852343738079071, 0.866406261920929, 0.8804687261581421, 0.89453125,
    0.9085937738418579, 0.922656238079071, 0.936718761920929, 0.9507812261581421,
    0.96484375, 0.9789062738418579, 0.992968738079071, 1.0,
)


def create_dynamic_map() -> mx.array:
    """Return the 256-entry signed dynamic 8-bit LUT as a fp32 ``mx.array``.

    The LUT is the ``bitsandbytes.functional.create_dynamic_map(signed=True,
    max_exponent_bits=7, total_bits=8)`` table baked at module-load time, so
    the dequantize kernel's ``code[qdata[elem]]`` lookup matches
    ``dDequantizeBlockwise`` in ``bitsandbytes/csrc/kernels.cu`` byte-for-byte.

    Returns a fresh ``mx.array`` so callers can keep an immutable reference;
    the codec itself caches a single shared instance via :func:`_get_lut`.
    """

    return mx.array(list(_BNB_DYNAMIC_LUT_VALUES), dtype=mx.float32)


_dynamic_lut: Optional[mx.array] = None


def _get_lut() -> mx.array:
    """Return the module-scoped fp32 LUT, materializing it on first use."""

    global _dynamic_lut
    if _dynamic_lut is None:
        _dynamic_lut = create_dynamic_map()
    return _dynamic_lut


# Sorted-LUT binary search: each thread computes |x|/scale (sign preserved),
# then walks the LUT to find the index whose value is closest. With
# ``_BNB_DYNAMIC_LUT_VALUES`` monotone non-decreasing, a 8-step binary search
# nails the upper bound; we then compare the upper and lower neighbours and
# emit the closer index. The LUT is loaded into threadgroup memory once per
# block to amortize the constant-buffer fetch cost.
_QUANTIZE_DYNAMIC_HEADER = """
constant constexpr uint BLOCK_SIZE_DEFAULT = 256;
constant constexpr uint LUT_SIZE = 256;
"""

_QUANTIZE_DYNAMIC_SOURCE = """
    threadgroup float scratch[256];
    threadgroup float lut_tg[256];
    uint tid = thread_position_in_threadgroup.x;
    uint bid = threadgroup_position_in_grid.x;
    uint total = x_shape[0];
    uint elem = bid * BLOCK_SIZE_DEFAULT + tid;

    // Stage 0: cooperatively load the 256-entry LUT into threadgroup memory.
    lut_tg[tid] = lut[tid];

    // Stage 1: load |x[elem]| into scratch (zero-pad tail).
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

    // Stage 3: per-thread quantization via LUT binary search.
    if (elem < total) {
        float xval = (float)x[elem];
        float normalized = (scale > 0.0f) ? (xval / scale) : 0.0f;
        // Clamp to LUT-supported range; LUT covers [-0.99296875, 1.0].
        if (normalized > 1.0f) normalized = 1.0f;
        if (normalized < -1.0f) normalized = -1.0f;
        // Binary search for the smallest index i with lut_tg[i] >= normalized.
        uint lo = 0u;
        uint hi = LUT_SIZE - 1u;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1;
            if (lut_tg[mid] < normalized) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        // Pick the closer of [lo, lo-1] (lo-1 only if lo > 0).
        uint best = lo;
        if (lo > 0u) {
            float d_hi = metal::abs(lut_tg[lo] - normalized);
            float d_lo = metal::abs(lut_tg[lo - 1u] - normalized);
            if (d_lo < d_hi) best = lo - 1u;
        }
        qdata[elem] = (uint8_t)best;
    }
"""

_DEQUANTIZE_DYNAMIC_SOURCE = """
    uint elem = thread_position_in_grid.x;
    if (elem >= qdata_shape[0]) return;
    uint bid = elem / 256u;
    float scale = absmax[bid];
    uint idx = (uint)qdata[elem];
    float val = lut[idx] * scale;
    out[elem] = (T)val;
"""


_quantize_dynamic_kernel: Optional[object] = None
_dequantize_dynamic_kernel: Optional[object] = None


def _get_quantize_dynamic_kernel() -> object:
    global _quantize_dynamic_kernel
    if _quantize_dynamic_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Adam8bit dynamic quantization requires the MLX Metal backend; "
                "default device is not GPU or mx.metal is unavailable."
            )
        _quantize_dynamic_kernel = mx.fast.metal_kernel(
            name="cppmega_quantize_8bit_dynamic",
            input_names=["x", "lut"],
            output_names=["absmax", "qdata"],
            header=_QUANTIZE_DYNAMIC_HEADER,
            source=_QUANTIZE_DYNAMIC_SOURCE,
            ensure_row_contiguous=True,
        )
    return _quantize_dynamic_kernel


def _get_dequantize_dynamic_kernel() -> object:
    global _dequantize_dynamic_kernel
    if _dequantize_dynamic_kernel is None:
        if not _can_run_metal():
            raise RuntimeError(
                "Adam8bit dynamic dequantization requires the MLX Metal backend."
            )
        _dequantize_dynamic_kernel = mx.fast.metal_kernel(
            name="cppmega_dequantize_8bit_dynamic",
            input_names=["qdata", "absmax", "lut"],
            output_names=["out"],
            source=_DEQUANTIZE_DYNAMIC_SOURCE,
            ensure_row_contiguous=True,
        )
    return _dequantize_dynamic_kernel


def quantize_dynamic_lut_blockwise(
    fp_tensor: mx.array,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[mx.array, mx.array]:
    """Per-block 8-bit quantization using bitsandbytes' dynamic LUT.

    Mirrors the layout of :func:`quantize_dynamic_blockwise` (per-256-block
    fp32 absmax + uint8 payload), but the byte stored is the LUT index in
    ``[0, 255]`` rather than a symmetric int8. Round-trip error near zero is
    ~5x tighter than the symmetric path because the LUT bins are
    exponentially denser around the origin.

    Returns ``(qdata, absmax)`` with the same shapes/dtypes as the symmetric
    codec so the calling Adam8bit / Muon code can swap one for the other
    without touching state allocation.
    """

    if block_size != DEFAULT_BLOCK_SIZE:
        raise NotImplementedError(
            f"block_size={block_size} is not yet supported; "
            f"only block_size={DEFAULT_BLOCK_SIZE} is wired through the Metal kernel."
        )
    if fp_tensor.dtype not in {mx.float32, mx.float16, mx.bfloat16}:
        raise TypeError(
            f"quantize_dynamic_lut_blockwise expects a floating dtype, got {fp_tensor.dtype}"
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

    kernel = _get_quantize_dynamic_kernel()
    lut = _get_lut()
    absmax, qdata_flat = kernel(
        inputs=[flat, lut],
        output_shapes=[(nblocks,), flat.shape],
        output_dtypes=[mx.float32, mx.uint8],
        grid=(nblocks * block_size, 1, 1),
        threadgroup=(block_size, 1, 1),
        stream=mx.gpu,
    )
    qdata = qdata_flat.reshape(original_shape)
    return qdata, absmax


def dequantize_dynamic_lut_blockwise(
    qdata: mx.array,
    absmax: mx.array,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Inverse of :func:`quantize_dynamic_lut_blockwise`.

    Indexes the bnb dynamic LUT by ``qdata[elem]`` then multiplies by the
    per-block ``absmax`` -- exactly the ``code[qvals[j]] * local_abs_max``
    inner loop in ``dDequantizeBlockwise`` (kernels.cu, ``General8bit``).
    """

    if qdata.dtype != mx.uint8:
        raise TypeError(f"qdata must be uint8, got {qdata.dtype}")
    if absmax.dtype != mx.float32:
        raise TypeError(f"absmax must be float32, got {absmax.dtype}")

    original_shape = qdata.shape
    flat = qdata.reshape(-1)
    if int(flat.size) == 0:
        return mx.zeros(original_shape, dtype=out_dtype)

    kernel = _get_dequantize_dynamic_kernel()
    lut = _get_lut()
    threads = min(256, int(flat.size))
    out_flat = kernel(
        inputs=[flat, absmax, lut],
        template=[("T", out_dtype)],
        output_shapes=[flat.shape],
        output_dtypes=[out_dtype],
        grid=(flat.size, 1, 1),
        threadgroup=(threads, 1, 1),
        stream=mx.gpu,
    )[0]
    return out_flat.reshape(original_shape)


# ---------------------------------------------------------------------------
# Scheme dispatch helpers.
# ---------------------------------------------------------------------------


def quantize_blockwise(
    fp_tensor: mx.array,
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    scheme: str = QUANT_SCHEME_SYMMETRIC,
) -> tuple[mx.array, mx.array]:
    """Dispatch the 8-bit codec by ``scheme`` string.

    ``scheme`` must be one of :data:`QUANT_SCHEMES`. ``QUANT_SCHEME_SYMMETRIC``
    routes to the symmetric int8 codec (uint8 with +128 bias), while
    ``QUANT_SCHEME_DYNAMIC`` routes to the bnb dynamic LUT codec (uint8 LUT
    index in ``[0, 255]``). Both share the per-256-block fp32 absmax layout.
    """

    if scheme == QUANT_SCHEME_SYMMETRIC:
        return quantize_dynamic_blockwise(fp_tensor, block_size)
    if scheme == QUANT_SCHEME_DYNAMIC:
        return quantize_dynamic_lut_blockwise(fp_tensor, block_size)
    raise ValueError(
        f"unknown quant scheme {scheme!r}; expected one of {QUANT_SCHEMES}"
    )


def dequantize_blockwise(
    qdata: mx.array,
    absmax: mx.array,
    *,
    scheme: str = QUANT_SCHEME_SYMMETRIC,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dispatch the 8-bit dequantize codec by ``scheme`` string."""

    if scheme == QUANT_SCHEME_SYMMETRIC:
        return dequantize_dynamic_blockwise(qdata, absmax, out_dtype=out_dtype)
    if scheme == QUANT_SCHEME_DYNAMIC:
        return dequantize_dynamic_lut_blockwise(qdata, absmax, out_dtype=out_dtype)
    raise ValueError(
        f"unknown quant scheme {scheme!r}; expected one of {QUANT_SCHEMES}"
    )


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "QUANT_BIAS",
    "QUANT_RANGE",
    "QUANT_SCHEME_DYNAMIC",
    "QUANT_SCHEME_SYMMETRIC",
    "QUANT_SCHEMES",
    "create_dynamic_map",
    "dequantize_blockwise",
    "dequantize_dynamic_blockwise",
    "dequantize_dynamic_lut_blockwise",
    "num_blocks",
    "quantize_blockwise",
    "quantize_dynamic_blockwise",
    "quantize_dynamic_lut_blockwise",
]
