"""Vendored FP8 (e4m3fn) MSL kernels for Apple Silicon.

.. note::
   **Migration wave-6 audit**: this module hosts **four** raw-MSL kernels
   (``_FP8_TO_HALF``, ``_HALF_TO_FP8``, ``_FP8_MATMUL``, ``_FP8_VECMAT``).
   None of them use Apple-silicon FP8 SIMDgroup MMA -- the matmul/vecmat
   kernels decode FP8 bytes via a 256-entry LUT in MSL constant memory and
   accumulate in plain ``float`` ALU registers. So the originally-suspected
   blocker (``simdgroup_a_fp8`` / ``simdgroup_b_fp8`` Fragment factories in
   ``tilelang/language/extern.py``) is **not** what gates flipping these
   kernels to the unified engine.

   The actual blocker is two-fold:

   1. ``tl.extern_intrinsic`` on the Metal target does not yet emit a
      ``mx.fast.metal_kernel``-shaped artifact (it emits a CUDA/HIP-shaped
      device-function call), and
   2. The unified engine has no mechanism to inject a 256-entry constant
      LUT (``constant float fp8_e4m3fn_lut[256]`` here) into the MSL
      header at codegen time -- TileLang would need a constant-table
      extern declaration that ``codegen_metal.cc`` knows to materialise.

   Until both land, callers stay on ``mx.fast.metal_kernel`` directly. The
   3-kernel classification below survives in source comments per kernel.

   TODO(wave-7): once Metal-target ``extern_intrinsic`` ships, wrap each
   kernel body in a thin ``@T.prim_func`` calling ``T.extern_intrinsic``
   with the existing MSL body verbatim, then route through
   ``_engine_dispatch.dispatch_lower(prim, "metal", return_msl=True)`` so
   the four wrappers participate in the unified ``CPPMEGA_MLX_TILELANG_ENGINE``
   env-flag dispatch. The ``_msl_extraction.extract_msl_from_engine_artifact``
   adapter (commit 00d6d90) is already prepared for this -- it reads
   ``artifact.kernel_source`` and reconstitutes a ``TileLangMSLLowering``.

   See ``MIGRATION_PLAN.md §2.4`` (FP8 vecmat consolidation) and gap #7
   (FP8 factories - not actually a blocker for THIS file).

Source attribution
------------------

The Metal Shading Language sources embedded in this module are direct,
near-verbatim ports (LUT tables and kernel bodies) of the kernels published
in two upstream repositories:

* AppMana/mps-fp8-for-torch-and-comfyui-python-package
  (commit a902571eca5362f5e2496cf33dcce52c8bac6a15) -- LUT-based decode and
  vectorized matmul. License: Apache 2.0.
  Source file: src/fp4_fp8_for_torch_mps/shaders/fp8_matmul.metal

* audiohacking/fp8-mps-metal
  (commit d4fbd40c48aa2a243e600d06627c7dd818150636) -- earlier branchy decode
  variant kept for reference. License: MIT (declared in README; LICENSE.txt
  referenced in pyproject is not present in tree).

Both upstream projects target torch.mps via torch.mps.compile_shader; this
module re-hosts the same MSL via mx.fast.metal_kernel so the kernels run
inside MLX/cppmega_mlx without any PyTorch dependency.

Why direct vendoring
--------------------

Apple Silicon (through M5 / MSL 4.0) has no native FP8 hardware path
exposed in MSL: there is no float8_e4m3 type and no simdgroup_matrix<float8>
MMA instruction. The TileLang TVM-Metal codegen explicitly errors on
``float8_e4m3 -> Metal type`` (3rdparty/tvm/src/target/source/codegen_metal.cc:271).

Instead we treat FP8 as **storage-only**:

1. Quantize: float -> uint8 via float_to_fp8_e4m3fn (integer bit
   manipulation, no transcendentals, banker's rounding).
2. Dequantize: uint8 -> float via a 256-entry LUT in MSL constant memory
   (single load, no branching, fast on Apple GPUs).
3. Matmul: dequantize Q/K bytes into fp32 ALU registers and run the matmul
   using regular float fma. Accumulator is fp32; output is fp32 (matmul) or
   fp16 (vecmat / dequantize convenience).

This complements -- but does not duplicate -- the manual bit-extraction
fused FP8 sparse-MLA forward in
``cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py``: that file fuses the dequant
inline inside an attention kernel for one specific layout (Q/KV with
per-token scales) using a custom decode helper. The kernels here are
**generic** scaled matmul / vecmat / quantize / dequantize building blocks
that can drop into any FP8 path (linear layers, MoE up-projections,
block-scaled MLA, etc.).

License contract
----------------

Both upstream projects are MIT- or Apache 2.0-licensed (permissive). The
embedded MSL sources are reproduced under their original license terms;
``__license_notice__`` carries the attribution that distributed binaries
must keep visible.

API surface
-----------

``fp8_to_half_kernel(fp8_uchar, count)`` -> fp16
``half_to_fp8_kernel(half_arr, count)`` -> uint8 (e4m3fn bytes)
``fp8_scaled_matmul(A_fp8, A_scale, B_fp8, B_scale, scale_mode)`` -> fp32 (M,N)
``fp8_scaled_vecmat(x_fp8, W_fp8, scale_x, scale_w, scale_mode)`` -> fp32 (N,)

The matmul wrapper carries an mx.custom_function VJP (forward-only kernel
on the GPU, but the host-side fp32 grad is computed via dequantize +
mx.matmul; gradients flow back as fp32 with the per-tensor scale factored
out). The vecmat wrapper is forward-only (used at inference time on M=1
batches; backward uses the matmul kernel with M=1 instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang._msl_transform import (
    MetalKernel,
    can_run_metal,
    dispatch,
    make_metal_kernel,
)


__license_notice__ = (
    "FP8 MSL kernels in cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py are "
    "ported from AppMana/mps-fp8-for-torch-and-comfyui-python-package "
    "(commit a902571e, Apache 2.0) and audiohacking/fp8-mps-metal "
    "(commit d4fbd40c, MIT). Original attribution and license terms apply "
    "to redistributed binaries that include this MSL."
)


# ---------------------------------------------------------------------------
# MSL header: e4m3fn LUTs (256 entries) + integer-bit-manipulation encode.
# ---------------------------------------------------------------------------
#
# The float and half LUTs are byte-for-byte the same constants as the
# AppMana fork: 256 entries indexed by the raw uint8 e4m3fn bit pattern.
# Indices 0x00..0x7F cover non-negative values, 0x80..0xFF cover negatives.
# Special values: 0x7F and 0xFF map to NaN. Index 0x80 is signed-zero (-0).
#
# float_to_fp8_e4m3fn() uses ``as_type<uint>(val)`` to read float32 bits,
# extracts the IEEE 754 sign / exponent / mantissa, and emits the e4m3fn
# byte with banker's rounding. No log2/exp2 transcendentals on the hot path.

_FP8_HEADER = """
#include <metal_stdlib>
using namespace metal;

// FP8 e4m3fn -> float32 LUT (256 entries, indexed by raw uint8).
// Source: AppMana/mps-fp8-for-torch-and-comfyui-python-package
//         (commit a902571e, Apache 2.0)
constant float fp8_e4m3fn_lut[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, NAN,
};

// Half-precision LUT (same data, half storage). Used by fp8_to_half_kernel
// for slightly faster dequant when the consumer is fp16-only.
constant half fp8_e4m3fn_lut_half[256] = {
    0.0h, 0.001953125h, 0.00390625h, 0.005859375h, 0.0078125h, 0.009765625h, 0.01171875h, 0.013671875h,
    0.015625h, 0.017578125h, 0.01953125h, 0.021484375h, 0.0234375h, 0.025390625h, 0.02734375h, 0.029296875h,
    0.03125h, 0.03515625h, 0.0390625h, 0.04296875h, 0.046875h, 0.05078125h, 0.0546875h, 0.05859375h,
    0.0625h, 0.0703125h, 0.078125h, 0.0859375h, 0.09375h, 0.1015625h, 0.109375h, 0.1171875h,
    0.125h, 0.140625h, 0.15625h, 0.171875h, 0.1875h, 0.203125h, 0.21875h, 0.234375h,
    0.25h, 0.28125h, 0.3125h, 0.34375h, 0.375h, 0.40625h, 0.4375h, 0.46875h,
    0.5h, 0.5625h, 0.625h, 0.6875h, 0.75h, 0.8125h, 0.875h, 0.9375h,
    1.0h, 1.125h, 1.25h, 1.375h, 1.5h, 1.625h, 1.75h, 1.875h,
    2.0h, 2.25h, 2.5h, 2.75h, 3.0h, 3.25h, 3.5h, 3.75h,
    4.0h, 4.5h, 5.0h, 5.5h, 6.0h, 6.5h, 7.0h, 7.5h,
    8.0h, 9.0h, 10.0h, 11.0h, 12.0h, 13.0h, 14.0h, 15.0h,
    16.0h, 18.0h, 20.0h, 22.0h, 24.0h, 26.0h, 28.0h, 30.0h,
    32.0h, 36.0h, 40.0h, 44.0h, 48.0h, 52.0h, 56.0h, 60.0h,
    64.0h, 72.0h, 80.0h, 88.0h, 96.0h, 104.0h, 112.0h, 120.0h,
    128.0h, 144.0h, 160.0h, 176.0h, 192.0h, 208.0h, 224.0h, 240.0h,
    256.0h, 288.0h, 320.0h, 352.0h, 384.0h, 416.0h, 448.0h, NAN,
    0.0h, -0.001953125h, -0.00390625h, -0.005859375h, -0.0078125h, -0.009765625h, -0.01171875h, -0.013671875h,
    -0.015625h, -0.017578125h, -0.01953125h, -0.021484375h, -0.0234375h, -0.025390625h, -0.02734375h, -0.029296875h,
    -0.03125h, -0.03515625h, -0.0390625h, -0.04296875h, -0.046875h, -0.05078125h, -0.0546875h, -0.05859375h,
    -0.0625h, -0.0703125h, -0.078125h, -0.0859375h, -0.09375h, -0.1015625h, -0.109375h, -0.1171875h,
    -0.125h, -0.140625h, -0.15625h, -0.171875h, -0.1875h, -0.203125h, -0.21875h, -0.234375h,
    -0.25h, -0.28125h, -0.3125h, -0.34375h, -0.375h, -0.40625h, -0.4375h, -0.46875h,
    -0.5h, -0.5625h, -0.625h, -0.6875h, -0.75h, -0.8125h, -0.875h, -0.9375h,
    -1.0h, -1.125h, -1.25h, -1.375h, -1.5h, -1.625h, -1.75h, -1.875h,
    -2.0h, -2.25h, -2.5h, -2.75h, -3.0h, -3.25h, -3.5h, -3.75h,
    -4.0h, -4.5h, -5.0h, -5.5h, -6.0h, -6.5h, -7.0h, -7.5h,
    -8.0h, -9.0h, -10.0h, -11.0h, -12.0h, -13.0h, -14.0h, -15.0h,
    -16.0h, -18.0h, -20.0h, -22.0h, -24.0h, -26.0h, -28.0h, -30.0h,
    -32.0h, -36.0h, -40.0h, -44.0h, -48.0h, -52.0h, -56.0h, -60.0h,
    -64.0h, -72.0h, -80.0h, -88.0h, -96.0h, -104.0h, -112.0h, -120.0h,
    -128.0h, -144.0h, -160.0h, -176.0h, -192.0h, -208.0h, -224.0h, -240.0h,
    -256.0h, -288.0h, -320.0h, -352.0h, -384.0h, -416.0h, -448.0h, NAN,
};

// float -> FP8 e4m3fn encode using integer bit manipulation (no log2/exp2).
// Source: AppMana fork, commit a902571e.
inline uchar float_to_fp8_e4m3fn(float val) {
    uint raw = as_type<uint>(val);
    uint sign = raw >> 31;
    val = abs(val);

    if (val >= 448.0f) return uchar((sign << 7) | 0x7E);
    if (val < (1.0f / 512.0f)) return uchar(sign << 7);

    uint bits = as_type<uint>(val);
    int f32_exp = int((bits >> 23) & 0xFF) - 127;
    uint f32_mant = bits & 0x7FFFFF;

    if (f32_exp < -6) {
        // Subnormal FP8 path: round-to-nearest-even via rint().
        float mant_f = val * 512.0f;
        uint mant = uint(rint(mant_f));
        if (mant >= 8) return uchar((sign << 7) | 0x08);
        return uchar((sign << 7) | mant);
    }

    // Normal FP8 path: shift float32 mantissa to 3 bits with banker's rounding.
    uint truncated = f32_mant & 0xFFFFF;
    uint halfway = 1u << 19;
    uint mant = f32_mant >> 20;
    if (truncated > halfway || (truncated == halfway && (mant & 1))) {
        mant++;
    }
    int fp8_exp = f32_exp + 7;

    if (mant > 7) { mant = 0; fp8_exp++; }
    fp8_exp = clamp(fp8_exp, 1, 15);
    if (fp8_exp == 15 && mant == 7) mant = 6;

    return uchar((sign << 7) | uint(fp8_exp << 3) | mant);
}
"""


# ---------------------------------------------------------------------------
# Kernel body sources (Apache 2.0 / AppMana commit a902571e).
# ---------------------------------------------------------------------------
#
# Each body is the contents of a kernel void; mx.fast.metal_kernel wraps it
# in a kernel signature derived from input_names / output_names. Buffer
# names below match the input_names / output_names declared in the
# corresponding factory call.

_FP8_TO_HALF_BODY = """
    uint gid = thread_position_in_grid.x;
    uint count = uint(input_shape[0]);
    if (gid >= count) return;
    output[gid] = fp8_e4m3fn_lut_half[uint(input[gid])];
"""

_HALF_TO_FP8_BODY = """
    uint gid = thread_position_in_grid.x;
    uint count = uint(input_shape[0]);
    if (gid >= count) return;
    output[gid] = float_to_fp8_e4m3fn(float(input[gid]));
"""

# Scaled MxN matmul: A is (M, K) e4m3fn, B is (N, K) e4m3fn (transposed),
# C is (M, N) fp32. Per-tensor scale (mode 0) or per-channel/row scale
# (mode 1). The 4-element K-axis unroll improves device memory bandwidth on
# Apple GPUs, mirroring the AppMana / metalQwen3 reference. Threadgroup
# shape is configurable but we default to (16, 16, 1) for general MxN.
_FP8_MATMUL_BODY = """
    uint row = thread_position_in_grid.y;
    uint col = thread_position_in_grid.x;
    uint M = uint(A_shape[0]);
    uint N = uint(B_shape[0]);
    uint K = uint(A_shape[1]);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    uint a_base = row * K;
    uint b_base = col * K;

    device const uint* A4 = reinterpret_cast<device const uint*>(A + a_base);
    device const uint* B4 = reinterpret_cast<device const uint*>(B + b_base);
    uint K4 = K / 4u;
    for (uint i = 0u; i < K4; i++) {
        uint pa = A4[i];
        uint pb = B4[i];
        sum += fp8_e4m3fn_lut[pa & 0xFFu]          * fp8_e4m3fn_lut[pb & 0xFFu]
             + fp8_e4m3fn_lut[(pa >> 8) & 0xFFu]   * fp8_e4m3fn_lut[(pb >> 8) & 0xFFu]
             + fp8_e4m3fn_lut[(pa >> 16) & 0xFFu]  * fp8_e4m3fn_lut[(pb >> 16) & 0xFFu]
             + fp8_e4m3fn_lut[(pa >> 24) & 0xFFu]  * fp8_e4m3fn_lut[(pb >> 24) & 0xFFu];
    }
    for (uint k = K4 * 4u; k < K; k++) {
        sum += fp8_e4m3fn_lut[uint(A[a_base + k])] * fp8_e4m3fn_lut[uint(B[b_base + k])];
    }

    uint scale_mode = uint(scale_mode_buf[0]);
    float sa = (scale_mode == 0u) ? float(scale_a[0]) : float(scale_a[row]);
    float sb = (scale_mode == 0u) ? float(scale_b[0]) : float(scale_b[col]);
    C[row * N + col] = sum * sa * sb;
"""

# Vec * matmul (M=1) with SIMD reduction across K. Each SIMD group (32
# lanes) handles one output row. We use vectorized 4-byte loads through
# uint32_t reinterpret to amortize the byte-level decode.
_FP8_VECMAT_BODY = """
    uint gid = thread_position_in_grid.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint N = uint(W_shape[0]);
    uint K = uint(W_shape[1]);
    uint row = gid / 32u;
    if (row >= N) return;

    uint row_offset = row * K;
    float sum = 0.0f;

    device const uint* x4 = reinterpret_cast<device const uint*>(x);
    device const uint* w4 = reinterpret_cast<device const uint*>(W + row_offset);
    uint K4 = K / 4u;
    for (uint i = simd_lane; i < K4; i += 32u) {
        uint px = x4[i];
        uint pw = w4[i];
        sum += fp8_e4m3fn_lut[px & 0xFFu]          * fp8_e4m3fn_lut[pw & 0xFFu]
             + fp8_e4m3fn_lut[(px >> 8) & 0xFFu]   * fp8_e4m3fn_lut[(pw >> 8) & 0xFFu]
             + fp8_e4m3fn_lut[(px >> 16) & 0xFFu]  * fp8_e4m3fn_lut[(pw >> 16) & 0xFFu]
             + fp8_e4m3fn_lut[(px >> 24) & 0xFFu]  * fp8_e4m3fn_lut[(pw >> 24) & 0xFFu];
    }
    for (uint k = K4 * 4u + simd_lane; k < K; k += 32u) {
        sum += fp8_e4m3fn_lut[uint(x[k])] * fp8_e4m3fn_lut[uint(W[row_offset + k])];
    }

    sum = simd_sum(sum);

    if (simd_lane == 0u) {
        uint scale_mode = uint(scale_mode_buf[0]);
        float sx = float(scale_x[0]);
        float sw = (scale_mode == 0u) ? float(scale_w[0]) : float(scale_w[row]);
        output[row] = sum * sx * sw;
    }
"""


# ---------------------------------------------------------------------------
# Compiled kernel handles (lazy: None when Metal unavailable).
# ---------------------------------------------------------------------------


# Wave-6 classification (per kernel, see module docstring for the 2-fold blocker):
#   _FP8_TO_HALF_KERNEL      : fp8-bordered (256-entry LUT lookup, elementwise)
#   _HALF_TO_FP8_KERNEL      : fp8-pure     (integer bit-manipulation encode)
#   _FP8_MATMUL_KERNEL       : fp8-bordered (LUT decode + fp32 fma + scale epilogue)
#   _FP8_VECMAT_KERNEL       : fp8-bordered (LUT decode + fp32 fma + simd_sum)
#
# All four stay on mx.fast.metal_kernel until wave-7 (Metal-target
# extern_intrinsic + constant-table extern in codegen_metal.cc). The
# wave-6 audit reclassifies them away from the originally-suspected
# "FP8 SIMDgroup factory" blocker -- none of the kernels here use
# simdgroup_matrix<float8> MMA.


_FP8_TO_HALF_KERNEL: MetalKernel | None = make_metal_kernel(
    name="cppmega_fp8_to_half",
    input_names=["input"],
    output_names=["output"],
    source=_FP8_TO_HALF_BODY,
    header=_FP8_HEADER,
)

_HALF_TO_FP8_KERNEL: MetalKernel | None = make_metal_kernel(
    name="cppmega_half_to_fp8",
    input_names=["input"],
    output_names=["output"],
    source=_HALF_TO_FP8_BODY,
    header=_FP8_HEADER,
)

_FP8_MATMUL_KERNEL: MetalKernel | None = make_metal_kernel(
    name="cppmega_fp8_scaled_matmul",
    input_names=["A", "B", "scale_a", "scale_b", "scale_mode_buf"],
    output_names=["C"],
    source=_FP8_MATMUL_BODY,
    header=_FP8_HEADER,
)

_FP8_VECMAT_KERNEL: MetalKernel | None = make_metal_kernel(
    name="cppmega_fp8_scaled_vecmat",
    input_names=["x", "W", "scale_x", "scale_w", "scale_mode_buf"],
    output_names=["output"],
    source=_FP8_VECMAT_BODY,
    header=_FP8_HEADER,
)


# ---------------------------------------------------------------------------
# Wave-7 flip pattern (placeholder — gated on Metal-target extern_intrinsic
# + constant-table extern landing in tilelang codegen_metal.cc).
# ---------------------------------------------------------------------------


def _wave7_engine_flip_blocked_reason() -> str:
    """Return a human-readable reason this module hasn't been engine-flipped.

    Used by tests and by ``fp8_msl_status()`` to report why the four kernels
    here remain on raw ``mx.fast.metal_kernel`` even after wave-3 landed the
    ``_msl_extraction`` adapter. The two prerequisites are described in the
    module docstring; this function exists so callers can branch precisely
    on the real blocker rather than the FP8-factory red herring.
    """

    return (
        "wave-7 blocked: tl.extern_intrinsic does not yet emit "
        "mx.fast.metal_kernel-shaped artifacts on the Metal target, and "
        "the unified engine has no constant-table extern (LUT injection) "
        "for codegen_metal.cc. The four FP8 MSL kernels here do NOT need "
        "FP8 SIMDgroup factories — they decode via a 256-entry uint8 LUT "
        "in MSL constant memory and accumulate in plain fp32. See module "
        "docstring TODO(wave-7) for the flip plan."
    )


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FP8MSLKernelStatus:
    """Runtime status of the vendored FP8 MSL kernels."""

    available: bool
    reason: str


def fp8_msl_status() -> FP8MSLKernelStatus:
    """Return whether the vendored FP8 MSL kernels are dispatchable."""

    if not can_run_metal():
        return FP8MSLKernelStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    if (
        _FP8_TO_HALF_KERNEL is None
        or _HALF_TO_FP8_KERNEL is None
        or _FP8_MATMUL_KERNEL is None
        or _FP8_VECMAT_KERNEL is None
    ):
        return FP8MSLKernelStatus(
            available=False,
            reason=(
                "mx.fast.metal_kernel is unavailable; FP8 MSL kernels did not "
                "compile in this environment."
            ),
        )
    return FP8MSLKernelStatus(
        available=True,
        reason=(
            "Vendored FP8 e4m3fn MSL kernels (LUT-based decode + integer-bit "
            "encode) are compiled and ready to dispatch."
        ),
    )


# ---------------------------------------------------------------------------
# Python-level wrappers
# ---------------------------------------------------------------------------


def _check_dtype(name: str, arr: mx.array, expected: mx.Dtype) -> None:
    if arr.dtype != expected:
        raise TypeError(
            f"fp8_msl_kernels.{name}: expected dtype {expected}, got {arr.dtype}"
        )


def fp8_to_half(fp8_uchar: mx.array) -> mx.array:
    """Dequantize a uint8-packed e4m3fn FP8 tensor to fp16.

    The byte ordering follows the e4m3fn bit pattern that ``mx.to_fp8``
    emits: ``[sign:1][exp:4][mantissa:3]``. The decode is a single LUT
    lookup per byte, with NaN values mapped through the LUT as ``NAN``.

    Args:
        fp8_uchar: uint8 array of any shape.

    Returns:
        fp16 array with the same shape as ``fp8_uchar``.
    """

    _check_dtype("fp8_to_half", fp8_uchar, mx.uint8)
    if fp8_uchar.size == 0:
        return mx.zeros(fp8_uchar.shape, dtype=mx.float16)
    if _FP8_TO_HALF_KERNEL is None or not can_run_metal():
        # Pure-MLX fallback: route via mx.from_fp8 (operates on stored FP8).
        return mx.from_fp8(fp8_uchar, dtype=mx.float16)

    flat = fp8_uchar.reshape(-1)
    n = flat.size
    threadgroup = (min(256, max(1, n)), 1, 1)
    grid = (((n + threadgroup[0] - 1) // threadgroup[0]) * threadgroup[0], 1, 1)
    outputs = dispatch(
        cast(MetalKernel, _FP8_TO_HALF_KERNEL),
        inputs=[flat],
        output_shapes=[(n,)],
        output_dtypes=[mx.float16],
        grid=grid,
        threadgroup=threadgroup,
    )
    return outputs[0].reshape(fp8_uchar.shape)


def half_to_fp8(half_arr: mx.array) -> mx.array:
    """Quantize an fp16 tensor to e4m3fn FP8 bytes.

    Rounding is round-half-to-even (banker's rounding) inside the MSL
    kernel; this matches the reference PyTorch CPU encoder used by the
    upstream MIT/Apache ports and our existing ``mx.to_fp8`` callsites.

    Args:
        half_arr: fp16 array of any shape.

    Returns:
        uint8 array with the same shape, holding e4m3fn bytes.
    """

    _check_dtype("half_to_fp8", half_arr, mx.float16)
    if half_arr.size == 0:
        return mx.zeros(half_arr.shape, dtype=mx.uint8)
    if _HALF_TO_FP8_KERNEL is None or not can_run_metal():
        # Pure-MLX fallback: mx.to_fp8 expects float32, hence the cast.
        return mx.to_fp8(half_arr.astype(mx.float32))

    flat = half_arr.reshape(-1)
    n = flat.size
    threadgroup = (min(256, max(1, n)), 1, 1)
    grid = (((n + threadgroup[0] - 1) // threadgroup[0]) * threadgroup[0], 1, 1)
    outputs = dispatch(
        cast(MetalKernel, _HALF_TO_FP8_KERNEL),
        inputs=[flat],
        output_shapes=[(n,)],
        output_dtypes=[mx.uint8],
        grid=grid,
        threadgroup=threadgroup,
    )
    return outputs[0].reshape(half_arr.shape)


def _resolve_scale(
    scale: mx.array | float, *, length: int, name: str
) -> Tuple[mx.array, int]:
    """Normalize ``scale`` to a 1D fp32 array; return ``(scale_array, mode)``.

    ``mode == 0`` is per-tensor (scalar); ``mode == 1`` is per-channel.
    """

    if isinstance(scale, (int, float)):
        return mx.array([float(scale)], dtype=mx.float32), 0
    if scale.size == 1:
        return scale.reshape(1).astype(mx.float32), 0
    if scale.size == length:
        return scale.reshape(length).astype(mx.float32), 1
    raise ValueError(
        f"fp8_msl_kernels: expected {name} to have size 1 (per-tensor) or "
        f"{length} (per-channel), got size {scale.size}"
    )


def fp8_scaled_matmul_raw(
    A_fp8: mx.array,
    B_fp8: mx.array,
    *,
    scale_a: mx.array | float,
    scale_b: mx.array | float,
) -> mx.array:
    """Forward-only scaled FP8 matmul kernel call (no autograd).

    ``A_fp8`` is (M, K) uint8 e4m3fn, ``B_fp8`` is (N, K) uint8 e4m3fn (the
    second operand is **transposed**: row j of B is the j-th column of the
    conceptual matmul). The output is (M, N) fp32.

    For autograd-aware use, see :func:`fp8_scaled_matmul`.
    """

    _check_dtype("fp8_scaled_matmul_raw[A]", A_fp8, mx.uint8)
    _check_dtype("fp8_scaled_matmul_raw[B]", B_fp8, mx.uint8)
    if A_fp8.ndim != 2 or B_fp8.ndim != 2:
        raise ValueError(
            f"fp8_scaled_matmul_raw expects 2D inputs; got A.ndim={A_fp8.ndim}, "
            f"B.ndim={B_fp8.ndim}"
        )
    M, K = A_fp8.shape
    N, K_b = B_fp8.shape
    if K != K_b:
        raise ValueError(
            f"fp8_scaled_matmul_raw shape mismatch: A is (M={M}, K={K}), "
            f"B is (N={N}, K={K_b})"
        )

    scale_a_arr, mode_a = _resolve_scale(scale_a, length=M, name="scale_a")
    scale_b_arr, mode_b = _resolve_scale(scale_b, length=N, name="scale_b")
    scale_mode = max(mode_a, mode_b)

    if _FP8_MATMUL_KERNEL is None or not can_run_metal():
        # Fallback: dequantize and use mx.matmul. Maintains numerical contract.
        a_full = mx.from_fp8(A_fp8, dtype=mx.float32)
        b_full = mx.from_fp8(B_fp8, dtype=mx.float32)
        if scale_mode == 0:
            sa = scale_a_arr[0] if scale_a_arr.size == 1 else scale_a_arr
            sb = scale_b_arr[0] if scale_b_arr.size == 1 else scale_b_arr
            return mx.matmul(a_full, mx.swapaxes(b_full, 0, 1)) * sa * sb
        sa = scale_a_arr.reshape(M, 1) if scale_a_arr.size == M else scale_a_arr
        sb = scale_b_arr.reshape(1, N) if scale_b_arr.size == N else scale_b_arr
        return mx.matmul(a_full, mx.swapaxes(b_full, 0, 1)) * sa * sb

    # If one side is per-tensor and the other is per-channel, broadcast the
    # scalar to the per-channel layout so the MSL path can use scale_mode=1.
    if scale_mode == 1:
        if mode_a == 0:
            scale_a_arr = mx.broadcast_to(scale_a_arr, (M,)).astype(mx.float32)
        if mode_b == 0:
            scale_b_arr = mx.broadcast_to(scale_b_arr, (N,)).astype(mx.float32)

    scale_mode_buf = mx.array([scale_mode], dtype=mx.uint32)

    threadgroup = (min(16, max(1, N)), min(16, max(1, M)), 1)
    grid_x = ((N + threadgroup[0] - 1) // threadgroup[0]) * threadgroup[0]
    grid_y = ((M + threadgroup[1] - 1) // threadgroup[1]) * threadgroup[1]
    outputs = dispatch(
        cast(MetalKernel, _FP8_MATMUL_KERNEL),
        inputs=[A_fp8, B_fp8, scale_a_arr, scale_b_arr, scale_mode_buf],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(grid_x, grid_y, 1),
        threadgroup=threadgroup,
    )
    return outputs[0]


def fp8_scaled_vecmat(
    x_fp8: mx.array,
    W_fp8: mx.array,
    *,
    scale_x: mx.array | float,
    scale_w: mx.array | float,
) -> mx.array:
    """Vector x FP8-matrix scaled multiply.

    ``x_fp8`` is (K,) uint8 e4m3fn; ``W_fp8`` is (N, K) uint8 e4m3fn (the
    matrix is **already transposed**: row j is the j-th output projection).
    Returns an (N,) fp32 vector. Uses one SIMD group per output row with
    ``simd_sum`` reduction.

    Forward-only (no autograd VJP). For training-time use, prefer
    :func:`fp8_scaled_matmul` with M=1, which carries a manual VJP.
    """

    _check_dtype("fp8_scaled_vecmat[x]", x_fp8, mx.uint8)
    _check_dtype("fp8_scaled_vecmat[W]", W_fp8, mx.uint8)
    if x_fp8.ndim != 1 or W_fp8.ndim != 2:
        raise ValueError(
            f"fp8_scaled_vecmat expects 1D x and 2D W; got x.ndim={x_fp8.ndim}, "
            f"W.ndim={W_fp8.ndim}"
        )
    (K,) = x_fp8.shape
    N, K_w = W_fp8.shape
    if K != K_w:
        raise ValueError(
            f"fp8_scaled_vecmat shape mismatch: x is (K={K},), W is (N={N}, K={K_w})"
        )
    if K % 4 != 0:
        raise ValueError(
            f"fp8_scaled_vecmat: K must be a multiple of 4 (vectorized 4-byte "
            f"loads); got K={K}"
        )

    scale_x_arr, mode_x = _resolve_scale(scale_x, length=1, name="scale_x")
    scale_w_arr, mode_w = _resolve_scale(scale_w, length=N, name="scale_w")
    scale_mode = mode_w  # x is always per-tensor in this kernel

    if _FP8_VECMAT_KERNEL is None or not can_run_metal():
        # Pure-MLX fallback.
        x_full = mx.from_fp8(x_fp8, dtype=mx.float32)
        w_full = mx.from_fp8(W_fp8, dtype=mx.float32)
        out = mx.matmul(w_full, x_full) * scale_x_arr[0]
        if scale_mode == 0:
            return out * scale_w_arr[0]
        return out * scale_w_arr.reshape(N)

    if scale_mode == 1 and mode_w == 0:
        scale_w_arr = mx.broadcast_to(scale_w_arr, (N,)).astype(mx.float32)
    scale_mode_buf = mx.array([scale_mode], dtype=mx.uint32)

    threads_per_row = 32
    threadgroup = (min(256, threads_per_row * 4), 1, 1)
    total_threads = N * threads_per_row
    grid_x = (
        ((total_threads + threadgroup[0] - 1) // threadgroup[0]) * threadgroup[0]
    )
    outputs = dispatch(
        cast(MetalKernel, _FP8_VECMAT_KERNEL),
        inputs=[x_fp8, W_fp8, scale_x_arr, scale_w_arr, scale_mode_buf],
        output_shapes=[(N,)],
        output_dtypes=[mx.float32],
        grid=(grid_x, 1, 1),
        threadgroup=threadgroup,
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Autograd-aware scaled matmul
# ---------------------------------------------------------------------------


@mx.custom_function
def fp8_scaled_matmul(
    A_fp8: mx.array,
    B_fp8: mx.array,
    scale_a: mx.array,
    scale_b: mx.array,
) -> mx.array:
    """Differentiable scaled FP8 matmul.

    Forward dispatches the LUT-based MSL kernel; backward dequantizes the
    inputs and uses ``mx.matmul`` to produce fp32 gradients (the upstream
    AppMana / audiohacking kernels are forward-only). The straight-through
    estimator path keeps gradients flowing through the FP8 cast at the
    Python boundary -- callers that ingest fp32 inputs typically pair this
    with the same per-tensor amax-based scale used in
    ``cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py``.

    Args:
        A_fp8: (M, K) uint8 e4m3fn.
        B_fp8: (N, K) uint8 e4m3fn (transposed: row j = output column j).
        scale_a: scalar fp32 per-tensor or (M,) fp32 per-row scale.
        scale_b: scalar fp32 per-tensor or (N,) fp32 per-row scale.

    Returns:
        (M, N) fp32 output.
    """

    return fp8_scaled_matmul_raw(
        A_fp8, B_fp8, scale_a=scale_a, scale_b=scale_b
    )


@fp8_scaled_matmul.vjp
def _fp8_scaled_matmul_vjp(primals, cotangent, output):  # noqa: ARG001
    A_fp8, B_fp8, scale_a, scale_b = primals
    # Dequantize once; gradients are fp32. The FP8 cast itself is treated
    # as straight-through (no VJP through the e4m3 quantization step).
    a_full = mx.from_fp8(A_fp8, dtype=mx.float32)
    b_full = mx.from_fp8(B_fp8, dtype=mx.float32)

    sa = scale_a.reshape(-1)
    sb = scale_b.reshape(-1)

    if sa.size == 1:
        a_scaled = a_full * sa[0]
    else:
        a_scaled = a_full * sa.reshape(-1, 1)
    if sb.size == 1:
        b_scaled = b_full * sb[0]
    else:
        b_scaled = b_full * sb.reshape(-1, 1)

    # Forward: C[m, n] = sum_k a_scaled[m, k] * b_scaled[n, k]
    # so dL/da_scaled[m, k] = sum_n cotangent[m, n] * b_scaled[n, k]
    #    dL/db_scaled[n, k] = sum_m cotangent[m, n] * a_scaled[m, k]
    grad_a = mx.matmul(cotangent, b_scaled)  # (M, K)
    grad_b = mx.matmul(mx.swapaxes(cotangent, 0, 1), a_scaled)  # (N, K)

    # FP8 inputs are storage-only; the cotangent into the uint8 buffers is
    # zero-valued (we treat the quantization as a non-differentiable cast
    # at this boundary). The scale gradients come from the chain rule.
    grad_A_fp8 = mx.zeros_like(A_fp8)
    grad_B_fp8 = mx.zeros_like(B_fp8)

    if sa.size == 1:
        grad_scale_a = mx.sum(grad_a * a_full, axis=None).reshape(scale_a.shape)
    else:
        grad_scale_a = mx.sum(grad_a * a_full, axis=1).reshape(scale_a.shape)
    if sb.size == 1:
        grad_scale_b = mx.sum(grad_b * b_full, axis=None).reshape(scale_b.shape)
    else:
        grad_scale_b = mx.sum(grad_b * b_full, axis=1).reshape(scale_b.shape)

    return (grad_A_fp8, grad_B_fp8, grad_scale_a, grad_scale_b)


__all__ = [
    "FP8MSLKernelStatus",
    "__license_notice__",
    "fp8_msl_status",
    "fp8_scaled_matmul",
    "fp8_scaled_matmul_raw",
    "fp8_scaled_vecmat",
    "fp8_to_half",
    "half_to_fp8",
]
