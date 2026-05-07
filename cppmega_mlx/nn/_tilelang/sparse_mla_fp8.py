"""Path B port of cppmega's FP8 sparse-MLA fwd/bwd TileLang pair.

.. note::
    **Wave-6 migration deferred — FP8 factory blocker.** This module's
    fwd/bwd kernels are pure raw-MSL strings (``_FP8_FWD/_BWD_KERNEL_SOURCE``
    → ``_msl_transform.make_metal_kernel``), with no ``@T.prim_func`` to
    feed into ``_engine_dispatch.dispatch_lower``. There is no Path-C
    sister: the kernel's whole point is inline ``e4m3`` byte-level
    dequantisation that has no TileLang DSL equivalent until
    ``simdgroup_a_fp8`` / ``simdgroup_b_fp8`` ``Fragment`` factories
    land in ``tilelang/language/extern.py`` AND Apple ships a
    documented MSL ``float8`` ``simdgroup_matrix`` MMA path.

    Same blocker as :mod:`fp8_msl_kernels` and
    :mod:`fp8_vecmat_path_c` (the latter routes through ``dispatch_lower``
    for non-FP8 surrounds and falls back to Path-B FP8 inner). Track the
    factory work in the migration plan §"FP8 SIMDgroup factories" item.

    Until that lands the raw-MSL kernel here is the only Apple-Metal
    fused FP8 sparse-MLA path. Public API stays stable so the 4 mlx call
    sites keep working unchanged.

Source attribution
------------------

Forward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd_fp8.py
Backward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd_fp8.py
Autograd glue on gb10:
    cppmega/megatron/sparse_mla_ops/sparse_mla.py (class SparseMLA_FP8)

These come from the upstream NVIDIA Megatron-LM PR #3674 (DSA "thd" branch),
in turn ported from tile-ai/tilelang/examples/deepseek_v32/. Q and KV ride in
torch.float8_e4m3fn with per-token FP32 scale factors; the kernel dequantizes
acc_s by ``q_scale * kv_scale`` after every Q@K tile.

Status on Apple Metal (M-series)
--------------------------------

Apple's Metal Shading Language 4.0 does not expose a native ``float8_e4m3``
type or a ``simdgroup_matrix<float8>`` MMA path. M4 Pro/Max silicon has FP8
hardware but no documented MSL surface as of Metal 4.0 / MLX 0.31. We
therefore can't dispatch a "real" FP8 simdgroup MMA from MSL.

Direct-MSL bypass (this module)
-------------------------------

Instead of waiting on hardware FP8 in MSL, we implement the FP8 forward
and backward by:

1. Storing the quantized Q and KV tensors as ``uint8`` arrays carrying
   ``e4m3`` bit patterns (the same storage layout ``mx.to_fp8`` produces).
2. Dequantizing those bytes inline inside MSL using a closed-form bit
   manipulation of the e4m3 format (sign, 4-bit exponent biased 7,
   3-bit mantissa) into fp32, multiplying by the per-token scale.
3. Performing the rest of the attention math (Q@K, softmax, S@V) at fp32
   accumulator with fp16 carrier for the output tensors. This matches the
   numerical contract of the gb10 kernel which also keeps ``acc_s`` in fp32
   and emits BF16 output.

The backward dequantizes Q/KV the same way and runs the same gradient flow
as the BF16 sparse-MLA backward. Gradients from FP8 inputs are propagated
through MLX autograd's straight-through estimator (the ``mx.from_fp8`` op
has no autograd VJP in MLX 0.31, so we apply STE explicitly when needed).

Numerical behavior
~~~~~~~~~~~~~~~~~~

* **Forward parity vs the BF16 reference**: at std=0.1 inputs the rtol=5e-3
  / atol=5e-3 tolerance from the task brief is met. At larger inputs the
  e4m3 mantissa noise dominates; tests use small-magnitude inputs.
* **Backward parity vs the BF16 reference (over recovered tensors)**:
  rtol=1e-2 / atol=5e-3 (FP8 noise tolerance) is met.
* **Speed**: the per-token dequant step adds ~1-2 ALU ops per byte read.
  Bench numbers in ``bench/tilelang_ports/sparse_mla_fp8.json`` show this
  is comparable to the BF16 direct-MSL kernel (within 20%) because the
  Q@K matmul time dominates over the dequant.

bf16 vs fp16 carrier note
~~~~~~~~~~~~~~~~~~~~~~~~~

We emit fp16 outputs and accumulate in fp32. The dequantized Q/KV values
go through fp32 ALUs inside MSL (no simdgroup_matrix involvement) so the
bf16 simdgroup miscompiles do not affect this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._msl_transform import (
    can_run_metal,
    msl_dispatch_status,
)
from cppmega_mlx.nn.sparse_mla import (
    SparseMLAShapes,
    _resolve_shapes,
    sparse_mla_attention_reference,
)


# ---------------------------------------------------------------------------
# Public status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMLAFp8MetalStatus:
    """Runtime status of the Path B FP8 sparse-MLA kernel."""

    available: bool
    reason: str
    fp8_dtype: str = "float8_e4m3"


_DIRECT_MSL_OK_REASON = (
    "sparse_mla_fp8 direct-MSL kernel built via mx.fast.metal_kernel is "
    "available; FP8 e4m3 dequant happens inline inside MSL on uint8 storage. "
    "Apple MSL 4.0 has no native float8 simdgroup matrix, so the matmuls run "
    "as plain register fma ops (still ~2x faster than the BF16-fallback "
    "reference because the dequant fuses with the QK loop)."
)
_DIRECT_MSL_BLOCKER_REASON = (
    "sparse_mla_fp8 direct-MSL kernel could not be constructed: "
    "mx.fast.metal_kernel is unavailable on this device."
)


# ---------------------------------------------------------------------------
# FP8 helpers (same ABI as the original module)
# ---------------------------------------------------------------------------


def _to_fp8_with_per_tensor_scale(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Cast ``x`` to FP8 (e4m3) with a per-tensor amax-based scale.

    Returns (fp8_uint8, scale_f32) with scale broadcasting to ``x.shape[:-1]``.
    """

    if x.size == 0:
        scale_shape = x.shape[:-1] if x.ndim > 1 else (1,)
        return mx.zeros(x.shape, dtype=mx.uint8), mx.ones(scale_shape, dtype=mx.float32)

    x_f32 = x.astype(mx.float32)
    amax = mx.max(mx.abs(x_f32))
    scale = mx.maximum(amax / mx.array(448.0, dtype=mx.float32), mx.array(1e-12, dtype=mx.float32))
    x_scaled = x_f32 / scale
    fp8 = mx.to_fp8(x_scaled)

    if x.ndim >= 2:
        scale_per_token = mx.broadcast_to(scale.reshape((1,) * (x.ndim - 1)), x.shape[:-1])
    else:
        scale_per_token = scale.reshape((1,))
    return fp8, scale_per_token.astype(mx.float32)


def _from_fp8_with_scale(
    fp8: mx.array,
    scale: mx.array,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize an FP8 tensor with per-token scale back to the given dtype."""

    rec = mx.from_fp8(fp8, dtype=mx.float32)
    rec = rec * scale[..., None].astype(mx.float32)
    return rec.astype(dtype)


@mx.custom_function
def _fp8_roundtrip_ste(x: mx.array) -> mx.array:
    """Quantize-then-dequantize FP8 roundtrip with straight-through gradients."""

    fp8, scale = _to_fp8_with_per_tensor_scale(x)
    rec = _from_fp8_with_scale(fp8, scale, dtype=x.dtype)
    return rec


@_fp8_roundtrip_ste.vjp
def _fp8_roundtrip_ste_vjp(primals, cotangent, output):  # noqa: ARG001
    return (cotangent.astype(primals[0].dtype),)


# ---------------------------------------------------------------------------
# FP8 e4m3 inline dequant helper for MSL
# ---------------------------------------------------------------------------
#
# e4m3 format: sign (1 bit) | exponent (4 bits, bias 7) | mantissa (3 bits)
# Special values:
#   exp=0xF, mant=0x7 -> NaN (-NaN)
#   exp=0, mant=0     -> +0/-0
#   exp=0, mant!=0    -> subnormal: value = (-1)^s * 2^(-6) * (mant / 8)
#   exp!=0            -> normal: value = (-1)^s * 2^(exp-7) * (1 + mant / 8)
#
# Float32 IEEE: sign | exponent (8 bits, bias 127) | mantissa (23 bits)
#
# Manual decode: (mirror of the GPU-friendly routine in TVM/cuda fp8 codegen)
#   sign = (b >> 7) & 1
#   exp  = (b >> 3) & 0xF
#   mant = b & 0x7
#   if (exp == 0 && mant == 0): result = ±0
#   if (exp == 0 && mant != 0): result = (-1)^s * 2^(-6) * (mant / 8.0)
#   if (exp == 0xF && mant == 0x7): result = NaN -> we map to +/- amax (clamp)
#   else: exp32 = exp + (127 - 7) = exp + 120
#         mant32 = mant << (23 - 3) = mant << 20
#         result = bitcast<float>(sign << 31 | exp32 << 23 | mant32)
#
# We inline this as a Metal function header.

_FP8_DEQUANT_HEADER = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fp8_e4m3_to_float(uchar b) {
        uint sign = (b >> 7) & 0x1u;
        uint exp_bits  = (b >> 3) & 0xFu;
        uint mant = b & 0x7u;
        uint result_bits;
        if (exp_bits == 0u) {
            if (mant == 0u) {
                // Signed zero.
                result_bits = sign << 31;
                return as_type<float>(result_bits);
            } else {
                // Subnormal: value = (-1)^s * 2^(-6) * (mant / 8)
                float v = float(mant) * (1.0f / 8.0f) * 0.015625f; // 2^-6 = 0.015625
                return sign ? -v : v;
            }
        } else if (exp_bits == 0xFu && mant == 0x7u) {
            // NaN -> map to amax to be benign in matmul reductions.
            return sign ? -448.0f : 448.0f;
        } else {
            uint exp32 = exp_bits + 120u; // bias adjust 127-7
            uint mant32 = mant << 20u;
            result_bits = (sign << 31) | (exp32 << 23) | mant32;
            return as_type<float>(result_bits);
        }
    }
"""


# ---------------------------------------------------------------------------
# Forward MSL kernel.
# ---------------------------------------------------------------------------
#
# Input layout:
#   q_fp8        [B, S, H, D_qk]    uint8   (e4m3 bit pattern)
#   q_scale      [B, S, H]          fp32    (per-token scale)
#   kv_fp8       [B, Skv, G, D_qk]  uint8
#   kv_scale     [B, Skv, G]        fp32
#   indices      [B, S, G, topk]    int32
#
# Outputs:
#   out          [B, S, H, D_v]     fp16
#   lse          [B, S, H]          fp32
#
# Effective dequant: q_dequant[..., d] = fp8_to_float(q_fp8[..., d]) * q_scale
#                    kv_dequant[..., d] = fp8_to_float(kv_fp8[..., d]) * kv_scale

_FP8_FWD_KERNEL_SOURCE = """
    threadgroup float scores[TOPK];
    threadgroup float reduce_buf[BLOCK_SIZE];

    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint threads = BLOCK_SIZE;

    uint h = gid % uint(HEADS);
    uint s = (gid / uint(HEADS)) % uint(SEQ_LEN);
    uint b = gid / (uint(HEADS) * uint(SEQ_LEN));
    if (b >= uint(BATCH)) {
        return;
    }

    uint kv_group = uint(KV_GROUP);
    uint head_kv = uint(HEAD_KV);
    uint g = h / head_kv;

    uint qk_dim = uint(QK_DIM);
    uint d_v = uint(D_V);

    uint q_row_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * qk_dim;
    uint kv_outer_stride = uint(SEQ_LEN_KV) * kv_group * qk_dim;
    uint kv_b_base = b * kv_outer_stride;
    uint idx_base = ((b * uint(SEQ_LEN) + s) * kv_group + g) * uint(TOPK);

    float qs = float(q_scale[(b * uint(SEQ_LEN) + s) * uint(HEADS) + h]);
    float sm_scale = float(sm_scale_buf[0]);

    // Phase 1: scores[k] = sm_scale * q_scale * kv_scale * sum_d q_fp8[d] * kv_fp8[k,d]
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            scores[k] = -INFINITY;
            continue;
        }
        uint kv_pos = uint(gather_idx);
        uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
        float ks = float(kv_scale[(b * uint(SEQ_LEN_KV) + kv_pos) * kv_group + g]);
        float acc = 0.0f;
        for (uint d = 0; d < qk_dim; ++d) {
            float qv = fp8_e4m3_to_float(q_fp8[q_row_base + d]);
            float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]);
            acc += qv * kv_v;
        }
        scores[k] = acc * (qs * ks) * sm_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: max reduction.
    float local_max = -INFINITY;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        if (scores[k] > local_max) local_max = scores[k];
    }
    reduce_buf[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = reduce_buf[tid];
            float b_v = reduce_buf[tid + stride];
            if (b_v > a) reduce_buf[tid] = b_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m_i = reduce_buf[0];

    if (m_i == -INFINITY) {
        for (uint d = tid; d < d_v; d += threads) {
            uint out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
            out[out_row + d] = T_OUT(0);
        }
        if (tid == 0) {
            uint lse_idx = (b * uint(SEQ_LEN) + s) * uint(HEADS) + h;
            lse[lse_idx] = 0.0f;
        }
        return;
    }

    // Phase 3: exp + sum.
    for (uint k = tid; k < uint(TOPK); k += threads) {
        float v = scores[k];
        scores[k] = (v == -INFINITY) ? 0.0f : exp(v - m_i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        local_sum += scores[k];
    }
    reduce_buf[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sumexp = reduce_buf[0];
    float inv_sum = (sumexp > 0.0f) ? (1.0f / sumexp) : 0.0f;

    // Phase 4: output[d] = sum_k p[k] * kv[k, d] * kv_scale[k] / sumexp
    uint out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
    for (uint d = tid; d < d_v; d += threads) {
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            int gather_idx = indices[idx_base + k];
            if (gather_idx < 0) continue;
            uint kv_pos = uint(gather_idx);
            uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
            float ks = float(kv_scale[(b * uint(SEQ_LEN_KV) + kv_pos) * kv_group + g]);
            float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]) * ks;
            acc += scores[k] * kv_v;
        }
        out[out_row + d] = T_OUT(acc * inv_sum);
    }

    if (tid == 0) {
        uint lse_idx = (b * uint(SEQ_LEN) + s) * uint(HEADS) + h;
        lse[lse_idx] = m_i + log(sumexp);
    }
"""


_FP8_FWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_sparse_mla_fp8_fwd",
    input_names=[
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
        "indices",
        "sm_scale_buf",
    ],
    output_names=["out", "lse"],
    source=_FP8_FWD_KERNEL_SOURCE,
    header=_FP8_DEQUANT_HEADER,
)


# ---------------------------------------------------------------------------
# Backward MSL kernel.
# ---------------------------------------------------------------------------
#
# Backward output is in fp32 dequantized space (gradients flow back through
# the dequant cast as straight-through). The host then re-quantizes if it
# wants FP8 grads, but typically gradients stay in fp32 for optimizer steps.

_FP8_BWD_KERNEL_SOURCE = """
    threadgroup float scores[TOPK];
    threadgroup float p[TOPK];
    threadgroup float dp[TOPK];
    threadgroup float ds[TOPK];
    threadgroup float reduce_buf[BLOCK_SIZE];

    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint threads = BLOCK_SIZE;

    uint h = gid % uint(HEADS);
    uint s = (gid / uint(HEADS)) % uint(SEQ_LEN);
    uint b = gid / (uint(HEADS) * uint(SEQ_LEN));
    if (b >= uint(BATCH)) {
        return;
    }

    uint kv_group = uint(KV_GROUP);
    uint head_kv = uint(HEAD_KV);
    uint g = h / head_kv;

    uint qk_dim = uint(QK_DIM);
    uint d_v = uint(D_V);

    uint q_row_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * qk_dim;
    uint kv_outer_stride = uint(SEQ_LEN_KV) * kv_group * qk_dim;
    uint kv_b_base = b * kv_outer_stride;
    uint idx_base = ((b * uint(SEQ_LEN) + s) * kv_group + g) * uint(TOPK);

    float qs = float(q_scale[(b * uint(SEQ_LEN) + s) * uint(HEADS) + h]);
    float sm_scale = float(sm_scale_buf[0]);

    // Recompute scores.
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            scores[k] = -INFINITY;
            continue;
        }
        uint kv_pos = uint(gather_idx);
        uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
        float ks = float(kv_scale[(b * uint(SEQ_LEN_KV) + kv_pos) * kv_group + g]);
        float acc = 0.0f;
        for (uint d = 0; d < qk_dim; ++d) {
            float qv = fp8_e4m3_to_float(q_fp8[q_row_base + d]);
            float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]);
            acc += qv * kv_v;
        }
        scores[k] = acc * (qs * ks) * sm_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // m_i.
    float local_max = -INFINITY;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        if (scores[k] > local_max) local_max = scores[k];
    }
    reduce_buf[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = reduce_buf[tid];
            float b_v = reduce_buf[tid + stride];
            if (b_v > a) reduce_buf[tid] = b_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m_i = reduce_buf[0];

    for (uint k = tid; k < uint(TOPK); k += threads) {
        float v = scores[k];
        p[k] = (v == -INFINITY) ? 0.0f : exp(v - m_i);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        local_sum += p[k];
    }
    reduce_buf[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sumexp = reduce_buf[0];

    if (sumexp <= 0.0f) {
        for (uint d = tid; d < qk_dim; d += threads) {
            dq_dequant[q_row_base + d] = 0.0f;
        }
        uint dkv_pb = (((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * uint(TOPK)) * qk_dim;
        for (uint k_off = tid; k_off < uint(TOPK) * qk_dim; k_off += threads) {
            dkv_partial[dkv_pb + k_off] = 0.0f;
        }
        return;
    }
    float inv_sum = 1.0f / sumexp;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        p[k] = p[k] * inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dp[k] = sum_d V[k, d] * d_out[d]
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            dp[k] = 0.0f;
            continue;
        }
        uint kv_pos = uint(gather_idx);
        uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
        float ks = float(kv_scale[(b * uint(SEQ_LEN_KV) + kv_pos) * kv_group + g]);
        float acc = 0.0f;
        uint d_out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
        for (uint d = 0; d < d_v; ++d) {
            float v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]) * ks;
            float dod = float(d_out[d_out_row + d]);
            acc += v * dod;
        }
        dp[k] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_rs = 0.0f;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        local_rs += p[k] * dp[k];
    }
    reduce_buf[tid] = local_rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowsum = reduce_buf[0];

    for (uint k = tid; k < uint(TOPK); k += threads) {
        ds[k] = p[k] * (dp[k] - rowsum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dQ_dequant[dd] = sm_scale * sum_k ds[k] * (kv_dequant_with_scale[k, dd])
    // Note: ds[k] already includes p[k] which depends on the FP8-recovered
    // scores; the multiplication chain stays in fp32.
    for (uint dd = tid; dd < qk_dim; dd += threads) {
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            int gather_idx = indices[idx_base + k];
            if (gather_idx < 0) continue;
            uint kv_pos = uint(gather_idx);
            uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
            float ks = float(kv_scale[(b * uint(SEQ_LEN_KV) + kv_pos) * kv_group + g]);
            float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + dd]) * ks;
            acc += ds[k] * kv_v;
        }
        dq_dequant[q_row_base + dd] = acc * sm_scale;
    }

    // dKV_dequant partial: per-(b, s, h, k, d)
    uint dkv_pb = (((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * uint(TOPK)) * qk_dim;
    for (uint kd = tid; kd < uint(TOPK) * qk_dim; kd += threads) {
        uint k = kd / qk_dim;
        uint d = kd % qk_dim;
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            dkv_partial[dkv_pb + kd] = 0.0f;
            continue;
        }
        float qv = fp8_e4m3_to_float(q_fp8[q_row_base + d]) * qs;
        float ks_q = sm_scale * ds[k] * qv;
        if (d < d_v) {
            uint d_out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
            float dod = float(d_out[d_out_row + d]);
            dkv_partial[dkv_pb + kd] = p[k] * dod + ks_q;
        } else {
            dkv_partial[dkv_pb + kd] = ks_q;
        }
    }
"""


_FP8_BWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_sparse_mla_fp8_bwd",
    input_names=[
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
        "indices",
        "d_out",
        "sm_scale_buf",
    ],
    output_names=["dq_dequant", "dkv_partial"],
    source=_FP8_BWD_KERNEL_SOURCE,
    header=_FP8_DEQUANT_HEADER,
)


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------


def sparse_mla_fp8_metal_status(*arrays: mx.array) -> SparseMLAFp8MetalStatus:
    """Return whether the Path B FP8 kernel is currently dispatchable."""

    if not can_run_metal():
        return SparseMLAFp8MetalStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    if _FP8_FWD_KERNEL is None or _FP8_BWD_KERNEL is None:
        return SparseMLAFp8MetalStatus(available=False, reason=_DIRECT_MSL_BLOCKER_REASON)
    float_arrays = [a for a in arrays if a.dtype in (mx.float16, mx.float32, mx.bfloat16)]
    if float_arrays:
        runtime = msl_dispatch_status(*float_arrays)
        if not runtime.available:
            return SparseMLAFp8MetalStatus(available=False, reason=runtime.reason)
    return SparseMLAFp8MetalStatus(available=True, reason=_DIRECT_MSL_OK_REASON)


# ---------------------------------------------------------------------------
# Forward Metal entry point (FP8-storage inputs)
# ---------------------------------------------------------------------------


def sparse_mla_fp8_fwd_metal_impl(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int,
    shapes: SparseMLAShapes,
) -> Tuple[mx.array, mx.array] | None:
    """Direct-MSL FP8 forward; takes pre-quantized inputs.

    Args:
        q_fp8: uint8 [B, S, H, D_qk] (e4m3 bytes)
        q_scale: fp32 [B, S, H]
        kv_fp8: uint8 [B, Skv, G, D_qk]
        kv_scale: fp32 [B, Skv, G]
        indices: int32 [B, S, G, topk]

    Returns:
        (out_fp16, lse_fp32) or None.
    """

    if _FP8_FWD_KERNEL is None or not can_run_metal():
        return None

    threads = min(64, max(1, shapes.topk))
    p = 1
    while (p << 1) <= threads:
        p <<= 1
    threads = p
    if threads < 1:
        threads = 1

    template = [
        ("T_OUT", mx.float16),
        ("BATCH", shapes.batch),
        ("SEQ_LEN", shapes.seq_len),
        ("SEQ_LEN_KV", shapes.seq_len_kv),
        ("HEADS", shapes.heads),
        ("HEAD_KV", shapes.head_kv),
        ("KV_GROUP", shapes.kv_group),
        ("QK_DIM", shapes.qk_dim),
        ("D_V", shapes.d_v),
        ("TOPK", shapes.topk),
        ("BLOCK_SIZE", threads),
    ]

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    grid_x = shapes.batch * shapes.seq_len * shapes.heads
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, _FP8_FWD_KERNEL),
        inputs=[
            q_fp8.astype(mx.uint8),
            q_scale.astype(mx.float32),
            kv_fp8.astype(mx.uint8),
            kv_scale.astype(mx.float32),
            indices.astype(mx.int32),
            sm_scale_buf,
        ],
        output_shapes=[
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.d_v),
            (shapes.batch, shapes.seq_len, shapes.heads),
        ],
        output_dtypes=[mx.float16, mx.float32],
        grid=(grid_x * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        template=template,
    )
    out, lse = outputs
    return out, lse


def sparse_mla_fp8_bwd_metal_impl(
    q_fp8: mx.array,
    q_scale: mx.array,
    kv_fp8: mx.array,
    kv_scale: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float,
    d_v: int,
    shapes: SparseMLAShapes,
) -> Tuple[mx.array, mx.array] | None:
    """Direct-MSL FP8 backward; returns dequantized gradients.

    Returns:
        (dq_dequant, dkv_dequant) in fp32. The host uses STE to map these
        back to the upstream input tensors.
    """

    if _FP8_BWD_KERNEL is None or not can_run_metal():
        return None

    threads = min(64, max(1, shapes.topk))
    p = 1
    while (p << 1) <= threads:
        p <<= 1
    threads = p
    if threads < 1:
        threads = 1

    template = [
        ("BATCH", shapes.batch),
        ("SEQ_LEN", shapes.seq_len),
        ("SEQ_LEN_KV", shapes.seq_len_kv),
        ("HEADS", shapes.heads),
        ("HEAD_KV", shapes.head_kv),
        ("KV_GROUP", shapes.kv_group),
        ("QK_DIM", shapes.qk_dim),
        ("D_V", shapes.d_v),
        ("TOPK", shapes.topk),
        ("BLOCK_SIZE", threads),
    ]

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    grid_x = shapes.batch * shapes.seq_len * shapes.heads
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, _FP8_BWD_KERNEL),
        inputs=[
            q_fp8.astype(mx.uint8),
            q_scale.astype(mx.float32),
            kv_fp8.astype(mx.uint8),
            kv_scale.astype(mx.float32),
            indices.astype(mx.int32),
            d_out.astype(mx.float32),
            sm_scale_buf,
        ],
        output_shapes=[
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.qk_dim),
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.topk, shapes.qk_dim),
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(grid_x * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        template=template,
    )
    dq_dequant, dkv_partial = outputs
    return dq_dequant, dkv_partial


# ---------------------------------------------------------------------------
# Public Path B forward / backward (matches BF16 sibling API)
# ---------------------------------------------------------------------------


def sparse_mla_fp8_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Quantize Q/KV to FP8 (e4m3 with per-tensor scale) and run Path B fwd.

    Returns ``(out_fp16, lse_fp32)`` or ``None`` if Metal is unavailable.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5
    status = sparse_mla_fp8_metal_status(q, kv, indices)
    if not status.available:
        return None

    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    return sparse_mla_fp8_fwd_metal_impl(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        indices,
        sm_scale=sm_scale,
        d_v=shapes.d_v,
        shapes=shapes,
    )


def sparse_mla_fp8_bwd_metal(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Quantize Q/KV and run the FP8 backward; returns dequantized gradients.

    Returns ``(dq, dkv)`` in fp32 (caller should cast to upstream dtype).
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5
    status = sparse_mla_fp8_metal_status(q, kv, indices)
    if not status.available:
        return None

    q_fp8, q_scale = _to_fp8_with_per_tensor_scale(q)
    kv_fp8, kv_scale = _to_fp8_with_per_tensor_scale(kv)
    metal_grads = sparse_mla_fp8_bwd_metal_impl(
        q_fp8,
        q_scale,
        kv_fp8,
        kv_scale,
        d_out,
        indices,
        sm_scale=sm_scale,
        d_v=shapes.d_v,
        shapes=shapes,
    )
    if metal_grads is None:
        return None
    dq, dkv_partial = metal_grads

    # Reduce dkv_partial -> dkv. Use the same scatter as the BF16 backward.
    indices_i32 = indices.astype(mx.int32)
    dkv = _reduce_dkv_partial_fp32(dkv_partial, indices_i32, shapes)
    return dq, dkv


def _reduce_dkv_partial_fp32(
    dkv_partial: mx.array, indices_i32: mx.array, shapes
) -> mx.array:
    """Scatter-add per-token dkv partials into [B, Skv, G, qk_dim] fp32."""

    B = shapes.batch
    S = shapes.seq_len
    Skv = shapes.seq_len_kv
    G = shapes.kv_group
    head_kv = shapes.head_kv
    topk = shapes.topk
    qk_dim = shapes.qk_dim

    dkv_per_group = dkv_partial.reshape(B, S, G, head_kv, topk, qk_dim).sum(axis=3)

    valid_mask = indices_i32 != -1  # [B, S, G, topk]
    safe_idx = mx.maximum(indices_i32, mx.array(0, dtype=mx.int32))
    dkv_per_group_masked = mx.where(
        valid_mask[..., None], dkv_per_group, mx.zeros_like(dkv_per_group)
    )

    batch_idx = mx.arange(B, dtype=mx.int32).reshape(B, 1, 1, 1)
    batch_idx = mx.broadcast_to(batch_idx, (B, S, G, topk))
    group_idx = mx.arange(G, dtype=mx.int32).reshape(1, 1, G, 1)
    group_idx = mx.broadcast_to(group_idx, (B, S, G, topk))

    dkv_dst = mx.zeros((B, Skv, G, qk_dim), dtype=mx.float32)
    flat_b = batch_idx.reshape(-1)
    flat_kv = safe_idx.reshape(-1)
    flat_g = group_idx.reshape(-1)
    flat_vals = dkv_per_group_masked.reshape(-1, qk_dim)
    dkv_dst = dkv_dst.at[flat_b, flat_kv, flat_g].add(flat_vals)
    return dkv_dst


# ---------------------------------------------------------------------------
# Reference pure-MLX path (kept for parity tests + fallback)
# ---------------------------------------------------------------------------


def sparse_mla_fp8_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Pure-MLX FP8 reference: roundtrip through ``mx.to_fp8``/``from_fp8`` + BF16 ref."""

    q_recovered = _fp8_roundtrip_ste(q)
    kv_recovered = _fp8_roundtrip_ste(kv)
    return sparse_mla_attention_reference(
        q_recovered,
        kv_recovered,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


def sparse_mla_quantized_matmul_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Hand-built sparse-MLA forward using ``mx.matmul`` on dequantized fp32."""

    shapes: SparseMLAShapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    qk_dim = shapes.qk_dim
    d_v_resolved = shapes.d_v
    if sm_scale is None:
        sm_scale = qk_dim ** -0.5

    indices_i32 = indices.astype(mx.int32)
    safe_indices = mx.maximum(indices_i32, mx.array(0, dtype=mx.int32))
    batch_idx = mx.arange(shapes.batch, dtype=mx.int32).reshape(shapes.batch, 1, 1, 1)
    batch_idx = mx.broadcast_to(batch_idx, indices_i32.shape)
    group_idx = mx.arange(shapes.kv_group, dtype=mx.int32).reshape(1, 1, shapes.kv_group, 1)
    group_idx = mx.broadcast_to(group_idx, indices_i32.shape)
    gathered = kv[batch_idx, safe_indices, group_idx]

    valid = (indices_i32 != -1)[:, :, :, None, :]

    q_grouped = q.reshape(shapes.batch, shapes.seq_len, shapes.kv_group, shapes.head_kv, qk_dim)
    q_f32 = q_grouped.astype(mx.float32)
    kv_f32 = gathered.astype(mx.float32)

    scores = mx.matmul(q_f32, mx.swapaxes(kv_f32, -1, -2)) * sm_scale
    neg_inf = mx.array(-mx.inf, dtype=mx.float32)
    scores = mx.where(valid, scores, neg_inf)
    m_i = mx.max(scores, axis=-1, keepdims=True)
    has_any_valid = mx.any(indices_i32 != -1, axis=-1, keepdims=True)[:, :, :, None, :]
    m_i_clean = mx.where(has_any_valid, m_i, mx.zeros_like(m_i))
    exp_scores = mx.exp(scores - m_i_clean)
    exp_scores = mx.where(valid, exp_scores, mx.zeros_like(exp_scores))
    sumexp = mx.sum(exp_scores, axis=-1, keepdims=True)
    safe_sumexp = mx.where(sumexp > 0, sumexp, mx.ones_like(sumexp))
    probs = exp_scores / safe_sumexp
    v_f32 = kv_f32[..., :d_v_resolved]
    out_f32 = mx.matmul(probs, v_f32)
    out_f32 = out_f32 * has_any_valid.astype(mx.float32)

    out = out_f32.reshape(shapes.batch, shapes.seq_len, shapes.heads, d_v_resolved).astype(q.dtype)

    if not return_lse:
        return out
    lse = (m_i_clean + mx.log(safe_sumexp)).reshape(shapes.batch, shapes.seq_len, shapes.heads)
    return out, lse


# ---------------------------------------------------------------------------
# Custom VJP wrapper
# ---------------------------------------------------------------------------


@mx.custom_function
def sparse_mla_fp8_metal_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
) -> mx.array:
    """Forward-only differentiable wrapper around the FP8 direct-MSL kernel."""

    result = sparse_mla_fp8_fwd_metal(q, kv, indices)
    if result is None:
        return sparse_mla_fp8_reference(q, kv, indices)
    out, _lse = result
    return out


@sparse_mla_fp8_metal_apply.vjp
def _sparse_mla_fp8_metal_apply_vjp(primals, cotangent, output):
    del output
    q, kv, indices = primals
    grads = sparse_mla_fp8_bwd_metal(q, kv, cotangent, indices)
    if grads is None:
        def _ref_apply(q_, kv_):
            return sparse_mla_fp8_reference(q_, kv_, indices)
        _, vjps = mx.vjp(_ref_apply, (q, kv), (cotangent,))
        return (vjps[0], vjps[1], mx.zeros_like(indices))
    dq, dkv = grads
    return (dq.astype(q.dtype), dkv.astype(kv.dtype), mx.zeros_like(indices))


def sparse_mla_fp8_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_metal: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Apply FP8 sparse MLA. Prefers the direct-MSL kernel.

    Note (Path C status — REDUCERS-ONLY):
        There is intentionally no ``sparse_mla_fp8_path_c_apply`` counterpart.
        ``sparse_mla_fp8_path_c.py`` exposes only QK / indexed-QK reducer
        surfaces, not an end-to-end attention apply. Therefore there is no
        ``force_metal`` -> ``force_path_c`` kwarg rename for FP8 sparse-MLA;
        the only callable wrapper is this Path B one and it keeps
        ``force_metal``. The FP8 Path C reducers are also currently broken at
        runtime — the ``tirx.metal.fp8_e4m3_dot4`` intrinsic is not registered
        in the in-tree TileLang/TVM build (agent-D report
        ``reports/2026-05-06-tilelang-tvm-review/agent-D-planning-vs-reality/grok__design__20260506T171408.md``
        finding #1). See ``docs/production_kernel_routing.md``.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    if return_lse:
        result = sparse_mla_fp8_fwd_metal(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            if force_metal:
                raise RuntimeError(
                    "sparse_mla_fp8_apply: Metal path unavailable: "
                    f"{sparse_mla_fp8_metal_status(q, kv, indices).reason}"
                )
            return sparse_mla_fp8_reference(
                q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=True
            )
        out, lse = result
        return out.astype(q.dtype), lse

    status = sparse_mla_fp8_metal_status(q, kv, indices)
    if not status.available:
        if force_metal:
            raise RuntimeError(
                f"sparse_mla_fp8_apply: Metal path unavailable: {status.reason}"
            )
        return sparse_mla_fp8_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False
        )

    is_default = (
        d_v is None or d_v == shapes.qk_dim
    ) and abs(sm_scale - shapes.qk_dim ** -0.5) < 1e-9
    if is_default:
        out = sparse_mla_fp8_metal_apply(q, kv, indices)
        return out.astype(q.dtype)

    return sparse_mla_fp8_reference(
        q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False
    )


__all__ = [
    "SparseMLAFp8MetalStatus",
    "sparse_mla_fp8_apply",
    "sparse_mla_fp8_bwd_metal",
    "sparse_mla_fp8_fwd_metal",
    "sparse_mla_fp8_metal_apply",
    "sparse_mla_fp8_metal_status",
    "sparse_mla_fp8_reference",
    "sparse_mla_quantized_matmul_reference",
]
