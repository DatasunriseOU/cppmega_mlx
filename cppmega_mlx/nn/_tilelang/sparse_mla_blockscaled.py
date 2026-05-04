"""Path B port of cppmega's block-scaled (MXFP8) sparse-MLA fwd/bwd TileLang pair.

Source attribution
------------------

Forward + backward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_fused.py
    (functions ``sparse_mla_blockscaled_mxfp8_fwd`` and
    ``sparse_mla_blockscaled_mxfp8_bwd_kernel``)

A separate experimental QK-only block-scaled scoring helper lives in
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py

Both consume MXFP8 layouts where the data tensor is a torch.float8_e4m3fn
matrix and a separate FP32 scale tensor with one scalar per 32-element block
along the last (head) axis::

    q_data:   [B, S,  H, D_total]      torch.float8_e4m3fn
    kv_data:  [B, SK, G, D_total]      torch.float8_e4m3fn
    q_scale:  [B, S,  H, D_total/32]   torch.float32
    kv_scale: [B, SK, G, D_total/32]   torch.float32
    indices:  [B, S,  G, topk]         torch.int32, -1 sentinel

The kernel walks the head dim in 32-element blocks (``MXFP8_BLOCK_SIZE = 32``),
runs ``T.gemm(q_block, kv_block, partial)`` for each block, and accumulates
``partial[h, j] * QScale[h, kb] * KVScale[j, kb]`` into ``acc_s``. After the
block walk it runs the standard online-softmax + S@V flow used by the BF16
sparse-MLA forward, with V dequantized from FP8 by ``KVScale[d/32]``.

Direct-MSL bypass (this module)
-------------------------------

Same approach as the tensorwise FP8 sibling: skip TileLang and emit MSL
directly through ``mx.fast.metal_kernel``. The block-scaled layout is
implemented with two extra inputs (``q_scale_block``, ``kv_scale_block``)
of shape ``[..., D_qk / 32]`` indexed by ``d / BLOCK_SIZE`` inside the MSL
QK loop. The MXFP8 storage is ``uint8`` e4m3 bytes (the same byte pattern
as the tensorwise FP8 module), and dequant uses the same ``fp8_e4m3_to_float``
inline function.

Apple's mxfp8 ``mx.quantize(mode='mxfp8')`` returns a packed ``uint32`` tensor
where each uint32 holds 4 fp8 e4m3 bytes plus the scale tensor as a separate
``uint8`` E8M0 (scale exponents). For the direct MSL kernel we expand the
packed uint32 back into raw uint8 e4m3 bytes (cheap, host-side single
``.view(mx.uint8)``-style cast) so the MSL inner loop can iterate one byte
at a time. The block scales arrive as ``uint8`` (E8M0 representation):
the MSL kernel decodes them via ``ldexp(1.0, int(scale_byte) - 127)``.

Numerical contract:

* ``acc_s[h, k] = sum_kb partial[h, k, kb] * Q_blk_scale[h, kb] * KV_blk_scale[k, kb]``
* fp16 carrier for outputs, fp32 internal accumulators.
* Block size 32 along the head dim (matches gb10).
* Inputs with non-multiple-of-32 head_dim fall back to BF16 reference.
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
from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
    blockscaled_sparse_mla_qk_path_c_status,
)
from cppmega_mlx.nn.sparse_mla import (
    SparseMLAShapes,
    _resolve_shapes,
    sparse_mla_attention_reference,
)


MXFP8_BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# Public status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMLABlockScaledMetalStatus:
    """Runtime status of the Path B block-scaled sparse-MLA kernel."""

    available: bool
    reason: str
    block_size: int = MXFP8_BLOCK_SIZE


_DIRECT_MSL_OK_REASON = (
    "sparse_mla_blockscaled direct-MSL kernel built via mx.fast.metal_kernel "
    "is available; block-scaled MXFP8 dequant happens inline inside MSL on "
    "uint8 e4m3 storage with E8M0 block-scales. Apple MSL 4.0 has no native "
    "float8 simdgroup matrix, so the matmuls run as plain register fma."
)
_DIRECT_MSL_BLOCKER_REASON = (
    "sparse_mla_blockscaled direct-MSL kernel could not be constructed: "
    "mx.fast.metal_kernel is unavailable on this device."
)


# ---------------------------------------------------------------------------
# MXFP8 helpers (same ABI as before)
# ---------------------------------------------------------------------------


def _quantize_mxfp8(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Quantize ``x`` to MXFP8 layout; returns ``(packed_uint32, scales_uint8)``."""

    if x.size == 0:
        last = max(x.shape[-1] // 4, 0) if x.ndim >= 1 else 0
        scales_last = max(x.shape[-1] // MXFP8_BLOCK_SIZE, 0) if x.ndim >= 1 else 0
        packed = mx.zeros(x.shape[:-1] + (last,), dtype=mx.uint32)
        scales = mx.zeros(x.shape[:-1] + (scales_last,), dtype=mx.uint8)
        return packed, scales

    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 last dim must be divisible by {MXFP8_BLOCK_SIZE}, "
            f"got shape {x.shape}."
        )
    res = mx.quantize(x, mode="mxfp8")
    return res[0], res[1]


def _dequantize_mxfp8(
    w_packed: mx.array,
    scales: mx.array,
    *,
    out_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Dequantize an MXFP8 tensor back to the requested dtype."""

    return mx.dequantize(w_packed, scales, mode="mxfp8", dtype=out_dtype)


@mx.custom_function
def _mxfp8_roundtrip_ste(x: mx.array) -> mx.array:
    """Quantize-then-dequantize MXFP8 roundtrip with straight-through gradients."""

    if x.shape[-1] % MXFP8_BLOCK_SIZE != 0:
        return x
    packed, scales = _quantize_mxfp8(x)
    return _dequantize_mxfp8(packed, scales, out_dtype=x.dtype)


@_mxfp8_roundtrip_ste.vjp
def _mxfp8_roundtrip_ste_vjp(primals, cotangent, output):  # noqa: ARG001
    return (cotangent.astype(primals[0].dtype),)


def _unpack_mxfp8_to_uint8(packed: mx.array, last_dim: int) -> mx.array:
    """Re-expand the packed uint32 form into a flat uint8 e4m3 byte array.

    ``mx.quantize(mode='mxfp8')`` packs 4 e4m3 bytes per uint32 in the order
    ``[byte0 | byte1 | byte2 | byte3]`` (low byte first). We unpack with
    ``mx.view(mx.uint8)`` and then reshape the trailing axis to ``last_dim``.
    """

    # mx.array.view changes the element dtype reinterpretation. uint32 -> uint8
    # multiplies the trailing axis by 4 (little-endian byte order).
    bytes_view = packed.view(mx.uint8)
    # The view's last dim is packed.shape[-1] * 4, which equals last_dim
    # when scales were emitted at last_dim // 32 granularity. Confirm:
    expected_last = packed.shape[-1] * 4
    if expected_last != last_dim:
        raise ValueError(
            f"_unpack_mxfp8_to_uint8: expected unpacked last_dim={expected_last} "
            f"to match {last_dim}"
        )
    return bytes_view


# ---------------------------------------------------------------------------
# Direct-MSL forward kernel (block-scaled FP8)
# ---------------------------------------------------------------------------
#
# Inputs:
#   q_fp8       [B, S, H, D_qk]            uint8 (e4m3 bytes)
#   q_scale_e8m0 [B, S, H, D_qk/32]        uint8 (E8M0 scale exponents)
#   kv_fp8      [B, Skv, G, D_qk]          uint8
#   kv_scale_e8m0 [B, Skv, G, D_qk/32]     uint8
#   indices     [B, S, G, topk]            int32
#   sm_scale_buf [1]                       fp32
#
# Outputs:
#   out         [B, S, H, D_v]             fp16
#   lse         [B, S, H]                  fp32

_BLOCKSCALED_FP8_HEADER = """
    #include <metal_stdlib>
    using namespace metal;

    inline float fp8_e4m3_to_float(uchar b) {
        uint sign = (b >> 7) & 0x1u;
        uint exp_bits  = (b >> 3) & 0xFu;
        uint mant = b & 0x7u;
        uint result_bits;
        if (exp_bits == 0u) {
            if (mant == 0u) {
                result_bits = sign << 31;
                return as_type<float>(result_bits);
            } else {
                float v = float(mant) * (1.0f / 8.0f) * 0.015625f;
                return sign ? -v : v;
            }
        } else if (exp_bits == 0xFu && mant == 0x7u) {
            return sign ? -448.0f : 448.0f;
        } else {
            uint exp32 = exp_bits + 120u;
            uint mant32 = mant << 20u;
            result_bits = (sign << 31) | (exp32 << 23) | mant32;
            return as_type<float>(result_bits);
        }
    }

    // E8M0 exponent: stored byte represents 2^(byte - 127).
    inline float e8m0_to_float(uchar b) {
        // Special: byte == 0 means scale = 0 (typically signals NaN/zero block);
        // we treat as 1.0 so it doesn't poison reductions when the data is also 0.
        if (b == 0u) return 0.0f;
        if (b == 0xFFu) return 0.0f;  // NaN sentinel
        int exp_signed = int(b) - 127;
        return ldexp(1.0f, exp_signed);
    }
"""


_BLOCKSCALED_FWD_KERNEL_SOURCE = """
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
    uint num_blocks = qk_dim / uint(MXFP8_BS);

    uint q_row_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * qk_dim;
    uint q_scale_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * num_blocks;
    uint kv_outer_stride = uint(SEQ_LEN_KV) * kv_group * qk_dim;
    uint kv_b_base = b * kv_outer_stride;
    uint kv_scale_outer_stride = uint(SEQ_LEN_KV) * kv_group * num_blocks;
    uint kv_scale_b_base = b * kv_scale_outer_stride;
    uint idx_base = ((b * uint(SEQ_LEN) + s) * kv_group + g) * uint(TOPK);
    float sm_scale = float(sm_scale_buf[0]);

    // Phase 1: scores[k] = sum_blocks (sum_d_in_block q[d] * kv[k,d]) * q_blk_scale * kv_blk_scale
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            scores[k] = -INFINITY;
            continue;
        }
        uint kv_pos = uint(gather_idx);
        uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
        uint kv_scale_row_base = kv_scale_b_base + (kv_pos * kv_group + g) * num_blocks;

        float acc = 0.0f;
        for (uint kb = 0; kb < num_blocks; ++kb) {
            float qs_b = e8m0_to_float(q_scale_e8m0[q_scale_base + kb]);
            float ks_b = e8m0_to_float(kv_scale_e8m0[kv_scale_row_base + kb]);
            float partial = 0.0f;
            uint d_start = kb * uint(MXFP8_BS);
            for (uint dd = 0; dd < uint(MXFP8_BS); ++dd) {
                uint d = d_start + dd;
                float qv = fp8_e4m3_to_float(q_fp8[q_row_base + d]);
                float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]);
                partial += qv * kv_v;
            }
            acc += partial * qs_b * ks_b;
        }
        scores[k] = acc * sm_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: max
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

    // Phase 3: exp, sum
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

    // Phase 4: out[d] = sum_k p[k] * V[k, d] / sumexp
    // V[k, d] = fp8_to_float(kv[k, d]) * kv_scale[k, d/32]
    uint out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
    for (uint d = tid; d < d_v; d += threads) {
        uint kb_d = d / uint(MXFP8_BS);
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            int gather_idx = indices[idx_base + k];
            if (gather_idx < 0) continue;
            uint kv_pos = uint(gather_idx);
            uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
            uint kv_scale_row_base = kv_scale_b_base + (kv_pos * kv_group + g) * num_blocks;
            float ks_b = e8m0_to_float(kv_scale_e8m0[kv_scale_row_base + kb_d]);
            float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]) * ks_b;
            acc += scores[k] * kv_v;
        }
        out[out_row + d] = T_OUT(acc * inv_sum);
    }

    if (tid == 0) {
        uint lse_idx = (b * uint(SEQ_LEN) + s) * uint(HEADS) + h;
        lse[lse_idx] = m_i + log(sumexp);
    }
"""


_BLOCKSCALED_FWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_sparse_mla_blockscaled_fwd",
    input_names=[
        "q_fp8",
        "q_scale_e8m0",
        "kv_fp8",
        "kv_scale_e8m0",
        "indices",
        "sm_scale_buf",
    ],
    output_names=["out", "lse"],
    source=_BLOCKSCALED_FWD_KERNEL_SOURCE,
    header=_BLOCKSCALED_FP8_HEADER,
)


# ---------------------------------------------------------------------------
# Direct-MSL backward kernel (block-scaled FP8)
# ---------------------------------------------------------------------------

_BLOCKSCALED_BWD_KERNEL_SOURCE = """
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
    uint num_blocks = qk_dim / uint(MXFP8_BS);

    uint q_row_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * qk_dim;
    uint q_scale_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * num_blocks;
    uint kv_outer_stride = uint(SEQ_LEN_KV) * kv_group * qk_dim;
    uint kv_b_base = b * kv_outer_stride;
    uint kv_scale_outer_stride = uint(SEQ_LEN_KV) * kv_group * num_blocks;
    uint kv_scale_b_base = b * kv_scale_outer_stride;
    uint idx_base = ((b * uint(SEQ_LEN) + s) * kv_group + g) * uint(TOPK);
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
        uint kv_scale_row_base = kv_scale_b_base + (kv_pos * kv_group + g) * num_blocks;
        float acc = 0.0f;
        for (uint kb = 0; kb < num_blocks; ++kb) {
            float qs_b = e8m0_to_float(q_scale_e8m0[q_scale_base + kb]);
            float ks_b = e8m0_to_float(kv_scale_e8m0[kv_scale_row_base + kb]);
            float partial = 0.0f;
            uint d_start = kb * uint(MXFP8_BS);
            for (uint dd = 0; dd < uint(MXFP8_BS); ++dd) {
                uint d = d_start + dd;
                float qv = fp8_e4m3_to_float(q_fp8[q_row_base + d]);
                float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]);
                partial += qv * kv_v;
            }
            acc += partial * qs_b * ks_b;
        }
        scores[k] = acc * sm_scale;
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

    // dp[k] = sum_d V[k, d] * d_out[d]; V uses block-scaled dequant for d in [0, d_v)
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            dp[k] = 0.0f;
            continue;
        }
        uint kv_pos = uint(gather_idx);
        uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
        uint kv_scale_row_base = kv_scale_b_base + (kv_pos * kv_group + g) * num_blocks;
        float acc = 0.0f;
        uint d_out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
        for (uint d = 0; d < d_v; ++d) {
            uint kb_d = d / uint(MXFP8_BS);
            float ks_b = e8m0_to_float(kv_scale_e8m0[kv_scale_row_base + kb_d]);
            float v = fp8_e4m3_to_float(kv_fp8[kv_row_base + d]) * ks_b;
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

    // dQ_dequant[dd] = sm_scale * sum_k ds[k] * KV_dequant[k, dd]  (block-scaled K)
    for (uint dd = tid; dd < qk_dim; dd += threads) {
        uint kb_d = dd / uint(MXFP8_BS);
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            int gather_idx = indices[idx_base + k];
            if (gather_idx < 0) continue;
            uint kv_pos = uint(gather_idx);
            uint kv_row_base = kv_b_base + (kv_pos * kv_group + g) * qk_dim;
            uint kv_scale_row_base = kv_scale_b_base + (kv_pos * kv_group + g) * num_blocks;
            float ks_b = e8m0_to_float(kv_scale_e8m0[kv_scale_row_base + kb_d]);
            float kv_v = fp8_e4m3_to_float(kv_fp8[kv_row_base + dd]) * ks_b;
            acc += ds[k] * kv_v;
        }
        dq_dequant[q_row_base + dd] = acc * sm_scale;
    }

    // dKV_dequant partial.
    uint dkv_pb = (((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * uint(TOPK)) * qk_dim;
    for (uint kd = tid; kd < uint(TOPK) * qk_dim; kd += threads) {
        uint k = kd / qk_dim;
        uint d = kd % qk_dim;
        uint kb_d = d / uint(MXFP8_BS);
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            dkv_partial[dkv_pb + kd] = 0.0f;
            continue;
        }
        float qs_b = e8m0_to_float(q_scale_e8m0[q_scale_base + kb_d]);
        float qv = fp8_e4m3_to_float(q_fp8[q_row_base + d]) * qs_b;
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


_BLOCKSCALED_BWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_sparse_mla_blockscaled_bwd",
    input_names=[
        "q_fp8",
        "q_scale_e8m0",
        "kv_fp8",
        "kv_scale_e8m0",
        "indices",
        "d_out",
        "sm_scale_buf",
    ],
    output_names=["dq_dequant", "dkv_partial"],
    source=_BLOCKSCALED_BWD_KERNEL_SOURCE,
    header=_BLOCKSCALED_FP8_HEADER,
)


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------


def sparse_mla_blockscaled_metal_status(
    *arrays: mx.array,
) -> SparseMLABlockScaledMetalStatus:
    """Return whether the Path B block-scaled FP8 kernel is dispatchable."""

    if not can_run_metal():
        return SparseMLABlockScaledMetalStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    if _BLOCKSCALED_FWD_KERNEL is None or _BLOCKSCALED_BWD_KERNEL is None:
        return SparseMLABlockScaledMetalStatus(
            available=False, reason=_DIRECT_MSL_BLOCKER_REASON
        )
    float_arrays = [a for a in arrays if a.dtype in (mx.float16, mx.float32, mx.bfloat16)]
    if float_arrays:
        runtime = msl_dispatch_status(*float_arrays)
        if not runtime.available:
            return SparseMLABlockScaledMetalStatus(
                available=False, reason=runtime.reason
            )
    return SparseMLABlockScaledMetalStatus(
        available=True, reason=_DIRECT_MSL_OK_REASON
    )


# ---------------------------------------------------------------------------
# Path B forward / backward (high level)
# ---------------------------------------------------------------------------


def sparse_mla_blockscaled_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Quantize Q/KV to MXFP8 and run Path B forward."""

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if shapes.qk_dim % MXFP8_BLOCK_SIZE != 0:
        return None  # callers fall back to BF16 reference
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5
    status = sparse_mla_blockscaled_metal_status(q, kv, indices)
    if not status.available:
        return None

    # Quantize Q and KV to MXFP8 layout.
    q_packed, q_scales = _quantize_mxfp8(q)
    kv_packed, kv_scales = _quantize_mxfp8(kv)
    # Unpack the uint32 packed bytes back into uint8 e4m3 storage so the MSL
    # kernel can iterate one byte per d.
    q_fp8 = _unpack_mxfp8_to_uint8(q_packed, shapes.qk_dim)
    kv_fp8 = _unpack_mxfp8_to_uint8(kv_packed, shapes.qk_dim)

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
        ("MXFP8_BS", MXFP8_BLOCK_SIZE),
    ]

    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    grid_x = shapes.batch * shapes.seq_len * shapes.heads
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, _BLOCKSCALED_FWD_KERNEL),
        inputs=[
            q_fp8.astype(mx.uint8),
            q_scales.astype(mx.uint8),
            kv_fp8.astype(mx.uint8),
            kv_scales.astype(mx.uint8),
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


def sparse_mla_blockscaled_bwd_metal(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Quantize and run the block-scaled FP8 backward; returns dequantized grads."""

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if shapes.qk_dim % MXFP8_BLOCK_SIZE != 0:
        return None
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5
    status = sparse_mla_blockscaled_metal_status(q, kv, indices)
    if not status.available:
        return None

    q_packed, q_scales = _quantize_mxfp8(q)
    kv_packed, kv_scales = _quantize_mxfp8(kv)
    q_fp8 = _unpack_mxfp8_to_uint8(q_packed, shapes.qk_dim)
    kv_fp8 = _unpack_mxfp8_to_uint8(kv_packed, shapes.qk_dim)

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
        ("MXFP8_BS", MXFP8_BLOCK_SIZE),
    ]
    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)
    grid_x = shapes.batch * shapes.seq_len * shapes.heads
    outputs = _msl_transform.dispatch(
        cast(_msl_transform.MetalKernel, _BLOCKSCALED_BWD_KERNEL),
        inputs=[
            q_fp8.astype(mx.uint8),
            q_scales.astype(mx.uint8),
            kv_fp8.astype(mx.uint8),
            kv_scales.astype(mx.uint8),
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
    indices_i32 = indices.astype(mx.int32)
    dkv = _reduce_dkv_partial_fp32(dkv_partial, indices_i32, shapes)
    return dq_dequant, dkv


def _reduce_dkv_partial_fp32(dkv_partial, indices_i32, shapes):
    """Scatter-add per-token dkv partials into [B, Skv, G, qk_dim] fp32."""

    B = shapes.batch
    S = shapes.seq_len
    Skv = shapes.seq_len_kv
    G = shapes.kv_group
    head_kv = shapes.head_kv
    topk = shapes.topk
    qk_dim = shapes.qk_dim

    dkv_per_group = dkv_partial.reshape(B, S, G, head_kv, topk, qk_dim).sum(axis=3)
    valid_mask = indices_i32 != -1
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
# Reference (kept for parity tests + fallback)
# ---------------------------------------------------------------------------


def sparse_mla_blockscaled_reference(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Pure-MLX MXFP8 block-scaled reference."""

    shapes: SparseMLAShapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if shapes.qk_dim % MXFP8_BLOCK_SIZE != 0:
        return sparse_mla_attention_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=return_lse
        )

    q_recovered = _mxfp8_roundtrip_ste(q)
    kv_recovered = _mxfp8_roundtrip_ste(kv)
    return sparse_mla_attention_reference(
        q_recovered,
        kv_recovered,
        indices,
        sm_scale=sm_scale,
        d_v=d_v,
        return_lse=return_lse,
    )


# ---------------------------------------------------------------------------
# Custom VJP wrapper
# ---------------------------------------------------------------------------


@mx.custom_function
def sparse_mla_blockscaled_metal_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
) -> mx.array:
    """Forward-only differentiable wrapper around the block-scaled MSL kernel."""

    result = sparse_mla_blockscaled_fwd_metal(q, kv, indices)
    if result is None:
        return sparse_mla_blockscaled_reference(q, kv, indices)
    out, _lse = result
    return out


@sparse_mla_blockscaled_metal_apply.vjp
def _sparse_mla_blockscaled_metal_apply_vjp(primals, cotangent, output):
    del output
    q, kv, indices = primals
    grads = sparse_mla_blockscaled_bwd_metal(q, kv, cotangent, indices)
    if grads is None:
        def _ref_apply(q_, kv_):
            return sparse_mla_blockscaled_reference(q_, kv_, indices)
        _, vjps = mx.vjp(_ref_apply, (q, kv), (cotangent,))
        return (vjps[0], vjps[1], mx.zeros_like(indices))
    dq, dkv = grads
    return (dq.astype(q.dtype), dkv.astype(kv.dtype), mx.zeros_like(indices))


def sparse_mla_blockscaled_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_metal: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Apply block-scaled FP8 sparse MLA, preferring Path B."""

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if shapes.qk_dim % MXFP8_BLOCK_SIZE != 0:
        # Fall back to BF16 reference (no MXFP8 path for misaligned dims).
        return sparse_mla_blockscaled_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=return_lse
        )

    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    if return_lse:
        result = sparse_mla_blockscaled_fwd_metal(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v
        )
        if result is None:
            if force_metal:
                raise RuntimeError(
                    f"sparse_mla_blockscaled_apply: Metal path unavailable: "
                    f"{sparse_mla_blockscaled_metal_status(q, kv, indices).reason}"
                )
            return sparse_mla_blockscaled_reference(
                q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=True
            )
        out, lse = result
        return out.astype(q.dtype), lse

    status = sparse_mla_blockscaled_metal_status(q, kv, indices)
    if not status.available:
        if force_metal:
            raise RuntimeError(
                f"sparse_mla_blockscaled_apply: Metal path unavailable: "
                f"{status.reason}"
            )
        return sparse_mla_blockscaled_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False
        )

    is_default = (
        d_v is None or d_v == shapes.qk_dim
    ) and abs(sm_scale - shapes.qk_dim ** -0.5) < 1e-9
    if is_default:
        out = sparse_mla_blockscaled_metal_apply(q, kv, indices)
        return out.astype(q.dtype)

    return sparse_mla_blockscaled_reference(
        q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False
    )


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "SparseMLABlockScaledMetalStatus",
    "blockscaled_sparse_mla_qk_path_c_status",
    "sparse_mla_blockscaled_apply",
    "sparse_mla_blockscaled_bwd_metal",
    "sparse_mla_blockscaled_fwd_metal",
    "sparse_mla_blockscaled_metal_apply",
    "sparse_mla_blockscaled_metal_status",
    "sparse_mla_blockscaled_reference",
]
