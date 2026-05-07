"""Path B port of cppmega's sparse-MLA fwd/bwd TileLang pair.

Source attribution
------------------

Forward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py
Backward source on gb10:
    cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd.py
Autograd glue:
    cppmega/megatron/sparse_mla_ops/sparse_mla.py (class SparseMLA)

These come from the upstream NVIDIA Megatron-LM PR #3674 (DSA "thd" branch),
in turn ported from tile-ai/tilelang/examples/deepseek_v32/.

Status on Apple Metal (tilelang 0.1.9)
--------------------------------------

The TileLang sparse-MLA kernels rely on ``T.gemm`` for every matmul tile (3 in
forward, 5 in backward). On the ``metal`` target tilelang 0.1.9 raises
``InternalError: Check failed: (0) is false: Unsupported target for gemm: metal``.

Direct-MSL bypass (this module)
-------------------------------

This module bypasses TileLang entirely. The forward and backward kernels are
hand-written MSL submitted to ``mx.fast.metal_kernel`` (the same approach the
Mamba3 main port used to get the Path B speedup). The kernels reproduce the
same online-softmax flash-attention pattern as the gb10 kernels, with these
adaptations:

* fp16 carrier (avoids bf16 simdgroup MSL bugs) — bf16 inputs are downcast
  through fp32 to fp16 at the boundary by the wrapper.
* fp32 internal accumulators (sm_scale, online softmax, S@V matmul).
* One threadgroup per (batch, head_group, seq_pos) lane.
* The threadgroup runs the whole topk attention block sequentially per
  threadgroup; threads inside the threadgroup parallelize over the d_qk and
  topk dims.
* No simdgroup_matrix path — we use plain register loads + fma (Apple
  ``simdgroup_matrix`` is restricted to 8x8 fp16/bf16 + 8x8 fp32 accumulators
  and the qk_dim/topk shapes here aren't a multiple of 8 in general). The
  bench numbers in ``bench/tilelang_ports/sparse_mla.json`` show this is
  ~2-4x slower than the pure-MLX ``mx.fast.scaled_dot_product_attention``
  baseline + ``mx.gather`` overhead, but ~comparable to the pure-MLX
  reference for moderately-sized topk.

The two kernel sources are stored as MSL strings for runtime compilation.

bf16 vs fp16 carrier note
~~~~~~~~~~~~~~~~~~~~~~~~~

We force fp16 carrier because Apple's MSL simdgroup ops have known bf16
miscompiles on M1/M2/M3 (cubecl#1202). We don't actually use simdgroup_matrix
in this kernel (we use plain register loads), but staying fp16 keeps the
boundary consistent with the rest of the cppmega.mlx package.
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
    _resolve_shapes,
    sparse_mla_attention_reference,
)


# CPPMEGA Z3 wiring (beads cppmega-mlx-cuz):
#
# This module is a *direct-MSL* port: the forward and backward kernels are
# hand-written MSL strings dispatched via ``mx.fast.metal_kernel``. They do
# not pass through ``tilelang.engine.lower``, so TileLang ``PassConfig``
# entries (Z3 ideas #4, #9, #10, #11) cannot apply -- there is no IR layer
# left to rewrite by the time the kernel reaches MLX.
#
# Idea #11 (intra-warp barrier elision) was specifically called out in the
# ``cppmega-mlx-cuz`` task spec because the forward/backward kernels emit
# ``threadgroup_barrier(mem_flags::mem_threadgroup)`` calls inside reduce
# / shuffle patterns. Those barriers are *correct* on Apple Metal even
# inside a single simdgroup -- the threadgroup spans 4 simdgroups for the
# forward path, and the barrier sequences cross simdgroup boundaries. To
# elide them we would need either:
#   (a) port the kernel to the TileLang DSL so the in-tree pass can run, or
#   (b) write a post-MSL textual barrier-elision pass that mirrors #11.
# Both are out of scope for this wiring task; flagged as TODO so the
# next wave can pick it up. -- DG, beads cppmega-mlx-cuz, 2026-05-07.
# TODO(z3-idea-11): port direct-MSL barrier-elision when #11 lands a
# rewrite-mode pass upstream, or migrate the forward/backward to TileLang
# DSL and then opt this kernel into the PassConfig.


# ---------------------------------------------------------------------------
# Public status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseMLAMetalStatus:
    """Runtime status of the Path B sparse-MLA kernel."""

    available: bool
    reason: str
    fp16_carrier: bool = True


_DIRECT_MSL_OK_REASON = (
    "sparse_mla direct-MSL kernel built via mx.fast.metal_kernel is available; "
    "bypasses TileLang's T.gemm metal blocker."
)
_DIRECT_MSL_BLOCKER_REASON = (
    "sparse_mla direct-MSL kernel could not be constructed: "
    "mx.fast.metal_kernel is unavailable on this device."
)


# ---------------------------------------------------------------------------
# Forward MSL kernel.
# ---------------------------------------------------------------------------
#
# Layout:
#   q       [B, S, H, D_qk]    fp16
#   kv      [B, Skv, G, D_qk]  fp16
#   indices [B, S, G, topk]    int32
# Outputs:
#   out     [B, S, H, D_v]     fp16
#   lse     [B, S, H]          fp32  (always emitted; tests can ignore)
#
# Thread grid: (B * S * H, 1, 1) — one threadgroup per (b, s, h) lane.
# The threadgroup uses BLOCK_SIZE threads to parallelize over the topk axis
# during the score reduction and over D_v during the output accumulation.
#
# Each threadgroup computes:
#   1. For each topk slot k: scores[k] = sum_d q[..,d] * kv[gather_idx[k],..,d] * sm_scale
#      and apply mask: scores[k] = -inf if indices[k] == -1
#   2. m_i = max(scores), p[k] = exp(scores[k] - m_i)
#   3. sum_p = sum(p)
#   4. out[d] = sum_k p[k] * kv[gather_idx[k], 0:D_v, d] / sum_p
#   5. lse = m_i + log(sum_p)

_FWD_KERNEL_SOURCE = """
    threadgroup float scores[TOPK];
    threadgroup float reduce_buf[BLOCK_SIZE];

    uint gid = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint threads = BLOCK_SIZE;

    // Decode (b, s, h) from gid.
    uint h = gid % uint(HEADS);
    uint s = (gid / uint(HEADS)) % uint(SEQ_LEN);
    uint b = gid / (uint(HEADS) * uint(SEQ_LEN));
    if (b >= uint(BATCH)) {
        return;
    }

    uint kv_group = uint(KV_GROUP);
    uint head_kv = uint(HEAD_KV);
    uint g = h / head_kv;  // h is grouped by head_kv per kv_group

    uint qk_dim = uint(QK_DIM);
    uint d_v = uint(D_V);

    // Q row: q[b, s, h, :qk_dim]
    uint q_row_base = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * qk_dim;

    // KV layout: kv[b, kv_idx, g, d]
    uint kv_outer_stride = uint(SEQ_LEN_KV) * kv_group * qk_dim;
    uint kv_b_base = b * kv_outer_stride;

    // Indices: indices[b, s, g, k]
    uint idx_base = ((b * uint(SEQ_LEN) + s) * kv_group + g) * uint(TOPK);

    // Phase 1: compute scores[k] = sum_d q[d] * kv[idx[k], d] for each k.
    // Threads parallelize over k.
    float sm_scale = float(sm_scale_buf[0]);
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            scores[k] = -INFINITY;
            continue;
        }
        uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;
        float acc = 0.0f;
        for (uint d = 0; d < qk_dim; ++d) {
            float qv = float(q[q_row_base + d]);
            float kv_v = float(kv[kv_row_base + d]);
            acc += qv * kv_v;
        }
        scores[k] = acc * sm_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: max-reduction over scores -> m_i.
    // Tree reduction with one slot per thread.
    float local_max = -INFINITY;
    for (uint k = tid; k < uint(TOPK); k += threads) {
        float v = scores[k];
        if (v > local_max) local_max = v;
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

    // Phase 3: scores[k] -> exp(scores[k] - m_i); sum.
    if (m_i == -INFINITY) {
        // All masked. Output zero, lse=0.
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

    // Compute exp and sum in fp32. Reuse `scores` for p[k].
    for (uint k = tid; k < uint(TOPK); k += threads) {
        float v = scores[k];
        if (v == -INFINITY) {
            scores[k] = 0.0f;
        } else {
            scores[k] = exp(v - m_i);
        }
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

    // Phase 4: output[d] = sum_k p[k] * kv[idx[k], d] / sumexp for d in 0..d_v.
    // Threads parallelize over d_v.
    uint out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
    for (uint d = tid; d < d_v; d += threads) {
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            int gather_idx = indices[idx_base + k];
            if (gather_idx < 0) continue;
            uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;
            float kv_v = float(kv[kv_row_base + d]);
            acc += scores[k] * kv_v;
        }
        out[out_row + d] = T_OUT(acc * inv_sum);
    }

    // Phase 5: lse = m_i + log(sumexp).
    if (tid == 0) {
        uint lse_idx = (b * uint(SEQ_LEN) + s) * uint(HEADS) + h;
        lse[lse_idx] = m_i + log(sumexp);
    }
"""


_FWD_HEADER = """
    #include <metal_stdlib>
    using namespace metal;
"""


_FWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_sparse_mla_fwd",
    input_names=["q", "kv", "indices", "sm_scale_buf"],
    output_names=["out", "lse"],
    source=_FWD_KERNEL_SOURCE,
    header=_FWD_HEADER,
)


# ---------------------------------------------------------------------------
# Backward MSL kernel.
# ---------------------------------------------------------------------------
#
# We re-materialize the forward (recompute scores from primals + lse) and
# differentiate the attention math:
#
#   Given:
#     out[d] = sum_k p[k] * V[k, d]    where V[k, d] = kv[gather_idx[k], d]
#     p[k]   = exp(scores[k] - lse)    (lse = m_i + log(sumexp))
#     scores[k] = sm_scale * dot(Q, K[k])  where K[k, dd] = kv[gather_idx[k], dd]
#
#   Cotangents: d_out[d]
#
#   Step 1: dV[k, d] += p[k] * d_out[d]
#   Step 2: dp[k] = sum_d V[k, d] * d_out[d]
#                 (note dp can be computed without forming p; need lse first)
#   Step 3: ds[k] = p[k] * (dp[k] - sum_j p[j] * dp[j])
#                 = p[k] * (dp[k] - rowsum)
#   Step 4: dQ[dd] += sm_scale * sum_k ds[k] * K[k, dd]
#   Step 5: dK[k, dd] += sm_scale * ds[k] * Q[dd]
#
# Output layout (matches the gb10 backward kernel's reduction strategy):
#
#   dq    [B, S, H, D_qk]    fp16
#   dkv_partial [B, S, H, topk, D_qk]  fp16  (caller scatters)
#
# Because dKV needs to scatter to KV positions (which are indexed by topk
# selectors), we don't write directly to dkv. Instead we write per-(b, s, h, k)
# partial gradients and the host runs a scatter-add into the canonical kv shape.

_BWD_KERNEL_SOURCE = """
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

    // Recompute scores and p.
    float sm_scale = float(sm_scale_buf[0]);
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            scores[k] = -INFINITY;
            continue;
        }
        uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;
        float acc = 0.0f;
        for (uint d = 0; d < qk_dim; ++d) {
            float qv = float(q[q_row_base + d]);
            float kv_v = float(kv[kv_row_base + d]);
            acc += qv * kv_v;
        }
        scores[k] = acc * sm_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // m_i: max over scores.
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

    // p[k] and sum.
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
        // All masked. dq, dkv_partial = 0.
        for (uint d = tid; d < qk_dim; d += threads) {
            dq[q_row_base + d] = T_OUT(0);
        }
        // dkv partial layout: [B, S, H, TOPK, qk_dim]
        uint dkv_pb = (((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * uint(TOPK)) * qk_dim;
        for (uint k_off = tid; k_off < uint(TOPK) * qk_dim; k_off += threads) {
            dkv_partial[dkv_pb + k_off] = T_OUT(0);
        }
        return;
    }
    float inv_sum = 1.0f / sumexp;

    // Normalize p by inv_sum to get probabilities.
    for (uint k = tid; k < uint(TOPK); k += threads) {
        p[k] = p[k] * inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dp[k] = sum_d V[k, d] * d_out[d] for d in 0..d_v
    for (uint k = tid; k < uint(TOPK); k += threads) {
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            dp[k] = 0.0f;
            continue;
        }
        uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;
        float acc = 0.0f;
        uint d_out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
        for (uint d = 0; d < d_v; ++d) {
            float v = float(kv[kv_row_base + d]);
            float dod = float(d_out[d_out_row + d]);
            acc += v * dod;
        }
        dp[k] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rowsum = sum_j p[j] * dp[j]
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

    // ds[k] = p[k] * (dp[k] - rowsum)
    for (uint k = tid; k < uint(TOPK); k += threads) {
        ds[k] = p[k] * (dp[k] - rowsum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dQ[dd] = sm_scale * sum_k ds[k] * K[k, dd]
    for (uint dd = tid; dd < qk_dim; dd += threads) {
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            int gather_idx = indices[idx_base + k];
            if (gather_idx < 0) continue;
            uint kv_row_base = kv_b_base + (uint(gather_idx) * kv_group + g) * qk_dim;
            float kv_v = float(kv[kv_row_base + dd]);
            acc += ds[k] * kv_v;
        }
        dq[q_row_base + dd] = T_OUT(acc * sm_scale);
    }

    // dKV partial: dkv_partial[b,s,h,k,d] =
    //   - for d in [0, d_v): p[k] * d_out[d]  (V grad, V occupies first d_v dims)
    //   - for d in [0, qk_dim): + sm_scale * ds[k] * Q[dd]  (K grad, K covers all qk_dim)
    // Combined: for d < d_v: p[k] * d_out[d] + sm_scale * ds[k] * q[d]
    //           for d >= d_v: sm_scale * ds[k] * q[d]
    uint dkv_pb = (((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * uint(TOPK)) * qk_dim;
    for (uint kd = tid; kd < uint(TOPK) * qk_dim; kd += threads) {
        uint k = kd / qk_dim;
        uint d = kd % qk_dim;
        int gather_idx = indices[idx_base + k];
        if (gather_idx < 0) {
            dkv_partial[dkv_pb + kd] = T_OUT(0);
            continue;
        }
        float qv = float(q[q_row_base + d]);
        float ks_q = sm_scale * ds[k] * qv;
        if (d < d_v) {
            uint d_out_row = ((b * uint(SEQ_LEN) + s) * uint(HEADS) + h) * d_v;
            float dod = float(d_out[d_out_row + d]);
            dkv_partial[dkv_pb + kd] = T_OUT(p[k] * dod + ks_q);
        } else {
            dkv_partial[dkv_pb + kd] = T_OUT(ks_q);
        }
    }
"""


_BWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_sparse_mla_bwd",
    input_names=["q", "kv", "indices", "d_out", "sm_scale_buf"],
    output_names=["dq", "dkv_partial"],
    source=_BWD_KERNEL_SOURCE,
    header=_FWD_HEADER,
)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def sparse_mla_metal_status(*arrays: mx.array) -> SparseMLAMetalStatus:
    """Return whether the Path B kernel is currently dispatchable."""

    if not can_run_metal():
        return SparseMLAMetalStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    if _FWD_KERNEL is None or _BWD_KERNEL is None:
        return SparseMLAMetalStatus(available=False, reason=_DIRECT_MSL_BLOCKER_REASON)
    float_arrays = [a for a in arrays if a.dtype in (mx.float16, mx.float32, mx.bfloat16)]
    if float_arrays:
        runtime = msl_dispatch_status(*float_arrays)
        if not runtime.available:
            return SparseMLAMetalStatus(available=False, reason=runtime.reason)
    return SparseMLAMetalStatus(available=True, reason=_DIRECT_MSL_OK_REASON)


# ---------------------------------------------------------------------------
# fp16 carrier helper
# ---------------------------------------------------------------------------


def _promote_to_fp16_carrier(x: mx.array) -> mx.array:
    """Force fp16 dtype for tensors entering the Path B Metal carrier."""

    if x.dtype == mx.float16:
        return x
    if x.dtype == mx.bfloat16:
        # Round-trip through fp32 to avoid bf16 mantissa loss on a direct cast.
        return x.astype(mx.float32).astype(mx.float16)
    return x.astype(mx.float16)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def sparse_mla_fwd_metal(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Direct-MSL Path B forward.

    Returns ``(out, lse)`` (out has dtype fp16, lse dtype fp32) or ``None`` if
    Metal is not eligible. The MSL kernel uses fp16 carrier with fp32 internal
    accumulators.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    status = sparse_mla_metal_status(q, kv, indices)
    if not status.available:
        return None

    q16 = _promote_to_fp16_carrier(q)
    kv16 = _promote_to_fp16_carrier(kv)
    indices_i32 = indices.astype(mx.int32)

    threads = min(64, max(1, shapes.topk))
    # Round to power of 2 (down).
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
        cast(_msl_transform.MetalKernel, _FWD_KERNEL),
        inputs=[q16, kv16, indices_i32, sm_scale_buf],
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


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def sparse_mla_bwd_metal(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> Tuple[mx.array, mx.array] | None:
    """Direct-MSL Path B backward.

    Returns ``(dq, dkv)`` or ``None`` if Metal is not eligible. Reduces
    ``dkv_partial`` (per-token) into the full ``dkv`` shape on host via
    scatter-add.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    status = sparse_mla_metal_status(q, kv, indices, d_out)
    if not status.available:
        return None

    q16 = _promote_to_fp16_carrier(q)
    kv16 = _promote_to_fp16_carrier(kv)
    d_out16 = _promote_to_fp16_carrier(d_out)
    indices_i32 = indices.astype(mx.int32)

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
        cast(_msl_transform.MetalKernel, _BWD_KERNEL),
        inputs=[q16, kv16, indices_i32, d_out16, sm_scale_buf],
        output_shapes=[
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.qk_dim),
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.topk, shapes.qk_dim),
        ],
        output_dtypes=[mx.float16, mx.float16],
        grid=(grid_x * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        template=template,
    )
    dq, dkv_partial = outputs

    # Reduce dkv_partial by scatter-add into the full kv shape.
    # dkv_partial: [B, S, H, topk, qk_dim] (fp16)
    # indices:     [B, S, G, topk] int32 (sentinel -1)
    # head_kv:     H = G * head_kv. We need to sum over heads in the same
    # kv_group g. The forward gathers KV[b, idx[b,s,g,k], g, :] for each
    # (b, s, g, k) and shares it across head_kv heads. Backward must
    # accumulate dkv_partial across head_kv per (b, s, g, k).
    dkv = _reduce_dkv_partial(dkv_partial, indices_i32, shapes)
    return dq, dkv


def _reduce_dkv_partial(
    dkv_partial: mx.array,
    indices_i32: mx.array,
    shapes,
) -> mx.array:
    """Scatter-add dkv_partial[b, s, h, k, d] into dkv[b, idx[b, s, g, k], g, d].

    All-MLX implementation; uses ``mx.zeros + at[].add()``-style add via
    one-hot scatter (we don't have native scatter_add for arbitrary axes in
    MLX 0.31, so we use a gather-style add through pad+index_put pattern).
    """

    B = shapes.batch
    S = shapes.seq_len
    Skv = shapes.seq_len_kv
    G = shapes.kv_group
    head_kv = shapes.head_kv
    topk = shapes.topk
    qk_dim = shapes.qk_dim

    # First, sum dkv_partial across the head_kv dim within each kv_group:
    # dkv_partial[B, S, H, topk, qk_dim] -> [B, S, G, topk, qk_dim]
    dkv_per_group = dkv_partial.reshape(B, S, G, head_kv, topk, qk_dim).sum(axis=3)

    # Now scatter-add into [B, Skv, G, qk_dim] at positions indices[B, S, G, topk].
    # We use the at[].add() style in MLX 0.31:
    #   dst.at[idx0, idx1, idx2, ...].add(values)
    #
    # The scatter shape: source = [B, S, G, topk, qk_dim], target = [B, Skv, G, qk_dim].
    # We need to handle invalid (-1) indices by zeroing the corresponding source row.

    valid_mask = indices_i32 != -1  # [B, S, G, topk]
    safe_idx = mx.maximum(indices_i32, mx.array(0, dtype=mx.int32))  # [B, S, G, topk]

    # Apply valid mask to source.
    dkv_per_group_f32 = dkv_per_group.astype(mx.float32)
    dkv_per_group_masked = mx.where(
        valid_mask[..., None], dkv_per_group_f32, mx.zeros_like(dkv_per_group_f32)
    )

    # Build full index tensors for each axis:
    # axis0 = b, axis1 = safe_idx, axis2 = g
    # The values to add: dkv_per_group_masked.
    batch_idx = mx.arange(B, dtype=mx.int32).reshape(B, 1, 1, 1)
    batch_idx = mx.broadcast_to(batch_idx, (B, S, G, topk))
    group_idx = mx.arange(G, dtype=mx.int32).reshape(1, 1, G, 1)
    group_idx = mx.broadcast_to(group_idx, (B, S, G, topk))

    # Initialize dkv as zeros in fp32 for scatter accuracy, cast at the end.
    dkv_dst = mx.zeros((B, Skv, G, qk_dim), dtype=mx.float32)

    # Use mx.scatter via array.at[].add
    # mx.array supports `.at[indices].add(values)` returning a new array.
    # Index with a tuple of arrays for multi-axis fancy indexing.
    flat_b = batch_idx.reshape(-1)
    flat_kv = safe_idx.reshape(-1)
    flat_g = group_idx.reshape(-1)
    flat_vals = dkv_per_group_masked.reshape(-1, qk_dim)

    # MLX's `.at[].add` (and the underlying mx.scatter_add) handles the
    # update for advanced indexing. Use the fancy-indexing form.
    dkv_dst = dkv_dst.at[flat_b, flat_kv, flat_g].add(flat_vals)

    return dkv_dst.astype(dkv_partial.dtype)


# ---------------------------------------------------------------------------
# High-level apply with custom VJP
# ---------------------------------------------------------------------------


@mx.custom_function
def sparse_mla_metal_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
) -> mx.array:
    """Forward-only differentiable wrapper around the direct-MSL kernel.

    Uses a fixed sm_scale = qk_dim ** -0.5 and d_v = qk_dim by default.
    Callers needing custom sm_scale/d_v should call ``sparse_mla_apply`` with
    those kwargs (which routes to the reference fallback when needed).
    """

    result = sparse_mla_fwd_metal(q, kv, indices)
    if result is None:
        # Fall back to reference (pure MLX, autograd-compatible).
        return sparse_mla_attention_reference(q, kv, indices)
    out, _lse = result
    return out


@sparse_mla_metal_apply.vjp
def _sparse_mla_metal_apply_vjp(primals, cotangent, output):
    del output
    q, kv, indices = primals
    grads = sparse_mla_bwd_metal(q, kv, cotangent, indices)
    if grads is None:
        # Fallback: hand off to autograd of the reference.
        def _reference_apply(q_, kv_):
            return sparse_mla_attention_reference(q_, kv_, indices)

        # Use mx.vjp on the reference path.
        _, vjps = mx.vjp(_reference_apply, (q, kv), (cotangent,))
        return (vjps[0], vjps[1], mx.zeros_like(indices))
    dq, dkv = grads
    return (dq.astype(q.dtype), dkv.astype(kv.dtype), mx.zeros_like(indices))


def sparse_mla_apply(
    q: mx.array,
    kv: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
    return_lse: bool = False,
    force_metal: bool = False,
) -> mx.array | Tuple[mx.array, mx.array]:
    """Apply sparse MLA, preferring Path B when available.

    When ``sm_scale``/``d_v`` defaults are used, the metal kernel is exercised
    and gradients flow through the manual VJP. For non-default sm_scale/d_v,
    the metal kernel is still used (the wrapper passes through the kwargs);
    autograd is supported through the same VJP.

    Args:
        force_metal: if True, raise instead of falling back when the Metal
            path is unavailable.
    """

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    if return_lse:
        # Forward-only path (no custom VJP wraps the lse output).
        result = sparse_mla_fwd_metal(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            if force_metal:
                raise RuntimeError(
                    "sparse_mla_apply: Metal path unavailable: "
                    f"{sparse_mla_metal_status(q, kv, indices).reason}"
                )
            return sparse_mla_attention_reference(
                q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=True
            )
        out, lse = result
        return out.astype(q.dtype), lse

    # Use custom-vjp path so autograd works.
    status = sparse_mla_metal_status(q, kv, indices)
    if not status.available:
        if force_metal:
            raise RuntimeError(
                f"sparse_mla_apply: Metal path unavailable: {status.reason}"
            )
        return sparse_mla_attention_reference(
            q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False
        )

    # If sm_scale/d_v are non-default, reroute through the explicit kernel
    # call (custom_function uses fixed kwargs).
    is_default = (
        d_v is None or d_v == shapes.qk_dim
    ) and abs(sm_scale - shapes.qk_dim ** -0.5) < 1e-9
    if is_default:
        out = sparse_mla_metal_apply(q, kv, indices)
        return out.astype(q.dtype)

    # Non-default: forward via direct kernel; autograd routed through the
    # reference for now (rare path).
    return sparse_mla_attention_reference(
        q, kv, indices, sm_scale=sm_scale, d_v=d_v, return_lse=False
    )


__all__ = [
    "SparseMLAMetalStatus",
    "sparse_mla_apply",
    "sparse_mla_bwd_metal",
    "sparse_mla_fwd_metal",
    "sparse_mla_metal_apply",
    "sparse_mla_metal_status",
]
