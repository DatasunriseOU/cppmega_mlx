"""Path B port of cppmega's M2RNN recurrent mixer.

This module implements the M2RNN per-token recurrence in vendor MSL via
:func:mx.fast.metal_kernel, paired with a manual VJP through
:class:mx.custom_function. The kernel matches the math of
:func:cppmega_mlx.nn.m2rnn.m2rnn_scan (the parity oracle) but does not
modify it.

TODO(wave-7): unified-pipeline migration deferred — both _FWD_KERNEL and
_BWD_KERNEL are hand-written MSL constructed via
``_msl_transform.make_metal_kernel`` (no ``@T.prim_func`` to feed
``dispatch_lower``). The MSL-extraction adapter (commit 00d6d90) only
applies to ``tilelang.engine.lower(prim, target)`` artifacts, which is the
inverse direction. Migrating m2rnn to the unified pipeline therefore
requires a full TileLang DSL rewrite of both the forward scan (per-kk row-
of-h fragment + threadgroup-shared W + ``T.serial(S)``) and the backward
two-pass walk (forward sweep persisting h_{t-1} to scratch, backward time
loop with cross-thread dW reduction). All required TileLang primitives
exist; the blocker is the size of the rewrite. See per-kernel TODO blocks
on _FWD_KERNEL and _BWD_KERNEL below.

Recurrence (per (batch, head) lane, looping over seq):
    z_t   = h_{t-1} @ W + outer(k_t, v_t)
    h_new = tanh(z_t)
    h_t   = f_t * h_{t-1} + (1 - f_t) * h_new
    y_t   = q_t^T h_t

Backward (walking backwards from seq-1 to 0, accumulating cotangents):
    dh_t   = q_t * dy_t + carry
    dh_new = (1 - f_t) * dh_t
    dz     = dh_new * (1 - tanh(z_t)^2)        (saved tanh(z_t) avoids recompute)
    df_t   = (h_{t-1} - tanh(z_t)) * dh_t      (sum over K, V)
    dk_t   = dz @ v_t                           (sum over V)
    dv_t   = k_t @ dz                           (sum over K)
    dW    += h_{t-1}^T @ dz
    carry  = f_t * dh_t + dz @ W^T

Numerical contract:
  - fp16/fp32 carrier with fp32 accumulators inside the kernel.
  - bf16 inputs cast to fp16 before launch (bf16 simdgroup MSL codegen bugs).
  - Saved tanh(z_t) per step in fwd kernel feeds 4-way reuse in bwd.

Public surface (mirrors :mod:cppmega_mlx.nn._tilelang.mamba3):
  - :func:m2rnn_metal_status: probe Metal eligibility.
  - :func:m2rnn_fwd_metal: forward kernel returning (y, h_last, tanh_cache).
  - :func:m2rnn_bwd_metal: backward kernel returning per-input grads.
  - :func:m2rnn_apply: differentiable wrapper returning y.
  - :func:m2rnn_apply_with_state: differentiable wrapper returning
    (y, h_last) for cache assembly (h_last cotangent ignored).
  - :func:m2rnn_reference: thin call into the parent reference scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform


@dataclass(frozen=True)
class M2RNNMetalStatus:
    """Capability probe result for the Path B M2RNN kernel."""

    available: bool
    reason: str


# ---------------------------------------------------------------------------
# Forward MSL kernel
# ---------------------------------------------------------------------------
#
# TODO(wave-7): port to TileLang DSL — complex-fused.
#
# This kernel is hand-written MSL constructed via
# ``_msl_transform.make_metal_kernel(name=..., source=..., header=...)``,
# not via ``_msl_transform.lower_tilelang_to_msl_inline(@T.prim_func)``.
# The MSL-extraction adapter (`_msl_extraction.extract_msl_from_engine_artifact`
# landed in commit 00d6d90) converts ``@T.prim_func`` engine artifacts INTO
# MSL — the inverse direction — so it does not apply here. To route this
# kernel through ``dispatch_lower(prim, target)`` we need a full TileLang
# DSL rewrite carrying:
#
#   * per-kk row-of-h register state (size V scalars per thread, no
#     cross-thread reduction during matmul) — expressible as
#     ``T.alloc_fragment((V,), accum_dtype)``;
#   * threadgroup-shared ``W`` tile (V*V) loaded once per (b,h) lane —
#     ``T.alloc_shared((V_DIM, V_DIM))``;
#   * sequential time loop with persistent ``h`` carried across steps —
#     ``T.serial(S)`` with the fragment surviving across iterations;
#   * cross-thread reduction over kk for ``y`` per step — TileLang already
#     supports threadgroup-shared accumulator + barrier;
#   * ``tanh_cache`` write per step ([B,S,H,K,V]) for bwd.
#
# Each piece is supported individually; the blocker is just the size of
# the rewrite (and the bwd kernel below has the same shape but inverted).
# Port deferred until at least one production caller demands the unified
# CUDA + Metal lowering.
#
# Parallelism strategy:
#   - One threadgroup per (batch, head) lane.
#   - K_DIM threads per group; each thread owns one row of h: h[kk, :] (size V).
#   - Per step: matmul h@W is computed independently per kk (no cross-thread
#     reduction needed for the matmul itself); the y reduction across kk uses
#     threadgroup memory + a single barrier.
#
# Per-thread state lives in registers (size V scalars), well within Apple
# GPU per-thread register budget. Across threads in the group we share W
# loads via threadgroup memory (V*V floats — small).
#
# Inputs (all cast to T_OUT before launch, fp32 internal accum):
#   q  [B, S, H, K]
#   k  [B, S, H, K]
#   v  [B, S, H, V]
#   W  [H, V, V]
#   xf [B, S, H]
#   h0 [B, H, K, V]
# Outputs:
#   y          [B, S, H, V]
#   h_last     [B, H, K, V]
#   tanh_cache [B, S, H, K, V]   -- per-step tanh(z_t) saved for bwd

_FWD_KERNEL_SOURCE = """
    uint group_id = threadgroup_position_in_grid.x;
    uint kk = thread_position_in_threadgroup.x;
    uint K_DIM = uint(KDIM);
    uint V_DIM = uint(VDIM);
    if (kk >= K_DIM) {
        return;
    }

    uint h = group_id % uint(HEADS);
    uint b = group_id / uint(HEADS);
    if (b >= uint(BATCH)) {
        return;
    }

    // Per-thread row of h_state: V floats.
    float h_row[VDIM];
    uint h_base = ((b * uint(HEADS) + h) * K_DIM + kk) * V_DIM;
    for (uint vv = 0; vv < V_DIM; ++vv) {
        h_row[vv] = float(h0[h_base + vv]);
    }

    // Threadgroup memory:
    //   - W_shared: [V, V] floats for the per-(B,H) W slice.
    //   - y_shared: [K, V] floats accumulating q^T h per step (sums over kk).
    threadgroup float W_shared[STATE_VV];
    threadgroup float y_shared[STATE_KV];

    // Cooperatively load W[h] into threadgroup memory (V*V elements, threads K).
    uint w_base_global = h * V_DIM * V_DIM;
    uint vv_total = V_DIM * V_DIM;
    for (uint i = kk; i < vv_total; i += K_DIM) {
        W_shared[i] = float(W[w_base_global + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qk_stride_t = uint(HEADS) * K_DIM;
    uint v_stride_t  = uint(HEADS) * V_DIM;
    uint xf_stride_t = uint(HEADS);
    uint qk_base = b * uint(SEQ) * qk_stride_t;
    uint v_base  = b * uint(SEQ) * v_stride_t;
    uint xf_base = b * uint(SEQ) * xf_stride_t;

    for (uint t = 0; t < uint(SEQ); ++t) {
        uint qk_idx = qk_base + t * qk_stride_t + h * K_DIM;
        uint v_idx  = v_base  + t * v_stride_t  + h * V_DIM;
        uint xf_idx = xf_base + t * xf_stride_t + h;

        float f_val = float(xf[xf_idx]);
        float one_minus_f = 1.0f - f_val;

        // Compute z[kk, vv] = sum_v0 h_row[v0] * W_shared[v0, vv] + k[kk] * v[vv]
        // Then tanh, blend, and emit per-thread.
        float k_val = float(k[qk_idx + kk]);
        float q_val = float(q[qk_idx + kk]);
        float tanh_z_row[VDIM];
        float h_new_row[VDIM];

        for (uint vv = 0; vv < V_DIM; ++vv) {
            float acc = 0.0f;
            for (uint v0 = 0; v0 < V_DIM; ++v0) {
                acc += h_row[v0] * W_shared[v0 * V_DIM + vv];
            }
            float v_val = float(v[v_idx + vv]);
            float z = acc + k_val * v_val;
            float tz;
            if (z > 20.0f) {
                tz = 1.0f;
            } else if (z < -20.0f) {
                tz = -1.0f;
            } else {
                float ez = exp(z);
                float enz = exp(-z);
                tz = (ez - enz) / (ez + enz);
            }
            tanh_z_row[vv] = tz;
            float h_new = f_val * h_row[vv] + one_minus_f * tz;
            h_new_row[vv] = h_new;
        }

        // Persist tanh_cache before overwriting h_row.
        uint cache_base = (((b * uint(SEQ) + t) * uint(HEADS) + h) * K_DIM + kk) * V_DIM;
        for (uint vv = 0; vv < V_DIM; ++vv) {
            tanh_cache[cache_base + vv] = T_OUT(tanh_z_row[vv]);
        }
        // Update h_row.
        for (uint vv = 0; vv < V_DIM; ++vv) {
            h_row[vv] = h_new_row[vv];
        }

        // Compute partial y contribution: y[vv] += q[kk] * h_new_row[vv] across all kk.
        // Stash q*h_new in y_shared[kk, vv] and reduce.
        for (uint vv = 0; vv < V_DIM; ++vv) {
            y_shared[kk * V_DIM + vv] = q_val * h_new_row[vv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction across kk -> produce y[vv] in thread 0's region.
        // For simplicity (K_DIM may be non-power-of-2), do sequential accumulation
        // assigning each thread one of V_DIM output channels.
        if (kk < V_DIM) {
            float y_acc = 0.0f;
            for (uint k_red = 0; k_red < K_DIM; ++k_red) {
                y_acc += y_shared[k_red * V_DIM + kk];
            }
            uint y_base = ((b * uint(SEQ) + t) * uint(HEADS) + h) * V_DIM;
            y[y_base + kk] = T_OUT(y_acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Persist final hidden state.
    for (uint vv = 0; vv < V_DIM; ++vv) {
        h_last[h_base + vv] = T_OUT(h_row[vv]);
    }
"""


_KERNEL_HEADER = """
    #include <metal_stdlib>
    using namespace metal;
"""


_FWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_m2rnn_fwd",
    input_names=["q", "k", "v", "W", "xf", "h0"],
    output_names=["y", "h_last", "tanh_cache"],
    source=_FWD_KERNEL_SOURCE,
    header=_KERNEL_HEADER,
)


# ---------------------------------------------------------------------------
# Backward MSL kernel
# ---------------------------------------------------------------------------
#
# TODO(wave-7): port to TileLang DSL — complex-fused (same blocker as fwd).
#
# This kernel walks the time loop both forward (Pass 1: persist h_{t-1} to
# scratch) and backward (Pass 2: cross-thread reductions over kk for dv and
# dW). To migrate, both passes must move into a single ``@T.prim_func`` with
# ``T.serial(S)`` (forward) followed by a second ``T.serial(S)`` reading the
# scratch in reverse. The cross-thread dW reduction maps to
# ``T.alloc_shared((V_DIM, V_DIM))`` + barrier, which TileLang supports.
# Same scope-of-rewrite reason as the fwd kernel.
#
# Parallelism strategy (mirrors fwd):
#   - One threadgroup per (batch, head) lane.
#   - K_DIM threads per group; each thread owns one row of h: dh[kk, :] (size V).
#
# Pass 1: forward sweep persists h_{t-1} per row to scratch.
# Pass 2: walks the time loop backwards, with cross-thread reduction (over kk)
#         for dv (sum_kk) and dW (sum_kk per (v0, vv)). dq[kk], dk[kk], dxf
#         contributions stay per-thread.
#
# Per-lane partial outputs (caller reduces over (B) for dW; dq/dk/dv/dxf are
# already (B,S,H,*) so no extra reduction):
#   dq         [B, S, H, K]
#   dk         [B, S, H, K]
#   dv         [B, S, H, V]
#   dW_partial [B, H, V, V]      -- caller sums over B
#   dxf        [B, S, H]
#   dh0        [B, H, K, V]
#
# Plus an internal scratch buffer h_steps_scratch [B*H, S, K, V] holding
# h_{t-1}.

_BWD_KERNEL_SOURCE = """
    uint group_id = threadgroup_position_in_grid.x;
    uint kk = thread_position_in_threadgroup.x;
    uint K_DIM = uint(KDIM);
    uint V_DIM = uint(VDIM);
    if (kk >= K_DIM) {
        return;
    }

    uint h = group_id % uint(HEADS);
    uint b = group_id / uint(HEADS);
    if (b >= uint(BATCH)) {
        return;
    }

    threadgroup float W_shared[STATE_VV];
    // dz_shared holds dz[kk, vv] (K * V floats), used for cross-thread reductions.
    threadgroup float dz_shared[STATE_KV];
    // dxf_partial[kk] holds per-thread df contribution; reduced by thread 0.
    threadgroup float dxf_partial[KDIM];
    // h_prev_shared holds h_{t-1}[kk, vv] for use by dW reduction across threads.
    threadgroup float h_prev_shared[STATE_KV];
    // dW_shared accumulates dW[v0, vv] across t (V * V floats per group).
    threadgroup float dW_shared[STATE_VV];

    // Cooperatively load W[h] once.
    uint w_base_global = h * V_DIM * V_DIM;
    uint vv_total = V_DIM * V_DIM;
    for (uint i = kk; i < vv_total; i += K_DIM) {
        W_shared[i] = float(W[w_base_global + i]);
        dW_shared[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint qk_stride_t = uint(HEADS) * K_DIM;
    uint v_stride_t  = uint(HEADS) * V_DIM;
    uint xf_stride_t = uint(HEADS);
    uint qk_base = b * uint(SEQ) * qk_stride_t;
    uint v_base  = b * uint(SEQ) * v_stride_t;
    uint xf_base = b * uint(SEQ) * xf_stride_t;

    // -- Pass 1: replay forward, persist h_{t-1}[kk, :] to scratch.
    uint h0_base_row = ((b * uint(HEADS) + h) * K_DIM + kk) * V_DIM;
    float h_row[VDIM];
    for (uint vv = 0; vv < V_DIM; ++vv) {
        h_row[vv] = float(h0[h0_base_row + vv]);
    }
    uint scratch_base_row = ((b * uint(HEADS) + h) * uint(SEQ) * K_DIM + kk) * V_DIM;
    // Note: layout is (B*H, S, K, V) row-major; for thread kk, addressing is:
    //   h_steps_scratch[((b*H + h) * S + t) * K_DIM * V_DIM + kk * V_DIM + vv]
    uint scratch_lane_base = (b * uint(HEADS) + h) * uint(SEQ) * K_DIM * V_DIM;
    for (uint t = 0; t < uint(SEQ); ++t) {
        uint xf_idx = xf_base + t * xf_stride_t + h;
        float f_val = float(xf[xf_idx]);
        float one_minus_f = 1.0f - f_val;

        // Persist current h_row (= h_{t-1}) at slot (t, kk, :)
        uint scratch_step_base = scratch_lane_base + t * K_DIM * V_DIM + kk * V_DIM;
        for (uint vv = 0; vv < V_DIM; ++vv) {
            h_steps_scratch[scratch_step_base + vv] = T_OUT(h_row[vv]);
        }

        // Read tanh_z[kk, :] and apply blend.
        uint cache_base_row = (((b * uint(SEQ) + t) * uint(HEADS) + h) * K_DIM + kk) * V_DIM;
        for (uint vv = 0; vv < V_DIM; ++vv) {
            float tz = float(tanh_cache[cache_base_row + vv]);
            h_row[vv] = f_val * h_row[vv] + one_minus_f * tz;
        }
    }

    // -- Pass 2: walk backwards.
    float dh_row[VDIM];
    for (uint vv = 0; vv < V_DIM; ++vv) {
        dh_row[vv] = 0.0f;
    }

    for (int t_signed = int(SEQ) - 1; t_signed >= 0; --t_signed) {
        uint t = uint(t_signed);
        uint qk_idx = qk_base + t * qk_stride_t + h * K_DIM;
        uint v_idx  = v_base  + t * v_stride_t  + h * V_DIM;
        uint xf_idx = xf_base + t * xf_stride_t + h;
        uint cache_base_row = (((b * uint(SEQ) + t) * uint(HEADS) + h) * K_DIM + kk) * V_DIM;
        uint scratch_step_kkbase = scratch_lane_base + t * K_DIM * V_DIM + kk * V_DIM;
        uint y_base = ((b * uint(SEQ) + t) * uint(HEADS) + h) * V_DIM;

        float f_val = float(xf[xf_idx]);
        float one_minus_f = 1.0f - f_val;

        // Load h_{t-1}[kk, :] and tanh_z[kk, :] for this thread; compute h_t[kk, :].
        float h_prev_row[VDIM];
        float tz_row[VDIM];
        float h_t_row[VDIM];
        for (uint vv = 0; vv < V_DIM; ++vv) {
            h_prev_row[vv] = float(h_steps_scratch[scratch_step_kkbase + vv]);
            tz_row[vv] = float(tanh_cache[cache_base_row + vv]);
            h_t_row[vv] = f_val * h_prev_row[vv] + one_minus_f * tz_row[vv];
        }

        // dY[t][vv] is shared across kk; load to per-thread.
        // dq[kk] = sum_vv dY[vv] * h_t[kk, vv]
        // dh[kk, vv] += q[kk] * dY[vv]
        float q_val = float(q[qk_idx + kk]);
        float dq_kk = 0.0f;
        for (uint vv = 0; vv < V_DIM; ++vv) {
            float dY_v = float(dy[y_base + vv]);
            dq_kk += dY_v * h_t_row[vv];
            dh_row[vv] += q_val * dY_v;
        }
        dq[qk_idx + kk] = T_OUT(dq_kk);

        // Compute df_t per-thread (sum over vv of this row), then reduce.
        float df_kk = 0.0f;
        float dz_row[VDIM];
        for (uint vv = 0; vv < V_DIM; ++vv) {
            float dh_kv = dh_row[vv];
            df_kk += dh_kv * (h_prev_row[vv] - tz_row[vv]);
            float dh_new = one_minus_f * dh_kv;
            float one_minus_t2 = 1.0f - tz_row[vv] * tz_row[vv];
            dz_row[vv] = dh_new * one_minus_t2;
        }
        dxf_partial[kk] = df_kk;

        // dk[kk] = sum_vv dz_row[vv] * v_t[vv]  (per-thread, no reduction needed).
        float dk_kk = 0.0f;
        for (uint vv = 0; vv < V_DIM; ++vv) {
            dk_kk += dz_row[vv] * float(v[v_idx + vv]);
        }
        dk[qk_idx + kk] = T_OUT(dk_kk);

        // Stash dz_shared and h_prev_shared for the cross-thread reductions below.
        for (uint vv = 0; vv < V_DIM; ++vv) {
            dz_shared[kk * V_DIM + vv] = dz_row[vv];
            h_prev_shared[kk * V_DIM + vv] = h_prev_row[vv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dxf reduction: thread 0 sums dxf_partial[0..K_DIM] -> dxf[t]
        if (kk == 0) {
            float df_total = 0.0f;
            for (uint kr = 0; kr < K_DIM; ++kr) {
                df_total += dxf_partial[kr];
            }
            dxf[xf_idx] = T_OUT(df_total);
        }

        // dv[vv] = sum_kk dz[kk, vv] * k_t[kk]; assign each thread one vv (kk < V_DIM).
        if (kk < V_DIM) {
            float dv_acc = 0.0f;
            for (uint kr = 0; kr < K_DIM; ++kr) {
                dv_acc += dz_shared[kr * V_DIM + kk] * float(k[qk_idx + kr]);
            }
            dv[v_idx + kk] = T_OUT(dv_acc);
        }

        // dW[v0, vv] += sum_kk h_prev[kk, v0] * dz[kk, vv].
        // Distribute (v0, vv) pairs across threads; each thread handles
        // multiple pairs if V*V > K_DIM.
        for (uint pair = kk; pair < V_DIM * V_DIM; pair += K_DIM) {
            uint v0 = pair / V_DIM;
            uint vv = pair % V_DIM;
            float w_acc = 0.0f;
            for (uint kr = 0; kr < K_DIM; ++kr) {
                w_acc += h_prev_shared[kr * V_DIM + v0] * dz_shared[kr * V_DIM + vv];
            }
            dW_shared[v0 * V_DIM + vv] += w_acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Update dh_row = f * dh_row + dz_row @ W^T.
        // (dz @ W^T)[kk, v_in] = sum_v_out dz[kk, v_out] * W[v_in, v_out]
        float dh_next[VDIM];
        for (uint v_in = 0; v_in < V_DIM; ++v_in) {
            float acc = f_val * dh_row[v_in];
            for (uint v_out = 0; v_out < V_DIM; ++v_out) {
                acc += dz_row[v_out] * W_shared[v_in * V_DIM + v_out];
            }
            dh_next[v_in] = acc;
        }
        for (uint vv = 0; vv < V_DIM; ++vv) {
            dh_row[vv] = dh_next[vv];
        }
    }

    // Persist dh0 row (gradient that propagated past t=0).
    for (uint vv = 0; vv < V_DIM; ++vv) {
        dh0[h0_base_row + vv] = T_OUT(dh_row[vv]);
    }
    // Persist dW per lane: write the threadgroup-shared tile to (b, h, V, V).
    uint dW_lane_base = (b * uint(HEADS) + h) * V_DIM * V_DIM;
    for (uint i = kk; i < V_DIM * V_DIM; i += K_DIM) {
        dW_partial[dW_lane_base + i] = T_OUT(dW_shared[i]);
    }
"""


_BWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_m2rnn_bwd",
    input_names=["dy", "q", "k", "v", "W", "xf", "h0", "tanh_cache"],
    output_names=[
        "dq",
        "dk",
        "dv",
        "dW_partial",
        "dxf",
        "dh0",
        "h_steps_scratch",
    ],
    source=_BWD_KERNEL_SOURCE,
    header=_KERNEL_HEADER,
)


# ---------------------------------------------------------------------------
# Validation / dtype helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
) -> tuple[int, int, int, int, int]:
    """Return (B, S, H, K_DIM, V_DIM) and validate post-broadcast shapes."""

    if q.ndim != 4:
        raise ValueError(f"q must be (B,S,H,K), got {q.shape}")
    batch, seq, heads, k_dim = q.shape
    if k.shape != (batch, seq, heads, k_dim):
        raise ValueError(f"k must be {(batch, seq, heads, k_dim)}, got {k.shape}")
    if v.ndim != 4:
        raise ValueError(f"v must be 4D, got {v.shape}")
    v_dim = v.shape[-1]
    if v.shape != (batch, seq, heads, v_dim):
        raise ValueError(f"v must be {(batch, seq, heads, v_dim)}, got {v.shape}")
    if W.shape != (heads, v_dim, v_dim):
        raise ValueError(f"W must be {(heads, v_dim, v_dim)}, got {W.shape}")
    if xf.shape != (batch, seq, heads):
        raise ValueError(f"xf must be {(batch, seq, heads)}, got {xf.shape}")
    if h0 is not None and h0.shape != (batch, heads, k_dim, v_dim):
        raise ValueError(
            f"h0 must be {(batch, heads, k_dim, v_dim)}, got {h0.shape}"
        )
    return batch, seq, heads, k_dim, v_dim


def _carrier_dtype(dtype: mx.Dtype) -> mx.Dtype:
    """Return the working dtype: bf16 inputs are upcast to fp16 to dodge MSL bugs."""

    if dtype == mx.bfloat16:
        return mx.float16
    return dtype


def _broadcast_inputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Reuse the parent module's head broadcast helper.

    We delegate so the kernel sees pre-broadcast tensors, matching the contract
    of m2rnn_scan / chunked_m2rnn_scan.
    """

    from cppmega_mlx.nn.m2rnn import broadcast_m2rnn_heads
    return broadcast_m2rnn_heads(q, k, v, W, xf)


def m2rnn_metal_status(*arrays: mx.array) -> M2RNNMetalStatus:
    """Report Path B Metal eligibility for the M2RNN kernel."""

    status = _msl_transform.msl_dispatch_status(*arrays)
    if not status.available:
        return M2RNNMetalStatus(False, status.reason)
    if _FWD_KERNEL is None:
        return M2RNNMetalStatus(False, "vendor MSL fwd kernel was not constructed")
    if _BWD_KERNEL is None:
        return M2RNNMetalStatus(False, "vendor MSL bwd kernel was not constructed")
    return M2RNNMetalStatus(True, status.reason)


def m2rnn_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    *,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Pure-MLX reference — thin call into :func:cppmega_mlx.nn.m2rnn.m2rnn_scan.

    This module is deliberately a Path B port; the reference math lives in
    the parent module so we have a single source of truth.
    """

    from cppmega_mlx.nn.m2rnn import m2rnn_scan
    return m2rnn_scan(q, k, v, W, xf, h0=h0)


# ---------------------------------------------------------------------------
# Forward / Backward Metal dispatch
# ---------------------------------------------------------------------------


def _materialize_h0(
    h0: mx.array | None,
    *,
    batch: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
) -> mx.array:
    if h0 is None:
        return mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype)
    return h0.astype(dtype)


def m2rnn_fwd_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Path B Metal forward, returning (y, h_last, tanh_cache).

    Falls back to the reference scan if Metal is unavailable; in that case
    tanh_cache is recomputed in pure MLX so the bwd contract still
    holds.
    """

    q, k, v, W, xf = _broadcast_inputs(q, k, v, W, xf)
    batch, seq, heads, k_dim, v_dim = _validate_inputs(q, k, v, W, xf, h0)
    out_dtype = q.dtype
    cast_dtype = _carrier_dtype(out_dtype)

    h0_full = _materialize_h0(
        h0, batch=batch, heads=heads, k_dim=k_dim, v_dim=v_dim, dtype=cast_dtype
    )

    if seq == 0:
        return (
            mx.zeros((batch, 0, heads, v_dim), dtype=out_dtype),
            h0_full.astype(out_dtype),
            mx.zeros((batch, 0, heads, k_dim, v_dim), dtype=cast_dtype),
        )

    status = m2rnn_metal_status(q)
    if not status.available or _FWD_KERNEL is None:
        return _m2rnn_fwd_pure_mlx(q, k, v, W, xf, h0_full, out_dtype=out_dtype)

    inputs = [
        q.astype(cast_dtype),
        k.astype(cast_dtype),
        v.astype(cast_dtype),
        W.astype(cast_dtype),
        xf.astype(cast_dtype),
        h0_full,
    ]

    # One threadgroup per (batch, head) lane, K_DIM threads per group.
    # Grid is laid out as (groups * threads_per_group, 1, 1) since MLX expects
    # the grid as the total thread count and threadgroup as the per-group size.
    groups = batch * heads
    threads_per_group = max(k_dim, v_dim)
    template = [
        ("T_OUT", cast_dtype),
        ("BATCH", batch),
        ("SEQ", seq),
        ("HEADS", heads),
        ("KDIM", k_dim),
        ("VDIM", v_dim),
        ("STATE_KV", k_dim * v_dim),
        ("STATE_VV", v_dim * v_dim),
    ]
    try:
        outputs = _msl_transform.dispatch(
            cast(_msl_transform.MetalKernel, _FWD_KERNEL),
            inputs=inputs,
            output_shapes=[
                (batch, seq, heads, v_dim),
                (batch, heads, k_dim, v_dim),
                (batch, seq, heads, k_dim, v_dim),
            ],
            output_dtypes=[cast_dtype, cast_dtype, cast_dtype],
            grid=(groups * threads_per_group, 1, 1),
            threadgroup=(threads_per_group, 1, 1),
            template=template,
        )
    except Exception:
        return _m2rnn_fwd_pure_mlx(q, k, v, W, xf, h0_full, out_dtype=out_dtype)

    y, h_last, tanh_cache = outputs
    return y.astype(out_dtype), h_last.astype(out_dtype), tanh_cache


def _m2rnn_fwd_pure_mlx(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
    *,
    out_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reference forward that also returns the tanh_cache the bwd kernel needs.

    Math identical to :func:cppmega_mlx.nn.m2rnn.m2rnn_scan. We cannot just
    call the parent helper because we also need the per-step tanh(z_t)
    saved out for backward.
    """

    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if seq == 0:
        return (
            mx.zeros((batch, 0, heads, v_dim), dtype=out_dtype),
            h0.astype(out_dtype),
            mx.zeros((batch, 0, heads, k_dim, v_dim), dtype=h0.dtype),
        )

    work_dtype = mx.float32
    h = h0.astype(work_dtype)
    W_e = W.astype(work_dtype)[None, :, :, :]  # (1, H, V, V)
    x_all = (
        mx.expand_dims(k.astype(work_dtype), -1)
        * mx.expand_dims(v.astype(work_dtype), -2)
    )  # (B, S, H, K, V)
    xf_5d = xf.astype(work_dtype)[:, :, :, None, None]
    out_steps: list[mx.array] = []
    tanh_steps: list[mx.array] = []
    for s in range(seq):
        f = xf_5d[:, s]  # (B, H, 1, 1)
        z = mx.matmul(h, W_e[0]) + x_all[:, s]
        tz = mx.tanh(z)
        tanh_steps.append(tz)
        h = f * h + (1.0 - f) * tz
        # y[s, vv] = sum_kk q[kk] * h[kk, vv] -> einsum bhk,bhkv->bhv
        y_s = mx.einsum("bhk,bhkv->bhv", q[:, s].astype(work_dtype), h)
        out_steps.append(y_s)
    y = mx.stack(out_steps, axis=1).astype(out_dtype)
    h_final = h.astype(out_dtype)
    tanh_cache = mx.stack(tanh_steps, axis=1).astype(h0.dtype)
    return y, h_final, tanh_cache


def m2rnn_bwd_metal(
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Backward pass returning gradients for (q, k, v, W, xf, h0).

    Tries the Metal kernel first; on dispatch failure or when Metal is
    unavailable, falls back to a pure-MLX implementation that reproduces
    the same math step-by-step (still uses tanh_cache from the fwd).
    """

    q, k, v, W, xf = _broadcast_inputs(q, k, v, W, xf)
    batch, seq, heads, k_dim, v_dim = _validate_inputs(q, k, v, W, xf, h0)
    if dy.shape != (batch, seq, heads, v_dim):
        raise ValueError(f"dy must be {(batch, seq, heads, v_dim)}, got {dy.shape}")

    out_dtypes = (q.dtype, k.dtype, v.dtype, W.dtype, xf.dtype)
    h0_dtype = h0.dtype if h0 is not None else q.dtype
    cast_dtype = _carrier_dtype(q.dtype)

    h0_full = _materialize_h0(
        h0, batch=batch, heads=heads, k_dim=k_dim, v_dim=v_dim, dtype=cast_dtype
    )

    if seq == 0:
        return (
            mx.zeros_like(q),
            mx.zeros_like(k),
            mx.zeros_like(v),
            mx.zeros_like(W),
            mx.zeros_like(xf),
            h0_full.astype(h0_dtype),
        )

    metal_grads = _m2rnn_bwd_metal_kernel(
        dy=dy,
        q=q,
        k=k,
        v=v,
        W=W,
        xf=xf,
        h0_full=h0_full,
        tanh_cache=tanh_cache,
        cast_dtype=cast_dtype,
    )
    if metal_grads is not None:
        dq, dk, dv, dW, dxf_, dh0 = metal_grads
        return (
            dq.astype(out_dtypes[0]),
            dk.astype(out_dtypes[1]),
            dv.astype(out_dtypes[2]),
            dW.astype(out_dtypes[3]),
            dxf_.astype(out_dtypes[4]),
            dh0.astype(h0_dtype),
        )
    # Fall back to pure-MLX bwd.
    return _m2rnn_bwd_pure_mlx(
        dy=dy,
        q=q,
        k=k,
        v=v,
        W=W,
        xf=xf,
        h0_full=h0_full,
        tanh_cache=tanh_cache,
        out_dtypes=out_dtypes,
        h0_dtype=h0_dtype,
    )


def _m2rnn_bwd_metal_kernel(
    *,
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0_full: mx.array,
    tanh_cache: mx.array,
    cast_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array] | None:
    """Try the Metal bwd kernel; return None if dispatch is not eligible."""

    if _BWD_KERNEL is None:
        return None
    status = m2rnn_metal_status(q)
    if not status.available:
        return None

    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if seq == 0:
        return None

    inputs = [
        dy.astype(cast_dtype),
        q.astype(cast_dtype),
        k.astype(cast_dtype),
        v.astype(cast_dtype),
        W.astype(cast_dtype),
        xf.astype(cast_dtype),
        h0_full,
        tanh_cache.astype(cast_dtype),
    ]

    groups = batch * heads
    threads_per_group = max(k_dim, v_dim)
    template = [
        ("T_OUT", cast_dtype),
        ("BATCH", batch),
        ("SEQ", seq),
        ("HEADS", heads),
        ("KDIM", k_dim),
        ("VDIM", v_dim),
        ("STATE_KV", k_dim * v_dim),
        ("STATE_VV", v_dim * v_dim),
    ]
    try:
        outputs = _msl_transform.dispatch(
            cast(_msl_transform.MetalKernel, _BWD_KERNEL),
            inputs=inputs,
            output_shapes=[
                (batch, seq, heads, k_dim),               # dq
                (batch, seq, heads, k_dim),               # dk
                (batch, seq, heads, v_dim),               # dv
                (batch, heads, v_dim, v_dim),             # dW_partial
                (batch, seq, heads),                      # dxf
                (batch, heads, k_dim, v_dim),             # dh0
                (batch * heads, seq, k_dim * v_dim),      # h_steps_scratch
            ],
            output_dtypes=[cast_dtype] * 7,
            grid=(groups * threads_per_group, 1, 1),
            threadgroup=(threads_per_group, 1, 1),
            template=template,
        )
    except Exception:
        return None
    dq, dk, dv, dW_partial, dxf_, dh0, _scratch = outputs
    # Reduce dW over batch.
    dW = mx.sum(dW_partial, axis=0)  # (H, V, V)
    return dq, dk, dv, dW, dxf_, dh0


def _m2rnn_bwd_pure_mlx(
    *,
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0_full: mx.array,
    tanh_cache: mx.array,
    out_dtypes: tuple[mx.Dtype, ...],
    h0_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Pure-MLX backward. Matches the kernel math step-by-step."""

    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    work_dtype = mx.float32

    q_f = q.astype(work_dtype)
    k_f = k.astype(work_dtype)
    v_f = v.astype(work_dtype)
    W_f = W.astype(work_dtype)
    xf_f = xf.astype(work_dtype)
    h0_f = h0_full.astype(work_dtype)
    dy_f = dy.astype(work_dtype)
    tanh_f = tanh_cache.astype(work_dtype)

    # Re-materialise h_{t-1} sequence (h_steps[t] is h_{t-1} at step t).
    h_steps: list[mx.array] = []
    h = h0_f
    for t in range(seq):
        h_steps.append(h)
        f = xf_f[:, t, :, None, None]
        h = f * h + (1.0 - f) * tanh_f[:, t]

    # Walk backwards.
    dh = mx.zeros_like(h0_f)
    dW_acc = mx.zeros((heads, v_dim, v_dim), dtype=work_dtype)
    dq_steps: list[mx.array] = [mx.zeros((batch, heads, k_dim), dtype=work_dtype)] * seq
    dk_steps: list[mx.array] = [mx.zeros((batch, heads, k_dim), dtype=work_dtype)] * seq
    dv_steps: list[mx.array] = [mx.zeros((batch, heads, v_dim), dtype=work_dtype)] * seq
    dxf_steps: list[mx.array] = [mx.zeros((batch, heads), dtype=work_dtype)] * seq

    for t in range(seq - 1, -1, -1):
        f = xf_f[:, t, :, None, None]            # (B, H, 1, 1)
        f_flat = xf_f[:, t]                       # (B, H)
        one_minus_f = 1.0 - f
        h_prev = h_steps[t]                      # (B, H, K, V)
        tz = tanh_f[:, t]                        # (B, H, K, V)
        h_t = f * h_prev + one_minus_f * tz      # (B, H, K, V)

        # dY[t] is (B, H, V); dq[kk] += sum_vv dY[vv] * h_t[kk, vv]
        dY_t = dy_f[:, t]                        # (B, H, V)
        dq_steps[t] = mx.einsum("bhv,bhkv->bhk", dY_t, h_t)

        # dh += q[..., None] * dY[..., None, :]
        dh = dh + q_f[:, t, :, :, None] * dY_t[:, :, None, :]

        # df_t = sum_kv dh * (h_prev - tz)
        df = mx.sum(dh * (h_prev - tz), axis=(-1, -2))  # (B, H)
        dxf_steps[t] = df

        # dh_new = (1 - f) * dh; dz = dh_new * (1 - tanh^2)
        dh_new = one_minus_f * dh
        dz = dh_new * (1.0 - tz * tz)

        # dk[kk] = sum_vv dz[kk, vv] * v_t[vv]
        dk_steps[t] = mx.sum(dz * v_f[:, t, :, None, :], axis=-1)
        # dv[vv] = sum_kk dz[kk, vv] * k_t[kk]
        dv_steps[t] = mx.sum(dz * k_f[:, t, :, :, None], axis=-2)
        # dW[h, v0, vv] += sum_b sum_kk h_prev[b, h, kk, v0] * dz[b, h, kk, vv]
        dW_step = mx.einsum("bhkv,bhku->bhvu", h_prev, dz)  # (B, H, V_in, V_out)
        dW_acc = dW_acc + mx.sum(dW_step, axis=0)

        # dh_next = f * dh + dz @ W^T  (where dz is (B,H,K,V_out), W is (H, V_in, V_out))
        dh = f * dh + mx.einsum("bhku,hvu->bhkv", dz, W_f)

    dq = mx.stack(dq_steps, axis=1).astype(out_dtypes[0])
    dk = mx.stack(dk_steps, axis=1).astype(out_dtypes[1])
    dv = mx.stack(dv_steps, axis=1).astype(out_dtypes[2])
    dxf_out = mx.stack(dxf_steps, axis=1).astype(out_dtypes[4])
    dW_out = dW_acc.astype(out_dtypes[3])
    dh0_out = dh.astype(h0_dtype)
    return dq, dk, dv, dW_out, dxf_out, dh0_out


# ---------------------------------------------------------------------------
# Differentiable wrappers
# ---------------------------------------------------------------------------


def _zeros_for_h0(
    q: mx.array,
    *,
    heads: int,
    k_dim: int,
    v_dim: int,
) -> mx.array:
    """Return a default zero h0 used when the caller passes None."""
    return mx.zeros((q.shape[0], heads, k_dim, v_dim), dtype=q.dtype)


@mx.custom_function
def m2rnn_apply(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> mx.array:
    """Forward-only wrapper returning the gated output y.

    The Metal forward also produces h_last and tanh_cache; we only
    expose y from the differentiable surface so the VJP signature stays
    aligned with autograd-through-reference. Callers that need h_last
    use :func:m2rnn_apply_with_state (custom_function returning a tuple).
    """

    y, _h, _t = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    return y


@m2rnn_apply.vjp
def _m2rnn_apply_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array, ...]:
    del output
    q, k, v, W, xf, h0 = primals
    # Re-run forward to obtain the saved tanh_cache (needed for bwd).
    _y, _h, tanh_cache = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    return m2rnn_bwd_metal(cotangent, q, k, v, W, xf, tanh_cache, h0)


@mx.custom_function
def m2rnn_apply_with_state(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Differentiable Path B forward returning (y, h_last).

    Wraps :func:m2rnn_fwd_metal and reuses the manual VJP via
    :func:m2rnn_bwd_metal. Gradients flow only through y; the
    cotangent for h_last is treated as zero. This matches the
    production model contract: training loss flows through y, while
    h_last is consumed by the inference cache only.
    """

    y, h_last, _ = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    return y, h_last


@m2rnn_apply_with_state.vjp
def _m2rnn_apply_with_state_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    del output
    q, k, v, W, xf, h0 = primals
    dy = cotangent[0]  # ignore the h_last cotangent.
    _y, _h, tanh_cache = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    return m2rnn_bwd_metal(dy, q, k, v, W, xf, tanh_cache, h0)


__all__ = [
    "M2RNNMetalStatus",
    "m2rnn_apply",
    "m2rnn_apply_with_state",
    "m2rnn_bwd_metal",
    "m2rnn_fwd_metal",
    "m2rnn_metal_status",
    "m2rnn_reference",
]
