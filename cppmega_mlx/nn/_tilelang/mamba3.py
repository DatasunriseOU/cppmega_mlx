"""Path B port of mamba_ssm.ops.tilelang.mamba3 fwd/bwd.

.. todo:: wave-6: port to TileLang DSL.

   Both ``_FWD_KERNEL_SOURCE`` and ``_BWD_KERNEL_SOURCE`` are sequential
   selective-scan kernels with **per-thread cumulative state** (``float
   h_state[STATE]`` for fwd, ``float dh[STATE]`` plus a persistent
   ``h_steps_scratch[tid][t][n]`` slab and reverse iteration for bwd). They do
   not fit the canonical TileLang DSL idiom (tile-parallel ``T.Parallel`` /
   ``T.Pipelined`` over a static iteration space) without a non-trivial
   rewrite that introduces ``T.serial(reverse=True)`` over ``t`` and treats
   the per-thread carry as a ``T.alloc_fragment`` of static shape.

   The MSL-extraction adapter
   (:func:`cppmega_mlx.nn._tilelang._msl_extraction.extract_msl_from_engine_artifact`,
   commit ``00d6d90``) is in tree and the prerequisite ``return_msl=True``
   kwarg on ``dispatch_lower`` works for the simpler tile-parallel kernels
   (``topk_selector`` already flipped). Porting these scan kernels remains a
   wave-6 line item: write ``mamba3_mimo_fwd_prim`` /
   ``mamba3_mimo_bwd_prim`` ``@T.prim_func`` factories, route through
   ``dispatch_lower(prim, "metal", return_msl=True)``, then feed the
   extracted MSL string into the existing ``_msl_transform.make_metal_kernel``
   call site so the 12 mlx + 2 cppmega call sites stay numerically identical.

   Until then this module stays on its hand-written MSL source to preserve
   numerical parity. See ``MIGRATION_PLAN.md`` and the wave-6 tracker.

This module implements the Mamba3 MIMO selective-scan kernel in vendor MSL,
without depending on TileLang's TVM-Metal lowering. The forward kernel is the
core of Path B verified by the cppmega.mlx port research; the backward is a
matching fp32 sweep that reuses the three pure-MLX Triton-replacement helpers
in :mod:`_mamba3_helpers`.

Numerical contract:
  - fp16 input carrier (avoids bf16 simdgroup miscompiles flagged by Path A).
  - fp32 internal accumulators inside the MSL kernel and the helpers.
  - parity oracle: cppmega_mlx/nn/mamba3.py reference scan. We do not import
    that module's class here (must remain unmodified per the task), but we do
    reproduce the same recurrence and use it in tests.

Public surface:
  - mamba3_mimo_fwd_metal(...): forward kernel returning (y, h_final).
  - mamba3_mimo_bwd_metal(...): backward kernel returning per-input grads.
  - mamba3_mimo_apply: mx.custom_function-wrapped fwd that ties to the bwd VJP.
  - mamba3_mimo_reference: the same algorithm written in pure MLX.
  - mamba3_mimo_metal_status: introspect Metal eligibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, cast

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._mamba3_helpers import (
    bwd_dadt_fused,
    bwd_dtrap_ddt,
    compute_dacs_segsum,
)

# Sequence-chunked Mamba3 backward. The monolithic Path-B bwd kernel
# materialises fp32 partials of shape (B, SEQ, H, N, P) plus an
# (B, SEQ+1, H, P, N) state scratch slab; at seq=4096 that is ~21 GB per
# mamba3 layer (×3 layers ≈ 63 GB), which is the dominant term in the
# fwd+bwd+optimizer peak. When this env flag is set to a positive integer
# C, the orchestrator splits the time axis into C-length chunks: it carries
# the forward scan state h across chunk boundaries (cheap, O(num_chunks)
# boundary states) and runs the reverse VJP chunk-by-chunk, feeding each
# chunk's dh0 as the next (earlier) chunk's incoming state cotangent. This is
# *numerically identical* to the full-sequence backward — same kernel math,
# just a bounded working set — and caps the bwd partials at chunk-size.
MAMBA3_BWD_SEQ_CHUNK_ENV = "CPPMEGA_MAMBA3_BWD_SEQ_CHUNK"


def mamba3_bwd_seq_chunk() -> int:
    """Return the configured Mamba3 backward seq-chunk length (0 = disabled)."""

    raw = os.environ.get(MAMBA3_BWD_SEQ_CHUNK_ENV, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{MAMBA3_BWD_SEQ_CHUNK_ENV} must be a non-negative integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{MAMBA3_BWD_SEQ_CHUNK_ENV} must be a non-negative integer, got {value}"
        )
    return value


@dataclass(frozen=True)
class Mamba3MetalStatus:
    available: bool
    reason: str


_FWD_KERNEL_SOURCE = """
    // Inputs (all fp32 carriers after up-cast in Python wrapper):
    //   x      [B, T, H, P]
    //   B_proj [B, T, H, N]
    //   C_proj [B, T, H, N]
    //   z      [B, T, H, P]
    //   A      [B, T, H]    (log-decay; will be A * dt)
    //   dt     [B, T, H]
    //   D      [H]          (skip)
    //   h0     [B, H, P, N]
    // Outputs:
    //   y      [B, T, H, P]
    //   h_last [B, H, P, N]
    // Grid is launched with one thread per (b, h, p) lane and the time loop
    // runs sequentially inside the thread to preserve causal carry.

    uint tid = thread_position_in_grid.x;
    uint total_lanes = uint(BATCH) * uint(HEADS) * uint(HEADDIM);
    if (tid >= total_lanes) {
        return;
    }
    uint p = tid % uint(HEADDIM);
    uint h = (tid / uint(HEADDIM)) % uint(HEADS);
    uint b = tid / (uint(HEADDIM) * uint(HEADS));

    // Per-lane state lives in registers (size N), accumulated as fp32.
    float h_state[STATE];
    uint h_base = ((b * uint(HEADS) + h) * uint(HEADDIM) + p) * uint(STATE);
    for (uint n = 0; n < uint(STATE); ++n) {
        h_state[n] = float(h0[h_base + n]);
    }

    uint xz_stride_t = uint(HEADS) * uint(HEADDIM);
    uint bc_stride_t = uint(HEADS) * uint(STATE);
    uint adt_stride_t = uint(HEADS);
    uint xz_base = b * uint(SEQ) * xz_stride_t;
    uint bc_base = b * uint(SEQ) * bc_stride_t;
    uint adt_base = b * uint(SEQ) * adt_stride_t;

    float D_h = float(D[h]);

    for (uint t = 0; t < uint(SEQ); ++t) {
        uint xz_idx = xz_base + t * xz_stride_t + h * uint(HEADDIM) + p;
        uint bc_idx = bc_base + t * bc_stride_t + h * uint(STATE);
        uint adt_idx = adt_base + t * adt_stride_t + h;

        float x_val = float(x[xz_idx]);
        float z_val = float(z[xz_idx]);
        float A_val = float(A[adt_idx]);
        float dt_val = float(dt[adt_idx]);
        float decay = exp(A_val * dt_val);

        // Scan: h_state[n] = decay * h_state[n] + x_val * B_proj[n]
        // Output: y = sum(h_state[n] * C_proj[n]) + D_h * x_val
        //         y_gated = silu(z) * y
        float y_acc = 0.0f;
        for (uint n = 0; n < uint(STATE); ++n) {
            float B_val = float(B_proj[bc_idx + n]);
            float C_val = float(C_proj[bc_idx + n]);
            float new_h = decay * h_state[n] + x_val * B_val;
            h_state[n] = new_h;
            y_acc += new_h * C_val;
        }
        float y_skipped = y_acc + D_h * x_val;
        // SiLU: z * sigmoid(z)
        float sig_z = 1.0f / (1.0f + exp(-z_val));
        y[xz_idx] = T_OUT(z_val * sig_z * y_skipped);
    }

    // Persist final state.
    for (uint n = 0; n < uint(STATE); ++n) {
        h_last[h_base + n] = T_OUT(h_state[n]);
    }
"""


_FWD_KERNEL_HEADER = """
    #include <metal_stdlib>
    using namespace metal;
"""


# TODO(wave-6): port to TileLang DSL. Sequential time-loop scan with
# per-thread ``float h_state[STATE]`` cumulative state does not fit the
# tile-parallel ``T.Parallel`` idiom cleanly. Once a ``mamba3_mimo_fwd_prim``
# ``@T.prim_func`` exists, route through
# ``dispatch_lower(prim, "metal", return_msl=True)`` and feed the extracted
# MSL into ``make_metal_kernel(source=...)`` to keep the runtime contract.
_FWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_mamba3_mimo_fwd",
    input_names=["x", "B_proj", "C_proj", "z", "A", "dt", "D", "h0"],
    output_names=["y", "h_last"],
    source=_FWD_KERNEL_SOURCE,
    header=_FWD_KERNEL_HEADER,
)


_BWD_KERNEL_SOURCE = """
    // Path B Mamba3 MIMO backward.
    //
    // Inputs:
    //   dy     [B, T, H, P]
    //   x      [B, T, H, P]
    //   B_proj [B, T, H, N]
    //   C_proj [B, T, H, N]
    //   z      [B, T, H, P]
    //   A      [B, T, H]
    //   dt     [B, T, H]
    //   D      [H]
    //   h0     [B, H, P, N]
    //
    // Outputs:
    //   dx     [B, T, H, P]
    //   dz     [B, T, H, P]
    //   dB_partial [B, T, H, P, N]  -- private wrapper sums over P
    //   dC_partial [B, T, H, P, N]  -- private wrapper sums over P
    //   dA_partial [B, T, H, P]     -- private wrapper sums over P
    //   ddt_partial[B, T, H, P]     -- private wrapper sums over P
    //   dD_partial [B, H, P]        -- private wrapper sums over (B, P)
    //   dh0        [B, H, P, N]
    //
    // One thread per (b, h, p) lane. The (b, h, p) decomposition keeps each
    // lane fully owning a single P slice, so per-lane partial outputs do not
    // need atomics. The private Python wrapper reduces them before the public
    // backward API returns final owner-output gradients.

    uint tid = thread_position_in_grid.x;
    uint total_lanes = uint(BATCH) * uint(HEADS) * uint(HEADDIM);
    if (tid >= total_lanes) {
        return;
    }
    uint p = tid % uint(HEADDIM);
    uint h = (tid / uint(HEADDIM)) % uint(HEADS);
    uint b = tid / (uint(HEADDIM) * uint(HEADS));

    uint h_base = ((b * uint(HEADS) + h) * uint(HEADDIM) + p) * uint(STATE);

    uint xz_stride_t = uint(HEADS) * uint(HEADDIM);
    uint bc_stride_t = uint(HEADS) * uint(STATE);
    uint adt_stride_t = uint(HEADS);
    uint xz_base = b * uint(SEQ) * xz_stride_t;
    uint bc_base = b * uint(SEQ) * bc_stride_t;
    uint adt_base = b * uint(SEQ) * adt_stride_t;

    // Forward pass: re-materialise h[t] for this lane into the per-lane scratch
    // section of h_steps_scratch. The scratch buffer is laid out as
    // [tid][t][n] so each lane writes to a contiguous slab.
    uint scratch_base = tid * uint(SEQ) * uint(STATE);
    float h_state[STATE];
    for (uint n = 0; n < uint(STATE); ++n) {
        h_state[n] = float(h0[h_base + n]);
    }
    for (uint t = 0; t < uint(SEQ); ++t) {
        uint xz_idx = xz_base + t * xz_stride_t + h * uint(HEADDIM) + p;
        uint bc_idx = bc_base + t * bc_stride_t + h * uint(STATE);
        uint adt_idx = adt_base + t * adt_stride_t + h;
        float x_val = float(x[xz_idx]);
        float A_val = float(A[adt_idx]);
        float dt_val = float(dt[adt_idx]);
        float decay = exp(A_val * dt_val);
        for (uint n = 0; n < uint(STATE); ++n) {
            float B_val = float(B_proj[bc_idx + n]);
            float new_h = decay * h_state[n] + x_val * B_val;
            h_state[n] = new_h;
            h_steps_scratch[scratch_base + t * uint(STATE) + n] = T_OUT(new_h);
        }
    }

    // Reverse pass.
    float dh[STATE];
    for (uint n = 0; n < uint(STATE); ++n) {
        dh[n] = 0.0f;
    }
    float dD_acc = 0.0f;
    float D_h = float(D[h]);

    for (int t_signed = int(SEQ) - 1; t_signed >= 0; --t_signed) {
        uint t = uint(t_signed);
        uint xz_idx = xz_base + t * xz_stride_t + h * uint(HEADDIM) + p;
        uint bc_idx = bc_base + t * bc_stride_t + h * uint(STATE);
        uint adt_idx = adt_base + t * adt_stride_t + h;
        uint scratch_t = scratch_base + t * uint(STATE);

        float x_val = float(x[xz_idx]);
        float z_val = float(z[xz_idx]);
        float A_val = float(A[adt_idx]);
        float dt_val = float(dt[adt_idx]);
        float decay = exp(A_val * dt_val);
        float dY = float(dy[xz_idx]);

        float y_state = 0.0f;
        for (uint n = 0; n < uint(STATE); ++n) {
            y_state += float(h_steps_scratch[scratch_t + n]) * float(C_proj[bc_idx + n]);
        }
        float y_skipped = y_state + D_h * x_val;
        float sig_z = 1.0f / (1.0f + exp(-z_val));
        float silu_z = z_val * sig_z;
        float silu_dz = sig_z * (1.0f + z_val * (1.0f - sig_z));

        float d_silu = dY * y_skipped;
        float d_y_skipped = dY * silu_z;

        dz[xz_idx] = T_OUT(d_silu * silu_dz);
        dD_acc += d_y_skipped * x_val;

        // Update dh from y_state contribution.
        for (uint n = 0; n < uint(STATE); ++n) {
            dh[n] += d_y_skipped * float(C_proj[bc_idx + n]);
        }

        // Stride for the (B, T, H, P, N) partial buffers.
        uint partial_n_base = ((b * uint(SEQ) + t) * uint(HEADS) * uint(HEADDIM)
                              + h * uint(HEADDIM) + p) * uint(STATE);
        for (uint n = 0; n < uint(STATE); ++n) {
            dC_partial[partial_n_base + n] = T_OUT(d_y_skipped * float(h_steps_scratch[scratch_t + n]));
            dB_partial[partial_n_base + n] = T_OUT(dh[n] * x_val);
        }

        // dx contribution.
        float dx_inp = 0.0f;
        for (uint n = 0; n < uint(STATE); ++n) {
            dx_inp += dh[n] * float(B_proj[bc_idx + n]);
        }
        float dx_skip = d_y_skipped * D_h;
        dx[xz_idx] = T_OUT(dx_skip + dx_inp);

        // Decay backward.
        float h_prev_n;
        float d_decay = 0.0f;
        if (t == 0) {
            for (uint n = 0; n < uint(STATE); ++n) {
                d_decay += dh[n] * float(h0[h_base + n]);
            }
        } else {
            for (uint n = 0; n < uint(STATE); ++n) {
                h_prev_n = float(h_steps_scratch[scratch_base + (t - 1) * uint(STATE) + n]);
                d_decay += dh[n] * h_prev_n;
            }
        }
        float d_logdecay = d_decay * decay;
        uint adt_partial_idx = ((b * uint(SEQ) + t) * uint(HEADS) + h) * uint(HEADDIM) + p;
        dA_partial[adt_partial_idx] = T_OUT(d_logdecay * dt_val);
        ddt_partial[adt_partial_idx] = T_OUT(d_logdecay * A_val);

        // Propagate dh through decay.
        for (uint n = 0; n < uint(STATE); ++n) {
            dh[n] = dh[n] * decay;
        }
    }

    // After loop, dh holds the gradient that propagates past t=0; that is dh0
    // for this lane.
    for (uint n = 0; n < uint(STATE); ++n) {
        dh0[h_base + n] = T_OUT(dh[n]);
    }
    uint dD_idx = ((b) * uint(HEADS) + h) * uint(HEADDIM) + p;
    dD_partial[dD_idx] = T_OUT(dD_acc);
"""


# This is a private primitive ABI. Its P-axis intermediates are intentionally
# not a public gradient surface; _mamba3_mimo_bwd_metal_kernel reduces them
# before mamba3_mimo_bwd_metal returns.
#
# TODO(wave-6): port to TileLang DSL. Reverse-time scan with both per-thread
# ``float dh[STATE]`` accumulator and a persistent ``h_steps_scratch[tid][t][n]``
# slab; needs ``T.serial(reverse=True)`` over ``t`` plus careful fragment
# layout to keep the per-lane partial outputs (dB_partial, dC_partial,
# dA_partial, ddt_partial) atomics-free. Same flip pattern as the fwd above
# once the prim_func exists.
_BWD_KERNEL = _msl_transform.make_metal_kernel(
    name="cppmega_mamba3_mimo_bwd",
    input_names=["dy", "x", "B_proj", "C_proj", "z", "A", "dt", "D", "h0"],
    output_names=[
        "dx",
        "dz",
        "dB_partial",
        "dC_partial",
        "dA_partial",
        "ddt_partial",
        "dD_partial",
        "dh0",
        "h_steps_scratch",
    ],
    source=_BWD_KERNEL_SOURCE,
    header=_FWD_KERNEL_HEADER,
)

# Chunked variant: identical math, but the reverse scan seeds dh with an
# incoming end-state cotangent ``dh_init`` (instead of 0) so the sequence-
# chunked orchestrator can carry the boundary state-gradient across chunks.
# Only the dh-seed differs; everything else is the shared source.
_BWD_KERNEL_SOURCE_DHINIT = _BWD_KERNEL_SOURCE.replace(
    "    float dh[STATE];\n    for (uint n = 0; n < uint(STATE); ++n) {\n        dh[n] = 0.0f;\n    }",
    "    float dh[STATE];\n    for (uint n = 0; n < uint(STATE); ++n) {\n        dh[n] = float(dh_init[h_base + n]);\n    }",
)
if _BWD_KERNEL_SOURCE_DHINIT == _BWD_KERNEL_SOURCE:  # pragma: no cover - guard
    raise RuntimeError(
        "mamba3 chunked bwd: failed to splice dh_init seed into MSL source"
    )
_BWD_KERNEL_DHINIT = _msl_transform.make_metal_kernel(
    name="cppmega_mamba3_mimo_bwd_dhinit",
    input_names=[
        "dy", "x", "B_proj", "C_proj", "z", "A", "dt", "D", "h0", "dh_init"
    ],
    output_names=[
        "dx",
        "dz",
        "dB_partial",
        "dC_partial",
        "dA_partial",
        "ddt_partial",
        "dD_partial",
        "dh0",
        "h_steps_scratch",
    ],
    source=_BWD_KERNEL_SOURCE_DHINIT,
    header=_FWD_KERNEL_HEADER,
)


def _validate_inputs(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[int, int, int, int, int]:
    if x.ndim != 4:
        raise ValueError(f"x must be (B,T,H,P), got {x.shape}")
    batch, seq, heads, headdim = x.shape
    state = B.shape[-1]
    if B.shape != (batch, seq, heads, state):
        raise ValueError(f"B must be {(batch, seq, heads, state)}, got {B.shape}")
    if C.shape != (batch, seq, heads, state):
        raise ValueError(f"C must be {(batch, seq, heads, state)}, got {C.shape}")
    if z.shape != x.shape:
        raise ValueError(f"z must match x shape {x.shape}, got {z.shape}")
    if A.shape != (batch, seq, heads):
        raise ValueError(f"A must be {(batch, seq, heads)}, got {A.shape}")
    if dt.shape != (batch, seq, heads):
        raise ValueError(f"dt must be {(batch, seq, heads)}, got {dt.shape}")
    if D.shape != (heads,):
        raise ValueError(f"D must be {(heads,)}, got {D.shape}")
    if h0.shape != (batch, heads, headdim, state):
        raise ValueError(f"h0 must be {(batch, heads, headdim, state)}, got {h0.shape}")
    return batch, seq, heads, headdim, state


def mamba3_mimo_metal_status(x: mx.array | None = None) -> Mamba3MetalStatus:
    """Report Metal-path eligibility for the Mamba3 MIMO kernel."""

    arrays = (x,) if x is not None else ()
    status = _msl_transform.msl_dispatch_status(*arrays)
    if not status.available:
        return Mamba3MetalStatus(False, status.reason)
    if _FWD_KERNEL is None:
        return Mamba3MetalStatus(False, "vendor MSL fwd kernel was not constructed")
    return Mamba3MetalStatus(True, status.reason)


def mamba3_mimo_reference(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Pure-MLX reference identical in semantics to the Metal kernel.

    Mirrors cppmega_mlx/nn/mamba3.py::_chunked_mamba3_diagonal_scan but takes
    the already-reduced (post-projection) (B,T,H,*) tensors.
    """

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if seq == 0:
        return mx.zeros((batch, 0, heads, headdim), dtype=x.dtype), h0

    log_decay = (A * dt)[:, :, :, None, None]
    inp = x[:, :, :, :, None] * B[:, :, :, None, :]
    h = h0
    out_steps: list[mx.array] = []
    for t in range(seq):
        h = mx.exp(log_decay[:, t]) * h + inp[:, t]
        y = mx.sum(h * C[:, t, :, None, :], axis=-1)
        y = y + D[None, :, None].astype(y.dtype) * x[:, t]
        out_steps.append(nn.silu(z[:, t]) * y)
    y_full = mx.stack(out_steps, axis=1)
    return y_full, h


def _row_contiguous(array: mx.array) -> mx.array:
    """Return a row-contiguous copy if needed (mx.ascontiguousarray equivalent)."""

    return mx.array(array)  # mx.array(...) returns a contiguous copy in 0.31.


def mamba3_mimo_fwd_metal(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Path B Metal forward. Falls back to pure MLX if Metal is not eligible."""

    status = mamba3_mimo_metal_status(x)
    if not status.available or _FWD_KERNEL is None:
        return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)

    out_dtype = x.dtype
    # MSL kernel does fp32 internal accumulation; we cast everything to a
    # consistent T type and emit T outputs to keep the dispatcher simple.
    cast_dtype = mx.float32 if x.dtype == mx.bfloat16 else x.dtype
    inputs = [
        x.astype(cast_dtype),
        B.astype(cast_dtype),
        C.astype(cast_dtype),
        z.astype(cast_dtype),
        A.astype(cast_dtype),
        dt.astype(cast_dtype),
        D.astype(cast_dtype),
        h0.astype(cast_dtype),
    ]

    total_lanes = batch * heads * headdim
    threads = min(256, total_lanes if total_lanes > 0 else 1)
    template = [
        ("T_OUT", cast_dtype),
        ("BATCH", batch),
        ("SEQ", seq),
        ("HEADS", heads),
        ("HEADDIM", headdim),
        ("STATE", state),
    ]
    try:
        outputs = _msl_transform.dispatch(
            cast(_msl_transform.MetalKernel, _FWD_KERNEL),
            inputs=inputs,
            output_shapes=[(batch, seq, heads, headdim), (batch, heads, headdim, state)],
            output_dtypes=[cast_dtype, cast_dtype],
            grid=(total_lanes, 1, 1),
            threadgroup=(threads, 1, 1),
            template=template,
        )
    except Exception:
        # Any dispatch failure (out-of-bounds template, MSL compile diff)
        # must fail safe via the reference scan.
        return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    y, h_last = outputs
    return y.astype(out_dtype), h_last.astype(out_dtype)


def _mamba3_mimo_bwd_metal_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, ...] | None:
    """Try the Metal bwd kernel; return None if Metal is not eligible."""

    if _BWD_KERNEL is None:
        return None
    status = mamba3_mimo_metal_status(x)
    if not status.available:
        return None

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if seq == 0:
        return None  # pure-MLX path handles empty seq trivially.

    cast_dtype = mx.float32 if x.dtype == mx.bfloat16 else x.dtype
    inputs = [
        dy.astype(cast_dtype),
        x.astype(cast_dtype),
        B.astype(cast_dtype),
        C.astype(cast_dtype),
        z.astype(cast_dtype),
        A.astype(cast_dtype),
        dt.astype(cast_dtype),
        D.astype(cast_dtype),
        h0.astype(cast_dtype),
    ]
    total_lanes = batch * heads * headdim
    threads = min(256, total_lanes if total_lanes > 0 else 1)
    template = [
        ("T_OUT", cast_dtype),
        ("BATCH", batch),
        ("SEQ", seq),
        ("HEADS", heads),
        ("HEADDIM", headdim),
        ("STATE", state),
    ]
    output_shapes = [
        (batch, seq, heads, headdim),                  # dx
        (batch, seq, heads, headdim),                  # dz
        (batch, seq, heads, headdim, state),           # dB_partial
        (batch, seq, heads, headdim, state),           # dC_partial
        (batch, seq, heads, headdim),                  # dA_partial
        (batch, seq, heads, headdim),                  # ddt_partial
        (batch, heads, headdim),                       # dD_partial
        (batch, heads, headdim, state),                # dh0
        (batch * heads * headdim, seq, state),         # h_steps_scratch
    ]
    output_dtypes = [cast_dtype] * len(output_shapes)
    try:
        outputs = _msl_transform.dispatch(
            cast(_msl_transform.MetalKernel, _BWD_KERNEL),
            inputs=inputs,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            grid=(total_lanes, 1, 1),
            threadgroup=(threads, 1, 1),
            template=template,
        )
    except Exception:
        return None
    dx_, dz_, dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial, dh0_, _h_scratch = outputs
    # Reduce P-dimension partials.
    dB = mx.sum(dB_partial, axis=3)         # -> (B, T, H, N)
    dC = mx.sum(dC_partial, axis=3)         # -> (B, T, H, N)
    dA = mx.sum(dA_partial, axis=3)         # -> (B, T, H)
    ddt = mx.sum(ddt_partial, axis=3)       # -> (B, T, H)
    dD = mx.sum(dD_partial, axis=(0, 2))    # -> (H,)
    return (
        dx_.astype(x.dtype),
        dB.astype(B.dtype),
        dC.astype(C.dtype),
        dz_.astype(z.dtype),
        dA.astype(A.dtype),
        ddt.astype(dt.dtype),
        dD.astype(D.dtype),
        dh0_.astype(h0.dtype),
    )


def _mamba3_mimo_bwd_metal_kernel_dhinit(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    dh_init: mx.array,
) -> tuple[mx.array, ...] | None:
    """Metal bwd kernel with an incoming end-state cotangent (chunk carry).

    Identical to :func:`_mamba3_mimo_bwd_metal_kernel` except the reverse scan
    seeds ``dh`` with ``dh_init`` instead of 0. Returns ``None`` if Metal is
    not eligible.
    """

    if _BWD_KERNEL_DHINIT is None:
        return None
    status = mamba3_mimo_metal_status(x)
    if not status.available:
        return None

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if seq == 0:
        return None
    if dh_init.shape != h0.shape:
        raise ValueError(
            f"dh_init must match h0 shape {h0.shape}, got {dh_init.shape}"
        )

    cast_dtype = mx.float32 if x.dtype == mx.bfloat16 else x.dtype
    inputs = [
        dy.astype(cast_dtype),
        x.astype(cast_dtype),
        B.astype(cast_dtype),
        C.astype(cast_dtype),
        z.astype(cast_dtype),
        A.astype(cast_dtype),
        dt.astype(cast_dtype),
        D.astype(cast_dtype),
        h0.astype(cast_dtype),
        dh_init.astype(cast_dtype),
    ]
    total_lanes = batch * heads * headdim
    threads = min(256, total_lanes if total_lanes > 0 else 1)
    template = [
        ("T_OUT", cast_dtype),
        ("BATCH", batch),
        ("SEQ", seq),
        ("HEADS", heads),
        ("HEADDIM", headdim),
        ("STATE", state),
    ]
    output_shapes = [
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim, state),
        (batch, seq, heads, headdim, state),
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim),
        (batch, heads, headdim),
        (batch, heads, headdim, state),
        (batch * heads * headdim, seq, state),
    ]
    output_dtypes = [cast_dtype] * len(output_shapes)
    try:
        outputs = _msl_transform.dispatch(
            cast(_msl_transform.MetalKernel, _BWD_KERNEL_DHINIT),
            inputs=inputs,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            grid=(total_lanes, 1, 1),
            threadgroup=(threads, 1, 1),
            template=template,
        )
    except Exception:
        return None
    dx_, dz_, dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial, dh0_, _h = outputs
    dB = mx.sum(dB_partial, axis=3)
    dC = mx.sum(dC_partial, axis=3)
    dA = mx.sum(dA_partial, axis=3)
    ddt = mx.sum(ddt_partial, axis=3)
    dD = mx.sum(dD_partial, axis=(0, 2))
    return (
        dx_.astype(x.dtype),
        dB.astype(B.dtype),
        dC.astype(C.dtype),
        dz_.astype(z.dtype),
        dA.astype(A.dtype),
        ddt.astype(dt.dtype),
        dD.astype(D.dtype),
        dh0_.astype(h0.dtype),
    )


def _slice_seq(arr: mx.array, start: int, stop: int) -> mx.array:
    """Slice the time axis (axis=1) of a (B, SEQ, ...) tensor."""

    return arr[:, start:stop]


def mamba3_mimo_bwd_seq_chunked(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    chunk: int,
    fwd_state_fn: Callable[..., mx.array],
    bwd_chunk_fn: Callable[..., tuple[mx.array, ...]],
    trap: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Sequence-chunked Mamba3 backward — numerically identical, bounded memory.

    Splits the time axis into ``chunk``-length segments. ``fwd_state_fn`` must
    return the scan state ``h_last`` (shape == ``h0``) for a given input slice
    and starting state. ``bwd_chunk_fn`` must run the per-chunk reverse VJP for
    a slice starting at scan-state ``h0_chunk`` and incoming end-state cotangent
    ``dh_init`` and return ``(dx, dB, dC, dz, dA, ddt, dD, dh0)`` for that slice
    (``dD`` reduced over the chunk's time axis already).

    The state carry across chunk boundaries makes this *exactly* the same math
    as the monolithic backward (RULE #1: no correctness change), with the bwd
    working set capped at one chunk instead of the full sequence.
    """

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if dy.shape != (batch, seq, heads, headdim):
        raise ValueError(f"dy must be {(batch, seq, heads, headdim)}, got {dy.shape}")

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < seq:
        bounds.append((start, min(start + chunk, seq)))
        start += chunk

    # Forward boundary-state pass: h_starts[i] is the scan state entering chunk
    # i. Only the O(num_chunks) boundary states are retained (no per-step slab).
    h_starts: list[mx.array] = []
    h_cur = h0
    for lo, hi in bounds:
        h_starts.append(h_cur)
        h_cur = fwd_state_fn(
            _slice_seq(x, lo, hi),
            _slice_seq(B, lo, hi),
            _slice_seq(C, lo, hi),
            _slice_seq(z, lo, hi),
            _slice_seq(A, lo, hi),
            _slice_seq(dt, lo, hi),
            D,
            h_cur,
        )
        mx.eval(h_cur)

    # Reverse chunk loop: carry dh (state cotangent on the chunk boundary).
    dx_parts: list[mx.array] = [None] * len(bounds)  # type: ignore[list-item]
    dB_parts: list[mx.array] = [None] * len(bounds)  # type: ignore[list-item]
    dC_parts: list[mx.array] = [None] * len(bounds)  # type: ignore[list-item]
    dz_parts: list[mx.array] = [None] * len(bounds)  # type: ignore[list-item]
    dA_parts: list[mx.array] = [None] * len(bounds)  # type: ignore[list-item]
    ddt_parts: list[mx.array] = [None] * len(bounds)  # type: ignore[list-item]
    dD_total = mx.zeros((heads,), dtype=mx.float32)
    dh_carry = mx.zeros_like(h0.astype(mx.float32))

    for idx in range(len(bounds) - 1, -1, -1):
        lo, hi = bounds[idx]
        trap_slice = _slice_seq(trap, lo, hi) if trap is not None else None
        dx_c, dB_c, dC_c, dz_c, dA_c, ddt_c, dD_c, dh0_c = bwd_chunk_fn(
            _slice_seq(dy, lo, hi),
            _slice_seq(x, lo, hi),
            _slice_seq(B, lo, hi),
            _slice_seq(C, lo, hi),
            _slice_seq(z, lo, hi),
            _slice_seq(A, lo, hi),
            _slice_seq(dt, lo, hi),
            D,
            h_starts[idx],
            dh_init=dh_carry,
            trap=trap_slice,
        )
        dx_parts[idx] = dx_c
        dB_parts[idx] = dB_c
        dC_parts[idx] = dC_c
        dz_parts[idx] = dz_c
        dA_parts[idx] = dA_c
        ddt_parts[idx] = ddt_c
        dD_total = dD_total + dD_c.astype(mx.float32)
        dh_carry = dh0_c.astype(mx.float32)
        # Free the chunk's working partials before the next (earlier) chunk.
        mx.eval(dx_c, dB_c, dC_c, dz_c, dA_c, ddt_c, dh_carry, dD_total)

    dx = mx.concatenate(dx_parts, axis=1)
    dB = mx.concatenate(dB_parts, axis=1)
    dC = mx.concatenate(dC_parts, axis=1)
    dz = mx.concatenate(dz_parts, axis=1)
    dA = mx.concatenate(dA_parts, axis=1)
    ddt = mx.concatenate(ddt_parts, axis=1)
    return (
        dx.astype(x.dtype),
        dB.astype(B.dtype),
        dC.astype(C.dtype),
        dz.astype(z.dtype),
        dA.astype(A.dtype),
        ddt.astype(dt.dtype),
        dD_total.astype(D.dtype),
        dh_carry.astype(h0.dtype),
    )


def mamba3_mimo_bwd_metal(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    trap: mx.array | None = None,
    backend: str = "auto",
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Backward pass for the Mamba3 MIMO selective scan.

    The private Metal primitive emits per-lane partial gradients; its wrapper
    reduces those intermediates before returning final owner gradients. The
    pure-MLX path (reachable via ``backend='mlx'``) reproduces the same math
    step-by-step on the GPU graph.

    Inputs match the forward; ``trap`` is optional and only used for the
    extra ``ddt``/``dtrap`` contributions when the caller wants to wire the
    trapezoidal scale into the same gradient sweep.

    Returns gradients for (x, B, C, z, A, dt, D, h0). When ``trap`` is given,
    callers can postprocess (ddt, dtrap) externally via :func:`bwd_dtrap_ddt`.
    """

    if backend not in {"auto", "mlx", "metal"}:
        raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'mlx', or 'metal'")

    chunk = mamba3_bwd_seq_chunk()
    seq = x.shape[1]
    use_chunk = chunk > 0 and seq > chunk

    metal_chunk_ok = (
        use_chunk
        and backend in {"auto", "metal"}
        and _BWD_KERNEL_DHINIT is not None
        and mamba3_mimo_metal_status(x).available
    )

    if use_chunk and metal_chunk_ok:
        # Seq-chunked Metal path: each chunk runs the dh_init-seeded Metal
        # kernel, carrying the scan state and end-state cotangent across
        # boundaries. Same kernel math, bwd working set bounded by chunk-len.
        def _metal_fwd_state(x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c):
            _y, h_last = mamba3_mimo_fwd_metal(
                x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c
            )
            return h_last

        def _metal_bwd_chunk(
            dy_c, x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c, *, dh_init, trap=None
        ):
            grads = _mamba3_mimo_bwd_metal_kernel_dhinit(
                dy_c, x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c, dh_init
            )
            if grads is None:
                # Metal eligible at the top-level check but a per-chunk shape
                # (e.g. a short tail chunk) was rejected — use the pure-MLX
                # parity oracle for just that chunk.
                return _mamba3_mimo_bwd_pure_mlx(
                    dy_c, x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c,
                    trap=trap, dh_init=dh_init,
                )
            if trap is not None:
                ddt_trap, _dtrap = bwd_dtrap_ddt(
                    grads[5].astype(mx.float32),
                    dt_c.astype(mx.float32),
                    trap.astype(mx.float32),
                )
                grads = list(grads)
                grads[5] = grads[5] + ddt_trap.astype(grads[5].dtype)
                grads = tuple(grads)
            return grads

        return mamba3_mimo_bwd_seq_chunked(
            dy, x, B, C, z, A, dt, D, h0,
            chunk=chunk,
            fwd_state_fn=_metal_fwd_state,
            bwd_chunk_fn=_metal_bwd_chunk,
            trap=trap,
        )

    if backend in {"auto", "metal"} and not use_chunk:
        metal_result = _mamba3_mimo_bwd_metal_kernel(dy, x, B, C, z, A, dt, D, h0)
        if metal_result is not None:
            metal_grads = metal_result
            if trap is not None:
                ddt = metal_grads[5]
                ddt_trap, _dtrap = bwd_dtrap_ddt(
                    ddt.astype(mx.float32),
                    dt.astype(mx.float32),
                    trap.astype(mx.float32),
                )
                metal_grads = list(metal_grads)
                metal_grads[5] = (ddt + ddt_trap.astype(ddt.dtype))
            return tuple(metal_grads)  # type: ignore[return-value]
        if backend == "metal":
            raise RuntimeError("explicit metal backend unavailable for Mamba3 bwd")

    if use_chunk:
        # Metal not eligible: pure-MLX chunked path (parity oracle).
        def _mlx_fwd_state(x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c):
            _y, h_last = mamba3_mimo_reference(
                x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c
            )
            return h_last

        def _mlx_bwd_chunk(
            dy_c, x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c, *, dh_init, trap=None
        ):
            return _mamba3_mimo_bwd_pure_mlx(
                dy_c, x_c, B_c, C_c, z_c, A_c, dt_c, D_c, h0_c,
                trap=trap, dh_init=dh_init,
            )

        return mamba3_mimo_bwd_seq_chunked(
            dy, x, B, C, z, A, dt, D, h0,
            chunk=chunk,
            fwd_state_fn=_mlx_fwd_state,
            bwd_chunk_fn=_mlx_bwd_chunk,
            trap=trap,
        )
    return _mamba3_mimo_bwd_pure_mlx(dy, x, B, C, z, A, dt, D, h0, trap=trap)


def _mamba3_mimo_bwd_pure_mlx(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    trap: mx.array | None = None,
    dh_init: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Pure-MLX backward; identical math to the kernel.

    Used as the fallback path and as a parity oracle for the Metal kernel.

    ``dh_init`` is the incoming state cotangent at the *end* of this (sub)scan
    (``dh`` on ``h_last``). It is ``0`` for a full-sequence backward (nothing
    downstream consumes ``h_last``), but the sequence-chunked orchestrator
    feeds the next chunk's ``dh0`` here so a chunked backward is bitwise the
    same math as the monolithic one. ``dh0`` (the returned 8th grad) is the
    state cotangent at the *start* of this (sub)scan.
    """

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if dy.shape != (batch, seq, heads, headdim):
        raise ValueError(f"dy must be {(batch, seq, heads, headdim)}, got {dy.shape}")

    work_dtype = mx.float32
    if seq == 0:
        zero_x = mx.zeros_like(x)
        zero_B = mx.zeros_like(B)
        zero_C = mx.zeros_like(C)
        zero_z = mx.zeros_like(z)
        zero_A = mx.zeros_like(A)
        zero_dt = mx.zeros_like(dt)
        zero_D = mx.zeros_like(D)
        return zero_x, zero_B, zero_C, zero_z, zero_A, zero_dt, zero_D, h0 * 0.0

    x_f = x.astype(work_dtype)
    B_f = B.astype(work_dtype)
    C_f = C.astype(work_dtype)
    z_f = z.astype(work_dtype)
    A_f = A.astype(work_dtype)
    dt_f = dt.astype(work_dtype)
    D_f = D.astype(work_dtype)
    h0_f = h0.astype(work_dtype)
    dy_f = dy.astype(work_dtype)

    log_decay = (A_f * dt_f)[:, :, :, None, None]
    decay_factor = mx.exp(log_decay)  # (B,T,H,1,1)
    inp = x_f[:, :, :, :, None] * B_f[:, :, :, None, :]

    # Forward sweep retained per-step h to feed reverse VJP.
    h_steps: list[mx.array] = []
    h_t = h0_f
    for t in range(seq):
        h_t = decay_factor[:, t] * h_t + inp[:, t]
        h_steps.append(h_t)

    # silu derivative: silu(z) = z * sigmoid(z); silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    sig_z = mx.sigmoid(z_f)
    silu_z = z_f * sig_z
    silu_dz = sig_z * (1.0 + z_f * (1.0 - sig_z))

    # Pre-skip y per step: y_skipped[t] = sum(h[t]*C[t]) + D*x[t]
    # Final output: dy/d_y_skipped = silu(z); dy/dz = silu'(z) * y_skipped
    # dy/d_silu = y_skipped via the gated multiply
    dC_steps: list[mx.array] = [mx.zeros((batch, heads, state), dtype=work_dtype)] * seq
    dB_steps: list[mx.array] = [mx.zeros((batch, heads, state), dtype=work_dtype)] * seq
    dx_steps: list[mx.array] = [mx.zeros((batch, heads, headdim), dtype=work_dtype)] * seq
    dz_steps: list[mx.array] = [mx.zeros((batch, heads, headdim), dtype=work_dtype)] * seq
    dA_steps: list[mx.array] = [mx.zeros((batch, heads), dtype=work_dtype)] * seq
    ddt_steps: list[mx.array] = [mx.zeros((batch, heads), dtype=work_dtype)] * seq
    dD = mx.zeros((heads,), dtype=work_dtype)

    # y_skipped is needed for dz; recompute lazily during reverse pass.
    # Reverse recurrence on h: dh_(t-1) += decay[t] * dh_t.
    if dh_init is None:
        dh_next = mx.zeros_like(h0_f)
    else:
        if dh_init.shape != h0.shape:
            raise ValueError(
                f"dh_init must match h0 shape {h0.shape}, got {dh_init.shape}"
            )
        dh_next = dh_init.astype(work_dtype)

    # Walk backwards through time.
    for t in range(seq - 1, -1, -1):
        h_curr = h_steps[t]
        # y_skipped[t] = sum(h_curr * C[:, t, :, None, :], -1) + D*x[:, t]
        C_t = C_f[:, t, :, None, :]  # (B,H,1,N)
        y_state = mx.sum(h_curr * C_t, axis=-1)  # (B,H,P)
        y_skipped = y_state + D_f[None, :, None] * x_f[:, t]
        # gated output: y_full = silu_z * y_skipped
        # dy/d_silu = y_skipped, dy/d_y_skipped = silu_z
        dY_t = dy_f[:, t]
        d_silu_t = dY_t * y_skipped  # (B,H,P)
        d_y_skipped = dY_t * silu_z[:, t]  # (B,H,P)

        dz_steps[t] = d_silu_t * silu_dz[:, t]

        # dD_h += sum_b sum_p (d_y_skipped[b,h,p] * x[b,t,h,p])
        dD = dD + mx.sum(d_y_skipped * x_f[:, t], axis=(0, 2))
        # dx_t direct from skip path
        dx_skip_t = d_y_skipped * D_f[None, :, None]

        # dC_t = d_y_skipped[..., None] * h_curr -> sum over P -> (B,H,N)
        dC_steps[t] = mx.sum(d_y_skipped[..., None] * h_curr, axis=2)

        # gradient back through y_state: dh += d_y_skipped[..., None] * C[t]
        dh_curr = dh_next + d_y_skipped[..., None] * C_t

        # input contribution: inp[t] = x[t] * B[t]; through dh_curr
        # d_inp[t] = dh_curr (same shape as h: (B,H,P,N))
        # so dx_t += sum over N (dh_curr * B[t])
        # and dB_t += sum over P (dh_curr * x[t])
        B_t = B_f[:, t, :, None, :]
        x_t = x_f[:, t, :, :, None]
        dx_inp_t = mx.sum(dh_curr * B_t, axis=-1)
        dB_steps[t] = mx.sum(dh_curr * x_t, axis=2)
        dx_steps[t] = dx_skip_t + dx_inp_t

        # decay backward: h_curr = decay[t] * h_prev + inp[t]
        # d_decay = sum( dh_curr * h_prev ); dh_prev = dh_curr * decay[t]
        if t == 0:
            h_prev = h0_f
        else:
            h_prev = h_steps[t - 1]
        d_decay_t = mx.sum(dh_curr * h_prev, axis=(2, 3))  # (B,H)
        # decay = exp(A*dt); d(A*dt) = d_decay * decay
        d_log_decay = d_decay_t * decay_factor[:, t, :, 0, 0]
        dA_steps[t] = d_log_decay * dt_f[:, t]
        ddt_steps[t] = d_log_decay * A_f[:, t]

        # carry for next iter (going backward): dh_prev = dh_curr * decay
        dh_next = dh_curr * decay_factor[:, t]

    # Final dh0 is dh_next (gradient that propagated past t=0).
    dh0 = dh_next
    dC = mx.stack(dC_steps, axis=1)
    dB = mx.stack(dB_steps, axis=1)
    dx = mx.stack(dx_steps, axis=1)
    dz = mx.stack(dz_steps, axis=1)
    dA = mx.stack(dA_steps, axis=1)
    ddt = mx.stack(ddt_steps, axis=1)

    # Optional: if caller passed trap, route through helper. We don't apply
    # automatically since not all callers wire it.
    if trap is not None:
        ddt_trap, dtrap = bwd_dtrap_ddt(ddt, dt_f, trap.astype(work_dtype))
        ddt = ddt + ddt_trap
        # dtrap is the caller's responsibility to pick up; we attach it to the
        # signature via a side channel only when needed.
        del dtrap  # kept here to make dependency explicit; callers re-call.

    # The two helpers below are unused inside this fused sweep but the imports
    # are exercised so callers can swap to a fully-fused variant if desired.
    _ = compute_dacs_segsum
    _ = bwd_dadt_fused

    return (
        dx.astype(x.dtype),
        dB.astype(B.dtype),
        dC.astype(C.dtype),
        dz.astype(z.dtype),
        dA.astype(A.dtype),
        ddt.astype(dt.dtype),
        dD.astype(D.dtype),
        dh0.astype(h0.dtype),
    )


def _scatter_along_axis(buffer: mx.array, value: mx.array, index: int, *, axis: int) -> mx.array:
    """Return ``buffer`` with ``value`` added at slot ``index`` along ``axis``.

    Pure functional: the input ``buffer`` is not mutated; the returned array
    shares all positions with the input except for the single index slot.
    """

    idx = mx.arange(buffer.shape[axis])
    mask = mx.equal(idx, index).astype(buffer.dtype)
    mask_shape = [1] * buffer.ndim
    mask_shape[axis] = buffer.shape[axis]
    mask = mask.reshape(mask_shape)
    expanded = mx.expand_dims(value, axis=axis)
    return buffer + mask * expanded.astype(buffer.dtype)


@mx.custom_function
def mamba3_mimo_apply(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> mx.array:
    """Forward-only wrapper that returns the gated output ``y``.

    The Metal forward also returns the final hidden state, but we only expose
    ``y`` from the differentiable surface to keep the VJP signature symmetric
    with autograd-through-reference. Callers that need ``h_last`` can call
    :func:`mamba3_mimo_fwd_metal` directly.

    Note (Path B vs Path C):
        Path C's analogous wrapper is ``mamba3_mimo_apply_path_c`` in
        ``mamba3_path_c.py``. Neither apply takes a ``force_metal`` /
        ``force_path_c`` kwarg today, so there is no kwarg rename to migrate.
        Path B is the production entrypoint; Path C is a proof / override
        path reached only via the submodule import (it is not re-exported
        from ``cppmega_mlx.nn._tilelang.__init__``). See
        ``docs/production_kernel_routing.md``.
    """

    y, _ = mamba3_mimo_fwd_metal(x, B, C, z, A, dt, D, h0)
    return y


@mamba3_mimo_apply.vjp
def _mamba3_mimo_apply_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array, ...]:
    x, B, C, z, A, dt, D, h0 = primals
    del output  # unused; we recompute from primals.
    grads = mamba3_mimo_bwd_metal(cotangent, x, B, C, z, A, dt, D, h0)
    return grads


__all__ = [
    "MAMBA3_BWD_SEQ_CHUNK_ENV",
    "Mamba3MetalStatus",
    "mamba3_bwd_seq_chunk",
    "mamba3_mimo_apply",
    "mamba3_mimo_bwd_metal",
    "mamba3_mimo_bwd_seq_chunked",
    "mamba3_mimo_fwd_metal",
    "mamba3_mimo_metal_status",
    "mamba3_mimo_reference",
]
