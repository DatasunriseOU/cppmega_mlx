"""Path-C mamba3 chunked-SSD FORWARD precompute (F0) + inter-chunk recurrence (F1).

Stage 2 of ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2/§7. These are the two
NEW single-entry grid kernels that precede the already-landed, Metal-validated F2
scan+combine core (``mamba3_chunked_scan_core.chunk_scan_fwd_metal_prim``):

  * **F0  ``mamba3_chunk_precompute``**  — grid-parallel, NO scan dependency. From
    the per-position SSD tensors (``x, B, C, A, dt``) it forms the F2 handoff
    inputs ``cb = C @ B^T`` (per chunk), ``dA_cumsum = cumsum(A*dt)`` (per chunk),
    and the per-chunk ``summary_states`` (the within-chunk decayed input states,
    h0-independent). This is exactly the SSD precompute the Stage-1 isolation
    harness did eagerly in ``_eager_precompute`` — now as a Metal kernel.

  * **F1  ``mamba3_inter_chunk_recur``** — the ONLY O(S/C) sequential stage. From
    ``summary_states`` + ``h0`` + the chunk-boundary decay (segsum over the
    per-chunk ``dA_cumsum`` tail) it produces ``prev_states`` (the inter-chunk
    entry state per chunk) and ``final_state``. ``prev_states``/``summary_states``
    are fp32 (precision; design §3.3/§6.6).

Numerical contract (the ONE clear path, RULE #1): the (F0 -> F1 -> F2) chain
reproduces the serial Path-C forward to ``max|abs| < 1e-5`` fp32 / ``< 5e-4`` fp16.
The handoff buffer shapes match ``chunk_scan_fwd_metal_prim`` exactly
(``mamba3_chunked_scan_core`` §"Inputs"):

  cb              : (batch, nchunks, ngroups, chunk, chunk)
  dA_cumsum       : (batch, nheads, nchunks, chunk)
  summary_states  : (batch, nchunks, nheads, headdim, dstate)   fp32
  prev_states     : (batch, nchunks, nheads, headdim, dstate)   fp32
  final_state     : (batch, nheads, headdim, dstate)            fp32

On any shape/compile failure the builders RAISE with where+what; there is NO
serial fallback (the chunked-vs-serial choice is an explicit gate elsewhere).
"""

# CRITICAL: no ``from __future__ import annotations`` — see the scan-core module
# docstring (TileLang eager frontend reads ``T.Tensor`` annotations as objects).

import math
from typing import Any

__all__ = [
    "MAMBA3_CHUNK_PRECOMPUTE_OP_NAME",
    "MAMBA3_INTER_CHUNK_RECUR_OP_NAME",
    "chunk_precompute_fwd_metal_prim",
    "inter_chunk_recur_fwd_metal_prim",
    "chunk_precompute_fwd_grid",
    "inter_chunk_recur_fwd_grid",
    "build_chunk_precompute_metal",
    "build_inter_chunk_recur_metal",
]

# Stable op-node names for the Path-C F0 / F1 forward segments (design §2/§3.2).
MAMBA3_CHUNK_PRECOMPUTE_OP_NAME = "mamba3_chunk_precompute"
MAMBA3_INTER_CHUNK_RECUR_OP_NAME = "mamba3_inter_chunk_recur"

# 1/ln(2): ``exp2(x*p) == exp(x)`` (matches the F2 scan-core convention).
_LOG2E = 1.4426950408889634


# --------------------------------------------------------------------------- #
# F0 — precompute (grid-parallel, NO scan dependency).                          #
# --------------------------------------------------------------------------- #


def chunk_precompute_fwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[int, tuple[int, int, int]]:
    """Return ``(total_threadgroups, (gx, gy, gz))`` for the F0 grid.

    Grid is ``(batch*nchunks, nheads, 1)``: one threadgroup per (batch, chunk,
    head). Each threadgroup forms this head's ``dA_cumsum`` row, its slice of
    ``cb`` (group-shared), and its ``summary_states`` block. Many threadgroups
    (vs the serial 1): for S=4096,chunk=64,H=112 -> 64*112 = 7168.
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_precompute_fwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_precompute_fwd_grid: nheads ({nheads}) must be divisible by "
            f"ngroups ({ngroups})"
        )
    nchunks = seqlen // chunk_size
    gx = batch * nchunks
    gy = nheads
    return gx * gy, (gx, gy, 1)


def chunk_precompute_fwd_metal_prim(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    *,
    threads: int = 128,
) -> Any:
    """Build the F0 ``mamba3_chunk_precompute`` Metal ``@T.prim_func``.

    Inputs (per-position SSD tensors the precompute stage already stages):
      x   : (batch, seqlen, nheads, headdim)
      B   : (batch, seqlen, ngroups, dstate)
      C   : (batch, seqlen, ngroups, dstate)
      A   : (nheads,)                              A = -softplus(...) <= 0
      dt  : (batch, seqlen, nheads)
    Outputs (caller-owned F2 handoff buffers):
      cb              : (batch, nchunks, ngroups, chunk, chunk)   = C @ B^T
      dA_cumsum       : (batch, nheads, nchunks, chunk)           = cumsum(A*dt)
      summary_states  : (batch, nchunks, nheads, headdim, dstate) fp32
                        = sum_l exp(dA_cs[-1]-dA_cs[l]) * dt[l] * (x[l] outer B[l])

    Grid ``(batch*nchunks, nheads, 1)``; ``cb`` is written once per group
    (head-0 of each group) since it is head-independent within a group.
    """
    import tilelang
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_precompute_fwd_metal_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_precompute_fwd_metal_prim: nheads ({nheads}) must be "
            f"divisible by ngroups ({ngroups})"
        )

    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    p = _LOG2E

    @T.prim_func
    def main(
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        A: T.Tensor((nheads), dtype),  # type: ignore
        dt: T.Tensor((batch, seqlen, nheads), dtype),  # type: ignore
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        summary_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size

            a_row = T.alloc_shared((L,), accum_dtype)
            dacs = T.alloc_shared((L,), accum_dtype)

            # --- dA_cumsum: a[l] = A[h]*dt[l]; inclusive cumsum over l. ---
            for l in T.Parallel(L):
                a_row[l] = T.Cast(accum_dtype, A[head_idx]) * T.Cast(
                    accum_dtype, dt[batch_idx, base + l, head_idx]
                )
            T.sync_threads()
            # serial inclusive scan (single lane) — L is small (=chunk_size).
            if T.get_thread_binding(0) == 0:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for l in T.serial(L):
                    acc[0] = acc[0] + a_row[l]
                    dacs[l] = acc[0]
            T.sync_threads()
            for l in T.Parallel(L):
                dA_cumsum[batch_idx, head_idx, chunk_idx, l] = T.Cast(dtype, dacs[l])

            # --- cb = C @ B^T per (group, chunk): write once per group. ---
            if head_idx % heads_per_group == 0:
                for ls in T.Parallel(L * L):
                    li = ls // L
                    si = ls % L
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)
                    for n in T.serial(dstate):
                        acc[0] = acc[0] + T.Cast(
                            accum_dtype, C[batch_idx, base + li, group_idx, n]
                        ) * T.Cast(accum_dtype, B[batch_idx, base + si, group_idx, n])
                    cb[batch_idx, chunk_idx, group_idx, li, si] = T.Cast(dtype, acc[0])
            T.sync_threads()

            # --- summary_states[p,n] = sum_l decay[l]*dt[l]*x[l,p]*B[l,n] ---
            # decay[l] = exp(dA_cs[L-1] - dA_cs[l]) = exp2((tail - dacs[l]) * p)
            for pn in T.Parallel(headdim * dstate):
                pp = pn // dstate
                nn = pn % dstate
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for l in T.serial(L):
                    decay = T.exp2((dacs[L - 1] - dacs[l]) * p)
                    dt_l = T.Cast(accum_dtype, dt[batch_idx, base + l, head_idx])
                    x_l = T.Cast(accum_dtype, x[batch_idx, base + l, head_idx, pp])
                    b_l = T.Cast(accum_dtype, B[batch_idx, base + l, group_idx, nn])
                    acc[0] = acc[0] + decay * dt_l * x_l * b_l
                summary_states[batch_idx, chunk_idx, head_idx, pp, nn] = acc[0]

    return main


def build_chunk_precompute_metal(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    **kwargs: Any,
) -> Any:
    """Compile the F0 precompute kernel to a Metal ``JITKernel``.

    Outputs (cb, dA_cumsum, summary_states) are the 6th/7th/8th params; pass
    PRE-ZEROED contiguous buffers positionally (no ``out_idx`` allocation), and
    ``torch.mps.synchronize()`` after. RULE #1: compile failures propagate.
    """
    import tilelang

    from cppmega_mlx.nn._tilelang import _msl_transform

    prim = chunk_precompute_fwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    return tilelang.compile(
        prim,
        out_idx=[5, 6, 7],
        target=_msl_transform._as_metal_target("metal -thread_warp_size=32"),
    )


# --------------------------------------------------------------------------- #
# F1 — inter-chunk recurrence (the ONLY O(S/C) sequential stage).               #
# --------------------------------------------------------------------------- #


def inter_chunk_recur_fwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[int, tuple[int, int, int]]:
    """Return ``(total_threadgroups, (gx, gy, gz))`` for the F1 grid.

    Grid is ``(batch, nheads, 1)``: one threadgroup per (batch, head) walks the
    O(nchunks) chunk axis sequentially carrying the inter-chunk state. Each
    (p,n) state cell is an independent thread; the chunk loop is the only serial
    axis. Many threadgroups (batch*nheads) — the sequential cost is O(S/C), not
    O(S). For S=4096,chunk=64,H=112 -> 112 threadgroups, each looping 64 chunks.
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"inter_chunk_recur_fwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    return batch * nheads, (batch, nheads, 1)


def inter_chunk_recur_fwd_metal_prim(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    *,
    threads: int = 128,
) -> Any:
    """Build the F1 ``mamba3_inter_chunk_recur`` Metal ``@T.prim_func``.

    Inputs:
      summary_states : (batch, nchunks, nheads, headdim, dstate) fp32  (from F0)
      dA_cumsum      : (batch, nheads, nchunks, chunk)                 (from F0)
      h0             : (batch, nheads, headdim, dstate)          fp32  initial state
    Outputs:
      prev_states    : (batch, nchunks, nheads, headdim, dstate) fp32  entry state/chunk
      final_state    : (batch, nheads, headdim, dstate)          fp32

    Recurrence (matches the proto / Stage-1 ``_eager_precompute``):
      state[0]      = h0
      prev[c]       = state[c]
      state[c+1]    = exp(chunk_tail[c]) * state[c] + summary_states[c]
    where ``chunk_tail[c] = dA_cumsum[..., c, L-1]`` is the per-chunk total decay.
    This is the segsum-over-chunk-tails recurrence written as a plain serial scan
    over the chunk axis (the ONLY sequential stage). Each thread owns a disjoint
    set of ``(p,n)`` state cells; the chunk axis is serial within each thread.
    """
    import tilelang
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"inter_chunk_recur_fwd_metal_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )

    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    L = chunk_size
    p = _LOG2E
    cells = headdim * dstate

    @T.prim_func
    def main(
        summary_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), T.float16),  # type: ignore
        h0: T.Tensor((batch, nheads, headdim, dstate), accum_dtype),  # type: ignore
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        final_state: T.Tensor((batch, nheads, headdim, dstate), accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch, nheads, threads=threads) as (bx, by):
            batch_idx = bx
            head_idx = by
            state = T.alloc_local((1,), accum_dtype)
            # Each thread strides over the (p,n) state cells; the chunk axis is
            # the serial carry. ``cells`` may exceed ``threads`` -> grid-stride.
            for cell0 in T.serial(0, cells, threads):
                lane = T.get_thread_binding(0)
                cell = cell0 + lane
                if cell < cells:
                    pp = cell // dstate
                    nn = cell % dstate
                    state[0] = h0[batch_idx, head_idx, pp, nn]
                    for c in T.serial(nchunks):
                        prev_states[batch_idx, c, head_idx, pp, nn] = state[0]
                        tail = T.Cast(
                            accum_dtype,
                            dA_cumsum[batch_idx, head_idx, c, L - 1],
                        )
                        decay = T.exp2(tail * p)
                        state[0] = (
                            decay * state[0]
                            + summary_states[batch_idx, c, head_idx, pp, nn]
                        )
                    final_state[batch_idx, head_idx, pp, nn] = state[0]

    return main


def build_inter_chunk_recur_metal(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    **kwargs: Any,
) -> Any:
    """Compile the F1 inter-chunk recurrence kernel to a Metal ``JITKernel``.

    Outputs (prev_states, final_state) are the 4th/5th params; pass PRE-ZEROED
    contiguous fp32 buffers positionally, ``torch.mps.synchronize()`` after.
    RULE #1: compile failures propagate (no serial fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang import _msl_transform

    prim = inter_chunk_recur_fwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    return tilelang.compile(
        prim,
        out_idx=[3, 4],
        target=_msl_transform._as_metal_target("metal -thread_warp_size=32"),
    )
