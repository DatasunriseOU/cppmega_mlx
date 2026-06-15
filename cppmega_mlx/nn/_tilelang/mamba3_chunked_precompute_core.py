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
import os
from typing import Any

__all__ = [
    "MAMBA3_CHUNK_PRECOMPUTE_OP_NAME",
    "MAMBA3_INTER_CHUNK_RECUR_OP_NAME",
    "chunk_precompute_fwd_metal_prim",
    "chunk_precompute_fwd_cuda_prim",
    "chunk_precompute_fwd_metal_gemm_prim",
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

            # --- summary_states[p,n] = sum_l decay[l]*x[l,p]*B[l,n] ---
            # decay[l] = exp(dA_cs[L-1] - dA_cs[l]) = exp2((tail - dacs[l]) * p)
            # PRODUCTION model: input term = x*B (NO dt). dt enters ONLY via decay
            # (=exp(A*dt) through dacs). prod fwd h_t = decay*h + x*B (path_c :1001).
            for pn in T.Parallel(headdim * dstate):
                pp = pn // dstate
                nn = pn % dstate
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for l in T.serial(L):
                    decay = T.exp2((dacs[L - 1] - dacs[l]) * p)
                    x_l = T.Cast(accum_dtype, x[batch_idx, base + l, head_idx, pp])
                    b_l = T.Cast(accum_dtype, B[batch_idx, base + l, group_idx, nn])
                    acc[0] = acc[0] + decay * x_l * b_l
                summary_states[batch_idx, chunk_idx, head_idx, pp, nn] = acc[0]

    return main


def chunk_precompute_fwd_cuda_prim(
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
    """CUDA / sm_121 twin of :func:`chunk_precompute_fwd_metal_prim` (gb10).

    IDENTICAL F0 math; the two head-independent serial scalar loops of the Metal
    prim are re-expressed as TENSOR-CORE ``T.gemm`` (the §22/§23 measured lever —
    the Metal prim's ``cb`` per-``(li,si)`` dot over ``dstate`` and its
    ``summary_states`` ``P*N=4096``-cell serial-``l`` accumulate were the 24.57 ms
    F0 term). ``dA_cumsum`` STAYS a single-lane serial inclusive scan (it is a
    scan, NOT a GEMM) and is copied verbatim from the Metal prim.

    The two GEMMs, both per the proven sibling CUDA SSD precedent
    (:func:`mamba3_chunked_scan_core.chunk_scan_fwd_cuda_prim`): register-fragment
    fp32 C accumulators (:func:`T.alloc_fragment`), plain ``scope="shared"`` fp16
    operands (NO ``make_swizzled_layout``), and ``disable_tma=True`` on every
    global<->shared copy (sm_121 TMA tensormap mis-aligns at these 64-tile dims ->
    "Invalid TMA descriptor arguments" at RUN). The compile site
    (:func:`build_chunk_precompute_metal`) selects this prim when the resolved
    target is CUDA and passes the matching
    ``{tl.disable_tma_lower, tl.disable_warp_specialized}`` pass_configs.

      (1) cb[L,L] = C[L,N] @ B[L,N]^T  (per (chunk, group); written once per group
          via the ``head_idx % heads_per_group == 0`` guard, exactly as the Metal
          prim, since cb is head-independent within a group):
            T.gemm(C_shared[L,N], B_shared[L,N], cb_frag[L,L], transpose_B=True)
          -> cb_frag[li,si] = sum_n C[li,n]*B[si,n]   (Tri-Dao _bmm_chunk_fwd).

      (2) summary_states[P,N] = (decay*dt-weighted x)^T @ B  (per (chunk, head),
          the DOMINANT term). The per-row decay*dt is folded into the x OPERAND
          (exactly Tri-Dao _chunk_state_fwd folds ``scale`` into its operand; this
          keeps the fp32 output exact and B in native fp16 as the serial reads it):
            x_dec[l,p] = exp2((dacs[L-1]-dacs[l])*p) * dt[l] * x[l,p]
            T.gemm(x_dec_shared[L,P], B_shared[L,N], states_frag[P,N], transpose_A=True)
          -> states_frag[p,n] = sum_l x_dec[l,p]*B[l,n].

    DTYPE CONTRACT (unchanged, §23 F1/F2): operands fp16, fragment accumulate
    fp32, ``cb`` stored fp16, ``summary_states`` stored fp32 (= accum_dtype; F1
    reads it fp32 — NO fp16 downcast on the store, the §23 N=64 segfault class).

    DECAY/SIGN: ``A = -softplus(...) <= 0`` so ``a_row <= 0`` and
    ``dacs[L-1]-dacs[l] <= 0``; the decay uses ``exp2((dacs[L-1]-dacs[l])*p)`` with
    NO ``minimum(...,0.0)`` clamp (a no-op for this sign convention; the serial
    prim omits it, so adding it would diverge from the parity reference).

    SMEM at prod ``L=P=N=64`` (fp16 = 2B, fp32 = 4B): a_row[L]+dacs[L] fp32 0.5 KB
    + C_shared[L,N] 8 KB + B_shared[L,N] 8 KB + cb_store[L,L] fp16 8 KB +
    x_dec_shared[L,P] 8 KB + states_store[P,N] fp32 16 KB = 48.5 KB combined
    (cb_frag/states_frag live in REGISTERS), << the gb10 ~99 KB budget. See the
    compile-site assertion in :func:`build_chunk_precompute_metal`.

    RULE #1: this is the ONE CUDA path when selected; a tile/MMA-divisibility/
    compile failure RAISES with where+what — it NEVER falls back to a serial loop.
    """
    import tilelang  # noqa: F401  (parity with the sibling cuda prim import block)
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_precompute_fwd_cuda_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_precompute_fwd_cuda_prim: nheads ({nheads}) must be "
            f"divisible by ngroups ({ngroups})"
        )
    # m16n8k16 fp16 MMA divisibility for both GEMMs: cb is M=L,N=L,K=dstate;
    # states is M=headdim,N=dstate,K=chunk_size. RAISE (no silent pad, RULE #1).
    for axis_name, axis_val in (
        ("chunk_size(L)", chunk_size),
        ("dstate(N/K)", dstate),
        ("headdim(P)", headdim),
    ):
        if axis_val % 16 != 0:
            raise ValueError(
                f"chunk_precompute_fwd_cuda_prim: {axis_name}={axis_val} is not a "
                f"multiple of 16; the fp16 m16n8k16 tensor-core MMA requires "
                f"K%16==0 and M%16==0/N%8==0 for the cb/summary_states GEMMs. No "
                f"silent padding (RULE #1) — re-tile or pad the caller buffers."
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
            # GEMM operands/outputs: plain scope="shared", fp16; register-fragment
            # fp32 C accumulators (the CUDA T.gemm asserts C is a fragment).
            C_shared = T.alloc_shared((L, dstate), dtype, scope="shared")
            B_shared = T.alloc_shared((L, dstate), dtype, scope="shared")
            cb_frag = T.alloc_fragment((L, L), accum_dtype)
            cb_store = T.alloc_shared((L, L), dtype, scope="shared")
            x_dec_shared = T.alloc_shared((L, headdim), dtype, scope="shared")
            states_frag = T.alloc_fragment((headdim, dstate), accum_dtype)
            states_store = T.alloc_shared((headdim, dstate), accum_dtype, scope="shared")

            # --- dA_cumsum: a[l] = A[h]*dt[l]; inclusive cumsum over l. ---
            # (STAYS a scan — copied verbatim from the Metal prim; NOT a GEMM.)
            for l in T.Parallel(L):
                a_row[l] = T.Cast(accum_dtype, A[head_idx]) * T.Cast(
                    accum_dtype, dt[batch_idx, base + l, head_idx]
                )
            T.sync_threads()
            if T.get_thread_binding(0) == 0:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for l in T.serial(L):
                    acc[0] = acc[0] + a_row[l]
                    dacs[l] = acc[0]
            T.sync_threads()
            for l in T.Parallel(L):
                dA_cumsum[batch_idx, head_idx, chunk_idx, l] = T.Cast(dtype, dacs[l])

            # --- (1) cb = C @ B^T per (group, chunk): T.gemm, write once/group. ---
            if head_idx % heads_per_group == 0:
                T.copy(
                    C[batch_idx, base : base + L, group_idx, 0:dstate],
                    C_shared,
                    disable_tma=True,
                )
                T.copy(
                    B[batch_idx, base : base + L, group_idx, 0:dstate],
                    B_shared,
                    disable_tma=True,
                )
                T.clear(cb_frag)
                # cb_frag[li,si] = sum_n C[li,n]*B[si,n] = (C @ B^T)[li,si].
                T.gemm(C_shared, B_shared, cb_frag, transpose_B=True)
                T.copy(cb_frag, cb_store)
                T.copy(
                    cb_store,
                    cb[batch_idx, chunk_idx, group_idx, 0:L, 0:L],
                    disable_tma=True,
                )
            T.sync_threads()

            # --- (2) summary_states[p,n] = sum_l (decay[l]*x[l,p]) * B[l,n] ---
            # PRODUCTION model: input term = x*B (NO dt); dt enters ONLY via decay.
            # Fold per-row decay into the x operand (cf. Tri-Dao _chunk_state_fwd but
            # WITHOUT the dt weight); decay[l] = exp2((dacs[L-1]-dacs[ll])*p).
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                decay = T.exp2((dacs[L - 1] - dacs[ll]) * p)
                x_l = T.Cast(accum_dtype, x[batch_idx, base + ll, head_idx, pp])
                x_dec_shared[ll, pp] = T.Cast(dtype, decay * x_l)
            # Re-stage B for the group (B_shared was last written for the cb GEMM
            # only under the head-0 guard; every head needs it here).
            T.copy(
                B[batch_idx, base : base + L, group_idx, 0:dstate],
                B_shared,
                disable_tma=True,
            )
            T.clear(states_frag)
            # states_frag[p,n] = sum_l x_dec[l,p]*B[l,n]; transpose_A contracts L.
            T.gemm(x_dec_shared, B_shared, states_frag, transpose_A=True)
            # Output stays fp32 (= accum_dtype); F1 reads it fp32. NO downcast.
            T.copy(states_frag, states_store)
            T.copy(
                states_store,
                summary_states[batch_idx, chunk_idx, head_idx, 0:headdim, 0:dstate],
                disable_tma=True,
            )

    return main


def chunk_precompute_fwd_metal_gemm_prim(
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
    """Metal TENSOR-OP twin of :func:`chunk_precompute_fwd_metal_prim`.

    IDENTICAL F0 math; the two head-independent serial scalar loops of the Metal
    prim (``cb`` per-``(li,si)`` dot over ``dstate`` and the ``summary_states``
    ``P*N``-cell serial-``l`` accumulate — the 24.57 ms F0 term on CUDA) are
    re-expressed as ``T.gemm``, mirroring the proven CUDA twin
    :func:`chunk_precompute_fwd_cuda_prim` (24.57 ms -> 1.13 ms, 14.5x, sm_121)
    and the Metal SSD precedent
    :func:`mamba3_chunked_scan_core.chunk_scan_fwd_metal_prim`.

    THE ONE LOAD-BEARING METAL-VS-CUDA DELTA (the Metal GEMM-instruction
    SELECTOR, ``src/backend/metal/op/gemm.cc::SelectInst``): the C accumulator
    must be a register FRAGMENT (``T.alloc_fragment``, scope ``local.fragment``).
    That is what forces the STABLE 8x8 ``metal.simdgroup`` path on this M4 Max.
    A plain ``scope="shared"`` fp32 C with N>=32 and K%16==0 (here N=dstate=64,
    N=L=64, K=64) instead satisfies ``CanUseCooperativeTensor`` and routes to the
    EXPERIMENTAL M5-only ``metal.cooperative_tensor`` (MPP matmul2d) path, which
    fails to compile on M1-M4 (``use of undeclared identifier '__pct_op'`` — the
    cooperative-tensor op is unavailable pre-M5). ``SelectInst`` returns the
    simdgroup path unconditionally when ``op.c_.scope()=="local.fragment"``, so a
    fragment C is the portable choice. (The F2 precedent dodged the same trap a
    different way — its shared C had block_N=16<32, failing the N%32 divisibility
    so the cooperative path was rejected; here the logical N is 64 so a fragment C
    is required.) Everything else (operand staging, decay/dt fold into the x
    operand, ``transpose_B`` / ``transpose_A`` flags, ``disable_tma`` copies, the
    serial ``dA_cumsum`` scan kept verbatim, fp32 ``summary_states`` store, the
    fragment->shared->global copy chain) is the same as the CUDA twin.

    The two GEMMs (per the CUDA twin / Tri-Dao ``_bmm_chunk_fwd`` +
    ``_chunk_state_fwd``):
      (1) cb[L,L] = C[L,N] @ B[L,N]^T  (per (chunk, group); written once per group
          via the ``head_idx % heads_per_group == 0`` guard — cb is head-independent
          within a group):
            T.gemm(C_shared[L,N], B_shared[L,N], cb_acc[L,L], transpose_B=True)
          -> cb_acc[li,si] = sum_n C[li,n]*B[si,n].
      (2) summary_states[P,N] = (decay*dt-weighted x)^T @ B  (per (chunk, head), the
          DOMINANT term). decay*dt folded into the x OPERAND:
            x_dec[l,p] = exp2((dacs[L-1]-dacs[l])*p) * dt[l] * x[l,p]
            T.gemm(x_dec_shared[L,P], B_shared[L,N], states_acc[P,N], transpose_A=True)
          -> states_acc[p,n] = sum_l x_dec[l,p]*B[l,n].

    DTYPE CONTRACT (unchanged): operands fp16, accumulate fp32 (fragment C),
    ``cb`` stored fp16, ``summary_states`` stored fp32 (= accum_dtype; F1 reads it
    fp32 — NO fp16 downcast). ``A = -softplus(...) <= 0`` so the decay uses
    ``exp2((dacs[L-1]-dacs[l])*p)`` with NO ``minimum(...,0)`` clamp (the serial
    parity reference omits it).

    RULE #1: this is the ONE Metal path when the GEMM flag is set; a
    tile/divisibility/compile failure RAISES with where+what — it NEVER falls back
    to the serial loop / the byte-identical serial prim (that is the parity
    reference, selected by an explicit gate, not a silent fallback).
    """
    import tilelang  # noqa: F401
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_precompute_fwd_metal_gemm_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_precompute_fwd_metal_gemm_prim: nheads ({nheads}) must be "
            f"divisible by ngroups ({ngroups})"
        )
    # Metal 8x8 simdgroup tensor-op divisibility (research §gemm.cc PARTIAL-TILE
    # guard): the ICHECK requires M%8==0 / N%8==0 / K%8==0 for the cb (M=L,N=L,
    # K=dstate) and summary (M=headdim,N=dstate,K=chunk) GEMMs. RAISE (no silent
    # pad, RULE #1).
    for axis_name, axis_val in (
        ("chunk_size(L)", chunk_size),
        ("dstate(N/K)", dstate),
        ("headdim(P)", headdim),
    ):
        if axis_val % 8 != 0:
            raise ValueError(
                f"chunk_precompute_fwd_metal_gemm_prim: {axis_name}={axis_val} is "
                f"not a multiple of 8; the Metal 8x8 simdgroup matmul requires "
                f"M/N/K % 8 == 0 for the cb / summary_states GEMMs. No silent "
                f"padding (RULE #1) — re-tile or pad the caller buffers."
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

            # GEMM operands fp16 (plain scope="shared"); the C accumulators are
            # register FRAGMENTS (scope local.fragment) — this is what forces the
            # stable 8x8 metal.simdgroup path (see the docstring delta). The
            # fragment is copied whole-tile to a shared staging buffer before the
            # global store (the F0 CUDA-twin copy chain).
            # SMEM-MINIMAL tiling (Apple 32 KB threadgroup-memory limit). The fp16
            # GEMM A-operand tile ``opA_shared`` is REUSED across the two GEMM phases
            # (C[L,N] for the cb GEMM under the head-0 guard, then x_dec[L,P] for the
            # summary GEMM) — disjoint lifetimes separated by T.sync_threads(), so one
            # [L, max(dstate,headdim)] fp16 tile holds either. ``B_shared`` is the
            # shared B-operand for both GEMMs (re-staged per head for the summary
            # GEMM). One fp32 staging tile ``store_f32`` is REUSED for the cb store
            # (runs FIRST) and the summary_states store (runs LAST). fragment(fp32)->
            # shared(fp16) is an illegal simdgroup cast, so staging is fp32 and the
            # global store downcasts (cb->fp16, summary_states->fp32 — supported
            # two-step copies). fragment->fp32-shared then fp32-shared->global is the
            # supported downcast path; a fragment->fp16 cast is an illegal
            # simdgroup_float8x8->half4 conversion. A simdgroup fragment can only be
            # copied WHOLE (partial row/col slices fail to lower), so the fp32 staging
            # is a full [L,L] (cb) / [P,N] (summary) tile, 16 KB at L=P=N=64.
            #
            # SMEM BUDGET (Apple 32 KB threadgroup limit, hard): the operand tiles
            # (opA fp16 [L,max(N,P)] 8 KB + B_shared fp16 [L,N] 8 KB) and the fp32
            # staging (16 KB) exactly fill a 32 KB pool. The single-lane cumsum scratch
            # would be a SEPARATE 256 B shared tile that overflows the cap, so the scan
            # instead uses the FIRST ROW of the fp32 staging tile (store_f32[0, 0:L]):
            # the scan runs FIRST (before cb/summary touch store_f32) and the dA_cumsum
            # global write completes before cb overwrites it — disjoint lifetimes, no
            # extra buffer. The summary decay then reloads the fp16 dA_cumsum (the F2
            # convention), so no persistent fp32 dacs tile is needed.
            _sm = headdim if headdim > L else L
            _sn = dstate if dstate > L else L
            store_f32 = T.alloc_shared((_sm, _sn), accum_dtype, scope="shared")
            _opn = dstate if dstate > headdim else headdim
            opA_shared = T.alloc_shared((L, _opn), dtype, scope="shared")
            B_shared = T.alloc_shared((L, dstate), dtype, scope="shared")
            cb_frag = T.alloc_fragment((L, L), accum_dtype)
            states_frag = T.alloc_fragment((headdim, dstate), accum_dtype)

            # --- dA_cumsum: a[l] = A[h]*dt[l]; inclusive cumsum over l. ---
            # (STAYS a scan — same math as the serial Metal prim; NOT a GEMM.) The
            # scratch is store_f32's FIRST ROW (no separate tile; the scan precedes any
            # cb/summary use of store_f32). a[l]=A*dt -> in-place inclusive cumsum.
            for l in T.Parallel(L):
                store_f32[0, l] = T.Cast(accum_dtype, A[head_idx]) * T.Cast(
                    accum_dtype, dt[batch_idx, base + l, head_idx]
                )
            T.sync_threads()
            if T.get_thread_binding(0) == 0:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for l in T.serial(L):
                    acc[0] = acc[0] + store_f32[0, l]  # read a[l] (pre-scan value)
                    store_f32[0, l] = acc[0]           # overwrite in place w/ cumsum
            T.sync_threads()
            for l in T.Parallel(L):
                dA_cumsum[batch_idx, head_idx, chunk_idx, l] = T.Cast(
                    dtype, store_f32[0, l]
                )
            T.sync_threads()  # dA_cumsum global write done before cb overwrites store_f32

            # --- (1) cb = C @ B^T per (group, chunk): T.gemm, write once/group. ---
            if head_idx % heads_per_group == 0:
                T.copy(
                    C[batch_idx, base : base + L, group_idx, 0:dstate],
                    opA_shared[0:L, 0:dstate],  # C operand (reused tile)
                    disable_tma=True,
                )
                T.copy(
                    B[batch_idx, base : base + L, group_idx, 0:dstate],
                    B_shared,
                    disable_tma=True,
                )
                T.clear(cb_frag)
                # cb_frag[li,si] = sum_n C[li,n]*B[si,n] = (C @ B^T)[li,si].
                T.gemm(opA_shared[0:L, 0:dstate], B_shared, cb_frag, transpose_B=True)
                T.copy(cb_frag, store_f32[0:L, 0:L])  # frag fp32 -> fp32 shared
                T.copy(
                    store_f32[0:L, 0:L],
                    cb[batch_idx, chunk_idx, group_idx, 0:L, 0:L],
                    disable_tma=True,  # fp32 shared -> fp16 global (downcast)
                )
            T.sync_threads()

            # --- (2) summary_states[p,n] = sum_l (decay[l]*dt[l]*x[l,p]) * B[l,n] ---
            # Fold per-row decay*dt into the x operand;
            # decay[l] = exp(dA_cs[L-1]-dA_cs[l]) = exp2((dacs[L-1]-dacs[l])*p).
            # The decay reads dacs back from the GLOBAL dA_cumsum (fp16) just written
            # above, NOT a persistent fp32 dacs shared tile: keeping dacs live to here
            # forces it OUT of the threadgroup pool (it overlaps the summary GEMM
            # staging) and pushes the kernel +256 B over Apple's 32 KB limit.
            # Reloading the fp16 dA_cumsum and computing the decay in fp32 is EXACTLY
            # the F2 forward-scan convention (chunk_scan_fwd_metal_prim reads fp16
            # dA_cumsum -> fp32 exp2). dacs is then live ONLY during the early cumsum
            # scan and the pool reclaims its bytes (pool = 32768 == the limit, OK).
            tail_dacs = T.alloc_local((1,), accum_dtype)
            tail_dacs[0] = T.Cast(
                accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, L - 1]
            )
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                dacs_l = T.Cast(
                    accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, ll]
                )
                decay = T.exp2((tail_dacs[0] - dacs_l) * p)
                dt_l = T.Cast(accum_dtype, dt[batch_idx, base + ll, head_idx])
                x_l = T.Cast(accum_dtype, x[batch_idx, base + ll, head_idx, pp])
                opA_shared[ll, pp] = T.Cast(dtype, decay * dt_l * x_l)
            # Re-stage B for the group (B_shared was last written for the cb GEMM
            # only under the head-0 guard; every head needs it here).
            T.copy(
                B[batch_idx, base : base + L, group_idx, 0:dstate],
                B_shared,
                disable_tma=True,
            )
            T.clear(states_frag)
            # states_frag[p,n] = sum_l x_dec[l,p]*B[l,n]; transpose_A contracts L.
            T.gemm(opA_shared[0:L, 0:headdim], B_shared, states_frag, transpose_A=True)
            T.sync_threads()  # operands (opA/B) dead after the GEMM -> let the
            #                    threadgroup-pool alias store_f32 onto their bytes.
            # Output stays fp32 (= accum_dtype); F1 reads it fp32. NO downcast.
            T.copy(states_frag, store_f32[0:headdim, 0:dstate])  # frag fp32 -> fp32 shared
            T.copy(
                store_f32[0:headdim, 0:dstate],
                summary_states[batch_idx, chunk_idx, head_idx, 0:headdim, 0:dstate],
                disable_tma=True,
            )

    return main


def build_chunk_precompute_metal(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    *,
    target: Any = None,
    **kwargs: Any,
) -> Any:
    """Compile the F0 precompute kernel to a ``JITKernel`` (Metal or CUDA).

    ``target`` selects the codegen backend (``None`` => Metal default, ``"cuda"``
    => CUDA / sm_121). Outputs (cb, dA_cumsum, summary_states) are the 6th/7th/8th
    params; pass PRE-ZEROED contiguous buffers positionally (no ``out_idx``
    allocation), and synchronize the device after. RULE #1: compile failures
    propagate.

    F0 is the SECOND chunked stage (after F2) whose prim BODY differs by backend:
    the Metal prim runs the two head-independent reductions (``cb`` dot,
    ``summary_states`` accumulate) as serial scalar loops; the CUDA prim
    (:func:`chunk_precompute_fwd_cuda_prim`) re-expresses them as tensor-core
    ``T.gemm`` (register-fragment C, plain shared operands, ``disable_tma`` copies)
    and compiles with the sm_121 ``{tl.disable_tma_lower, tl.disable_warp_specialized}``
    escape hatch — the SAME per-target codegen choice
    :func:`mamba3_chunked_scan_core.compile_chunk_scan_fwd_metal` makes for F2
    (RULE #1: explicit per-target branch, NOT a silent fallback to the serial
    loop). ``dA_cumsum`` stays a serial scan on both backends.
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )

    resolved_target = _resolve_chunked_compile_target(target)
    kind = str(getattr(getattr(resolved_target, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        # SMEM BUDGET GATE (gb10 ~99 KB per-threadgroup dynamic smem). The CUDA
        # prim's shared allocs at the requested shapes (fp16=2B, fp32=4B):
        #   a_row[L]+dacs[L] fp32 + C_shared[L,N]fp16 + B_shared[L,N]fp16 +
        #   cb_store[L,L]fp16 + x_dec_shared[L,P]fp16 + states_store[P,N]fp32.
        # RAISE (RULE #1) if it would exceed the budget — never silently re-tile.
        smem_bytes = (
            2 * chunk_size * 4                  # a_row + dacs (fp32)
            + chunk_size * dstate * 2           # C_shared (fp16)
            + chunk_size * dstate * 2           # B_shared (fp16)
            + chunk_size * chunk_size * 2       # cb_store (fp16)
            + chunk_size * headdim * 2          # x_dec_shared (fp16)
            + headdim * dstate * 4              # states_store (fp32)
        )
        _GB10_SMEM_BUDGET = 99 * 1024
        if smem_bytes >= _GB10_SMEM_BUDGET:
            raise ValueError(
                f"build_chunk_precompute_metal(cuda): per-threadgroup smem "
                f"{smem_bytes} B (L={chunk_size},P={headdim},N={dstate}) >= the "
                f"gb10 budget {_GB10_SMEM_BUDGET} B; re-tile the F0 cuda prim "
                f"(RULE #1: no silent over-budget launch)."
            )
        prim = chunk_precompute_fwd_cuda_prim(
            batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
        )
        # sm_121: disable TMA + warp-specialized lowering (the GEMM operand copies
        # otherwise lower to a TMA tensormap that mis-aligns at these 64-tile dims
        # -> "Invalid TMA descriptor arguments" at RUN). Same escape hatch as F2.
        pass_configs = {
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
        }
        return tilelang.compile(
            prim,
            out_idx=[5, 6, 7],
            target=resolved_target,
            pass_configs=pass_configs,
        )

    # METAL PRIM SELECTION (RULE #1 — the ONE path, failure RAISES, no fallback):
    #   env ``CPPMEGA_PATH_C_METAL_GEMM`` truthy selects the TENSOR-OP Metal prim
    #   :func:`chunk_precompute_fwd_metal_gemm_prim` (the two head-independent
    #   reductions — cb dot + summary_states accumulate — as ``T.gemm`` with a
    #   SHARED fp32 C accumulator, the Metal 8x8 simdgroup route). When OFF the
    #   byte-identical serial Metal prim is the parity reference. Explicit gate, NOT
    #   a silent fallback; a compile/parity failure PROPAGATES.
    _gemm_flag = str(os.environ.get("CPPMEGA_PATH_C_METAL_GEMM", "")).strip().lower()
    if _gemm_flag in ("1", "true", "yes", "on"):
        prim = chunk_precompute_fwd_metal_gemm_prim(
            batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
        )
    else:
        prim = chunk_precompute_fwd_metal_prim(
            batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
        )
    return tilelang.compile(
        prim,
        out_idx=[5, 6, 7],
        target=resolved_target,
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
    *,
    target: Any = None,
    **kwargs: Any,
) -> Any:
    """Compile the F1 inter-chunk recurrence kernel to a ``JITKernel``.

    ``target`` selects Metal (default) or CUDA (``"cuda"`` / sm_121). Outputs
    (prev_states, final_state) are the 4th/5th params; pass PRE-ZEROED contiguous
    fp32 buffers positionally, synchronize the device after. RULE #1: compile
    failures propagate (no serial fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )

    prim = inter_chunk_recur_fwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    return tilelang.compile(
        prim,
        out_idx=[3, 4],
        target=_resolve_chunked_compile_target(target),
    )
