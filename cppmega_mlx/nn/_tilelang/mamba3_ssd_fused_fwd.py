"""Path-C mamba3 FUSED SSD forward — ONE smem-resident CUDA kernel (gb10/sm_121).

This is the fused replacement for the un-fused F0 (precompute) + F1 (inter-chunk
recurrence) + F2 (scan+combine) chain
(``mamba3_chunked_precompute_core`` / ``mamba3_chunked_scan_core``). It mirrors
the canonical Mamba2/Tri-Dao SSD per-(batch,chunk,head) structure but keeps the
running chunk state ``state[headdim, dstate]`` RESIDENT IN SHARED MEMORY across
the whole chunk axis (the PyTorch-blog "Accelerating Mamba2 with Kernel Fusion"
design): a persistent per-(batch,head) threadgroup loops the chunks serially and
carries the state without round-tripping ``cb / dA_cumsum / summary_states /
prev_states`` through global memory. Only ``Output`` and ``final_state`` are
written to global; every SSD intermediate stays in shared/registers.

WHY (the MEASURED lever, docs/MAMBA3-PATHC-VS-CPPMEGA.md): the un-fused F0 alone
is 16.37 ms (>=5.7x cppmega's WHOLE fused mamba forward 3.11 ms) because it
re-reads/re-writes the chunk tensors through global memory between F0/F1/F2. The
fix is to fuse the three regions so the chunk state never leaves shared memory —
exactly what cppmega's Triton ``mamba_chunk_scan_combined`` does (state resident),
re-tiled for the GB10 sm_121 ~99 KB (101,376 B) per-block opt-in dynamic-smem cap.

ENV GATE (default OFF): ``CPPMEGA_MAMBA3_SSD_FUSED_FWD`` selects this fused path.
When OFF (default) callers keep using the existing F0/F1/F2 grid kernels + the
Metal path BYTE-IDENTICAL; nothing here runs. This module adds NO new default
behavior. RULE #1: when the gate is ON the fused kernel is the ONE path — on any
tile/smem/compile failure it RAISES with where+what; it NEVER truncates the state
columns and NEVER silently falls back to the slow un-fused chain.

PER-(batch,head) ALGORITHM (serial over chunk c = 0..nchunks-1):
  state[P,N] = h0  (fp32, resident in shared across the loop)
  for c in chunks:
    1. dA_cumsum[l] = inclusive_scan_l(A[h]*dt[c,l])           (F0 cumsum, fused)
    2. cb[i,s]      = sum_n C[c,i,n]*B[c,s,n]   (per-chunk C@B^T; F0 cb, fused)
    3. Y_diag[i,p]  = sum_{s<=i} cb[i,s]*exp2((dA[i]-dA[s])*p)*dt[s]*x[c,s,p]
    4. Y_off[i,p]   = exp2(dA[i]*p) * sum_n C[c,i,n]*state[p,n]   (uses RESIDENT state)
    5. Output[c,i,p]= Y_diag + Y_off + D[h]*x[c,i,p]
    6. summary[p,n] = sum_l exp2((dA[L-1]-dA[l])*p)*dt[l]*x[c,l,p]*B[c,l,n]  (F0 summary)
       state[p,n]   = exp2(dA[L-1]*p)*state[p,n] + summary[p,n]   (F1 carry, in smem)
  final_state[p,n]  = state[p,n]

INPUT ABI (the raw per-position SSD tensors F0 consumes — NOT the F2 handoff
buffers; the fused kernel IS F0+F1+F2):
  x   : (batch, seqlen, nheads, headdim)            fp16
  B   : (batch, seqlen, ngroups, dstate)            fp16
  C   : (batch, seqlen, ngroups, dstate)            fp16
  A   : (nheads,)                                   fp16   A = -softplus(...) <= 0
  dt  : (batch, seqlen, nheads)                     fp16
  D   : (nheads,)                                   fp16
  h0  : (batch, nheads, headdim, dstate)            fp32   initial inter-chunk state
OUTPUTS (caller-owned, PRE-ZEROED, passed positionally):
  Output      : (batch, seqlen, nheads, headdim)    fp16
  final_state : (batch, nheads, headdim, dstate)    fp32

Numerical contract (RULE #1): the fused forward reproduces the un-fused
F0->F1->F2 chain (and the serial reference) to fp16 ``< 5e-4`` over ALL elements.
The carried state is fp32 (the §3.3 precision rule — never downcast it).
"""

# CRITICAL: no ``from __future__ import annotations`` — the TileLang eager
# frontend reads the ``T.Tensor`` prim_func parameter annotations as real objects
# (see ``mamba3_chunked_scan_core`` docstring); PEP 563 string annotations would
# defer evaluation to a globalns lacking the factory closure vars and raise
# ``NameError: name 'seqlen' is not defined`` at prim_func build time.

import math
import os
from typing import Any

__all__ = [
    "MAMBA3_SSD_FUSED_FWD_OP_NAME",
    "MAMBA3_SSD_FUSED_FWD_ENV",
    "ssd_fused_fwd_enabled",
    "ssd_fused_fwd_grid",
    "ssd_fused_fwd_smem_budget_bytes",
    "ssd_fused_fwd_cuda_prim",
    "build_ssd_fused_fwd",
]

# Stable op-node name for the fused Path-C SSD forward segment.
MAMBA3_SSD_FUSED_FWD_OP_NAME = "mamba3_ssd_fused_fwd"

# Env gate (default OFF). When unset/falsey the existing un-fused F0/F1/F2 grid
# kernels + the Metal path stay byte-identical; this fused kernel does not run.
MAMBA3_SSD_FUSED_FWD_ENV = "CPPMEGA_MAMBA3_SSD_FUSED_FWD"

# 1/ln(2): ``exp2(x*p) == exp(x)`` (matches the F0/F1/F2 core convention exactly).
_LOG2E = 1.4426950408889634

# GB10 / sm_121 per-block opt-in dynamic-smem ceiling (docs/GB10-SMEM-LIMIT.md:
# cudaDevAttrMaxSharedMemoryPerBlockOptin = 101,376 B). The fused kernel MUST fit
# under this; the builder RAISES (RULE #1) if the computed budget exceeds it
# rather than letting the launch fail opaquely.
_GB10_SMEM_CAP_BYTES = 101_376


def ssd_fused_fwd_enabled() -> bool:
    """Return True iff the fused SSD forward path is env-gated ON (default OFF)."""
    return os.environ.get(MAMBA3_SSD_FUSED_FWD_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def ssd_fused_fwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[int, tuple[int, int]]:
    """Return ``(total_threadgroups, (gx, gy))`` for the fused forward grid.

    Grid is ``(batch, nheads)``: one PERSISTENT threadgroup per (batch, head) owns
    ALL nchunks for that (batch, head) and carries ``state[headdim, dstate]`` in
    shared memory across the chunk loop (the inter-chunk F1 carry pulled INTO the
    scan kernel so the state never round-trips to global). For prod
    (batch=1, nheads=112) this is 112 persistent threadgroups, each looping
    nchunks = seqlen/chunk_size = 64.

    NOTE this is FEWER threadgroups than the un-fused F2 3D grid (which is
    nheads*ceil(C/bM)*ceil(P/bN)*batch*nchunks); the trade is occupancy-for-no-
    global-round-trip. The serial chunk axis lives INSIDE the group (no cross-block
    atomics), which is legal because each (batch,head) state carry is independent.
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"ssd_fused_fwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"ssd_fused_fwd_grid: nheads ({nheads}) must be divisible by "
            f"ngroups ({ngroups})"
        )
    return batch * nheads, (batch, nheads)


def ssd_fused_fwd_smem_budget_bytes(
    chunk_size: int,
    headdim: int,
    dstate: int,
) -> dict[str, int]:
    """Compute the per-threadgroup static shared-memory budget (bytes).

    Returns a per-buffer breakdown plus ``total`` so the budget is auditable
    against the GB10 ~99 KB (101,376 B) opt-in cap. fp16 = 2 B/elem, fp32 = 4
    B/elem. ``L = chunk_size`` (the in-chunk axis), ``P = headdim``, ``N = dstate``.

    At prod (chunk=64, headdim=64, dstate=64): total = 82,688 B (80.75 KiB), which
    is 18,688 B (18.25 KiB) under the 101,376 B cap. dstate=64 fits whole — NO
    dstate streaming is required (the §-measured dstate-split was a NO-GO for perf
    anyway). The operand tiles (cb/x/C/B) are fp16; an all-fp32 staging would be
    115,456 B (112.75 KiB) and OVERFLOW the cap — hence fp16 operands + fp32
    accumulation (cast-on-read), which keeps the §3.3 precision and fits.
    """
    L = chunk_size
    P = headdim
    N = dstate
    f16 = 2
    f32 = 4
    budget = {
        # The ONE thing that persists across the chunk loop (fp32, the §3.3
        # precision rule — never downcast the carried state).
        "state_PN_f32": P * N * f32,
        # per-chunk cb = C@B^T; staged fp16 (cb is fp16 in the F0 handoff; the diag
        # scan casts it to fp32 per-element on read). Recomputed each chunk, never
        # to global. Matches the kernel's ``T.alloc_shared(..., dtype)``.
        "cb_LL_f16": L * L * f16,
        # in-chunk operand tiles staged fp16 once per chunk (they ARE fp16 in
        # global; the explicit serial reductions cast each element to fp32 on read,
        # so the accumulation still happens in fp32 — the §3.3 precision rule).
        "x_LP_f16": L * P * f16,
        "C_LN_f16": L * N * f16,
        "B_LN_f16": L * N * f16,
        # fp32 output accumulator for this chunk's L*P out tile (fp32: accumulator).
        "acc_o_LP_f32": L * P * f32,
        # per-chunk summary (decayed x^T@B) before the carry merge (fp32: accum).
        "summary_PN_f32": P * N * f32,
        # small per-chunk row buffers (fp32).
        "a_row_L_f32": L * f32,
        "dacs_L_f32": L * f32,
        "dt_L_f32": L * f32,
    }
    budget["total"] = sum(budget.values())
    return budget


def ssd_fused_fwd_cuda_prim(
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
    """Build the fused SSD forward CUDA ``@T.prim_func`` (gb10 / sm_121).

    ONE smem-resident kernel: F0 precompute (dA_cumsum + cb) + intra-chunk scan
    (Y_diag) + inter-chunk carry (Y_off using the resident state, then state
    update) per (batch, chunk, head). The state[headdim, dstate] is fp32 in shared
    memory and is carried across the serial chunk loop — it never touches global.

    RULE #1: on a tile/smem/shape failure this RAISES with where+what (it never
    truncates the dstate columns and never falls back to the un-fused chain). The
    SSD math is identical to F0->F1->F2; only the data-residency changes.
    """
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"ssd_fused_fwd_cuda_prim: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"ssd_fused_fwd_cuda_prim: nheads ({nheads}) must be divisible by "
            f"ngroups ({ngroups})"
        )
    # RULE #1: the fused kernel keeps the FULL dstate state resident — it does NOT
    # tile/truncate the dstate columns. If the smem budget would exceed the GB10
    # cap we RAISE here (where+what) rather than launch a kernel that overflows.
    budget = ssd_fused_fwd_smem_budget_bytes(chunk_size, headdim, dstate)
    if budget["total"] > _GB10_SMEM_CAP_BYTES:
        raise ValueError(
            f"ssd_fused_fwd_cuda_prim: per-threadgroup smem budget "
            f"{budget['total']} B (chunk={chunk_size}, headdim={headdim}, "
            f"dstate={dstate}) exceeds the GB10 sm_121 opt-in cap "
            f"{_GB10_SMEM_CAP_BYTES} B. Re-tile headdim/chunk to fit; this kernel "
            f"does NOT truncate dstate state columns (RULE #1). Breakdown: {budget}"
        )

    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    P = headdim
    N = dstate
    p = _LOG2E

    @T.prim_func
    def main(
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        A: T.Tensor((nheads), dtype),  # type: ignore
        dt: T.Tensor((batch, seqlen, nheads), dtype),  # type: ignore
        D: T.Tensor((nheads), dtype),  # type: ignore
        h0: T.Tensor((batch, nheads, headdim, dstate), accum_dtype),  # type: ignore
        Output: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        final_state: T.Tensor((batch, nheads, headdim, dstate), accum_dtype),  # type: ignore
    ):
        # Grid: one PERSISTENT threadgroup per (batch, head); serial over chunks.
        with T.Kernel(batch, nheads, threads=threads) as (bx, by):
            batch_idx = bx
            head_idx = by
            group_idx = head_idx // heads_per_group

            # ---- resident across the chunk loop (fp32 state carry) ----
            state = T.alloc_shared((P, N), accum_dtype)  # the inter-chunk carry

            # ---- per-chunk scratch (re-used every chunk; never to global) ----
            # Operand tiles are fp16 (they ARE fp16 in global); the explicit serial
            # reductions cast each element to fp32 on read so the accumulation still
            # runs in fp32 (the §3.3 precision rule). cb is stored fp16 to match the
            # un-fused F0 cb handoff dtype exactly (same rounding as F0->F2). This
            # fp16 staging keeps the budget at 80.75 KiB < the 99 KB GB10 cap; an
            # all-fp32 staging would be 112.75 KiB and OVERFLOW (RULE #1: fit, never
            # truncate dstate).
            a_row = T.alloc_shared((L,), accum_dtype)       # A[h]*dt[c,l]
            dacs = T.alloc_shared((L,), accum_dtype)        # inclusive cumsum
            dt_row = T.alloc_shared((L,), accum_dtype)      # dt[c,l] (fp32)
            cb_sh = T.alloc_shared((L, L), dtype)           # cb = C@B^T (fp16, this chunk)
            x_sh = T.alloc_shared((L, P), dtype)            # x[c, :, :] (fp16)
            C_sh = T.alloc_shared((L, N), dtype)            # C[c, :, :] (fp16)
            B_sh = T.alloc_shared((L, N), dtype)            # B[c, :, :] (fp16)
            acc_o = T.alloc_shared((L, P), accum_dtype)     # Y_diag+Y_off+D out tile (fp32)
            summary = T.alloc_shared((P, N), accum_dtype)   # per-chunk decayed x^T@B (fp32)

            a_scalar = T.alloc_local((1,), accum_dtype)
            d_scalar = T.alloc_local((1,), accum_dtype)
            acc_scan = T.alloc_local((1,), accum_dtype)

            tid = T.get_thread_binding(0)

            # ---- init the carried state from h0 (fp32, kept in smem) ----
            for cell0 in T.serial(0, P * N, threads):
                cell = cell0 + tid
                if cell < P * N:
                    pp = cell // N
                    nn = cell % N
                    state[pp, nn] = h0[batch_idx, head_idx, pp, nn]
            T.sync_threads()

            a_scalar[0] = T.Cast(accum_dtype, A[head_idx])
            d_scalar[0] = T.Cast(accum_dtype, D[head_idx])

            # ===================== SERIAL CHUNK LOOP =====================
            for c in T.serial(nchunks):
                base = c * chunk_size

                # --- 1. dA_cumsum: a[l] = A[h]*dt[c,l]; inclusive scan over l. ---
                for l in T.Parallel(L):
                    a_row[l] = a_scalar[0] * T.Cast(
                        accum_dtype, dt[batch_idx, base + l, head_idx]
                    )
                    dt_row[l] = T.Cast(
                        accum_dtype, dt[batch_idx, base + l, head_idx]
                    )
                T.sync_threads()
                # serial inclusive scan on a single lane (L == chunk_size is small).
                if tid == 0:
                    acc_scan[0] = T.Cast(accum_dtype, 0)
                    for l in T.serial(L):
                        acc_scan[0] = acc_scan[0] + a_row[l]
                        dacs[l] = acc_scan[0]
                T.sync_threads()

                # --- stage x[c], C[c], B[c] into shared (fp16; cast on read) ---
                for cell0 in T.serial(0, L * P, threads):
                    cell = cell0 + tid
                    if cell < L * P:
                        li = cell // P
                        pp = cell % P
                        x_sh[li, pp] = x[batch_idx, base + li, head_idx, pp]
                for cell0 in T.serial(0, L * N, threads):
                    cell = cell0 + tid
                    if cell < L * N:
                        li = cell // N
                        nn = cell % N
                        C_sh[li, nn] = C[batch_idx, base + li, group_idx, nn]
                        B_sh[li, nn] = B[batch_idx, base + li, group_idx, nn]
                T.sync_threads()

                # --- 2. cb[i,s] = sum_n C[c,i,n]*B[c,s,n]  (per-chunk C@B^T) ---
                # accumulate in fp32, store cb fp16 (matches F0's cb handoff dtype).
                for cell0 in T.serial(0, L * L, threads):
                    cell = cell0 + tid
                    if cell < L * L:
                        li = cell // L
                        si = cell % L
                        cb_acc = T.alloc_local((1,), accum_dtype)
                        cb_acc[0] = T.Cast(accum_dtype, 0)
                        for nn in T.serial(N):
                            cb_acc[0] = cb_acc[0] + T.Cast(
                                accum_dtype, C_sh[li, nn]
                            ) * T.Cast(accum_dtype, B_sh[si, nn])
                        cb_sh[li, si] = T.Cast(dtype, cb_acc[0])
                T.sync_threads()

                # --- 3+4. Output[i,p] = Y_diag + Y_off + D-skip, per (i,p) cell. ---
                #   Y_diag[i,p] = sum_{s<=i} cb[i,s]*exp2((dacs[i]-dacs[s])*p)*dt[s]*x[s,p]
                #   Y_off[i,p]  = exp2(dacs[i]*p) * sum_n C[i,n]*state[p,n]   (resident)
                for cell0 in T.serial(0, L * P, threads):
                    cell = cell0 + tid
                    if cell < L * P:
                        li = cell // P
                        pp = cell % P
                        o_acc = T.alloc_local((1,), accum_dtype)
                        o_acc[0] = T.Cast(accum_dtype, 0)
                        # intra-chunk causal diagonal (cast fp16 operands to fp32)
                        for si in T.serial(L):
                            if si <= li:
                                decay = T.exp2((dacs[li] - dacs[si]) * p)
                                o_acc[0] = o_acc[0] + (
                                    T.Cast(accum_dtype, cb_sh[li, si])
                                    * decay
                                    * dt_row[si]
                                    * T.Cast(accum_dtype, x_sh[si, pp])
                                )
                        # inter-chunk off-diagonal against the RESIDENT state
                        off_acc = T.alloc_local((1,), accum_dtype)
                        off_acc[0] = T.Cast(accum_dtype, 0)
                        for nn in T.serial(N):
                            off_acc[0] = off_acc[0] + (
                                T.Cast(accum_dtype, C_sh[li, nn]) * state[pp, nn]
                            )
                        o_acc[0] = o_acc[0] + off_acc[0] * T.exp2(dacs[li] * p)
                        # D skip
                        o_acc[0] = o_acc[0] + (
                            d_scalar[0] * T.Cast(accum_dtype, x_sh[li, pp])
                        )
                        acc_o[li, pp] = o_acc[0]
                T.sync_threads()

                # --- write this chunk's Output tile (fp16) ---
                for cell0 in T.serial(0, L * P, threads):
                    cell = cell0 + tid
                    if cell < L * P:
                        li = cell // P
                        pp = cell % P
                        Output[batch_idx, base + li, head_idx, pp] = T.Cast(
                            dtype, acc_o[li, pp]
                        )
                T.sync_threads()

                # --- 5. summary[p,n] = sum_l exp2((dacs[L-1]-dacs[l])*p)*dt[l]*x[l,p]*B[l,n]
                for cell0 in T.serial(0, P * N, threads):
                    cell = cell0 + tid
                    if cell < P * N:
                        pp = cell // N
                        nn = cell % N
                        s_acc = T.alloc_local((1,), accum_dtype)
                        s_acc[0] = T.Cast(accum_dtype, 0)
                        for l in T.serial(L):
                            decay = T.exp2((dacs[L - 1] - dacs[l]) * p)
                            s_acc[0] = s_acc[0] + (
                                decay
                                * dt_row[l]
                                * T.Cast(accum_dtype, x_sh[l, pp])
                                * T.Cast(accum_dtype, B_sh[l, nn])
                            )
                        summary[pp, nn] = s_acc[0]
                T.sync_threads()

                # --- 6. state[p,n] = exp2(dacs[L-1]*p)*state[p,n] + summary[p,n] ---
                #   (the F1 inter-chunk carry, done IN shared — never to global)
                tail_decay = T.alloc_local((1,), accum_dtype)
                for cell0 in T.serial(0, P * N, threads):
                    cell = cell0 + tid
                    if cell < P * N:
                        pp = cell // N
                        nn = cell % N
                        tail_decay[0] = T.exp2(dacs[L - 1] * p)
                        state[pp, nn] = (
                            tail_decay[0] * state[pp, nn] + summary[pp, nn]
                        )
                T.sync_threads()

            # ===================== after the chunk loop =====================
            # write the carried state out as final_state (the ONLY state write).
            for cell0 in T.serial(0, P * N, threads):
                cell = cell0 + tid
                if cell < P * N:
                    pp = cell // N
                    nn = cell % N
                    final_state[batch_idx, head_idx, pp, nn] = state[pp, nn]

    return main


def build_ssd_fused_fwd(
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
    """Compile the fused SSD forward kernel to a ``JITKernel`` (CUDA / sm_121).

    ``target`` selects the codegen backend; this fused kernel is CUDA-only (the
    gb10 fusion target). Outputs (Output, final_state) are the 8th/9th params; pass
    PRE-ZEROED contiguous buffers positionally (no ``out_idx`` allocation) and
    synchronize the device after dispatch.

    RULE #1: this builder does NOT silently default to Metal — a non-CUDA target
    RAISES (the fused kernel is the gb10 fusion path; the Metal F0/F1/F2 stay the
    Metal path). Compile failures propagate (no swallow / no un-fused fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )

    resolved_target = _resolve_chunked_compile_target(target)
    kind = str(getattr(getattr(resolved_target, "kind", None), "name", "")).lower()
    if "cuda" not in kind:
        raise ValueError(
            "build_ssd_fused_fwd: the fused SSD forward kernel is CUDA-only "
            f"(gb10 / sm_121); got target kind {kind!r}. The Metal path keeps the "
            "un-fused F0/F1/F2 kernels (RULE #1: no silent Metal default)."
        )

    prim = ssd_fused_fwd_cuda_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    # sm_121 (Blackwell): the elementwise/serial copies otherwise risk lowering to
    # TMA (``__tvm_tensormap_create_tiled``) where the tensormap descriptor is
    # 32-byte aligned vs TMA's 64-byte requirement -> "Invalid TMA descriptor
    # arguments" at RUN. The fused kernel has NO T.gemm (the scans are explicit
    # serial reductions over shared tiles), so there is nothing TMA-eligible, but
    # disabling the TMA + warp-specialized lowering is a no-op-or-safer escape
    # hatch matching the F2/B2 cuda compile path (RULE #1: explicit codegen choice,
    # not a silent fallback).
    pass_configs = {
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True,
    }
    return tilelang.compile(
        prim, out_idx=[7, 8], target=resolved_target, pass_configs=pass_configs
    )
