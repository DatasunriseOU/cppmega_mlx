"""Path-C mamba3 chunked-SSD BACKWARD core (B2 / B1 / B0) — Metal grid kernels.

Stage 3 of ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2/§7. These are the three
NEW single-entry grid kernels that are the EXACT ANALYTIC TRANSPOSE of the proven
forward F0/F1/F2 (``mamba3_chunked_precompute_core`` /
``mamba3_chunked_scan_core``). They mirror the validated MLX backward prototype on
branch ``mamba3-chunked-backward``
(``scratch/mamba3_chunked_backward_proto.py``: worst grad ``1.30e-4`` vs the
serial VJP, 34.5-155x, chunk {64,128,256}).

The 3 segments (design §2 backward table; time-reversed transpose of F0/F1/F2):

  * **B2  ``mamba3_chunk_scan_combine_bwd``** — grid-parallel (same grid family as
    the forward F2). The transpose of the output/gate + Y_diag + Y_off + skip
    forward path. From ``dout`` it produces the per-position output-side grads:
    ``dz`` (gate), ``dx`` (D skip), ``dD``, the per-chunk-state grad
    ``dchunk_states`` (= grad wrt the F1 ``prev_states``), the diag+off ``dC``,
    the input-outer grads ``dinp_diag``/``dinp_off`` (-> dx_inp, dB), and the
    Y_off/Y_diag contributions to ``dA_cumsum``.

  * **B1  ``mamba3_inter_chunk_recur_bwd``** — the NEW O(S/C) **reverse** scan
    (upper-tri ``decay_chunk`` contraction). From ``dchunk_states`` (+ optional
    ``dh_last``) it produces ``dstates`` (grad of the per-chunk summary states),
    ``dh0``, and the chunk-tail contribution to ``dA_cumsum``. This is the ONE
    genuinely new kernel (the forward F1 lower-tri recurrence transposes to an
    upper-tri reverse combine).

  * **B0  ``mamba3_chunk_precompute_bwd``** — grid-parallel (same grid as F0).
    The transpose of the precompute: folds ``dstates`` (from B1) and
    ``dinp_diag``/``dinp_off`` (from B2) into ``dinp``, applies the
    ``decay_states`` transpose (-> the ``dA_cumsum`` scatter), the segsum/cumsum
    VJP for the log-space decay chain rule, and assembles the final input grads
    ``dlog_decay`` (= dA*dt path), ``dx``, ``dB``, ``dC``.

Numerical contract (the ONE clear path, RULE #1): the (B2 -> B1 -> B0) chain
reproduces the serial Path-C backward per-grad-tensor ``max|abs| < 1e-3`` (the MLX
proto is ``1.30e-4``). On any shape/compile failure the builders RAISE with
where+what; there is NO serial fallback (the chunked-vs-serial choice is an
explicit gate elsewhere).

The handoff buffer shapes match the forward F0/F1/F2 cache exactly (the backward
REUSES the forward-materialized ``cb / dA_cumsum / prev_states / summary_states``
boundary states instead of the 8x checkpoint-replay — the elimination the design
§3/§6 calls the dominant backward win).
"""

# CRITICAL: no ``from __future__ import annotations`` — see the scan-core module
# docstring (TileLang eager frontend reads ``T.Tensor`` annotations as objects).

import math
from typing import Any

__all__ = [
    "MAMBA3_CHUNK_SCAN_COMBINE_BWD_OP_NAME",
    "MAMBA3_INTER_CHUNK_RECUR_BWD_OP_NAME",
    "MAMBA3_CHUNK_PRECOMPUTE_BWD_OP_NAME",
    "chunk_scan_combine_bwd_metal_prim",
    "inter_chunk_recur_bwd_metal_prim",
    "chunk_precompute_bwd_metal_prim",
    "chunk_scan_combine_bwd_grid",
    "inter_chunk_recur_bwd_grid",
    "chunk_precompute_bwd_grid",
    "build_chunk_scan_combine_bwd_metal",
    "build_inter_chunk_recur_bwd_metal",
    "build_chunk_precompute_bwd_metal",
]

# Stable op-node names for the Path-C B0/B1/B2 backward segments (design §2/§3.2).
# These are the time-reversed transpose mirrors of the forward op-names.
MAMBA3_CHUNK_SCAN_COMBINE_BWD_OP_NAME = "mamba3_chunk_scan_combine_bwd"
MAMBA3_INTER_CHUNK_RECUR_BWD_OP_NAME = "mamba3_inter_chunk_recur_bwd"
MAMBA3_CHUNK_PRECOMPUTE_BWD_OP_NAME = "mamba3_chunk_precompute_bwd"

# 1/ln(2): ``exp2(x*p) == exp(x)`` (matches the forward core convention).
_LOG2E = 1.4426950408889634


def _silu_grad_expr(T: Any, z_val: Any, accum_dtype: Any) -> Any:
    """d/dz silu(z) = sigmoid(z)*(1 + z*(1 - sigmoid(z))) — proto ``_silu_grad``."""
    one = T.Cast(accum_dtype, 1.0)
    s = one / (one + T.exp(-z_val))
    return s * (one + z_val * (one - s))


# --------------------------------------------------------------------------- #
# B2 — scan+combine backward (output/gate + Y_diag + Y_off transpose).          #
#      Grid-parallel; the transpose of the forward F2 scan+combine.             #
# --------------------------------------------------------------------------- #


def chunk_scan_combine_bwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[int, tuple[int, int, int]]:
    """Return ``(total_threadgroups, (gx, gy, gz))`` for the B2 grid.

    Grid is ``(batch*nchunks, nheads, 1)`` — one threadgroup per (batch, chunk,
    head), each transposing this head's chunk of the output/Y path. Many
    threadgroups vs the serial backward's 1.
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_scan_combine_bwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_scan_combine_bwd_grid: nheads ({nheads}) must be divisible by "
            f"ngroups ({ngroups})"
        )
    nchunks = seqlen // chunk_size
    return batch * nchunks * nheads, (batch * nchunks, nheads, 1)


def chunk_scan_combine_bwd_metal_prim(
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
    """Build the B2 ``mamba3_chunk_scan_combine_bwd`` Metal ``@T.prim_func``.

    Inputs (forward cache + upstream cotangent):
      dout        : (batch, seqlen, nheads, headdim)        cotangent of out
      cb          : (batch, nchunks, ngroups, chunk, chunk) = C@B^T (forward F0)
      x           : (batch, seqlen, nheads, headdim)
      z           : (batch, seqlen, nheads, headdim)
      dt          : (batch, nheads, nchunks, chunk)
      dA_cumsum   : (batch, nheads, nchunks, chunk)
      C           : (batch, seqlen, ngroups, dstate)
      prev_states : (batch, nchunks, nheads, headdim, dstate) entry state (F1)
      D           : (nheads,)
      y           : (batch, seqlen, nheads, headdim)         forward pre-gate y
    Outputs (per-position output-side grads; the B0/B1 consumers):
      dC           : (batch, seqlen, ngroups, dstate)        diag+off
      dx           : (batch, seqlen, nheads, headdim)        D-skip path
      dz           : (batch, seqlen, nheads, headdim)        gate
      dchunk_states: (batch, nchunks, nheads, headdim, dstate) grad of prev_states
      dinp         : (batch, seqlen, nheads, headdim, dstate) diag input-outer grad
      dA_cumsum_y  : (batch, nheads, nchunks, chunk)         Y_off+Y_diag dA grad
      dD           : (nheads,)

    Transpose of the forward F2 path (proto ``chunked_mamba3_backward`` up to and
    including the Y_diag / Y_off / state-decay transposes). One threadgroup per
    (batch, chunk, head). Per-position l is parallel; reductions over the chunk
    (s axis) and state (n) are serial within a thread.
    """
    import tilelang
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_scan_combine_bwd_metal_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_scan_combine_bwd_metal_prim: nheads ({nheads}) must be "
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
        dout: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),  # type: ignore
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        z: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        D: T.Tensor((nheads), dtype),  # type: ignore
        y: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        dC: T.Tensor((batch, seqlen, nheads, dstate), accum_dtype),  # type: ignore
        dx: T.Tensor((batch, seqlen, nheads, headdim), accum_dtype),  # type: ignore
        dz: T.Tensor((batch, seqlen, nheads, headdim), accum_dtype),  # type: ignore
        dchunk_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dinp: T.Tensor((batch, seqlen, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dA_cumsum_y: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),  # type: ignore
        dD: T.Tensor((nheads), accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size

            dacs = T.alloc_shared((L,), accum_dtype)        # dA_cumsum row (this head/chunk)
            dY = T.alloc_shared((L, headdim), accum_dtype)  # dY[l,p] (post split)
            dAcs_acc = T.alloc_shared((L,), accum_dtype)     # accumulated dA_cumsum grad

            # --- load dA_cumsum row, init dA grad accumulator ---
            for l in T.Parallel(L):
                dacs[l] = T.Cast(accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, l])
                dAcs_acc[l] = T.Cast(accum_dtype, 0)
            T.sync_threads()

            # --- output/gate + D-skip transpose: dY[l,p], dz, dx, dD ---
            #   out = gate*y ; gate=silu(z) ; y = Y + D*x
            #   dgate = dout*y ; dy = dout*gate ; dz = dgate*silu'(z)
            #   dx_skip = D*dy ; dD += dy*x ; dY = dy
            dD_local = T.alloc_local((1,), accum_dtype)
            dD_local[0] = T.Cast(accum_dtype, 0)
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                s = base + ll
                z_v = T.Cast(accum_dtype, z[batch_idx, s, head_idx, pp])
                gate = z_v / (T.Cast(accum_dtype, 1.0) + T.exp(-z_v))
                y_v = T.Cast(accum_dtype, y[batch_idx, s, head_idx, pp])
                dout_v = T.Cast(accum_dtype, dout[batch_idx, s, head_idx, pp])
                dgate = dout_v * y_v
                dy_v = dout_v * gate
                dz[batch_idx, s, head_idx, pp] = dgate * _silu_grad_expr(T, z_v, accum_dtype)
                d_v = T.Cast(accum_dtype, D[head_idx])
                x_v = T.Cast(accum_dtype, x[batch_idx, s, head_idx, pp])
                dx[batch_idx, s, head_idx, pp] = d_v * dy_v
                dD_local[0] = dD_local[0] + dy_v * x_v
                dY[ll, pp] = dy_v
            T.sync_threads()
            T.atomic_add(dD[head_idx], dD_local[0])

            # The Y_off / Y_diag transpose is done with explicit serial reductions
            # below to keep a single, race-free accumulation per (l,n), (p,n) and l
            # cell. First zero the dC / dinp / dchunk_states slices this
            # threadgroup owns (each (batch,chunk,head) owns disjoint slices).
            for ln in T.Parallel(L * dstate):
                ll = ln // dstate
                nn = ln % dstate
                dC[batch_idx, base + ll, head_idx, nn] = T.Cast(accum_dtype, 0)
            for pn in T.Parallel(headdim * dstate):
                pp = pn // dstate
                nn = pn % dstate
                dchunk_states[batch_idx, chunk_idx, head_idx, pp, nn] = T.Cast(
                    accum_dtype, 0
                )
            for lpn0 in T.serial(0, L * headdim * dstate, threads):
                lane = T.get_thread_binding(0)
                idx = lpn0 + lane
                if idx < L * headdim * dstate:
                    ll = idx // (headdim * dstate)
                    rem = idx % (headdim * dstate)
                    pp = rem // dstate
                    nn = rem % dstate
                    dinp[batch_idx, base + ll, head_idx, pp, nn] = T.Cast(
                        accum_dtype, 0
                    )
            T.sync_threads()

            # ---- dC = dC_off + dC_diag (per (l,n)); dchunk_states + state_decay dA ----
            #   dC_off[l,n]  = sum_p dY[l,p]*chunk_states[p,n]*state_decay[l]
            #   dC_diag[l,n] = sum_{s<=l} Lmat[l,s]*dt_s*(sum_p dY[l,p]*x[s,p])*B[s,n]
            #     Lmat[l,s] = exp(dacs[l]-dacs[s]).  inp=dt*x⊗B so dt_s appears here.
            #   dstate_decay[l] = sum_n dC_off-inner * C[l,n] ; dA += dstate_decay*sd
            for ll in T.serial(L):
                s = base + ll
                sd = T.exp2(dacs[ll] * p)
                dsd = T.alloc_local((1,), accum_dtype)
                dsd[0] = T.Cast(accum_dtype, 0)
                if T.get_thread_binding(0) == 0:
                    for nn in T.serial(dstate):
                        # dC_off inner: sum_p dY[l,p]*chunk_states[p,n]
                        accn = T.alloc_local((1,), accum_dtype)
                        accn[0] = T.Cast(accum_dtype, 0)
                        for pp in T.serial(headdim):
                            cs = prev_states[batch_idx, chunk_idx, head_idx, pp, nn]
                            accn[0] = accn[0] + dY[ll, pp] * cs
                        c_v = T.Cast(accum_dtype, C[batch_idx, s, group_idx, nn])
                        # dC_diag inner: sum_{s2<=l} Lmat[l,s2]*dt_s2*(sum_p dY*x)*B[s2,n]
                        cdiag = T.alloc_local((1,), accum_dtype)
                        cdiag[0] = T.Cast(accum_dtype, 0)
                        for ss in T.serial(0, ll + 1):
                            lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                            dt_s = T.Cast(
                                accum_dtype, dt[batch_idx, head_idx, chunk_idx, ss]
                            )
                            b_v = T.Cast(
                                accum_dtype, B[batch_idx, base + ss, group_idx, nn]
                            )
                            dyx = T.alloc_local((1,), accum_dtype)
                            dyx[0] = T.Cast(accum_dtype, 0)
                            for pp in T.serial(headdim):
                                dyx[0] = dyx[0] + dY[ll, pp] * T.Cast(
                                    accum_dtype, x[batch_idx, base + ss, head_idx, pp]
                                )
                            cdiag[0] = cdiag[0] + lmat * dt_s * dyx[0] * b_v
                        dC[batch_idx, s, head_idx, nn] = (
                            dC[batch_idx, s, head_idx, nn] + accn[0] * sd + cdiag[0]
                        )
                        dsd[0] = dsd[0] + accn[0] * c_v
                    dAcs_acc[ll] = dAcs_acc[ll] + dsd[0] * sd
            T.sync_threads()

            # dchunk_states[p,n] += sum_l dY[l,p]*C[l,n]*state_decay[l]
            for pn in T.Parallel(headdim * dstate):
                pp = pn // dstate
                nn = pn % dstate
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for ll in T.serial(L):
                    sd = T.exp2(dacs[ll] * p)
                    c_v = T.Cast(accum_dtype, C[batch_idx, base + ll, group_idx, nn])
                    acc[0] = acc[0] + dY[ll, pp] * c_v * sd
                dchunk_states[batch_idx, chunk_idx, head_idx, pp, nn] = acc[0]
            T.sync_threads()

            # ---- Y_diag transpose (intra-chunk) ----
            #   Y_diag[l,p] = sum_{s<=l} sum_n C[l,n]*Lmat[l,s]*inp[s,p,n]
            #   Lmat[l,s] = exp(segsum) = exp(dA_cumsum[l]-dA_cumsum[s]) for s<=l, masked by cb.
            #   In OUR form Y_diag is realized by cb (=C@B^T) folded with dt and decay
            #   (see scan-core): M[l,s] = cb[l,s]*exp(dacs[l]-dacs[s])*dt[s] (s<=l).
            #   Y_diag[l,p] = sum_s M[l,s] * x[s,p].  (inp[s,p,n]=x[s,p]*B[s,n]; cb folds B,C)
            #   dinp[s,p,n] (grad wrt inp=dt*x⊗B) = sum_{l>=s} dY[l,p]*C[l,n]*Lmat[l,s]
            #   (NO extra dt: inp ALREADY includes dt — the proto einsum has none).
            #   dA grad from Lmat handled in the dseg loop below.
            for sp in T.serial(0, L * headdim, threads):
                lane = T.get_thread_binding(0)
                spi = sp + lane
                if spi < L * headdim:
                    ss = spi // headdim
                    pp = spi % headdim
                    sidx = base + ss
                    for nn in T.serial(dstate):
                        acc = T.alloc_local((1,), accum_dtype)
                        acc[0] = T.Cast(accum_dtype, 0)
                        for ll in T.serial(ss, L):  # l >= s (lower-tri)
                            lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                            c_v = T.Cast(
                                accum_dtype, C[batch_idx, base + ll, group_idx, nn]
                            )
                            acc[0] = acc[0] + dY[ll, pp] * c_v * lmat
                        dinp[batch_idx, sidx, head_idx, pp, nn] = (
                            dinp[batch_idx, sidx, head_idx, pp, nn] + acc[0]
                        )
            T.sync_threads()

            # Y_diag dA grad: dLmat[l,s] = sum_{p,n} dY[l,p]*C[l,n]*inp[s,p,n]*dt[s]
            #   then dseg = dLmat*Lmat ; segsum-vjp: +dacs[l] over l>=s, -dacs[s] over s<l
            # Realize via: contribution to dAcs_acc[l] += sum_{s<l} dseg[l,s];
            #              contribution to dAcs_acc[s] -= sum_{l>s} dseg[l,s].
            if T.get_thread_binding(0) == 0:
                for ll in T.serial(L):
                    for ss in T.serial(0, ll + 1):
                        # dLmat[l,s]
                        cb_v = T.Cast(
                            accum_dtype, cb[batch_idx, chunk_idx, group_idx, ll, ss]
                        )
                        lmat = T.exp2((dacs[ll] - dacs[ss]) * p)
                        dt_s = T.Cast(
                            accum_dtype, dt[batch_idx, head_idx, chunk_idx, ss]
                        )
                        dlmat = T.alloc_local((1,), accum_dtype)
                        dlmat[0] = T.Cast(accum_dtype, 0)
                        for pp in T.serial(headdim):
                            # sum_n C[l,n]*inp[s,p,n] == (cb folds C@B^T); use the
                            # cb*x realization: M=cb*lmat*dt, so dM=dY*x; dseg=dM*M.
                            dlmat[0] = dlmat[0] + dY[ll, pp] * T.Cast(
                                accum_dtype, x[batch_idx, base + ss, head_idx, pp]
                            )
                        # M[l,s] = cb*lmat*dt_s ; dseg = (dM * M) where the lmat
                        # factor carries the dA dependence.
                        m_ls = cb_v * lmat * dt_s
                        dseg = dlmat[0] * m_ls
                        if ll > ss:
                            dAcs_acc[ll] = dAcs_acc[ll] + dseg
                            dAcs_acc[ss] = dAcs_acc[ss] - dseg
            T.sync_threads()

            for l in T.Parallel(L):
                dA_cumsum_y[batch_idx, head_idx, chunk_idx, l] = dAcs_acc[l]

    return main


def build_chunk_scan_combine_bwd_metal(
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
    """Compile the B2 scan+combine backward kernel to a ``JITKernel``.

    ``target`` selects Metal (default) or CUDA (``"cuda"`` / sm_121). Outputs
    (dC, dx, dz, dchunk_states, dinp, dA_cumsum_y, dD) are the trailing 7 params;
    pass PRE-ZEROED contiguous buffers positionally. RULE #1: compile failures
    propagate (no serial fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )

    prim = chunk_scan_combine_bwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    resolved_target = _resolve_chunked_compile_target(target)
    # CUDA (sm_121): mirror the forward F2 compile-site EXACTLY — thread the same
    # two pass_configs so the CUDA codegen surface is identical to the validated
    # forward. The B2 prim BODY has NO T.gemm / TMA-eligible copy (grep-confirmed:
    # zero T.gemm / shared.dyn / make_swizzled_layout), so there is no tensormap
    # descriptor to mis-align; disabling the TMA + warp-specialized lowering is a
    # no-op-or-safer escape hatch that keeps the compile path byte-identical to
    # the forward. The Metal branch is UNCHANGED (no pass_configs). RULE #1:
    # explicit per-target codegen choice, never a silent fallback.
    kind = str(getattr(getattr(resolved_target, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        pass_configs = {
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
        }
        return tilelang.compile(
            prim,
            out_idx=[11, 12, 13, 14, 15, 16, 17],
            target=resolved_target,
            pass_configs=pass_configs,
        )
    return tilelang.compile(
        prim,
        out_idx=[11, 12, 13, 14, 15, 16, 17],
        target=resolved_target,
    )


# --------------------------------------------------------------------------- #
# B1 — inter-chunk recurrence backward (the NEW O(S/C) REVERSE upper-tri scan). #
#      The transpose of the forward F1 lower-tri inter-chunk recurrence.         #
# --------------------------------------------------------------------------- #


def inter_chunk_recur_bwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[int, tuple[int, int, int]]:
    """Return ``(total_threadgroups, (gx, gy, gz))`` for the B1 reverse grid.

    Grid is ``(batch, nheads, 1)``: one threadgroup per (batch, head) walks the
    O(nchunks) chunk axis in REVERSE carrying the adjoint state. Mirrors the
    forward F1 grid; the chunk axis is the only serial axis (O(S/C), not O(S)).
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"inter_chunk_recur_bwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    return batch * nheads, (batch, nheads, 1)


def inter_chunk_recur_bwd_metal_prim(
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
    """Build the B1 ``mamba3_inter_chunk_recur_bwd`` Metal ``@T.prim_func``.

    Inputs (forward cache + the B2 grad of prev_states):
      dchunk_states : (batch, nchunks, nheads, headdim, dstate) grad of prev_states (B2)
      dA_cumsum     : (batch, nheads, nchunks, chunk)           forward (F0)
      dh_last       : (batch, nheads, headdim, dstate)          cotangent of final_state
      prev_states   : (batch, nchunks, nheads, headdim, dstate) entry state[c] (F1; REUSED)
    Outputs:
      dstates       : (batch, nchunks, nheads, headdim, dstate) grad of summary_states
      dh0           : (batch, nheads, headdim, dstate)          grad of h0
      dA_cumsum_tail: (batch, nheads, nchunks, chunk)           chunk-tail dA grad

    The forward F1 recurrence is
        state[0] = h0 ; prev[c] = state[c] ; state[c+1] = decay[c]*state[c]+summary[c]
    with decay[c] = exp(dA_cumsum[...,c,L-1]). Its adjoint is the REVERSE scan
    (proto's inter-chunk transpose / ssd_state_passing _bwd). With g = grad wrt
    state[c+1] carried in reverse:
        g = dh_last                         # grad wrt state[nchunks]
        for c in reversed(range(nchunks)):
            dstates[c] = g                  # summary[c] -> state[c+1] additively
            dtail[c]  = sum_{p,n} g * state[c] * decay[c]   # decay=exp(tail), chain rule
            g = decay[c]*g + dchunk_states[c]               # grad wrt state[c]
        dh0 = g
    The boundary states ``state[c]`` are the forward-materialized ``prev_states``
    (the design §3/§6 reuse that ELIMINATES the 8x checkpoint-replay) — read
    directly, NOT recomputed. Each thread owns disjoint (p,n) state cells; the
    chunk axis is the serial REVERSE carry. ``dtail[c]`` is scattered into
    ``dA_cumsum_tail[...,c,L-1]`` (the per-cell sum via atomic_add).
    """
    import tilelang
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"inter_chunk_recur_bwd_metal_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )

    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    L = chunk_size
    p = _LOG2E
    cells = headdim * dstate

    @T.prim_func
    def main(
        dchunk_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), T.float16),  # type: ignore
        dh_last: T.Tensor((batch, nheads, headdim, dstate), accum_dtype),  # type: ignore
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dstates: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dh0: T.Tensor((batch, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dA_cumsum_tail: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch, nheads, threads=threads) as (bx, by):
            batch_idx = bx
            head_idx = by
            g = T.alloc_local((1,), accum_dtype)

            # zero the dA_cumsum_tail row (only the L-1 slot is written below).
            for cl in T.serial(0, nchunks * L, threads):
                lane = T.get_thread_binding(0)
                idx = cl + lane
                if idx < nchunks * L:
                    cc = idx // L
                    ll = idx % L
                    dA_cumsum_tail[batch_idx, head_idx, cc, ll] = T.Cast(
                        accum_dtype, 0
                    )
            T.sync_threads()

            for cell0 in T.serial(0, cells, threads):
                lane = T.get_thread_binding(0)
                cell = cell0 + lane
                if cell < cells:
                    pp = cell // dstate
                    nn = cell % dstate
                    # ---- reverse adjoint scan (g = grad wrt state[c+1]) ----
                    g[0] = dh_last[batch_idx, head_idx, pp, nn]
                    for c in T.serial(nchunks):
                        cc = nchunks - 1 - c
                        # summary[cc] enters state[cc+1] additively -> dstates = g
                        dstates[batch_idx, cc, head_idx, pp, nn] = g[0]
                        tail = T.Cast(
                            accum_dtype, dA_cumsum[batch_idx, head_idx, cc, L - 1]
                        )
                        decay = T.exp2(tail * p)
                        state_cc = prev_states[batch_idx, cc, head_idx, pp, nn]
                        # dtail = g * state[cc] * decay (decay=exp(tail), chain rule)
                        T.atomic_add(
                            dA_cumsum_tail[batch_idx, head_idx, cc, L - 1],
                            g[0] * state_cc * decay,
                        )
                        # grad wrt state[cc] = decay*g + dchunk_states[cc]
                        g[0] = decay * g[0] + dchunk_states[
                            batch_idx, cc, head_idx, pp, nn
                        ]
                    dh0[batch_idx, head_idx, pp, nn] = g[0]

    return main


def build_inter_chunk_recur_bwd_metal(
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
    """Compile the B1 reverse inter-chunk recurrence kernel to a ``JITKernel``.

    ``target`` selects Metal (default) or CUDA (``"cuda"`` / sm_121). Outputs
    (dstates, dh0, dA_cumsum_tail) are the trailing 3 params; pass PRE-ZEROED
    contiguous fp32 buffers positionally. RULE #1: compile failures propagate
    (no serial fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )

    prim = inter_chunk_recur_bwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    resolved_target = _resolve_chunked_compile_target(target)
    # CUDA (sm_121): mirror the forward F2 compile-site EXACTLY — thread the same
    # two pass_configs. The B1 prim BODY has NO T.gemm / TMA-eligible copy
    # (grep-confirmed: zero T.gemm / shared.dyn / make_swizzled_layout — it does
    # not even alloc_shared), so there is no tensormap descriptor to mis-align;
    # the pass_configs are a no-op-or-safer escape hatch keeping the CUDA compile
    # path byte-identical to the forward. The Metal branch is UNCHANGED (no
    # pass_configs). RULE #1: explicit per-target codegen choice, never a silent
    # fallback.
    kind = str(getattr(getattr(resolved_target, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        pass_configs = {
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
        }
        return tilelang.compile(
            prim,
            out_idx=[4, 5, 6],
            target=resolved_target,
            pass_configs=pass_configs,
        )
    return tilelang.compile(
        prim,
        out_idx=[4, 5, 6],
        target=resolved_target,
    )


# --------------------------------------------------------------------------- #
# B0 — precompute backward (assemble dinp -> dlog_decay/dx/dB/dC + dA chain).    #
#      The transpose of the forward F0 precompute (grid-parallel).              #
# --------------------------------------------------------------------------- #


def chunk_precompute_bwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
) -> tuple[int, tuple[int, int, int]]:
    """Return ``(total_threadgroups, (gx, gy, gz))`` for the B0 grid.

    Grid is ``(batch*nchunks, nheads, 1)`` — same family as the forward F0.
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_precompute_bwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    nchunks = seqlen // chunk_size
    return batch * nchunks * nheads, (batch * nchunks, nheads, 1)


def chunk_precompute_bwd_metal_prim(
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
    """Build the B0 ``mamba3_chunk_precompute_bwd`` Metal ``@T.prim_func``.

    Inputs (forward cache + the B1/B2 partials):
      dstates       : (batch, nchunks, nheads, headdim, dstate) grad of summary_states (B1)
      dinp_diag     : (batch, seqlen, nheads, headdim, dstate)  diag input-outer grad (B2)
      dA_cumsum_y   : (batch, nheads, nchunks, chunk)           Y-path dA grad (B2)
      dA_cumsum_tail: (batch, nheads, nchunks, chunk)           chunk-tail dA grad (B1)
      dA_cumsum     : (batch, nheads, nchunks, chunk)           forward (F0)
      x             : (batch, seqlen, nheads, headdim)
      B             : (batch, seqlen, ngroups, dstate)
      dt            : (batch, nheads, nchunks, chunk)
      A             : (nheads,)
    Outputs (the final input grads):
      dx            : (batch, seqlen, nheads, headdim)   accumulated x-inp grad (NB: B2 wrote D-skip dx separately)
      dB            : (batch, seqlen, ngroups, dstate)
      dlog_decay    : (batch, seqlen, nheads)            = dA*dt path (proto dlog_decay)
      ddt           : (batch, seqlen, nheads)            dt grad (inp + Lmat dt-scale)

    Transpose of the precompute. Folds the per-chunk-state ``decay_states``
    transpose (``dstates`` -> ``dinp_states`` + ``dA`` scatter), assembles
    ``dinp = dinp_diag + dinp_states``, reduces ``dinp[s,p,n]`` to ``dx_inp[s,p] =
    sum_n dinp*B[s,n]`` and ``dB[s,n] = sum_p dinp*x[s,p]``, and assembles the
    full ``dA_cumsum`` (Y-path + chunk-tail + decay_states) into ``da`` via the
    reverse-cumsum VJP (cumsum adjoint) -> ``dlog_decay`` and ``ddt``.
    """
    import tilelang
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_precompute_bwd_metal_prim: seqlen ({seqlen}) must be "
            f"divisible by chunk_size ({chunk_size}); no padding (RULE #1)"
        )

    dtype = T.float16
    accum_dtype = T.float32
    nchunks = seqlen // chunk_size
    heads_per_group = nheads // ngroups
    L = chunk_size
    p = _LOG2E

    @T.prim_func
    def main(
        dstates: T.Tensor((batch, nchunks, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dinp_diag: T.Tensor((batch, seqlen, nheads, headdim, dstate), accum_dtype),  # type: ignore
        dA_cumsum_y: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),  # type: ignore
        dA_cumsum_tail: T.Tensor((batch, nheads, nchunks, chunk_size), accum_dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        B: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        A: T.Tensor((nheads), dtype),  # type: ignore
        dx: T.Tensor((batch, seqlen, nheads, headdim), accum_dtype),  # type: ignore
        dB: T.Tensor((batch, seqlen, nheads, dstate), accum_dtype),  # type: ignore
        dlog_decay: T.Tensor((batch, seqlen, nheads), accum_dtype),  # type: ignore
        ddt: T.Tensor((batch, seqlen, nheads), accum_dtype),  # type: ignore
    ):
        with T.Kernel(batch * nchunks, nheads, threads=threads) as (bx, by):
            batch_idx = bx % batch
            chunk_idx = bx // batch
            head_idx = by
            group_idx = head_idx // heads_per_group
            base = chunk_idx * chunk_size

            dacs = T.alloc_shared((L,), accum_dtype)
            dA_full = T.alloc_shared((L,), accum_dtype)  # total dA_cumsum grad per l
            ddt_inp_sh = T.alloc_shared((L,), accum_dtype)  # inp-path dt grad per l

            for l in T.Parallel(L):
                dacs[l] = T.Cast(accum_dtype, dA_cumsum[batch_idx, head_idx, chunk_idx, l])
                ddt_inp_sh[l] = T.Cast(accum_dtype, 0)
            T.sync_threads()

            # ---- decay_states transpose: dstates -> dinp_states + dA scatter ----
            #   decay_states[l] = exp(dacs[L-1]-dacs[l])
            #   states[p,n]     = sum_l decay_states[l]*dt[l]*x[l,p]*B[l,n]   (forward F0)
            #   dinp_states[l,p,n] = dstates[p,n]*decay_states[l]*dt[l] (folded into dx/dB)
            #   ddecay_states[l]   = sum_{p,n} dstates[p,n]*dt[l]*x[l,p]*B[l,n]
            #   t = ddecay_states*decay_states ; dA[l] += -t ; dA[L-1] += sum_l t
            # Plus the Y-path + chunk-tail dA grads.
            # We accumulate dA_full[l] = dA_cumsum_y[l] + (-t[l]); dA_full[L-1] += sum t + tail.
            t_sum = T.alloc_shared((1,), accum_dtype)
            if T.get_thread_binding(0) == 0:
                t_sum[0] = T.Cast(accum_dtype, 0)
            T.sync_threads()

            for l in T.Parallel(L):
                decay_l = T.exp2((dacs[L - 1] - dacs[l]) * p)
                dt_l = T.Cast(accum_dtype, dt[batch_idx, head_idx, chunk_idx, l])
                # ddecay_states[l] = sum_{p,n} dstates[p,n]*dt_l*x[l,p]*B[l,n]
                dds = T.alloc_local((1,), accum_dtype)
                dds[0] = T.Cast(accum_dtype, 0)
                for pn in T.serial(headdim * dstate):
                    pp = pn // dstate
                    nn = pn % dstate
                    ds = dstates[batch_idx, chunk_idx, head_idx, pp, nn]
                    x_v = T.Cast(accum_dtype, x[batch_idx, base + l, head_idx, pp])
                    b_v = T.Cast(accum_dtype, B[batch_idx, base + l, group_idx, nn])
                    dds[0] = dds[0] + ds * dt_l * x_v * b_v
                tt = dds[0] * decay_l
                dA_full[l] = dA_cumsum_y[batch_idx, head_idx, chunk_idx, l] - tt
                T.atomic_add(t_sum[0], tt)
            T.sync_threads()
            if T.get_thread_binding(0) == 0:
                dA_full[L - 1] = (
                    dA_full[L - 1]
                    + t_sum[0]
                    + dA_cumsum_tail[batch_idx, head_idx, chunk_idx, L - 1]
                )
            T.sync_threads()

            # ---- assemble dinp = dinp_diag + dinp_states ; reduce to dx_inp, dB, ddt
            #   dinp[l,p,n] = grad wrt inp = dt*x⊗B (proto native). Two sources:
            #     dinp_diag (from B2 Y_diag, NO dt baked in)
            #     dinp_states[l,p,n] = dstates[p,n]*decay_states[l]  (summary chain;
            #        forward summary = sum_l decay*dt*x*B, so grad wrt inp drops dt)
            #   Then chain inp = dt*(x⊗B):
            #     dx_inp[l,p] = sum_n dinp[l,p,n]*dt[l]*B[l,n]
            #     dB[l,n]     = sum_p dinp[l,p,n]*dt[l]*x[l,p]
            #     ddt_inp[l]  = sum_{p,n} dinp[l,p,n]*x[l,p]*B[l,n]
            for lp in T.Parallel(L * headdim):
                ll = lp // headdim
                pp = lp % headdim
                decay_l = T.exp2((dacs[L - 1] - dacs[ll]) * p)
                dt_l = T.Cast(accum_dtype, dt[batch_idx, head_idx, chunk_idx, ll])
                x_lp = T.Cast(accum_dtype, x[batch_idx, base + ll, head_idx, pp])
                acc = T.alloc_local((1,), accum_dtype)      # sum_n dinp*B  (no dt yet)
                acc[0] = T.Cast(accum_dtype, 0)
                for nn in T.serial(dstate):
                    ds = dstates[batch_idx, chunk_idx, head_idx, pp, nn]
                    dinp_v = (
                        dinp_diag[batch_idx, base + ll, head_idx, pp, nn]
                        + ds * decay_l
                    )
                    b_v = T.Cast(accum_dtype, B[batch_idx, base + ll, group_idx, nn])
                    acc[0] = acc[0] + dinp_v * b_v
                # dx_inp = dt * sum_n dinp*B ; ddt_inp += x * sum_n dinp*B
                T.atomic_add(dx[batch_idx, base + ll, head_idx, pp], acc[0] * dt_l)
                T.atomic_add(ddt_inp_sh[ll], acc[0] * x_lp)
            T.sync_threads()

            for ln in T.Parallel(L * dstate):
                ll = ln // dstate
                nn = ln % dstate
                decay_l = T.exp2((dacs[L - 1] - dacs[ll]) * p)
                dt_l = T.Cast(accum_dtype, dt[batch_idx, head_idx, chunk_idx, ll])
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for pp in T.serial(headdim):
                    ds = dstates[batch_idx, chunk_idx, head_idx, pp, nn]
                    dinp_v = (
                        dinp_diag[batch_idx, base + ll, head_idx, pp, nn]
                        + ds * decay_l
                    )
                    x_v = T.Cast(accum_dtype, x[batch_idx, base + ll, head_idx, pp])
                    acc[0] = acc[0] + dinp_v * x_v
                # dB[l,n] = dt * sum_p dinp*x ; accumulate over heads in a group
                T.atomic_add(dB[batch_idx, base + ll, head_idx, nn], acc[0] * dt_l)
            T.sync_threads()

            # ---- A_cumsum = cumsum(a) over l -> da via reverse-cumsum (adjoint) ----
            #   da[l] = sum_{k>=l} dA_full[k]   (cumsum adjoint = reverse-cumsum)
            #   a[l]  = A[h]*dt[l] -> dlog_decay[l] = da[l] (this is dA*dt = d log_decay)
            #   dA contribution: da[l]*dt[l] ; ddt contribution: da[l]*A[h]
            if T.get_thread_binding(0) == 0:
                a_h = T.Cast(accum_dtype, A[head_idx])
                # explicit reverse cumsum into dlog_decay (cumsum adjoint):
                #   da[l] = sum_{k>=l} dA_full[k]
                racc = T.alloc_local((1,), accum_dtype)
                racc[0] = T.Cast(accum_dtype, 0)
                for l in T.serial(L):
                    ll = L - 1 - l
                    racc[0] = racc[0] + dA_full[ll]
                    s = base + ll
                    # a[l] = A[h]*dt[l]; dlog_decay = da (=dA*dt path, proto).
                    dlog_decay[batch_idx, s, head_idx] = racc[0]
                    # ddt = decay-chain path (da*A) + inp path (sum dinp*x*B).
                    ddt[batch_idx, s, head_idx] = racc[0] * a_h + ddt_inp_sh[ll]

    return main


def build_chunk_precompute_bwd_metal(
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
    """Compile the B0 precompute backward kernel to a ``JITKernel``.

    ``target`` selects Metal (default) or CUDA (``"cuda"`` / sm_121). Outputs
    (dx, dB, dlog_decay, ddt) are the trailing 4 params; pass PRE-ZEROED
    contiguous fp32 buffers positionally (dx is accumulated into — B2 wrote the
    D-skip path first). RULE #1: compile failures propagate (no serial fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang.mamba3_chunked_scan_core import (
        _resolve_chunked_compile_target,
    )

    prim = chunk_precompute_bwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    resolved_target = _resolve_chunked_compile_target(target)
    # CUDA (sm_121): mirror the forward F2 compile-site EXACTLY — thread the same
    # two pass_configs. The B0 prim BODY has NO T.gemm / TMA-eligible copy
    # (grep-confirmed: zero T.gemm / shared.dyn / make_swizzled_layout; all
    # alloc_shared are plain default scope), so there is no tensormap descriptor
    # to mis-align; the pass_configs are a no-op-or-safer escape hatch keeping the
    # CUDA compile path byte-identical to the forward. The Metal branch is
    # UNCHANGED (no pass_configs). RULE #1: explicit per-target codegen choice,
    # never a silent fallback.
    kind = str(getattr(getattr(resolved_target, "kind", None), "name", "")).lower()
    if "cuda" in kind:
        pass_configs = {
            "tl.disable_tma_lower": True,
            "tl.disable_warp_specialized": True,
        }
        return tilelang.compile(
            prim,
            out_idx=[9, 10, 11, 12],
            target=resolved_target,
            pass_configs=pass_configs,
        )
    return tilelang.compile(
        prim,
        out_idx=[9, 10, 11, 12],
        target=resolved_target,
    )
