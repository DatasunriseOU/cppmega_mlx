"""PROVEN chunked-parallel forward scan-core for the Path-C mamba3 forward.

This is the productionized SSD 4-step chunked forward intra/inter-chunk
scan-core, promoted from the validated prototype on branch
``mamba3-chunked-forward`` (commit 351ebee). It replaces the O(S) serial
single-threadgroup forward scan in the Path-C emitter
(``_append_row_phased_mamba3_body``) with a GRID over
``(nheads, ceildiv(chunk,bM)*ceildiv(headdim,bN), batch*nchunks)`` — i.e. many
threadgroups instead of one.

Validation (M4 Max, this worktree, tilelang 5952468a):
  * MLX numerical contract (``mamba3_chunked_forward`` below, the proto algebra)
    matches OUR serial ``_chunked_mamba3_diagonal_scan`` to out max|abs|≈1e-5,
    h_last≈3e-6 in fp32, no NaN (the production accumulation dtype).
  * The TileLang ``chunk_scan_fwd_metal`` prim_func COMPILES to MSL and RUNS on
    Metal: max|abs diff| vs the torch SSD reference ≈ 2.4e-4 (S=256) / 4.9e-4
    (S=4096) fp16, no NaN.
  * Threadgroup count for S=4096,C=256,H=8,P=64,N=128 is **2048** (grid
    ``(8,16,16)``) vs the serial Path-C forward's **1**; ~37.6x forward speedup
    (from the bench sweep) — the Amdahl signature of removing the O(S) serial
    dependency.

RULE #1 (NO FALLBACK): callers MUST select this path explicitly. On a
chunking/parity/codegen failure the helpers RAISE with where+what — they never
silently degrade to the serial scan. The per-target codegen choice (grid vs the
serial launcher) is a legitimate gate, not a silent fallback.

Calling convention for the compiled Metal kernel (verified):
  * Pass a PRE-ZEROED contiguous fp16 output buffer POSITIONALLY as the 8th arg
    (the tilelang torch-mps adapter does not allocate/return via ``out_idx``).
  * All inputs contiguous fp16; ``dA_cumsum`` is the in-chunk cumsum of A*dt.
  * ``torch.mps.synchronize()`` after dispatch before reading the output.

The original prototype/handoff lives at
``scratch/mamba3_chunked_forward_tilelang.py`` /
``scratch/mamba3_chunked_forward_proto.py`` /
``scratch/MAMBA3-CHUNKED-FORWARD-RESULTS.md``.
"""

import math
from typing import Any

# CRITICAL: do NOT add ``from __future__ import annotations`` to this module.
# The TileLang eager frontend reads the ``T.Tensor((...), dtype)`` prim_func
# parameter annotations as real objects via ``get_type_hints``. PEP 563 string
# annotations would defer their evaluation to a globalns that lacks the factory
# closure vars (``seqlen``, ``nchunks``, ...) and raise
# ``NameError: name 'seqlen' is not defined`` at prim_func build time.

__all__ = [
    "chunk_scan_fwd_metal_prim",
    "chunk_scan_fwd_grid",
    "compile_chunk_scan_fwd_metal",
    "build_chunk_scan_combine_metal",
    "MAMBA3_CHUNKED_FWD_BLOCK_M",
    "MAMBA3_CHUNKED_FWD_BLOCK_N",
    "MAMBA3_CHUNKED_FWD_BLOCK_K",
    "MAMBA3_CHUNKED_FWD_BLOCK_DSTATE",
    "MAMBA3_CHUNKED_FWD_THREADS",
]

# Stable op-node name for the Path-C F2 scan+combine segment (design doc
# ``docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md`` §2/§3.2). The Path-C brick-schedule
# descriptor registered under this name DELEGATES its kernel build to
# ``chunk_scan_fwd_metal_prim`` below (the proven, Metal-validated SSD chunked
# forward scan+combine core). This single binding is the one clear path from the
# descriptor/emitter and the isolation harness to the scan core (RULE #1: no
# second/fallback codegen path).
MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME = "mamba3_chunk_scan_combine"
__all__.append("MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME")

# Metal-validated tile config. block_N=16 keeps the GEMM N below the 32-column
# threshold so the legacy 8x8 simdgroup path (M1-M4) is selected instead of the
# M5-only cooperative-tensor path. More headdim tiles => more threadgroups, so
# occupancy only improves.
MAMBA3_CHUNKED_FWD_BLOCK_M = 64
MAMBA3_CHUNKED_FWD_BLOCK_N = 16
MAMBA3_CHUNKED_FWD_BLOCK_K = 64
MAMBA3_CHUNKED_FWD_BLOCK_DSTATE = 128
MAMBA3_CHUNKED_FWD_THREADS = 128


def chunk_scan_fwd_grid(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    *,
    block_M: int = MAMBA3_CHUNKED_FWD_BLOCK_M,
    block_N: int = MAMBA3_CHUNKED_FWD_BLOCK_N,
) -> tuple[int, tuple[int, int, int]]:
    """Return ``(total_threadgroups, (gx, gy, gz))`` for the chunked scan grid.

    This is the occupancy proof vs the serial Path-C forward (1 threadgroup).
    For S=4096,C=256,H=8,P=64,N=128 this is ``(8, 16, 16) -> 2048``.
    """
    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_scan_fwd_grid: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback is permitted (RULE #1)"
        )
    nchunks = seqlen // chunk_size
    gx = nheads
    gy = math.ceil(chunk_size / block_M) * math.ceil(headdim / block_N)
    gz = batch * nchunks
    return gx * gy * gz, (gx, gy, gz)


def chunk_scan_fwd_metal_prim(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    *,
    block_M: int = MAMBA3_CHUNKED_FWD_BLOCK_M,
    block_N: int = MAMBA3_CHUNKED_FWD_BLOCK_N,
    block_K: int = MAMBA3_CHUNKED_FWD_BLOCK_K,
    block_Dstate: int = MAMBA3_CHUNKED_FWD_BLOCK_DSTATE,
    threads: int = MAMBA3_CHUNKED_FWD_THREADS,
) -> Any:
    """Build the Metal-compatible chunk-scan forward ``@T.prim_func``.

    Inputs (SSD intermediates produced by the Path-C precompute stage):
      cb         : (batch, nchunks, ngroups, chunk, chunk)  = C @ B^T per chunk
      x          : (batch, seqlen, nheads, headdim)
      dt         : (batch, nheads, nchunks, chunk)
      dA_cumsum  : (batch, nheads, nchunks, chunk)          = cumsum(A*dt) per chunk
      C          : (batch, seqlen, ngroups, dstate)
      prev_states: (batch, nchunks, nheads, headdim, dstate) inter-chunk entry states
      D          : (nheads,)
    Output       : (batch, seqlen, nheads, headdim)

    Two Metal-codegen-preserving deltas vs the CUDA upstream form (both
    numerics-preserving, see the prototype docstring): C accumulator lives in
    *shared* (element-addressable for the elementwise scale/mask) and the
    K-reduction is a plain ``T.serial`` loop (no software pipelining, which
    trips a Metal storage-sync / cast-buffer SSA bug). ``T.gemm(cb_local,
    x_shared, acc_o)`` (A register-fragment, B shared) routes through the
    RS->staged-shared lowering in ``tilelang/tileop/gemm/gemm_metal.py``.
    """
    import tilelang
    import tilelang.language as T

    if seqlen % chunk_size != 0:
        raise ValueError(
            f"chunk_scan_fwd_metal_prim: seqlen ({seqlen}) must be divisible by "
            f"chunk_size ({chunk_size}); no padding fallback (RULE #1)"
        )
    if nheads % ngroups != 0:
        raise ValueError(
            f"chunk_scan_fwd_metal_prim: nheads ({nheads}) must be divisible by "
            f"ngroups ({ngroups})"
        )

    dtype = T.float16
    accum_dtype = T.float32
    nchunks = T.ceildiv(seqlen, chunk_size)
    p = 1.44269504  # 1/ln(2): exp2(x*p) == exp(x)

    @T.prim_func
    def main(
        cb: T.Tensor((batch, nchunks, ngroups, chunk_size, chunk_size), dtype),  # type: ignore
        x: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
        dt: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        dA_cumsum: T.Tensor((batch, nheads, nchunks, chunk_size), dtype),  # type: ignore
        C: T.Tensor((batch, seqlen, ngroups, dstate), dtype),  # type: ignore
        prev_states: T.Tensor((batch, nchunks, nheads, headdim, dstate), dtype),  # type: ignore
        D: T.Tensor((nheads), dtype),  # type: ignore
        Output: T.Tensor((batch, seqlen, nheads, headdim), dtype),  # type: ignore
    ):
        with T.Kernel(
            nheads,
            T.ceildiv(chunk_size, block_M) * T.ceildiv(headdim, block_N),
            batch * nchunks,
            threads=threads,
        ) as (bz, bx, by):
            acc_o = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared")
            acc_o_shared = T.alloc_shared((block_M, block_N), dtype)
            cb_shared = T.alloc_shared((block_M, block_K), dtype)
            cb_local = T.alloc_fragment((block_M, block_K), dtype)
            dA_cs_k_shared = T.alloc_shared((block_K), dtype)
            dA_cs_k_local = T.alloc_fragment((block_K), accum_dtype)
            dA_cs_m_local = T.alloc_fragment((block_M), accum_dtype)
            dt_shared = T.alloc_shared((block_K), dtype)
            dt_local = T.alloc_fragment((block_K), accum_dtype)
            x_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.dyn")
            dA_cs_m_shared = T.alloc_shared((block_M), dtype)
            scale_m_local = T.alloc_fragment((block_M), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_Dstate), dtype)
            prev_state_shared = T.alloc_shared((block_N, block_Dstate), dtype)
            D_local = T.alloc_fragment((1), accum_dtype)
            x_residual_shared = T.alloc_shared((block_M, block_N), dtype)
            x_residual_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            batch_idx = by % batch
            chunk_idx = by // batch
            m_idx = bx // T.ceildiv(headdim, block_N)
            n_idx = bx % T.ceildiv(headdim, block_N)

            T.annotate_layout(
                {
                    cb_shared: tilelang.layout.make_swizzled_layout(cb_shared),
                    x_residual_shared: tilelang.layout.make_swizzled_layout(
                        x_residual_shared
                    ),
                }
            )

            T.copy(
                dA_cumsum[
                    batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M
                ],
                dA_cs_m_shared,
            )
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_M : chunk_idx * chunk_size
                    + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0:block_Dstate,
                ],
                C_shared,
            )
            T.copy(
                prev_states[
                    batch_idx,
                    chunk_idx,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                    0:block_Dstate,
                ],
                prev_state_shared,
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)
            for k in T.serial(loop_range):
                T.copy(
                    cb[
                        batch_idx,
                        chunk_idx,
                        bz // (nheads // ngroups),
                        m_idx * block_M : (m_idx + 1) * block_M,
                        k * block_K : (k + 1) * block_K,
                    ],
                    cb_shared,
                )
                T.copy(cb_shared, cb_local)
                T.copy(
                    dA_cumsum[
                        batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K
                    ],
                    dA_cs_k_shared,
                )
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(
                        dA_cs_m_local[i] * p - dA_cs_k_local[j] * p
                    )
                T.copy(
                    dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K],
                    dt_shared,
                )
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(
                        m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0
                    )
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size
                        + k * block_K : chunk_idx * chunk_size
                        + (k + 1) * block_K,
                        bz,
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    x_shared,
                )
                # A in register fragment (cb_local), B in shared (x_shared), C in
                # shared (acc_o): RS->staged-shared Metal GEMM (gemm_metal.py fix).
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_M : chunk_idx * chunk_size
                    + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
                x_residual_shared,
            )
            T.copy(x_residual_shared, x_residual_local)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] += x_residual_local[i, j] * D_local[0]

            T.copy(acc_o, acc_o_shared)
            T.copy(
                acc_o_shared,
                Output[
                    batch_idx,
                    chunk_idx * chunk_size
                    + m_idx * block_M : chunk_idx * chunk_size
                    + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
            )

    return main


def compile_chunk_scan_fwd_metal(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    **kwargs: Any,
) -> Any:
    """Compile the chunked scan-core to a Metal ``JITKernel``.

    Uses the repo's ``_as_metal_target`` builder (the bare ``"metal
    -thread_warp_size=32"`` string is rejected by this tilelang's
    ``determine_target`` allowlist; the Target object bypasses it). Returns a
    callable ``JITKernel``; pass a PRE-ZEROED contiguous fp16 output buffer
    positionally as the 8th argument and ``torch.mps.synchronize()`` after.

    RULE #1: compile failures propagate (no swallow / no serial fallback).
    """
    import tilelang

    from cppmega_mlx.nn._tilelang import _msl_transform

    prim = chunk_scan_fwd_metal_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
    kernel = tilelang.compile(
        prim,
        out_idx=[7],
        target=_msl_transform._as_metal_target("metal -thread_warp_size=32"),
    )
    return kernel


def build_chunk_scan_combine_metal(
    batch: int,
    seqlen: int,
    chunk_size: int,
    ngroups: int,
    nheads: int,
    headdim: int,
    dstate: int,
    **kwargs: Any,
) -> Any:
    """Build the Path-C F2 ``mamba3_chunk_scan_combine`` Metal kernel.

    This is the SINGLE named delegation that the Path-C brick-schedule
    descriptor registered under :data:`MAMBA3_CHUNK_SCAN_COMBINE_OP_NAME` and the
    Stage-1 isolation parity harness both call. It compiles the proven
    :func:`chunk_scan_fwd_metal_prim` scan+combine core (inputs
    ``cb, x, dt, dA_cumsum, C, prev_states, D`` -> ``Output``) — there is exactly
    ONE codegen path; on any compile/shape failure the underlying builder RAISES
    with where+what and there is NO serial fallback (RULE #1).

    Returns a callable Metal ``JITKernel``; pass a PRE-ZEROED contiguous fp16
    output buffer positionally as the 8th argument and ``torch.mps.synchronize()``
    after dispatch.
    """
    return compile_chunk_scan_fwd_metal(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate, **kwargs
    )
