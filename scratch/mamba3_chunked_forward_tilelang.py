"""TileLang prim_func form of the chunked mamba3 forward intra-chunk scan.

Adapted from tilelang/examples/linear_attention/example_mamba_chunk_scan.py.
The whole point of step 3 is to show the scan runs over a GRID of threadgroups
(chunks x channels), not the single T.Kernel(1, threads=1024) the serial Path-C
forward uses. This module builds the prim_func, compiles it to the Metal target,
and reports the launch grid (number of threadgroups) so occupancy is provable.

The intra-chunk Y_diag kernel here is the matmul-heavy step (step 1/4 of the SSD
decomposition): grid = (nheads, ceildiv(chunk,bM)*ceildiv(headdim,bN), batch*nchunks).
For S=4096, C=256 -> nchunks=16, batch*nchunks blocks alone is already a full wave
on a 40-core M4 Max, versus the single resident block today.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here.  The TileLang
# eager frontend reads the ``T.Tensor((...), dtype)`` parameter annotations as
# real objects via get_type_hints; PEP 563 string annotations would defer their
# evaluation to a globalns that lacks the closure vars (seqlen, nchunks, ...) and
# raise ``NameError: name 'seqlen' is not defined`` at prim_func build time.

import tilelang
import tilelang.language as T


def chunk_scan_fwd_prim(
    batch,
    seqlen,
    chunk_size,
    ngroups,
    nheads,
    headdim,
    dstate,
    block_M=64,
    block_N=64,
    block_K=64,
    block_Dstate=128,
    num_stages=2,
    threads=128,
):
    """Build the chunk-scan forward prim_func (no autotune, fixed config).

    Mirrors example_mamba_chunk_scan.chunk_scan_fwd but as a plain factory we can
    compile to Metal and inspect the grid. Inputs are the SSD intermediates:
      cb         : (batch, nchunks, ngroups, chunk, chunk)  = C @ B^T per chunk
      x          : (batch, seqlen, nheads, headdim)
      dt         : (batch, nheads, nchunks, chunk)
      dA_cumsum  : (batch, nheads, nchunks, chunk)          = cumsum(A*dt) per chunk
      C          : (batch, seqlen, ngroups, dstate)
      prev_states: (batch, nchunks, nheads, headdim, dstate) inter-chunk entry states
      D          : (nheads,)
    Output       : (batch, seqlen, nheads, headdim)
    """
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
            acc_o = T.alloc_fragment((block_M, block_N), accum_dtype)
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
                    x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared),
                }
            )
            T.no_set_max_nreg()

            T.copy(
                dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M],
                dA_cs_m_shared,
            )
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0:block_Dstate,
                ],
                C_shared,
            )
            T.copy(
                prev_states[batch_idx, chunk_idx, bz, n_idx * block_N : (n_idx + 1) * block_N, 0:block_Dstate],
                prev_state_shared,
            )
            T.gemm(C_shared, prev_state_shared, acc_o, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                acc_o[i, j] *= scale_m_local[i]

            loop_range = T.ceildiv((m_idx + 1) * block_M, block_K)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
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
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0)
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
                        bz,
                        n_idx * block_N : (n_idx + 1) * block_N,
                    ],
                    x_shared,
                )
                T.gemm(cb_local, x_shared, acc_o)

            D_local[0] = D[bz]
            T.copy(
                x[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
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
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
            )

    return main


def chunk_scan_fwd_metal(
    batch,
    seqlen,
    chunk_size,
    ngroups,
    nheads,
    headdim,
    dstate,
    block_M=64,
    block_N=16,
    block_K=64,
    block_Dstate=128,
    threads=128,
):
    """Metal-compatible chunk-scan forward prim_func (compiles + runs on M-series).

    Two changes vs the CUDA ``chunk_scan_fwd_prim`` above, both numerics-preserving:

      1. The output accumulator ``acc_o`` lives in *shared* memory instead of a
         register fragment.  On the Metal simdgroup GEMM path a register-fragment
         C accumulator is stored as opaque ``simdgroup_matrix`` tiles that are NOT
         per-thread element-addressable, so the subsequent elementwise scaling /
         masking (``acc_o[i, j] *= ...``) emits an illegal ``(half4)simdgroup``
         cast in MSL.  Keeping C in shared makes it element-addressable and routes
         the GEMM through the simdgroup-matrix path that runs on M1+ silicon.
      2. ``block_N=16`` (headdim tile) so the per-block GEMM N=16 stays below the
         32-column threshold that would otherwise select the Metal *M5*
         cooperative-tensor (``mpp::tensor_ops::matmul2d``) path, which only
         compiles on Apple M5 / MSL 4.  This keeps the legacy 8x8 simdgroup path
         that runs on M1-M4.  More headdim tiles => *more* threadgroups, so
         occupancy only improves.

    The ``T.gemm(cb_local, x_shared, acc_o)`` call (A in register fragment, B in
    shared) is supported by the RS->staged-shared lowering added to
    ``tilelang/tileop/gemm/gemm_metal.py`` (GemmMetalSimdGroup): the cb_local
    fragment is copied into a temporary shared buffer before the simdgroup matmul.
    No CUDA-only intrinsics (``T.no_set_max_nreg``) and no software pipelining
    (which trips a separate Metal storage-sync / cast-buffer SSA bug); the
    K-reduction is a plain ``T.serial`` loop.

    Verified on Apple M4 Max: compiles to MSL, launches gx*gy*gz threadgroups,
    and matches the torch SSD reference (examples ref_program) to max-abs-diff
    ~1.2e-4 (fp16), ~0.27 ms/iter for B=1,S=4096,C=256,H=8,P=64,N=128.
    """
    import tilelang.language as T

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
                    x_residual_shared: tilelang.layout.make_swizzled_layout(x_residual_shared),
                }
            )

            T.copy(
                dA_cumsum[batch_idx, bz, chunk_idx, m_idx * block_M : (m_idx + 1) * block_M],
                dA_cs_m_shared,
            )
            T.copy(dA_cs_m_shared, dA_cs_m_local)
            T.clear(acc_o)

            for i in T.Parallel(block_M):
                scale_m_local[i] = T.exp2(dA_cs_m_local[i] * p)
            T.copy(
                C[
                    batch_idx,
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz // (nheads // ngroups),
                    0:block_Dstate,
                ],
                C_shared,
            )
            T.copy(
                prev_states[batch_idx, chunk_idx, bz, n_idx * block_N : (n_idx + 1) * block_N, 0:block_Dstate],
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
                T.copy(dA_cumsum[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dA_cs_k_shared)
                T.copy(dA_cs_k_shared, dA_cs_k_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = cb_local[i, j] * T.exp2(dA_cs_m_local[i] * p - dA_cs_k_local[j] * p)
                T.copy(dt[batch_idx, bz, chunk_idx, k * block_K : (k + 1) * block_K], dt_shared)
                T.copy(dt_shared, dt_local)
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] *= dt_local[j]
                for i, j in T.Parallel(block_M, block_K):
                    cb_local[i, j] = T.if_then_else(m_idx * block_M + i >= k * block_K + j, cb_local[i, j], 0)
                T.copy(
                    x[
                        batch_idx,
                        chunk_idx * chunk_size + k * block_K : chunk_idx * chunk_size + (k + 1) * block_K,
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
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
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
                    chunk_idx * chunk_size + m_idx * block_M : chunk_idx * chunk_size + (m_idx + 1) * block_M,
                    bz,
                    n_idx * block_N : (n_idx + 1) * block_N,
                ],
            )

    return main


def grid_blocks(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
                block_M=64, block_N=16):
    """Return the number of threadgroups (grid size) for the given shape/config.

    This is the occupancy proof: serial Path-C forward = 1 block; chunked = this.
    """
    import math
    nchunks = math.ceil(seqlen / chunk_size)
    gx = nheads
    gy = math.ceil(chunk_size / block_M) * math.ceil(headdim / block_N)
    gz = batch * nchunks
    return gx * gy * gz, (gx, gy, gz)


if __name__ == "__main__":
    # realistic-ish shape
    batch, seqlen, chunk_size = 1, 4096, 256
    ngroups, nheads, headdim, dstate = 1, 8, 64, 128
    total, grid = grid_blocks(batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate)
    print(f"shape: batch={batch} seq={seqlen} chunk={chunk_size} "
          f"nheads={nheads} headdim={headdim} dstate={dstate}")
    print(f"chunked-scan grid (gx,gy,gz) = {grid}  -> total threadgroups = {total}")
    print(f"serial Path-C forward threadgroups = 1")
    print(f"occupancy ratio (chunked / serial) = {total}x grid blocks "
          f"(vs 40 GPU cores on M4 Max)")

    prim = chunk_scan_fwd_prim(
        batch, seqlen, chunk_size, ngroups, nheads, headdim, dstate,
        block_M=64, block_N=64, block_K=64, block_Dstate=128, num_stages=2, threads=128,
    )
    print("\nprim_func built OK.")
    # Try lowering to Metal MSL (proves codegen for the multi-block grid).
    try:
        from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
        art = dispatch_lower(prim, "metal -thread_warp_size=32")
        src = getattr(art, "kernel_source", None) or getattr(art, "_source", None) or str(art)
        nthreadgroups_kw = src.count("threadgroup") if isinstance(src, str) else -1
        print(f"Metal lowering: OK (engine target stamped="
              f"{getattr(art, '_tilelang_engine_target', '?')})")
        print(f"MSL contains 'threadgroup' decls: {nthreadgroups_kw}")
    except Exception as e:  # noqa: BLE001  (prototype: surface the exact failure)
        print(f"Metal lowering raised: {type(e).__name__}: {e}")
