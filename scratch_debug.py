import math
from cppmega_mlx.nn._tilelang.dsa_splitk_indexer_loss import make_dsa_splitk_stage1_kernel
import tilelang
import torch

AB, AH, AD = 1, 2, 32
ASq, Sk = 64, 128
softmax_scale = 1.0 / math.sqrt(AD)
in_dtype = "float16"

# Let's override the make_dsa_splitk_stage1_kernel to add the dummy read and see!
import tilelang.language as T
from typing import Any, cast

def make_dsa_splitk_stage1_kernel_debug(
    *,
    AB: int,
    AH: int,
    AD: int,
    Sk: int,
    ASq: int,
    sparse_loss: bool,
    softmax_scale: float,
    in_dtype: str = "float16",
    BLOCK_SQ: int = 128,
    BLOCK_SK: int = 128,
    BLOCK_D: int = 64,
    threads: int = 256,
    num_stages: int = 3,
    compute_index_path: bool = True,
) -> Any:
    NUM_SQ_BLOCKS = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    SK_TILES = (Sk + BLOCK_SK - 1) // BLOCK_SK
    SCALE = float(softmax_scale)
    SPARSE = bool(sparse_loss)
    COMPUTE_INDEX = bool(compute_index_path)

    @T.prim_func
    def dsa_stage1(
        Q: T.Tensor((ASq, AB, AH, AD), in_dtype),
        K: T.Tensor((Sk, AB, AH, AD), in_dtype),
        IndexScores: T.Tensor((AB, ASq, Sk), "float32"),
        IndexMask: T.Tensor((AB, ASq, Sk), "float32"),
        M: T.Tensor((AB, AH, ASq), "float32"),
        D: T.Tensor((AB, AH, ASq), "float32"),
        M1: T.Tensor((AB, ASq), "float32"),
        D1: T.Tensor((AB, ASq), "float32"),
    ):
        with T.Kernel(AB, NUM_SQ_BLOCKS, AH, threads=threads) as (b, sq_block_id, h):
            Q_s = T.alloc_shared((BLOCK_SQ, BLOCK_D), in_dtype)
            K_s = T.alloc_shared((BLOCK_D, BLOCK_SK), in_dtype)
            dummy_pad = T.alloc_shared((3096,), "float16")
            Q_full = T.alloc_shared((BLOCK_SQ, AD), in_dtype)
            scores_f = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            row_max_local = T.alloc_shared((BLOCK_SQ,), "float32")
            row_sum_local = T.alloc_shared((BLOCK_SQ,), "float32")
            m_i = T.alloc_shared((BLOCK_SQ,), "float32")
            d_i = T.alloc_shared((BLOCK_SQ,), "float32")
            m_i_prev = T.alloc_shared((BLOCK_SQ,), "float32")
            if COMPUTE_INDEX:
                m1_i = T.alloc_shared((BLOCK_SQ,), "float32")
                d1_i = T.alloc_shared((BLOCK_SQ,), "float32")
                m1_i_prev = T.alloc_shared((BLOCK_SQ,), "float32")
                idx_scores_f = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), "float32")
            else:
                m1_i = T.alloc_shared((1,), "float32")
                d1_i = T.alloc_shared((1,), "float32")
                m1_i_prev = T.alloc_shared((1,), "float32")
                idx_scores_f = T.alloc_fragment((1, 1), "float32")

            if COMPUTE_INDEX:
                for i in T.Parallel(BLOCK_SQ):
                    m1_i[i] = T.cast(-3.4028234663852886e38, "float32")
                    d1_i[i] = T.cast(0, "float32")

            for i in T.Parallel(BLOCK_SQ):
                m_i[i] = T.cast(-3.4028234663852886e38, "float32")
                d_i[i] = T.cast(0, "float32")

            for i in T.Parallel(1):
                dummy_pad[i] = T.cast(0, "float16")

            # Force compiler to preserve IndexMask argument in compiled C/CUDA signature
            if IndexMask[b, 0, 0] > 1e10:
                m_i[0] = m_i[0] + T.cast(1e-20, "float32")

            for i, dd in T.Parallel(BLOCK_SQ, AD):
                safe_sq_idx = T.min(sq_block_id * BLOCK_SQ + i, ASq - 1)
                if sq_block_id * BLOCK_SQ + i < ASq:
                    Q_full[i, dd] = Q[safe_sq_idx, b, h, dd]
                else:
                    Q_full[i, dd] = T.cast(0, in_dtype)

            _max_sq_in_block = sq_block_id * BLOCK_SQ + (BLOCK_SQ - 1)
            _max_useful_sk = T.min(_max_sq_in_block, ASq - 1)
            _active_sk_tiles = T.max(
                T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1), 1
            )
            for sk_tile in T.serial(_active_sk_tiles):
                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    scores_f[i, j] = T.cast(0, "float32")

                for d_tile in T.serial((AD + BLOCK_D - 1) // BLOCK_D):
                    for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
                        d_idx = d_tile * BLOCK_D + dd
                        if d_idx < AD:
                            Q_s[i, dd] = Q_full[i, d_idx]
                        else:
                            Q_s[i, dd] = T.cast(0, in_dtype)

                    for dd, j in T.Parallel(BLOCK_D, BLOCK_SK):
                        sk_idx = sk_tile * BLOCK_SK + j
                        d_idx = d_tile * BLOCK_D + dd
                        safe_sk_idx = T.min(sk_idx, Sk - 1)
                        safe_d_idx = T.min(d_idx, AD - 1)
                        if (sk_idx < Sk) and (d_idx < AD):
                            K_s[dd, j] = K[safe_sk_idx, b, h, safe_d_idx]
                        else:
                            K_s[dd, j] = T.cast(0, in_dtype)

                    T.gemm(Q_s, K_s, scores_f)

                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    scores_f[i, j] = T.if_then_else(
                        (sq_block_id * BLOCK_SQ + i < ASq) and (sk_tile * BLOCK_SK + j < Sk) and (sq_block_id * BLOCK_SQ + i >= sk_tile * BLOCK_SK + j),
                        scores_f[i, j] * T.cast(SCALE, "float32"),
                        T.cast(-3.4028234663852886e38, "float32")
                    )

                T.reduce_max(scores_f, row_max_local, dim=1, clear=True)
                T.sync_threads()
                for i in T.Parallel(BLOCK_SQ):
                    m_i_prev[i] = m_i[i]
                    m_i[i] = T.if_then_else(
                        T.max(m_i[i], row_max_local[i]) <= T.cast(-3.4028234663852886e38, "float32"),
                        T.cast(0, "float32"),
                        T.max(m_i[i], row_max_local[i]),
                    )

                for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                    scores_f[i, j] = T.exp(scores_f[i, j] - m_i[i])

                T.reduce_sum(scores_f, row_sum_local, dim=1, clear=True)
                T.sync_threads()
                for i in T.Parallel(BLOCK_SQ):
                    d_i[i] = d_i[i] * T.exp(m_i_prev[i] - m_i[i]) + row_sum_local[i]

                if COMPUTE_INDEX and h == 0:
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        idx_scores_f[i, j] = T.cast(-3.4028234663852886e38, "float32")

                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        safe_sq_idx = T.min(sq_block_id * BLOCK_SQ + i, ASq - 1)
                        safe_sk_idx = T.min(sk_tile * BLOCK_SK + j, Sk - 1)
                        idx_scores_f[i, j] = T.if_then_else(
                            (sq_block_id * BLOCK_SQ + i < ASq) and (sk_tile * BLOCK_SK + j < Sk),
                            IndexScores[b, safe_sq_idx, safe_sk_idx],
                            T.cast(-3.4028234663852886e38, "float32")
                        )

                    T.reduce_max(idx_scores_f, row_max_local, dim=1, clear=True)
                    T.sync_threads()
                    for i in T.Parallel(BLOCK_SQ):
                        m1_i_prev[i] = m1_i[i]
                        m1_i[i] = T.if_then_else(
                            T.max(m1_i[i], row_max_local[i]) <= T.cast(-3.4028234663852886e38, "float32"),
                            T.cast(0, "float32"),
                            T.max(m1_i[i], row_max_local[i]),
                        )

                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        idx_scores_f[i, j] = T.exp(idx_scores_f[i, j] - m1_i[i])
                    T.reduce_sum(idx_scores_f, row_sum_local, dim=1, clear=True)
                    T.sync_threads()
                    for i in T.Parallel(BLOCK_SQ):
                        d1_i[i] = d1_i[i] * T.exp(m1_i_prev[i] - m1_i[i]) + row_sum_local[i]

            if T.get_thread_binding(0) == 0:
                for i in T.serial(BLOCK_SQ):
                    sq_idx = sq_block_id * BLOCK_SQ + i
                    if sq_idx < ASq:
                        M[b, h, sq_idx] = m_i[i]
                        D[b, h, sq_idx] = d_i[i]
                        if COMPUTE_INDEX and h == 0:
                            M1[b, sq_idx] = m1_i[i]
                            D1[b, sq_idx] = d1_i[i]
                        if sq_idx == 0:
                            dummy_pad[0] = dummy_pad[0] + T.cast(1e-20, "float16")

    return dsa_stage1

prim = make_dsa_splitk_stage1_kernel_debug(
    AB=AB,
    AH=AH,
    AD=AD,
    Sk=Sk,
    ASq=ASq,
    sparse_loss=False,
    softmax_scale=softmax_scale,
    in_dtype=in_dtype,
)

jit_kernel = tilelang.compile(prim, target="cuda")
print("Saved generated kernel.")

if __name__ == "__main__":
    device = torch.device("cuda")
    query = torch.randn(ASq, AB, AH, AD, dtype=torch.float16, device=device)
    key = torch.randn(Sk, AB, AH, AD, dtype=torch.float16, device=device)
    index_scores = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)
    index_mask = torch.randn(AB, ASq, Sk, dtype=torch.float32, device=device)
    softmax_m = torch.full((AB, AH, ASq), float("-inf"), dtype=torch.float32, device=device)
    softmax_d = torch.zeros((AB, AH, ASq), dtype=torch.float32, device=device)
    softmax_m1 = torch.full((AB, ASq), float("-inf"), dtype=torch.float32, device=device)
    softmax_d1 = torch.zeros((AB, ASq), dtype=torch.float32, device=device)

    # Execute via JIT wrapper directly
    jit_kernel(query, key, index_scores, index_mask, softmax_m, softmax_d, softmax_m1, softmax_d1)

    print("Kernel executed successfully via JIT wrapper directly!")
