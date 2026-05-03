"""Probe script for the metal pipeline 3D-buffer fix.

Repros the original blocker (``Buffer A_shared is 3-dimensional``) and
verifies the macro_generator.py fix that propagates leading region
dims (the version index inserted by InjectSoftwarePipeline) into
``T.access_ptr`` in MPSIntrinEmitter's ldmatrix_a/ldmatrix_b/simdgroup_copy.

Run:
    .venv/bin/python docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py
"""
from __future__ import annotations

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm_inner
from tilelang.engine.lower import lower

target = tvm_inner.target.Target("metal")


def try_lower(name, func):
    try:
        lower(func, target=target)
        return (name, "OK", "")
    except Exception as exc:
        return (name, "FAILED", f"{type(exc).__name__}: {str(exc)[:200]}")


# -----------------------------------------------------------------------
# k_pipe_2: T.Pipelined num_stages=2 with 16x16 fragment.
# Pre-fix: IndexError "Buffer A_shared is 3-dimensional".
# Post-fix: lowers to MSL successfully.
# -----------------------------------------------------------------------
@T.prim_func
def k_pipe_2(
    A: T.Tensor((64, 128), "float16"),
    B: T.Tensor((128, 64), "float16"),
    C: T.Tensor((64, 64), "float16"),
):
    with T.Kernel(T.ceildiv(64, 16), T.ceildiv(64, 16), threads=64) as (bx, by):
        A_shared = T.alloc_shared((16, 32), "float16", scope="shared")
        B_shared = T.alloc_shared((32, 16), "float16", scope="shared")
        C_local = T.alloc_fragment((16, 16), "float16")
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(128, 32), num_stages=2):
            T.copy(A[by * 16, ko * 32], A_shared)
            T.copy(B[ko * 32, bx * 16], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 16, bx * 16])


# -----------------------------------------------------------------------
# k_pipe_3: T.Pipelined num_stages=3 (deeper pipelining).
# -----------------------------------------------------------------------
@T.prim_func
def k_pipe_3(
    A: T.Tensor((64, 128), "float16"),
    B: T.Tensor((128, 64), "float16"),
    C: T.Tensor((64, 64), "float16"),
):
    with T.Kernel(T.ceildiv(64, 16), T.ceildiv(64, 16), threads=64) as (bx, by):
        A_shared = T.alloc_shared((16, 32), "float16", scope="shared")
        B_shared = T.alloc_shared((32, 16), "float16", scope="shared")
        C_local = T.alloc_fragment((16, 16), "float16")
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(128, 32), num_stages=3):
            T.copy(A[by * 16, ko * 32], A_shared)
            T.copy(B[ko * 32, bx * 16], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 16, bx * 16])


# -----------------------------------------------------------------------
# k_attn: pipelined Q*K^T (the sparse-MLA pattern).
# -----------------------------------------------------------------------
@T.prim_func
def k_attn(
    Q: T.Tensor((64, 64), "float16"),
    K: T.Tensor((64, 64), "float16"),
    O: T.Tensor((64, 64), "float16"),
):
    with T.Kernel(T.ceildiv(64, 16), threads=64) as (bx,):
        Q_shared = T.alloc_shared((16, 16), "float16", scope="shared")
        K_shared = T.alloc_shared((16, 16), "float16", scope="shared")
        S_local = T.alloc_fragment((16, 16), "float16")
        T.clear(S_local)
        T.copy(Q[bx * 16, 0], Q_shared)
        for ki in T.Pipelined(T.ceildiv(64, 16), num_stages=2):
            T.copy(K[bx * 16, ki * 16], K_shared)
            T.gemm(Q_shared, K_shared, S_local, transpose_B=True)
        T.copy(S_local, O[bx * 16, 0])


def main():
    results = [
        try_lower("k_pipe_2 (num_stages=2)", k_pipe_2),
        try_lower("k_pipe_3 (num_stages=3)", k_pipe_3),
        try_lower("k_attn (pipelined Q*K^T)", k_attn),
    ]
    print("=" * 80)
    print(f"{'Kernel':<30} {'Status':<10} {'Error':<40}")
    print("=" * 80)
    for name, status, err in results:
        print(f"{name:<30} {status:<10} {err[:60]}")


if __name__ == "__main__":
    main()
