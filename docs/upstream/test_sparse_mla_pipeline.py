"""Probe what TileLang's metal target accepts now (post-PR #2118).

Tests progressively heavier kernels that a sparse-MLA fwd would need:
1. T.gemm in T.Pipelined (the pattern sparse_mla_fwd uses)
2. T.alloc_shared + T.alloc_fragment + T.gemm + T.copy chain
3. Multi-stage T.Pipelined with K-loop and accumulator
"""
import tilelang
import tilelang.language as T
from tilelang import tvm as tvm_inner
from tilelang.engine.lower import lower

target = tvm_inner.target.Target("metal")
results = []


def try_lower(name, func):
    try:
        result = lower(func, target=target)
        return (name, "OK", "")
    except Exception as e:
        return (name, "FAILED", f"{type(e).__name__}: {str(e)[:300]}")


# Test 1: simple gemm (already verified, baseline)
@T.prim_func
def k1_simple_gemm(
    A: T.Tensor((64, 32), "float16"),
    B: T.Tensor((32, 64), "float16"),
    C: T.Tensor((64, 64), "float16"),
):
    with T.Kernel(T.ceildiv(64, 16), T.ceildiv(64, 16), threads=64) as (bx, by):
        A_shared = T.alloc_shared((16, 32), "float16", scope="shared")
        B_shared = T.alloc_shared((32, 16), "float16", scope="shared")
        C_local = T.alloc_fragment((16, 16), "float16")
        T.clear(C_local)
        T.copy(A[by * 16, 0], A_shared)
        T.copy(B[0, bx * 16], B_shared)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 16, bx * 16])


# Test 2: Pipelined K-loop with multiple gemms (what sparse-MLA fwd does)
@T.prim_func
def k2_pipelined_gemm(
    A: T.Tensor((128, 256), "float16"),
    B: T.Tensor((256, 128), "float16"),
    C: T.Tensor((128, 128), "float16"),
):
    with T.Kernel(T.ceildiv(128, 32), T.ceildiv(128, 32), threads=128) as (bx, by):
        A_shared = T.alloc_shared((32, 32), "float16", scope="shared")
        B_shared = T.alloc_shared((32, 32), "float16", scope="shared")
        C_local = T.alloc_fragment((32, 32), "float32")
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(256, 32), num_stages=2):
            T.copy(A[by * 32, ko * 32], A_shared)
            T.copy(B[ko * 32, bx * 32], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * 32, bx * 32])


# Test 3: multiple gemms in same kernel (sparse-MLA fwd has 3 gemms)
@T.prim_func
def k3_multi_gemm(
    Q: T.Tensor((32, 64), "float16"),
    K: T.Tensor((32, 64), "float16"),
    V: T.Tensor((32, 64), "float16"),
    O: T.Tensor((32, 64), "float16"),
):
    with T.Kernel(1, threads=128) as (bx,):
        Q_shared = T.alloc_shared((32, 64), "float16", scope="shared")
        K_shared = T.alloc_shared((32, 64), "float16", scope="shared")
        V_shared = T.alloc_shared((32, 64), "float16", scope="shared")
        S_local = T.alloc_fragment((32, 32), "float32")
        O_local = T.alloc_fragment((32, 64), "float32")
        T.clear(S_local)
        T.clear(O_local)
        T.copy(Q, Q_shared)
        T.copy(K, K_shared)
        T.copy(V, V_shared)
        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)
        T.gemm(S_local, V_shared, O_local)
        T.copy(O_local, O)


for name, kernel in [("k1_simple_gemm", k1_simple_gemm), ("k2_pipelined_gemm", k2_pipelined_gemm), ("k3_multi_gemm", k3_multi_gemm)]:
    n, status, err = try_lower(name, kernel)
    results.append((n, status, err))

print("=" * 70)
print(f"{'Kernel':<25} {'Status':<10} {'Error':<40}")
print("=" * 70)
for n, s, e in results:
    print(f"{n:<25} {s:<10} {e[:60]}")
