"""Probe script for Metal shared.dyn storage scope support.

Documents three regimes:

1. static_size_dyn: a single T.alloc_shared(scope="shared.dyn") with
   compile-time-known extent. MergeSharedMemoryAllocations collapses
   the dynamic alloc into a constant-sized merged buffer that codegen
   prints as threadgroup half buf_dyn_shmem[128]. Lowers cleanly.

2. merged_dyn: two distinct dynamic allocs that the merge pass
   coalesces into a single backing buffer (still constant size).
   Lowers cleanly.

3. symbolic_dyn: an alloc_shared whose extent depends on a
   symbolic dimension. The merged buffer's extent stays symbolic and
   codegen_metal.cc::VisitStmt_(AllocateNode) fires the
   ICHECK_GT(constant_size, 0) assertion. Documented as a known
   limitation; not on the cppmega Path B path because the topk_selector
   port (the original consumer of dynamic shmem) was rewritten to use
   direct mx.fast.metal_kernel (see
   cppmega_mlx/nn/_tilelang/topk_selector.py).
"""
from __future__ import annotations

import pytest

pytest.importorskip("tilelang")

import tilelang.language as T
from tilelang import tvm as tvm_inner
from tilelang.engine.lower import lower

TARGET = tvm_inner.target.Target("metal")


def try_lower(name, func):
    try:
        result = lower(func, target=TARGET)
        return (name, "OK", result.kernel_source)
    except Exception as exc:
        return (name, "FAILED", f"{type(exc).__name__}: {str(exc)[:200]}")


# 1. Static-size shared.dyn.
@T.prim_func
def static_size_dyn(A: T.Tensor((128,), "float16"), B: T.Tensor((128,), "float16")):
    with T.Kernel(1, threads=128) as (bx,):
        S = T.alloc_shared((128,), "float16", scope="shared.dyn")
        for i in T.Parallel(128):
            S[i] = A[i] * 2.0
        for i in T.Parallel(128):
            B[i] = S[i]


# 2. Merged dyn shmem (two allocs).
@T.prim_func
def merged_dyn(A: T.Tensor((128,), "float16"), B: T.Tensor((128,), "float16")):
    with T.Kernel(1, threads=128) as (bx,):
        S = T.alloc_shared((128,), "float16", scope="shared.dyn")
        T2 = T.alloc_shared((128,), "float16", scope="shared.dyn")
        for i in T.Parallel(128):
            S[i] = A[i] * 2.0
        for i in T.Parallel(128):
            T2[i] = S[i] + 1.0
        for i in T.Parallel(128):
            B[i] = T2[i]


# 3. Symbolic-size dyn shmem (the documented limitation).
N = T.symbolic("N")


@T.prim_func
def symbolic_dyn(A: T.Tensor((N,), "float16"), B: T.Tensor((N,), "float16")):
    with T.Kernel(1, threads=128) as (bx,):
        S = T.alloc_shared((N,), "float16", scope="shared.dyn")
        for i in T.Parallel(N):
            S[i] = A[i] * 2.0
        for i in T.Parallel(N):
            B[i] = S[i]


def main():
    cases = [
        ("static_size_dyn", static_size_dyn),
        ("merged_dyn", merged_dyn),
        ("symbolic_dyn (known limitation)", symbolic_dyn),
    ]
    print("=" * 80)
    print(f"{'Case':<40} {'Status':<10} {'Notes':<30}")
    print("=" * 80)
    for name, fn in cases:
        n, status, payload = try_lower(name, fn)
        if status == "OK":
            # Confirm the source actually contains a threadgroup decl
            # (i.e. shared.dyn really did lower to a static-sized buf).
            if "threadgroup" in payload and "buf_dyn_shmem" in payload:
                note = "merged into static threadgroup decl"
            else:
                note = "lowered (no shared decl)"
        else:
            note = payload[:60]
        print(f"{n:<40} {status:<10} {note:<30}")


@pytest.mark.parametrize(
    ("name", "kernel"),
    [
        ("static_size_dyn", static_size_dyn),
        ("merged_dyn", merged_dyn),
    ],
)
def test_const_extent_shared_dyn_lowers_to_threadgroup(name, kernel) -> None:
    _, status, payload = try_lower(name, kernel)
    assert status == "OK", payload
    assert "threadgroup" in payload
    assert "buf_dyn_shmem" in payload


@pytest.mark.xfail(reason="symbolic shared.dyn extent is a documented Metal codegen limitation")
def test_symbolic_extent_shared_dyn_lowers_when_runtime_dyn_shmem_is_implemented() -> None:
    _, status, payload = try_lower("symbolic_dyn", symbolic_dyn)
    assert status == "OK", payload


if __name__ == "__main__":
    main()
