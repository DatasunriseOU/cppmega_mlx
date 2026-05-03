"""Probe script for the metal pipeline 3D-buffer fix.

Repros the original blocker (``Buffer A_shared is 3-dimensional``) and
verifies the macro_generator.py fix that propagates leading region
dims (the version index inserted by InjectSoftwarePipeline) into
``T.access_ptr`` in MPSIntrinEmitter's ldmatrix_a/ldmatrix_b/simdgroup_copy.

Run:
    .venv/bin/python docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py
"""
from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable

import pytest

pytest.importorskip("tilelang")

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm_inner
from tilelang.engine.lower import lower

TARGET = tvm_inner.target.Target("metal")
EXPECTED_METRICS = {
    "k_pipe_2": {
        "threadgroups": {"A_shared": 1024, "B_shared": 1024},
        "simdgroup_load": 4,
        "simdgroup_multiply_accumulate": 2,
        "simdgroup_store": 2,
    },
    "k_pipe_3": {
        "threadgroups": {"A_shared": 1536, "B_shared": 1536},
        "simdgroup_load": 6,
        "simdgroup_multiply_accumulate": 3,
        "simdgroup_store": 2,
    },
    "k_attn": {
        "threadgroups": {"Q_shared": 256, "K_shared": 512},
        "simdgroup_load": 4,
        "simdgroup_multiply_accumulate": 2,
        "simdgroup_store": 2,
    },
}
RUNTIME_SHAPES = {
    "k_pipe_2": ((64, 128), (128, 64), (64, 64)),
    "k_pipe_3": ((64, 128), (128, 64), (64, 64)),
    "k_attn": ((64, 64), (64, 64), (64, 64)),
}


def try_lower(name, func):
    try:
        lower(func, target=TARGET)
        return (name, "OK", "")
    except Exception as exc:
        return (name, "FAILED", f"{type(exc).__name__}: {str(exc)[:200]}")


def _extract_kernel_source(artifact) -> str:
    source = getattr(artifact, "kernel_source", None)
    if source:
        return source
    rt_mod = getattr(artifact, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "get_source"):
        return rt_mod.get_source()
    return ""


def _threadgroup_allocations(source: str) -> dict[str, int]:
    allocations: dict[str, int] = {}
    for line in source.splitlines():
        line = line.strip()
        if not line.startswith("threadgroup "):
            continue
        if "[" not in line or "]" not in line:
            continue
        name = line.split("[", 1)[0].split()[-1]
        size = line.split("[", 1)[1].split("]", 1)[0]
        if size.isdigit():
            allocations[name] = int(size)
    return allocations


def collect_kernel_metrics(name, func):
    try:
        artifact = lower(func, target=TARGET)
        source = _extract_kernel_source(artifact)
        return {
            "name": name,
            "status": "OK",
            "error": "",
            "source": source,
            "source_bytes": len(source.encode("utf-8")),
            "source_lines": len(source.splitlines()),
            "threadgroups": _threadgroup_allocations(source),
            "simdgroup_load": source.count("simdgroup_load"),
            "simdgroup_multiply_accumulate": source.count(
                "simdgroup_multiply_accumulate"
            ),
            "simdgroup_store": source.count("simdgroup_store"),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "source": "",
            "source_bytes": 0,
            "source_lines": 0,
            "threadgroups": {},
            "simdgroup_load": 0,
            "simdgroup_multiply_accumulate": 0,
            "simdgroup_store": 0,
        }


def _profile_lowering(repeats: int) -> list[tuple[str, float, float]]:
    rows = []
    for name, _, kernel in KERNELS:
        samples_ms = []
        for _ in range(repeats):
            start = time.perf_counter()
            lower(kernel, target=TARGET)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
        rows.append((name, statistics.median(samples_ms), min(samples_ms)))
    return rows


def _bench_wall_ms(
    fn: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    reps: int,
    warmups: int,
    rounds: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    synchronize()

    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / reps)
    return statistics.median(samples), min(samples)


def _profile_runtime(reps: int, warmups: int, rounds: int):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is unavailable")

    rows = []
    for name, _, kernel_func in KERNELS:
        start = time.perf_counter()
        compiled = tilelang.compile(
            kernel_func,
            out_idx=[2],
            target="metal",
            execution_backend="torch",
        )
        compile_ms = (time.perf_counter() - start) * 1000.0

        tensors = [
            torch.randn(shape, dtype=torch.float16, device="mps")
            for shape in RUNTIME_SHAPES[name][:2]
        ]
        tensors.append(
            torch.empty(RUNTIME_SHAPES[name][2], dtype=torch.float16, device="mps")
        )
        launch_median_ms, launch_min_ms = _bench_wall_ms(
            lambda: compiled(*tensors),
            torch.mps.synchronize,
            reps=reps,
            warmups=warmups,
            rounds=rounds,
        )
        rows.append(
            (
                name,
                type(compiled.adapter).__name__,
                compile_ms,
                launch_median_ms,
                launch_min_ms,
            )
        )
    return rows


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


KERNELS = (
    ("k_pipe_2", "num_stages=2", k_pipe_2),
    ("k_pipe_3", "num_stages=3", k_pipe_3),
    ("k_attn", "pipelined Q*K^T", k_attn),
)


def main():
    results = [
        collect_kernel_metrics(f"{name} ({label})", kernel)
        for name, label, kernel in KERNELS
    ]
    print("=" * 80)
    print(
        f"{'Kernel':<30} {'Status':<8} {'Lines':>5} {'Bytes':>6} "
        f"{'Load':>4} {'MMA':>4} {'Store':>5} {'Threadgroup allocs'}"
    )
    print("=" * 80)
    for row in results:
        allocs = ", ".join(
            f"{name}[{size}]" for name, size in sorted(row["threadgroups"].items())
        )
        print(
            f"{row['name']:<30} {row['status']:<8} {row['source_lines']:>5} "
            f"{row['source_bytes']:>6} {row['simdgroup_load']:>4} "
            f"{row['simdgroup_multiply_accumulate']:>4} "
            f"{row['simdgroup_store']:>5} {allocs or row['error'][:80]}"
        )

    if os.environ.get("TILELANG_PIPELINED_PROFILE"):
        repeats = int(os.environ.get("TILELANG_PIPELINED_PROFILE_REPEATS", "3"))
        print("=" * 80)
        print(f"lower() profile, repeats={repeats}")
        print(f"{'Kernel':<30} {'median_ms':>10} {'min_ms':>10}")
        print("=" * 80)
        for name, median_ms, min_ms in _profile_lowering(repeats):
            print(f"{name:<30} {median_ms:>10.2f} {min_ms:>10.2f}")

    if os.environ.get("TILELANG_PIPELINED_RUNTIME_PROFILE"):
        os.environ.setdefault("TILELANG_DISABLE_CACHE", "1")
        reps = int(os.environ.get("TILELANG_PIPELINED_RUNTIME_REPS", "200"))
        warmups = int(os.environ.get("TILELANG_PIPELINED_RUNTIME_WARMUPS", "20"))
        rounds = int(os.environ.get("TILELANG_PIPELINED_RUNTIME_ROUNDS", "5"))
        print("=" * 80)
        print(
            "torch/MPS runtime profile, "
            f"reps={reps}, warmups={warmups}, rounds={rounds}"
        )
        print(
            f"{'Kernel':<30} {'Adapter':<20} {'compile_ms':>10} "
            f"{'launch_median_ms':>16} {'launch_min_ms':>13}"
        )
        print("=" * 80)
        for name, adapter, compile_ms, launch_median_ms, launch_min_ms in (
            _profile_runtime(reps, warmups, rounds)
        ):
            print(
                f"{name:<30} {adapter:<20} {compile_ms:>10.2f} "
                f"{launch_median_ms:>16.4f} {launch_min_ms:>13.4f}"
            )


@pytest.mark.parametrize(("name", "_label", "kernel"), KERNELS)
def test_pipelined_metal_kernel_lowers(name, _label, kernel) -> None:
    _, status, err = try_lower(name, kernel)
    assert status == "OK", err


@pytest.mark.parametrize(("name", "_label", "kernel"), KERNELS)
def test_pipelined_metal_source_metrics(name, _label, kernel) -> None:
    row = collect_kernel_metrics(name, kernel)
    assert row["status"] == "OK", row["error"]
    assert "3-dimensional" not in row["error"]
    assert row["source"], "lower() did not expose generated Metal source"

    expected = EXPECTED_METRICS[name]
    for alloc_name, alloc_size in expected["threadgroups"].items():
        assert row["threadgroups"].get(alloc_name) == alloc_size
    for counter in (
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    ):
        assert row[counter] == expected[counter]


if __name__ == "__main__":
    main()
