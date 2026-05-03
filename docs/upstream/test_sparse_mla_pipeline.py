"""Probe what TileLang's metal target accepts now (post-PR #2118).

Tests progressively heavier kernels that a sparse-MLA fwd would need:
1. T.gemm in T.Pipelined (the pattern sparse_mla_fwd uses)
2. T.alloc_shared + T.alloc_fragment + T.gemm + T.copy chain
3. Multi-stage T.Pipelined with K-loop and accumulator
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import statistics
import sys
import time
import traceback

import pytest

pytest.importorskip("tilelang")

import tilelang.language as T
from tilelang import tvm as tvm_inner
from tilelang.engine.lower import lower

TARGET = tvm_inner.target.Target("metal")
VERBOSE_ERRORS = os.environ.get("SPARSE_MLA_PIPELINE_VERBOSE") == "1"
LOWERING_NOTE = "lower/source-codegen only: artifact detail shows rt_mod; this probe does not launch Metal kernels"
DEVICE_COMPILE_NOTE = "lower + requested Metal device-compile path: artifact detail shows rt_mod; this probe still does not launch kernels"


def describe_artifact(artifact) -> str:
    rt_mod_state = "present" if getattr(artifact, "rt_mod", None) is not None else "none"
    kernel_source = getattr(artifact, "kernel_source", "") or ""
    kernel_source_bytes = len(kernel_source.encode("utf-8"))
    return f"{type(artifact).__name__}; rt_mod={rt_mod_state}; kernel_source_bytes={kernel_source_bytes}"


def format_error(exc: Exception) -> str:
    if VERBOSE_ERRORS:
        return "".join(traceback.format_exception(exc)).rstrip()
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return f"{type(exc).__name__}: {message}"


def try_lower(name, func, *, enable_device_compile: bool = False):
    try:
        artifact = lower(func, target=TARGET, enable_device_compile=enable_device_compile)
        return (name, "OK", describe_artifact(artifact))
    except Exception as e:
        return (name, "FAILED", format_error(e))


def timed_try_lower(name, func, *, enable_device_compile: bool = False):
    start = time.perf_counter()
    result = try_lower(name, func, enable_device_compile=enable_device_compile)
    return (*result, (time.perf_counter() - start) * 1000.0)


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


# Control for Test 2: same 32x32/fp32 accumulator fragment without the software
# pipeline transform. This isolates the current failure to Metal simdgroup vector
# dtype lowering, not to runtime execution or the 3-D pipeline-buffer rewrite.
@T.prim_func
def k2_32x32_no_pipeline(
    A: T.Tensor((32, 32), "float16"),
    B: T.Tensor((32, 32), "float16"),
    C: T.Tensor((32, 32), "float16"),
):
    with T.Kernel(1, 1, threads=128) as (bx, by):
        A_shared = T.alloc_shared((32, 32), "float16", scope="shared")
        B_shared = T.alloc_shared((32, 32), "float16", scope="shared")
        C_local = T.alloc_fragment((32, 32), "float32")
        T.clear(C_local)
        T.copy(A, A_shared)
        T.copy(B, B_shared)
        T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C)


# Control for Test 2: same software-pipeline pattern, but with the 16x16/fp16
# fragment shape that avoids the known float32x4 simdgroup-vector path.
@T.prim_func
def k2_pipelined_16x16_control(
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


def test_simple_metal_gemm_lowers() -> None:
    _, status, err = try_lower("k1_simple_gemm", k1_simple_gemm)
    assert status == "OK", err


@pytest.mark.xfail(reason="32x32 pipelined fragment still hits the known float32x4 simdgroup-vector limitation")
def test_pipelined_sparse_mla_pattern_lowers_when_simdgroup_vector_bug_is_fixed() -> None:
    _, status, err = try_lower("k2_pipelined_gemm", k2_pipelined_gemm)
    assert status == "OK", err


@pytest.mark.xfail(reason="32x32 fp32 accumulator fragment hits the known float32x4 simdgroup-vector limitation")
def test_32x32_no_pipeline_sparse_mla_shape_lowers_when_simdgroup_vector_bug_is_fixed() -> None:
    _, status, err = try_lower("k2_32x32_no_pipeline", k2_32x32_no_pipeline)
    assert status == "OK", err


def test_pipelined_16x16_control_lowers() -> None:
    _, status, err = try_lower("k2_pipelined_16x16_control", k2_pipelined_16x16_control)
    assert status == "OK", err


def test_chained_mixed_dtype_attention_gemms_lower() -> None:
    _, status, err = try_lower("k3_multi_gemm", k3_multi_gemm)
    assert status == "OK", err


KERNELS = [
    ("k1_simple_gemm", k1_simple_gemm),
    ("k2_pipelined_gemm", k2_pipelined_gemm),
    ("k2_32x32_no_pipeline", k2_32x32_no_pipeline),
    ("k2_pipelined_16x16_control", k2_pipelined_16x16_control),
    ("k3_multi_gemm", k3_multi_gemm),
]


def selected_kernels(name: str):
    if name == "all":
        return KERNELS
    prefix = f"{name}_"
    return [(kernel_name, func) for kernel_name, func in KERNELS if kernel_name.startswith(prefix)]


def runtime_note(enable_device_compile: bool) -> str:
    return DEVICE_COMPILE_NOTE if enable_device_compile else LOWERING_NOTE


def print_status(kernels, *, enable_device_compile: bool) -> None:
    results = [
        try_lower(name, func, enable_device_compile=enable_device_compile)
        for name, func in kernels
    ]
    print(runtime_note(enable_device_compile))
    print("=" * 70)
    print(f"{'Kernel':<25} {'Status':<10} {'Detail':<40}")
    print("=" * 70)
    for n, s, e in results:
        first_line = e.splitlines()[0] if e else ""
        print(f"{n:<25} {s:<10} {first_line[:100]}")
        if VERBOSE_ERRORS and "\n" in e:
            print("\n".join(f"  {line}" for line in e.splitlines()[1:]))


def print_timing(kernels, repeats: int, *, enable_device_compile: bool) -> None:
    print(runtime_note(enable_device_compile))
    print("=" * 96)
    print(f"{'Kernel':<25} {'Status':<10} {'mean ms':>10} {'median ms':>10} {'min ms':>10} {'max ms':>10} Detail")
    print("=" * 96)
    for name, func in kernels:
        samples = []
        status = ""
        err = ""
        for _ in range(repeats):
            _, status, err, elapsed_ms = timed_try_lower(
                name,
                func,
                enable_device_compile=enable_device_compile,
            )
            samples.append(elapsed_ms)
        first_line = err.splitlines()[0] if err else ""
        print(
            f"{name:<25} {status:<10} "
            f"{statistics.mean(samples):10.1f} {statistics.median(samples):10.1f} "
            f"{min(samples):10.1f} {max(samples):10.1f} {first_line[:80]}"
        )


def print_profile(kernels, limit: int, *, enable_device_compile: bool) -> None:
    print(runtime_note(enable_device_compile))
    for name, func in kernels:
        profile = cProfile.Profile()
        profile.enable()
        _, status, err, elapsed_ms = timed_try_lower(
            name,
            func,
            enable_device_compile=enable_device_compile,
        )
        profile.disable()
        first_line = err.splitlines()[0] if err else ""
        print("=" * 96)
        print(f"{name}: {status} in {elapsed_ms:.1f} ms {first_line}")
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(limit)
        sys.stdout.write(stream.getvalue())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["all", "k1", "k2", "k3"], default="all")
    parser.add_argument("--time", action="store_true", help="time TileLang lower/codegen for each selected kernel")
    parser.add_argument("--repeat", type=int, default=3, help="repeat count for --time")
    parser.add_argument("--profile", action="store_true", help="print cProfile cumulative hot frames for lower/codegen")
    parser.add_argument("--profile-limit", type=int, default=12)
    parser.add_argument(
        "--device-compile",
        action="store_true",
        help="request TileLang's Metal device-compile path; still does not launch kernels",
    )
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    kernels = selected_kernels(args.kernel)
    if args.time:
        print_timing(kernels, args.repeat, enable_device_compile=args.device_compile)
    elif args.profile:
        print_profile(kernels, args.profile_limit, enable_device_compile=args.device_compile)
    else:
        print_status(kernels, enable_device_compile=args.device_compile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
