"""Pytest and timing probe for the TileLang Metal T.gemm lowering unlock."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import pytest

pytest.importorskip("tilelang")

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm_inner
from tilelang.engine.lower import lower


@T.prim_func
def metal_gemm_test(
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


TARGET = tvm_inner.target.Target("metal")


def lower_metal_gemm_test():
    return lower(metal_gemm_test, target=TARGET)


def test_metal_t_gemm_lowers_to_compiled_artifact() -> None:
    result = lower_metal_gemm_test()
    assert type(result).__name__ == "CompiledArtifact"
    kernel_source = getattr(result, "kernel_source", "")
    if kernel_source:
        assert "kernel void" in kernel_source


def _bench_wall_ms(
    fn: Callable[[], object],
    synchronize: Callable[[], None],
    *,
    reps: int,
    warmups: int,
    rounds: int,
) -> list[float]:
    for _ in range(warmups):
        fn()
    synchronize()

    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3 / reps)
    return samples


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _profile_lowering(repeats: int) -> None:
    samples = []
    source = ""
    for idx in range(repeats):
        start = time.perf_counter()
        result = lower_metal_gemm_test()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        source = getattr(result, "kernel_source", "") or ""
        samples.append(elapsed_ms)
        print(f"lower_iter_{idx + 1}_ms={elapsed_ms:.3f}")

    print(f"lower_min_ms={min(samples):.3f}")
    print(f"lower_median_ms={statistics.median(samples):.3f}")
    print(f"lower_mean_ms={statistics.mean(samples):.3f}")
    print(f"kernel_source_len={len(source)}")
    print(f"kernel_source_has_kernel_void={'kernel void' in source}")
    print(f"kernel_source_has_simdgroup_mma={'simdgroup_multiply_accumulate' in source}")


def _set_tilelang_cache(enabled: bool) -> str:
    previous = tilelang.env.TILELANG_DISABLE_CACHE
    tilelang.env.TILELANG_DISABLE_CACHE = "0" if enabled else "1"
    return previous


def _profile_runtime(
    reps: int,
    warmups: int,
    rounds: int,
    atol: float,
    *,
    enable_tilelang_cache: bool,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is unavailable")

    previous_cache_setting = _set_tilelang_cache(enable_tilelang_cache)
    try:
        start = time.perf_counter()
        kernel = tilelang.compile(
            metal_gemm_test,
            out_idx=[2],
            target="metal",
            execution_backend="torch",
        )
        compile_ms = (time.perf_counter() - start) * 1e3
    finally:
        tilelang.env.TILELANG_DISABLE_CACHE = previous_cache_setting

    source = kernel.get_kernel_source() or ""

    a = torch.randn((64, 32), dtype=torch.float16, device="mps")
    b = torch.randn((32, 64), dtype=torch.float16, device="mps")
    c = torch.empty((64, 64), dtype=torch.float16, device="mps")
    torch_out = torch.empty_like(c)
    torch.mps.synchronize()

    out = kernel(a, b, c)
    torch.mps.synchronize()
    actual = out if hasattr(out, "shape") else c
    ref = a @ b
    torch.mps.synchronize()
    max_abs = (actual - ref).abs().max().item()
    if max_abs > atol:
        raise AssertionError(f"Metal GEMM max_abs={max_abs} exceeds atol={atol}")

    torch_ref_out = torch.matmul(a, b, out=torch_out)
    torch.mps.synchronize()
    if torch_ref_out.data_ptr() != torch_out.data_ptr():
        raise AssertionError("torch.matmul(..., out=...) did not return the preallocated output")
    torch_out_max_abs = (torch_out - ref).abs().max().item()
    if torch_out_max_abs > atol:
        raise AssertionError(f"torch.matmul(out=...) max_abs={torch_out_max_abs} exceeds atol={atol}")

    kernel_samples = _bench_wall_ms(
        lambda: kernel(a, b, c),
        torch.mps.synchronize,
        reps=reps,
        warmups=warmups,
        rounds=rounds,
    )
    torch_samples = _bench_wall_ms(
        lambda: torch.matmul(a, b, out=torch_out),
        torch.mps.synchronize,
        reps=reps,
        warmups=warmups,
        rounds=rounds,
    )
    kernel_median = statistics.median(kernel_samples)
    torch_median = statistics.median(torch_samples)

    print(f"runtime_compile_ms={compile_ms:.3f}")
    print(f"runtime_adapter_type={type(kernel.adapter).__name__}")
    print(f"runtime_tilelang_disk_cache_enabled={enable_tilelang_cache}")
    print(f"runtime_kernel_source_len={len(source)}")
    print(f"runtime_kernel_source_has_simdgroup_mma={'simdgroup_multiply_accumulate' in source}")
    print(f"runtime_max_abs_vs_torch_matmul={max_abs}")
    print(f"runtime_torch_matmul_out_max_abs_vs_torch_matmul={torch_out_max_abs}")
    print("runtime_torch_baseline=torch.matmul_out_preallocated")
    print("runtime_torch_baseline_allocation_free=True")
    print("runtime_kernel_wall_ms_samples=" + ",".join(f"{v:.6f}" for v in kernel_samples))
    print(f"runtime_kernel_wall_ms_median={kernel_median:.6f}")
    print("runtime_torch_matmul_out_wall_ms_samples=" + ",".join(f"{v:.6f}" for v in torch_samples))
    print(f"runtime_torch_matmul_out_wall_ms_median={torch_median:.6f}")
    print(f"runtime_speedup_vs_torch_matmul_out={torch_median / kernel_median:.3f}x")
    print(
        "runtime_profiler_note=TileLang do_bench currently calls CUDA synchronize/cache "
        "paths on MPS, and torch.mps.Event elapsed timing hung in a smoke test on this "
        "machine; this mode reports torch.mps.synchronize wall-clock timing."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-lowering", action="store_true")
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--lowering-repeats", type=_positive_int, default=7)
    parser.add_argument("--runtime-reps", type=_positive_int, default=300)
    parser.add_argument("--runtime-warmups", type=_nonnegative_int, default=50)
    parser.add_argument("--runtime-rounds", type=_positive_int, default=5)
    parser.add_argument("--runtime-atol", type=float, default=0.125)
    parser.add_argument(
        "--enable-tilelang-cache",
        action="store_true",
        help=(
            "Leave TileLang disk cache enabled during runtime profiling. The current "
            "MetalKernelAdapter has no libpath, so this can log a non-fatal cache-save error."
        ),
    )
    args = parser.parse_args()

    if not args.profile_lowering and not args.profile_runtime:
        return pytest.main([__file__, "-q", "-s"])

    if args.profile_lowering:
        _profile_lowering(args.lowering_repeats)
    if args.profile_runtime:
        _profile_runtime(
            args.runtime_reps,
            args.runtime_warmups,
            args.runtime_rounds,
            args.runtime_atol,
            enable_tilelang_cache=args.enable_tilelang_cache,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
