# pyright: reportInvalidTypeForm=false, reportMissingImports=false, reportUndefinedVariable=false
"""Profile TileLang Path C FP8 kernels against Path B hand-written MSL.

This is the local Apple-Silicon/Metal Path C FP8 harness. It intentionally
lives outside the TileLang/TVM trees: the script profiles the current
apple-head TileLang Metal lowering without modifying core codegen.

Default output:
    bench/tilelang_ports/fp8_path_c_vs_path_b.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT", "/private/tmp/tilelang_apple_head/tilelang"))
TVM_ROOT = Path(os.environ.get("TVM_ROOT", "/private/tmp/tvm_apple_head/tvm"))
TILELANG_METAL_TARGET = "metal"
TILELANG_METAL_VECMAT_TARGET = "metal -thread_warp_size=32"
DEFAULT_PARITY_MAX_ABS = 1.0e-5
DEFAULT_PARITY_MAX_REL = 1.0e-5

sys.path.insert(0, str(REPO_ROOT))
if TILELANG_ROOT.exists():
    sys.path.insert(0, str(TILELANG_ROOT))

np: Any = None
mx: Any = None
torch: Any = None
fp8_msl_status: Any = None
fp8_scaled_matmul_raw: Any = None
fp8_scaled_vecmat: Any = None
canonical_vecmat_runtime_body: Any = None
fp8_scaled_vecmat_path_c: Any = None
fp8_vecmat_msl_blockers: Any = None
make_fp8_vecmat_reduce_kernel: Any = None


def _require_bench_deps() -> None:
    """Import GPU/runtime deps only for real benchmarks, not config tests."""

    global TILELANG_METAL_VECMAT_TARGET
    global canonical_vecmat_runtime_body
    global fp8_msl_status
    global fp8_scaled_matmul_raw
    global fp8_scaled_vecmat
    global fp8_scaled_vecmat_path_c
    global fp8_vecmat_msl_blockers
    global make_fp8_vecmat_reduce_kernel
    global mx
    global np
    global torch

    if np is None:
        import numpy as _np

        np = _np
    if mx is None:
        import mlx.core as _mx

        mx = _mx
    if torch is None:
        import torch as _torch

        torch = _torch
    if fp8_msl_status is None:
        from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (
            fp8_msl_status as _fp8_msl_status,
            fp8_scaled_matmul_raw as _fp8_scaled_matmul_raw,
            fp8_scaled_vecmat as _fp8_scaled_vecmat,
        )

        fp8_msl_status = _fp8_msl_status
        fp8_scaled_matmul_raw = _fp8_scaled_matmul_raw
        fp8_scaled_vecmat = _fp8_scaled_vecmat
    if fp8_scaled_vecmat_path_c is None:
        from cppmega_mlx.nn._tilelang.fp8_vecmat_path_c import (
            TILELANG_METAL_VECMAT_TARGET as _TILELANG_METAL_VECMAT_TARGET,
            canonical_vecmat_runtime_body as _canonical_vecmat_runtime_body,
            fp8_scaled_vecmat_path_c as _fp8_scaled_vecmat_path_c,
            fp8_vecmat_msl_blockers as _fp8_vecmat_msl_blockers,
            make_fp8_vecmat_reduce_kernel as _make_fp8_vecmat_reduce_kernel,
        )

        TILELANG_METAL_VECMAT_TARGET = _TILELANG_METAL_VECMAT_TARGET
        canonical_vecmat_runtime_body = _canonical_vecmat_runtime_body
        fp8_scaled_vecmat_path_c = _fp8_scaled_vecmat_path_c
        fp8_vecmat_msl_blockers = _fp8_vecmat_msl_blockers
        make_fp8_vecmat_reduce_kernel = _make_fp8_vecmat_reduce_kernel


Shape = dict[str, Any]

SHAPES: dict[str, Shape] = {
    "tiny_128": {
        "kind": "matmul",
        "M": 128,
        "N": 128,
        "K": 128,
        "BM": 16,
        "BN": 16,
        "BK": 16,
        "num_stages": 0,
        "parity": True,
    },
    "matmul_128": {
        "kind": "matmul",
        "M": 128,
        "N": 128,
        "K": 128,
        "BM": 16,
        "BN": 16,
        "BK": 16,
        "num_stages": 0,
        "parity": True,
    },
    "matmul_512": {
        "kind": "matmul",
        "M": 512,
        "N": 512,
        "K": 512,
        "BM": 32,
        "BN": 32,
        "BK": 32,
        "num_stages": 0,
        "parity": False,
    },
    "vecmat_4096": {
        "kind": "vecmat",
        "M": 1,
        "N": 4096,
        "K": 4096,
        "BM": 1,
        "BN": 32,
        "BK": 32,
        "num_stages": 0,
        "parity": True,
    },
}


def _fp8_scaled_matmul_kernel_template(
    A_fp8: T.Tensor((_M, _K), "float8_e4m3"),
    A_scale: T.Tensor((1,), "float32"),
    B_fp8: T.Tensor((_K, _N), "float8_e4m3"),
    B_scale: T.Tensor((1,), "float32"),
    C: T.Tensor((_M, _N), "float32"),
):
    with T.Kernel(T.ceildiv(_N, _BN), T.ceildiv(_M, _BM), threads=128) as (bx, by):
        A_shared = T.alloc_shared((_BM, _BK), "float8_e4m3", scope="shared")
        B_shared = T.alloc_shared((_BK, _BN), "float8_e4m3", scope="shared")
        C_local = T.alloc_fragment((_BM, _BN), "float32")
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(_K, _BK), num_stages=_NUM_STAGES):
            T.copy(A_fp8[by * _BM, ko * _BK], A_shared)
            T.copy(B_fp8[ko * _BK, bx * _BN], B_shared)
            T.fp8_scaled_matmul(A_shared, A_scale, B_shared, B_scale, C_local)
        T.copy(C_local, C[by * _BM, bx * _BN])


@dataclass(frozen=True)
class BenchStats:
    label: str
    ok: bool
    median_ms: float | None = None
    min_ms: float | None = None
    p90_ms: float | None = None
    max_ms: float | None = None
    tflops: float | None = None
    warmup: int = 0
    iters: int = 0
    paired: bool = False
    error: str | None = None


@dataclass(frozen=True)
class PairedBenchResult:
    stats: dict[str, BenchStats]
    paired_ratios: dict[str, float]


def _sync_mlx() -> None:
    _require_bench_deps()
    sync = getattr(mx, "synchronize", None)
    if sync is not None:
        sync()


def _sync_torch_mps() -> None:
    _require_bench_deps()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _sync_all() -> None:
    _sync_mlx()
    _sync_torch_mps()


def _percentile(sorted_samples: list[float], pct: float) -> float:
    if not sorted_samples:
        raise ValueError("empty samples")
    idx = min(len(sorted_samples) - 1, max(0, int(round((len(sorted_samples) - 1) * pct))))
    return sorted_samples[idx]


def _bench_callable(
    label: str,
    fn: Callable[[], Any],
    sync: Callable[[], None],
    *,
    flops: float,
    warmup: int,
    iters: int,
) -> BenchStats:
    try:
        for _ in range(warmup):
            fn()
            sync()
        samples = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter() - t0) * 1000.0)
        return _stats_from_samples(label, samples, flops=flops, warmup=warmup, iters=iters)
    except Exception as exc:  # pragma: no cover - used for local profiling receipts
        return BenchStats(
            label=label,
            ok=False,
            warmup=warmup,
            iters=iters,
            error=f"{type(exc).__name__}: {exc}",
        )


def _stats_from_samples(
    label: str,
    samples: list[float],
    *,
    flops: float,
    warmup: int,
    iters: int,
    paired: bool = False,
) -> BenchStats:
    samples.sort()
    median = statistics.median(samples)
    return BenchStats(
        label=label,
        ok=True,
        median_ms=float(median),
        min_ms=float(samples[0]),
        p90_ms=float(_percentile(samples, 0.90)),
        max_ms=float(samples[-1]),
        tflops=float(flops / (median / 1000.0) / 1.0e12) if median > 0 else None,
        warmup=warmup,
        iters=iters,
        paired=paired,
    )


def _bench_paired_callables(
    strategies: tuple[tuple[str, Callable[[], Any]], ...],
    sync: Callable[[], None],
    *,
    flops: float,
    warmup: int,
    iters: int,
) -> PairedBenchResult:
    """Time peer kernels in alternating order to reduce launch-order bias."""

    failed: dict[str, str] = {}
    for step in range(warmup):
        order = strategies if step % 2 == 0 else tuple(reversed(strategies))
        for label, fn in order:
            if label in failed:
                continue
            try:
                fn()
                sync()
            except Exception as exc:
                failed[label] = f"{type(exc).__name__}: {exc}"

    samples: dict[str, list[float]] = {label: [] for label, _ in strategies}
    samples_by_step: dict[str, dict[int, float]] = {label: {} for label, _ in strategies}
    for step in range(iters):
        order = strategies if step % 2 == 0 else tuple(reversed(strategies))
        for label, fn in order:
            if label in failed:
                continue
            try:
                sync()
                t0 = time.perf_counter()
                fn()
                sync()
                elapsed = (time.perf_counter() - t0) * 1000.0
                samples[label].append(elapsed)
                samples_by_step[label][step] = elapsed
            except Exception as exc:
                failed[label] = f"{type(exc).__name__}: {exc}"

    out: dict[str, BenchStats] = {}
    for label, _ in strategies:
        if label in failed or not samples[label]:
            out[label] = BenchStats(
                label=label,
                ok=False,
                warmup=warmup,
                iters=iters,
                paired=True,
                error=failed.get(label, "kernel did not produce timing samples"),
            )
        else:
            out[label] = _stats_from_samples(
                label,
                samples[label],
                flops=flops,
                warmup=warmup,
                iters=iters,
                paired=True,
            )
    paired_ratios: dict[str, float] = {}
    if len(strategies) >= 2:
        base_label = strategies[0][0]
        base_samples = samples_by_step.get(base_label, {})
        for label, _ in strategies[1:]:
            ratios = [
                samples_by_step[label][step] / base_samples[step]
                for step in sorted(base_samples)
                if step in samples_by_step[label] and base_samples[step] > 0
            ]
            if ratios:
                paired_ratios[f"{label}_over_{base_label}_paired_median"] = float(statistics.median(ratios))
                paired_ratios[f"{label}_over_{base_label}_paired_p90"] = float(
                    _percentile(sorted(ratios), 0.90)
                )
    return PairedBenchResult(stats=out, paired_ratios=paired_ratios)


def _run_cmd(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _git_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "head": _run_cmd(["git", "rev-parse", "HEAD"], cwd=path),
        "status_short": _run_cmd(["git", "status", "--short"], cwd=path),
    }


def _require_runtime() -> None:
    _require_bench_deps()
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"MLX default device is not GPU: {mx.default_device()}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch.backends.mps.is_available() is false")
    status = fp8_msl_status()
    if not status.available:
        raise RuntimeError(f"Path B FP8 MSL unavailable: {status.reason}")


def _import_tilelang() -> tuple[Any, Any, Any, Any]:
    _require_bench_deps()
    import tilelang
    import tilelang.language as T
    from tilelang import tvm
    from tvm.target import Target

    return tilelang, T, tvm, Target


def _make_scaled_matmul_kernel(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    num_stages: int,
):
    _, T, _, _ = _import_tilelang()
    g = globals()
    g.update(
        T=T,
        _M=M,
        _N=N,
        _K=K,
        _BM=BM,
        _BN=BN,
        _BK=BK,
        _NUM_STAGES=num_stages,
    )
    return T.prim_func(_fp8_scaled_matmul_kernel_template)


def _lower_source(prim_func: Any, *, target: str = TILELANG_METAL_TARGET) -> str:
    tilelang, _, _, Target = _import_tilelang()
    artifact = tilelang.lower(prim_func, target=Target(target))
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    if hasattr(artifact, "rt_mod") and hasattr(artifact.rt_mod, "get_source"):
        return str(artifact.rt_mod.get_source())
    return str(artifact)


def _compile_tilelang(prim_func: Any, *, target: str = TILELANG_METAL_TARGET) -> Any:
    tilelang, _, _, _ = _import_tilelang()
    return tilelang.compile(prim_func, target=target)


def _source_metrics(src: str) -> dict[str, Any]:
    lowered = src.lower()
    markers = {
        "simdgroup_multiply_accumulate": src.count("simdgroup_multiply_accumulate"),
        "simdgroup_load": src.count("simdgroup_load"),
        "simdgroup_store": src.count("simdgroup_store"),
        "threadgroup_half": lowered.count("threadgroup half"),
        "threadgroup_uchar": lowered.count("threadgroup uchar"),
        "threadgroup_barrier": src.count("threadgroup_barrier"),
        "fp8_e4m3_decode_helper": src.count("__tvm_fp8_e4m3_to_half"),
        "fp8_e5m2_decode_helper": src.count("__tvm_fp8_e5m2_to_half"),
        "A_scale_loads": src.count("A_scale["),
        "B_scale_loads": src.count("B_scale["),
        "scalar_float_a_val": lowered.count("float a_val"),
        "scalar_float_b_val": lowered.count("float b_val"),
        "tvm_thread_allreduce": src.count("tvm_thread_allreduce"),
        "simd_sum": src.count("simd_sum"),
        "kernel_void": src.count("kernel void"),
        "packed_uint_loads": src.count("reinterpret_cast<device const uint*>"),
        "fp8_e4m3_lut": src.count("fp8_e4m3fn_lut"),
    }
    return {"source_len": len(src), "markers": markers}


def _path_c_vecmat_runtime_source_metrics(*, N: int, K: int) -> dict[str, Any]:
    src = canonical_vecmat_runtime_body(N=N, K=K, scale_w_per_row=True)
    out = _source_metrics(src)
    out["source"] = "mlx_runtime_canonical_body"
    return out


def _xcrun_compile(src: str, *, label: str, dump_dir: Path | None) -> dict[str, Any]:
    if shutil.which("xcrun") is None:
        return {"ok": False, "skipped": True, "reason": "xcrun not found"}
    find_res = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
    )
    if find_res.returncode != 0:
        return {
            "ok": False,
            "skipped": True,
            "reason": "xcrun --sdk macosx --find metal failed",
            "stderr": find_res.stderr.strip(),
        }

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        msl_path = dump_dir / f"{label}.metal"
        air_path = dump_dir / f"{label}.air"
        msl_path.write_text(src)
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory()
        msl_path = Path(tmp.name) / f"{label}.metal"
        air_path = Path(tmp.name) / f"{label}.air"
        msl_path.write_text(src)
        cleanup = True

    env = os.environ.copy()
    env.setdefault("MAKEFLAGS", "-j4")
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    try:
        res = subprocess.run(
            ["xcrun", "--sdk", "macosx", "metal", "-c", str(msl_path), "-o", str(air_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        return {
            "ok": res.returncode == 0,
            "returncode": res.returncode,
            "stderr": res.stderr.strip(),
            "msl_path": str(msl_path) if dump_dir is not None else None,
            "air_path": str(air_path) if dump_dir is not None and air_path.exists() else None,
        }
    finally:
        if cleanup:
            tmp.cleanup()  # type: ignore[name-defined]


def _torch_fp8(x: torch.Tensor) -> torch.Tensor:
    _require_bench_deps()
    return x.detach().cpu().to(torch.float8_e4m3fn)


def _build_inputs(
    shape: Shape,
    *,
    seed: int,
    input_scale: float,
    scale_a: float,
    scale_b: float,
) -> dict[str, Any]:
    _require_bench_deps()
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    a_orig = torch.randn((M, K), generator=gen, dtype=torch.float32) * input_scale
    b_orig = torch.randn((K, N), generator=gen, dtype=torch.float32) * input_scale
    a_fp8_cpu = _torch_fp8(a_orig)
    b_fp8_cpu = _torch_fp8(b_orig)
    b_t_fp8_cpu = _torch_fp8(b_orig.t().contiguous())

    a_fp8_mps = a_fp8_cpu.to("mps")
    b_fp8_mps = b_fp8_cpu.to("mps")
    b_t_fp8_mps = b_t_fp8_cpu.to("mps")
    a_scale_mps = torch.tensor([scale_a], dtype=torch.float32, device="mps")
    b_scale_mps = torch.tensor([scale_b], dtype=torch.float32, device="mps")
    c_out_mps = torch.empty((M, N), dtype=torch.float32, device="mps")
    _sync_torch_mps()

    a_mx = mx.array(a_fp8_cpu.view(torch.uint8).numpy())
    b_t_mx = mx.array(np.ascontiguousarray(b_fp8_cpu.view(torch.uint8).numpy().T))
    x_mx = mx.array(np.ascontiguousarray(a_fp8_cpu[:1].view(torch.uint8).numpy().reshape(K)))
    scale_a_mx = mx.array([scale_a], dtype=mx.float32)
    scale_b_mx = mx.array([scale_b], dtype=mx.float32)
    mx.eval(a_mx, b_t_mx, x_mx, scale_a_mx, scale_b_mx)
    _sync_mlx()

    return {
        "a_orig": a_orig,
        "b_orig": b_orig,
        "a_fp8_cpu": a_fp8_cpu,
        "b_fp8_cpu": b_fp8_cpu,
        "a_fp8_mps": a_fp8_mps,
        "b_fp8_mps": b_fp8_mps,
        "b_t_fp8_mps": b_t_fp8_mps,
        "a_scale_mps": a_scale_mps,
        "b_scale_mps": b_scale_mps,
        "c_out_mps": c_out_mps,
        "a_mx": a_mx,
        "b_t_mx": b_t_mx,
        "x_mx": x_mx,
        "scale_a_mx": scale_a_mx,
        "scale_b_mx": scale_b_mx,
        "scale_a": scale_a,
        "scale_b": scale_b,
    }


def _max_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    _require_bench_deps()
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    denom = np.abs(b.astype(np.float32)) + 1.0e-6
    return {"max_abs": float(diff.max()), "max_rel": float((diff / denom).max())}


def _parity_for_matmul(inputs: dict[str, Any], actual: torch.Tensor | mx.array) -> dict[str, Any]:
    _require_bench_deps()
    a_ref = inputs["a_fp8_cpu"].to(torch.float32)
    b_ref = inputs["b_fp8_cpu"].to(torch.float32)
    ref = (a_ref @ b_ref).numpy() * float(inputs["scale_a"]) * float(inputs["scale_b"])
    if isinstance(actual, torch.Tensor):
        actual_np = actual.detach().cpu().numpy()
    else:
        actual_np = np.asarray(actual)
    return _max_error(actual_np, ref)


def _bench_path_b_matmul(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[BenchStats, mx.array | None]:
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    flops = 2.0 * M * N * K
    last: mx.array | None = None

    def run() -> None:
        nonlocal last
        last = fp8_scaled_matmul_raw(
            inputs["a_mx"],
            inputs["b_t_mx"],
            scale_a=inputs["scale_a_mx"],
            scale_b=inputs["scale_b_mx"],
        )
        mx.eval(last)

    stats = _bench_callable("path_b_msl_fp8_scaled_matmul", run, _sync_mlx, flops=flops, warmup=warmup, iters=iters)
    return stats, last


def _make_path_b_matmul_runner(
    inputs: dict[str, Any],
    last_ref: list[mx.array | None],
) -> Callable[[], None]:
    def run() -> None:
        last_ref[0] = fp8_scaled_matmul_raw(
            inputs["a_mx"],
            inputs["b_t_mx"],
            scale_a=inputs["scale_a_mx"],
            scale_b=inputs["scale_b_mx"],
        )
        mx.eval(last_ref[0])

    return run


def _make_path_c_scaled_matmul_runner(
    compiled: Callable[..., Any],
    inputs: dict[str, Any],
    last_ref: list[torch.Tensor | None],
) -> Callable[[], None]:
    c_out = inputs["c_out_mps"]

    def run() -> None:
        compiled(
            inputs["a_fp8_mps"],
            inputs["a_scale_mps"],
            inputs["b_fp8_mps"],
            inputs["b_scale_mps"],
            c_out,
        )
        last_ref[0] = c_out

    return run


def _bench_paired_scaled_matmul(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    skip_xcrun: bool,
    dump_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    b_label = "path_b_msl_fp8_scaled_matmul"
    c_label = "matmul_tl_fp8_scaled_matmul"
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    flops = 2.0 * M * N * K
    row_b: dict[str, Any] = {"label": b_label}
    row_c: dict[str, Any] = {"label": c_label, "variant": "T.fp8_scaled_matmul"}

    try:
        prim = _make_scaled_matmul_kernel(
            M=M,
            N=N,
            K=K,
            BM=int(shape["BM"]),
            BN=int(shape["BN"]),
            BK=int(shape["BK"]),
            num_stages=int(shape.get("num_stages", 0)),
        )
        src = _lower_source(prim)
        row_c["source_metrics"] = _source_metrics(src)
        row_c["xcrun_compile"] = (
            {"skipped": True, "reason": "--skip-xcrun"}
            if skip_xcrun
            else _xcrun_compile(src, label=c_label, dump_dir=dump_dir)
        )
        compiled = _compile_tilelang(prim)

        b_last_ref: list[mx.array | None] = [None]
        c_last_ref: list[torch.Tensor | None] = [None]
        paired = _bench_paired_callables(
            (
                (b_label, _make_path_b_matmul_runner(inputs, b_last_ref)),
                (c_label, _make_path_c_scaled_matmul_runner(compiled, inputs, c_last_ref)),
            ),
            _sync_all,
            flops=flops,
            warmup=warmup,
            iters=iters,
        )
        stats = paired.stats
        row_c["paired_ratios"] = paired.paired_ratios
        row_b["bench"] = asdict(stats[b_label])
        row_c["bench"] = asdict(stats[c_label])

        if stats[b_label].ok and b_last_ref[0] is not None and bool(shape.get("parity", False)):
            row_b["parity_vs_torch_cpu_dequant"] = _parity_for_matmul(inputs, b_last_ref[0])
        if (
            stats[c_label].ok
            and c_last_ref[0] is not None
            and b_last_ref[0] is not None
            and bool(shape.get("parity", False))
        ):
            _sync_all()
            row_c["parity_vs_path_b_msl"] = _max_error(
                c_last_ref[0].detach().cpu().numpy(),
                np.asarray(b_last_ref[0]),
            )
            row_c["parity_vs_torch_cpu_dequant"] = _parity_for_matmul(inputs, c_last_ref[0])
    except Exception as exc:
        row_b.setdefault(
            "bench",
            asdict(
                BenchStats(
                    label=b_label,
                    ok=False,
                    warmup=warmup,
                    iters=iters,
                    paired=True,
                    error=f"Path C setup failed before paired timing: {type(exc).__name__}: {exc}",
                )
            ),
        )
        row_c["bench"] = asdict(
            BenchStats(
                label=c_label,
                ok=False,
                warmup=warmup,
                iters=iters,
                paired=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        row_c["traceback"] = traceback.format_exc(limit=12)
    return row_b, row_c


def _bench_path_b_vecmat(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[BenchStats, mx.array | None]:
    N, K = int(shape["N"]), int(shape["K"])
    flops = 2.0 * N * K
    last: mx.array | None = None

    def run() -> None:
        nonlocal last
        last = fp8_scaled_vecmat(
            inputs["x_mx"],
            inputs["b_t_mx"],
            scale_x=inputs["scale_a_mx"],
            scale_w=inputs["scale_b_mx"],
        )
        mx.eval(last)

    stats = _bench_callable("path_b_msl_fp8_scaled_vecmat", run, _sync_mlx, flops=flops, warmup=warmup, iters=iters)
    return stats, last


def _make_path_b_vecmat_runner(
    inputs: dict[str, Any],
    last_ref: list[mx.array | None],
) -> Callable[[], None]:
    def run() -> None:
        last_ref[0] = fp8_scaled_vecmat(
            inputs["x_mx"],
            inputs["b_t_mx"],
            scale_x=inputs["scale_a_mx"],
            scale_w=inputs["scale_b_mx"],
        )
        mx.eval(last_ref[0])

    return run


def _make_path_c_vecmat_runner(
    inputs: dict[str, Any],
    last_ref: list[mx.array | None],
) -> Callable[[], None]:
    def run() -> None:
        out = fp8_scaled_vecmat_path_c(
            inputs["x_mx"],
            inputs["b_t_mx"],
            scale_x=inputs["scale_a_mx"],
            scale_w=inputs["scale_b_mx"],
        )
        if out is None:
            raise RuntimeError("fp8_scaled_vecmat_path_c returned None")
        last_ref[0] = out
        mx.eval(last_ref[0])

    return run


def _bench_paired_vecmat_mlx(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[tuple[BenchStats, mx.array | None], tuple[dict[str, Any], mx.array | None]]:
    N, K = int(shape["N"]), int(shape["K"])
    flops = 2.0 * N * K
    b_last_ref: list[mx.array | None] = [None]
    c_last_ref: list[mx.array | None] = [None]
    b_label = "path_b_msl_fp8_scaled_vecmat"
    c_label = "path_c_mlx_tilelang_fp8_scaled_vecmat"
    paired = _bench_paired_callables(
        (
            (b_label, _make_path_b_vecmat_runner(inputs, b_last_ref)),
            (c_label, _make_path_c_vecmat_runner(inputs, c_last_ref)),
        ),
        _sync_mlx,
        flops=flops,
        warmup=warmup,
        iters=iters,
    )
    stats = paired.stats
    row_c: dict[str, Any] = {
        "label": c_label,
        "variant": "MLX-dispatched TileLang fp8 vecmat packed dot4",
        "target": TILELANG_METAL_VECMAT_TARGET,
        "bench": asdict(stats[c_label]),
        "paired_ratios": paired.paired_ratios,
    }
    if stats[c_label].ok:
        try:
            prim = make_fp8_vecmat_reduce_kernel(N=N, K=K)
            src = _lower_source(prim, target=TILELANG_METAL_VECMAT_TARGET)
            row_c["source_metrics"] = _path_c_vecmat_runtime_source_metrics(N=N, K=K)
            row_c["diagnostic_tilelang_source_metrics"] = _source_metrics(src)
            row_c["path_c_blockers"] = fp8_vecmat_msl_blockers(src)
        except Exception as exc:  # pragma: no cover - profiling metadata only
            row_c["source_metrics_error"] = f"{type(exc).__name__}: {exc}"
    return (stats[b_label], b_last_ref[0]), (row_c, c_last_ref[0])


def _bench_path_c_vecmat_mlx(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], mx.array | None]:
    N, K = int(shape["N"]), int(shape["K"])
    flops = 2.0 * N * K
    last: mx.array | None = None
    label = "path_c_mlx_tilelang_fp8_scaled_vecmat"

    def run() -> None:
        nonlocal last
        out = fp8_scaled_vecmat_path_c(
            inputs["x_mx"],
            inputs["b_t_mx"],
            scale_x=inputs["scale_a_mx"],
            scale_w=inputs["scale_b_mx"],
        )
        if out is None:
            raise RuntimeError("fp8_scaled_vecmat_path_c returned None")
        last = out
        mx.eval(last)

    stats = _bench_callable(label, run, _sync_mlx, flops=flops, warmup=warmup, iters=iters)
    row: dict[str, Any] = {
        "label": label,
        "variant": "MLX-dispatched TileLang fp8 vecmat packed dot4",
        "target": TILELANG_METAL_VECMAT_TARGET,
        "bench": asdict(stats),
    }
    if stats.ok:
        try:
            prim = make_fp8_vecmat_reduce_kernel(N=N, K=K)
            src = _lower_source(prim, target=TILELANG_METAL_VECMAT_TARGET)
            row["source_metrics"] = _path_c_vecmat_runtime_source_metrics(N=N, K=K)
            row["diagnostic_tilelang_source_metrics"] = _source_metrics(src)
            row["path_c_blockers"] = fp8_vecmat_msl_blockers(src)
        except Exception as exc:  # pragma: no cover - profiling metadata only
            row["source_metrics_error"] = f"{type(exc).__name__}: {exc}"
    return row, last


def _bench_path_c_scaled_matmul(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    skip_xcrun: bool,
    dump_dir: Path | None,
    include_vecmat_diagnostics: bool,
    path_b_last: mx.array | None = None,
) -> dict[str, Any]:
    label = f"{shape['kind']}_tl_fp8_scaled_matmul"
    M, N, K = int(shape["M"]), int(shape["N"]), int(shape["K"])
    flops = 2.0 * M * N * K
    result: dict[str, Any] = {"label": label, "variant": "T.fp8_scaled_matmul"}
    try:
        prim = _make_scaled_matmul_kernel(
            M=M,
            N=N,
            K=K,
            BM=int(shape["BM"]),
            BN=int(shape["BN"]),
            BK=int(shape["BK"]),
            num_stages=int(shape.get("num_stages", 0)),
        )
        src = _lower_source(prim)
        result["source_metrics"] = _source_metrics(src)
        result["xcrun_compile"] = (
            {"skipped": True, "reason": "--skip-xcrun"}
            if skip_xcrun
            else _xcrun_compile(src, label=label, dump_dir=dump_dir)
        )
        compiled = _compile_tilelang(prim)
        c_out = inputs["c_out_mps"]

        def run() -> None:
            compiled(
                inputs["a_fp8_mps"],
                inputs["a_scale_mps"],
                inputs["b_fp8_mps"],
                inputs["b_scale_mps"],
                c_out,
            )

        stats = _bench_callable(label, run, _sync_torch_mps, flops=flops, warmup=warmup, iters=iters)
        result["bench"] = asdict(stats)
        if stats.ok and bool(shape.get("parity", False)):
            _sync_torch_mps()
            if path_b_last is not None:
                result["parity_vs_path_b_msl"] = _max_error(
                    c_out.detach().cpu().numpy(),
                    np.asarray(path_b_last),
                )
            result["parity_vs_torch_cpu_dequant"] = _parity_for_matmul(inputs, c_out)
    except Exception as exc:
        result["bench"] = asdict(
            BenchStats(label=label, ok=False, warmup=warmup, iters=iters, error=f"{type(exc).__name__}: {exc}")
        )
        result["traceback"] = traceback.format_exc(limit=12)
    return result


def _bench_path_c_vecmat_reduce(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    skip_xcrun: bool,
    dump_dir: Path | None,
) -> dict[str, Any]:
    label = "vecmat_tl_reduce_fp8"
    N, K = int(shape["N"]), int(shape["K"])
    flops = 2.0 * N * K
    result: dict[str, Any] = {
        "label": label,
        "variant": "TileLang fp8 vecmat thread_allreduce",
        "target": TILELANG_METAL_VECMAT_TARGET,
    }
    try:
        prim = make_fp8_vecmat_reduce_kernel(N=N, K=K)
        src = _lower_source(prim, target=TILELANG_METAL_VECMAT_TARGET)
        result["source_metrics"] = _source_metrics(src)
        result["path_c_blockers"] = fp8_vecmat_msl_blockers(src)
        result["xcrun_compile"] = (
            {"skipped": True, "reason": "--skip-xcrun"}
            if skip_xcrun
            else _xcrun_compile(src, label=label, dump_dir=dump_dir)
        )
        compiled = _compile_tilelang(prim, target=TILELANG_METAL_VECMAT_TARGET)
        c_out = inputs["c_out_mps"]

        def run() -> None:
            compiled(
                inputs["a_fp8_mps"],
                inputs["a_scale_mps"],
                inputs["b_t_fp8_mps"],
                inputs["b_scale_mps"],
                c_out,
            )

        stats = _bench_callable(label, run, _sync_torch_mps, flops=flops, warmup=warmup, iters=iters)
        result["bench"] = asdict(stats)
        if stats.ok and bool(shape.get("parity", False)):
            _sync_torch_mps()
            result["parity_vs_torch_cpu_dequant"] = _parity_for_matmul(inputs, c_out)
    except Exception as exc:
        result["bench"] = asdict(
            BenchStats(label=label, ok=False, warmup=warmup, iters=iters, error=f"{type(exc).__name__}: {exc}")
        )
        result["traceback"] = traceback.format_exc(limit=12)
    return result


def _compare_ratios(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    benches: dict[str, float] = {}
    paired: dict[str, dict[str, float]] = {}
    for row in rows:
        label = str(row.get("label", ""))
        bench = row.get("bench") or {}
        if bench.get("ok") and bench.get("median_ms") is not None:
            benches[str(row.get("label", bench.get("label")))] = float(bench["median_ms"])
        paired_ratios = row.get("paired_ratios") or {}
        if isinstance(paired_ratios, dict):
            paired[label] = {
                str(name): float(value)
                for name, value in paired_ratios.items()
                if isinstance(value, int | float) and math.isfinite(float(value))
            }
    out: dict[str, float] = {}
    b_matmul = benches.get("path_b_msl_fp8_scaled_matmul")
    b_vecmat = benches.get("path_b_msl_fp8_scaled_vecmat")
    for label, median in benches.items():
        if label.startswith("path_b_"):
            continue
        base_label = (
            "path_b_msl_fp8_scaled_vecmat"
            if "vecmat" in label
            else "path_b_msl_fp8_scaled_matmul"
        )
        paired_key = f"{label}_over_{base_label}_paired_median"
        paired_ratio = paired.get(label, {}).get(paired_key)
        if paired_ratio is not None:
            out[f"{label}_over_path_b"] = paired_ratio
            continue
        base = b_vecmat if "vecmat" in label and b_vecmat is not None else b_matmul
        if base is not None and base > 0:
            out[f"{label}_over_path_b"] = median / base
    return out


def _finite_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _bench_ok(labels: dict[str, dict[str, Any]], label: str) -> bool:
    return bool(labels.get(label, {}).get("bench", {}).get("ok"))


def _status_payload(status: Any, fields: Iterable[str] | None = None) -> dict[str, Any]:
    if isinstance(status, dict):
        data = dict(status)
    elif fields is None:
        data = asdict(status)
    else:
        data = {field: getattr(status, field) for field in fields if hasattr(status, field)}
    if "available" in data:
        data["available"] = bool(data["available"])
    features = data.get("features")
    if isinstance(features, dict):
        data["features"] = dict(features)
    return data


def _full_dispatch_strict_failures(
    payload: dict[str, Any],
    *,
    status_key: str,
    label: str,
) -> list[str]:
    status = payload[status_key]
    failures: list[str] = []
    if not status["available"]:
        failures.append(f"{status_key}.available=false blocks full Path C {label} dispatch")
        return failures
    features = status.get("features", {})
    dispatch_surface = features.get("dispatch_surface")
    if dispatch_surface != "full_fwd_bwd":
        failures.append(
            f"{status_key}.features.dispatch_surface={dispatch_surface!r} "
            f"is not full_fwd_bwd Path C {label} dispatch"
        )
    if features.get("full_fwd_bwd_available") is not True:
        failures.append(f"{status_key}.features.full_fwd_bwd_available is not true")
    return failures


def _sparse_path_c_status_payload(
    *,
    fp8_qk_status: Any,
    fp8_qk_reduce_status: Any,
    fp8_indexed_qk_reduce_status: Any,
    e8m0_qk_status: Any,
    e8m0_qk_reduce_status: Any,
) -> dict[str, Any]:
    payload = {
        "path_c_tilelang_qk_status": _status_payload(
            fp8_qk_status,
            fields=("available", "reason", "target", "m", "n", "k", "transpose_B", "features"),
        ),
        "path_c_tilelang_qk_reduce_status": _status_payload(
            fp8_qk_reduce_status,
            fields=("available", "reason", "target", "n", "k", "outputs_per_block", "reduce_threads", "vec", "features"),
        ),
        "path_c_tilelang_indexed_qk_reduce_status": _status_payload(
            fp8_indexed_qk_reduce_status,
            fields=(
                "available",
                "reason",
                "target",
                "batch",
                "seq_len",
                "heads",
                "seq_len_kv",
                "kv_group",
                "head_kv",
                "topk",
                "k",
                "outputs_per_block",
                "reduce_threads",
                "vec",
                "features",
            ),
        ),
        "path_c_tilelang_e8m0_qk_status": _status_payload(
            e8m0_qk_status,
            fields=(
                "available",
                "reason",
                "target",
                "m",
                "n",
                "k",
                "transpose_B",
                "scale_block_size",
                "scale_layout",
                "features",
            ),
        ),
        "path_c_tilelang_e8m0_qk_reduce_status": _status_payload(
            e8m0_qk_reduce_status,
            fields=(
                "available",
                "reason",
                "target",
                "n",
                "k",
                "outputs_per_block",
                "reduce_threads",
                "vec",
                "scale_block_size",
                "scale_layout",
                "features",
            ),
        ),
    }
    payload["path_c_status"] = payload["path_c_tilelang_qk_status"]
    failures = _full_dispatch_strict_failures(
        payload,
        status_key="path_c_tilelang_qk_status",
        label="FP8",
    )
    failures.extend(
        _full_dispatch_strict_failures(
            payload,
            status_key="path_c_tilelang_e8m0_qk_status",
            label="blockscaled",
        )
    )
    payload["strict"] = {
        "enabled": True,
        "scope": "full_path_c_dispatch",
        "passed": not failures,
        "failures": failures,
    }
    return payload


def _shape_row_strict_ok(
    shape_row: dict[str, Any],
    *,
    max_ratio: float = 1.0,
    parity_max_abs: float = DEFAULT_PARITY_MAX_ABS,
    parity_max_rel: float = DEFAULT_PARITY_MAX_REL,
) -> bool:
    labels = {str(row.get("label")): row for row in shape_row.get("rows", [])}
    kind = shape_row.get("shape", {}).get("kind")
    if kind == "vecmat":
        path_b_label = "path_b_msl_fp8_scaled_vecmat"
        path_c_label = "path_c_mlx_tilelang_fp8_scaled_vecmat"
    else:
        path_b_label = "path_b_msl_fp8_scaled_matmul"
        path_c_label = "matmul_tl_fp8_scaled_matmul"
    ratio_key = f"{path_c_label}_over_path_b"
    ratio = shape_row.get("ratios", {}).get(ratio_key)
    if kind == "vecmat":
        # Vecmat timings are collected as paired A/B launches in the same
        # loop. Use that paired median for the strict gate so launch-order
        # jitter cannot fail an otherwise faster paired run.
        paired_ratio_key = f"{path_c_label}_over_{path_b_label}_paired_median"
        paired_ratio = labels.get(path_c_label, {}).get("paired_ratios", {}).get(
            paired_ratio_key
        )
        if _finite_float(paired_ratio):
            ratio = paired_ratio
    if not _bench_ok(labels, path_b_label) or not _bench_ok(labels, path_c_label):
        return False
    if not _finite_float(ratio) or ratio > max_ratio:
        return False

    shape = shape_row.get("shape", {})
    parity_required = bool(shape_row.get("parity_required", shape.get("parity", True)))
    if parity_required:
        parity = labels.get(path_c_label, {}).get("parity_vs_path_b_msl")
        if not isinstance(parity, dict):
            return False
        max_abs = parity.get("max_abs")
        max_rel = parity.get("max_rel")
        if not isinstance(max_abs, float) or not isinstance(max_rel, float):
            return False
        if not math.isfinite(max_abs) or not math.isfinite(max_rel):
            return False
        return max_abs <= parity_max_abs and max_rel <= parity_max_rel
    return True


def _bench_shape(
    shape_name: str,
    shape: Shape,
    *,
    warmup: int,
    iters: int,
    seed: int,
    input_scale: float,
    scale_a: float,
    scale_b: float,
    skip_xcrun: bool,
    dump_dir: Path | None,
    include_vecmat_diagnostics: bool,
) -> dict[str, Any]:
    print(f"[bench] {shape_name}: M={shape['M']} N={shape['N']} K={shape['K']} kind={shape['kind']}")
    inputs = _build_inputs(shape, seed=seed, input_scale=input_scale, scale_a=scale_a, scale_b=scale_b)
    rows: list[dict[str, Any]] = []

    if shape["kind"] == "vecmat":
        (b_stats, b_last), (row_c_mlx, c_mlx_last) = _bench_paired_vecmat_mlx(
            shape,
            inputs,
            warmup=warmup,
            iters=iters,
        )
        row_b = {"label": "path_b_msl_fp8_scaled_vecmat", "bench": asdict(b_stats)}
        if b_stats.ok and b_last is not None and bool(shape.get("parity", False)):
            ref = _parity_for_matmul(inputs, b_last.reshape(1, int(shape["N"])))
            row_b["parity_vs_torch_cpu_dequant"] = ref
        rows.append(row_b)
        if (
            row_c_mlx.get("bench", {}).get("ok")
            and c_mlx_last is not None
            and b_last is not None
            and bool(shape.get("parity", False))
        ):
            row_c_mlx["parity_vs_path_b_msl"] = _max_error(np.asarray(c_mlx_last), np.asarray(b_last))
            row_c_mlx["parity_vs_torch_cpu_dequant"] = _parity_for_matmul(
                inputs,
                c_mlx_last.reshape(1, int(shape["N"])),
            )
        rows.append(row_c_mlx)
        if include_vecmat_diagnostics:
            rows.append(
                _bench_path_c_scaled_matmul(
                    shape,
                    inputs,
                    warmup=warmup,
                    iters=iters,
                    skip_xcrun=skip_xcrun,
                    dump_dir=dump_dir,
                    include_vecmat_diagnostics=include_vecmat_diagnostics,
                    path_b_last=None,
                )
            )
            rows.append(
                _bench_path_c_vecmat_reduce(
                    shape,
                    inputs,
                    warmup=warmup,
                    iters=iters,
                    skip_xcrun=skip_xcrun,
                    dump_dir=dump_dir,
                )
            )
    else:
        row_b, row_c = _bench_paired_scaled_matmul(
            shape,
            inputs,
            warmup=warmup,
            iters=iters,
            skip_xcrun=skip_xcrun,
            dump_dir=dump_dir,
        )
        rows.append(row_b)
        rows.append(row_c)

    return {
        "shape_name": shape_name,
        "shape": {k: v for k, v in shape.items() if k != "parity"},
        "parity_required": bool(shape.get("parity", False)),
        "rows": rows,
        "ratios": _compare_ratios(rows),
    }


def _bench_sparse_status(*, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    _require_bench_deps()
    try:
        from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
            blockscaled_sparse_mla_qk_path_c_status,
            blockscaled_sparse_mla_qk_reduce_path_c_status,
        )
        from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
            sparse_mla_fp8_fwd_metal,
            sparse_mla_fp8_metal_status,
        )
        from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
            fp8_sparse_mla_indexed_qk_reduce_path_c_status,
            fp8_sparse_mla_qk_path_c_status,
            fp8_sparse_mla_qk_reduce_path_c_status,
        )
        from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference

        path_c_payload = _sparse_path_c_status_payload(
            fp8_qk_status=fp8_sparse_mla_qk_path_c_status(),
            fp8_qk_reduce_status=fp8_sparse_mla_qk_reduce_path_c_status(),
            fp8_indexed_qk_reduce_status=fp8_sparse_mla_indexed_qk_reduce_path_c_status(),
            e8m0_qk_status=blockscaled_sparse_mla_qk_path_c_status(),
            e8m0_qk_reduce_status=blockscaled_sparse_mla_qk_reduce_path_c_status(),
        )
        rng = np.random.default_rng(seed)
        q = mx.array((rng.standard_normal((1, 64, 4, 64)) * 0.1).astype(np.float16))
        kv = mx.array((rng.standard_normal((1, 64, 1, 64)) * 0.1).astype(np.float16))
        indices_np = np.tile(np.arange(16, dtype=np.int32).reshape(1, 1, 1, 16), (1, 64, 1, 1))
        indices_np[:, :, :, 8:] = -1
        indices = mx.array(indices_np)
        mx.eval(q, kv, indices)
        _sync_mlx()
        status = sparse_mla_fp8_metal_status(q, kv, indices)
        result: dict[str, Any] = {
            "path_b_status": {"available": status.available, "reason": status.reason},
            **path_c_payload,
        }
        if status.available:
            def run_b() -> None:
                out = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=32)
                if out is None:
                    raise RuntimeError("sparse_mla_fp8_fwd_metal returned None")
                mx.eval(out[0], out[1])

            stats = _bench_callable(
                "path_b_sparse_mla_fp8_fwd",
                run_b,
                _sync_mlx,
                flops=0.0,
                warmup=warmup,
                iters=iters,
            )
            result["path_b_bench"] = asdict(stats)
            ref = sparse_mla_attention_reference(q, kv, indices, d_v=32)
            metal = sparse_mla_fp8_fwd_metal(q, kv, indices, d_v=32)
            if metal is not None:
                mx.eval(ref, metal[0])
                result["path_b_parity_vs_bf16_reference"] = _max_error(np.asarray(metal[0]), np.asarray(ref))
        return result
    except Exception as exc:
        unavailable = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "features": {"dispatch_surface": "unavailable", "full_fwd_bwd_available": False},
        }
        return {
            "path_b_status": {"available": False, "reason": f"{type(exc).__name__}: {exc}"},
            **_sparse_path_c_status_payload(
                fp8_qk_status=unavailable,
                fp8_qk_reduce_status=unavailable,
                fp8_indexed_qk_reduce_status=unavailable,
                e8m0_qk_status=unavailable,
                e8m0_qk_reduce_status=unavailable,
            ),
            "traceback": traceback.format_exc(limit=8),
        }


def _print_summary(payload: dict[str, Any]) -> None:
    print("\n=== Path C vs Path B FP8 Summary ===")
    for row in payload["results"]:
        print(f"\n{row['shape_name']} {row['shape']}")
        for item in row["rows"]:
            bench = item.get("bench") or {}
            if bench.get("ok"):
                print(
                    f"  {item['label']}: median={bench['median_ms']:.6f} ms "
                    f"p90={bench['p90_ms']:.6f} ms tflops={bench['tflops']:.6f}"
                )
            else:
                print(f"  {item['label']}: FAIL {bench.get('error')}")
            metrics = item.get("source_metrics", {}).get("markers")
            if metrics:
                print(
                    "    markers: "
                    f"mma={metrics['simdgroup_multiply_accumulate']} "
                    f"loads={metrics['simdgroup_load']} stores={metrics['simdgroup_store']} "
                    f"tg_half={metrics['threadgroup_half']} "
                    f"allreduce={metrics['tvm_thread_allreduce']} "
                    f"packed_uint={metrics['packed_uint_loads']} "
                    f"lut={metrics['fp8_e4m3_lut']} "
                    f"simd_sum={metrics['simd_sum']} "
                    f"scalar_a={metrics['scalar_float_a_val']}"
                )
            blockers = item.get("path_c_blockers")
            if blockers and not blockers.get("path_b_fast_path_ready", False):
                print(f"    path-c blockers: {', '.join(blockers.get('missing', []))}")
        for ratio_name, value in row.get("ratios", {}).items():
            print(f"  ratio {ratio_name}={value:.3f}x")
    sparse = payload.get("sparse_mla")
    if sparse:
        print("\nSparse-MLA FP8:")
        print(f"  Path B: {sparse['path_b_status']}")
        print(f"  Path C full FP8 dispatch: {sparse.get('path_c_tilelang_qk_status')}")
        print(f"  Path C FP8 QK reducer: {sparse.get('path_c_tilelang_qk_reduce_status')}")
        print(f"  Path C FP8 indexed QK reducer: {sparse.get('path_c_tilelang_indexed_qk_reduce_status')}")
        print(f"  Path C e8m0 full dispatch: {sparse.get('path_c_tilelang_e8m0_qk_status')}")
        print(f"  Path C e8m0 QK reducer: {sparse.get('path_c_tilelang_e8m0_qk_reduce_status')}")
        print(f"  Full Path C strict gate: {sparse.get('strict')}")
        bench = sparse.get("path_b_bench")
        if bench and bench.get("ok"):
            print(f"  Path B median={bench['median_ms']:.6f} ms p90={bench['p90_ms']:.6f} ms")


def main() -> int:
    _require_bench_deps()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=2.0)
    parser.add_argument("--scale-a", type=float, default=1.0)
    parser.add_argument("--scale-b", type=float, default=1.0)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["matmul_128", "vecmat_4096"],
        choices=sorted(SHAPES),
        help="Shape labels to run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "bench" / "tilelang_ports" / "fp8_path_c_vs_path_b.json",
    )
    parser.add_argument("--dump-msl", type=Path, default=None, help="Directory for emitted .metal/.air files.")
    parser.add_argument("--skip-xcrun", action="store_true", help="Skip offline xcrun Metal compilation.")
    parser.add_argument("--skip-sparse", action="store_true", help="Skip Sparse-MLA Path B status/bench probe.")
    parser.add_argument(
        "--include-vecmat-diagnostics",
        action="store_true",
        help="Also run legacy Torch/TileLang vecmat diagnostic routes; off by default because they do not match production Path C dispatch.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any requested Path C benchmark fails.")
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed Path C / Path B median ratio for --strict.",
    )
    parser.add_argument(
        "--parity-max-abs",
        type=float,
        default=DEFAULT_PARITY_MAX_ABS,
        help="Maximum allowed Path C vs Path B max_abs parity error for --strict.",
    )
    parser.add_argument(
        "--parity-max-rel",
        type=float,
        default=DEFAULT_PARITY_MAX_REL,
        help="Maximum allowed Path C vs Path B max_rel parity error for --strict.",
    )
    args = parser.parse_args()

    _require_runtime()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "path_c_vs_path_b_fp8_profile",
        "host": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx": getattr(mx, "__version__", "unknown"),
            "torch": getattr(torch, "__version__", "unknown"),
        },
        "env_contract": {
            "MAKEFLAGS": os.environ.get("MAKEFLAGS"),
            "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "repos": {
            "cppmega_mlx": _git_meta(REPO_ROOT),
            "tilelang": _git_meta(TILELANG_ROOT),
            "tvm": _git_meta(TVM_ROOT),
        },
        "path_b_status": asdict(fp8_msl_status()),
        "strict_policy": {
            "path_c_over_path_b_max_ratio": float(args.max_ratio),
            "path_c_vs_path_b_parity_max_abs": float(args.parity_max_abs),
            "path_c_vs_path_b_parity_max_rel": float(args.parity_max_rel),
            "requires_path_b_and_path_c": True,
        },
        "results": [],
    }

    for shape_name in args.shapes:
        payload["results"].append(
            _bench_shape(
                shape_name,
                SHAPES[shape_name],
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                input_scale=args.input_scale,
                scale_a=args.scale_a,
                scale_b=args.scale_b,
                skip_xcrun=args.skip_xcrun,
                dump_dir=args.dump_msl,
                include_vecmat_diagnostics=args.include_vecmat_diagnostics,
            )
        )

    if not args.skip_sparse:
        payload["sparse_mla"] = _bench_sparse_status(warmup=args.warmup, iters=args.iters, seed=args.seed)

    if args.strict:
        for shape_row in payload["results"]:
            if not _shape_row_strict_ok(
                shape_row,
                max_ratio=float(args.max_ratio),
                parity_max_abs=float(args.parity_max_abs),
                parity_max_rel=float(args.parity_max_rel),
            ):
                print(
                    "strict FP8 Path C gate failed: "
                    f"{shape_row['shape_name']} ratios={shape_row.get('ratios', {})}",
                    file=sys.stderr,
                )
                return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    _print_summary(payload)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
