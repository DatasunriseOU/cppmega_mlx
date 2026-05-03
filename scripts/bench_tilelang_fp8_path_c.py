"""Profile TileLang Path C FP8 kernels against Path B hand-written MSL.

This is Lane F's local M4/MPS harness. It intentionally lives outside the
TileLang/TVM trees: the script profiles the current apple-head TileLang Metal
lowering without modifying core codegen.

Default output:
    bench/tilelang_ports/fp8_path_c_vs_path_b.json
"""

from __future__ import annotations

import argparse
import json
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT", "/private/tmp/tilelang_apple_head/tilelang"))
TVM_ROOT = Path(os.environ.get("TVM_ROOT", "/private/tmp/tvm_apple_head/tvm"))
TILELANG_METAL_TARGET = "metal"
TILELANG_METAL_VECMAT_TARGET = "metal -thread_warp_size=32"

sys.path.insert(0, str(REPO_ROOT))
if TILELANG_ROOT.exists():
    sys.path.insert(0, str(TILELANG_ROOT))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (  # noqa: E402
    fp8_msl_status,
    fp8_scaled_matmul_raw,
    fp8_scaled_vecmat,
)


Shape = dict[str, Any]

SHAPES: dict[str, Shape] = {
    "tiny_128": {
        "kind": "matmul",
        "M": 128,
        "N": 128,
        "K": 128,
        "BM": 32,
        "BN": 32,
        "BK": 32,
        "num_stages": 0,
        "parity": True,
    },
    "matmul_128": {
        "kind": "matmul",
        "M": 128,
        "N": 128,
        "K": 128,
        "BM": 32,
        "BN": 32,
        "BK": 32,
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
    error: str | None = None


def _sync_mlx() -> None:
    sync = getattr(mx, "synchronize", None)
    if sync is not None:
        sync()


def _sync_torch_mps() -> None:
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
        )
    except Exception as exc:  # pragma: no cover - used for local profiling receipts
        return BenchStats(
            label=label,
            ok=False,
            warmup=warmup,
            iters=iters,
            error=f"{type(exc).__name__}: {exc}",
        )


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
    if mx.default_device() != mx.gpu:
        raise RuntimeError(f"MLX default device is not GPU: {mx.default_device()}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch.backends.mps.is_available() is false")
    status = fp8_msl_status()
    if not status.available:
        raise RuntimeError(f"Path B FP8 MSL unavailable: {status.reason}")


def _import_tilelang() -> tuple[Any, Any, Any, Any]:
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
        _M=M,
        _N=N,
        _K=K,
        _BM=BM,
        _BN=BN,
        _BK=BK,
        _NUM_STAGES=num_stages,
    )

    @T.prim_func
    def fp8_scaled_kernel(
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

    return fp8_scaled_kernel


def _make_vecmat_reduce_kernel(*, N: int, K: int, outputs_per_block: int = 4):
    """Build the experimental Path C vecmat DSL reducer from the local probe."""

    _, T, _, _ = _import_tilelang()
    reduce_threads = 32
    vec = 4
    block_k = reduce_threads * vec
    g = globals()
    g.update(
        _VM_N=N,
        _VM_K=K,
        _VM_NP=outputs_per_block,
        _VM_RT=reduce_threads,
        _VM_VEC=vec,
        _VM_BLOCK_K=block_k,
    )

    @T.prim_func
    def fp8_vecmat_reduce(
        A: T.Tensor((1, _VM_K), "float8_e4m3"),
        A_scale: T.Tensor((1,), "float32"),
        B: T.Tensor((_VM_N, _VM_K), "float8_e4m3"),
        B_scale: T.Tensor((1,), "float32"),
        C: T.Tensor((1, _VM_N), "float32"),
    ):
        with T.Kernel(T.ceildiv(_VM_N, _VM_NP), threads=(_VM_RT, _VM_NP)) as bx:
            accum = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            kr = T.get_thread_binding(0)
            ni = T.get_thread_binding(1)
            col = bx * _VM_NP + ni
            T.clear(accum)
            for ko in T.serial(T.ceildiv(_VM_K, _VM_BLOCK_K)):
                for v in T.serial(_VM_VEC):
                    k = ko * _VM_BLOCK_K + kr * _VM_VEC + v
                    if col < _VM_N and k < _VM_K:
                        accum[0] += T.cast(A[0, k], "float32") * T.cast(B[col, k], "float32")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum[0],
                        True,
                        reduced[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0 and col < _VM_N:
                C[0, col] = reduced[0] * A_scale[0] * B_scale[0]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_vecmat_reduce)
    except Exception:
        return fp8_vecmat_reduce


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
    }
    return {"source_len": len(src), "markers": markers}


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
    return x.detach().cpu().to(torch.float8_e4m3fn)


def _build_inputs(
    shape: Shape,
    *,
    seed: int,
    input_scale: float,
    scale_a: float,
    scale_b: float,
) -> dict[str, Any]:
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
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    denom = np.abs(b.astype(np.float32)) + 1.0e-6
    return {"max_abs": float(diff.max()), "max_rel": float((diff / denom).max())}


def _parity_for_matmul(inputs: dict[str, Any], actual: torch.Tensor | mx.array) -> dict[str, Any]:
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


def _bench_path_c_scaled_matmul(
    shape: Shape,
    inputs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    skip_xcrun: bool,
    dump_dir: Path | None,
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
        prim = _make_vecmat_reduce_kernel(N=N, K=K)
        src = _lower_source(prim, target=TILELANG_METAL_VECMAT_TARGET)
        result["source_metrics"] = _source_metrics(src)
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
    for row in rows:
        bench = row.get("bench") or {}
        if bench.get("ok") and bench.get("median_ms") is not None:
            benches[str(row.get("label", bench.get("label")))] = float(bench["median_ms"])
    out: dict[str, float] = {}
    b_matmul = benches.get("path_b_msl_fp8_scaled_matmul")
    b_vecmat = benches.get("path_b_msl_fp8_scaled_vecmat")
    for label, median in benches.items():
        if label.startswith("path_b_"):
            continue
        base = b_vecmat if "vecmat" in label and b_vecmat is not None else b_matmul
        if base is not None and base > 0:
            out[f"{label}_over_path_b"] = median / base
    return out


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
) -> dict[str, Any]:
    print(f"[bench] {shape_name}: M={shape['M']} N={shape['N']} K={shape['K']} kind={shape['kind']}")
    inputs = _build_inputs(shape, seed=seed, input_scale=input_scale, scale_a=scale_a, scale_b=scale_b)
    rows: list[dict[str, Any]] = []

    if shape["kind"] == "vecmat":
        b_stats, b_last = _bench_path_b_vecmat(shape, inputs, warmup=warmup, iters=iters)
        row_b = {"label": "path_b_msl_fp8_scaled_vecmat", "bench": asdict(b_stats)}
        if b_stats.ok and b_last is not None and bool(shape.get("parity", False)):
            ref = _parity_for_matmul(inputs, b_last.reshape(1, int(shape["N"])))
            row_b["parity_vs_torch_cpu_dequant"] = ref
        rows.append(row_b)
        rows.append(
            _bench_path_c_scaled_matmul(
                shape,
                inputs,
                warmup=warmup,
                iters=iters,
                skip_xcrun=skip_xcrun,
                dump_dir=dump_dir,
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
        b_stats, b_last = _bench_path_b_matmul(shape, inputs, warmup=warmup, iters=iters)
        row_b = {"label": "path_b_msl_fp8_scaled_matmul", "bench": asdict(b_stats)}
        if b_stats.ok and b_last is not None and bool(shape.get("parity", False)):
            row_b["parity_vs_torch_cpu_dequant"] = _parity_for_matmul(inputs, b_last)
        rows.append(row_b)
        rows.append(
            _bench_path_c_scaled_matmul(
                shape,
                inputs,
                warmup=warmup,
                iters=iters,
                skip_xcrun=skip_xcrun,
                dump_dir=dump_dir,
            )
        )

    return {
        "shape_name": shape_name,
        "shape": {k: v for k, v in shape.items() if k != "parity"},
        "rows": rows,
        "ratios": _compare_ratios(rows),
    }


def _bench_sparse_status(*, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    try:
        from cppmega_mlx.nn._tilelang.sparse_mla_fp8 import (
            sparse_mla_fp8_fwd_metal,
            sparse_mla_fp8_metal_status,
        )
        from cppmega_mlx.nn.sparse_mla import sparse_mla_attention_reference

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
            "path_c_status": {
                "available": False,
                "reason": "No checked-in TileLang DSL Sparse-MLA Path C reference found in this lane.",
            },
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
        return {
            "path_b_status": {"available": False, "reason": f"{type(exc).__name__}: {exc}"},
            "path_c_status": {
                "available": False,
                "reason": "No checked-in TileLang DSL Sparse-MLA Path C reference found in this lane.",
            },
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
                    f"scalar_a={metrics['scalar_float_a_val']}"
                )
        for ratio_name, value in row.get("ratios", {}).items():
            print(f"  ratio {ratio_name}={value:.3f}x")
    sparse = payload.get("sparse_mla")
    if sparse:
        print("\nSparse-MLA FP8:")
        print(f"  Path B: {sparse['path_b_status']}")
        print(f"  Path C: {sparse['path_c_status']}")
        bench = sparse.get("path_b_bench")
        if bench and bench.get("ok"):
            print(f"  Path B median={bench['median_ms']:.6f} ms p90={bench['p90_ms']:.6f} ms")


def main() -> int:
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
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any requested Path C benchmark fails.")
    args = parser.parse_args()

    _require_runtime()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "lane_f_path_c_vs_path_b_fp8_profile",
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
            )
        )

    if not args.skip_sparse:
        payload["sparse_mla"] = _bench_sparse_status(warmup=args.warmup, iters=args.iters, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    _print_summary(payload)
    print(f"\nwrote {args.out}")

    if args.strict:
        for shape_row in payload["results"]:
            for row in shape_row["rows"]:
                if row["label"].startswith("path_b_"):
                    continue
                if not row.get("bench", {}).get("ok", False):
                    return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
