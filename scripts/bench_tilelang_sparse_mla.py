"""Bench script for the cppmega sparse-MLA Path B and Path C ports.

Compares pure-MLX reference forward/backward against the Path B direct-MSL
kernel and the Path C TileLang DSL forward/backward kernels.

Output: bench/tilelang_ports/sparse_mla.json by default.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, TypedDict, cast

import numpy as np

import mlx.core as mx

# Ensure repo root is importable when called from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cppmega_mlx.nn._tilelang.sparse_mla import (
    _BWD_KERNEL,
    _promote_to_fp16_carrier,
    _reduce_dkv_partial,
    sparse_mla_metal_status,
    sparse_mla_apply,
    sparse_mla_bwd_metal,
    sparse_mla_fwd_metal,
)
from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang.sparse_mla_path_c import (
    sparse_mla_bwd_path_c,
    sparse_mla_fwd_path_c,
    sparse_mla_path_c_status,
)
from cppmega_mlx.nn.sparse_mla import (
    _resolve_shapes,
    sparse_mla_attention,
    sparse_mla_attention_reference,
)


DEFAULT_SHAPES = [
    {"name": "B2_S128_H8_D64", "B": 2, "S": 128, "H": 8, "D": 64, "G": 1, "topk": 16, "Skv": 128},
    {"name": "B4_S512_H8_D64", "B": 4, "S": 512, "H": 8, "D": 64, "G": 1, "topk": 32, "Skv": 512},
    {"name": "B4_S1024_H8_D64", "B": 4, "S": 1024, "H": 8, "D": 64, "G": 1, "topk": 64, "Skv": 1024},
]


class BenchResult(TypedDict, total=False):
    label: str
    ok: bool
    median_ms: float | None
    min_ms: float | None
    max_ms: float | None
    iters: int
    warmup: int
    error: str | None
    pair_ratio_median: float
    pair_ratio_min: float
    pair_ratio_max: float


def _sync_mlx() -> None:
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def _eval_result(out: Any) -> None:
    if isinstance(out, (list, tuple)):
        mx.eval(*out)
    else:
        mx.eval(out)
    _sync_mlx()


def _bench_failure(label: str, *, warmup: int, iters: int, error: str) -> BenchResult:
    return {
        "label": label,
        "ok": False,
        "median_ms": None,
        "min_ms": None,
        "max_ms": None,
        "iters": iters,
        "warmup": warmup,
        "error": error,
    }


def _bench_callable(
    label: str, fn: Callable[[], Any], *, warmup: int = 5, iters: int = 20
) -> BenchResult:
    try:
        for _ in range(warmup):
            _eval_result(fn())
        times = []
        for _ in range(iters):
            _sync_mlx()
            t0 = time.perf_counter()
            _eval_result(fn())
            times.append((time.perf_counter() - t0) * 1000.0)
    except Exception as exc:
        return _bench_failure(
            label,
            warmup=warmup,
            iters=iters,
            error=f"{type(exc).__name__}: {exc}",
        )
    return {
        "label": label,
        "ok": True,
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "iters": iters,
        "warmup": warmup,
    }


def _time_one(fn: Callable[[], Any]) -> float:
    _sync_mlx()
    t0 = time.perf_counter()
    _eval_result(fn())
    return (time.perf_counter() - t0) * 1000.0


def _result_from_times(
    label: str,
    times: list[float],
    *,
    warmup: int,
    iters: int,
) -> BenchResult:
    return {
        "label": label,
        "ok": True,
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "iters": iters,
        "warmup": warmup,
    }


def _bench_pair_interleaved(
    label_a: str,
    fn_a: Callable[[], Any],
    label_b: str,
    fn_b: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[BenchResult, BenchResult, float]:
    """Benchmark two same-shape kernels in one alternating run.

    Sequential Path-B-then-Path-C medians on Apple GPUs are noisy enough to
    trip a strict C/B gate even when both wrappers dispatch the same compiled
    Metal kernel. Alternating order balances first/second-run cache and power
    state effects while keeping the per-call timing/eval contract unchanged.
    """

    try:
        for i in range(warmup):
            if i & 1:
                _eval_result(fn_b())
                _eval_result(fn_a())
            else:
                _eval_result(fn_a())
                _eval_result(fn_b())

        times_a: list[float] = []
        times_b: list[float] = []
        for i in range(iters):
            if i & 1:
                times_b.append(_time_one(fn_b))
                times_a.append(_time_one(fn_a))
            else:
                times_a.append(_time_one(fn_a))
                times_b.append(_time_one(fn_b))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return (
            _bench_failure(label_a, warmup=warmup, iters=iters, error=error),
            _bench_failure(label_b, warmup=warmup, iters=iters, error=error),
            float("inf"),
        )

    pair_ratios = [
        time_b / time_a
        for time_a, time_b in zip(times_a, times_b, strict=True)
        if time_a > 0.0 and math.isfinite(time_a) and math.isfinite(time_b)
    ]
    pair_ratio_median = (
        float(statistics.median(pair_ratios)) if pair_ratios else float("inf")
    )
    result_a = _result_from_times(label_a, times_a, warmup=warmup, iters=iters)
    result_b = _result_from_times(label_b, times_b, warmup=warmup, iters=iters)
    if pair_ratios:
        result_b["pair_ratio_median"] = pair_ratio_median
        result_b["pair_ratio_min"] = float(min(pair_ratios))
        result_b["pair_ratio_max"] = float(max(pair_ratios))
    return result_a, result_b, pair_ratio_median


def _make_inputs(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    B, S, H, D = cfg["B"], cfg["S"], cfg["H"], cfg["D"]
    G = cfg["G"]
    topk = cfg["topk"]
    Skv = cfg["Skv"]
    d_v = cfg.get("d_v", D)
    qk_dim = D
    q = mx.array(rng.standard_normal((B, S, H, qk_dim)).astype(np.float16))
    kv = mx.array(rng.standard_normal((B, Skv, G, qk_dim)).astype(np.float16))
    indices_np = rng.integers(0, Skv, size=(B, S, G, topk)).astype(np.int32)
    indices = mx.array(indices_np)
    sm_scale = qk_dim ** -0.5
    return {
        "q": q,
        "kv": kv,
        "indices": indices,
        "sm_scale": sm_scale,
        "d_v": d_v,
    }


def _threadgroup_size(topk: int) -> int:
    threads = min(64, max(1, topk))
    power = 1
    while (power << 1) <= threads:
        power <<= 1
    return power


def _sparse_mla_bwd_msl_partial(
    q: mx.array,
    kv: mx.array,
    d_out: mx.array,
    indices: mx.array,
    *,
    sm_scale: float | None = None,
    d_v: int | None = None,
) -> tuple[mx.array, mx.array, mx.array, Any] | None:
    """Run Path B's direct-MSL backward kernel without the shared dKV reduction."""

    shapes = _resolve_shapes(q, kv, indices, d_v=d_v)
    scale = shapes.qk_dim ** -0.5 if sm_scale is None else sm_scale

    status = sparse_mla_metal_status(q, kv, indices, d_out)
    if not status.available or _BWD_KERNEL is None:
        return None

    q16 = _promote_to_fp16_carrier(q)
    kv16 = _promote_to_fp16_carrier(kv)
    d_out16 = _promote_to_fp16_carrier(d_out)
    indices_i32 = indices.astype(mx.int32)
    threads = _threadgroup_size(shapes.topk)
    sm_scale_buf = mx.array([float(scale)], dtype=mx.float32)

    template = [
        ("T_OUT", mx.float16),
        ("BATCH", shapes.batch),
        ("SEQ_LEN", shapes.seq_len),
        ("SEQ_LEN_KV", shapes.seq_len_kv),
        ("HEADS", shapes.heads),
        ("HEAD_KV", shapes.head_kv),
        ("KV_GROUP", shapes.kv_group),
        ("QK_DIM", shapes.qk_dim),
        ("D_V", shapes.d_v),
        ("TOPK", shapes.topk),
        ("BLOCK_SIZE", threads),
    ]

    grid_x = shapes.batch * shapes.seq_len * shapes.heads
    dq, dkv_partial = _msl_transform.dispatch(
        _BWD_KERNEL,
        inputs=[q16, kv16, indices_i32, d_out16, sm_scale_buf],
        output_shapes=[
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.qk_dim),
            (shapes.batch, shapes.seq_len, shapes.heads, shapes.topk, shapes.qk_dim),
        ],
        output_dtypes=[mx.float16, mx.float16],
        grid=(grid_x * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        template=template,
    )
    return dkv_partial, dq, indices_i32, shapes


def _empty_bench(label: str, *, warmup: int, iters: int) -> BenchResult:
    return _bench_failure(label, warmup=warmup, iters=iters, error="kernel did not dispatch")


def _bench_median(result: BenchResult) -> float | None:
    value = result.get("median_ms")
    return value if isinstance(value, float) and math.isfinite(value) and value > 0.0 else None


def _bench_ratio(numerator: BenchResult, denominator: BenchResult) -> float:
    num = _bench_median(numerator)
    den = _bench_median(denominator)
    if not numerator.get("ok") or not denominator.get("ok") or num is None or den is None:
        return float("inf")
    return num / den


def _bench_shape(
    cfg: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    fwd_only: bool = False,
    max_ratio: float = 1.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    inputs = _make_inputs(cfg, rng)
    q, kv, indices = inputs["q"], inputs["kv"], inputs["indices"]
    sm_scale = inputs["sm_scale"]
    d_v = inputs["d_v"]

    # Pre-eval inputs so allocation isn't counted.
    mx.eval(q, kv, indices)

    def fwd_reference():
        return sparse_mla_attention_reference(q, kv, indices, sm_scale=sm_scale, d_v=d_v)

    def fwd_apply():
        return sparse_mla_apply(q, kv, indices, sm_scale=sm_scale, d_v=d_v)

    def fwd_msl():
        result = sparse_mla_fwd_metal(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            raise RuntimeError("Path B sparse_mla_fwd_metal did not dispatch")
        return result[0]

    def fwd_path_c():
        result = sparse_mla_fwd_path_c(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            raise RuntimeError("Path C sparse_mla_fwd_path_c did not dispatch")
        return result[0]

    fwd_ref_bench = _bench_callable("reference_fwd", fwd_reference, warmup=warmup, iters=iters)
    fwd_apply_bench = _bench_callable("apply_fwd", fwd_apply, warmup=warmup, iters=iters)
    fwd_msl_bench = _bench_callable("path_b_msl_fwd", fwd_msl, warmup=warmup, iters=iters)
    fwd_path_c_bench = _bench_callable(
        "path_c_tilelang_fwd", fwd_path_c, warmup=warmup, iters=iters
    )
    fwd_path_c_over_path_b = _bench_ratio(fwd_path_c_bench, fwd_msl_bench)
    (
        paired_fwd_msl_bench,
        paired_fwd_path_c_bench,
        paired_fwd_path_c_over_path_b,
    ) = _bench_pair_interleaved(
        "path_b_msl_fwd_paired",
        fwd_msl,
        "path_c_tilelang_fwd_paired",
        fwd_path_c,
        warmup=warmup,
        iters=iters,
    )

    metal_status = sparse_mla_metal_status(q, kv, indices)
    path_c_status = sparse_mla_path_c_status()
    path_b = {
        "available": metal_status.available,
        "reason": metal_status.reason,
    }
    path_c = {
        "available": path_c_status.available,
        "reason": path_c_status.reason,
    }

    row: dict[str, Any] = {
        "shape": cfg,
        "fwd_reference_ms": fwd_ref_bench,
        "fwd_apply_ms": fwd_apply_bench,
        "fwd_msl_ms": fwd_msl_bench,
        "fwd_path_c_ms": fwd_path_c_bench,
        "fwd_msl_paired_ms": paired_fwd_msl_bench,
        "fwd_path_c_paired_ms": paired_fwd_path_c_bench,
        "fwd_path_c_over_path_b_ratio": float(fwd_path_c_over_path_b),
        "fwd_path_c_over_path_b_paired_ratio": float(paired_fwd_path_c_over_path_b),
        "path_c_over_path_b_max_ratio": float(max_ratio),
        "fwd_path_c_no_worse_than_path_b": bool(
            paired_fwd_path_c_over_path_b <= max_ratio
        ),
        "fwd_path_c_no_worse_than_path_b_paired": bool(
            paired_fwd_path_c_over_path_b <= max_ratio
        ),
        "fwd_path_c_unpaired_within_max_ratio": bool(fwd_path_c_over_path_b <= max_ratio),
        "path_b": path_b,
        "path_c": path_c,
    }
    if fwd_only:
        return row

    # Backward via mx.value_and_grad on a scalar mean.
    q_fp32 = q.astype(mx.float32)
    kv_fp32 = kv.astype(mx.float32)
    mx.eval(q_fp32, kv_fp32)

    def loss_ref(q_in: mx.array, kv_in: mx.array) -> mx.array:
        out = sparse_mla_attention(q_in, kv_in, indices, sm_scale=sm_scale, d_v=d_v)
        out = cast(mx.array, out)
        return mx.mean(out * out)

    grad_fn = mx.grad(loss_ref, argnums=(0, 1))

    def bwd_reference():
        dq, dkv = grad_fn(q_fp32, kv_fp32)
        return dq, dkv

    bwd_ref_bench = _bench_callable("reference_bwd", bwd_reference, warmup=warmup, iters=iters)

    # Direct MSL backward.
    d_out = mx.array(np.random.default_rng(0).standard_normal(
        (cfg["B"], cfg["S"], cfg["H"], cfg.get("d_v", cfg["D"]))
    ).astype(np.float16))
    mx.eval(d_out)

    def bwd_msl():
        result = sparse_mla_bwd_metal(q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            raise RuntimeError("Path B sparse_mla_bwd_metal did not dispatch")
        return result

    bwd_msl_bench = _bench_callable("path_b_msl_bwd", bwd_msl, warmup=warmup, iters=iters)

    def bwd_msl_kernel():
        result = _sparse_mla_bwd_msl_partial(
            q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
        )
        if result is None:
            raise RuntimeError("Path B backward partial kernel did not dispatch")
        dkv_partial, dq, _indices_i32, _shapes = result
        return dkv_partial, dq

    bwd_msl_kernel_bench = _bench_callable(
        "path_b_msl_bwd_kernel_only",
        bwd_msl_kernel,
        warmup=warmup,
        iters=iters,
    )

    def bwd_msl_fresh_reduce():
        result = _sparse_mla_bwd_msl_partial(
            q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
        )
        if result is None:
            raise RuntimeError("Path B backward partial kernel did not dispatch")
        dkv_partial, dq, indices_i32, shapes = result
        dkv = _reduce_dkv_partial(dkv_partial, indices_i32, shapes)
        return dq, dkv

    bwd_msl_fresh_reduce_bench = _bench_callable(
        "path_b_msl_bwd_fresh_reduce",
        bwd_msl_fresh_reduce,
        warmup=warmup,
        iters=iters,
    )

    path_b_partial = _sparse_mla_bwd_msl_partial(
        q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
    )
    if path_b_partial is None:
        bwd_msl_reduce_bench = _empty_bench(
            "path_b_msl_bwd_reduce_only", warmup=warmup, iters=iters
        )
    else:
        dkv_partial_b, _dq_b, indices_i32_b, shapes_b = path_b_partial
        mx.eval(dkv_partial_b, indices_i32_b)

        def bwd_msl_reduce():
            return _reduce_dkv_partial(dkv_partial_b, indices_i32_b, shapes_b)

        bwd_msl_reduce_bench = _bench_callable(
            "path_b_msl_bwd_reduce_only",
            bwd_msl_reduce,
            warmup=warmup,
            iters=iters,
        )

    def bwd_path_c():
        result = sparse_mla_bwd_path_c(q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            raise RuntimeError("Path C sparse_mla_bwd_path_c did not dispatch")
        return result

    bwd_path_c_bench = _bench_callable(
        "path_c_tilelang_bwd", bwd_path_c, warmup=warmup, iters=iters
    )
    (
        paired_bwd_msl_bench,
        paired_bwd_path_c_bench,
        paired_bwd_path_c_over_path_b,
    ) = _bench_pair_interleaved(
        "path_b_msl_bwd_paired",
        bwd_msl,
        "path_c_tilelang_bwd_paired",
        bwd_path_c,
        warmup=warmup,
        iters=iters,
    )

    def bwd_path_c_kernel():
        result = sparse_mla_bwd_path_c(
            q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
        )
        if result is None:
            raise RuntimeError("Path C backward owner-output kernel did not dispatch")
        return result

    bwd_path_c_kernel_bench = _bench_callable(
        "path_c_tilelang_bwd_owner_output_kernel",
        bwd_path_c_kernel,
        warmup=warmup,
        iters=iters,
    )

    path_c_no_public_reduce = (
        "not applicable: Path C backward returns final owner-output dKV"
    )
    bwd_path_c_fresh_reduce_bench = _bench_failure(
        "path_c_tilelang_bwd_fresh_reduce",
        warmup=warmup,
        iters=iters,
        error=path_c_no_public_reduce,
    )
    bwd_path_c_reduce_bench = _bench_failure(
        "path_c_tilelang_bwd_reduce_only",
        warmup=warmup,
        iters=iters,
        error=path_c_no_public_reduce,
    )

    bwd_path_c_over_path_b = _bench_ratio(bwd_path_c_bench, bwd_msl_bench)
    bwd_path_c_kernel_over_path_b_kernel = _bench_ratio(
        bwd_path_c_kernel_bench,
        bwd_msl_kernel_bench,
    )
    bwd_path_c_reduce_over_path_b_reduce = _bench_ratio(
        bwd_path_c_reduce_bench,
        bwd_msl_reduce_bench,
    )
    bwd_path_c_fresh_reduce_over_path_b_fresh_reduce = _bench_ratio(
        bwd_path_c_fresh_reduce_bench,
        bwd_msl_fresh_reduce_bench,
    )
    path_c_fresh_reduce_ok = bool(bwd_path_c_fresh_reduce_bench.get("ok"))
    path_c_reduce_ok = bool(bwd_path_c_reduce_bench.get("ok"))
    if paired_bwd_path_c_over_path_b <= max_ratio:
        bwd_blocker = f"none: Path C backward is within strict ratio {max_ratio:.3g}"
    elif bwd_path_c_kernel_over_path_b_kernel > 1.10:
        bwd_blocker = (
            "TileLang owner-output backward slower than direct-MSL Path B "
            "partial kernel"
        )
    elif (
        path_c_fresh_reduce_ok
        and bwd_path_c_fresh_reduce_over_path_b_fresh_reduce > 1.10
    ):
        bwd_blocker = "fresh shared dKV scatter-reduction dominates measured Path C backward"
    elif path_c_reduce_ok and bwd_path_c_reduce_over_path_b_reduce > 1.10:
        bwd_blocker = (
            "pre-materialized shared dKV scatter-reduction is slower, but not "
            "the fresh total-backward blocker"
        )
    else:
        bwd_blocker = "mixed overhead: Path C backward slower without one phase exceeding 10%"

    row.update(
        {
            "bwd_reference_ms": bwd_ref_bench,
            "bwd_msl_ms": bwd_msl_bench,
            "bwd_msl_paired_ms": paired_bwd_msl_bench,
            "bwd_msl_kernel_ms": bwd_msl_kernel_bench,
            "bwd_msl_fresh_reduce_ms": bwd_msl_fresh_reduce_bench,
            "bwd_msl_reduce_ms": bwd_msl_reduce_bench,
            "bwd_path_c_ms": bwd_path_c_bench,
            "bwd_path_c_paired_ms": paired_bwd_path_c_bench,
            "bwd_path_c_kernel_ms": bwd_path_c_kernel_bench,
            "bwd_path_c_fresh_reduce_ms": bwd_path_c_fresh_reduce_bench,
            "bwd_path_c_reduce_ms": bwd_path_c_reduce_bench,
            "bwd_path_c_over_path_b_ratio": float(bwd_path_c_over_path_b),
            "bwd_path_c_over_path_b_paired_ratio": float(paired_bwd_path_c_over_path_b),
            "bwd_path_c_kernel_over_path_b_kernel_ratio": float(
                bwd_path_c_kernel_over_path_b_kernel
            ),
            "bwd_path_c_fresh_reduce_over_path_b_fresh_reduce_ratio": float(
                bwd_path_c_fresh_reduce_over_path_b_fresh_reduce
            ),
            "bwd_path_c_reduce_over_path_b_reduce_ratio": float(
                bwd_path_c_reduce_over_path_b_reduce
            ),
            "bwd_path_c_no_worse_than_path_b": bool(
                paired_bwd_path_c_over_path_b <= max_ratio
            ),
            "bwd_path_c_no_worse_than_path_b_paired": bool(
                paired_bwd_path_c_over_path_b <= max_ratio
            ),
            "bwd_blocker": bwd_blocker,
        }
    )
    return row


def _finite_float(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _bench_ok(row: dict[str, Any], key: str) -> bool:
    bench = row.get(key, {})
    return bool(bench.get("ok"))


def _strict_row_failures(
    row: dict[str, Any],
    *,
    fwd_only: bool,
    max_ratio: float,
    strict_phase: str,
) -> list[str]:
    shape = row.get("shape", {}).get("name", "<unknown>")
    failures: list[str] = []
    if not row.get("path_b", {}).get("available"):
        failures.append(f"{shape}: Path B unavailable: {row.get('path_b', {}).get('reason')}")
    if not row.get("path_c", {}).get("available"):
        failures.append(f"{shape}: Path C unavailable: {row.get('path_c', {}).get('reason')}")
    if strict_phase in ("all", "fwd"):
        fwd_ratio = row.get("fwd_path_c_over_path_b_paired_ratio")
        fwd_ratio_ok = _finite_float(fwd_ratio)
        if (
            not _bench_ok(row, "fwd_msl_paired_ms")
            or not _bench_ok(row, "fwd_path_c_paired_ms")
            or not fwd_ratio_ok
            or (cast(float, fwd_ratio) > max_ratio)
        ):
            failures.append(
                f"{shape}: forward strict gate failed paired C/B={fwd_ratio} "
                f"path_b_ok={_bench_ok(row, 'fwd_msl_paired_ms')} "
                f"path_c_ok={_bench_ok(row, 'fwd_path_c_paired_ms')}"
            )
    if strict_phase in ("all", "bwd") and not fwd_only:
        bwd_ratio = row.get("bwd_path_c_over_path_b_paired_ratio")
        bwd_ratio_ok = _finite_float(bwd_ratio)
        if (
            not _bench_ok(row, "bwd_msl_paired_ms")
            or not _bench_ok(row, "bwd_path_c_paired_ms")
            or not bwd_ratio_ok
            or (cast(float, bwd_ratio) > max_ratio)
        ):
            failures.append(
                f"{shape}: backward strict gate failed paired C/B={bwd_ratio} "
                f"path_b_ok={_bench_ok(row, 'bwd_msl_paired_ms')} "
                f"path_c_ok={_bench_ok(row, 'bwd_path_c_paired_ms')}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bench/tilelang_ports/sparse_mla.json"),
        help="output JSON path",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--shape",
        action="append",
        choices=[cfg["name"] for cfg in DEFAULT_SHAPES],
        help="shape name to benchmark; may be passed more than once",
    )
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        help="measure only forward kernels; avoids unrelated backward compile/reporting paths",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless Path C runs and is no slower than Path B for every selected row.",
    )
    parser.add_argument(
        "--strict-phase",
        choices=("all", "fwd", "bwd"),
        default="all",
        help="Phase checked by --strict. Default preserves the historical all-phase gate.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed Path C / Path B median ratio for --strict.",
    )
    args = parser.parse_args()
    if args.fwd_only and args.strict_phase == "bwd":
        parser.error("--strict-phase bwd requires backward measurements; drop --fwd-only")

    rows = []
    selected_shapes = set(args.shape or [cfg["name"] for cfg in DEFAULT_SHAPES])
    for cfg in DEFAULT_SHAPES:
        if cfg["name"] not in selected_shapes:
            continue
        print(f"benching shape {cfg['name']} ...")
        rows.append(
            _bench_shape(
                cfg,
                warmup=args.warmup,
                iters=args.iters,
                fwd_only=args.fwd_only,
                max_ratio=float(args.max_ratio),
            )
        )

    metal_status = sparse_mla_metal_status()
    path_c_status = sparse_mla_path_c_status()
    strict_failures = [
        failure
        for row in rows
        for failure in _strict_row_failures(
            row,
            fwd_only=bool(args.fwd_only),
            max_ratio=float(args.max_ratio),
            strict_phase=str(args.strict_phase),
        )
    ]
    payload = {
        "schema": 1,
        "kernel": "sparse_mla",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "fp16_carrier": True,
        "fwd_only": bool(args.fwd_only),
        "strict_policy": {
            "path_c_over_path_b_max_ratio": float(args.max_ratio),
            "requires_path_b_and_path_c": True,
            "fwd_only": bool(args.fwd_only),
            "phase": str(args.strict_phase),
        },
        "strict": {
            "enabled": bool(args.strict),
            "passed": not strict_failures,
            "path_c_over_path_b_max_ratio": float(args.max_ratio),
            "phase": str(args.strict_phase),
            "fwd_only": bool(args.fwd_only),
            "failures": strict_failures,
        },
        "path_b_status": {
            "available": metal_status.available,
            "reason": metal_status.reason,
        },
        "path_c_status": {
            "available": path_c_status.available,
            "reason": path_c_status.reason,
        },
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    if args.strict:
        if strict_failures:
            print("strict Sparse-MLA Path C gate failed:", file=sys.stderr)
            for failure in strict_failures:
                print(f"  {failure}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
