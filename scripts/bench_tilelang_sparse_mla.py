"""Bench script for the cppmega sparse-MLA Path B and Path C ports.

Compares pure-MLX reference forward/backward against the Path B direct-MSL
kernel and the Path C TileLang DSL forward/backward kernels.

Output: bench/tilelang_ports/sparse_mla.json by default.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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
    _sparse_mla_bwd_path_c_partial,
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


def _bench_callable(
    label: str, fn, *, warmup: int = 5, iters: int = 20
) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "label": label,
        "median_ms": float(statistics.median(times)),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "iters": iters,
        "warmup": warmup,
    }


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
    if sm_scale is None:
        sm_scale = shapes.qk_dim ** -0.5

    status = sparse_mla_metal_status(q, kv, indices, d_out)
    if not status.available or _BWD_KERNEL is None:
        return None

    q16 = _promote_to_fp16_carrier(q)
    kv16 = _promote_to_fp16_carrier(kv)
    d_out16 = _promote_to_fp16_carrier(d_out)
    indices_i32 = indices.astype(mx.int32)
    threads = _threadgroup_size(shapes.topk)
    sm_scale_buf = mx.array([float(sm_scale)], dtype=mx.float32)

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


def _empty_bench(label: str, *, warmup: int, iters: int) -> dict[str, float]:
    return {
        "label": label,
        "median_ms": 0.0,
        "min_ms": 0.0,
        "max_ms": 0.0,
        "iters": iters,
        "warmup": warmup,
    }


def _bench_shape(cfg: dict[str, Any], *, warmup: int, iters: int) -> dict[str, Any]:
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
            return mx.zeros((1,))
        return result[0]

    def fwd_path_c():
        result = sparse_mla_fwd_path_c(q, kv, indices, sm_scale=sm_scale, d_v=d_v)
        if result is None:
            return mx.zeros((1,))
        return result[0]

    fwd_ref_bench = _bench_callable("reference_fwd", fwd_reference, warmup=warmup, iters=iters)
    fwd_apply_bench = _bench_callable("apply_fwd", fwd_apply, warmup=warmup, iters=iters)
    fwd_msl_bench = _bench_callable("path_b_msl_fwd", fwd_msl, warmup=warmup, iters=iters)
    fwd_path_c_bench = _bench_callable(
        "path_c_tilelang_fwd", fwd_path_c, warmup=warmup, iters=iters
    )
    fwd_path_c_over_path_b = (
        fwd_path_c_bench["median_ms"] / fwd_msl_bench["median_ms"]
        if fwd_msl_bench["median_ms"] > 0.0
        else float("inf")
    )

    # Backward via mx.value_and_grad on a scalar mean.
    q_fp32 = q.astype(mx.float32)
    kv_fp32 = kv.astype(mx.float32)
    mx.eval(q_fp32, kv_fp32)

    def loss_ref(q_in: mx.array, kv_in: mx.array) -> mx.array:
        out = sparse_mla_attention(q_in, kv_in, indices, sm_scale=sm_scale, d_v=d_v)
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
            return mx.zeros((1,))
        return result

    bwd_msl_bench = _bench_callable("path_b_msl_bwd", bwd_msl, warmup=warmup, iters=iters)

    def bwd_msl_kernel():
        result = _sparse_mla_bwd_msl_partial(
            q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
        )
        if result is None:
            return mx.zeros((1,))
        dkv_partial, dq, _indices_i32, _shapes = result
        return dkv_partial, dq

    bwd_msl_kernel_bench = _bench_callable(
        "path_b_msl_bwd_kernel_only",
        bwd_msl_kernel,
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
            return mx.zeros((1,))
        return result

    bwd_path_c_bench = _bench_callable(
        "path_c_tilelang_bwd", bwd_path_c, warmup=warmup, iters=iters
    )

    def bwd_path_c_kernel():
        result = _sparse_mla_bwd_path_c_partial(
            q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
        )
        if result is None:
            return mx.zeros((1,))
        dkv_partial, dq, _indices_i32, _shapes = result
        return dkv_partial, dq

    bwd_path_c_kernel_bench = _bench_callable(
        "path_c_tilelang_bwd_kernel_only",
        bwd_path_c_kernel,
        warmup=warmup,
        iters=iters,
    )

    path_c_partial = _sparse_mla_bwd_path_c_partial(
        q, kv, d_out, indices, sm_scale=sm_scale, d_v=d_v
    )
    if path_c_partial is None:
        bwd_path_c_reduce_bench = _empty_bench(
            "path_c_tilelang_bwd_reduce_only", warmup=warmup, iters=iters
        )
    else:
        dkv_partial, _dq, indices_i32, shapes = path_c_partial
        mx.eval(dkv_partial, indices_i32)

        def bwd_path_c_reduce():
            return _reduce_dkv_partial(dkv_partial, indices_i32, shapes)

        bwd_path_c_reduce_bench = _bench_callable(
            "path_c_tilelang_bwd_reduce_only",
            bwd_path_c_reduce,
            warmup=warmup,
            iters=iters,
        )

    bwd_path_c_over_path_b = (
        bwd_path_c_bench["median_ms"] / bwd_msl_bench["median_ms"]
        if bwd_msl_bench["median_ms"] > 0.0
        else float("inf")
    )
    bwd_path_c_kernel_over_path_b_kernel = (
        bwd_path_c_kernel_bench["median_ms"] / bwd_msl_kernel_bench["median_ms"]
        if bwd_msl_kernel_bench["median_ms"] > 0.0
        else float("inf")
    )
    bwd_path_c_reduce_over_path_b_reduce = (
        bwd_path_c_reduce_bench["median_ms"] / bwd_msl_reduce_bench["median_ms"]
        if bwd_msl_reduce_bench["median_ms"] > 0.0
        else float("inf")
    )
    if bwd_path_c_over_path_b <= 1.05:
        bwd_blocker = "none: Path C backward is within 5% of Path B"
    elif bwd_path_c_kernel_over_path_b_kernel > 1.10:
        bwd_blocker = "TileLang-lowered backward kernel slower than direct-MSL Path B kernel"
    elif bwd_path_c_reduce_over_path_b_reduce > 1.10:
        bwd_blocker = "shared dKV scatter-reduction dominates measured Path C backward"
    else:
        bwd_blocker = "mixed overhead: Path C backward slower without one phase exceeding 10%"

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

    return {
        "shape": cfg,
        "fwd_reference_ms": fwd_ref_bench,
        "fwd_apply_ms": fwd_apply_bench,
        "fwd_msl_ms": fwd_msl_bench,
        "fwd_path_c_ms": fwd_path_c_bench,
        "fwd_path_c_over_path_b_ratio": float(fwd_path_c_over_path_b),
        "fwd_path_c_no_worse_than_path_b": bool(fwd_path_c_over_path_b <= 1.05),
        "bwd_reference_ms": bwd_ref_bench,
        "bwd_msl_ms": bwd_msl_bench,
        "bwd_msl_kernel_ms": bwd_msl_kernel_bench,
        "bwd_msl_reduce_ms": bwd_msl_reduce_bench,
        "bwd_path_c_ms": bwd_path_c_bench,
        "bwd_path_c_kernel_ms": bwd_path_c_kernel_bench,
        "bwd_path_c_reduce_ms": bwd_path_c_reduce_bench,
        "bwd_path_c_over_path_b_ratio": float(bwd_path_c_over_path_b),
        "bwd_path_c_kernel_over_path_b_kernel_ratio": float(
            bwd_path_c_kernel_over_path_b_kernel
        ),
        "bwd_path_c_reduce_over_path_b_reduce_ratio": float(
            bwd_path_c_reduce_over_path_b_reduce
        ),
        "bwd_path_c_no_worse_than_path_b": bool(bwd_path_c_over_path_b <= 1.05),
        "bwd_blocker": bwd_blocker,
        "path_b": path_b,
        "path_c": path_c,
    }


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
    args = parser.parse_args()

    rows = []
    for cfg in DEFAULT_SHAPES:
        print(f"benching shape {cfg['name']} ...")
        rows.append(_bench_shape(cfg, warmup=args.warmup, iters=args.iters))

    metal_status = sparse_mla_metal_status()
    path_c_status = sparse_mla_path_c_status()
    payload = {
        "schema": 1,
        "kernel": "sparse_mla",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "fp16_carrier": True,
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
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
