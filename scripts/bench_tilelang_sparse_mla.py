"""Bench script for the cppmega sparse-MLA Path B port.

Compares pure-MLX reference forward and backward against the (currently
gated) Path B Metal kernel. While the GEMM blocker on tilelang 0.1.9 is in
effect, only the reference numbers are produced; ``path_b_*`` rows record
the blocker reason instead of timings so downstream parity reports can show
the gap.

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
    sparse_mla_apply,
    sparse_mla_bwd_metal,
    sparse_mla_fwd_metal,
    sparse_mla_metal_status,
)
from cppmega_mlx.nn.sparse_mla import (
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

    fwd_ref_bench = _bench_callable("reference_fwd", fwd_reference, warmup=warmup, iters=iters)
    fwd_apply_bench = _bench_callable("apply_fwd", fwd_apply, warmup=warmup, iters=iters)
    fwd_msl_bench = _bench_callable("path_b_msl_fwd", fwd_msl, warmup=warmup, iters=iters)

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
        return result[0]

    bwd_msl_bench = _bench_callable("path_b_msl_bwd", bwd_msl, warmup=warmup, iters=iters)

    metal_status = sparse_mla_metal_status(q, kv, indices)
    path_b = {
        "available": metal_status.available,
        "reason": metal_status.reason,
    }

    return {
        "shape": cfg,
        "fwd_reference_ms": fwd_ref_bench,
        "fwd_apply_ms": fwd_apply_bench,
        "fwd_msl_ms": fwd_msl_bench,
        "bwd_reference_ms": bwd_ref_bench,
        "bwd_msl_ms": bwd_msl_bench,
        "path_b": path_b,
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
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
