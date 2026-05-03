#!/usr/bin/env python3
"""Bench the Path B M2RNN Metal port against the pure-MLX reference scan.

Measures forward-only and forward+backward latency across two shapes:
  - Smoke: B=2 T=512 H=4 K=64 V=16
  - Mini: B=2 T=2048 H=8 K=64 V=32 (closer to the production mini-config)

Writes a JSON receipt to bench/tilelang_ports/m2rnn_path_b.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.nn._tilelang import (  # noqa: E402
    m2rnn_apply,
    m2rnn_fwd_metal,
    m2rnn_metal_status,
    m2rnn_reference,
)


DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _make_inputs(
    *,
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
    seed: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    q = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    k = (mx.random.normal((batch, seq, heads, k_dim)) * 0.1).astype(dtype)
    v = (mx.random.normal((batch, seq, heads, v_dim)) * 0.1).astype(dtype)
    eye = mx.broadcast_to(mx.eye(v_dim)[None], (heads, v_dim, v_dim))
    W = (eye + mx.random.normal((heads, v_dim, v_dim)) * 0.05).astype(dtype)
    xf = mx.random.uniform(0.1, 0.9, (batch, seq, heads)).astype(dtype)
    h0 = mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype)
    mx.eval(q, k, v, W, xf, h0)
    return q, k, v, W, xf, h0


def _run_iter(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    out = fn()
    if isinstance(out, tuple):
        mx.eval(*out)
    elif isinstance(out, mx.array):
        mx.eval(out)
    return time.perf_counter() - start


def _bench(label: str, fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
        mx.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        samples.append(_run_iter(fn))
        mx.synchronize()
    return {
        "label": label,
        "mean_ms": statistics.mean(samples) * 1000.0,
        "median_ms": statistics.median(samples) * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "iters": iters,
        "warmup": warmup,
    }


def _value_norm(*ys: mx.array) -> float:
    total = 0.0
    for y in ys:
        f = y.astype(mx.float32) if y.dtype != mx.float32 else y
        total += float(mx.sum(f * f))
    return total**0.5


def _parity_diff(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32) if a.dtype != mx.float32 else a
    bf = b.astype(mx.float32) if b.dtype != mx.float32 else b
    return float(mx.max(mx.abs(af - bf)))


def _gflops_estimate_fwd(
    *, batch: int, seq: int, heads: int, k_dim: int, v_dim: int
) -> float:
    """Rough FLOPS estimate per forward pass.

    Per step: matmul h@W is K*V*V mul-add per (B,H), outer product is K*V,
    blend is K*V, q^T h is K*V. So ~ 2*K*V*V + 4*K*V multiplies per (B,H,t).
    """

    return 2.0 * batch * seq * heads * (2 * k_dim * v_dim * v_dim + 4 * k_dim * v_dim) / 1e9


def _peak_memory_mb() -> float | None:
    metal = getattr(mx, "metal", None)
    fn = getattr(metal, "get_peak_memory", None) if metal is not None else None
    if fn is None:
        return None
    try:
        return float(fn()) / (1024.0 * 1024.0)
    except Exception:
        return None


def _reset_peak_memory() -> None:
    metal = getattr(mx, "metal", None)
    fn = getattr(metal, "reset_peak_memory", None) if metal is not None else None
    if fn is not None:
        try:
            fn()
        except Exception:
            pass


def _bench_one_shape(
    *,
    batch: int,
    seq: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
    dtype_name: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    inputs = _make_inputs(
        batch=batch, seq=seq, heads=heads, k_dim=k_dim, v_dim=v_dim, dtype=dtype, seed=seed
    )
    q, k, v, W, xf, h0 = inputs

    # Parity check first.
    y_ref, h_ref = m2rnn_reference(q, k, v, W, xf, h0=h0)
    y_met, h_met, _ = m2rnn_fwd_metal(q, k, v, W, xf, h0)
    mx.eval(y_ref, h_ref, y_met, h_met)
    parity = {
        "y_max_abs": _parity_diff(y_met, y_ref),
        "y_ref_norm": _value_norm(y_ref),
        "h_max_abs": _parity_diff(h_met, h_ref),
        "h_ref_norm": _value_norm(h_ref),
    }
    parity["y_rel"] = parity["y_max_abs"] / (parity["y_ref_norm"] + 1e-12)
    parity["h_rel"] = parity["h_max_abs"] / (parity["h_ref_norm"] + 1e-12)

    _reset_peak_memory()
    fwd_ref = _bench(
        "fwd_reference",
        lambda: m2rnn_reference(q, k, v, W, xf, h0=h0),
        warmup=warmup,
        iters=iters,
    )
    fwd_ref_peak = _peak_memory_mb()

    _reset_peak_memory()
    fwd_met = _bench(
        "fwd_metal",
        lambda: m2rnn_fwd_metal(q, k, v, W, xf, h0),
        warmup=warmup,
        iters=iters,
    )
    fwd_met_peak = _peak_memory_mb()

    # Forward + backward.
    def grad_ref_loss(
        q: mx.array, k: mx.array, v: mx.array, W: mx.array, xf: mx.array
    ) -> mx.array:
        y, _ = m2rnn_reference(q, k, v, W, xf)
        return mx.sum(y * y) * 0.5

    def grad_met_loss(
        q: mx.array, k: mx.array, v: mx.array, W: mx.array, xf: mx.array, h0: mx.array
    ) -> mx.array:
        y = cast(mx.array, m2rnn_apply(q, k, v, W, xf, h0))
        return mx.sum(y * y) * 0.5

    grad_ref = mx.value_and_grad(grad_ref_loss, argnums=(0, 1, 2, 3, 4))
    grad_met = mx.value_and_grad(grad_met_loss, argnums=(0, 1, 2, 3, 4))

    _reset_peak_memory()
    fwd_bwd_ref = _bench(
        "fwd_bwd_reference",
        lambda: grad_ref(q, k, v, W, xf),
        warmup=warmup,
        iters=iters,
    )
    fwd_bwd_ref_peak = _peak_memory_mb()

    _reset_peak_memory()
    fwd_bwd_met = _bench(
        "fwd_bwd_metal",
        lambda: grad_met(q, k, v, W, xf, h0),
        warmup=warmup,
        iters=iters,
    )
    fwd_bwd_met_peak = _peak_memory_mb()

    gflops = _gflops_estimate_fwd(
        batch=batch, seq=seq, heads=heads, k_dim=k_dim, v_dim=v_dim
    )

    speedup_fwd = (
        fwd_ref["median_ms"] / fwd_met["median_ms"]
        if fwd_met["median_ms"] > 0 else None
    )
    speedup_fb = (
        fwd_bwd_ref["median_ms"] / fwd_bwd_met["median_ms"]
        if fwd_bwd_met["median_ms"] > 0 else None
    )

    return {
        "shape": {
            "batch": batch,
            "seq": seq,
            "heads": heads,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "dtype": dtype_name,
        },
        "parity": parity,
        "timings": {
            "fwd_reference": fwd_ref,
            "fwd_metal": fwd_met,
            "fwd_bwd_reference": fwd_bwd_ref,
            "fwd_bwd_metal": fwd_bwd_met,
        },
        "peak_memory_mb": {
            "fwd_reference": fwd_ref_peak,
            "fwd_metal": fwd_met_peak,
            "fwd_bwd_reference": fwd_bwd_ref_peak,
            "fwd_bwd_metal": fwd_bwd_met_peak,
        },
        "speedups": {
            "fwd_median": speedup_fwd,
            "fwd_bwd_median": speedup_fb,
        },
        "gflops_fwd_est": gflops,
        "gflops_per_sec_fwd_metal": (
            (gflops * 1000.0 / fwd_met["median_ms"]) if fwd_met["median_ms"] > 0 else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench" / "tilelang_ports" / "m2rnn_path_b.json",
    )
    parser.add_argument("--hardware-label", type=str, default=platform.node() or "unknown")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    status = m2rnn_metal_status()

    shapes = [
        # Smoke: B=2 T=512 H=4 K=64 V=16
        dict(batch=2, seq=512, heads=4, k_dim=64, v_dim=16),
        # Mini: B=2 T=2048 H=8 K=64 V=32
        dict(batch=2, seq=2048, heads=8, k_dim=64, v_dim=32),
    ]
    shape_results: list[dict[str, Any]] = []
    for shape in shapes:
        shape_results.append(
            _bench_one_shape(
                batch=int(shape["batch"]),
                seq=int(shape["seq"]),
                heads=int(shape["heads"]),
                k_dim=int(shape["k_dim"]),
                v_dim=int(shape["v_dim"]),
                dtype=dtype,
                dtype_name=args.dtype,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
        )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local_only",
        "kernel": "m2rnn_path_b",
        "hardware_label": args.hardware_label,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": _safe_version("mlx"),
        },
        "metal_status": {
            "available": status.available,
            "reason": status.reason,
        },
        "shapes": shape_results,
        "matched_run_guard": (
            "Compare M4 Max and GB10 only when both rows were collected with "
            "identical kernel inputs and dtype. This is a local_only receipt."
        ),
    }

    text = json.dumps(receipt, indent=2)
    print(text)
    if not args.print_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
