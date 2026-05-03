#!/usr/bin/env python3
"""Bench the Path B Mamba3 MIMO Metal port against the pure-MLX reference scan.

The script measures forward-only and forward+backward latency at the spec shape
(B=2, T=512, D=128, state=64) and writes a JSON receipt under
bench/tilelang_ports/mamba3.json. The receipt format mirrors the existing
bench scripts (scope flag, shape metadata, mean/median timing).
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
    mamba3_mimo_apply,
    mamba3_mimo_fwd_metal,
    mamba3_mimo_metal_status,
    mamba3_mimo_reference,
)


DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _make_inputs(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: mx.Dtype,
    seed: int,
) -> tuple[mx.array, ...]:
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    B = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    C = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(dtype)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    D = mx.ones((heads,), dtype=dtype)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=dtype)
    return x, B, C, z, A, dt, D, h0


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--state", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="float16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "bench" / "tilelang_ports" / "mamba3.json",
    )
    parser.add_argument("--hardware-label", type=str, default=platform.node() or "unknown")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    inputs = _make_inputs(
        batch=args.batch,
        seq=args.seq,
        heads=args.heads,
        headdim=args.headdim,
        state=args.state,
        dtype=dtype,
        seed=args.seed,
    )
    status = mamba3_mimo_metal_status()

    # Parity check first.
    y_ref, h_ref = mamba3_mimo_reference(*inputs)
    y_met, h_met = mamba3_mimo_fwd_metal(*inputs)
    mx.eval(y_ref, h_ref, y_met, h_met)
    parity = {
        "y_max_abs": _parity_diff(y_met, y_ref),
        "y_ref_norm": _value_norm(y_ref),
        "h_max_abs": _parity_diff(h_met, h_ref),
        "h_ref_norm": _value_norm(h_ref),
    }
    parity["y_rel"] = parity["y_max_abs"] / (parity["y_ref_norm"] + 1e-12)
    parity["h_rel"] = parity["h_max_abs"] / (parity["h_ref_norm"] + 1e-12)

    fwd_ref = _bench(
        "fwd_reference",
        lambda: mamba3_mimo_reference(*inputs),
        warmup=args.warmup,
        iters=args.iters,
    )
    fwd_met = _bench(
        "fwd_metal",
        lambda: mamba3_mimo_fwd_metal(*inputs),
        warmup=args.warmup,
        iters=args.iters,
    )

    # Forward + backward through autograd-through-reference.
    def loss_ref() -> mx.array:
        y, _ = mamba3_mimo_reference(*inputs)
        return mx.sum(y * y) * 0.5

    def loss_met() -> mx.array:
        y = mamba3_mimo_apply(*inputs)
        return mx.sum(y * y) * 0.5

    def grad_ref_loss(
        x: mx.array,
        B: mx.array,
        C: mx.array,
        z: mx.array,
        A: mx.array,
        dt: mx.array,
        D: mx.array,
        h0: mx.array,
    ) -> mx.array:
        y, _ = mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)
        return mx.sum(y * y) * 0.5

    def grad_met_loss(
        x: mx.array,
        B: mx.array,
        C: mx.array,
        z: mx.array,
        A: mx.array,
        dt: mx.array,
        D: mx.array,
        h0: mx.array,
    ) -> mx.array:
        y = cast(mx.array, mamba3_mimo_apply(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    grad_ref = mx.value_and_grad(grad_ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    grad_met = mx.value_and_grad(grad_met_loss, argnums=(0, 1, 2, 3, 4, 5, 6, 7))

    fwd_bwd_ref = _bench(
        "fwd_bwd_reference",
        lambda: grad_ref(*inputs),
        warmup=args.warmup,
        iters=args.iters,
    )
    fwd_bwd_met = _bench(
        "fwd_bwd_metal",
        lambda: grad_met(*inputs),
        warmup=args.warmup,
        iters=args.iters,
    )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local_only",
        "kernel": "mamba3_mimo_path_b",
        "hardware_label": args.hardware_label,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": _safe_version("mlx"),
        },
        "shape": {
            "batch": args.batch,
            "seq": args.seq,
            "heads": args.heads,
            "headdim": args.headdim,
            "state": args.state,
            "dtype": args.dtype,
            "d_inner": args.heads * args.headdim,
            "d_model_equivalent": args.heads * args.headdim // 2,
        },
        "metal_status": {
            "available": status.available,
            "reason": status.reason,
        },
        "parity": parity,
        "timings": {
            "fwd_reference": fwd_ref,
            "fwd_metal": fwd_met,
            "fwd_bwd_reference": fwd_bwd_ref,
            "fwd_bwd_metal": fwd_bwd_met,
        },
        "speedups": {
            "fwd": fwd_ref["mean_ms"] / fwd_met["mean_ms"]
            if fwd_met["mean_ms"] > 0 else None,
            "fwd_bwd": fwd_bwd_ref["mean_ms"] / fwd_bwd_met["mean_ms"]
            if fwd_bwd_met["mean_ms"] > 0 else None,
        },
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
