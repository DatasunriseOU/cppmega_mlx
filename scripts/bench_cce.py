#!/usr/bin/env python3
"""Benchmark cut-cross-entropy variants vs `mlx.nn.losses.cross_entropy`.

Apple's upstream `cut_cross_entropy` Triton kernel rejects MacOS at runtime
(``RuntimeError: CCE does not support MacOS``) and the package targets torch
tensors regardless. This script compares three MLX-native paths instead:

* ``materialized``         -- ``cross_entropy(e @ c.T, targets)``; today's default.
* ``chunked_forward``      -- :func:`linear_cross_entropy`; chunked forward only
  (backward via ``mx.grad`` keeps every chunk's trace live).
* ``chunked_eager_grad``   -- :func:`linear_cross_entropy_value_and_grad`;
  eager chunked forward+backward outside MLX autograd, so each chunk's
  ``[chunk_rows, V]`` tile is freed mid-backward.

For each path we time forward-only and forward+backward at a fixed shape
(default ``B=4 T=512 V=65536 D=3584``), record peak memory via
``mx.get_peak_memory``, and emit JSON to stdout (or ``--output``).

Run from the repo root::

    .venv/bin/python scripts/bench_cce.py
    .venv/bin/python scripts/bench_cce.py --batch-size 1 --seq-len 64 --vocab-size 1024 --hidden 256
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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.training.cut_cross_entropy import (  # noqa: E402
    DEFAULT_CHUNK_ROWS,
    linear_cross_entropy,
    linear_cross_entropy_value_and_grad,
    materialized_cross_entropy,
)

DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}
BENCH_RECEIPT_SCHEMA_VERSION = 1


def _try_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _hardware_label() -> str:
    machine = platform.machine() or "unknown"
    system = platform.system() or "unknown"
    return f"{system}-{machine}"


def _build_inputs(
    *,
    batch_size: int,
    seq_len: int,
    hidden: int,
    vocab: int,
    dtype: mx.Dtype,
    seed: int,
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    e = mx.random.normal((batch_size, seq_len, hidden)).astype(dtype)
    c = mx.random.normal((vocab, hidden)).astype(dtype)
    targets = mx.random.randint(0, vocab, (batch_size, seq_len))
    mx.eval(e, c, targets)
    return e, c, targets


def _time_forward(
    fn: Callable[[mx.array, mx.array, mx.array], mx.array],
    e: mx.array,
    c: mx.array,
    targets: mx.array,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    # Warm-up + cache priming.
    for _ in range(warmup):
        loss = fn(e, c, targets)
        mx.eval(loss)
    mx.reset_peak_memory()
    samples_ms: list[float] = []
    last_loss: float | None = None
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = fn(e, c, targets)
        mx.eval(loss)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss.item())
    peak_bytes = int(mx.get_peak_memory())
    return {
        "loss": last_loss,
        "peak_memory_bytes": peak_bytes,
        "wall_ms_mean": statistics.fmean(samples_ms),
        "wall_ms_min": min(samples_ms),
        "wall_ms_samples": samples_ms,
        "iters": iters,
        "warmup": warmup,
    }


def _time_forward_backward(
    grad_fn: Callable[
        [mx.array, mx.array, mx.array],
        tuple[mx.array, tuple[mx.array, mx.array]],
    ],
    e: mx.array,
    c: mx.array,
    targets: mx.array,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        loss, (de, dc) = grad_fn(e, c, targets)
        mx.eval(loss, de, dc)
    mx.reset_peak_memory()
    samples_ms: list[float] = []
    last_loss: float | None = None
    for _ in range(iters):
        t0 = time.perf_counter()
        loss, (de, dc) = grad_fn(e, c, targets)
        mx.eval(loss, de, dc)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
        last_loss = float(loss.item())
    peak_bytes = int(mx.get_peak_memory())
    return {
        "loss": last_loss,
        "peak_memory_bytes": peak_bytes,
        "wall_ms_mean": statistics.fmean(samples_ms),
        "wall_ms_min": min(samples_ms),
        "wall_ms_samples": samples_ms,
        "iters": iters,
        "warmup": warmup,
    }


def _forward_factory(
    name: str, *, chunk_rows: int
) -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    if name == "materialized":
        return lambda e, c, t: materialized_cross_entropy(e, c, t)
    if name == "chunked_forward":
        return lambda e, c, t: linear_cross_entropy(e, c, t, chunk_rows=chunk_rows)
    if name == "chunked_eager_grad":
        # For forward-only timing of the eager path we just call the chunked
        # forward; the eager grad path's gain is on the backward side.
        return lambda e, c, t: linear_cross_entropy(e, c, t, chunk_rows=chunk_rows)
    raise ValueError(f"unknown bench path: {name}")


def _grad_factory(
    name: str, *, chunk_rows: int
) -> Callable[
    [mx.array, mx.array, mx.array],
    tuple[mx.array, tuple[mx.array, mx.array]],
]:
    if name == "materialized":
        fn = lambda e, c, t: materialized_cross_entropy(e, c, t)  # noqa: E731
        return mx.value_and_grad(fn, argnums=(0, 1))
    if name == "chunked_forward":
        fn = lambda e, c, t: linear_cross_entropy(  # noqa: E731
            e, c, t, chunk_rows=chunk_rows
        )
        return mx.value_and_grad(fn, argnums=(0, 1))
    if name == "chunked_eager_grad":
        def grad_fn(
            e: mx.array, c: mx.array, t: mx.array
        ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
            loss, de, dc = linear_cross_entropy_value_and_grad(
                e, c, t, chunk_rows=chunk_rows
            )
            return loss, (de, dc)

        return grad_fn
    raise ValueError(f"unknown bench path: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MLX cross-entropy loss variants. The MLX-native chunked "
            "implementations stand in for Apple's cut-cross-entropy package, "
            "whose Triton kernel rejects MacOS."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="Rows per chunk for the chunked_forward path.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["materialized", "chunked_forward", "chunked_eager_grad"],
        help="Subset of paths to benchmark.",
    )
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help="Skip the forward+backward measurement (forward only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path; otherwise print to stdout.",
    )
    return parser


def run_bench(args: argparse.Namespace) -> dict[str, Any]:
    dtype = DTYPES[args.dtype]
    e, c, targets = _build_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden=args.hidden,
        vocab=args.vocab_size,
        dtype=dtype,
        seed=args.seed,
    )

    results: dict[str, dict[str, Any]] = {}
    for name in args.paths:
        fwd_fn = _forward_factory(name, chunk_rows=args.chunk_rows)
        path_result: dict[str, Any] = {
            "forward_only": _time_forward(
                fwd_fn, e, c, targets, warmup=args.warmup, iters=args.iters
            ),
        }
        if not args.skip_backward:
            grad_fn = _grad_factory(name, chunk_rows=args.chunk_rows)
            path_result["forward_backward"] = _time_forward_backward(
                grad_fn, e, c, targets, warmup=args.warmup, iters=args.iters
            )
        results[name] = path_result

    payload = {
        "schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "scope": "local_only",
        "policy": (
            "MLX-native cut-cross-entropy comparison; Apple's upstream Triton "
            "kernel rejects MacOS so this benchmark stands in for it."
        ),
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "vocab_size": args.vocab_size,
            "hidden": args.hidden,
            "dtype": args.dtype,
        },
        "config": {
            "chunk_rows": args.chunk_rows,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "skip_backward": bool(args.skip_backward),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx_version": _try_version("mlx") or "unknown",
            "cppmega_mlx_version": _try_version("cppmega-mlx") or "unknown",
            "cut_cross_entropy_version": _try_version("cut-cross-entropy"),
            "hardware_label": _hardware_label(),
        },
        "results": results,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run_bench(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
