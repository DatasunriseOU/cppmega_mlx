#!/usr/bin/env python3
"""Benchmark cppmega's topk_selector forward across MLX/Metal strategies.

The cppmega TileLang ``topk_selector`` kernel returns the indices of the
``k`` largest values per batch row. TileLang's own Metal lowering is still
blocked by tilelang 0.1.9's ``shared.dyn`` / ``LowerTileOp`` failures, so the
Mac path uses a direct-MSL bypass through ``mx.fast.metal_kernel`` plus pure-MLX
fallback strategies.

Strategies:

* ``argpartition`` -- ``mx.argpartition(-scores, k, axis=-1)[..., :k]``;
  matches the reference implementation in
  ``cppmega_mlx/nn/_tilelang/topk_selector.topk_selector_reference``.
* ``argsort_slice`` -- ``mx.argsort(-scores, axis=-1)[..., :k]``; the
  cleanest baseline. Higher work, but kept honest as an upper bound.
* ``topk_take_along`` -- ``mx.argpartition(...)[..., :k]`` followed by
  ``mx.take_along_axis(scores, indices, axis=-1)`` so the values are
  materialized too. This is the "fused selector" shape callers use.
* ``path_b_msl`` -- hand-written direct-MSL Metal kernel. If it cannot dispatch
  for a shape/device, the bench records ``ran=false`` instead of timing a
  fallback as if it were Metal.

Run from the repo root::

    .venv/bin/python scripts/bench_tilelang_topk.py
    .venv/bin/python scripts/bench_tilelang_topk.py --json

JSON output is written to ``bench/tilelang_ports/topk_selector.json`` by
default and the same payload is emitted on stdout when ``--json`` is set.
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

from cppmega_mlx.nn._tilelang.topk_selector import (  # noqa: E402
    topk_selector_metal,
    topk_selector_path_b_status,
    topk_selector_reference,
)

DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

DEFAULT_OUTPUT = ROOT / "bench" / "tilelang_ports" / "topk_selector.json"
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
    batch: int,
    seq_len: int,
    dtype: mx.Dtype,
    seed: int,
) -> mx.array:
    mx.random.seed(seed)
    scores = mx.random.normal((batch, seq_len)).astype(dtype)
    mx.eval(scores)
    return scores


def _strategy_argpartition(scores: mx.array, k: int) -> mx.array:
    return topk_selector_reference(scores, k)


def _strategy_argsort_slice(scores: mx.array, k: int) -> mx.array:
    seq_len = int(scores.shape[1])
    if k == seq_len:
        return mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], scores.shape)
    return mx.argsort(-scores, axis=-1)[..., :k].astype(mx.int32)


def _strategy_topk_take_along(scores: mx.array, k: int) -> tuple[mx.array, mx.array]:
    indices = topk_selector_reference(scores, k)
    values = mx.take_along_axis(scores, indices.astype(mx.int32), axis=-1)
    return indices, values


def _strategy_path_b_msl(scores: mx.array, k: int) -> mx.array:
    """Direct-MSL Path B kernel (bypasses TileLang)."""

    out = topk_selector_metal(scores, k)
    if out is None:
        raise RuntimeError("direct-MSL Path B kernel did not dispatch")
    return out


_STRATEGIES: dict[str, Callable[[mx.array, int], Any]] = {
    "argpartition": _strategy_argpartition,
    "argsort_slice": _strategy_argsort_slice,
    "topk_take_along": _strategy_topk_take_along,
    "path_b_msl": _strategy_path_b_msl,
}


def _time_strategy(
    fn: Callable[[mx.array, int], Any],
    scores: mx.array,
    k: int,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    ran = True
    error: str | None = None
    try:
        for _ in range(warmup):
            out = fn(scores, k)
            if isinstance(out, tuple):
                mx.eval(*out)
            else:
                mx.eval(out)
    except RuntimeError as exc:
        ran = False
        error = str(exc)
    if not ran:
        return {
            "ran": False,
            "error": error,
            "iters": iters,
            "warmup": warmup,
            "median_ms": None,
            "min_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "peak_gib": None,
        }
    samples_ms: list[float] = []
    peaks: list[int] = []
    reset_fn = getattr(mx, "reset_peak_memory", None)
    get_peak_fn = getattr(mx, "get_peak_memory", None)
    for _ in range(iters):
        if callable(reset_fn):
            reset_fn()
        t0 = time.perf_counter()
        out = fn(scores, k)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
        if callable(get_peak_fn):
            peak_value = cast(Callable[[], int | float], get_peak_fn)()
            peaks.append(int(peak_value))
    samples_ms.sort()
    peak_bytes = max(peaks) if peaks else 0
    return {
        "iters": iters,
        "warmup": warmup,
        "ran": True,
        "median_ms": float(statistics.median(samples_ms)),
        "min_ms": float(samples_ms[0]),
        "max_ms": float(samples_ms[-1]),
        "mean_ms": float(statistics.fmean(samples_ms)),
        "peak_gib": float(peak_bytes / (1024**3)) if peak_bytes else None,
    }


def _bench_shape(
    *,
    batch: int,
    seq_len: int,
    k: int,
    dtype_name: str,
    seed: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    dtype = DTYPES[dtype_name]
    scores = _build_inputs(batch=batch, seq_len=seq_len, dtype=dtype, seed=seed)
    rows: dict[str, dict[str, Any]] = {}
    for label, fn in _STRATEGIES.items():
        rows[label] = _time_strategy(fn, scores, k, warmup=warmup, iters=iters)
    return {
        "batch": batch,
        "seq_len": seq_len,
        "k": k,
        "dtype": dtype_name,
        "strategies": rows,
    }


def _build_payload(
    *,
    shapes: list[dict[str, int | str]],
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in shapes:
        rows.append(
            _bench_shape(
                batch=int(spec["batch"]),
                seq_len=int(spec["seq_len"]),
                k=int(spec["k"]),
                dtype_name=str(spec["dtype"]),
                seed=seed,
                warmup=warmup,
                iters=iters,
            )
        )
    return {
        "schema_version": BENCH_RECEIPT_SCHEMA_VERSION,
        "kernel": "tilelang_topk_selector",
        "source": "cppmega/megatron/tilelang_sparse_mla/topk_selector.py",
        "path_b_status": {
            "available": (status := topk_selector_path_b_status()).available,
            "reason": status.reason,
        },
        "hardware_label": _hardware_label(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "package_versions": {
            "mlx": _try_version("mlx"),
            "mlx_metal": _try_version("mlx-metal"),
            "tilelang": _try_version("tilelang"),
            "numpy": _try_version("numpy"),
        },
        "warmup": warmup,
        "iters": iters,
        "seed": seed,
        "rows": rows,
    }


def _default_shapes() -> list[dict[str, int | str]]:
    return [
        {"batch": 1, "seq_len": 64, "k": 8, "dtype": "float32"},
        {"batch": 1, "seq_len": 512, "k": 32, "dtype": "float32"},
        {"batch": 1, "seq_len": 2048, "k": 64, "dtype": "float32"},
        {"batch": 4, "seq_len": 2048, "k": 64, "dtype": "float32"},
        {"batch": 4, "seq_len": 2048, "k": 64, "dtype": "float16"},
        {"batch": 4, "seq_len": 2048, "k": 64, "dtype": "bfloat16"},
        {"batch": 4, "seq_len": 4096, "k": 256, "dtype": "float32"},
    ]


def _format_table(payload: dict[str, Any]) -> str:
    headers = ["B", "T", "k", "dtype", "argpart_ms", "argsort_ms", "fused_ms", "msl_ms", "peak_gib"]
    width = [3, 6, 5, 9, 12, 12, 12, 12, 9]
    out_lines = ["  ".join(h.ljust(w) for h, w in zip(headers, width))]
    for row in payload["rows"]:
        ap = row["strategies"]["argpartition"]["median_ms"]
        ass = row["strategies"]["argsort_slice"]["median_ms"]
        fused = row["strategies"]["topk_take_along"]["median_ms"]
        msl = row["strategies"].get("path_b_msl", {}).get("median_ms")
        msl_str = f"{msl:.4f}" if isinstance(msl, float) else "SKIP"
        peak = row["strategies"]["argpartition"].get("peak_gib")
        peak_str = f"{peak:.4f}" if peak else "-"
        line = "  ".join([
            str(row["batch"]).ljust(width[0]),
            str(row["seq_len"]).ljust(width[1]),
            str(row["k"]).ljust(width[2]),
            row["dtype"].ljust(width[3]),
            f"{ap:.4f}".ljust(width[4]),
            f"{ass:.4f}".ljust(width[5]),
            f"{fused:.4f}".ljust(width[6]),
            msl_str.ljust(width[7]),
            peak_str.ljust(width[8]),
        ])
        out_lines.append(line)
    return "\n".join(out_lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path (default: bench/tilelang_ports/topk_selector.json)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout in addition to writing the output file.",
    )
    p.add_argument(
        "--no-output-file",
        action="store_true",
        help="Skip writing to the output JSON file (useful for ad-hoc runs).",
    )
    args = p.parse_args(argv)

    payload = _build_payload(
        shapes=_default_shapes(),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
    )

    if not args.no_output_file:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        if not args.json:
            print(f"wrote {args.output}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("# topk_selector MLX/Metal strategy comparison")
        print(f"# Path B available: {payload['path_b_status']['available']}")
        print(f"# warmup={payload['warmup']} iters={payload['iters']} seed={payload['seed']}")
        print()
        print(_format_table(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
