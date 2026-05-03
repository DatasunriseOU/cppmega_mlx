"""Benchmark the three Mamba3 backward helpers: TileLang/Metal vs pure-MLX.

Compares the Path B TileLang kernel implementations
(``cppmega_mlx.nn._tilelang._mamba3_helpers_tilelang``) against the pure-MLX
sibling rewrites (``cppmega_mlx.nn._tilelang._mamba3_helpers``) at a few
representative shapes drawn from real Mamba3 configurations.

Output JSON layout::

    {
      "machine": {...},
      "tilelang_status": {...},
      "shapes": [
        {
          "label": "mamba3-default",
          "B": 2, "T": 128, "H": 4, "P": 64, "N": 16,
          "compute_dacs_segsum": {
            "pure_mlx": {"median_ms": ..., "min_ms": ..., "max_ms": ...},
            "tilelang_metal": {"median_ms": ..., "peak_gib": ...},
            "speedup_tilelang_over_pure": float,
            "max_abs_err": float
          },
          "bwd_dadt_fused": {...},
          "bwd_dtrap_ddt": {...}
        },
        ...
      ]
    }

Run::

    python scripts/bench_tilelang_mamba3_helpers.py

The output file ``bench/tilelang_ports/mamba3_helpers.json`` is overwritten on
each run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.nn._tilelang import _mamba3_helpers as pure_helpers  # noqa: E402
from cppmega_mlx.nn._tilelang import _mamba3_helpers_tilelang as tl_helpers  # noqa: E402

RESULTS_PATH = REPO_ROOT / "bench" / "tilelang_ports" / "mamba3_helpers.json"


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


SHAPE_PRESETS = [
    # (label, B, T, H, P, N) — N is unused for dtrap (which is (B,T,H) only).
    ("tiny", 1, 16, 4, 4, 8),
    ("small", 2, 32, 4, 4, 8),
    ("medium", 4, 64, 8, 4, 16),
    ("mamba3-default", 2, 128, 4, 64, 16),
]


# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------


def _bench(fn: Callable[[], Any], *, warmup: int = 5, iters: int = 30) -> dict[str, float]:
    """Run fn(), evaluate every output, and return median/min/max ms."""

    for _ in range(warmup):
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return {
        "median_ms": float(times[len(times) // 2]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
    }


def _max_abs_err(a: mx.array, b: mx.array) -> float:
    a32 = np.array(a).astype(np.float32)
    b32 = np.array(b).astype(np.float32)
    return float(np.max(np.abs(a32 - b32)))


def _peak_gib(arrays: list[mx.array]) -> float:
    total = 0
    for a in arrays:
        total += int(a.nbytes)
    return total / (1024 ** 3)


# ---------------------------------------------------------------------------
# Per-helper benches
# ---------------------------------------------------------------------------


def _bench_segsum(B: int, T_: int, H: int, P: int, N: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    A_np = (rng.standard_normal((B, T_, H)) * 0.1 - 0.5).astype(np.float32)
    dt_np = (np.abs(rng.standard_normal((B, T_, H))) * 0.1 + 1e-3).astype(np.float32)
    dh_np = rng.standard_normal((B, T_, H, P, N)).astype(np.float16)
    A = mx.array(A_np)
    dt = mx.array(dt_np)
    dh = mx.array(dh_np)
    mx.eval(A, dt, dh)

    out_pure = pure_helpers.compute_dacs_segsum(A, dt, dh)
    out_tl = tl_helpers.compute_dacs_segsum(A, dt, dh)
    mx.eval(out_pure, out_tl)
    err = _max_abs_err(out_tl, out_pure)

    pure_stats = _bench(lambda: pure_helpers.compute_dacs_segsum(A, dt, dh))
    tl_stats = _bench(lambda: tl_helpers.compute_dacs_segsum(A, dt, dh))

    return {
        "pure_mlx": pure_stats,
        "tilelang_metal": {**tl_stats, "peak_gib": _peak_gib([A, dt, dh, out_tl])},
        "speedup_tilelang_over_pure": pure_stats["median_ms"] / tl_stats["median_ms"]
            if tl_stats["median_ms"] > 0 else float("inf"),
        "max_abs_err": err,
    }


def _bench_dadt(B: int, T_: int, H: int, P: int, N: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    A = mx.array((rng.standard_normal((B, T_, H)) * 0.1 - 0.5).astype(np.float32))
    dt = mx.array((np.abs(rng.standard_normal((B, T_, H))) * 0.1 + 1e-3).astype(np.float32))
    dY = mx.array(rng.standard_normal((B, T_, H, P, N)).astype(np.float16))
    h = mx.array(rng.standard_normal((B, T_, H, P, N)).astype(np.float16))
    mx.eval(A, dt, dY, h)

    dA_p, ddt_p = pure_helpers.bwd_dadt_fused(dY, A, dt, h)
    dA_t, ddt_t = tl_helpers.bwd_dadt_fused(dY, A, dt, h)
    mx.eval(dA_p, ddt_p, dA_t, ddt_t)
    err = max(_max_abs_err(dA_t, dA_p), _max_abs_err(ddt_t, ddt_p))

    pure_stats = _bench(lambda: pure_helpers.bwd_dadt_fused(dY, A, dt, h))
    tl_stats = _bench(lambda: tl_helpers.bwd_dadt_fused(dY, A, dt, h))

    return {
        "pure_mlx": pure_stats,
        "tilelang_metal": {**tl_stats, "peak_gib": _peak_gib([A, dt, dY, h, dA_t, ddt_t])},
        "speedup_tilelang_over_pure": pure_stats["median_ms"] / tl_stats["median_ms"]
            if tl_stats["median_ms"] > 0 else float("inf"),
        "max_abs_err": err,
    }


def _bench_dtrap(B: int, T_: int, H: int, *, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dB = mx.array(rng.standard_normal((B, T_, H)).astype(np.float16))
    dt = mx.array((np.abs(rng.standard_normal((B, T_, H))) * 0.1 + 1e-3).astype(np.float16))
    trap = mx.array(rng.standard_normal((B, T_, H)).astype(np.float16))
    mx.eval(dB, dt, trap)

    ddt_p, dtrap_p = pure_helpers.bwd_dtrap_ddt(dB, dt, trap)
    ddt_t, dtrap_t = tl_helpers.bwd_dtrap_ddt(dB, dt, trap)
    mx.eval(ddt_p, dtrap_p, ddt_t, dtrap_t)
    err = max(_max_abs_err(ddt_t, ddt_p), _max_abs_err(dtrap_t, dtrap_p))

    pure_stats = _bench(lambda: pure_helpers.bwd_dtrap_ddt(dB, dt, trap))
    tl_stats = _bench(lambda: tl_helpers.bwd_dtrap_ddt(dB, dt, trap))

    return {
        "pure_mlx": pure_stats,
        "tilelang_metal": {**tl_stats, "peak_gib": _peak_gib([dB, dt, trap, ddt_t, dtrap_t])},
        "speedup_tilelang_over_pure": pure_stats["median_ms"] / tl_stats["median_ms"]
            if tl_stats["median_ms"] > 0 else float("inf"),
        "max_abs_err": err,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    metal = getattr(mx, "metal", None)
    if metal is not None and metal.is_available():
        try:
            info["metal_device_info"] = str(metal.device_info())
        except Exception:
            info["metal_device_info"] = "available"
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help=f"Path to write JSON output (default: {RESULTS_PATH}).",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status = tl_helpers.helpers_metal_status()
    print(f"TileLang status: available={status.available} reason={status.reason}")

    if not status.available:
        result = {
            "machine": _machine_info(),
            "tilelang_status": asdict(status),
            "skip_reason": status.reason,
            "shapes": [],
        }
        output_path.write_text(json.dumps(result, indent=2))
        print(f"Wrote skip notice to {output_path}")
        return

    # Re-bind warmup/iters defaults via the closures.
    global _bench
    _orig_bench = _bench

    def _patched_bench(fn, *, warmup=args.warmup, iters=args.iters):
        return _orig_bench(fn, warmup=warmup, iters=iters)

    _bench = _patched_bench  # type: ignore[assignment]

    shapes_out: list[dict[str, Any]] = []
    for label, B, T_, H, P, N in SHAPE_PRESETS:
        print(f"\n=== {label} (B={B}, T={T_}, H={H}, P={P}, N={N}) ===")
        seg = _bench_segsum(B, T_, H, P, N, seed=hash(label) & 0xFFFF)
        dadt = _bench_dadt(B, T_, H, P, N, seed=(hash(label) ^ 1) & 0xFFFF)
        dtrap = _bench_dtrap(B, T_, H, seed=(hash(label) ^ 2) & 0xFFFF)
        for name, payload in (("compute_dacs_segsum", seg), ("bwd_dadt_fused", dadt), ("bwd_dtrap_ddt", dtrap)):
            pure_ms = payload["pure_mlx"]["median_ms"]
            tl_ms = payload["tilelang_metal"]["median_ms"]
            err = payload["max_abs_err"]
            print(
                f"  {name}: pure={pure_ms:.4f}ms  tilelang={tl_ms:.4f}ms  "
                f"speedup={payload['speedup_tilelang_over_pure']:.2f}x  err={err:.2e}"
            )
        shapes_out.append({
            "label": label,
            "B": B,
            "T": T_,
            "H": H,
            "P": P,
            "N": N,
            "compute_dacs_segsum": seg,
            "bwd_dadt_fused": dadt,
            "bwd_dtrap_ddt": dtrap,
        })

    payload = {
        "machine": _machine_info(),
        "tilelang_status": asdict(status),
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": shapes_out,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults -> {output_path}")


if __name__ == "__main__":
    main()
