"""Benchmark the retired FP8 helper surface against ``mx.matmul`` baselines.

Output JSON: ``bench/tilelang_ports/fp8_msl_kernels.json``

The helpers under test live in
``cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py``. The historical direct-MSL
surface is retired; this script now records the pure-MLX reference timings and
the explicit unavailable status.

Apple Silicon (through M5 / MSL 4.0) has no native FP8 hardware, so this
bench measures the realistic overhead of treating FP8 as storage-only:
LUT-based decode in MSL constant memory plus regular fp32 fma in
register. The fp16 ``mx.matmul`` baseline is the practical performance
ceiling for any FP8 path on this hardware.

Bench shapes mirror the canonical FP8 inference shapes from the upstream
projects:

  * 4096 x 4096 x 4096 -- square (decoder-style)
  * 2048 x  512 x 4096 -- vec-mat-ish (cross-attention column projection)
  *  512 x  512 x 8192 -- Mamba-like K-major projection

Each shape is run for: encode (fp16 -> fp8), decode (fp8 -> fp16),
reference fp8_scaled_matmul, reference fp8_scaled_vecmat (M=1 only), and an
mx.matmul fp16 baseline computed on the dequantized inputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (  # noqa: E402
    fp8_msl_status,
    fp8_scaled_matmul_raw,
    fp8_scaled_vecmat,
    fp8_to_half,
    half_to_fp8,
)


SHAPES = [
    {"label": "square_4096", "M": 4096, "N": 4096, "K": 4096},
    {"label": "tall_skinny", "M": 2048, "N": 512, "K": 4096},
    {"label": "fat_K", "M": 512, "N": 512, "K": 8192},
    {"label": "tiny_smoke", "M": 64, "N": 64, "K": 128},
]


def _bench(
    label: str,
    fn: Callable[[], mx.array | tuple[mx.array, ...]],
    *,
    warmup: int = 5,
    iters: int = 25,
) -> dict[str, Any]:
    for _ in range(warmup):
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "label": label,
        "median_ms": float(times[len(times) // 2]),
        "min_ms": float(min(times)),
        "max_ms": float(max(times)),
        "iters": iters,
        "warmup": warmup,
    }


def _bytes(arr: mx.array) -> int:
    n = 1
    for d in arr.shape:
        n *= d
    return n * arr.dtype.size


def _build_inputs(M: int, N: int, K: int, seed: int = 0) -> dict[str, mx.array]:
    rng = np.random.default_rng(seed)
    A = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    B = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    A_fp8 = mx.to_fp8(A)
    B_fp8 = mx.to_fp8(B)
    A_f16 = A.astype(mx.float16)
    B_f16 = B.astype(mx.float16)
    sa = mx.array([1.0], dtype=mx.float32)
    sb = mx.array([1.0], dtype=mx.float32)
    mx.eval(A, B, A_fp8, B_fp8, A_f16, B_f16, sa, sb)
    return {
        "A_f16": A_f16,
        "B_f16": B_f16,
        "A_fp8": A_fp8,
        "B_fp8": B_fp8,
        "scale_a": sa,
        "scale_b": sb,
    }


def _bench_shape(shape: dict[str, Any], *, warmup: int, iters: int) -> dict[str, Any]:
    M, N, K = shape["M"], shape["N"], shape["K"]
    label = shape["label"]
    inputs = _build_inputs(M, N, K)

    rows: list[dict[str, Any]] = []

    # Encode (fp16 -> fp8) and decode (fp8 -> fp16).
    rows.append(
        _bench(
            f"{label}/half_to_fp8 (A {M}x{K})",
            lambda: half_to_fp8(inputs["A_f16"]),
            warmup=warmup,
            iters=iters,
        )
    )
    rows.append(
        _bench(
            f"{label}/fp8_to_half (A {M}x{K})",
            lambda: fp8_to_half(inputs["A_fp8"]),
            warmup=warmup,
            iters=iters,
        )
    )

    # Scaled FP8 matmul, fp8 -> fp32 output.
    def _matmul_fp8():
        return fp8_scaled_matmul_raw(
            inputs["A_fp8"],
            inputs["B_fp8"],
            scale_a=inputs["scale_a"],
            scale_b=inputs["scale_b"],
        )

    rows.append(
        _bench(
            f"{label}/fp8_scaled_matmul ({M}x{K} @ {K}x{N})",
            _matmul_fp8,
            warmup=warmup,
            iters=iters,
        )
    )

    # mx.matmul fp16 baseline.
    def _matmul_fp16():
        return mx.matmul(inputs["A_f16"], mx.swapaxes(inputs["B_f16"], 0, 1))

    rows.append(
        _bench(
            f"{label}/mx_matmul_fp16 ({M}x{K} @ {K}x{N})",
            _matmul_fp16,
            warmup=warmup,
            iters=iters,
        )
    )

    # Vecmat (M=1 slice) -- the upstream FP8 inference path's hot loop.
    if K % 4 == 0:
        x_fp8 = mx.to_fp8(mx.zeros(K, dtype=mx.float32))
        sx = mx.array([1.0], dtype=mx.float32)
        sw = mx.array([1.0], dtype=mx.float32)
        mx.eval(x_fp8)

        def _vecmat_fp8():
            return fp8_scaled_vecmat(
                x_fp8, inputs["B_fp8"], scale_x=sx, scale_w=sw
            )

        rows.append(
            _bench(
                f"{label}/fp8_scaled_vecmat (1x{K} @ {K}x{N})",
                _vecmat_fp8,
                warmup=warmup,
                iters=iters,
            )
        )

    # Per-shape memory accounting for the scaled matmul path.
    fp8_bytes = _bytes(inputs["A_fp8"]) + _bytes(inputs["B_fp8"])
    fp16_bytes = _bytes(inputs["A_f16"]) + _bytes(inputs["B_f16"])

    return {
        "shape_label": label,
        "M": M,
        "N": N,
        "K": K,
        "fp8_input_bytes": fp8_bytes,
        "fp16_input_bytes": fp16_bytes,
        "memory_savings_pct": float(1.0 - fp8_bytes / fp16_bytes) * 100.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="bench/tilelang_ports/fp8_msl_kernels.json",
        help="Output JSON path (relative to repo root).",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="Optional subset of shape labels (e.g. tiny_smoke).",
    )
    args = parser.parse_args()

    selected = (
        SHAPES if args.shapes is None else [s for s in SHAPES if s["label"] in args.shapes]
    )
    if not selected:
        raise SystemExit("No matching shapes selected")

    status = fp8_msl_status()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": "local_only",
        "kernel": "fp8_reference_helpers",
        "license_notice": (
            "Historical direct-MSL sources were vendored from "
            "AppMana/mps-fp8-for-torch-and-comfyui-python-package "
            "(commit a902571e, Apache 2.0) and audiohacking/fp8-mps-metal "
            "(commit d4fbd40c, MIT); current helpers are pure MLX references."
        ),
        "hardware_label": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
        },
        "metal_status": {
            "available": status.available,
            "reason": status.reason,
        },
        "shapes": [
            _bench_shape(s, warmup=args.warmup, iters=args.iters) for s in selected
        ],
    }

    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote bench JSON: {out_path}")
    for shape in payload["shapes"]:
        print(f"\n=== {shape['shape_label']} (M={shape['M']} N={shape['N']} K={shape['K']}) ===")
        for row in shape["rows"]:
            print(
                f"  {row['label']:<60} median {row['median_ms']:8.3f} ms  "
                f"min {row['min_ms']:8.3f} ms  iters {row['iters']}"
            )


if __name__ == "__main__":
    main()
