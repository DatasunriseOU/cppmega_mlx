#!/usr/bin/env python3
"""Local receipt for Muon Newton-Schulz carrier policy.

This is a local Apple Silicon receipt, not a GB10 parity claim. It exercises
the repo-local ``MuonWithNSCarrier`` implementation with fp32 and bf16 carriers,
records numerical drift on the orthogonalized matrix, times the NS loop, and
runs a deterministic tiny 100-step Muon smoke for loss-regression evidence.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from cppmega_mlx.training.optimizers import (  # noqa: E402
    MUON_NS_CARRIER_ENV,
    _muon_zeropower_newtonschulz5,
    make_muon,
)


SCHEMA_VERSION = 1
SOURCE_BEAD = "cppmega-mlx-c08.4"
DEFAULT_OUTPUT = Path("bench/baselines/c08_4_muon_ns_carrier_m4.json")


class _TinyMuonSmokeModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


def _try_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _metal_device_info() -> str | None:
    device_info = getattr(mx, "device_info", None)
    if device_info is not None:
        try:
            return str(device_info())
        except Exception:  # pragma: no cover - diagnostic metadata only
            pass
    metal = getattr(mx, "metal", None)
    device_info = getattr(metal, "device_info", None)
    if device_info is None:
        return None
    try:
        return str(device_info())
    except Exception:  # pragma: no cover - diagnostic metadata only
        return None


def _make_update(*, matrix_size: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    update = mx.random.normal((matrix_size, matrix_size)).astype(mx.float32) * 0.02
    mx.eval(update)
    return update


def _sync_peak_memory() -> int | None:
    get_peak_memory = getattr(mx, "get_peak_memory", None)
    if get_peak_memory is None:
        return None
    try:
        return int(get_peak_memory())
    except Exception:  # pragma: no cover - runtime diagnostic only
        return None


def _reset_peak_memory() -> None:
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if reset_peak_memory is not None:
        reset_peak_memory()


def _stats(samples_ms: list[float], *, warmup: int, iters: int) -> dict[str, Any]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "warmup": warmup,
        "iters": iters,
        "peak_memory_bytes": _sync_peak_memory(),
    }


def _time_ns_loop(
    update: mx.array,
    *,
    ns_carrier: str,
    ns_steps: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        out = _muon_zeropower_newtonschulz5(
            update,
            steps=ns_steps,
            ns_carrier=ns_carrier,  # type: ignore[arg-type]
            output_dtype=mx.float32,
        )
        mx.eval(out)

    _reset_peak_memory()
    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = _muon_zeropower_newtonschulz5(
            update,
            steps=ns_steps,
            ns_carrier=ns_carrier,  # type: ignore[arg-type]
            output_dtype=mx.float32,
        )
        mx.eval(out)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    stats = _stats(samples_ms, warmup=warmup, iters=iters)
    stats["median_per_ns_iter_ms"] = stats["median_ms"] / ns_steps
    return stats


def _max_abs_orthogonalization_error(
    update: mx.array,
    *,
    ns_steps: int,
) -> float:
    fp32 = _muon_zeropower_newtonschulz5(
        update,
        steps=ns_steps,
        ns_carrier="fp32",
        output_dtype=mx.float32,
    )
    bf16 = _muon_zeropower_newtonschulz5(
        update,
        steps=ns_steps,
        ns_carrier="bf16",
        output_dtype=mx.float32,
    )
    err = mx.max(mx.abs(fp32 - bf16))
    mx.eval(err)
    return float(err.item())


def _make_smoke_batch(
    *,
    batch_size: int,
    hidden: int,
    seed: int,
) -> tuple[mx.array, mx.array]:
    mx.random.seed(seed)
    x = mx.random.normal((batch_size, hidden)).astype(mx.bfloat16)
    true_w = mx.random.normal((hidden, hidden)).astype(mx.float32) * 0.1
    true_b = mx.random.normal((hidden,)).astype(mx.float32) * 0.01
    y = x.astype(mx.float32) @ true_w.T + true_b
    mx.eval(x, y)
    return x, y


def _run_smoke(
    *,
    ns_carrier: str,
    steps: int,
    hidden: int,
    batch_size: int,
    seed: int,
    ns_steps: int,
    lr_muon: float,
    lr_adamw: float,
) -> dict[str, Any]:
    x, y = _make_smoke_batch(batch_size=batch_size, hidden=hidden, seed=seed + 101)
    mx.random.seed(seed)
    model = _TinyMuonSmokeModel(hidden)
    model.set_dtype(mx.bfloat16)
    optimizer = make_muon(
        ns_carrier=ns_carrier,
        ns_steps=ns_steps,
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
        weight_decay=0.0,
    )
    optimizer.init(model.trainable_parameters())

    def loss_fn(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        pred = m(x).astype(mx.float32)
        return mx.mean(mx.square(pred - y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []
    for _ in range(steps):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))

    return {
        "ns_carrier": ns_carrier,
        "steps": steps,
        "hidden": hidden,
        "batch_size": batch_size,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "mean_last10_loss": statistics.fmean(losses[-min(10, len(losses)) :]),
        "loss_decreased": losses[-1] < losses[0],
        "losses": losses,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a local-only Muon NS carrier receipt for cppmega-mlx-c08.4. "
            "No GB10 parity or cross-host throughput claim is made."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=100)
    parser.add_argument("--smoke-hidden", type=int, default=16)
    parser.add_argument("--smoke-batch-size", type=int, default=32)
    parser.add_argument("--smoke-lr-muon", type=float, default=2e-3)
    parser.add_argument("--smoke-lr-adamw", type=float, default=1e-4)
    parser.add_argument(
        "--orthogonalization-atol",
        type=float,
        default=1e-2,
        help="Acceptance threshold for bf16-vs-fp32 orthogonalized matrix drift.",
    )
    parser.add_argument(
        "--speedup-lower-bound",
        type=float,
        default=1.45,
        help="Local M4 median speedup floor for 'around 1.5x' acceptance.",
    )
    parser.add_argument(
        "--smoke-loss-regression-rtol",
        type=float,
        default=0.05,
        help="Allowed relative final-loss regression for bf16 vs fp32 smoke.",
    )
    parser.add_argument(
        "--smoke-loss-regression-atol",
        type=float,
        default=1e-3,
        help="Allowed absolute final-loss regression for bf16 vs fp32 smoke.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Write JSON receipt to this path. Default: {DEFAULT_OUTPUT}",
    )
    return parser


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    update = _make_update(matrix_size=args.matrix_size, seed=args.seed)
    max_abs_error = _max_abs_orthogonalization_error(update, ns_steps=args.ns_steps)
    fp32_timing = _time_ns_loop(
        update,
        ns_carrier="fp32",
        ns_steps=args.ns_steps,
        warmup=args.warmup,
        iters=args.iters,
    )
    bf16_timing = _time_ns_loop(
        update,
        ns_carrier="bf16",
        ns_steps=args.ns_steps,
        warmup=args.warmup,
        iters=args.iters,
    )
    median_speedup = fp32_timing["median_ms"] / bf16_timing["median_ms"]

    fp32_smoke = _run_smoke(
        ns_carrier="fp32",
        steps=args.smoke_steps,
        hidden=args.smoke_hidden,
        batch_size=args.smoke_batch_size,
        seed=args.seed,
        ns_steps=args.ns_steps,
        lr_muon=args.smoke_lr_muon,
        lr_adamw=args.smoke_lr_adamw,
    )
    bf16_smoke = _run_smoke(
        ns_carrier="bf16",
        steps=args.smoke_steps,
        hidden=args.smoke_hidden,
        batch_size=args.smoke_batch_size,
        seed=args.seed,
        ns_steps=args.ns_steps,
        lr_muon=args.smoke_lr_muon,
        lr_adamw=args.smoke_lr_adamw,
    )
    allowed_final_loss = (
        fp32_smoke["final_loss"] * (1.0 + args.smoke_loss_regression_rtol)
        + args.smoke_loss_regression_atol
    )
    bf16_no_regression = bf16_smoke["final_loss"] <= allowed_final_loss

    acceptance = {
        "max_abs_orthogonalization_error": max_abs_error,
        "orthogonalization_atol": args.orthogonalization_atol,
        "orthogonalization_within_atol": max_abs_error <= args.orthogonalization_atol,
        "median_ns_loop_speedup_fp32_over_bf16": median_speedup,
        "speedup_lower_bound": args.speedup_lower_bound,
        "meets_local_speedup_gate": median_speedup >= args.speedup_lower_bound,
        "bf16_no_loss_regression_vs_fp32": bf16_no_regression,
        "bf16_loss_decreased": bool(bf16_smoke["loss_decreased"]),
        "fp32_loss_decreased": bool(fp32_smoke["loss_decreased"]),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "cppmega.mlx.local_m4_muon_ns_carrier_receipt",
        "source_bead": SOURCE_BEAD,
        "guards": {
            "local_only": True,
            "gb10_parity_claim": False,
            "m4_vs_gb10_throughput_parity_claim": False,
            "gb10_training_correctness_claim": False,
            "first_call_compile_time_excluded": True,
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "mlx": _try_version("mlx"),
            "cppmega_mlx": _try_version("cppmega-mlx"),
            "metal_device_info": _metal_device_info(),
        },
        "config": {
            "matrix_size": args.matrix_size,
            "ns_steps": args.ns_steps,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "smoke_steps": args.smoke_steps,
            "smoke_hidden": args.smoke_hidden,
            "smoke_batch_size": args.smoke_batch_size,
            "smoke_lr_muon": args.smoke_lr_muon,
            "smoke_lr_adamw": args.smoke_lr_adamw,
            "muon_ns_carrier_env": MUON_NS_CARRIER_ENV,
        },
        "timing": {
            "fp32": fp32_timing,
            "bf16": bf16_timing,
        },
        "smoke": {
            "fp32": fp32_smoke,
            "bf16": bf16_smoke,
            "allowed_bf16_final_loss": allowed_final_loss,
        },
        "acceptance": acceptance,
        "notes": [
            "Local Apple Silicon receipt only; does not claim GB10 parity.",
            "The bf16 carrier only changes the Newton-Schulz polynomial carrier dtype.",
            "Muon momentum state remains fp32 and parameter dtype is preserved by unit tests.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = build_receipt(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    output = args.output
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
