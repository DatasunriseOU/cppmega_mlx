#!/usr/bin/env python3
"""Local receipt for the symmetric 8-bit Lion (Lion8bit) optimizer state.

Measures the optimizer-state byte total and 1-step time for
``LionFP32Moments``, the unfused (Python chain) :class:`Lion8bit` and the
fused-kernel :class:`Lion8bit` against the full ``local_gb10_quarter``
1.797B-param bf16 model. The point is to capture the local M4 Apple-Silicon
evidence that Lion8bit cuts the optimizer-state footprint by ~3.7-3.9x and
that the fused Metal kernel cuts step time below the LionFP32Moments
baseline, mirroring ``bitsandbytes.optim.Lion8bit`` on the GB10 CUDA stack
-- see ``cppmega/docs/lion8bit_ab_2026_04_25.md`` for the reference run.

This is a **local-only** receipt -- it does not claim GB10 parity or compare
M4 throughput against GB10. The compaction ratio is the load-bearing number;
the absolute timings are diagnostic. With the fused kernel we expect a 3-4x
speedup over the unfused Lion8bit Python chain and step time at or below the
LionFP32Moments baseline; with the unfused chain we expect step time to be a
touch slower than LionFP32Moments because every apply_single goes through a
uint8 -> fp32 dequant + fp32 -> uint8 re-quant round trip.
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
from mlx.utils import tree_flatten  # noqa: E402

from cppmega_mlx.recipes.model_factory import local_gb10_quarter  # noqa: E402
from cppmega_mlx.training.optimizers import (  # noqa: E402
    LION8BIT_QUANT_KIND,
    Lion8bit,
    LionFP32Moments,
    make_lion,
    make_lion8bit,
)


SCHEMA_VERSION = 1
SOURCE_BEAD = "cppmega-mlx-lion8bit"
DEFAULT_OUTPUT = Path("bench/baselines/lion8bit_state_size_m4.json")


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
        except Exception:  # pragma: no cover - diagnostic only
            pass
    metal = getattr(mx, "metal", None)
    device_info = getattr(metal, "device_info", None)
    if device_info is None:
        return None
    try:
        return str(device_info())
    except Exception:  # pragma: no cover - diagnostic only
        return None


def _state_bytes(state: object) -> int:
    """Total bytes in mx.array leaves of an optimizer state pytree."""

    total = 0
    for _, value in tree_flatten(state):
        if isinstance(value, mx.array):
            total += int(value.nbytes)
    return total


def _state_dtype_breakdown(state: object) -> dict[str, dict[str, int]]:
    """Per-suffix dtype histogram, e.g. ``m_quant -> {uint8: 1.83 GiB}``."""

    by_key: dict[str, dict[str, int]] = {}
    for path, value in tree_flatten(state):
        if not isinstance(value, mx.array):
            continue
        key = path.rsplit(".", 1)[-1]
        dtype = str(value.dtype).removeprefix("mlx.core.")
        bucket = by_key.setdefault(key, {})
        bucket[dtype] = bucket.get(dtype, 0) + int(value.nbytes)
    return by_key


def _count_param_bytes(model: nn.Module) -> tuple[int, int]:
    """Return (parameter element count, bytes)."""

    elements = 0
    nbytes = 0
    for _, value in tree_flatten(model.trainable_parameters()):
        if isinstance(value, mx.array):
            elements += int(value.size)
            nbytes += int(value.nbytes)
    return elements, nbytes


def _sync_peak_memory() -> int | None:
    get_peak_memory = getattr(mx, "get_peak_memory", None)
    if get_peak_memory is None:
        return None
    try:
        return int(get_peak_memory())
    except Exception:  # pragma: no cover - diagnostic only
        return None


def _reset_peak_memory() -> None:
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if reset_peak_memory is not None:
        reset_peak_memory()


def _make_step_batch(
    model: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
) -> mx.array:
    vocab_size = int(model.config.vocab_size)  # type: ignore[attr-defined]
    return mx.random.randint(
        low=0,
        high=vocab_size,
        shape=(batch_size, seq_len),
    )


def _measure_optimizer(
    model: nn.Module,
    optimizer: Any,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)

    state_bytes = _state_bytes(optimizer.state)
    dtype_breakdown = _state_dtype_breakdown(optimizer.state)

    def loss_fn(m: nn.Module, ids: mx.array) -> mx.array:
        logits = m(ids)
        return mx.mean(logits.astype(mx.float32))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    samples_ms: list[float] = []
    for _ in range(warmup):
        ids = _make_step_batch(model, batch_size=batch_size, seq_len=seq_len)
        loss, grads = loss_and_grad(model, ids)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    _reset_peak_memory()
    for _ in range(iters):
        ids = _make_step_batch(model, batch_size=batch_size, seq_len=seq_len)
        t0 = time.perf_counter()
        loss, grads = loss_and_grad(model, ids)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "optimizer_class": type(optimizer).__name__,
        "state_bytes": state_bytes,
        "state_gib": state_bytes / (1024**3),
        "state_dtype_breakdown_bytes": dtype_breakdown,
        "step_ms": {
            "median": statistics.median(samples_ms),
            "mean": statistics.fmean(samples_ms),
            "min": min(samples_ms),
            "max": max(samples_ms),
            "samples_ms": samples_ms,
            "warmup": warmup,
            "iters": iters,
        },
        "peak_memory_bytes": _sync_peak_memory(),
    }


def _measure_state_only(
    model: nn.Module,
    optimizer: Any,
) -> dict[str, Any]:
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)
    state_bytes = _state_bytes(optimizer.state)
    return {
        "optimizer_class": type(optimizer).__name__,
        "state_bytes": state_bytes,
        "state_gib": state_bytes / (1024**3),
        "state_dtype_breakdown_bytes": _state_dtype_breakdown(optimizer.state),
        "step_ms": None,
        "peak_memory_bytes": _sync_peak_memory(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a local-only Lion8bit state-size + step-time receipt against "
            "the full local_gb10_quarter 1.797B-param bf16 model. No GB10 "
            "parity or cross-host throughput claim is made."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Micro-batch size for the 1-step timing run (default: 1).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for the 1-step timing run (default: 128).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup steps before timing begins (default: 2).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Timed iterations (default: 3).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for both optimizers (default: 1e-4).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for both optimizers (default: 0.01).",
    )
    parser.add_argument(
        "--skip-fp32",
        action="store_true",
        help=(
            "Skip the LionFP32Moments measurement -- useful when the M4 host "
            "cannot fit fp32 momentum alongside the 1.797B-param bf16 weights."
        ),
    )
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Skip the Lion8bit measurement (diagnostic; usually leave off).",
    )
    parser.add_argument(
        "--skip-unfused",
        action="store_true",
        help=(
            "Skip the Lion8bit unfused-Python-chain measurement. The unfused "
            "row is the baseline that the fused kernel is judged against; skip "
            "it only when the host is too tight on memory to run all three rows."
        ),
    )
    parser.add_argument(
        "--skip-fused",
        action="store_true",
        help="Skip the Lion8bit fused-kernel measurement (diagnostic).",
    )
    parser.add_argument(
        "--skip-symmetric",
        action="store_true",
        help=(
            "Skip the symmetric Lion8bit measurements (both fused and unfused). "
            "Useful when the host cannot afford to allocate the symmetric "
            "optimizer state alongside the dynamic one."
        ),
    )
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help=(
            "Skip the dynamic-LUT (bitsandbytes-style) Lion8bit measurements. "
            "By default the bench captures both symmetric and dynamic codec "
            "rows so the LUT-vs-symmetric step-time gap is visible."
        ),
    )
    parser.add_argument(
        "--skip-step-timing",
        action="store_true",
        help=(
            "Skip the 1-step forward+backward+update timing and only report "
            "the static state-size measurement. Useful when the host cannot "
            "fit a forward pass for the chosen batch_size + seq_len."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Write JSON receipt to this path. Default: {DEFAULT_OUTPUT}",
    )
    return parser


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    print("Building local_gb10_quarter (bf16) ...", file=sys.stderr)
    model = local_gb10_quarter(dtype=mx.bfloat16)
    param_count, param_bytes = _count_param_bytes(model)
    print(
        f"  model param count = {param_count:,} ({param_bytes / 1024**3:.2f} GiB)",
        file=sys.stderr,
    )

    fp32_result: dict[str, Any] | None = None
    unfused_result: dict[str, Any] | None = None
    fused_result: dict[str, Any] | None = None
    dynamic_unfused_result: dict[str, Any] | None = None
    dynamic_fused_result: dict[str, Any] | None = None

    if not args.skip_fp32:
        print("Measuring LionFP32Moments ...", file=sys.stderr)
        fp32_optimizer = make_lion(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if args.skip_step_timing:
            fp32_result = _measure_state_only(model, fp32_optimizer)
        else:
            fp32_result = _measure_optimizer(
                model,
                fp32_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        # Free fp32 state before allocating quant state (M4 RAM matters here).
        del fp32_optimizer

    if not args.skip_quant and not args.skip_unfused and not args.skip_symmetric:
        print("Measuring Lion8bit (unfused, Python chain) ...", file=sys.stderr)
        unfused_optimizer = make_lion8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_fused_kernel=False,
        )
        if args.skip_step_timing:
            unfused_result = _measure_state_only(model, unfused_optimizer)
        else:
            unfused_result = _measure_optimizer(
                model,
                unfused_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del unfused_optimizer

    if not args.skip_quant and not args.skip_fused and not args.skip_symmetric:
        print("Measuring Lion8bit (FUSED Metal kernel) ...", file=sys.stderr)
        fused_optimizer = make_lion8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_fused_kernel=True,
        )
        if args.skip_step_timing:
            fused_result = _measure_state_only(model, fused_optimizer)
        else:
            fused_result = _measure_optimizer(
                model,
                fused_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del fused_optimizer

    # Dynamic-LUT (bnb-style) Lion8bit rows. Same uint8 + fp32 absmax layout
    # as the symmetric path; the fused dynamic kernel loads the 256-entry
    # signed LUT into threadgroup memory and binary-searches it on quant.
    if (
        not args.skip_quant
        and not args.skip_unfused
        and not args.skip_dynamic
    ):
        print(
            "Measuring Lion8bit dynamic (unfused, Python chain) ...",
            file=sys.stderr,
        )
        dyn_unfused_optimizer = make_lion8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_fused_kernel=False,
            quant_scheme="dynamic_int8_v1",
        )
        if args.skip_step_timing:
            dynamic_unfused_result = _measure_state_only(model, dyn_unfused_optimizer)
        else:
            dynamic_unfused_result = _measure_optimizer(
                model,
                dyn_unfused_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del dyn_unfused_optimizer

    if not args.skip_quant and not args.skip_fused and not args.skip_dynamic:
        print(
            "Measuring Lion8bit dynamic (FUSED Metal kernel) ...",
            file=sys.stderr,
        )
        dyn_fused_optimizer = make_lion8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_fused_kernel=True,
            quant_scheme="dynamic_int8_v1",
        )
        if args.skip_step_timing:
            dynamic_fused_result = _measure_state_only(model, dyn_fused_optimizer)
        else:
            dynamic_fused_result = _measure_optimizer(
                model,
                dyn_fused_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del dyn_fused_optimizer

    # The "lion8bit" payload retains the previous schema's role as the
    # default-configuration row (fused if available, otherwise unfused).
    quant_result: dict[str, Any] | None = fused_result or unfused_result

    state_compaction = None
    step_speedup_fp32_over_fused = None
    step_speedup_unfused_over_fused = None
    if fp32_result is not None and quant_result is not None:
        state_compaction = (
            fp32_result["state_bytes"] / max(quant_result["state_bytes"], 1)
        )
    if (
        fp32_result is not None
        and fp32_result.get("step_ms") is not None
        and fused_result is not None
        and fused_result.get("step_ms") is not None
    ):
        step_speedup_fp32_over_fused = (
            fp32_result["step_ms"]["median"] / fused_result["step_ms"]["median"]
        )
    if (
        unfused_result is not None
        and unfused_result.get("step_ms") is not None
        and fused_result is not None
        and fused_result.get("step_ms") is not None
    ):
        step_speedup_unfused_over_fused = (
            unfused_result["step_ms"]["median"] / fused_result["step_ms"]["median"]
        )

    # Fused-vs-unfused speedup for the dynamic-LUT path.
    step_speedup_unfused_over_fused_dynamic = None
    if (
        dynamic_unfused_result is not None
        and dynamic_unfused_result.get("step_ms") is not None
        and dynamic_fused_result is not None
        and dynamic_fused_result.get("step_ms") is not None
    ):
        step_speedup_unfused_over_fused_dynamic = (
            dynamic_unfused_result["step_ms"]["median"]
            / dynamic_fused_result["step_ms"]["median"]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "cppmega.mlx.local_m4_lion8bit_state_receipt",
        "source_bead": SOURCE_BEAD,
        "guards": {
            "local_only": True,
            "gb10_parity_claim": False,
            "m4_vs_gb10_throughput_parity_claim": False,
            "bitsandbytes_bit_exact_claim": False,
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
            "model_profile": "local_gb10_quarter",
            "model_dtype": "bfloat16",
            "param_count": param_count,
            "param_bytes": param_bytes,
            "param_gib": param_bytes / (1024**3),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "warmup": args.warmup,
            "iters": args.iters,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "block_size": 256,
            "quant_kind": LION8BIT_QUANT_KIND,
        },
        "fp32_moments_lion": fp32_result,
        "lion8bit": quant_result,
        "lion8bit_unfused": unfused_result,
        "lion8bit_fused": fused_result,
        "lion8bit_dynamic_fused": dynamic_fused_result,
        "lion8bit_dynamic_unfused": dynamic_unfused_result,
        "acceptance": {
            "state_compaction_ratio_fp32_over_quant": state_compaction,
            "step_speedup_fp32_over_quant_median": step_speedup_fp32_over_fused,
            "step_speedup_unfused_over_fused_median": step_speedup_unfused_over_fused,
            "step_speedup_unfused_over_fused_dynamic_median": (
                step_speedup_unfused_over_fused_dynamic
            ),
            "state_compaction_lower_bound": 3.0,
            "meets_state_compaction_gate": (
                state_compaction is not None and state_compaction >= 3.0
            ),
            "fused_vs_unfused_dynamic_lower_bound": 2.0,
            "meets_fused_vs_unfused_dynamic_gate": (
                step_speedup_unfused_over_fused_dynamic is not None
                and step_speedup_unfused_over_fused_dynamic >= 2.0
            ),
        },
        "notes": [
            "Local Apple Silicon receipt only; does not claim GB10 parity.",
            "Both schemes have fused single-kernel-launch MSL implementations: "
            "the symmetric kernel embeds the +128 bias inline, the dynamic "
            "kernel loads the bnb 256-entry signed LUT into threadgroup memory "
            "and binary-searches it on the quant step (8 cmps/elem).",
            "LionFP32Moments needs ~6.69 GiB of optimizer state for "
            "1.797B-param bf16 weights; pass --skip-fp32 if the host runs "
            "out of memory.",
            "lion8bit_unfused: Python chain (3 Metal launches per param). "
            "lion8bit_fused: single fused-kernel launch per param "
            "(dequant -> Lion math -> apply -> quant fused into one MSL kernel). "
            "Default optimizer instance uses the fused kernel.",
            "lion8bit_dynamic_fused / lion8bit_dynamic_unfused: same paths but "
            "using the bnb-style 256-entry dynamic LUT codec. Skip with "
            "--skip-dynamic when host memory is tight.",
            "CUDA reference: cppmega/docs/lion8bit_ab_2026_04_25.md.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = build_receipt(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT
        if not output.is_absolute():
            output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
