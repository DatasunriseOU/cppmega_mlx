#!/usr/bin/env python3
"""Local receipt for the 8-bit AdamW (Adam8bit) optimizer state.

Measures the optimizer-state byte total and 1-step time for three
configurations against the full ``local_gb10_quarter`` 1.797B-param bf16
model:

1. ``AdamWFP32Moments`` -- the fp32-moments baseline (~14.4 GiB state).
2. ``Adam8bit`` symmetric -- the existing M0-grade symmetric-int8 codec.
3. ``Adam8bit`` dynamic -- the bnb-style dynamic LUT codec
   (``QUANT_SCHEME_DYNAMIC``), denser bins near zero. Same memory layout
   as symmetric (uint8 + per-256-block fp32 absmax), so the state size
   should be identical; the step time is expected to be slightly slower
   because the LUT lookup adds an extra dependent load and the fused
   kernel only supports the symmetric path.

This is a **local-only** receipt -- it does not claim GB10 parity or compare
M4 throughput against GB10. The compaction ratio is the load-bearing number;
the absolute timings are diagnostic.
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
from cppmega_mlx.training._quantize_8bit import (  # noqa: E402
    QUANT_SCHEME_DYNAMIC,
    QUANT_SCHEME_SYMMETRIC,
)
from cppmega_mlx.training.optimizers import (  # noqa: E402
    ADAM8BIT_QUANT_KIND,
    Adam8bit,
    AdamWFP32Moments,
    make_adam8bit,
    make_adamw,
)


SCHEMA_VERSION = 1
SOURCE_BEAD = "cppmega-mlx-adam8bit"
DEFAULT_OUTPUT = Path("bench/baselines/adam8bit_state_size_m4.json")


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
    """Per-suffix dtype histogram, e.g. ``m_quant -> {uint8: 18 GiB}``."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit a local-only Adam8bit state-size + step-time receipt against "
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
            "Skip the AdamWFP32Moments measurement -- useful when the M4 host "
            "cannot fit fp32 m+v alongside the 1.797B-param bf16 weights."
        ),
    )
    parser.add_argument(
        "--skip-quant",
        action="store_true",
        help="Skip the Adam8bit measurement (diagnostic; usually leave off).",
    )
    parser.add_argument(
        "--skip-unfused",
        action="store_true",
        help=(
            "Skip the unfused (legacy Python-chain) Adam8bit measurement. "
            "By default the bench captures both unfused and fused rows so the "
            "kernel-fusion speedup is visible."
        ),
    )
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help=(
            "Skip the Adam8bit dynamic-LUT (bitsandbytes-style) measurement. "
            "By default the bench captures the dynamic row alongside the "
            "symmetric ones so the LUT codec's state-size + step-time can be "
            "compared directly. Pass when the host cannot afford the extra "
            "iteration."
        ),
    )
    parser.add_argument(
        "--skip-symmetric",
        action="store_true",
        help=(
            "Skip the Adam8bit symmetric measurements (both fused and "
            "unfused). Useful when the host cannot afford to allocate the "
            "symmetric optimizer state alongside the dynamic one. The fp32 "
            "and dynamic rows are unaffected."
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


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    print("Building local_gb10_quarter (bf16) ...", file=sys.stderr)
    model = local_gb10_quarter(dtype=mx.bfloat16)
    param_count, param_bytes = _count_param_bytes(model)
    print(
        f"  model param count = {param_count:,} ({param_bytes / 1024**3:.2f} GiB)",
        file=sys.stderr,
    )

    fp32_result: dict[str, Any] | None = None
    quant_unfused_result: dict[str, Any] | None = None
    quant_result: dict[str, Any] | None = None  # fused (default) Adam8bit
    quant_dynamic_result: dict[str, Any] | None = None  # FUSED bnb-style dynamic LUT
    quant_dynamic_unfused_result: dict[str, Any] | None = None  # unfused dynamic chain

    if not args.skip_fp32:
        print("Measuring AdamWFP32Moments ...", file=sys.stderr)
        fp32_optimizer = make_adamw(
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

    # Measure the unfused (legacy Python-chain) Adam8bit path so the bench
    # captures the pre-fusion baseline alongside the fused number. This is
    # the row that quantifies the kernel-fusion speedup.
    if not args.skip_quant and not args.skip_unfused and not args.skip_symmetric:
        print("Measuring Adam8bit (unfused, Python chain) ...", file=sys.stderr)
        quant_unfused_optimizer = make_adam8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_fused_kernel=False,
        )
        if args.skip_step_timing:
            quant_unfused_result = _measure_state_only(model, quant_unfused_optimizer)
        else:
            quant_unfused_result = _measure_optimizer(
                model,
                quant_unfused_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del quant_unfused_optimizer

    if not args.skip_quant and not args.skip_symmetric:
        print("Measuring Adam8bit (fused Metal kernel) ...", file=sys.stderr)
        quant_optimizer = make_adam8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            use_fused_kernel=True,
        )
        if args.skip_step_timing:
            quant_result = _measure_state_only(model, quant_optimizer)
        else:
            quant_result = _measure_optimizer(
                model,
                quant_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del quant_optimizer

    # The dynamic-LUT (bnb-style) Adam8bit rows. Same memory layout as the
    # symmetric path but a different codec; the fused dynamic kernel loads
    # the 256-entry signed LUT into threadgroup memory once per block and
    # binary-searches it on the quant step. We capture both the unfused
    # chain (legacy, 4-5 separate launches per parameter) and the fused
    # single-launch path so the speedup gate is visible.
    if not args.skip_quant and not args.skip_dynamic and not args.skip_unfused:
        print(
            "Measuring Adam8bit dynamic (unfused, Python chain) ...",
            file=sys.stderr,
        )
        quant_dynamic_unfused_optimizer = make_adam8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            quant_scheme=QUANT_SCHEME_DYNAMIC,
            use_fused_kernel=False,
        )
        if args.skip_step_timing:
            quant_dynamic_unfused_result = _measure_state_only(
                model, quant_dynamic_unfused_optimizer
            )
        else:
            quant_dynamic_unfused_result = _measure_optimizer(
                model,
                quant_dynamic_unfused_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del quant_dynamic_unfused_optimizer

    if not args.skip_quant and not args.skip_dynamic:
        print(
            "Measuring Adam8bit dynamic (fused Metal kernel) ...",
            file=sys.stderr,
        )
        quant_dynamic_optimizer = make_adam8bit(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            quant_scheme=QUANT_SCHEME_DYNAMIC,
            use_fused_kernel=True,
        )
        if args.skip_step_timing:
            quant_dynamic_result = _measure_state_only(model, quant_dynamic_optimizer)
        else:
            quant_dynamic_result = _measure_optimizer(
                model,
                quant_dynamic_optimizer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                iters=args.iters,
            )
        del quant_dynamic_optimizer

    state_compaction = None
    step_speedup = None
    fused_vs_unfused_speedup = None
    fused_vs_fp32_step_ratio = None
    if fp32_result is not None and quant_result is not None:
        state_compaction = (
            fp32_result["state_bytes"] / max(quant_result["state_bytes"], 1)
        )
        if (
            fp32_result.get("step_ms") is not None
            and quant_result.get("step_ms") is not None
        ):
            step_speedup = (
                fp32_result["step_ms"]["median"] / quant_result["step_ms"]["median"]
            )
            fused_vs_fp32_step_ratio = (
                quant_result["step_ms"]["median"] / fp32_result["step_ms"]["median"]
            )
    if (
        quant_unfused_result is not None
        and quant_result is not None
        and quant_unfused_result.get("step_ms") is not None
        and quant_result.get("step_ms") is not None
    ):
        fused_vs_unfused_speedup = (
            quant_unfused_result["step_ms"]["median"]
            / quant_result["step_ms"]["median"]
        )

    dynamic_vs_symmetric_state_ratio = None
    dynamic_vs_symmetric_step_ratio = None
    if quant_dynamic_result is not None and quant_result is not None:
        dynamic_vs_symmetric_state_ratio = (
            quant_dynamic_result["state_bytes"]
            / max(quant_result["state_bytes"], 1)
        )
        if (
            quant_dynamic_result.get("step_ms") is not None
            and quant_result.get("step_ms") is not None
        ):
            dynamic_vs_symmetric_step_ratio = (
                quant_dynamic_result["step_ms"]["median"]
                / quant_result["step_ms"]["median"]
            )

    # Fused-vs-unfused speedup for the dynamic path. Mirrors the symmetric
    # path's gate but compares the fused dynamic kernel to the unfused
    # dynamic chain (both at scheme=dynamic_int8_v1).
    fused_vs_unfused_dynamic_speedup = None
    if (
        quant_dynamic_unfused_result is not None
        and quant_dynamic_result is not None
        and quant_dynamic_unfused_result.get("step_ms") is not None
        and quant_dynamic_result.get("step_ms") is not None
    ):
        fused_vs_unfused_dynamic_speedup = (
            quant_dynamic_unfused_result["step_ms"]["median"]
            / quant_dynamic_result["step_ms"]["median"]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "cppmega.mlx.local_m4_adam8bit_state_receipt",
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
            "quant_kind": ADAM8BIT_QUANT_KIND,
        },
        "fp32_moments_adamw": fp32_result,
        "adam8bit_unfused": quant_unfused_result,
        "adam8bit": quant_result,
        "adam8bit_dynamic": quant_dynamic_result,
        "adam8bit_dynamic_unfused": quant_dynamic_unfused_result,
        "acceptance": {
            "state_compaction_ratio_fp32_over_quant": state_compaction,
            "step_speedup_fp32_over_quant_median": step_speedup,
            "fused_vs_unfused_speedup_median": fused_vs_unfused_speedup,
            "fused_vs_unfused_dynamic_speedup_median": fused_vs_unfused_dynamic_speedup,
            "fused_vs_fp32_step_ratio_median": fused_vs_fp32_step_ratio,
            "fused_vs_unfused_lower_bound": 1.10,
            "meets_fused_vs_unfused_gate": (
                fused_vs_unfused_speedup is not None
                and fused_vs_unfused_speedup >= 1.10
            ),
            "fused_vs_unfused_dynamic_lower_bound": 2.0,
            "meets_fused_vs_unfused_dynamic_gate": (
                fused_vs_unfused_dynamic_speedup is not None
                and fused_vs_unfused_dynamic_speedup >= 2.0
            ),
            "state_compaction_lower_bound": 3.0,
            "meets_state_compaction_gate": (
                state_compaction is not None and state_compaction >= 3.0
            ),
            "dynamic_vs_symmetric_state_ratio_median": dynamic_vs_symmetric_state_ratio,
            "dynamic_vs_symmetric_step_ratio_median": dynamic_vs_symmetric_step_ratio,
            "dynamic_vs_symmetric_state_lower_bound": 0.99,
            "dynamic_vs_symmetric_state_upper_bound": 1.01,
            "meets_dynamic_vs_symmetric_state_gate": (
                dynamic_vs_symmetric_state_ratio is not None
                and 0.99 <= dynamic_vs_symmetric_state_ratio <= 1.01
            ),
        },
        "schemes": {
            "adam8bit": QUANT_SCHEME_SYMMETRIC,
            "adam8bit_unfused": QUANT_SCHEME_SYMMETRIC,
            "adam8bit_dynamic": QUANT_SCHEME_DYNAMIC,
            "adam8bit_dynamic_unfused": QUANT_SCHEME_DYNAMIC,
        },
        "notes": [
            "Local Apple Silicon receipt only; does not claim GB10 parity.",
            "adam8bit (default) uses symmetric int8 blockwise quantization, "
            "matching the existing M0 codec. adam8bit_dynamic uses the "
            "bitsandbytes-style 256-entry signed dynamic LUT (denser bins "
            "near zero); both share the per-256-block uint8 + fp32-absmax "
            "memory layout so the state size is identical.",
            "AdamWFP32Moments needs ~14.4 GiB of optimizer state for "
            "1.797B-param bf16 weights; pass --skip-fp32 if the host runs "
            "out of memory.",
            "adam8bit_unfused is the legacy Python-chain dequant -> math -> "
            "quant -> apply path; adam8bit is the fused single-kernel-launch "
            "MSL implementation. Both produce identical updates within fp32 "
            "noise. Both schemes (symmetric + dynamic) have fused kernels: "
            "adam8bit_dynamic uses the dynamic-LUT fused MSL kernel that "
            "loads the bnb 256-entry signed LUT into threadgroup memory and "
            "binary-searches it on the quant step; adam8bit_dynamic_unfused "
            "is the legacy Python chain for the dynamic codec.",
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
