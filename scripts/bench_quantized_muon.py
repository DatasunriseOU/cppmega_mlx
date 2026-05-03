#!/usr/bin/env python3
"""Local receipt for the int8-quantized Muon momentum buffer.

Builds two ``MuonAdamWMulti`` optimizers on the production-shape
``local_gb10_quarter`` model -- one with the default fp32 momentum and
one with ``quantize_momentum=True`` -- and measures the Muon-group
optimizer state size + total optimizer state size before and after.
The receipt is local-only on Apple Silicon; it does not claim GB10
parity. The numbers track cppmega CUDA's
``quantized_muon_momentum_update_multi_and_normalize_groups_`` budget
(~5.24 GiB -> ~1.33 GiB on the Muon group at 1.797B params).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.recipes.model_factory import local_gb10_quarter  # noqa: E402
from cppmega_mlx.training.optimizers import (  # noqa: E402
    MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE,
    MUON_QUANTIZED_MOMENTUM_SCHEME,
    make_muon,
)


SCHEMA_VERSION = 1
SOURCE_BEAD = "cppmega-mlx-quantized-muon-momentum"
DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "quantized_muon_state_size_m4.json"


def _flatten_arrays(tree: Any) -> Iterable[mx.array]:
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _flatten_arrays(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _flatten_arrays(value)
    elif isinstance(tree, mx.array):
        yield tree


def _bytes(tree: Any) -> int:
    return sum(int(arr.size * arr.dtype.size) for arr in _flatten_arrays(tree))


def _peak_memory_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None)
    if fn is None:
        metal = getattr(mx, "metal", None)
        if metal is None:
            return 0
        fn = getattr(metal, "get_peak_memory", None)
        if fn is None:
            return 0
    try:
        return int(fn())
    except Exception:  # pragma: no cover - diagnostic only
        return 0


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _measure(
    *,
    quantize_momentum: bool,
    cppmega_cuda_parity: bool,
    do_step: bool,
) -> dict[str, Any]:
    _reset_peak_memory()
    t0 = time.perf_counter()
    model = local_gb10_quarter()
    model.set_dtype(mx.bfloat16)

    optimizer = make_muon(
        cppmega_cuda_parity=cppmega_cuda_parity,
        quantize_momentum=quantize_momentum,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)

    muon_state_bytes = _bytes(optimizer.state["muon"])
    adamw_state_bytes = _bytes(optimizer.state["adamw"])
    total_state_bytes = _bytes(optimizer.state)
    peak_after_init = _peak_memory_bytes()

    step_peak: int | None = None
    if do_step:
        # Synthetic gradients matching the parameter pytree -- enough to
        # exercise the apply path without paying for a full forward/backward.
        params = model.trainable_parameters()
        grads = _zero_like_tree(params)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        step_peak = _peak_memory_bytes()

    elapsed = time.perf_counter() - t0
    return {
        "quantize_momentum": quantize_momentum,
        "cppmega_cuda_parity": cppmega_cuda_parity,
        "muon_state_bytes": muon_state_bytes,
        "muon_state_gib": muon_state_bytes / (1024**3),
        "adamw_state_bytes": adamw_state_bytes,
        "adamw_state_gib": adamw_state_bytes / (1024**3),
        "total_state_bytes": total_state_bytes,
        "total_state_gib": total_state_bytes / (1024**3),
        "peak_memory_after_init_bytes": peak_after_init,
        "peak_memory_after_init_gib": peak_after_init / (1024**3),
        "peak_memory_after_step_bytes": step_peak,
        "peak_memory_after_step_gib": (
            step_peak / (1024**3) if step_peak is not None else None
        ),
        "elapsed_s": elapsed,
    }


def _zero_like_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _zero_like_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_zero_like_tree(v) for v in tree)
    if isinstance(tree, mx.array):
        return mx.zeros(tree.shape, dtype=tree.dtype)
    return tree


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Local Apple Silicon receipt comparing the fp32 vs int8 Muon "
            "momentum buffer on the local_gb10_quarter profile. No GB10 "
            "parity claim is made."
        )
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON receipt path. Default: {DEFAULT_OUTPUT.relative_to(ROOT)}",
    )
    p.add_argument(
        "--cppmega-cuda-parity",
        action="store_true",
        default=True,
        help="Use the cppmega CUDA parity LR/Nesterov knobs (default).",
    )
    p.add_argument(
        "--no-cppmega-cuda-parity",
        dest="cppmega_cuda_parity",
        action="store_false",
    )
    p.add_argument(
        "--skip-step",
        action="store_true",
        help="Skip the synthetic apply step; init-only measurement.",
    )
    return p


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    fp32 = _measure(
        quantize_momentum=False,
        cppmega_cuda_parity=args.cppmega_cuda_parity,
        do_step=not args.skip_step,
    )
    quant = _measure(
        quantize_momentum=True,
        cppmega_cuda_parity=args.cppmega_cuda_parity,
        do_step=not args.skip_step,
    )
    muon_delta = fp32["muon_state_bytes"] - quant["muon_state_bytes"]
    total_delta = fp32["total_state_bytes"] - quant["total_state_bytes"]
    muon_ratio = fp32["muon_state_bytes"] / max(quant["muon_state_bytes"], 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "cppmega.mlx.local_m4_quantized_muon_momentum_state_receipt",
        "source_bead": SOURCE_BEAD,
        "guards": {
            "local_only": True,
            "gb10_parity_claim": False,
            "m4_vs_gb10_throughput_parity_claim": False,
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "config": {
            "model_profile": "local_gb10_quarter",
            "block_size": MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE,
            "scheme": MUON_QUANTIZED_MOMENTUM_SCHEME,
            "cppmega_cuda_parity": args.cppmega_cuda_parity,
            "do_step": not args.skip_step,
        },
        "fp32_momentum": fp32,
        "int8_momentum": quant,
        "delta": {
            "muon_state_saved_bytes": muon_delta,
            "muon_state_saved_gib": muon_delta / (1024**3),
            "total_state_saved_bytes": total_delta,
            "total_state_saved_gib": total_delta / (1024**3),
            "muon_state_compression_ratio": muon_ratio,
        },
        "notes": [
            "Local Apple Silicon receipt only; does not claim GB10 parity.",
            "Mirrors cppmega CUDA quantized_muon_momentum_update_multi_and_"
            "normalize_groups_ in megatron/core/optimizer/emerging_optimizers.py.",
            "AdamW group is untouched (parallel work-stream owns Adam8bit).",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = build_receipt(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
