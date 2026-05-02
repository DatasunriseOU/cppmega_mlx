#!/usr/bin/env python3
"""M0.4 local MLX bf16 training-step receipt.

This is a correctness smoke for the local MLX training plumbing. It intentionally
uses the existing tiny hybrid model path until the full M0 tokenizer and
local_gb10_quarter factory blockers are closed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_recipes_import_shim() -> None:
    """Bypass package-level recipe exports when importing training smoke code."""

    if "cppmega_mlx.recipes" in sys.modules:
        return
    module = ModuleType("cppmega_mlx.recipes")
    setattr(module, "__path__", [str(ROOT / "cppmega_mlx" / "recipes")])
    sys.modules["cppmega_mlx.recipes"] = module


_install_recipes_import_shim()

import mlx.core as mx  # noqa: E402

from scripts.train_hybrid_tiny import (  # noqa: E402
    TrainHybridTinyConfig,
    device_info,
    dry_run_payload,
    train_hybrid_tiny,
    validation_dataset_path,
    validate_config,
)


TARGET_PARQUET = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
DEFAULT_OUTPUT = ROOT / "bench" / "baselines" / "m04_train_step.json"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_SCOPE = "local_mlx_m04_train_step"
OPEN_M0_BLOCKERS = (
    {
        "id": "cppmega-mlx-t8f.1",
        "title": "M0.1 tokenizer is still open",
        "impact": "full local_gb10_quarter M0.4 acceptance cannot be claimed",
    },
    {
        "id": "cppmega-mlx-t8f.2",
        "title": "M0.2 local_gb10_quarter model factory is still open",
        "impact": "this lane uses HybridTinyLM instead of the target 3.79B profile",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or dry-run the M0.4 local bf16 MLX training-step smoke and "
            "write a bench/baselines-compatible JSON receipt."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=TARGET_PARQUET,
        help=f"Token dataset path. Defaults to {TARGET_PARQUET.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--data-format",
        choices=("npz", "parquet", "megatron"),
        default="parquet",
    )
    parser.add_argument("--token-key", default="token_ids")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--vocab-size", type=int, default=131_072)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--pattern", default="M")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=1)
    parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--mamba-head-dim", type=int, default=4)
    parser.add_argument("--mamba-state-dim", type=int, default=4)
    parser.add_argument("--mamba-groups", type=int, default=1)
    parser.add_argument("--mamba-chunk-size", type=int, default=4)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Request mlx.core.compile for the train step. Eager is default for local reliability.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use an explicit temporary NPZ with repeated tiny samples instead of --data-path.",
    )
    parser.add_argument(
        "--dry-run-json",
        action="store_true",
        help="Validate configuration and emit/write receipt JSON without training.",
    )
    parser.add_argument(
        "--require-loss-decrease",
        action="store_true",
        help="Exit non-zero unless final_loss < initial_loss. Useful for 100-step gates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print compact JSON receipt to stdout. The output file is always written.",
    )
    return parser


def config_from_args(args: argparse.Namespace, *, data_path: Path) -> TrainHybridTinyConfig:
    return TrainHybridTinyConfig(
        npz_path=str(data_path),
        data_format=args.data_format,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        dtype=args.dtype,
        compile=args.compile,
        seed=args.seed,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        pattern=args.pattern,
        depth=args.depth,
        num_attention_heads=args.num_attention_heads,
        mamba_expand=args.mamba_expand,
        mamba_head_dim=args.mamba_head_dim,
        mamba_state_dim=args.mamba_state_dim,
        mamba_groups=args.mamba_groups,
        mamba_chunk_size=args.mamba_chunk_size,
        token_key=args.token_key,
    )


def write_synthetic_npz(path: Path, *, steps: int, batch_size: int, seq_len: int, vocab_size: int) -> None:
    samples = max(batch_size * max(steps, 1), batch_size, 4)
    base = np.arange(seq_len, dtype=np.int32) % max(vocab_size, 2)
    tokens = np.tile(base.reshape(1, seq_len), (samples, 1))
    arrays: dict[str, Any] = {
        "tokens": tokens,
        "attention_mask": np.ones_like(tokens, dtype=np.float32),
        "structure_ids": (tokens % 7).astype(np.int32),
        "dep_levels": (tokens % 3).astype(np.int32),
        "ast_depth_ids": (tokens % 5).astype(np.int32),
        "sibling_index_ids": (tokens % 11).astype(np.int32),
        "node_type_ids": (tokens % 13).astype(np.int32),
        "vocab_size": np.array(vocab_size, dtype=np.int64),
        "tokenizer_contract": np.array("local_profile"),
    }
    np.savez(path, **arrays)


def run_receipt(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.steps < 1:
        return blocked_receipt(args, "steps must be positive", "invalid_cli"), 2
    if args.batch_size < 1:
        return blocked_receipt(args, "batch_size must be positive", "invalid_cli"), 2
    if args.seq_len < 2:
        return blocked_receipt(args, "seq_len must be at least 2", "invalid_cli"), 2

    if args.synthetic:
        with tempfile.TemporaryDirectory(prefix="cppmega_mlx_m04_") as tmp:
            data_path = Path(tmp) / "tokens.npz"
            write_synthetic_npz(
                data_path,
                steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
            )
            original_format = args.data_format
            args.data_format = "npz"
            args.token_key = "tokens"
            try:
                payload, exit_code = _run_existing_training(args, data_path=data_path)
            finally:
                args.data_format = original_format
            return payload, exit_code

    data_path = args.data_path
    if not data_path.exists():
        payload = blocked_receipt(
            args,
            f"dataset path does not exist: {data_path}",
            "missing_dataset",
        )
        return payload, 0 if args.dry_run_json else 2

    return _run_existing_training(args, data_path=data_path)


def _run_existing_training(args: argparse.Namespace, *, data_path: Path) -> tuple[dict[str, Any], int]:
    config = config_from_args(args, data_path=data_path)
    try:
        validate_config(config)
    except Exception as exc:
        return blocked_receipt(args, str(exc), type(exc).__name__), 0 if args.dry_run_json else 2

    reset_peak_memory()
    memory_before = metal_memory_payload()
    try:
        if args.dry_run_json:
            train_payload = dry_run_payload(
                config,
                npz_path=str(data_path),
                valid_path=validation_dataset_path(config),
            )
        else:
            train_payload = train_hybrid_tiny(
                config,
                npz_path=str(data_path),
                valid_path=validation_dataset_path(config),
            )
    except Exception as exc:
        return blocked_receipt(args, str(exc), type(exc).__name__), 0 if args.dry_run_json else 2
    finally:
        mx.synchronize()

    memory_after = metal_memory_payload()
    receipt = receipt_from_train_payload(
        args,
        config=config,
        train_payload=train_payload,
        memory_before=memory_before,
        memory_after=memory_after,
    )
    if args.require_loss_decrease and not receipt["training"]["loss_decreased"]:
        receipt["status"] = "failed"
        receipt["training"]["loss_decrease_required"] = True
        receipt["training"]["loss_decrease_satisfied"] = False
        return receipt, 2
    return receipt, 0


def receipt_from_train_payload(
    args: argparse.Namespace,
    *,
    config: TrainHybridTinyConfig,
    train_payload: dict[str, Any],
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
) -> dict[str, Any]:
    step_metrics = list(train_payload.get("step_metrics", []))
    losses = [float(item["loss"]) for item in step_metrics if "loss" in item]
    step_times = [float(item["seconds"]) for item in step_metrics if "seconds" in item]
    tokens_per_second = [
        float(item["tokens_per_second"])
        for item in step_metrics
        if "tokens_per_second" in item
    ]
    all_finite = bool(losses) and all(math.isfinite(value) for value in losses)
    optimizer_updated = bool(step_metrics) and all(bool(item.get("updated")) for item in step_metrics)
    loss_decreased = bool(len(losses) >= 2 and losses[-1] < losses[0])
    status = "dry_run" if train_payload.get("status") == "dry_run" else "ok"
    dataset = train_payload.get("dataset", {})
    model_config = train_payload.get("model_config", {})
    mode = "compiled" if train_payload.get("compile_enabled") else "eager"

    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_scope": RECEIPT_SCOPE,
        "status": status,
        "issue": {
            "id": "cppmega-mlx-t8f.4",
            "title": "M0.4: one bf16 training step + 100-step loss decrease on local parquet samples",
        },
        "local_only": True,
        "gb10_training_correctness_claim": False,
        "m4_vs_gb10_throughput_parity_claim": False,
        "full_m0_4_acceptance_claim": False,
        "acceptance_blockers": list(OPEN_M0_BLOCKERS),
        "workload": {
            "target_data_path": str(TARGET_PARQUET.relative_to(ROOT)),
            "data_path": str(config.npz_path),
            "data_format": config.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": config.dtype,
            "steps_requested": config.steps,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "tokens_per_step": train_payload.get("tokens_per_step"),
            "compile_requested": config.compile,
            "mode": mode,
            "require_loss_decrease": bool(args.require_loss_decrease),
        },
        "training": {
            "steps_completed": len(step_metrics),
            "optimizer_updated": optimizer_updated,
            "all_finite": all_finite,
            "losses": losses,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else train_payload.get("final_loss"),
            "mean_loss": train_payload.get("mean_loss"),
            "loss_decreased": loss_decreased,
            "loss_decrease_required": bool(args.require_loss_decrease),
            "loss_decrease_satisfied": (not args.require_loss_decrease) or loss_decreased,
            "trained_tokens": train_payload.get("trained_tokens"),
            "step_metrics": step_metrics,
        },
        "timing": {
            "step_times_s": step_times,
            "mean_step_time_s": statistics.fmean(step_times) if step_times else None,
            "median_step_time_s": statistics.median(step_times) if step_times else None,
            "tokens_per_second": statistics.fmean(tokens_per_second)
            if tokens_per_second
            else train_payload.get("tokens_per_second"),
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "peak_memory_bytes": memory_after.get("peak_memory_bytes"),
        },
        "dataset": dataset,
        "model": {
            "source": train_payload.get("model_source"),
            "parameter_count": train_payload.get("parameter_count"),
            "route_symbols": train_payload.get("route_symbols"),
            "route_roles": train_payload.get("route_roles"),
            "backend_plan": train_payload.get("backend_plan"),
            "config": model_config,
        },
        "software": {
            "git_commit": git_commit(),
            "device": train_payload.get("device", device_info()),
        },
        "baseline_row": baseline_row(train_payload, config=config, mode=mode),
    }
    return json_ready(receipt)


def blocked_receipt(args: argparse.Namespace, reason: str, reason_type: str) -> dict[str, Any]:
    return {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_scope": RECEIPT_SCOPE,
        "status": "blocked",
        "issue": {
            "id": "cppmega-mlx-t8f.4",
            "title": "M0.4: one bf16 training step + 100-step loss decrease on local parquet samples",
        },
        "local_only": True,
        "gb10_training_correctness_claim": False,
        "m4_vs_gb10_throughput_parity_claim": False,
        "full_m0_4_acceptance_claim": False,
        "blockers": [
            {
                "type": reason_type,
                "reason": reason,
                "recoverable": True,
            },
            *OPEN_M0_BLOCKERS,
        ],
        "workload": {
            "target_data_path": str(TARGET_PARQUET.relative_to(ROOT)),
            "data_path": str(args.data_path),
            "data_format": args.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": args.dtype,
            "steps_requested": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "compile_requested": bool(args.compile),
            "require_loss_decrease": bool(args.require_loss_decrease),
        },
        "training": {
            "steps_completed": 0,
            "optimizer_updated": False,
            "all_finite": False,
            "losses": [],
            "initial_loss": None,
            "final_loss": None,
            "loss_decreased": False,
            "loss_decrease_required": bool(args.require_loss_decrease),
            "loss_decrease_satisfied": False,
        },
        "timing": {
            "step_times_s": [],
            "mean_step_time_s": None,
            "median_step_time_s": None,
            "tokens_per_second": None,
        },
        "memory": {
            "before": metal_memory_payload(),
            "after": metal_memory_payload(),
            "peak_memory_bytes": None,
        },
        "software": {
            "git_commit": git_commit(),
            "device": device_info(),
        },
    }


def baseline_row(
    train_payload: dict[str, Any],
    *,
    config: TrainHybridTinyConfig,
    mode: str,
) -> dict[str, Any]:
    device = train_payload.get("device", {})
    hardware = str(device.get("machine") or "local-mac")
    if device.get("mlx_device_info") and isinstance(device["mlx_device_info"], dict):
        hardware = str(device["mlx_device_info"].get("device_name") or hardware)
    return {
        "hardware": hardware,
        "commit": git_commit() or "unknown",
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "route": str(train_payload.get("route_symbols") or "unknown"),
        "model": "HybridTinyLM",
        "mode": mode,
        "tokens_per_second": float(train_payload.get("tokens_per_second") or 0.0),
        "local_only": True,
        "gb10_parity_claim": False,
    }


def reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        metal.reset_peak_memory()


def metal_memory_payload() -> dict[str, Any]:
    metal = getattr(mx, "metal", None)
    if metal is None:
        return {
            "active_memory_bytes": None,
            "cache_memory_bytes": None,
            "peak_memory_bytes": None,
        }
    return {
        "active_memory_bytes": _call_optional_int(mx, "get_active_memory")
        if hasattr(mx, "get_active_memory")
        else _call_optional_int(metal, "get_active_memory"),
        "cache_memory_bytes": _call_optional_int(mx, "get_cache_memory")
        if hasattr(mx, "get_cache_memory")
        else _call_optional_int(metal, "get_cache_memory"),
        "peak_memory_bytes": _call_optional_int(mx, "get_peak_memory")
        if hasattr(mx, "get_peak_memory")
        else _call_optional_int(metal, "get_peak_memory"),
    }


def _call_optional_int(obj: Any, name: str) -> int | None:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:
        return None


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt, exit_code = run_receipt(args)
    write_json(args.output, receipt)
    if args.json or args.dry_run_json or exit_code != 0:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
        print(f"status: {receipt['status']}")
        print(f"steps_completed: {receipt['training']['steps_completed']}")
        print(f"final_loss: {receipt['training']['final_loss']}")
        print(f"peak_memory_bytes: {receipt['memory']['peak_memory_bytes']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
