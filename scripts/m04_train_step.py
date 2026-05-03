#!/usr/bin/env python3
"""M0.4 local MLX bf16 training-step receipt.

This is a correctness smoke for the local MLX training plumbing. It intentionally
uses the existing tiny hybrid model path until the full local_gb10_quarter
grad-checkpoint target-parquet gate is captured.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from cppmega_mlx.runtime.memory import (  # noqa: E402
    DEFAULT_METAL_RATIO,
    DEFAULT_WIRED_RATIO,
    memory_limit_api_status,
)
from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    local_gb10_quarter,
    local_gb10_quarter_profile,
)
from cppmega_mlx.training.optimizers import (  # noqa: E402
    ADAMW_BASE_CLASS,
    ADAMW_FP32_MOMENTS_CLASS,
    ADAMW_FP32_MOMENTS_SOURCE,
    collect_adamw_moment_dtypes,
    dtype_name,
    make_adamw,
)
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
TARGET_DATASET_NAME = "clang_semantic_4k_v10"
REQUIRED_MODEL_PROFILE = "local_gb10_quarter"
REQUIRED_MODEL_SOURCE = "cppmega_mlx.recipes.model_factory"
REQUIRED_DTYPE = "bfloat16"
REQUIRED_MODEL_GEOMETRY: dict[str, Any] = {
    "depth": 13,
    "hidden_size": 3584,
    "ffn_hidden_size": 18_944,
    "num_attention_heads": 28,
    "head_dim": 128,
    "vocab_size": 65_536,
    "pattern": "AEMEAEMEAEMR",
    "mtp": {
        "depth": 2,
        "beta": 0.6,
        "loss_weight": 0.3,
    },
}
CURRENT_MODEL_NAME = "HybridTinyLM"
FULL_PROFILE_ALLOCATION_MODE = "full_profile_allocation_probe"
ALLOCATION_PROBE_EVAL_SCOPE = "parameters_only_no_forward_no_training"
UNSUPPORTED_REQUIRED_MODEL_PROFILE_ROUTE_REASON = (
    "requested model_profile=local_gb10_quarter requires the real "
    "cppmega_mlx.recipes.model_factory local_gb10_quarter training route; "
    "the current HybridTinyLM smoke route is training-plumbing evidence only"
)
REQUIRED_OPTIMIZER_NAME = "AdamW"
REQUIRED_ADAMW_MASTER_MOMENT_DTYPE = "float32"
OBSERVED_OPTIMIZER_IDENTITY = {
    "name": REQUIRED_OPTIMIZER_NAME,
    "class": ADAMW_FP32_MOMENTS_CLASS,
    "base_class": ADAMW_BASE_CLASS,
    "source": ADAMW_FP32_MOMENTS_SOURCE,
    "construction": (
        "repo-local make_adamw(learning_rate=config.learning_rate, "
        "weight_decay=config.weight_decay) with fp32 AdamW moments"
    ),
}
GRAD_CHECKPOINT_EXPECTATION = {
    "required": True,
    "source": (
        "TrainHybridTinyConfig.grad_checkpoint -> HybridTinyConfig.grad_checkpoint "
        "-> HybridTinyLM mx.checkpoint block wrapper"
    ),
}
OPEN_M0_BLOCKERS = (
    {
        "id": "cppmega-mlx-t8f.4.local_gb10_quarter_gate",
        "title": (
            "full local_gb10_quarter bf16 AdamW + grad-checkpoint "
            "100-step target-parquet receipt is not captured"
        ),
        "impact": "HybridTinyLM receipts remain training-plumbing evidence only",
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
    parser.add_argument(
        "--model-profile",
        default=TrainHybridTinyConfig.model_profile,
        help=(
            "Receipt model/profile identity label passed through the training "
            "smoke. This does not by itself satisfy the local_gb10_quarter gate."
        ),
    )
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
        "--grad-checkpoint",
        action="store_true",
        help="Enable HybridTinyLM block checkpointing for the M0.4 smoke receipt.",
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
        "--memory-limit-total-bytes",
        type=int,
        default=None,
        help=(
            "Plan MLX wired/Metal memory limits from this total byte count. "
            "Does not apply unless --apply-memory-limit-plan is also set."
        ),
    )
    parser.add_argument(
        "--memory-limit-wired-ratio",
        type=float,
        default=DEFAULT_WIRED_RATIO,
        help="Wired-limit ratio for --memory-limit-total-bytes planning.",
    )
    parser.add_argument(
        "--memory-limit-metal-ratio",
        type=float,
        default=DEFAULT_METAL_RATIO,
        help="Metal allocator ratio for --memory-limit-total-bytes planning.",
    )
    parser.add_argument(
        "--apply-memory-limit-plan",
        action="store_true",
        help="Apply the planned MLX memory limits before training.",
    )
    parser.add_argument(
        "--clear-cache-every-steps",
        type=int,
        default=None,
        help=(
            "Run mx.clear_cache when the receipt wrapper observes a completed "
            "step whose number is divisible by this cadence."
        ),
    )
    parser.add_argument(
        "--probe-local-gb10-quarter-allocation",
        action="store_true",
        help=(
            "Opt-in M0.4 preflight: instantiate the full local_gb10_quarter "
            "profile and evaluate its parameter allocations. This records "
            "allocation evidence only; it does not run forward/training or "
            "close M0.4 by itself."
        ),
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
        model_profile=args.model_profile,
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
        grad_checkpoint=args.grad_checkpoint,
        token_key=args.token_key,
        memory_limit_total_bytes=args.memory_limit_total_bytes,
        memory_limit_wired_ratio=args.memory_limit_wired_ratio,
        memory_limit_metal_ratio=args.memory_limit_metal_ratio,
        apply_memory_limit_plan=args.apply_memory_limit_plan,
        clear_cache_every_steps=args.clear_cache_every_steps,
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
        return (
            blocked_receipt(
                args,
                "steps must be positive",
                "invalid_cli",
                probe_allocation=False,
            ),
            2,
        )
    if args.batch_size < 1:
        return (
            blocked_receipt(
                args,
                "batch_size must be positive",
                "invalid_cli",
                probe_allocation=False,
            ),
            2,
        )
    if args.seq_len < 2:
        return (
            blocked_receipt(
                args,
                "seq_len must be at least 2",
                "invalid_cli",
                probe_allocation=False,
            ),
            2,
        )

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
            original_token_key = args.token_key
            args.data_format = "npz"
            args.token_key = "tokens"
            try:
                payload, exit_code = _run_existing_training(args, data_path=data_path)
            finally:
                args.data_format = original_format
                args.token_key = original_token_key
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
    if config.model_profile == REQUIRED_MODEL_PROFILE:
        return (
            blocked_receipt(
                args,
                UNSUPPORTED_REQUIRED_MODEL_PROFILE_ROUTE_REASON,
                "unsupported_model_profile_route",
            ),
            0 if args.dry_run_json else 2,
        )

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
    memory_limit = train_payload.get("memory_limit")
    trainer_memory = train_payload.get("memory", {})
    if not isinstance(trainer_memory, dict):
        trainer_memory = {}
    clear_cache_events = list(trainer_memory.get("clear_cache_events") or [])
    api_status = memory_limit_api_status(mx)
    applied_memory_limit_api_path = applied_memory_limit_api_path_from_payload(
        memory_limit
    )
    optimizer = optimizer_identity(config, optimizer_updated=optimizer_updated)
    grad_checkpoint = grad_checkpoint_payload(config)
    local_gb10_preflight = local_gb10_quarter_preflight_from_args(args)
    observed_model_profile = train_payload.get("model_profile")
    model_config_for_gate = dict(model_config)
    if isinstance(observed_model_profile, str):
        model_config_for_gate.setdefault("profile", observed_model_profile)
    acceptance_gate = acceptance_gate_payload(
        data_path=config.npz_path,
        data_format=config.data_format,
        dtype=config.dtype,
        dataset=dataset,
        steps_requested=config.steps,
        steps_completed=len(step_metrics),
        loss_decreased=loss_decreased,
        all_finite=all_finite,
        optimizer_updated=optimizer_updated,
        model_name=CURRENT_MODEL_NAME,
        model_source=train_payload.get("model_source"),
        model_config=model_config_for_gate,
        optimizer=optimizer,
        grad_checkpoint=grad_checkpoint,
        device=train_payload.get("device", device_info()),
        local_gb10_quarter_preflight=local_gb10_preflight,
    )

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
        "local_gb10_quarter_preflight": local_gb10_preflight,
        "acceptance_gate": acceptance_gate,
        "workload": {
            "target_data_path": target_dataset_path(),
            "data_path": str(config.npz_path),
            "data_format": config.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": config.dtype,
            "steps_requested": config.steps,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "tokens_per_step": train_payload.get("tokens_per_step"),
            "compile_requested": config.compile,
            "model_profile": config.model_profile,
            "grad_checkpoint": config.grad_checkpoint,
            "mode": mode,
            "require_loss_decrease": bool(args.require_loss_decrease),
            "memory_limit_total_bytes": args.memory_limit_total_bytes,
            "memory_limit_wired_ratio": args.memory_limit_wired_ratio,
            "memory_limit_metal_ratio": args.memory_limit_metal_ratio,
            "apply_memory_limit_plan": bool(args.apply_memory_limit_plan),
            "clear_cache_every_steps": args.clear_cache_every_steps,
            "probe_local_gb10_quarter_allocation": bool(
                args.probe_local_gb10_quarter_allocation
            ),
        },
        "training": {
            "steps_completed": len(step_metrics),
            "optimizer_updated": optimizer_updated,
            "optimizer": optimizer,
            "grad_checkpoint": grad_checkpoint,
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
            "memory_limit": memory_limit,
            "memory_limit_api_status": api_status.to_dict(),
            "applied_memory_limit_api_path": applied_memory_limit_api_path,
            "clear_cache_every_steps": args.clear_cache_every_steps,
            "clear_cache_cadence_recorded": args.clear_cache_every_steps is not None,
            "clear_cache_events": clear_cache_events,
            "clear_cache_event_count": len(clear_cache_events),
            "clear_cache_event": clear_cache_events[-1] if clear_cache_events else None,
            "clear_cache_event_recorded": bool(clear_cache_events),
            "clear_cache_event_scope": (
                "train_hybrid_tiny_step_loop"
                if clear_cache_events
                else None
            ),
            "trainer_memory": trainer_memory or None,
        },
        "dataset": dataset,
        "model": {
            "source": train_payload.get("model_source"),
            "name": CURRENT_MODEL_NAME,
            "required_profile": REQUIRED_MODEL_PROFILE,
            "profile": observed_model_profile,
            "profile_matches_required": observed_model_profile == REQUIRED_MODEL_PROFILE,
            "local_gb10_quarter_preflight": local_gb10_preflight,
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


def target_dataset_path() -> str:
    return str(TARGET_PARQUET.relative_to(ROOT))


def applied_memory_limit_api_path_from_payload(memory_limit: Any) -> str | None:
    """Return the actual setter path recorded by the trainer payload."""

    if not isinstance(memory_limit, dict) or memory_limit.get("applied") is not True:
        return None
    api_path = memory_limit.get("metal_limit_api_path")
    return api_path if isinstance(api_path, str) and api_path else None


def _resolve_repo_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _string_from_mapping(mapping: Any, key: str) -> str | None:
    if isinstance(mapping, dict) and isinstance(mapping.get(key), str):
        return str(mapping[key])
    return None


def _dataset_receipt(dataset: Any) -> dict[str, Any]:
    if not isinstance(dataset, dict):
        return {}
    receipt = dataset.get("dataset_receipt")
    return receipt if isinstance(receipt, dict) else {}


def _dataset_source_path(dataset: Any) -> str | None:
    receipt = _dataset_receipt(dataset)
    return (
        _string_from_mapping(receipt, "source_path")
        or _string_from_mapping(dataset, "path")
        or _string_from_mapping(dataset, "source_path")
    )


def _dataset_source_format(dataset: Any) -> str | None:
    receipt = _dataset_receipt(dataset)
    metadata = dataset.get("metadata") if isinstance(dataset, dict) else None
    return (
        _string_from_mapping(receipt, "source_format")
        or _string_from_mapping(metadata, "source_format")
        or _string_from_mapping(dataset, "data_format")
    )


def _dataset_name(dataset: Any) -> str | None:
    receipt = _dataset_receipt(dataset)
    return _string_from_mapping(receipt, "source_dataset_name")


def _device_info_mapping(device: Any) -> dict[str, Any]:
    return device if isinstance(device, dict) else {}


def _mlx_device_info_mapping(device: Any) -> dict[str, Any]:
    device_info_payload = _device_info_mapping(device).get("mlx_device_info")
    return device_info_payload if isinstance(device_info_payload, dict) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _model_config_value(model_config: dict[str, Any], key: str) -> Any:
    if key in model_config:
        return model_config[key]
    config = model_config.get("config")
    if isinstance(config, dict):
        return config.get(key)
    return None


def _model_geometry_matches(model_config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    observed = {
        key: _model_config_value(model_config, key)
        for key in REQUIRED_MODEL_GEOMETRY
        if key != "mtp"
    }
    mtp_payload = _model_config_value(model_config, "mtp")
    if not isinstance(mtp_payload, dict):
        mtp_payload = _model_config_value(model_config, "mtp_profile")
    if not isinstance(mtp_payload, dict):
        mtp_payload = {}
    observed["mtp"] = {
        key: mtp_payload.get(key)
        for key in _mapping(REQUIRED_MODEL_GEOMETRY.get("mtp"))
    }
    return observed == REQUIRED_MODEL_GEOMETRY, observed


def _local_gb10_quarter_profile_geometry() -> dict[str, Any]:
    profile = local_gb10_quarter_profile()
    return {
        "depth": profile.depth,
        "hidden_size": profile.hidden_size,
        "ffn_hidden_size": profile.ffn_hidden_size,
        "num_attention_heads": profile.num_attention_heads,
        "head_dim": profile.head_dim,
        "vocab_size": profile.vocab_size,
        "pattern": profile.pattern,
        "mtp": {
            "depth": profile.mtp.depth,
            "beta": profile.mtp.beta,
            "loss_weight": profile.mtp.loss_weight,
        },
    }


def local_gb10_quarter_preflight_payload(
    *,
    allocation_attempted: bool = False,
    allocation_ready: bool | None = None,
    allocation_mode: str | None = None,
    allocation_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record target-profile readiness.

    The default preflight is allocation-free; the opt-in probe records a
    parameter-allocation-only check with no forward or training execution.
    """

    profile = local_gb10_quarter_profile()
    tokenizer_contract = profile.tokenizer_contract
    profile_geometry = _local_gb10_quarter_profile_geometry()
    geometry_matches_required = profile_geometry == REQUIRED_MODEL_GEOMETRY
    tokenizer_resolved = bool(tokenizer_contract.is_resolved)
    resolved_allocation_ready = bool(allocation_ready) if allocation_ready is not None else False
    resolved_allocation_mode = allocation_mode
    if resolved_allocation_mode is None:
        resolved_allocation_mode = (
            "caller_supplied_allocation_evidence"
            if allocation_attempted
            else "allocation_free_preflight"
        )
    blockers = []
    if not allocation_attempted:
        blockers.append("allocation_attempted")
    if not resolved_allocation_ready:
        blockers.append("allocation_ready")
    if resolved_allocation_mode != FULL_PROFILE_ALLOCATION_MODE:
        blockers.append("allocation_mode")
    if not tokenizer_resolved:
        blockers.append("tokenizer_contract_resolved")
    if not geometry_matches_required:
        blockers.append("geometry_matches_required")
    ok = bool(
        allocation_attempted
        and resolved_allocation_ready
        and resolved_allocation_mode == FULL_PROFILE_ALLOCATION_MODE
        and tokenizer_resolved
        and geometry_matches_required
    )
    payload = {
        "profile_name": profile.name,
        "source": REQUIRED_MODEL_SOURCE,
        "allocation_attempted": allocation_attempted,
        "allocation_ready": resolved_allocation_ready,
        "allocation_mode": resolved_allocation_mode,
        "required_geometry": REQUIRED_MODEL_GEOMETRY,
        "profile_geometry": profile_geometry,
        "geometry_matches_required": geometry_matches_required,
        "tokenizer_contract": {
            "resolved": tokenizer_resolved,
            "expected_vocab_size": tokenizer_contract.expected_vocab_size,
            "required_special_tokens": dict(tokenizer_contract.required_special_tokens),
            "milestone": tokenizer_contract.milestone,
            "blocker_id": tokenizer_contract.blocker_id,
            "reason": tokenizer_contract.reason,
        },
        "ok": ok,
        "blockers": blockers,
    }
    if allocation_probe is not None:
        payload["allocation_probe"] = allocation_probe
    return payload


def local_gb10_quarter_preflight_from_args(
    args: argparse.Namespace,
    *,
    probe_allocation: bool | None = None,
) -> dict[str, Any]:
    should_probe = (
        bool(args.probe_local_gb10_quarter_allocation)
        if probe_allocation is None
        else probe_allocation
    )
    if not should_probe:
        return local_gb10_quarter_preflight_payload()

    allocation_probe = probe_local_gb10_quarter_allocation()
    return local_gb10_quarter_preflight_payload(
        allocation_attempted=True,
        allocation_ready=allocation_probe.get("allocation_ready") is True,
        allocation_mode=FULL_PROFILE_ALLOCATION_MODE,
        allocation_probe=allocation_probe,
    )


def probe_local_gb10_quarter_allocation() -> dict[str, Any]:
    """Instantiate the full M0.4 target profile without forward or optimizer work."""

    model: Any | None = None
    memory_before = metal_memory_payload()
    profile_geometry = _local_gb10_quarter_profile_geometry()
    geometry_matches_required = profile_geometry == REQUIRED_MODEL_GEOMETRY
    identity_payload = {
        "source": REQUIRED_MODEL_SOURCE,
        "allocation_mode": FULL_PROFILE_ALLOCATION_MODE,
        "required_geometry": REQUIRED_MODEL_GEOMETRY,
        "profile_geometry": profile_geometry,
        "geometry_matches_required": geometry_matches_required,
    }
    try:
        model = local_gb10_quarter()
        mx.eval(model.parameters())
        mx.synchronize()
        memory_after = metal_memory_payload()
        return {
            "status": "ok",
            "allocation_ready": True,
            **identity_payload,
            "profile_name": REQUIRED_MODEL_PROFILE,
            "model_class": type(model).__name__,
            "eval_scope": ALLOCATION_PROBE_EVAL_SCOPE,
            "forward_executed": False,
            "training_executed": False,
            "memory_before": memory_before,
            "memory_after": memory_after,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "allocation_ready": False,
            **identity_payload,
            "profile_name": REQUIRED_MODEL_PROFILE,
            "eval_scope": ALLOCATION_PROBE_EVAL_SCOPE,
            "forward_executed": False,
            "training_executed": False,
            "memory_before": memory_before,
            "memory_after": metal_memory_payload(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if model is not None:
            del model
        try:
            mx.synchronize()
        except Exception:
            pass
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            try:
                clear_cache()
            except Exception:
                pass


def m4_runtime_metadata_ok(device: Any) -> bool:
    device_payload = _device_info_mapping(device)
    mlx_device_info = _mlx_device_info_mapping(device)
    device_name = str(mlx_device_info.get("device_name") or "")
    memory_size = mlx_device_info.get("memory_size")
    return bool(
        device_payload.get("metal_available") is True
        and device_payload.get("machine") == "arm64"
        and "macOS" in str(device_payload.get("platform") or "")
        and "M4" in device_name
        and isinstance(memory_size, int)
        and memory_size > 0
    )


def optimizer_identity(
    config: TrainHybridTinyConfig | argparse.Namespace,
    *,
    optimizer_updated: bool,
    master_moment_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    moment_evidence = master_moment_evidence or adamw_master_moment_evidence()
    return {
        **OBSERVED_OPTIMIZER_IDENTITY,
        "required_name": REQUIRED_OPTIMIZER_NAME,
        "name_matches_required": OBSERVED_OPTIMIZER_IDENTITY["name"] == REQUIRED_OPTIMIZER_NAME,
        "adamw": OBSERVED_OPTIMIZER_IDENTITY["name"] == "AdamW",
        "learning_rate": getattr(config, "learning_rate", getattr(config, "lr", None)),
        "weight_decay": getattr(config, "weight_decay", None),
        "update_observed": optimizer_updated,
        "required_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "master_moment_evidence": moment_evidence,
        "master_moment_dtype_ok": moment_evidence.get("ok") is True,
    }


class _AdamWMomentProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.ones((2, 2), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sum(x @ self.weight)


def adamw_master_moment_evidence() -> dict[str, Any]:
    """Probe installed MLX AdamW state dtype for bf16 parameters."""

    try:
        model = _AdamWMomentProbe()
        optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.0)

        def loss_fn(probe: _AdamWMomentProbe, x: mx.array) -> mx.array:
            return probe(x)

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        _, grads = loss_and_grad(model, mx.ones((2, 2), dtype=mx.bfloat16))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        moment_dtypes = collect_adamw_moment_dtypes(optimizer.state)
        ok = bool(
            moment_dtypes
            and all(
                dtype == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
                for dtype in moment_dtypes.values()
            )
        )
        return {
            "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
            "observed_parameter_dtype": dtype_name(model.weight),
            "observed_moment_dtypes": moment_dtypes,
            "optimizer_class": OBSERVED_OPTIMIZER_IDENTITY["class"],
            "optimizer_base_class": OBSERVED_OPTIMIZER_IDENTITY["base_class"],
            "state_keys": sorted(str(key) for key in optimizer.state),
            "ok": ok,
        }
    except Exception as exc:
        return {
            "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
            "observed_parameter_dtype": None,
            "observed_moment_dtypes": {},
            "optimizer_class": OBSERVED_OPTIMIZER_IDENTITY["class"],
            "optimizer_base_class": OBSERVED_OPTIMIZER_IDENTITY["base_class"],
            "state_keys": [],
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def grad_checkpoint_payload(
    config: TrainHybridTinyConfig | argparse.Namespace | None = None,
) -> dict[str, Any]:
    observed_enabled = bool(getattr(config, "grad_checkpoint", False))
    return {
        **GRAD_CHECKPOINT_EXPECTATION,
        "observed_enabled": observed_enabled,
        "expectation_satisfied": (
            observed_enabled if GRAD_CHECKPOINT_EXPECTATION["required"] else True
        ),
    }


def acceptance_gate_payload(
    *,
    data_path: str | None,
    data_format: str | None,
    dtype: str | None,
    dataset: dict[str, Any] | None,
    steps_requested: int,
    steps_completed: int,
    loss_decreased: bool,
    all_finite: bool,
    optimizer_updated: bool,
    model_name: str | None,
    model_source: str | None,
    model_config: dict[str, Any] | None,
    optimizer: dict[str, Any] | None,
    grad_checkpoint: dict[str, Any] | None,
    device: dict[str, Any] | None,
    local_gb10_quarter_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_path = target_dataset_path()
    resolved_target = TARGET_PARQUET.resolve()
    resolved_data_path = _resolve_repo_path(data_path)
    dataset_path = _dataset_source_path(dataset)
    resolved_dataset_path = _resolve_repo_path(dataset_path)
    uses_full_target_dataset = bool(
        resolved_data_path is not None and resolved_data_path == resolved_target
    )
    dataset_source_path_ok = bool(
        resolved_dataset_path is not None and resolved_dataset_path == resolved_target
    )
    dataset_format = _dataset_source_format(dataset)
    target_parquet_path_ok = uses_full_target_dataset and dataset_source_path_ok
    dataset_format_ok = data_format == "parquet" and dataset_format == "parquet"
    dataset_name = _dataset_name(dataset)
    dataset_name_ok = dataset_name == TARGET_DATASET_NAME
    dtype_ok = dtype == REQUIRED_DTYPE
    real_parquet_source_identity_ok = bool(
        target_parquet_path_ok and dataset_format_ok and dataset_name_ok
    )
    step_count_ok = steps_requested >= 100 and steps_completed >= steps_requested
    loss_fields_ok = all_finite and loss_decreased
    optimizer_update_ok = bool(optimizer_updated)
    full_target_100_step_completed = bool(
        real_parquet_source_identity_ok
        and dtype_ok
        and step_count_ok
        and loss_fields_ok
        and optimizer_update_ok
    )
    model_profile = None
    model_config_payload = _mapping(model_config)
    if isinstance(model_config_payload.get("profile"), str):
        model_profile = str(model_config_payload["profile"])
    model_source_ok = model_source == REQUIRED_MODEL_SOURCE
    model_geometry_ok, observed_model_geometry = _model_geometry_matches(model_config_payload)
    model_identity_ok = bool(
        model_name == REQUIRED_MODEL_PROFILE
        and model_source_ok
        and model_profile == REQUIRED_MODEL_PROFILE
        and model_geometry_ok
    )
    preflight_payload = _mapping(local_gb10_quarter_preflight)
    preflight_tokenizer = _mapping(preflight_payload.get("tokenizer_contract"))
    preflight_allocation_probe = _mapping(preflight_payload.get("allocation_probe"))
    preflight_profile = _string_from_mapping(preflight_payload, "profile_name")
    preflight_source = _string_from_mapping(preflight_payload, "source")
    local_gb10_quarter_preflight_ok = bool(
        preflight_payload.get("ok") is True
        and preflight_profile == REQUIRED_MODEL_PROFILE
        and preflight_source == REQUIRED_MODEL_SOURCE
        and preflight_payload.get("allocation_attempted") is True
        and preflight_payload.get("allocation_ready") is True
        and preflight_payload.get("allocation_mode") == FULL_PROFILE_ALLOCATION_MODE
        and preflight_allocation_probe.get("status") == "ok"
        and preflight_allocation_probe.get("allocation_ready") is True
        and preflight_allocation_probe.get("source") == REQUIRED_MODEL_SOURCE
        and preflight_allocation_probe.get("allocation_mode")
        == FULL_PROFILE_ALLOCATION_MODE
        and preflight_allocation_probe.get("profile_name") == REQUIRED_MODEL_PROFILE
        and preflight_allocation_probe.get("model_class") == CURRENT_MODEL_NAME
        and preflight_allocation_probe.get("eval_scope") == ALLOCATION_PROBE_EVAL_SCOPE
        and preflight_allocation_probe.get("forward_executed") is False
        and preflight_allocation_probe.get("training_executed") is False
        and preflight_allocation_probe.get("geometry_matches_required") is True
        and preflight_allocation_probe.get("required_geometry") == REQUIRED_MODEL_GEOMETRY
        and preflight_allocation_probe.get("profile_geometry") == REQUIRED_MODEL_GEOMETRY
        and preflight_payload.get("geometry_matches_required") is True
        and preflight_payload.get("required_geometry") == REQUIRED_MODEL_GEOMETRY
        and preflight_payload.get("profile_geometry") == REQUIRED_MODEL_GEOMETRY
        and preflight_tokenizer.get("resolved") is True
    )
    optimizer_payload = _mapping(optimizer)
    observed_optimizer_name = _string_from_mapping(optimizer_payload, "name")
    observed_optimizer_class = _string_from_mapping(optimizer_payload, "class")
    observed_optimizer_source = _string_from_mapping(optimizer_payload, "source")
    master_moment_evidence = _mapping(optimizer_payload.get("master_moment_evidence"))
    observed_master_moment_dtypes = _mapping(
        master_moment_evidence.get("observed_moment_dtypes")
    )
    optimizer_identity_ok = bool(
        observed_optimizer_name == REQUIRED_OPTIMIZER_NAME
        and observed_optimizer_class == OBSERVED_OPTIMIZER_IDENTITY["class"]
        and observed_optimizer_source == OBSERVED_OPTIMIZER_IDENTITY["source"]
        and optimizer_payload.get("required_name") == REQUIRED_OPTIMIZER_NAME
        and optimizer_payload.get("name_matches_required") is True
    )
    fp32_adamw_master_moments_ok = bool(
        optimizer_identity_ok
        and optimizer_payload.get("required_master_moment_dtype")
        == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
        and master_moment_evidence.get("required_dtype")
        == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
        and optimizer_payload.get("master_moment_dtype_ok") is True
        and master_moment_evidence.get("ok") is True
        and observed_master_moment_dtypes
        and all(
            dtype == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
            for dtype in observed_master_moment_dtypes.values()
        )
    )
    adamw_ok = bool(
        optimizer_identity_ok
        and optimizer_payload.get("adamw") is True
        and optimizer_update_ok
    )
    grad_checkpoint_payload_value = _mapping(grad_checkpoint)
    grad_checkpoint_enabled = grad_checkpoint_payload_value.get("observed_enabled")
    grad_checkpoint_expectation_ok = bool(
        grad_checkpoint_payload_value.get("required") is True
        and grad_checkpoint_payload_value.get("source")
        == GRAD_CHECKPOINT_EXPECTATION["source"]
        and grad_checkpoint_enabled is True
        and grad_checkpoint_payload_value.get("expectation_satisfied") is True
    )
    runtime_metadata_ok = m4_runtime_metadata_ok(device)
    gate_checks = {
        "real_parquet_source_identity_ok": real_parquet_source_identity_ok,
        "target_parquet_path_ok": target_parquet_path_ok,
        "dataset_name_ok": dataset_name_ok,
        "dataset_format_ok": dataset_format_ok,
        "dtype_ok": dtype_ok,
        "local_gb10_quarter_preflight_ok": local_gb10_quarter_preflight_ok,
        "model_identity_ok": model_identity_ok,
        "optimizer_identity_ok": optimizer_identity_ok,
        "fp32_adamw_master_moments_ok": fp32_adamw_master_moments_ok,
        "adamw_ok": adamw_ok,
        "grad_checkpoint_expectation_ok": grad_checkpoint_expectation_ok,
        "step_count_ok": step_count_ok,
        "loss_decrease_ok": loss_decreased,
        "loss_fields_ok": loss_fields_ok,
        "all_finite_ok": all_finite,
        "optimizer_update_ok": optimizer_update_ok,
        "m4_runtime_metadata_ok": runtime_metadata_ok,
    }
    full_local_gb10_quarter_gate_completed = all(gate_checks.values())
    failed_checks = sorted(key for key, value in gate_checks.items() if not value)
    return {
        "full_target_dataset": target_path,
        "uses_full_target_dataset": uses_full_target_dataset,
        "full_target_dataset_100_step_completed": full_target_100_step_completed,
        "full_target_dataset_100_step_required": True,
        "full_target_dataset_blocker": None
        if full_target_100_step_completed
        else (
            "receipt did not complete >=100 decreasing steps on the full target parquet; "
            "treat this as a partial training-plumbing smoke only"
        ),
        "local_gb10_quarter_required": True,
        "required_model_profile": REQUIRED_MODEL_PROFILE,
        "required_dtype": REQUIRED_DTYPE,
        "observed_dtype": dtype,
        "dtype_ok": dtype_ok,
        "local_gb10_quarter_preflight": preflight_payload,
        "local_gb10_quarter_preflight_ok": local_gb10_quarter_preflight_ok,
        "observed_model_name": model_name,
        "observed_model_source": model_source,
        "observed_model_profile": model_profile,
        "model_identity_ok": model_identity_ok,
        "model_identity": {
            "required_name": REQUIRED_MODEL_PROFILE,
            "observed_name": model_name,
            "required_source": REQUIRED_MODEL_SOURCE,
            "observed_source": model_source,
            "source_ok": model_source_ok,
            "required_profile": REQUIRED_MODEL_PROFILE,
            "observed_profile": model_profile,
            "profile_ok": model_profile == REQUIRED_MODEL_PROFILE,
            "required_geometry": REQUIRED_MODEL_GEOMETRY,
            "observed_geometry": observed_model_geometry,
            "geometry_ok": model_geometry_ok,
            "ok": model_identity_ok,
        },
        "required_optimizer_name": REQUIRED_OPTIMIZER_NAME,
        "observed_optimizer_name": observed_optimizer_name,
        "required_adamw_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "observed_adamw_master_moment_dtypes": observed_master_moment_dtypes,
        "fp32_adamw_master_moments_ok": fp32_adamw_master_moments_ok,
        "optimizer_identity_ok": optimizer_identity_ok,
        "adamw_ok": adamw_ok,
        "optimizer_identity": {
            "required_name": REQUIRED_OPTIMIZER_NAME,
            "observed_name": observed_optimizer_name,
            "observed_class": observed_optimizer_class,
            "observed_source": observed_optimizer_source,
            "observed_adamw": optimizer_payload.get("adamw"),
            "observed_update": optimizer_payload.get("update_observed"),
            "required_master_moment_dtype": optimizer_payload.get(
                "required_master_moment_dtype"
            ),
            "master_moment_evidence": master_moment_evidence,
            "master_moment_dtype_ok": optimizer_payload.get("master_moment_dtype_ok"),
            "ok": optimizer_identity_ok,
        },
        "grad_checkpoint_required": True,
        "grad_checkpoint_observed_enabled": grad_checkpoint_enabled,
        "grad_checkpoint_expectation_ok": grad_checkpoint_expectation_ok,
        "grad_checkpoint_identity": {
            "required": grad_checkpoint_payload_value.get("required"),
            "observed_enabled": grad_checkpoint_enabled,
            "expectation_satisfied": grad_checkpoint_payload_value.get(
                "expectation_satisfied"
            ),
            "source": grad_checkpoint_payload_value.get("source"),
            "ok": grad_checkpoint_expectation_ok,
        },
        "real_parquet_source_identity": {
            "required_path": target_path,
            "observed_data_path": data_path,
            "observed_dataset_source_path": dataset_path,
            "required_dataset_name": TARGET_DATASET_NAME,
            "observed_dataset_name": dataset_name,
            "required_format": "parquet",
            "observed_data_format": data_format,
            "observed_dataset_format": dataset_format,
            "ok": real_parquet_source_identity_ok,
        },
        "target_parquet_path_ok": target_parquet_path_ok,
        "dataset_name_ok": dataset_name_ok,
        "dataset_format_ok": dataset_format_ok,
        "step_count_ok": step_count_ok,
        "loss_decrease_ok": loss_decreased,
        "loss_fields_ok": loss_fields_ok,
        "all_finite_ok": all_finite,
        "optimizer_update_ok": optimizer_update_ok,
        "m4_runtime_metadata": {
            "required_device_family": "Apple M4",
            "observed_device_name": _mlx_device_info_mapping(device).get("device_name"),
            "observed_memory_size": _mlx_device_info_mapping(device).get("memory_size"),
            "observed_platform": _device_info_mapping(device).get("platform"),
            "observed_machine": _device_info_mapping(device).get("machine"),
            "metal_available": _device_info_mapping(device).get("metal_available"),
            "ok": runtime_metadata_ok,
        },
        "m4_runtime_metadata_ok": runtime_metadata_ok,
        "full_local_gb10_quarter_gate_completed": full_local_gb10_quarter_gate_completed,
        "full_local_gb10_quarter_gate_required": True,
        "full_local_gb10_quarter_gate_blockers": failed_checks,
    }


def blocked_receipt(
    args: argparse.Namespace,
    reason: str,
    reason_type: str,
    *,
    probe_allocation: bool | None = None,
) -> dict[str, Any]:
    local_gb10_preflight = local_gb10_quarter_preflight_from_args(
        args,
        probe_allocation=probe_allocation,
    )
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
        "local_gb10_quarter_preflight": local_gb10_preflight,
        "acceptance_gate": acceptance_gate_payload(
            data_path=str(args.data_path),
            data_format=args.data_format,
            dtype=args.dtype,
            dataset=None,
            steps_requested=args.steps,
            steps_completed=0,
            loss_decreased=False,
            all_finite=False,
            optimizer_updated=False,
            model_name=None,
            model_source=None,
            model_config=None,
            optimizer=optimizer_identity(args, optimizer_updated=False),
            grad_checkpoint=grad_checkpoint_payload(args),
            device=device_info(),
            local_gb10_quarter_preflight=local_gb10_preflight,
        ),
        "blockers": [
            {
                "type": reason_type,
                "reason": reason,
                "recoverable": True,
            },
            *OPEN_M0_BLOCKERS,
        ],
        "workload": {
            "target_data_path": target_dataset_path(),
            "data_path": str(args.data_path),
            "data_format": args.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": args.dtype,
            "steps_requested": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "compile_requested": bool(args.compile),
            "model_profile": args.model_profile,
            "grad_checkpoint": bool(args.grad_checkpoint),
            "require_loss_decrease": bool(args.require_loss_decrease),
            "memory_limit_total_bytes": args.memory_limit_total_bytes,
            "memory_limit_wired_ratio": args.memory_limit_wired_ratio,
            "memory_limit_metal_ratio": args.memory_limit_metal_ratio,
            "apply_memory_limit_plan": bool(args.apply_memory_limit_plan),
            "clear_cache_every_steps": args.clear_cache_every_steps,
            "probe_local_gb10_quarter_allocation": bool(
                args.probe_local_gb10_quarter_allocation
            ),
        },
        "training": {
            "steps_completed": 0,
            "optimizer_updated": False,
            "optimizer": optimizer_identity(args, optimizer_updated=False),
            "grad_checkpoint": grad_checkpoint_payload(args),
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
            "memory_limit": None,
            "memory_limit_api_status": memory_limit_api_status(mx).to_dict(),
            "clear_cache_every_steps": args.clear_cache_every_steps,
            "clear_cache_cadence_recorded": args.clear_cache_every_steps is not None,
            "clear_cache_event": None,
            "clear_cache_event_recorded": False,
            "clear_cache_event_scope": None,
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
