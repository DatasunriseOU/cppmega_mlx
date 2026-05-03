#!/usr/bin/env python3
"""Tiny token-dataset-backed MLX training smoke.

This is intentionally a local smoke CLI, not a production trainer. It verifies
that pre-tokenized local data can flow through the token dataset readers,
TinyLM, and the eager/compiled pretraining step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

from cppmega_mlx.data.token_dataset import (  # noqa: E402
    TokenBatchDataset,
    TokenDatasetFormat,
    open_token_dataset,
)
from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig  # noqa: E402
from cppmega_mlx.runtime.memory import (  # noqa: E402
    DEFAULT_METAL_RATIO,
    DEFAULT_WIRED_RATIO,
    apply_memory_limit_plan,
    device_total_memory_bytes,
    memory_limit_plan,
)
from cppmega_mlx.runtime.seed import capture_rng_state  # noqa: E402
from cppmega_mlx.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from cppmega_mlx.training.compiled import CompiledPretrainingStep  # noqa: E402
from cppmega_mlx.training.eval import EvalMetrics, evaluate_batches  # noqa: E402


DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


@dataclass(frozen=True)
class TrainTinyNpzConfig:
    npz_path: str
    dataset_format: TokenDatasetFormat | None = None
    batch_size: int = 2
    seq_len: int = 64
    steps: int = 1
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    vocab_size: int | None = None
    hidden_size: int = 64
    num_layers: int = 1
    num_heads: int = 4
    ffn_hidden_size: int = 128
    shuffle: bool = False
    token_key: str = "tokens"
    checkpoint_dir: str | None = None
    checkpoint_path: str | None = None
    checkpoint_save_interval: int = 0
    resume_from: str | None = None
    valid_npz_path: str | None = None
    valid_dataset_path: str | None = None
    valid_dataset_format: TokenDatasetFormat | None = None
    eval_batches: int = 0
    memory_limit_total_bytes: int | None = None
    memory_limit_wired_ratio: float = DEFAULT_WIRED_RATIO
    memory_limit_metal_ratio: float = DEFAULT_METAL_RATIO
    apply_memory_limit_plan: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train TinyLM for a few steps from a pre-tokenized local token dataset."
        ),
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help=(
            "Path to an NPZ/parquet token shard or suffixless Megatron "
            ".bin/.idx prefix."
        ),
    )
    parser.add_argument(
        "--dataset-format",
        choices=("npz", "parquet", "megatron"),
        default=None,
        help="Override dataset format inference.",
    )
    parser.add_argument("--batch-size", type=int, default=TrainTinyNpzConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=TrainTinyNpzConfig.seq_len)
    parser.add_argument("--steps", type=int, default=TrainTinyNpzConfig.steps)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=TrainTinyNpzConfig.dtype)
    parser.add_argument("--lr", type=float, default=TrainTinyNpzConfig.learning_rate)
    parser.add_argument(
        "--weight-decay", type=float, default=TrainTinyNpzConfig.weight_decay
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override dataset metadata vocab_size for the TinyLM head.",
    )
    parser.add_argument("--hidden-size", type=int, default=TrainTinyNpzConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=TrainTinyNpzConfig.num_layers)
    parser.add_argument("--num-heads", type=int, default=TrainTinyNpzConfig.num_heads)
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=TrainTinyNpzConfig.ffn_hidden_size
    )
    parser.add_argument("--seed", type=int, default=TrainTinyNpzConfig.seed)
    parser.add_argument("--token-key", default=TrainTinyNpzConfig.token_key)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory for interval checkpoints. Saves "
            "checkpoint-000001-style subdirectories when "
            "--checkpoint-save-interval is set."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Write a final full-training checkpoint to this directory/path.",
    )
    parser.add_argument(
        "--checkpoint-save-interval",
        type=int,
        default=TrainTinyNpzConfig.checkpoint_save_interval,
        help="Save an interval checkpoint every N global steps; 0 disables it.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume model, optimizer, and dataset cursor from a checkpoint.",
    )
    parser.add_argument(
        "--valid-npz-path",
        type=str,
        default=None,
        help="Optional validation dataset path; kept for NPZ CLI compatibility.",
    )
    parser.add_argument(
        "--valid-dataset-path",
        type=str,
        default=None,
        help="Optional validation dataset path for NPZ/parquet/Megatron data.",
    )
    parser.add_argument(
        "--valid-dataset-format",
        choices=("npz", "parquet", "megatron"),
        default=None,
        help="Override validation dataset format inference.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=TrainTinyNpzConfig.eval_batches,
        help=(
            "Maximum validation batches to evaluate; 0 evaluates the full "
            "validation dataset."
        ),
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
        default=TrainTinyNpzConfig.memory_limit_wired_ratio,
        help="Wired-limit ratio for --memory-limit-total-bytes planning.",
    )
    parser.add_argument(
        "--memory-limit-metal-ratio",
        type=float,
        default=TrainTinyNpzConfig.memory_limit_metal_ratio,
        help="Metal allocator ratio for --memory-limit-total-bytes planning.",
    )
    parser.add_argument(
        "--apply-memory-limit-plan",
        action="store_true",
        help=(
            "Apply the planned MLX wired and Metal memory limits before training. "
            "Dry-runs always report the plan without applying it."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only the metrics JSON object.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and dataset shape without training.",
    )
    parser.add_argument(
        "--dry-run-json",
        action="store_true",
        help="Validate configuration and emit the dry-run plan as JSON.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainTinyNpzConfig:
    return TrainTinyNpzConfig(
        npz_path=args.npz_path,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        dtype=args.dtype,
        compile=not args.no_compile,
        seed=args.seed,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        shuffle=args.shuffle,
        token_key=args.token_key,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_save_interval=args.checkpoint_save_interval,
        resume_from=args.resume_from,
        valid_npz_path=args.valid_npz_path,
        valid_dataset_path=args.valid_dataset_path,
        valid_dataset_format=args.valid_dataset_format,
        eval_batches=args.eval_batches,
        memory_limit_total_bytes=args.memory_limit_total_bytes,
        memory_limit_wired_ratio=args.memory_limit_wired_ratio,
        memory_limit_metal_ratio=args.memory_limit_metal_ratio,
        apply_memory_limit_plan=args.apply_memory_limit_plan,
    )


def validate_config(config: TrainTinyNpzConfig) -> None:
    path = Path(config.npz_path)
    if not _dataset_path_exists(path):
        raise ValueError(f"dataset path does not exist: {path}")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if config.steps < 1:
        raise ValueError("steps must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if config.vocab_size is not None and config.vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if config.hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    if config.num_layers < 1:
        raise ValueError("num_layers must be positive")
    if config.num_heads < 1:
        raise ValueError("num_heads must be positive")
    if config.ffn_hidden_size < 1:
        raise ValueError("ffn_hidden_size must be positive")
    if config.hidden_size % config.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if config.checkpoint_save_interval < 0:
        raise ValueError("checkpoint_save_interval must be >= 0")
    if config.checkpoint_save_interval and not config.checkpoint_dir:
        raise ValueError(
            "checkpoint_dir is required when checkpoint_save_interval is enabled"
        )
    if config.resume_from is not None and not Path(config.resume_from).exists():
        raise ValueError(f"resume checkpoint does not exist: {config.resume_from}")
    valid_path = validation_dataset_path(config)
    if (
        config.valid_npz_path is not None
        and config.valid_dataset_path is not None
        and config.valid_npz_path != config.valid_dataset_path
    ):
        raise ValueError(
            "valid_npz_path and valid_dataset_path must match when both are set"
        )
    if valid_path is not None and not _dataset_path_exists(Path(valid_path)):
        raise ValueError(f"validation dataset path does not exist: {valid_path}")
    if config.valid_dataset_format is not None and valid_path is None:
        raise ValueError(
            "valid_dataset_format requires valid_dataset_path or valid_npz_path"
        )
    if config.eval_batches < 0:
        raise ValueError("eval_batches must be >= 0")
    if config.memory_limit_total_bytes is not None and config.memory_limit_total_bytes <= 0:
        raise ValueError("memory_limit_total_bytes must be positive")
    memory_limit_plan(
        config.memory_limit_total_bytes or 1,
        wired_ratio=config.memory_limit_wired_ratio,
        metal_ratio=config.memory_limit_metal_ratio,
    )


def _dataset_path_exists(path: Path) -> bool:
    if path.exists():
        return True
    if path.suffix:
        return False
    return path.with_suffix(".bin").exists() or path.with_suffix(".idx").exists()


def validation_dataset_path(config: TrainTinyNpzConfig) -> str | None:
    return config.valid_dataset_path or config.valid_npz_path


def make_dataset(
    config: TrainTinyNpzConfig, *, loop: bool, resume_batch: int = 0
) -> TokenBatchDataset:
    return open_token_dataset(
        config.npz_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        format=config.dataset_format,
        token_key=config.token_key,
        shuffle=config.shuffle,
        seed=config.seed,
        loop=loop,
        resume_batch=resume_batch,
    )


def make_eval_dataset(config: TrainTinyNpzConfig) -> TokenBatchDataset | None:
    valid_path = validation_dataset_path(config)
    if valid_path is None:
        return None
    return open_token_dataset(
        valid_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        format=config.valid_dataset_format,
        token_key=config.token_key,
        shuffle=False,
        seed=config.seed,
        loop=False,
    )


def resolved_vocab_size(config: TrainTinyNpzConfig, dataset: TokenBatchDataset) -> int:
    vocab_size = config.vocab_size or dataset.metadata.vocab_size
    if vocab_size < 2:
        raise ValueError("resolved vocab_size must be at least 2")
    return vocab_size


def validate_dataset_for_training(
    config: TrainTinyNpzConfig, dataset: TokenBatchDataset, vocab_size: int
) -> None:
    if dataset.num_batches < 1:
        raise ValueError(
            "dataset does not contain enough full batches for the requested batch_size"
        )
    token_min, token_max = dataset.token_id_range()
    if token_min < 0:
        raise ValueError(f"tokens must be non-negative, found {token_min}")
    if token_max >= vocab_size:
        raise ValueError(
            f"tokens contain id {token_max}, which exceeds vocab_size={vocab_size}"
        )


def tiny_model_config(
    config: TrainTinyNpzConfig, dataset: TokenBatchDataset, vocab_size: int
) -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_hidden_size=config.ffn_hidden_size,
        max_seq_length=config.seq_len,
        structure_vocab_size=max(2, min(vocab_size, 32)),
    )


def parameter_count(model: TinyLM) -> int:
    return _nested_parameter_count(model.parameters())


def _nested_parameter_count(tree: Any) -> int:
    if hasattr(tree, "size"):
        return int(tree.size)
    if isinstance(tree, list | tuple):
        return sum(_nested_parameter_count(value) for value in tree)
    if not isinstance(tree, dict):
        return 0
    return sum(_nested_parameter_count(value) for value in tree.values())


def metadata_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def device_info() -> dict[str, Any]:
    info = {
        "default_device": str(mx.default_device()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx": metadata_version("mlx"),
        "mlx_lm": metadata_version("mlx-lm"),
        "mlx_disable_compile": os.environ.get("MLX_DISABLE_COMPILE"),
    }
    if hasattr(mx, "device_info"):
        info["mlx_device_info"] = mx.device_info()
    return info


def env_flag_enabled(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def compile_payload(config: TrainTinyNpzConfig, device: dict[str, Any]) -> dict[str, Any]:
    disabled_by_env = env_flag_enabled(device.get("mlx_disable_compile"))
    enabled = config.compile and not disabled_by_env
    return {
        "requested": config.compile,
        "enabled": enabled,
        "disabled_by_env": disabled_by_env,
        "backend": "mlx.core.compile" if enabled else "eager",
        "pattern": "mlx_lm_tuner_stateful_step" if enabled else "python_eager_step",
        "state_inputs_outputs": [
            "model.state",
            "optimizer.state",
            "mx.random.state",
        ]
        if enabled
        else [],
        "fixed_batch_signature": enabled,
        "mlx_disable_compile": device.get("mlx_disable_compile"),
    }


def memory_limit_payload(
    config: TrainTinyNpzConfig,
    *,
    apply: bool,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    should_plan = config.memory_limit_total_bytes is not None or config.apply_memory_limit_plan
    if not should_plan:
        return {
            "mode": "off",
            "apply_requested": config.apply_memory_limit_plan,
            "applied": False,
            "total_bytes_source": None,
            "plan": None,
            "previous_wired_limit_bytes": None,
            "previous_metal_limit_bytes": None,
            "metal_limit_api_path": None,
        }

    total_bytes = config.memory_limit_total_bytes
    total_bytes_source = "cli"
    if total_bytes is None:
        total_bytes = device_total_memory_bytes(mx_module or mx)
        total_bytes_source = "mlx.device_info"
    if total_bytes is None:
        raise ValueError(
            "memory_limit_total_bytes is required when MLX device memory_size is unavailable"
        )

    plan = memory_limit_plan(
        total_bytes,
        wired_ratio=config.memory_limit_wired_ratio,
        metal_ratio=config.memory_limit_metal_ratio,
    )
    applied = apply_memory_limit_plan(
        plan,
        mx_module=mx_module or mx,
        apply=apply and config.apply_memory_limit_plan,
    )
    return {
        "mode": "planned",
        "apply_requested": config.apply_memory_limit_plan,
        "applied": applied.applied,
        "total_bytes_source": total_bytes_source,
        "plan": plan.to_dict(),
        "previous_wired_limit_bytes": applied.previous_wired_limit_bytes,
        "previous_metal_limit_bytes": applied.previous_metal_limit_bytes,
        "metal_limit_api_path": applied.metal_limit_api_path,
    }


def metadata_non_negative_int(
    metadata: dict[str, Any],
    key: str,
    *,
    default: int,
    source: Path,
) -> int:
    value = metadata.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"checkpoint metadata {source}: {key} must be a non-negative integer"
        )
    return value


def metadata_batch_cursor_offset(
    metadata: dict[str, Any],
    *,
    default: int,
    source: Path,
) -> int:
    cursor = metadata.get("batch_cursor", {})
    if cursor is None:
        return default
    if not isinstance(cursor, dict):
        raise ValueError(
            f"checkpoint metadata {source}: batch_cursor must be an object"
        )
    if "global_batch_offset" not in cursor:
        return default
    value = cursor["global_batch_offset"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"checkpoint metadata {source}: "
            "batch_cursor.global_batch_offset must be a non-negative integer"
        )
    return value


def checkpoint_path_for_step(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"checkpoint-{step:06d}"


def checkpoint_metadata(
    *,
    config: TrainTinyNpzConfig,
    dataset: TokenBatchDataset,
    stepper: CompiledPretrainingStep,
    step: int,
    consumed_batches: int | None = None,
    rng: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cursor = dataset.cursor_after(step if consumed_batches is None else consumed_batches)
    return {
        "step": step,
        "trained_tokens": stepper.state.trained_tokens,
        "batch_cursor": cursor.__dict__,
        "training_config": asdict(config),
        "dataset": dataset_payload(dataset),
        "rng": rng if rng is not None else {
            "mode": "snapshot",
            "snapshot": capture_rng_state(),
        },
    }


def save_training_checkpoint(
    *,
    model: TinyLM,
    optimizer: optim.Optimizer,
    path: str | Path,
    config: TrainTinyNpzConfig,
    dataset: TokenBatchDataset,
    stepper: CompiledPretrainingStep,
    step: int,
    consumed_batches: int | None = None,
    rng: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return save_checkpoint(
        model,
        path,
        optimizer=optimizer,
        training_step=stepper,
        metadata=checkpoint_metadata(
            config=config,
            dataset=dataset,
            stepper=stepper,
            step=step,
            consumed_batches=consumed_batches,
            rng=rng,
        ),
    )


def dataset_payload(dataset: TokenBatchDataset) -> dict[str, Any]:
    side_channels = getattr(dataset, "_side_channels", {})
    side_channel_names = (
        sorted(side_channels) if isinstance(side_channels, dict) else []
    )
    payload = {
        "path": str(dataset.path),
        "token_key": dataset.token_key,
        "seq_len": dataset.seq_len,
        "batch_size": dataset.batch_size,
        "num_samples": dataset.num_samples,
        "num_batches": dataset.num_batches,
        "dropped_samples": dataset.dropped_samples,
        "shuffle": dataset.shuffle,
        "loop": dataset.loop,
        "metadata": _json_ready(dataset.metadata),
        "side_channels": side_channel_names,
        "dataset_receipt": dataset_receipt_payload(
            dataset,
            side_channels=side_channel_names,
        ),
    }
    index_metadata = getattr(dataset, "index_metadata", None)
    if index_metadata is not None:
        payload["index_metadata"] = _json_ready(index_metadata)
    return payload


def dataset_source_format(dataset: TokenBatchDataset) -> str:
    source_format = getattr(dataset.metadata, "source_format", None)
    if source_format is not None:
        return str(source_format)
    index_metadata = getattr(dataset, "index_metadata", None)
    index_source_format = getattr(index_metadata, "source_format", None)
    return str(index_source_format or "unknown")


def source_dataset_name(path: Path, *, source_format: str) -> str:
    stem = path.stem if path.suffix else path.name
    if source_format == "parquet":
        if stem == "val_00000" and path.parent.name:
            return path.parent.name
        for suffix in ("_local_train_eval_head", "_train_head", "_head"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
    return stem


def dataset_receipt_payload(
    dataset: TokenBatchDataset,
    *,
    side_channels: list[str] | None = None,
) -> dict[str, Any]:
    source_format = dataset_source_format(dataset)
    payload = {
        "source_format": source_format,
        "source_dataset_name": source_dataset_name(
            dataset.path,
            source_format=source_format,
        ),
        "source_path": str(dataset.path),
        "token_key": dataset.token_key,
        "seq_len": dataset.seq_len,
        "batch_size": dataset.batch_size,
        "num_samples": dataset.num_samples,
        "num_batches": dataset.num_batches,
        "dropped_samples": dataset.dropped_samples,
        "side_channels": (
            side_channels
            if side_channels is not None
            else sorted(getattr(dataset, "_side_channels", {}))
        ),
    }
    index_metadata = getattr(dataset, "index_metadata", None)
    if index_metadata is not None:
        payload["index_metadata"] = _json_ready(index_metadata)
        payload["megatron_indexed_receipt"] = {
            "ingress": "MegatronIndexedDataset",
            "path_accepts_suffixless_prefix": True,
            "sidecar_schema": "explicit_token_aligned_binary_side_channel_paths",
            "local_only": True,
            "receipt_scope": "local_mlx_training_ingress",
            "megatron_runtime_imported": False,
            "distributed_megatron_parity_claim": False,
            "gb10_training_correctness_claim": False,
            "m4_vs_gb10_throughput_parity_claim": False,
        }
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def eval_metrics_payload(metrics: EvalMetrics) -> dict[str, Any]:
    return {
        "loss": metrics.loss,
        "ntokens": metrics.ntokens,
        "batches": metrics.batches,
        "seconds": metrics.seconds,
        "tokens_per_second": metrics.tokens_per_second,
    }


def assert_finite_metric(name: str, value: Any) -> None:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, found {value!r}")


def evaluation_payload(
    *,
    config: TrainTinyNpzConfig,
    model: TinyLM,
    dataset: TokenBatchDataset,
) -> dict[str, Any]:
    requested_batches = (
        dataset.num_batches if config.eval_batches == 0 else config.eval_batches
    )
    batches_to_eval = min(requested_batches, dataset.num_batches)
    if batches_to_eval < 1:
        raise ValueError("validation dataset does not contain any full batches")

    batches = dataset.iter_batches(loop=False)
    metrics = evaluate_batches(model, (next(batches) for _ in range(batches_to_eval)))
    return {
        "dataset": dataset_payload(dataset),
        "requested_batches": config.eval_batches,
        "evaluated_batches": batches_to_eval,
        "metrics": eval_metrics_payload(metrics),
    }


def dry_run_payload(config: TrainTinyNpzConfig) -> dict[str, Any]:
    validate_config(config)
    memory_limit = memory_limit_payload(config, apply=False)
    dataset = make_dataset(config, loop=False)
    vocab_size = resolved_vocab_size(config, dataset)
    validate_dataset_for_training(config, dataset, vocab_size)
    eval_dataset = make_eval_dataset(config)
    if eval_dataset is not None:
        validate_dataset_for_training(config, eval_dataset, vocab_size)
    model_config = tiny_model_config(config, dataset, vocab_size)
    device = device_info()
    compile_plan = compile_payload(config, device)
    payload = {
        "status": "dry_run",
        "config": asdict(config),
        "dataset": dataset_payload(dataset),
        "model_config": model_config.to_dict(),
        "tokens_per_step": config.batch_size * (config.seq_len - 1),
        "planned_steps": config.steps,
        "compile": config.compile,
        "compile_enabled": compile_plan["enabled"],
        "compile_plan": compile_plan,
        "dtype": config.dtype,
        "device": device,
        "memory_limit": memory_limit,
    }
    if eval_dataset is not None:
        payload["evaluation"] = {
            "dataset": dataset_payload(eval_dataset),
            "requested_batches": config.eval_batches,
            "planned_batches": min(
                eval_dataset.num_batches,
                eval_dataset.num_batches
                if config.eval_batches == 0
                else config.eval_batches,
            ),
        }
    return payload


def train_tiny_npz(config: TrainTinyNpzConfig) -> dict[str, Any]:
    validate_config(config)
    memory_limit = memory_limit_payload(config, apply=True)
    mx.random.seed(config.seed)

    resume_metadata: dict[str, Any] | None = None
    resume_metadata_path: Path | None = None
    resume_step = 0
    resume_trained_tokens = 0
    resume_batch = 0
    if config.resume_from:
        resume_metadata_path = Path(config.resume_from)
        if resume_metadata_path.suffix == ".safetensors":
            resume_metadata_path = resume_metadata_path.with_suffix(".json")
        else:
            resume_metadata_path = resume_metadata_path / "metadata.json"
        if resume_metadata_path.exists():
            loaded_metadata: dict[str, Any] = json.loads(
                resume_metadata_path.read_text()
            )
            resume_metadata = loaded_metadata
            resume_step = metadata_non_negative_int(
                loaded_metadata,
                "step",
                default=0,
                source=resume_metadata_path,
            )
            resume_trained_tokens = metadata_non_negative_int(
                loaded_metadata,
                "trained_tokens",
                default=0,
                source=resume_metadata_path,
            )
            resume_batch = metadata_batch_cursor_offset(
                loaded_metadata,
                default=resume_step,
                source=resume_metadata_path,
            )

    dataset = make_dataset(config, loop=True, resume_batch=resume_batch)
    vocab_size = resolved_vocab_size(config, dataset)
    validate_dataset_for_training(config, dataset, vocab_size)
    eval_dataset = make_eval_dataset(config)
    if eval_dataset is not None:
        validate_dataset_for_training(config, eval_dataset, vocab_size)
    model_config = tiny_model_config(config, dataset, vocab_size)
    model = TinyLM(model_config)
    model.set_dtype(DTYPES[config.dtype])
    device = device_info()
    compile_plan = compile_payload(config, device)
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )
    stepper = CompiledPretrainingStep(
        model,
        optimizer,
        compile=bool(compile_plan["enabled"]),
        state={"step": resume_step, "trained_tokens": resume_trained_tokens},
    )
    if config.resume_from:
        if resume_metadata_path is None:
            raise ValueError("resume metadata path was not resolved")
        training_step = (
            stepper
            if isinstance((resume_metadata or {}).get("training_state"), dict)
            else None
        )
        resume_metadata = load_checkpoint(
            model,
            config.resume_from,
            optimizer=optimizer,
            training_step=training_step,
        )
        if training_step is not None:
            resume_step = stepper.state.step
            resume_trained_tokens = stepper.state.trained_tokens
        else:
            resume_step = metadata_non_negative_int(
                resume_metadata,
                "step",
                default=resume_step,
                source=resume_metadata_path,
            )
            resume_trained_tokens = metadata_non_negative_int(
                resume_metadata,
                "trained_tokens",
                default=resume_trained_tokens,
                source=resume_metadata_path,
            )
        resume_batch = metadata_batch_cursor_offset(
            resume_metadata,
            default=resume_batch,
            source=resume_metadata_path,
        )
    mx.eval(model.state, optimizer.state)

    step_metrics = []
    saved_checkpoints: list[dict[str, Any]] = []
    batches = dataset.iter_batches(loop=True)
    for _ in range(config.steps):
        metrics = stepper(next(batches))
        step_metrics.append(asdict(metrics))
        mx.synchronize()
        if (
            config.checkpoint_dir
            and config.checkpoint_save_interval
            and metrics.step % config.checkpoint_save_interval == 0
        ):
            path = checkpoint_path_for_step(config.checkpoint_dir, metrics.step)
            manifest = save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                path=path,
                config=config,
                dataset=dataset,
                stepper=stepper,
                step=metrics.step,
                consumed_batches=metrics.step - resume_step,
            )
            saved_checkpoints.append(
                {
                    "path": str(path),
                    "step": manifest["step"],
                    "trained_tokens": manifest["trained_tokens"],
                }
            )

    final_checkpoint: dict[str, Any] | None = None
    if config.checkpoint_path:
        manifest = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            path=config.checkpoint_path,
            config=config,
            dataset=dataset,
            stepper=stepper,
            step=stepper.state.step,
            consumed_batches=stepper.state.step - resume_step,
        )
        final_checkpoint = {
            "path": str(config.checkpoint_path),
            "step": manifest["step"],
            "trained_tokens": manifest["trained_tokens"],
        }

    losses = [item["loss"] for item in step_metrics]
    step_times = [item["seconds"] for item in step_metrics]
    tps_values = [item["tokens_per_second"] for item in step_metrics]
    final = step_metrics[-1]
    for index, item in enumerate(step_metrics, start=1):
        assert_finite_metric(f"step_metrics[{index}].loss", item["loss"])
        assert_finite_metric(f"step_metrics[{index}].ntokens", item["ntokens"])
        assert_finite_metric(
            f"step_metrics[{index}].tokens_per_second",
            item["tokens_per_second"],
        )
        if int(item["ntokens"]) <= 0:
            raise ValueError(f"step_metrics[{index}].ntokens must be positive")
        if int(item["trained_tokens"]) <= 0:
            raise ValueError(f"step_metrics[{index}].trained_tokens must be positive")
    eval_payload = (
        evaluation_payload(config=config, model=model, dataset=eval_dataset)
        if eval_dataset is not None
        else None
    )

    payload = {
        "status": "ok",
        "config": asdict(config),
        "dataset": dataset_payload(dataset),
        "model_source": "cppmega_mlx.models.tiny_lm",
        "model_config": model_config.to_dict(),
        "parameter_count": parameter_count(model),
        "device": device,
        "memory_limit": memory_limit,
        "steps": config.steps,
        "start_step": resume_step,
        "end_step": final["step"],
        "compile": config.compile,
        "compile_enabled": compile_plan["enabled"],
        "compile_plan": compile_plan,
        "dtype": config.dtype,
        "tokens_per_step": final["ntokens"],
        "trained_tokens": final["trained_tokens"],
        "final_loss": final["loss"],
        "mean_loss": statistics.fmean(losses),
        "mean_step_time_s": statistics.fmean(step_times),
        "median_step_time_s": statistics.median(step_times),
        "tokens_per_second": statistics.fmean(tps_values),
        "step_metrics": step_metrics,
        "resume": {
            "path": config.resume_from,
            "loaded": config.resume_from is not None,
            "step": resume_step,
            "trained_tokens": resume_trained_tokens,
            "batch_cursor": resume_metadata.get("batch_cursor")
            if resume_metadata
            else None,
        },
        "checkpoints": {
            "save_interval": config.checkpoint_save_interval,
            "checkpoint_dir": config.checkpoint_dir,
            "saved": saved_checkpoints,
            "final": final_checkpoint,
        },
    }
    if eval_payload is not None:
        payload["evaluation"] = eval_payload
    return payload


def print_human(payload: dict[str, Any]) -> None:
    config = payload["config"]
    dataset = payload["dataset"]
    print("cppmega.mlx tiny token training smoke")
    print(f"status: {payload['status']}")
    print(f"dataset_path: {dataset['path']}")
    print(
        "shape: "
        f"batch={config['batch_size']} seq={config['seq_len']} "
        f"vocab={payload['model_config']['vocab_size']} "
        f"hidden={config['hidden_size']} heads={config['num_heads']} "
        f"layers={config['num_layers']} dtype={config['dtype']}"
    )
    print(
        "dataset: "
        f"samples={dataset['num_samples']} batches={dataset['num_batches']} "
        f"dropped={dataset['dropped_samples']} side_channels={dataset['side_channels']}"
    )
    print(f"compile: {payload['compile']}")
    print(f"parameter_count: {payload['parameter_count']:,}")
    print(f"trained_tokens: {payload['trained_tokens']}")
    if payload.get("resume", {}).get("loaded"):
        print(f"resumed_from: {payload['resume']['path']}")
    if payload.get("checkpoints", {}).get("final"):
        print(f"checkpoint: {payload['checkpoints']['final']['path']}")
    print(f"mean_step_time_s: {payload['mean_step_time_s']:.6f}")
    print(f"tokens_per_second: {payload['tokens_per_second']:.2f}")
    print(f"final_loss: {payload['final_loss']:.6f}")
    if payload.get("evaluation"):
        eval_metrics = payload["evaluation"]["metrics"]
        print(
            "eval: "
            f"loss={eval_metrics['loss']:.6f} "
            f"batches={payload['evaluation']['evaluated_batches']} "
            f"ntokens={eval_metrics['ntokens']}"
        )
    print("\njson:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        payload = (
            dry_run_payload(config)
            if args.dry_run or args.dry_run_json
            else train_tiny_npz(config)
        )
    except Exception as exc:
        device = device_info()
        compile_plan = compile_payload(config, device)
        payload = {
            "status": "error",
            "error": str(exc),
            "config": asdict(config),
            "compile": config.compile,
            "compile_enabled": compile_plan["enabled"],
            "compile_plan": compile_plan,
            "device": device,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    if args.json or args.dry_run_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
