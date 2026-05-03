#!/usr/bin/env python3
"""Tiny HybridTinyLM MLX training smoke.

This is a local smoke path, not a production trainer. It reuses the existing
fixed-shape token dataset, next-token loss, and compiled/eager pretraining step
to prove that HybridTinyLM can consume a tiny NPZ shard end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

from cppmega_mlx.data.token_dataset import TokenBatchDataset, open_token_dataset  # noqa: E402
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM  # noqa: E402
from cppmega_mlx.runtime.memory import (  # noqa: E402
    DEFAULT_METAL_RATIO,
    DEFAULT_WIRED_RATIO,
    apply_memory_limit_plan,
    device_total_memory_bytes,
    maybe_clear_cache_after_step,
    memory_limit_plan,
)
from cppmega_mlx.runtime.seed import capture_rng_state  # noqa: E402
from cppmega_mlx.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from cppmega_mlx.training.compiled import CompiledPretrainingStep  # noqa: E402
from cppmega_mlx.training.cut_cross_entropy import DEFAULT_CHUNK_ROWS  # noqa: E402
from cppmega_mlx.training.eval import evaluate_batches  # noqa: E402
from cppmega_mlx.training.loss import (  # noqa: E402
    next_token_cross_entropy,
    next_token_cut_cross_entropy,
)
from cppmega_mlx.training.optimizers import (  # noqa: E402
    ADAMW_FP32_MOMENTS_CLASS,
    ADAMW_FP32_MOMENTS_SOURCE,
    MUON_ADAMW_MULTI_CLASS,
    MUON_ADAMW_MULTI_SOURCE,
    MUON_NS_CARRIER_ENV,
    make_adamw,
    make_muon,
)


DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}
STRUCTURE_MODEL_KWARG_NAMES = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
OPTIMIZER_ENV = "CPPMEGA_OPTIMIZER"
OPTIMIZER_NAMES = ("adamw", "muon")
OPTIMIZER_SOURCES = ("default", "cli", "env")
LOSS_BACKENDS = ("cross_entropy", "cce")
MODEL_NAME = "HybridTinyLM"
MODEL_SOURCE = "cppmega_mlx.models.hybrid_lm"
DEFAULT_MODEL_PROFILE = "hybrid_tiny"


@dataclass(frozen=True)
class TrainHybridTinyConfig:
    npz_path: str | None = None
    data_format: Literal["npz", "parquet", "megatron"] | None = None
    model_profile: str = DEFAULT_MODEL_PROFILE
    batch_size: int = 1
    seq_len: int = 8
    steps: int = 1
    dtype: str = "float32"
    compile: bool = False
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    optimizer_source: str = "default"
    loss_backend: Literal["cross_entropy", "cce"] = "cross_entropy"
    cce_chunk_rows: int = DEFAULT_CHUNK_ROWS
    vocab_size: int | None = None
    hidden_size: int = 8
    pattern: str = "AEMR"
    depth: int = 4
    dsa_a_layer_ranks: tuple[int, ...] = ()
    num_attention_heads: int = 1
    moe_num_experts: int = 2
    moe_top_k: int = 1
    moe_expert_hidden_size: int = 16
    moe_shared_expert_hidden_size: int | None = 8
    mamba_expand: int = 1
    mamba_head_dim: int = 4
    mamba_state_dim: int = 4
    mamba_groups: int = 1
    mamba_chunk_size: int = 4
    m2rnn_k_head_dim: int = 2
    m2rnn_v_head_dim: int = 2
    m2rnn_num_v_heads: int = 1
    m2rnn_num_f_heads: int = 1
    m2rnn_chunk_size: int = 4
    ngram_hash_enabled: bool = False
    ngram_hash_orders: tuple[int, ...] = (2, 3)
    ngram_hash_heads: int = 8
    ngram_hash_table_size: int = 500_000
    ngram_hash_embed_dim: int = 16
    ngram_hash_dropout: float = 0.0
    ngram_hash_seed: int | None = None
    include_structure: bool = True
    grad_checkpoint: bool = False
    shuffle: bool = False
    token_key: str = "tokens"
    valid_npz_path: str | None = None
    valid_dataset_path: str | None = None
    valid_dataset_format: Literal["npz", "parquet", "megatron"] | None = None
    eval_batches: int = 0
    checkpoint_dir: str | None = None
    checkpoint_path: str | None = None
    checkpoint_save_interval: int = 0
    resume_from: str | None = None
    memory_limit_total_bytes: int | None = None
    memory_limit_wired_ratio: float = DEFAULT_WIRED_RATIO
    memory_limit_metal_ratio: float = DEFAULT_METAL_RATIO
    apply_memory_limit_plan: bool = False
    clear_cache_every_steps: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train HybridTinyLM for a dry-run or a few tiny MLX smoke steps. "
            "If npz_path is omitted, a deterministic temporary shard is generated."
        ),
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default=None,
        help=(
            "Optional token shard path. NPZ, Parquet, and Megatron .bin/.idx "
            "prefixes are inferred by suffix; "
            "Parquet uses token_ids plus token-aligned side-channel aliases."
        ),
    )
    parser.add_argument(
        "--data-format",
        choices=("npz", "parquet", "megatron"),
        default=None,
        help="Override token shard format inference.",
    )
    parser.add_argument(
        "--model-profile",
        default=TrainHybridTinyConfig.model_profile,
        help=(
            "Receipt identity label for the model/profile being exercised. "
            "The default is the tiny local smoke profile, not local_gb10_quarter."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=TrainHybridTinyConfig.batch_size)
    parser.add_argument("--seq-len", type=int, default=TrainHybridTinyConfig.seq_len)
    parser.add_argument("--steps", type=int, default=TrainHybridTinyConfig.steps)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=TrainHybridTinyConfig.dtype)
    parser.add_argument("--lr", type=float, default=TrainHybridTinyConfig.learning_rate)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TrainHybridTinyConfig.weight_decay,
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_NAMES,
        default=None,
        help=(
            "Training optimizer. Defaults to AdamW; can also be selected with "
            f"{OPTIMIZER_ENV}=muon for opt-in local Muon+AdamW smokes."
        ),
    )
    parser.add_argument(
        "--loss-backend",
        choices=LOSS_BACKENDS,
        default=TrainHybridTinyConfig.loss_backend,
        help=(
            "Training loss backend. Defaults to materialized next-token "
            "cross-entropy; cce opts into the local MLX chunked CE forward."
        ),
    )
    parser.add_argument(
        "--cce-chunk-rows",
        type=int,
        default=TrainHybridTinyConfig.cce_chunk_rows,
        help="Rows per logits chunk for --loss-backend cce.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override NPZ metadata vocab_size for the HybridTinyLM head.",
    )
    parser.add_argument("--hidden-size", type=int, default=TrainHybridTinyConfig.hidden_size)
    parser.add_argument(
        "--pattern",
        default=TrainHybridTinyConfig.pattern,
        help=(
            "Hybrid layer route pattern. A=attention/transformer, E=MoE, "
            "M=Mamba3, R=M2RNN. Use --depth 1 with M, R, or A for single-route "
            "Apple GPU compile smokes."
        ),
    )
    parser.add_argument("--depth", type=int, default=TrainHybridTinyConfig.depth)
    parser.add_argument(
        "--dsa-a-layer-ranks",
        default="",
        help="Comma-separated 1-based A-layer ranks to mark as DSA routes.",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=TrainHybridTinyConfig.num_attention_heads,
    )
    parser.add_argument(
        "--moe-num-experts",
        type=int,
        default=TrainHybridTinyConfig.moe_num_experts,
    )
    parser.add_argument("--moe-top-k", type=int, default=TrainHybridTinyConfig.moe_top_k)
    parser.add_argument(
        "--moe-expert-hidden-size",
        type=int,
        default=TrainHybridTinyConfig.moe_expert_hidden_size,
    )
    parser.add_argument(
        "--moe-shared-expert-hidden-size",
        type=int,
        default=TrainHybridTinyConfig.moe_shared_expert_hidden_size,
        help="Set to 0 to disable the shared expert.",
    )
    parser.add_argument(
        "--mamba-expand",
        type=int,
        default=TrainHybridTinyConfig.mamba_expand,
    )
    parser.add_argument(
        "--mamba-head-dim",
        type=int,
        default=TrainHybridTinyConfig.mamba_head_dim,
    )
    parser.add_argument(
        "--mamba-state-dim",
        type=int,
        default=TrainHybridTinyConfig.mamba_state_dim,
    )
    parser.add_argument(
        "--mamba-groups",
        type=int,
        default=TrainHybridTinyConfig.mamba_groups,
    )
    parser.add_argument(
        "--mamba-chunk-size",
        type=int,
        default=TrainHybridTinyConfig.mamba_chunk_size,
    )
    parser.add_argument(
        "--m2rnn-k-head-dim",
        type=int,
        default=TrainHybridTinyConfig.m2rnn_k_head_dim,
    )
    parser.add_argument(
        "--m2rnn-v-head-dim",
        type=int,
        default=TrainHybridTinyConfig.m2rnn_v_head_dim,
    )
    parser.add_argument(
        "--m2rnn-num-v-heads",
        type=int,
        default=TrainHybridTinyConfig.m2rnn_num_v_heads,
    )
    parser.add_argument(
        "--m2rnn-num-f-heads",
        type=int,
        default=TrainHybridTinyConfig.m2rnn_num_f_heads,
    )
    parser.add_argument(
        "--m2rnn-chunk-size",
        type=int,
        default=TrainHybridTinyConfig.m2rnn_chunk_size,
    )
    parser.add_argument(
        "--ngram-hash",
        action="store_true",
        help=(
            "Enable cppmega n-gram hash enrichment. Hashes are derived from "
            "input_ids inside the model; NPZ ngram sidecars are rejected."
        ),
    )
    parser.add_argument(
        "--ngram-hash-orders",
        default=",".join(str(order) for order in TrainHybridTinyConfig.ngram_hash_orders),
        help="Comma-separated n-gram orders for --ngram-hash.",
    )
    parser.add_argument(
        "--ngram-hash-heads",
        type=int,
        default=TrainHybridTinyConfig.ngram_hash_heads,
    )
    parser.add_argument(
        "--ngram-hash-table-size",
        type=int,
        default=TrainHybridTinyConfig.ngram_hash_table_size,
    )
    parser.add_argument(
        "--ngram-hash-embed-dim",
        type=int,
        default=TrainHybridTinyConfig.ngram_hash_embed_dim,
    )
    parser.add_argument(
        "--ngram-hash-dropout",
        type=float,
        default=TrainHybridTinyConfig.ngram_hash_dropout,
    )
    parser.add_argument(
        "--ngram-hash-seed",
        type=int,
        default=TrainHybridTinyConfig.ngram_hash_seed,
    )
    parser.add_argument("--token-key", default=TrainHybridTinyConfig.token_key)
    parser.add_argument("--seed", type=int, default=TrainHybridTinyConfig.seed)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-structure", action="store_true")
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Wrap each block forward in mx.checkpoint to trade compute for memory.",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Run the training step through mlx.core.compile.",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Force eager training even when a caller supplies compile defaults.",
    )
    parser.set_defaults(compile=TrainHybridTinyConfig.compile)
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
        default=TrainHybridTinyConfig.eval_batches,
        help="Number of validation batches to evaluate; 0 disables eval.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--checkpoint-save-interval",
        type=int,
        default=TrainHybridTinyConfig.checkpoint_save_interval,
    )
    parser.add_argument("--resume-from", type=str, default=None)
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
        default=TrainHybridTinyConfig.memory_limit_wired_ratio,
        help="Wired-limit ratio for --memory-limit-total-bytes planning.",
    )
    parser.add_argument(
        "--memory-limit-metal-ratio",
        type=float,
        default=TrainHybridTinyConfig.memory_limit_metal_ratio,
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
        "--clear-cache-every-steps",
        type=int,
        default=TrainHybridTinyConfig.clear_cache_every_steps,
        help="Run mx.clear_cache after training steps divisible by this cadence.",
    )
    parser.add_argument("--json", action="store_true", help="Emit metrics JSON only.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-json", action="store_true")
    return parser


def parse_csv_ints(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def optimizer_from_args(args: argparse.Namespace) -> tuple[str, str]:
    raw_optimizer = getattr(args, "optimizer", None)
    if raw_optimizer is not None:
        return str(raw_optimizer).strip().lower(), "cli"

    env_optimizer = os.environ.get(OPTIMIZER_ENV)
    if env_optimizer is not None and env_optimizer.strip():
        return env_optimizer.strip().lower(), "env"

    return "adamw", "default"


def config_from_args(args: argparse.Namespace) -> TrainHybridTinyConfig:
    shared_hidden = args.moe_shared_expert_hidden_size
    optimizer, optimizer_source = optimizer_from_args(args)
    return TrainHybridTinyConfig(
        npz_path=args.npz_path,
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
        optimizer=optimizer,
        optimizer_source=optimizer_source,
        loss_backend=args.loss_backend,
        cce_chunk_rows=args.cce_chunk_rows,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        pattern=args.pattern,
        depth=args.depth,
        dsa_a_layer_ranks=parse_csv_ints(args.dsa_a_layer_ranks),
        num_attention_heads=args.num_attention_heads,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_expert_hidden_size=args.moe_expert_hidden_size,
        moe_shared_expert_hidden_size=shared_hidden if shared_hidden > 0 else None,
        mamba_expand=args.mamba_expand,
        mamba_head_dim=args.mamba_head_dim,
        mamba_state_dim=args.mamba_state_dim,
        mamba_groups=args.mamba_groups,
        mamba_chunk_size=args.mamba_chunk_size,
        m2rnn_k_head_dim=args.m2rnn_k_head_dim,
        m2rnn_v_head_dim=args.m2rnn_v_head_dim,
        m2rnn_num_v_heads=args.m2rnn_num_v_heads,
        m2rnn_num_f_heads=args.m2rnn_num_f_heads,
        m2rnn_chunk_size=args.m2rnn_chunk_size,
        ngram_hash_enabled=args.ngram_hash,
        ngram_hash_orders=parse_csv_ints(args.ngram_hash_orders),
        ngram_hash_heads=args.ngram_hash_heads,
        ngram_hash_table_size=args.ngram_hash_table_size,
        ngram_hash_embed_dim=args.ngram_hash_embed_dim,
        ngram_hash_dropout=args.ngram_hash_dropout,
        ngram_hash_seed=args.ngram_hash_seed,
        include_structure=not args.no_structure,
        grad_checkpoint=args.grad_checkpoint,
        shuffle=args.shuffle,
        token_key=args.token_key,
        valid_npz_path=args.valid_npz_path,
        valid_dataset_path=args.valid_dataset_path,
        valid_dataset_format=args.valid_dataset_format,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_save_interval=args.checkpoint_save_interval,
        resume_from=args.resume_from,
        memory_limit_total_bytes=args.memory_limit_total_bytes,
        memory_limit_wired_ratio=args.memory_limit_wired_ratio,
        memory_limit_metal_ratio=args.memory_limit_metal_ratio,
        apply_memory_limit_plan=args.apply_memory_limit_plan,
        clear_cache_every_steps=args.clear_cache_every_steps,
    )


def validate_config(config: TrainHybridTinyConfig) -> None:
    valid_path = validation_dataset_path(config)
    if (
        config.valid_npz_path is not None
        and config.valid_dataset_path is not None
        and config.valid_npz_path != config.valid_dataset_path
    ):
        raise ValueError(
            "valid_npz_path and valid_dataset_path must match when both are set"
        )
    if config.npz_path is not None and not _token_shard_exists(
        Path(config.npz_path),
        data_format=config.data_format,
    ):
        raise ValueError(f"token shard path does not exist: {config.npz_path}")
    if valid_path is not None and not _token_shard_exists(
        Path(valid_path),
        data_format=config.valid_dataset_format,
    ):
        raise ValueError(f"token shard path does not exist: {valid_path}")
    if config.valid_dataset_format is not None and valid_path is None:
        raise ValueError(
            "valid_dataset_format requires valid_dataset_path or valid_npz_path"
        )
    if config.data_format is not None and config.data_format not in {
        "npz",
        "parquet",
        "megatron",
    }:
        raise ValueError(f"unsupported data_format={config.data_format!r}")
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
    if config.optimizer not in OPTIMIZER_NAMES:
        allowed = ", ".join(OPTIMIZER_NAMES)
        raise ValueError(
            f"unsupported optimizer={config.optimizer!r}; expected one of: {allowed}"
        )
    if config.optimizer_source not in OPTIMIZER_SOURCES:
        allowed = ", ".join(OPTIMIZER_SOURCES)
        raise ValueError(
            "unsupported optimizer_source="
            f"{config.optimizer_source!r}; expected one of: {allowed}"
        )
    if config.loss_backend not in LOSS_BACKENDS:
        allowed = ", ".join(LOSS_BACKENDS)
        raise ValueError(
            f"unsupported loss_backend={config.loss_backend!r}; "
            f"expected one of: {allowed}"
        )
    if config.cce_chunk_rows <= 0:
        raise ValueError("cce_chunk_rows must be positive")
    if config.vocab_size is not None and config.vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if config.eval_batches < 0:
        raise ValueError("eval_batches must be >= 0")
    if config.checkpoint_save_interval < 0:
        raise ValueError("checkpoint_save_interval must be >= 0")
    if config.checkpoint_save_interval and not config.checkpoint_dir:
        raise ValueError(
            "checkpoint_dir is required when checkpoint_save_interval is enabled"
        )
    if config.resume_from is not None and not Path(config.resume_from).exists():
        raise ValueError(f"resume checkpoint does not exist: {config.resume_from}")
    if config.ngram_hash_enabled:
        if not config.ngram_hash_orders:
            raise ValueError("ngram_hash_orders must contain at least one n-gram order")
        if any(order <= 0 for order in config.ngram_hash_orders):
            raise ValueError("ngram_hash_orders must be positive")
        if config.ngram_hash_heads <= 0:
            raise ValueError("ngram_hash_heads must be positive")
        if config.ngram_hash_table_size <= 0:
            raise ValueError("ngram_hash_table_size must be positive")
        if config.ngram_hash_embed_dim <= 0:
            raise ValueError("ngram_hash_embed_dim must be positive")
        if not 0.0 <= config.ngram_hash_dropout < 1.0:
            raise ValueError("ngram_hash_dropout must be in [0, 1)")
    if config.memory_limit_total_bytes is not None and config.memory_limit_total_bytes <= 0:
        raise ValueError("memory_limit_total_bytes must be positive")
    memory_limit_plan(
        config.memory_limit_total_bytes or 1,
        wired_ratio=config.memory_limit_wired_ratio,
        metal_ratio=config.memory_limit_metal_ratio,
    )
    if config.clear_cache_every_steps is not None and config.clear_cache_every_steps <= 0:
        raise ValueError("clear_cache_every_steps must be positive when provided")


def validation_dataset_path(config: TrainHybridTinyConfig) -> str | None:
    return config.valid_dataset_path or config.valid_npz_path


def _token_shard_exists(
    path: Path,
    *,
    data_format: Literal["npz", "parquet", "megatron"] | None,
) -> bool:
    if path.exists():
        return True
    if data_format == "megatron" or path.suffix in {"", ".bin", ".idx", ".json"}:
        if path.suffix == ".bin":
            prefix = path.with_suffix("")
        elif path.suffix == ".idx":
            prefix = path.with_suffix("")
        elif path.name.endswith(".idx.json"):
            prefix = Path(str(path)[: -len(".idx.json")])
        elif path.suffix == ".json":
            prefix = path.with_suffix("")
        else:
            prefix = path
        return prefix.with_suffix(".bin").exists() or prefix.with_suffix(".idx").exists()
    return False


def write_synthetic_npz(path: Path, config: TrainHybridTinyConfig) -> None:
    vocab_size = config.vocab_size or 32
    samples = max(config.batch_size * max(config.steps, config.eval_batches, 1), 4)
    total_tokens = samples * config.seq_len
    tokens = (np.arange(total_tokens, dtype=np.int32) % vocab_size).reshape(
        samples,
        config.seq_len,
    )
    arrays: dict[str, Any] = {
        "tokens": tokens,
        "attention_mask": np.ones_like(tokens, dtype=np.float32),
        "vocab_size": np.array(vocab_size, dtype=np.int64),
        "tokenizer_contract": np.array("local_profile"),
    }
    if config.include_structure:
        arrays["structure_ids"] = (tokens % 7).astype(np.int32)
        arrays["dep_levels"] = (tokens % 3).astype(np.int32)
        arrays["ast_depth_ids"] = (tokens % 5).astype(np.int32)
        arrays["sibling_index_ids"] = (tokens % 11).astype(np.int32)
        arrays["node_type_ids"] = (tokens % 13).astype(np.int32)
    np.savez(path, **arrays)


def make_dataset(
    config: TrainHybridTinyConfig,
    *,
    path: str,
    loop: bool,
    data_format: Literal["npz", "parquet", "megatron"] | None = None,
    inherit_config_format: bool = True,
    resume_batch: int = 0,
) -> TokenBatchDataset:
    return open_token_dataset(
        path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        format=config.data_format
        if inherit_config_format and data_format is None
        else data_format,
        token_key=config.token_key,
        shuffle=config.shuffle,
        seed=config.seed,
        loop=loop,
        resume_batch=resume_batch,
    )


def resolved_vocab_size(
    config: TrainHybridTinyConfig,
    dataset: TokenBatchDataset,
) -> int:
    vocab_size = config.vocab_size or dataset.metadata.vocab_size
    if vocab_size < 2:
        raise ValueError("resolved vocab_size must be at least 2")
    return vocab_size


def validate_dataset_for_training(dataset: TokenBatchDataset, vocab_size: int) -> None:
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


def validate_side_channel_contract(
    config: TrainHybridTinyConfig,
    dataset: TokenBatchDataset,
) -> None:
    """Fail closed when caller intent conflicts with model-threaded side channels."""

    if config.include_structure:
        return
    structure_channels = dataset_structure_side_channels(dataset)
    if structure_channels:
        keys = ", ".join(structure_channels)
        raise ValueError(
            "dataset contains structure side channels "
            f"({keys}) but --no-structure was set; refusing to silently drop "
            "or train with them"
        )


def hybrid_model_config(
    config: TrainHybridTinyConfig,
    vocab_size: int,
) -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        pattern=config.pattern,
        depth=config.depth,
        dsa_a_layer_ranks=config.dsa_a_layer_ranks,
        num_attention_heads=config.num_attention_heads,
        max_seq_length=config.seq_len,
        structure_vocab_size=max(2, min(vocab_size, 32)),
        moe_num_experts=config.moe_num_experts,
        moe_top_k=config.moe_top_k,
        moe_expert_hidden_size=config.moe_expert_hidden_size,
        moe_shared_expert_hidden_size=config.moe_shared_expert_hidden_size,
        mamba_expand=config.mamba_expand,
        mamba_head_dim=config.mamba_head_dim,
        mamba_state_dim=config.mamba_state_dim,
        mamba_groups=config.mamba_groups,
        mamba_chunk_size=config.mamba_chunk_size,
        m2rnn_k_head_dim=config.m2rnn_k_head_dim,
        m2rnn_v_head_dim=config.m2rnn_v_head_dim,
        m2rnn_num_v_heads=config.m2rnn_num_v_heads,
        m2rnn_num_f_heads=config.m2rnn_num_f_heads,
        m2rnn_chunk_size=config.m2rnn_chunk_size,
        ngram_hash_enabled=config.ngram_hash_enabled,
        ngram_hash_orders=config.ngram_hash_orders,
        ngram_hash_heads=config.ngram_hash_heads,
        ngram_hash_table_size=config.ngram_hash_table_size,
        ngram_hash_embed_dim=config.ngram_hash_embed_dim,
        ngram_hash_dropout=config.ngram_hash_dropout,
        ngram_hash_seed=config.ngram_hash_seed,
        grad_checkpoint=config.grad_checkpoint,
    )


def parameter_count(model: HybridTinyLM) -> int:
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
        "mlx_metal": metadata_version("mlx-metal"),
        "mlx_disable_compile": os.environ.get("MLX_DISABLE_COMPILE"),
    }
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "is_available"):
        info["metal_available"] = bool(metal.is_available())
    if hasattr(mx, "device_info"):
        info["mlx_device_info"] = mx.device_info()
    return info


def env_flag_enabled(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def compile_payload(config: TrainHybridTinyConfig, device: dict[str, Any]) -> dict[str, Any]:
    disabled_by_env = env_flag_enabled(device.get("mlx_disable_compile"))
    enabled = config.compile and not disabled_by_env
    return {
        "requested": config.compile,
        "enabled": enabled,
        "disabled_by_env": disabled_by_env,
        "backend": "mlx.core.compile" if enabled else "eager",
        "pattern": (
            "mlx_lm_tuner_stateful_step"
            if enabled
            else "python_eager_step"
        ),
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


def training_optimizer_payload(config: TrainHybridTinyConfig) -> dict[str, Any]:
    if config.optimizer == "adamw":
        return {
            "name": "adamw",
            "source": config.optimizer_source,
            "class": ADAMW_FP32_MOMENTS_CLASS,
            "factory": ADAMW_FP32_MOMENTS_SOURCE,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "groups": {
                "adamw": {
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "betas": [0.9, 0.999],
                    "moment_dtype": "float32",
                },
            },
        }
    if config.optimizer == "muon":
        ns_carrier = os.environ.get(MUON_NS_CARRIER_ENV, "fp32")
        return {
            "name": "muon",
            "source": config.optimizer_source,
            "class": MUON_ADAMW_MULTI_CLASS,
            "factory": MUON_ADAMW_MULTI_SOURCE,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "cppmega_cuda_parity": False,
            "groups": {
                "muon": {
                    "learning_rate": config.learning_rate,
                    "momentum": 0.95,
                    "nesterov": True,
                    "ns_steps": 5,
                    "ns_carrier": ns_carrier,
                    "weight_decay": config.weight_decay,
                },
                "adamw": {
                    "learning_rate": 1e-4,
                    "weight_decay": config.weight_decay,
                    "betas": [0.9, 0.95],
                    "moment_dtype": "float32",
                },
            },
        }
    raise ValueError(
        f"unsupported optimizer={config.optimizer!r}; expected one of: "
        f"{', '.join(OPTIMIZER_NAMES)}"
    )


def safe_training_optimizer_payload(config: TrainHybridTinyConfig) -> dict[str, Any]:
    try:
        return training_optimizer_payload(config)
    except ValueError as exc:
        return {
            "name": config.optimizer,
            "source": config.optimizer_source,
            "error": str(exc),
        }


def training_loss_payload(config: TrainHybridTinyConfig) -> dict[str, Any]:
    if config.loss_backend == "cross_entropy":
        return {
            "backend": "cross_entropy",
            "source": "cppmega_mlx.training.loss.next_token_cross_entropy",
            "default": True,
            "chunk_rows": None,
            "forward_memory_saving_claim": False,
            "manual_chunked_backward": False,
            "eval_backend": "cross_entropy",
        }
    if config.loss_backend == "cce":
        return {
            "backend": "cce",
            "source": "cppmega_mlx.training.loss.next_token_cut_cross_entropy",
            "default": False,
            "chunk_rows": config.cce_chunk_rows,
            "forward_memory_saving_claim": True,
            "manual_chunked_backward": False,
            "eval_backend": "cross_entropy",
        }
    raise ValueError(
        f"unsupported loss_backend={config.loss_backend!r}; expected one of: "
        f"{', '.join(LOSS_BACKENDS)}"
    )


def safe_training_loss_payload(config: TrainHybridTinyConfig) -> dict[str, Any]:
    try:
        return training_loss_payload(config)
    except ValueError as exc:
        return {
            "backend": config.loss_backend,
            "chunk_rows": config.cce_chunk_rows,
            "error": str(exc),
        }


def make_training_loss_fn(config: TrainHybridTinyConfig) -> Any:
    if config.loss_backend == "cross_entropy":
        return next_token_cross_entropy
    if config.loss_backend == "cce":
        return partial(
            next_token_cut_cross_entropy,
            chunk_rows=config.cce_chunk_rows,
        )
    raise ValueError(
        f"unsupported loss_backend={config.loss_backend!r}; expected one of: "
        f"{', '.join(LOSS_BACKENDS)}"
    )


def make_training_optimizer(config: TrainHybridTinyConfig) -> Any:
    if config.optimizer == "adamw":
        return make_adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "muon":
        return make_muon(
            lr_muon=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(
        f"unsupported optimizer={config.optimizer!r}; expected one of: "
        f"{', '.join(OPTIMIZER_NAMES)}"
    )


def memory_limit_payload(
    config: TrainHybridTinyConfig,
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
    }


def reset_peak_memory() -> bool:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return True
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        metal.reset_peak_memory()
        return True
    return False


def metal_memory_payload() -> dict[str, int | None]:
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


def validate_resume_training_optimizer(
    metadata: dict[str, Any],
    config: TrainHybridTinyConfig,
    *,
    source: Path,
) -> None:
    payload = metadata.get("training_optimizer")
    if payload is None:
        return
    if not isinstance(payload, dict):
        raise ValueError(
            f"checkpoint metadata {source}: training_optimizer must be an object"
        )
    checkpoint_optimizer = payload.get("name")
    if not isinstance(checkpoint_optimizer, str) or not checkpoint_optimizer:
        raise ValueError(
            f"checkpoint metadata {source}: training_optimizer.name must be a string"
        )
    if checkpoint_optimizer != config.optimizer:
        raise ValueError(
            "checkpoint optimizer "
            f"{checkpoint_optimizer!r} does not match requested optimizer "
            f"{config.optimizer!r}"
        )


def assert_finite_metric(name: str, value: Any) -> None:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, found {value!r}")


def checkpoint_path_for_step(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"checkpoint-{step:06d}"


def dataset_side_channel_names(dataset: TokenBatchDataset) -> list[str]:
    side_channels = getattr(dataset, "_side_channels", {})
    return sorted(str(name) for name in side_channels)


def dataset_structure_side_channels(dataset: TokenBatchDataset) -> list[str]:
    side_channels = set(dataset_side_channel_names(dataset))
    return sorted(name for name in STRUCTURE_MODEL_KWARG_NAMES if name in side_channels)


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
            side_channels if side_channels is not None else dataset_side_channel_names(dataset)
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
    parquet_receipt = getattr(dataset, "parquet_receipt", None)
    if parquet_receipt is not None:
        payload["parquet_receipt"] = _json_ready(parquet_receipt)
    return payload


def side_channel_contract_payload(
    dataset: TokenBatchDataset,
    config: TrainHybridTinyConfig | None = None,
) -> dict[str, Any]:
    threaded_structure = dataset_structure_side_channels(dataset)
    ngram_enabled = bool(config.ngram_hash_enabled) if config is not None else False
    return {
        "structure_side_channels": {
            "threaded_to_model": threaded_structure,
            "model_kwarg_names": list(STRUCTURE_MODEL_KWARG_NAMES),
            "batch_slice": "tokens[:, :-1]",
            "attention_mask_is_loss_only": "attention_mask" in dataset_side_channel_names(dataset),
        },
        "ngram_hash": {
            "enabled": ngram_enabled,
            "source": "input_ids",
            "threaded_to_model": "HybridTinyLM.__call__(input_ids)",
            "batch_slice": "tokens[:, :-1]",
            "model_derived": True,
            "sidecars_supported": False,
            "orders": list(config.ngram_hash_orders) if config is not None else list(TrainHybridTinyConfig.ngram_hash_orders),
            "heads": config.ngram_hash_heads if config is not None else TrainHybridTinyConfig.ngram_hash_heads,
        },
        "unsupported_sidecars_fail_closed": True,
    }


def dataset_payload(
    dataset: TokenBatchDataset,
    config: TrainHybridTinyConfig | None = None,
) -> dict[str, Any]:
    side_channels = dataset_side_channel_names(dataset)
    return {
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
        "side_channels": side_channels,
        "dataset_receipt": dataset_receipt_payload(
            dataset,
            side_channels=side_channels,
        ),
        "side_channel_contract": side_channel_contract_payload(dataset, config),
    }


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


def route_backend_payload(model: HybridTinyLM) -> dict[str, Any]:
    layer_backends = [str(layer.backend) for layer in model.layers]
    backend_summary = {
        backend: layer_backends.count(backend) for backend in sorted(set(layer_backends))
    }
    attention_backends = [
        str(route_info.backend)
        for layer in model.layers
        if (route_info := getattr(layer.block, "route_info", None)) is not None
    ]
    return {
        "route_symbols": "".join(model.route_symbols),
        "route_roles": list(model.route_roles),
        "layer_backends": layer_backends,
        "backend_summary": backend_summary,
        "attention_backends": attention_backends,
        "execution_backend": "mlx"
        if not attention_backends
        else "+".join(sorted(set(["mlx", *attention_backends]))),
    }


def checkpoint_metadata(
    *,
    config: TrainHybridTinyConfig,
    dataset: TokenBatchDataset,
    stepper: CompiledPretrainingStep,
    step: int,
    consumed_batches: int | None = None,
    evaluation: dict[str, Any] | None = None,
    rng: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cursor = dataset.cursor_after(step if consumed_batches is None else consumed_batches)
    cursor_payload = cursor.__dict__
    payload = {
        "step": step,
        "trained_tokens": stepper.state.trained_tokens,
        "batch_cursor": cursor_payload,
        "resume_cursor": {
            "step": step,
            "trained_tokens": stepper.state.trained_tokens,
            "batch_cursor": cursor_payload,
        },
        "training_config": asdict(config),
        "training_optimizer": training_optimizer_payload(config),
        "training_loss": training_loss_payload(config),
        "model_name": MODEL_NAME,
        "model_profile": config.model_profile,
        "model_source": MODEL_SOURCE,
        "dataset": dataset_payload(dataset, config),
        "rng": rng if rng is not None else {
            "mode": "snapshot",
            "snapshot": capture_rng_state(),
        },
    }
    if evaluation is not None:
        payload["evaluation"] = evaluation
    return payload


def save_training_checkpoint(
    *,
    model: HybridTinyLM,
    optimizer: Any,
    path: str | Path,
    config: TrainHybridTinyConfig,
    dataset: TokenBatchDataset,
    stepper: CompiledPretrainingStep,
    step: int,
    consumed_batches: int | None = None,
    evaluation: dict[str, Any] | None = None,
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
            evaluation=evaluation,
            rng=rng,
        ),
    )


def eval_payload(
    *,
    model: HybridTinyLM,
    config: TrainHybridTinyConfig,
    valid_path: str | None,
) -> dict[str, Any] | None:
    if config.eval_batches == 0:
        return None
    path = valid_path
    if path is None:
        path = config.npz_path
    if path is None:
        raise ValueError("eval_batches requires a validation or training NPZ path")

    data_format = (
        config.valid_dataset_format
        if valid_path is not None
        else config.data_format
    )
    dataset = make_dataset(
        config,
        path=path,
        loop=False,
        data_format=data_format,
        inherit_config_format=False,
    )
    validate_side_channel_contract(config, dataset)
    vocab_size = resolved_vocab_size(config, dataset)
    validate_dataset_for_training(dataset, vocab_size)
    planned_batches = min(config.eval_batches, dataset.num_batches)
    batches = dataset.iter_batches(loop=False)
    metrics = evaluate_batches(model, (next(batches) for _ in range(planned_batches)))
    return {
        "dataset": dataset_payload(dataset, config),
        "requested_batches": config.eval_batches,
        "planned_batches": planned_batches,
        "evaluated_batches": planned_batches,
        "metrics": asdict(metrics),
    }


def dry_run_payload(
    config: TrainHybridTinyConfig,
    *,
    npz_path: str,
    valid_path: str | None,
) -> dict[str, Any]:
    validate_config(config)
    memory_limit = memory_limit_payload(config, apply=False)
    dataset = make_dataset(config, path=npz_path, loop=False)
    validate_side_channel_contract(config, dataset)
    vocab_size = resolved_vocab_size(config, dataset)
    validate_dataset_for_training(dataset, vocab_size)
    model_config = hybrid_model_config(config, vocab_size)
    model = HybridTinyLM(model_config)
    route_backend = route_backend_payload(model)
    device = device_info()
    compile_plan = compile_payload(config, device)
    evaluation = None
    if config.eval_batches:
        eval_path = valid_path or npz_path
        eval_format = config.valid_dataset_format if valid_path is not None else None
        eval_dataset = make_dataset(
            config,
            path=eval_path,
            loop=False,
            data_format=eval_format,
            inherit_config_format=valid_path is None,
        )
        validate_side_channel_contract(config, eval_dataset)
        eval_vocab_size = resolved_vocab_size(config, eval_dataset)
        validate_dataset_for_training(eval_dataset, eval_vocab_size)
        evaluation = {
            "dataset": dataset_payload(eval_dataset, config),
            "requested_batches": config.eval_batches,
            "planned_batches": min(config.eval_batches, eval_dataset.num_batches),
        }
    return {
        "status": "dry_run",
        "config": asdict(config),
        "training_optimizer": training_optimizer_payload(config),
        "training_loss": training_loss_payload(config),
        "synthetic_npz": config.npz_path is None,
        "dataset": dataset_payload(dataset, config),
        "model_name": MODEL_NAME,
        "model_profile": config.model_profile,
        "model_source": MODEL_SOURCE,
        "model_config": model_config.to_dict(),
        "route_symbols": route_backend["route_symbols"],
        "route_roles": route_backend["route_roles"],
        "backend_plan": route_backend,
        "parameter_count": parameter_count(model),
        "tokens_per_step": config.batch_size * (config.seq_len - 1),
        "planned_steps": config.steps,
        "compile": config.compile,
        "compile_enabled": compile_plan["enabled"],
        "compile_plan": compile_plan,
        "dtype": config.dtype,
        "device": device,
        "memory_limit": memory_limit,
        "evaluation": evaluation,
    }


def train_hybrid_tiny(
    config: TrainHybridTinyConfig,
    *,
    npz_path: str,
    valid_path: str | None,
) -> dict[str, Any]:
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
            validate_resume_training_optimizer(
                loaded_metadata,
                config,
                source=resume_metadata_path,
            )
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

    dataset = make_dataset(config, path=npz_path, loop=True, resume_batch=resume_batch)
    validate_side_channel_contract(config, dataset)
    vocab_size = resolved_vocab_size(config, dataset)
    validate_dataset_for_training(dataset, vocab_size)
    model_config = hybrid_model_config(config, vocab_size)
    model = HybridTinyLM(model_config)
    model.set_dtype(DTYPES[config.dtype])
    route_backend = route_backend_payload(model)
    device = device_info()
    compile_plan = compile_payload(config, device)
    peak_memory_reset = reset_peak_memory()
    memory_before = metal_memory_payload()
    requested_resume_cursor = (
        {
            "step": resume_step,
            "trained_tokens": resume_trained_tokens,
            "batch_cursor": dataset.cursor_after(0).__dict__,
        }
        if config.resume_from
        else None
    )
    optimizer = make_training_optimizer(config)
    loss_fn = make_training_loss_fn(config)

    stepper = CompiledPretrainingStep(
        model,
        optimizer,
        compile=bool(compile_plan["enabled"]),
        state={"step": resume_step, "trained_tokens": resume_trained_tokens},
        loss_fn=loss_fn,
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
        requested_resume_cursor = {
            "step": resume_step,
            "trained_tokens": resume_trained_tokens,
            "batch_cursor": dataset.cursor_after(0).__dict__,
        }
    mx.eval(model.state, optimizer.state)

    step_metrics = []
    clear_cache_events: list[dict[str, Any]] = []
    saved_checkpoints: list[dict[str, Any]] = []
    batches = dataset.iter_batches(loop=True)
    for _ in range(config.steps):
        metrics = stepper(next(batches))
        step_metrics.append(asdict(metrics))
        mx.synchronize()
        clear_cache_event = maybe_clear_cache_after_step(
            metrics.step,
            config.clear_cache_every_steps,
            mx_module=mx,
            synchronize=False,
        )
        if clear_cache_event is not None:
            clear_cache_events.append(clear_cache_event.to_dict())
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

    evaluation = eval_payload(
        model=model,
        config=config,
        valid_path=valid_path,
    )
    mx.synchronize()
    memory_after = metal_memory_payload()

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
            evaluation=evaluation,
        )
        final_checkpoint = {
            "path": str(config.checkpoint_path),
            "step": manifest["step"],
            "trained_tokens": manifest["trained_tokens"],
        }

    return {
        "status": "ok",
        "config": asdict(config),
        "training_optimizer": training_optimizer_payload(config),
        "training_loss": training_loss_payload(config),
        "synthetic_npz": config.npz_path is None,
        "dataset": dataset_payload(dataset, config),
        "model_name": MODEL_NAME,
        "model_profile": config.model_profile,
        "model_source": MODEL_SOURCE,
        "model_config": model_config.to_dict(),
        "route_symbols": route_backend["route_symbols"],
        "route_roles": route_backend["route_roles"],
        "backend_plan": route_backend,
        "parameter_count": parameter_count(model),
        "device": device,
        "memory_limit": memory_limit,
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "peak_memory_bytes": memory_after.get("peak_memory_bytes"),
            "peak_memory_reset": peak_memory_reset,
            "clear_cache_every_steps": config.clear_cache_every_steps,
            "clear_cache_events": clear_cache_events,
            "clear_cache_event_count": len(clear_cache_events),
        },
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
        "evaluation": evaluation,
        "resume": {
            "path": config.resume_from,
            "loaded": config.resume_from is not None,
            "step": resume_step,
            "trained_tokens": resume_trained_tokens,
            "batch_cursor": resume_metadata.get("batch_cursor")
            if resume_metadata
            else None,
            "resume_cursor": requested_resume_cursor,
        },
        "checkpoints": {
            "save_interval": config.checkpoint_save_interval,
            "checkpoint_dir": config.checkpoint_dir,
            "saved": saved_checkpoints,
            "final": final_checkpoint,
        },
    }


def print_human(payload: dict[str, Any]) -> None:
    config = payload["config"]
    dataset = payload["dataset"]
    print("cppmega.mlx HybridTinyLM training smoke")
    print(f"status: {payload['status']}")
    print(f"data_path: {dataset['path']}")
    print(
        "shape: "
        f"batch={config['batch_size']} seq={config['seq_len']} "
        f"vocab={payload['model_config']['vocab_size']} "
        f"hidden={config['hidden_size']} heads={config['num_attention_heads']} "
        f"depth={config['depth']} dtype={config['dtype']}"
    )
    print(
        "dataset: "
        f"samples={dataset['num_samples']} batches={dataset['num_batches']} "
        f"dropped={dataset['dropped_samples']} side_channels={dataset['side_channels']}"
    )
    print(f"route: {payload['route_symbols']}")
    print(f"compile: {payload['compile']}")
    print(f"parameter_count: {payload['parameter_count']:,}")
    if payload["status"] == "ok":
        print(f"trained_tokens: {payload['trained_tokens']}")
        print(f"mean_step_time_s: {payload['mean_step_time_s']:.6f}")
        print(f"tokens_per_second: {payload['tokens_per_second']:.2f}")
        print(f"final_loss: {payload['final_loss']:.6f}")
    print("\njson:")
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_with_optional_synthetic_npz(
    config: TrainHybridTinyConfig,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    if config.npz_path is not None:
        valid_path = validation_dataset_path(config)
        return (
            dry_run_payload(
                config,
                npz_path=config.npz_path,
                valid_path=valid_path,
            )
            if dry_run
            else train_hybrid_tiny(
                config,
                npz_path=config.npz_path,
                valid_path=valid_path,
            )
        )

    with tempfile.TemporaryDirectory(prefix="cppmega_mlx_hybrid_tiny_") as tmp:
        npz_path = Path(tmp) / "tokens.npz"
        write_synthetic_npz(npz_path, config)
        valid_path = validation_dataset_path(config) or str(npz_path)
        return (
            dry_run_payload(config, npz_path=str(npz_path), valid_path=valid_path)
            if dry_run
            else train_hybrid_tiny(config, npz_path=str(npz_path), valid_path=valid_path)
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        payload = run_with_optional_synthetic_npz(
            config,
            dry_run=args.dry_run or args.dry_run_json,
        )
    except Exception as exc:
        payload = {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "config": asdict(config),
            "training_optimizer": safe_training_optimizer_payload(config),
            "training_loss": safe_training_loss_payload(config),
            "compile": config.compile,
            "device": device_info(),
        }
        payload["compile_plan"] = compile_payload(config, payload["device"])
        payload["compile_enabled"] = payload["compile_plan"]["enabled"]
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
