#!/usr/bin/env python3
"""M0.4 local MLX bf16 training-step receipt.

This is a correctness smoke for the local MLX training plumbing. It intentionally
uses the existing tiny hybrid model path until the full local_gb10_quarter
grad-checkpoint target-parquet gate is captured.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

from cppmega_mlx.data.batch import ensure_lm_batch  # noqa: E402
from cppmega_mlx.data.packing import mlx_document_boundary_mask  # noqa: E402
from cppmega_mlx.nn.attention import (  # noqa: E402
    CausalSelfAttention,
    sparse_mla_fp8_route_enabled,
)
from cppmega_mlx.data.parquet_dataset import (  # noqa: E402
    MultiShardTokenParquetDataset,
    TokenParquetDataset,
)
from cppmega_mlx.runtime.memory import (  # noqa: E402
    DEFAULT_METAL_RATIO,
    DEFAULT_WIRED_RATIO,
    maybe_clear_cache_after_step,
    memory_limit_api_status,
)
from cppmega_mlx.runtime.kernel_policy import KernelPath, selected_path  # noqa: E402
from cppmega_mlx.runtime.path_c_fusion import (  # noqa: E402
    CompiledPathCRegion,
    PathCFusionMode,
    build_path_c_model_regions_from_model,
    compile_path_c_region,
    mark_path_c_schedule_template_for_region,
    selected_path_c_fusion_mode,
    tilelang_single_entry_lowerer,
)
from cppmega_mlx.runtime.path_c_physical_abi import (  # noqa: E402
    PathCLogicalBufferOwner,
    compose_path_c_logical_buffer_owner,
    make_physical_abi_bank_owner,
    path_c_kernel_buffer_order,
    physical_abi_bank_specs,
    plan_physical_abi_runtime_bridge,
    validate_physical_abi_runtime_bindings,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (  # noqa: E402
    DESCRIPTOR_EXECUTION_STAGE_ALL,
    DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
    DESCRIPTOR_EXECUTION_STAGE_FORWARD,
    DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
    DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
    PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR,
    make_path_c_descriptor_schedule_template,
    make_path_c_descriptor_stage_schedule_template,
    merge_path_c_physical_abi_for_prim_funcs,
    path_c_descriptor_stage_prim_funcs,
    path_c_fusion_schedule_spec,
    plan_path_c_descriptor_stage_groups,
    plan_path_c_direct_fusion_chain_for_region,
    plan_path_c_direct_fusion_chains_for_model,
    plan_path_c_fusion_schedule_for_region,
)
from cppmega_mlx.models.hybrid_lm import PathCActivationBufferCapture  # noqa: E402
from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    local_gb10_quarter,
    local_gb10_quarter_profile,
)
from cppmega_mlx.recipes.pattern import expand_nam_pattern  # noqa: E402
from cppmega_mlx.training.compiled import (  # noqa: E402
    CompiledPretrainingStep,
    PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS,
    PathCFusedPlusEagerTrainingRuntime,
    PathCFusedTrainBlockCallableArtifact,
    PathCFusedTrainBlockTrainingRuntime,
    PathCGeneratedStageTrainBlockCallableArtifact,
    PathCGradientBufferCapture,
)
from cppmega_mlx.training.cut_cross_entropy import (  # noqa: E402
    DEFAULT_CHUNK_ROWS,
    linear_cross_entropy,
)
from cppmega_mlx.training.loss import next_token_cut_cross_entropy  # noqa: E402
from cppmega_mlx.training.optimizers import (  # noqa: E402
    ADAM8BIT_CLASS,
    ADAM8BIT_SOURCE,
    ADAMW_BASE_CLASS,
    ADAMW_FP32_MOMENTS_CLASS,
    ADAMW_FP32_MOMENTS_SOURCE,
    LION8BIT_CLASS,
    LION8BIT_SOURCE,
    MUON_ADAMW_MULTI_CLASS,
    MUON_ADAMW_MULTI_SOURCE,
    MUON_QUANTIZED_MOMENTUM_SCHEMES,
    collect_adamw_moment_dtypes,
    dtype_name,
    make_adam8bit,
    make_adamw,
    make_lion,
    make_lion8bit,
    make_muon,
)
from scripts.train_hybrid_tiny import (  # noqa: E402
    DTYPES,
    TrainHybridTinyConfig,
    compile_payload,
    dataset_payload,
    device_info,
    dry_run_payload,
    memory_limit_payload as train_memory_limit_payload,
    parameter_count,
    parse_csv_ints,
    route_backend_payload,
    train_hybrid_tiny,
    validate_dataset_for_training,
    validate_side_channel_contract,
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
CACHE_LIMIT_ENV = "CPPMEGA_MLX_CACHE_LIMIT_BYTES"
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
FP8_PATH_B_DTYPE = "fp8_path_b"
FP8_PATH_C_DTYPE = "fp8_path_c"
FP8_PATH_B_E2E_TRAINING_STATUS = "m04_path_b_fp8_reference_baseline_available"
FP8_PATH_C_ROUTE_BLOCKER_TYPE = "fp8_path_c_training_route_unavailable"
FP8_PATH_C_KERNEL_SURFACE_STATUS = "prepared_buffer_path_c_available"
FP8_PATH_C_E2E_TRAINING_STATUS = "m04_path_c_training_route_available"
FP8_PATH_C_SPLIT_TRAINING_STATUS = "m04_path_c_split_training_route_available"
FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS = (
    "model_owned_physical_abi_banks_missing"
)
FP8_PATH_C_FUSED_TRAIN_BLOCK_ARTIFACT_MISSING_STATUS = (
    "fused_train_block_artifact_missing"
)
FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS = (
    "fused_train_block_training_runtime_missing"
)
FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_INCOMPLETE_STATUS = (
    "fused_train_block_training_runtime_incomplete"
)
FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_NOT_CRITICAL_STATUS = (
    "fused_train_block_training_runtime_not_on_critical_path"
)
FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_HIDDEN_ALLOCATION_STATUS = (
    "fused_train_block_training_runtime_hidden_allocation"
)
FP8_PATH_C_DIRECT_CHAIN_LOGICAL_BUFFERS_MISSING_STATUS = (
    "direct_fusion_chain_logical_buffers_missing"
)
FP8_PATH_C_DIRECT_CHAIN_ARTIFACTS_MISSING_STATUS = (
    "direct_fusion_chain_artifacts_missing"
)
FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_MISSING_STATUS = (
    "direct_fusion_chain_training_runtime_missing"
)
FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS = (
    "direct_fusion_chain_training_runtime_incomplete"
)
FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_NOT_CRITICAL_STATUS = (
    "direct_fusion_chain_training_runtime_not_on_critical_path"
)
FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_HIDDEN_ALLOCATION_STATUS = (
    "direct_fusion_chain_training_runtime_hidden_allocation"
)
PATH_C_DIRECT_FUSION_TRAINING_RUNTIME_CONTRACT = (
    "path_c_direct_fusion_training_runtime_v1"
)
PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT = (
    "path_c_direct_fusion_value_and_grad_v1"
)
PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT = (
    "path_c_fused_train_block_training_runtime_v1"
)
PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT = (
    "path_c_fused_train_block_value_and_grad_v1"
)
PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT = "path_c_loss_cotangent_bridge_v1"
PATH_C_FUSION_COMPILE_RECEIPT_ENV = "CPPMEGA_PATH_C_FUSION_COMPILE_RECEIPT"
PATH_C_FUSION_COMPILE_RECEIPT_PATH = (
    ROOT / "reports" / "path_c_fusion_compile_receipt.json"
)
PATH_C_FUSION_MATRIX_PROFILE_RECEIPT_ENV = (
    "CPPMEGA_PATH_C_FUSION_MATRIX_PROFILE_RECEIPT"
)
PATH_C_FUSION_MATRIX_PROFILE_RECEIPT_PATH = (
    ROOT / "reports" / "path_c_fusion_matrix_profile_receipt.json"
)
FP8_PATH_C_BRIDGE_TARGET = "native_mlx_tvm_ffi_graph_bridge"
FP8_PATH_C_BRIDGE_STATUS = "m04_wired_for_native_tvm_ffi_graph_outputs"
FP8_PATH_C_CARRIER_DTYPE = "bfloat16"
FP8_PATH_C_NATIVE_PRODUCER_STATUS = "attention_sparse_mla_fp8_producer_wired"
FP8_PATH_C_PRODUCER_MISSING_STATUS = "producer_missing"
FP8_PATH_C_PRODUCER_UNOBSERVED_STATUS = "producer_unobserved"
FP8_PATH_C_PRODUCER_OWNER = (
    "cppmega_mlx.nn.attention.CausalSelfAttention.prepare_sparse_mla_fp8"
)
FP8_PATH_C_PRODUCER_STAGE = "attention_qkv_projection"
FP8_PATH_C_PRODUCER_QUANTIZATION = "producer_owned_single_pass_metal_fp8_quant"
FP8_PATH_C_REQUIRED_PREPARED_BUFFERS = (
    "q_fp8",
    "q_scale",
    "kv_fp8",
    "kv_scale",
)
FP8_PATH_C_KERNEL_POLICY_ENV = {
    "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
    "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
    "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
}
FP8_PATH_B_KERNEL_POLICY_ENV = {
    "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_b",
}
SPARSE_MLA_FP8_ROUTE_ENV = "CPPMEGA_SPARSE_MLA_FP8_ROUTE"
SPARSE_MLA_FP8_BWD_ENV = "CPPMEGA_SPARSE_MLA_FP8_BWD"
MAMBA3_PATH_C_BWD_ENV = "CPPMEGA_MAMBA3_PATH_C_BWD"
FP8_PATH_C_RUNTIME_ENV: dict[str, str] = {
    SPARSE_MLA_FP8_ROUTE_ENV: "path_c",
    SPARSE_MLA_FP8_BWD_ENV: "path_c",
    MAMBA3_PATH_C_BWD_ENV: "path_b",
}
FP8_PATH_B_RUNTIME_ENV: dict[str, str] = {SPARSE_MLA_FP8_ROUTE_ENV: "path_b"}
FP8_PATH_C_SPLIT_GRAD_UPDATE_EVAL_REASON = (
    "Path C runs custom TileLang VJP nodes inside the same eager train step as "
    "optimizer.update; splitting the eval at the gradient/update boundary keeps "
    "backward activations out of the optimizer update peak."
)
FP8_PATH_C_ROUTE_BLOCKER_REASON = (
    "FP8 Path C has an m04 route for prepared Sparse-MLA Path C model ops, "
    "Mamba3 TileLang Path C selective scan, and M2RNN TileLang Path C recurrence. "
    "HybridTinyLM DSA A-layers now create prepared q_fp8/q_scale/kv_fp8/kv_scale "
    "tensors before Sparse-MLA Path C and the backward path scatters into final "
    "owner buffers. Remaining FP8 ownership work is parameter/weight producer "
    "coverage without hidden large tensor staging."
)
OPTIMIZER_CHOICES = (
    "adamw",
    "muon_adamw",
    "muon",
    "nam56r",
    "lion",
    "adam8bit",
    "lion8bit",
    "int8",
)
LION_FP32_MOMENTS_CLASS = "cppmega_mlx.training.optimizers.LionFP32Moments"
LION_FP32_MOMENTS_SOURCE = "cppmega_mlx.training.optimizers.make_lion"
MUON_INT8_SOURCE = "cppmega_mlx.training.optimizers.make_muon(int8_state)"
DEFAULT_SMOKE_LR = 1e-3
DEFAULT_LOCAL_GB10_QUARTER_LR = 1e-4
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
MATRIX_DTYPE_ROUTES = ("bf16", "fp8_path_b", "fp8_path_c", "int8")
MATRIX_OPTIMIZERS = ("adamw", "muon", "muon_adamw", "lion", "lion8bit", "adam8bit")
MATRIX_PROFILE_ROW_CHECKS = (
    "row_status_ok",
    "path_b_baseline_clean",
    "path_c_default_gate_passed",
    "path_c_peak_memory_non_regression",
    "path_c_warm_cache_hit_observed",
    "path_c_cold_cache_miss_observed",
    "profiling_trace_captured",
)
MATRIX_STEPS = 20
MATRIX_ACCEPTANCE_STEPS = 100
MATRIX_SMOKE_STEPS = 1
MATRIX_BATCH_SIZE = 1
MATRIX_SEQ_LEN = 4096
MATRIX_LR = "1e-4"
MATRIX_QUANT_SCHEME = "dynamic_int8_v1"
MATRIX_BASELINE_TOKENS_PER_SECOND = 900.0
MATRIX_BASELINE_RECEIPTS = (
    {
        "receipt": "bench/baselines/m04_optimizer_matrix/lion8bit_sym_lr1e-4.json",
        "case_id": "lion8bit_sym_lr1e-4",
        "optimizer": "lion8bit",
        "quant_scheme": "symmetric_int8_v1",
        "tokens_per_second": 900.6977464886402,
        "loss_decreased": True,
    },
    {
        "receipt": "bench/baselines/m04_optimizer_matrix/adam8bit_sym_lr1e-4.json",
        "case_id": "adam8bit_sym_lr1e-4",
        "optimizer": "adam8bit",
        "quant_scheme": "symmetric_int8_v1",
        "tokens_per_second": 894.2881681665949,
        "loss_decreased": False,
    },
    {
        "receipt": "bench/baselines/m04_optimizer_matrix/adam8bit_dyn_lr1e-4.json",
        "case_id": "adam8bit_dyn_lr1e-4",
        "optimizer": "adam8bit",
        "quant_scheme": "dynamic_int8_v1",
        "tokens_per_second": 890.0726520621305,
        "loss_decreased": True,
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
        "--data-shard",
        dest="data_shards",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional parquet shard path for deterministic sequential streaming. "
            "Repeat to pass the full corpus order."
        ),
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
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16", FP8_PATH_B_DTYPE, FP8_PATH_C_DTYPE),
        default="bfloat16",
        help=(
            "Training dtype/precision route. fp8_path_b enables the explicit "
            "non-Path-C FP8 reference baseline; fp8_path_c enables existing "
            "Path C model ops with a bf16 carrier."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "AdamW learning rate. Defaults to 1e-3 for tiny smoke routes and "
            "1e-4 for local_gb10_quarter unless set explicitly."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="adamw",
        help=(
            "Optimizer for the real local_gb10_quarter route. AdamW remains "
            "the default M0.4 acceptance optimizer; non-AdamW choices are "
            "recorded as optimizer-matrix variants."
        ),
    )
    parser.add_argument(
        "--optimizer-quant-scheme",
        choices=MUON_QUANTIZED_MOMENTUM_SCHEMES,
        default="dynamic_int8_v1",
        help=(
            "Blockwise int8 codec for adam8bit, lion8bit, and int8 "
            "optimizer variants. The default uses the bitsandbytes-style "
            "dynamic LUT; pass symmetric_int8_v1 for the older local codec."
        ),
    )
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--vocab-size", type=int, default=131_072)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--pattern", default="M")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--dsa-a-layer-ranks",
        default="",
        help=(
            "Comma-separated zero-based A-layer ranks that should own DSA/"
            "Sparse-MLA prepared FP8 producers."
        ),
    )
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
        "--cce-chunk-rows",
        type=int,
        default=TrainHybridTinyConfig.cce_chunk_rows,
        help=(
            "Rows per chunk for the MLX cut-cross-entropy loss. The value is "
            "recorded in receipts because it is a performance/memory knob."
        ),
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
        "--cache-limit-bytes",
        type=int,
        default=None,
        help=(
            "Set the MLX allocator cache limit before model allocation. "
            "Unset keeps MLX defaults except Path C local_gb10_quarter runs, "
            "which default to 0 to avoid retained IOAccelerator cache pressure. "
            f"Override with {CACHE_LIMIT_ENV} or this flag."
        ),
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
    parser.add_argument(
        "--profile-hold-seconds",
        type=float,
        default=0.0,
        help=(
            "Sleep after writing the receipt so external memory profilers can "
            "inspect the live MLX allocator/cache state. Normal runs leave this at 0."
        ),
    )
    parser.add_argument(
        "--profile-path-c-direct-chain-runtime",
        action="store_true",
        help=(
            "After an eager fp8_path_c local_gb10_quarter step captures the "
            "direct-chain logical buffers, compile the direct-chain runtime and "
            "execute its non-eager value_and_grad bridge as a profiling probe. "
            "This is post-step evidence only and does not mark Path C as the "
            "training critical path."
        ),
    )
    parser.add_argument(
        "--use-path-c-direct-chain-runtime",
        action="store_true",
        help=(
            "Opt in to the dynamic pre-step direct-chain value_and_grad runtime "
            "for fp8_path_c local_gb10_quarter training. This installs the "
            "runtime on the actual m04 train-step critical path, requires "
            "compile=False, and remains off by default."
        ),
    )
    parser.add_argument(
        "--use-path-c-fused-train-block-runtime",
        action="store_true",
        help=(
            "Opt in to the generated fused train-block value_and_grad runtime "
            "for fp8_path_c local_gb10_quarter training. This compiles the "
            "descriptor-generated train-block artifact on the actual critical "
            "path; recurrent backward routes use launcher-chunk generated "
            "stages instead of the unsafe monolithic grid-chunk shader."
        ),
    )
    return parser


def config_from_args(
    args: argparse.Namespace, *, data_path: Path
) -> TrainHybridTinyConfig:
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
        learning_rate=learning_rate_from_args(args),
        weight_decay=args.weight_decay,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        pattern=args.pattern,
        depth=args.depth,
        dsa_a_layer_ranks=parse_csv_ints(args.dsa_a_layer_ranks),
        num_attention_heads=args.num_attention_heads,
        mamba_expand=args.mamba_expand,
        mamba_head_dim=args.mamba_head_dim,
        mamba_state_dim=args.mamba_state_dim,
        mamba_groups=args.mamba_groups,
        mamba_chunk_size=args.mamba_chunk_size,
        grad_checkpoint=args.grad_checkpoint,
        cce_chunk_rows=args.cce_chunk_rows,
        token_key=args.token_key,
        memory_limit_total_bytes=args.memory_limit_total_bytes,
        memory_limit_wired_ratio=args.memory_limit_wired_ratio,
        memory_limit_metal_ratio=args.memory_limit_metal_ratio,
        apply_memory_limit_plan=args.apply_memory_limit_plan,
        clear_cache_every_steps=args.clear_cache_every_steps,
    )


def parquet_shard_paths_from_args(
    args: argparse.Namespace,
    *,
    data_path: Path,
) -> tuple[Path, ...]:
    paths = [data_path, *(Path(path) for path in getattr(args, "data_shards", ()) or ())]
    return tuple(dict.fromkeys(paths))


def training_dataset_from_args(
    args: argparse.Namespace,
    *,
    config: TrainHybridTinyConfig,
    data_path: Path,
    loop: bool,
) -> TokenParquetDataset | MultiShardTokenParquetDataset:
    shard_paths = parquet_shard_paths_from_args(args, data_path=data_path)
    if len(shard_paths) > 1:
        if config.data_format != "parquet":
            raise ValueError("multi-shard streaming requires --data-format parquet")
        return MultiShardTokenParquetDataset(
            shard_paths,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            token_key=config.token_key,
            shuffle=config.shuffle,
            seed=config.seed,
            loop=loop,
        )
    return TokenParquetDataset(
        data_path,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        token_key=config.token_key,
        shuffle=config.shuffle,
        seed=config.seed,
        loop=loop,
    )


def learning_rate_from_args(args: argparse.Namespace) -> float:
    if args.lr is not None:
        return float(args.lr)
    if args.model_profile == REQUIRED_MODEL_PROFILE:
        return DEFAULT_LOCAL_GB10_QUARTER_LR
    return DEFAULT_SMOKE_LR


def optimizer_key_from_args(args: argparse.Namespace) -> str:
    key = str(getattr(args, "optimizer", "adamw")).strip().lower()
    if key == "muon":
        return "muon_adamw"
    if key == "nam56r":
        return "muon_adamw"
    return key


def optimizer_variant_payload(args: argparse.Namespace) -> dict[str, Any]:
    requested = str(getattr(args, "optimizer", "adamw")).strip().lower()
    key = optimizer_key_from_args(args)
    return {
        "requested": requested,
        "key": key,
        "quant_scheme": getattr(args, "optimizer_quant_scheme", None),
        "source": "cli" if requested != "adamw" else "default",
    }


def fp8_path_c_route_requested(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> bool:
    return str(getattr(args, "dtype", "")).strip().lower() == FP8_PATH_C_DTYPE


def fp8_path_b_route_requested(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> bool:
    return str(getattr(args, "dtype", "")).strip().lower() == FP8_PATH_B_DTYPE


def fp8_training_route_requested(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> bool:
    return fp8_path_b_route_requested(args) or fp8_path_c_route_requested(args)


def path_c_training_route_requested(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> bool:
    return fp8_path_c_route_requested(args) or path_c_kernel_policy_requested()


def path_c_direct_chain_capture_requested(
    args: argparse.Namespace | TrainHybridTinyConfig,
    *,
    compile_enabled: bool,
) -> bool:
    """Return whether direct-chain activation/gradient evidence must be retained."""

    return (
        fp8_path_c_route_requested(args)
        and not compile_enabled
        and (
            bool(getattr(args, "use_path_c_direct_chain_runtime", False))
            or bool(getattr(args, "profile_path_c_direct_chain_runtime", False))
        )
    )


def path_c_kernel_policy_requested() -> bool:
    path_c_values = {"path_c", "c"}
    for env_name in (
        "CPPMEGA_KERNEL_PATH",
        "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO",
        "CPPMEGA_KERNEL_PATH__M2RNN",
        "CPPMEGA_KERNEL_PATH__SPARSE_MLA",
    ):
        if os.environ.get(env_name, "").strip().lower() in path_c_values:
            return True
    return False


def path_c_training_sequence_length(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> int | None:
    """Return the decoder input length used by the training loss graph."""

    seq_len = int(getattr(args, "seq_len", 0) or 0)
    if seq_len <= 1:
        return None
    return seq_len - 1


def path_c_batch_sequence_length(
    batch: Mapping[str, mx.array] | mx.array | None,
) -> int | None:
    if batch is None:
        return None
    lm_batch = ensure_lm_batch(batch)
    inputs = lm_batch.inputs
    if not isinstance(inputs, mx.array) or inputs.ndim != 2:
        return None
    return int(inputs.shape[1])


def _validate_cache_limit_bytes(limit: int, *, source: str) -> int:
    if not isinstance(limit, int):
        raise TypeError(f"cache limit from {source} must be an integer byte count")
    if limit < 0:
        raise ValueError(f"cache limit from {source} must be >= 0")
    return limit


def resolved_cache_limit_bytes(args: argparse.Namespace) -> tuple[int | None, str]:
    explicit = getattr(args, "cache_limit_bytes", None)
    if explicit is not None:
        return _validate_cache_limit_bytes(explicit, source="cli"), "cli"

    env_value = os.environ.get(CACHE_LIMIT_ENV)
    if env_value is not None and env_value.strip() != "":
        try:
            parsed = int(env_value)
        except ValueError as exc:
            raise ValueError(f"{CACHE_LIMIT_ENV} must be an integer byte count") from exc
        return _validate_cache_limit_bytes(parsed, source=CACHE_LIMIT_ENV), CACHE_LIMIT_ENV

    if (
        str(getattr(args, "model_profile", "")).strip() == REQUIRED_MODEL_PROFILE
        and path_c_kernel_policy_requested()
    ):
        return 0, "path_c_default"
    return None, "mlx_default"


def apply_cache_limit_payload(
    args: argparse.Namespace,
    *,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    limit, source = resolved_cache_limit_bytes(args)
    payload: dict[str, Any] = {
        "configured": limit is not None,
        "applied": False,
        "limit_bytes": limit,
        "source": source,
        "api_path": None,
        "previous_limit_bytes": None,
    }
    if limit is None:
        return payload

    mx_backend = mx if mx_module is None else mx_module
    set_cache_limit = getattr(mx_backend, "set_cache_limit", None)
    if not callable(set_cache_limit):
        raise RuntimeError("mlx.core.set_cache_limit is unavailable")

    previous = int(set_cache_limit(limit))
    payload.update(
        {
            "applied": True,
            "api_path": "mx.set_cache_limit",
            "previous_limit_bytes": previous,
        }
    )
    return payload


def carrier_dtype_for_acceptance(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> str:
    if fp8_training_route_requested(args):
        return FP8_PATH_C_CARRIER_DTYPE
    return str(getattr(args, "dtype", ""))


def _coerce_dsa_a_layer_ranks(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return parse_csv_ints(value)
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    return tuple(int(item) for item in value)


def sparse_mla_fp8_producer_payload(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> dict[str, Any]:
    requested = fp8_path_c_route_requested(args)
    if str(getattr(args, "model_profile", "")).strip() == REQUIRED_MODEL_PROFILE:
        profile = local_gb10_quarter_profile()
        pattern = profile.pattern
        depth = profile.depth
        dsa_ranks = profile.dsa_a_layer_ranks
        route_source = "cppmega_mlx.recipes.model_factory.local_gb10_quarter"
    else:
        pattern = str(getattr(args, "pattern", ""))
        depth = int(getattr(args, "depth", 0) or 0)
        route_source = "scripts.m04_train_step.cli"
        try:
            dsa_ranks = _coerce_dsa_a_layer_ranks(
                getattr(args, "dsa_a_layer_ranks", ())
            )
        except Exception as exc:
            return {
                "requested": requested,
                "configured": False,
                "status": "producer_config_invalid"
                if requested
                else "not_requested",
                "owner": FP8_PATH_C_PRODUCER_OWNER,
                "route_source": route_source,
                "pattern": pattern,
                "depth": depth,
                "dsa_a_layer_ranks": [],
                "dsa_layer_numbers": [],
                "required_prepared_buffers": list(FP8_PATH_C_REQUIRED_PREPARED_BUFFERS),
                "prepared_buffers_configured": False,
                "producer_stage": FP8_PATH_C_PRODUCER_STAGE,
                "producer_quantization": FP8_PATH_C_PRODUCER_QUANTIZATION,
                "reason": (
                    "producer_config_invalid: unable to parse "
                    f"dsa_a_layer_ranks ({exc})"
                )
                if requested
                else None,
                "large_tensor_staging_allowed": False,
                "hidden_wrapper_quantization_allowed": False,
                "kernel_boundary_quantization_allowed": False,
            }

    try:
        expanded = expand_nam_pattern(
            pattern,
            depth,
            dsa_a_layer_ranks=dsa_ranks,
        )
        dsa_layer_numbers = list(expanded.dsa_layer_numbers)
        attention_layer_numbers = list(expanded.a_layer_numbers)
    except Exception as exc:
        return {
            "requested": requested,
            "configured": False,
            "status": "producer_config_invalid" if requested else "not_requested",
            "owner": FP8_PATH_C_PRODUCER_OWNER,
            "route_source": route_source,
            "pattern": pattern,
            "depth": depth,
            "dsa_a_layer_ranks": list(dsa_ranks),
            "dsa_layer_numbers": [],
            "attention_layer_numbers": [],
            "required_prepared_buffers": list(FP8_PATH_C_REQUIRED_PREPARED_BUFFERS),
            "prepared_buffers_configured": False,
            "producer_stage": FP8_PATH_C_PRODUCER_STAGE,
            "producer_quantization": FP8_PATH_C_PRODUCER_QUANTIZATION,
            "reason": (
                "producer_config_invalid: unable to expand NAM route "
                f"for Sparse-MLA FP8 producers ({exc})"
            )
            if requested
            else None,
            "large_tensor_staging_allowed": False,
            "hidden_wrapper_quantization_allowed": False,
            "kernel_boundary_quantization_allowed": False,
        }

    configured = bool(dsa_layer_numbers)
    if not requested:
        status = "not_requested"
        reason = None
    elif configured:
        status = FP8_PATH_C_NATIVE_PRODUCER_STATUS
        reason = None
    else:
        status = FP8_PATH_C_PRODUCER_MISSING_STATUS
        reason = (
            "producer_missing: fp8_path_c requested Sparse-MLA Path C, but "
            "the current model graph has no DSA A-layer to own prepared "
            "q_fp8/q_scale/kv_fp8/kv_scale buffers"
        )

    return {
        "requested": requested,
        "configured": configured,
        "status": status,
        "owner": FP8_PATH_C_PRODUCER_OWNER,
        "route_source": route_source,
        "pattern": pattern,
        "depth": depth,
        "attention_layer_numbers": attention_layer_numbers,
        "dsa_a_layer_ranks": list(dsa_ranks),
        "dsa_layer_numbers": dsa_layer_numbers,
        "required_prepared_buffers": list(FP8_PATH_C_REQUIRED_PREPARED_BUFFERS),
        "prepared_buffers_configured": configured,
        "producer_stage": FP8_PATH_C_PRODUCER_STAGE,
        "producer_quantization": FP8_PATH_C_PRODUCER_QUANTIZATION,
        "reason": reason,
        "large_tensor_staging_allowed": False,
        "hidden_wrapper_quantization_allowed": False,
        "kernel_boundary_quantization_allowed": False,
        "design_refs": [
            "vLLM BaseKVCacheMethod owns q/k/v scale attributes on attention layers",
            "MLX SDPA vector kernels consume q/k/v pointers and scale inputs directly",
        ],
    }


def sparse_mla_fp8_producer_configured(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> bool:
    return bool(sparse_mla_fp8_producer_payload(args)["configured"])


def fp8_path_c_producer_gate_payload(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> dict[str, Any]:
    producer = sparse_mla_fp8_producer_payload(args)
    requested = fp8_path_c_route_requested(args)
    configured = bool(producer["configured"])
    required = bool(requested)
    status = str(producer["status"]) if required else "not_requested"
    return {
        "name": "fp8_path_c_sparse_mla_producer",
        "required": required,
        "ok": (not required) or configured,
        "status": status,
        "configured": configured,
        "fail_closed": bool(required and not configured),
        "reason": producer["reason"] if required and not configured else None,
        "owner": producer["owner"],
        "route_source": producer["route_source"],
        "required_prepared_buffers": list(FP8_PATH_C_REQUIRED_PREPARED_BUFFERS),
        "prepared_buffers_configured": bool(producer["prepared_buffers_configured"]),
        "producer_stage": producer["producer_stage"],
        "producer_quantization": producer["producer_quantization"],
        "producer": producer,
        "fallback_to_path_b_allowed": False if required else None,
        "large_tensor_staging_allowed": False,
        "hidden_wrapper_quantization_allowed": False,
        "kernel_boundary_quantization_allowed": False,
        "receipt_field_paths": [
            "workload.precision_route.sparse_mla_fp8_producer",
            "training.fp8_path_c_training_route.sparse_mla_fp8_producer",
            "regression_report.route_dispatch.fp8_sparse_mla_producer",
            "regression_report.fp8_path_c_producer_gate",
        ],
    }


def _tilelang_dev_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    env_root = os.environ.get("TILELANG_DEV_BUILD_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend(
        [
            ROOT.parent / "tl_apache_tvm_swap",
            Path("/private/tmp/tl_apache_tvm_swap"),
            Path.home() / "sources" / "tl_apache_tvm_swap",
        ]
    )

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _tilelang_source_and_build_root(root: Path) -> tuple[Path, Path] | None:
    if (root / "tilelang").exists():
        return root, root / "build"
    if root.name == "build" and (root.parent / "tilelang").exists():
        return root.parent, root
    return None


def _prepend_env_path(name: str, path: Path) -> None:
    path_str = str(path)
    values = [
        value
        for value in os.environ.get(name, "").split(os.pathsep)
        if value
    ]
    if path_str not in values:
        os.environ[name] = os.pathsep.join([path_str, *values])


def ensure_tilelang_dev_env_for_path_c() -> None:
    for root in _tilelang_dev_roots():
        normalized = _tilelang_source_and_build_root(root)
        if normalized is None:
            continue
        source_root, build_root = normalized
        lib_dir = build_root / "lib"
        tvm_dir = build_root / "tvm"
        if not lib_dir.exists() or not tvm_dir.exists():
            continue
        os.environ["TILELANG_DEV_BUILD_ROOT"] = str(build_root)
        os.environ.setdefault("TVM_LIBRARY_PATH", str(lib_dir))
        _prepend_env_path("DYLD_LIBRARY_PATH", lib_dir)
        for path in (source_root, source_root / "3rdparty" / "tvm" / "python"):
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)
        return


@contextmanager
def fp8_path_c_kernel_policy(
    args: argparse.Namespace | TrainHybridTinyConfig,
    *,
    ensure_dev_env: Callable[[], None] | None = None,
):
    if not fp8_path_c_route_requested(args):
        yield
        return

    if ensure_dev_env is None:
        ensure_dev_env = ensure_tilelang_dev_env_for_path_c
    ensure_dev_env()
    policy_env = {**FP8_PATH_C_KERNEL_POLICY_ENV, **FP8_PATH_C_RUNTIME_ENV}
    previous = {key: os.environ.get(key) for key in policy_env}
    os.environ.update(FP8_PATH_C_KERNEL_POLICY_ENV)
    os.environ.update(FP8_PATH_C_RUNTIME_ENV)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def fp8_path_b_kernel_policy(
    args: argparse.Namespace | TrainHybridTinyConfig,
):
    if not fp8_path_b_route_requested(args):
        yield
        return

    policy_env = {**FP8_PATH_B_KERNEL_POLICY_ENV, **FP8_PATH_B_RUNTIME_ENV}
    previous = {key: os.environ.get(key) for key in policy_env}
    os.environ.update(FP8_PATH_B_KERNEL_POLICY_ENV)
    os.environ.update(FP8_PATH_B_RUNTIME_ENV)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def fp8_path_c_stdio_suppressed(
    args: argparse.Namespace | TrainHybridTinyConfig,
):
    if not fp8_path_c_route_requested(args):
        yield
        return

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def precision_route_payload(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> dict[str, Any]:
    if fp8_path_b_route_requested(args):
        return {
            "requested": FP8_PATH_B_DTYPE,
            "kind": "fp8_path_b_reference_baseline",
            "status": FP8_PATH_B_E2E_TRAINING_STATUS,
            "carrier_dtype": FP8_PATH_C_CARRIER_DTYPE,
            "kernel_policy_env": dict(FP8_PATH_B_KERNEL_POLICY_ENV),
            "baseline_surface": "sparse_mla_fp8_apply",
            "baseline_module": "cppmega_mlx.nn._tilelang.sparse_mla_fp8",
            "path_c_used": False,
            "zero_copy_required": False,
            "large_tensor_staging_allowed": False,
            "hidden_wrapper_quantization_allowed": False,
            "kernel_boundary_quantization_allowed": False,
        }
    if fp8_path_c_route_requested(args):
        producer = sparse_mla_fp8_producer_payload(args)
        producer_configured = bool(producer["configured"])
        return {
            "requested": FP8_PATH_C_DTYPE,
            "kind": "fp8_path_c",
            "status": (
                FP8_PATH_C_E2E_TRAINING_STATUS
                if producer_configured
                else producer["status"]
            ),
            "blocker_type": None
            if producer_configured
            else str(producer["status"]),
            "carrier_dtype": FP8_PATH_C_CARRIER_DTYPE,
            "native_fp8_producer_status": producer["status"],
            "sparse_mla_fp8_producer": producer,
            "prepared_buffers_configured": bool(
                producer["prepared_buffers_configured"]
            ),
            "producer_stage": producer["producer_stage"],
            "producer_quantization": producer["producer_quantization"],
            "kernel_surface_status": FP8_PATH_C_KERNEL_SURFACE_STATUS,
            "kernel_surface_available": True,
            "full_end_to_end_training_available": producer_configured,
            "bridge_target": FP8_PATH_C_BRIDGE_TARGET,
            "bridge_status": FP8_PATH_C_BRIDGE_STATUS,
            "zero_copy_required": True,
            "large_tensor_staging_allowed": False,
            "hidden_wrapper_quantization_allowed": False,
            "kernel_boundary_quantization_allowed": False,
        }
    return {
        "requested": str(getattr(args, "dtype", "")),
        "kind": "native_mlx_dtype",
        "status": "available",
        "zero_copy_required": False,
        "large_tensor_staging_allowed": False,
    }


def path_c_direct_fusion_chain_training_runtime_contract_payload(
    *,
    training_runtime: Any | None,
    runtime_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate that a direct-chain runtime is wired into the training graph."""

    runtime_binding = runtime_binding or {}
    runtime_installed = training_runtime is not None
    forward = getattr(training_runtime, "forward", None)
    backward = getattr(training_runtime, "backward", None)
    vjp = getattr(training_runtime, "vjp", None)
    value_and_grad = getattr(training_runtime, "value_and_grad", None)
    forward_callable = callable(forward)
    backward_callable = callable(backward)
    vjp_callable = callable(vjp)
    value_and_grad_callable = callable(value_and_grad)
    reverse_callable = backward_callable or vjp_callable
    declared_contract = str(getattr(training_runtime, "contract", ""))
    contract_matches = (
        declared_contract == PATH_C_DIRECT_FUSION_TRAINING_RUNTIME_CONTRACT
    )
    declared_training_critical_path = bool(
        getattr(
            training_runtime,
            "training_critical_path",
            getattr(training_runtime, "critical_path", False),
        )
    )
    hidden_packing_performed = bool(
        getattr(training_runtime, "hidden_packing_performed", False)
    )
    no_hidden_allocation_policy = bool(
        getattr(training_runtime, "no_hidden_allocation_policy", True)
    )
    runtime_uses_direct_chain = bool(
        runtime_binding.get("runtime_uses_direct_fusion_chain")
    )
    graph_binding = _direct_chain_training_graph_binding_payload(training_runtime)
    graph_binding_ok = bool(graph_binding.get("status") == "ok")
    value_and_grad_contract = _direct_chain_value_and_grad_contract_payload(
        training_runtime
    )
    value_and_grad_contract_ok = bool(
        value_and_grad_contract.get("status") == "ok"
    )
    returns_full_model_grads = bool(
        value_and_grad_contract.get("returns_full_model_grads", False)
    )
    full_model_gradient_coverage = value_and_grad_contract.get(
        "full_model_gradient_coverage"
    )
    training_critical_path = bool(
        declared_training_critical_path
        and graph_binding_ok
        and value_and_grad_callable
        and value_and_grad_contract_ok
        and returns_full_model_grads
    )
    if not runtime_installed:
        status = FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_MISSING_STATUS
    elif not runtime_uses_direct_chain:
        status = str(
            runtime_binding.get(
                "status",
                FP8_PATH_C_DIRECT_CHAIN_LOGICAL_BUFFERS_MISSING_STATUS,
            )
        )
    elif (
        not contract_matches
        or not forward_callable
        or not reverse_callable
        or not value_and_grad_callable
        or not value_and_grad_contract_ok
        or not returns_full_model_grads
    ):
        status = FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS
    elif hidden_packing_performed or not no_hidden_allocation_policy:
        status = FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_HIDDEN_ALLOCATION_STATUS
    elif not training_critical_path:
        status = FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_NOT_CRITICAL_STATUS
    else:
        status = "ok"
    training_runtime_available = status == "ok"
    return {
        "contract": PATH_C_DIRECT_FUSION_TRAINING_RUNTIME_CONTRACT,
        "status": status,
        "training_runtime_available": training_runtime_available,
        "runtime_installed": runtime_installed,
        "declared_contract": declared_contract if runtime_installed else None,
        "contract_matches": contract_matches,
        "runtime_class": type(training_runtime).__name__ if runtime_installed else None,
        "runtime_owner": getattr(training_runtime, "owner_name", None)
        if runtime_installed
        else None,
        "forward_callable": forward_callable,
        "backward_callable": backward_callable,
        "vjp_callable": vjp_callable,
        "value_and_grad_callable": value_and_grad_callable,
        "value_and_grad_contract": value_and_grad_contract,
        "value_and_grad_contract_ok": value_and_grad_contract_ok,
        "returns_full_model_grads": returns_full_model_grads,
        "full_model_gradient_coverage": full_model_gradient_coverage,
        "training_critical_path_declared": declared_training_critical_path,
        "training_graph_bound": graph_binding_ok,
        "training_graph_binding": graph_binding,
        "training_critical_path_verified": training_critical_path,
        "critical_path_ready": training_runtime_available,
        "runtime_uses_direct_fusion_chain": runtime_uses_direct_chain,
        "hidden_packing_performed": hidden_packing_performed,
        "no_hidden_allocation_policy": no_hidden_allocation_policy,
        "reason": (
            "direct-chain forward, backward/vjp, and value_and_grad hooks are "
            "installed in the training graph and backed by callable artifacts "
            "plus caller-owned logical buffers"
            if training_runtime_available
            else "direct-chain artifacts or standalone dispatch are not enough; "
            "Path C needs explicit training-graph forward, backward/vjp, and "
            "value_and_grad hooks with caller-owned buffers, no hidden packing, "
            "and no delegation to eager loss_and_grad; m04 critical path also "
            "requires full-model gradients, not only selected-region gradients"
        ),
    }


def path_c_fused_train_block_training_runtime_contract_payload(
    *,
    training_runtime: Any | None,
    runtime_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate that a single fused train-block runtime owns training execution."""

    runtime_binding = runtime_binding or {}
    runtime_installed = training_runtime is not None
    forward = getattr(training_runtime, "forward", None)
    backward = getattr(training_runtime, "backward", None)
    vjp = getattr(training_runtime, "vjp", None)
    value_and_grad = getattr(training_runtime, "value_and_grad", None)
    forward_callable = callable(forward)
    backward_callable = callable(backward)
    vjp_callable = callable(vjp)
    value_and_grad_callable = callable(value_and_grad)
    reverse_callable = backward_callable or vjp_callable
    declared_contract = str(getattr(training_runtime, "contract", ""))
    contract_matches = (
        declared_contract == PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT
    )
    declared_training_critical_path = bool(
        getattr(
            training_runtime,
            "training_critical_path",
            getattr(training_runtime, "critical_path", False),
        )
    )
    hidden_packing_performed = bool(
        getattr(training_runtime, "hidden_packing_performed", False)
    )
    no_hidden_allocation_policy = bool(
        getattr(training_runtime, "no_hidden_allocation_policy", True)
    )
    runtime_uses_fused_train_block = bool(
        runtime_binding.get("runtime_uses_fused_train_block")
    )
    graph_binding = _fused_train_block_training_graph_binding_payload(
        training_runtime
    )
    graph_binding_ok = bool(graph_binding.get("status") == "ok")
    value_and_grad_contract = _fused_train_block_value_and_grad_contract_payload(
        training_runtime
    )
    value_and_grad_contract_ok = bool(
        value_and_grad_contract.get("status") == "ok"
    )
    returns_full_model_grads = bool(
        value_and_grad_contract.get("returns_full_model_grads", False)
    )
    full_model_gradient_coverage = value_and_grad_contract.get(
        "full_model_gradient_coverage"
    )
    training_critical_path = bool(
        declared_training_critical_path
        and graph_binding_ok
        and value_and_grad_callable
        and value_and_grad_contract_ok
        and returns_full_model_grads
    )
    if not runtime_installed:
        status = FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS
    elif not runtime_uses_fused_train_block:
        status = str(
            runtime_binding.get(
                "status",
                FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS,
            )
        )
    elif (
        not contract_matches
        or not forward_callable
        or not reverse_callable
        or not value_and_grad_callable
        or not value_and_grad_contract_ok
        or not returns_full_model_grads
    ):
        status = FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_INCOMPLETE_STATUS
    elif hidden_packing_performed or not no_hidden_allocation_policy:
        status = FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_HIDDEN_ALLOCATION_STATUS
    elif not training_critical_path:
        status = FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_NOT_CRITICAL_STATUS
    else:
        status = "ok"
    training_runtime_available = status == "ok"
    return {
        "contract": PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT,
        "status": status,
        "training_runtime_available": training_runtime_available,
        "runtime_installed": runtime_installed,
        "declared_contract": declared_contract if runtime_installed else None,
        "contract_matches": contract_matches,
        "runtime_class": type(training_runtime).__name__ if runtime_installed else None,
        "runtime_owner": getattr(training_runtime, "owner_name", None)
        if runtime_installed
        else None,
        "forward_callable": forward_callable,
        "backward_callable": backward_callable,
        "vjp_callable": vjp_callable,
        "value_and_grad_callable": value_and_grad_callable,
        "value_and_grad_contract": value_and_grad_contract,
        "value_and_grad_contract_ok": value_and_grad_contract_ok,
        "returns_full_model_grads": returns_full_model_grads,
        "full_model_gradient_coverage": full_model_gradient_coverage,
        "training_critical_path_declared": declared_training_critical_path,
        "training_graph_bound": graph_binding_ok,
        "training_graph_binding": graph_binding,
        "training_critical_path_verified": training_critical_path,
        "critical_path_ready": training_runtime_available,
        "runtime_uses_fused_train_block": runtime_uses_fused_train_block,
        "hidden_packing_performed": hidden_packing_performed,
        "no_hidden_allocation_policy": no_hidden_allocation_policy,
        "reason": (
            "single fused train-block value_and_grad is installed in the "
            "training graph and backed by callable generated artifact plus "
            "model-owned physical ABI banks"
            if training_runtime_available
            else "fused train-block bank/artifact binding is not enough; m04 "
            "needs an explicit training-graph value_and_grad runtime with "
            "full-model gradients, no hidden packing, and no delegation to "
            "eager loss_and_grad"
        ),
    }


def _direct_chain_training_graph_binding_payload(
    training_runtime: Any | None,
) -> dict[str, Any]:
    if training_runtime is None:
        return {
            "status": "missing",
            "owner": None,
            "uses_direct_chain_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
        }
    raw_binding = getattr(training_runtime, "training_graph_binding", None)
    if callable(raw_binding):
        raw_binding = raw_binding()
    if not isinstance(raw_binding, Mapping):
        return {
            "status": "missing",
            "owner": None,
            "uses_direct_chain_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
        }
    payload = dict(raw_binding)
    owner = str(payload.get("owner", ""))
    uses_runtime = bool(payload.get("uses_direct_chain_runtime"))
    uses_forward = bool(payload.get("uses_forward_hook"))
    uses_reverse = bool(payload.get("uses_backward_or_vjp_hook"))
    status = (
        "ok"
        if owner == "CompiledPretrainingStep"
        and uses_runtime
        and uses_forward
        and uses_reverse
        else "incomplete"
    )
    return {
        **payload,
        "status": status,
        "owner": owner or None,
        "uses_direct_chain_runtime": uses_runtime,
        "uses_forward_hook": uses_forward,
        "uses_backward_or_vjp_hook": uses_reverse,
    }


def _direct_chain_value_and_grad_contract_payload(
    training_runtime: Any | None,
) -> dict[str, Any]:
    if training_runtime is None:
        return {
            "status": "missing",
            "contract": PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT,
            "owner": None,
            "uses_direct_chain_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
            "returns_model_grads": False,
            "returns_full_model_grads": False,
            "loss_cotangent_bridge_ready": False,
            "model_gradient_tree_ready": False,
            "delegates_to_eager_loss_and_grad": True,
            "hidden_packing_performed": False,
        }
    raw_contract = getattr(training_runtime, "value_and_grad_contract", None)
    if callable(raw_contract):
        raw_contract = raw_contract()
    if not isinstance(raw_contract, Mapping):
        return {
            "status": "missing",
            "contract": PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT,
            "owner": None,
            "uses_direct_chain_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
            "returns_model_grads": False,
            "returns_full_model_grads": False,
            "loss_cotangent_bridge_ready": False,
            "model_gradient_tree_ready": False,
            "delegates_to_eager_loss_and_grad": True,
            "hidden_packing_performed": False,
        }
    payload = dict(raw_contract)
    owner = str(payload.get("owner", ""))
    contract = str(payload.get("contract", ""))
    uses_runtime = bool(payload.get("uses_direct_chain_runtime"))
    uses_forward = bool(payload.get("uses_forward_hook"))
    uses_reverse = bool(payload.get("uses_backward_or_vjp_hook"))
    returns_model_grads = bool(payload.get("returns_model_grads"))
    returns_full_model_grads = bool(payload.get("returns_full_model_grads", False))
    loss_cotangent_bridge_ready = bool(payload.get("loss_cotangent_bridge_ready"))
    model_gradient_tree_ready = bool(payload.get("model_gradient_tree_ready"))
    delegates_to_eager = bool(payload.get("delegates_to_eager_loss_and_grad", True))
    hidden_packing = bool(payload.get("hidden_packing_performed", False))
    status = (
        "ok"
        if contract == PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT
        and owner == "CompiledPretrainingStep"
        and uses_runtime
        and uses_forward
        and uses_reverse
        and returns_model_grads
        and loss_cotangent_bridge_ready
        and model_gradient_tree_ready
        and not delegates_to_eager
        and not hidden_packing
        else "incomplete"
    )
    return {
        **payload,
        "status": status,
        "contract": contract or PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT,
        "owner": owner or None,
        "uses_direct_chain_runtime": uses_runtime,
        "uses_forward_hook": uses_forward,
        "uses_backward_or_vjp_hook": uses_reverse,
        "returns_model_grads": returns_model_grads,
        "returns_full_model_grads": returns_full_model_grads,
        "loss_cotangent_bridge_ready": loss_cotangent_bridge_ready,
        "model_gradient_tree_ready": model_gradient_tree_ready,
        "delegates_to_eager_loss_and_grad": delegates_to_eager,
        "hidden_packing_performed": hidden_packing,
    }


def _fused_train_block_training_graph_binding_payload(
    training_runtime: Any | None,
) -> dict[str, Any]:
    if training_runtime is None:
        return {
            "status": "missing",
            "owner": None,
            "uses_fused_train_block_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
        }
    raw_binding = getattr(training_runtime, "training_graph_binding", None)
    if callable(raw_binding):
        raw_binding = raw_binding()
    if not isinstance(raw_binding, Mapping):
        return {
            "status": "missing",
            "owner": None,
            "uses_fused_train_block_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
        }
    payload = dict(raw_binding)
    owner = str(payload.get("owner", ""))
    uses_runtime = bool(payload.get("uses_fused_train_block_runtime"))
    uses_forward = bool(payload.get("uses_forward_hook"))
    uses_reverse = bool(payload.get("uses_backward_or_vjp_hook"))
    status = (
        "ok"
        if owner == "CompiledPretrainingStep"
        and uses_runtime
        and uses_forward
        and uses_reverse
        else "incomplete"
    )
    return {
        **payload,
        "status": status,
        "owner": owner or None,
        "uses_fused_train_block_runtime": uses_runtime,
        "uses_forward_hook": uses_forward,
        "uses_backward_or_vjp_hook": uses_reverse,
    }


def _fused_train_block_value_and_grad_contract_payload(
    training_runtime: Any | None,
) -> dict[str, Any]:
    if training_runtime is None:
        return {
            "status": "missing",
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": None,
            "uses_fused_train_block_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
            "returns_model_grads": False,
            "returns_full_model_grads": False,
            "loss_cotangent_bridge_ready": False,
            "model_gradient_tree_ready": False,
            "delegates_to_eager_loss_and_grad": True,
            "hidden_packing_performed": False,
        }
    raw_contract = getattr(training_runtime, "value_and_grad_contract", None)
    if callable(raw_contract):
        raw_contract = raw_contract()
    if not isinstance(raw_contract, Mapping):
        return {
            "status": "missing",
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": None,
            "uses_fused_train_block_runtime": False,
            "uses_forward_hook": False,
            "uses_backward_or_vjp_hook": False,
            "returns_model_grads": False,
            "returns_full_model_grads": False,
            "loss_cotangent_bridge_ready": False,
            "model_gradient_tree_ready": False,
            "delegates_to_eager_loss_and_grad": True,
            "hidden_packing_performed": False,
        }
    payload = dict(raw_contract)
    owner = str(payload.get("owner", ""))
    contract = str(payload.get("contract", ""))
    uses_runtime = bool(payload.get("uses_fused_train_block_runtime"))
    uses_forward = bool(payload.get("uses_forward_hook"))
    uses_reverse = bool(payload.get("uses_backward_or_vjp_hook"))
    returns_model_grads = bool(payload.get("returns_model_grads"))
    returns_full_model_grads = bool(payload.get("returns_full_model_grads", False))
    loss_cotangent_bridge_ready = bool(payload.get("loss_cotangent_bridge_ready"))
    model_gradient_tree_ready = bool(payload.get("model_gradient_tree_ready"))
    delegates_to_eager = bool(payload.get("delegates_to_eager_loss_and_grad", True))
    hidden_packing = bool(payload.get("hidden_packing_performed", False))
    status = (
        "ok"
        if contract == PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT
        and owner == "CompiledPretrainingStep"
        and uses_runtime
        and uses_forward
        and uses_reverse
        and returns_model_grads
        and loss_cotangent_bridge_ready
        and model_gradient_tree_ready
        and not delegates_to_eager
        and not hidden_packing
        else "incomplete"
    )
    return {
        **payload,
        "status": status,
        "contract": contract or PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
        "owner": owner or None,
        "uses_fused_train_block_runtime": uses_runtime,
        "uses_forward_hook": uses_forward,
        "uses_backward_or_vjp_hook": uses_reverse,
        "returns_model_grads": returns_model_grads,
        "returns_full_model_grads": returns_full_model_grads,
        "loss_cotangent_bridge_ready": loss_cotangent_bridge_ready,
        "model_gradient_tree_ready": model_gradient_tree_ready,
        "delegates_to_eager_loss_and_grad": delegates_to_eager,
        "hidden_packing_performed": hidden_packing,
    }


def _path_c_loss_cotangent_bridge_contract_payload(
    bridge: Any | None,
) -> dict[str, Any]:
    if bridge is None or not callable(bridge):
        return {
            "status": "missing",
            "contract": PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT,
            "returns_required_loss_cotangents": False,
            "delegates_to_eager_loss_and_grad": True,
            "hidden_packing_performed": False,
        }
    raw_contract = getattr(bridge, "loss_cotangent_bridge_contract", None)
    if callable(raw_contract):
        raw_contract = raw_contract()
    if not isinstance(raw_contract, Mapping):
        return {
            "status": "missing",
            "contract": PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT,
            "returns_required_loss_cotangents": False,
            "delegates_to_eager_loss_and_grad": True,
            "hidden_packing_performed": False,
        }
    payload = dict(raw_contract)
    contract = str(payload.get("contract", ""))
    returns_cotangents = bool(payload.get("returns_required_loss_cotangents"))
    delegates_to_eager = bool(payload.get("delegates_to_eager_loss_and_grad", True))
    hidden_packing = bool(payload.get("hidden_packing_performed", False))
    status = (
        "ok"
        if contract == PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT
        and returns_cotangents
        and not delegates_to_eager
        and not hidden_packing
        else "incomplete"
    )
    return {
        **payload,
        "status": status,
        "contract": contract or PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT,
        "returns_required_loss_cotangents": returns_cotangents,
        "delegates_to_eager_loss_and_grad": delegates_to_eager,
        "hidden_packing_performed": hidden_packing,
    }


class PathCResidualSumSuffixLossCotangentBridge:
    """Compute suffix loss cotangents for direct-chain residual output buffers."""

    def __init__(self, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> None:
        self.chunk_rows = max(1, int(chunk_rows))

    def loss_cotangent_bridge_contract(self) -> dict[str, Any]:
        return {
            "contract": PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT,
            "returns_required_loss_cotangents": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
            "suffix": "residual_sum_norm_lm_head_cut_cross_entropy",
            "parameter_gradient_names": ("norm.weight_grad", "lm_head.weight_grad"),
            "chunk_rows": self.chunk_rows,
        }

    def __call__(
        self,
        *,
        model: nn.Module,
        batch: Mapping[str, mx.array] | mx.array,
        logical_buffers: Mapping[str, mx.array],
        required_loss_cotangent_buffers: Sequence[str],
        chain: Any,
    ) -> dict[str, Any]:
        required_names = tuple(str(name) for name in required_loss_cotangent_buffers)
        if not required_names:
            raise ValueError("loss cotangent bridge requires at least one grad buffer")
        chain_required = tuple(
            str(name) for name in _path_c_direct_chain_loss_cotangent_seed_buffers(chain)
        )
        if chain_required and set(required_names) != set(chain_required):
            raise ValueError(
                "loss cotangent bridge required buffers do not match direct-chain "
                f"seed buffers: required={required_names!r}, chain={chain_required!r}"
            )
        source_names: list[str] = []
        for name in required_names:
            if not name.endswith("_grad"):
                raise ValueError(
                    f"loss cotangent seed buffer must end with '_grad': {name!r}"
                )
            source_names.append(name.removesuffix("_grad"))
        sources: list[mx.array] = []
        for name in source_names:
            value = logical_buffers.get(name)
            if not isinstance(value, mx.array):
                raise ValueError(
                    f"loss cotangent bridge missing source buffer {name!r}"
                )
            sources.append(value)
        first_shape = tuple(sources[0].shape)
        for name, value in zip(source_names[1:], sources[1:], strict=True):
            if tuple(value.shape) != first_shape:
                raise ValueError(
                    "loss cotangent bridge source shapes must match; "
                    f"{source_names[0]!r} has {first_shape}, {name!r} has "
                    f"{tuple(value.shape)}"
                )

        lm_batch = ensure_lm_batch(batch)
        targets = lm_batch.targets
        if tuple(sources[0].shape[:2]) != tuple(targets.shape):
            raise ValueError(
                "loss cotangent bridge suffix input prefix shape "
                f"{tuple(sources[0].shape[:2])} must match targets "
                f"{tuple(targets.shape)}"
            )
        mask = lm_batch.target_mask
        norm = getattr(model, "norm", None)
        lm_head = getattr(model, "lm_head", None)
        norm_weight = getattr(norm, "weight", None)
        head_weight = getattr(lm_head, "weight", None)
        if not isinstance(norm_weight, mx.array):
            raise TypeError(
                "loss cotangent bridge requires model.norm.weight as an mx.array"
            )
        if not isinstance(head_weight, mx.array):
            raise TypeError(
                "loss cotangent bridge requires model.lm_head.weight as an mx.array"
            )
        norm_eps = float(getattr(norm, "eps", 1e-5))

        def suffix_loss(*suffix_args: mx.array) -> tuple[mx.array, mx.array]:
            boundary_arrays = suffix_args[: len(sources)]
            norm_weight_arg = suffix_args[-2]
            head_weight_arg = suffix_args[-1]
            hidden = boundary_arrays[0]
            for boundary in boundary_arrays[1:]:
                hidden = hidden + boundary
            inv_rms = mx.rsqrt(
                mx.mean(hidden * hidden, axis=-1, keepdims=True)
                + mx.array(norm_eps, dtype=hidden.dtype)
            )
            normed = hidden * inv_rms * norm_weight_arg
            token_losses = linear_cross_entropy(
                normed,
                head_weight_arg,
                targets,
                reduction="none",
                chunk_rows=self.chunk_rows,
                eval_chunks=False,
            )
            ntokens = mask.sum()
            denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
            loss = (token_losses * mask).astype(mx.float32).sum() / denom
            return loss, ntokens

        suffix_args = (*sources, norm_weight, head_weight)
        argnums = tuple(range(len(suffix_args)))
        (loss, ntokens), raw_grads = mx.value_and_grad(
            suffix_loss,
            argnums=argnums,
        )(*suffix_args)
        grads = raw_grads if isinstance(raw_grads, tuple) else (raw_grads,)
        if len(grads) != len(suffix_args):
            raise RuntimeError(
                "loss cotangent bridge gradient arity mismatch: "
                f"got {len(grads)}, expected {len(suffix_args)}"
            )
        cotangent_grads = tuple(
            grad.astype(mx.float32) if isinstance(grad, mx.array) else grad
            for grad in grads[: len(required_names)]
        )
        norm_grad = grads[-2]
        head_grad = grads[-1]
        return {
            "loss": loss,
            "ntokens": ntokens,
            "cotangents": dict(zip(required_names, cotangent_grads, strict=True)),
            "parameter_grads": {
                "norm.weight_grad": norm_grad,
                "lm_head.weight_grad": head_grad,
            },
            "source_buffers": tuple(source_names),
            "required_loss_cotangent_buffers": required_names,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }


_DOCUMENT_ID_ALIASES = ("document_ids", "doc_ids", "packing_document_ids")


def _path_c_extract_document_ids(
    batch: Mapping[str, Any] | mx.array,
    *,
    tokens: mx.array,
) -> mx.array | None:
    if not isinstance(batch, Mapping):
        return None
    present = [
        name
        for name in _DOCUMENT_ID_ALIASES
        if name in batch and batch[name] is not None
    ]
    if len(present) > 1:
        raise ValueError(
            "batch mapping must provide only one document-id alias; got "
            f"{present}"
        )
    if not present:
        return None
    value = batch[present[0]]
    if not isinstance(value, mx.array):
        raise TypeError(f"{present[0]} must be an mx.array")
    if value.shape != tokens.shape:
        raise ValueError(
            f"{present[0]} must match tokens shape {tokens.shape}, got "
            f"{value.shape}"
        )
    has_negative = mx.any(value.astype(mx.int32) < 0)
    mx.eval(has_negative)
    if bool(has_negative.item()):
        raise ValueError(f"{present[0]} must be non-negative")
    return value.astype(mx.int32)


def _path_c_direct_chain_first_brick_name(chain: Any) -> str | None:
    source_region = getattr(chain, "source_region", None)
    metadata = getattr(source_region, "metadata", {}) or {}
    raw_bricks = metadata.get("path_c_bricks", ())
    try:
        bricks = tuple(raw_bricks)
    except TypeError:
        bricks = ()
    if bricks:
        first = bricks[0]
        if isinstance(first, Mapping) and first.get("name") is not None:
            return str(first["name"])
    nodes = getattr(source_region, "nodes", ()) or ()
    if nodes:
        return str(getattr(nodes[0], "name", ""))
    return None


def _path_c_direct_chain_start_layer_index(model: Any, chain: Any) -> int | None:
    first_brick = _path_c_direct_chain_first_brick_name(chain)
    if not first_brick:
        return None
    for index, layer in enumerate(getattr(model, "layers", ())):
        names = {
            str(getattr(layer, "path_c_brick_name", "")),
            str(getattr(layer, "path_c_profile_brick_name", "")),
        }
        if first_brick in names:
            return index
    return None


def _path_c_direct_chain_selected_layer_indices(
    model: Any,
    chain: Any,
) -> tuple[int, ...]:
    source_region = getattr(chain, "source_region", None)
    metadata = getattr(source_region, "metadata", {}) or {}
    raw_bricks = metadata.get("path_c_bricks", ())
    try:
        bricks = tuple(raw_bricks)
    except TypeError:
        bricks = ()
    if not bricks:
        start = _path_c_direct_chain_start_layer_index(model, chain)
        return () if start is None else (start,)

    name_to_index: dict[str, int] = {}
    for index, layer in enumerate(getattr(model, "layers", ())):
        for raw_name in (
            getattr(layer, "path_c_brick_name", None),
            getattr(layer, "path_c_profile_brick_name", None),
        ):
            if raw_name is not None:
                name_to_index[str(raw_name)] = index

    indices: list[int] = []
    for brick in bricks:
        brick_name = None
        if isinstance(brick, Mapping):
            brick_name = brick.get("name")
        else:
            brick_name = getattr(brick, "name", None)
        if brick_name is None:
            continue
        index = name_to_index.get(str(brick_name))
        if index is not None:
            indices.append(index)
    return tuple(dict.fromkeys(indices))


def _path_c_model_trainable_parameter_names(model: Any) -> frozenset[str]:
    return frozenset(
        str(name)
        for name, value in tree_flatten(model.trainable_parameters())
        if isinstance(value, mx.array)
    )


def _path_c_model_prefix_parameter_names(
    model: Any,
    *,
    end_layer_index: int,
    include_boundary_norm: bool = False,
) -> frozenset[str]:
    names: set[str] = set()
    for raw_name, value in tree_flatten(model.trainable_parameters()):
        if not isinstance(value, mx.array):
            continue
        name = str(raw_name)
        if name.startswith("layers."):
            parts = name.split(".", 2)
            if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) < end_layer_index:
                names.add(name)
            continue
        if name.startswith(("norm.", "lm_head.", "mtp_head.")):
            continue
        names.add(name)
    if include_boundary_norm:
        boundary_norm_name = f"layers.{end_layer_index}.norm.weight"
        trainable_names = _path_c_model_trainable_parameter_names(model)
        if boundary_norm_name in trainable_names:
            names.add(boundary_norm_name)
    return frozenset(names)


def _path_c_strip_gradient_suffix(name: str) -> str:
    return name[: -len("_grad")] if name.endswith("_grad") else name


def _path_c_gradient_tree_subset(grads: Any, names: frozenset[str]) -> Any:
    pairs = [
        (str(name), value)
        for name, value in tree_flatten(grads)
        if str(name) in names and isinstance(value, mx.array)
    ]
    present = {name for name, _ in pairs}
    missing = sorted(names.difference(present))
    if missing:
        raise ValueError(
            "prefix VJP did not return gradients for required parameters: "
            f"{missing[:8]}"
        )
    return tree_unflatten(pairs)


def _path_c_model_gradient_tree_strip_grad_suffixes(grads: Any) -> Any:
    pairs: list[tuple[str, mx.array]] = []
    seen: set[str] = set()
    for raw_name, value in tree_flatten(grads):
        if not isinstance(value, mx.array):
            continue
        name = _path_c_strip_gradient_suffix(str(raw_name))
        if name in seen:
            raise ValueError(f"duplicate model gradient name after suffix strip: {name}")
        seen.add(name)
        pairs.append((name, value))
    return tree_unflatten(pairs)


def _path_c_model_gradient_tree_from_parameter_grads(
    parameter_grads: Mapping[str, Any],
) -> Any:
    pairs: list[tuple[str, mx.array]] = []
    seen: set[str] = set()
    for raw_name, value in sorted(parameter_grads.items()):
        name = _path_c_strip_gradient_suffix(str(raw_name))
        if name in seen:
            raise ValueError(f"duplicate bridge parameter gradient {name!r}")
        if not isinstance(value, mx.array):
            raise TypeError(
                f"loss cotangent bridge parameter grad {raw_name!r} must be an mx.array"
            )
        seen.add(name)
        pairs.append((name, value))
    return tree_unflatten(pairs)


def _path_c_merge_model_gradient_trees(*trees: Any) -> Any:
    merged: list[tuple[str, mx.array]] = []
    seen: set[str] = set()
    for tree in trees:
        for raw_name, value in tree_flatten(tree):
            if not isinstance(value, mx.array):
                continue
            name = str(raw_name)
            if name in seen:
                raise ValueError(f"duplicate model gradient {name!r}")
            seen.add(name)
            merged.append((name, value))
    return tree_unflatten(sorted(merged, key=lambda item: item[0]))


def _path_c_model_gradient_tree_array_names(grads: Any) -> frozenset[str]:
    return frozenset(
        str(name)
        for name, value in tree_flatten(grads)
        if isinstance(value, mx.array)
    )


def _path_c_inactive_sparse_dsa_dense_parameter_names(
    model: Any,
    chain: Any,
) -> frozenset[str]:
    trainable_names = _path_c_model_trainable_parameter_names(model)
    inactive: set[str] = set()
    for index in _path_c_direct_chain_selected_layer_indices(model, chain):
        try:
            layer = tuple(getattr(model, "layers", ()))[index]
        except IndexError:
            continue
        attention = getattr(layer, "attention_block", None)
        if not isinstance(attention, CausalSelfAttention):
            continue
        if attention.config.mode != "dsa" or attention.sparse_kv_proj is None:
            continue
        for suffix in (
            "block.k_proj.weight",
            "block.k_proj.bias",
            "block.v_proj.weight",
            "block.v_proj.bias",
        ):
            name = f"layers.{index}.{suffix}"
            if name in trainable_names:
                inactive.add(name)
    return frozenset(inactive)


def _path_c_zero_gradient_tree_for_parameters(
    model: Any,
    names: frozenset[str],
) -> Any:
    if not names:
        return tree_unflatten([])
    parameters = {
        str(name): value
        for name, value in tree_flatten(model.trainable_parameters())
        if isinstance(value, mx.array)
    }
    pairs: list[tuple[str, mx.array]] = []
    missing = sorted(names.difference(parameters))
    if missing:
        raise ValueError(
            "cannot build zero gradients for unknown parameters: "
            f"{missing[:8]}"
        )
    for name in sorted(names):
        pairs.append((name, mx.zeros_like(parameters[name])))
    return tree_unflatten(pairs)


def path_c_model_prefix_hidden_states(
    model: nn.Module,
    batch: Mapping[str, mx.array] | mx.array,
    *,
    end_layer_index: int,
) -> mx.array:
    """Return decoder hidden states immediately before a Path C route region."""

    lm_batch = ensure_lm_batch(batch)
    input_ids = lm_batch.inputs
    layers = tuple(getattr(model, "layers", ()))
    if end_layer_index < 0 or end_layer_index > len(layers):
        raise ValueError(
            f"end_layer_index must be within [0, {len(layers)}], got "
            f"{end_layer_index}"
        )
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be shaped (B, S), got {input_ids.shape}")

    seq_length = input_ids.shape[1]
    config = getattr(model, "config", None)
    max_seq_length = int(getattr(config, "max_seq_length", seq_length))
    if seq_length > max_seq_length:
        raise ValueError(
            f"sequence length {seq_length} exceeds max_seq_length {max_seq_length}"
        )

    positions = mx.arange(seq_length)[None, :]
    hidden_states = model.token_embedding(input_ids) + model.position_embedding(
        positions
    )
    ngram_hash_embedding = getattr(model, "ngram_hash_embedding", None)
    if ngram_hash_embedding is not None:
        hidden_states = hidden_states + ngram_hash_embedding(input_ids)

    structure_embedding = getattr(model, "structure_embedding", None)
    if structure_embedding is not None:
        structure_embeddings = structure_embedding(
            structure_ids=lm_batch.structure_ids[:, :-1]
            if lm_batch.structure_ids is not None
            else None,
            dep_levels=lm_batch.dep_levels[:, :-1]
            if lm_batch.dep_levels is not None
            else None,
            ast_depth_ids=lm_batch.ast_depth_ids[:, :-1]
            if lm_batch.ast_depth_ids is not None
            else None,
            sibling_index_ids=lm_batch.sibling_index_ids[:, :-1]
            if lm_batch.sibling_index_ids is not None
            else None,
            node_type_ids=lm_batch.node_type_ids[:, :-1]
            if lm_batch.node_type_ids is not None
            else None,
            target_dtype=hidden_states.dtype,
        )
        if structure_embeddings.ndim == hidden_states.ndim:
            hidden_states = hidden_states + structure_embeddings

    document_ids = _path_c_extract_document_ids(batch, tokens=lm_batch.tokens)
    document_ids = document_ids[:, :-1] if document_ids is not None else None
    prefix_layers = layers[:end_layer_index]
    mask: mx.array | str | None = None
    if any(getattr(layer, "backend", None) == "attention" for layer in prefix_layers):
        if document_ids is None:
            dsa_path_c = (
                selected_path("sparse_mla") is KernelPath.PATH_C
                and sparse_mla_fp8_route_enabled(KernelPath.PATH_C)
                and any(
                    getattr(layer, "backend", None) == "attention"
                    and isinstance(getattr(layer, "block", None), CausalSelfAttention)
                    and layer.block.config.mode == "dsa"
                    for layer in prefix_layers
                )
            )
            mask = (
                "causal"
                if dsa_path_c
                else nn.MultiHeadAttention.create_additive_causal_mask(
                    seq_length,
                    dtype=hidden_states.dtype,
                )
            )
        else:
            mask = mlx_document_boundary_mask(
                document_ids,
                causal=True,
                expand_heads=True,
            )

    if bool(getattr(config, "grad_checkpoint", False)):
        for layer in prefix_layers:
            if getattr(layer, "backend", None) == "engram" and document_ids is not None:
                hidden_states = mx.checkpoint(layer)(
                    hidden_states,
                    mask,
                    doc_ids=document_ids,
                )
            else:
                hidden_states = mx.checkpoint(layer)(hidden_states, mask)
        return hidden_states

    attention_layer_idx = 0
    for layer in prefix_layers:
        if getattr(layer, "backend", None) == "attention":
            hidden_states = layer(
                hidden_states,
                mask,
                attention_layer_idx=None,
            )
            attention_layer_idx += 1
        elif getattr(layer, "backend", None) == "engram":
            hidden_states = layer(hidden_states, mask, doc_ids=document_ids)
        else:
            hidden_states = layer(hidden_states, mask)
    return hidden_states


def path_c_prefix_gradient_tree_from_hidden_cotangent(
    *,
    model: nn.Module,
    batch: Mapping[str, mx.array] | mx.array,
    hidden_cotangent: mx.array,
    normed_hidden_cotangent: mx.array | None = None,
    chain: Any,
) -> Any:
    start_layer_index = _path_c_direct_chain_start_layer_index(model, chain)
    if start_layer_index is None:
        raise ValueError("cannot resolve Path C direct-chain start layer")
    prefix_parameter_names = _path_c_model_prefix_parameter_names(
        model,
        end_layer_index=start_layer_index,
        include_boundary_norm=normed_hidden_cotangent is not None,
    )

    def prefix_vjp_loss(
        prefix_model: nn.Module,
        prefix_batch: Mapping[str, mx.array] | mx.array,
    ) -> mx.array:
        hidden = path_c_model_prefix_hidden_states(
            prefix_model,
            prefix_batch,
            end_layer_index=start_layer_index,
        )
        if tuple(hidden.shape) != tuple(hidden_cotangent.shape):
            raise ValueError(
                "prefix hidden shape must match direct-chain hidden_grad shape: "
                f"{tuple(hidden.shape)} vs {tuple(hidden_cotangent.shape)}"
            )
        loss = (
            hidden.astype(mx.float32) * hidden_cotangent.astype(mx.float32)
        ).sum()
        if normed_hidden_cotangent is not None:
            layers = tuple(getattr(prefix_model, "layers", ()))
            boundary_layer = layers[start_layer_index]
            normed_hidden = boundary_layer.norm(hidden)
            if tuple(normed_hidden.shape) != tuple(normed_hidden_cotangent.shape):
                raise ValueError(
                    "prefix normed hidden shape must match direct-chain "
                    "residual_norm_hidden_grad shape: "
                    f"{tuple(normed_hidden.shape)} vs "
                    f"{tuple(normed_hidden_cotangent.shape)}"
                )
            loss = loss + (
                normed_hidden.astype(mx.float32)
                * normed_hidden_cotangent.astype(mx.float32)
            ).sum()
        return loss

    _value, grads = nn.value_and_grad(model, prefix_vjp_loss)(model, batch)
    return _path_c_gradient_tree_subset(grads, prefix_parameter_names)


def _path_c_direct_chain_full_gradient_coverage_payload(
    *,
    model: Any | None,
    chain: Any,
    bridge_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if model is None:
        return {
            "full_model_gradient_tree_ready": False,
            "reason": "runtime was not constructed with a model",
            "covered_parameter_count": 0,
            "trainable_parameter_count": 0,
            "missing_parameter_names": [],
        }
    start_layer_index = _path_c_direct_chain_start_layer_index(model, chain)
    if start_layer_index is None:
        return {
            "full_model_gradient_tree_ready": False,
            "reason": "cannot resolve direct-chain start layer",
            "covered_parameter_count": 0,
            "trainable_parameter_count": len(_path_c_model_trainable_parameter_names(model)),
            "missing_parameter_names": [],
        }
    bridge_plan = path_c_direct_fusion_chain_value_and_grad_bridge_plan(
        chain=chain,
        model=model,
    )
    suffix_bridge_parameter_names = {"norm.weight_grad", "lm_head.weight_grad"}
    direct_names = {
        _path_c_strip_gradient_suffix(name)
        for name in bridge_plan.get("parameter_gradient_tree_names", ())
        if str(name) not in suffix_bridge_parameter_names
    }
    boundary_norm_name = f"layers.{start_layer_index}.norm.weight"
    prefix_names = _path_c_model_prefix_parameter_names(
        model,
        end_layer_index=start_layer_index,
        include_boundary_norm=boundary_norm_name not in direct_names,
    )
    suffix_names = {
        _path_c_strip_gradient_suffix(str(name))
        for name in bridge_contract.get("parameter_gradient_names", ())
    }
    inactive_zero_names = _path_c_inactive_sparse_dsa_dense_parameter_names(
        model,
        chain,
    )
    trainable_names = _path_c_model_trainable_parameter_names(model)
    covered = direct_names | prefix_names | suffix_names | inactive_zero_names
    missing = sorted(trainable_names.difference(covered))
    return {
        "full_model_gradient_tree_ready": not missing,
        "reason": "prefix, selected direct-chain, suffix, and inactive zero gradients cover all trainable parameters"
        if not missing
        else "prefix, selected direct-chain, suffix, and inactive zero gradients do not cover all trainable parameters",
        "start_layer_index": start_layer_index,
        "covered_parameter_count": len(covered.intersection(trainable_names)),
        "trainable_parameter_count": len(trainable_names),
        "missing_parameter_names": missing,
        "prefix_parameter_count": len(prefix_names),
        "direct_chain_parameter_count": len(direct_names),
        "suffix_parameter_count": len(suffix_names),
        "inactive_zero_gradient_parameter_count": len(inactive_zero_names),
        "inactive_zero_gradient_parameter_names": sorted(inactive_zero_names),
    }


def _path_c_fusion_planner_unavailable_payload(
    *,
    args: argparse.Namespace | TrainHybridTinyConfig,
    sequence_length: int,
    exception: Exception,
) -> dict[str, Any]:
    mode = selected_path_c_fusion_mode()
    profile_name = str(getattr(args, "model_profile", "hybrid_tiny"))
    route_symbols: tuple[str, ...] = ()
    if profile_name == REQUIRED_MODEL_PROFILE:
        route_symbols = tuple(local_gb10_quarter_profile().expanded_pattern.symbols)
    exception_payload = {
        "type": type(exception).__name__,
        "message": str(exception),
    }
    status = "path_c_fusion_planner_unavailable"
    reason = (
        "Path C fusion route metadata failed closed because the planner could "
        f"not be imported or executed: {exception_payload['type']}: "
        f"{exception_payload['message']}"
    )
    runtime_binding = {
        "status": status,
        "runtime_uses_fused_train_block": False,
        "runtime_binding_ready": False,
        "physical_abi_binding_ready": False,
        "provided_bank_buffers": [],
        "required_bank_buffers": [],
        "no_hidden_allocation_policy": True,
        "reason": reason,
    }
    training_contract = {
        "status": status,
        "training_runtime_available": False,
        "runtime_installed": False,
        "critical_path_ready": False,
        "reason": reason,
    }
    direct_chain_binding = {
        "status": status,
        "runtime_uses_direct_fusion_chain": False,
        "logical_tensor_binding_ready": False,
        "direct_chain_artifacts_bound": False,
        "reason": reason,
    }
    return {
        "mode": mode.value,
        "status": status,
        "reason": reason,
        "region_name": None,
        "backend": "tilelang",
        "compiler": "tilelang.engine.fusion",
        "fusion_kind": "model_route_path_c",
        "sequence_length": int(sequence_length),
        "planner_exception": exception_payload,
        "graph_construction": {
            "builder": "PathCFusionScheduleOptimizer",
            "input_model": profile_name,
            "route_symbols": list(route_symbols),
            "region_source": "planner_unavailable",
            "selected_model_region": None,
            "selected_model_region_op_signature": [],
            "selected_model_region_schedule_id": None,
            "preset_only": False,
        },
        "model_route_candidates": {
            "profile": profile_name,
            "route_symbols": list(route_symbols),
            "region_source": "planner_unavailable",
            "selection_policy": "largest_supported_contiguous_route_segment",
            "selected_candidate": None,
            "candidate_regions": [],
            "planner_exception": exception_payload,
        },
        "runtime_training_binding": runtime_binding,
        "fused_train_block_training_runtime_contract": training_contract,
        "direct_chained_fusion": {
            "status": status,
            "runtime_binding": direct_chain_binding,
            "training_runtime_contract": training_contract,
        },
        "schedule_blockers": [
            {
                "kind": status,
                "reason": reason,
                "exception": exception_payload,
            }
        ],
        "single_kernel_fused": False,
        "production_compile_receipt": {
            "status": "not_evaluated",
            "verified": False,
            "reason": reason,
        },
        "production_matrix_profile_receipt": {
            "status": "not_evaluated",
            "verified": False,
            "reason": reason,
        },
        "zero_copy_required": True,
    }


def fp8_path_c_training_route_payload(
    args: argparse.Namespace | TrainHybridTinyConfig,
    *,
    model: Any | None = None,
    compile_receipt_path: str | Path | None = None,
    bank_buffers: Mapping[str, Any] | None = None,
    bank_buffer_owner: str | None = None,
    bank_owner: Any | None = None,
    fused_artifact: Any | None = None,
    direct_chain_artifacts: Any | None = None,
    direct_chain_logical_buffers: Mapping[str, Any] | None = None,
    direct_chain_logical_buffer_owner: str | None = None,
    direct_chain_logical_owner: Any | None = None,
    direct_chain_training_runtime: Any | None = None,
    fused_train_block_training_runtime: Any | None = None,
    path_c_fusion_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if path_c_fusion_fn is None:
        path_c_fusion_fn = path_c_fusion_payload
    requested = path_c_training_route_requested(args)
    fp8_producer_required = fp8_path_c_route_requested(args)
    producer = sparse_mla_fp8_producer_payload(args)
    fp8_producer_configured = bool(producer["configured"])
    producer_configured = bool(
        fp8_producer_configured or not fp8_producer_required
    )
    sequence_length = path_c_training_sequence_length(args)
    try:
        path_c_fusion = dict(
            path_c_fusion_fn(
                model=model,
                compile_receipt_path=compile_receipt_path,
                bank_buffers=bank_buffers,
                bank_buffer_owner=bank_buffer_owner,
                bank_owner=bank_owner,
                fused_artifact=fused_artifact,
                direct_chain_artifacts=direct_chain_artifacts,
                direct_chain_logical_buffers=direct_chain_logical_buffers,
                direct_chain_logical_buffer_owner=direct_chain_logical_buffer_owner,
                direct_chain_logical_owner=direct_chain_logical_owner,
                direct_chain_training_runtime=direct_chain_training_runtime,
                fused_train_block_training_runtime=fused_train_block_training_runtime,
                sequence_length=sequence_length,
            )
        )
    except Exception as exc:
        path_c_fusion = _path_c_fusion_planner_unavailable_payload(
            args=args,
            sequence_length=sequence_length or 0,
            exception=exc,
        )
    runtime_training_binding = path_c_fusion.get("runtime_training_binding", {})
    single_fused_train_block_standalone_dispatch_available = bool(
        isinstance(runtime_training_binding, dict)
        and runtime_training_binding.get("runtime_uses_fused_train_block")
    )
    fused_train_block_training_contract: Mapping[str, Any] = {}
    candidate_fused_contract = path_c_fusion.get(
        "fused_train_block_training_runtime_contract",
        {},
    )
    if isinstance(candidate_fused_contract, Mapping):
        fused_train_block_training_contract = candidate_fused_contract
    fused_train_block_training_critical_path = bool(
        fused_train_block_training_contract.get("critical_path_ready")
    )
    fused_train_block_training_runtime_available = bool(
        single_fused_train_block_standalone_dispatch_available
        and fused_train_block_training_contract.get("training_runtime_available")
    )
    single_fused_train_block_runtime_available = (
        fused_train_block_training_runtime_available
    )
    direct_chain_binding = {}
    direct_chain_training_contract: Mapping[str, Any] = {}
    direct_chained_fusion = path_c_fusion.get("direct_chained_fusion", {})
    if isinstance(direct_chained_fusion, dict):
        direct_chain_binding = direct_chained_fusion.get("runtime_binding", {})
        candidate_contract = direct_chained_fusion.get("training_runtime_contract", {})
        if isinstance(candidate_contract, Mapping):
            direct_chain_training_contract = candidate_contract
    direct_fusion_chain_runtime_available = bool(
        isinstance(direct_chain_binding, dict)
        and direct_chain_binding.get("runtime_uses_direct_fusion_chain")
    )
    direct_chain_training_critical_path = bool(
        direct_chain_training_contract.get("critical_path_ready")
    )
    direct_fusion_chain_training_runtime_available = bool(
        direct_fusion_chain_runtime_available
        and direct_chain_training_contract.get("training_runtime_available")
    )
    fused_train_block_runtime_available = bool(
        single_fused_train_block_runtime_available
    )
    path_c_training_runtime_available = bool(
        fused_train_block_runtime_available
        or direct_fusion_chain_training_runtime_available
    )
    split_training_available = bool(requested and producer_configured)
    full_training_available = bool(
        split_training_available and path_c_training_runtime_available
    )
    route_status = (
        FP8_PATH_C_E2E_TRAINING_STATUS
        if full_training_available
        else FP8_PATH_C_SPLIT_TRAINING_STATUS
        if requested and producer_configured
        else producer["status"]
        if requested and fp8_producer_required
        else FP8_PATH_C_SPLIT_TRAINING_STATUS
        if requested
        else "not_requested"
    )
    fused_train_block_blocker_type = (
        None
        if not requested
        else None
        if full_training_available
        else str(fused_train_block_training_contract.get("status"))
        if single_fused_train_block_standalone_dispatch_available
        else str(runtime_training_binding.get("status"))
        if producer_configured
        else str(producer["status"])
        if fp8_producer_required
        else str(runtime_training_binding.get("status"))
    )
    direct_call_status = (
        "m04_uses_fused_train_block_route"
        if full_training_available and single_fused_train_block_runtime_available
        else "m04_uses_direct_fusion_chain_route"
        if (
            full_training_available
            and direct_fusion_chain_training_runtime_available
        )
        else "m04_fused_train_block_standalone_only_not_training_route"
        if single_fused_train_block_standalone_dispatch_available
        else "m04_direct_fusion_chain_standalone_only_not_training_route"
        if direct_fusion_chain_runtime_available
        else "m04_uses_split_model_graph_route_not_fused_train_block"
        if split_training_available
        else str(producer["status"])
        if requested and fp8_producer_required
        else "m04_uses_split_model_graph_route_not_fused_train_block"
        if requested
        else "not_requested"
    )
    selected_action = (
        "run_path_c_fused_train_block_route"
        if full_training_available and single_fused_train_block_runtime_available
        else "run_path_c_direct_fusion_chain_route"
        if (
            full_training_available
            and direct_fusion_chain_training_runtime_available
        )
        else "run_path_c_split_training_route"
        if split_training_available
        else f"fail_closed_{producer['status']}"
        if requested and fp8_producer_required
        else "run_path_c_split_training_route"
        if requested
        else None
    )
    return {
        "requested": requested,
        "dtype": str(getattr(args, "dtype", "")),
        "status": route_status,
        "blocker_type": None
        if (not requested or producer_configured)
        else str(producer["status"]),
        "reason": producer["reason"]
        if requested and fp8_producer_required and not producer_configured
        else None,
        "carrier_dtype": FP8_PATH_C_CARRIER_DTYPE,
        "native_fp8_producer_status": producer["status"],
        "sparse_mla_fp8_producer": producer,
        "prepared_buffers_configured": bool(producer["prepared_buffers_configured"]),
        "producer_stage": producer["producer_stage"],
        "producer_quantization": producer["producer_quantization"],
        "kernel_surface_status": FP8_PATH_C_KERNEL_SURFACE_STATUS,
        "kernel_surface_available": True,
        "split_end_to_end_training_available": split_training_available,
        "full_end_to_end_training_available": full_training_available,
        "fused_train_block_runtime_available": (
            fused_train_block_runtime_available
        ),
        "path_c_training_runtime_available": (
            path_c_training_runtime_available
        ),
        "single_fused_train_block_runtime_available": (
            single_fused_train_block_runtime_available
        ),
        "single_fused_train_block_standalone_dispatch_available": (
            single_fused_train_block_standalone_dispatch_available
        ),
        "fused_train_block_training_critical_path": (
            fused_train_block_training_critical_path
        ),
        "fused_train_block_training_runtime_available": (
            fused_train_block_training_runtime_available
        ),
        "fused_train_block_training_runtime_contract": dict(
            fused_train_block_training_contract
        ),
        "direct_fusion_chain_runtime_available": (
            direct_fusion_chain_runtime_available
        ),
        "direct_fusion_chain_standalone_dispatch_available": (
            direct_fusion_chain_runtime_available
        ),
        "direct_fusion_chain_training_critical_path": (
            direct_chain_training_critical_path
        ),
        "direct_fusion_chain_training_runtime_available": (
            direct_fusion_chain_training_runtime_available
        ),
        "direct_fusion_chain_training_runtime_contract": dict(
            direct_chain_training_contract
        ),
        "fused_train_block_blocker_type": fused_train_block_blocker_type,
        "end_to_end_training_status": route_status,
        "direct_mx_array_artifact_call_status": direct_call_status,
        "bridge_target": FP8_PATH_C_BRIDGE_TARGET,
        "bridge_status": FP8_PATH_C_BRIDGE_STATUS,
        "bridge_evidence": {
            "mlx_array_exports_dlpack": True,
            "mlx_public_from_dlpack_available": False,
            "tvm_ffi_from_dlpack_available": True,
            "mlx_metal_dlpack_device": "kDLMetal:0",
            "tvm_from_dlpack_device": "metal:0",
            "native_mlx_array_wrapper_linked": True,
            "native_tvm_ffi_graph_outputs": True,
            "dlpack_used_for_path_c_graph_bridge": False,
            "standalone_mlx_to_tvm_metal_kernel_verified": True,
            "m04_bridge_wired": bool(requested),
        },
        "contract": "end_to_end_training_route_over_existing_gpu_buffers",
        "zero_copy_required": True,
        "large_tensor_staging_allowed": False,
        "hidden_wrapper_quantization_allowed": False,
        "kernel_boundary_quantization_allowed": False,
        "hidden_dtype_cast_allowed": False,
        "hidden_shape_staging_allowed": False,
        "fallback_to_path_b_allowed": False,
        "available_path_c_surfaces": [
            {
                "name": "fp8_scaled_vecmat_path_c",
                "module": "cppmega_mlx.nn._tilelang.fp8_vecmat_path_c",
                "shape_surface": "M=1, W=(N,K), forward prepared buffers",
                "training_surface": False,
            },
            {
                "name": "sparse_mla_fp8_path_c_apply",
                "module": "cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c",
                "shape_surface": "prepared q_fp8/q_scale/kv_fp8/kv_scale sparse-MLA buffers",
                "training_surface": fp8_producer_configured,
                "producer_required": True,
                "producer_status": producer["status"],
                "backward_surface": "native_tvm_ffi_graph_output_scatter",
                "fallback_backward_surface": "prepared_fp8_path_b_reference_vjp",
                "default_backward_route": "path_c",
                "default_backward_reason": (
                    "native Sparse-MLA Path C backward uses graph outputs for "
                    "the no-owner VJP path; caller-owned buffers remain explicit "
                    "for fused runtimes"
                ),
                "kernel_policy_env": {
                    SPARSE_MLA_FP8_ROUTE_ENV: "path_c",
                    SPARSE_MLA_FP8_BWD_ENV: "path_c",
                },
            },
            {
                "name": "mamba3_mimo_path_c",
                "module": "cppmega_mlx.nn._tilelang.mamba3_path_c",
                "shape_surface": "HybridTinyLM M-layer selective scan",
                "kernel_policy_env": {
                    "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
                },
                "fp8_route_auto_selected": True,
                "fp8_route_reason": (
                    "Path C now uses direct tvm-ffi owner-output for contiguous "
                    "selective-scan forward buffers and keeps the performance-safe "
                    "Path B backward route until full Path C backward wins the "
                    "1B training matrix"
                ),
                "training_surface": False,
                "full_path_c_backward_available": True,
                "default_backward_route": "path_b",
            },
            {
                "name": "m2rnn_path_c",
                "module": "cppmega_mlx.nn._tilelang.m2rnn_path_c",
                "shape_surface": (
                    "HybridTinyLM R-layer packed recurrence with explicit h0 "
                    "and TileLang owner-output forward/backward"
                ),
                "kernel_policy_env": {
                    "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
                },
                "fp8_route_auto_selected": True,
                "fp8_route_reason": (
                    "Path C uses the packed TileLang DSL recurrence through "
                    "the native MLX/tvm-ffi graph bridge; transform-boundary "
                    "fallback to Path B is not allowed"
                ),
                "training_surface": True,
                "fallback_to_path_b_allowed": False,
            },
            {
                "name": "matmul_tl_fp8_scaled_matmul",
                "module": "scripts.bench_tilelang_fp8_path_c",
                "shape_surface": (
                    "M>1 T.fp8_scaled_matmul prepared "
                    "A_fp8/A_scale/B_fp8/B_scale buffers"
                ),
                "kernel_surface_available": True,
                "training_surface": False,
                "reason": (
                    "prepared-buffer kernel surface is available, but m04 has no "
                    "model producer/autograd route wired through DLPack/tvm-ffi "
                    "to reach it without hidden large tensor staging"
                ),
            },
        ],
        "missing_training_surfaces": [
            "FP8 parameter/weight producers that create the required dtype/layout "
            "before matmul kernel boundaries",
            "absorbed MLA producer split for NoPE/RoPE KV layout and calibrated "
            "separate K/V scale lifecycle",
        ],
        "higher_level_owner": {
            "current_m04_route_owner": (
                "scripts.m04_train_step -> HybridTinyLM -> "
                "CausalSelfAttention.prepare_sparse_mla_fp8"
                if fp8_producer_configured
                else (
                    "scripts.m04_train_step -> scripts.train_hybrid_tiny -> "
                    "HybridTinyLM without DSA Sparse-MLA producer"
                )
            ),
            "sparse_mla_fp8_next_owner": (
                FP8_PATH_C_PRODUCER_OWNER
            ),
            "model_factory_owner": (
                "cppmega_mlx.recipes.model_factory.local_gb10_quarter wires "
                "HybridTinyLM; DSA A layers use prepared Sparse-MLA FP8 when "
                "CPPMEGA_KERNEL_PATH__SPARSE_MLA=path_c"
            ),
        },
        "kernel_policy_env": dict(FP8_PATH_C_KERNEL_POLICY_ENV),
        "selected_action": selected_action,
        "path_c_fusion": path_c_fusion,
    }


def _as_path_c_logical_owner_tuple(raw_owners: Any) -> tuple[Any, ...]:
    if raw_owners is None:
        return ()
    if isinstance(raw_owners, Mapping) or hasattr(raw_owners, "buffers"):
        return (raw_owners,)
    return tuple(raw_owners)


def _path_c_direct_chain_runtime_logical_owner_for_model(model: Any) -> Any | None:
    runtime = getattr(model, "path_c_direct_fusion_chain_training_runtime", None)
    if runtime is None:
        return None
    for attr_name in ("last_pre_step_owner", "logical_owner"):
        owner = getattr(runtime, attr_name, None)
        owner_buffers = getattr(owner, "buffers", None)
        if isinstance(owner_buffers, Mapping):
            return owner
    return None


def _path_c_direct_chain_logical_owner_for_model(model: Any) -> Any | None:
    runtime_owner = _path_c_direct_chain_runtime_logical_owner_for_model(model)
    if runtime_owner is not None:
        return runtime_owner

    owners: list[Any] = []
    base_owner = getattr(
        model,
        "path_c_direct_fusion_chain_logical_buffer_owner",
        None,
    )
    if base_owner is None:
        make_logical_owner = getattr(
            model,
            "make_path_c_direct_fusion_chain_logical_buffer_owner",
            None,
        )
        if callable(make_logical_owner):
            base_owner = make_logical_owner()
    if base_owner is not None:
        owners.append(base_owner)
    owners.extend(
        _as_path_c_logical_owner_tuple(
            getattr(
                model,
                "path_c_direct_fusion_chain_logical_buffer_owners",
                None,
            )
        )
    )
    if not owners:
        return None
    if len(owners) == 1:
        return owners[0]
    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    return compose_path_c_logical_buffer_owner(
        f"{profile_name}.path_c_direct_fusion_chain_buffers",
        *owners,
    )


def _path_c_physical_abi_bank_owner_for_model(
    model: Any,
    *,
    sequence_length: int | None,
    allocate_if_missing: bool = True,
) -> Any | None:
    bank_owner = getattr(model, "path_c_physical_abi_bank_owner", None)
    if bank_owner is not None:
        return bank_owner

    if not allocate_if_missing:
        return None

    make_bank_owner = getattr(model, "make_path_c_physical_abi_bank_owner", None)
    if callable(make_bank_owner):
        return make_bank_owner(sequence_length=sequence_length)
    return None


def _path_c_kernel_buffer_dtype(name: str, *, default: str = "float32") -> Any:
    """Return the MLX dtype implied by a generated Path C kernel buffer name."""

    text = str(name)
    if "uint8" in text:
        return mx.uint8
    if "int32" in text:
        return mx.int32
    if "bfloat16" in text:
        return mx.bfloat16
    if "float16" in text:
        return mx.float16
    if "float32" in text:
        return mx.float32
    return getattr(mx, default)


def _path_c_shape_numel(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size)


def _path_c_bank_owner_buffers(bank_owner: Any | None) -> Mapping[str, Any]:
    if bank_owner is None:
        return {}
    if isinstance(bank_owner, Mapping):
        return {str(name): value for name, value in bank_owner.items()}
    buffers = getattr(bank_owner, "buffers", None)
    if isinstance(buffers, Mapping):
        return {str(name): value for name, value in buffers.items()}
    return {}


def _path_c_artifact_kernel_buffer_shape_requirements(
    artifact: Any,
) -> dict[str, tuple[int, ...]]:
    """Return largest per-buffer shape required by a selected runtime artifact."""

    requirements: dict[str, tuple[int, ...]] = {}

    def add_shapes(shapes: Any) -> None:
        if not isinstance(shapes, Mapping):
            return
        for raw_name, raw_shape in shapes.items():
            name = str(raw_name)
            if name in PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS:
                continue
            shape = tuple(int(dim) for dim in tuple(raw_shape))
            current = requirements.get(name)
            if current is None or _path_c_shape_numel(shape) > _path_c_shape_numel(
                current
            ):
                requirements[name] = shape

    add_shapes(getattr(artifact, "kernel_buffer_shapes", None))
    add_shapes(getattr(artifact, "forward_kernel_buffer_shapes", None))
    add_shapes(getattr(artifact, "backward_kernel_buffer_shapes", None))
    for attr_name in (
        "forward_kernel_buffer_shapes_by_stage",
        "backward_kernel_buffer_shapes_by_stage",
    ):
        for shapes in getattr(artifact, attr_name, ()) or ():
            add_shapes(shapes)
    return requirements


def _path_c_artifact_bank_owner_kernel_buffer_binding(
    *,
    artifact: Any | None,
    bank_owner: Any | None,
) -> dict[str, Any]:
    """Validate top-level kernel buffers that are outside physical ABI banks."""

    if artifact is None:
        return {
            "status": "not_bound",
            "reason": "no fused train-block artifact is selected",
            "missing_kernel_buffers": [],
            "undersized_kernel_buffers": [],
            "required_kernel_buffers": [],
        }

    requirements = _path_c_artifact_kernel_buffer_shape_requirements(artifact)
    required_names = sorted(requirements)
    buffers = _path_c_bank_owner_buffers(bank_owner)
    missing: list[str] = []
    undersized: list[dict[str, Any]] = []
    for name in required_names:
        value = buffers.get(name)
        if value is None:
            missing.append(name)
            continue
        expected_shape = requirements[name]
        expected_size = _path_c_shape_numel(expected_shape)
        actual_shape = tuple(int(dim) for dim in tuple(getattr(value, "shape", ())))
        actual_size = int(getattr(value, "size", 0) or _path_c_shape_numel(actual_shape))
        if actual_size < expected_size:
            undersized.append(
                {
                    "name": name,
                    "expected_shape": list(expected_shape),
                    "expected_size": expected_size,
                    "actual_shape": list(actual_shape),
                    "actual_size": actual_size,
                }
            )
    ok = not missing and not undersized
    return {
        "status": "ok" if ok else "failed",
        "reason": (
            "caller/model-owned kernel buffers satisfy selected artifact"
            if ok
            else "caller/model-owned kernel buffers do not satisfy selected artifact"
        ),
        "required_kernel_buffers": required_names,
        "missing_kernel_buffers": missing,
        "undersized_kernel_buffers": undersized,
        "bank_buffer_owner": getattr(bank_owner, "owner_name", None),
    }


def _path_c_physical_abi_bank_owner_for_artifact(
    *,
    artifact: Any,
    owner_name: str,
) -> Any | None:
    """Allocate model-owned ABI banks sized to the selected compiled artifact."""

    physical_abi_map = getattr(artifact, "physical_abi_map", None)
    physical_abi_shapes = getattr(artifact, "physical_abi_shapes", None)
    if not isinstance(physical_abi_map, Mapping) or not isinstance(
        physical_abi_shapes,
        Mapping,
    ):
        return None
    if not physical_abi_map or not physical_abi_shapes:
        return None

    bank_dtypes: dict[str, str] = {}
    for placement in physical_abi_map.values():
        if not isinstance(placement, Mapping):
            continue
        bank = str(placement.get("bank", ""))
        dtype = str(placement.get("dtype", ""))
        if not bank or not dtype:
            continue
        existing = bank_dtypes.setdefault(bank, dtype)
        if existing != dtype:
            raise ValueError(
                f"conflicting bank dtype for {bank!r}: {existing!r} vs {dtype!r}"
            )

    buffers: dict[str, mx.array] = {}
    for bank, shape in physical_abi_shapes.items():
        bank_name = str(bank)
        dtype_name = bank_dtypes.get(bank_name)
        if dtype_name is None:
            raise ValueError(
                f"no logical buffer is placed inside physical bank {bank_name!r}"
            )
        buffers[bank_name] = mx.zeros(
            tuple(int(dim) for dim in tuple(shape)),
            dtype=_path_c_kernel_buffer_dtype(bank_name, default=dtype_name),
        )

    kernel_buffer_shapes = getattr(artifact, "kernel_buffer_shapes", None)
    if isinstance(kernel_buffer_shapes, Mapping):
        for name, shape in kernel_buffer_shapes.items():
            buffer_name = str(name)
            if (
                buffer_name in buffers
                or buffer_name in physical_abi_shapes
                or buffer_name in PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS
            ):
                continue
            buffers[buffer_name] = mx.zeros(
                tuple(int(dim) for dim in tuple(shape)),
                dtype=_path_c_kernel_buffer_dtype(buffer_name),
            )

    return make_physical_abi_bank_owner(
        owner_name,
        physical_abi_map,
        physical_abi_shapes,
        buffers,
    )


def _path_c_direct_chain_runtime_capture_aliases_for_model(
    model: Any,
) -> dict[str, tuple[str, ...]]:
    """Return direct-chain aliases that can be captured during eager runtime."""

    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    region_prefix = _path_c_direct_chain_region_prefix(model, profile_name)
    try:
        regions = build_path_c_model_regions_from_model(
            model,
            region_prefix=region_prefix,
        )
        selected_region = _select_path_c_model_route_region(regions)
    except Exception:
        selected_region = None
    if selected_region is None or not getattr(selected_region, "nodes", ()):
        return {}

    aliases: dict[str, tuple[str, ...]] = {}
    first_node = selected_region.nodes[0]
    if "hidden" in first_node.inputs:
        aliases[f"{first_node.name}_hidden"] = ("hidden",)
    return aliases


def _path_c_direct_chain_runtime_capture_owners_for_model(
    model: Any,
) -> tuple[PathCActivationBufferCapture, PathCGradientBufferCapture]:
    """Create zero-copy runtime capture owners for a model direct-chain audit."""

    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    activation_capture = PathCActivationBufferCapture(
        aliases=_path_c_direct_chain_runtime_capture_aliases_for_model(model),
        owner_name=f"{profile_name}.path_c_forward_activation_capture",
    )
    gradient_aliases: Mapping[str, Any] = {}
    if callable(getattr(model, "path_c_parameter_gradient_aliases", None)):
        gradient_aliases = model.path_c_parameter_gradient_aliases()
    gradient_capture = PathCGradientBufferCapture(
        aliases=gradient_aliases,
        owner_name=f"{profile_name}.path_c_parameter_gradient_capture",
    )
    return activation_capture, gradient_capture


def fp8_path_c_training_route_payload_for_model(
    args: argparse.Namespace | TrainHybridTinyConfig,
    model: Any,
    *,
    compile_receipt_path: str | Path | None = None,
    auto_install_fused_train_block: bool = False,
    fused_train_block_artifact_lowerer: Callable[..., Any] | None = None,
    fused_train_block_artifact_target_name: str = "metal",
    fused_train_block_artifact_execution_backend: str = "tvm_ffi",
) -> dict[str, Any]:
    """Return Path C training route metadata using model-owned ABI banks."""

    sequence_length = path_c_training_sequence_length(args)
    auto_install_requested = bool(
        auto_install_fused_train_block
        or getattr(args, "use_path_c_fused_train_block_runtime", False)
    )
    auto_install_report: dict[str, Any] | None = None
    model_fused_artifact = getattr(model, "path_c_fused_train_block_artifact", None)
    model_fused_training_runtime = getattr(
        model,
        "path_c_fused_train_block_training_runtime",
        None,
    )
    artifact_can_back_training_runtime = (
        _path_c_fused_train_block_artifact_has_training_runtime_contract(
            model_fused_artifact
        )
    )
    model_bank_owner = _path_c_physical_abi_bank_owner_for_model(
        model,
        sequence_length=sequence_length,
        allocate_if_missing=(
            (
                auto_install_requested
                and callable(model_fused_artifact)
            )
            or artifact_can_back_training_runtime
            or model_fused_training_runtime is not None
        ),
    )
    if (
        auto_install_requested
        and (
            not callable(model_fused_artifact)
            or (
                model_fused_training_runtime is None
                and artifact_can_back_training_runtime
            )
        )
    ):
        compile_missing_artifact = not callable(model_fused_artifact)
        auto_install_report = install_path_c_fused_train_block_runtime_for_model(
            model=model,
            bank_owner=None if compile_missing_artifact else model_bank_owner,
            training_runtime=model_fused_training_runtime,
            compile_artifact=compile_missing_artifact,
            artifact_lowerer=fused_train_block_artifact_lowerer,
            artifact_target_name=fused_train_block_artifact_target_name,
            artifact_execution_backend=(
                fused_train_block_artifact_execution_backend
            ),
            sequence_length=sequence_length,
        )
        model_bank_owner = _path_c_physical_abi_bank_owner_for_model(
            model,
            sequence_length=sequence_length,
            allocate_if_missing=False,
        )

    direct_chain_logical_owner = _path_c_direct_chain_logical_owner_for_model(model)

    route = fp8_path_c_training_route_payload(
        args,
        model=model,
        compile_receipt_path=compile_receipt_path,
        bank_owner=model_bank_owner,
        fused_artifact=getattr(model, "path_c_fused_train_block_artifact", None),
        direct_chain_artifacts=getattr(
            model,
            "path_c_direct_fusion_chain_artifacts",
            None,
        ),
        direct_chain_logical_buffers=getattr(
            model,
            "path_c_direct_fusion_chain_logical_buffers",
            None,
        ),
        direct_chain_logical_buffer_owner=getattr(
            model,
            "path_c_direct_fusion_chain_logical_buffer_owner_name",
            None,
        ),
        direct_chain_logical_owner=direct_chain_logical_owner,
        direct_chain_training_runtime=getattr(
            model,
            "path_c_direct_fusion_chain_training_runtime",
            None,
        ),
        fused_train_block_training_runtime=getattr(
            model,
            "path_c_fused_train_block_training_runtime",
            None,
        ),
    )
    if auto_install_report is not None:
        path_c_fusion = route.get("path_c_fusion")
        if isinstance(path_c_fusion, dict):
            path_c_fusion["fused_train_block_auto_install"] = auto_install_report
    return route


def _path_c_fused_train_block_artifact_has_training_runtime_contract(
    artifact: Any,
) -> bool:
    return bool(
        callable(artifact)
        and callable(getattr(artifact, "forward", None))
        and (
            callable(getattr(artifact, "backward", None))
            or callable(getattr(artifact, "vjp", None))
        )
        and callable(getattr(artifact, "value_and_grad", None))
        and callable(getattr(artifact, "value_and_grad_contract", None))
    )


def _build_path_c_fused_suffix_loss_fn_for_model(
    *,
    model: Any,
    artifact: Any,
    bank_owner: Any,
    in_region_parameter_bank_aliases: Mapping[str, Mapping[str, Any]],
    sequence_length: int | None,
) -> Callable[[Any, Mapping[str, Any]], tuple[mx.array, mx.array]] | None:
    """Compose the fused-suffix custom function and attach it to the model.

    Returns the model's ``path_c_fused_suffix_loss`` method bound to its
    own ``self`` so the training runtime can call it as ``loss_fn(model,
    batch)``. Returns ``None`` when the model does not expose the
    suffix-bridge surface or the ABI map does not declare the required
    fused-suffix inputs / outputs.
    """

    from cppmega_mlx.training.path_c_fused_suffix import (
        build_fused_suffix_custom_function,
    )

    if not callable(getattr(model, "path_c_fused_suffix_loss", None)):
        return None
    if not callable(getattr(model, "attach_path_c_fused_suffix_custom_function", None)):
        return None

    prim_func = model.path_c_fused_train_block_prim_func(
        sequence_length=sequence_length,
    )
    if prim_func is None:
        return None
    abi_map = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
        or {}
    )
    if not abi_map:
        return None
    # Discover the canonical hidden entry name.
    hidden_entry_logical_name: str | None = None
    for name in abi_map:
        if (
            "_hidden" in name
            and not name.endswith("_grad")
            and not name.endswith("_after")
            and not name.endswith("_after_grad")
        ):
            hidden_entry_logical_name = name
            break
    if hidden_entry_logical_name is None:
        return None
    required_inputs = ("target_ids", "target_mask", "loss", "ntokens")
    if not all(name in abi_map for name in required_inputs):
        return None
    suffix_input_abi = getattr(
        prim_func,
        "_cppmega_path_c_train_step_suffix_loss_input_abi",
        {},
    )
    output_abi = getattr(
        prim_func,
        "_cppmega_path_c_train_step_output_abi",
        {},
    )
    if isinstance(output_abi, Mapping) and not bool(
        output_abi.get("outputs_computed", False)
    ):
        return None

    first_in_region_layer_index_fn = getattr(
        model, "path_c_fused_first_in_region_layer_index", None
    )
    if not callable(first_in_region_layer_index_fn):
        return None
    first_in_region_layer_index = first_in_region_layer_index_fn(
        sequence_length=sequence_length
    )
    if first_in_region_layer_index is None:
        return None
    first_in_region_layer_index_int = int(cast(int, first_in_region_layer_index))

    parameter_order = tuple(sorted(in_region_parameter_bank_aliases.keys()))
    try:
        fused_suffix = build_fused_suffix_custom_function(
            artifact=artifact,
            bank_owner=bank_owner,
            abi_map=abi_map,
            hidden_entry_logical_name=hidden_entry_logical_name,
            target_ids_logical_name="target_ids",
            target_mask_logical_name="target_mask",
            loss_logical_name="loss",
            ntokens_logical_name="ntokens",
            in_region_parameter_bank_aliases=in_region_parameter_bank_aliases,
            parameter_order=parameter_order,
        )
    except Exception:
        return None
    model.attach_path_c_fused_suffix_custom_function(
        fused_suffix,
        parameter_order=parameter_order,
        first_in_region_layer_index=first_in_region_layer_index_int,
    )

    def _loss_fn(loss_model: Any, batch: Mapping[str, Any]) -> tuple[mx.array, mx.array]:
        # MLX nn.value_and_grad always calls loss_fn(model, batch). Forward
        # the call to the model's bound suffix-loss helper.
        return loss_model.path_c_fused_suffix_loss(batch)

    return _loss_fn


def _build_path_c_fused_replay_loss_fn_for_model(
    *,
    model: Any,
    artifact: Any,
    bank_owner: Any,
    in_region_parameter_bank_aliases: Mapping[str, Mapping[str, Any]],
    sequence_length: int | None,
) -> Callable[[Any, Mapping[str, Any]], tuple[mx.array, mx.array]] | None:
    """Attach replay/cotangent custom function for the fused block boundary."""

    from cppmega_mlx.training.path_c_fused_replay import (
        build_fused_replay_boundary_custom_function,
    )

    if not callable(getattr(model, "path_c_fused_replay_loss", None)):
        return None
    if not callable(getattr(model, "attach_path_c_fused_replay_custom_function", None)):
        return None

    prim_func = model.path_c_fused_train_block_prim_func(
        sequence_length=sequence_length,
    )
    if prim_func is None:
        return None
    abi_map = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
        or {}
    )
    if not abi_map:
        return None
    hidden_entry_logical_name: str | None = None
    for name in abi_map:
        if (
            "_hidden" in name
            and not name.endswith("_grad")
            and not name.endswith("_after")
            and not name.endswith("_after_grad")
        ):
            hidden_entry_logical_name = name
            break
    if hidden_entry_logical_name is None:
        return None
    boundary_abi = dict(
        getattr(
            prim_func,
            "_cppmega_path_c_train_step_loss_cotangent_abi",
            {},
        )
        or {}
    )
    boundary_outputs = tuple(
        str(name) for name in boundary_abi.get("source_logical_buffers", ())
    )
    boundary_cotangents = tuple(
        str(name) for name in boundary_abi.get("logical_cotangent_buffers", ())
    )
    if (
        not boundary_outputs
        or len(boundary_outputs) != len(boundary_cotangents)
        or any(name not in abi_map for name in boundary_outputs)
        or any(name not in abi_map for name in boundary_cotangents)
    ):
        return None

    first_in_region_layer_index_fn = getattr(
        model, "path_c_fused_first_in_region_layer_index", None
    )
    if not callable(first_in_region_layer_index_fn):
        return None
    first_in_region_layer_index = first_in_region_layer_index_fn(
        sequence_length=sequence_length
    )
    if first_in_region_layer_index is None:
        return None
    first_in_region_layer_index_int = int(cast(int, first_in_region_layer_index))

    parameter_order = tuple(sorted(in_region_parameter_bank_aliases.keys()))
    try:
        fused_replay = build_fused_replay_boundary_custom_function(
            artifact=artifact,
            bank_owner=bank_owner,
            abi_map=abi_map,
            hidden_entry_logical_name=hidden_entry_logical_name,
            boundary_output_logical_names=boundary_outputs,
            boundary_cotangent_logical_names=boundary_cotangents,
            in_region_parameter_bank_aliases=in_region_parameter_bank_aliases,
            parameter_order=parameter_order,
            row_chunk_count=getattr(
                artifact,
                "_cppmega_path_c_row_chunk_count",
                getattr(artifact, "row_chunk_count", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_row_chunk_count",
                None,
            ),
            row_chunk_index_param=getattr(
                artifact,
                "_cppmega_path_c_row_chunk_index_param",
                getattr(artifact, "row_chunk_index_param", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_row_chunk_index_param",
                None,
            ),
            row_subchunk_count=getattr(
                artifact,
                "_cppmega_path_c_row_subchunk_count",
                getattr(artifact, "row_subchunk_count", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_row_subchunk_count",
                None,
            ),
            row_subchunk_index_param=getattr(
                artifact,
                "_cppmega_path_c_row_subchunk_index_param",
                getattr(artifact, "row_subchunk_index_param", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_row_subchunk_index_param",
                None,
            ),
            backward_stage_count=getattr(
                artifact,
                "_cppmega_path_c_backward_stage_count",
                getattr(artifact, "backward_stage_count", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_backward_stage_count",
                None,
            ),
            backward_stage_index_param=getattr(
                artifact,
                "_cppmega_path_c_backward_stage_index_param",
                getattr(artifact, "backward_stage_index_param", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_backward_stage_index_param",
                None,
            ),
            backward_gate_param=getattr(
                artifact,
                "_cppmega_path_c_backward_gate_param",
                getattr(artifact, "backward_gate_param", None),
            )
            or getattr(
                prim_func,
                "_cppmega_path_c_backward_gate_param",
                "path_c_run_backward",
            ),
        )
    except Exception:
        return None
    model.attach_path_c_fused_replay_custom_function(
        fused_replay,
        parameter_order=parameter_order,
        first_in_region_layer_index=first_in_region_layer_index_int,
        boundary_output_logical_names=boundary_outputs,
    )

    def _loss_fn(loss_model: Any, batch: Mapping[str, Any]) -> tuple[mx.array, mx.array]:
        return loss_model.path_c_fused_replay_loss(batch)

    return _loss_fn


def _path_c_fused_train_block_training_runtime_from_artifact(
    *,
    artifact: Any,
    bank_owner: Any,
    runtime_owner: str,
    model: Any | None = None,
    sequence_length: int | None = None,
) -> Any | None:
    """Build the training runtime that owns m04's fused Path C train step.

    Today HybridTinyLM's fused region only covers a subset of trainable
    parameters (3 layers of 16, no embedding / lm_head), so the artifact's
    own value_and_grad_contract reports returns_full_model_grads=False. The
    strict PathCFusedTrainBlockTrainingRuntime then trips the m04 install
    gate's FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_INCOMPLETE check.

    The mixed-mode PathCFusedPlusEagerTrainingRuntime takes the same
    artifact and bank owner, and routes value_and_grad through the
    trainer-supplied eager closure for residual parameters. It surfaces a
    closed full-model gradient contract so the gate flips to 'ok' without
    monkeypatching or hidden allocation.

    When ``model`` is provided and exposes the parameter-bank residency
    surface (``bind_path_c_in_region_parameter_views_into_bank`` plus
    ``sync_path_c_in_region_parameters_into_bank``), this function also
    binds the in-region trainable parameters as zero-copy views into the
    model-owned bank and passes the alias map plus a per-step sync
    callable into the runtime. That activates merged-gradient mode: the
    fused artifact populates bank-resident gradient slots, the eager
    closure runs for the residual parameters, and the runtime returns one
    merged gradient tree backed by the same bank storage the kernel reads
    and writes. When the model lacks the bank-residency surface, the
    runtime falls back to warmup-only mode and the eager closure remains
    the source of every gradient.

    When the artifact already returns full-coverage grads (no in-region
    gaps), the strict runtime is used instead so the artifact retains
    exclusive ownership of training execution.
    """
    if not _path_c_fused_train_block_artifact_has_training_runtime_contract(
        artifact
    ):
        return None
    inner_contract: Mapping[str, Any] = {}
    contract_fn = getattr(artifact, "value_and_grad_contract", None)
    if callable(contract_fn):
        try:
            raw = contract_fn()
        except Exception:
            raw = None
        if isinstance(raw, Mapping):
            inner_contract = raw
    returns_full = bool(inner_contract.get("returns_full_model_grads", False))
    in_region_names = tuple(
        sorted(
            str(name)
            for name in inner_contract.get("full_model_gradient_coverage", {}).get(
                "covered_parameter_names", ()
            )
        )
    )
    in_region_aliases: dict[str, dict[str, Any]] | None = None
    sync_callable: Callable[[], Mapping[str, Any]] | None = None
    bind_report: dict[str, Any] | None = None
    fused_suffix_loss_fn: Callable[
        [Any, Mapping[str, Any]], tuple[mx.array, mx.array]
    ] | None = None
    fused_replay_loss_fn: Callable[
        [Any, Mapping[str, Any]], tuple[mx.array, mx.array]
    ] | None = None
    if model is not None and bank_owner is not None:
        binder = getattr(
            model,
            "bind_path_c_in_region_parameter_views_into_bank",
            None,
        )
        sync_fn = getattr(
            model,
            "sync_path_c_in_region_parameters_into_bank",
            None,
        )
        alias_fn = getattr(
            model,
            "path_c_fused_in_region_parameter_bank_aliases",
            None,
        )
        if (
            callable(binder)
            and callable(sync_fn)
            and callable(alias_fn)
        ):
            try:
                raw_bind_report = binder(
                    bank_owner, sequence_length=sequence_length
                )
            except Exception as exc:
                raw_bind_report = {
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "bound": [],
                    "skipped": [],
                    "in_region_parameter_count": 0,
                }
            if isinstance(raw_bind_report, Mapping):
                bind_report = {
                    str(key): value for key, value in raw_bind_report.items()
                }
            else:
                bind_report = None
            if (
                bind_report is not None
                and bind_report.get("status") in {"ok", "partial"}
            ):
                raw_aliases = bind_report.get(
                    "in_region_parameter_bank_aliases", {}
                ) or {}
                if isinstance(raw_aliases, Mapping):
                    in_region_aliases = {
                        str(name): (
                            {str(k): v for k, v in dict(info).items()}
                            if isinstance(info, Mapping)
                            else {}
                        )
                        for name, info in raw_aliases.items()
                    }
                else:
                    in_region_aliases = {}
                if not in_region_aliases:
                    # Refresh from the model if the bind report did not carry it.
                    try:
                        raw_aliases = alias_fn(
                            sequence_length=sequence_length
                        )
                    except Exception:
                        raw_aliases = None
                    if isinstance(raw_aliases, Mapping):
                        in_region_aliases = {
                            str(name): (
                                {str(k): v for k, v in dict(info).items()}
                                if isinstance(info, Mapping)
                                else {}
                            )
                            for name, info in raw_aliases.items()
                        }
                if in_region_aliases:
                    raw_bound = bind_report.get("bound", ()) or ()
                    bound_names = tuple(
                        sorted(
                            str(name) for name in cast(
                                Sequence[Any], raw_bound
                            )
                        )
                    )
                    if bound_names:
                        in_region_names = bound_names

                    aliases_for_sync = in_region_aliases

                    def _sync() -> Mapping[str, Any]:
                        report = sync_fn(
                            bank_owner,
                            in_region_aliases=aliases_for_sync,
                            sequence_length=sequence_length,
                        )
                        if isinstance(report, Mapping):
                            return cast(Mapping[str, Any], report)
                        return {
                            "status": "ok",
                            "reason": (
                                "sync callable returned non-mapping value"
                            ),
                            "synced": [],
                            "skipped": [],
                        }

                    sync_callable = _sync
                    fused_replay_loss_fn = (
                        _build_path_c_fused_replay_loss_fn_for_model(
                            model=model,
                            artifact=artifact,
                            bank_owner=bank_owner,
                            in_region_parameter_bank_aliases=in_region_aliases,
                            sequence_length=sequence_length,
                        )
                    )
                else:
                    in_region_aliases = None
    try:
        if returns_full:
            return PathCFusedTrainBlockTrainingRuntime(
                artifact=artifact,
                bank_owner=bank_owner,
                owner_name=runtime_owner,
            )
        runtime = PathCFusedPlusEagerTrainingRuntime(
            artifact=artifact,
            bank_owner=bank_owner,
            owner_name=runtime_owner,
            in_region_parameter_names=in_region_names,
            in_region_parameter_bank_aliases=in_region_aliases,
            model_bank_sync_callable=sync_callable,
            fused_suffix_loss_fn=fused_suffix_loss_fn,
            fused_replay_loss_fn=fused_replay_loss_fn,
        )
        if bind_report is not None:
            runtime.last_parameter_bank_bind_report = dict(bind_report)
        return runtime
    except TypeError:
        return None


def _path_c_physical_abi_bank_plan_payload(
    *,
    physical_abi_map: Mapping[str, Any],
    physical_abi_shapes: Mapping[str, Any],
    owner_name: str | None,
) -> dict[str, Any]:
    """Describe required caller/model-owned physical ABI banks without allocation."""

    try:
        specs = physical_abi_bank_specs(physical_abi_map, physical_abi_shapes)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": str(exc),
            "owner_attribute": "path_c_physical_abi_bank_owner",
            "owner_name": owner_name,
            "allocation_required_before_binding": False,
            "required_bank_buffers": [],
            "bank_count": 0,
            "bank_specs": [],
            "total_elements": 0,
            "total_nbytes": 0,
            "hidden_packing_performed": False,
            "no_hidden_allocation_policy": True,
        }

    bank_specs: list[dict[str, Any]] = []
    for spec in specs:
        logical_buffers = [str(name) for name in spec.logical_buffers]
        bank_specs.append(
            {
                "name": spec.name,
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "elements": spec.elements,
                "nbytes": spec.nbytes,
                "logical_buffer_count": len(logical_buffers),
                "logical_buffers": logical_buffers,
            }
        )
    total_elements = sum(int(spec["elements"]) for spec in bank_specs)
    total_nbytes = sum(int(spec["nbytes"]) for spec in bank_specs)
    return {
        "status": "model_owned_physical_abi_banks_required"
        if bank_specs
        else "no_physical_abi_banks_required",
        "reason": (
            "generated fused train-block ABI requires the model or caller to "
            "own these physical bank buffers before runtime binding; hidden "
            "packing and implicit allocation remain forbidden"
            if bank_specs
            else "generated fused train-block ABI did not require physical banks"
        ),
        "owner_attribute": "path_c_physical_abi_bank_owner",
        "owner_name": owner_name,
        "allocation_required_before_binding": bool(bank_specs),
        "required_bank_buffers": [str(spec["name"]) for spec in bank_specs],
        "bank_count": len(bank_specs),
        "bank_specs": bank_specs,
        "total_elements": total_elements,
        "total_nbytes": total_nbytes,
        "hidden_packing_performed": False,
        "no_hidden_allocation_policy": True,
    }


def path_c_fusion_runtime_training_binding_payload(
    *,
    region: Any,
    schedule_target: Any,
    bank_buffers: Mapping[str, Any] | None = None,
    bank_buffer_owner: str | None = None,
    bank_owner: Any | None = None,
    fused_artifact: Any | None = None,
) -> dict[str, Any]:
    """Return executable-runtime binding status for the fused train-block."""

    try:
        resolved_bank_buffers = bank_buffers
        resolved_bank_owner = bank_buffer_owner
        if bank_owner is not None:
            if bank_buffers is not None:
                raise ValueError(
                    "bank_owner and bank_buffers are mutually exclusive"
                )
            resolved_bank_buffers = getattr(bank_owner, "buffers", None)
            resolved_bank_owner = str(
                getattr(bank_owner, "owner_name", resolved_bank_owner)
            )
        artifact_abi_map = getattr(fused_artifact, "physical_abi_map", None)
        artifact_abi_shapes = getattr(fused_artifact, "physical_abi_shapes", None)
        if artifact_abi_map is not None and artifact_abi_shapes is not None:
            physical_abi_map = dict(artifact_abi_map)
            physical_abi_shapes = dict(artifact_abi_shapes)
            physical_abi_policy = "fused_artifact_physical_abi"
        else:
            prim_func = schedule_target.schedule_template(region)
            physical_abi_map = dict(
                getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
                or {}
            )
            physical_abi_shapes = dict(
                getattr(
                    prim_func,
                    "_cppmega_path_c_physical_buffer_abi_shapes",
                    {},
                )
                or {}
            )
            physical_abi_policy = str(
                getattr(prim_func, "_cppmega_path_c_physical_abi_policy", "unknown")
            )
        bridge = plan_physical_abi_runtime_bridge(
            physical_abi_map,
            physical_abi_shapes,
        )
        bank_plan = _path_c_physical_abi_bank_plan_payload(
            physical_abi_map=physical_abi_map,
            physical_abi_shapes=physical_abi_shapes,
            owner_name=resolved_bank_owner,
        )
        binding = validate_physical_abi_runtime_bindings(
            physical_abi_map,
            physical_abi_shapes,
            resolved_bank_buffers,
        )
        required_bank_buffers = list(bridge.get("required_bank_buffers", ()))
        missing_bank_buffers = list(binding.get("missing_bank_buffers", ()))
        ordered_kernel_buffers = list(binding.get("ordered_kernel_buffers", ()))
        provided_names = set(str(name) for name in (resolved_bank_buffers or {}))
        provided_bank_buffers = [
            name for name in ordered_kernel_buffers if name in provided_names
        ]
        physical_abi_binding_ready = (
            binding.get("status") == "ok" and not missing_bank_buffers
        )
        fused_artifact_bound = callable(fused_artifact)
        runtime_uses_fused_train_block = (
            physical_abi_binding_ready and fused_artifact_bound
        )
        status = (
            "ok"
            if runtime_uses_fused_train_block
            else FP8_PATH_C_FUSED_TRAIN_BLOCK_ARTIFACT_MISSING_STATUS
            if physical_abi_binding_ready
            else FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS
        )
        return {
            "status": status,
            "binding_status": binding.get("status"),
            "bridge_status": bridge.get("status"),
            "physical_abi_binding_ready": physical_abi_binding_ready,
            "fused_artifact_bound": fused_artifact_bound,
            "runtime_uses_fused_train_block": runtime_uses_fused_train_block,
            "physical_abi_policy": physical_abi_policy,
            "required_bank_buffers": required_bank_buffers,
            "missing_bank_buffers": missing_bank_buffers,
            "shape_mismatch_buffers": list(
                binding.get("shape_mismatch_buffers", ())
            ),
            "dtype_mismatch_buffers": list(
                binding.get("dtype_mismatch_buffers", ())
            ),
            "unexpected_buffers": list(binding.get("unexpected_buffers", ())),
            "provided_bank_buffers": provided_bank_buffers,
            "bank_buffer_owner": resolved_bank_owner,
            "model_owned_physical_abi_bank_plan": bank_plan,
            "hidden_packing_performed": False,
            "no_hidden_allocation_policy": bool(
                bridge.get("no_hidden_allocation_policy", True)
            ),
            "reason": (
                "caller/model-owned physical ABI banks and callable fused "
                "train-block artifact are bound in generated kernel argument order"
                if runtime_uses_fused_train_block
                else
                "caller/model-owned physical ABI banks are bound, but no callable "
                "fused train-block artifact is attached to the training runtime"
                if physical_abi_binding_ready
                else
                "fused train-block runtime requires caller/model-owned physical "
                "ABI banks; the current m04/HybridTinyLM route still owns "
                "separate tensors, and hidden packing or copying is forbidden"
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive receipt metadata
        return {
            "status": "fused_train_block_runtime_binding_unavailable",
            "binding_status": "unavailable",
            "bridge_status": "unavailable",
            "physical_abi_binding_ready": False,
            "fused_artifact_bound": False,
            "runtime_uses_fused_train_block": False,
            "physical_abi_policy": "unknown",
            "required_bank_buffers": [],
            "missing_bank_buffers": [],
            "provided_bank_buffers": [],
            "bank_buffer_owner": bank_buffer_owner
            if bank_owner is None
            else str(getattr(bank_owner, "owner_name", bank_buffer_owner)),
            "model_owned_physical_abi_bank_plan": {
                "status": "unavailable",
                "reason": str(exc),
                "owner_attribute": "path_c_physical_abi_bank_owner",
                "owner_name": bank_buffer_owner
                if bank_owner is None
                else str(getattr(bank_owner, "owner_name", bank_buffer_owner)),
                "allocation_required_before_binding": False,
                "required_bank_buffers": [],
                "bank_count": 0,
                "bank_specs": [],
                "total_elements": 0,
                "total_nbytes": 0,
                "hidden_packing_performed": False,
                "no_hidden_allocation_policy": True,
            },
            "hidden_packing_performed": False,
            "no_hidden_allocation_policy": True,
            "reason": str(exc),
        }


def _path_c_fused_train_block_training_abi_contract_payload(
    prim_func: Any | None,
) -> dict[str, Any]:
    if prim_func is None:
        return {
            "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "status": "unavailable",
            "can_back_value_and_grad": False,
            "loss_output_available": False,
            "ntokens_output_available": False,
            "returns_model_grads": False,
            "returns_full_model_grads": False,
            "train_step_output_abi_declared": False,
            "train_step_suffix_loss_input_abi_declared": False,
            "train_step_outputs_computed": False,
            "train_step_computed_outputs": [],
            "train_step_pending_outputs": [],
            "train_step_loss_source_buffers": [],
            "train_step_loss_cotangents_computed": False,
            "train_step_loss_cotangent_abi": {},
            "replay_cotangent_boundary_available": False,
            "replay_boundary_source_buffers": [],
            "replay_boundary_cotangent_buffers": [],
            "train_step_suffix_loss_parameter_grads_computed": False,
            "train_step_suffix_loss_parameter_grad_abi": {},
            "train_step_suffix_loss_parameter_gradient_buffers": [],
            "missing_train_step_suffix_loss_parameter_gradient_buffers": [
                "final_norm_weight_grad",
                "lm_head_weight_grad",
            ],
            "suffix_loss_inputs_available": False,
            "logical_buffer_count": 0,
            "kernel_parameter_count": 0,
            "gradient_output_count": 0,
            "missing_value_and_grad_outputs": ["loss", "ntokens", "model_grads"],
            "missing_suffix_loss_inputs": [
                "target_ids",
                "target_mask",
                "final_norm_weight",
                "lm_head_weight",
            ],
            "loss_output_candidates": [],
            "ntokens_output_candidates": [],
            "sample_gradient_outputs": [],
            "reason": "generated fused train-block PrimFunc was unavailable",
        }
    physical_abi_map = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {}) or {}
    )
    physical_abi_shapes = dict(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes", {}) or {}
    )
    train_step_output_abi = dict(
        getattr(prim_func, "_cppmega_path_c_train_step_output_abi", {}) or {}
    )
    train_step_suffix_loss_input_abi = dict(
        getattr(prim_func, "_cppmega_path_c_train_step_suffix_loss_input_abi", {})
        or {}
    )
    train_step_output_abi_declared = bool(train_step_output_abi.get("declared"))
    train_step_suffix_loss_input_abi_declared = bool(
        train_step_suffix_loss_input_abi.get("declared")
    )
    train_step_outputs_computed = bool(
        train_step_output_abi.get("outputs_computed")
    )
    train_step_computed_outputs = [
        str(name)
        for name in train_step_output_abi.get("computed_logical_outputs", ())
    ]
    train_step_pending_outputs = [
        str(name)
        for name in train_step_output_abi.get("pending_logical_outputs", ())
    ]
    train_step_loss_source_buffers = [
        str(name)
        for name in (
            getattr(
                prim_func,
                "_cppmega_path_c_train_step_suffix_loss_source_buffers",
                (),
            )
            or ()
        )
    ]
    train_step_loss_cotangent_abi = dict(
        getattr(
            prim_func,
            "_cppmega_path_c_train_step_loss_cotangent_abi",
            {},
        )
        or {}
    )
    train_step_loss_cotangents_computed = bool(
        train_step_loss_cotangent_abi.get("cotangents_computed")
    )
    replay_boundary_source_buffers = [
        str(name)
        for name in train_step_loss_cotangent_abi.get(
            "source_logical_buffers",
            (),
        )
    ]
    replay_boundary_cotangent_buffers = [
        str(name)
        for name in train_step_loss_cotangent_abi.get(
            "logical_cotangent_buffers",
            (),
        )
    ]
    replay_cotangent_boundary_available = bool(
        replay_boundary_source_buffers
        and len(replay_boundary_source_buffers)
        == len(replay_boundary_cotangent_buffers)
        and all(name in physical_abi_map for name in replay_boundary_source_buffers)
        and all(name in physical_abi_map for name in replay_boundary_cotangent_buffers)
    )
    train_step_suffix_loss_parameter_grad_abi = dict(
        getattr(
            prim_func,
            "_cppmega_path_c_train_step_suffix_loss_parameter_grad_abi",
            {},
        )
        or {}
    )
    train_step_suffix_loss_parameter_gradient_buffers = [
        str(name)
        for name in train_step_suffix_loss_parameter_grad_abi.get(
            "logical_gradient_buffers",
            (),
        )
    ]
    missing_train_step_suffix_loss_parameter_gradient_buffers = [
        name
        for name in train_step_suffix_loss_parameter_gradient_buffers
        if name not in physical_abi_map
    ]
    train_step_suffix_loss_parameter_grads_computed = bool(
        train_step_suffix_loss_parameter_grad_abi.get("gradients_computed")
        and train_step_suffix_loss_parameter_gradient_buffers
        and not missing_train_step_suffix_loss_parameter_gradient_buffers
    )
    suffix_loss_inputs = tuple(
        str(name)
        for name in train_step_suffix_loss_input_abi.get("logical_inputs", ())
    )
    logical_names = tuple(str(name) for name in physical_abi_map)

    def value_output_candidates(semantic_name: str) -> list[str]:
        accepted_leaf_names = {
            "loss": {"loss", "train_loss", "total_loss"},
            "ntokens": {"ntokens", "num_tokens", "token_count"},
        }[semantic_name]
        matches: list[str] = []
        for logical_name in logical_names:
            leaf = logical_name.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
            if leaf not in accepted_leaf_names:
                continue
            entry = physical_abi_map.get(logical_name, {})
            size = entry.get("size") if isinstance(entry, Mapping) else None
            shape = entry.get("shape") if isinstance(entry, Mapping) else None
            if size == 1 or tuple(shape or ()) in ((), (1,)):
                matches.append(logical_name)
        return sorted(matches)

    gradient_outputs = sorted(
        logical_name for logical_name in logical_names if logical_name.endswith("_grad")
    )
    loss_outputs = value_output_candidates("loss")
    ntokens_outputs = value_output_candidates("ntokens")
    missing_outputs: list[str] = []
    if not loss_outputs and not replay_cotangent_boundary_available:
        missing_outputs.append("loss")
    if not ntokens_outputs and not replay_cotangent_boundary_available:
        missing_outputs.append("ntokens")
    if not gradient_outputs:
        missing_outputs.append("model_grads")
    if (
        not replay_cotangent_boundary_available
        and not train_step_suffix_loss_parameter_grads_computed
    ):
        missing_outputs.append("suffix_parameter_grads")
    missing_suffix_loss_inputs = [
        name for name in suffix_loss_inputs if name not in physical_abi_map
    ]
    suffix_loss_inputs_available = (
        train_step_suffix_loss_input_abi_declared
        and bool(suffix_loss_inputs)
        and not missing_suffix_loss_inputs
    )
    legacy_suffix_value_and_grad = (
        not missing_outputs
        and suffix_loss_inputs_available
        and train_step_outputs_computed
        and train_step_loss_cotangents_computed
        and train_step_suffix_loss_parameter_grads_computed
    )
    can_back_value_and_grad = bool(
        legacy_suffix_value_and_grad
        or (
            replay_cotangent_boundary_available
            and bool(gradient_outputs)
        )
    )
    return {
        "contract": PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
        "status": "ok" if can_back_value_and_grad else "incomplete",
        "can_back_value_and_grad": can_back_value_and_grad,
        "loss_output_available": bool(loss_outputs),
        "ntokens_output_available": bool(ntokens_outputs),
        "returns_model_grads": bool(gradient_outputs),
        "returns_full_model_grads": False,
        "train_step_output_abi_declared": train_step_output_abi_declared,
        "train_step_suffix_loss_input_abi_declared": (
            train_step_suffix_loss_input_abi_declared
        ),
        "train_step_outputs_computed": train_step_outputs_computed,
        "train_step_computed_outputs": train_step_computed_outputs,
        "train_step_pending_outputs": train_step_pending_outputs,
        "train_step_loss_source_buffers": train_step_loss_source_buffers,
        "train_step_loss_cotangents_computed": train_step_loss_cotangents_computed,
        "train_step_loss_cotangent_abi": train_step_loss_cotangent_abi,
        "replay_cotangent_boundary_available": replay_cotangent_boundary_available,
        "replay_boundary_source_buffers": replay_boundary_source_buffers,
        "replay_boundary_cotangent_buffers": replay_boundary_cotangent_buffers,
        "train_step_suffix_loss_parameter_grads_computed": (
            train_step_suffix_loss_parameter_grads_computed
        ),
        "train_step_suffix_loss_parameter_grad_abi": (
            train_step_suffix_loss_parameter_grad_abi
        ),
        "train_step_suffix_loss_parameter_gradient_buffers": (
            train_step_suffix_loss_parameter_gradient_buffers
        ),
        "missing_train_step_suffix_loss_parameter_gradient_buffers": (
            missing_train_step_suffix_loss_parameter_gradient_buffers
        ),
        "train_step_output_abi": train_step_output_abi,
        "train_step_suffix_loss_input_abi": train_step_suffix_loss_input_abi,
        "suffix_loss_inputs_available": suffix_loss_inputs_available,
        "logical_buffer_count": len(logical_names),
        "kernel_parameter_count": len(physical_abi_shapes),
        "gradient_output_count": len(gradient_outputs),
        "missing_value_and_grad_outputs": missing_outputs,
        "missing_suffix_loss_inputs": missing_suffix_loss_inputs,
        "loss_output_candidates": loss_outputs,
        "ntokens_output_candidates": ntokens_outputs,
        "loss_outputs_source": (
            "fused_train_step_output_abi"
            if loss_outputs and ntokens_outputs
            else "eager_suffix_replay_cotangent_bridge"
            if replay_cotangent_boundary_available
            else "unavailable"
        ),
        "sample_gradient_outputs": gradient_outputs[:8],
        "reason": (
            "generated fused train-block ABI exposes replay boundary outputs, "
            "cotangent seed buffers, and model-gradient buffers; loss and "
            "ntokens are computed by the eager suffix outside the TileLang "
            "artifact"
            if replay_cotangent_boundary_available and gradient_outputs
            else
            "generated fused train-block ABI exposes loss, ntokens, and gradient "
            "buffers for value_and_grad"
            if can_back_value_and_grad
            else "generated fused train-block ABI declares scalar outputs, but "
            "is missing suffix loss inputs required to compute them"
            if not missing_outputs and not suffix_loss_inputs_available
            else "generated fused train-block ABI computes loss and ntokens, "
            "but suffix loss cotangents are not generated into the backward "
            "seed buffers yet"
            if (
                not missing_outputs
                and suffix_loss_inputs_available
                and train_step_outputs_computed
                and not train_step_loss_cotangents_computed
            )
            else "generated fused train-block ABI computes suffix loss "
            "cotangents, but final norm and lm-head suffix parameter gradients "
            "are not generated into model-gradient buffers yet"
            if (
                not missing_outputs
                and suffix_loss_inputs_available
                and train_step_outputs_computed
                and train_step_loss_cotangents_computed
                and not train_step_suffix_loss_parameter_grads_computed
            )
            else "generated fused train-block ABI computes ntokens from "
            "target_mask, but loss suffix codegen has not populated loss yet"
            if not missing_outputs and train_step_computed_outputs
            else "generated fused train-block ABI declares the train-step "
            "loss/ntokens scalar slots, but suffix loss codegen has not populated "
            "them yet"
            if not missing_outputs
            else "generated fused train-block ABI exposes gradient buffers but "
            "does not yet expose all outputs required for a single-kernel "
            "train-step value_and_grad runtime"
        ),
    }


def _path_c_fused_train_block_plan_payload(plan: Any) -> dict[str, Any]:
    contract = getattr(plan, "schedule_contract", None)
    return {
        "region_name": getattr(plan, "region_name", None),
        "schedule_name": getattr(plan, "schedule_name", None),
        "schedule_status": getattr(plan, "schedule_status", None),
        "single_kernel_fused": bool(getattr(plan, "single_kernel_fused", False)),
        "backward_graph": getattr(plan, "backward_graph", None),
        "autograd_status": getattr(plan, "autograd_status", None),
        "schedule_contract_status": getattr(contract, "status", None),
        "declared_required_real_abi_inputs": list(
            getattr(contract, "declared_required_real_abi_inputs", ()) or ()
        ),
        "missing_real_abi_inputs": list(
            getattr(contract, "missing_real_abi_inputs", ()) or ()
        ),
    }


def _path_c_fused_train_block_selected_region_metadata(region: Any) -> dict[str, Any]:
    metadata = getattr(region, "metadata", {})
    bricks = (
        tuple(metadata.get("path_c_bricks", ()))
        if isinstance(metadata, Mapping)
        else ()
    )
    brick_names = [
        str(brick.get("name"))
        for brick in bricks
        if isinstance(brick, Mapping) and brick.get("name")
    ]
    brick_route_symbols = [
        str(brick.get("route_symbol"))
        for brick in bricks
        if isinstance(brick, Mapping) and brick.get("route_symbol")
    ]
    node_names = [str(name) for name in getattr(region, "node_names", ())]
    return {
        "name": getattr(region, "name", None),
        "node_count": len(getattr(region, "nodes", ()) or ()),
        "edge_count": len(getattr(region, "edges", ()) or ()),
        "first_node_name": node_names[0] if node_names else None,
        "last_node_name": node_names[-1] if node_names else None,
        "brick_names": brick_names,
        "first_brick_name": brick_names[0] if brick_names else None,
        "last_brick_name": brick_names[-1] if brick_names else None,
        "brick_route_symbols": brick_route_symbols,
    }


def _path_c_fused_train_block_artifact_compile_report(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    report = dict(payload)
    artifact = report.pop("artifact", None)
    report["artifact_bound"] = callable(artifact)
    report["artifact_type"] = (
        type(artifact).__name__ if artifact is not None else None
    )
    return report


def _path_c_generated_stage_schedule_template(
    *,
    schedule_target: Any,
    region: Any,
    abi_prim_func: Any,
    execution_stage: str,
    active_node_names: Sequence[str] | None = None,
    stage_suffix: str = "",
    row_dispatch_mode: str = DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    rows_per_kernel_launch: int | None = None,
) -> Callable[[Any], Any]:
    """Build a stage PrimFunc template with the full train-block ABI map."""
    return make_path_c_descriptor_stage_schedule_template(
        schedule_target=schedule_target,
        region=region,
        abi_prim_func=abi_prim_func,
        execution_stage=execution_stage,
        active_node_names=active_node_names,
        stage_suffix=stage_suffix,
        row_dispatch_mode=row_dispatch_mode,
        rows_per_kernel_launch=(
            DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH
            if rows_per_kernel_launch is None
            else int(rows_per_kernel_launch)
        ),
    )


def _path_c_kernel_buffer_shapes(prim_func: Any) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    buffer_map = getattr(prim_func, "buffer_map", {}) or {}
    for param in tuple(getattr(prim_func, "params", ())):
        buffer = buffer_map.get(param)
        if buffer is None:
            continue
        name = str(getattr(buffer, "name", param))
        shape = tuple(int(dim) for dim in tuple(getattr(buffer, "shape", ())))
        shapes[name] = shape
    return shapes


def _path_c_merged_physical_abi_for_prim_funcs(
    prim_funcs: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    return merge_path_c_physical_abi_for_prim_funcs(prim_funcs)


_PATH_C_RECURRENT_BACKWARD_OPS = frozenset({"m2rnn_bwd", "mamba3_mimo_bwd"})


def _path_c_int_prim_attr(prim_func: Any, attr_name: str) -> int | None:
    raw_value = getattr(prim_func, attr_name, None)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _path_c_str_prim_attr(prim_func: Any, attr_name: str) -> str | None:
    raw_value = getattr(prim_func, attr_name, None)
    if raw_value is None:
        return None
    text = str(raw_value)
    return text or None


def _path_c_first_str_prim_attr(
    prim_funcs: Sequence[Any],
    attr_name: str,
) -> str | None:
    for prim_func in prim_funcs:
        value = _path_c_str_prim_attr(prim_func, attr_name)
        if value:
            return value
    return None


def _path_c_max_int_prim_attr(
    prim_funcs: Sequence[Any],
    attr_name: str,
) -> int | None:
    values = tuple(
        value
        for prim_func in prim_funcs
        if (value := _path_c_int_prim_attr(prim_func, attr_name)) is not None
    )
    return max(values) if values else None


def _path_c_min_int_prim_attr(
    prim_funcs: Sequence[Any],
    attr_name: str,
) -> int | None:
    values = tuple(
        value
        for prim_func in prim_funcs
        if (value := _path_c_int_prim_attr(prim_func, attr_name)) is not None
    )
    return min(values) if values else None


def _path_c_generated_stage_row_launch_attrs(
    prim_funcs: Sequence[Any],
) -> dict[str, Any]:
    """Extract launcher-chunk scalar ABI metadata from generated stage PrimFuncs."""

    return {
        "row_chunk_count": _path_c_max_int_prim_attr(
            prim_funcs,
            "_cppmega_path_c_row_chunk_count",
        ),
        "row_chunk_index_param": _path_c_first_str_prim_attr(
            prim_funcs,
            "_cppmega_path_c_row_chunk_index_param",
        ),
        "row_subchunk_count": _path_c_min_int_prim_attr(
            prim_funcs,
            "_cppmega_path_c_row_subchunk_count",
        ),
        "row_subchunk_index_param": _path_c_first_str_prim_attr(
            prim_funcs,
            "_cppmega_path_c_row_subchunk_index_param",
        ),
        "rows_per_kernel_launch": _path_c_max_int_prim_attr(
            prim_funcs,
            "_cppmega_path_c_rows_per_kernel_launch",
        ),
        "backward_gate_param": _path_c_first_str_prim_attr(
            prim_funcs,
            "_cppmega_path_c_backward_gate_param",
        ),
    }


def _path_c_stage_row_launch_specs(
    prim_funcs: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for index, prim_func in enumerate(prim_funcs):
        specs.append(
            {
                "stage_index": index,
                "row_chunk_count": _path_c_int_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_chunk_count",
                ),
                "row_chunk_index_param": _path_c_str_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_chunk_index_param",
                ),
                "row_subchunk_count": _path_c_int_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_subchunk_count",
                ),
                "row_subchunk_index_param": _path_c_str_prim_attr(
                    prim_func,
                    "_cppmega_path_c_row_subchunk_index_param",
                ),
                "rows_per_kernel_launch": _path_c_int_prim_attr(
                    prim_func,
                    "_cppmega_path_c_rows_per_kernel_launch",
                ),
            }
        )
    return tuple(specs)


def _path_c_monolithic_recurrent_backward_runtime_blocker(
    prim_func: Any | None,
) -> dict[str, Any] | None:
    """Return a runtime blocker for the scalarized monolithic recurrent bwd form."""

    if prim_func is None:
        return None
    row_dispatch_mode = _path_c_str_prim_attr(
        prim_func,
        "_cppmega_path_c_row_dispatch_mode",
    )
    execution_stage = _path_c_str_prim_attr(
        prim_func,
        "_cppmega_path_c_execution_stage",
    )
    if (
        row_dispatch_mode != DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS
        or execution_stage != DESCRIPTOR_EXECUTION_STAGE_ALL
    ):
        return None
    ops = tuple(
        str(op)
        for op in (getattr(prim_func, "_cppmega_path_c_brick_ops", ()) or ())
    )
    recurrent_backward_ops = tuple(
        op for op in ops if op in _PATH_C_RECURRENT_BACKWARD_OPS
    )
    if not recurrent_backward_ops:
        return None
    backward_stage_count = _path_c_int_prim_attr(
        prim_func,
        "_cppmega_path_c_backward_stage_count",
    )
    backward_stage_index_param = _path_c_str_prim_attr(
        prim_func,
        "_cppmega_path_c_backward_stage_index_param",
    )
    if backward_stage_count is not None and backward_stage_count > 1 and (
        backward_stage_index_param
    ):
        return None
    shape_env = getattr(prim_func, "_cppmega_path_c_shape_env", None)
    sequence_length = getattr(shape_env, "sequence_length", None)
    try:
        sequence_length = int(sequence_length)
    except (TypeError, ValueError):
        sequence_length = None
    return {
        "status": "blocked",
        "kind": "monolithic_grid_chunks_recurrent_backward_scalar_replay",
        "reason": (
            "monolithic grid-chunk recurrent backward scalarizes exact replay "
            "inside one all-stage generated TileLang call; production runtime "
            "must use descriptor launcher-chunk stages instead"
        ),
        "row_dispatch_mode": row_dispatch_mode,
        "execution_stage": execution_stage,
        "row_chunk_count": _path_c_int_prim_attr(
            prim_func,
            "_cppmega_path_c_row_chunk_count",
        ),
        "sequence_length": sequence_length,
        "recurrent_backward_ops": list(dict.fromkeys(recurrent_backward_ops)),
    }


def _path_c_single_launcher_recurrent_backward_runtime_blocker(
    prim_func: Any | None,
) -> dict[str, Any] | None:
    """Return a runtime blocker for all-stage launcher recurrent backward."""

    if prim_func is None:
        return None
    row_dispatch_mode = _path_c_str_prim_attr(
        prim_func,
        "_cppmega_path_c_row_dispatch_mode",
    )
    execution_stage = _path_c_str_prim_attr(
        prim_func,
        "_cppmega_path_c_execution_stage",
    )
    if (
        row_dispatch_mode != DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        or execution_stage != DESCRIPTOR_EXECUTION_STAGE_ALL
    ):
        return None
    ops = tuple(
        str(op)
        for op in (getattr(prim_func, "_cppmega_path_c_brick_ops", ()) or ())
    )
    recurrent_backward_ops = tuple(
        op for op in ops if op in _PATH_C_RECURRENT_BACKWARD_OPS
    )
    if not recurrent_backward_ops:
        return None
    backward_stage_count = _path_c_int_prim_attr(
        prim_func,
        "_cppmega_path_c_backward_stage_count",
    )
    backward_stage_index_param = _path_c_str_prim_attr(
        prim_func,
        "_cppmega_path_c_backward_stage_index_param",
    )
    if backward_stage_count is not None and backward_stage_count > 1 and (
        backward_stage_index_param
    ):
        return None
    shape_env = getattr(prim_func, "_cppmega_path_c_shape_env", None)
    sequence_length = getattr(shape_env, "sequence_length", None)
    try:
        sequence_length = int(sequence_length)
    except (TypeError, ValueError):
        sequence_length = None
    return {
        "status": "blocked",
        "kind": "single_launcher_chunks_recurrent_backward_scalar_replay",
        "reason": (
            "single all-stage launcher still runs recurrent backward scalar "
            "replay inside one generated TileLang call; production runtime "
            "must use descriptor backward stage fragments until the recurrent "
            "backward fragments are tiled"
        ),
        "row_dispatch_mode": row_dispatch_mode,
        "execution_stage": execution_stage,
        "row_chunk_count": _path_c_int_prim_attr(
            prim_func,
            "_cppmega_path_c_row_chunk_count",
        ),
        "row_subchunk_count": _path_c_int_prim_attr(
            prim_func,
            "_cppmega_path_c_row_subchunk_count",
        ),
        "rows_per_kernel_launch": _path_c_int_prim_attr(
            prim_func,
            "_cppmega_path_c_rows_per_kernel_launch",
        ),
        "sequence_length": sequence_length,
        "recurrent_backward_ops": list(dict.fromkeys(recurrent_backward_ops)),
    }


def compile_path_c_fused_train_block_artifact_for_model(
    *,
    model: Any,
    sequence_length: int | None = None,
    target_name: str = "metal",
    execution_backend: str = "tvm_ffi",
    lowerer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Compile the selected model Path C train block as one callable artifact.

    This compiles only the generated fused TileLang/TVM schedule. It does not
    allocate physical ABI banks or attach the artifact to the model; callers
    must still bind caller/model-owned banks before using the train-block route.
    """

    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    runtime_owner = f"{profile_name}.path_c_fused_train_block_runtime"
    route_context = _path_c_model_route_context(
        model,
        sequence_length=sequence_length,
    )
    selected_region = _select_path_c_model_route_region(route_context.regions)
    if selected_region is None:
        return {
            "status": "blocked",
            "reason": "model did not expose a Path C route region",
            "runtime_owner": runtime_owner,
            "route_region": None,
            "native_compile_ok": False,
            "hidden_packing_performed": False,
            "artifact": None,
        }

    scheduled = plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    schedule_target = getattr(scheduled, "schedule_target", None)
    if schedule_target is None:
        return {
            "status": "blocked",
            "reason": str(getattr(scheduled, "reason", "Path C schedule unavailable")),
            "runtime_owner": runtime_owner,
            "route_region": getattr(selected_region, "name", None),
            "native_compile_ok": False,
            "hidden_packing_performed": False,
            "artifact": None,
        }

    native_lowerer = lowerer
    if native_lowerer is None:

        def native_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
            return tilelang_single_entry_lowerer(
                func_or_mod,
                target=target,
                execution_backend=execution_backend,
                **kwargs,
            )

    schedule_template = mark_path_c_schedule_template_for_region(
        schedule_target.schedule_template,
        scheduled.region,
        implementation_kind=schedule_target.implementation_kind,
        production_schedule_id=schedule_target.schedule_id
        if schedule_target.implementation_kind == "production"
        else "",
        required_real_abi_inputs=schedule_target.required_real_abi_inputs,
    )
    training_abi_prim_func: Any | None = None
    try:
        training_abi_prim_func = schedule_template(scheduled.region)
        training_abi_contract = _path_c_fused_train_block_training_abi_contract_payload(
            training_abi_prim_func
        )
    except Exception as exc:
        training_abi_contract = {
            **_path_c_fused_train_block_training_abi_contract_payload(None),
            "reason": str(exc),
        }
    try:
        if training_abi_prim_func is None:
            raise RuntimeError(str(training_abi_contract.get("reason", "")))
        monolithic_runtime_blocker = (
            _path_c_monolithic_recurrent_backward_runtime_blocker(
                training_abi_prim_func
            )
        )
        monolithic_compile_kwargs: dict[str, Any] = {
            "schedule_template": schedule_template,
            "schedule_name": schedule_target.schedule_name,
            "schedule_status": schedule_target.schedule_status,
            "target": target_name,
        }
        if monolithic_runtime_blocker is None:
            monolithic_compile_kwargs["tilelang_lowerer"] = native_lowerer
        compiled = compile_path_c_region(
            scheduled.region,
            **monolithic_compile_kwargs,
        )
        descriptor_stage_groups = plan_path_c_descriptor_stage_groups(
            scheduled.region
        )
        forward_stage_groups = tuple(
            group
            for group in descriptor_stage_groups
            if group.execution_stage == DESCRIPTOR_EXECUTION_STAGE_FORWARD
        )
        forward_stage_templates = tuple(
            _path_c_generated_stage_schedule_template(
                schedule_target=schedule_target,
                region=scheduled.region,
                abi_prim_func=training_abi_prim_func,
                execution_stage=group.execution_stage,
                active_node_names=group.active_node_names,
                stage_suffix=group.stage_suffix,
                row_dispatch_mode=group.row_dispatch_mode,
                rows_per_kernel_launch=group.rows_per_kernel_launch,
            )
            for group in forward_stage_groups
        )
        backward_stage_groups = tuple(
            group
            for group in descriptor_stage_groups
            if group.execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
        )
        backward_stage_templates = tuple(
            _path_c_generated_stage_schedule_template(
                schedule_target=schedule_target,
                region=scheduled.region,
                abi_prim_func=training_abi_prim_func,
                execution_stage=group.execution_stage,
                active_node_names=group.active_node_names,
                stage_suffix=group.stage_suffix,
                row_dispatch_mode=group.row_dispatch_mode,
                rows_per_kernel_launch=group.rows_per_kernel_launch,
            )
            for group in backward_stage_groups
        )
        forward_stage_prim_funcs = tuple(
            template(scheduled.region) for template in forward_stage_templates
        )
        backward_stage_prim_funcs = tuple(
            template(scheduled.region) for template in backward_stage_templates
        )
        monolithic_artifact = (
            compiled.artifact if isinstance(compiled, CompiledPathCRegion) else None
        )
        plan = compiled.plan if isinstance(compiled, CompiledPathCRegion) else compiled
        monolithic_plan_payload = _path_c_fused_train_block_plan_payload(plan)
        if monolithic_runtime_blocker is not None:
            monolithic_plan_payload = {
                **monolithic_plan_payload,
                "monolithic_native_compile_skipped": True,
                "monolithic_runtime_blocked": True,
                "monolithic_runtime_blocker": monolithic_runtime_blocker,
                "selected_runtime_artifact": "generated_stage_launcher_chunks",
            }
        physical_abi_map, physical_abi_shapes = (
            _path_c_merged_physical_abi_for_prim_funcs(
                (
                    training_abi_prim_func,
                    *forward_stage_prim_funcs,
                    *backward_stage_prim_funcs,
                )
            )
        )
        parameter_gradient_aliases: Mapping[str, Any] = {}
        if callable(getattr(model, "path_c_parameter_gradient_aliases", None)):
            parameter_gradient_aliases = model.path_c_parameter_gradient_aliases()
        selected_region_metadata = (
            _path_c_fused_train_block_selected_region_metadata(scheduled.region)
        )
        trainable_parameter_names = tuple(
            sorted(_path_c_model_trainable_parameter_names(model))
        )
        generated_stage_groups_payload = [
            {
                "index": group.index,
                "execution_stage": group.execution_stage,
                "active_node_names": list(group.active_node_names),
                "stage_suffix": group.stage_suffix,
                "row_dispatch_mode": group.row_dispatch_mode,
                "rows_per_kernel_launch": group.rows_per_kernel_launch,
                "reason": group.reason,
            }
            for group in (*forward_stage_groups, *backward_stage_groups)
        ]
        if monolithic_runtime_blocker is not None:
            try:
                single_launcher_template = _path_c_generated_stage_schedule_template(
                    schedule_target=schedule_target,
                    region=scheduled.region,
                    abi_prim_func=training_abi_prim_func,
                    execution_stage=DESCRIPTOR_EXECUTION_STAGE_ALL,
                    active_node_names=None,
                    stage_suffix="launcher",
                    row_dispatch_mode=DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
                    rows_per_kernel_launch=1,
                )
                single_launcher_prim_func = single_launcher_template(
                    scheduled.region
                )
                attested_single_launcher_template = (
                    mark_path_c_schedule_template_for_region(
                        single_launcher_template,
                        scheduled.region,
                        implementation_kind=schedule_target.implementation_kind,
                        production_schedule_id=schedule_target.schedule_id
                        if schedule_target.implementation_kind == "production"
                        else "",
                        required_real_abi_inputs=(
                            schedule_target.required_real_abi_inputs
                        ),
                    )
                )
                single_launcher_compiled = compile_path_c_region(
                    scheduled.region,
                    schedule_template=attested_single_launcher_template,
                    schedule_name=f"{schedule_target.schedule_name}_launcher_chunks",
                    schedule_status=schedule_target.schedule_status,
                    tilelang_lowerer=native_lowerer,
                    target=target_name,
                )
                single_launcher_artifact = (
                    single_launcher_compiled.artifact
                    if isinstance(single_launcher_compiled, CompiledPathCRegion)
                    else None
                )
                single_launcher_plan = (
                    single_launcher_compiled.plan
                    if isinstance(single_launcher_compiled, CompiledPathCRegion)
                    else single_launcher_compiled
                )
                single_launcher_plan_payload = (
                    _path_c_fused_train_block_plan_payload(single_launcher_plan)
                )
                if callable(single_launcher_artifact) and bool(
                    single_launcher_plan_payload.get("single_kernel_fused")
                ):
                    single_launcher_runtime_blocker = (
                        _path_c_single_launcher_recurrent_backward_runtime_blocker(
                            single_launcher_prim_func
                        )
                    )
                    launcher_physical_abi_map, launcher_physical_abi_shapes = (
                        _path_c_merged_physical_abi_for_prim_funcs(
                            (single_launcher_prim_func,)
                        )
                    )
                    launcher_row_launch_attrs = (
                        _path_c_generated_stage_row_launch_attrs(
                            (single_launcher_prim_func,)
                        )
                    )
                    plan_payload = {
                        **single_launcher_plan_payload,
                        "monolithic_native_compile_skipped": True,
                        "monolithic_runtime_blocked": False,
                        "monolithic_runtime_blocker": None,
                        "monolithic_grid_runtime_blocked": True,
                        "monolithic_grid_runtime_blocker": (
                            monolithic_runtime_blocker
                        ),
                        "selected_runtime_artifact": (
                            "single_generated_launcher_chunks"
                        ),
                        "single_generated_artifact": True,
                        "generated_stage_artifact": False,
                        "generated_stage_count": len(forward_stage_prim_funcs)
                        + len(backward_stage_prim_funcs),
                        "generated_stage_groups": generated_stage_groups_payload,
                        "single_launcher_compile_verified": True,
                        "single_launcher_row_launch": {
                            key: value
                            for key, value in launcher_row_launch_attrs.items()
                            if value is not None
                        },
                        "single_launcher_backward_stage": {
                            "backward_stage_count": getattr(
                                single_launcher_prim_func,
                                "_cppmega_path_c_backward_stage_count",
                                None,
                            ),
                            "backward_stage_index_param": getattr(
                                single_launcher_prim_func,
                                "_cppmega_path_c_backward_stage_index_param",
                                None,
                            ),
                            "backward_stage_node_groups": [
                                list(group)
                                for group in getattr(
                                    single_launcher_prim_func,
                                    "_cppmega_path_c_backward_stage_node_groups",
                                    (),
                                )
                            ],
                        },
                        "single_launcher_runtime_blocked": bool(
                            single_launcher_runtime_blocker
                        ),
                        "single_launcher_runtime_blocker": (
                            single_launcher_runtime_blocker
                        ),
                        "monolithic_contract_artifact_type": type(
                            single_launcher_artifact
                        ).__name__,
                        "all_stages_single_kernel_fused": True,
                        "runtime_schedule_contract_status": (
                            "single_launcher_verified"
                        ),
                    }
                    if single_launcher_runtime_blocker is not None:
                        monolithic_plan_payload = {
                            **monolithic_plan_payload,
                            **plan_payload,
                            "selected_runtime_artifact": (
                                "generated_stage_launcher_chunks"
                            ),
                            "single_generated_artifact": False,
                            "generated_stage_artifact": True,
                            "runtime_schedule_contract_status": (
                                "single_launcher_runtime_blocked"
                            ),
                        }
                        raise RuntimeError(
                            single_launcher_runtime_blocker["reason"]
                        )
                    artifact = PathCFusedTrainBlockCallableArtifact(
                        kernel=single_launcher_artifact,
                        physical_abi_map=launcher_physical_abi_map,
                        physical_abi_shapes=launcher_physical_abi_shapes,
                        training_abi_contract=training_abi_contract,
                        parameter_gradient_aliases=parameter_gradient_aliases,
                        trainable_parameter_names=trainable_parameter_names,
                        selected_region_metadata=selected_region_metadata,
                        kernel_buffer_order=path_c_kernel_buffer_order(
                            single_launcher_prim_func
                        ),
                        kernel_buffer_shapes=_path_c_kernel_buffer_shapes(
                            single_launcher_prim_func
                        ),
                        backward_gate_param=(
                            launcher_row_launch_attrs["backward_gate_param"]
                            or getattr(
                                single_launcher_prim_func,
                                "_cppmega_path_c_backward_gate_param",
                                "path_c_run_backward",
                            )
                        ),
                        row_chunk_count=launcher_row_launch_attrs[
                            "row_chunk_count"
                        ],
                        row_chunk_index_param=launcher_row_launch_attrs[
                            "row_chunk_index_param"
                        ],
                        row_subchunk_count=launcher_row_launch_attrs[
                            "row_subchunk_count"
                        ],
                        row_subchunk_index_param=launcher_row_launch_attrs[
                            "row_subchunk_index_param"
                        ],
                        rows_per_kernel_launch=launcher_row_launch_attrs[
                            "rows_per_kernel_launch"
                        ],
                        backward_stage_count=getattr(
                            single_launcher_prim_func,
                            "_cppmega_path_c_backward_stage_count",
                            None,
                        ),
                        backward_stage_index_param=getattr(
                            single_launcher_prim_func,
                            "_cppmega_path_c_backward_stage_index_param",
                            None,
                        ),
                    )
                    return {
                        "status": "ok",
                        "reason": (
                            "selected Path C model train block compiled to one "
                            "callable generated TileLang/TVM launcher artifact"
                        ),
                        "runtime_owner": runtime_owner,
                        "route_region": getattr(scheduled.region, "name", None),
                        "schedule_id": getattr(schedule_target, "schedule_id", None),
                        "schedule_name": getattr(schedule_target, "schedule_name", None),
                        "implementation_kind": getattr(
                            schedule_target,
                            "implementation_kind",
                            None,
                        ),
                        "native_compile_ok": True,
                        "hidden_packing_performed": False,
                        "plan": plan_payload,
                        "training_abi_contract": training_abi_contract,
                        "artifact": artifact,
                    }
                monolithic_plan_payload = {
                    **monolithic_plan_payload,
                    "single_launcher_compile_verified": False,
                    "single_launcher_rejected_reason": (
                        "single generated launcher did not produce a verified "
                        "callable fused artifact"
                    ),
                    "single_launcher_plan": single_launcher_plan_payload,
                }
            except Exception as exc:
                if monolithic_plan_payload.get("single_launcher_runtime_blocked"):
                    monolithic_plan_payload = {
                        **monolithic_plan_payload,
                        "single_launcher_rejected_reason": str(exc),
                    }
                else:
                    monolithic_plan_payload = {
                        **monolithic_plan_payload,
                        "single_launcher_compile_verified": False,
                        "single_launcher_compile_error": str(exc),
                        "single_launcher_compile_error_type": type(exc).__name__,
                    }
        if (
            monolithic_runtime_blocker is None
            and callable(monolithic_artifact)
            and bool(monolithic_plan_payload.get("single_kernel_fused"))
        ):
            plan_payload = {
                **monolithic_plan_payload,
                "monolithic_native_compile_skipped": False,
                "monolithic_runtime_blocked": False,
                "monolithic_runtime_blocker": None,
                "selected_runtime_artifact": "single_monolithic_grid_chunks",
                "single_generated_artifact": True,
                "generated_stage_artifact": False,
                "generated_stage_count": len(forward_stage_prim_funcs)
                + len(backward_stage_prim_funcs),
                "generated_stage_groups": generated_stage_groups_payload,
                "monolithic_contract_artifact_type": type(
                    monolithic_artifact
                ).__name__,
                "all_stages_single_kernel_fused": True,
            }
            artifact = PathCFusedTrainBlockCallableArtifact(
                kernel=monolithic_artifact,
                physical_abi_map=physical_abi_map,
                physical_abi_shapes=physical_abi_shapes,
                training_abi_contract=training_abi_contract,
                parameter_gradient_aliases=parameter_gradient_aliases,
                trainable_parameter_names=trainable_parameter_names,
                selected_region_metadata=selected_region_metadata,
                kernel_buffer_order=path_c_kernel_buffer_order(
                    training_abi_prim_func
                ),
                kernel_buffer_shapes=_path_c_kernel_buffer_shapes(
                    training_abi_prim_func
                ),
                backward_gate_param=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_backward_gate_param",
                    "path_c_run_backward",
                ),
                row_chunk_count=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_row_chunk_count",
                    None,
                ),
                row_chunk_index_param=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_row_chunk_index_param",
                    None,
                ),
                row_subchunk_count=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_row_subchunk_count",
                    None,
                ),
                row_subchunk_index_param=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_row_subchunk_index_param",
                    None,
                ),
                rows_per_kernel_launch=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_rows_per_kernel_launch",
                    None,
                ),
                backward_stage_count=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_backward_stage_count",
                    None,
                ),
                backward_stage_index_param=getattr(
                    training_abi_prim_func,
                    "_cppmega_path_c_backward_stage_index_param",
                    None,
                ),
            )
            return {
                "status": "ok",
                "reason": (
                    "selected Path C model train block compiled to one callable "
                    "generated TileLang/TVM artifact"
                ),
                "runtime_owner": runtime_owner,
                "route_region": getattr(scheduled.region, "name", None),
                "schedule_id": getattr(schedule_target, "schedule_id", None),
                "schedule_name": getattr(schedule_target, "schedule_name", None),
                "implementation_kind": getattr(
                    schedule_target,
                    "implementation_kind",
                    None,
                ),
                "native_compile_ok": True,
                "hidden_packing_performed": False,
                "plan": plan_payload,
                "training_abi_contract": training_abi_contract,
                "artifact": artifact,
            }
        forward_artifacts = tuple(
            native_lowerer(
                prim_func,
                target=target_name,
            )
            for prim_func in forward_stage_prim_funcs
        )
        backward_artifacts = tuple(
            native_lowerer(
                prim_func,
                target=target_name,
            )
            for prim_func in backward_stage_prim_funcs
        )
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": str(exc),
            "error_type": type(exc).__name__,
            "runtime_owner": runtime_owner,
            "route_region": getattr(scheduled.region, "name", None),
            "schedule_id": getattr(schedule_target, "schedule_id", None),
            "schedule_name": getattr(schedule_target, "schedule_name", None),
            "implementation_kind": getattr(
                schedule_target,
                "implementation_kind",
                None,
            ),
            "native_compile_ok": False,
            "hidden_packing_performed": False,
            "artifact": None,
        }

    monolithic_artifact = (
        compiled.artifact if isinstance(compiled, CompiledPathCRegion) else None
    )
    plan_payload = {
        **monolithic_plan_payload,
        "single_kernel_fused": False,
    }
    if monolithic_runtime_blocker is not None:
        plan_payload = {
            **plan_payload,
            "monolithic_native_compile_skipped": True,
            "monolithic_runtime_blocked": True,
            "monolithic_runtime_blocker": monolithic_runtime_blocker,
            "monolithic_grid_runtime_blocked": True,
            "monolithic_grid_runtime_blocker": monolithic_runtime_blocker,
            "selected_runtime_artifact": "generated_stage_launcher_chunks",
        }
    else:
        plan_payload = {
            **plan_payload,
            "monolithic_native_compile_skipped": False,
            "monolithic_runtime_blocked": False,
            "monolithic_runtime_blocker": None,
            "monolithic_grid_runtime_blocked": False,
            "monolithic_grid_runtime_blocker": None,
            "selected_runtime_artifact": "generated_stage_launcher_chunks",
        }
    stage_row_launch_attrs = _path_c_generated_stage_row_launch_attrs(
        (*forward_stage_prim_funcs, *backward_stage_prim_funcs)
    )
    plan_payload = {
        **plan_payload,
        "single_generated_artifact": False,
        "generated_stage_artifact": True,
        "generated_stage_count": len(forward_stage_prim_funcs)
        + len(backward_stage_prim_funcs),
        "generated_stage_groups": [
            {
                "index": group.index,
                "execution_stage": group.execution_stage,
                "active_node_names": list(group.active_node_names),
                "stage_suffix": group.stage_suffix,
                "row_dispatch_mode": group.row_dispatch_mode,
                "rows_per_kernel_launch": group.rows_per_kernel_launch,
                "reason": group.reason,
            }
            for group in (*forward_stage_groups, *backward_stage_groups)
        ],
        "stage_source_bytes": {
            **{
                f"forward_{index}": len(
                    (
                        getattr(
                            prim_func,
                            "_cppmega_path_c_generated_source",
                            "",
                        )
                        or ""
                    ).encode("utf-8")
                )
                for index, prim_func in enumerate(forward_stage_prim_funcs)
            },
            **{
                f"backward_{index}": len(
                    (
                        getattr(
                            prim_func,
                            "_cppmega_path_c_generated_source",
                            "",
                        )
                        or ""
                    ).encode("utf-8")
                )
                for index, prim_func in enumerate(backward_stage_prim_funcs)
            },
        },
        "monolithic_contract_artifact_type": type(monolithic_artifact).__name__
        if monolithic_artifact is not None
        else None,
        "all_stages_single_kernel_fused": bool(
            all(callable(artifact) for artifact in forward_artifacts)
            and all(callable(artifact) for artifact in backward_artifacts)
        ),
        "generated_stage_compile_verified": bool(
            all(callable(artifact) for artifact in forward_artifacts)
            and all(callable(artifact) for artifact in backward_artifacts)
        ),
        "runtime_schedule_contract_status": "generated_stages_verified"
        if (
            all(callable(artifact) for artifact in forward_artifacts)
            and all(callable(artifact) for artifact in backward_artifacts)
        )
        else "generated_stages_unverified",
        "generated_stage_row_launch": {
            key: value
            for key, value in stage_row_launch_attrs.items()
            if value is not None
        },
        "generated_forward_stage_row_launches": list(
            _path_c_stage_row_launch_specs(forward_stage_prim_funcs)
        ),
        "generated_backward_stage_row_launches": list(
            _path_c_stage_row_launch_specs(backward_stage_prim_funcs)
        ),
    }
    if (
        not forward_artifacts
        or not all(callable(artifact) for artifact in forward_artifacts)
        or not backward_artifacts
        or not all(callable(artifact) for artifact in backward_artifacts)
    ):
        return {
            "status": "blocked",
            "reason": (
                "generated Path C stage compile did not produce callable "
                "forward/backward artifacts"
            ),
            "runtime_owner": runtime_owner,
            "route_region": getattr(scheduled.region, "name", None),
            "schedule_id": getattr(schedule_target, "schedule_id", None),
            "schedule_name": getattr(schedule_target, "schedule_name", None),
            "implementation_kind": getattr(
                schedule_target,
                "implementation_kind",
                None,
            ),
            "native_compile_ok": False,
            "hidden_packing_performed": False,
            "plan": plan_payload,
            "training_abi_contract": training_abi_contract,
            "artifact": None,
        }
    if not bool(plan_payload["all_stages_single_kernel_fused"]):
        return {
            "status": "blocked",
            "reason": "compiled train-block stages are not verified fused kernels",
            "runtime_owner": runtime_owner,
            "route_region": getattr(scheduled.region, "name", None),
            "schedule_id": getattr(schedule_target, "schedule_id", None),
            "schedule_name": getattr(schedule_target, "schedule_name", None),
            "implementation_kind": getattr(
                schedule_target,
                "implementation_kind",
                None,
            ),
            "native_compile_ok": False,
            "hidden_packing_performed": False,
            "plan": plan_payload,
            "training_abi_contract": training_abi_contract,
            "artifact": None,
        }
    parameter_gradient_aliases: Mapping[str, Any] = {}
    if callable(getattr(model, "path_c_parameter_gradient_aliases", None)):
        parameter_gradient_aliases = model.path_c_parameter_gradient_aliases()
    physical_abi_map, physical_abi_shapes = _path_c_merged_physical_abi_for_prim_funcs(
        (
            training_abi_prim_func,
            *forward_stage_prim_funcs,
            *backward_stage_prim_funcs,
        )
    )
    artifact = PathCGeneratedStageTrainBlockCallableArtifact(
        forward_kernel=forward_artifacts[0],
        forward_kernels=forward_artifacts,
        backward_kernel=backward_artifacts[0],
        backward_kernels=backward_artifacts,
        physical_abi_map=physical_abi_map,
        physical_abi_shapes=physical_abi_shapes,
        training_abi_contract=training_abi_contract,
        parameter_gradient_aliases=parameter_gradient_aliases,
        trainable_parameter_names=tuple(
            sorted(_path_c_model_trainable_parameter_names(model))
        ),
        selected_region_metadata=(
            _path_c_fused_train_block_selected_region_metadata(scheduled.region)
        ),
        forward_kernel_buffer_order=path_c_kernel_buffer_order(
            forward_stage_prim_funcs[0]
        ),
        forward_kernel_buffer_orders=tuple(
            path_c_kernel_buffer_order(prim_func)
            for prim_func in forward_stage_prim_funcs
        ),
        backward_kernel_buffer_order=path_c_kernel_buffer_order(
            backward_stage_prim_funcs[0]
        ),
        backward_kernel_buffer_orders=tuple(
            path_c_kernel_buffer_order(prim_func)
            for prim_func in backward_stage_prim_funcs
        ),
        forward_kernel_buffer_shapes=_path_c_kernel_buffer_shapes(
            forward_stage_prim_funcs[0]
        ),
        forward_kernel_buffer_shapes_by_stage=tuple(
            _path_c_kernel_buffer_shapes(prim_func)
            for prim_func in forward_stage_prim_funcs
        ),
        backward_kernel_buffer_shapes=_path_c_kernel_buffer_shapes(
            backward_stage_prim_funcs[0]
        ),
        backward_kernel_buffer_shapes_by_stage=tuple(
            _path_c_kernel_buffer_shapes(prim_func)
            for prim_func in backward_stage_prim_funcs
        ),
        backward_gate_param=getattr(
            training_abi_prim_func,
            "_cppmega_path_c_backward_gate_param",
            "path_c_run_backward",
        )
        or "path_c_run_backward",
        row_chunk_count=stage_row_launch_attrs["row_chunk_count"],
        row_chunk_index_param=stage_row_launch_attrs["row_chunk_index_param"],
        row_subchunk_count=stage_row_launch_attrs["row_subchunk_count"],
        row_subchunk_index_param=stage_row_launch_attrs["row_subchunk_index_param"],
        rows_per_kernel_launch=stage_row_launch_attrs["rows_per_kernel_launch"],
        forward_stage_row_launches=_path_c_stage_row_launch_specs(
            forward_stage_prim_funcs
        ),
        backward_stage_row_launches=_path_c_stage_row_launch_specs(
            backward_stage_prim_funcs
        ),
    )

    return {
        "status": "ok",
        "reason": (
            "selected Path C model train block compiled to one callable "
            "generated TileLang/TVM artifact"
        ),
        "runtime_owner": runtime_owner,
        "route_region": getattr(scheduled.region, "name", None),
        "schedule_id": getattr(schedule_target, "schedule_id", None),
        "schedule_name": getattr(schedule_target, "schedule_name", None),
        "implementation_kind": getattr(schedule_target, "implementation_kind", None),
        "native_compile_ok": True,
        "hidden_packing_performed": False,
        "plan": plan_payload,
        "training_abi_contract": training_abi_contract,
        "artifact": artifact,
    }


def install_path_c_fused_train_block_runtime_for_model(
    *,
    model: Any,
    bank_owner: Any | None = None,
    bank_buffers: Mapping[str, Any] | None = None,
    bank_buffer_owner: str | None = None,
    fused_artifact: Any | None = None,
    training_runtime: Any | None = None,
    compile_artifact: bool = False,
    artifact_lowerer: Callable[..., Any] | None = None,
    artifact_target_name: str = "metal",
    artifact_execution_backend: str = "tvm_ffi",
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Attach a single fused train-block runtime when ABI binding is executable.

    The fused route requires caller/model-owned physical ABI banks plus a
    callable compiled artifact. This installer validates those objects against
    the generated schedule ABI and never allocates, packs, reshapes, or casts
    tensors to satisfy the binding.
    """

    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    explicit_caller_banks = bank_owner is not None or bank_buffers is not None
    if bank_owner is not None and bank_buffers is not None:
        return {
            "status": "blocked",
            "reason": "bank_owner and bank_buffers are mutually exclusive",
            "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
            "runtime_uses_fused_train_block": False,
            "hidden_packing_performed": False,
            "artifact_compile": None,
            "execution": None,
        }

    route_context = _path_c_model_route_context(
        model,
        sequence_length=sequence_length,
    )
    selected_region = _select_path_c_model_route_region(route_context.regions)
    if selected_region is None:
        return {
            "status": "blocked",
            "reason": "model did not expose a Path C route region",
            "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
            "runtime_uses_fused_train_block": False,
            "hidden_packing_performed": False,
            "artifact_compile": None,
            "execution": None,
        }

    scheduled = plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    schedule_target = getattr(scheduled, "schedule_target", None)
    if schedule_target is None:
        return {
            "status": "blocked",
            "reason": str(getattr(scheduled, "reason", "Path C schedule unavailable")),
            "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
            "route_region": getattr(selected_region, "name", None),
            "runtime_uses_fused_train_block": False,
            "hidden_packing_performed": False,
            "artifact_compile": None,
            "execution": None,
        }

    resolved_artifact = (
        fused_artifact
        if fused_artifact is not None
        else getattr(model, "path_c_fused_train_block_artifact", None)
    )
    if bank_owner is not None:
        resolved_bank_owner = bank_owner
    elif bank_buffers is not None:
        resolved_bank_owner = None
    else:
        # When we are about to compile/select the artifact, do not allocate
        # model banks from the generic schedule first: single launcher chunks
        # can require larger top-level scratch than the per-stage ABI template.
        resolved_bank_owner = _path_c_physical_abi_bank_owner_for_model(
            model,
            sequence_length=sequence_length,
            allocate_if_missing=not (compile_artifact and resolved_artifact is None),
        )
    resolved_bank_buffers = bank_buffers
    resolved_bank_buffer_owner = bank_buffer_owner
    if resolved_bank_owner is None and bank_buffers is not None:
        try:
            prim_func = schedule_target.schedule_template(scheduled.region)
            physical_abi_map = dict(
                getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
                or {}
            )
            physical_abi_shapes = dict(
                getattr(
                    prim_func,
                    "_cppmega_path_c_physical_buffer_abi_shapes",
                    {},
                )
                or {}
            )
            resolved_bank_owner = make_physical_abi_bank_owner(
                bank_buffer_owner
                or f"{profile_name}.path_c_physical_abi_banks",
                physical_abi_map,
                physical_abi_shapes,
                bank_buffers,
            )
            resolved_bank_buffers = None
            resolved_bank_buffer_owner = None
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"physical ABI bank owner validation failed: {exc}",
                "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
                "route_region": getattr(selected_region, "name", None),
                "runtime_uses_fused_train_block": False,
                "hidden_packing_performed": False,
                "artifact_compile": None,
                "execution": None,
            }

    artifact_compile_payload: dict[str, Any] | None = None
    if resolved_artifact is None and compile_artifact:
        artifact_compile_payload = (
            compile_path_c_fused_train_block_artifact_for_model(
                model=model,
                sequence_length=sequence_length,
                target_name=artifact_target_name,
                execution_backend=artifact_execution_backend,
                lowerer=artifact_lowerer,
            )
        )
        if artifact_compile_payload.get("status") != "ok":
            return {
                "status": "blocked",
                "reason": artifact_compile_payload.get("reason"),
                "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
                "route_region": getattr(selected_region, "name", None),
                "runtime_uses_fused_train_block": False,
                "hidden_packing_performed": bool(
                    artifact_compile_payload.get(
                        "hidden_packing_performed",
                        False,
                    )
                ),
                "artifact_compile": (
                    _path_c_fused_train_block_artifact_compile_report(
                        artifact_compile_payload
                    )
                ),
                "execution": None,
            }
        resolved_artifact = artifact_compile_payload.get("artifact")

    artifact_kernel_buffer_binding: dict[str, Any] | None = None
    if resolved_artifact is not None:
        artifact_kernel_buffer_binding = (
            _path_c_artifact_bank_owner_kernel_buffer_binding(
                artifact=resolved_artifact,
                bank_owner=resolved_bank_owner,
            )
        )
    if (
        resolved_artifact is not None
        and artifact_kernel_buffer_binding is not None
        and artifact_kernel_buffer_binding.get("status") != "ok"
        and not explicit_caller_banks
    ):
        try:
            resolved_bank_owner = _path_c_physical_abi_bank_owner_for_artifact(
                artifact=resolved_artifact,
                owner_name=f"{profile_name}.path_c_physical_abi_banks",
            )
            artifact_kernel_buffer_binding = (
                _path_c_artifact_bank_owner_kernel_buffer_binding(
                    artifact=resolved_artifact,
                    bank_owner=resolved_bank_owner,
                )
            )
            if artifact_compile_payload is not None:
                artifact_compile_payload = {
                    **artifact_compile_payload,
                    "artifact_physical_abi_bank_owner_allocated": (
                        resolved_bank_owner is not None
                    ),
                    "artifact_physical_abi_bank_owner": getattr(
                        resolved_bank_owner,
                        "owner_name",
                        None,
                    ),
                    "artifact_kernel_buffer_binding": (
                        artifact_kernel_buffer_binding
                    ),
                }
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"artifact physical ABI bank allocation failed: {exc}",
                "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
                "route_region": getattr(selected_region, "name", None),
                "runtime_uses_fused_train_block": False,
                "hidden_packing_performed": False,
                "artifact_compile": _path_c_fused_train_block_artifact_compile_report(
                    artifact_compile_payload
                ),
                "execution": None,
            }
    if (
        resolved_artifact is not None
        and artifact_kernel_buffer_binding is not None
        and artifact_kernel_buffer_binding.get("status") != "ok"
    ):
        return {
            "status": "blocked",
            "reason": artifact_kernel_buffer_binding.get("reason"),
            "runtime_owner": f"{profile_name}.path_c_fused_train_block_runtime",
            "route_region": getattr(selected_region, "name", None),
            "runtime_uses_fused_train_block": False,
            "hidden_packing_performed": False,
            "artifact_compile": _path_c_fused_train_block_artifact_compile_report(
                artifact_compile_payload
            ),
            "artifact_kernel_buffer_binding": artifact_kernel_buffer_binding,
            "execution": None,
        }

    runtime_binding = path_c_fusion_runtime_training_binding_payload(
        region=scheduled.region,
        schedule_target=schedule_target,
        bank_buffers=resolved_bank_buffers,
        bank_buffer_owner=resolved_bank_buffer_owner,
        bank_owner=resolved_bank_owner,
        fused_artifact=resolved_artifact,
    )
    runtime_owner = f"{profile_name}.path_c_fused_train_block_runtime"
    if not bool(runtime_binding.get("runtime_uses_fused_train_block")):
        return {
            "status": "blocked",
            "reason": runtime_binding.get("reason"),
            "runtime_owner": runtime_owner,
            "route_region": getattr(selected_region, "name", None),
            "binding_status": runtime_binding.get("status"),
            "runtime_uses_fused_train_block": False,
            "runtime_binding": runtime_binding,
            "artifact_kernel_buffer_binding": artifact_kernel_buffer_binding,
            "hidden_packing_performed": bool(
                runtime_binding.get("hidden_packing_performed", False)
            ),
            "artifact_compile": _path_c_fused_train_block_artifact_compile_report(
                artifact_compile_payload
            ),
            "execution": None,
        }

    if training_runtime is None and resolved_bank_owner is not None:
        training_runtime = (
            _path_c_fused_train_block_training_runtime_from_artifact(
                artifact=resolved_artifact,
                bank_owner=resolved_bank_owner,
                runtime_owner=runtime_owner,
                model=model,
                sequence_length=sequence_length,
            )
        )

    training_runtime_contract = (
        path_c_fused_train_block_training_runtime_contract_payload(
            training_runtime=training_runtime,
            runtime_binding=runtime_binding,
        )
    )
    training_runtime_bound = False
    if training_runtime is not None:
        bind_training_graph = getattr(training_runtime, "bind_training_graph", None)
        if callable(bind_training_graph):
            bind_training_graph(
                owner="CompiledPretrainingStep",
                uses_fused_train_block_runtime=True,
                uses_forward_hook=True,
                uses_backward_or_vjp_hook=True,
            )
            training_runtime_bound = True
            training_runtime_contract = (
                path_c_fused_train_block_training_runtime_contract_payload(
                    training_runtime=training_runtime,
                    runtime_binding=runtime_binding,
                )
            )
    if training_runtime_contract.get("status") != "ok":
        if training_runtime_bound:
            unbind_training_graph = getattr(
                training_runtime,
                "unbind_training_graph",
                None,
            )
            if callable(unbind_training_graph):
                unbind_training_graph(owner="CompiledPretrainingStep")
        model.path_c_physical_abi_bank_owner = resolved_bank_owner
        model.path_c_fused_train_block_artifact = resolved_artifact
        return {
            "status": "blocked",
            "reason": training_runtime_contract.get("reason"),
            "runtime_owner": runtime_owner,
            "route_region": getattr(selected_region, "name", None),
            "binding_status": runtime_binding.get("status"),
            "runtime_uses_fused_train_block": True,
            "runtime_binding": runtime_binding,
            "artifact_kernel_buffer_binding": artifact_kernel_buffer_binding,
            "training_runtime_available": False,
            "training_runtime_contract": training_runtime_contract,
            "hidden_packing_performed": bool(
                runtime_binding.get("hidden_packing_performed", False)
                or training_runtime_contract.get(
                    "hidden_packing_performed",
                    False,
                )
            ),
            "artifact_compile": (
                _path_c_fused_train_block_artifact_compile_report(
                    artifact_compile_payload
                )
            ),
            "execution": None,
        }

    model.path_c_physical_abi_bank_owner = resolved_bank_owner
    model.path_c_fused_train_block_artifact = resolved_artifact
    if training_runtime is not None:
        model.path_c_fused_train_block_training_runtime = training_runtime
    return {
        "status": "ok",
        "runtime_owner": runtime_owner,
        "route_region": getattr(selected_region, "name", None),
        "binding_status": runtime_binding.get("status"),
        "runtime_uses_fused_train_block": True,
        "runtime_binding": runtime_binding,
        "artifact_kernel_buffer_binding": artifact_kernel_buffer_binding,
        "training_runtime_available": bool(
            training_runtime_contract.get("training_runtime_available")
        ),
        "training_runtime_contract": training_runtime_contract,
        "bank_buffer_owner": runtime_binding.get("bank_buffer_owner"),
        "hidden_packing_performed": bool(
            runtime_binding.get("hidden_packing_performed", False)
            or training_runtime_contract.get("hidden_packing_performed", False)
        ),
        "no_hidden_allocation_policy": bool(
            runtime_binding.get("no_hidden_allocation_policy", True)
        ),
        "artifact_compile": _path_c_fused_train_block_artifact_compile_report(
            artifact_compile_payload
        ),
        "execution": None,
        "reason": (
            "single fused train-block runtime is attached from model/caller-owned "
            "physical ABI banks and a callable generated artifact"
        ),
    }


def _path_c_direct_chain_plan_payload(chain: Any) -> dict[str, Any]:
    segments = []
    for segment in getattr(chain, "segments", ()):
        target = getattr(segment, "schedule_target", None)
        plan = getattr(segment, "plan", None)
        contract = getattr(plan, "schedule_contract", None)
        segments.append(
            {
                "index": int(segment.index),
                "node_start": int(segment.node_start),
                "node_end": int(segment.node_end),
                "region_name": getattr(segment.region, "name", ""),
                "node_names": list(getattr(segment.region, "node_names", ())),
                "op_signature": [
                    str(node.op_name)
                    for node in getattr(segment.region, "nodes", ())
                ],
                "execution_phase": str(
                    getattr(segment, "execution_phase", "unknown")
                ),
                "status": str(segment.status),
                "reason": str(segment.reason),
                "physical_abi_policy": str(segment.physical_abi_policy),
                "kernel_parameter_count": segment.kernel_parameter_count,
                "schedule_id": getattr(target, "schedule_id", None),
                "schedule_name": getattr(target, "schedule_name", None),
                "schedule_contract_status": getattr(contract, "status", None),
            }
        )
    source_nodes = list(getattr(chain.source_region, "node_names", ()))
    covers_full_region = bool(segments) and (
        segments[0]["node_start"] == 0
        and segments[-1]["node_end"] == len(source_nodes)
        and all(
            left["node_end"] == right["node_start"]
            for left, right in zip(segments[:-1], segments[1:], strict=True)
        )
    )
    return {
        "status": str(chain.status),
        "reason": str(chain.reason),
        "max_kernel_buffers": int(chain.max_kernel_buffers),
        "segment_count": len(segments),
        "covers_full_region": covers_full_region,
        "source_region_name": getattr(chain.source_region, "name", ""),
        "source_node_count": len(source_nodes),
        "segments": segments,
    }


def _select_path_c_direct_chain_for_region(
    chains: tuple[Any, ...],
    selected_region: Any,
) -> Any | None:
    if not chains:
        return None
    selected_name = str(getattr(selected_region, "name", ""))
    for chain in chains:
        source_region = getattr(chain, "source_region", None)
        if str(getattr(source_region, "name", "")) == selected_name:
            return chain
    return max(
        chains,
        key=lambda chain: (
            len(getattr(getattr(chain, "source_region", None), "nodes", ())),
            len(getattr(getattr(chain, "source_region", None), "edges", ())),
            str(getattr(getattr(chain, "source_region", None), "name", "")),
        ),
    )


def _path_c_direct_chain_region_prefix(
    model: Any,
    profile_name: str,
) -> str:
    return str(
        getattr(model, "path_c_region_prefix", None)
        or f"{profile_name}_path_c"
    )


def _path_c_direct_chain_artifact_for_segment(
    artifacts: Any | None,
    segment: Any,
) -> Any | None:
    if artifacts is None:
        return None
    if isinstance(artifacts, Mapping):
        target = getattr(segment, "schedule_target", None)
        keys = (
            segment.index,
            str(segment.index),
            getattr(segment.region, "name", None),
            getattr(target, "schedule_id", None),
            getattr(target, "schedule_name", None),
        )
        for key in keys:
            if key is not None and key in artifacts:
                return artifacts[key]
        return None
    if isinstance(artifacts, (list, tuple)):
        try:
            return artifacts[int(segment.index)]
        except IndexError:
            return None
    return None


def compile_path_c_direct_fusion_chain_artifacts(
    chain: Any,
    *,
    target_name: str = "metal",
    execution_backend: str = "tvm_ffi",
    lowerer: Callable[..., Any] | None = None,
) -> tuple[Any, ...]:
    """Native-compile direct-chain segment artifacts as callable TileLang kernels."""

    native_lowerer = lowerer
    if native_lowerer is None:

        def native_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
            return tilelang_single_entry_lowerer(
                func_or_mod,
                target=target,
                execution_backend=execution_backend,
                **kwargs,
            )

    artifacts: list[Any] = []
    for segment in getattr(chain, "segments", ()):
        target = getattr(segment, "schedule_target", None)
        if target is None or str(segment.status) != "ok":
            raise RuntimeError(
                f"direct-chain segment {getattr(segment, 'index', '?')} is not compilable: "
                f"{getattr(segment, 'reason', '')}"
            )
        schedule_template = mark_path_c_schedule_template_for_region(
            target.schedule_template,
            segment.region,
            implementation_kind=target.implementation_kind,
            production_schedule_id=target.schedule_id
            if target.implementation_kind == "production"
            else "",
            required_real_abi_inputs=target.required_real_abi_inputs,
        )
        compiled = compile_path_c_region(
            segment.region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name,
            schedule_status=target.schedule_status,
            tilelang_lowerer=native_lowerer,
            target=target_name,
        )
        artifact = (
            compiled.artifact if isinstance(compiled, CompiledPathCRegion) else None
        )
        if artifact is None or not callable(artifact):
            raise RuntimeError(
                f"direct-chain segment {segment.index} native compile did not "
                "produce a callable artifact"
            )
        artifacts.append(artifact)
    return tuple(artifacts)


def run_path_c_direct_fusion_chain_route(
    *,
    chain: Any,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
    artifacts: Any,
    mx_module: Any = mx,
    execution_phases: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute callable direct-chain segment artifacts with caller-owned buffers."""

    binding = path_c_direct_fusion_chain_runtime_binding_payload(
        chain=chain,
        logical_buffers=logical_buffers,
        logical_owner=logical_owner,
        artifacts=artifacts,
    )
    if not binding.get("runtime_uses_direct_fusion_chain"):
        detail_parts = [
            *(str(error) for error in binding.get("binding_errors", ())[:8]),
            *(
                f"missing={name}"
                for name in binding.get("missing_logical_buffers", ())[:8]
            ),
        ]
        detail = "; ".join(detail_parts)
        raise ValueError(
            "direct-chain route is not executable: "
            + str(binding.get("reason", binding.get("status")))
            + (f": {detail}" if detail else "")
        )
    resolved_buffers = (
        getattr(logical_owner, "buffers", None)
        if logical_owner is not None
        else logical_buffers
    )
    if resolved_buffers is None:
        raise ValueError("direct-chain route requires logical buffers")
    buffers = {str(name): value for name, value in resolved_buffers.items()}
    selected_phases = (
        None if execution_phases is None else {str(phase) for phase in execution_phases}
    )
    segment_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for segment in getattr(chain, "segments", ()):
        execution_phase = str(getattr(segment, "execution_phase", "unknown"))
        if selected_phases is not None and execution_phase not in selected_phases:
            continue
        artifact = _path_c_direct_chain_artifact_for_segment(artifacts, segment)
        target = getattr(segment, "schedule_target", None)
        if target is None or not callable(artifact):
            raise ValueError(f"direct-chain segment {segment.index} is not executable")
        prim_func = target.schedule_template(segment.region)
        physical_abi_map = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        physical_abi_shapes = dict(
            getattr(
                prim_func,
                "_cppmega_path_c_physical_buffer_abi_shapes",
                {},
            )
            or {}
        )
        segment_binding = validate_physical_abi_runtime_bindings(
            physical_abi_map,
            physical_abi_shapes,
            buffers,
        )
        if segment_binding.get("status") != "ok":
            detail = "; ".join(str(error) for error in segment_binding.get("errors", ()))
            raise ValueError(
                f"direct-chain segment {segment.index} buffers are invalid"
                + (f": {detail}" if detail else "")
            )
        scratch_specs = _path_c_internal_scratch_abi_specs(prim_func)
        scratch_binding = _path_c_validate_internal_scratch_abi_buffers(
            scratch_specs,
            buffers,
        )
        if scratch_binding.get("status") != "ok":
            detail = "; ".join(
                str(error) for error in scratch_binding.get("errors", ())
            )
            raise ValueError(
                f"direct-chain segment {segment.index} scratch buffers are invalid"
                + (f": {detail}" if detail else "")
            )
        ordered_names = tuple(
            str(name) for name in segment_binding.get("ordered_kernel_buffers", ())
        )
        ordered_names = (
            *ordered_names,
            *(
                str(name)
                for name in scratch_binding.get(
                    "required_internal_scratch_buffers",
                    (),
                )
            ),
        )
        arrays = tuple(buffers[name] for name in ordered_names)
        mx_module.eval(*arrays)
        segment_started = time.perf_counter()
        result = artifact(*arrays)
        mx_module.eval(*arrays)
        segment_results.append(
            {
                "index": int(segment.index),
                "status": "ok",
                "region_name": segment.region.name,
                "execution_phase": execution_phase,
                "schedule_id": target.schedule_id,
                "kernel_arg_count": len(arrays),
                "kernel_args": list(ordered_names),
                "elapsed_s": time.perf_counter() - segment_started,
                "result_type": type(result).__name__,
            }
        )
    return {
        "status": "ok",
        "runtime_uses_direct_fusion_chain": True,
        "segment_count": len(segment_results),
        "elapsed_s": time.perf_counter() - started,
        "execution_phases": None
        if selected_phases is None
        else sorted(selected_phases),
        "binding": binding,
        "segments": segment_results,
    }


class PathCDirectFusionChainTrainingRuntime:
    """Explicit direct-chain runtime object for m04 route binding checks."""

    contract = PATH_C_DIRECT_FUSION_TRAINING_RUNTIME_CONTRACT
    hidden_packing_performed = False
    no_hidden_allocation_policy = True

    def __init__(
        self,
        *,
        chain: Any,
        artifacts: Any,
        logical_buffers: Mapping[str, Any] | None = None,
        logical_owner: Any | None = None,
        owner_name: str,
        training_critical_path: bool = False,
        loss_cotangent_bridge: Any | None = None,
        model: Any | None = None,
        pre_step_owner_factory: Callable[[nn.Module, Mapping[str, mx.array]], Any]
        | None = None,
    ) -> None:
        if logical_buffers is not None and logical_owner is not None:
            raise ValueError("logical_buffers and logical_owner are mutually exclusive")
        self.chain = chain
        self.artifacts = artifacts
        self.logical_buffers = logical_buffers
        self.logical_owner = logical_owner
        self.owner_name = str(owner_name)
        self.training_critical_path = bool(training_critical_path)
        self.loss_cotangent_bridge = loss_cotangent_bridge
        self.model = model
        self.pre_step_owner_factory = pre_step_owner_factory
        self.last_pre_step_owner: Any | None = None
        self.last_pre_step_binding: Mapping[str, Any] | None = None
        self._training_graph_binding: dict[str, Any] | None = None
        self.binding = path_c_direct_fusion_chain_runtime_binding_payload(
            chain=chain,
            logical_buffers=logical_buffers,
            logical_owner=logical_owner,
            artifacts=artifacts,
        )
        if not bool(self.binding.get("runtime_uses_direct_fusion_chain")):
            raise ValueError(
                "direct-chain training runtime requires executable direct-chain "
                f"binding; got {self.binding.get('status')}"
            )

    def _value_and_grad_logical_owner(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
    ) -> tuple[Mapping[str, Any] | None, Any | None]:
        factory = self.pre_step_owner_factory
        if factory is None:
            return self.logical_buffers, self.logical_owner
        owner = factory(model, batch)
        owner_buffers = getattr(owner, "buffers", None)
        if not isinstance(owner_buffers, Mapping):
            raise TypeError("pre-step owner factory must return a buffers owner")
        binding = path_c_direct_fusion_chain_runtime_binding_payload(
            chain=self.chain,
            logical_owner=owner,
            artifacts=self.artifacts,
        )
        if not bool(binding.get("runtime_uses_direct_fusion_chain")):
            raise ValueError(
                "pre-step direct-chain owner is not executable: "
                f"{binding.get('status')}"
            )
        self.last_pre_step_owner = owner
        self.last_pre_step_binding = binding
        self.binding = binding
        return None, owner

    def forward(self) -> dict[str, Any]:
        return run_path_c_direct_fusion_chain_route(
            chain=self.chain,
            logical_buffers=self.logical_buffers,
            logical_owner=self.logical_owner,
            artifacts=self.artifacts,
        )

    def backward(self) -> dict[str, Any]:
        return run_path_c_direct_fusion_chain_route(
            chain=self.chain,
            logical_buffers=self.logical_buffers,
            logical_owner=self.logical_owner,
            artifacts=self.artifacts,
        )

    def value_and_grad(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        loss_and_grad: Any,
    ) -> tuple[tuple[mx.array, mx.array], Any]:
        del loss_and_grad
        bridge_contract = _path_c_loss_cotangent_bridge_contract_payload(
            self.loss_cotangent_bridge
        )
        if bridge_contract.get("status") != "ok":
            raise ValueError(
                "direct-chain loss cotangent bridge is incomplete: "
                f"{bridge_contract.get('status')}"
            )
        logical_buffers, logical_owner = self._value_and_grad_logical_owner(
            model,
            batch,
        )
        run_path_c_direct_fusion_chain_route(
            chain=self.chain,
            logical_buffers=logical_buffers,
            logical_owner=logical_owner,
            artifacts=self.artifacts,
            execution_phases=("forward",),
        )
        buffers, _ = _path_c_direct_chain_resolved_logical_buffers(
            logical_buffers=logical_buffers,
            logical_owner=logical_owner,
        )
        required_loss_cotangent_buffers = (
            _path_c_direct_chain_loss_cotangent_seed_buffers(self.chain)
        )
        bridge_payload = self.loss_cotangent_bridge(
            model=model,
            batch=batch,
            logical_buffers=buffers,
            required_loss_cotangent_buffers=tuple(required_loss_cotangent_buffers),
            chain=self.chain,
        )
        if not isinstance(bridge_payload, Mapping):
            raise TypeError("loss cotangent bridge must return a Mapping payload")
        loss = bridge_payload.get("loss")
        ntokens = bridge_payload.get("ntokens")
        if not isinstance(loss, mx.array) or not isinstance(ntokens, mx.array):
            raise TypeError("loss cotangent bridge must return mx.array loss/ntokens")
        cotangents = bridge_payload.get("cotangents", {})
        if cotangents is None:
            cotangents = {}
        if not isinstance(cotangents, Mapping):
            raise TypeError("loss cotangent bridge cotangents must be a Mapping")
        required_specs = _path_c_direct_chain_required_logical_buffer_specs(self.chain)
        for name in required_loss_cotangent_buffers:
            value = cotangents.get(name, buffers.get(name))
            if not isinstance(value, mx.array):
                raise ValueError(
                    f"loss cotangent bridge did not provide {name!r}"
                )
            expected_dtype = required_specs.get(str(name), {}).get("dtype")
            if expected_dtype is not None:
                target_dtype = _mx_dtype_from_path_c_abi(str(expected_dtype))
                if value.dtype != target_dtype:
                    value = value.astype(target_dtype)
            buffers[name] = value
        run_path_c_direct_fusion_chain_route(
            chain=self.chain,
            logical_buffers=buffers,
            artifacts=self.artifacts,
            execution_phases=("backward",),
        )
        bridge_plan = path_c_direct_fusion_chain_value_and_grad_bridge_plan(
            chain=self.chain,
            model=model,
            logical_buffers=buffers,
        )
        parameter_grads = bridge_payload.get("parameter_grads", {})
        if parameter_grads is None:
            parameter_grads = {}
        if not isinstance(parameter_grads, Mapping):
            raise TypeError("loss cotangent bridge parameter_grads must be a Mapping")
        bridge_parameter_gradient_names = {
            _path_c_strip_gradient_suffix(str(name)) for name in parameter_grads
        }
        direct_parameter_gradient_names = [
            str(name)
            for name in bridge_plan["parameter_gradient_tree_names"]
            if _path_c_strip_gradient_suffix(str(name))
            not in bridge_parameter_gradient_names
        ]
        direct_grads = path_c_model_gradient_tree_from_direct_buffers(
            model=model,
            logical_buffers=buffers,
            parameter_gradient_names=direct_parameter_gradient_names,
        )
        coverage = _path_c_direct_chain_full_gradient_coverage_payload(
            model=model,
            chain=self.chain,
            bridge_contract=bridge_contract,
        )
        if coverage.get("full_model_gradient_tree_ready"):
            required_bridge_parameter_names = {
                str(name)
                for name in bridge_contract.get("parameter_gradient_names", ())
            }
            missing_bridge_parameters = sorted(
                required_bridge_parameter_names.difference(
                    str(name) for name in parameter_grads
                )
            )
            if missing_bridge_parameters:
                raise ValueError(
                    "loss cotangent bridge did not return required parameter "
                    f"grads: {missing_bridge_parameters}"
                )
            first_brick_name = _path_c_direct_chain_first_brick_name(self.chain)
            hidden_cotangent = (
                buffers.get(f"{first_brick_name}_hidden_grad")
                if first_brick_name is not None
                else None
            )
            if not isinstance(hidden_cotangent, mx.array):
                hidden_cotangent = buffers.get("hidden_grad")
            if not isinstance(hidden_cotangent, mx.array):
                raise ValueError(
                    "full-model Path C gradients require direct-chain hidden_grad"
                )
            direct_gradient_names = {
                _path_c_strip_gradient_suffix(str(name))
                for name in direct_parameter_gradient_names
            }
            start_layer_index = _path_c_direct_chain_start_layer_index(
                model,
                self.chain,
            )
            boundary_norm_name = (
                f"layers.{start_layer_index}.norm.weight"
                if start_layer_index is not None
                else None
            )
            direct_chain_owns_entry_rmsnorm_bwd = (
                boundary_norm_name is not None
                and boundary_norm_name in direct_gradient_names
            )
            raw_hidden_cotangent = (
                hidden_cotangent
                if direct_chain_owns_entry_rmsnorm_bwd
                else mx.zeros_like(hidden_cotangent)
            )
            normed_hidden_cotangent = (
                None
                if direct_chain_owns_entry_rmsnorm_bwd
                else hidden_cotangent
            )
            prefix_grads = path_c_prefix_gradient_tree_from_hidden_cotangent(
                model=model,
                batch=batch,
                hidden_cotangent=raw_hidden_cotangent,
                normed_hidden_cotangent=normed_hidden_cotangent,
                chain=self.chain,
            )
            direct_model_grads = _path_c_model_gradient_tree_strip_grad_suffixes(
                direct_grads
            )
            bridge_model_grads = _path_c_model_gradient_tree_from_parameter_grads(
                parameter_grads
            )
            covered_gradient_names = (
                _path_c_model_gradient_tree_array_names(prefix_grads)
                | _path_c_model_gradient_tree_array_names(direct_model_grads)
                | _path_c_model_gradient_tree_array_names(bridge_model_grads)
            )
            grads = _path_c_merge_model_gradient_trees(
                prefix_grads,
                direct_model_grads,
                bridge_model_grads,
                _path_c_zero_gradient_tree_for_parameters(
                    model,
                    frozenset(
                        str(name)
                        for name in coverage.get(
                            "inactive_zero_gradient_parameter_names",
                            (),
                        )
                        if str(name) not in covered_gradient_names
                    ),
                ),
            )
            returned_names = {
                str(name)
                for name, value in tree_flatten(grads)
                if isinstance(value, mx.array)
            }
            missing_names = sorted(
                _path_c_model_trainable_parameter_names(model).difference(
                    returned_names
                )
            )
            if missing_names:
                raise RuntimeError(
                    "Path C full-model gradient merge missed trainable "
                    f"parameters: {missing_names[:8]}"
                )
            return (loss, ntokens), grads

        grads = direct_grads
        if parameter_grads:
            grad_pairs = list(tree_flatten(grads))
            existing = {name for name, _ in grad_pairs}
            for raw_name, value in sorted(parameter_grads.items()):
                name = str(raw_name)
                if name in existing:
                    raise ValueError(
                        f"loss cotangent bridge duplicate parameter grad {name!r}"
                    )
                if not isinstance(value, mx.array):
                    raise TypeError(
                        f"loss cotangent bridge parameter grad {name!r} "
                        "must be an mx.array"
                    )
                grad_pairs.append((name, value))
            grads = tree_unflatten(grad_pairs)
        return (loss, ntokens), grads

    def value_and_grad_contract(self) -> Mapping[str, Any]:
        binding = self.training_graph_binding() or {}
        bridge_contract = _path_c_loss_cotangent_bridge_contract_payload(
            self.loss_cotangent_bridge
        )
        resolved_buffers, _ = _path_c_direct_chain_resolved_logical_buffers(
            logical_buffers=self.logical_buffers,
            logical_owner=self.logical_owner,
        )
        pre_step_owner_factory_available = callable(self.pre_step_owner_factory)
        model_gradient_tree_ready = (
            bool(resolved_buffers) or pre_step_owner_factory_available
        ) and bridge_contract.get("status") == "ok"
        loss_cotangent_bridge_ready = bridge_contract.get("status") == "ok"
        coverage = _path_c_direct_chain_full_gradient_coverage_payload(
            model=self.model,
            chain=self.chain,
            bridge_contract=bridge_contract,
        )
        returns_full_model_grads = bool(
            model_gradient_tree_ready
            and loss_cotangent_bridge_ready
            and coverage.get("full_model_gradient_tree_ready")
        )
        return {
            "contract": PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT,
            "owner": binding.get("owner"),
            "uses_direct_chain_runtime": bool(
                binding.get("uses_direct_chain_runtime")
            ),
            "uses_forward_hook": bool(binding.get("uses_forward_hook")),
            "uses_backward_or_vjp_hook": bool(
                binding.get("uses_backward_or_vjp_hook")
            ),
            "returns_model_grads": model_gradient_tree_ready,
            "returns_full_model_grads": returns_full_model_grads,
            "gradient_scope": "full_model"
            if returns_full_model_grads
            else "selected_region",
            "loss_cotangent_bridge_ready": loss_cotangent_bridge_ready,
            "model_gradient_tree_ready": model_gradient_tree_ready,
            "pre_step_owner_factory_available": pre_step_owner_factory_available,
            "last_pre_step_binding_status": None
            if self.last_pre_step_binding is None
            else self.last_pre_step_binding.get("status"),
            "selected_region_gradient_tree_ready": model_gradient_tree_ready,
            "full_model_gradient_tree_ready": returns_full_model_grads,
            "full_model_gradient_coverage": coverage,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
            "loss_cotangent_bridge_contract": bridge_contract,
            "reason": (
                "PathCDirectFusionChainTrainingRuntime returns a full MLX "
                "model-gradient tree from prefix VJP, caller-owned direct-chain "
                "buffers, and a contracted suffix loss bridge"
                if returns_full_model_grads
                else
                "PathCDirectFusionChainTrainingRuntime owns a contracted "
                "loss-cotangent bridge and returns an MLX model-gradient tree "
                "from caller-owned direct-chain buffers"
                if loss_cotangent_bridge_ready and model_gradient_tree_ready
                else
                "PathCDirectFusionChainTrainingRuntime can execute direct-chain "
                "segments, but it does not yet own a contracted loss-to-region "
                "cotangent bridge and caller-owned MLX model-gradient buffers"
            ),
        }

    def bind_training_graph(
        self,
        *,
        owner: str,
        uses_direct_chain_runtime: bool,
        uses_forward_hook: bool,
        uses_backward_or_vjp_hook: bool,
    ) -> None:
        self._training_graph_binding = {
            "owner": str(owner),
            "uses_direct_chain_runtime": bool(uses_direct_chain_runtime),
            "uses_forward_hook": bool(uses_forward_hook),
            "uses_backward_or_vjp_hook": bool(uses_backward_or_vjp_hook),
        }

    def unbind_training_graph(self, *, owner: str) -> None:
        if (
            self._training_graph_binding is not None
            and self._training_graph_binding.get("owner") == str(owner)
        ):
            self._training_graph_binding = None

    def training_graph_binding(self) -> Mapping[str, Any] | None:
        return self._training_graph_binding


def install_path_c_direct_chain_training_runtime_for_model(
    *,
    model: Any,
    chain: Any | None = None,
    artifacts: Any | None = None,
    logical_owner: Any | None = None,
    artifact_compiler: Callable[[Any], Any] = compile_path_c_direct_fusion_chain_artifacts,
    owner_name: str | None = None,
    sequence_length: int | None = None,
    training_critical_path: bool = False,
    run_probe: bool = False,
    loss_cotangent_bridge: Any | None = None,
    pre_step_owner_factory: Callable[[nn.Module, Mapping[str, mx.array]], Any]
    | None = None,
) -> dict[str, Any]:
    """Attach an explicit direct-chain runtime when binding is executable.

    This helper is deliberately injectable for tests and receipt probes; it
    never infers that the runtime is on the training critical path.
    """

    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    selected_chain = chain
    if selected_chain is None:
        direct_chain_region_prefix = _path_c_direct_chain_region_prefix(
            model,
            profile_name,
        )
        direct_chains = plan_path_c_direct_fusion_chains_for_model(
            model,
            region_prefix=direct_chain_region_prefix,
            include_backward=True,
            sequence_length=sequence_length,
        )
        regions = build_path_c_model_regions_from_model(
            model,
            region_prefix=direct_chain_region_prefix,
            include_backward=False,
            sequence_length=sequence_length,
        )
        selected_region = _select_path_c_model_route_region(regions)
        if selected_region is None:
            return {
                "status": "blocked",
                "reason": "model did not expose a Path C route region",
                "training_critical_path": False,
            }
        selected_chain = _select_path_c_direct_chain_for_region(
            direct_chains,
            selected_region,
        )
    if selected_chain is None:
        return {
            "status": "blocked",
            "reason": "model did not expose a direct Path C chain",
            "training_critical_path": False,
        }
    if str(getattr(selected_chain, "status", "")) != "ready":
        return {
            "status": "blocked",
            "reason": str(getattr(selected_chain, "reason", "direct chain is blocked")),
            "chain_status": str(getattr(selected_chain, "status", "")),
            "training_critical_path": False,
        }

    resolved_owner = (
        logical_owner
        if logical_owner is not None
        else _path_c_direct_chain_logical_owner_for_model(model)
    )
    if resolved_owner is None:
        return {
            "status": "blocked",
            "reason": "direct-chain runtime requires a logical buffer owner",
            "chain_status": str(getattr(selected_chain, "status", "")),
            "training_critical_path": False,
        }

    compiled_artifacts = artifacts if artifacts is not None else artifact_compiler(selected_chain)
    resolved_owner_name = owner_name or (
        f"{profile_name}.path_c_direct_fusion_chain_training_runtime"
    )
    runtime_binding = path_c_direct_fusion_chain_runtime_binding_payload(
        chain=selected_chain,
        logical_owner=resolved_owner,
        artifacts=compiled_artifacts,
    )
    if not bool(runtime_binding.get("runtime_uses_direct_fusion_chain")):
        return {
            "status": "blocked",
            "reason": "direct-chain runtime requires executable direct-chain binding",
            "runtime_owner": resolved_owner_name,
            "training_critical_path": False,
            "chain_status": str(getattr(selected_chain, "status", "")),
            "segment_count": len(tuple(getattr(selected_chain, "segments", ()))),
            "logical_buffer_owner": str(
                getattr(resolved_owner, "owner_name", "direct_chain_logical_owner")
            ),
            "artifact_count": len(tuple(compiled_artifacts))
            if isinstance(compiled_artifacts, (list, tuple))
            else None,
            "binding_status": runtime_binding.get("status"),
            "runtime_uses_direct_fusion_chain": False,
            "runtime_binding": runtime_binding,
            "execution": None,
        }
    runtime = PathCDirectFusionChainTrainingRuntime(
        chain=selected_chain,
        artifacts=compiled_artifacts,
        logical_owner=resolved_owner,
        owner_name=resolved_owner_name,
        training_critical_path=training_critical_path,
        loss_cotangent_bridge=loss_cotangent_bridge,
        model=model,
        pre_step_owner_factory=pre_step_owner_factory,
    )
    if training_critical_path:
        runtime.bind_training_graph(
            owner="CompiledPretrainingStep",
            uses_direct_chain_runtime=True,
            uses_forward_hook=True,
            uses_backward_or_vjp_hook=True,
        )
        value_and_grad_contract = _direct_chain_value_and_grad_contract_payload(runtime)
        training_runtime_contract = (
            path_c_direct_fusion_chain_training_runtime_contract_payload(
                training_runtime=runtime,
                runtime_binding=runtime.binding,
            )
        )
        contract_block_reason = None
        if value_and_grad_contract.get("status") != "ok":
            contract_block_reason = "direct-chain value_and_grad runtime incomplete"
        elif not bool(value_and_grad_contract.get("returns_full_model_grads", False)):
            contract_block_reason = "direct-chain full-model gradients incomplete"
        elif training_runtime_contract.get("status") != "ok":
            contract_block_reason = "direct-chain training runtime incomplete"
        if contract_block_reason is not None:
            runtime.unbind_training_graph(owner="CompiledPretrainingStep")
            return {
                "status": "blocked",
                "reason": contract_block_reason,
                "runtime_owner": resolved_owner_name,
                "runtime_class": type(runtime).__name__,
                "training_critical_path": False,
                "chain_status": str(getattr(selected_chain, "status", "")),
                "segment_count": len(tuple(getattr(selected_chain, "segments", ()))),
                "logical_buffer_owner": str(
                    getattr(resolved_owner, "owner_name", "direct_chain_logical_owner")
                ),
                "artifact_count": len(tuple(compiled_artifacts))
                if isinstance(compiled_artifacts, (list, tuple))
                else None,
                "binding_status": runtime.binding.get("status"),
                "runtime_uses_direct_fusion_chain": bool(
                    runtime.binding.get("runtime_uses_direct_fusion_chain")
                ),
                "runtime_binding": runtime.binding,
                "value_and_grad_contract": value_and_grad_contract,
                "training_runtime_contract": training_runtime_contract,
                "execution": None,
            }
    model.path_c_direct_fusion_chain_logical_buffer_owner = resolved_owner
    model.path_c_direct_fusion_chain_logical_buffer_owner_name = str(
        getattr(resolved_owner, "owner_name", "direct_chain_logical_owner")
    )
    model.path_c_direct_fusion_chain_artifacts = compiled_artifacts
    model.path_c_direct_fusion_chain_training_runtime = runtime
    execution_payload = runtime.forward() if run_probe else None
    return {
        "status": "ok",
        "runtime_owner": resolved_owner_name,
        "runtime_class": type(runtime).__name__,
        "training_critical_path": bool(training_critical_path),
        "chain_status": str(getattr(selected_chain, "status", "")),
        "segment_count": len(tuple(getattr(selected_chain, "segments", ()))),
        "logical_buffer_owner": str(
            getattr(resolved_owner, "owner_name", "direct_chain_logical_owner")
        ),
        "artifact_count": len(tuple(compiled_artifacts))
        if isinstance(compiled_artifacts, (list, tuple))
        else None,
        "binding_status": runtime.binding.get("status"),
        "runtime_uses_direct_fusion_chain": bool(
            runtime.binding.get("runtime_uses_direct_fusion_chain")
        ),
        "runtime_binding": runtime.binding,
        "value_and_grad_contract": _direct_chain_value_and_grad_contract_payload(
            runtime
        ),
        "execution": execution_payload,
        "reason": (
            "direct-chain runtime was attached for explicit profiling/binding "
            "evidence; critical-path readiness is reported separately"
        ),
    }


def path_c_direct_chain_value_and_grad_probe_payload(
    *,
    runtime: Any,
    model: nn.Module,
    batch: Mapping[str, mx.array] | mx.array,
) -> dict[str, Any]:
    """Execute the direct-chain value-and-grad bridge as post-step evidence."""

    def _forbidden_loss_and_grad(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Path C direct-chain probe must not delegate to eager")

    started = time.perf_counter()
    (loss, ntokens), grads = runtime.value_and_grad(
        model,
        batch,
        _forbidden_loss_and_grad,
    )
    flat_grads = [
        (str(name), value)
        for name, value in tree_flatten(grads)
        if isinstance(value, mx.array)
    ]
    mx.eval(loss, ntokens, *(value for _, value in flat_grads))
    value_and_grad_contract = _direct_chain_value_and_grad_contract_payload(runtime)
    training_runtime_contract = path_c_direct_fusion_chain_training_runtime_contract_payload(
        training_runtime=runtime,
        runtime_binding=getattr(runtime, "binding", {}),
    )
    return {
        "status": "ok",
        "execution_phase": "post_step_profile_probe",
        "training_critical_path": False,
        "delegated_to_eager_loss_and_grad": False,
        "elapsed_s": time.perf_counter() - started,
        "loss": float(loss.item()),
        "ntokens": int(ntokens.item()),
        "gradient_count": len(flat_grads),
        "gradient_name_examples": [name for name, _ in flat_grads[:8]],
        "value_and_grad_contract": value_and_grad_contract,
        "training_runtime_contract": training_runtime_contract,
    }


def _path_c_shape_tuple(shape: Any) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in tuple(shape))


def _path_c_shape_extent(shape: Any) -> int:
    extent = 1
    for dim in _path_c_shape_tuple(shape):
        extent *= int(dim)
    return extent


def _path_c_preferred_direct_chain_shape(
    current: tuple[int, ...],
    candidate: tuple[int, ...],
) -> tuple[int, ...]:
    if current == candidate:
        return current
    if len(candidate) > len(current):
        return candidate
    return current


def _path_c_merge_direct_chain_buffer_spec(
    specs: dict[str, dict[str, Any]],
    *,
    name: str,
    shape: Any,
    dtype: str,
    category: str,
    segment_index: int,
    source: str,
) -> None:
    shape_tuple = _path_c_shape_tuple(shape)
    dtype_name = str(dtype)
    existing = specs.setdefault(
        name,
        {
            "name": name,
            "shape": shape_tuple,
            "dtype": dtype_name,
            "category": str(category),
            "segments": [],
        },
    )
    if str(existing["dtype"]) != dtype_name:
        raise ValueError(
            f"conflicting direct-chain {source} buffer dtype {name!r}: "
            f"{existing['dtype']!r} vs {dtype_name!r}"
        )
    existing_shape = _path_c_shape_tuple(existing["shape"])
    if existing_shape != shape_tuple:
        existing_extent = _path_c_shape_extent(existing_shape)
        shape_extent = _path_c_shape_extent(shape_tuple)
        if source == "scratch":
            existing["shape"] = (
                shape_tuple if shape_extent > existing_extent else existing_shape
            )
        elif existing_extent != shape_extent:
            raise ValueError(
                f"conflicting direct-chain {source} buffer shape {name!r}: "
                f"{existing_shape!r} vs {shape_tuple!r}"
            )
        else:
            existing["shape"] = _path_c_preferred_direct_chain_shape(
                existing_shape,
                shape_tuple,
            )
    existing_category = str(existing.get("category", ""))
    if existing_category == "runtime_scratch" and str(category) != "runtime_scratch":
        existing["category"] = str(category)
    existing["segments"].append(int(segment_index))


def _path_c_direct_chain_required_logical_buffer_specs(
    chain: Any,
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    suffix_shape_env: Any | None = None
    suffix_segment_index = -1
    for segment in getattr(chain, "segments", ()):
        target = getattr(segment, "schedule_target", None)
        if target is None or str(segment.status) != "ok":
            continue
        prim_func = target.schedule_template(segment.region)
        suffix_segment_index = max(suffix_segment_index, int(segment.index))
        if suffix_shape_env is None:
            suffix_shape_env = getattr(prim_func, "_cppmega_path_c_shape_env", None)
        physical_abi_map = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        physical_abi_shapes = dict(
            getattr(
                prim_func,
                "_cppmega_path_c_physical_buffer_abi_shapes",
                {},
            )
            or {}
        )
        bridge = plan_physical_abi_runtime_bridge(
            physical_abi_map,
            physical_abi_shapes,
        )
        bank_dtypes = dict(bridge.get("bank_dtypes", {}))
        for raw_name in bridge.get("required_bank_buffers", ()):
            name = str(raw_name)
            shape = tuple(int(dim) for dim in tuple(physical_abi_shapes[name]))
            dtype = str(bank_dtypes[name])
            _path_c_merge_direct_chain_buffer_spec(
                specs,
                name=name,
                shape=shape,
                dtype=dtype,
                category=_path_c_direct_chain_buffer_category(name),
                segment_index=int(segment.index),
                source="logical",
            )
        for name, scratch_spec in _path_c_internal_scratch_abi_specs(
            prim_func
        ).items():
            _path_c_merge_direct_chain_buffer_spec(
                specs,
                name=str(name),
                shape=scratch_spec["shape"],
                dtype=str(scratch_spec["dtype"]),
                category=_path_c_direct_chain_buffer_category(str(name)),
                segment_index=int(segment.index),
                source="scratch",
            )
    if (
        suffix_shape_env is not None
        and _path_c_direct_chain_loss_cotangent_seed_buffers(chain)
    ):
        hidden = int(getattr(suffix_shape_env, "hidden_size", 0) or 0)
        vocab = int(getattr(suffix_shape_env, "vocab_size", 0) or 0)
        if hidden > 0:
            suffix_specs = {
                "final_norm_weight_grad": (hidden,),
                "lm_head_weight_grad": (max(1, vocab) * hidden,),
            }
            for name, shape in suffix_specs.items():
                _path_c_merge_direct_chain_buffer_spec(
                    specs,
                    name=name,
                    shape=shape,
                    dtype="float32",
                    category="runtime_activation_or_grad",
                    segment_index=suffix_segment_index,
                    source="suffix grad",
                )
    return specs


def _path_c_internal_scratch_abi_specs(prim_func: Any) -> dict[str, dict[str, Any]]:
    scratch_shapes = dict(
        getattr(prim_func, "_cppmega_path_c_spilled_shared_scratch_shapes", {})
        or {}
    )
    specs: dict[str, dict[str, Any]] = {}
    coalesced_banks: dict[str, dict[str, Any]] = {}
    for raw_name, raw in scratch_shapes.items():
        if not isinstance(raw, Mapping):
            continue
        dtype = str(raw.get("dtype", "float32"))
        shape = _path_c_shape_tuple(raw.get("shape", ()))
        if bool(raw.get("coalesced_scratch_bank")):
            name = str(raw.get("param_name") or raw.get("bank") or raw_name)
            offset = int(raw.get("offset", 0) or 0)
            extent = _path_c_shape_extent(shape)
            existing = coalesced_banks.setdefault(
                name,
                {
                    "shape": (0,),
                    "dtype": dtype,
                },
            )
            if str(existing["dtype"]) != dtype:
                raise ValueError(
                    f"conflicting internal scratch bank dtype {name!r}: "
                    f"{existing['dtype']!r} vs {dtype!r}"
                )
            existing["shape"] = (
                max(_path_c_shape_extent(existing["shape"]), offset + extent),
            )
            continue
        name = str(raw.get("param_name", raw_name))
        specs[name] = {
            "shape": shape,
            "dtype": dtype,
        }
    specs.update(coalesced_banks)
    return specs


def _path_c_tensor_dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).rsplit(".", 1)[-1]


def _path_c_validate_internal_scratch_abi_buffers(
    scratch_specs: Mapping[str, Mapping[str, Any]],
    buffers: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provided = {str(name): value for name, value in (buffers or {}).items()}
    missing: list[str] = []
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    errors: list[str] = []
    for name, spec in scratch_specs.items():
        if name not in provided:
            missing.append(name)
            errors.append(f"{name}: missing caller-owned internal scratch buffer")
            continue
        expected_shape = _path_c_shape_tuple(spec.get("shape", ()))
        actual_shape_raw = getattr(provided[name], "shape", None)
        actual_shape = (
            None if actual_shape_raw is None else _path_c_shape_tuple(actual_shape_raw)
        )
        expected_extent = _path_c_shape_extent(expected_shape)
        actual_extent = (
            None if actual_shape is None else _path_c_shape_extent(actual_shape)
        )
        coalesced_scratch_bank = (
            name.startswith("path_c_") and name.endswith("_scratch_bank")
        )
        shape_compatible = bool(
            actual_shape == expected_shape
            or actual_extent == expected_extent
            or (
                coalesced_scratch_bank
                and actual_extent is not None
                and actual_extent >= expected_extent
            )
        )
        if not shape_compatible:
            shape_mismatches.append(name)
            errors.append(
                f"{name}: shape {actual_shape} is not compatible with expected "
                f"{expected_shape}"
            )
        expected_dtype = str(spec.get("dtype", "float32"))
        actual_dtype = _path_c_tensor_dtype_name(provided[name])
        if actual_dtype != expected_dtype:
            dtype_mismatches.append(name)
            errors.append(
                f"{name}: dtype {actual_dtype!r} does not match expected {expected_dtype!r}"
            )
    return {
        "status": "ok" if not errors else "failed",
        "required_internal_scratch_buffers": list(scratch_specs),
        "missing_internal_scratch_buffers": missing,
        "shape_mismatch_internal_scratch_buffers": shape_mismatches,
        "dtype_mismatch_internal_scratch_buffers": dtype_mismatches,
        "errors": errors,
    }


def _path_c_direct_chain_loss_cotangent_seed_buffers(chain: Any) -> list[str]:
    """Return external gradient inputs that seed direct-chain backward.

    These are boundary cotangents consumed by owner-output backward nodes. They
    are inputs to the full region but are not produced by any node inside it.
    """

    source_region = getattr(chain, "source_region", None)
    regions = (
        (source_region,)
        if source_region is not None and getattr(source_region, "nodes", None)
        else tuple(
            getattr(segment, "region", None)
            for segment in getattr(chain, "segments", ())
        )
    )
    nodes = [
        node
        for region in regions
        if region is not None
        for node in getattr(region, "nodes", ())
    ]
    produced = {
        str(output)
        for node in nodes
        for output in getattr(node, "outputs", ())
    }
    external_gradient_inputs = {
        str(input_name)
        for node in nodes
        if str(getattr(node, "backward", "")) == "owner_output"
        for input_name in getattr(node, "inputs", ())
        if str(input_name).endswith("_grad") and str(input_name) not in produced
    }
    return sorted(external_gradient_inputs)


def _path_c_direct_chain_resolved_logical_buffers(
    *,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
) -> tuple[dict[str, Any], str | None]:
    if logical_buffers is not None and logical_owner is not None:
        raise ValueError("logical_buffers and logical_owner are mutually exclusive")
    if logical_owner is not None:
        owner_buffers = getattr(logical_owner, "buffers", None)
        if not isinstance(owner_buffers, Mapping):
            raise TypeError("logical_owner must expose a Mapping buffers attribute")
        return {str(name): value for name, value in owner_buffers.items()}, str(
            getattr(logical_owner, "owner_name", type(logical_owner).__name__)
        )
    if logical_buffers is None:
        return {}, None
    return {str(name): value for name, value in logical_buffers.items()}, None


def _path_c_model_gradient_tree_pairs_from_direct_buffers(
    *,
    model: Any,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
    parameter_gradient_names: Sequence[str] | None = None,
) -> tuple[list[tuple[str, mx.array]], dict[str, Any]]:
    buffers, owner_name = _path_c_direct_chain_resolved_logical_buffers(
        logical_buffers=logical_buffers,
        logical_owner=logical_owner,
    )
    if not callable(getattr(model, "path_c_parameter_gradient_aliases", None)):
        return [], {
            "status": "blocked",
            "reason": "model does not expose path_c_parameter_gradient_aliases",
            "owner_name": owner_name,
            "parameter_gradient_alias_count": 0,
            "gradient_tree_ready": False,
            "missing_parameter_gradient_names": [],
            "missing_logical_gradient_buffers": [],
            "non_array_logical_gradient_buffers": [],
            "mapped_parameter_gradient_count": 0,
        }
    parameter_gradient_aliases = model.path_c_parameter_gradient_aliases()
    if parameter_gradient_names is not None:
        selected_names = {str(name) for name in parameter_gradient_names}
        parameter_gradient_aliases = {
            str(name): targets
            for name, targets in parameter_gradient_aliases.items()
            if str(name) in selected_names
        }
    pairs: list[tuple[str, mx.array]] = []
    mapped_logical_buffers: dict[str, str] = {}
    missing_parameter_gradient_names: list[str] = []
    missing_logical_gradient_buffers: list[str] = []
    non_array_logical_gradient_buffers: list[str] = []
    for parameter_name, raw_targets in sorted(parameter_gradient_aliases.items()):
        targets = (raw_targets,) if isinstance(raw_targets, str) else tuple(raw_targets)
        selected: tuple[str, mx.array] | None = None
        missing_targets: list[str] = []
        for raw_target in targets:
            target = str(raw_target)
            if target not in buffers:
                missing_targets.append(target)
                continue
            value = buffers[target]
            if not isinstance(value, mx.array):
                non_array_logical_gradient_buffers.append(target)
                continue
            selected = (target, value)
            break
        if selected is None:
            missing_parameter_gradient_names.append(str(parameter_name))
            missing_logical_gradient_buffers.extend(missing_targets)
            continue
        logical_name, tensor = selected
        pairs.append((str(parameter_name), tensor))
        mapped_logical_buffers[str(parameter_name)] = logical_name
    status = (
        "ok"
        if len(pairs) == len(parameter_gradient_aliases)
        and not non_array_logical_gradient_buffers
        else "blocked"
    )
    return pairs, {
        "status": status,
        "reason": "all Path C parameter gradient buffers map to MLX gradient tree"
        if status == "ok"
        else "missing or non-array Path C parameter gradient buffers",
        "owner_name": owner_name,
        "parameter_gradient_alias_count": len(parameter_gradient_aliases),
        "gradient_tree_ready": status == "ok",
        "mapped_parameter_gradient_count": len(pairs),
        "mapped_logical_gradient_buffers": mapped_logical_buffers,
        "missing_parameter_gradient_names": missing_parameter_gradient_names,
        "missing_logical_gradient_buffers": sorted(set(missing_logical_gradient_buffers)),
        "non_array_logical_gradient_buffers": sorted(
            set(non_array_logical_gradient_buffers)
        ),
    }


def path_c_model_gradient_tree_extraction_payload(
    *,
    model: Any,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
    parameter_gradient_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report whether direct-chain logical grad buffers can form MLX grads."""

    _, payload = _path_c_model_gradient_tree_pairs_from_direct_buffers(
        model=model,
        logical_buffers=logical_buffers,
        logical_owner=logical_owner,
        parameter_gradient_names=parameter_gradient_names,
    )
    return payload


def path_c_model_gradient_tree_from_direct_buffers(
    *,
    model: Any,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
    parameter_gradient_names: Sequence[str] | None = None,
) -> Any:
    """Build an MLX model-gradient tree from direct-chain logical buffers."""

    pairs, payload = _path_c_model_gradient_tree_pairs_from_direct_buffers(
        model=model,
        logical_buffers=logical_buffers,
        logical_owner=logical_owner,
        parameter_gradient_names=parameter_gradient_names,
    )
    if payload.get("status") != "ok":
        raise ValueError(str(payload.get("reason", "gradient tree is not ready")))
    return tree_unflatten(pairs)


def _mx_dtype_from_path_c_abi(dtype: str) -> mx.Dtype:
    mapping = {
        "bool": mx.bool_,
        "uint8": mx.uint8,
        "int8": mx.int8,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
        "int32": mx.int32,
    }
    try:
        return mapping[str(dtype)]
    except KeyError as exc:
        raise ValueError(f"unsupported Path C logical workspace dtype {dtype!r}") from exc


def make_path_c_direct_fusion_chain_workspace_owner(
    *,
    chain: Any,
    logical_buffer_names: Sequence[str],
    owner_name: str,
) -> PathCLogicalBufferOwner:
    """Allocate explicit caller-owned direct-chain workspace/output buffers.

    These buffers are not packed or copied from model tensors. They are explicit
    output targets for direct-chain segment execution, primarily backward bridge
    gradients that MLX autograd does not expose as model parameter gradients.
    """

    specs = _path_c_direct_chain_required_logical_buffer_specs(chain)
    buffers: dict[str, mx.array] = {}
    for raw_name in logical_buffer_names:
        name = str(raw_name)
        spec = specs.get(name)
        if spec is None:
            raise KeyError(f"unknown direct-chain logical buffer {name!r}")
        if not name.endswith("_grad"):
            raise ValueError(
                "direct-chain workspace allocation is limited to explicit "
                f"gradient output buffers; got {name!r}"
            )
        buffers[name] = mx.zeros(
            tuple(spec["shape"]),
            dtype=_mx_dtype_from_path_c_abi(str(spec["dtype"])),
        )
    return PathCLogicalBufferOwner(
        owner_name=str(owner_name),
        buffers=buffers,
        hidden_packing_performed=False,
        no_hidden_allocation_policy=True,
    )


def make_path_c_direct_chain_pre_step_runtime_owner(
    *,
    chain: Any,
    model: nn.Module,
    batch: Mapping[str, mx.array] | mx.array,
    owner_name: str | None = None,
) -> PathCLogicalBufferOwner:
    """Build per-step direct-chain buffers from model params and the batch.

    This owner is derived from the dynamic chain ABI.  It does not depend on
    post-step captures and does not use named acceptance fixtures.
    """

    start_layer_index = _path_c_direct_chain_start_layer_index(model, chain)
    if start_layer_index is None:
        raise ValueError("cannot resolve Path C direct-chain start layer")
    first_brick_name = _path_c_direct_chain_first_brick_name(chain)
    if not first_brick_name:
        raise ValueError("cannot resolve Path C direct-chain first brick")

    required_specs = _path_c_direct_chain_required_logical_buffer_specs(chain)
    by_category = _path_c_direct_chain_names_by_category(required_specs)
    make_model_owner = getattr(
        model,
        "make_path_c_direct_fusion_chain_logical_buffer_owner",
        None,
    )
    if not callable(make_model_owner):
        raise TypeError(
            "model must expose make_path_c_direct_fusion_chain_logical_buffer_owner"
        )
    model_owner = make_model_owner()
    model_buffers = getattr(model_owner, "buffers", None)
    if not isinstance(model_buffers, Mapping):
        raise TypeError("model direct-chain owner must expose Mapping buffers")

    prefix_hidden = path_c_model_prefix_hidden_states(
        model,
        batch,
        end_layer_index=start_layer_index,
    )
    layers = tuple(getattr(model, "layers", ()))
    boundary_hidden = prefix_hidden
    if start_layer_index < len(layers):
        boundary_hidden = layers[start_layer_index].norm(prefix_hidden)
    hidden_seed_names = {
        "hidden",
        f"{first_brick_name}_hidden",
    }
    buffers: dict[str, mx.array] = {}
    missing_model_buffers: list[str] = []
    model_parameter_names = set(by_category.get("model_parameter_or_constant", ()))
    for name in sorted(required_specs):
        spec = required_specs[name]
        if name in model_parameter_names:
            value = model_buffers.get(name)
            if isinstance(value, mx.array):
                buffers[name] = value
            else:
                missing_model_buffers.append(name)
            continue
        if name in hidden_seed_names:
            buffers[name] = boundary_hidden
            continue
        buffers[name] = mx.zeros(
            tuple(spec["shape"]),
            dtype=_mx_dtype_from_path_c_abi(str(spec["dtype"])),
        )
    if missing_model_buffers:
        raise ValueError(
            "model direct-chain owner is missing required parameter buffers: "
            f"{missing_model_buffers[:8]}"
        )

    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    return PathCLogicalBufferOwner(
        owner_name=owner_name or f"{profile_name}.path_c_pre_step_runtime_buffers",
        buffers=buffers,
        hidden_packing_performed=False,
        no_hidden_allocation_policy=True,
    )


def _path_c_direct_chain_buffer_category(name: str) -> str:
    if name in {"hidden", "hidden_grad"}:
        return "runtime_activation_or_grad"
    if name.endswith("_grad"):
        return "runtime_activation_or_grad"
    if any(
        name.endswith(suffix)
        for suffix in (
            "_sparse_mla_sm_scale",
            "_sparse_mla_sinks",
            "_sparse_mla_has_sinks",
        )
    ):
        return "runtime_activation_or_grad"
    if any(
        name.endswith(suffix)
        for suffix in (
            "_weight",
            "_bias",
            "_D",
            "_A_log",
            "_dt_bias",
            "_rope_inv_freq",
        )
    ):
        return "model_parameter_or_constant"
    if any(
        name.endswith(suffix)
        for suffix in (
            "_h0",
            "_state",
            "_state_in",
            "_conv_state",
        )
    ):
        return "runtime_state"
    return "runtime_activation_or_grad"


def _path_c_direct_chain_names_by_category(
    required_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = {}
    for name, spec in required_specs.items():
        by_category.setdefault(str(spec.get("category")), []).append(str(name))
    for names in by_category.values():
        names.sort()
    return by_category


def path_c_direct_chain_pre_step_owner_plan(
    *,
    chain: Any,
    model: Any | None = None,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
) -> dict[str, Any]:
    """Classify which direct-chain buffers can exist before the train step."""

    if logical_buffers is None and logical_owner is None and model is not None:
        make_logical_owner = getattr(
            model,
            "make_path_c_direct_fusion_chain_logical_buffer_owner",
            None,
        )
        if callable(make_logical_owner):
            logical_owner = make_logical_owner()

    resolved_buffers, owner_name = _path_c_direct_chain_resolved_logical_buffers(
        logical_buffers=logical_buffers,
        logical_owner=logical_owner,
    )
    required_specs = _path_c_direct_chain_required_logical_buffer_specs(chain)
    by_category = _path_c_direct_chain_names_by_category(required_specs)
    runtime_activation_or_grad = by_category.get("runtime_activation_or_grad", [])
    model_parameter_names = by_category.get("model_parameter_or_constant", [])
    runtime_state_names = by_category.get("runtime_state", [])
    runtime_scratch_names = by_category.get("runtime_scratch", [])
    forward_or_prepared_names = sorted(
        name for name in runtime_activation_or_grad if not name.endswith("_grad")
    )
    backward_gradient_names = sorted(
        name for name in runtime_activation_or_grad if name.endswith("_grad")
    )

    def available(names: Sequence[str]) -> list[str]:
        return sorted(
            name for name in names if isinstance(resolved_buffers.get(name), mx.array)
        )

    model_parameter_available = available(model_parameter_names)
    forward_or_prepared_available = available(forward_or_prepared_names)
    runtime_state_available = available(runtime_state_names)
    runtime_scratch_available = available(runtime_scratch_names)
    backward_gradient_available = available(backward_gradient_names)
    forward_or_prepared_missing = sorted(
        set(forward_or_prepared_names).difference(forward_or_prepared_available)
    )
    runtime_state_missing = sorted(
        set(runtime_state_names).difference(runtime_state_available)
    )
    runtime_scratch_missing = sorted(
        set(runtime_scratch_names).difference(runtime_scratch_available)
    )
    pre_step_runtime_missing = [
        *forward_or_prepared_missing,
        *runtime_state_missing,
        *runtime_scratch_missing,
    ]
    status = (
        "pre_step_runtime_owner_missing"
        if pre_step_runtime_missing
        else "pre_step_runtime_owner_ready"
    )
    return {
        "status": status,
        "training_critical_path_ready": status == "pre_step_runtime_owner_ready",
        "logical_buffer_owner": owner_name,
        "required_logical_buffer_count": len(required_specs),
        "model_parameter_or_constant_count": len(model_parameter_names),
        "model_parameter_or_constant_available_count": len(model_parameter_available),
        "model_parameter_or_constant_missing_count": (
            len(model_parameter_names) - len(model_parameter_available)
        ),
        "batch_dependent_forward_or_prepared_count": len(forward_or_prepared_names),
        "batch_dependent_forward_or_prepared_available_count": len(
            forward_or_prepared_available
        ),
        "batch_dependent_forward_or_prepared_missing_count": len(
            forward_or_prepared_missing
        ),
        "runtime_state_count": len(runtime_state_names),
        "runtime_state_available_count": len(runtime_state_available),
        "runtime_state_missing_count": len(runtime_state_missing),
        "runtime_scratch_count": len(runtime_scratch_names),
        "runtime_scratch_available_count": len(runtime_scratch_available),
        "runtime_scratch_missing_count": len(runtime_scratch_missing),
        "pre_step_runtime_missing_count": len(pre_step_runtime_missing),
        "backward_workspace_gradient_count": len(backward_gradient_names),
        "backward_workspace_gradient_available_count": len(
            backward_gradient_available
        ),
        "backward_workspace_gradient_missing_count": (
            len(backward_gradient_names) - len(backward_gradient_available)
        ),
        "batch_dependent_forward_or_prepared_missing_examples": (
            forward_or_prepared_missing[:12]
        ),
        "runtime_state_missing_examples": runtime_state_missing[:12],
        "runtime_scratch_missing_examples": runtime_scratch_missing[:12],
        "backward_workspace_gradient_examples": backward_gradient_names[:12],
        "reason": (
            "direct-chain training cannot be bound before the step until a "
            "runtime owner produces the batch-dependent forward/prepared "
            "buffers and recurrent state buffers without relying on a prior "
            "eager Path C step"
            if pre_step_runtime_missing
            else "all batch-dependent direct-chain inputs have an explicit "
            "pre-step owner; remaining gradient buffers can be allocated as "
            "workspace for the VJP"
        ),
    }


def path_c_direct_fusion_chain_model_binding_audit(
    *,
    chain: Any,
    model: Any | None = None,
) -> dict[str, Any]:
    """Classify direct-chain buffers by the owner needed at live runtime."""

    required_buffers = _path_c_direct_chain_required_logical_buffer_specs(chain)
    by_category = _path_c_direct_chain_names_by_category(required_buffers)
    runtime_activation_or_grad = by_category.get("runtime_activation_or_grad", [])
    backward_grad_names = sorted(
        name for name in runtime_activation_or_grad if name.endswith("_grad")
    )
    forward_activation_names = sorted(
        name for name in runtime_activation_or_grad if not name.endswith("_grad")
    )
    runtime_state_names = by_category.get("runtime_state", [])
    runtime_scratch_names = by_category.get("runtime_scratch", [])
    runtime_names = sorted(
        [*runtime_activation_or_grad, *runtime_state_names, *runtime_scratch_names]
    )
    requires_runtime_owner = bool(runtime_names)
    forward_probe_available = bool(
        model is not None
        and callable(getattr(model, "attach_path_c_activation_probe", None))
    )
    profile_bricks_attached = bool(
        model is not None
        and all(
            hasattr(layer, "path_c_profile_brick_name")
            for layer in getattr(model, "layers", ())
        )
    )
    parameter_gradient_probe_available = callable(
        getattr(CompiledPretrainingStep, "attach_path_c_gradient_probe", None)
    )
    parameter_gradient_aliases: Mapping[str, Any] = {}
    if model is not None and callable(
        getattr(model, "path_c_parameter_gradient_aliases", None)
    ):
        parameter_gradient_aliases = model.path_c_parameter_gradient_aliases()
    model_logical_owner = None
    make_logical_owner = (
        getattr(model, "make_path_c_direct_fusion_chain_logical_buffer_owner", None)
        if model is not None
        else None
    )
    if callable(make_logical_owner):
        model_logical_owner = make_logical_owner()
    model_logical_owner_buffers = getattr(model_logical_owner, "buffers", {}) or {}
    model_parameter_names = by_category.get("model_parameter_or_constant", [])
    model_logical_owner_required_buffers = sorted(
        name for name in model_parameter_names if name in model_logical_owner_buffers
    )
    parameter_gradient_alias_targets = sorted(
        {
            str(target)
            for targets in parameter_gradient_aliases.values()
            for target in (
                (targets,) if isinstance(targets, str) else tuple(targets)
            )
        }
    )
    covered_parameter_gradient_names = sorted(
        set(backward_grad_names).intersection(parameter_gradient_alias_targets)
    )
    uncovered_backward_grad_names = sorted(
        set(backward_grad_names).difference(covered_parameter_gradient_names)
    )
    status = (
        "runtime_backward_or_state_owner_missing"
        if requires_runtime_owner and forward_probe_available
        else "runtime_activation_owner_missing"
        if requires_runtime_owner
        else "model_parameter_owner_sufficient"
    )
    return {
        "status": status,
        "required_logical_buffer_count": len(required_buffers),
        "model_parameter_or_constant_count": len(
            by_category.get("model_parameter_or_constant", [])
        ),
        "runtime_activation_or_grad_count": len(
            by_category.get("runtime_activation_or_grad", [])
        ),
        "forward_activation_or_prepared_count": len(forward_activation_names),
        "backward_gradient_count": len(backward_grad_names),
        "runtime_state_count": len(by_category.get("runtime_state", [])),
        "requires_runtime_activation_owner": requires_runtime_owner,
        "forward_activation_probe_surface_available": forward_probe_available,
        "parameter_gradient_probe_surface_available": (
            parameter_gradient_probe_available
        ),
        "parameter_gradient_alias_count": len(parameter_gradient_aliases),
        "model_parameter_logical_owner_available": model_logical_owner is not None,
        "model_parameter_logical_owner": None
        if model_logical_owner is None
        else str(getattr(model_logical_owner, "owner_name", "")),
        "model_parameter_logical_owner_buffer_count": len(
            model_logical_owner_required_buffers
        ),
        "model_parameter_logical_owner_total_buffer_count": len(
            model_logical_owner_buffers
        ),
        "model_parameter_logical_owner_buffer_examples": (
            model_logical_owner_required_buffers[:12]
        ),
        "backward_gradient_parameter_alias_coverage_count": len(
            covered_parameter_gradient_names
        ),
        "backward_gradient_uncovered_count": len(uncovered_backward_grad_names),
        "profile_brick_names_attached": profile_bricks_attached,
        "forward_activation_or_prepared_examples": forward_activation_names[:12],
        "backward_gradient_examples": backward_grad_names[:12],
        "backward_gradient_parameter_alias_coverage_examples": (
            covered_parameter_gradient_names[:12]
        ),
        "backward_gradient_uncovered_examples": uncovered_backward_grad_names[:12],
        "runtime_activation_or_grad_examples": runtime_names[:12],
        "categories": {category: len(names) for category, names in by_category.items()},
        "reason": (
            "model exposes an explicit zero-copy forward activation probe, but "
            "direct-chain runtime still needs the remaining gradient/state/"
            "prepared-buffer owners produced inside the train step before it "
            "can execute"
            if requires_runtime_owner and forward_probe_available
            else "direct-chain runtime needs activations, prepared FP8 buffers, states, "
            "and gradients produced inside the train step; model parameters alone "
            "cannot satisfy this ABI without hidden staging"
            if requires_runtime_owner
            else "direct-chain runtime can be satisfied by model-owned parameters"
        ),
    }


def path_c_direct_fusion_chain_value_and_grad_bridge_plan(
    *,
    chain: Any,
    model: Any | None = None,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_owner: Any | None = None,
) -> dict[str, Any]:
    """Plan the non-eager loss/VJP bridge required for direct-chain training."""

    required_specs = _path_c_direct_chain_required_logical_buffer_specs(chain)
    required_gradient_buffers = sorted(
        name
        for name, spec in required_specs.items()
        if str(spec.get("category")) == "runtime_activation_or_grad"
        and name.endswith("_grad")
    )
    parameter_gradient_aliases: Mapping[str, Any] = {}
    if model is not None and callable(
        getattr(model, "path_c_parameter_gradient_aliases", None)
    ):
        parameter_gradient_aliases = model.path_c_parameter_gradient_aliases()
    target_to_parameter: dict[str, str] = {}
    for parameter_name, raw_targets in parameter_gradient_aliases.items():
        targets = (raw_targets,) if isinstance(raw_targets, str) else tuple(raw_targets)
        for raw_target in targets:
            target_to_parameter[str(raw_target)] = str(parameter_name)
    covered_parameter_gradient_buffers = sorted(
        name for name in required_gradient_buffers if name in target_to_parameter
    )
    parameter_gradient_tree_names = sorted(
        {target_to_parameter[name] for name in covered_parameter_gradient_buffers}
    )
    bridge_only_gradient_buffers = sorted(
        set(required_gradient_buffers).difference(covered_parameter_gradient_buffers)
    )
    required_loss_cotangent_buffers = [
        name
        for name in _path_c_direct_chain_loss_cotangent_seed_buffers(chain)
        if name in required_specs
    ]
    required_runtime_bridge_gradients = sorted(
        name
        for name in bridge_only_gradient_buffers
        if name not in set(required_loss_cotangent_buffers)
    )
    resolved_buffers, _ = _path_c_direct_chain_resolved_logical_buffers(
        logical_buffers=logical_buffers,
        logical_owner=logical_owner,
    )
    runtime_bridge_gradient_outputs_ready = (
        not required_runtime_bridge_gradients
    ) or all(
        isinstance(resolved_buffers.get(name), mx.array)
        for name in required_runtime_bridge_gradients
    )
    model_gradient_tree_extraction = (
        path_c_model_gradient_tree_extraction_payload(
            model=model,
            logical_buffers=logical_buffers,
            logical_owner=logical_owner,
            parameter_gradient_names=parameter_gradient_tree_names,
        )
        if model is not None
        and (logical_buffers is not None or logical_owner is not None)
        else {
            "status": "blocked",
            "reason": "no direct-chain logical gradient buffers were provided",
            "gradient_tree_ready": False,
            "mapped_parameter_gradient_count": 0,
            "parameter_gradient_alias_count": len(parameter_gradient_aliases),
            "missing_parameter_gradient_names": [],
            "missing_logical_gradient_buffers": [],
            "non_array_logical_gradient_buffers": [],
        }
    )
    model_gradient_tree_ready = bool(
        model_gradient_tree_extraction.get("gradient_tree_ready")
    )
    blockers = [
        {
            "kind": "loss_cotangent_bridge_missing",
            "required_buffers": required_loss_cotangent_buffers,
            "reason": (
                "direct-chain backward needs a cotangent for the selected region "
                "output; m04 still gets that from eager MLX loss/backward instead "
                "of a Path C loss-to-region bridge"
            ),
        },
    ]
    if not model_gradient_tree_ready:
        blockers.append(
            {
                "kind": "model_gradient_tree_extraction_missing",
                "covered_parameter_gradient_buffer_count": len(
                    covered_parameter_gradient_buffers
                ),
                "parameter_gradient_tree_name_count": len(parameter_gradient_tree_names),
                "reason": (
                    "direct-chain kernels write logical gradient buffers, but no "
                    "production runtime yet returns an MLX model-gradient tree for "
                    "CompiledPretrainingStep/optimizer.update"
                ),
            }
        )
    if required_runtime_bridge_gradients and not runtime_bridge_gradient_outputs_ready:
        blockers.append(
            {
                "kind": "runtime_bridge_gradient_outputs_required",
                "required_buffers": required_runtime_bridge_gradients,
                "reason": (
                    "these non-parameter bridge gradients must be caller-owned "
                    "direct-chain inputs/outputs when the fused VJP runs"
                ),
            }
        )
    return {
        "status": "blocked",
        "contract": PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT,
        "loss_cotangent_bridge_ready": False,
        "model_gradient_tree_ready": model_gradient_tree_ready,
        "runtime_bridge_gradient_outputs_ready": runtime_bridge_gradient_outputs_ready,
        "delegates_to_eager_loss_and_grad": False,
        "required_gradient_buffer_count": len(required_gradient_buffers),
        "required_gradient_buffers": required_gradient_buffers,
        "required_loss_cotangent_buffers": required_loss_cotangent_buffers,
        "required_runtime_bridge_gradients": required_runtime_bridge_gradients,
        "covered_parameter_gradient_buffer_count": len(
            covered_parameter_gradient_buffers
        ),
        "covered_parameter_gradient_buffers": covered_parameter_gradient_buffers,
        "parameter_gradient_tree_name_count": len(parameter_gradient_tree_names),
        "parameter_gradient_tree_names": parameter_gradient_tree_names,
        "model_gradient_tree_extraction": model_gradient_tree_extraction,
        "bridge_only_gradient_buffer_count": len(bridge_only_gradient_buffers),
        "bridge_only_gradient_buffers": bridge_only_gradient_buffers,
        "blockers": blockers,
        "reason": (
            "standalone direct-chain dispatch is not a train-step value_and_grad; "
            "the missing production work is the loss cotangent bridge"
            if model_gradient_tree_ready and runtime_bridge_gradient_outputs_ready
            else "standalone direct-chain dispatch is not a train-step value_and_grad; "
            "the missing production work is the loss cotangent bridge and MLX "
            "model-gradient tree extraction"
        ),
    }


def path_c_direct_fusion_chain_runtime_binding_payload(
    *,
    chain: Any,
    logical_buffers: Mapping[str, Any] | None = None,
    logical_buffer_owner: str | None = None,
    logical_owner: Any | None = None,
    artifacts: Any | None = None,
) -> dict[str, Any]:
    """Return executable-runtime binding status for a direct-buffer chain."""

    try:
        resolved_buffers = logical_buffers
        resolved_owner = logical_buffer_owner
        hidden_packing_performed = False
        no_hidden_allocation_policy = True
        if logical_owner is not None:
            if logical_buffers is not None:
                raise ValueError(
                    "logical_owner and logical_buffers are mutually exclusive"
                )
            resolved_buffers = getattr(logical_owner, "buffers", None)
            resolved_owner = str(
                getattr(logical_owner, "owner_name", resolved_owner)
            )
            hidden_packing_performed = bool(
                getattr(logical_owner, "hidden_packing_performed", False)
            )
            no_hidden_allocation_policy = bool(
                getattr(logical_owner, "no_hidden_allocation_policy", True)
            )
        provided_names = set(str(name) for name in (resolved_buffers or {}))
        segment_payloads = []
        all_segments_ready = str(getattr(chain, "status", "")) == "ready"
        all_direct_logical = True
        all_bindings_ready = True
        all_artifacts_bound = True
        required_logical_buffer_names: set[str] = set()
        missing_logical_buffers: set[str] = set()
        shape_mismatch_buffers: set[str] = set()
        dtype_mismatch_buffers: set[str] = set()
        binding_errors: set[str] = set()
        missing_artifact_segments: list[int] = []
        for segment in getattr(chain, "segments", ()):
            target = getattr(segment, "schedule_target", None)
            if target is None or str(segment.status) != "ok":
                all_segments_ready = False
                all_direct_logical = False
                all_bindings_ready = False
                all_artifacts_bound = False
                missing_artifact_segments.append(int(segment.index))
                segment_payloads.append(
                    {
                        "index": int(segment.index),
                        "status": "blocked",
                        "reason": str(segment.reason),
                        "execution_phase": str(
                            getattr(segment, "execution_phase", "unknown")
                        ),
                        "runtime_ready": False,
                        "artifact_bound": False,
                        "logical_tensor_binding_ready": False,
                        "required_logical_buffers": [],
                        "missing_logical_buffers": [],
                        "provided_logical_buffers": [],
                    }
                )
                continue
            prim_func = target.schedule_template(segment.region)
            physical_abi_map = dict(
                getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
                or {}
            )
            physical_abi_shapes = dict(
                getattr(
                    prim_func,
                    "_cppmega_path_c_physical_buffer_abi_shapes",
                    {},
                )
                or {}
            )
            physical_abi_policy = str(
                getattr(prim_func, "_cppmega_path_c_physical_abi_policy", "unknown")
            )
            bridge = plan_physical_abi_runtime_bridge(
                physical_abi_map,
                physical_abi_shapes,
            )
            binding = validate_physical_abi_runtime_bindings(
                physical_abi_map,
                physical_abi_shapes,
                resolved_buffers,
            )
            scratch_specs = _path_c_internal_scratch_abi_specs(prim_func)
            scratch_binding = _path_c_validate_internal_scratch_abi_buffers(
                scratch_specs,
                resolved_buffers,
            )
            required_logical_buffers = list(bridge.get("required_bank_buffers", ()))
            required_internal_scratch_buffers = list(
                scratch_binding.get("required_internal_scratch_buffers", ())
            )
            required_logical_buffers.extend(required_internal_scratch_buffers)
            required_logical_buffer_names.update(
                str(name) for name in required_logical_buffers
            )
            segment_missing = list(binding.get("missing_bank_buffers", ()))
            segment_missing.extend(
                str(name)
                for name in scratch_binding.get("missing_internal_scratch_buffers", ())
            )
            ordered_kernel_buffers = list(binding.get("ordered_kernel_buffers", ()))
            ordered_kernel_buffers.extend(required_internal_scratch_buffers)
            provided_logical_buffers = [
                name for name in ordered_kernel_buffers if name in provided_names
            ]
            logical_binding_ready = (
                binding.get("status") == "ok" and not segment_missing
                and scratch_binding.get("status") == "ok"
            )
            direct_logical_supported = bool(
                bridge.get("logical_tensor_binding_supported")
            )
            artifact = _path_c_direct_chain_artifact_for_segment(
                artifacts,
                segment,
            )
            artifact_bound = callable(artifact)
            runtime_ready = bool(
                physical_abi_policy == "direct_buffers"
                and direct_logical_supported
                and logical_binding_ready
                and artifact_bound
            )
            if not direct_logical_supported or physical_abi_policy != "direct_buffers":
                all_direct_logical = False
            shape_mismatch_buffers.update(
                str(name) for name in binding.get("shape_mismatch_buffers", ())
            )
            shape_mismatch_buffers.update(
                str(name)
                for name in scratch_binding.get(
                    "shape_mismatch_internal_scratch_buffers",
                    (),
                )
            )
            dtype_mismatch_buffers.update(
                str(name) for name in binding.get("dtype_mismatch_buffers", ())
            )
            dtype_mismatch_buffers.update(
                str(name)
                for name in scratch_binding.get(
                    "dtype_mismatch_internal_scratch_buffers",
                    (),
                )
            )
            binding_errors.update(str(error) for error in binding.get("errors", ()))
            binding_errors.update(
                str(error) for error in scratch_binding.get("errors", ())
            )
            if not logical_binding_ready:
                all_bindings_ready = False
                missing_logical_buffers.update(str(name) for name in segment_missing)
            if not artifact_bound:
                all_artifacts_bound = False
                missing_artifact_segments.append(int(segment.index))
            segment_payloads.append(
                {
                    "index": int(segment.index),
                    "status": "ok" if runtime_ready else "not_bound",
                    "reason": (
                        "segment has callable artifact and caller-owned direct "
                        "logical buffers"
                        if runtime_ready
                        else "segment requires callable artifact and caller-owned "
                        "direct logical buffers"
                    ),
                    "runtime_ready": runtime_ready,
                    "artifact_bound": artifact_bound,
                    "logical_tensor_binding_supported": direct_logical_supported,
                    "logical_tensor_binding_ready": logical_binding_ready,
                    "bridge_status": bridge.get("status"),
                    "binding_status": binding.get("status"),
                    "internal_scratch_binding_status": scratch_binding.get("status"),
                    "physical_abi_policy": physical_abi_policy,
                    "execution_phase": str(
                        getattr(segment, "execution_phase", "unknown")
                    ),
                    "required_logical_buffers": required_logical_buffers,
                    "missing_logical_buffers": segment_missing,
                    "required_internal_scratch_buffers": (
                        required_internal_scratch_buffers
                    ),
                    "missing_internal_scratch_buffers": list(
                        scratch_binding.get("missing_internal_scratch_buffers", ())
                    ),
                    "unexpected_logical_buffers": list(
                        binding.get("unexpected_buffers", ())
                    ),
                    "shape_mismatch_buffers": list(
                        [
                            *binding.get("shape_mismatch_buffers", ()),
                            *scratch_binding.get(
                                "shape_mismatch_internal_scratch_buffers",
                                (),
                            ),
                        ]
                    ),
                    "dtype_mismatch_buffers": list(
                        [
                            *binding.get("dtype_mismatch_buffers", ()),
                            *scratch_binding.get(
                                "dtype_mismatch_internal_scratch_buffers",
                                (),
                            ),
                        ]
                    ),
                    "binding_errors": [
                        *binding.get("errors", ()),
                        *scratch_binding.get("errors", ()),
                    ],
                    "provided_logical_buffers": provided_logical_buffers,
                    "schedule_id": target.schedule_id,
                    "region_name": segment.region.name,
                }
            )
        runtime_uses_direct_chain = bool(
            all_segments_ready
            and all_direct_logical
            and all_bindings_ready
            and all_artifacts_bound
            and segment_payloads
        )
        status = (
            "ok"
            if runtime_uses_direct_chain
            else "direct_fusion_chain_plan_blocked"
            if not all_segments_ready or not all_direct_logical
            else FP8_PATH_C_DIRECT_CHAIN_LOGICAL_BUFFERS_MISSING_STATUS
            if not all_bindings_ready
            else FP8_PATH_C_DIRECT_CHAIN_ARTIFACTS_MISSING_STATUS
        )
        supplemental_train_step_buffers = {
            "final_norm_weight_grad",
            "lm_head_weight_grad",
        }
        unexpected_logical_buffers = provided_names.difference(
            required_logical_buffer_names
        ).difference(supplemental_train_step_buffers)
        return {
            "status": status,
            "chain_status": str(getattr(chain, "status", "unknown")),
            "runtime_uses_direct_fusion_chain": runtime_uses_direct_chain,
            "runtime_uses_fused_train_block": False,
            "logical_tensor_binding_ready": all_bindings_ready,
            "direct_logical_tensor_binding_supported": all_direct_logical,
            "direct_chain_artifacts_bound": all_artifacts_bound,
            "logical_buffer_owner": resolved_owner,
            "segment_count": len(segment_payloads),
            "missing_artifact_segments": missing_artifact_segments,
            "missing_logical_buffers": sorted(missing_logical_buffers),
            "missing_logical_buffer_count": len(missing_logical_buffers),
            "provided_logical_buffer_count": len(
                set(provided_names).intersection(
                    {
                        str(name)
                        for payload in segment_payloads
                        for name in payload.get("required_logical_buffers", ())
                    }
                )
            ),
            "unexpected_logical_buffers": sorted(
                unexpected_logical_buffers
            ),
            "unexpected_logical_buffer_count": len(unexpected_logical_buffers),
            "shape_mismatch_buffers": sorted(shape_mismatch_buffers),
            "shape_mismatch_count": len(shape_mismatch_buffers),
            "dtype_mismatch_buffers": sorted(dtype_mismatch_buffers),
            "dtype_mismatch_count": len(dtype_mismatch_buffers),
            "binding_errors": sorted(binding_errors),
            "hidden_packing_performed": hidden_packing_performed,
            "no_hidden_allocation_policy": no_hidden_allocation_policy,
            "segments": segment_payloads,
            "reason": (
                "all direct-chain segments have callable artifacts and caller-owned "
                "logical buffers in generated kernel argument order"
                if runtime_uses_direct_chain
                else "direct-chain runtime requires caller/model-owned logical "
                "buffers and callable segment artifacts; hidden packing or copying "
                "is forbidden"
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive receipt metadata
        return {
            "status": "direct_fusion_chain_runtime_binding_unavailable",
            "chain_status": "unknown",
            "runtime_uses_direct_fusion_chain": False,
            "runtime_uses_fused_train_block": False,
            "logical_tensor_binding_ready": False,
            "direct_logical_tensor_binding_supported": False,
            "direct_chain_artifacts_bound": False,
            "logical_buffer_owner": logical_buffer_owner
            if logical_owner is None
            else str(getattr(logical_owner, "owner_name", logical_buffer_owner)),
            "segment_count": 0,
            "missing_artifact_segments": [],
            "missing_logical_buffers": [],
            "missing_logical_buffer_count": 0,
            "provided_logical_buffer_count": 0,
            "unexpected_logical_buffers": [],
            "unexpected_logical_buffer_count": 0,
            "shape_mismatch_buffers": [],
            "shape_mismatch_count": 0,
            "dtype_mismatch_buffers": [],
            "dtype_mismatch_count": 0,
            "binding_errors": [],
            "hidden_packing_performed": False,
            "no_hidden_allocation_policy": True,
            "segments": [],
            "reason": str(exc),
        }


def _path_c_fusion_compile_receipt_path() -> Path:
    override = os.environ.get(PATH_C_FUSION_COMPILE_RECEIPT_ENV)
    if override:
        return Path(override).expanduser()
    return PATH_C_FUSION_COMPILE_RECEIPT_PATH


def _path_c_fusion_compile_receipt_payload(
    *,
    schedule_spec: Any,
    plan: Any,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the production fused-schedule compile receipt for this route."""

    receipt_path = (
        Path(receipt_path).expanduser()
        if receipt_path is not None
        else _path_c_fusion_compile_receipt_path()
    )
    base: dict[str, Any] = {
        "path": str(receipt_path),
        "status": "missing",
        "verified": False,
        "schedule_id": None,
        "schedule_name": None,
        "native_compile_ok": False,
        "cache_key_recompile_status": None,
        "reason": "production fused schedule compile receipt is missing",
    }
    if not receipt_path.exists():
        return base
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "unreadable",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    receipt_schedule = receipt.get("schedule_spec", {})
    receipt_plan = receipt.get("compile_plan", {})
    receipt_contract = receipt_plan.get("schedule_contract", {}) or {}
    receipt_cache_audit = receipt.get("cache_key_recompile_audit", {}) or {}
    runtime_execution_contract = receipt.get("runtime_execution_contract", {}) or {}
    reporting_contract = receipt.get("reporting_contract", {}) or {}
    runtime_smoke = receipt.get("runtime_smoke", {}) or {}
    schedule_id = str(receipt_schedule.get("schedule_id", ""))
    schedule_name = str(receipt_plan.get("schedule_name", ""))
    cache_key_recompile_status = receipt_cache_audit.get("status")
    runtime_execution_status = runtime_execution_contract.get("status")
    runtime_route_uses_fused_region = runtime_execution_contract.get(
        "runtime_route_uses_fused_region"
    )
    runtime_smoke_status = runtime_smoke.get("status")
    runtime_smoke_mode = runtime_smoke.get("mode")
    runtime_smoke_actually_executed = runtime_smoke.get("actually_executed")
    production_runtime_smoke_uses_fused_train_block = reporting_contract.get(
        "production_runtime_smoke_uses_fused_train_block"
    )

    expected_schedule_id = str(schedule_spec.schedule_id)
    expected_schedule_name = str(schedule_spec.schedule_name)
    checks = {
        "receipt_status_ok": receipt.get("status") == "ok",
        "native_compile_ok": receipt.get("native_compile_requested") is True
        and receipt.get("native_compile_ok") is True,
        "schedule_id_matches": schedule_id == expected_schedule_id,
        "schedule_name_matches": schedule_name == expected_schedule_name,
        "plan_region_matches": receipt_plan.get("region_name") == plan.region_name,
        "single_kernel_fused": receipt_plan.get("single_kernel_fused") is True,
        "compile_plan_ready": receipt_plan.get("schedule_status") == "ready",
        "schedule_contract_verified": receipt_contract.get("status") == "verified",
        "declared_schedule_id_matches": (
            receipt_contract.get("declared_schedule_id") == expected_schedule_id
        ),
        "real_abi_complete": (
            receipt_schedule.get("real_abi_contract_complete") is True
            and receipt_schedule.get("missing_real_abi_inputs") == []
            and receipt_contract.get("missing_real_abi_inputs") == []
        ),
        "real_abi_input_shapes_match": (
            tuple(receipt_schedule.get("required_real_abi_input_shapes", ()))
            == tuple(schedule_spec.required_real_abi_input_shapes)
        ),
        "production_fragments_complete": (
            receipt_schedule.get("production_fragments_complete") is True
        ),
        "cache_key_recompile_stable": cache_key_recompile_status == "key_stable",
        "runtime_execution_ready": (
            runtime_execution_status == "runtime_ready"
            and runtime_route_uses_fused_region is True
        ),
        "production_runtime_smoke_ok": (
            runtime_smoke_status == "ok"
            and runtime_smoke_mode == "production_1b"
            and runtime_smoke_actually_executed is True
        ),
        "production_smoke_uses_fused_train_block": (
            production_runtime_smoke_uses_fused_train_block is True
        ),
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    verified = not failed_checks
    return {
        **base,
        "status": "verified" if verified else "mismatch",
        "verified": verified,
        "schedule_id": schedule_id or None,
        "schedule_name": schedule_name or None,
        "native_compile_ok": bool(receipt.get("native_compile_ok")),
        "cache_key_recompile_status": cache_key_recompile_status,
        "runtime_execution_status": runtime_execution_status,
        "runtime_route_uses_fused_region": bool(runtime_route_uses_fused_region),
        "runtime_smoke_status": runtime_smoke_status,
        "runtime_smoke_mode": runtime_smoke_mode,
        "runtime_smoke_actually_executed": bool(runtime_smoke_actually_executed),
        "production_runtime_smoke_uses_fused_train_block": bool(
            production_runtime_smoke_uses_fused_train_block
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "reason": (
            "production fused schedule compile receipt matches the selected "
            "schedule, verified contract, native lowerer, production runtime "
            "smoke, and stable cache-key recompile audit"
            if verified
            else "production fused schedule compile receipt does not match this route"
        ),
    }


def _path_c_fusion_matrix_profile_receipt_path() -> Path:
    override = os.environ.get(PATH_C_FUSION_MATRIX_PROFILE_RECEIPT_ENV)
    if override:
        return Path(override).expanduser()
    return PATH_C_FUSION_MATRIX_PROFILE_RECEIPT_PATH


def _matrix_profile_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _matrix_profile_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                return float(text)
            except ValueError:
                continue
    return None


def _matrix_profile_bool_is(
    row: Mapping[str, Any],
    keys: Sequence[str],
    *,
    expected: bool,
) -> bool:
    for key in keys:
        if key in row:
            return row.get(key) is expected
    return False


def _matrix_profile_status_ok(value: Any) -> bool:
    if value is True:
        return True
    text = _matrix_profile_string(value)
    if text is None:
        return False
    return text.lower() in {"ok", "pass", "passed", "success", "verified"}


def _matrix_profile_trace_captured(row: Mapping[str, Any]) -> bool:
    if row.get("profiling_trace_captured") is True:
        return True
    for key in (
        "profiling_trace_path",
        "profile_trace_path",
        "trace_path",
        "xctrace_path",
    ):
        if _matrix_profile_string(row.get(key)):
            return True
    return False


def _matrix_profile_row_check_summary(
    matrix_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    summary = {"total_rows": len(matrix_rows)}
    for check_name in MATRIX_PROFILE_ROW_CHECKS:
        summary[check_name] = sum(
            1
            for row in matrix_rows
            if isinstance(row.get("checks"), Mapping)
            and bool(row["checks"].get(check_name))
        )
    return summary


def _matrix_profile_row_key(row: Mapping[str, Any]) -> str | None:
    dtype_route = _matrix_profile_string(row.get("dtype_route"))
    optimizer_name = _matrix_profile_string(row.get("optimizer"))
    if dtype_route is None or optimizer_name is None:
        return None
    return f"{dtype_route}:{optimizer_name}"


def _matrix_profile_failed_rows_by_check(
    matrix_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    failed_rows: dict[str, list[str]] = {}
    for row in matrix_rows:
        row_key = _matrix_profile_row_key(row)
        checks = row.get("checks")
        if row_key is None or not isinstance(checks, Mapping):
            continue
        for check_name in MATRIX_PROFILE_ROW_CHECKS:
            if not bool(checks.get(check_name)):
                failed_rows.setdefault(check_name, []).append(row_key)
    return failed_rows


def _matrix_profile_failed_rows_payload(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {}
    failed_rows: dict[str, list[str]] = {}
    for check_name, rows in value.items():
        if not isinstance(check_name, str):
            continue
        if not isinstance(rows, Sequence) or isinstance(
            rows,
            (str, bytes, bytearray),
        ):
            continue
        failed_rows[check_name] = [str(row) for row in rows]
    return failed_rows


def _matrix_profile_report_rows(
    matrix_report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    for key in ("matrix_rows", "results", "rows", "cells"):
        rows = matrix_report.get(key)
        if isinstance(rows, Sequence) and not isinstance(
            rows, (str, bytes, bytearray)
        ):
            mapped_rows = [row for row in rows if isinstance(row, Mapping)]
            if mapped_rows:
                return mapped_rows
    return []


def _matrix_profile_row_commit(
    *,
    row: Mapping[str, Any],
    report_software: Mapping[str, Any],
) -> str | None:
    row_software = row.get("software")
    if not isinstance(row_software, Mapping):
        row_software = {}
    for source in (row, row_software, report_software):
        for key in ("cppmega_sha", "git_commit"):
            value = _matrix_profile_string(source.get(key))
            if value:
                return value
    return None


def _matrix_profile_direct_receipt_row(
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    dtype_route = _matrix_profile_string(row.get("dtype_route"))
    optimizer_name = _matrix_profile_string(row.get("optimizer"))
    if dtype_route is None or optimizer_name is None:
        return None
    dtype_route = dtype_route.lower()
    optimizer_name = optimizer_name.lower()
    if dtype_route not in MATRIX_DTYPE_ROUTES:
        return None
    if optimizer_name not in MATRIX_OPTIMIZERS:
        return None

    path_b_peak_memory_gb = _matrix_profile_float(
        row,
        (
            "path_b_peak_memory_gb",
            "baseline_peak_memory_gb",
            "path_b_peak_gb",
        ),
    )
    path_c_peak_memory_gb = _matrix_profile_float(
        row,
        (
            "path_c_peak_memory_gb",
            "path_c_warm_peak_memory_gb",
            "path_c_peak_gb",
        ),
    )
    path_b_tok_sec = _matrix_profile_float(
        row,
        (
            "path_b_tok_sec",
            "baseline_tok_sec",
            "path_b_tokens_per_second",
            "baseline_tokens_per_second",
        ),
    )
    path_c_warm_tok_sec = _matrix_profile_float(
        row,
        (
            "path_c_warm_tok_sec",
            "path_c_tok_sec",
            "path_c_tokens_per_second",
            "path_c_warm_tokens_per_second",
        ),
    )
    path_b_status_ok = _matrix_profile_status_ok(
        row.get("path_b_status", row.get("baseline_status", row.get("status")))
    )
    path_c_warm_status_ok = _matrix_profile_status_ok(
        row.get("path_c_warm_status", row.get("path_c_status", row.get("status")))
    )
    path_b_reason = _matrix_profile_string(
        row.get("path_b_pass_fail_reason", row.get("pass_fail_reason", "ok"))
    )
    path_c_reason = _matrix_profile_string(
        row.get(
            "path_c_warm_pass_fail_reason",
            row.get("path_c_pass_fail_reason", "ok"),
        )
    )
    path_b_baseline_clean = path_b_status_ok and path_b_reason == "ok"
    path_c_default_gate_passed = (
        path_c_warm_status_ok
        and path_c_reason == "ok"
        and path_b_tok_sec is not None
        and path_c_warm_tok_sec is not None
        and path_c_warm_tok_sec >= path_b_tok_sec
    )
    path_c_peak_memory_non_regression = (
        path_b_peak_memory_gb is not None
        and path_c_peak_memory_gb is not None
        and path_c_peak_memory_gb <= path_b_peak_memory_gb
    )
    path_c_warm_cache_hit_observed = _matrix_profile_bool_is(
        row,
        ("path_c_warm_cache_hit", "warm_cache_hit"),
        expected=True,
    )
    path_c_cold_cache_miss_observed = _matrix_profile_bool_is(
        row,
        ("path_c_cold_cache_hit", "cold_cache_hit"),
        expected=False,
    )
    profiling_trace_captured = _matrix_profile_trace_captured(row)
    row_status_ok = _matrix_profile_status_ok(row.get("status"))
    checks = {
        "row_status_ok": row_status_ok,
        "path_b_baseline_clean": path_b_baseline_clean,
        "path_c_default_gate_passed": path_c_default_gate_passed,
        "path_c_peak_memory_non_regression": path_c_peak_memory_non_regression,
        "path_c_warm_cache_hit_observed": path_c_warm_cache_hit_observed,
        "path_c_cold_cache_miss_observed": path_c_cold_cache_miss_observed,
        "profiling_trace_captured": profiling_trace_captured,
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    row_observed_ok = (
        row_status_ok
        and path_b_baseline_clean
        and path_c_warm_status_ok
        and path_c_reason == "ok"
    )
    return {
        "dtype_route": dtype_route,
        "optimizer": optimizer_name,
        "status": "ok" if row_observed_ok else "mismatch",
        "path_b_tok_sec": path_b_tok_sec,
        "path_c_warm_tok_sec": path_c_warm_tok_sec,
        "path_b_peak_memory_gb": path_b_peak_memory_gb,
        "path_c_peak_memory_gb": path_c_peak_memory_gb,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _matrix_profile_path_matrix_optimizer_alias(
    dtype_route: str,
    optimizer_name: str,
) -> str:
    if dtype_route != "int8":
        return optimizer_name
    if optimizer_name in {"adamw", "adam8bit"}:
        return "adam8bit"
    if optimizer_name in {"lion", "lion8bit"}:
        return "lion8bit"
    if optimizer_name in {"muon", "muon_adamw"}:
        return "muon_int8"
    return optimizer_name


def _matrix_profile_path_matrix_dtype(dtype_route: str) -> str:
    if dtype_route in {"fp8_path_b", "fp8_path_c"}:
        return "fp8"
    return "bf16"


def _matrix_profile_path_matrix_receipt_row(
    *,
    dtype_route: str,
    optimizer_name: str,
    baseline_row: Mapping[str, Any],
    warm_row: Mapping[str, Any] | None,
    cold_row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if dtype_route == "fp8_path_b":
        warm_row = baseline_row
    if warm_row is None:
        return None
    normalized_row: dict[str, Any] = {
        "dtype_route": dtype_route,
        "optimizer": optimizer_name,
        "status": baseline_row.get("status"),
        "path_b_status": baseline_row.get("status"),
        "path_b_pass_fail_reason": baseline_row.get("pass_fail_reason", "ok"),
        "path_b_tok_sec": baseline_row.get("tok_sec"),
        "path_b_peak_memory_gb": baseline_row.get("peak_memory_gb"),
        "path_c_warm_status": warm_row.get("status"),
        "path_c_warm_pass_fail_reason": warm_row.get("pass_fail_reason", "ok"),
        "path_c_warm_tok_sec": warm_row.get("tok_sec"),
        "path_c_peak_memory_gb": warm_row.get("peak_memory_gb"),
        "path_c_warm_cache_hit": (
            True if dtype_route == "fp8_path_b" else warm_row.get("cache_hit")
        ),
        "path_c_cold_cache_hit": (
            False
            if dtype_route == "fp8_path_b" or cold_row is None
            else cold_row.get("cache_hit")
        ),
    }
    for key in (
        "profiling_trace_path",
        "profile_trace_path",
        "trace_path",
        "xctrace_path",
        "profiling_trace_captured",
    ):
        if key in warm_row:
            normalized_row[key] = warm_row[key]
        elif key in baseline_row:
            normalized_row[key] = baseline_row[key]
    return _matrix_profile_direct_receipt_row(normalized_row)


def _matrix_profile_path_matrix_receipt_rows(
    report_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    path_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in report_rows:
        dtype = _matrix_profile_string(row.get("dtype"))
        optimizer_name = _matrix_profile_string(row.get("optimizer"))
        path_name = _matrix_profile_string(row.get("path"))
        if dtype is None or optimizer_name is None or path_name is None:
            continue
        path_rows[(dtype.lower(), optimizer_name.lower(), path_name.lower())] = row

    matrix_rows: list[dict[str, Any]] = []
    for dtype_route in MATRIX_DTYPE_ROUTES:
        dtype = _matrix_profile_path_matrix_dtype(dtype_route)
        for optimizer_name in MATRIX_OPTIMIZERS:
            optimizer_alias = _matrix_profile_path_matrix_optimizer_alias(
                dtype_route,
                optimizer_name,
            )
            baseline_row = path_rows.get((dtype, optimizer_alias, "path_b"))
            if baseline_row is None:
                continue
            warm_row = path_rows.get((dtype, optimizer_alias, "path_c_warm"))
            cold_row = path_rows.get((dtype, optimizer_alias, "path_c_cold"))
            receipt_row = _matrix_profile_path_matrix_receipt_row(
                dtype_route=dtype_route,
                optimizer_name=optimizer_name,
                baseline_row=baseline_row,
                warm_row=warm_row,
                cold_row=cold_row,
            )
            if receipt_row is not None:
                matrix_rows.append(receipt_row)
    return matrix_rows


def path_c_fusion_matrix_profile_receipt_from_report(
    matrix_report: Mapping[str, Any],
    *,
    schedule_id: str,
    schedule_name: str,
    model_profile: str = REQUIRED_MODEL_PROFILE,
) -> dict[str, Any]:
    """Derive a production matrix/profile receipt from benchmark row evidence."""

    report_software = matrix_report.get("software")
    if not isinstance(report_software, Mapping):
        report_software = {}
    report_rows = _matrix_profile_report_rows(matrix_report)
    required_matrix_grid = {
        (str(dtype_route), str(optimizer_name))
        for dtype_route in MATRIX_DTYPE_ROUTES
        for optimizer_name in MATRIX_OPTIMIZERS
    }
    row_by_grid: dict[tuple[str, str], dict[str, Any]] = {}
    commits: set[str] = set()
    missing_commit_rows = 0
    for row in report_rows:
        commit = _matrix_profile_row_commit(
            row=row,
            report_software=report_software,
        )
        if commit is None:
            missing_commit_rows += 1
        else:
            commits.add(commit)
        receipt_row = _matrix_profile_direct_receipt_row(row)
        if receipt_row is None:
            continue
        row_by_grid[(receipt_row["dtype_route"], receipt_row["optimizer"])] = (
            receipt_row
        )
    if not row_by_grid:
        for receipt_row in _matrix_profile_path_matrix_receipt_rows(report_rows):
            row_by_grid[(receipt_row["dtype_route"], receipt_row["optimizer"])] = (
                receipt_row
            )

    matrix_rows = [
        row_by_grid[(dtype_route, optimizer_name)]
        for dtype_route in MATRIX_DTYPE_ROUTES
        for optimizer_name in MATRIX_OPTIMIZERS
        if (dtype_route, optimizer_name) in row_by_grid
    ]
    missing_matrix_grid = sorted(
        f"{dtype_route}:{optimizer_name}"
        for dtype_route, optimizer_name in required_matrix_grid.difference(
            row_by_grid
        )
    )
    full_1b_matrix_captured = not missing_matrix_grid and bool(matrix_rows)
    grid_rows_ok = (
        len(matrix_rows) == len(required_matrix_grid)
        and all(row.get("status") == "ok" for row in matrix_rows)
    )
    single_cppmega_commit = len(commits) == 1 and missing_commit_rows == 0
    checks = {
        "single_cppmega_commit": single_cppmega_commit,
        "full_1b_matrix_captured": full_1b_matrix_captured and grid_rows_ok,
        "profiling_traces_captured": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("profiling_trace_captured"))
            for row in matrix_rows
        ),
        "memory_non_regression_ok": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("path_c_peak_memory_non_regression"))
            for row in matrix_rows
        ),
        "cache_receipts_captured": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("path_c_warm_cache_hit_observed"))
            and bool(row.get("checks", {}).get("path_c_cold_cache_miss_observed"))
            for row in matrix_rows
        ),
        "path_b_baselines_clean": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("path_b_baseline_clean"))
            for row in matrix_rows
        ),
        "path_c_default_gate_rows_passed": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("path_c_default_gate_passed"))
            for row in matrix_rows
        ),
        "path_c_peak_memory_non_regression": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("path_c_peak_memory_non_regression"))
            for row in matrix_rows
        ),
        "path_c_warm_cache_hit_observed": grid_rows_ok
        and all(
            bool(row.get("checks", {}).get("path_c_warm_cache_hit_observed"))
            for row in matrix_rows
        ),
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    row_check_summary = _matrix_profile_row_check_summary(matrix_rows)
    failed_rows_by_check = _matrix_profile_failed_rows_by_check(matrix_rows)
    return {
        "kind": "cppmega_path_c_fusion_matrix_profile_receipt",
        "status": "ok" if not failed_checks else "mismatch",
        "model_profile": model_profile,
        "schedule_id": schedule_id,
        "schedule_name": schedule_name,
        "cppmega_sha": next(iter(commits)) if len(commits) == 1 else None,
        "cppmega_shas": sorted(commits),
        "missing_commit_rows": missing_commit_rows,
        "single_cppmega_commit": single_cppmega_commit,
        "full_1b_matrix_captured": checks["full_1b_matrix_captured"],
        "profiling_traces_captured": checks["profiling_traces_captured"],
        "memory_non_regression_ok": checks["memory_non_regression_ok"],
        "cache_receipts_captured": checks["cache_receipts_captured"],
        "path_b_baselines_clean": checks["path_b_baselines_clean"],
        "path_c_default_gate_rows_passed": checks[
            "path_c_default_gate_rows_passed"
        ],
        "path_c_peak_memory_non_regression": checks[
            "path_c_peak_memory_non_regression"
        ],
        "path_c_warm_cache_hit_observed": checks[
            "path_c_warm_cache_hit_observed"
        ],
        "row_check_summary": row_check_summary,
        "failed_rows_by_check": failed_rows_by_check,
        "matrix_rows": matrix_rows,
        "missing_matrix_rows": missing_matrix_grid,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _path_c_fusion_matrix_profile_payload(
    *,
    schedule_spec: Any,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate 1B matrix/profile/cache evidence for a production schedule."""

    receipt_path = (
        Path(receipt_path).expanduser()
        if receipt_path is not None
        else _path_c_fusion_matrix_profile_receipt_path()
    )
    expected_schedule_id = str(schedule_spec.schedule_id)
    expected_schedule_name = str(schedule_spec.schedule_name)
    base: dict[str, Any] = {
        "path": str(receipt_path),
        "status": "missing",
        "verified": False,
        "schedule_id": None,
        "schedule_name": None,
        "failed_checks": [],
        "failed_rows_by_check": {},
        "reason": "production fused schedule 1B matrix/profile receipt is missing",
    }
    if not receipt_path.exists():
        return base
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "unreadable",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    row_check_summary = receipt.get("row_check_summary")
    if not isinstance(row_check_summary, Mapping):
        row_check_summary = {}
    failed_rows_by_check = _matrix_profile_failed_rows_payload(
        receipt.get("failed_rows_by_check")
    )
    matrix_rows = receipt.get("matrix_rows", ())
    required_matrix_grid = {
        (str(dtype_route), str(optimizer_name))
        for dtype_route in MATRIX_DTYPE_ROUTES
        for optimizer_name in MATRIX_OPTIMIZERS
    }
    observed_matrix_grid: set[tuple[str, str]] = set()
    if isinstance(matrix_rows, list):
        for row in matrix_rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("status") != "ok":
                continue
            dtype_route = str(
                row.get("dtype_route", row.get("dtype", ""))
            ).strip()
            optimizer_name = str(row.get("optimizer", "")).strip()
            if dtype_route and optimizer_name:
                observed_matrix_grid.add((dtype_route, optimizer_name))
    missing_matrix_grid = sorted(
        f"{dtype_route}:{optimizer_name}"
        for dtype_route, optimizer_name in required_matrix_grid.difference(
            observed_matrix_grid
        )
    )
    checks = {
        "receipt_kind_ok": (
            receipt.get("kind")
            == "cppmega_path_c_fusion_matrix_profile_receipt"
        ),
        "receipt_status_ok": receipt.get("status") == "ok",
        "model_profile_matches": (
            receipt.get("model_profile") == REQUIRED_MODEL_PROFILE
        ),
        "schedule_id_matches": (
            str(receipt.get("schedule_id", "")) == expected_schedule_id
        ),
        "schedule_name_matches": (
            str(receipt.get("schedule_name", "")) == expected_schedule_name
        ),
        "single_cppmega_commit": (
            receipt.get("single_cppmega_commit") is True
            and bool(str(receipt.get("cppmega_sha", "")).strip())
        ),
        "full_1b_matrix_captured": (
            receipt.get("full_1b_matrix_captured") is True
        ),
        "profiling_traces_captured": (
            receipt.get("profiling_traces_captured") is True
        ),
        "memory_non_regression_ok": (
            receipt.get("memory_non_regression_ok") is True
        ),
        "cache_receipts_captured": (
            receipt.get("cache_receipts_captured") is True
        ),
        "path_b_baselines_clean": (
            receipt.get("path_b_baselines_clean") is True
        ),
        "path_c_default_gate_rows_passed": (
            receipt.get("path_c_default_gate_rows_passed") is True
        ),
        "path_c_peak_memory_non_regression": (
            receipt.get("path_c_peak_memory_non_regression") is True
        ),
        "path_c_warm_cache_hit_observed": (
            receipt.get("path_c_warm_cache_hit_observed") is True
        ),
        "matrix_rows_present": isinstance(matrix_rows, list) and bool(matrix_rows),
        "matrix_rows_cover_required_grid": not missing_matrix_grid,
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    verified = not failed_checks
    return {
        **base,
        "status": "verified" if verified else "mismatch",
        "verified": verified,
        "schedule_id": receipt.get("schedule_id"),
        "schedule_name": receipt.get("schedule_name"),
        "checks": checks,
        "failed_checks": failed_checks,
        "matrix_row_count": len(matrix_rows) if isinstance(matrix_rows, list) else 0,
        "required_matrix_row_count": len(required_matrix_grid),
        "covered_matrix_row_count": len(observed_matrix_grid),
        "missing_matrix_rows": missing_matrix_grid,
        "row_check_summary": dict(row_check_summary),
        "failed_rows_by_check": failed_rows_by_check,
        "reason": (
            "production fused schedule has matching full 1B matrix, profiling, "
            "memory non-regression, and cache receipt evidence"
            if verified
            else "production fused schedule matrix/profile receipt does not match this route"
        ),
    }


def path_c_fusion_payload(
    *,
    model: Any | None = None,
    compile_receipt_path: str | Path | None = None,
    matrix_profile_receipt_path: str | Path | None = None,
    bank_buffers: Mapping[str, Any] | None = None,
    bank_buffer_owner: str | None = None,
    bank_owner: Any | None = None,
    fused_artifact: Any | None = None,
    direct_chain_artifacts: Any | None = None,
    direct_chain_logical_buffers: Mapping[str, Any] | None = None,
    direct_chain_logical_buffer_owner: str | None = None,
    direct_chain_logical_owner: Any | None = None,
    direct_chain_training_runtime: Any | None = None,
    fused_train_block_training_runtime: Any | None = None,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Return receipt metadata for the high-level Path C fusion planner."""

    mode = selected_path_c_fusion_mode()
    route_context = _path_c_model_route_context(
        model,
        sequence_length=sequence_length,
    )
    profile_name = route_context.profile_name
    route_symbols = route_context.route_symbols
    model_regions = route_context.regions
    selected_region = _select_path_c_model_route_region(model_regions)
    if selected_region is None:
        raise RuntimeError(f"{profile_name} did not expose any Path C model region")
    model_route_candidates = _path_c_model_route_candidates_payload(
        profile_name=profile_name,
        route_symbols=route_symbols,
        regions=model_regions,
        selected_region=selected_region,
        region_source=route_context.region_source,
    )
    selected_model_region = model_route_candidates["selected_candidate"]
    scheduled = plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    if scheduled.schedule_target is None:
        raise RuntimeError(
            "selected Path C model region did not resolve to a schedule target"
        )
    region = scheduled.region
    plan = scheduled.plan
    schedule_spec = path_c_fusion_schedule_spec(
        region,
        contract=plan.schedule_contract,
        target=scheduled.schedule_target,
    )
    compile_receipt = _path_c_fusion_compile_receipt_payload(
        schedule_spec=schedule_spec,
        plan=plan,
        receipt_path=compile_receipt_path,
    )
    production_compile_verified = bool(compile_receipt.get("verified"))
    matrix_profile_receipt = _path_c_fusion_matrix_profile_payload(
        schedule_spec=schedule_spec,
        receipt_path=matrix_profile_receipt_path,
    )
    production_matrix_profile_verified = bool(
        matrix_profile_receipt.get("verified")
    )
    expected_bank_buffer_owner = (
        bank_buffer_owner
        if bank_buffer_owner is not None
        else f"{profile_name}.path_c_physical_abi_banks"
    )
    resolved_bank_owner = bank_owner
    if resolved_bank_owner is None and bank_buffers is None and model is not None:
        resolved_bank_owner = getattr(model, "path_c_physical_abi_bank_owner", None)
    resolved_fused_artifact = fused_artifact
    if resolved_fused_artifact is None and model is not None:
        resolved_fused_artifact = getattr(
            model,
            "path_c_fused_train_block_artifact",
            None,
        )
    resolved_fused_train_block_training_runtime = fused_train_block_training_runtime
    if resolved_fused_train_block_training_runtime is None and model is not None:
        resolved_fused_train_block_training_runtime = getattr(
            model,
            "path_c_fused_train_block_training_runtime",
            None,
        )
    resolved_direct_chain_training_runtime = direct_chain_training_runtime
    if resolved_direct_chain_training_runtime is None and model is not None:
        resolved_direct_chain_training_runtime = getattr(
            model,
            "path_c_direct_fusion_chain_training_runtime",
            None,
        )
    resolved_direct_chain_artifacts = direct_chain_artifacts
    if resolved_direct_chain_artifacts is None:
        resolved_direct_chain_artifacts = getattr(
            resolved_direct_chain_training_runtime,
            "artifacts",
            None,
        )
    if resolved_direct_chain_artifacts is None and model is not None:
        resolved_direct_chain_artifacts = getattr(
            model,
            "path_c_direct_fusion_chain_artifacts",
            None,
        )
    resolved_direct_chain_logical_buffers = direct_chain_logical_buffers
    if resolved_direct_chain_logical_buffers is None and model is not None:
        resolved_direct_chain_logical_buffers = getattr(
            model,
            "path_c_direct_fusion_chain_logical_buffers",
            None,
        )
    resolved_direct_chain_logical_buffer_owner = direct_chain_logical_buffer_owner
    if resolved_direct_chain_logical_buffer_owner is None and model is not None:
        resolved_direct_chain_logical_buffer_owner = getattr(
            model,
            "path_c_direct_fusion_chain_logical_buffer_owner_name",
            None,
        )
    resolved_direct_chain_logical_owner = direct_chain_logical_owner
    if (
        resolved_direct_chain_logical_owner is None
        and resolved_direct_chain_logical_buffers is None
        and model is not None
    ):
        resolved_direct_chain_logical_owner = (
            _path_c_direct_chain_logical_owner_for_model(model)
        )
    runtime_training_binding = path_c_fusion_runtime_training_binding_payload(
        region=region,
        schedule_target=scheduled.schedule_target,
        bank_buffers=bank_buffers,
        bank_buffer_owner=expected_bank_buffer_owner,
        bank_owner=resolved_bank_owner,
        fused_artifact=resolved_fused_artifact,
    )
    fused_train_block_training_contract = (
        path_c_fused_train_block_training_runtime_contract_payload(
            training_runtime=resolved_fused_train_block_training_runtime,
            runtime_binding=runtime_training_binding,
        )
    )
    fused_train_block_training_critical_path = bool(
        fused_train_block_training_contract.get("critical_path_ready")
    )
    try:
        if model is not None:
            direct_chain_region_prefix = _path_c_direct_chain_region_prefix(
                model,
                profile_name,
            )
            runtime_chain = getattr(resolved_direct_chain_training_runtime, "chain", None)
            direct_chains = ()
            direct_chain = runtime_chain
            if direct_chain is None:
                direct_chains = plan_path_c_direct_fusion_chains_for_model(
                    model,
                    region_prefix=direct_chain_region_prefix,
                    include_backward=True,
                    sequence_length=sequence_length,
                )
                direct_chain = _select_path_c_direct_chain_for_region(
                    direct_chains,
                    selected_region,
                )
            if direct_chain is None:
                raise RuntimeError(
                    f"{profile_name} did not produce any direct Path C chain"
                )
            direct_chain_construction = {
                "planner": (
                    "training_runtime.chain"
                    if runtime_chain is not None
                    else "plan_path_c_direct_fusion_chains_for_model"
                ),
                "region_prefix": direct_chain_region_prefix,
                "candidate_chain_count": len(direct_chains)
                if direct_chains
                else 1,
                "selected_source_region": getattr(
                    getattr(direct_chain, "source_region", None),
                    "name",
                    None,
                ),
            }
        else:
            direct_chain = plan_path_c_direct_fusion_chain_for_region(
                region,
                include_backward=True,
            )
            direct_chain_construction = {
                "planner": "plan_path_c_direct_fusion_chain_for_region",
                "region_prefix": None,
                "candidate_chain_count": 1,
                "selected_source_region": getattr(
                    getattr(direct_chain, "source_region", None),
                    "name",
                    None,
                ),
            }
        direct_chain_runtime_binding = (
            path_c_direct_fusion_chain_runtime_binding_payload(
                chain=direct_chain,
                logical_buffers=resolved_direct_chain_logical_buffers,
                logical_buffer_owner=resolved_direct_chain_logical_buffer_owner,
                logical_owner=resolved_direct_chain_logical_owner,
                artifacts=resolved_direct_chain_artifacts,
            )
        )
        direct_chain_training_contract = (
            path_c_direct_fusion_chain_training_runtime_contract_payload(
                training_runtime=resolved_direct_chain_training_runtime,
                runtime_binding=direct_chain_runtime_binding,
            )
        )
    except Exception as exc:
        direct_chain = SimpleNamespace(
            status="blocked",
            reason=f"direct-chain planner unavailable: {type(exc).__name__}: {exc}",
            max_kernel_buffers=0,
            segments=(),
            source_region=selected_region,
        )
        direct_chain_construction = {
            "planner": "direct_chain_planner_unavailable",
            "region_prefix": (
                _path_c_direct_chain_region_prefix(model, profile_name)
                if model is not None
                else None
            ),
            "candidate_chain_count": 0,
            "selected_source_region": getattr(selected_region, "name", None),
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        direct_chain_runtime_binding = {
            "status": "direct_chain_planner_unavailable",
            "reason": str(exc),
            "runtime_uses_direct_fusion_chain": False,
            "hidden_packing_performed": False,
            "no_hidden_allocation_policy": True,
        }
        direct_chain_training_contract = (
            path_c_direct_fusion_chain_training_runtime_contract_payload(
                training_runtime=None,
                runtime_binding=direct_chain_runtime_binding,
            )
        )
    direct_chain_training_critical_path = bool(
        direct_chain_training_contract.get("critical_path_ready")
    )
    try:
        direct_chain_model_binding_audit = (
            path_c_direct_fusion_chain_model_binding_audit(
                chain=direct_chain,
                model=model,
            )
        )
    except Exception as exc:
        direct_chain_model_binding_audit = {
            "status": "direct_chain_audit_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        direct_chain_pre_step_owner_plan_payload = (
            path_c_direct_chain_pre_step_owner_plan(
                chain=direct_chain,
                model=model,
                logical_buffers=direct_chain_logical_buffers,
                logical_owner=direct_chain_logical_owner,
            )
        )
    except Exception as exc:
        direct_chain_pre_step_owner_plan_payload = {
            "status": "direct_chain_pre_step_owner_plan_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        direct_chain_value_and_grad_bridge_plan_payload = (
            path_c_direct_fusion_chain_value_and_grad_bridge_plan(
                chain=direct_chain,
                model=model,
                logical_buffers=direct_chain_logical_buffers,
                logical_owner=direct_chain_logical_owner,
            )
        )
    except Exception as exc:
        direct_chain_value_and_grad_bridge_plan_payload = {
            "status": "direct_chain_value_and_grad_bridge_plan_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    runtime_fused_route_bound = bool(
        fused_train_block_training_contract.get("training_runtime_available")
    )
    real_schedule_unverified = (
        not production_compile_verified and not plan.single_kernel_fused
    )
    force_blocked = mode is PathCFusionMode.FORCE and real_schedule_unverified
    force_runtime_blocked = mode is PathCFusionMode.FORCE and not runtime_fused_route_bound
    scaffold_blocked = not schedule_spec.production_fragments_complete
    status = (
        "force_blocked_schedule_unverified"
        if force_blocked
        else "force_blocked_runtime_not_bound"
        if force_runtime_blocked
        else "plan_scaffold_not_default"
        if scaffold_blocked
        else "runtime_bound_not_default"
        if runtime_fused_route_bound
        else "off"
        if mode is PathCFusionMode.OFF
        else "plan_ready_not_default"
    )
    if scaffold_blocked:
        reason = (
            "Path C fusion selected a model-route-derived single-entry schedule "
            "scaffold, but the production body is not complete because at least "
            "one brick fragment is not production-inlined"
        )
    elif not production_compile_verified:
        reason = (
            "real fused train-block schedule is registered but is not yet trusted, "
            "compile-verified, benchmarked, and memory-profiled for default use"
        )
    elif not runtime_fused_route_bound:
        reason = (
            "real fused train-block schedule has a matching native compile "
            "receipt and stable cache-key audit, but the training runtime still "
            "has no bound generated fused artifact, physical ABI banks, and "
            "single fused train-block value_and_grad runtime"
        )
    else:
        reason = (
            "real fused train-block schedule is compile-verified and runtime-bound, "
            "but the full 1B benchmark, profiling, and memory receipts are still "
            "required before default use"
        )
    return {
        "mode": mode.value,
        "status": status,
        "reason": reason,
        "region_name": plan.region_name,
        "lowering_boundary": plan.lowering_boundary,
        "backend": plan.backend,
        "compiler": plan.compiler,
        "fusion_kind": plan.fusion_kind,
        "graph_construction": {
            "builder": "PathCFusionScheduleOptimizer",
            "input_model": route_context.input_model,
            "route_symbols": model_route_candidates["route_symbols"],
            "region_source": model_route_candidates["region_source"],
            "edge_policy": "infer_from_outputs_to_inputs",
            "dependency_ordering": "topological",
            "schedule_construction": "dynamic_brick_descriptors",
            "optimization_scope": "all_discovered_supported_path_c_brick_segments",
            "static_acceptance_fixture_used_for_selection": False,
            "selected_model_region": (
                selected_model_region["name"] if selected_model_region else None
            ),
            "selected_model_region_op_signature": (
                selected_model_region["op_signature"]
                if selected_model_region
                else []
            ),
            "selected_model_region_schedule_id": (
                selected_model_region["schedule_target"]["schedule_id"]
                if selected_model_region
                and selected_model_region["schedule_target"] is not None
                else None
            ),
            "preset_only": False,
        },
        "model_route_candidates": model_route_candidates,
        "schedule_name": plan.schedule_name,
        "schedule_status": plan.schedule_status,
        "schedule_registry": {
            "selector": "PathCFusionScheduleRegistry",
            "match_policy": "op_signature_or_descriptor_chain",
            "selected_schedule_id": schedule_spec.schedule_id,
            "selected_schedule_name": schedule_spec.schedule_name,
            "selected_from": "selected_model_region",
        },
        "production_schedule": {
            "source": "selected_model_region",
            "schedule_id": schedule_spec.schedule_id,
            "schedule_name": schedule_spec.schedule_name,
            "implementation_kind": schedule_spec.implementation_kind,
            "implementation_status": schedule_spec.implementation_status,
            "missing_reason": schedule_spec.missing_reason,
            "trusted_by_default": schedule_spec.trusted_by_default,
            "contract_name": schedule_spec.contract_name,
            "contract_key": schedule_spec.contract_key,
            "shape_env_key": schedule_spec.shape_env_key,
            "op_signature": list(schedule_spec.op_signature),
            "required_internal_buffers": list(
                schedule_spec.required_internal_buffers
            ),
            "required_external_buffers": list(
                schedule_spec.required_external_buffers
            ),
            "required_real_abi_inputs": list(
                schedule_spec.required_real_abi_inputs
            ),
            "required_real_abi_input_shapes": list(
                schedule_spec.required_real_abi_input_shapes
            ),
            "missing_real_abi_inputs": list(
                schedule_spec.missing_real_abi_inputs
            ),
            "real_abi_contract_complete": (
                schedule_spec.real_abi_contract_complete
            ),
            "required_codegen_steps": list(
                schedule_spec.required_codegen_steps
            ),
            "schedule_generator": schedule_spec.schedule_generator,
            "schedule_generator_status": (
                schedule_spec.schedule_generator_status
            ),
            "internal_buffer_policy": schedule_spec.internal_buffer_policy,
            "loop_policy": schedule_spec.loop_policy,
            "buffer_extent": schedule_spec.buffer_extent,
            "loop_extent": schedule_spec.loop_extent,
            "brick_ops": list(schedule_spec.brick_ops),
            "brick_schedule_families": list(
                schedule_spec.brick_schedule_families
            ),
            "brick_descriptor_statuses": list(
                schedule_spec.brick_descriptor_statuses
            ),
            "brick_production_fragment_statuses": list(
                schedule_spec.brick_production_fragment_statuses
            ),
            "brick_production_fragment_reasons": list(
                schedule_spec.brick_production_fragment_reasons
            ),
            "brick_production_fragment_blockers": list(
                schedule_spec.brick_production_fragment_blockers
            ),
            "production_fragments_complete": (
                schedule_spec.production_fragments_complete
            ),
        },
        "schedule_contract": (
            {
                "name": plan.schedule_contract.name,
                "key": plan.schedule_contract.key,
                "status": plan.schedule_contract.status,
                "reason": plan.schedule_contract.reason,
                "shape_env_key": plan.schedule_contract.shape_env_key,
                "declared_key": plan.schedule_contract.declared_key,
                "declared_implementation_kind": (
                    plan.schedule_contract.declared_implementation_kind
                ),
                "declared_schedule_id": plan.schedule_contract.declared_schedule_id,
                "declared_required_real_abi_inputs": list(
                    plan.schedule_contract.declared_required_real_abi_inputs
                ),
                "missing_real_abi_inputs": list(
                    plan.schedule_contract.missing_real_abi_inputs
                ),
                "op_signature": list(plan.schedule_contract.op_signature),
                "required_internal_buffers": list(
                    plan.schedule_contract.required_internal_buffers
                ),
                "required_external_buffers": list(
                    plan.schedule_contract.required_external_buffers
                ),
            }
            if plan.schedule_contract is not None
            else None
        ),
        "runtime_training_binding": runtime_training_binding,
        "fused_train_block_training_critical_path": bool(
            fused_train_block_training_critical_path
        ),
        "fused_train_block_training_runtime_available": bool(
            fused_train_block_training_contract.get("training_runtime_available")
        ),
        "fused_train_block_training_runtime_contract": (
            fused_train_block_training_contract
        ),
        "production_compile_receipt": compile_receipt,
        "production_matrix_profile_receipt": matrix_profile_receipt,
        "direct_chained_fusion": {
            **_path_c_direct_chain_plan_payload(direct_chain),
            "construction": direct_chain_construction,
            "standalone_dispatch_available": bool(
                direct_chain_runtime_binding.get("runtime_uses_direct_fusion_chain")
            ),
            "training_critical_path": bool(direct_chain_training_critical_path),
            "training_runtime_available": bool(
                direct_chain_training_contract.get("training_runtime_available")
            ),
            "training_runtime_contract": direct_chain_training_contract,
            "model_binding_audit": direct_chain_model_binding_audit,
            "pre_step_owner_plan": direct_chain_pre_step_owner_plan_payload,
            "value_and_grad_bridge_plan": direct_chain_value_and_grad_bridge_plan_payload,
            "runtime_binding": direct_chain_runtime_binding,
        },
        "schedule_blockers": [
            {
                "kind": (
                    "production_schedule_scaffold_not_default"
                    if schedule_spec.implementation_kind == "scaffold"
                    else "selected_model_schedule_not_default"
                ),
                "schedule_id": schedule_spec.schedule_id,
                "schedule_name": schedule_spec.schedule_name,
                "reason": (
                    "the selected model-route schedule is only a scaffold because "
                    "at least one brick fragment is not production-inlined"
                    if schedule_spec.implementation_kind == "scaffold"
                    else (
                        "the descriptor-generated schedule is selected from the "
                        "model route graph, but it is not a trusted default until "
                        "compile, benchmark, profiling, and memory receipts pass"
                    )
                ),
                "schedule_generator": PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR,
            },
            *(
                [
                    {
                        "kind": "production_schedule_uses_descriptor_loop_fragments",
                        "schedule_id": schedule_spec.schedule_id,
                        "schedule_name": schedule_spec.schedule_name,
                        "schedule_generator": schedule_spec.schedule_generator,
                        "schedule_generator_status": (
                            schedule_spec.schedule_generator_status
                        ),
                        "brick_schedule_families": list(
                            schedule_spec.brick_schedule_families
                        ),
                        "brick_production_fragment_statuses": list(
                            schedule_spec.brick_production_fragment_statuses
                        ),
                        "brick_production_fragment_reasons": list(
                            schedule_spec.brick_production_fragment_reasons
                        ),
                        "brick_production_fragment_blockers": list(
                            schedule_spec.brick_production_fragment_blockers
                        ),
                        "production_fragments_complete": (
                            schedule_spec.production_fragments_complete
                        ),
                        "reason": (
                            "the selected schedule is dynamically assembled from "
                            "Path C brick descriptors, but at least one brick "
                            "fragment is not production-inlined"
                        ),
                    }
                ]
                if not schedule_spec.production_fragments_complete
                else []
            ),
            *(
                [
                    {
                        "kind": "production_schedule_not_compile_verified",
                        "schedule_id": schedule_spec.schedule_id,
                        "schedule_name": schedule_spec.schedule_name,
                        "schedule_contract_status": plan.schedule_contract.status
                        if plan.schedule_contract is not None
                        else "missing",
                        "compile_receipt_status": compile_receipt.get("status"),
                        "compile_receipt_path": compile_receipt.get("path"),
                        "failed_checks": list(
                            compile_receipt.get("failed_checks", ())
                        ),
                        "reason": (
                            "the planner has not produced a matching lowered "
                            "production compile receipt with native_compile_ok, "
                            "schedule_contract.status=verified, production "
                            "runtime smoke, and a stable cache-key recompile audit"
                        ),
                    }
                ]
                if not production_compile_verified
                else []
            ),
            *(
                [
                    {
                        "kind": "fused_train_block_runtime_not_bound",
                        "schedule_id": schedule_spec.schedule_id,
                        "schedule_name": schedule_spec.schedule_name,
                        "runtime_binding_status": (
                            runtime_training_binding.get("status")
                        ),
                        "required_bank_buffers": list(
                            runtime_training_binding.get(
                                "required_bank_buffers",
                                (),
                            )
                        ),
                        "missing_bank_buffers": list(
                            runtime_training_binding.get(
                                "missing_bank_buffers",
                                (),
                            )
                        ),
                        "training_runtime_contract_status": (
                            fused_train_block_training_contract.get("status")
                        ),
                        "training_runtime_available": bool(
                            fused_train_block_training_contract.get(
                                "training_runtime_available"
                            )
                        ),
                        "reason": (
                            fused_train_block_training_contract.get("reason")
                            if runtime_training_binding.get("status") == "ok"
                            else runtime_training_binding.get("reason")
                        ),
                    }
                ]
                if not runtime_fused_route_bound
                else []
            ),
            *(
                [
                    {
                        "kind": "production_1b_matrix_profile_missing",
                        "schedule_id": schedule_spec.schedule_id,
                        "schedule_name": schedule_spec.schedule_name,
                        "matrix_profile_receipt_status": (
                            matrix_profile_receipt.get("status")
                        ),
                        "matrix_profile_receipt_path": (
                            matrix_profile_receipt.get("path")
                        ),
                        "failed_checks": list(
                            matrix_profile_receipt.get("failed_checks", ())
                        ),
                        "row_check_summary": dict(
                            matrix_profile_receipt.get("row_check_summary", {})
                        ),
                        "failed_rows_by_check": dict(
                            matrix_profile_receipt.get("failed_rows_by_check", {})
                        ),
                        "reason": (
                            "the full 1B Path B/Path C matrix, profiling traces, "
                            "memory non-regression evidence, and cache receipts "
                            "have not been captured for this production schedule"
                        ),
                    }
                ]
                if not production_matrix_profile_verified
                else []
            ),
            *(
                [
                    {
                        "kind": "missing_real_abi_inputs",
                        "schedule_id": schedule_spec.schedule_id,
                        "schedule_name": schedule_spec.schedule_name,
                        "missing_inputs": list(
                            schedule_spec.missing_real_abi_inputs
                        ),
                        "reason": (
                            "the model-semantic fusion contract still exposes "
                            "symbolic buffers instead of the real Mamba3, M2RNN, "
                            "attention projection, and Sparse-MLA apply inputs "
                            "needed by a production train-block schedule"
                        ),
                    }
                ]
                if schedule_spec.missing_real_abi_inputs
                else []
            ),
            *[
                {
                    "kind": blocker.kind,
                    "producer": blocker.producer,
                    "consumer": blocker.consumer,
                    "required_node": blocker.required_node,
                    "reason": blocker.reason,
                }
                for blocker in plan.semantic_blockers
            ],
        ],
        "single_kernel_fused": plan.single_kernel_fused,
        "fullgraph_required": True,
        "graph_break_policy": "fail_closed",
        "autograd_plan": {
            "mode": plan.autograd_mode,
            "status": plan.autograd_status,
            "backward_nodes": list(plan.autograd_backward_nodes),
            "backward_edges": [
                list(edge) for edge in plan.autograd_backward_edges
            ],
            "missing_backward_nodes": list(plan.autograd_missing_backward_nodes),
        },
        "semantic_blockers": [
            {
                "kind": blocker.kind,
                "producer": blocker.producer,
                "consumer": blocker.consumer,
                "required_node": blocker.required_node,
                "reason": blocker.reason,
            }
            for blocker in plan.semantic_blockers
        ],
        "default_allowed": False,
        "node_names": list(region.node_names),
        "fusion_groups": [
            {
                "node_names": list(group.node_names),
                "schedule_template": group.schedule_template,
            }
            for group in plan.fusion_groups
        ],
        "backward_graph": plan.backward_graph,
        "requires_msl_post_fusion": plan.requires_msl_post_fusion,
        "large_tensor_staging_allowed": plan.large_tensor_staging_allowed,
        "cache_key_parts": list(plan.cache_key_parts),
        "z3_sync": {
            "enabled": plan.z3_sync.enabled,
            "objective": plan.z3_sync.objective,
            "candidates": list(plan.z3_sync.candidates),
            "proof_required": plan.z3_sync.proof_required,
        },
        "acceptance_gate": {
            "ignores_bad_path_b": True,
            "requires_clean_path_b_baseline": True,
            "requires_ready_fusion_plan": True,
            "requires_compile_verified_single_kernel": True,
            "current_compile_receipt_verified": production_compile_verified,
            "current_matrix_profile_verified": (
                production_matrix_profile_verified
            ),
            "requires_verified_schedule_contract": True,
            "requires_complete_real_abi_contract": True,
            "current_plan_default_eligible": False,
            "requires_real_c_over_b_win": True,
            "requires_peak_memory_non_regression": True,
        },
        "cache_audit_required": True,
    }


def path_c_model_route_candidates_payload() -> dict[str, Any]:
    """Return Path C fusion candidate regions derived from local_gb10_quarter."""

    profile, route_symbols, regions = _local_gb10_path_c_model_regions()
    return _path_c_model_route_candidates_payload(
        profile_name=profile.name,
        route_symbols=route_symbols,
        regions=regions,
        selected_region=_select_path_c_model_route_region(regions),
        region_source="build_path_c_model_regions_from_model",
    )


def _path_c_model_route_context(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> SimpleNamespace:
    if model is None:
        profile, route_symbols, regions = _local_gb10_path_c_model_regions()
        return SimpleNamespace(
            profile_name=profile.name,
            input_model=f"{profile.name}_profile_path_c_bricks",
            route_symbols=route_symbols,
            regions=regions,
            region_source="build_path_c_model_regions_from_model",
        )

    profile_name = str(
        getattr(model, "path_c_profile_name", None)
        or getattr(type(model), "__name__", "model")
    )
    input_model = str(
        getattr(model, "path_c_input_model_name", None)
        or f"{profile_name}.path_c_bricks"
    )
    route_symbols = tuple(str(symbol) for symbol in getattr(model, "route_symbols", ()))
    region_builder = getattr(model, "path_c_fusion_regions", None)
    if callable(region_builder):
        regions = tuple(
            region_builder(
                include_backward=False,
                min_route_bricks=2,
                sequence_length=sequence_length,
            )
        )
        region_source = f"{profile_name}.path_c_fusion_regions"
    else:
        regions = build_path_c_model_regions_from_model(
            model,
            region_prefix=f"{profile_name}_path_c",
            sequence_length=sequence_length,
        )
        region_source = "build_path_c_model_regions_from_model"

    if not route_symbols:
        route_symbols = _route_symbols_from_regions(regions)
    return SimpleNamespace(
        profile_name=profile_name,
        input_model=input_model,
        route_symbols=route_symbols,
        regions=regions,
        region_source=region_source,
    )


def _route_symbols_from_regions(regions: tuple[Any, ...]) -> tuple[str, ...]:
    for region in regions:
        metadata = getattr(region, "metadata", {})
        bricks = metadata.get("path_c_bricks", ()) if isinstance(metadata, dict) else ()
        symbols = tuple(
            str(brick.get("route_symbol"))
            for brick in bricks
            if isinstance(brick, dict) and brick.get("route_symbol")
        )
        if symbols:
            return symbols
    return ()


def _local_gb10_path_c_model_regions() -> tuple[Any, tuple[str, ...], tuple[Any, ...]]:
    profile = local_gb10_quarter_profile()
    route_symbols = tuple(profile.expanded_pattern.symbols)
    model_descriptor = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    regions = build_path_c_model_regions_from_model(
        model_descriptor,
        region_prefix=f"{profile.name}_path_c",
    )
    return profile, route_symbols, regions


def _path_c_model_route_candidates_payload(
    *,
    profile_name: str,
    route_symbols: tuple[str, ...],
    regions: tuple[Any, ...],
    selected_region: Any | None,
    region_source: str,
) -> dict[str, Any]:
    candidate_regions = [
        _path_c_model_route_candidate_payload(region)
        for region in regions
    ]
    selected_name = getattr(selected_region, "name", None)
    selected_candidate = next(
        (
            candidate
            for candidate in candidate_regions
            if candidate["name"] == selected_name
        ),
        None,
    )
    return {
        "profile": profile_name,
        "route_symbols": list(route_symbols),
        "region_source": region_source,
        "selection_policy": "largest_supported_contiguous_route_segment",
        "selected_candidate": selected_candidate,
        "candidate_regions": candidate_regions,
    }


def _select_path_c_model_route_region(
    regions: tuple[Any, ...],
) -> Any | None:
    if not regions:
        return None
    return max(
        regions,
        key=lambda region: (
            len(region.nodes),
            len(region.edges),
            region.name,
        ),
    )


def _path_c_model_route_candidate_payload(region: Any) -> dict[str, Any]:
    planned = plan_path_c_fusion_schedule_for_region(
        region,
        include_backward=True,
    )
    target = planned.schedule_target
    contract = planned.plan.schedule_contract
    metadata = getattr(region, "metadata", {})
    bricks = (
        metadata.get("path_c_bricks", ())
        if isinstance(metadata, dict)
        else ()
    )
    return {
        "name": region.name,
        "brick_names": [brick.get("name") for brick in bricks],
        "brick_kinds": [brick.get("kind") for brick in bricks],
        "brick_route_symbols": [brick.get("route_symbol") for brick in bricks],
        "node_names": list(region.node_names),
        "op_signature": [node.op_name for node in region.nodes],
        "edge_count": len(region.edges),
        "z3_sync": {
            "enabled": region.z3_sync.enabled,
            "objective": region.z3_sync.objective,
            "proof_required": region.z3_sync.proof_required,
        },
        "schedule_target": None
        if target is None
        else {
            "schedule_id": target.schedule_id,
            "schedule_name": target.schedule_name,
            "schedule_status": target.schedule_status,
            "implementation_kind": target.implementation_kind,
            "schedule_generator": target.schedule_generator,
            "required_real_abi_inputs": list(target.required_real_abi_inputs),
            "brick_ops": [descriptor.op_name for descriptor in target.brick_descriptors],
        },
        "plan": {
            "schedule_name": planned.plan.schedule_name,
            "schedule_status": planned.plan.schedule_status,
            "schedule_contract_status": contract.status if contract is not None else None,
            "single_kernel_fused": planned.plan.single_kernel_fused,
            "backward_graph": planned.plan.backward_graph,
            "autograd_status": planned.plan.autograd_status,
        },
    }


def matrix_command_argv(
    *,
    dtype_arg: str,
    cli_optimizer: str,
    output_path: str,
    steps: int,
    dry_run: bool = False,
    require_loss_decrease: bool = False,
) -> list[str]:
    command_argv = [
        ".venv/bin/python",
        "scripts/m04_train_step.py",
        "--model-profile",
        REQUIRED_MODEL_PROFILE,
        "--data-path",
        target_dataset_path(),
        "--data-format",
        "parquet",
        "--token-key",
        "token_ids",
        "--steps",
        str(steps),
        "--batch-size",
        str(MATRIX_BATCH_SIZE),
        "--seq-len",
        str(MATRIX_SEQ_LEN),
        "--dtype",
        dtype_arg,
        "--optimizer",
        cli_optimizer,
        "--optimizer-quant-scheme",
        MATRIX_QUANT_SCHEME,
        "--lr",
        MATRIX_LR,
        "--grad-checkpoint",
        "--output",
        output_path,
        "--json",
    ]
    if require_loss_decrease:
        command_argv.insert(-3, "--require-loss-decrease")
    if dry_run:
        command_argv.insert(-3, "--dry-run-json")
    return command_argv


def shell_command(command_argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command_argv)


def matrix_case_payload(
    dtype_route: str,
    optimizer_name: str,
    *,
    steps: int = MATRIX_STEPS,
    require_loss_decrease: bool = False,
) -> dict[str, Any]:
    dtype_route = dtype_route.strip().lower()
    optimizer_name = optimizer_name.strip().lower()
    dtype_arg = "bfloat16"
    cli_optimizer = optimizer_name
    unsupported_reason = None

    if dtype_route == "fp8_path_b":
        dtype_arg = FP8_PATH_B_DTYPE
    elif dtype_route == "fp8_path_c":
        dtype_arg = FP8_PATH_C_DTYPE
    elif dtype_route in {"int8", "int8_state"}:
        dtype_route = "int8"
        if optimizer_name in {"muon", "muon_adamw"}:
            cli_optimizer = "int8"
        elif optimizer_name in {"adamw", "adam8bit"}:
            cli_optimizer = "adam8bit"
        elif optimizer_name in {"lion", "lion8bit"}:
            cli_optimizer = "lion8bit"
        else:
            unsupported_reason = (
                "int8 is an optimizer-state precision route; use "
                "MuonAdamWInt8, Adam8bit, or Lion8bit rather than fp32-state "
                f"{optimizer_name}."
            )
    elif dtype_route != "bf16":
        unsupported_reason = f"unknown dtype route {dtype_route!r}"

    case_id = f"{dtype_route}_{optimizer_name}_{steps}step"
    output_path = f"bench/baselines/m04_optimizer_matrix/{case_id}.json"
    supported = unsupported_reason is None
    command_argv: list[str] = []
    dry_run_command_argv: list[str] = []
    smoke_command_argv: list[str] = []
    command = None
    dry_run_command = None
    smoke_command = None
    if supported:
        dry_run_output_path = (
            f"bench/baselines/m04_optimizer_matrix/{case_id}_dry_run.json"
        )
        smoke_output_path = (
            f"bench/baselines/m04_optimizer_matrix/{case_id}_smoke1.json"
        )
        command_argv = matrix_command_argv(
            dtype_arg=dtype_arg,
            cli_optimizer=cli_optimizer,
            output_path=output_path,
            steps=steps,
            require_loss_decrease=require_loss_decrease,
        )
        dry_run_command_argv = matrix_command_argv(
            dtype_arg=dtype_arg,
            cli_optimizer=cli_optimizer,
            output_path=dry_run_output_path,
            steps=steps,
            dry_run=True,
            require_loss_decrease=require_loss_decrease,
        )
        smoke_command_argv = matrix_command_argv(
            dtype_arg=dtype_arg,
            cli_optimizer=cli_optimizer,
            output_path=smoke_output_path,
            steps=MATRIX_SMOKE_STEPS,
        )
        command = shell_command(command_argv)
        dry_run_command = shell_command(dry_run_command_argv)
        smoke_command = shell_command(smoke_command_argv)

    payload: dict[str, Any] = {
        "case_id": case_id,
        "dtype_route": dtype_route,
        "dtype_arg": dtype_arg,
        "optimizer": optimizer_name,
        "cli_optimizer": cli_optimizer if supported else None,
        "optimizer_quant_scheme": MATRIX_QUANT_SCHEME if supported else None,
        "steps": steps,
        "batch_size": MATRIX_BATCH_SIZE,
        "seq_len": MATRIX_SEQ_LEN,
        "learning_rate": MATRIX_LR,
        "require_loss_decrease": require_loss_decrease,
        "supported": supported,
        "unsupported_reason": unsupported_reason,
        "output": output_path if supported else None,
        "command_argv": command_argv,
        "command": command,
        "real_step_command": command,
        "dry_run_command_argv": dry_run_command_argv,
        "dry_run_command": dry_run_command,
        "smoke_command_argv": smoke_command_argv,
        "smoke_command": smoke_command,
    }
    payload[f"real_{steps}step_command"] = command
    if steps == MATRIX_STEPS:
        payload["real_20step_command"] = command
    if steps == MATRIX_ACCEPTANCE_STEPS:
        payload["real_100step_command"] = command
    return payload


def matrix_baseline_comparison_payload() -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    for receipt in MATRIX_BASELINE_RECEIPTS:
        tokens_per_second = float(receipt["tokens_per_second"])
        references.append(
            {
                **receipt,
                "baseline_tokens_per_second": MATRIX_BASELINE_TOKENS_PER_SECOND,
                "delta_tokens_per_second": (
                    tokens_per_second - MATRIX_BASELINE_TOKENS_PER_SECOND
                ),
                "ratio_to_900_tok_s": (
                    tokens_per_second / MATRIX_BASELINE_TOKENS_PER_SECOND
                ),
                "meets_900_tok_s_baseline": (
                    tokens_per_second >= MATRIX_BASELINE_TOKENS_PER_SECOND
                ),
            }
        )
    return {
        "baseline_tokens_per_second": MATRIX_BASELINE_TOKENS_PER_SECOND,
        "baseline_kind": "existing_real_parquet_bs1_seq4096_20step_receipts",
        "baseline_scope": "local_m4_only_not_gb10_parity",
        "reference_receipts": references,
        "readiness_rule": (
            "new 20-step rows should be compared against the checked-in "
            "900 tok/s-class receipts using identical real parquet, bs=1, "
            "seq=4096, local_gb10_quarter, grad-checkpoint workload shape"
        ),
    }


def m04_20step_matrix_payload() -> dict[str, Any]:
    cases = [
        matrix_case_payload(dtype_route, optimizer_name)
        for dtype_route in MATRIX_DTYPE_ROUTES
        for optimizer_name in MATRIX_OPTIMIZERS
    ]
    acceptance_cases = [
        matrix_case_payload(
            dtype_route,
            optimizer_name,
            steps=MATRIX_ACCEPTANCE_STEPS,
            require_loss_decrease=True,
        )
        for dtype_route in MATRIX_DTYPE_ROUTES
        for optimizer_name in MATRIX_OPTIMIZERS
    ]
    return {
        "name": "m04_local_gb10_20step_dtype_optimizer_matrix",
        "status": "commands_prepared_not_executed_by_this_receipt",
        "profile": REQUIRED_MODEL_PROFILE,
        "dataset": target_dataset_path(),
        "steps": MATRIX_STEPS,
        "acceptance_steps": MATRIX_ACCEPTANCE_STEPS,
        "smoke_steps": MATRIX_SMOKE_STEPS,
        "batch_size": MATRIX_BATCH_SIZE,
        "seq_len": MATRIX_SEQ_LEN,
        "learning_rate": MATRIX_LR,
        "optimizer_quant_scheme": MATRIX_QUANT_SCHEME,
        "baseline_comparison": matrix_baseline_comparison_payload(),
        "command_sets": ["dry_run", "smoke_1step", "real_20step", "real_100step"],
        "dtype_routes": list(MATRIX_DTYPE_ROUTES),
        "optimizers": list(MATRIX_OPTIMIZERS),
        "receipt_directory": "bench/baselines/m04_optimizer_matrix",
        "cases": cases,
        "supported_case_ids": [
            str(case["case_id"]) for case in cases if case["supported"] is True
        ],
        "unsupported_case_ids": [
            str(case["case_id"]) for case in cases if case["supported"] is False
        ],
        "real_20step_commands": [
            str(case["real_20step_command"])
            for case in cases
            if case["supported"] is True
        ],
        "real_100step_commands": [
            str(case["real_100step_command"])
            for case in acceptance_cases
            if case["supported"] is True
        ],
        "dry_run_commands": [
            str(case["dry_run_command"])
            for case in cases
            if case["supported"] is True
        ],
        "smoke_commands": [
            str(case["smoke_command"]) for case in cases if case["supported"] is True
        ],
        "notes": [
            "bf16 and fp8_path_c commands run model weights/activations through "
            "the requested dtype route; fp8_path_c uses a bf16 carrier plus "
            "Path C policy overrides for prepared-buffer ops.",
            "int8 is optimizer-state precision, so the model dtype remains "
            "bfloat16. Logical adamw/lion int8 rows map to Adam8bit/Lion8bit, "
            "and muon/muon_adamw int8 rows map to MuonAdamWInt8.",
            "These 20-step receipts are regression evidence only; the M0.4 "
            "acceptance gate still requires the 100-step bf16 AdamW "
            "grad-checkpoint target-parquet receipt.",
            "100-step commands add --require-loss-decrease so red rows fail "
            "closed instead of only writing diagnostic JSON.",
        ],
    }


def _path_c_policy_ops(
    args: argparse.Namespace | TrainHybridTinyConfig,
) -> list[str]:
    if not path_c_training_route_requested(args):
        return []
    return sorted(
        key.split("__", 1)[-1].lower()
        for key, value in FP8_PATH_C_KERNEL_POLICY_ENV.items()
        if value == "path_c"
    )


def _first_blocker_reason(blockers: list[dict[str, Any]]) -> str | None:
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        reason = blocker.get("reason") or blocker.get("impact") or blocker.get("title")
        reason_type = blocker.get("type") or blocker.get("id")
        if reason:
            return (
                f"{reason_type}: {reason}"
                if isinstance(reason_type, str) and reason_type
                else str(reason)
            )
    return None


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def throughput_claim_gate_payload(
    *,
    step_metrics: list[dict[str, Any]],
    tokens_per_second: float | None,
) -> dict[str, Any]:
    claim_present = tokens_per_second is not None
    reported_tokens_per_second_finite = _finite_number(tokens_per_second)
    checks: list[dict[str, Any]] = []
    total_tokens = 0
    total_seconds = 0.0
    reported_step_rates: list[float] = []

    for index, item in enumerate(step_metrics):
        ntokens = item.get("ntokens")
        seconds = item.get("seconds")
        reported_rate = item.get("tokens_per_second")
        ntokens_ok = (
            isinstance(ntokens, int | float)
            and not isinstance(ntokens, bool)
            and int(ntokens) > 0
        )
        seconds_ok = _finite_number(seconds) and float(seconds) > 0.0
        expected_rate = (
            int(ntokens) / float(seconds) if ntokens_ok and seconds_ok else None
        )
        reported_rate_finite = _finite_number(reported_rate)
        rate_consistent = (
            reported_rate_finite
            and expected_rate is not None
            and math.isclose(
                float(reported_rate),
                float(expected_rate),
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
        )
        if ntokens_ok:
            total_tokens += int(ntokens)
        if seconds_ok:
            total_seconds += float(seconds)
        if reported_rate_finite:
            reported_step_rates.append(float(reported_rate))
        checks.append(
            {
                "index": index,
                "ntokens": int(ntokens) if ntokens_ok else ntokens,
                "seconds": float(seconds) if seconds_ok else seconds,
                "reported_tokens_per_second": (
                    float(reported_rate) if reported_rate_finite else reported_rate
                ),
                "expected_tokens_per_second": expected_rate,
                "ntokens_positive": ntokens_ok,
                "seconds_positive": seconds_ok,
                "reported_tokens_per_second_finite": reported_rate_finite,
                "rate_consistent_with_ntokens_and_seconds": rate_consistent,
            }
        )

    reported_step_mean = (
        statistics.fmean(reported_step_rates) if reported_step_rates else None
    )
    weighted_tokens_per_second = (
        total_tokens / total_seconds if total_tokens > 0 and total_seconds > 0 else None
    )
    reported_matches_step_mean = (
        not claim_present
        or (
            reported_tokens_per_second_finite
            and reported_step_mean is not None
            and math.isclose(
                float(tokens_per_second),
                float(reported_step_mean),
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
        )
    )
    step_gate_ok = bool(checks) and all(
        bool(check["rate_consistent_with_ntokens_and_seconds"]) for check in checks
    )
    ok = (not claim_present) or (
        reported_tokens_per_second_finite and step_gate_ok and reported_matches_step_mean
    )
    return {
        "ok": ok,
        "claim_present": claim_present,
        "bogus_tok_sec_claim_detected": bool(claim_present and not ok),
        "reported_tokens_per_second": tokens_per_second,
        "reported_tokens_per_second_finite": reported_tokens_per_second_finite,
        "reported_tokens_per_second_matches_step_mean": reported_matches_step_mean,
        "reported_step_tokens_per_second_mean": reported_step_mean,
        "weighted_tokens_per_second_from_steps": weighted_tokens_per_second,
        "total_target_tokens": total_tokens if checks else None,
        "total_measured_seconds": total_seconds if checks else None,
        "step_metrics_count": len(checks),
        "step_rates_consistent": step_gate_ok,
        "step_checks": checks,
        "required_fields": [
            "step_metrics[].ntokens",
            "step_metrics[].seconds",
            "step_metrics[].tokens_per_second",
        ],
        "reported_kind": "mean_step_loss_target_tokens_per_second",
        "notes": [
            "Each step token/sec must equal ntokens / seconds.",
            "The receipt-level token/sec must equal the mean of recorded step rates.",
        ],
    }


def kernel_dispatch_report(
    args: argparse.Namespace | TrainHybridTinyConfig,
    *,
    train_payload: dict[str, Any],
    status: str,
    blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_dispatch = list(train_payload.get("kernel_dispatch") or [])
    dispatch_log = [entry for entry in raw_dispatch if isinstance(entry, dict)]
    requested_path_c_ops = _path_c_policy_ops(args)
    path_counts: dict[str, int] = {}
    kernel_counts: dict[str, int] = {}
    observed_ops: set[str] = set()
    observed_path_c_ops: set[str] = set()
    observed_reference_ops: set[str] = set()
    observed_path_b_ops: set[str] = set()
    unexpected_policy_entries: list[dict[str, Any]] = []

    for entry in dispatch_log:
        op_name = str(entry.get("op_name") or "")
        path = str(entry.get("path") or "")
        kernel_used = str(entry.get("kernel_used") or "")
        if op_name:
            observed_ops.add(op_name)
        if path:
            path_counts[path] = path_counts.get(path, 0) + 1
        if kernel_used:
            kernel_counts[kernel_used] = kernel_counts.get(kernel_used, 0) + 1
        if path == "path_c" or "path_c" in kernel_used:
            observed_path_c_ops.add(op_name)
        if path == "path_b" or kernel_used == "metal_kernel_fwd_v1":
            observed_path_b_ops.add(op_name)
        if kernel_used == "reference_pure_mlx":
            observed_reference_ops.add(op_name)
        if op_name in requested_path_c_ops and path != "path_c":
            unexpected_policy_entries.append(dict(entry))

    fallback_entries = [
        dict(entry)
        for entry in dispatch_log
        if entry.get("kernel_used") == "reference_pure_mlx"
    ]
    fallback_entries.extend(unexpected_policy_entries)
    sparse_mla_producer = sparse_mla_fp8_producer_payload(args)
    producer_missing = bool(
        fp8_path_c_route_requested(args)
        and not bool(sparse_mla_producer["configured"])
    )
    producer_unobserved = bool(
        status == "ok"
        and fp8_path_c_route_requested(args)
        and bool(sparse_mla_producer["configured"])
        and "sparse_mla" in requested_path_c_ops
        and "sparse_mla" not in observed_ops
    )
    blocker_reason = _first_blocker_reason(list(blockers or []))
    fallback_reason = None
    if blocker_reason is not None:
        fallback_reason = blocker_reason
    elif producer_missing:
        fallback_reason = str(sparse_mla_producer["reason"])
    elif producer_unobserved:
        fallback_reason = (
            f"{FP8_PATH_C_PRODUCER_UNOBSERVED_STATUS}: DSA Sparse-MLA FP8 "
            "producer is configured, "
            "but no sparse_mla Path C dispatch was recorded"
        )
    elif fallback_entries:
        first = fallback_entries[0]
        fallback_reason = (
            f"{first.get('op_name')} dispatched {first.get('kernel_used')} "
            f"under policy {first.get('path')}"
        )

    return {
        "dispatch_observed": bool(dispatch_log),
        "dispatch_clean": bool(
            status in {"ok", "dry_run"}
            and not fallback_entries
            and not producer_missing
            and not producer_unobserved
        ),
        "raw": dispatch_log,
        "observed_ops": sorted(op for op in observed_ops if op),
        "path_counts": dict(sorted(path_counts.items())),
        "kernel_counts": dict(sorted(kernel_counts.items())),
        "path_b_observed": bool(observed_path_b_ops),
        "path_c_observed": bool(observed_path_c_ops),
        "reference_observed": bool(observed_reference_ops),
        "path_summary": {
            "path_b": {
                "observed": bool(observed_path_b_ops),
                "ops": sorted(op for op in observed_path_b_ops if op),
            },
            "path_c": {
                "observed": bool(observed_path_c_ops),
                "ops": sorted(op for op in observed_path_c_ops if op),
            },
            "reference": {
                "observed": bool(observed_reference_ops),
                "ops": sorted(op for op in observed_reference_ops if op),
            },
        },
        "requested_path_c_ops": requested_path_c_ops,
        "observed_path_c_ops": sorted(op for op in observed_path_c_ops if op),
        "observed_path_b_ops": sorted(op for op in observed_path_b_ops if op),
        "observed_reference_ops": sorted(op for op in observed_reference_ops if op),
        "unobserved_requested_path_c_ops": [
            op for op in requested_path_c_ops if op not in observed_ops
        ],
        "fp8_sparse_mla_producer": sparse_mla_producer,
        "producer_missing": producer_missing,
        "producer_unobserved": producer_unobserved,
        "unexpected_policy_entries": unexpected_policy_entries,
        "fallback_detected": bool(
            fallback_entries
            or blocker_reason
            or producer_missing
            or producer_unobserved
        ),
        "fallback_entries": fallback_entries,
        "fallback_reason": fallback_reason,
        "kernel_policy_env": (
            dict(FP8_PATH_C_KERNEL_POLICY_ENV)
            if path_c_training_route_requested(args)
            else {}
        ),
    }


def regression_report_payload(
    args: argparse.Namespace | TrainHybridTinyConfig,
    *,
    config: TrainHybridTinyConfig | argparse.Namespace,
    train_payload: dict[str, Any],
    optimizer: dict[str, Any],
    memory_after: dict[str, Any],
    tokens_per_second: float | None,
    status: str,
    blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dispatch = kernel_dispatch_report(
        config,
        train_payload=train_payload,
        status=status,
        blockers=blockers,
    )
    fp8_producer_gate = fp8_path_c_producer_gate_payload(config)
    requested_dtype = str(getattr(config, "dtype", ""))
    optimizer_variant = optimizer_variant_payload(args)
    optimizer_key = str(optimizer.get("key") or optimizer_variant["key"])
    peak_memory_bytes = memory_after.get("peak_memory_bytes")
    step_metrics = [
        item for item in train_payload.get("step_metrics", []) if isinstance(item, dict)
    ]
    losses = [
        float(item["loss"])
        for item in step_metrics
        if _finite_number(item.get("loss"))
    ]
    all_finite = bool(losses) and len(losses) == len(step_metrics)
    final_loss = losses[-1] if losses else train_payload.get("final_loss")
    mean_loss = train_payload.get("mean_loss")
    loss_decreased = bool(len(losses) >= 2 and losses[-1] < losses[0])
    tokens_per_second_finite = _finite_number(tokens_per_second)
    throughput_claim_gate = throughput_claim_gate_payload(
        step_metrics=step_metrics,
        tokens_per_second=tokens_per_second,
    )
    return {
        "route_dispatch": dispatch,
        "fp8_path_c_producer_gate": fp8_producer_gate,
        "dtype": {
            "requested": requested_dtype,
            "carrier": carrier_dtype_for_acceptance(config),
            "precision_route": precision_route_payload(config),
        },
        "optimizer": {
            "requested": optimizer_variant["requested"],
            "key": optimizer_key,
            "name": optimizer.get("name"),
            "class": optimizer.get("class"),
            "quant_scheme": optimizer_variant["quant_scheme"],
            "quantized_state": optimizer.get("quantized_state"),
            "update_observed": optimizer.get("update_observed"),
        },
        "memory": {
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_gib": (
                float(peak_memory_bytes) / (1024**3)
                if isinstance(peak_memory_bytes, int)
                else None
            ),
        },
        "training": {
            "steps_completed": len(step_metrics),
            "loss_eval_chunks": train_payload.get("loss_eval_chunks"),
            "split_grad_update_eval": train_payload.get("split_grad_update_eval"),
            "split_grad_update_eval_reason": train_payload.get(
                "split_grad_update_eval_reason"
            ),
            "all_finite": all_finite,
            "losses": losses,
            "initial_loss": losses[0] if losses else None,
            "final_loss": final_loss,
            "final_loss_finite": _finite_number(final_loss),
            "mean_loss": mean_loss,
            "mean_loss_finite": _finite_number(mean_loss),
            "loss_decreased": loss_decreased,
            "loss_decrease_required": bool(getattr(args, "require_loss_decrease", False)),
            "loss_decrease_satisfied": (
                not bool(getattr(args, "require_loss_decrease", False))
            )
            or loss_decreased,
        },
        "throughput": {
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_finite": tokens_per_second_finite,
            "claim_gate": throughput_claim_gate,
        },
        "fallback_reason": dispatch["fallback_reason"],
        "gate_summary": {
            "dtype": requested_dtype,
            "optimizer": optimizer_key,
            "path_b_observed": dispatch["path_b_observed"],
            "path_c_observed": dispatch["path_c_observed"],
            "fp8_path_c_producer_status": fp8_producer_gate["status"],
            "fp8_path_c_producer_ok": fp8_producer_gate["ok"],
            "fallback_reason": dispatch["fallback_reason"],
            "all_finite": all_finite,
            "final_loss": final_loss,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_finite": tokens_per_second_finite,
            "tokens_per_second_claim_ok": throughput_claim_gate["ok"],
            "bogus_tok_sec_claim_detected": throughput_claim_gate[
                "bogus_tok_sec_claim_detected"
            ],
        },
        "visibility_gate": {
            "route_dispatch_visible": "raw" in dispatch,
            "dtype_visible": bool(requested_dtype),
            "optimizer_visible": bool(optimizer_key),
            "memory_peak_visible": "peak_memory_bytes" in memory_after,
            "tokens_per_second_visible": tokens_per_second is not None,
            "finite_visible": True,
            "loss_visible": True,
            "fallback_reason_visible": "fallback_reason" in dispatch,
        },
    }


def write_synthetic_npz(
    path: Path, *, steps: int, batch_size: int, seq_len: int, vocab_size: int
) -> None:
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


def run_receipt(
    args: argparse.Namespace,
    *,
    dry_run_payload_fn: Callable[..., dict[str, Any]] | None = None,
    train_hybrid_tiny_fn: Callable[..., dict[str, Any]] | None = None,
    local_gb10_route_fn: Callable[..., tuple[dict[str, Any], int]] | None = None,
    allocation_probe_fn: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
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
                payload, exit_code = _run_existing_training(
                    args,
                    data_path=data_path,
                    dry_run_payload_fn=dry_run_payload_fn,
                    train_hybrid_tiny_fn=train_hybrid_tiny_fn,
                    local_gb10_route_fn=local_gb10_route_fn,
                    allocation_probe_fn=allocation_probe_fn,
                )
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

    return _run_existing_training(
        args,
        data_path=data_path,
        dry_run_payload_fn=dry_run_payload_fn,
        train_hybrid_tiny_fn=train_hybrid_tiny_fn,
        local_gb10_route_fn=local_gb10_route_fn,
        allocation_probe_fn=allocation_probe_fn,
    )


def _run_existing_training(
    args: argparse.Namespace,
    *,
    data_path: Path,
    dry_run_payload_fn: Callable[..., dict[str, Any]] | None = None,
    train_hybrid_tiny_fn: Callable[..., dict[str, Any]] | None = None,
    local_gb10_route_fn: Callable[..., tuple[dict[str, Any], int]] | None = None,
    allocation_probe_fn: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    if dry_run_payload_fn is None:
        dry_run_payload_fn = dry_run_payload
    if train_hybrid_tiny_fn is None:
        train_hybrid_tiny_fn = train_hybrid_tiny
    if local_gb10_route_fn is None:
        local_gb10_route_fn = run_local_gb10_quarter_training
    if path_c_kernel_policy_requested():
        ensure_tilelang_dev_env_for_path_c()
    config = config_from_args(args, data_path=data_path)
    try:
        validate_config(config)
    except Exception as exc:
        return blocked_receipt(
            args, str(exc), type(exc).__name__
        ), 0 if args.dry_run_json else 2
    fp8_producer_gate = fp8_path_c_producer_gate_payload(config)
    if fp8_producer_gate["fail_closed"]:
        reason_type = str(fp8_producer_gate["status"])
        reason = str(fp8_producer_gate["reason"] or reason_type)
        prefix = f"{reason_type}: "
        if reason.startswith(prefix):
            reason = reason[len(prefix) :]
        return (
            blocked_receipt(
                args,
                reason,
                reason_type,
                probe_allocation=False,
            ),
            0 if args.dry_run_json else 2,
        )
    if config.model_profile == REQUIRED_MODEL_PROFILE:
        if args.dry_run_json:
            receipt = local_gb10_quarter_metadata_dry_run_receipt(
                args,
                config=config,
                data_path=data_path,
                allocation_probe_fn=allocation_probe_fn,
            )
            return enforce_loss_decrease_requirement(args, receipt)
        with (
            fp8_path_b_kernel_policy(config),
            fp8_path_c_kernel_policy(config),
            fp8_path_c_stdio_suppressed(config),
        ):
            return local_gb10_route_fn(
                args,
                config=config,
                data_path=data_path,
            )
    if optimizer_key_from_args(args) != "adamw":
        return (
            blocked_receipt(
                args,
                "non-default --optimizer choices are supported only with "
                "--model-profile local_gb10_quarter in this receipt path",
                "unsupported_optimizer_route",
            ),
            2,
        )

    reset_peak_memory()
    memory_before = metal_memory_payload()
    try:
        with (
            fp8_path_b_kernel_policy(config),
            fp8_path_c_kernel_policy(config),
            fp8_path_c_stdio_suppressed(config),
        ):
            if args.dry_run_json:
                train_payload = dry_run_payload_fn(
                    config,
                    npz_path=str(data_path),
                    valid_path=validation_dataset_path(config),
                )
            else:
                train_payload = train_hybrid_tiny_fn(
                    config,
                    npz_path=str(data_path),
                    valid_path=validation_dataset_path(config),
                )
    except Exception as exc:
        return blocked_receipt(
            args, str(exc), type(exc).__name__
        ), 0 if args.dry_run_json else 2
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
        return enforce_loss_decrease_requirement(args, receipt)
    return receipt, 0


def run_local_gb10_quarter_training(
    args: argparse.Namespace,
    *,
    config: TrainHybridTinyConfig,
    data_path: Path,
) -> tuple[dict[str, Any], int]:
    """Run the real full-profile M0.4 parquet training route."""

    if config.data_format != "parquet":
        return (
            blocked_receipt(
                args,
                "local_gb10_quarter training requires --data-format parquet; "
                f"got {config.data_format!r}",
                "unsupported_data_format",
                probe_allocation=False,
            ),
            2,
        )
    profile = local_gb10_quarter_profile()
    if config.seq_len > profile.max_seq_length:
        return (
            blocked_receipt(
                args,
                "local_gb10_quarter seq_len must not exceed "
                f"{profile.max_seq_length}; got {config.seq_len}",
                "invalid_cli",
                probe_allocation=False,
            ),
            2,
        )
    if config.dtype not in DTYPES:
        return (
            blocked_receipt(
                args,
                f"unsupported dtype={config.dtype!r}",
                "invalid_cli",
                probe_allocation=False,
            ),
            2,
        )

    model: Any | None = None
    optimizer: Any | None = None
    try:
        from cppmega_mlx.runtime.kernel_policy import clear_dispatch_log, get_dispatch_log

        clear_dispatch_log()
        memory_limit = train_memory_limit_payload(config, apply=True)
        cache_limit = apply_cache_limit_payload(args, mx_module=mx)
        mx.random.seed(config.seed)
        dataset = training_dataset_from_args(
            args,
            config=config,
            data_path=data_path,
            loop=True,
        )
        validate_side_channel_contract(config, dataset)
        validate_dataset_for_training(dataset, profile.vocab_size)

        device = device_info()
        compile_plan = compile_payload(config, device)
        loss_eval_chunks = not bool(compile_plan["enabled"])
        direct_chain_capture_requested = path_c_direct_chain_capture_requested(
            config,
            compile_enabled=bool(compile_plan["enabled"]),
        )
        if fp8_path_c_route_requested(config):
            # MLX 0.32 rejects explicit evals inside value_and_grad once the
            # FP8 Path C graph uses custom TileLang-backed nodes. Keep chunking
            # for memory shape, but let the enclosing eval own scheduling.
            loss_eval_chunks = False
        peak_memory_reset = bool(reset_peak_memory())
        memory_before = metal_memory_payload()

        model = local_gb10_quarter(
            dtype=DTYPES[config.dtype],
            grad_checkpoint=config.grad_checkpoint,
        )
        direct_chain_activation_capture: PathCActivationBufferCapture | None = None
        direct_chain_gradient_capture: PathCGradientBufferCapture | None = None
        route_backend = route_backend_payload(model)
        if direct_chain_capture_requested:
            (
                direct_chain_activation_capture,
                direct_chain_gradient_capture,
            ) = _path_c_direct_chain_runtime_capture_owners_for_model(model)
            model.attach_path_c_activation_probe(direct_chain_activation_capture)
        use_path_c_fused_train_block_runtime = bool(
            getattr(args, "use_path_c_fused_train_block_runtime", False)
        )
        fp8_path_c_training_route = fp8_path_c_training_route_payload_for_model(
            config,
            model,
            auto_install_fused_train_block=use_path_c_fused_train_block_runtime,
        )
        mx.eval(model.parameters())
        mx.synchronize()
        memory_after_parameters = metal_memory_payload()
        local_gb10_preflight = local_gb10_preflight_from_allocated_model(
            model,
            memory_before=memory_before,
            memory_after=memory_after_parameters,
        )

        optimizer = make_local_gb10_optimizer(
            args,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer.init(model.trainable_parameters())
        mx.eval(model.parameters(), optimizer.state)

        def loss_fn(model_arg: nn.Module, batch: Any) -> tuple[mx.array, mx.array]:
            return next_token_cut_cross_entropy(
                model_arg,
                batch,
                chunk_rows=config.cce_chunk_rows,
                eval_chunks=loss_eval_chunks,
            )

        batches = dataset.iter_batches(loop=True)
        first_batch: Any | None = next(batches) if config.steps > 0 else None
        fp8_path_c_direct_chain_critical_path_install = None
        path_c_training_runtime = None
        if (
            path_c_training_route_requested(config)
            and not bool(compile_plan["enabled"])
            and use_path_c_fused_train_block_runtime
        ):
            path_c_training_runtime = getattr(
                model,
                "path_c_fused_train_block_training_runtime",
                None,
            )
        if bool(getattr(args, "use_path_c_direct_chain_runtime", False)):
            try:
                if not path_c_training_route_requested(config):
                    fp8_path_c_direct_chain_critical_path_install = {
                        "status": "blocked",
                        "reason": (
                            "direct-chain critical path requires Path C kernel "
                            "policy or dtype=fp8_path_c"
                        ),
                        "training_critical_path": False,
                    }
                elif bool(compile_plan["enabled"]):
                    fp8_path_c_direct_chain_critical_path_install = {
                        "status": "blocked",
                        "reason": "direct-chain critical path requires compile=False",
                        "training_critical_path": False,
                    }
                elif first_batch is None:
                    fp8_path_c_direct_chain_critical_path_install = {
                        "status": "blocked",
                        "reason": "direct-chain critical path requires at least one batch",
                        "training_critical_path": False,
                    }
                else:
                    path_c_runtime_sequence_length = (
                        path_c_batch_sequence_length(first_batch)
                        or path_c_training_sequence_length(config)
                    )
                    profile_name = str(
                        getattr(model, "path_c_profile_name", "HybridTinyLM")
                    )
                    direct_chain_region_prefix = _path_c_direct_chain_region_prefix(
                        model,
                        profile_name,
                    )
                    direct_chains = plan_path_c_direct_fusion_chains_for_model(
                        model,
                        region_prefix=direct_chain_region_prefix,
                        include_backward=True,
                        max_segment_nodes=1,
                        sequence_length=path_c_runtime_sequence_length,
                    )
                    regions = build_path_c_model_regions_from_model(
                        model,
                        region_prefix=direct_chain_region_prefix,
                        include_backward=False,
                        sequence_length=path_c_runtime_sequence_length,
                    )
                    selected_region = _select_path_c_model_route_region(regions)
                    selected_chain = (
                        None
                        if selected_region is None
                        else _select_path_c_direct_chain_for_region(
                            direct_chains,
                            selected_region,
                        )
                    )
                    if selected_chain is None:
                        fp8_path_c_direct_chain_critical_path_install = {
                            "status": "blocked",
                            "reason": "model did not expose a direct Path C chain",
                            "training_critical_path": False,
                        }
                    else:
                        initial_owner = (
                            make_path_c_direct_chain_pre_step_runtime_owner(
                                chain=selected_chain,
                                model=model,
                                batch=first_batch,
                            )
                        )

                        def pre_step_owner_factory(
                            model_arg: nn.Module,
                            batch_arg: Mapping[str, mx.array],
                            *,
                            chain: Any = selected_chain,
                        ) -> PathCLogicalBufferOwner:
                            return make_path_c_direct_chain_pre_step_runtime_owner(
                                chain=chain,
                                model=model_arg,
                                batch=batch_arg,
                            )

                        fp8_path_c_direct_chain_critical_path_install = (
                            install_path_c_direct_chain_training_runtime_for_model(
                                model=model,
                                chain=selected_chain,
                                logical_owner=initial_owner,
                                sequence_length=path_c_runtime_sequence_length,
                                training_critical_path=True,
                                run_probe=False,
                                loss_cotangent_bridge=PathCResidualSumSuffixLossCotangentBridge(
                                    chunk_rows=config.cce_chunk_rows,
                                ),
                                pre_step_owner_factory=pre_step_owner_factory,
                            )
                        )
                        if (
                            fp8_path_c_direct_chain_critical_path_install.get(
                                "status"
                            )
                            == "ok"
                        ):
                            path_c_training_runtime = getattr(
                                model,
                                "path_c_direct_fusion_chain_training_runtime",
                                None,
                            )
            except Exception as exc:
                fp8_path_c_direct_chain_critical_path_install = {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "training_critical_path": False,
                }

        split_grad_update_eval = bool(
            direct_chain_capture_requested or path_c_training_runtime is not None
        )
        stepper = CompiledPretrainingStep(
            model,
            optimizer,
            state={"step": 0, "trained_tokens": 0},
            loss_fn=loss_fn,
            compile=bool(compile_plan["enabled"]),
            split_grad_update_eval=split_grad_update_eval,
            path_c_gradient_probe=direct_chain_gradient_capture,
            path_c_training_runtime=path_c_training_runtime,
        )
        clear_cache_events: list[dict[str, Any]] = []
        step_metrics: list[dict[str, Any]] = []
        last_batch: Any | None = None
        for step_index in range(config.steps):
            if direct_chain_activation_capture is not None:
                direct_chain_activation_capture.clear()
            if direct_chain_gradient_capture is not None:
                direct_chain_gradient_capture.clear()
            last_batch = first_batch if step_index == 0 else next(batches)
            metrics = stepper(last_batch)
            step_metrics.append(asdict(metrics))
            clear_cache_event = maybe_clear_cache_after_step(
                metrics.step,
                config.clear_cache_every_steps,
                mx_module=mx,
                synchronize=False,
            )
            if clear_cache_event is not None:
                clear_cache_events.append(clear_cache_event.to_dict())

        if not step_metrics:
            raise RuntimeError("local_gb10_quarter route completed zero steps")
        losses = [float(item["loss"]) for item in step_metrics]
        step_times = [float(item["seconds"]) for item in step_metrics]
        tps_values = [float(item["tokens_per_second"]) for item in step_metrics]
        final = step_metrics[-1]
        for index, item in enumerate(step_metrics, start=1):
            if not math.isfinite(float(item["loss"])):
                raise ValueError(f"step_metrics[{index}].loss must be finite")
            if int(item["ntokens"]) <= 0:
                raise ValueError(f"step_metrics[{index}].ntokens must be positive")
            if not math.isfinite(float(item["tokens_per_second"])):
                raise ValueError(
                    f"step_metrics[{index}].tokens_per_second must be finite"
                )

        mx.synchronize()
        memory_after = metal_memory_payload()
        fp8_path_c_post_step_runtime_capture = None
        fp8_path_c_direct_chain_runtime_probe = None
        if (
            direct_chain_activation_capture is not None
            and direct_chain_gradient_capture is not None
        ):
            existing_owners = _as_path_c_logical_owner_tuple(
                getattr(
                    model,
                    "path_c_direct_fusion_chain_logical_buffer_owners",
                    None,
                )
            )
            model.path_c_direct_fusion_chain_logical_buffer_owners = (
                *existing_owners,
                direct_chain_activation_capture,
                direct_chain_gradient_capture,
            )
            fp8_path_c_post_step_runtime_capture = (
                fp8_path_c_training_route_payload_for_model(
                    config,
                    model,
                    auto_install_fused_train_block=(
                        use_path_c_fused_train_block_runtime
                    ),
                )
            )
            direct_binding = (
                fp8_path_c_post_step_runtime_capture.get("path_c_fusion", {})
                .get("direct_chained_fusion", {})
                .get("runtime_binding", {})
            )
            missing_direct_buffers = tuple(
                str(name)
                for name in direct_binding.get("missing_logical_buffers", ())
            )
            if (
                missing_direct_buffers
                and all(name.endswith("_grad") for name in missing_direct_buffers)
            ):
                direct_chain_region_prefix = _path_c_direct_chain_region_prefix(
                    model,
                    str(getattr(model, "path_c_profile_name", "HybridTinyLM")),
                )
                direct_chains = plan_path_c_direct_fusion_chains_for_model(
                    model,
                    region_prefix=direct_chain_region_prefix,
                    include_backward=True,
                    sequence_length=path_c_training_sequence_length(config),
                )
                direct_chain = _select_path_c_direct_chain_for_region(
                    direct_chains,
                    (
                        fp8_path_c_post_step_runtime_capture.get("path_c_fusion", {})
                        .get("selected_forward_region", {})
                        .get("name")
                    ),
                )
                if direct_chain is not None:
                    workspace_owner = (
                        make_path_c_direct_fusion_chain_workspace_owner(
                            chain=direct_chain,
                            logical_buffer_names=missing_direct_buffers,
                            owner_name=(
                                f"{getattr(model, 'path_c_profile_name', 'HybridTinyLM')}"
                                ".path_c_direct_fusion_chain_workspace"
                            ),
                        )
                    )
                    model.path_c_direct_fusion_chain_logical_buffer_owners = (
                        *existing_owners,
                        direct_chain_activation_capture,
                        direct_chain_gradient_capture,
                        workspace_owner,
                    )
                    fp8_path_c_post_step_runtime_capture = (
                        fp8_path_c_training_route_payload_for_model(
                            config,
                            model,
                            auto_install_fused_train_block=(
                                use_path_c_fused_train_block_runtime
                            ),
                        )
                    )
            if bool(getattr(args, "profile_path_c_direct_chain_runtime", False)):
                try:
                    fp8_path_c_direct_chain_runtime_probe = (
                        install_path_c_direct_chain_training_runtime_for_model(
                            model=model,
                            training_critical_path=False,
                            run_probe=False,
                            loss_cotangent_bridge=PathCResidualSumSuffixLossCotangentBridge(
                                chunk_rows=config.cce_chunk_rows,
                            ),
                        )
                    )
                    runtime = getattr(
                        model,
                        "path_c_direct_fusion_chain_training_runtime",
                        None,
                    )
                    if (
                        fp8_path_c_direct_chain_runtime_probe.get("status") == "ok"
                        and runtime is not None
                        and last_batch is not None
                    ):
                        fp8_path_c_direct_chain_runtime_probe[
                            "value_and_grad_probe"
                        ] = path_c_direct_chain_value_and_grad_probe_payload(
                            runtime=runtime,
                            model=model,
                            batch=last_batch,
                        )
                    fp8_path_c_post_step_runtime_capture = (
                        fp8_path_c_training_route_payload_for_model(
                            config,
                            model,
                            auto_install_fused_train_block=(
                                use_path_c_fused_train_block_runtime
                            ),
                        )
                    )
                except Exception as exc:
                    fp8_path_c_direct_chain_runtime_probe = {
                        "status": "blocked",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "training_critical_path": False,
                    }
        if fp8_path_c_direct_chain_critical_path_install is not None:
            fp8_path_c_training_route = fp8_path_c_training_route_payload_for_model(
                config,
                model,
                auto_install_fused_train_block=(
                    use_path_c_fused_train_block_runtime
                ),
            )
        optimizer_evidence = optimizer_identity_for_selected_optimizer(
            args,
            config,
            optimizer,
            model,
            optimizer_updated=True,
        )
        train_payload = {
            "status": "ok",
            "config": asdict(config),
            "model_name": REQUIRED_MODEL_PROFILE,
            "model_profile": REQUIRED_MODEL_PROFILE,
            "model_source": REQUIRED_MODEL_SOURCE,
            "model_config": local_gb10_quarter_model_config_payload(model),
            "route_symbols": route_backend["route_symbols"],
            "route_roles": route_backend["route_roles"],
            "backend_plan": route_backend,
            "fp8_path_c_training_route": fp8_path_c_training_route,
            "fp8_path_c_post_step_runtime_capture": (
                fp8_path_c_post_step_runtime_capture
            ),
            "fp8_path_c_direct_chain_runtime_probe": (
                fp8_path_c_direct_chain_runtime_probe
            ),
            "fp8_path_c_direct_chain_critical_path_install": (
                fp8_path_c_direct_chain_critical_path_install
            ),
            "parameter_count": parameter_count(model),
            "tokens_per_step": final["ntokens"],
            "trained_tokens": final["trained_tokens"],
            "final_loss": final["loss"],
            "mean_loss": statistics.fmean(losses),
            "mean_step_time_s": statistics.fmean(step_times),
            "median_step_time_s": statistics.median(step_times),
            "tokens_per_second": statistics.fmean(tps_values),
            "step_metrics": step_metrics,
            "kernel_dispatch": get_dispatch_log(),
            "stepper_state": stepper.state_dict(),
            "compile": config.compile,
            "compile_enabled": compile_plan["enabled"],
            "compile_plan": compile_plan,
            "loss_eval_chunks": loss_eval_chunks,
            "split_grad_update_eval": split_grad_update_eval,
            "split_grad_update_eval_reason": (
                FP8_PATH_C_SPLIT_GRAD_UPDATE_EVAL_REASON
                if split_grad_update_eval
                else None
            ),
            "dtype": config.dtype,
            "dataset": dataset_payload(dataset, config),
            "device": device,
            "memory_limit": memory_limit,
            "memory": {
                "before": memory_before,
                "after": memory_after,
                "allocation_after_parameters": memory_after_parameters,
                "peak_memory_bytes": memory_after.get("peak_memory_bytes"),
                "peak_memory_reset": peak_memory_reset,
                "cache_limit": cache_limit,
                "clear_cache_every_steps": config.clear_cache_every_steps,
                "clear_cache_events": clear_cache_events,
                "clear_cache_event_count": len(clear_cache_events),
            },
            "optimizer_identity": optimizer_evidence,
            "local_gb10_quarter_preflight": local_gb10_preflight,
        }
        receipt = receipt_from_train_payload(
            args,
            config=config,
            train_payload=train_payload,
            memory_before=memory_before,
            memory_after=memory_after,
        )
        if args.require_loss_decrease and not receipt["training"]["loss_decreased"]:
            return enforce_loss_decrease_requirement(args, receipt)
        return receipt, 0
    except Exception as exc:
        reason = str(exc) or repr(exc)
        reason = f"{reason}\n{traceback.format_exc()}"
        return (
            blocked_receipt(
                args,
                reason,
                type(exc).__name__,
                probe_allocation=False,
            ),
            2,
        )
    finally:
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        try:
            mx.synchronize()
        except Exception:
            pass


def make_local_gb10_optimizer(
    args: argparse.Namespace,
    *,
    learning_rate: float,
    weight_decay: float,
) -> Any:
    key = optimizer_key_from_args(args)
    quant_scheme = str(getattr(args, "optimizer_quant_scheme", "dynamic_int8_v1"))
    if key == "adamw":
        return make_adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    if key == "muon_adamw":
        return make_muon(
            lr_muon=learning_rate,
            lr_adamw=learning_rate,
            weight_decay=weight_decay,
            cppmega_cuda_parity=True,
        )
    if key == "lion":
        return make_lion(learning_rate=learning_rate, weight_decay=weight_decay)
    if key == "adam8bit":
        return make_adam8bit(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            quant_scheme=quant_scheme,
            min_8bit_size=4096,
        )
    if key == "lion8bit":
        return make_lion8bit(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            quant_scheme=quant_scheme,
        )
    if key == "int8":
        return make_muon(
            lr_muon=learning_rate,
            lr_adamw=learning_rate,
            weight_decay=weight_decay,
            cppmega_cuda_parity=True,
            quantize_momentum=True,
            quantize_momentum_scheme=quant_scheme,
            scalar_optimizer="adam8bit",
            adam8bit_quant_scheme=quant_scheme,
            adam8bit_min_8bit_size=4096,
        )
    raise ValueError(f"unsupported optimizer={key!r}")


def selected_optimizer_static_identity(args: argparse.Namespace) -> dict[str, Any]:
    variant = optimizer_variant_payload(args)
    key = variant["key"]
    if key == "adamw":
        return {
            **OBSERVED_OPTIMIZER_IDENTITY,
            "key": key,
            "variant": variant,
            "adamw_family": True,
            "quantized_state": False,
        }
    if key == "muon_adamw":
        return {
            "name": "MuonAdamW",
            "key": key,
            "class": MUON_ADAMW_MULTI_CLASS,
            "base_class": "mlx.optimizers.Optimizer",
            "source": MUON_ADAMW_MULTI_SOURCE,
            "construction": (
                "repo-local make_muon(cppmega_cuda_parity=True, "
                "lr_muon=config.learning_rate, lr_adamw=config.learning_rate)"
            ),
            "variant": variant,
            "adamw_family": False,
            "quantized_state": False,
            "nam56r_style": True,
        }
    if key == "lion":
        return {
            "name": "Lion",
            "key": key,
            "class": LION_FP32_MOMENTS_CLASS,
            "base_class": "mlx.optimizers.Lion",
            "source": LION_FP32_MOMENTS_SOURCE,
            "construction": (
                "repo-local make_lion(learning_rate=config.learning_rate, "
                "weight_decay=config.weight_decay) with fp32 momentum"
            ),
            "variant": variant,
            "adamw_family": False,
            "quantized_state": False,
        }
    if key == "adam8bit":
        return {
            "name": "Adam8bit",
            "key": key,
            "class": ADAM8BIT_CLASS,
            "base_class": "mlx.optimizers.Optimizer",
            "source": ADAM8BIT_SOURCE,
            "construction": (
                "repo-local make_adam8bit(learning_rate=config.learning_rate, "
                "weight_decay=config.weight_decay, quant_scheme=..., "
                "min_8bit_size=4096)"
            ),
            "variant": variant,
            "adamw_family": True,
            "quantized_state": True,
        }
    if key == "lion8bit":
        return {
            "name": "Lion8bit",
            "key": key,
            "class": LION8BIT_CLASS,
            "base_class": "mlx.optimizers.Optimizer",
            "source": LION8BIT_SOURCE,
            "construction": (
                "repo-local make_lion8bit(learning_rate=config.learning_rate, "
                "weight_decay=config.weight_decay, quant_scheme=...)"
            ),
            "variant": variant,
            "adamw_family": False,
            "quantized_state": True,
        }
    if key == "int8":
        return {
            "name": "MuonAdamWInt8",
            "key": key,
            "class": MUON_ADAMW_MULTI_CLASS,
            "base_class": "mlx.optimizers.Optimizer",
            "source": MUON_INT8_SOURCE,
            "construction": (
                "repo-local make_muon(cppmega_cuda_parity=True, "
                "quantize_momentum=True, scalar_optimizer='adam8bit', "
                "adam8bit_min_8bit_size=4096)"
            ),
            "variant": variant,
            "adamw_family": False,
            "quantized_state": True,
            "nam56r_style": True,
        }
    raise ValueError(f"unsupported optimizer={key!r}")


def optimizer_state_dtype_breakdown(state: Any) -> dict[str, dict[str, int]]:
    breakdown: dict[str, dict[str, int]] = {}

    def walk(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk((*path, str(key)), item)
            return
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk((*path, str(index)), item)
            return
        if isinstance(value, mx.array):
            leaf = path[-1] if path else "<root>"
            dtype = dtype_name(value)
            by_dtype = breakdown.setdefault(leaf, {})
            by_dtype[dtype] = by_dtype.get(dtype, 0) + int(value.nbytes)

    walk((), state)
    return breakdown


def optimizer_state_evidence(optimizer: Any, model: Any) -> dict[str, Any]:
    state = optimizer.state if isinstance(optimizer.state, dict) else {}
    moment_dtypes = collect_adamw_moment_dtypes(state)
    sampled_moment_dtypes = dict(sorted(moment_dtypes.items())[:64])
    return {
        "observed_parameter_dtype": first_parameter_dtype(model),
        "state_keys": sorted(str(key) for key in state),
        "state_dtype_breakdown_bytes": optimizer_state_dtype_breakdown(state),
        "observed_adamw_moment_dtypes": sampled_moment_dtypes,
        "observed_adamw_moment_dtype_count": len(moment_dtypes),
        "observed_adamw_moment_dtypes_sampled": len(sampled_moment_dtypes),
        "observed_adamw_moment_dtypes_truncated": (
            len(sampled_moment_dtypes) < len(moment_dtypes)
        ),
    }


def optimizer_identity_for_selected_optimizer(
    args: argparse.Namespace,
    config: TrainHybridTinyConfig | argparse.Namespace,
    optimizer: Any,
    model: Any,
    *,
    optimizer_updated: bool,
) -> dict[str, Any]:
    if optimizer_key_from_args(args) == "adamw":
        return optimizer_identity(
            config,
            optimizer_updated=optimizer_updated,
            master_moment_evidence=adamw_moment_evidence_from_optimizer(
                optimizer,
                model,
            ),
        )
    identity = selected_optimizer_static_identity(args)
    state_evidence = optimizer_state_evidence(optimizer, model)
    return {
        **identity,
        "required_name": REQUIRED_OPTIMIZER_NAME,
        "name_matches_required": False,
        "adamw": False,
        "learning_rate": getattr(config, "learning_rate", getattr(config, "lr", None)),
        "weight_decay": getattr(config, "weight_decay", None),
        "update_observed": optimizer_updated,
        "required_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "master_moment_evidence": {
            "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
            "observed_parameter_dtype": state_evidence["observed_parameter_dtype"],
            "observed_moment_dtypes": state_evidence["observed_adamw_moment_dtypes"],
            "observed_moment_dtype_count": state_evidence[
                "observed_adamw_moment_dtype_count"
            ],
            "observed_moment_dtypes_sampled": state_evidence[
                "observed_adamw_moment_dtypes_sampled"
            ],
            "observed_moment_dtypes_truncated": state_evidence[
                "observed_adamw_moment_dtypes_truncated"
            ],
            "optimizer_class": identity["class"],
            "optimizer_base_class": identity["base_class"],
            "state_keys": state_evidence["state_keys"],
            "ok": False,
            "reason": (
                "M0.4 acceptance still requires repo-local AdamW fp32 moments; "
                "this receipt records an optimizer-matrix variant."
            ),
        },
        "master_moment_dtype_ok": False,
        "state_evidence": state_evidence,
    }


def local_gb10_preflight_from_allocated_model(
    model: Any,
    *,
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
) -> dict[str, Any]:
    profile_geometry = _local_gb10_quarter_profile_geometry()
    allocation_probe = {
        "status": "ok",
        "allocation_ready": True,
        "source": REQUIRED_MODEL_SOURCE,
        "allocation_mode": FULL_PROFILE_ALLOCATION_MODE,
        "required_geometry": REQUIRED_MODEL_GEOMETRY,
        "profile_geometry": profile_geometry,
        "geometry_matches_required": profile_geometry == REQUIRED_MODEL_GEOMETRY,
        "profile_name": REQUIRED_MODEL_PROFILE,
        "model_class": type(model).__name__,
        "eval_scope": ALLOCATION_PROBE_EVAL_SCOPE,
        "forward_executed": False,
        "training_executed": False,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    return local_gb10_quarter_preflight_payload(
        allocation_attempted=True,
        allocation_ready=True,
        allocation_mode=FULL_PROFILE_ALLOCATION_MODE,
        allocation_probe=allocation_probe,
    )


def local_gb10_quarter_model_config_payload(model: Any) -> dict[str, Any]:
    profile = local_gb10_quarter_profile()
    geometry = _local_gb10_quarter_profile_geometry()
    model_config = getattr(model, "config", None)
    to_dict = getattr(model_config, "to_dict", None)
    config_payload = to_dict() if callable(to_dict) else None
    return {
        "profile": REQUIRED_MODEL_PROFILE,
        "source": REQUIRED_MODEL_SOURCE,
        "max_seq_length": profile.max_seq_length,
        "dsa_a_layer_ranks": list(profile.dsa_a_layer_ranks),
        **geometry,
        "mtp_profile": geometry["mtp"],
        "config": config_payload,
    }


def adamw_moment_evidence_from_optimizer(
    optimizer: Any,
    model: Any,
) -> dict[str, Any]:
    try:
        moment_dtypes = collect_adamw_moment_dtypes(optimizer.state)
        sampled_moment_dtypes = dict(sorted(moment_dtypes.items())[:64])
        ok = bool(
            moment_dtypes
            and all(
                dtype == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
                for dtype in moment_dtypes.values()
            )
        )
        state = optimizer.state if isinstance(optimizer.state, dict) else {}
        return {
            "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
            "observed_parameter_dtype": first_parameter_dtype(model),
            "observed_moment_dtypes": sampled_moment_dtypes,
            "observed_moment_dtype_count": len(moment_dtypes),
            "observed_moment_dtypes_sampled": len(sampled_moment_dtypes),
            "observed_moment_dtypes_truncated": (
                len(sampled_moment_dtypes) < len(moment_dtypes)
            ),
            "optimizer_class": OBSERVED_OPTIMIZER_IDENTITY["class"],
            "optimizer_base_class": OBSERVED_OPTIMIZER_IDENTITY["base_class"],
            "state_keys": sorted(str(key) for key in state),
            "ok": ok,
        }
    except Exception as exc:
        return {
            "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
            "observed_parameter_dtype": first_parameter_dtype(model),
            "observed_moment_dtypes": {},
            "optimizer_class": OBSERVED_OPTIMIZER_IDENTITY["class"],
            "optimizer_base_class": OBSERVED_OPTIMIZER_IDENTITY["base_class"],
            "state_keys": [],
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def first_parameter_dtype(model: Any) -> str | None:
    for array in iter_mx_arrays(getattr(model, "parameters", lambda: {})()):
        return dtype_name(array)
    return None


def iter_mx_arrays(tree: Any):
    if isinstance(tree, mx.array):
        yield tree
        return
    if isinstance(tree, dict):
        for value in tree.values():
            yield from iter_mx_arrays(value)
        return
    if isinstance(tree, list | tuple):
        for value in tree:
            yield from iter_mx_arrays(value)


def enforce_loss_decrease_requirement(
    args: argparse.Namespace,
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    training = receipt.get("training")
    if not isinstance(training, dict):
        training = {}
        receipt["training"] = training
    if args.require_loss_decrease and not bool(training.get("loss_decreased")):
        receipt["status"] = "failed"
        training["loss_decrease_required"] = True
        training["loss_decrease_satisfied"] = False
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
    optimizer_updated = bool(step_metrics) and all(
        bool(item.get("updated")) for item in step_metrics
    )
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
    optimizer_payload = train_payload.get("optimizer_identity")
    optimizer = (
        optimizer_payload
        if isinstance(optimizer_payload, dict)
        else optimizer_identity(config, optimizer_updated=optimizer_updated)
    )
    grad_checkpoint = grad_checkpoint_payload(config)
    preflight_payload = train_payload.get("local_gb10_quarter_preflight")
    local_gb10_preflight = (
        preflight_payload
        if isinstance(preflight_payload, dict)
        else local_gb10_quarter_preflight_from_args(args)
    )
    observed_model_profile = train_payload.get("model_profile")
    observed_model_name = train_payload.get("model_name") or CURRENT_MODEL_NAME
    model_config_for_gate = dict(model_config)
    if isinstance(observed_model_profile, str):
        model_config_for_gate.setdefault("profile", observed_model_profile)
    acceptance_gate = acceptance_gate_payload(
        data_path=config.npz_path,
        data_format=config.data_format,
        dtype=carrier_dtype_for_acceptance(config),
        dataset=dataset,
        steps_requested=config.steps,
        steps_completed=len(step_metrics),
        loss_decreased=loss_decreased,
        all_finite=all_finite,
        optimizer_updated=optimizer_updated,
        model_name=observed_model_name,
        model_source=train_payload.get("model_source"),
        model_config=model_config_for_gate,
        optimizer=optimizer,
        grad_checkpoint=grad_checkpoint,
        device=train_payload.get("device", device_info()),
        local_gb10_quarter_preflight=local_gb10_preflight,
    )
    full_acceptance_claim = bool(
        acceptance_gate.get("full_local_gb10_quarter_gate_completed")
    )
    timing_tokens_per_second = (
        statistics.fmean(tokens_per_second)
        if tokens_per_second
        else train_payload.get("tokens_per_second")
    )
    regression_report = regression_report_payload(
        args,
        config=config,
        train_payload=train_payload,
        optimizer=optimizer,
        memory_after=memory_after,
        tokens_per_second=timing_tokens_per_second,
        status=status,
    )
    fp8_path_c_route = train_payload.get("fp8_path_c_training_route")
    if not isinstance(fp8_path_c_route, dict):
        fp8_path_c_route = fp8_path_c_training_route_payload(config)
    fp8_path_c_post_step_runtime_capture = train_payload.get(
        "fp8_path_c_post_step_runtime_capture"
    )
    if not isinstance(fp8_path_c_post_step_runtime_capture, dict):
        fp8_path_c_post_step_runtime_capture = None
    fp8_path_c_direct_chain_runtime_probe = train_payload.get(
        "fp8_path_c_direct_chain_runtime_probe"
    )
    if not isinstance(fp8_path_c_direct_chain_runtime_probe, dict):
        fp8_path_c_direct_chain_runtime_probe = None
    fp8_path_c_direct_chain_critical_path_install = train_payload.get(
        "fp8_path_c_direct_chain_critical_path_install"
    )
    if not isinstance(fp8_path_c_direct_chain_critical_path_install, dict):
        fp8_path_c_direct_chain_critical_path_install = None

    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_scope": RECEIPT_SCOPE,
        "status": status,
        "issue": {
            "id": "cppmega-mlx-t8f.4",
            "title": "M0.4: one bf16 training step + 100-step loss decrease on local parquet samples",
        },
        "local_only": True,
        "gb10_training_correctness_claim": full_acceptance_claim,
        "m4_vs_gb10_throughput_parity_claim": False,
        "full_m0_4_acceptance_claim": full_acceptance_claim,
        "acceptance_blockers": [] if full_acceptance_claim else list(OPEN_M0_BLOCKERS),
        "local_gb10_quarter_preflight": local_gb10_preflight,
        "acceptance_gate": acceptance_gate,
        "regression_report": regression_report,
        "m04_20step_matrix": m04_20step_matrix_payload(),
        "workload": {
            "target_data_path": target_dataset_path(),
            "data_path": str(config.npz_path),
            "data_format": config.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": config.dtype,
            "precision_route": precision_route_payload(config),
            "fp8_path_c_training_route": fp8_path_c_route,
            "fp8_path_c_post_step_runtime_capture": (
                fp8_path_c_post_step_runtime_capture
            ),
            "fp8_path_c_direct_chain_runtime_probe": (
                fp8_path_c_direct_chain_runtime_probe
            ),
            "fp8_path_c_direct_chain_critical_path_install": (
                fp8_path_c_direct_chain_critical_path_install
            ),
            "stepper_state": train_payload.get("stepper_state"),
            "steps_requested": config.steps,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "tokens_per_step": train_payload.get("tokens_per_step"),
            "compile_requested": config.compile,
            "learning_rate": config.learning_rate,
            "model_profile": config.model_profile,
            "optimizer": optimizer_variant_payload(args),
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
            "precision_route": precision_route_payload(config),
            "fp8_path_c_training_route": fp8_path_c_route,
            "fp8_path_c_post_step_runtime_capture": (
                fp8_path_c_post_step_runtime_capture
            ),
            "fp8_path_c_direct_chain_runtime_probe": (
                fp8_path_c_direct_chain_runtime_probe
            ),
            "fp8_path_c_direct_chain_critical_path_install": (
                fp8_path_c_direct_chain_critical_path_install
            ),
            "stepper_state": train_payload.get("stepper_state"),
            "all_finite": all_finite,
            "losses": losses,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else train_payload.get("final_loss"),
            "mean_loss": train_payload.get("mean_loss"),
            "loss_decreased": loss_decreased,
            "loss_decrease_required": bool(args.require_loss_decrease),
            "loss_decrease_satisfied": (not args.require_loss_decrease)
            or loss_decreased,
            "trained_tokens": train_payload.get("trained_tokens"),
            "loss_eval_chunks": train_payload.get("loss_eval_chunks"),
            "split_grad_update_eval": train_payload.get("split_grad_update_eval"),
            "split_grad_update_eval_reason": train_payload.get(
                "split_grad_update_eval_reason"
            ),
            "step_metrics": step_metrics,
            "kernel_dispatch": list(train_payload.get("kernel_dispatch") or []),
        },
        "timing": {
            "step_times_s": step_times,
            "mean_step_time_s": statistics.fmean(step_times) if step_times else None,
            "median_step_time_s": statistics.median(step_times) if step_times else None,
            "tokens_per_second": timing_tokens_per_second,
            "throughput_interpretation": throughput_interpretation_payload(
                config,
                train_payload=train_payload,
                step_metrics=step_metrics,
                tokens_per_second_values=tokens_per_second,
            ),
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
                "train_hybrid_tiny_step_loop" if clear_cache_events else None
            ),
            "trainer_memory": trainer_memory or None,
        },
        "dataset": dataset,
        "model": {
            "source": train_payload.get("model_source"),
            "name": observed_model_name,
            "required_profile": REQUIRED_MODEL_PROFILE,
            "profile": observed_model_profile,
            "profile_matches_required": observed_model_profile
            == REQUIRED_MODEL_PROFILE,
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


def _model_geometry_matches(
    model_config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
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
    resolved_allocation_ready = (
        bool(allocation_ready) if allocation_ready is not None else False
    )
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
    allocation_probe_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    should_probe = (
        bool(args.probe_local_gb10_quarter_allocation)
        if probe_allocation is None
        else probe_allocation
    )
    if not should_probe:
        return local_gb10_quarter_preflight_payload()

    if allocation_probe_fn is None:
        allocation_probe_fn = probe_local_gb10_quarter_allocation
    allocation_probe = allocation_probe_fn()
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
        "key": "adamw",
        "variant": {
            "requested": "adamw",
            "key": "adamw",
            "quant_scheme": getattr(config, "optimizer_quant_scheme", None),
            "source": "default",
        },
        "adamw_family": True,
        "quantized_state": False,
        "required_name": REQUIRED_OPTIMIZER_NAME,
        "name_matches_required": OBSERVED_OPTIMIZER_IDENTITY["name"]
        == REQUIRED_OPTIMIZER_NAME,
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


def metadata_only_optimizer_identity(
    args: argparse.Namespace,
    config: TrainHybridTinyConfig | argparse.Namespace,
) -> dict[str, Any]:
    identity = selected_optimizer_static_identity(args)
    moment_evidence = {
        "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "observed_parameter_dtype": None,
        "observed_moment_dtypes": {},
        "optimizer_class": identity["class"],
        "optimizer_base_class": identity["base_class"],
        "state_keys": [],
        "ok": False,
        "skipped": True,
        "reason": "metadata-only dry-run does not allocate optimizer state",
    }
    if optimizer_key_from_args(args) == "adamw":
        return optimizer_identity(
            config,
            optimizer_updated=False,
            master_moment_evidence=moment_evidence,
        )
    return {
        **identity,
        "required_name": REQUIRED_OPTIMIZER_NAME,
        "name_matches_required": False,
        "adamw": False,
        "learning_rate": getattr(config, "learning_rate", getattr(config, "lr", None)),
        "weight_decay": getattr(config, "weight_decay", None),
        "update_observed": False,
        "required_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "master_moment_evidence": moment_evidence,
        "master_moment_dtype_ok": False,
        "state_evidence": {
            "observed_parameter_dtype": None,
            "state_keys": [],
            "state_dtype_breakdown_bytes": {},
            "observed_adamw_moment_dtypes": {},
            "observed_adamw_moment_dtype_count": 0,
            "observed_adamw_moment_dtypes_sampled": 0,
            "observed_adamw_moment_dtypes_truncated": False,
        },
    }


def local_gb10_quarter_metadata_dry_run_receipt(
    args: argparse.Namespace,
    *,
    config: TrainHybridTinyConfig,
    data_path: Path,
    allocation_probe_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Emit a metadata-only preflight receipt for the full M0.4 target profile.

    The opt-in allocation probe may instantiate parameters. This route never
    runs forward/training or allocates optimizer state.
    """

    local_gb10_preflight = local_gb10_quarter_preflight_from_args(
        args,
        allocation_probe_fn=allocation_probe_fn,
    )
    optimizer = metadata_only_optimizer_identity(args, config)
    grad_checkpoint = grad_checkpoint_payload(config)
    memory_snapshot = metal_memory_payload()
    device = device_info()
    commit = git_commit()
    acceptance_gate = acceptance_gate_payload(
        data_path=str(data_path),
        data_format=config.data_format,
        dtype=carrier_dtype_for_acceptance(config),
        dataset=None,
        steps_requested=config.steps,
        steps_completed=0,
        loss_decreased=False,
        all_finite=False,
        optimizer_updated=False,
        model_name=None,
        model_source=None,
        model_config=None,
        optimizer=optimizer,
        grad_checkpoint=grad_checkpoint,
        device=device,
        local_gb10_quarter_preflight=local_gb10_preflight,
    )
    regression_report = regression_report_payload(
        args,
        config=config,
        train_payload={},
        optimizer=optimizer,
        memory_after=memory_snapshot,
        tokens_per_second=None,
        status="dry_run",
    )
    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_scope": RECEIPT_SCOPE,
        "status": "dry_run",
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
        "regression_report": regression_report,
        "m04_20step_matrix": m04_20step_matrix_payload(),
        "workload": {
            "target_data_path": target_dataset_path(),
            "data_path": str(data_path),
            "data_format": config.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": config.dtype,
            "precision_route": precision_route_payload(config),
            "fp8_path_c_training_route": fp8_path_c_training_route_payload(config),
            "steps_requested": config.steps,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "tokens_per_step": config.batch_size * max(config.seq_len - 1, 0),
            "compile_requested": config.compile,
            "learning_rate": config.learning_rate,
            "model_profile": config.model_profile,
            "optimizer": optimizer_variant_payload(args),
            "grad_checkpoint": config.grad_checkpoint,
            "mode": "metadata_only_no_forward_no_training",
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
            "optimizer": optimizer,
            "grad_checkpoint": grad_checkpoint,
            "precision_route": precision_route_payload(config),
            "fp8_path_c_training_route": fp8_path_c_training_route_payload(config),
            "all_finite": False,
            "losses": [],
            "initial_loss": None,
            "final_loss": None,
            "mean_loss": None,
            "loss_decreased": False,
            "loss_decrease_required": bool(args.require_loss_decrease),
            "loss_decrease_satisfied": False,
            "trained_tokens": 0,
            "step_metrics": [],
            "kernel_dispatch": [],
        },
        "timing": {
            "step_times_s": [],
            "mean_step_time_s": None,
            "median_step_time_s": None,
            "tokens_per_second": None,
            "throughput_interpretation": throughput_interpretation_payload(
                config,
                train_payload={},
                step_metrics=[],
                tokens_per_second_values=[],
            ),
        },
        "memory": {
            "before": memory_snapshot,
            "after": memory_snapshot,
            "peak_memory_bytes": memory_snapshot.get("peak_memory_bytes"),
            "memory_limit": None,
            "memory_limit_api_status": memory_limit_api_status(mx).to_dict(),
            "applied_memory_limit_api_path": None,
            "clear_cache_every_steps": args.clear_cache_every_steps,
            "clear_cache_cadence_recorded": args.clear_cache_every_steps is not None,
            "clear_cache_events": [],
            "clear_cache_event_count": 0,
            "clear_cache_event": None,
            "clear_cache_event_recorded": False,
            "clear_cache_event_scope": None,
            "trainer_memory": None,
        },
        "dataset": {},
        "model": {
            "source": None,
            "name": None,
            "observed_source": None,
            "observed_name": None,
            "required_source": REQUIRED_MODEL_SOURCE,
            "required_name": REQUIRED_MODEL_PROFILE,
            "required_profile": REQUIRED_MODEL_PROFILE,
            "requested_profile": config.model_profile,
            "profile": None,
            "requested_profile_matches_required": (
                config.model_profile == REQUIRED_MODEL_PROFILE
            ),
            "profile_matches_required": False,
            "local_gb10_quarter_preflight": local_gb10_preflight,
            "parameter_count": None,
            "route_symbols": None,
            "route_roles": None,
            "backend_plan": None,
            "config": None,
            "metadata_only": True,
            "forward_executed": False,
            "training_executed": False,
        },
        "software": {
            "git_commit": commit,
            "device": device,
        },
        "baseline_row": {
            "hardware": str(device.get("machine") or "local-mac"),
            "commit": commit or "unknown",
            "dtype": config.dtype,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "route": "metadata_only_no_forward_no_training",
            "model": "metadata_only_no_observed_model",
            "mode": "metadata_only_no_forward_no_training",
            "tokens_per_second": 0.0,
            "local_only": True,
            "gb10_parity_claim": False,
        },
    }
    return json_ready(receipt)


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
    model_geometry_ok, observed_model_geometry = _model_geometry_matches(
        model_config_payload
    )
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
        and preflight_allocation_probe.get("required_geometry")
        == REQUIRED_MODEL_GEOMETRY
        and preflight_allocation_probe.get("profile_geometry")
        == REQUIRED_MODEL_GEOMETRY
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
    optimizer = metadata_only_optimizer_identity(args, args)
    blockers = [
        {
            "type": reason_type,
            "reason": reason,
            "recoverable": True,
        },
        *OPEN_M0_BLOCKERS,
    ]
    memory_before = metal_memory_payload()
    memory_after = metal_memory_payload()
    regression_report = regression_report_payload(
        args,
        config=args,
        train_payload={},
        optimizer=optimizer,
        memory_after=memory_after,
        tokens_per_second=None,
        status="blocked",
        blockers=blockers,
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
            dtype=carrier_dtype_for_acceptance(args),
            dataset=None,
            steps_requested=args.steps,
            steps_completed=0,
            loss_decreased=False,
            all_finite=False,
            optimizer_updated=False,
            model_name=None,
            model_source=None,
            model_config=None,
            optimizer=optimizer,
            grad_checkpoint=grad_checkpoint_payload(args),
            device=device_info(),
            local_gb10_quarter_preflight=local_gb10_preflight,
        ),
        "regression_report": regression_report,
        "m04_20step_matrix": m04_20step_matrix_payload(),
        "blockers": blockers,
        "workload": {
            "target_data_path": target_dataset_path(),
            "data_path": str(args.data_path),
            "data_format": args.data_format,
            "synthetic": bool(args.synthetic),
            "dtype": args.dtype,
            "precision_route": precision_route_payload(args),
            "fp8_path_c_training_route": fp8_path_c_training_route_payload(args),
            "steps_requested": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "compile_requested": bool(args.compile),
            "learning_rate": learning_rate_from_args(args),
            "model_profile": args.model_profile,
            "optimizer": optimizer_variant_payload(args),
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
            "optimizer": optimizer,
            "grad_checkpoint": grad_checkpoint_payload(args),
            "precision_route": precision_route_payload(args),
            "fp8_path_c_training_route": fp8_path_c_training_route_payload(args),
            "all_finite": False,
            "losses": [],
            "initial_loss": None,
            "final_loss": None,
            "mean_loss": None,
            "loss_decreased": False,
            "loss_decrease_required": bool(args.require_loss_decrease),
            "loss_decrease_satisfied": False,
            "kernel_dispatch": [],
        },
        "timing": {
            "step_times_s": [],
            "mean_step_time_s": None,
            "median_step_time_s": None,
            "tokens_per_second": None,
            "throughput_interpretation": throughput_interpretation_payload(
                args,
                train_payload={},
                step_metrics=[],
                tokens_per_second_values=[],
            ),
        },
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "peak_memory_bytes": memory_after.get("peak_memory_bytes"),
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
        "model": str(train_payload.get("model_name") or CURRENT_MODEL_NAME),
        "mode": mode,
        "tokens_per_second": float(train_payload.get("tokens_per_second") or 0.0),
        "local_only": True,
        "gb10_parity_claim": False,
    }


def throughput_interpretation_payload(
    config: TrainHybridTinyConfig | argparse.Namespace,
    *,
    train_payload: dict[str, Any],
    step_metrics: list[dict[str, Any]],
    tokens_per_second_values: list[float],
) -> dict[str, Any]:
    """Explain what m04 token/sec does and does not measure."""

    batch_size = int(getattr(config, "batch_size", 0) or 0)
    seq_len = int(getattr(config, "seq_len", 0) or 0)
    model_profile = str(getattr(config, "model_profile", ""))
    synthetic = bool(getattr(config, "synthetic", False))
    if not synthetic:
        synthetic = bool(train_payload.get("synthetic_npz", False))
    production_seq_len = int(local_gb10_quarter_profile().max_seq_length)
    input_tokens_per_step = (
        batch_size * seq_len if batch_size > 0 and seq_len > 0 else None
    )
    nominal_target_tokens_per_step = (
        batch_size * max(seq_len - 1, 0) if batch_size > 0 and seq_len > 0 else None
    )
    measured_target_tokens = [
        int(item["ntokens"])
        for item in step_metrics
        if isinstance(item, dict) and "ntokens" in item
    ]
    measured_seconds = [
        float(item["seconds"])
        for item in step_metrics
        if isinstance(item, dict) and "seconds" in item
    ]
    total_measured_seconds = sum(measured_seconds)
    total_target_tokens = sum(measured_target_tokens)
    total_input_tokens = (
        input_tokens_per_step * len(step_metrics)
        if input_tokens_per_step is not None
        else None
    )
    input_tokens_per_second = (
        total_input_tokens / total_measured_seconds
        if total_input_tokens is not None and total_measured_seconds > 0
        else None
    )
    target_tokens_per_second = (
        statistics.fmean(tokens_per_second_values)
        if tokens_per_second_values
        else train_payload.get("tokens_per_second")
    )
    if target_tokens_per_second is not None:
        target_tokens_per_second = float(target_tokens_per_second)

    production_shape = (
        model_profile == REQUIRED_MODEL_PROFILE
        and seq_len == production_seq_len
        and not synthetic
    )
    if model_profile != REQUIRED_MODEL_PROFILE:
        scope = "tiny_or_hybrid_smoke"
        warning = (
            "This is a tiny/hybrid training-plumbing smoke, not local_gb10_quarter "
            "throughput evidence."
        )
    elif synthetic:
        scope = "synthetic_full_profile_smoke"
        warning = (
            "This uses synthetic data; it can isolate model/optimizer cost but does "
            "not prove target-parquet acceptance."
        )
    elif seq_len < production_seq_len:
        scope = "short_sequence_full_profile_smoke"
        warning = (
            f"seq_len={seq_len} underfills the local_gb10_quarter production "
            f"shape ({production_seq_len}); low tok/sec here is a short-sequence "
            "latency smoke, not the 4k production throughput denominator."
        )
    else:
        scope = "production_sequence_full_profile"
        warning = None

    return json_ready(
        {
            "reported_tokens_per_second_kind": "loss_target_tokens_per_second",
            "denominator": "sum(step_metrics[].ntokens)",
            "denominator_note": (
                "ntokens is the number of next-token loss targets after shifting; "
                "for an unmasked dense batch it is batch_size * (seq_len - 1)."
            ),
            "input_tokens_per_step": input_tokens_per_step,
            "nominal_target_tokens_per_step": nominal_target_tokens_per_step,
            "measured_target_tokens_per_step": measured_target_tokens,
            "total_input_tokens": total_input_tokens,
            "total_target_tokens": total_target_tokens,
            "total_measured_seconds": (
                total_measured_seconds if measured_seconds else None
            ),
            "input_tokens_per_second": input_tokens_per_second,
            "target_tokens_per_second": target_tokens_per_second,
            "timed_scope": (
                "CompiledPretrainingStep.__call__: loss/value_and_grad, "
                "optimizer.update, mx.eval(model.state, optimizer.state, "
                "mx.random.state, loss, ntokens, grad_accum), and scalar "
                "loss/ntokens materialization."
            ),
            "excluded_from_step_timer": [
                "dataset construction",
                "next(batches) parquet/npz batch fetch",
                "model allocation",
                "optimizer initialization",
                "receipt JSON serialization",
                "post-step cache clear cadence",
            ],
            "post_step_loop_synchronize": "not_needed_after_step_metrics_materialize",
            "compile_first_call_included_when_compile_enabled": bool(
                getattr(config, "compile", False)
            ),
            "production_seq_len": production_seq_len,
            "production_shape": production_shape,
            "workload_scope": scope,
            "warning": warning,
        }
    )


def reset_peak_memory() -> bool:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return True
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        metal.reset_peak_memory()
        return True
    return False


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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    if args.profile_hold_seconds > 0:
        print(f"profile_hold_seconds: {args.profile_hold_seconds}")
        sys.stdout.flush()
        time.sleep(args.profile_hold_seconds)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
