"""Training, loss, and checkpoint helpers.

The package exports the same convenience names as before, but resolves them
on first access.  Keeping package import lightweight matters for Path C/TVM
compile paths: importing ``training.mtp`` for model config metadata must not
also load the native optimizer extension and Metal tooling.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASELINES = frozenset(
    {
        "BASELINE_ARCHIVE_KIND",
        "BASELINE_INDEX_FILENAME",
        "BASELINE_INDEX_KIND",
        "BASELINE_INDEX_SCHEMA_VERSION",
        "BASELINE_ROW_SCHEMA_VERSION",
        "PARITY_EVIDENCE_POLICY",
        "REQUIRED_BASELINE_ROW_KEYS",
        "VALID_MODES",
        "BaselineValidationError",
        "archive_baseline_row",
        "baseline_filename",
        "validate_baseline_row",
    }
)
_COMPILED = frozenset(
    {
        "CompileTarget",
        "CompiledPretrainingStep",
        "PretrainingMetrics",
        "PretrainingState",
        "REGIONAL_COMPILE_TARGETS",
        "STABLE_BATCH_KEYS",
        "maybe_compile_region",
        "normalize_compiled_batch",
        "regional_compile",
        "should_compile_region",
    }
)
_CHECKPOINT = frozenset(
    {
        "FORMAT_NAME",
        "FORMAT_VERSION",
        "GRAD_ACCUM_NAME",
        "METADATA_NAME",
        "OPTIMIZER_NAME",
        "RNG_MODE_NOT_SAVED",
        "RNG_MODE_SEED",
        "SHARDING_MODE_SINGLE_FILE",
        "SHARD_INDEX_NAME",
        "WEIGHTS_NAME",
        "load_checkpoint",
        "save_checkpoint",
    }
)
_EVAL = frozenset({"EvalMetrics", "evaluate_batches"})
_LOOP = frozenset(
    {
        "TrainStepResult",
        "assert_grad_dtype_matches_param_dtype",
        "make_adamw",
        "one_step_train",
    }
)
_LOSS = frozenset(
    {
        "next_token_cross_entropy",
        "next_token_cross_entropy_mtp_loss",
        "next_token_cross_entropy_with_mtp",
        "next_token_cross_entropy_with_stp",
    }
)
_MTP = frozenset(
    {
        "DEFAULT_MTP_DECAY",
        "DEFAULT_MTP_DEPTH",
        "DEFAULT_MTP_LAMBDA",
        "MTP_IGNORE_INDEX",
        "MTPLossConfig",
        "MTPLossMetrics",
        "MinimalMTPHead",
        "MinimalMTPSharedBlock",
        "attach_mtp_head",
        "compute_mtp_step_weights",
        "compute_weighted_mtp_loss",
        "get_or_attach_mtp_head",
        "mtp_cross_entropy_from_logits",
        "mtp_loss_for_model",
        "next_token_and_mtp_loss",
        "roll_and_mask_mtp_ids",
        "roll_and_mask_mtp_labels",
    }
)
_STP = frozenset(
    {
        "DEFAULT_STP_LAMBDA",
        "DEFAULT_STP_SPANS",
        "STP_COSINE_EPSILON",
        "STPLossConfig",
        "STPLossMetrics",
        "compute_stp_loss",
        "next_token_and_stp_loss",
    }
)
_OPTIMIZERS = frozenset(
    {
        "ADAM8BIT_CLASS",
        "ADAM8BIT_QUANT_KIND",
        "ADAM8BIT_SOURCE",
        "ADAMW_BASE_CLASS",
        "ADAMW_FP32_MOMENTS_CLASS",
        "ADAMW_FP32_MOMENTS_SOURCE",
        "ADAMW_MOMENT_STATE_KEYS",
        "Adam8bit",
        "AdamWFP32Moments",
        "LION8BIT_CLASS",
        "LION8BIT_QUANT_KIND",
        "LION8BIT_SOURCE",
        "Lion8bit",
        "LionFP32Moments",
        "MUON_ADAMW_MULTI_CLASS",
        "MUON_ADAMW_MULTI_SOURCE",
        "MUON_QUANTIZED_MOMENTUM_BLOCK_SIZE",
        "MUON_QUANTIZED_MOMENTUM_SCHEME",
        "MuonAdamWMulti",
        "QuantizedMuonWithNSCarrier",
        "adamw_moment_dtypes_ok",
        "collect_adamw_moment_dtypes",
        "dtype_name",
        "make_adam8bit",
        "make_lion",
        "make_lion8bit",
        "make_muon",
    }
)
_DISTRIBUTED = frozenset(
    {
        "DISTRIBUTED_OPTIMIZER_SUPPORTED_INNER_CLASSES",
        "ZERO1_STREAM_F_POLICY",
        "DistributedZeRO1Optimizer",
        "make_distributed_optimizer",
    }
)
_MLX_LM_ADAPTER = frozenset(
    {
        "MLX_LM_DENSE_BATCH_KEYS",
        "MLXLMAPIInfo",
        "MLXLMBatchRouteMetadata",
        "MLXLMTrainerIntegrationUnsupported",
        "REQUIRED_TRAINER_PARAMETERS",
        "STRUCTURE_FIELD_NAMES",
        "TRAINER_ADAPTER_SAVE_FIELDS",
        "TRAINER_API_NAMES",
        "TRAINER_MODULE",
        "TRAINING_ARGS_MEMORY_POLICY_FIELDS",
        "UNSUPPORTED_TRAINER_INTEGRATION_REASON",
        "as_mlx_lm_loss_args",
        "as_mlx_lm_token_mapping",
        "describe_mlx_lm_batch_route_metadata",
        "describe_mlx_lm_trainer_apis",
        "require_supported_mlx_lm_trainer_integration",
    }
)
_PARITY = frozenset(
    {
        "LOCAL_ONLY_POLICY",
        "M05_MTP_BETA",
        "M05_MTP_CUDA_ARTIFACT_CONTRACT",
        "M05_MTP_CUDA_ARTIFACT_FORMAT",
        "M05_MTP_CUDA_ARTIFACT_PATH",
        "M05_MTP_CUDA_ARTIFACT_PREFLIGHT_STATUSES",
        "M05_MTP_CUDA_LOSS_VALUE_FIELDS",
        "M05_MTP_CUDA_REFERENCE_SOURCES",
        "M05_MTP_DEPTH",
        "M05_MTP_LAMBDA",
        "M05_MTP_PARITY_ISSUE_ID",
        "M05_MTP_PARITY_OUTPUT",
        "M05_MTP_PARITY_POLICY",
        "M05_MTP_PARITY_PROFILE",
        "M05_MTP_PARITY_RECEIPT_SCOPE",
        "PARITY_MANIFEST_FORMAT",
        "PARITY_MANIFEST_VERSION",
        "PARITY_RECEIPT_SCOPE",
        "REQUIRED_RECEIPT_FIELDS",
        "VALID_PARITY_STATUSES",
        "TensorParityReceipt",
        "build_m05_mtp_parity_manifest",
        "build_parity_manifest",
        "coerce_parity_receipt",
        "m05_loss_values_sha256",
        "validate_m05_cuda_reference_artifact_dict",
        "validate_m05_mtp_parity_manifest_dict",
        "validate_parity_manifest_dict",
        "validate_parity_receipt_dict",
        "write_m05_mtp_parity_manifest_json",
        "write_parity_manifest_json",
        "write_parity_manifest_jsonl",
    }
)
_PROFILE = frozenset(
    {
        "HotspotEvidence",
        "KernelAdoptionAssessment",
        "KernelAdoptionBlocked",
        "MemorySnapshot",
        "ProfileContext",
        "ProfileMetrics",
        "StepProfiler",
        "assess_kernel_adoption",
        "hotspot_from_profile_metrics",
        "profile_context",
        "profile_step",
        "require_kernel_hotspot_evidence",
        "reset_peak_memory",
        "summarize_hotspots",
        "synchronize",
    }
)

_MODULE_BY_EXPORT: dict[str, str] = {
    **dict.fromkeys(_BASELINES, "cppmega_mlx.training.baselines"),
    **dict.fromkeys(_COMPILED, "cppmega_mlx.training.compiled"),
    **dict.fromkeys(_CHECKPOINT, "cppmega_mlx.training.checkpoint"),
    **dict.fromkeys(_EVAL, "cppmega_mlx.training.eval"),
    **dict.fromkeys(_LOOP, "cppmega_mlx.training.loop"),
    **dict.fromkeys(_LOSS, "cppmega_mlx.training.loss"),
    **dict.fromkeys(_MTP, "cppmega_mlx.training.mtp"),
    **dict.fromkeys(_STP, "cppmega_mlx.training.stp_loss"),
    **dict.fromkeys(_OPTIMIZERS, "cppmega_mlx.training.optimizers"),
    **dict.fromkeys(_DISTRIBUTED, "cppmega_mlx.training.distributed_optimizer"),
    **dict.fromkeys(_MLX_LM_ADAPTER, "cppmega_mlx.training.mlx_lm_adapter"),
    **dict.fromkeys(_PARITY, "cppmega_mlx.training.parity"),
    **dict.fromkeys(_PROFILE, "cppmega_mlx.training.profile"),
}
_ALIASES = {
    "DISTRIBUTED_OPTIMIZER_SUPPORTED_INNER_CLASSES": "SUPPORTED_INNER_CLASSES",
}

__all__ = sorted(_MODULE_BY_EXPORT)


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_EXPORT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, _ALIASES.get(name, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
