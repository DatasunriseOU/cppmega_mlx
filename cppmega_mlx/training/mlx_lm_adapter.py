"""Optional MLX-LM trainer compatibility probes.

The helpers in this module keep MLX-LM at the edge of cppmega's training
surface.  They import MLX-LM lazily, report the trainer APIs that are actually
installed, and normalize cppmega token batches into the simple token/length
mapping used by MLX-LM loss and batching code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import importlib.metadata
import inspect
from types import ModuleType
from typing import Any, Mapping

import mlx.core as mx

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch


TRAINER_MODULE = "mlx_lm.tuner.trainer"
TRAINER_API_NAMES = (
    "TrainingArgs",
    "default_loss",
    "iterate_batches",
    "evaluate",
    "train",
    "grad_checkpoint",
)
REQUIRED_TRAINER_PARAMETERS: dict[str, tuple[str, ...]] = {
    "TrainingArgs": (
        "batch_size",
        "iters",
        "max_seq_length",
        "grad_accumulation_steps",
    ),
    "default_loss": ("model", "batch", "lengths"),
    "iterate_batches": ("dataset", "batch_size", "max_seq_length"),
    "evaluate": ("model", "dataset", "batch_size", "num_batches", "loss"),
    "train": ("model", "optimizer", "train_dataset", "args", "loss", "iterate_batches"),
    "grad_checkpoint": ("layer",),
}
TRAINING_ARGS_MEMORY_POLICY_FIELDS = (
    "grad_checkpoint",
    "grad_accumulation_steps",
    "clear_cache_threshold",
)
TRAINER_ADAPTER_SAVE_FIELDS = ("adapter_file",)
STRUCTURE_FIELD_NAMES = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
MLX_LM_DENSE_BATCH_KEYS = ("tokens", "lengths")
UNSUPPORTED_TRAINER_INTEGRATION_REASON = (
    "mlx-lm trainer integration is unsupported for cppmega_mlx models: "
    "the installed trainer accepts dense token batches and saves adapter "
    "weights, but cppmega_mlx batches carry attention masks, structure side "
    "channels, route metadata, and full-pretraining checkpoint state"
)

OffsetSpec = int | Sequence[int] | mx.array


@dataclass(frozen=True)
class MLXLMAPIInfo:
    """Small, serializable description of the installed MLX-LM trainer surface."""

    available: bool
    module: str = TRAINER_MODULE
    module_file: str | None = None
    api_signatures: dict[str, str] | None = None
    missing_apis: tuple[str, ...] = ()
    incompatible_apis: tuple[str, ...] = ()
    compatibility_errors: tuple[str, ...] = ()
    package_versions: dict[str, str] | None = None
    error: str | None = None


@dataclass(frozen=True)
class MLXLMBatchRouteMetadata:
    """Route and side-channel facts that must stay outside MLX-LM loss args."""

    token_shape: tuple[int, int]
    token_dtype: str
    dropped_fields: tuple[str, ...]
    structure_fields: tuple[str, ...]
    route_symbols: tuple[str, ...] = ()
    route_roles: tuple[str, ...] = ()
    has_attention_mask: bool = False


class MLXLMTrainerIntegrationUnsupported(NotImplementedError):
    """Raised when callers try to use MLX-LM as the cppmega trainer."""


def _load_trainer_module() -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(TRAINER_MODULE), None
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch tests.
        return None, f"{type(exc).__name__}: {exc}"


def _signature_for(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("mlx", "mlx-lm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "<not installed>"
    return versions


def _missing_required_parameters(name: str, obj: Any) -> tuple[str, ...]:
    try:
        parameters = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return REQUIRED_TRAINER_PARAMETERS[name]
    return tuple(
        parameter
        for parameter in REQUIRED_TRAINER_PARAMETERS[name]
        if parameter not in parameters
    )


def describe_mlx_lm_trainer_apis() -> MLXLMAPIInfo:
    """Inspect the installed MLX-LM trainer without making it a hard dependency."""

    package_versions = _package_versions()
    trainer, error = _load_trainer_module()
    if trainer is None:
        return MLXLMAPIInfo(
            available=False,
            api_signatures={},
            missing_apis=TRAINER_API_NAMES,
            package_versions=package_versions,
            error=error,
        )

    signatures: dict[str, str] = {}
    missing: list[str] = []
    incompatible: list[str] = []
    compatibility_errors: list[str] = []
    for name in TRAINER_API_NAMES:
        obj = getattr(trainer, name, None)
        if obj is None:
            missing.append(name)
        else:
            signatures[name] = _signature_for(obj)
            missing_parameters = _missing_required_parameters(name, obj)
            if missing_parameters:
                incompatible.append(name)
                compatibility_errors.append(
                    f"{name} missing required parameter(s): "
                    + ", ".join(missing_parameters)
                )

    return MLXLMAPIInfo(
        available=not missing and not incompatible,
        module_file=getattr(trainer, "__file__", None),
        api_signatures=signatures,
        missing_apis=tuple(missing),
        incompatible_apis=tuple(incompatible),
        compatibility_errors=tuple(compatibility_errors),
        package_versions=package_versions,
        error=None,
    )


def _offset_vector(offset: OffsetSpec, batch_size: int) -> mx.array:
    if isinstance(offset, int):
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return mx.broadcast_to(mx.array(offset, dtype=mx.int32), (batch_size,))

    if isinstance(offset, mx.array):
        offsets = offset
    else:
        offsets = mx.array(list(offset), dtype=mx.int32)

    if offsets.shape != (batch_size,):
        raise ValueError(f"offset must be a scalar or shaped ({batch_size},)")
    if bool(mx.any(offsets < 0).item()):
        raise ValueError("offset must be non-negative")
    return offsets.astype(mx.int32)


def _sequence_lengths(batch_size: int, seq_len: int) -> mx.array:
    return mx.broadcast_to(mx.array(seq_len, dtype=mx.int32), (batch_size,))


def _validate_mlx_lm_tokens(tokens: mx.array) -> None:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be shaped (B, S), got {tokens.shape}")
    if tokens.shape[1] < 2:
        raise ValueError("tokens sequence length must be at least 2")
    if tokens.dtype != mx.int32:
        raise TypeError(
            "MLX-LM dense token mapping requires int32 tokens; "
            f"got {tokens.dtype}"
        )


def _dtype_name(value: mx.array) -> str:
    return str(value.dtype).removeprefix("mlx.core.")


def as_mlx_lm_token_mapping(
    batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    *,
    offset: OffsetSpec = 0,
) -> dict[str, mx.array]:
    """Return a minimal MLX-LM-like token batch mapping.

    MLX-LM's trainer passes (batch, lengths) to default_loss where
    batch is a dense token matrix and lengths stores (offset, length)
    per row.  cppmega keeps richer side channels in LMTokenBatch; this
    adapter intentionally exports only the common token/length subset.  It does
    not make cppmega models or side-channel batches drop-in MLX-LM trainer
    inputs.
    """

    lm_batch = ensure_lm_batch(batch)
    _validate_mlx_lm_tokens(lm_batch.tokens)
    batch_size, seq_len = lm_batch.tokens.shape
    lengths = mx.stack(
        [_offset_vector(offset, batch_size), _sequence_lengths(batch_size, seq_len)],
        axis=1,
    )
    return {
        "tokens": lm_batch.tokens,
        "lengths": lengths,
    }


def as_mlx_lm_loss_args(
    batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    *,
    offset: OffsetSpec = 0,
) -> tuple[mx.array, mx.array]:
    """Return (tokens, lengths) for probes against MLX-LM default_loss."""

    mapping = as_mlx_lm_token_mapping(batch, offset=offset)
    return mapping["tokens"], mapping["lengths"]


def require_supported_mlx_lm_trainer_integration(
    batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    *,
    model: Any | None = None,
) -> None:
    """Fail closed for full MLX-LM trainer integration attempts.

    The adapter supports only dense token/loss-argument conversion for API
    probes and documentation receipts.  Full mlx_lm.tuner.trainer.train use
    would drop cppmega masks, structure side channels, route semantics, and
    checkpoint state, so callers must stay on the repo-local trainer until that
    boundary is explicitly implemented and tested.
    """

    metadata = describe_mlx_lm_batch_route_metadata(batch, model=model)
    dropped = ", ".join(metadata.dropped_fields) or "none"
    routes = "".join(metadata.route_symbols) or "unknown"
    raise MLXLMTrainerIntegrationUnsupported(
        f"{UNSUPPORTED_TRAINER_INTEGRATION_REASON}; "
        f"dropped_fields={dropped}; route_symbols={routes}"
    )


def _route_tuple(value: Any, *, split_string: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value) if split_string else (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def describe_mlx_lm_batch_route_metadata(
    batch: LMTokenBatch | Mapping[str, Any] | mx.array,
    *,
    model: Any | None = None,
    route_symbols: Sequence[str] | str | None = None,
    route_roles: Sequence[str] | str | None = None,
) -> MLXLMBatchRouteMetadata:
    """Describe local cppmega metadata that MLX-LM dense loss args cannot carry."""

    lm_batch = ensure_lm_batch(batch)
    structure_fields = tuple(
        name for name, value in lm_batch.structure_fields().items() if value is not None
    )
    dropped_fields = (
        ("attention_mask",) if lm_batch.attention_mask is not None else ()
    ) + structure_fields

    if model is not None:
        if route_symbols is None:
            route_symbols = getattr(model, "route_symbols", None)
        if route_roles is None:
            route_roles = getattr(model, "route_roles", None)

    return MLXLMBatchRouteMetadata(
        token_shape=(int(lm_batch.tokens.shape[0]), int(lm_batch.tokens.shape[1])),
        token_dtype=_dtype_name(lm_batch.tokens),
        dropped_fields=dropped_fields,
        structure_fields=structure_fields,
        route_symbols=_route_tuple(route_symbols, split_string=True),
        route_roles=_route_tuple(route_roles),
        has_attention_mask=lm_batch.attention_mask is not None,
    )


__all__ = [
    "MLX_LM_DENSE_BATCH_KEYS",
    "MLXLMAPIInfo",
    "MLXLMBatchRouteMetadata",
    "MLXLMTrainerIntegrationUnsupported",
    "REQUIRED_TRAINER_PARAMETERS",
    "STRUCTURE_FIELD_NAMES",
    "TRAINER_ADAPTER_SAVE_FIELDS",
    "TRAINER_API_NAMES",
    "TRAINING_ARGS_MEMORY_POLICY_FIELDS",
    "TRAINER_MODULE",
    "UNSUPPORTED_TRAINER_INTEGRATION_REASON",
    "as_mlx_lm_loss_args",
    "as_mlx_lm_token_mapping",
    "describe_mlx_lm_batch_route_metadata",
    "describe_mlx_lm_trainer_apis",
    "require_supported_mlx_lm_trainer_integration",
]
