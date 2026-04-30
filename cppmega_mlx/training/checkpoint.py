"""Small safetensors checkpoint helpers for MLX modules."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


WEIGHTS_NAME = "model.safetensors"
OPTIMIZER_NAME = "optimizer.safetensors"
GRAD_ACCUM_NAME = "gradient_accumulator.safetensors"
METADATA_NAME = "metadata.json"
SHARD_INDEX_NAME = "model.safetensors.index.json"
FORMAT_NAME = "cppmega_mlx_checkpoint_v1"
FORMAT_VERSION = 1
RNG_MODE_NOT_SAVED = "not_saved"
RNG_MODE_SEED = "seed"
SHARDING_MODE_SINGLE_FILE = "single_file"

_STANDALONE_RNG_KEYS = {
    "rng_state",
    "random_state",
    "mx_random_state",
    "numpy_random_state",
    "python_random_state",
}
_RNG_PAYLOAD_KEYS = {
    "state",
    "payload",
    "mx_state",
    "numpy_state",
    "python_state",
}
_SHARDING_PAYLOAD_KEYS = {
    "shards",
    "weight_map",
    "index_file",
    "shard_index",
    "max_file_size_gb",
}


def _checkpoint_paths(path: str | Path) -> tuple[Path, Path, Path | None]:
    base = Path(path)
    if base.suffix == ".safetensors":
        return base, base.with_suffix(".json"), None
    return base / WEIGHTS_NAME, base / METADATA_NAME, base / OPTIMIZER_NAME


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("cppmega-mlx", "mlx", "mlx-lm", "safetensors", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _tokenizer_contract(
    model: nn.Module,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tokenizer = metadata.get("tokenizer") or metadata.get("tokenizer_contract")
    contract: dict[str, Any] = _jsonable(tokenizer) if isinstance(tokenizer, dict) else {}

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if "vocab_size" not in contract and vocab_size is not None:
        contract["vocab_size"] = vocab_size
    max_seq_length = getattr(config, "max_seq_length", None)
    if "max_seq_length" not in contract and max_seq_length is not None:
        contract["max_seq_length"] = max_seq_length
    structure_vocab_size = getattr(config, "structure_vocab_size", None)
    if "structure_vocab_size" not in contract and structure_vocab_size is not None:
        contract["structure_vocab_size"] = structure_vocab_size

    for field in ("tokenizer_path", "tokenizer_name", "bos_token_id", "eos_token_id", "pad_token_id"):
        if field in metadata and field not in contract:
            contract[field] = _jsonable(metadata[field])

    return contract


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    flat = dict(tree_flatten(state))
    return {
        "num_tensors": len(flat),
        "tensors": sorted(flat.keys()),
    }


def _require_non_negative_int(
    value: Any,
    *,
    name: str,
    metadata_path: Path,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: {name} must be a non-negative integer"
        )


def _require_positive_int(
    value: Any,
    *,
    name: str,
    metadata_path: Path,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: {name} must be a positive integer"
        )


def _require_string(
    value: Any,
    *,
    name: str,
    metadata_path: Path,
) -> None:
    if not isinstance(value, str):
        raise ValueError(f"checkpoint metadata {metadata_path}: {name} must be a string")


def _reject_standalone_rng_payload(metadata: dict[str, Any], metadata_path: Path) -> None:
    for key in sorted(_STANDALONE_RNG_KEYS):
        if key in metadata:
            raise ValueError(
                f"checkpoint metadata {metadata_path}: standalone RNG payloads are "
                "not supported; use rng.mode='seed' for seed provenance or omit rng"
            )


def _rng_contract(metadata: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    _reject_standalone_rng_payload(metadata, metadata_path)
    raw_rng = metadata.get("rng")
    if raw_rng is None:
        return {"mode": RNG_MODE_NOT_SAVED}
    if not isinstance(raw_rng, dict):
        raise ValueError(f"checkpoint metadata {metadata_path}: rng must be an object")

    payload_keys = sorted(set(raw_rng) & _RNG_PAYLOAD_KEYS)
    if payload_keys:
        keys = ", ".join(payload_keys)
        raise ValueError(
            f"checkpoint metadata {metadata_path}: standalone RNG payloads are "
            f"not supported ({keys}); use rng.mode='seed' for seed provenance"
        )

    allowed_keys = {"mode", "seed", "source", "note"}
    unknown_keys = sorted(set(raw_rng) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: unsupported rng fields "
            f"{unknown_keys!r}; supported fields are {sorted(allowed_keys)!r}"
        )

    mode = raw_rng.get("mode", RNG_MODE_NOT_SAVED)
    if mode not in {RNG_MODE_NOT_SAVED, RNG_MODE_SEED}:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: unsupported rng.mode "
            f"{mode!r}; expected {RNG_MODE_NOT_SAVED!r} or {RNG_MODE_SEED!r}"
        )

    contract: dict[str, Any] = {"mode": mode}
    if mode == RNG_MODE_SEED:
        if "seed" not in raw_rng:
            raise ValueError(
                f"checkpoint metadata {metadata_path}: rng.seed is required "
                "when rng.mode='seed'"
            )
        _require_non_negative_int(raw_rng["seed"], name="rng.seed", metadata_path=metadata_path)
        contract["seed"] = raw_rng["seed"]
    elif "seed" in raw_rng:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: rng.seed requires rng.mode='seed'"
        )

    for field in ("source", "note"):
        if field in raw_rng:
            _require_string(raw_rng[field], name=f"rng.{field}", metadata_path=metadata_path)
            contract[field] = raw_rng[field]
    return contract


def _sharding_contract(
    metadata: dict[str, Any],
    metadata_path: Path,
    *,
    weights_name: str,
) -> dict[str, Any]:
    raw_sharding = metadata.get("sharding")
    if raw_sharding is None:
        return {
            "mode": SHARDING_MODE_SINGLE_FILE,
            "num_shards": 1,
            "weights": [weights_name],
            "index": None,
        }
    if not isinstance(raw_sharding, dict):
        raise ValueError(
            f"checkpoint metadata {metadata_path}: sharding must be an object"
        )

    payload_keys = sorted(set(raw_sharding) & _SHARDING_PAYLOAD_KEYS)
    if payload_keys:
        keys = ", ".join(payload_keys)
        raise ValueError(
            f"checkpoint metadata {metadata_path}: checkpoint sharding payloads "
            f"are not supported yet ({keys}); only single-file checkpoints are supported"
        )

    allowed_keys = {"mode", "num_shards", "weights", "index", "source", "note"}
    unknown_keys = sorted(set(raw_sharding) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: unsupported sharding fields "
            f"{unknown_keys!r}; supported fields are {sorted(allowed_keys)!r}"
        )

    mode = raw_sharding.get("mode", SHARDING_MODE_SINGLE_FILE)
    if mode != SHARDING_MODE_SINGLE_FILE:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: unsupported sharding.mode "
            f"{mode!r}; expected {SHARDING_MODE_SINGLE_FILE!r}"
        )

    num_shards = raw_sharding.get("num_shards", 1)
    _require_positive_int(num_shards, name="sharding.num_shards", metadata_path=metadata_path)
    if num_shards != 1:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: sharding.num_shards must be 1 "
            "for single-file checkpoints"
        )

    weights = raw_sharding.get("weights", [weights_name])
    if (
        not isinstance(weights, list)
        or len(weights) != 1
        or not all(isinstance(name, str) for name in weights)
    ):
        raise ValueError(
            f"checkpoint metadata {metadata_path}: sharding.weights must be a "
            "single-item string list for single-file checkpoints"
        )
    if weights != [weights_name]:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: sharding.weights must be "
            f"{[weights_name]!r} for this checkpoint"
        )

    index = raw_sharding.get("index")
    if index is not None:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: sharding.index must be null "
            "for single-file checkpoints"
        )

    contract: dict[str, Any] = {
        "mode": SHARDING_MODE_SINGLE_FILE,
        "num_shards": 1,
        "weights": [weights_name],
        "index": None,
    }
    for field in ("source", "note"):
        if field in raw_sharding:
            _require_string(
                raw_sharding[field],
                name=f"sharding.{field}",
                metadata_path=metadata_path,
            )
            contract[field] = raw_sharding[field]
    return contract


def _validate_checkpoint_metadata(payload: dict[str, Any], metadata_path: Path) -> None:
    if not payload:
        return

    checkpoint_format = payload.get("format")
    if checkpoint_format != FORMAT_NAME:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: unsupported format "
            f"{checkpoint_format!r}; expected {FORMAT_NAME!r}"
        )

    version = payload.get("version", FORMAT_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint metadata {metadata_path}: unsupported version "
            f"{version!r}; expected {FORMAT_VERSION}"
        )

    tokenizer_contract = payload.get("tokenizer_contract")
    if not isinstance(tokenizer_contract, dict):
        raise ValueError(
            f"checkpoint metadata {metadata_path}: tokenizer_contract must be an object"
        )
    for field in ("vocab_size", "max_seq_length", "structure_vocab_size"):
        if field in tokenizer_contract:
            _require_non_negative_int(
                tokenizer_contract[field],
                name=f"tokenizer_contract.{field}",
                metadata_path=metadata_path,
            )
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = tokenizer_contract.get(field)
        if value is not None:
            _require_non_negative_int(
                value,
                name=f"tokenizer_contract.{field}",
                metadata_path=metadata_path,
            )

    _rng_contract(payload, metadata_path)

    weights_name = payload.get("weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise ValueError(f"checkpoint metadata {metadata_path}: weights must be a string")
    _sharding_contract(payload, metadata_path, weights_name=weights_name)

    batch_cursor = payload.get("batch_cursor")
    if batch_cursor is not None:
        if not isinstance(batch_cursor, dict):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: batch_cursor must be an object"
            )
        for field in ("epoch", "batch_offset", "global_batch_offset"):
            if field not in batch_cursor:
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: batch_cursor.{field} is required"
                )
            _require_non_negative_int(
                batch_cursor[field],
                name=f"batch_cursor.{field}",
                metadata_path=metadata_path,
            )

    training_state = payload.get("training_state")
    if training_state is not None:
        if not isinstance(training_state, dict):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: training_state must be an object"
            )
        cursor = training_state.get("state")
        if not isinstance(cursor, dict):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: training_state.state must be an object"
            )
        for field in ("step", "trained_tokens"):
            if field not in cursor:
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: training_state.state.{field} is required"
                )
            _require_non_negative_int(
                cursor[field],
                name=f"training_state.state.{field}",
                metadata_path=metadata_path,
            )

        grad_accum_steps = training_state.get("grad_accum_steps")
        _require_positive_int(
            grad_accum_steps,
            name="training_state.grad_accum_steps",
            metadata_path=metadata_path,
        )
        assert isinstance(grad_accum_steps, int)
        pending_microbatches = training_state.get("pending_microbatches")
        _require_non_negative_int(
            pending_microbatches,
            name="training_state.pending_microbatches",
            metadata_path=metadata_path,
        )
        assert isinstance(pending_microbatches, int)
        if pending_microbatches >= grad_accum_steps:
            raise ValueError(
                f"checkpoint metadata {metadata_path}: "
                "training_state.pending_microbatches must be less than "
                "training_state.grad_accum_steps"
            )

        accumulator_present = training_state.get("gradient_accumulator_present")
        if not isinstance(accumulator_present, bool):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: "
                "training_state.gradient_accumulator_present must be a boolean"
            )
        if pending_microbatches > 0 and not accumulator_present:
            raise ValueError(
                f"checkpoint metadata {metadata_path}: pending training_state "
                "requires a gradient accumulator"
            )

        accumulator = training_state.get("gradient_accumulator")
        if accumulator is not None:
            if not isinstance(accumulator, dict):
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: "
                    "training_state.gradient_accumulator must be an object"
                )
            accumulator_file = accumulator.get("file")
            accumulator_file_present = bool(accumulator.get("present"))
            if accumulator_file_present != accumulator_present:
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: "
                    "training_state gradient accumulator presence mismatch"
                )
            if accumulator_file_present and not isinstance(accumulator_file, str):
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: "
                    "training_state.gradient_accumulator.file must be a string"
                )
            _require_non_negative_int(
                accumulator.get("num_tensors", 0),
                name="training_state.gradient_accumulator.num_tensors",
                metadata_path=metadata_path,
            )
            tensors = accumulator.get("tensors", [])
            if not isinstance(tensors, list) or not all(
                isinstance(name, str) for name in tensors
            ):
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: "
                    "training_state.gradient_accumulator.tensors must be a string list"
                )

    optimizer_metadata = payload.get("optimizer")
    if optimizer_metadata is not None:
        if not isinstance(optimizer_metadata, dict):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: optimizer must be an object"
            )
        present = optimizer_metadata.get("present")
        if not isinstance(present, bool):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: optimizer.present must be a boolean"
            )
        optimizer_file = optimizer_metadata.get("file")
        if present:
            if not isinstance(optimizer_file, str) or not optimizer_file:
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: optimizer.file must be a string"
                )
        elif optimizer_file is not None:
            raise ValueError(
                f"checkpoint metadata {metadata_path}: optimizer.file requires optimizer.present=true"
            )
        _require_non_negative_int(
            optimizer_metadata.get("num_tensors", 0),
            name="optimizer.num_tensors",
            metadata_path=metadata_path,
        )
        tensors = optimizer_metadata.get("tensors", [])
        if not isinstance(tensors, list) or not all(
            isinstance(name, str) for name in tensors
        ):
            raise ValueError(
                f"checkpoint metadata {metadata_path}: optimizer.tensors must be a string list"
            )


def _training_step_state(training_step: Any | None) -> dict[str, Any] | None:
    if training_step is None:
        return None
    state_dict = getattr(training_step, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("training_step must expose state_dict()")
    state = state_dict()
    if not isinstance(state, dict):
        raise TypeError("training_step.state_dict() must return a dict")
    return _jsonable(state)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    optimizer: optim.Optimizer | None = None,
    training_step: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save full model weights and optional optimizer state for resume."""

    metadata = metadata or {}
    weights_path, metadata_path, optimizer_path = _checkpoint_paths(path)
    rng_contract = _rng_contract(metadata, metadata_path)
    sharding_contract = _sharding_contract(
        metadata,
        metadata_path,
        weights_name=weights_path.name,
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(weights_path), weights, metadata={"format": "mlx"})

    optimizer_state_present = False
    optimizer_summary: dict[str, Any] | None = None
    if optimizer is not None and optimizer_path is not None:
        optimizer_state_present = True
        mx.eval(optimizer.state)
        optimizer_state = dict(tree_flatten(optimizer.state))
        mx.save_safetensors(
            str(optimizer_path),
            optimizer_state,
            metadata={"format": "mlx"},
        )
        optimizer_summary = _state_summary(optimizer.state)

    training_state = _training_step_state(training_step)
    grad_accum_summary: dict[str, Any] | None = None
    grad_accum_file: str | None = None
    if training_state is not None:
        grad_accum = getattr(training_step, "gradient_accumulator", None)
        if grad_accum is not None:
            if optimizer_path is None:
                raise ValueError(
                    "gradient accumulator checkpoints require a directory checkpoint"
                )
            grad_accum_path = weights_path.parent / GRAD_ACCUM_NAME
            mx.eval(grad_accum)
            grad_accum_state = dict(tree_flatten(grad_accum))
            mx.save_safetensors(
                str(grad_accum_path),
                grad_accum_state,
                metadata={"format": "mlx"},
            )
            grad_accum_summary = _state_summary(grad_accum)
            grad_accum_file = GRAD_ACCUM_NAME
        training_state["gradient_accumulator"] = {
            "present": grad_accum_summary is not None,
            "file": grad_accum_file,
            "num_tensors": grad_accum_summary["num_tensors"] if grad_accum_summary else 0,
            "tensors": grad_accum_summary["tensors"] if grad_accum_summary else [],
        }

    payload: dict[str, Any] = _jsonable(metadata) if metadata else {}
    payload.update({
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "weights": weights_path.name,
        "num_tensors": len(weights),
        "step": metadata.get("step"),
        "optimizer": {
            "present": optimizer_state_present,
            "file": optimizer_path.name if optimizer_state_present and optimizer_path else None,
            "num_tensors": optimizer_summary["num_tensors"] if optimizer_summary else 0,
            "tensors": optimizer_summary["tensors"] if optimizer_summary else [],
        },
        "package_versions": _package_versions(),
        "tokenizer_contract": _tokenizer_contract(model, metadata),
        "rng": rng_contract,
        "sharding": sharding_contract,
    })
    if training_state is not None:
        payload["training_state"] = training_state
    if hasattr(model, "config"):
        payload["model_config"] = _jsonable(model.config)
    _validate_checkpoint_metadata(payload, metadata_path)

    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    optimizer: optim.Optimizer | None = None,
    training_step: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` and optional ``optimizer``."""

    weights_path, metadata_path, default_optimizer_path = _checkpoint_paths(path)
    payload = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint metadata {metadata_path}: expected a JSON object")
    _validate_checkpoint_metadata(payload, metadata_path)

    if payload:
        weights_name = payload.get("weights")
        if weights_name != weights_path.name:
            raise ValueError(
                f"checkpoint metadata {metadata_path}: weights {weights_name!r} "
                f"does not match expected file {weights_path.name!r}"
            )
    shard_index_path = weights_path.parent / SHARD_INDEX_NAME
    if not weights_path.exists() and shard_index_path.exists():
        raise ValueError(
            f"checkpoint {path}: sharded checkpoint layout "
            f"{SHARD_INDEX_NAME!r} is not supported; expected single file "
            f"{weights_path.name!r}"
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No model weights found for checkpoint {path}: {weights_path}"
        )

    model.load_weights(str(weights_path), strict=strict)
    mx.eval(model.parameters())

    if optimizer is not None:
        optimizer_metadata = payload.get("optimizer")
        optimizer_file = (
            optimizer_metadata.get("file")
            if isinstance(optimizer_metadata, dict)
            else None
        )
        optimizer_path = (
            metadata_path.parent / optimizer_file
            if isinstance(optimizer_file, str) and optimizer_file
            else default_optimizer_path
        )
        if optimizer_path is None or not optimizer_path.exists():
            raise FileNotFoundError(f"No optimizer state found for checkpoint {path}")
        optimizer_state = mx.load(str(optimizer_path))
        if not isinstance(optimizer_state, dict):
            raise TypeError(f"Optimizer checkpoint must be a tensor mapping: {optimizer_path}")
        optimizer.state = tree_unflatten(optimizer_state)
        mx.eval(optimizer.state)

    if training_step is not None:
        training_state = payload.get("training_state")
        if not isinstance(training_state, dict):
            raise ValueError(f"No training_state found for checkpoint {path}")
        accumulator_metadata = training_state.get("gradient_accumulator")
        accumulator = None
        if isinstance(accumulator_metadata, dict) and accumulator_metadata.get("present"):
            accumulator_file = accumulator_metadata.get("file")
            if not isinstance(accumulator_file, str) or not accumulator_file:
                raise ValueError(
                    f"checkpoint metadata {metadata_path}: "
                    "training_state.gradient_accumulator.file must be a string"
                )
            accumulator_path = metadata_path.parent / accumulator_file
            if not accumulator_path.exists():
                raise FileNotFoundError(
                    f"No gradient accumulator state found for checkpoint {path}"
                )
            accumulator_state = mx.load(str(accumulator_path))
            if not isinstance(accumulator_state, dict):
                raise TypeError(
                    "Gradient accumulator checkpoint must be a tensor mapping: "
                    f"{accumulator_path}"
                )
            accumulator = tree_unflatten(accumulator_state)
            mx.eval(accumulator)

        load_state_dict = getattr(training_step, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError("training_step must expose load_state_dict()")
        load_state_dict(training_state, gradient_accumulator=accumulator)

    return payload


__all__ = [
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
]
