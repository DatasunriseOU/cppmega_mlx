"""Optional PyTorch DataLoader bridge for local token batches.

This module is safe to import from the MLX data package: it does not import
PyTorch until ``build_spawn_dataloader`` is called explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import importlib.util
from typing import Any

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.batch import LMTokenBatch

_BATCH_KEYS = (
    "tokens",
    "attention_mask",
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
_STRUCTURE_KEYS = tuple(key for key in _BATCH_KEYS if key not in {"tokens", "attention_mask"})


class TorchDataLoaderBridgeError(RuntimeError):
    """Raised when the optional PyTorch DataLoader bridge cannot be built."""


@dataclass(frozen=True)
class TorchDataLoaderBridgeConfig:
    """Configuration for the optional spawn-only PyTorch DataLoader bridge."""

    num_workers: int = 0
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    multiprocessing_context: str | Any | None = None
    pin_memory: bool = False


class LocalTokenBatchDataset:
    """Map-style dataset of already-local token batches.

    The dataset materializes each input batch as NumPy arrays so it is safe for
    PyTorch's spawn workers to pickle. It is intentionally for local handoff
    batches, not a scale replacement for the repository's real dataset readers.
    """

    def __init__(self, batches: Iterable[LMTokenBatch | Mapping[str, Any] | mx.array]):
        self._batches = tuple(_numpy_batch(batch) for batch in batches)
        if not self._batches:
            raise ValueError("batches must contain at least one local token batch")

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> Mapping[str, np.ndarray]:
        return self._batches[index]


def is_torch_dataloader_available() -> bool:
    """Return whether the optional PyTorch dependency can be imported."""

    return importlib.util.find_spec("torch") is not None


def build_spawn_dataloader(
    batches: Iterable[LMTokenBatch | Mapping[str, Any] | mx.array],
    *,
    config: TorchDataLoaderBridgeConfig | None = None,
    num_workers: int | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | Any | None = None,
    pin_memory: bool | None = None,
) -> Any:
    """Build a PyTorch DataLoader over local token batches.

    ``torch`` is imported lazily inside this function. When workers are enabled,
    the multiprocessing context is forced to ``spawn`` so callers cannot fork
    after MLX/Metal state has been initialized.
    """

    resolved = _resolve_config(
        config=config,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        pin_memory=pin_memory,
    )
    _validate_spawn_config(resolved)

    dataset = LocalTokenBatchDataset(batches)
    DataLoader = _load_torch_dataloader()

    kwargs: dict[str, Any] = {
        "batch_size": None,
        "collate_fn": _identity_collate,
        "num_workers": resolved.num_workers,
        "persistent_workers": (
            bool(resolved.persistent_workers)
            if resolved.persistent_workers is not None
            else resolved.num_workers > 0
        ),
        "pin_memory": resolved.pin_memory,
    }
    if resolved.num_workers > 0:
        kwargs["multiprocessing_context"] = (
            resolved.multiprocessing_context
            if resolved.multiprocessing_context is not None
            else "spawn"
        )
        if resolved.prefetch_factor is not None:
            kwargs["prefetch_factor"] = resolved.prefetch_factor
    return DataLoader(dataset, **kwargs)


def iter_mlx_batches(loader: Iterable[Mapping[str, Any]]) -> Iterator[LMTokenBatch]:
    """Yield MLX ``LMTokenBatch`` objects from bridge DataLoader outputs."""

    for batch in loader:
        yield _mlx_batch(batch)


def _resolve_config(
    *,
    config: TorchDataLoaderBridgeConfig | None,
    num_workers: int | None,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
    multiprocessing_context: str | Any | None,
    pin_memory: bool | None,
) -> TorchDataLoaderBridgeConfig:
    base = config or TorchDataLoaderBridgeConfig()
    return TorchDataLoaderBridgeConfig(
        num_workers=base.num_workers if num_workers is None else num_workers,
        persistent_workers=(
            base.persistent_workers
            if persistent_workers is None
            else persistent_workers
        ),
        prefetch_factor=(
            base.prefetch_factor if prefetch_factor is None else prefetch_factor
        ),
        multiprocessing_context=(
            base.multiprocessing_context
            if multiprocessing_context is None
            else multiprocessing_context
        ),
        pin_memory=base.pin_memory if pin_memory is None else pin_memory,
    )


def _validate_spawn_config(config: TorchDataLoaderBridgeConfig) -> None:
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.persistent_workers and config.num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if config.prefetch_factor is not None and config.num_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")
    if config.prefetch_factor is not None and config.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if config.multiprocessing_context is not None:
        start_method = _context_start_method(config.multiprocessing_context)
        if start_method != "spawn":
            raise ValueError(
                "PyTorch DataLoader multiprocessing_context must be 'spawn'; "
                f"got {start_method!r}"
            )


def _context_start_method(context: str | Any) -> str:
    if isinstance(context, str):
        return context
    get_start_method = getattr(context, "get_start_method", None)
    if callable(get_start_method):
        return str(get_start_method())
    raise ValueError(
        "multiprocessing_context must be 'spawn' or a context object with "
        "get_start_method()"
    )


def _load_torch_dataloader() -> Any:
    if not is_torch_dataloader_available():
        raise TorchDataLoaderBridgeError(
            "PyTorch DataLoader bridge requires optional dependency 'torch'. "
            "Install torch explicitly or keep using native MLX iterators."
        )
    from torch.utils.data import DataLoader

    return DataLoader


def _identity_collate(sample: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return sample


def _numpy_batch(batch: LMTokenBatch | Mapping[str, Any] | mx.array) -> Mapping[str, np.ndarray]:
    if isinstance(batch, LMTokenBatch):
        return _numpy_mapping(batch.as_dict())
    if isinstance(batch, mx.array):
        return _numpy_mapping({"tokens": batch})
    if isinstance(batch, Mapping):
        return _numpy_mapping(batch)
    raise TypeError(f"unsupported token batch type: {type(batch)!r}")


def _numpy_mapping(batch: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    keys = set(batch)
    unknown = sorted(keys - set(_BATCH_KEYS))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unsupported DataLoader bridge batch keys: {joined}")
    if "tokens" not in batch:
        raise ValueError("DataLoader bridge batches must include 'tokens'")

    arrays = {
        key: _as_numpy_array(key, batch[key])
        for key in _BATCH_KEYS
        if key in batch and batch[key] is not None
    }
    tokens = arrays["tokens"]
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be shaped (B, S), got {tokens.shape}")
    if tokens.shape[1] < 2:
        raise ValueError("tokens sequence length must be at least 2")
    for key in _BATCH_KEYS:
        if key == "tokens" or key not in arrays:
            continue
        if arrays[key].shape != tokens.shape:
            raise ValueError(
                f"{key} must match tokens shape {tokens.shape}, got {arrays[key].shape}"
            )
    return arrays


def _as_numpy_array(key: str, value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    if key == "attention_mask":
        return array.astype(np.float32, copy=False)
    _require_integer_array(key, array)
    if np.any(array < 0):
        raise ValueError(f"{key} IDs must be non-negative")
    if np.any(array > np.iinfo(np.int32).max):
        raise ValueError(f"{key} IDs exceed int32 range")
    if key == "tokens":
        return array.astype(np.int32, copy=False)
    if key in _STRUCTURE_KEYS:
        return array.astype(np.int32, copy=False)
    raise ValueError(f"unsupported DataLoader bridge batch key: {key}")


def _require_integer_array(key: str, value: np.ndarray) -> None:
    if value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{key} IDs must use an integer dtype")


def _mlx_batch(batch: Mapping[str, Any]) -> LMTokenBatch:
    arrays = _numpy_mapping(batch)
    kwargs = {
        key: mx.array(arrays[key])
        for key in _BATCH_KEYS
        if key != "tokens" and key in arrays
    }
    return LMTokenBatch(tokens=mx.array(arrays["tokens"]), **kwargs)


__all__ = [
    "LocalTokenBatchDataset",
    "TorchDataLoaderBridgeConfig",
    "TorchDataLoaderBridgeError",
    "build_spawn_dataloader",
    "is_torch_dataloader_available",
    "iter_mlx_batches",
]
