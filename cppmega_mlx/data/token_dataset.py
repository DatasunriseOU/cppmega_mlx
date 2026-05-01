"""Fixed-shape token datasets for local MLX training.

The H200 cppmega pipeline stores tokenized C/C++ documents as parquet and then
formats them into Megatron .bin/.idx files.  The Mac lane starts from a
smaller NPZ handoff: it keeps the same token IDs and tokenizer metadata, but
emits fixed-shape :class:LMTokenBatch objects directly for MLX.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

import mlx.core as mx
import numpy as np
from numpy.lib.npyio import NpzFile

from cppmega_mlx.config.model import (
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
)
from cppmega_mlx.data.batch import LMTokenBatch

TokenizerContract = Literal["megacpp", "local_profile", "custom"]
TokenDatasetFormat = Literal["npz", "parquet", "megatron"]


class TokenNpzDatasetOptions(TypedDict, total=False):
    token_key: str
    shuffle: bool
    seed: int
    loop: bool
    resume_batch: int
    metadata: TokenDatasetMetadata | None


class TokenParquetDatasetOptions(TokenNpzDatasetOptions, total=False):
    text_key: str | None
    tokenizer: NotRequired[Any | None]
    eos_token_id: int | None


class TokenBatchDataset(Protocol):
    path: Path
    seq_len: int
    batch_size: int
    token_key: str
    shuffle: bool
    loop: bool
    metadata: "TokenDatasetMetadata"

    @property
    def num_samples(self) -> int: ...

    @property
    def num_batches(self) -> int: ...

    @property
    def dropped_samples(self) -> int: ...

    def iter_batches(
        self,
        *,
        resume_batch: int | None = None,
        epoch: int = 0,
        loop: bool | None = None,
    ) -> Iterator[LMTokenBatch]: ...

    def cursor_after(self, consumed_batches: int, *, epoch: int = 0) -> "BatchCursor": ...

    def token_id_range(self) -> tuple[int, int]: ...


_SIDE_CHANNEL_KEYS = (
    "attention_mask",
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
_AMBIGUOUS_SIDECAR_KEYS = (
    "side_channels",
)
_UNSUPPORTED_NGRAM_SIDECAR_KEYS = (
    "ngram_ids",
    "ngram_hash",
    "ngram_hash_ids",
    "ngram_sidecar",
    "ngrams",
)


@dataclass(frozen=True)
class TokenDatasetMetadata:
    """Tokenizer/data contract carried with local token shards."""

    vocab_size: int = MEGACPP_TOKENIZER_VOCAB_SIZE
    tokenizer_contract: TokenizerContract = "megacpp"
    local_profile_vocab_size: int = LOCAL_PROFILE_VOCAB_SIZE
    megacpp_tokenizer_vocab_size: int = MEGACPP_TOKENIZER_VOCAB_SIZE
    source_format: str = "npz"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.local_profile_vocab_size <= 0:
            raise ValueError("local_profile_vocab_size must be positive")
        if self.megacpp_tokenizer_vocab_size <= 0:
            raise ValueError("megacpp_tokenizer_vocab_size must be positive")
        if self.tokenizer_contract not in {"megacpp", "local_profile", "custom"}:
            raise ValueError(
                f"unsupported tokenizer_contract={self.tokenizer_contract!r}"
            )

    @classmethod
    def from_npz(cls, data: NpzFile) -> "TokenDatasetMetadata":
        """Read optional scalar metadata from an NPZ file."""

        return cls(
            vocab_size=_npz_scalar_int(data, "vocab_size", MEGACPP_TOKENIZER_VOCAB_SIZE),
            tokenizer_contract=_npz_scalar_str(
                data, "tokenizer_contract", "megacpp"
            ),
            local_profile_vocab_size=_npz_scalar_int(
                data, "local_profile_vocab_size", LOCAL_PROFILE_VOCAB_SIZE
            ),
            megacpp_tokenizer_vocab_size=_npz_scalar_int(
                data, "megacpp_tokenizer_vocab_size", MEGACPP_TOKENIZER_VOCAB_SIZE
            ),
            source_format="npz",
        )


@dataclass(frozen=True)
class BatchCursor:
    """Resume cursor for deterministic fixed-shape batch iteration."""

    epoch: int
    batch_offset: int
    global_batch_offset: int


class TokenNpzDataset:
    """NPZ-backed fixed-shape token batch iterator.

    tokens is required in the NPZ.  A flat array is split into contiguous
    windows of seq_len.  A 2D array is treated as one or more documents and
    split row-wise into windows.  Optional side-channel arrays with matching
    shape are sliced alongside tokens and passed through to LMTokenBatch.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        batch_size: int,
        token_key: str = "tokens",
        shuffle: bool = False,
        seed: int = 0,
        loop: bool = False,
        resume_batch: int = 0,
        metadata: TokenDatasetMetadata | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if resume_batch < 0:
            raise ValueError("resume_batch must be non-negative")

        self.path = Path(path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_key = token_key
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.resume_batch = resume_batch

        with np.load(self.path, allow_pickle=False) as data:
            if token_key not in data:
                raise ValueError(f"NPZ file must contain a {token_key!r} array")
            _reject_unsupported_npz_sidecars(data)
            token_windows = _fixed_windows(np.asarray(data[token_key]), seq_len)
            side_channels = {
                key: _fixed_windows(np.asarray(data[key]), seq_len)
                for key in _SIDE_CHANNEL_KEYS
                if key in data
            }
            loaded_metadata = TokenDatasetMetadata.from_npz(data)

        if not len(token_windows):
            raise ValueError("token array does not contain a full fixed-shape sample")
        for key, value in side_channels.items():
            if value.shape != token_windows.shape:
                raise ValueError(
                    f"{key} windows must match tokens shape {token_windows.shape}, "
                    f"got {value.shape}"
                )

        self._tokens = _to_int32_token_ids(token_windows)
        self._side_channels = {
            key: _to_side_channel_values(key, value)
            for key, value in side_channels.items()
        }
        self.metadata = metadata if metadata is not None else loaded_metadata

    def __len__(self) -> int:
        return self.num_batches

    @property
    def num_samples(self) -> int:
        return int(self._tokens.shape[0])

    @property
    def num_batches(self) -> int:
        return self.num_samples // self.batch_size

    @property
    def dropped_samples(self) -> int:
        return self.num_samples - self.num_batches * self.batch_size

    def sample_order(self, *, epoch: int = 0) -> np.ndarray:
        """Return deterministic sample order for an epoch."""

        order = np.arange(self.num_samples, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(order)
        return order

    def iter_batches(
        self,
        *,
        resume_batch: int | None = None,
        epoch: int = 0,
        loop: bool | None = None,
    ) -> Iterator[LMTokenBatch]:
        """Yield fixed-shape LMTokenBatch objects.

        resume_batch counts full batches already consumed in this dataset
        epoch.  The same seed, epoch, and resume value reconstruct the same
        next batch after checkpoint restore.
        """

        effective_loop = self.loop if loop is None else loop
        start_batch = self.resume_batch if resume_batch is None else resume_batch
        if start_batch < 0:
            raise ValueError("resume_batch must be non-negative")

        current_epoch = epoch
        batch_offset = start_batch
        while True:
            if self.num_batches == 0:
                return
            if batch_offset >= self.num_batches:
                if not effective_loop:
                    return
                current_epoch += batch_offset // self.num_batches
                batch_offset = batch_offset % self.num_batches

            order = self.sample_order(epoch=current_epoch)
            for local_batch in range(batch_offset, self.num_batches):
                sample_idx = order[
                    local_batch * self.batch_size : (local_batch + 1) * self.batch_size
                ]
                yield self._make_batch(sample_idx)

            if not effective_loop:
                return
            current_epoch += 1
            batch_offset = 0

    def cursor_after(self, consumed_batches: int, *, epoch: int = 0) -> BatchCursor:
        """Return the deterministic cursor after consumed_batches."""

        if consumed_batches < 0:
            raise ValueError("consumed_batches must be non-negative")
        total_batches = self.resume_batch + consumed_batches
        if self.num_batches == 0:
            return BatchCursor(
                epoch=epoch,
                batch_offset=0,
                global_batch_offset=total_batches,
            )
        return BatchCursor(
            epoch=epoch + total_batches // self.num_batches,
            batch_offset=total_batches % self.num_batches,
            global_batch_offset=total_batches,
        )

    def token_id_range(self) -> tuple[int, int]:
        """Return the min/max token IDs present in the loaded fixed windows."""

        return int(self._tokens.min()), int(self._tokens.max())

    def _make_batch(self, sample_idx: np.ndarray) -> LMTokenBatch:
        kwargs = {
            key: mx.array(value[sample_idx])
            for key, value in self._side_channels.items()
        }
        return LMTokenBatch(tokens=mx.array(self._tokens[sample_idx]), **kwargs)


def open_token_dataset(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    format: TokenDatasetFormat | None = None,
    **kwargs: Any,
) -> TokenBatchDataset:
    """Open a local token dataset.

    Parquet and Megatron readers are optional seams loaded only when selected.
    Megatron .bin/.idx support uses the local fail-closed MMIDIDX reader
    and does not import megatron-core.
    """

    path = Path(path)
    inferred = format or _infer_format_from_path(path)
    if inferred == "npz":
        return TokenNpzDataset(
            path,
            seq_len=seq_len,
            batch_size=batch_size,
            **cast(TokenNpzDatasetOptions, kwargs),
        )
    if inferred in {"parquet", "pq"}:
        from cppmega_mlx.data.parquet_dataset import TokenParquetDataset

        return TokenParquetDataset(
            path,
            seq_len=seq_len,
            batch_size=batch_size,
            **cast(TokenParquetDatasetOptions, kwargs),
        )
    if inferred in {"megatron", "bin", "idx"}:
        from cppmega_mlx.data.megatron_indexed import MegatronIndexedDataset

        return MegatronIndexedDataset(
            path,
            seq_len=seq_len,
            batch_size=batch_size,
            **kwargs,
        )
    raise ValueError(f"unsupported token dataset format: {inferred!r}")


def _infer_format_from_path(path: Path) -> str:
    suffix = path.suffix.lstrip(".").lower()
    if suffix:
        return suffix
    if path.with_suffix(".bin").exists() or path.with_suffix(".idx").exists():
        return "megatron"
    return suffix


def iterate_token_batches(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    format: TokenDatasetFormat | None = None,
    **kwargs: Any,
) -> Iterator[LMTokenBatch]:
    """Convenience wrapper mirroring MLX-LM's generator-style batch loops."""

    return open_token_dataset(
        path, seq_len=seq_len, batch_size=batch_size, format=format, **kwargs
    ).iter_batches()


def _fixed_windows(values: np.ndarray, seq_len: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        full = values.shape[0] // seq_len
        return values[: full * seq_len].reshape(full, seq_len)
    if values.ndim == 2:
        full = values.shape[1] // seq_len
        if full == 0:
            return np.empty((0, seq_len), dtype=values.dtype)
        trimmed = values[:, : full * seq_len]
        return trimmed.reshape(values.shape[0] * full, seq_len)
    raise ValueError(f"token arrays must be 1D or 2D, got shape {values.shape}")


def _npz_scalar_int(
    data: NpzFile, key: str, default: int
) -> int:
    if key not in data:
        return default
    return int(np.asarray(data[key]).reshape(()).item())


def _npz_scalar_str(
    data: NpzFile, key: str, default: TokenizerContract
) -> TokenizerContract:
    if key not in data:
        return default
    value = str(np.asarray(data[key]).reshape(()).item())
    if value not in {"megacpp", "local_profile", "custom"}:
        raise ValueError(f"unsupported tokenizer_contract={value!r}")
    return value  # type: ignore[return-value]


def _side_channel_dtype(key: str) -> np.dtype[np.float32] | np.dtype[np.int32]:
    if key == "attention_mask":
        return np.dtype(np.float32)
    return np.dtype(np.int32)


def _to_int32_token_ids(values: np.ndarray) -> np.ndarray:
    _require_integer_array("token IDs", values)
    if np.any(values < 0):
        raise ValueError("token IDs must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError("token IDs exceed int32 range")
    return values.astype(np.int32, copy=False)


def _to_side_channel_values(key: str, values: np.ndarray) -> np.ndarray:
    if key == "attention_mask":
        return values.astype(np.float32, copy=False)
    _require_integer_array(f"{key} side-channel IDs", values)
    if np.any(values < 0):
        raise ValueError(f"{key} side-channel IDs must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{key} side-channel IDs exceed int32 range")
    return values.astype(_side_channel_dtype(key), copy=False)


def _require_integer_array(label: str, values: np.ndarray) -> None:
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must use an integer dtype")


def _reject_unsupported_npz_sidecars(data: NpzFile) -> None:
    ambiguous = sorted(key for key in _AMBIGUOUS_SIDECAR_KEYS if key in data)
    if ambiguous:
        keys = ", ".join(ambiguous)
        raise NotImplementedError(
            "NPZ side-channel arrays must use explicit token-aligned keys; "
            f"ambiguous keys: {keys}"
        )

    ngram = sorted(key for key in _UNSUPPORTED_NGRAM_SIDECAR_KEYS if key in data)
    if ngram:
        keys = ", ".join(ngram)
        raise NotImplementedError(
            "NPZ ngram sidecars are not supported; ngram hashes are derived "
            f"from input_ids in the model path. Unsupported keys: {keys}"
        )


__all__ = [
    "BatchCursor",
    "TokenDatasetMetadata",
    "TokenNpzDataset",
    "iterate_token_batches",
    "open_token_dataset",
]
