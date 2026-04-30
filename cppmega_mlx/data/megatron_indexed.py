"""Megatron indexed token dataset reader for local MLX training."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import numpy as np

from cppmega_mlx.config.model import (
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
)
from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.token_dataset import BatchCursor, TokenDatasetMetadata

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_ATTENTION_SIDE_CHANNEL_KEY = "attention_mask"
_STRUCTURE_SIDE_CHANNEL_KEYS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
_SIDE_CHANNEL_KEYS = (_ATTENTION_SIDE_CHANNEL_KEY, *_STRUCTURE_SIDE_CHANNEL_KEYS)
_SIDE_CHANNEL_SCHEMA_VERSION = 1
_SIDE_CHANNEL_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    _ATTENTION_SIDE_CHANNEL_KEY: ("token_attention_mask",),
    "structure_ids": ("token_structure_ids",),
    "dep_levels": ("token_dep_levels",),
    "ast_depth_ids": ("token_ast_depth",),
    "sibling_index_ids": ("token_sibling_index",),
    "node_type_ids": ("token_ast_node_type",),
}
_SIDE_CHANNEL_ALIAS_TO_KEY = {
    alias: key
    for key, aliases in _SIDE_CHANNEL_KEY_ALIASES.items()
    for alias in aliases
}
_SUPPORTED_SIDE_CHANNEL_ENTRY_KEYS = (
    *_SIDE_CHANNEL_KEYS,
    *_SIDE_CHANNEL_ALIAS_TO_KEY,
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

_INDEX_DTYPES: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    8: np.dtype(np.uint16),
}

_NAMED_DTYPES: dict[str, np.dtype] = {
    "uint8": np.dtype(np.uint8),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
}

_SIDE_CHANNEL_NAMED_DTYPES: dict[str, np.dtype] = {
    **_NAMED_DTYPES,
    "float32": np.dtype(np.float32),
}


@dataclass(frozen=True)
class MegatronIndexedMetadata:
    """Parsed on-disk layout metadata for a Megatron indexed token shard."""

    bin_path: Path
    idx_path: Path | None
    dtype: str
    sequence_count: int
    document_count: int
    token_count: int
    source_format: str = "megatron"


class MegatronIndexedDataset:
    """Read fixed token windows from Megatron ``.bin/.idx`` shards.

    The reader intentionally implements only the stable ``MMIDIDX`` layout and
    explicit raw-binary handoffs.  Unknown headers, dtype codes, and ambiguous
    raw binaries fail closed instead of pulling in the original training stack.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        batch_size: int,
        dtype: str | np.dtype | None = None,
        metadata_path: str | Path | None = None,
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
        if token_key != "tokens":
            raise ValueError("Megatron indexed datasets use fixed token_key='tokens'")

        resolved = _resolve_paths(path, metadata_path=metadata_path)
        sidecar = _load_sidecar(resolved.metadata_path)

        self.path = resolved.prefix
        self.bin_path = resolved.bin_path
        self.idx_path = resolved.idx_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.token_key = token_key
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.resume_batch = resume_batch

        if self.idx_path is not None and self.idx_path.exists():
            index = _parse_mmididx(self.idx_path)
            token_dtype = index.dtype
            sequence_offsets = index.sequence_offsets
            sequence_lengths = index.sequence_lengths
            document_count = int(index.document_indices.shape[0] - 1)
            source_format = str(sidecar.get("source_format", "megatron"))
        else:
            raw_dtype = _coerce_dtype(dtype or sidecar.get("dtype"))
            if raw_dtype is None:
                raise ValueError(
                    "raw .bin datasets require an explicit dtype or JSON sidecar dtype"
                )
            token_count = _raw_token_count(
                self.bin_path, dtype=raw_dtype, sidecar=sidecar
            )
            token_dtype = raw_dtype
            sequence_offsets = np.array([0], dtype=np.int64)
            sequence_lengths = np.array([token_count], dtype=np.int64)
            document_count = int(sidecar.get("document_count", 1))
            source_format = str(sidecar.get("source_format", "megatron-raw"))

        _validate_bin_references(
            self.bin_path,
            dtype=token_dtype,
            offsets=sequence_offsets,
            lengths=sequence_lengths,
        )
        windows = _build_windows(
            sequence_offsets,
            sequence_lengths,
            self.seq_len,
            itemsize=token_dtype.itemsize,
        )
        if not len(windows):
            raise ValueError("Megatron token data does not contain a full fixed-shape sample")

        self._dtype = token_dtype
        self._bin_mmap = np.memmap(self.bin_path, mode="r", dtype=np.uint8)
        self._windows = windows
        self._side_channels = _load_side_channels(
            sidecar,
            prefix=resolved.prefix,
            metadata_path=resolved.metadata_path,
            token_dtype=token_dtype,
            token_count=int(sequence_lengths.sum()),
            token_windows=windows,
        )
        self.index_metadata = MegatronIndexedMetadata(
            bin_path=self.bin_path,
            idx_path=self.idx_path if self.idx_path and self.idx_path.exists() else None,
            dtype=self._dtype.name,
            sequence_count=int(sequence_lengths.shape[0]),
            document_count=document_count,
            token_count=int(sequence_lengths.sum()),
            source_format=source_format,
        )
        self.metadata = metadata if metadata is not None else _token_metadata(
            sidecar, source_format=source_format
        )

    def __len__(self) -> int:
        return self.num_batches

    @property
    def num_samples(self) -> int:
        return int(self._windows.shape[0])

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
        """Yield fixed-shape ``LMTokenBatch`` objects."""

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
        """Return the deterministic cursor after ``consumed_batches``."""

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
        """Return the min/max token IDs in the indexed token storage."""

        token_count = self.index_metadata.token_count
        if token_count <= 0:
            raise ValueError("Megatron token data does not contain any tokens")
        token_view = np.memmap(self.bin_path, mode="r", dtype=self._dtype)[:token_count]
        token_min = int(token_view.min())
        token_max = int(token_view.max())
        if token_min < 0:
            raise ValueError("token IDs must be non-negative")
        if token_max > np.iinfo(np.int32).max:
            raise ValueError("token IDs exceed int32 range")
        return token_min, token_max

    def _make_batch(self, sample_idx: np.ndarray) -> LMTokenBatch:
        tokens = np.empty((sample_idx.shape[0], self.seq_len), dtype=np.int32)
        side_channels = {
            key: np.empty(
                (sample_idx.shape[0], self.seq_len),
                dtype=_target_side_channel_dtype(key),
            )
            for key in self._side_channels
        }
        for row, window_index in enumerate(sample_idx):
            byte_offset = int(self._windows[int(window_index)])
            token_view = np.frombuffer(
                self._bin_mmap,
                dtype=self._dtype,
                count=self.seq_len,
                offset=byte_offset,
            )
            tokens[row] = _to_int32_tokens(token_view)
            for key, storage in self._side_channels.items():
                side_view = np.frombuffer(
                    storage.mmap,
                    dtype=storage.dtype,
                    count=self.seq_len,
                    offset=int(storage.windows[int(window_index)]),
                )
                side_channels[key][row] = _to_side_channel_values(key, side_view)

        kwargs = {key: mx.array(value) for key, value in side_channels.items()}
        return LMTokenBatch(tokens=mx.array(tokens), **kwargs)


def open_megatron_indexed_dataset(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    dtype: str | np.dtype | None = None,
    metadata_path: str | Path | None = None,
    token_key: str = "tokens",
    shuffle: bool = False,
    seed: int = 0,
    loop: bool = False,
    resume_batch: int = 0,
    metadata: TokenDatasetMetadata | None = None,
) -> MegatronIndexedDataset:
    """Open a standalone local Megatron-indexed shard for CLI/training code.

    This is the explicit fail-closed ingress for macOS/MLX paths that already
    have Megatron ``.bin/.idx`` token shards.  It intentionally depends only on
    the local reader, NumPy, and MLX; it does not import Megatron or Torch
    runtime modules.
    """

    return MegatronIndexedDataset(
        path,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
        metadata_path=metadata_path,
        token_key=token_key,
        shuffle=shuffle,
        seed=seed,
        loop=loop,
        resume_batch=resume_batch,
        metadata=metadata,
    )


def megatron_indexed_side_channel_schema() -> dict[str, dict[str, object]]:
    """Return the documented token-aligned side-channel schema.

    Canonical keys are the names delivered on :class:`LMTokenBatch`.  Aliases
    match cppmega Parquet token-level column names and are normalized at the
    JSON sidecar boundary.
    """

    return {
        key: {
            "aliases": list(_SIDE_CHANNEL_KEY_ALIASES.get(key, ())),
            "default_dtype": _default_side_channel_dtype(key).name,
            "target_dtype": _target_side_channel_dtype(key).name,
            "allowed_dtypes": _allowed_side_channel_dtype_names(key),
            "model_kwarg": key in _STRUCTURE_SIDE_CHANNEL_KEYS,
        }
        for key in _SIDE_CHANNEL_KEYS
    }


@dataclass(frozen=True)
class _ResolvedPaths:
    prefix: Path
    bin_path: Path
    idx_path: Path | None
    metadata_path: Path | None


@dataclass(frozen=True)
class _ParsedIndex:
    dtype: np.dtype
    sequence_lengths: np.ndarray
    sequence_offsets: np.ndarray
    document_indices: np.ndarray


@dataclass(frozen=True)
class _SideChannelStorage:
    path: Path
    dtype: np.dtype
    windows: np.ndarray
    mmap: np.memmap


def _resolve_paths(
    path: str | Path, *, metadata_path: str | Path | None
) -> _ResolvedPaths:
    raw_path = Path(path)
    explicit_metadata = Path(metadata_path) if metadata_path is not None else None

    if raw_path.suffix == ".idx":
        prefix = raw_path.with_suffix("")
        idx_path: Path | None = raw_path
        bin_path = prefix.with_suffix(".bin")
    elif raw_path.suffix == ".bin":
        prefix = raw_path.with_suffix("")
        bin_path = raw_path
        candidate_idx = prefix.with_suffix(".idx")
        idx_path = candidate_idx if candidate_idx.exists() else None
    elif raw_path.suffix == ".json":
        explicit_metadata = raw_path
        if raw_path.name.endswith(".idx.json"):
            prefix = Path(str(raw_path)[: -len(".idx.json")])
        else:
            prefix = raw_path.with_suffix("")
        bin_path = prefix.with_suffix(".bin")
        candidate_idx = prefix.with_suffix(".idx")
        idx_path = candidate_idx if candidate_idx.exists() else None
    else:
        prefix = raw_path
        bin_path = prefix.with_suffix(".bin")
        candidate_idx = prefix.with_suffix(".idx")
        idx_path = candidate_idx if candidate_idx.exists() else None

    sidecar = explicit_metadata or _find_sidecar(prefix)
    return _ResolvedPaths(
        prefix=prefix,
        bin_path=bin_path,
        idx_path=idx_path,
        metadata_path=sidecar,
    )


def _find_sidecar(prefix: Path) -> Path | None:
    for candidate in (
        Path(str(prefix) + ".idx.json"),
        prefix.with_suffix(".json"),
        Path(str(prefix) + ".bin.json"),
    ):
        if candidate.exists():
            return candidate
    return None


def _load_sidecar(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata sidecar must be a JSON object: {path}")
    return payload


def _load_side_channels(
    sidecar: dict[str, Any],
    *,
    prefix: Path,
    metadata_path: Path | None,
    token_dtype: np.dtype,
    token_count: int,
    token_windows: np.ndarray,
) -> dict[str, _SideChannelStorage]:
    _reject_ambiguous_side_channel_metadata(sidecar)
    entries = _side_channel_entries(sidecar)
    base_dir = metadata_path.parent if metadata_path is not None else prefix.parent
    storages: dict[str, _SideChannelStorage] = {}
    for key, entry in entries.items():
        path, dtype = _parse_side_channel_entry(key, entry, base_dir=base_dir)
        capacity, remainder = divmod(path.stat().st_size, dtype.itemsize)
        if remainder:
            raise ValueError(
                f"{key} side-channel file size is not divisible by dtype itemsize {dtype.itemsize}"
            )
        if int(capacity) != token_count:
            raise ValueError(
                f"{key} side-channel token count {int(capacity)} does not match "
                f"token shard count {token_count}"
            )
        token_item_offset = token_windows // token_dtype.itemsize
        windows = token_item_offset * dtype.itemsize
        mmap = np.memmap(path, mode="r", dtype=np.uint8)
        storages[key] = _SideChannelStorage(
            path=path,
            dtype=dtype,
            windows=windows.astype(np.int64, copy=False),
            mmap=mmap,
        )
    return storages


def _reject_ambiguous_side_channel_metadata(sidecar: dict[str, Any]) -> None:
    ambiguous = [
        key
        for key in _AMBIGUOUS_SIDECAR_KEYS
        if key in sidecar and _declares_side_channel(sidecar[key])
    ]
    if ambiguous:
        keys = ", ".join(sorted(ambiguous))
        raise NotImplementedError(
            "Megatron indexed side-channel metadata must use explicit "
            f"'side_channel_paths' or top-level path entries; ambiguous keys: {keys}"
        )

    ngram = [
        key
        for key in _UNSUPPORTED_NGRAM_SIDECAR_KEYS
        if key in sidecar and _declares_side_channel(sidecar[key])
    ]
    if ngram:
        keys = ", ".join(sorted(ngram))
        raise NotImplementedError(
            "Megatron indexed ngram sidecars are not supported; ngram hashes are "
            f"derived from input_ids in the model path. Unsupported keys: {keys}"
        )


def _side_channel_entries(sidecar: dict[str, Any]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    mapping = sidecar.get("side_channel_paths")
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise ValueError("side_channel_paths must be a mapping of key to path metadata")
        for raw_key, entry in mapping.items():
            key = _canonical_side_channel_key(str(raw_key))
            if key is None:
                raise NotImplementedError(f"unsupported side-channel key {raw_key!r}")
            if key in entries:
                raise ValueError(f"{key} side-channel declared more than once")
            entries[key] = entry
    for raw_key in _SUPPORTED_SIDE_CHANNEL_ENTRY_KEYS:
        if raw_key in sidecar and _declares_side_channel(sidecar[raw_key]):
            key = _canonical_side_channel_key(raw_key)
            if key is None:
                raise NotImplementedError(f"unsupported side-channel key {raw_key!r}")
            if key in entries:
                raise ValueError(f"{key} side-channel declared more than once")
            entries[key] = sidecar[raw_key]
    return entries


def _canonical_side_channel_key(key: str) -> str | None:
    if key in _SIDE_CHANNEL_KEYS:
        return key
    return _SIDE_CHANNEL_ALIAS_TO_KEY.get(key)


def _parse_side_channel_entry(
    key: str, entry: Any, *, base_dir: Path
) -> tuple[Path, np.dtype]:
    if isinstance(entry, str):
        path = Path(entry)
        dtype_value = None
    elif isinstance(entry, dict):
        if "path" not in entry:
            raise ValueError(f"{key} side-channel entry must include a path")
        path = Path(str(entry["path"]))
        dtype_value = entry.get("dtype")
    else:
        raise ValueError(
            f"{key} side-channel entry must be a path string or object with path/dtype"
        )
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    dtype = _coerce_side_channel_dtype(key, dtype_value)
    return path, dtype


def _coerce_side_channel_dtype(key: str, value: Any | None) -> np.dtype:
    if value is None:
        return _default_side_channel_dtype(key)
    if not isinstance(value, str):
        try:
            dtype = np.dtype(value)
        except TypeError as error:
            raise ValueError(f"unsupported {key} side-channel dtype {value!r}") from error
    else:
        dtype = _SIDE_CHANNEL_NAMED_DTYPES.get(value)
        if dtype is None:
            raise ValueError(f"unsupported {key} side-channel dtype {value!r}")
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        if dtype != np.dtype(np.float32):
            raise ValueError("attention_mask side-channel dtype must be float32")
        return dtype
    if dtype.kind not in {"i", "u"}:
        raise ValueError(f"{key} side-channel dtype must be an integer dtype")
    if dtype.itemsize > np.dtype(np.int64).itemsize:
        raise ValueError(f"{key} side-channel dtype {dtype.name!r} is too wide")
    return dtype


def _default_side_channel_dtype(key: str) -> np.dtype:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return np.dtype(np.float32)
    return np.dtype(np.int32)


def _target_side_channel_dtype(key: str) -> np.dtype:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return np.dtype(np.float32)
    return np.dtype(np.int32)


def _allowed_side_channel_dtype_names(key: str) -> list[str]:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return ["float32"]
    return [
        name
        for name, dtype in _SIDE_CHANNEL_NAMED_DTYPES.items()
        if dtype.kind in {"i", "u"}
    ]


def _to_side_channel_values(key: str, values: np.ndarray) -> np.ndarray:
    if key == _ATTENTION_SIDE_CHANNEL_KEY:
        return values.astype(np.float32, copy=False)
    if np.any(values < 0):
        raise ValueError(f"{key} side-channel IDs must be non-negative")
    if np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{key} side-channel IDs exceed int32 range")
    return values.astype(np.int32, copy=False)


def _declares_side_channel(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, dict | list | tuple | set):
        return bool(value)
    return bool(value)


def _parse_mmididx(path: Path) -> _ParsedIndex:
    data = np.memmap(path, mode="r", dtype=np.uint8)
    view = memoryview(data)
    offset = 0

    header = bytes(view[offset : offset + len(_INDEX_HEADER)])
    offset += len(_INDEX_HEADER)
    if header != _INDEX_HEADER:
        raise NotImplementedError(f"unsupported Megatron .idx header in {path}")

    version = _unpack("<Q", view, offset)
    offset += 8
    if version != _INDEX_VERSION:
        raise NotImplementedError(
            f"unsupported Megatron .idx version {version}; expected {_INDEX_VERSION}"
        )

    dtype_code = _unpack("<B", view, offset)
    offset += 1
    dtype = _INDEX_DTYPES.get(dtype_code)
    if dtype is None:
        raise NotImplementedError(
            f"unsupported Megatron token dtype code {dtype_code}"
        )

    sequence_count = _unpack("<Q", view, offset)
    offset += 8
    document_count = _unpack("<Q", view, offset)
    offset += 8

    if sequence_count > np.iinfo(np.int64).max or document_count > np.iinfo(np.int64).max:
        raise ValueError("Megatron index counts exceed supported range")

    lengths = np.frombuffer(
        view, dtype=np.int32, count=int(sequence_count), offset=offset
    ).astype(np.int64, copy=True)
    offset += int(sequence_count) * np.dtype(np.int32).itemsize

    pointers = np.frombuffer(
        view, dtype=np.int64, count=int(sequence_count), offset=offset
    ).astype(np.int64, copy=True)
    offset += int(sequence_count) * np.dtype(np.int64).itemsize

    documents = np.frombuffer(
        view, dtype=np.int64, count=int(document_count), offset=offset
    ).astype(np.int64, copy=True)
    offset += int(document_count) * np.dtype(np.int64).itemsize

    remaining = int(data.size) - offset
    if remaining not in {0, int(sequence_count)}:
        raise NotImplementedError(
            f"unsupported Megatron .idx trailer length {remaining} bytes"
        )
    if remaining == int(sequence_count):
        np.frombuffer(view, dtype=np.int8, count=int(sequence_count), offset=offset)

    _validate_index_arrays(lengths, pointers, documents, itemsize=dtype.itemsize)
    return _ParsedIndex(
        dtype=dtype,
        sequence_lengths=lengths,
        sequence_offsets=pointers,
        document_indices=documents,
    )


def _unpack(fmt: str, data: memoryview, offset: int) -> int:
    size = struct.calcsize(fmt)
    if offset + size > len(data):
        raise ValueError("truncated Megatron .idx header")
    return int(struct.unpack_from(fmt, data, offset)[0])


def _validate_index_arrays(
    lengths: np.ndarray, pointers: np.ndarray, documents: np.ndarray, *, itemsize: int
) -> None:
    if pointers.shape != lengths.shape:
        raise ValueError("Megatron .idx sequence pointer count must match sequence lengths")
    if np.any(lengths < 0):
        raise ValueError("Megatron .idx contains negative sequence lengths")
    if np.any(pointers < 0):
        raise ValueError("Megatron .idx contains negative sequence pointers")
    if lengths.shape[0]:
        expected = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)]
        )
        expected *= itemsize
        if not np.array_equal(pointers, expected):
            raise ValueError("Megatron .idx sequence pointers do not match token dtype")
    if documents.shape[0] == 0:
        raise ValueError("Megatron .idx document_indices must include a sentinel")
    if documents[0] != 0 or documents[-1] != lengths.shape[0]:
        raise ValueError("Megatron .idx document_indices must span all sequences")
    if np.any(np.diff(documents) < 0):
        raise ValueError("Megatron .idx document_indices must be monotonic")


def _coerce_dtype(value: str | np.dtype | type[np.generic] | None) -> np.dtype | None:
    if value is None:
        return None
    if isinstance(value, np.dtype):
        dtype = value
    elif isinstance(value, str):
        dtype = _NAMED_DTYPES.get(value)
        if dtype is None:
            raise ValueError(f"unsupported token dtype {value!r}")
    else:
        dtype = np.dtype(value)
    if dtype.name not in _NAMED_DTYPES:
        raise ValueError(f"unsupported token dtype {dtype.name!r}")
    return dtype


def _raw_token_count(
    bin_path: Path, *, dtype: np.dtype, sidecar: dict[str, Any]
) -> int:
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    capacity, remainder = divmod(bin_path.stat().st_size, dtype.itemsize)
    if remainder:
        raise ValueError(
            f"{bin_path} size is not divisible by dtype itemsize {dtype.itemsize}"
        )
    token_count = int(sidecar.get("token_count", capacity))
    if token_count < 0:
        raise ValueError("metadata token_count must be non-negative")
    if token_count > capacity:
        raise ValueError("metadata token_count exceeds raw .bin capacity")
    return token_count


def _validate_bin_references(
    bin_path: Path, *, dtype: np.dtype, offsets: np.ndarray, lengths: np.ndarray
) -> None:
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)
    file_size = bin_path.stat().st_size
    itemsize = dtype.itemsize
    ends = offsets + lengths * itemsize
    if np.any(ends > file_size):
        raise ValueError(f"Megatron .idx references bytes past {bin_path}")
    if np.any(offsets % itemsize != 0):
        raise ValueError("Megatron .idx sequence pointers must align to dtype size")


def _build_windows(
    sequence_offsets: np.ndarray,
    sequence_lengths: np.ndarray,
    seq_len: int,
    *,
    itemsize: int,
) -> np.ndarray:
    windows: list[int] = []
    for byte_offset, length in zip(sequence_offsets, sequence_lengths, strict=True):
        full = int(length) // seq_len
        if full == 0:
            continue
        start = int(byte_offset)
        for sample in range(full):
            windows.append(start + sample * seq_len * itemsize)
    return np.asarray(windows, dtype=np.int64)


def _to_int32_tokens(tokens: np.ndarray) -> np.ndarray:
    if np.any(tokens < 0):
        raise ValueError("token IDs must be non-negative")
    if np.any(tokens > np.iinfo(np.int32).max):
        raise ValueError("token IDs exceed int32 range")
    return tokens.astype(np.int32, copy=False)


def _token_metadata(
    sidecar: dict[str, Any], *, source_format: str
) -> TokenDatasetMetadata:
    return TokenDatasetMetadata(
        vocab_size=int(sidecar.get("vocab_size", MEGACPP_TOKENIZER_VOCAB_SIZE)),
        tokenizer_contract=str(sidecar.get("tokenizer_contract", "megacpp")),  # type: ignore[arg-type]
        local_profile_vocab_size=int(
            sidecar.get("local_profile_vocab_size", LOCAL_PROFILE_VOCAB_SIZE)
        ),
        megacpp_tokenizer_vocab_size=int(
            sidecar.get("megacpp_tokenizer_vocab_size", MEGACPP_TOKENIZER_VOCAB_SIZE)
        ),
        source_format=source_format,
    )


__all__ = [
    "MegatronIndexedDataset",
    "MegatronIndexedMetadata",
    "megatron_indexed_side_channel_schema",
    "open_megatron_indexed_dataset",
]
