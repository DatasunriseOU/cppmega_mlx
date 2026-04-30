"""Optional Parquet token datasets for local MLX training.

The cppmega CUDA pipeline materializes token shards as Parquet before building
Megatron indexed datasets.  This reader keeps that handoff available on macOS
without making PyArrow, pandas, or Hugging Face Datasets required imports for
the base MLX package.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.token_dataset import (
    BatchCursor,
    TokenDatasetMetadata,
    _SIDE_CHANNEL_KEYS,
    _fixed_windows,
    _to_int32_token_ids,
    _to_side_channel_values,
)


TextEncoder = Callable[[str], Sequence[int]]

_SIDE_CHANNEL_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "attention_mask": ("token_attention_mask", "attention_mask"),
    "structure_ids": ("token_structure_ids", "structure_ids"),
    "dep_levels": ("token_dep_levels", "dep_levels"),
    "ast_depth_ids": ("token_ast_depth", "ast_depth_ids"),
    "sibling_index_ids": ("token_sibling_index", "sibling_index_ids"),
    "node_type_ids": ("token_ast_node_type", "node_type_ids"),
}
_TOKEN_LEVEL_SIDE_CHANNEL_ALIASES = {
    alias
    for aliases in _SIDE_CHANNEL_COLUMN_ALIASES.values()
    for alias in aliases
    if alias.startswith("token_")
}


@dataclass(frozen=True)
class ParquetColumns:
    """In-memory Parquet columns normalized to Python lists."""

    values: Mapping[str, list[Any]]
    types: Mapping[str, str] | None = None

    def require(self, key: str) -> list[Any]:
        if key not in self.values:
            available = ", ".join(sorted(self.values))
            raise ValueError(f"parquet column {key!r} not found; available: {available}")
        return self.values[key]

    def type_label(self, key: str) -> str | None:
        if self.types is None:
            return None
        return self.types.get(key)


class TokenParquetDataset:
    """Parquet-backed fixed-shape token batch iterator.

    ``token_key`` accepts either one integer token per row or a list-like token
    sequence per row.  ``text_key`` accepts source text and requires a tokenizer
    object with ``encode`` or a callable ``str -> Sequence[int]``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        batch_size: int,
        token_key: str = "tokens",
        text_key: str | None = None,
        tokenizer: Any | None = None,
        eos_token_id: int | None = None,
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
        self.text_key = text_key
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.resume_batch = resume_batch

        columns = _read_parquet_columns(self.path)
        token_rows = _token_rows_from_columns(
            columns,
            token_key=token_key,
            text_key=text_key,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
        )
        token_windows = _fixed_windows_from_rows(token_rows, seq_len)
        side_channels = _side_channel_windows(columns, token_rows, seq_len)

        if not len(token_windows):
            raise ValueError("parquet data does not contain a full fixed-shape sample")
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
        self.metadata = metadata or TokenDatasetMetadata(source_format="parquet")

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
        if self.num_batches == 0:
            return BatchCursor(epoch=epoch, batch_offset=0, global_batch_offset=0)
        return BatchCursor(
            epoch=epoch + consumed_batches // self.num_batches,
            batch_offset=consumed_batches % self.num_batches,
            global_batch_offset=consumed_batches,
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


def _read_parquet_columns(path: Path) -> ParquetColumns:
    try:
        pq = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as pyarrow_error:
        if pyarrow_error.name and not pyarrow_error.name.startswith("pyarrow"):
            raise
        try:
            pd = importlib.import_module("pandas")
        except ModuleNotFoundError as pandas_error:
            if pandas_error.name and pandas_error.name != "pandas":
                raise
            raise ImportError(
                "TokenParquetDataset requires optional dependency 'pyarrow' "
                "or 'pandas' to read parquet files"
            ) from pandas_error
        dataframe = pd.read_parquet(path)
        return ParquetColumns(
            {name: dataframe[name].tolist() for name in dataframe.columns},
            {
                name: str(getattr(dataframe[name], "dtype", "unknown"))
                for name in dataframe.columns
            },
        )
    else:
        table = pq.read_table(path)
        schema = getattr(table, "schema", None)
        return ParquetColumns(
            {name: table[name].to_pylist() for name in table.column_names},
            None
            if schema is None
            else {name: str(schema.field(name).type) for name in table.column_names},
        )


def _token_rows_from_columns(
    columns: ParquetColumns,
    *,
    token_key: str,
    text_key: str | None,
    tokenizer: Any | None,
    eos_token_id: int | None,
) -> list[list[int]]:
    if token_key in columns.values:
        _reject_non_integer_parquet_type(columns, token_key, "token IDs")
        return [_coerce_token_row(value) for value in columns.require(token_key)]
    if text_key is None:
        available = ", ".join(sorted(columns.values))
        raise ValueError(
            f"parquet column {token_key!r} not found and no text_key was provided; "
            f"available: {available}"
        )
    if tokenizer is None:
        raise ValueError("text_key parquet loading requires tokenizer or encode callable")
    rows = []
    for value in columns.require(text_key):
        tokens = list(_encode_text(str(value), tokenizer))
        if eos_token_id is not None and (not tokens or tokens[-1] != eos_token_id):
            tokens.append(eos_token_id)
        rows.append([int(token) for token in tokens])
    return rows


def _side_channel_windows(
    columns: ParquetColumns,
    token_rows: list[list[int]],
    seq_len: int,
) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for key in _SIDE_CHANNEL_KEYS:
        matched: list[tuple[str, np.ndarray]] = []
        for column_key in _SIDE_CHANNEL_COLUMN_ALIASES.get(key, (key,)):
            if column_key not in columns.values:
                continue
            if key != "attention_mask":
                _reject_non_integer_parquet_type(
                    columns, column_key, f"{key} side-channel IDs"
                )
            rows = [
                _coerce_side_channel_row(
                    key, value, label=f"{column_key} side-channel"
                )
                for value in columns.require(column_key)
            ]
            if not _rows_are_token_aligned(rows, token_rows):
                if column_key in _TOKEN_LEVEL_SIDE_CHANNEL_ALIASES:
                    raise ValueError(
                        f"{column_key} side-channel rows must be token-aligned with "
                        f"{len(token_rows)} token rows"
                    )
                continue
            matched.append((column_key, _fixed_windows_from_rows(rows, seq_len)))
        if len(matched) > 1:
            aliases = ", ".join(column_key for column_key, _ in matched)
            raise ValueError(
                f"{key} side-channel declared more than once via columns: {aliases}"
            )
        if matched:
            channels[key] = matched[0][1]
    return channels


def _rows_are_token_aligned(
    rows: list[list[int | float]], token_rows: list[list[int]]
) -> bool:
    if len(rows) != len(token_rows):
        return False
    return all(len(row) == len(token_row) for row, token_row in zip(rows, token_rows))


def _fixed_windows_from_rows(rows: list[list[int | float]], seq_len: int) -> np.ndarray:
    if not rows:
        return np.empty((0, seq_len), dtype=np.int32)
    if all(len(row) == 1 for row in rows):
        return _fixed_windows(np.asarray([row[0] for row in rows]), seq_len)

    windows = [
        _fixed_windows(np.asarray(row), seq_len)
        for row in rows
        if len(row) >= seq_len
    ]
    if not windows:
        return np.empty((0, seq_len), dtype=np.int32)
    return np.concatenate(windows, axis=0)


def _coerce_token_row(value: Any, *, label: str = "token IDs") -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [_coerce_integral_value(token, label=label) for token in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [_coerce_integral_value(token, label=label) for token in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            decoded = json.loads(stripped)
            if not isinstance(decoded, list):
                raise ValueError("serialized token-list parquet values must decode to a list")
            return [_coerce_integral_value(token, label=label) for token in decoded]
        return [
            _coerce_integral_value(token, label=label)
            for token in stripped.replace(",", " ").split()
        ]
    return [_coerce_integral_value(value, label=label)]


def _coerce_integral_value(value: Any, *, label: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{label} must be integer-valued, got boolean {value!r}")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.floating | float):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        raise ValueError(f"{label} must be integer-valued, got {value!r}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label} must be integer-valued, got empty string")
        try:
            return int(stripped, 10)
        except ValueError as error:
            raise ValueError(
                f"{label} must be integer-valued, got {value!r}"
            ) from error
    raise ValueError(f"{label} must be integer-valued, got {type(value).__name__}")


def _coerce_side_channel_row(
    key: str, value: Any, *, label: str
) -> list[int | float]:
    if key != "attention_mask":
        return _coerce_token_row(value, label=label)
    return _coerce_numeric_row(value, label=label)


def _coerce_numeric_row(value: Any, *, label: str) -> list[int | float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [
            _coerce_numeric_value(token, label=label)
            for token in value.reshape(-1).tolist()
        ]
    if isinstance(value, (list, tuple)):
        return [_coerce_numeric_value(token, label=label) for token in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            decoded = json.loads(stripped)
            if not isinstance(decoded, list):
                raise ValueError("serialized token-list parquet values must decode to a list")
            return [_coerce_numeric_value(token, label=label) for token in decoded]
        return [
            _coerce_numeric_value(token, label=label)
            for token in stripped.replace(",", " ").split()
        ]
    return [_coerce_numeric_value(value, label=label)]


def _coerce_numeric_value(value: Any, *, label: str) -> int | float:
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{label} must be numeric, got empty string")
        try:
            return int(stripped, 10)
        except ValueError:
            try:
                return float(stripped)
            except ValueError as error:
                raise ValueError(f"{label} must be numeric, got {value!r}") from error
    raise ValueError(f"{label} must be numeric, got {type(value).__name__}")


def _reject_non_integer_parquet_type(
    columns: ParquetColumns, key: str, label: str
) -> None:
    type_label = columns.type_label(key)
    if type_label is None:
        return
    lowered = type_label.lower()
    if any(fragment in lowered for fragment in ("bool", "float", "double", "decimal")):
        raise ValueError(
            f"{label} parquet column {key!r} must use an integer dtype, "
            f"got {type_label}"
        )


def _encode_text(text: str, tokenizer: Any) -> Sequence[int]:
    if hasattr(tokenizer, "encode"):
        return cast(Sequence[int], tokenizer.encode(text))
    if callable(tokenizer):
        return cast(Sequence[int], tokenizer(text))
    raise TypeError("tokenizer must expose encode(text) or be callable")


__all__ = ["ParquetColumns", "TokenParquetDataset"]
