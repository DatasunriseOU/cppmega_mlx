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
from typing import Any, Literal, cast

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.platform_context import MAX_PLATFORM_IDS
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
_TOKEN_CHUNK_METADATA_COLUMNS = (
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)
_TOKEN_GRAPH_METADATA_COLUMNS = (
    "token_call_edges",
    "token_type_edges",
)
_TOKEN_SEMANTIC_METADATA_COLUMNS = (
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
    "token_def_use",
)
_TOKEN_TEMPORAL_METADATA_COLUMNS = (
    "token_change_mask_pre",
    "token_change_mask_post",
    "hunk_id_per_token",
    "edit_op_per_token",
)
_ROW_METADATA_COLUMNS = (
    "actual_token_count",
    "pack_id",
    "valid_token_count",
    "num_docs",
    "platform_ids",
    "platform_info",
    "language_info",
    "build_profile",
    "build_info",
    "build_id",
    "compiler",
    "compiler_args",
    "compile_flags",
    "repo",
    "filepath",
    "commit",
    "commit_hash",
    "timestamp",
    "parent_hashes",
    "parent_count",
    "is_merge_commit",
    "author_timestamp",
    "commit_timestamp",
    "repo_stable_id",
    "filepath_stable_id",
    "file_local_commit_index",
    "has_ambiguous_reconstruction",
    "has_rename_ambiguity",
    "constituent_provenance",
    "constituent_provenance_json",
    "provenance",
    "provenance_json",
    "changed_chunk_ids",
    "changed_chunk_spans",
)
_RECOGNIZED_BATCH_METADATA_COLUMNS = (
    *_TOKEN_SEMANTIC_METADATA_COLUMNS,
    *_TOKEN_TEMPORAL_METADATA_COLUMNS,
    *_TOKEN_CHUNK_METADATA_COLUMNS,
    *_TOKEN_GRAPH_METADATA_COLUMNS,
    *_ROW_METADATA_COLUMNS,
)
_MODEL_INPUT_PARQUET_COLUMNS = frozenset(
    alias for aliases in _SIDE_CHANNEL_COLUMN_ALIASES.values() for alias in aliases
)
_MODEL_METADATA_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "platform_ids": ("platform_ids",),
}
_MODEL_INPUT_PARQUET_COLUMNS = frozenset(
    (
        *_MODEL_INPUT_PARQUET_COLUMNS,
        *(
            alias
            for aliases in _MODEL_METADATA_COLUMN_ALIASES.values()
            for alias in aliases
        ),
    )
)
_TOKEN_CONTENT_PARQUET_COLUMNS = frozenset(
    ("tokens", "token_ids", "input_ids", "target_ids", "text")
)
_TOKEN_LEVEL_METADATA_COLUMNS = frozenset(
    (
        *_TOKEN_SEMANTIC_METADATA_COLUMNS,
        *_TOKEN_TEMPORAL_METADATA_COLUMNS,
        "doc_ids",
        "document_ids",
        "packing_document_ids",
        "loss_mask",
    )
)
BatchMetadataColumnSelection = Sequence[str] | Literal["all"] | None


@dataclass(frozen=True)
class ParquetColumns:
    """In-memory Parquet columns normalized to Python lists."""

    values: Mapping[str, list[Any]]
    types: Mapping[str, str] | None = None
    all_column_names: Sequence[str] | None = None

    def require(self, key: str) -> list[Any]:
        if key not in self.values:
            available = ", ".join(sorted(self.all_column_names or self.values))
            raise ValueError(f"parquet column {key!r} not found; available: {available}")
        return self.values[key]

    def type_label(self, key: str) -> str | None:
        if self.types is None:
            return None
        return self.types.get(key)


@dataclass(frozen=True)
class _SideChannelColumns:
    channels: Mapping[str, np.ndarray]
    sources: Mapping[str, Mapping[str, str | None]]
    skipped: Sequence[Mapping[str, str | None]]


@dataclass(frozen=True)
class _ModelMetadataColumns:
    channels: Mapping[str, np.ndarray]
    sources: Mapping[str, Mapping[str, str | None]]


@dataclass(frozen=True)
class _WindowSpec:
    row_index: int | None
    token_start: int
    token_end: int


class TokenParquetDataset:
    """Parquet-backed fixed-shape token batch iterator.

    token_key accepts either one integer token per row or a list-like token
    sequence per row.  text_key accepts source text and requires a tokenizer
    object with encode or a callable str -> Sequence[int].
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
        metadata_columns: BatchMetadataColumnSelection = None,
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

        columns = _read_parquet_columns(
            self.path,
            candidate_columns=_candidate_parquet_columns(
                token_key=token_key,
                text_key=text_key,
                metadata_columns=metadata_columns,
            ),
        )
        token_rows = _token_rows_from_columns(
            columns,
            token_key=token_key,
            text_key=text_key,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
        )
        token_windows = _fixed_windows_from_rows(token_rows, seq_len)
        side_channel_columns = _side_channel_windows(columns, token_rows, seq_len)
        side_channels = side_channel_columns.channels
        model_metadata_columns = _model_metadata_windows(columns, token_rows, seq_len)
        batch_metadata_columns = _resolve_batch_metadata_columns(
            columns,
            token_key=token_key,
            text_key=text_key,
            metadata_columns=metadata_columns,
        )
        batch_metadata_windows = _batch_metadata_windows(
            columns,
            token_rows,
            seq_len=seq_len,
            metadata_columns=batch_metadata_columns,
        )

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
        self._model_metadata_channels = {
            key: value.astype(np.int32, copy=False)
            for key, value in model_metadata_columns.channels.items()
        }
        self._metadata_columns = batch_metadata_columns
        self._metadata_windows = batch_metadata_windows
        self.metadata = metadata or TokenDatasetMetadata(source_format="parquet")
        self.parquet_receipt = _parquet_receipt(
            columns,
            token_key=token_key,
            text_key=text_key,
            side_channel_sources=side_channel_columns.sources,
            skipped_side_channels=side_channel_columns.skipped,
            model_metadata_sources=model_metadata_columns.sources,
            batch_metadata_columns=batch_metadata_columns,
        )

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
        """Yield fixed-shape LMTokenBatch objects."""

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
        kwargs.update(
            {
                key: mx.array(value[sample_idx])
                for key, value in self._model_metadata_channels.items()
            }
        )
        return LMTokenBatch(
            tokens=mx.array(self._tokens[sample_idx]),
            metadata=self._make_batch_metadata(sample_idx),
            **kwargs,
        )

    def _make_batch_metadata(self, sample_idx: np.ndarray) -> Mapping[str, Any] | None:
        if not self._metadata_windows:
            return None

        selected = tuple(self._metadata_windows[int(index)] for index in sample_idx)
        windows = tuple(
            {
                "row_index": window.get("row_index"),
                "token_start": window.get("token_start"),
                "token_end": window.get("token_end"),
            }
            for window in selected
        )
        columns: dict[str, tuple[Any, ...]] = {}
        for column in self._metadata_columns:
            values = tuple(window.get(column) for window in selected)
            if any(value is not None for value in values):
                columns[column] = values
        return {
            "parquet": {
                "windows": windows,
                "columns": columns,
            }
        }


def _candidate_parquet_columns(
    *,
    token_key: str,
    text_key: str | None,
    metadata_columns: BatchMetadataColumnSelection = None,
) -> tuple[str, ...] | None:
    if metadata_columns == "all":
        return None

    candidates = [token_key]
    if text_key is not None:
        candidates.append(text_key)
    for aliases in _SIDE_CHANNEL_COLUMN_ALIASES.values():
        candidates.extend(aliases)
    for aliases in _MODEL_METADATA_COLUMN_ALIASES.values():
        candidates.extend(aliases)
    candidates.extend(str(column) for column in metadata_columns or ())
    return tuple(dict.fromkeys(candidates))


def _read_parquet_columns(
    path: Path,
    *,
    candidate_columns: Sequence[str] | None = None,
) -> ParquetColumns:
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
            tuple(str(name) for name in dataframe.columns),
        )
    else:
        requested = set(candidate_columns or ())
        parquet_file_cls = getattr(pq, "ParquetFile", None)
        full_schema = None
        if parquet_file_cls is not None:
            parquet_file = parquet_file_cls(path)
            full_schema = getattr(parquet_file, "schema_arrow", None)
            all_column_names = (
                list(getattr(full_schema, "names", ()))
                if full_schema is not None
                else []
            )
            selected = [
                name for name in all_column_names if not requested or name in requested
            ]
            table = parquet_file.read(columns=selected)
        else:
            table = pq.read_table(path)
            schema = getattr(table, "schema", None)
            all_column_names = list(table.column_names)
        schema = full_schema or getattr(table, "schema", None)
        loaded_names = list(table.column_names)
        return ParquetColumns(
            {name: table[name].to_pylist() for name in loaded_names},
            None
            if schema is None
            else _schema_type_labels(schema, loaded_names, all_column_names),
            tuple(all_column_names or loaded_names),
        )


def _schema_type_labels(
    schema: Any,
    loaded_names: Sequence[str],
    all_column_names: Sequence[str],
) -> dict[str, str]:
    names = all_column_names or loaded_names
    return {name: str(schema.field(name).type) for name in names}


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
) -> _SideChannelColumns:
    channels: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, str | None]] = {}
    skipped: list[dict[str, str | None]] = []
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
                skipped.append(
                    {
                        "field": key,
                        "column": column_key,
                        "type": columns.type_label(column_key),
                        "reason": "not_token_aligned",
                    }
                )
                continue
            matched.append((column_key, _fixed_windows_from_rows(rows, seq_len)))
        if len(matched) > 1:
            aliases = ", ".join(column_key for column_key, _ in matched)
            raise ValueError(
                f"{key} side-channel declared more than once via columns: {aliases}"
            )
        if matched:
            column_key, windows = matched[0]
            channels[key] = windows
            sources[key] = {
                "column": column_key,
                "type": columns.type_label(column_key),
            }
    return _SideChannelColumns(channels=channels, sources=sources, skipped=skipped)


def _model_metadata_windows(
    columns: ParquetColumns,
    token_rows: list[list[int]],
    seq_len: int,
) -> _ModelMetadataColumns:
    channels: dict[str, np.ndarray] = {}
    sources: dict[str, dict[str, str | None]] = {}
    specs = _fixed_window_specs_from_rows(token_rows, seq_len)
    for key, aliases in _MODEL_METADATA_COLUMN_ALIASES.items():
        matched = [column_key for column_key in aliases if column_key in columns.values]
        if len(matched) > 1:
            joined = ", ".join(matched)
            raise ValueError(f"{key} model metadata declared more than once: {joined}")
        if not matched or not specs:
            continue
        column_key = matched[0]
        rows = [
            _coerce_token_row(value, label=f"{column_key} model metadata")
            for value in columns.require(column_key)
        ]
        if any(len(row) > MAX_PLATFORM_IDS for row in rows):
            raise ValueError(
                f"{column_key} model metadata exceeds MAX_PLATFORM_IDS={MAX_PLATFORM_IDS}"
            )
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            continue
        windows = np.zeros((len(specs), width), dtype=np.int32)
        for out_index, spec in enumerate(specs):
            if spec.row_index is None:
                continue
            row = rows[spec.row_index]
            windows[out_index, : len(row)] = np.asarray(row, dtype=np.int32)
        channels[key] = windows
        sources[key] = {
            "column": column_key,
            "type": columns.type_label(column_key),
        }
    return _ModelMetadataColumns(channels=channels, sources=sources)


def _resolve_batch_metadata_columns(
    columns: ParquetColumns,
    *,
    token_key: str,
    text_key: str | None,
    metadata_columns: BatchMetadataColumnSelection,
) -> tuple[str, ...]:
    if metadata_columns is None:
        return ()

    loaded = set(columns.values)
    if metadata_columns == "all":
        excluded = {
            token_key,
            *_TOKEN_CONTENT_PARQUET_COLUMNS,
            *(_MODEL_INPUT_PARQUET_COLUMNS & loaded),
        }
        if text_key is not None:
            excluded.add(text_key)
        return tuple(
            name
            for name in columns.all_column_names or columns.values
            if name in loaded and name not in excluded
        )

    return tuple(
        dict.fromkeys(str(column) for column in metadata_columns if str(column) in loaded)
    )


def _batch_metadata_windows(
    columns: ParquetColumns,
    token_rows: list[list[int]],
    *,
    seq_len: int,
    metadata_columns: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    resolved_columns = tuple(
        dict.fromkeys(column for column in metadata_columns if column in columns.values)
    )
    if not resolved_columns:
        return ()

    specs = _fixed_window_specs_from_rows(token_rows, seq_len)
    windows: list[dict[str, Any]] = []
    for spec in specs:
        window: dict[str, Any] = {
            "row_index": spec.row_index,
            "token_start": spec.token_start,
            "token_end": spec.token_end,
        }
        if spec.row_index is None:
            windows.append(window)
            continue

        _add_chunk_metadata_window(
            window,
            columns,
            row_index=spec.row_index,
            token_start=spec.token_start,
            token_end=spec.token_end,
            metadata_columns=resolved_columns,
        )
        handled = set(_TOKEN_CHUNK_METADATA_COLUMNS) | set(_TOKEN_GRAPH_METADATA_COLUMNS)
        for column in resolved_columns:
            if column in handled:
                continue
            value = columns.require(column)[spec.row_index]
            if column in _TOKEN_LEVEL_METADATA_COLUMNS and _sequence_length(value) >= spec.token_end:
                window[column] = _metadata_sequence(value)[spec.token_start : spec.token_end]
            else:
                window[column] = _metadata_value(value)
        windows.append(window)
    return tuple(windows)


def _fixed_window_specs_from_rows(
    rows: Sequence[Sequence[int]], seq_len: int
) -> tuple[_WindowSpec, ...]:
    if not rows:
        return ()
    if all(len(row) == 1 for row in rows):
        return tuple(
            _WindowSpec(
                row_index=None,
                token_start=start,
                token_end=start + seq_len,
            )
            for start in range(0, len(rows) - seq_len + 1, seq_len)
        )

    specs: list[_WindowSpec] = []
    for row_index, row in enumerate(rows):
        for start in range(0, len(row) - seq_len + 1, seq_len):
            specs.append(
                _WindowSpec(
                    row_index=row_index,
                    token_start=start,
                    token_end=start + seq_len,
                )
            )
    return tuple(specs)


def _add_chunk_metadata_window(
    window: dict[str, Any],
    columns: ParquetColumns,
    *,
    row_index: int,
    token_start: int,
    token_end: int,
    metadata_columns: Sequence[str],
) -> None:
    requested = set(metadata_columns)
    starts_column = "token_chunk_starts"
    ends_column = "token_chunk_ends"
    if starts_column not in columns.values or ends_column not in columns.values:
        for column in (*_TOKEN_CHUNK_METADATA_COLUMNS, *_TOKEN_GRAPH_METADATA_COLUMNS):
            if column in requested and column in columns.values:
                window[column] = _metadata_value(columns.require(column)[row_index])
        return

    starts = _coerce_token_row(
        columns.require(starts_column)[row_index],
        label=f"{starts_column} metadata",
    )
    ends = _coerce_token_row(
        columns.require(ends_column)[row_index],
        label=f"{ends_column} metadata",
    )
    kinds = _optional_chunk_column(
        columns, "token_chunk_kinds", row_index=row_index, default=0, count=len(starts)
    )
    dep_levels = _optional_chunk_column(
        columns,
        "token_chunk_dep_levels",
        row_index=row_index,
        default=0,
        count=len(starts),
    )
    if not (len(starts) == len(ends) == len(kinds) == len(dep_levels)):
        raise ValueError("token chunk metadata columns must have matching lengths")

    included: list[tuple[int, int, int, int, int]] = []
    index_map: dict[int, int] = {}
    for chunk_index, (start, end, kind, dep_level) in enumerate(
        zip(starts, ends, kinds, dep_levels)
    ):
        if end <= token_start or start >= token_end:
            continue
        local_start = max(0, start - token_start)
        local_end = min(token_end - token_start, end - token_start)
        if local_start >= local_end:
            continue
        index_map[chunk_index] = len(included)
        included.append(
            (int(local_start), int(local_end), int(kind), int(dep_level), chunk_index)
        )

    if starts_column in requested:
        window[starts_column] = [start for start, _, _, _, _ in included]
    if ends_column in requested:
        window[ends_column] = [end for _, end, _, _, _ in included]
    if "token_chunk_kinds" in requested:
        window["token_chunk_kinds"] = [kind for _, _, kind, _, _ in included]
    if "token_chunk_dep_levels" in requested:
        window["token_chunk_dep_levels"] = [
            dep_level for _, _, _, dep_level, _ in included
        ]

    for edge_column in _TOKEN_GRAPH_METADATA_COLUMNS:
        if edge_column not in requested or edge_column not in columns.values:
            continue
        edges = _normalize_edge_pairs(columns.require(edge_column)[row_index])
        window[edge_column] = [
            {"from": index_map[src], "to": index_map[dst]}
            for src, dst in edges
            if src in index_map and dst in index_map
        ]


def _optional_chunk_column(
    columns: ParquetColumns,
    key: str,
    *,
    row_index: int,
    default: int,
    count: int,
) -> list[int]:
    if key not in columns.values:
        return [default] * count
    return _coerce_token_row(columns.require(key)[row_index], label=f"{key} metadata")


def _normalize_edge_pairs(value: Any) -> list[tuple[int, int]]:
    decoded = _decode_json_like(value)
    if decoded is None:
        return []
    if isinstance(decoded, np.ndarray):
        decoded = decoded.tolist()
    if not isinstance(decoded, (list, tuple)):
        return []

    pairs: list[tuple[int, int]] = []
    for item in decoded:
        item = _decode_json_like(item)
        if isinstance(item, Mapping) and "from" in item and "to" in item:
            pairs.append((int(item["from"]), int(item["to"])))
        elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
            pairs.append((int(item[0]), int(item[1])))
    return pairs


def _decode_json_like(value: Any) -> Any:
    while isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        if decoded == value:
            return value
        value = decoded
    return value


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_metadata_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_metadata_value(item) for item in value]
    if isinstance(value, list):
        return [_metadata_value(item) for item in value]
    return value


def _metadata_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [_metadata_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return []


def _sequence_length(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1).shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _parquet_receipt(
    columns: ParquetColumns,
    *,
    token_key: str,
    text_key: str | None,
    side_channel_sources: Mapping[str, Mapping[str, str | None]],
    skipped_side_channels: Sequence[Mapping[str, str | None]],
    model_metadata_sources: Mapping[str, Mapping[str, str | None]],
    batch_metadata_columns: Sequence[str],
) -> dict[str, Any]:
    token_source: dict[str, str | None]
    if token_key in columns.values:
        token_source = {
            "mode": "token_column",
            "column": token_key,
            "type": columns.type_label(token_key),
        }
    else:
        token_source = {
            "mode": "text_column",
            "column": text_key,
            "type": None if text_key is None else columns.type_label(text_key),
        }
    receipt = {
        "source_format": "parquet",
        "columns": sorted(columns.all_column_names or columns.values),
        "column_types": dict(columns.types or {}),
        "token_source": token_source,
        "side_channel_sources": {
            key: dict(value) for key, value in sorted(side_channel_sources.items())
        },
        "skipped_side_channel_columns": [
            dict(value)
            for value in sorted(
                skipped_side_channels,
                key=lambda item: (str(item.get("field")), str(item.get("column"))),
            )
        ],
    }
    if model_metadata_sources:
        receipt["model_metadata_sources"] = {
            key: dict(value) for key, value in sorted(model_metadata_sources.items())
        }
    if batch_metadata_columns:
        receipt["batch_metadata_sources"] = {
            column: {
                "column": column,
                "type": columns.type_label(column),
            }
            for column in batch_metadata_columns
            if column in columns.values
        }
    return receipt


def _rows_are_token_aligned(
    rows: Sequence[Sequence[int | float]], token_rows: Sequence[Sequence[int]]
) -> bool:
    if len(rows) != len(token_rows):
        return False
    return all(len(row) == len(token_row) for row, token_row in zip(rows, token_rows))


def _fixed_windows_from_rows(
    rows: Sequence[Sequence[int | float]], seq_len: int
) -> np.ndarray:
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
) -> Sequence[int | float]:
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
