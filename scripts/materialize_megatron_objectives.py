#!/usr/bin/env python3
"""Materialize the typed objective schedule as Megatron-ready parquet.

The output rows use ``shifted_lm_document_v1``: ``input_ids`` contains
``[objective_input[0], *objective_targets]`` and ``loss_mask`` contains the
objective mask followed by a zero sentinel. The adjacent JSON receipt is
validated again by the root Megatron converter before indexed data is emitted.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
import glob
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cppmega_mlx.data.domain_schema import (
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
    validate_case5_contract_metadata,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    SOURCE_IDENTITY_REGISTRY_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SOURCE_DOC_IDS_COLUMN,
    TOKEN_SOURCE_IDENTITY_IDS_COLUMN,
)
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    INPUT_IDS_COLUMN,
    NUM_DOCS_COLUMN,
    PACKED_ROWS_OBJECTIVE_SOURCE_COLUMNS,
    VALID_TOKEN_COUNT_COLUMN,
)
from cppmega_mlx.data.source_identity import (
    MAX_ROW_LOCAL_DOC_ID,
    MAX_SOURCE_ID,
    validate_source_identity_registry,
)
from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)
from cppmega_mlx.training.megatron_objectives import (
    MaterializedMegatronDocument,
    OBJECTIVE_TOKEN_SIDE_CHANNELS,
    materialize_megatron_document,
    write_objective_materialization_artifact,
)
from cppmega_mlx.training.objective_contract_accumulator import (
    ObjectiveContractAccumulator,
)
from cppmega_mlx.training.objective_data import (
    OBJECTIVE_ROUTE_MAPPING_SCHEMA,
    OBJECTIVE_SOURCE_COLUMNS,
    normalize_megatron_objective_source_row,
    objective_source_from_tokenized_row,
    require_megatron_objective_source_columns,
)
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    GraphAuxLossConfig,
    ObjectiveQuotaUnsatisfiedError,  # noqa: F401 - retained for test/API compatibility
    ObjectiveSource,
)
from cppmega_mlx.training.objective_schedule import (
    CanonicalObjectivePlanner,
    assess_graph_positive_capability,
    validate_graph_receipt_against_document,
)
from cppmega_mlx.data.graph_recipe import (
    STAGE1_GRAPH_RELATIONS,
    stage1_graph_config_kwargs,
)
from cppmega_mlx.training.task_mixer import STAGE1_DEFAULT_RATES
from cppmega_mlx.training.task_mixer import TaskKind

_ARROW_DTYPES = {
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
}
TOKEN_SIDECAR_TYPES = {
    column: _ARROW_DTYPES[dtype] for column, dtype in OBJECTIVE_TOKEN_SIDE_CHANNELS
}
PAIR_TYPE = pa.struct([pa.field("from", pa.uint16()), pa.field("to", pa.uint16())])
TRIPLE_TYPE = pa.struct(
    [
        pa.field("from", pa.uint32()),
        pa.field("to", pa.uint32()),
        pa.field("kind", pa.int32()),
    ]
)
SYMBOL_IDENTITY_TYPE = pa.struct(
    [
        pa.field("symbol_id", pa.uint64(), nullable=False),
        pa.field("symbol_key", pa.string(), nullable=False),
    ]
)
SOURCE_IDENTITY_TYPE = pa.struct(
    [
        pa.field("source_identity_id", pa.uint64(), nullable=False),
        pa.field("canonical_sha256", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)
_SourceStat = tuple[int, int, int, int, int]
_LEGACY_SAMPLING_MODE = "deterministic_epoch_shuffle_v1"
_BOUNDED_SAMPLING_MODE = "deterministic_shard_row_group_record_batch_shuffle_v2"
_BOUNDED_SAMPLING_PRODUCER = "pyarrow.parquet.ParquetFile.iter_batches"
_BOUNDED_SAMPLING_PRODUCER_VERSION = 1
_DEFAULT_SOURCE_BATCH_ROWS = 64
_DEFAULT_WRITE_BATCH_ROWS = 8
_DEFAULT_MAX_BUFFER_BYTES = 256 * 1024 * 1024
_SOURCE_POOL_MANIFEST_SCHEMA = "cppmega_ci_objective_pool_manifest_v1"
_TWO_POOL_SOURCE_SNAPSHOT_SCHEMA = "cppmega_objective_source_snapshot_v2"
_TWO_POOL_SCHEDULE = "alternate_primary_seed_v1"
_PRIMARY_POOL = "primary_ci"
_SEED_POOL = "objective_seed"
_SOURCE_POOL_MANIFEST_COPY = "objective_source_pool_manifest.json"
_CI_EXPORT_RECEIPT_COPY = "ci_export_receipt.json"
_CI_EXPORT_SCHEMAS = {
    "cppmega_ci_content_store_case5_export_v2",
    "cppmega_ci_content_store_case5_export_v4",
}


def _source_stat(path: Path) -> _SourceStat:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifact_set_sha256(records: list[dict[str, object]]) -> str:
    canonical = [
        {
            "path": str(record["path"]),
            "size": int(record["size_bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(records, key=lambda item: str(item["path"]))
    ]
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _build_source_snapshot(
    shards: list[str],
    *,
    sequence_length: int,
    requested_samples: int,
    seed: int,
    sampling_mode: str = _LEGACY_SAMPLING_MODE,
    source_batch_rows: int | None = None,
    source_root: Path | None = None,
) -> tuple[dict[str, object], dict[Path, _SourceStat]]:
    paths = sorted(Path(shard).resolve() for shard in shards)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("objective source shards must be non-empty and unique")
    canonical_root = None if source_root is None else source_root.resolve()
    if canonical_root is not None and not canonical_root.is_dir():
        raise ValueError(f"objective source root is not a directory: {canonical_root}")
    records: list[dict[str, object]] = []
    row_group_rows: list[list[int]] = []
    signatures: dict[Path, _SourceStat] = {}
    row_count = 0
    for path in paths:
        before = _source_stat(path)
        digest = _file_sha256(path)
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        shard_row_groups = [
            int(parquet.metadata.row_group(index).num_rows)
            for index in range(parquet.metadata.num_row_groups)
        ]
        after = _source_stat(path)
        if before != after:
            raise RuntimeError(f"objective source changed while hashing: {path}")
        if rows < 1:
            raise ValueError(f"objective source parquet is empty: {path}")
        signatures[path] = after
        if canonical_root is None:
            canonical_path = path.as_posix()
        else:
            try:
                canonical_path = path.relative_to(canonical_root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"objective source shard is outside --source-root: {path}"
                ) from exc
        records.append(
            {
                "path": canonical_path,
                "size_bytes": after[2],
                "sha256": digest,
                "rows": rows,
            }
        )
        row_group_rows.append(shard_row_groups)
        row_count += rows
    full_passes, tail_rows = divmod(requested_samples, row_count)
    sampling: dict[str, object] = {
        "mode": sampling_mode,
        "seed": int(seed),
        "requested_samples": int(requested_samples),
        "full_passes": full_passes,
        "tail_rows": tail_rows,
        "min_row_reuse": full_passes,
        "max_row_reuse": full_passes + int(tail_rows > 0),
    }
    if sampling_mode == _BOUNDED_SAMPLING_MODE:
        if source_batch_rows is None or source_batch_rows < 1:
            raise ValueError("bounded source sampling requires source_batch_rows >= 1")
        sampling.update(
            {
                "record_batch_rows": int(source_batch_rows),
                "producer": {
                    "name": _BOUNDED_SAMPLING_PRODUCER,
                    "version": _BOUNDED_SAMPLING_PRODUCER_VERSION,
                    "row_group_rows": row_group_rows,
                },
                "ordering": {
                    "permutation": "sha256_sort_key_v1",
                    "epochs": "ascending",
                    "shards": "seeded_permutation_per_epoch",
                    "row_groups": "seeded_permutation_per_shard_epoch",
                    "record_batches": "physical_order_within_row_group",
                    "rows": "seeded_permutation_within_record_batch",
                },
                "cursor_semantics": "last_yielded_row_v1",
                "final_cursor": None,
            }
        )
    elif sampling_mode != _LEGACY_SAMPLING_MODE:
        raise ValueError(f"unsupported objective source sampling mode: {sampling_mode}")
    return (
        {
            "schema": "cppmega_objective_source_snapshot_v1",
            "sequence_length": int(sequence_length),
            "file_count": len(records),
            "row_count": row_count,
            "files": records,
            "sampling": sampling,
            "artifact_set_sha256": _source_artifact_set_sha256(records),
        },
        signatures,
    )


def _bind_source_sampling_cursor(
    source_snapshot: dict[str, object],
    *,
    cursor: Mapping[str, int] | None,
    consumed_samples: int | None = None,
) -> None:
    sampling = source_snapshot.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("mode") != _BOUNDED_SAMPLING_MODE:
        raise ValueError("source snapshot does not use bounded deterministic sampling")
    if cursor is None:
        raise ValueError("bounded source sampling did not yield a replay cursor")
    requested_samples = (
        int(sampling["requested_samples"])
        if consumed_samples is None
        else int(consumed_samples)
    )
    if requested_samples < 1:
        raise ValueError("bounded source sampling must consume at least one row")
    row_count = int(source_snapshot["row_count"])
    full_passes, tail_rows = divmod(requested_samples, row_count)
    sampling.update(
        {
            "requested_samples": requested_samples,
            "full_passes": full_passes,
            "tail_rows": tail_rows,
            "min_row_reuse": full_passes,
            "max_row_reuse": full_passes + int(tail_rows > 0),
        }
    )
    if int(cursor.get("source_index", -1)) != requested_samples - 1:
        raise ValueError(
            "source replay cursor does not match requested sample count: "
            f"cursor={cursor.get('source_index')}, requested={requested_samples}"
        )
    sampling["final_cursor"] = dict(cursor)


def _require_source_snapshot_unchanged(
    signatures: dict[Path, _SourceStat],
) -> None:
    for path, expected in signatures.items():
        if _source_stat(path) != expected:
            raise RuntimeError(
                f"objective source changed while materializing documents: {path}"
            )


def _manifest_parquet_paths(
    raw_records: object,
    *,
    root: Path,
    where: str,
) -> tuple[list[str], list[dict[str, object]]]:
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{where} must be a non-empty list")
    canonical_root = root.resolve()
    if not canonical_root.is_dir():
        raise ValueError(f"{where} root is not a directory: {canonical_root}")
    paths: list[str] = []
    records: list[dict[str, object]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "rows",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(
                f"{where}[{index}] must contain path, rows, size_bytes, and sha256"
            )
        raw_path = raw_record["path"]
        rows = raw_record["rows"]
        size_bytes = raw_record["size_bytes"]
        sha256 = raw_record["sha256"]
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or Path(raw_path).is_absolute()
            or ".." in Path(raw_path).parts
            or Path(raw_path).suffix != ".parquet"
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows < 1
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 1
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"{where}[{index}] has an invalid file binding")
        path = (canonical_root / raw_path).resolve()
        try:
            path.relative_to(canonical_root)
        except ValueError as exc:
            raise ValueError(f"{where}[{index}] escapes its source root") from exc
        if not path.is_file() or path.stat().st_size != size_bytes:
            raise ValueError(f"{where}[{index}] file size binding drifted: {raw_path}")
        paths.append(str(path))
        records.append(dict(raw_record))
    if len(paths) != len(set(paths)):
        raise ValueError(f"{where} contains duplicate parquet paths")
    return paths, records


def _load_source_pool_manifest(
    path: Path,
    *,
    primary_root: Path,
    objective_seed_root: Path,
    sequence_length: int,
) -> tuple[
    list[str],
    list[dict[str, object]],
    list[str],
    list[dict[str, object]],
    bytes,
    bytes,
    dict[Path, _SourceStat],
]:
    manifest_path = path.resolve()
    before = _source_stat(manifest_path)
    raw = manifest_path.read_bytes()
    after = _source_stat(manifest_path)
    if before != after:
        raise RuntimeError(f"source pool manifest changed while reading: {manifest_path}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid source pool manifest: {manifest_path}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "algorithm",
        "sequence_lengths",
        "ci_export",
        _PRIMARY_POOL,
        _SEED_POOL,
        "producer",
    }:
        raise ValueError("source pool manifest has unsupported top-level fields")
    if (
        payload["schema"] != _SOURCE_POOL_MANIFEST_SCHEMA
        or payload["algorithm"] != _TWO_POOL_SCHEDULE
        or not isinstance(payload["sequence_lengths"], list)
        or sequence_length not in payload["sequence_lengths"]
    ):
        raise ValueError("source pool manifest schema, schedule, or bucket is invalid")

    primary = payload[_PRIMARY_POOL]
    objective_seed = payload[_SEED_POOL]
    if (
        not isinstance(primary, Mapping)
        or set(primary) != {"files_by_sequence_length"}
        or not isinstance(primary["files_by_sequence_length"], Mapping)
        or not isinstance(objective_seed, Mapping)
        or set(objective_seed) != {"files"}
        or not isinstance(payload["producer"], Mapping)
    ):
        raise ValueError("source pool manifest pool inventory is invalid")
    raw_primary_records = primary["files_by_sequence_length"].get(
        str(sequence_length)
    )
    primary_paths, primary_records = _manifest_parquet_paths(
        raw_primary_records,
        root=primary_root,
        where=f"{_PRIMARY_POOL}.files_by_sequence_length.{sequence_length}",
    )
    seed_paths, seed_records = _manifest_parquet_paths(
        objective_seed["files"],
        root=objective_seed_root,
        where=f"{_SEED_POOL}.files",
    )

    ci_export = payload["ci_export"]
    if not isinstance(ci_export, Mapping) or set(ci_export) != {
        "path",
        "sha256",
        "schema",
        "status",
        "source_completion",
    }:
        raise ValueError("source pool manifest CI export binding is invalid")
    if (
        ci_export["path"] != "export_receipt.json"
        or ci_export["schema"] not in _CI_EXPORT_SCHEMAS
        or ci_export["status"] != "complete"
        or not isinstance(ci_export["source_completion"], Mapping)
    ):
        raise ValueError("source pool manifest CI export semantics are invalid")
    receipt_path = (primary_root.resolve() / "export_receipt.json").resolve()
    receipt_before = _source_stat(receipt_path)
    receipt_raw = receipt_path.read_bytes()
    receipt_after = _source_stat(receipt_path)
    if receipt_before != receipt_after:
        raise RuntimeError(f"CI export receipt changed while reading: {receipt_path}")
    if hashlib.sha256(receipt_raw).hexdigest() != ci_export["sha256"]:
        raise ValueError("source pool manifest CI export receipt hash drifted")
    try:
        receipt = json.loads(receipt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid CI export receipt: {receipt_path}") from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != ci_export["schema"]
        or receipt.get("status") != "complete"
    ):
        raise ValueError("source pool manifest differs from its CI export receipt")
    source_completion = ci_export["source_completion"]
    if (
        source_completion.get("schema") != receipt["schema"]
        or source_completion.get("status") != "complete"
    ):
        raise ValueError("source pool manifest CI completion binding is invalid")
    if receipt["schema"].endswith("_v4"):
        if (
            source_completion.get("completion_mode") != "inventory-exhaustive"
            or source_completion.get("production_complete") is not True
        ):
            raise ValueError("production CI source pool is not inventory-exhaustive")
    elif (
        source_completion.get("production_complete") is True
        or source_completion.get("completion_mode") == "inventory-exhaustive"
    ):
        raise ValueError("threshold CI source pool cannot claim exhaustive completion")
    return (
        primary_paths,
        primary_records,
        seed_paths,
        seed_records,
        raw,
        receipt_raw,
        {manifest_path: after, receipt_path: receipt_after},
    )


def _build_two_pool_source_snapshot(
    manifest_path: Path,
    *,
    primary_root: Path,
    objective_seed_root: Path,
    sequence_length: int,
    requested_source_rows: int,
    seed: int,
    source_batch_rows: int,
) -> tuple[
    dict[str, object],
    dict[Path, _SourceStat],
    list[str],
    list[str],
    bytes,
    bytes,
]:
    (
        primary_shards,
        primary_records,
        seed_shards,
        seed_records,
        manifest_raw,
        receipt_raw,
        binding_signatures,
    ) = _load_source_pool_manifest(
        manifest_path,
        primary_root=primary_root,
        objective_seed_root=objective_seed_root,
        sequence_length=sequence_length,
    )
    primary_requested = (requested_source_rows + 1) // 2
    seed_requested = requested_source_rows // 2
    primary_snapshot, primary_signatures = _build_source_snapshot(
        primary_shards,
        sequence_length=sequence_length,
        requested_samples=primary_requested,
        seed=seed,
        sampling_mode=_BOUNDED_SAMPLING_MODE,
        source_batch_rows=source_batch_rows,
        source_root=primary_root,
    )
    seed_snapshot, seed_signatures = _build_source_snapshot(
        seed_shards,
        sequence_length=sequence_length,
        requested_samples=seed_requested,
        seed=seed,
        sampling_mode=_BOUNDED_SAMPLING_MODE,
        source_batch_rows=source_batch_rows,
        source_root=objective_seed_root,
    )
    if (
        primary_snapshot["files"] != primary_records
        or seed_snapshot["files"] != seed_records
    ):
        raise ValueError("source pool manifest parquet bytes or row counts drifted")
    snapshot = {
        "schema": _TWO_POOL_SOURCE_SNAPSHOT_SCHEMA,
        "sequence_length": int(sequence_length),
        "algorithm": _TWO_POOL_SCHEDULE,
        "pool_order": [_PRIMARY_POOL, _SEED_POOL],
        "source_pool_manifest": {
            "path": _SOURCE_POOL_MANIFEST_COPY,
            "size_bytes": len(manifest_raw),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
        "ci_export_receipt": {
            "path": _CI_EXPORT_RECEIPT_COPY,
            "size_bytes": len(receipt_raw),
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
        },
        "pools": {
            _PRIMARY_POOL: primary_snapshot,
            _SEED_POOL: seed_snapshot,
        },
    }
    signatures = {
        **primary_signatures,
        **seed_signatures,
        **binding_signatures,
    }
    return (
        snapshot,
        signatures,
        primary_shards,
        seed_shards,
        manifest_raw,
        receipt_raw,
    )


def materialized_schema() -> pa.Schema:
    fields = [
        pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
        pa.field("valid_token_count", pa.uint32(), nullable=False),
        pa.field("objective_kind", pa.string(), nullable=False),
        *[
            pa.field(column, pa.list_(dtype), nullable=False)
            for column, dtype in TOKEN_SIDECAR_TYPES.items()
        ],
        pa.field(
            "source_platform_ids", pa.list_(pa.list_(pa.uint16())), nullable=False
        ),
        pa.field("token_call_edges", pa.list_(PAIR_TYPE), nullable=False),
        pa.field("token_type_edges", pa.list_(PAIR_TYPE), nullable=False),
        pa.field("token_domain_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_build_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_shell_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_diagnostic_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_cross_domain_edges", pa.list_(TRIPLE_TYPE), nullable=False),
        pa.field("token_chunk_starts", pa.list_(pa.uint32()), nullable=False),
        pa.field("token_chunk_ends", pa.list_(pa.uint32()), nullable=False),
        pa.field("token_chunk_kinds", pa.list_(pa.uint8()), nullable=False),
        pa.field("token_chunk_dep_levels", pa.list_(pa.uint16()), nullable=False),
        pa.field(
            SYMBOL_IDENTITIES_COLUMN,
            pa.list_(SYMBOL_IDENTITY_TYPE),
            nullable=False,
        ),
        pa.field(
            SOURCE_IDENTITY_REGISTRY_COLUMN,
            pa.list_(SOURCE_IDENTITY_TYPE),
            nullable=False,
        ),
    ]
    return pa.schema(fields).with_metadata(
        {
            SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii"): str(
                SYMBOL_IDENTITY_SCHEMA_VERSION
            ).encode("ascii"),
            DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode("utf-8"): (
                DOMAIN_DELIMITER_CONTRACT_SHA256.encode("ascii")
            ),
            DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8"): (
                DOMAIN_SCHEMA_SHA256.encode("ascii")
            ),
            TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8"): (
                TOKENIZER_CONTRACT_SHA256.encode("ascii")
            ),
            b"cppmega.case5_schema": b"case5_domain_routes_v1",
            b"cppmega.macro_routes_version": b"full_macro_concept_routes_v1",
            b"cppmega.objective_route_mapping": OBJECTIVE_ROUTE_MAPPING_SCHEMA.encode(
                "ascii"
            ),
        }
    )


def _pad(values: object, capacity: int, *, fill: int) -> list[int]:
    result = [int(value) for value in values]  # type: ignore[union-attr]
    if len(result) > capacity:
        raise ValueError(
            f"materialized token-aligned row length {len(result)} exceeds {capacity}"
        )
    return result + [fill] * (capacity - len(result))


def padded_row(
    document: MaterializedMegatronDocument, *, capacity: int
) -> dict[str, object]:
    row = dict(document.row)
    valid = int(row["valid_token_count"])
    if valid != len(document.token_ids) or valid > capacity:
        raise ValueError(
            f"invalid materialized valid_token_count={valid}, capacity={capacity}"
        )
    row["input_ids"] = _pad(row["input_ids"], capacity, fill=0)
    for column in TOKEN_SIDECAR_TYPES:
        fill = 0
        row[column] = _pad(row[column], capacity, fill=fill)
    return row


def _schedule_key(seed: int, *components: object) -> bytes:
    encoded = json.dumps(
        [seed, *components], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).digest()


def _deterministic_permutation(size: int, seed: int, *components: object) -> list[int]:
    return sorted(
        range(size),
        key=lambda index: (_schedule_key(seed, *components, index), index),
    )


def _record_batch_rows(batch: pa.RecordBatch) -> list[dict[str, object]]:
    return batch.to_pylist()


def _iter_parquet_source_rows(
    shards: list[str],
    *,
    seed: int,
    source_batch_rows: int,
) -> Iterator[tuple[dict[str, object], dict[str, int]]]:
    """Yield the snapshot-declared hierarchy with one bounded batch resident."""

    if source_batch_rows < 1:
        raise ValueError("source_batch_rows must be >= 1")
    paths = sorted(Path(shard).resolve() for shard in shards)
    identity_columns = (
        "source_doc_id",
        "source_document_id",
        "document_id",
        "doc_id",
        "repo_stable_id",
        "filepath_stable_id",
        "commit_hash",
        "file_local_commit_index",
        "text",
    )
    epoch = 0
    while True:
        shard_order = _deterministic_permutation(len(paths), seed, "shards", epoch)
        for shard_position, shard_index in enumerate(shard_order):
            path = paths[shard_index]
            parquet = pq.ParquetFile(path)
            validate_case5_contract_metadata(
                parquet.schema_arrow.metadata,
                where=str(path),
            )
            available = tuple(parquet.schema_arrow.names)
            require_megatron_objective_source_columns(available)
            selected = [
                column for column in OBJECTIVE_SOURCE_COLUMNS if column in available
            ]
            selected.extend(
                column
                for column in (
                    INPUT_IDS_COLUMN,
                    VALID_TOKEN_COUNT_COLUMN,
                    NUM_DOCS_COLUMN,
                    *PACKED_ROWS_OBJECTIVE_SOURCE_COLUMNS,
                )
                if column in available and column not in selected
            )
            if "doc_ids" in available:
                selected.append("doc_ids")
            selected.extend(
                column
                for column in identity_columns
                if column in available and column not in selected
            )
            row_group_order = _deterministic_permutation(
                parquet.num_row_groups,
                seed,
                "row_groups",
                epoch,
                shard_index,
            )
            for row_group_position, row_group_index in enumerate(row_group_order):
                batches = parquet.iter_batches(
                    batch_size=source_batch_rows,
                    row_groups=[row_group_index],
                    columns=selected,
                    use_threads=False,
                )
                for record_batch_index, batch in enumerate(batches):
                    rows = _record_batch_rows(batch)
                    row_order = _deterministic_permutation(
                        len(rows),
                        seed,
                        "rows",
                        epoch,
                        shard_index,
                        row_group_index,
                        record_batch_index,
                    )
                    for row_shuffle_position, row_index in enumerate(row_order):
                        yield (
                            rows[row_index],
                            {
                                "epoch": epoch,
                                "shard_position": shard_position,
                                "shard_index": shard_index,
                                "row_group_position": row_group_position,
                                "row_group_index": row_group_index,
                                "record_batch_index": record_batch_index,
                                "row_shuffle_position": row_shuffle_position,
                                "row_index_in_record_batch": row_index,
                            },
                        )
        epoch += 1


class _AlternatingSourceRows(
    Iterator[tuple[dict[str, object], dict[str, int]]]
):
    def __init__(
        self,
        primary_rows: Iterator[tuple[dict[str, object], dict[str, int]]],
        seed_rows: Iterator[tuple[dict[str, object], dict[str, int]]],
    ) -> None:
        self._rows = (primary_rows, seed_rows)
        self._next_pool = 0
        self.rows_yielded = [0, 0]
        self.last_pool_cursors: list[dict[str, int] | None] = [None, None]

    def __iter__(self) -> _AlternatingSourceRows:
        return self

    def __next__(self) -> tuple[dict[str, object], dict[str, int]]:
        pool_index = self._next_pool
        row, raw_cursor = next(self._rows[pool_index])
        pool_source_index = self.rows_yielded[pool_index]
        pool_cursor = {
            **raw_cursor,
            "source_index": pool_source_index,
        }
        self.rows_yielded[pool_index] += 1
        self.last_pool_cursors[pool_index] = pool_cursor
        self._next_pool = 1 - pool_index
        return (
            row,
            {
                "pool_index": pool_index,
                "pool_source_index": pool_source_index,
                "primary_rows_yielded": self.rows_yielded[0],
                "objective_seed_rows_yielded": self.rows_yielded[1],
                "next_pool_index": self._next_pool,
                **{
                    f"pool_{key}": value
                    for key, value in raw_cursor.items()
                },
            },
        )


class _ObjectiveSourceIterator(Iterator[ObjectiveSource]):
    def __init__(
        self,
        shards: list[str],
        *,
        seed: int,
        source_batch_rows: int,
        rows: Iterator[tuple[dict[str, object], dict[str, int]]] | None = None,
    ) -> None:
        self._rows = (
            _iter_parquet_source_rows(
                shards,
                seed=seed,
                source_batch_rows=source_batch_rows,
            )
            if rows is None
            else rows
        )
        self._source_index = 0
        self.last_cursor: dict[str, int] | None = None

    def __iter__(self) -> _ObjectiveSourceIterator:
        return self

    def __next__(self) -> ObjectiveSource:
        raw_row, raw_cursor = next(self._rows)
        row = normalize_megatron_objective_source_row(
            raw_row,
            source_index=self._source_index,
        )
        token_count = len(row[TOKEN_IDS_COLUMN])  # type: ignore[arg-type]
        source_doc_ids = [
            int(value)
            for value in row[TOKEN_SOURCE_DOC_IDS_COLUMN]  # type: ignore[union-attr]
        ]
        if len(source_doc_ids) != token_count:
            raise ValueError(
                f"{TOKEN_SOURCE_DOC_IDS_COLUMN} length "
                f"{len(source_doc_ids)} != token count {token_count}"
            )
        if any(not 0 < value <= MAX_ROW_LOCAL_DOC_ID for value in source_doc_ids):
            raise ValueError(
                f"{TOKEN_SOURCE_DOC_IDS_COLUMN} must contain positive "
                "uint32 row-local constituent IDs"
            )

        source_identity_ids = [
            int(value)
            for value in row[TOKEN_SOURCE_IDENTITY_IDS_COLUMN]  # type: ignore[union-attr]
        ]
        if len(source_identity_ids) != token_count:
            raise ValueError(
                f"{TOKEN_SOURCE_IDENTITY_IDS_COLUMN} length "
                f"{len(source_identity_ids)} != token count {token_count}"
            )
        if any(not 0 < value <= MAX_SOURCE_ID for value in source_identity_ids):
            raise ValueError(
                f"{TOKEN_SOURCE_IDENTITY_IDS_COLUMN} must contain positive "
                "uint64 physical source IDs"
            )
        registry = row[SOURCE_IDENTITY_REGISTRY_COLUMN]
        if not isinstance(registry, list):
            raise ValueError(
                f"{SOURCE_IDENTITY_REGISTRY_COLUMN} must be a list of records"
            )
        validate_source_identity_registry(
            registry,
            referenced_ids=source_identity_ids,
        )

        raw_document_ids = row.get("doc_ids")
        document_ids = (
            [1] * token_count
            if raw_document_ids is None
            else [int(value) for value in raw_document_ids]  # type: ignore[union-attr]
        )
        if len(document_ids) != token_count or any(
            not 0 < value <= MAX_ROW_LOCAL_DOC_ID for value in document_ids
        ):
            raise ValueError(
                "doc_ids must contain one positive uint32 attention segment "
                "ID per token"
            )

        source = objective_source_from_tokenized_row(
            raw_row, source_index=self._source_index
        )
        if source.code_packet is None:  # pragma: no cover - typed row invariant
            raise ValueError("objective source row did not produce a CodePacket")
        code_packet = replace(
            source.code_packet,
            document_ids=mx.array(np.asarray(document_ids, dtype=np.uint32)),
            source_doc_ids=mx.array(np.asarray(source_doc_ids, dtype=np.uint32)),
            source_identity_ids=mx.array(
                np.asarray(source_identity_ids, dtype=np.uint64)
            ),
        )
        result = replace(source, code_packet=code_packet)
        self.last_cursor = {
            **raw_cursor,
            "source_index": self._source_index,
        }
        self._source_index += 1
        return result


def _iter_sources(
    shards: list[str],
    *,
    seed: int,
    source_batch_rows: int = _DEFAULT_SOURCE_BATCH_ROWS,
) -> _ObjectiveSourceIterator:
    return _ObjectiveSourceIterator(
        shards,
        seed=seed,
        source_batch_rows=source_batch_rows,
    )


def _iter_two_pool_sources(
    primary_shards: list[str],
    seed_shards: list[str],
    *,
    seed: int,
    source_batch_rows: int,
) -> tuple[_ObjectiveSourceIterator, _AlternatingSourceRows]:
    rows = _AlternatingSourceRows(
        _iter_parquet_source_rows(
            primary_shards,
            seed=seed,
            source_batch_rows=source_batch_rows,
        ),
        _iter_parquet_source_rows(
            seed_shards,
            seed=seed,
            source_batch_rows=source_batch_rows,
        ),
    )
    return (
        _ObjectiveSourceIterator(
            [],
            seed=seed,
            source_batch_rows=source_batch_rows,
            rows=rows,
        ),
        rows,
    )


def _bind_two_pool_sampling_cursors(
    source_snapshot: dict[str, object],
    *,
    rows: _AlternatingSourceRows,
    global_cursor: Mapping[str, int],
    consumed_samples: int,
) -> None:
    if (
        source_snapshot.get("schema") != _TWO_POOL_SOURCE_SNAPSHOT_SCHEMA
        or source_snapshot.get("algorithm") != _TWO_POOL_SCHEDULE
    ):
        raise ValueError("source snapshot does not use deterministic two-pool sampling")
    if sum(rows.rows_yielded) != consumed_samples:
        raise ValueError("two-pool source counters do not match consumed samples")
    if int(global_cursor.get("source_index", -1)) != consumed_samples - 1:
        raise ValueError("two-pool global cursor does not match consumed samples")
    if (
        int(global_cursor.get("primary_rows_yielded", -1)) != rows.rows_yielded[0]
        or int(global_cursor.get("objective_seed_rows_yielded", -1))
        != rows.rows_yielded[1]
    ):
        raise ValueError("two-pool global cursor counters drifted")
    pools = source_snapshot.get("pools")
    if not isinstance(pools, dict):
        raise ValueError("two-pool source snapshot is missing its pools")
    for pool_index, pool_name in enumerate((_PRIMARY_POOL, _SEED_POOL)):
        snapshot = pools.get(pool_name)
        cursor = rows.last_pool_cursors[pool_index]
        if not isinstance(snapshot, dict) or cursor is None:
            raise ValueError(f"two-pool source snapshot is missing {pool_name}")
        _bind_source_sampling_cursor(
            snapshot,
            cursor=cursor,
            consumed_samples=rows.rows_yielded[pool_index],
        )


def _bind_case5_contract_hashes(receipt: dict[str, object]) -> None:
    """Bind a generated receipt to the exact frozen CASE5 contract bytes."""

    expected = {
        "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
    }
    for key, digest in expected.items():
        actual = receipt.get(key)
        if actual is not None and actual != digest:
            raise ValueError(
                f"objective receipt has stale {key}: {actual!r}, expected {digest!r}"
            )
        receipt[key] = digest


def _nested_python_bytes(value: object) -> int:
    if isinstance(value, Mapping):
        return sys.getsizeof(value) + sum(
            _nested_python_bytes(key) + _nested_python_bytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(_nested_python_bytes(item) for item in value)
    return sys.getsizeof(value)


def _estimate_padded_row_buffer_bytes(
    document: MaterializedMegatronDocument,
    *,
    capacity: int,
) -> int:
    valid = int(document.row["valid_token_count"])
    if valid != len(document.token_ids) or valid > capacity:
        raise ValueError(
            f"invalid materialized valid_token_count={valid}, capacity={capacity}"
        )

    fixed_columns = {"input_ids", *TOKEN_SIDECAR_TYPES}
    fixed_list_count = len(fixed_columns)
    pointer_bytes = struct.calcsize("P")
    python_integer_bytes = sys.getsizeof(0)
    projected_padded_vectors = fixed_list_count * (
        sys.getsizeof([]) + capacity * (pointer_bytes + python_integer_bytes)
    )
    variable_python = sys.getsizeof(document.row)
    for key, value in document.row.items():
        if key not in fixed_columns:
            variable_python += _nested_python_bytes(key)
            variable_python += _nested_python_bytes(value)

    fixed_arrow_bytes_per_token = 4 + sum(
        dtype.bit_width // 8 for dtype in TOKEN_SIDECAR_TYPES.values()
    )
    projected_arrow = capacity * fixed_arrow_bytes_per_token + fixed_list_count * 8
    # Variable Python objects and their Arrow buffers coexist during conversion.
    return projected_padded_vectors + 2 * variable_python + projected_arrow + 64 * 1024


class _StreamingParquetShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        schema: pa.Schema,
        capacity: int,
        shard_rows: int,
        write_batch_rows: int,
        max_buffer_bytes: int,
    ) -> None:
        if shard_rows < 1 or write_batch_rows < 1 or max_buffer_bytes < 1:
            raise ValueError("streaming parquet limits must all be positive")
        self._output_dir = output_dir
        self._schema = schema
        self._capacity = capacity
        self._shard_rows = shard_rows
        self._write_batch_rows = write_batch_rows
        self._max_buffer_bytes = max_buffer_bytes
        self._pending_rows: list[dict[str, object]] = []
        self._pending_bytes = 0
        self._rows_in_shard = 0
        self._writer: pq.ParquetWriter | None = None
        self._paths: list[Path] = []
        self._closed = False

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._paths)

    def __enter__(self) -> _StreamingParquetShardWriter:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _open_shard(self) -> None:
        path = self._output_dir / f"objectives_{len(self._paths):05d}.parquet"
        self._writer = pq.ParquetWriter(path, self._schema, compression="zstd")
        self._paths.append(path)
        self._rows_in_shard = 0

    def _close_shard(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _flush(self) -> None:
        if not self._pending_rows:
            return
        if self._writer is None:
            self._open_shard()
        table = pa.Table.from_pylist(self._pending_rows, schema=self._schema)
        if table.nbytes > self._pending_bytes:
            raise RuntimeError(
                "write-buffer estimator undercounted Arrow payload: "
                f"estimated={self._pending_bytes}, actual={table.nbytes}"
            )
        assert self._writer is not None
        self._writer.write_table(table, row_group_size=table.num_rows)
        self._rows_in_shard += table.num_rows
        self._pending_rows.clear()
        self._pending_bytes = 0
        if self._rows_in_shard == self._shard_rows:
            self._close_shard()

    def add(self, document: MaterializedMegatronDocument) -> None:
        if self._closed:
            raise RuntimeError("cannot add rows to a closed parquet writer")
        estimated_bytes = _estimate_padded_row_buffer_bytes(
            document,
            capacity=self._capacity,
        )
        if estimated_bytes > self._max_buffer_bytes:
            raise MemoryError(
                "one padded objective row exceeds --max-buffer-bytes before "
                "Arrow conversion: "
                f"estimated={estimated_bytes}, budget={self._max_buffer_bytes}"
            )

        shard_capacity = self._shard_rows - self._rows_in_shard
        batch_capacity = min(self._write_batch_rows, shard_capacity)
        if self._pending_rows and (
            len(self._pending_rows) >= batch_capacity
            or self._pending_bytes + estimated_bytes > self._max_buffer_bytes
        ):
            self._flush()
            shard_capacity = self._shard_rows - self._rows_in_shard
            batch_capacity = min(self._write_batch_rows, shard_capacity)

        self._pending_rows.append(padded_row(document, capacity=self._capacity))
        self._pending_bytes += estimated_bytes
        if len(self._pending_rows) >= batch_capacity:
            self._flush()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._close_shard()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._pending_rows.clear()
        self._pending_bytes = 0
        self._close_shard()
        self._closed = True


@contextmanager
def _atomic_output_directory(output_dir: Path) -> Iterator[Path]:
    final = output_dir.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"objective output directory already exists: {final}")
    partial = Path(tempfile.mkdtemp(prefix=f".{final.name}.partial-", dir=final.parent))
    committed = False
    try:
        yield partial
        if final.exists():
            raise FileExistsError(
                f"objective output directory appeared before publish: {final}"
            )
        partial.replace(final)
        committed = True
    finally:
        if not committed and partial.exists():
            shutil.rmtree(partial)


def _materialized_assignment_has_graph_positive(
    source: ObjectiveSource,
    task: TaskKind,
    *,
    graph_relations: tuple[str, ...],
) -> bool:
    """Apply the graph-positive predicate to the task's realized route map."""

    mixer = EligibilityAwareTaskMixer({task: 1.0}, seed=0)
    realized = mixer.materialize(source, step_index=0)
    receipt = assess_graph_positive_capability(
        realized,
        source,
        graph_relations=graph_relations,
        require_route_sidecars=False,
    )
    return bool(receipt["eligible"])


def _materialize_stream(
    *,
    mixer: EligibilityAwareTaskMixer,
    source_iter: Iterator[ObjectiveSource],
    accumulator: ObjectiveContractAccumulator,
    writer: _StreamingParquetShardWriter,
    samples: int,
    quota_window_samples: int,
    quota_lookahead_samples: int,
    graph_relations: tuple[str, ...] = (),
    require_route_sidecars: bool = True,
) -> dict[str, object]:
    planner = CanonicalObjectivePlanner(
        mixer=mixer,
        source_iter=source_iter,
        quota_window_samples=quota_window_samples,
        quota_lookahead_samples=quota_lookahead_samples,
        graph_relations=graph_relations,
        require_route_sidecars=require_route_sidecars,
    )
    for start_step in range(0, samples, quota_window_samples):
        window = planner.plan_window(start_step=start_step)
        for assignment in window.assignments:
            document = materialize_megatron_document(
                assignment.realized,
                assignment.source,
                require_production_sidecars=require_route_sidecars,
            )
            if assignment.graph_eligibility is not None:
                validate_graph_receipt_against_document(
                    assignment.graph_eligibility,
                    document,
                    graph_relations=graph_relations,
                )
            accumulator.add(document)
            writer.add(document)
            del document
    return planner.source_selection_receipt(output_samples=samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-glob")
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Immutable root used to record canonical relative source paths; "
            "production bundles require code/<bucket>/... and commits/<bucket>/..."
        ),
    )
    parser.add_argument(
        "--source-pool-manifest",
        type=Path,
        help=(
            "Receipt-bound CI and objective-seed allowlist; when set, --data-glob "
            "is forbidden and rows alternate primary CI then objective seed"
        ),
    )
    parser.add_argument(
        "--objective-seed-root",
        type=Path,
        help="Immutable root for objective seed paths declared by the pool manifest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--quota-window-samples", type=int, default=60)
    parser.add_argument(
        "--quota-lookahead-samples",
        type=int,
        default=None,
        help=(
            "Maximum extra source rows retained while satisfying one exact quota "
            "window (default: three quota windows)"
        ),
    )
    parser.add_argument("--shard-rows", type=int, default=1024)
    parser.add_argument(
        "--source-batch-rows",
        type=int,
        default=_DEFAULT_SOURCE_BATCH_ROWS,
        help="Maximum source rows decoded in one Parquet record batch",
    )
    parser.add_argument(
        "--write-batch-rows",
        type=int,
        default=_DEFAULT_WRITE_BATCH_ROWS,
        help="Maximum padded rows converted in one Arrow write batch",
    )
    parser.add_argument(
        "--max-buffer-bytes",
        type=int,
        default=_DEFAULT_MAX_BUFFER_BYTES,
        help="Hard conservative byte budget for one padded Arrow write batch",
    )
    parser.add_argument("--seed", type=int, default=17)
    graph_recipe = stage1_graph_config_kwargs()
    parser.add_argument(
        "--graph-aux-weight", type=float, default=graph_recipe["global_weight"]
    )
    parser.add_argument(
        "--graph-indexer-weight", type=float, default=graph_recipe["indexer_weight"]
    )
    parser.add_argument(
        "--graph-layer-weight", type=float, default=graph_recipe["layer_weight"]
    )
    parser.add_argument(
        "--graph-bce-weight", type=float, default=graph_recipe["bce_weight"]
    )
    parser.add_argument(
        "--graph-coverage-weight", type=float, default=graph_recipe["coverage_weight"]
    )
    parser.add_argument("--graph-topk", type=int, default=graph_recipe["topk"])
    parser.add_argument(
        "--graph-relations",
        default=",".join(STAGE1_GRAPH_RELATIONS),
        help="Comma-separated graph relations included in the auxiliary loss",
    )
    args = parser.parse_args()

    quota_lookahead_samples = (
        3 * args.quota_window_samples
        if args.quota_lookahead_samples is None
        else args.quota_lookahead_samples
    )

    if (
        args.samples < 1
        or args.quota_window_samples < 1
        or quota_lookahead_samples < 0
        or args.samples % args.quota_window_samples
    ):
        raise ValueError(
            "--samples must be positive and divisible by --quota-window-samples; "
            "--quota-lookahead-samples must be non-negative"
        )
    if (
        args.seq_len < 2
        or args.shard_rows < 1
        or args.source_batch_rows < 1
        or args.write_batch_rows < 1
        or args.max_buffer_bytes < 1
    ):
        raise ValueError(
            "--seq-len must be >=2 and all source/write buffer limits must be >=1"
        )
    two_pool_rows: _AlternatingSourceRows | None = None
    source_pool_manifest_raw: bytes | None = None
    ci_export_receipt_raw: bytes | None = None
    if args.source_pool_manifest is None:
        if args.data_glob is None:
            raise ValueError(
                "--data-glob is required without --source-pool-manifest"
            )
        if args.objective_seed_root is not None:
            raise ValueError(
                "--objective-seed-root requires --source-pool-manifest"
            )
        shards = sorted(glob.glob(args.data_glob))
        if not shards:
            raise FileNotFoundError(f"no parquet shards match {args.data_glob!r}")
        source_snapshot, source_signatures = _build_source_snapshot(
            shards,
            sequence_length=args.seq_len,
            requested_samples=args.samples,
            seed=args.seed,
            sampling_mode=_BOUNDED_SAMPLING_MODE,
            source_batch_rows=args.source_batch_rows,
            source_root=args.source_root,
        )
        source_iter = _iter_sources(
            shards,
            seed=args.seed,
            source_batch_rows=args.source_batch_rows,
        )
    else:
        if args.data_glob is not None or args.objective_seed_root is None:
            raise ValueError(
                "--source-pool-manifest requires --objective-seed-root and "
                "forbids --data-glob"
            )
        (
            source_snapshot,
            source_signatures,
            primary_shards,
            seed_shards,
            source_pool_manifest_raw,
            ci_export_receipt_raw,
        ) = _build_two_pool_source_snapshot(
            args.source_pool_manifest,
            primary_root=args.source_root,
            objective_seed_root=args.objective_seed_root,
            sequence_length=args.seq_len,
            requested_source_rows=(
                args.samples + quota_lookahead_samples
            ),
            seed=args.seed,
            source_batch_rows=args.source_batch_rows,
        )
        source_iter, two_pool_rows = _iter_two_pool_sources(
            primary_shards,
            seed_shards,
            seed=args.seed,
            source_batch_rows=args.source_batch_rows,
        )

    mixer = EligibilityAwareTaskMixer(
        STAGE1_DEFAULT_RATES,
        seed=args.seed,
        max_input_tokens=args.seq_len,
    )
    graph_relations = tuple(
        relation.strip()
        for relation in args.graph_relations.split(",")
        if relation.strip()
    )
    graph_config = GraphAuxLossConfig(
        relations=graph_relations,
        topk=args.graph_topk,
        bias_beta=graph_recipe["bias_beta"],
        global_weight=args.graph_aux_weight,
        indexer_weight=args.graph_indexer_weight,
        layer_weight=args.graph_layer_weight,
        bce_weight=args.graph_bce_weight,
        coverage_weight=args.graph_coverage_weight,
    )
    accumulator = ObjectiveContractAccumulator(
        rates=mixer.rates,
        seed=args.seed,
        quota_window_samples=args.quota_window_samples,
        graph_config=graph_config,
        graph_weight=args.graph_aux_weight,
    )
    final_output_dir = args.output_dir.resolve()
    with _atomic_output_directory(final_output_dir) as partial_output_dir:
        if source_pool_manifest_raw is not None:
            (partial_output_dir / _SOURCE_POOL_MANIFEST_COPY).write_bytes(
                source_pool_manifest_raw
            )
            assert ci_export_receipt_raw is not None
            (partial_output_dir / _CI_EXPORT_RECEIPT_COPY).write_bytes(
                ci_export_receipt_raw
            )
        writer = _StreamingParquetShardWriter(
            partial_output_dir,
            schema=materialized_schema(),
            capacity=args.seq_len + 1,
            shard_rows=args.shard_rows,
            write_batch_rows=args.write_batch_rows,
            max_buffer_bytes=args.max_buffer_bytes,
        )
        with writer:
            source_selection = _materialize_stream(
                mixer=mixer,
                source_iter=source_iter,
                accumulator=accumulator,
                writer=writer,
                samples=args.samples,
                quota_window_samples=args.quota_window_samples,
                quota_lookahead_samples=quota_lookahead_samples,
                graph_relations=graph_relations,
            )
        _require_source_snapshot_unchanged(source_signatures)
        resume_state = source_selection.get("resume")
        if not isinstance(resume_state, Mapping):
            raise ValueError("objective source selection is missing resume state")
        receipt_cursor = resume_state.get("last_yielded_cursor")
        if not isinstance(receipt_cursor, Mapping):
            raise ValueError("objective source resume state is missing its read cursor")
        if source_iter.last_cursor != receipt_cursor:
            raise ValueError(
                "objective source selection receipt cursor differs from the "
                "iterator read cursor"
            )
        consumed_source_rows = int(source_selection["source_rows_consumed"])
        if two_pool_rows is None:
            _bind_source_sampling_cursor(
                source_snapshot,
                cursor=receipt_cursor,
                consumed_samples=consumed_source_rows,
            )
        else:
            _bind_two_pool_sampling_cursors(
                source_snapshot,
                rows=two_pool_rows,
                global_cursor=receipt_cursor,
                consumed_samples=consumed_source_rows,
            )
        contract = accumulator.finalize()
        contract["source_snapshot"] = source_snapshot
        contract["source_selection"] = source_selection
        _bind_case5_contract_hashes(contract)
        artifact_partial_path = write_objective_materialization_artifact(
            partial_output_dir,
            contract=contract,
            parquet_paths=writer.paths,
        )
        _require_source_snapshot_unchanged(source_signatures)

    artifact_path = final_output_dir / artifact_partial_path.name
    contract_path = final_output_dir / "objective_contract.json"
    print(
        json.dumps(
            {
                "documents": accumulator.samples,
                "input_tokens": contract["totals"]["input_tokens"],  # type: ignore[index]
                "loss_tokens": contract["totals"]["loss_tokens"],  # type: ignore[index]
                "contract": str(contract_path),
                "artifact": str(artifact_path),
                "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
                "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
