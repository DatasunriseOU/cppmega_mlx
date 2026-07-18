#!/usr/bin/env python3
# ruff: noqa: E402
"""Migrate identity-complete clang commit parquet files to the v12 layout.

Reads existing shards, validates their canonical symbol identity channels, adds
missing non-identity v12 columns with appropriate null/empty defaults, and
writes upgraded v12 parquet files. Legacy shards without canonical identity are
rejected because those identities cannot be reconstructed from numeric IDs.

The v12 schema is defined in ``scripts/data/clang_enriched_to_4k_parquet.py``
and corresponds to the unified clang semantic family that covers both static
code and temporal commit data.

This script is idempotent: re-running it on already-migrated v12 files preserves
existing columns and metadata.

Usage:
    # Migrate local directory
    python -m scripts.data.migrate_clang_commits_v1_to_v12 \
        --input_dir /path/to/clang_commits_4k_v1 \
        --output_dir /path/to/clang_commits_4k_v12

    # In-place migration (output_dir == input_dir)
    python -m scripts.data.migrate_clang_commits_v1_to_v12 \
        --input_dir /path/to/clang_commits_4k_v1 \
        --output_dir /path/to/clang_commits_4k_v1

    # Dry run (report missing columns without writing)
    python -m scripts.data.migrate_clang_commits_v1_to_v12 \
        --input_dir /path/to/clang_commits_4k_v1 \
        --output_dir /path/to/clang_commits_4k_v12 \
        --dry_run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    AUTHOR_TIMESTAMP_COLUMN,
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    COMMIT_HASH_COLUMN,
    COMMIT_TIMESTAMP_COLUMN,
    EDIT_OP_PER_TOKEN_COLUMN,
    FILEPATH_COLUMN,
    FILEPATH_STABLE_ID_COLUMN,
    FILE_LOCAL_COMMIT_INDEX_COLUMN,
    HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN,
    HAS_RENAME_AMBIGUITY_COLUMN,
    HUNK_ID_PER_TOKEN_COLUMN,
    IS_MERGE_COMMIT_COLUMN,
    PARENT_COUNT_COLUMN,
    PARENT_HASHES_COLUMN,
    PLATFORM_IDS_COLUMN,
    REPO_COLUMN,
    REPO_STABLE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_AST_NODE_TYPE_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CHANGE_MASK_POST_COLUMN,
    TOKEN_CHANGE_MASK_PRE_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_SIBLING_INDEX_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_SYMBOL_IDS_COLUMN,
    TOKEN_CALL_TARGETS_COLUMN,
    TOKEN_TYPE_REFS_COLUMN,
    TOKEN_DEF_USE_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
)
from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityError,
    SymbolIdentityRegistry,
)
from scripts.nanochat_data.atomic_publish import (
    atomic_output_directory,
    atomic_output_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("migrate_clang_commits_v1_to_v12")

# ---------------------------------------------------------------------------
# Target v12 schema — matches clang_enriched_to_4k_parquet._SCHEMA plus
# the four semantic token columns from the v12 contract.
# ---------------------------------------------------------------------------

V12_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("actual_token_count", pa.int32()),
    pa.field("structure_ids", pa.list_(pa.int8())),
    pa.field("chunk_boundaries", pa.list_(pa.struct([
        pa.field("start", pa.int32()),
        pa.field("end", pa.int32()),
        pa.field("kind", pa.int8()),
        pa.field("dep_level", pa.int32()),
        pa.field("name", pa.string()),
    ]))),
    pa.field("call_edges", pa.list_(pa.struct([
        pa.field("from", pa.int32()),
        pa.field("to", pa.int32()),
    ]))),
    pa.field("type_edges", pa.list_(pa.struct([
        pa.field("from", pa.int32()),
        pa.field("to", pa.int32()),
    ]))),
    pa.field("platform_info", pa.string()),
    pa.field("language_info", pa.string()),
    pa.field("build_info", pa.string()),
    pa.field("constituent_provenance", pa.list_(pa.struct([
        pa.field("filepath", pa.string()),
        pa.field("language_info", pa.string()),
        pa.field("build_info", pa.string()),
    ]))),
    pa.field("constituent_provenance_json", pa.string()),
    # Chronology
    pa.field(REPO_COLUMN, pa.string()),
    pa.field(FILEPATH_COLUMN, pa.string()),
    pa.field(COMMIT_HASH_COLUMN, pa.string()),
    pa.field(TIMESTAMP_COLUMN, pa.string()),
    pa.field(PARENT_HASHES_COLUMN, pa.list_(pa.string())),
    pa.field(PARENT_COUNT_COLUMN, pa.int32()),
    pa.field(IS_MERGE_COMMIT_COLUMN, pa.bool_()),
    pa.field(AUTHOR_TIMESTAMP_COLUMN, pa.string()),
    pa.field(COMMIT_TIMESTAMP_COLUMN, pa.string()),
    pa.field(REPO_STABLE_ID_COLUMN, pa.string()),
    pa.field(FILEPATH_STABLE_ID_COLUMN, pa.string()),
    pa.field(FILE_LOCAL_COMMIT_INDEX_COLUMN, pa.int32()),
    pa.field(HAS_AMBIGUOUS_RECONSTRUCTION_COLUMN, pa.bool_()),
    pa.field(HAS_RENAME_AMBIGUITY_COLUMN, pa.bool_()),
    # Token-aligned arrays
    pa.field(TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(PLATFORM_IDS_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_STRUCTURE_IDS_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_AST_DEPTH_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_SIBLING_INDEX_COLUMN, pa.list_(pa.uint16())),
    pa.field(TOKEN_AST_NODE_TYPE_COLUMN, pa.list_(pa.uint16())),
    # Semantic token columns (v12 additions)
    pa.field(TOKEN_SYMBOL_IDS_COLUMN, pa.list_(pa.uint64())),
    pa.field(TOKEN_CALL_TARGETS_COLUMN, pa.list_(pa.uint64())),
    pa.field(TOKEN_TYPE_REFS_COLUMN, pa.list_(pa.uint64())),
    pa.field(SYMBOL_IDENTITIES_COLUMN, pa.list_(pa.struct([
        pa.field("symbol_id", pa.uint64()),
        pa.field("symbol_key", pa.string()),
    ]))),
    pa.field(TOKEN_DEF_USE_COLUMN, pa.list_(pa.uint8())),
    # Temporal token columns
    pa.field(TOKEN_CHANGE_MASK_PRE_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_CHANGE_MASK_POST_COLUMN, pa.list_(pa.uint8())),
    pa.field(HUNK_ID_PER_TOKEN_COLUMN, pa.list_(pa.int32())),
    pa.field(EDIT_OP_PER_TOKEN_COLUMN, pa.list_(pa.uint8())),
    # Chunk metadata
    pa.field(TOKEN_CHUNK_STARTS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_CHUNK_ENDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(TOKEN_CHUNK_KINDS_COLUMN, pa.list_(pa.uint8())),
    pa.field(TOKEN_CHUNK_DEP_LEVELS_COLUMN, pa.list_(pa.uint16())),
    pa.field(CHANGED_CHUNK_IDS_COLUMN, pa.list_(pa.uint32())),
    pa.field(
        CHANGED_CHUNK_SPANS_COLUMN,
        pa.list_(
            pa.struct([
                pa.field("start", pa.uint32()),
                pa.field("end", pa.uint32()),
            ])
        ),
    ),
    pa.field(TOKEN_CALL_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint16()),
        pa.field("to", pa.uint16()),
    ]))),
    pa.field(TOKEN_TYPE_EDGES_COLUMN, pa.list_(pa.struct([
        pa.field("from", pa.uint16()),
        pa.field("to", pa.uint16()),
    ]))),
])

# Map of column name -> default value factory for missing columns.
# Lists get empty-list defaults; scalars get null/zero/False.
_EMPTY_LIST_COLUMNS: dict[str, pa.DataType] = {}
_SCALAR_DEFAULTS: dict[str, tuple[pa.DataType, object]] = {}

for field in V12_SCHEMA:
    if pa.types.is_list(field.type):
        _EMPTY_LIST_COLUMNS[field.name] = field.type
    elif pa.types.is_string(field.type):
        _SCALAR_DEFAULTS[field.name] = (field.type, None)
    elif pa.types.is_boolean(field.type):
        _SCALAR_DEFAULTS[field.name] = (field.type, False)
    elif pa.types.is_integer(field.type):
        _SCALAR_DEFAULTS[field.name] = (field.type, None)


def _make_null_column(
    name: str, pa_type: pa.DataType, num_rows: int
) -> pa.Array:
    """Create a column of nulls or empty lists for the given type."""
    if pa.types.is_list(pa_type):
        # Empty list for each row
        return pa.array([[] for _ in range(num_rows)], type=pa_type)
    if pa.types.is_boolean(pa_type):
        return pa.array([False] * num_rows, type=pa_type)
    # Null for string/int scalars
    return pa.array([None] * num_rows, type=pa_type)


def migrate_table(
    table: pa.Table,
    *,
    corpus_registry: SymbolIdentityRegistry | None = None,
) -> pa.Table:
    """Add non-identity v12 columns to an identity-complete table.

    Existing columns and metadata are preserved, and extra columns remain at the
    end. Canonical identity columns are mandatory and are never synthesized.
    """
    existing_names = set(table.column_names)
    num_rows = table.num_rows
    semantic_columns = (
        TOKEN_SYMBOL_IDS_COLUMN,
        TOKEN_CALL_TARGETS_COLUMN,
        TOKEN_TYPE_REFS_COLUMN,
    )
    required_identity_columns = {*semantic_columns, SYMBOL_IDENTITIES_COLUMN}
    missing_identity = sorted(required_identity_columns - existing_names)
    if missing_identity:
        raise SymbolIdentityError(
            "legacy parquet cannot be promoted to v12 because canonical identity "
            f"cannot be reconstructed; missing_columns={missing_identity}"
        )
    raw_version = (table.schema.metadata or {}).get(
        SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    )
    expected_version = str(SYMBOL_IDENTITY_SCHEMA_VERSION).encode("ascii")
    if raw_version != expected_version:
        raise SymbolIdentityError(
            "legacy parquet cannot be promoted to v12 without canonical identity "
            f"metadata; got={raw_version!r}, expected={expected_version!r}"
        )
    for name in semantic_columns:
        field_type = table.schema.field(name).type
        if not pa.types.is_list(field_type) or field_type.value_type != pa.uint64():
            raise SymbolIdentityError(
                f"{name} must preserve uint64 semantic identity, got {field_type}"
            )
    identities_type = table.schema.field(SYMBOL_IDENTITIES_COLUMN).type
    if not pa.types.is_list(identities_type):
        raise SymbolIdentityError(
            f"{SYMBOL_IDENTITIES_COLUMN} must be a list, got {identities_type}"
        )
    registry = corpus_registry or SymbolIdentityRegistry()
    identities = table.column(SYMBOL_IDENTITIES_COLUMN).to_pylist()
    semantic_rows = {
        name: table.column(name).to_pylist() for name in semantic_columns
    }
    for row_index, claims in enumerate(identities):
        row_registry = SymbolIdentityRegistry()
        normalized = row_registry.register_records(
            claims, source=f"migration row {row_index}"
        )
        registry.register_records(normalized, source=f"migration row {row_index}")
        used_ids = {
            int(value)
            for values in semantic_rows.values()
            for value in (values[row_index] or [])
            if int(value) != 0
        }
        row_registry.require_ids(used_ids, source=f"migration row {row_index}")

    # Build output columns in v12 schema order
    columns: dict[str, pa.Array] = {}
    added_columns: list[str] = []

    for field in V12_SCHEMA:
        if field.name in existing_names:
            # Keep existing column, cast if needed
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type, safe=True)
            columns[field.name] = col
        else:
            # Add missing column with defaults
            columns[field.name] = _make_null_column(
                field.name, field.type, num_rows
            )
            added_columns.append(field.name)

    # Preserve extra columns that exist in source but not in v12 schema
    # (e.g., changed_symbol_ids, ripple_candidates from v1 raw output)
    extra_columns: dict[str, pa.Array] = {}
    for name in table.column_names:
        if name not in {f.name for f in V12_SCHEMA}:
            extra_columns[name] = table.column(name)

    if added_columns:
        log.info("  Added %d missing columns: %s", len(added_columns), added_columns)

    if extra_columns:
        log.info(
            "  Preserved %d extra v1 columns: %s",
            len(extra_columns),
            list(extra_columns.keys()),
        )

    # Build the output table: v12 columns first, then extras
    all_names = list(columns.keys()) + list(extra_columns.keys())
    all_arrays = list(columns.values()) + list(extra_columns.values())
    migrated = pa.Table.from_arrays(all_arrays, names=all_names)
    return migrated.replace_schema_metadata(table.schema.metadata)


def migrate_shard(
    input_path: Path,
    output_path: Path,
    *,
    dry_run: bool = False,
    row_group_size: int = 1024,
    corpus_registry: SymbolIdentityRegistry | None = None,
) -> dict:
    """Migrate a single parquet shard from v1 to v12.

    Returns a summary dict with stats about the migration.
    """
    pf = pq.ParquetFile(str(input_path))
    table = pf.read()
    existing_columns = set(table.column_names)
    v12_columns = {f.name for f in V12_SCHEMA}
    missing = sorted(v12_columns - existing_columns)
    extra = sorted(existing_columns - v12_columns)

    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_rows": table.num_rows,
        "num_row_groups": pf.num_row_groups,
        "existing_columns": len(existing_columns),
        "v12_columns": len(v12_columns),
        "missing_columns": missing,
        "extra_columns": extra,
        "dry_run": dry_run,
    }

    migrated = migrate_table(table, corpus_registry=corpus_registry)

    if dry_run:
        log.info(
            "  [DRY RUN] %s: %d rows, %d missing columns, %d extra columns",
            input_path.name,
            table.num_rows,
            len(missing),
            len(extra),
        )
        return stats

    if not missing:
        log.info("  %s: already v12, no migration needed", input_path.name)
        if input_path != output_path:
            with atomic_output_file(output_path) as staged_output:
                pq.write_table(
                    migrated, str(staged_output), row_group_size=row_group_size
                )
        stats["action"] = "copied" if input_path != output_path else "skipped"
        return stats

    with atomic_output_file(output_path) as staged_output:
        pq.write_table(migrated, str(staged_output), row_group_size=row_group_size)
        check = pq.ParquetFile(str(staged_output))
        if check.metadata.num_rows != table.num_rows:
            raise RuntimeError(
                f"row count changed during migration: {table.num_rows} -> "
                f"{check.metadata.num_rows}"
            )
        if check.num_row_groups:
            check.read_row_group(0)

    log.info(
        "  %s: migrated %d rows, added %d columns",
        input_path.name,
        table.num_rows,
        len(missing),
    )
    stats["action"] = "migrated"
    return stats


def migrate_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    row_group_size: int = 1024,
) -> list[dict]:
    """Migrate all parquet shards in a directory from v1 to v12.

    Returns a list of per-shard stats dicts.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Collect parquet files (train shards + val shard)
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    log.info(
        "Found %d parquet files in %s", len(parquet_files), input_dir
    )

    def migrate_into(target_dir: Path) -> tuple[list[dict], int]:
        all_stats: list[dict] = []
        total_rows = 0
        corpus_registry = SymbolIdentityRegistry()
        for pf_path in parquet_files:
            out_path = target_dir / pf_path.name
            stats = migrate_shard(
                pf_path,
                out_path,
                dry_run=dry_run,
                row_group_size=row_group_size,
                corpus_registry=corpus_registry,
            )
            all_stats.append(stats)
            total_rows += stats["num_rows"]
        return all_stats, total_rows

    if dry_run:
        all_stats, total_rows = migrate_into(output_dir)
    else:
        with atomic_output_directory(output_dir) as staged_output:
            all_stats, total_rows = migrate_into(staged_output)
            for item in input_dir.iterdir():
                if (
                    item.is_file()
                    and not item.name.endswith(".parquet")
                    and item.name != "_COMPLETE"
                ):
                    shutil.copy2(item, staged_output / item.name)
            (staged_output / "_COMPLETE").write_text(
                f"Migrated from {input_dir.name} to v12 schema.\n"
                f"{len(parquet_files)} shards, {total_rows} total rows.\n",
                encoding="utf-8",
            )

    log.info(
        "Migration %s: %d shards, %d total rows",
        "preview" if dry_run else "complete",
        len(parquet_files),
        total_rows,
    )

    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate clang_commits_4k_v1 parquet to v12 schema"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to v1 parquet directory (e.g., clang_commits_4k_v1)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output v12 directory (may equal input_dir for in-place)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report missing columns without writing",
    )
    parser.add_argument(
        "--row_group_size",
        type=int,
        default=1024,
        help="Row group size for output parquet (default: 1024)",
    )
    args = parser.parse_args()

    t0 = time.time()
    stats = migrate_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        dry_run=args.dry_run,
        row_group_size=args.row_group_size,
    )
    elapsed = time.time() - t0

    # Print summary
    all_missing = set()
    all_extra = set()
    for s in stats:
        all_missing.update(s["missing_columns"])
        all_extra.update(s["extra_columns"])

    if all_missing:
        log.info("Columns added across all shards: %s", sorted(all_missing))
    if all_extra:
        log.info("Extra v1 columns preserved: %s", sorted(all_extra))
    log.info("Elapsed: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
