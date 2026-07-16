from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from cppmega_mlx.data.symbol_identity import (
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityError,
    compute_symbol_id,
)
from cppmega_mlx.data.domain_schema import (
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
)
from cppmega_mlx.data.tokenizer_contract import (
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY,
    DOMAIN_DELIMITER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)
from scripts.nanochat_data import materialize_tokenized_enriched_parquet as materializer
from scripts.nanochat_data import migrate_clang_commits_v1_to_v12 as migration
from scripts.nanochat_data import pack_enriched_rows as packer


_IDENTITY_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("symbol_id", pa.uint64()),
            pa.field("symbol_key", pa.string()),
        ]
    )
)
_IDENTITY_METADATA = {
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii"): str(
        SYMBOL_IDENTITY_SCHEMA_VERSION
    ).encode("ascii"),
    DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8"): DOMAIN_SCHEMA_SHA256.encode(
        "ascii"
    ),
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode(
        "utf-8"
    ): TOKENIZER_CONTRACT_SHA256.encode("ascii"),
    DOMAIN_DELIMITER_CONTRACT_METADATA_KEY.encode(
        "utf-8"
    ): DOMAIN_DELIMITER_CONTRACT_SHA256.encode("ascii"),
    b"cppmega.test.metadata": b"preserve-me",
}


def _identity(project: str, qname: str) -> tuple[int, str]:
    key = (
        f"usr=c:@F@{qname}#I#|qname={qname}|kind=FUNCTION_DECL|"
        f"project={project}"
    )
    return compute_symbol_id(key), key


def _raw_identity_table(symbol_id: int, symbol_key: str) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("symbol_ids", pa.list_(pa.uint64())),
            pa.field("call_targets", pa.list_(pa.uint64())),
            pa.field("type_refs", pa.list_(pa.uint64())),
            pa.field(SYMBOL_IDENTITIES_COLUMN, _IDENTITY_TYPE),
        ],
        metadata=_IDENTITY_METADATA,
    )
    return pa.Table.from_pylist(
        [
            {
                "text": "int route(int value) { return value; }",
                "symbol_ids": [symbol_id],
                "call_targets": [0],
                "type_refs": [0],
                SYMBOL_IDENTITIES_COLUMN: [
                    {"symbol_id": symbol_id, "symbol_key": symbol_key}
                ],
            }
        ],
        schema=schema,
    )


def _tokenized_identity_table(symbol_id: int, symbol_key: str) -> pa.Table:
    schema = pa.schema(
        [
            pa.field(packer.TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
            pa.field("token_symbol_ids", pa.list_(pa.uint64())),
            pa.field("token_call_targets", pa.list_(pa.uint64())),
            pa.field("token_type_refs", pa.list_(pa.uint64())),
            pa.field("token_def_use", pa.list_(pa.uint8())),
            pa.field(SYMBOL_IDENTITIES_COLUMN, _IDENTITY_TYPE),
        ],
        metadata=_IDENTITY_METADATA,
    )
    return pa.Table.from_pylist(
        [
            {
                packer.TOKEN_IDS_COLUMN: [1, 2],
                "token_symbol_ids": [symbol_id, symbol_id],
                "token_call_targets": [0, 0],
                "token_type_refs": [0, 0],
                "token_def_use": [1, 1],
                SYMBOL_IDENTITIES_COLUMN: [
                    {"symbol_id": symbol_id, "symbol_key": symbol_key}
                ],
            }
        ],
        schema=schema,
    )


def _migration_table(
    symbol_id: int,
    symbol_key: str,
    *,
    symbol_id_type: pa.DataType = pa.uint64(),
) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field(migration.TOKEN_IDS_COLUMN, pa.list_(pa.uint32())),
            pa.field(migration.TOKEN_SYMBOL_IDS_COLUMN, pa.list_(symbol_id_type)),
            pa.field(migration.TOKEN_CALL_TARGETS_COLUMN, pa.list_(pa.uint64())),
            pa.field(migration.TOKEN_TYPE_REFS_COLUMN, pa.list_(pa.uint64())),
            pa.field(SYMBOL_IDENTITIES_COLUMN, _IDENTITY_TYPE),
            pa.field("legacy_receipt", pa.string()),
        ],
        metadata=_IDENTITY_METADATA,
    )
    return pa.Table.from_pylist(
        [
            {
                "text": "route",
                migration.TOKEN_IDS_COLUMN: [1],
                migration.TOKEN_SYMBOL_IDS_COLUMN: [symbol_id],
                migration.TOKEN_CALL_TARGETS_COLUMN: [0],
                migration.TOKEN_TYPE_REFS_COLUMN: [0],
                SYMBOL_IDENTITIES_COLUMN: [
                    {"symbol_id": symbol_id, "symbol_key": symbol_key}
                ],
                "legacy_receipt": "keep",
            }
        ],
        schema=schema,
    )


def _stub_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(materializer, "load_cppmega_tokenizer", lambda _path: object())
    monkeypatch.setattr(
        materializer,
        "materialize_tokenized_enriched_batch",
        lambda docs, _tokenizer: [{} for _ in docs],
    )
    monkeypatch.setattr(
        materializer,
        "_merge_table_with_tokenized",
        lambda table, _rows: table,
    )


def test_packer_late_identity_failure_preserves_published_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol_id, symbol_key = _identity("owner/repo", "route")
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "packed.parquet"
    pq.write_table(_tokenized_identity_table(symbol_id, symbol_key), input_path)
    output_path.write_bytes(b"previously-published")

    doc = packer.normalize_document_record(
        {
            packer.TOKEN_IDS_COLUMN: [1, 2],
            "token_symbol_ids": [0, 0],
            "token_call_targets": [0, 0],
            "token_type_refs": [0, 0],
            "token_def_use": [0, 0],
        },
        source_doc_index=0,
    )

    def fail_after_first_window(*_args, **_kwargs):
        yield doc
        raise SymbolIdentityError("late symbol ID collision")

    monkeypatch.setattr(packer, "iter_tokenized_documents", fail_after_first_window)

    with pytest.raises(SymbolIdentityError, match="late symbol ID collision"):
        packer.pack_parquet_dataset(
            input_path,
            output_path,
            target_length=4,
            pack_token_window=2,
        )

    assert output_path.read_bytes() == b"previously-published"
    assert not list(tmp_path.glob(".packed.*.staged.parquet"))


def test_materializer_max_files_never_copies_complete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol_id, symbol_key = _identity("owner/repo", "route")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for index in range(2):
        pq.write_table(
            _raw_identity_table(symbol_id, symbol_key),
            input_dir / f"shard_{index:05d}.parquet",
        )
    (input_dir / "_COMPLETE").write_text("complete\n", encoding="utf-8")
    _stub_materializer(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_tokenized_enriched_parquet.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--max-files",
            "1",
            "--tokenizer-path",
            "unused.json",
        ],
    )

    materializer.main()

    assert [path.name for path in output_dir.glob("*.parquet")] == [
        "shard_00000.parquet"
    ]
    assert not (output_dir / "_COMPLETE").exists()


def test_materializer_late_collision_preserves_published_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol_id, first_key = _identity("owner/repo-a", "route")
    _, second_key = _identity("owner/repo-b", "route")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    pq.write_table(
        _raw_identity_table(symbol_id, first_key),
        input_dir / "a.parquet",
    )
    pq.write_table(
        _raw_identity_table(symbol_id, second_key),
        input_dir / "z.parquet",
    )
    (input_dir / "_COMPLETE").write_text("new\n", encoding="utf-8")
    (output_dir / "_COMPLETE").write_text("old\n", encoding="utf-8")
    (output_dir / "keep.txt").write_text("published\n", encoding="utf-8")
    _stub_materializer(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_tokenized_enriched_parquet.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--tokenizer-path",
            "unused.json",
        ],
    )

    with pytest.raises(SymbolIdentityError, match="does not match"):
        materializer.main()

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "_COMPLETE",
        "keep.txt",
    ]
    assert (output_dir / "_COMPLETE").read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".output.*.staged"))


def test_migration_requires_canonical_uint64_identity_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    symbol_id, symbol_key = _identity("owner/repo", "route")
    source = _migration_table(symbol_id, symbol_key)

    migrated = migration.migrate_table(source)

    assert migrated.schema.metadata == source.schema.metadata
    assert migrated.column("legacy_receipt").to_pylist() == ["keep"]
    assert (
        migrated.schema.field(migration.TOKEN_SYMBOL_IDS_COLUMN).type.value_type
        == pa.uint64()
    )
    migrated_path = tmp_path / "migrated.parquet"
    pq.write_table(migrated, migrated_path)
    packer._require_symbol_identity_schema(migrated_path)

    with pytest.raises(SymbolIdentityError, match="missing_columns"):
        migration.migrate_table(pa.table({"text": ["legacy"]}))
    with pytest.raises(SymbolIdentityError, match="preserve uint64"):
        migration.migrate_table(
            _migration_table(1, symbol_key, symbol_id_type=pa.uint32())
        )


def test_migration_late_identity_failure_preserves_published_directory(
    tmp_path: Path,
) -> None:
    symbol_id, symbol_key = _identity("owner/repo", "route")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    pq.write_table(
        _migration_table(symbol_id, symbol_key),
        input_dir / "a_valid.parquet",
    )
    pq.write_table(
        pa.table({"text": ["legacy-without-identity"]}),
        input_dir / "z_invalid.parquet",
    )
    (input_dir / "dataset.json").write_text('{"source":"legacy"}\n', encoding="utf-8")
    (output_dir / "_COMPLETE").write_text("old\n", encoding="utf-8")
    (output_dir / "keep.txt").write_text("published\n", encoding="utf-8")

    with pytest.raises(SymbolIdentityError, match="missing_columns"):
        migration.migrate_directory(input_dir, output_dir)

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "_COMPLETE",
        "keep.txt",
    ]
    assert (output_dir / "_COMPLETE").read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".output.*.staged"))
