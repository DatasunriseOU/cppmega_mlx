from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from cppmega_mlx.data.nanochat_pipeline import platform_vocab
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from scripts.nanochat_data import clang_enriched_to_parquet


def _load_nanochat_module(relative_path: str) -> ModuleType:
    source = Path("/Users/dave/sources/nanochat") / relative_path
    if not source.exists():
        pytest.skip(f"nanochat source file is not available: {source}")
    spec = importlib.util.spec_from_file_location(
        f"_nanochat_{source.stem}",
        source,
    )
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot import nanochat source file: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_platform_vocab_matches_nanochat_source_of_truth() -> None:
    nanochat_vocab = _load_nanochat_module("nanochat/platform_vocab.py")

    assert platform_vocab.PLATFORM_VOCAB == nanochat_vocab.PLATFORM_VOCAB
    assert platform_vocab.PLATFORM_VOCAB_SIZE == nanochat_vocab.PLATFORM_VOCAB_SIZE
    assert platform_vocab.MAX_PLATFORM_IDS == nanochat_vocab.MAX_PLATFORM_IDS


def test_local_tokenized_schema_keeps_full_nanochat_column_contract() -> None:
    nanochat_schema = _load_nanochat_module("nanochat/tokenized_enriched_schema.py")

    assert schema.TOKENIZED_ENRICHED_COLUMNS == nanochat_schema.TOKENIZED_ENRICHED_COLUMNS


def test_clang_enriched_parquet_schema_preserves_token_semantic_columns() -> None:
    required = {
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        schema.TOKEN_SYMBOL_IDS_COLUMN,
        schema.TOKEN_CALL_TARGETS_COLUMN,
        schema.TOKEN_TYPE_REFS_COLUMN,
        schema.TOKEN_DEF_USE_COLUMN,
    }

    assert required <= set(clang_enriched_to_parquet._SCHEMA.names)


def test_clang_enriched_docs_to_table_carries_token_semantic_columns() -> None:
    rows = [
        {
            "text": "int main() { return f(); }",
            "actual_token_count": 4,
            "structure_ids": [3, 3, 3, 3],
            "chunk_boundaries": [{"start": 0, "end": 24, "kind": 3, "dep_level": 0}],
            "call_edges": [],
            "type_edges": [],
            "ast_depth": [0, 1, 2, 1],
            "sibling_index": [0, 0, 1, 2],
            "ast_node_type": [1, 2, 3, 4],
            "symbol_ids": [0, 11, 11, 0],
            "call_targets": [0, 22, 0, 0],
            "type_refs": [0, 0, 33, 0],
            "def_use": [0, 1, 2, 0],
        }
    ]
    tokenized_rows = [
        {
            schema.TOKEN_IDS_COLUMN: [1, 2, 3, 4],
            schema.TOKEN_SYMBOL_IDS_COLUMN: [0, 11, 11, 0],
            schema.TOKEN_CALL_TARGETS_COLUMN: [0, 22, 0, 0],
            schema.TOKEN_TYPE_REFS_COLUMN: [0, 0, 33, 0],
            schema.TOKEN_DEF_USE_COLUMN: [0, 1, 2, 0],
        }
    ]

    table = clang_enriched_to_parquet.rows_to_table(
        rows,
        tokenized_rows=tokenized_rows,
    )

    for column in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        schema.TOKEN_SYMBOL_IDS_COLUMN,
        schema.TOKEN_CALL_TARGETS_COLUMN,
        schema.TOKEN_TYPE_REFS_COLUMN,
        schema.TOKEN_DEF_USE_COLUMN,
    ):
        expected = (
            tokenized_rows[0][column] if column in tokenized_rows[0] else rows[0][column]
        )
        assert table.column(column).to_pylist() == [expected]
