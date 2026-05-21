from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from cppmega_mlx.data.nanochat_pipeline import platform_vocab
from cppmega_mlx.data.nanochat_pipeline import tokenized_enriched_schema as schema
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)
from scripts.nanochat_data import clang_enriched_to_parquet
from scripts.nanochat_data.token_budget import chunk_enriched_document, load_tokenizer
from tools.clang_indexer.index_project import FunctionDef, PartInfo
from tools.clang_indexer.process_commits import FileAnalysis, _build_enriched_from_parts


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]


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


def test_converter_header_alignment_preserves_char_metadata_coordinates() -> None:
    record = {
        "text": "abc",
        "structure_ids": [1, 2, 3],
        "chunk_boundaries": [{"start": 0, "end": 3, "kind": 3, "dep_level": 0}],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [4, 5, 6],
        "sibling_index": [7, 8, 9],
        "ast_node_type": [10, 11, 12],
        "symbol_ids": [13, 14, 15],
        "call_targets": [16, 17, 18],
        "type_refs": [19, 20, 21],
        "def_use": [1, 2, 1],
    }

    docs = clang_enriched_to_parquet.process_record_with_policy(
        record,
        _CharTokenizer(),
        max_tokens=4096,
        overflow_policy="drop",
    )

    assert len(docs) == 1
    doc = docs[0]
    header_len = len(doc["text"]) - len(record["text"])
    assert header_len > 0
    assert doc["structure_ids"] == [0] * header_len + record["structure_ids"]
    assert doc["platform_info"] == clang_enriched_to_parquet._DEFAULT_PLATFORM_INFO
    for key in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
    ):
        assert doc[key] == [0] * header_len + record[key]


def test_token_budget_slices_semantic_char_metadata() -> None:
    doc = {
        "text": "abcdef",
        "structure_ids": [1, 2, 3, 4, 5, 6],
        "chunk_boundaries": [],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [10, 11, 12, 13, 14, 15],
        "sibling_index": [20, 21, 22, 23, 24, 25],
        "ast_node_type": [30, 31, 32, 33, 34, 35],
        "symbol_ids": [40, 41, 42, 43, 44, 45],
        "call_targets": [50, 51, 52, 53, 54, 55],
        "type_refs": [60, 61, 62, 63, 64, 65],
        "def_use": [0, 1, 2, 0, 1, 2],
    }

    pieces = chunk_enriched_document(doc, max_tokens=3, tokenizer=_CharTokenizer())

    assert [piece["text"] for piece in pieces] == ["abc", "def"]
    assert pieces[0]["symbol_ids"] == [40, 41, 42]
    assert pieces[1]["symbol_ids"] == [43, 44, 45]
    assert pieces[0]["def_use"] == [0, 1, 2]
    assert pieces[1]["def_use"] == [0, 1, 2]


def test_commit_enriched_builder_emits_semantic_columns_without_libclang_runtime() -> None:
    helper_text = "int helper() { return 1; }"
    main_text = "int main() { return helper(); }"
    helper = FunctionDef(
        "helper",
        "helper",
        "src/demo.cc",
        1,
        helper_text,
        [],
    )
    main = FunctionDef(
        "main",
        "main",
        "src/demo.cc",
        3,
        main_text,
        ["helper"],
    )
    analysis = FileAnalysis("", functions=[helper, main])
    parts: list[PartInfo] = [
        (helper_text, 3, 0, "helper", "helper"),
        (main_text, 3, 1, "main", "main"),
    ]

    doc = _build_enriched_from_parts(
        parts,
        analysis,
        None,
        {"filepath": "src/demo.cc"},
    )

    text_len = len(doc["text"])
    for key in ("symbol_ids", "call_targets", "type_refs", "def_use"):
        assert len(doc[key]) == text_len
    assert any(doc["symbol_ids"])
    assert any(doc["call_targets"])
    assert any(value == 1 for value in doc["def_use"])


def test_commit_enriched_builder_emits_temporal_char_annotations() -> None:
    old_text = "int main() { return 1; }"
    new_text = "int main() { return 2; }"
    old_main = FunctionDef("main", "main", "src/demo.cc", 1, old_text, [])
    new_main = FunctionDef("main", "main", "src/demo.cc", 1, new_text, [])
    diff = "\n".join(
        [
            "diff --git a/src/demo.cc b/src/demo.cc",
            "--- a/src/demo.cc",
            "+++ b/src/demo.cc",
            "@@ -1 +1 @@",
            "-int main() { return 1; }",
            "+int main() { return 2; }",
        ]
    )
    parts: list[PartInfo] = [
        ("// === PRE-COMMIT ===", 0, 0, "", None),
        (old_text, 3, 0, "main", "main"),
        ("// === POST-COMMIT ===", 0, 0, "", None),
        (new_text, 3, 0, "main", "main"),
    ]

    doc = _build_enriched_from_parts(
        parts,
        FileAnalysis("", functions=[old_main]),
        FileAnalysis("", functions=[new_main]),
        {
            "filepath": "src/demo.cc",
            "old_content": old_text,
            "new_content": new_text,
            "diff": diff,
        },
        section_kinds=["c", "o", "c", "n"],
    )

    assert len(doc["change_mask_pre"]) == len(doc["text"])
    assert len(doc["change_mask_post"]) == len(doc["text"])
    assert any(doc["change_mask_pre"])
    assert any(doc["change_mask_post"])
    assert any(value == 2 for value in doc["edit_op_per_char"])
    assert any(doc["hunk_id_per_char"])


def test_tokenized_materializer_maps_temporal_char_annotations_to_tokens() -> None:
    text = "int main() { return 2; }"
    change_mask = [0] * len(text)
    edit_ops = [3] * len(text)
    start = text.index("2")
    change_mask[start] = 1
    edit_ops[start] = 2
    docs = [
        {
            "text": text,
            "structure_ids": [3] * len(text),
            "chunk_boundaries": [{"start": 0, "end": len(text), "kind": 3, "dep_level": 0}],
            "call_edges": [],
            "type_edges": [],
            "change_mask_post": change_mask,
            "hunk_id_per_char": change_mask,
            "edit_op_per_char": edit_ops,
        }
    ]

    row = materialize_tokenized_enriched_batch(docs, load_tokenizer())[0]

    assert any(row[schema.TOKEN_CHANGE_MASK_POST_COLUMN])
    assert any(value == 2 for value in row[schema.EDIT_OP_PER_TOKEN_COLUMN])
    assert row[schema.CHANGED_CHUNK_IDS_COLUMN] == [0]
    assert row[schema.CHANGED_CHUNK_SPANS_COLUMN]
