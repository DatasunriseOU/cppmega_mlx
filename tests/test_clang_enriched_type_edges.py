"""Verify clang_enriched_to_parquet propagates real type_edges to parquet.

These tests feed a tiny row dict (the shape build_enriched_doc emits) through
the converter's rows_to_table() edge-handling code path and assert:
  - the parquet ``type_edges`` column is non-empty for a row that carries edges,
  - it has the expected struct dtype (list<struct<from:int32, to:int32>>),
  - the {from,to} integer endpoints round-trip through PyArrow intact,
  - the semantic char-level columns (symbol_ids / call_targets / type_refs /
    def_use) also propagate from the row dict into their typed columns.

No libclang is required: we exercise only the pure converter.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pyarrow as pa  # type: ignore[import-not-found]
import pytest

from cppmega_mlx.data.symbol_identity import compute_symbol_id

# Make the cppmega.mlx ``scripts.nanochat_data`` package importable regardless
# of pytest's rootdir / pythonpath. When this test is collected together with
# tests from a *different* repo (e.g. the nanochat indexer tests), pytest picks
# that repo's pyproject as the configfile / rootdir, so cppmega.mlx's
# ``pythonpath = ["."]`` is never applied. The nanochat repo additionally ships
# a *regular* top-level ``scripts`` package (its own ``__init__.py``), whereas
# cppmega.mlx's ``scripts`` is a namespace package (no ``__init__.py``). When
# both repo roots are on sys.path, ``import scripts`` binds to nanochat's
# regular package and the cppmega.mlx ``scripts.nanochat_data`` (plus the
# converter's own internal ``from scripts.nanochat_data.token_budget import``)
# become permanently unreachable — a namespace portion is never merged into a
# regular package. We therefore force-register cppmega.mlx's ``scripts`` and
# ``scripts.nanochat_data`` as namespace packages whose ``__path__`` points at
# this repo, BEFORE importing the converter, so every ``scripts.nanochat_data.*``
# import resolves against cppmega.mlx. (cppmega_mlx.* deps still require the
# repo root on sys.path — provided by pytest's pythonpath when run alone, or by
# PYTHONPATH when run cross-repo.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_NANOCHAT_DATA_DIR = _SCRIPTS_DIR / "nanochat_data"


def _bind_namespace_pkg(name: str, location: Path) -> None:
    """(Re)bind ``name`` to a namespace package rooted at ``location``."""
    existing = sys.modules.get(name)
    existing_paths = list(getattr(existing, "__path__", []) or [])
    if existing is not None and str(location) in existing_paths:
        return  # already correctly bound to this repo
    mod = types.ModuleType(name)
    mod.__path__ = [str(location)]  # type: ignore[attr-defined]
    sys.modules[name] = mod


_bind_namespace_pkg("scripts", _SCRIPTS_DIR)
_bind_namespace_pkg("scripts.nanochat_data", _NANOCHAT_DATA_DIR)

conv = importlib.import_module("scripts.nanochat_data.clang_enriched_to_parquet")


def _minimal_row(**overrides):
    """A row dict with just enough fields for rows_to_table to build a table."""
    row = {
        "symbol_identity_schema_version": 3,
        "symbol_identities": [],
        "text": "struct B{int x;};\n\nint D::f(){return x;}",
        "source_doc_id": "doc-1",
        "tokenizer_fingerprint": "fp-1",
        "actual_token_count": 7,
        "structure_ids": [],
        "chunk_boundaries": [],
        "call_edges": [],
        "type_edges": [],
        "ast_depth": [],
        "sibling_index": [],
        "ast_node_type": [],
        "symbol_ids": [],
        "call_targets": [],
        "type_refs": [],
        "def_use": [],
    }
    row.update(overrides)
    return row


def test_type_edges_column_populated_with_struct_dtype():
    row = _minimal_row(type_edges=[{"from": 1, "to": 0}, {"from": 1, "to": 0}])
    table = conv.rows_to_table([row])

    assert "type_edges" in table.column_names
    col = table.column("type_edges")

    # Expected dtype: list<struct<from:int32, to:int32>>.
    field = conv._SCHEMA.field("type_edges")
    assert col.type == field.type
    assert pa.types.is_list(col.type)
    struct_t = col.type.value_type
    assert pa.types.is_struct(struct_t)
    names = {struct_t.field(i).name: struct_t.field(i).type for i in range(struct_t.num_fields)}
    assert names["from"] == pa.int32()
    assert names["to"] == pa.int32()

    # Non-empty and endpoints round-trip.
    edges = col.to_pylist()[0]
    assert edges, "type_edges column must be non-empty for a row that carries edges"
    assert {"from": 1, "to": 0} in edges
    for e in edges:
        assert isinstance(e["from"], int) and isinstance(e["to"], int)


def test_type_edges_empty_when_row_has_none():
    table = conv.rows_to_table([_minimal_row(type_edges=[])])
    assert table.column("type_edges").to_pylist()[0] == []


def test_type_edges_coerces_non_int_endpoints():
    # The converter casts endpoints via int(); strings/floats must coerce, and
    # the schema dtype must still hold (int32 struct fields).
    row = _minimal_row(type_edges=[{"from": "2", "to": 3.0}])
    table = conv.rows_to_table([row])
    edges = table.column("type_edges").to_pylist()[0]
    assert edges == [{"from": 2, "to": 3}]
    assert table.column("type_edges").type == conv._SCHEMA.field("type_edges").type


def test_semantic_char_columns_propagate():
    keys = [f"usr:schema=v3\x1fproject=test\x1fusr=c:@F@edge{i}#" for i in range(3)]
    ids = [compute_symbol_id(key) for key in keys]
    row = _minimal_row(
        symbol_identities=[
            {"symbol_id": symbol_id, "symbol_key": key}
            for key, symbol_id in zip(keys, ids, strict=True)
        ],
        symbol_ids=[ids[0], ids[0], 0],
        call_targets=[0, 0, ids[1]],
        type_refs=[ids[2], 0, 0],
        def_use=[1, 1, 2],
    )
    table = conv.rows_to_table([row])
    assert table.column("symbol_ids").to_pylist()[0] == [ids[0], ids[0], 0]
    assert table.column("call_targets").to_pylist()[0] == [0, 0, ids[1]]
    assert table.column("type_refs").to_pylist()[0] == [ids[2], 0, 0]
    assert table.column("def_use").to_pylist()[0] == [1, 1, 2]
    # Confirm the declared schema dtypes are honored.
    assert table.column("symbol_ids").type == conv._SCHEMA.field("symbol_ids").type
    assert table.column("type_refs").type == conv._SCHEMA.field("type_refs").type
    assert table.column("def_use").type == conv._SCHEMA.field("def_use").type
    assert table.schema.metadata[
        conv.SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
    ] == b"3"


def test_converter_rejects_stale_symbol_identity_rows():
    row = _minimal_row(symbol_identity_schema_version=1)
    with pytest.raises(RuntimeError, match="regenerate.*clang USR"):
        conv.rows_to_table([row])


def test_pr_discussion_audit_columns_propagate():
    row = _minimal_row(
        pr_number=17,
        pr_discussion="title\nbody line",
    )
    table = conv.rows_to_table([row])

    assert table.column("pr_number").to_pylist() == [17]
    assert table.column("has_pr_discussion").to_pylist() == [True]
    assert table.column("pr_discussion_chars").to_pylist() == [15]
    assert table.column("pr_discussion_lines").to_pylist() == [2]
