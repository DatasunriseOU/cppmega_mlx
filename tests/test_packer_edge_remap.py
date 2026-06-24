"""Edge-remap regression tests for the whole-document packer.

These cover the gap where, after several whole docs are topologically ordered and
concatenated into one packed block, each doc's intra-doc graph/chunk metadata
must be shifted into block-coordinate space:

  * chunk_starts / chunk_ends -> TOKEN coordinates (shift by the doc's token
    start in the block).
  * token_call_edges / token_type_edges endpoints and changed_chunk_ids -> CHUNK
    indices (shift by the count of chunks emitted by preceding docs).

They also pin the whole-function invariant: a doc longer than target_length is
kept WHOLE in its own packed row (never split / truncated).
"""

from __future__ import annotations

import pytest

from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
)
from scripts.nanochat_data.pack_enriched_rows import (
    DOC_DEP_EDGES_COLUMN,
    NUM_DOCS_COLUMN,
    SOURCE_DOC_INDICES_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
    normalize_document_record,
    pack_documents,
)


def _doc(
    index: int,
    token_ids: list[int],
    *,
    chunk_starts: list[int],
    chunk_ends: list[int],
    chunk_kinds: list[int] | None = None,
    chunk_dep_levels: list[int] | None = None,
    call_edges: list[dict[str, int]] | None = None,
    type_edges: list[dict[str, int]] | None = None,
    doc_dep_edges: list[int] | None = None,
):
    n_chunks = len(chunk_starts)
    record = {
        TOKEN_IDS_COLUMN: list(token_ids),
        TOKEN_CHUNK_STARTS_COLUMN: list(chunk_starts),
        TOKEN_CHUNK_ENDS_COLUMN: list(chunk_ends),
        TOKEN_CHUNK_KINDS_COLUMN: list(chunk_kinds or [1] * n_chunks),
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: list(chunk_dep_levels or [0] * n_chunks),
        TOKEN_CALL_EDGES_COLUMN: list(call_edges or []),
        TOKEN_TYPE_EDGES_COLUMN: list(type_edges or []),
        DOC_DEP_EDGES_COLUMN: list(doc_dep_edges or []),
    }
    return normalize_document_record(record, source_doc_index=index)


def _chunk_token_span(row, edge_endpoint: int) -> tuple[int, int]:
    """Resolve a global chunk index to its (start, end) global token span."""

    return (
        int(row[TOKEN_CHUNK_STARTS_COLUMN][edge_endpoint]),
        int(row[TOKEN_CHUNK_ENDS_COLUMN][edge_endpoint]),
    )


def test_intra_doc_edges_remap_to_correct_global_tokens() -> None:
    # Three docs forced into one block via doc_dep_edges: doc0 -> doc1 -> doc2.
    # Each carries an intra-doc edge from a *dependency* chunk to a *dependent*
    # chunk. After packing we require: every edge endpoint resolves to the chunk
    # whose global token span belongs to the doc it came from, and the
    # dependency chunk's tokens precede the dependent chunk's tokens.
    #
    # doc0: tokens [10,11,12], chunks c0=[0,1) c1=[1,3); edge c0(dep)->c1(use)
    # doc1: tokens [20,21],    chunks c0=[0,1) c1=[1,2); edge c0(dep)->c1(use)
    # doc2: tokens [30,31,32,33], chunks c0=[0,2) c1=[2,4); edge c0(dep)->c1(use)
    doc0 = _doc(
        0,
        [10, 11, 12],
        chunk_starts=[0, 1],
        chunk_ends=[1, 3],
        call_edges=[{"from": 0, "to": 1}],
    )
    doc1 = _doc(
        1,
        [20, 21],
        chunk_starts=[0, 1],
        chunk_ends=[1, 2],
        type_edges=[{"from": 0, "to": 1}],
        doc_dep_edges=[0],
    )
    doc2 = _doc(
        2,
        [30, 31, 32, 33],
        chunk_starts=[0, 2],
        chunk_ends=[2, 4],
        call_edges=[{"from": 0, "to": 1}],
        doc_dep_edges=[1],
    )

    rows, overflow = pack_documents(
        [doc0, doc1, doc2], target_length=16, pad_token_id=0, strategy="best_fit"
    )

    assert overflow == []
    assert len(rows) == 1
    row = rows[0]

    # Topological order must be preserved: doc0, then doc1, then doc2.
    assert row[SOURCE_DOC_INDICES_COLUMN] == [0, 1, 2]
    assert row[NUM_DOCS_COLUMN] == 3
    assert row["input_ids"][:9] == [10, 11, 12, 20, 21, 30, 31, 32, 33]

    # doc0 contributed 2 chunks (global 0,1), doc1 2 (global 2,3), doc2 2 (4,5).
    assert row[TOKEN_CHUNK_STARTS_COLUMN] == [0, 1, 3, 4, 5, 7]
    assert row[TOKEN_CHUNK_ENDS_COLUMN] == [1, 3, 4, 5, 7, 9]

    # Edges are remapped into block CHUNK-index space.
    assert row[TOKEN_CALL_EDGES_COLUMN] == [
        {"from": 0, "to": 1},  # doc0: chunk_offset 0
        {"from": 4, "to": 5},  # doc2: chunk_offset 4
    ]
    assert row[TOKEN_TYPE_EDGES_COLUMN] == [
        {"from": 2, "to": 3},  # doc1: chunk_offset 2
    ]

    # Every endpoint resolves to a real chunk, and for each edge the dependency
    # chunk's tokens strictly precede the dependent chunk's tokens in the block.
    for edge in row[TOKEN_CALL_EDGES_COLUMN] + row[TOKEN_TYPE_EDGES_COLUMN]:
        dep_start, dep_end = _chunk_token_span(row, edge["from"])
        use_start, use_end = _chunk_token_span(row, edge["to"])
        assert 0 <= dep_start < dep_end <= len(row["input_ids"])
        assert 0 <= use_start < use_end <= len(row["input_ids"])
        # dependency precedes dependent
        assert dep_start < use_start
        assert dep_end <= use_start

    # The doc2 call edge must point at doc2's own tokens (30..33), proving the
    # endpoint was shifted by doc2's token/chunk offset and not left aliasing
    # doc0's chunk 0/1.
    doc2_dep_span = _chunk_token_span(row, 4)
    doc2_use_span = _chunk_token_span(row, 5)
    assert row["input_ids"][doc2_dep_span[0] : doc2_dep_span[1]] == [30, 31]
    assert row["input_ids"][doc2_use_span[0] : doc2_use_span[1]] == [32, 33]


def test_overlong_doc_gets_its_own_block_unsplit() -> None:
    # A doc longer than target_length must be emitted WHOLE in its own row
    # (never split/truncated), while a fitting doc packs normally. Verified for
    # both packing strategies.
    big = _doc(
        0,
        [1, 2, 3, 4, 5, 6, 7],
        chunk_starts=[0, 4],
        chunk_ends=[4, 7],
        call_edges=[{"from": 0, "to": 1}],
    )
    small = _doc(1, [8, 9], chunk_starts=[0], chunk_ends=[2])

    for strategy in ("best_fit", "sequential"):
        rows, overflow = pack_documents(
            [big, small], target_length=4, pad_token_id=0, strategy=strategy
        )

        # Oversized doc is still recorded for tracking.
        assert [int(rec["token_count"]) for rec in overflow] == [7]

        by_first_doc = {row[SOURCE_DOC_INDICES_COLUMN][0]: row for row in rows}
        assert set(by_first_doc) == {0, 1}

        big_row = by_first_doc[0]
        # Whole and unsplit: the over-long doc's tokens appear intact, with the
        # row width grown to hold them rather than clipped to target_length.
        assert big_row[SOURCE_DOC_INDICES_COLUMN] == [0]
        assert big_row[VALID_TOKEN_COUNT_COLUMN] == 7
        assert big_row["input_ids"][:7] == [1, 2, 3, 4, 5, 6, 7]
        assert len(big_row["input_ids"]) >= 7
        # Its intra-doc edge survives and stays in-range against its chunks.
        assert big_row[TOKEN_CALL_EDGES_COLUMN] == [{"from": 0, "to": 1}]
        assert big_row[TOKEN_CHUNK_STARTS_COLUMN] == [0, 4]
        assert big_row[TOKEN_CHUNK_ENDS_COLUMN] == [4, 7]

        small_row = by_first_doc[1]
        assert small_row[VALID_TOKEN_COUNT_COLUMN] == 2
        assert len(small_row["input_ids"]) == 4  # padded to target_length
        assert small_row["input_ids"][:2] == [8, 9]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
