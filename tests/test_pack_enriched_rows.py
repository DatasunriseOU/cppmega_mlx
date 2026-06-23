from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
from cppmega_mlx.data.nanochat_pipeline.platform_vocab import MAX_PLATFORM_IDS
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched_schema import (
    CHANGED_CHUNK_IDS_COLUMN,
    CHANGED_CHUNK_SPANS_COLUMN,
    PLATFORM_IDS_COLUMN,
    TOKEN_AST_DEPTH_COLUMN,
    TOKEN_CALL_EDGES_COLUMN,
    TOKEN_CHUNK_DEP_LEVELS_COLUMN,
    TOKEN_CHUNK_ENDS_COLUMN,
    TOKEN_CHUNK_KINDS_COLUMN,
    TOKEN_CHUNK_STARTS_COLUMN,
    TOKEN_DEP_LEVELS_COLUMN,
    TOKEN_IDS_COLUMN,
    TOKEN_STRUCTURE_IDS_COLUMN,
    TOKEN_TYPE_EDGES_COLUMN,
)
from scripts.nanochat_data.pack_enriched_rows import (
    DOC_IDS_COLUMN,
    INPUT_IDS_COLUMN,
    LOSS_MASK_COLUMN,
    NUM_DOCS_COLUMN,
    PACK_ID_COLUMN,
    SOURCE_DOC_INDICES_COLUMN,
    TARGET_IDS_COLUMN,
    VALID_TOKEN_COUNT_COLUMN,
    normalize_document_record,
    pack_documents,
    pack_parquet_dataset,
)


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _doc(
    token_ids: list[int],
    *,
    platform_ids: list[int] | None = None,
    token_structure_ids: list[int] | None = None,
    token_dep_levels: list[int] | None = None,
    token_ast_depth: list[int] | None = None,
    token_chunk_starts: list[int] | None = None,
    token_chunk_ends: list[int] | None = None,
    token_chunk_kinds: list[int] | None = None,
    token_chunk_dep_levels: list[int] | None = None,
    token_call_edges: list[dict[str, int]] | None = None,
    token_type_edges: list[dict[str, int]] | None = None,
    changed_chunk_ids: list[int] | None = None,
    changed_chunk_spans: list[dict[str, int]] | None = None,
) -> dict[str, object]:
    return {
        TOKEN_IDS_COLUMN: list(token_ids),
        PLATFORM_IDS_COLUMN: list(platform_ids or []),
        TOKEN_STRUCTURE_IDS_COLUMN: list(token_structure_ids or []),
        TOKEN_DEP_LEVELS_COLUMN: list(token_dep_levels or []),
        TOKEN_AST_DEPTH_COLUMN: list(token_ast_depth or []),
        TOKEN_CHUNK_STARTS_COLUMN: list(token_chunk_starts or []),
        TOKEN_CHUNK_ENDS_COLUMN: list(token_chunk_ends or []),
        TOKEN_CHUNK_KINDS_COLUMN: list(token_chunk_kinds or []),
        TOKEN_CHUNK_DEP_LEVELS_COLUMN: list(token_chunk_dep_levels or []),
        TOKEN_CALL_EDGES_COLUMN: list(token_call_edges or []),
        TOKEN_TYPE_EDGES_COLUMN: list(token_type_edges or []),
        CHANGED_CHUNK_IDS_COLUMN: list(changed_chunk_ids or []),
        CHANGED_CHUNK_SPANS_COLUMN: list(changed_chunk_spans or []),
    }


def _normalize(records: list[dict[str, object]]):
    return [
        normalize_document_record(record, source_doc_index=index)
        for index, record in enumerate(records)
    ]


def _write_input_parquet(path: Path, records: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(records), path)


def test_best_fit_strategy_matches_cppmega_largest_fitting_row_order() -> None:
    docs = _normalize(
        [
            _doc([1, 2]),
            _doc([10, 11, 12, 13, 14]),
            _doc([20, 21, 22]),
            _doc([30]),
        ]
    )

    rows, overflow = pack_documents(
        docs,
        target_length=6,
        pad_token_id=0,
        strategy="best_fit",
    )

    assert overflow == []
    # Best-fit-decreasing groups {doc1(len5)+doc3(len1)} and {doc0(len2)+doc2(len3)};
    # rows are emitted deterministically ordered by smallest contained
    # source_doc_index, and within each row documents are ordered topologically
    # (here all dep_levels are 0, so by source_doc_index).
    assert [row[SOURCE_DOC_INDICES_COLUMN] for row in rows] == [[0, 2], [1, 3]]
    assert rows[0][INPUT_IDS_COLUMN] == [1, 2, 20, 21, 22, 0]
    assert rows[1][INPUT_IDS_COLUMN] == [10, 11, 12, 13, 14, 30]


def test_sequential_strategy_preserves_document_order() -> None:
    docs = _normalize(
        [
            _doc([1, 2]),
            _doc([10, 11, 12, 13, 14]),
            _doc([20, 21, 22]),
            _doc([30]),
        ]
    )

    rows, overflow = pack_documents(
        docs,
        target_length=6,
        pad_token_id=0,
        strategy="sequential",
    )

    assert overflow == []
    assert [row[SOURCE_DOC_INDICES_COLUMN] for row in rows] == [[0], [1], [2, 3]]
    assert rows[2][INPUT_IDS_COLUMN] == [20, 21, 22, 30, 0, 0]


def test_loss_mask_excludes_padding_and_cross_document_targets() -> None:
    docs = _normalize([_doc([1, 2]), _doc([10, 11])])

    rows, overflow = pack_documents(docs, target_length=5, pad_token_id=0)

    assert overflow == []
    row = rows[0]
    assert row[INPUT_IDS_COLUMN] == [1, 2, 10, 11, 0]
    assert row[TARGET_IDS_COLUMN] == [2, 10, 11, 0, 0]
    assert row[DOC_IDS_COLUMN] == [1, 1, 2, 2, 2]
    assert row[LOSS_MASK_COLUMN] == [1, 0, 1, 0, 0]


def test_pack_documents_carries_token_and_chunk_metadata_with_offsets() -> None:
    docs = _normalize(
        [
            _doc(
                [1, 2],
                platform_ids=[7, 8],
                token_structure_ids=[3, 3],
                token_dep_levels=[0, 1],
                token_ast_depth=[2, 3],
                token_chunk_starts=[0],
                token_chunk_ends=[2],
                token_chunk_kinds=[4],
                token_chunk_dep_levels=[1],
                changed_chunk_ids=[0],
                changed_chunk_spans=[{"start": 0, "end": 2}],
            ),
            _doc(
                [10, 11, 12],
                platform_ids=[7, 8],
                token_structure_ids=[5, 5, 6],
                token_dep_levels=[2, 2, 3],
                token_ast_depth=[4, 4, 5],
                token_chunk_starts=[0, 1],
                token_chunk_ends=[1, 3],
                token_chunk_kinds=[1, 2],
                token_chunk_dep_levels=[0, 2],
                token_call_edges=[{"from": 1, "to": 0}],
                token_type_edges=[{"from": 0, "to": 1}],
                changed_chunk_ids=[1],
                changed_chunk_spans=[{"start": 1, "end": 3}],
            ),
        ]
    )

    rows, overflow = pack_documents(docs, target_length=6, pad_token_id=0)

    assert overflow == []
    row = rows[0]
    assert row[PLATFORM_IDS_COLUMN] == [7, 8]
    # doc1 has min chunk_dep_level 0 and doc0 has min 1, so the dependency-first
    # topological order places doc1 (struct ids [5,5,6]) before doc0 ([3,3]).
    assert row[TOKEN_STRUCTURE_IDS_COLUMN] == [5, 5, 6, 3, 3, 0]
    assert row[TOKEN_DEP_LEVELS_COLUMN] == [2, 2, 3, 0, 1, 0]
    assert row[TOKEN_AST_DEPTH_COLUMN] == [4, 4, 5, 2, 3, 0]
    assert row[TOKEN_CHUNK_STARTS_COLUMN] == [0, 1, 3]
    assert row[TOKEN_CHUNK_ENDS_COLUMN] == [1, 3, 5]
    assert row[TOKEN_CHUNK_KINDS_COLUMN] == [1, 2, 4]
    assert row[TOKEN_CHUNK_DEP_LEVELS_COLUMN] == [0, 2, 1]
    assert row[TOKEN_CALL_EDGES_COLUMN] == [{"from": 1, "to": 0}]
    assert row[TOKEN_TYPE_EDGES_COLUMN] == [{"from": 0, "to": 1}]
    assert row[CHANGED_CHUNK_IDS_COLUMN] == [1, 2]
    assert row[CHANGED_CHUNK_SPANS_COLUMN] == [
        {"start": 1, "end": 3},
        {"start": 3, "end": 5},
    ]


@pytest.mark.parametrize(
    ("edge_column", "edge"),
    [
        (TOKEN_CALL_EDGES_COLUMN, {"from": 0, "to": 1}),
        (TOKEN_TYPE_EDGES_COLUMN, {"from": 1, "to": 0}),
    ],
)
def test_normalize_document_record_rejects_graph_edges_without_chunk_layout(
    edge_column: str,
    edge: dict[str, int],
) -> None:
    record = _doc([1, 2, 3])
    record[edge_column] = [edge]

    with pytest.raises(ValueError, match="requires non-empty token_chunk"):
        normalize_document_record(record, source_doc_index=0)


@pytest.mark.parametrize(
    ("edge_column", "edge"),
    [
        (TOKEN_CALL_EDGES_COLUMN, {"from": 0, "to": 2}),
        (TOKEN_TYPE_EDGES_COLUMN, {"from": -1, "to": 0}),
    ],
)
def test_normalize_document_record_rejects_graph_edges_out_of_chunk_range(
    edge_column: str,
    edge: dict[str, int],
) -> None:
    record = _doc(
        [1, 2, 3],
        token_chunk_starts=[0],
        token_chunk_ends=[3],
        token_chunk_kinds=[1],
        token_chunk_dep_levels=[0],
    )
    record[edge_column] = [edge]

    with pytest.raises(ValueError, match=f"{edge_column} edge out of range"):
        normalize_document_record(record, source_doc_index=0)


def test_normalize_document_record_rejects_changed_chunk_ids_without_chunk_layout() -> None:
    record = _doc(
        [1, 2, 3],
        changed_chunk_ids=[0],
        changed_chunk_spans=[{"start": 0, "end": 1}],
    )

    with pytest.raises(ValueError, match="requires non-empty token_chunk"):
        normalize_document_record(record, source_doc_index=0)


def test_normalize_document_record_rejects_changed_chunk_ids_out_of_range() -> None:
    record = _doc(
        [1, 2, 3],
        token_chunk_starts=[0],
        token_chunk_ends=[3],
        token_chunk_kinds=[1],
        token_chunk_dep_levels=[0],
        changed_chunk_ids=[1],
        changed_chunk_spans=[{"start": 0, "end": 1}],
    )

    with pytest.raises(ValueError, match=f"{CHANGED_CHUNK_IDS_COLUMN} out of range"):
        normalize_document_record(record, source_doc_index=0)


def test_pack_documents_merges_mixed_platform_ids_for_packed_row() -> None:
    docs = _normalize(
        [
            _doc([1, 2], platform_ids=[2, 64, 94]),
            _doc([10, 11], platform_ids=[3, 64, 94]),
        ]
    )

    rows, overflow = pack_documents(docs, target_length=5, pad_token_id=0)

    assert overflow == []
    assert rows[0][PLATFORM_IDS_COLUMN] == [2, 3, 64, 94]


def test_pack_parquet_dataset_writes_rows_consumed_by_training_reader(tmp_path: Path) -> None:
    input_path = tmp_path / "enriched.parquet"
    output_path = tmp_path / "packed.parquet"
    _write_input_parquet(
        input_path,
        [
            _doc(
                [1, 2, 3],
                platform_ids=[4],
                token_structure_ids=[1, 1, 2],
                token_dep_levels=[0, 0, 1],
                token_ast_depth=[2, 2, 3],
            ),
            _doc(
                [10, 11],
                platform_ids=[4],
                token_structure_ids=[3, 3],
                token_dep_levels=[2, 2],
                token_ast_depth=[4, 4],
            ),
        ],
    )

    summary = pack_parquet_dataset(
        input_path,
        output_path,
        target_length=6,
        pad_token_id=0,
        strategy="best_fit",
    )

    assert summary == {"input_docs": 2, "packed_rows": 1, "overflow_docs": 0}
    table = pq.read_table(output_path)
    assert {
        PACK_ID_COLUMN,
        INPUT_IDS_COLUMN,
        TARGET_IDS_COLUMN,
        LOSS_MASK_COLUMN,
        DOC_IDS_COLUMN,
        VALID_TOKEN_COUNT_COLUMN,
        NUM_DOCS_COLUMN,
    }.issubset(set(table.column_names))

    dataset = TokenParquetDataset(
        output_path,
        seq_len=6,
        batch_size=1,
        token_key=INPUT_IDS_COLUMN,
    )
    batch = next(dataset.iter_batches())

    assert tuple(batch.tokens.shape) == (1, 6)
    assert tuple(batch.targets.shape) == (1, 6)
    assert batch.platform_ids is not None
    platform_ids = np.array(batch.platform_ids)
    assert platform_ids.shape == (1, 6, MAX_PLATFORM_IDS)
    np.testing.assert_array_equal(platform_ids[0, :5, 0], np.array([4, 4, 4, 4, 4]))
    assert np.count_nonzero(platform_ids[0, 5]) == 0
    np.testing.assert_array_equal(
        np.array(batch.structure_ids),
        np.array([[1, 1, 2, 3, 3, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.loss_mask),
        np.array([[1, 1, 0, 1, 0, 0]], dtype=np.float32),
    )


def test_pack_parquet_dataset_threads_doc_local_platform_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "mixed_platform_enriched.parquet"
    output_path = tmp_path / "mixed_platform_packed.parquet"
    _write_input_parquet(
        input_path,
        [
            _doc([1, 2], platform_ids=[2, 64, 94]),
            _doc([10, 11], platform_ids=[3, 64, 94]),
        ],
    )

    pack_parquet_dataset(
        input_path,
        output_path,
        target_length=5,
        pad_token_id=0,
        strategy="best_fit",
    )

    dataset = TokenParquetDataset(
        output_path,
        seq_len=5,
        batch_size=1,
        token_key=INPUT_IDS_COLUMN,
    )
    batch = next(dataset.iter_batches())

    assert batch.platform_ids is not None
    platform_ids = np.array(batch.model_kwargs()["platform_ids"])
    assert platform_ids.shape == (1, 5, MAX_PLATFORM_IDS)
    np.testing.assert_array_equal(platform_ids[0, 0, :3], [2, 64, 94])
    np.testing.assert_array_equal(platform_ids[0, 1, :3], [2, 64, 94])
    np.testing.assert_array_equal(platform_ids[0, 2, :3], [3, 64, 94])
    np.testing.assert_array_equal(platform_ids[0, 3, :3], [3, 64, 94])
    assert np.count_nonzero(platform_ids[0, 4]) == 0
