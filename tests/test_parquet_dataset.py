from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
import cppmega_mlx.data.parquet_dataset as parquet_dataset
from cppmega_mlx.training.loss import next_token_cross_entropy


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _FakeTable:
    def __init__(self, values, types=None):
        self._values = values
        self.column_names = list(values)
        self.schema = _FakeSchema(types) if types is not None else None

    def __getitem__(self, key):
        return _FakeColumn(self._values[key])


class _FakeField:
    def __init__(self, type_label):
        self.type = type_label


class _FakeSchema:
    def __init__(self, types):
        self._types = types

    def field(self, key):
        return _FakeField(self._types[key])


def _fake_pyarrow_reader(values, types=None):
    return SimpleNamespace(read_table=lambda path: _FakeTable(values, types=types))


def test_token_list_parquet_rows_yield_fixed_lm_batches(monkeypatch) -> None:
    values = {
        "tokens": [
            [0, 1, 2, 3, 99],
            [10, 11, 12, 13, 14, 15, 16, 17],
        ],
        "structure_ids": [
            [1, 1, 1, 1, 9],
            [2, 2, 2, 2, 3, 3, 3, 3],
        ],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=2)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 3
    assert dataset.num_batches == 1
    assert dataset.dropped_samples == 1
    assert dataset.metadata.source_format == "parquet"
    np.testing.assert_array_equal(
        np.array(batch.tokens),
        np.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.structure_ids),
        np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int32),
    )


def test_cppmega_token_side_channel_aliases_are_normalized(monkeypatch) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "structure_ids": [[101, 102]],
        "token_structure_ids": [[1, 1, 1, 1, 2, 2, 2, 2]],
        "token_dep_levels": [[0, 1, 2, 3, 0, 1, 2, 3]],
        "token_ast_depth": [[3, 3, 2, 2, 1, 1, 0, 0]],
        "token_sibling_index": [[0, 1, 0, 1, 0, 1, 0, 1]],
        "token_ast_node_type": [[9, 8, 7, 6, 5, 4, 3, 2]],
    }
    types = {
        "token_ids": "large_list<element: uint32>",
        "structure_ids": "large_list<element: int8>",
        "token_structure_ids": "large_list<element: uint8>",
        "token_dep_levels": "large_list<element: uint16>",
        "token_ast_depth": "large_list<element: uint16>",
        "token_sibling_index": "large_list<element: uint16>",
        "token_ast_node_type": "large_list<element: uint16>",
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values, types=types)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset(
        "cppmega.parquet", seq_len=4, batch_size=2, token_key="token_ids"
    )
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 2
    np.testing.assert_array_equal(
        np.array(batch.structure_ids),
        np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.dep_levels),
        np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.ast_depth_ids),
        np.array([[3, 3, 2, 2], [1, 1, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.sibling_index_ids),
        np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.node_type_ids),
        np.array([[9, 8, 7, 6], [5, 4, 3, 2]], dtype=np.int32),
    )
    assert dataset.parquet_receipt == {
        "source_format": "parquet",
        "columns": sorted(values),
        "column_types": types,
        "token_source": {
            "mode": "token_column",
            "column": "token_ids",
            "type": "large_list<element: uint32>",
        },
        "side_channel_sources": {
            "ast_depth_ids": {
                "column": "token_ast_depth",
                "type": "large_list<element: uint16>",
            },
            "dep_levels": {
                "column": "token_dep_levels",
                "type": "large_list<element: uint16>",
            },
            "node_type_ids": {
                "column": "token_ast_node_type",
                "type": "large_list<element: uint16>",
            },
            "sibling_index_ids": {
                "column": "token_sibling_index",
                "type": "large_list<element: uint16>",
            },
            "structure_ids": {
                "column": "token_structure_ids",
                "type": "large_list<element: uint8>",
            },
        },
        "skipped_side_channel_columns": [
            {
                "field": "structure_ids",
                "column": "structure_ids",
                "type": "large_list<element: int8>",
                "reason": "not_token_aligned",
            },
        ],
    }


def test_batch_metadata_windows_keep_chunk_edges_out_of_model_kwargs() -> None:
    columns = parquet_dataset.ParquetColumns(
        {
            "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "token_structure_ids": [[1, 1, 1, 1, 2, 2, 2, 2]],
            "token_dep_levels": [[0, 0, 0, 0, 1, 1, 1, 1]],
            "token_chunk_starts": [[0, 4]],
            "token_chunk_ends": [[4, 8]],
            "token_chunk_kinds": [[2, 3]],
            "token_chunk_dep_levels": [[0, 1]],
            "token_call_edges": [[{"from": 0, "to": 1}]],
            "token_type_edges": [[{"from": 1, "to": 0}]],
            "language_info": ['{"language": "cpp"}'],
            "build_profile": ["gb10-clang"],
            "constituent_provenance_json": ['{"repo": "llvm"}'],
        },
        all_column_names=(
            "token_ids",
            "token_structure_ids",
            "token_dep_levels",
            "token_chunk_starts",
            "token_chunk_ends",
            "token_chunk_kinds",
            "token_chunk_dep_levels",
            "token_call_edges",
            "token_type_edges",
            "language_info",
            "build_profile",
            "constituent_provenance_json",
        ),
    )
    token_rows = parquet_dataset._token_rows_from_columns(
        columns,
        token_key="token_ids",
        text_key=None,
        tokenizer=None,
        eos_token_id=None,
    )

    metadata_windows = parquet_dataset._batch_metadata_windows(
        columns,
        token_rows,
        seq_len=4,
        metadata_columns=(
            "token_chunk_starts",
            "token_chunk_ends",
            "token_chunk_kinds",
            "token_chunk_dep_levels",
            "token_call_edges",
            "token_type_edges",
            "language_info",
            "build_profile",
            "constituent_provenance_json",
        ),
    )

    assert metadata_windows == (
        {
            "row_index": 0,
            "token_start": 0,
            "token_end": 4,
            "token_chunk_starts": [0],
            "token_chunk_ends": [4],
            "token_chunk_kinds": [2],
            "token_chunk_dep_levels": [0],
            "token_call_edges": [],
            "token_type_edges": [],
            "language_info": '{"language": "cpp"}',
            "build_profile": "gb10-clang",
            "constituent_provenance_json": '{"repo": "llvm"}',
        },
        {
            "row_index": 0,
            "token_start": 4,
            "token_end": 8,
            "token_chunk_starts": [0],
            "token_chunk_ends": [4],
            "token_chunk_kinds": [3],
            "token_chunk_dep_levels": [1],
            "token_call_edges": [],
            "token_type_edges": [],
            "language_info": '{"language": "cpp"}',
            "build_profile": "gb10-clang",
            "constituent_provenance_json": '{"repo": "llvm"}',
        },
    )


def test_candidate_columns_keep_row_metadata_opt_in_but_load_graph_inputs() -> None:
    default_columns = parquet_dataset._candidate_parquet_columns(
        token_key="token_ids",
        text_key=None,
        metadata_columns=(),
    )
    metadata_columns = parquet_dataset._candidate_parquet_columns(
        token_key="token_ids",
        text_key=None,
        metadata_columns=("repo", "token_call_edges"),
    )

    assert default_columns is not None
    assert metadata_columns is not None
    assert "token_chunk_starts" in default_columns
    assert "token_call_edges" in default_columns
    assert "repo" not in default_columns
    assert "token_chunk_starts" in metadata_columns
    assert "token_call_edges" in metadata_columns
    assert "repo" in metadata_columns


def test_all_metadata_excludes_token_content_and_slices_token_level_fields() -> None:
    columns = parquet_dataset.ParquetColumns(
        {
            "text": ["int main() {}"],
            "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "token_symbol_ids": [[10, 11, 12, 13, 14, 15, 16, 17]],
            "doc_ids": [[0, 0, 0, 0, 1, 1, 1, 1]],
            "platform_ids": [[4, 9]],
            "repo": ["llvm"],
        },
        all_column_names=(
            "text",
            "token_ids",
            "token_symbol_ids",
            "doc_ids",
            "platform_ids",
            "repo",
        ),
    )
    token_rows = parquet_dataset._token_rows_from_columns(
        columns,
        token_key="token_ids",
        text_key=None,
        tokenizer=None,
        eos_token_id=None,
    )

    metadata_columns = parquet_dataset._resolve_batch_metadata_columns(
        columns,
        token_key="token_ids",
        text_key=None,
        metadata_columns="all",
    )
    metadata_windows = parquet_dataset._batch_metadata_windows(
        columns,
        token_rows,
        seq_len=4,
        metadata_columns=metadata_columns,
    )

    assert "text" not in metadata_columns
    assert "token_ids" not in metadata_columns
    assert metadata_windows[0]["token_symbol_ids"] == [10, 11, 12, 13]
    assert metadata_windows[1]["token_symbol_ids"] == [14, 15, 16, 17]
    assert "doc_ids" not in metadata_columns
    assert "platform_ids" not in metadata_columns
    assert metadata_windows[1]["repo"] == "llvm"


def test_parquet_token_semantic_and_temporal_metadata_reach_side_channel_map(
    tmp_path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "semantic_temporal.parquet"
    high_identity = (1 << 63) + 1000
    table = pa.table(
        {
            "token_ids": pa.array(
                [[0, 1, 2, 3, 4, 5, 6, 7]],
                type=pa.large_list(pa.int32()),
            ),
            "token_symbol_ids": pa.array(
                [[high_identity + offset for offset in range(8)]],
                type=pa.large_list(pa.uint64()),
            ),
            "token_call_targets": pa.array(
                [[high_identity + 100 + offset for offset in range(8)]],
                type=pa.large_list(pa.uint64()),
            ),
            "token_change_mask_post": pa.array(
                [[0, 0, 1, 1, 0, 1, 0, 1]],
                type=pa.large_list(pa.int32()),
            ),
            "edit_op_per_token": pa.array(
                [[0, 0, 2, 2, 0, 3, 0, 3]],
                type=pa.large_list(pa.int32()),
            ),
        }
    )
    pq.write_table(table, path)

    dataset = TokenParquetDataset(path, seq_len=4, batch_size=2, token_key="token_ids")
    batch = next(dataset.iter_batches())

    assert batch.side_channels is not None
    assert tuple(batch.model_kwargs()) == ()
    assert set(batch.side_channels) == {"semantic_graph", "temporal_diff"}
    np.testing.assert_array_equal(
        np.array(batch.side_channels["semantic_graph"]["token_symbol_ids"]),
        np.array(
            [
                [high_identity + offset for offset in range(4)],
                [high_identity + offset for offset in range(4, 8)],
            ],
            dtype=np.uint64,
        ),
    )
    np.testing.assert_array_equal(
        np.array(batch.side_channels["semantic_graph"]["token_call_targets"]),
        np.array(
            [
                [high_identity + 100 + offset for offset in range(4)],
                [high_identity + 100 + offset for offset in range(4, 8)],
            ],
            dtype=np.uint64,
        ),
    )
    assert (
        np.array(batch.side_channels["semantic_graph"]["token_symbol_ids"]).dtype
        == np.dtype(np.uint64)
    )
    np.testing.assert_array_equal(
        np.array(batch.side_channels["temporal_diff"]["token_change_mask_post"]),
        [[0, 0, 1, 1], [0, 1, 0, 1]],
    )
    np.testing.assert_array_equal(
        np.array(batch.side_channels["temporal_diff"]["edit_op_per_token"]),
        [[0, 0, 2, 2], [0, 3, 0, 3]],
    )

    dropped = batch.with_side_channel_dropout(
        {"semantic_graph": 1.0, "temporal_diff": 1.0},
        seed=7,
    )
    assert dropped.side_channels is not None
    assert float(
        mx.sum(
            mx.abs(dropped.side_channels["semantic_graph"]["token_symbol_ids"])
        ).item()
    ) == 0.0
    assert float(
        mx.sum(
            mx.abs(dropped.side_channels["temporal_diff"]["edit_op_per_token"])
        ).item()
    ) == 0.0
    assert (
        dataset.parquet_receipt["family_side_channel_sources"]["semantic_graph"][
            "token_symbol_ids"
        ]["column"]
        == "token_symbol_ids"
    )
    assert (
        dataset.parquet_receipt["family_side_channel_sources"]["temporal_diff"][
            "edit_op_per_token"
        ]["column"]
        == "edit_op_per_token"
    )


def test_parquet_token_graph_edges_reach_semantic_side_channel_map(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "semantic_edges.parquet"
    edge_type = pa.large_list(
        pa.struct([("from", pa.int32()), ("to", pa.int32())])
    )
    table = pa.table(
        {
            "token_ids": pa.array(
                [[0, 1, 2, 3, 4, 5, 6, 7]],
                type=pa.large_list(pa.int32()),
            ),
            "token_chunk_starts": pa.array(
                [[0, 2, 4, 6]],
                type=pa.large_list(pa.int32()),
            ),
            "token_chunk_ends": pa.array(
                [[2, 4, 6, 8]],
                type=pa.large_list(pa.int32()),
            ),
            "token_call_edges": pa.array(
                [[{"from": 0, "to": 1}, {"from": 2, "to": 3}]],
                type=edge_type,
            ),
            "token_type_edges": pa.array(
                [[{"from": 1, "to": 0}, {"from": 3, "to": 2}]],
                type=edge_type,
            ),
        }
    )
    pq.write_table(table, path)

    dataset = TokenParquetDataset(path, seq_len=4, batch_size=2, token_key="token_ids")
    batch = next(dataset.iter_batches())

    assert batch.side_channels is not None
    graph = batch.side_channels["semantic_graph"]
    assert set(graph) == {
        "token_call_edges",
        "token_call_edges_mask",
        "token_type_edges",
        "token_type_edges_mask",
    }
    np.testing.assert_array_equal(
        np.array(graph["token_call_edges"]),
        np.array([[[0, 1]], [[0, 1]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(graph["token_call_edges_mask"]),
        np.array([[1], [1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(graph["token_type_edges"]),
        np.array([[[1, 0]], [[1, 0]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(graph["token_type_edges_mask"]),
        np.array([[1], [1]], dtype=np.int32),
    )
    assert (
        dataset.parquet_receipt["family_side_channel_sources"]["semantic_graph"][
            "token_call_edges"
        ]["column"]
        == "token_call_edges"
    )
    assert (
        dataset.parquet_receipt["family_side_channel_sources"]["semantic_graph"][
            "token_type_edges"
        ]["column"]
        == "token_type_edges"
    )


def test_parquet_token_semantic_side_channel_fails_closed_when_not_token_aligned(
    tmp_path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "bad_semantic.parquet"
    table = pa.table(
        {
            "token_ids": pa.array(
                [[0, 1, 2, 3, 4, 5, 6, 7]],
                type=pa.large_list(pa.int32()),
            ),
            "token_symbol_ids": pa.array(
                [[10, 11]],
                type=pa.large_list(pa.int32()),
            ),
        }
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="token_symbol_ids.*token-aligned"):
        TokenParquetDataset(path, seq_len=4, batch_size=1, token_key="token_ids")


def test_platform_ids_parquet_column_threads_to_model_kwargs(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "platform_ids.parquet"
    table = pa.table(
        {
            "token_ids": pa.array(
                [[[0, 1, 2, 3, 4, 5, 6, 7]]][0],
                type=pa.large_list(pa.int32()),
            ),
            "platform_ids": pa.array(
                [[[2, 64, 94]]][0],
                type=pa.large_list(pa.int32()),
            ),
            "repo": pa.array(["llvm"]),
        }
    )
    pq.write_table(table, path)

    dataset = TokenParquetDataset(
        path,
        seq_len=4,
        batch_size=2,
        token_key="token_ids",
        metadata_columns="all",
    )
    batch = next(dataset.iter_batches())

    assert batch.platform_ids is not None
    np.testing.assert_array_equal(
        np.array(batch.platform_ids),
        np.array([[2, 64, 94], [2, 64, 94]], dtype=np.int32),
    )
    assert "platform_ids" in batch.model_kwargs()
    assert "platform_ids" not in batch.training_metadata()["parquet"]["columns"]
    assert dataset.parquet_receipt["model_metadata_sources"] == {
        "platform_ids": {
            "column": "platform_ids",
            "type": "large_list<element: int32>",
        }
    }


def test_nanochat_input_target_loss_docids_parquet_drive_training_batch(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "packed.parquet"
    table = pa.table(
        {
            "input_ids": pa.array([[10, 11, 12, 13]], type=pa.large_list(pa.int32())),
            "target_ids": pa.array([[11, 12, 13, 14]], type=pa.large_list(pa.int32())),
            "loss_mask": pa.array([[1, 0, 1, 1]], type=pa.large_list(pa.int8())),
            "doc_ids": pa.array([[0, 0, 1, 1]], type=pa.large_list(pa.int32())),
            "token_structure_ids": pa.array(
                [[2, 2, 3, 3]], type=pa.large_list(pa.int16())
            ),
            "token_symbol_ids": pa.array(
                [[100, 101, 102, 103]], type=pa.large_list(pa.int64())
            ),
        }
    )
    pq.write_table(table, path)

    dataset = TokenParquetDataset(
        path,
        seq_len=4,
        batch_size=1,
        metadata_columns="all",
    )
    batch = next(dataset.iter_batches())

    np.testing.assert_array_equal(np.array(batch.inputs), [[10, 11, 12, 13]])
    np.testing.assert_array_equal(np.array(batch.targets), [[11, 12, 13, 14]])
    np.testing.assert_array_equal(np.array(batch.target_mask), [[1, 0, 1, 1]])
    np.testing.assert_array_equal(np.array(batch.input_document_ids), [[0, 0, 1, 1]])
    np.testing.assert_array_equal(
        np.array(batch.model_kwargs()["structure_ids"]),
        [[2, 2, 3, 3]],
    )
    metadata_columns = batch.training_metadata()["parquet"]["columns"]
    assert "token_symbol_ids" in metadata_columns
    assert "target_ids" not in metadata_columns
    assert "loss_mask" not in metadata_columns
    assert "doc_ids" not in metadata_columns
    assert dataset.parquet_receipt["token_source"]["column"] == "input_ids"
    assert dataset.parquet_receipt["training_column_sources"] == {
        "target_tokens": {"column": "target_ids", "type": "large_list<element: int32>"},
        "loss_mask": {"column": "loss_mask", "type": "large_list<element: int8>"},
        "document_ids": {"column": "doc_ids", "type": "large_list<element: int32>"},
    }

    class UniformLogitModel(nn.Module):
        def __call__(self, input_ids, **kwargs):
            assert tuple(input_ids.shape) == (1, 4)
            assert "document_ids" in kwargs
            return parquet_dataset.mx.zeros((1, 4, 32), dtype=parquet_dataset.mx.float32)

    loss, ntokens = next_token_cross_entropy(UniformLogitModel(), batch)
    parquet_dataset.mx.eval(loss, ntokens)

    assert int(ntokens.item()) == 3
    assert float(loss.item()) > 0


@pytest.mark.parametrize(
    ("document_ids", "attention_mask", "loss_mask", "valid_count", "match"),
    [
        ([1, 2, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1], None, "reused non-contiguously"),
        ([1, 1, 2, 2], [1, 0, 1, 0], [0, 0, 0, 0], None, "padding must be trailing"),
        ([1, 1, 1, 1], [1, float("nan"), 1, 1], [0, 0, 0, 0], None, "finite binary"),
        ([1, 1, 1, 1], [1, 1, 1, 1], [1, float("nan"), 1, 1], None, "finite binary"),
        ([1, 1, 1, 1], [1, 1, 1, 1], [1, 0.5, 1, 1], None, "finite binary"),
        ([1, 1, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1], None, "cross-document"),
        ([1, 1, 0, 0], None, [1, 1, 0, 0], 2, "padding transitions"),
    ],
)
def test_packed_parquet_rejects_invalid_boundary_contract(
    tmp_path,
    document_ids,
    attention_mask,
    loss_mask,
    valid_count,
    match,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    columns = {
        "input_ids": pa.array([[10, 11, 12, 13]], type=pa.large_list(pa.int32())),
        "doc_ids": pa.array([document_ids], type=pa.large_list(pa.int32())),
        "loss_mask": pa.array([loss_mask], type=pa.large_list(pa.float32())),
    }
    if attention_mask is not None:
        columns["attention_mask"] = pa.array(
            [attention_mask],
            type=pa.large_list(pa.float32()),
        )
    if valid_count is not None:
        columns["valid_token_count"] = pa.array([valid_count], type=pa.int32())
    path = tmp_path / "invalid-packed.parquet"
    pq.write_table(pa.table(columns), path)

    with pytest.raises(ValueError, match=match):
        TokenParquetDataset(path, seq_len=4, batch_size=1)


def test_source_level_structure_ids_are_ignored_when_not_token_aligned(
    monkeypatch,
) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "structure_ids": [[10, 20]],
    }
    types = {
        "token_ids": "large_list<element: uint32>",
        "structure_ids": "large_list<element: int8>",
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values, types=types)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset(
        "cppmega.parquet", seq_len=4, batch_size=2, token_key="token_ids"
    )
    batch = next(dataset.iter_batches())

    assert batch.structure_ids is None
    assert dataset.parquet_receipt["side_channel_sources"] == {}
    assert dataset.parquet_receipt["skipped_side_channel_columns"] == [
        {
            "field": "structure_ids",
            "column": "structure_ids",
            "type": "large_list<element: int8>",
            "reason": "not_token_aligned",
        },
    ]


def test_token_aligned_alias_and_canonical_side_channel_collision_fails_closed(
    monkeypatch,
) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "structure_ids": [[10, 10, 10, 10, 20, 20, 20, 20]],
        "token_structure_ids": [[1, 1, 1, 1, 2, 2, 2, 2]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="structure_ids side-channel declared more than once"):
        TokenParquetDataset(
            "duplicate_structure.parquet",
            seq_len=4,
            batch_size=1,
            token_key="token_ids",
        )


def test_token_level_alias_shape_mismatch_fails_closed(monkeypatch) -> None:
    values = {
        "token_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "token_structure_ids": [[1, 1, 1, 1]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token_structure_ids side-channel rows must be token-aligned"):
        TokenParquetDataset(
            "short_token_structure.parquet",
            seq_len=4,
            batch_size=1,
            token_key="token_ids",
        )


def test_scalar_token_column_is_treated_as_one_contiguous_stream(monkeypatch) -> None:
    values = {"tokens": list(range(12))}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=3)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 3
    np.testing.assert_array_equal(np.array(batch.tokens[2]), np.arange(8, 12))


def test_cursor_after_includes_dataset_resume_batch_across_epoch_rollover(
    monkeypatch,
) -> None:
    values = {"tokens": list(range(40))}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    dataset = TokenParquetDataset(
        "resume_cursor.parquet",
        seq_len=5,
        batch_size=2,
        shuffle=True,
        seed=7,
        loop=True,
        resume_batch=3,
    )
    stream = dataset.iter_batches()
    next(stream)
    next(stream)
    expected_next = next(stream)

    cursor = dataset.cursor_after(2)
    restored = TokenParquetDataset(
        "resume_cursor.parquet",
        seq_len=5,
        batch_size=2,
        shuffle=True,
        seed=7,
        loop=True,
    )
    actual_next = next(
        restored.iter_batches(
            resume_batch=cursor.batch_offset,
            epoch=cursor.epoch,
        )
    )

    assert cursor.epoch == 1
    assert cursor.batch_offset == 1
    assert cursor.global_batch_offset == 5
    np.testing.assert_array_equal(
        np.array(actual_next.tokens),
        np.array(expected_next.tokens),
    )


def test_text_column_requires_and_uses_tokenizer(monkeypatch) -> None:
    values = {"text": ["abcd", "efgh"]}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    class TinyTokenizer:
        def encode(self, text: str):
            return [ord(char) - 96 for char in text]

    dataset = TokenParquetDataset(
        "text.parquet",
        seq_len=4,
        batch_size=1,
        text_key="text",
        tokenizer=TinyTokenizer(),
    )
    batch = next(dataset.iter_batches())

    np.testing.assert_array_equal(np.array(batch.tokens[0]), np.array([1, 2, 3, 4]))


def test_pandas_backend_is_used_when_pyarrow_is_absent(monkeypatch) -> None:
    class FakeDataFrame:
        columns = ["tokens"]

        def __getitem__(self, key):
            assert key == "tokens"
            return SimpleNamespace(tolist=lambda: [[0, 1, 2, 3]])

    def fake_import(name):
        if name == "pyarrow.parquet":
            raise ModuleNotFoundError(name)
        if name == "pandas":
            return SimpleNamespace(read_parquet=lambda path: FakeDataFrame())
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(parquet_dataset.importlib, "import_module", fake_import)

    dataset = TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=1)
    batch = next(dataset.iter_batches())

    np.testing.assert_array_equal(np.array(batch.tokens[0]), np.arange(4))


def test_parquet_rejects_token_ids_outside_int32_range(monkeypatch) -> None:
    values = {
        "tokens": [[0, np.iinfo(np.int32).max + 1, 2, 3]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token IDs exceed int32 range"):
        TokenParquetDataset("too_large_tokens.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_non_integer_token_ids(monkeypatch) -> None:
    values = {"tokens": [[0, 1.5, 2, 3]]}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token IDs must be integer-valued"):
        TokenParquetDataset("float_tokens.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_declared_float_token_dtype(monkeypatch) -> None:
    values = {"tokens": [[0, 1, 2, 3]]}
    types = {"tokens": "large_list<element: double>"}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values, types=types)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token IDs parquet column 'tokens' must use an integer dtype"):
        TokenParquetDataset("float_token_type.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_negative_structure_side_channels(monkeypatch) -> None:
    values = {
        "tokens": [[0, 1, 2, 3]],
        "token_structure_ids": [[0, 1, -1, 3]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(
        ValueError,
        match="structure_ids side-channel IDs must be non-negative",
    ):
        TokenParquetDataset("negative_structure.parquet", seq_len=4, batch_size=1)


def test_parquet_rejects_non_integer_structure_side_channels(monkeypatch) -> None:
    values = {
        "tokens": [[0, 1, 2, 3]],
        "token_structure_ids": [[0, 1.25, 2, 3]],
    }
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="token_structure_ids side-channel must be integer-valued"):
        TokenParquetDataset("float_structure.parquet", seq_len=4, batch_size=1)


def test_missing_optional_parquet_backends_raise_clear_import_error(monkeypatch) -> None:
    def fake_import(name):
        if name in {"pyarrow.parquet", "pandas"}:
            raise ModuleNotFoundError(name)
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(parquet_dataset.importlib, "import_module", fake_import)

    with pytest.raises(ImportError, match="pyarrow.*pandas"):
        TokenParquetDataset("tokens.parquet", seq_len=4, batch_size=1)


def test_text_column_without_tokenizer_is_rejected(monkeypatch) -> None:
    values = {"text": ["abcd"]}
    monkeypatch.setattr(
        parquet_dataset.importlib,
        "import_module",
        lambda name: _fake_pyarrow_reader(values)
        if name == "pyarrow.parquet"
        else pytest.fail(f"unexpected import {name}"),
    )

    with pytest.raises(ValueError, match="requires tokenizer"):
        TokenParquetDataset("text.parquet", seq_len=4, batch_size=1, text_key="text")
