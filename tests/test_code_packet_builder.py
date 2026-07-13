"""Tests for building CodePacket / CommitPacket from LMTokenBatch + parquet rows."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.code_packet_builder import (
    build_code_packet_from_row,
    build_code_packets,
    build_commit_packet_from_row,
)
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex


SEQ = 6
GOLDEN_MINI = Path(__file__).parent / "fixtures" / "golden_mini"


def _synthetic_packed_table() -> "pa.Table":
    """Two hand-built rows mimicking the v12/packed enriched parquet schema."""

    def tok(seed: int) -> list[int]:
        return [seed + i for i in range(SEQ)]

    return pa.table(
        {
            # Token-aligned semantic side-channels.
            "token_symbol_ids": [tok(10), tok(20)],
            "token_call_targets": [[0] * SEQ, [1] * SEQ],
            "token_type_refs": [[2] * SEQ, [0] * SEQ],
            "token_def_use": [[0] * SEQ, [3] * SEQ],
            # Chunk metadata (2 chunks per row over SEQ=6 tokens).
            "token_chunk_starts": [[0, 3], [0, 4]],
            "token_chunk_ends": [[3, 6], [4, 6]],
            "token_chunk_kinds": [[1, 2], [2, 1]],
            "token_chunk_dep_levels": [[0, 1], [0, 1]],
            # Graph edges over chunk indices, stored as list-of-pairs.
            "token_call_edges": [[[0, 1]], [[1, 0]]],
            "token_type_edges": [[[1, 0]], []],
            # Domain-routing token sidecars and token-local edge triples.
            "token_domain_ids": [[1] * SEQ, [2] * SEQ],
            "token_role_ids": [[0, 1, 2, 3, 4, 5], [0] * SEQ],
            "token_entity_ids": [[0, 10, 10, 0, 0, 0], [0] * SEQ],
            "token_scope_ids": [[0, 0, 1, 1, 0, 0], [0] * SEQ],
            "token_source_doc_ids": [[7] * SEQ, [8] * SEQ],
            "token_confidence_ids": [[4] * SEQ, [2] * SEQ],
            "token_domain_edges": [[{"from": 1, "to": 2, "kind": 20}], []],
            "token_build_edges": [[{"from": 2, "to": 3, "kind": 21}], []],
            "token_shell_edges": [[], []],
            "token_diagnostic_edges": [[], []],
            "token_cross_domain_edges": [[], []],
            # Provenance.
            "repo": ["acme/widgets", "acme/gadgets"],
            "filepath": ["src/a.c", "src/b.c"],
            "commit_hash": ["deadbeef", "cafef00d"],
        }
    )


def _synthetic_batch() -> LMTokenBatch:
    rng = np.random.default_rng(0)
    tokens = mx.array(rng.integers(0, 100, size=(2, SEQ), dtype=np.int32))
    return LMTokenBatch(
        tokens=tokens,
        target_tokens=mx.array(rng.integers(0, 100, size=(2, SEQ), dtype=np.int32)),
        loss_mask=mx.array(np.ones((2, SEQ), dtype=np.float32)),
        document_ids=mx.array(np.zeros((2, SEQ), dtype=np.int32)),
        structure_ids=mx.array(np.ones((2, SEQ), dtype=np.int32)),
        ast_depth_ids=mx.array(np.full((2, SEQ), 2, dtype=np.int32)),
        sibling_index_ids=mx.array(np.zeros((2, SEQ), dtype=np.int32)),
        node_type_ids=mx.array(np.full((2, SEQ), 3, dtype=np.int32)),
        dep_levels=mx.array(np.zeros((2, SEQ), dtype=np.int32)),
    )


def test_build_code_packets_populates_all_fields() -> None:
    batch = _synthetic_batch()
    table = _synthetic_packed_table()
    packets = build_code_packets(batch, table)
    assert len(packets) == 2

    p0 = packets[0]
    assert isinstance(p0, CodePacket)
    assert p0.token_axis_len == SEQ
    assert p0.repo == "acme/widgets"
    assert p0.filepath == "src/a.c"
    assert p0.commit_or_ref == "deadbeef"

    # Token-aligned semantic channels mapped and validated.
    assert p0.symbol_ids is not None
    assert np.asarray(p0.symbol_ids).tolist() == [10, 11, 12, 13, 14, 15]
    assert p0.type_refs is not None

    # Structure channels carried over from the LMTokenBatch.
    assert p0.structure_ids is not None
    assert int(np.asarray(p0.ast_depth)[0]) == 2

    # Chunk metadata.
    assert np.asarray(p0.chunk_starts).tolist() == [0, 3]
    assert np.asarray(p0.chunk_ends).tolist() == [3, 6]

    # Edges parse to EdgeIndex.
    assert isinstance(p0.call_edges, EdgeIndex)
    assert p0.call_edges.relation == "call"
    assert p0.call_edges.to_pairs() == [(0, 1)]
    assert p0.type_edges.to_pairs() == [(1, 0)]
    assert p0.domain_ids is not None
    assert np.asarray(p0.domain_ids).tolist() == [1] * SEQ
    assert p0.role_ids is not None
    assert np.asarray(p0.role_ids).tolist() == [0, 1, 2, 3, 4, 5]
    assert isinstance(p0.domain_edges, DomainEdgeIndex)
    assert p0.domain_edges.to_triples() == [(1, 2, 20)]
    assert p0.build_edges.to_triples() == [(2, 3, 21)]

    # Row 1 has an empty type_edges list.
    assert packets[1].type_edges.num_edges == 0
    assert packets[1].call_edges.to_pairs() == [(1, 0)]
    assert packets[1].domain_edges.num_edges == 0

    # GraphPacket bundling + block aggregation works on parsed edges.
    gp = p0.graph_packet()
    assert set(gp.relations) == {"call", "type"}
    assert gp.num_nodes == 2


def test_build_code_packet_records_absent_columns() -> None:
    batch = _synthetic_batch()
    # Drop the graph + semantic columns; keep tokens-shaped chunk metadata only.
    table = pa.table(
        {
            "token_chunk_starts": [[0, 3], [0, 4]],
            "token_chunk_ends": [[3, 6], [4, 6]],
            "token_chunk_kinds": [[1, 2], [2, 1]],
            "token_chunk_dep_levels": [[0, 1], [0, 1]],
            "repo": ["a", "b"],
            "filepath": ["x", "y"],
            "commit_hash": ["c0", "c1"],
        }
    )
    packets = build_code_packets(batch, table)
    p0 = packets[0]
    assert p0.symbol_ids is None
    assert p0.call_edges is None
    assert p0.domain_ids is None
    assert p0.domain_edges is None
    assert "token_symbol_ids" in p0.metadata["absent_columns"]
    assert "token_call_edges" in p0.metadata["absent_columns"]
    assert "token_domain_ids" in p0.metadata["absent_columns"]
    assert "token_domain_edges" in p0.metadata["absent_columns"]
    # Absent columns are recorded, NOT fabricated.
    assert "token_chunk_starts" in p0.metadata["present_columns"]


def test_build_code_packet_preserves_full_width_symbol_ids() -> None:
    batch = _synthetic_batch()
    high_id = 0xF123456789ABCDEF
    table = pa.table(
        {
            "token_symbol_ids": pa.array(
                [[high_id] * SEQ, [high_id - 1] * SEQ],
                type=pa.list_(pa.uint64()),
            ),
            "token_call_targets": pa.array(
                [[0] * SEQ, [high_id] * SEQ], type=pa.list_(pa.uint64())
            ),
            "token_type_refs": pa.array(
                [[high_id - 2] * SEQ, [0] * SEQ], type=pa.list_(pa.uint64())
            ),
            "token_def_use": [[0] * SEQ, [1] * SEQ],
        }
    )

    packets = build_code_packets(batch, table)

    assert packets[0].symbol_ids is not None
    assert packets[0].symbol_ids.dtype == mx.uint64
    assert np.asarray(packets[0].symbol_ids).tolist() == [high_id] * SEQ
    assert packets[1].call_targets is not None
    assert packets[1].call_targets.dtype == mx.uint64


def test_build_code_packet_misaligned_semantic_column_raises() -> None:
    batch = _synthetic_batch()
    bad = pa.table(
        {
            # Wrong length: 5 != SEQ (6).
            "token_symbol_ids": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        }
    )
    with pytest.raises(ValueError, match="token-aligned length 5 != token_ids length 6"):
        build_code_packets(batch, bad)


def test_build_code_packets_row_count_mismatch_raises() -> None:
    batch = _synthetic_batch()  # batch size 2
    one_row = pa.table({"token_symbol_ids": [[10, 11, 12, 13, 14, 15]]})
    with pytest.raises(ValueError, match="row count 1 != batch size 2"):
        build_code_packets(batch, one_row)


def test_build_commit_packet_from_temporal_row() -> None:
    table = pa.table(
        {
            "token_change_mask_pre": [[1, 0, 1, 0, 1, 0]],
            "token_change_mask_post": [[0, 1, 0, 1, 0, 1]],
            "hunk_id_per_token": [[0, 0, 1, 1, 1, 1]],
            "edit_op_per_token": [[1, 1, 2, 2, 0, 0]],
            "changed_chunk_ids": [[0, 1]],
            "changed_chunk_spans": [[[0, 3], [3, 6]]],
            "repo": ["acme/widgets"],
            "filepath": ["src/a.c"],
            "commit_hash": ["deadbeef"],
        }
    )
    columns = {name: table[name].to_pylist() for name in table.column_names}
    post = mx.array(np.arange(6, dtype=np.int32))
    pre = mx.array(np.arange(6, dtype=np.int32))
    packet = build_commit_packet_from_row(
        columns=columns,
        row_index=0,
        pre_token_ids=pre,
        post_token_ids=post,
    )
    assert packet.repo == "acme/widgets"
    assert packet.commit_or_ref == "deadbeef"
    assert np.asarray(packet.change_mask_pre).tolist() == [1, 0, 1, 0, 1, 0]
    assert np.asarray(packet.hunk_ids).tolist() == [0, 0, 1, 1, 1, 1]
    assert np.asarray(packet.changed_chunk_spans).tolist() == [[0, 3], [3, 6]]
    assert "token_change_mask_pre" in packet.metadata["present_columns"]


def test_build_commit_packet_misaligned_raises() -> None:
    columns = {
        # length 4 against post length 6
        "hunk_id_per_token": [[0, 0, 1, 1]],
    }
    post = mx.array(np.arange(6, dtype=np.int32))
    with pytest.raises(ValueError, match="hunk_ids: token-aligned length 4 != post_token_ids length 6"):
        build_commit_packet_from_row(columns=columns, row_index=0, post_token_ids=post)


@pytest.mark.skipif(
    not GOLDEN_MINI.exists(), reason="golden_mini fixture not present"
)
def test_build_from_golden_mini() -> None:
    import pyarrow.parquet as pq

    parquet_paths = sorted(GOLDEN_MINI.rglob("*.parquet"))
    if not parquet_paths:
        pytest.skip("golden_mini fixture has no parquet files")
    table = pq.read_table(parquet_paths[0])
    columns = {name: table[name].to_pylist() for name in table.column_names}
    # Build a CodePacket directly from row 0 with its own token ids if present.
    token_col = next(
        (c for c in ("token_ids", "input_ids", "tokens") if c in columns), None
    )
    if token_col is None:
        pytest.skip("golden_mini has no recognizable token column")
    token_ids = mx.array(np.asarray(columns[token_col][0], dtype=np.int32))
    packet = build_code_packet_from_row(
        token_ids=token_ids,
        columns=columns,
        row_index=0,
    )
    assert isinstance(packet, CodePacket)
    assert packet.token_axis_len == int(token_ids.shape[0])
