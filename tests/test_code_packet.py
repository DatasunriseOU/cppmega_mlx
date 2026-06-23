"""Tests for the typed CodePacket / CommitPacket / GraphPacket contract."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.graph_packet import EdgeIndex, GraphPacket


def _ids(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def test_edge_index_from_pairs_roundtrip() -> None:
    edge = EdgeIndex.from_pairs([[0, 1], [2, 3]], relation="call", num_nodes=4)
    assert edge.num_edges == 2
    assert edge.relation == "call"
    assert edge.to_pairs() == [(0, 1), (2, 3)]


def test_edge_index_empty() -> None:
    edge = EdgeIndex.from_pairs([], relation="type")
    assert edge.num_edges == 0
    assert edge.to_pairs() == []


def test_edge_index_from_padded_with_mask() -> None:
    values = np.array([[0, 1], [2, 3], [0, 0]], dtype=np.int32)
    mask = np.array([1, 1, 0], dtype=np.int32)
    edge = EdgeIndex.from_padded(values, mask, relation="call", num_nodes=4)
    assert edge.num_edges == 3
    assert edge.to_pairs() == [(0, 1), (2, 3)]  # masked padding dropped


def test_edge_index_src_dst_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="src/dst length mismatch"):
        EdgeIndex(src=_ids([0, 1]), dst=_ids([0]), relation="call")


def test_graph_packet_block_aggregate() -> None:
    edge = EdgeIndex.from_pairs([[0, 1], [0, 65], [64, 1]], relation="call", num_nodes=128)
    packet = GraphPacket(edges={"call": edge}, num_nodes=128)
    agg = packet.block_aggregate("call", block_size=64)
    arr = np.asarray(agg)
    assert arr.shape == (2, 2)
    assert int(arr[0, 0]) == 1  # (0,1)
    assert int(arr[0, 1]) == 1  # (0,65)
    assert int(arr[1, 0]) == 1  # (64,1)


def test_graph_packet_block_aggregate_out_of_range_raises() -> None:
    edge = EdgeIndex.from_pairs([[0, 99]], relation="call", num_nodes=4)
    packet = GraphPacket(edges={"call": edge}, num_nodes=4)
    with pytest.raises(ValueError, match="out of range"):
        packet.block_aggregate("call", block_size=64)


def test_graph_packet_relation_key_mismatch_raises() -> None:
    edge = EdgeIndex.from_pairs([[0, 1]], relation="call")
    with pytest.raises(ValueError, match="disagrees with"):
        GraphPacket(edges={"type": edge})


def test_code_packet_full_population_and_validation() -> None:
    n = 6
    tokens = _ids(list(range(n)))
    packet = CodePacket(
        token_ids=tokens,
        target_ids=_ids([1, 2, 3, 4, 5, 0]),
        loss_mask=_ids([1, 1, 1, 1, 1, 0]),
        document_ids=_ids([0, 0, 0, 1, 1, 1]),
        repo="acme/widgets",
        filepath="src/main.c",
        commit_or_ref="deadbeef",
        structure_ids=_ids([1] * n),
        ast_depth=_ids([2] * n),
        sibling_index=_ids([0] * n),
        ast_node_type=_ids([3] * n),
        dep_levels=_ids([0] * n),
        symbol_ids=_ids([7] * n),
        call_targets=_ids([0] * n),
        type_refs=_ids([0] * n),
        def_use=_ids([0] * n),
        call_edges=EdgeIndex.from_pairs([[0, 1]], relation="call", num_nodes=2),
        type_edges=EdgeIndex.from_pairs([[1, 0]], relation="type", num_nodes=2),
        chunk_starts=_ids([0, 3]),
        chunk_ends=_ids([3, 6]),
        chunk_kinds=_ids([1, 2]),
        chunk_dep_levels=_ids([0, 1]),
        metadata={"source": "synthetic"},
    )
    assert packet.token_axis_len == n
    assert packet.repo == "acme/widgets"
    assert set(packet.structure_fields()) == {
        "structure_ids", "ast_depth", "sibling_index", "ast_node_type", "dep_levels",
    }
    assert set(packet.semantic_fields()) == {
        "symbol_ids", "call_targets", "type_refs", "def_use",
    }
    present = packet.present_fields()
    assert "call_edges" in present and "type_edges" in present
    assert "symbol_ids" in present and "chunk_starts" in present

    gp = packet.graph_packet()
    assert set(gp.relations) == {"call", "type"}
    assert gp.edge("call").to_pairs() == [(0, 1)]


def test_code_packet_misaligned_channel_raises() -> None:
    tokens = _ids(list(range(6)))
    with pytest.raises(ValueError, match="token-aligned length 5 != token_ids length 6"):
        CodePacket(token_ids=tokens, symbol_ids=_ids([0, 0, 0, 0, 0]))


def test_code_packet_misaligned_loss_mask_raises() -> None:
    tokens = _ids(list(range(4)))
    with pytest.raises(ValueError, match="token-aligned length 3 != token_ids length 4"):
        CodePacket(token_ids=tokens, loss_mask=_ids([1, 1, 1]))


def test_code_packet_chunk_length_disagreement_raises() -> None:
    tokens = _ids(list(range(6)))
    with pytest.raises(ValueError, match="chunk-metadata channels must share length"):
        CodePacket(
            token_ids=tokens,
            chunk_starts=_ids([0, 3]),
            chunk_ends=_ids([3]),
        )


def test_code_packet_absent_optionals_are_none() -> None:
    packet = CodePacket(token_ids=_ids([1, 2, 3]))
    assert packet.symbol_ids is None
    assert packet.call_edges is None
    assert packet.chunk_starts is None
    assert packet.present_fields() == ()


def test_commit_packet_full_population() -> None:
    pre = _ids(list(range(5)))
    post = _ids(list(range(5)))
    packet = CommitPacket(
        pre_token_ids=pre,
        post_token_ids=post,
        diff_token_ids=_ids([9, 9, 9]),
        commit_msg=_ids([4, 5]),
        change_mask_pre=_ids([1, 0, 1, 0, 1]),
        change_mask_post=_ids([0, 1, 0, 1, 0]),
        hunk_ids=_ids([0, 0, 1, 1, 1]),
        edit_ops=_ids([1, 1, 2, 2, 0]),
        changed_chunk_ids=_ids([0, 1]),
        changed_chunk_spans=mx.array(np.array([[0, 2], [2, 5]], dtype=np.int32)),
        repo="acme/widgets",
        commit_or_ref="cafe",
    )
    present = packet.present_fields()
    assert "change_mask_pre" in present and "changed_chunk_spans" in present


def test_commit_packet_misaligned_post_channel_raises() -> None:
    post = _ids(list(range(5)))
    with pytest.raises(ValueError, match="hunk_ids: token-aligned length 4 != post_token_ids length 5"):
        CommitPacket(post_token_ids=post, hunk_ids=_ids([0, 0, 1, 1]))


def test_commit_packet_changed_chunk_pair_imbalance_raises() -> None:
    with pytest.raises(ValueError, match="must both be present or both absent"):
        CommitPacket(changed_chunk_ids=_ids([0, 1]))
