"""Tests for the fixed graph routing prior (code_graph_routes)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.graph_packet import EdgeIndex, GraphPacket
from cppmega_mlx.nn.code_graph_routes import (
    CodeGraphRouter,
    GraphRouteConfig,
    build_attention_bias,
    build_block_candidates,
)


def _call_packet() -> GraphPacket:
    # 8 chunks; call edges 0->4, 1->5, 6->2.
    return GraphPacket(
        edges={
            "call": EdgeIndex.from_pairs(
                [[0, 4], [1, 5], [6, 2]], relation="call", num_nodes=8
            )
        },
        num_nodes=8,
    )


def test_block_aggregate_routes_to_correct_blocks():
    pkt = _call_packet()
    cfg = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="binary")
    bias = build_attention_bias(pkt, config=cfg)
    assert tuple(bias.shape) == (4, 4)
    b = np.asarray(bias)
    # block_size = ceil(8/4) = 2: chunk0,1 -> block0 ; chunk4,5 -> block2 ;
    # chunk6 -> block3 ; chunk2 -> block1.
    assert b[0, 2] == 1.0  # 0->4 and 1->5 both collapse to block0->block2
    assert b[3, 1] == 1.0  # 6->2
    assert b.sum() == 2.0  # exactly two distinct block-edges (binary)


def test_count_vs_binary_normalize():
    pkt = _call_packet()
    cfg_bin = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="binary")
    cfg_cnt = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="count")
    b_bin = np.asarray(build_attention_bias(pkt, config=cfg_bin))
    b_cnt = np.asarray(build_attention_bias(pkt, config=cfg_cnt))
    # block0->block2 has TWO raw edges (0->4, 1->5): count keeps 2, binary clamps 1.
    assert b_cnt[0, 2] == 2.0
    assert b_bin[0, 2] == 1.0


def test_build_block_candidates_matches_routes():
    pkt = _call_packet()
    cfg = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="binary")
    cands = build_block_candidates(pkt, config=cfg)
    assert cands == [[2], [], [], [1]]


def test_router_learned_channel_mixing():
    pkt = GraphPacket(
        edges={
            "call": EdgeIndex.from_pairs([[0, 4]], relation="call", num_nodes=8),
            "type": EdgeIndex.from_pairs([[1, 5]], relation="type", num_nodes=8),
        },
        num_nodes=8,
    )
    cfg = GraphRouteConfig(num_blocks=4, relations=("call", "type"), normalize="binary")
    router = CodeGraphRouter(cfg)
    base = router(pkt)
    assert float(mx.sum(base)) == 2.0  # alpha=1 for both relations
    # Scale the type channel down: total drops by exactly the type contribution.
    router.alpha = mx.array([1.0, 0.0], dtype=mx.float32)
    only_call = router(pkt)
    assert float(mx.sum(only_call)) == 1.0


def test_router_alpha_is_differentiable():
    pkt = _call_packet()
    cfg = GraphRouteConfig(num_blocks=4, relations=("call",), normalize="binary")
    router = CodeGraphRouter(cfg)

    def loss_fn(alpha):
        router.alpha = alpha
        return mx.sum(router(pkt))

    grad = mx.grad(loss_fn)(router.alpha)
    # d/d_alpha sum(alpha * A) = sum(A) = number of block-edges = 2.
    assert float(grad[0]) == 2.0


def test_missing_relation_raises():
    pkt = _call_packet()
    cfg = GraphRouteConfig(num_blocks=4, relations=("type",), normalize="binary")
    with pytest.raises(KeyError):
        build_attention_bias(pkt, config=cfg)


def test_router_and_config_mutually_exclusive():
    pkt = _call_packet()
    router = CodeGraphRouter(GraphRouteConfig(num_blocks=4, relations=("call",)))
    with pytest.raises(ValueError):
        build_attention_bias(pkt, router=router, config=GraphRouteConfig())
