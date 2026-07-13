from __future__ import annotations

import numpy as np

from cppmega_mlx.data.domain_packet import DomainEdgeIndex, DomainPacket
from cppmega_mlx.data.domain_schema import DomainEdgeKind, DomainKind
from cppmega_mlx.nn.domain_graph_routes import (
    DomainGraphRouteConfig,
    build_domain_attention_bias,
    build_domain_block_candidates,
)


def test_default_config_prioritizes_diagnostic_locations():
    assert DomainGraphRouteConfig().edge_weights[
        int(DomainEdgeKind.DIAG_PRIMARY_LOCATION)
    ] == 2.0


def test_domain_attention_bias_routes_edge_triples_to_blocks() -> None:
    packet = DomainPacket.filled(
        list(range(8)),
        domain=DomainKind.CMAKE,
        graph_edges=DomainEdgeIndex.from_triples(
            [
                (0, 4, DomainEdgeKind.BUILD_TARGET_SOURCE),
                (1, 5, DomainEdgeKind.BUILD_TARGET_SOURCE),
                (6, 2, DomainEdgeKind.BUILD_TARGET_DEP),
            ]
        ),
    )

    cfg = DomainGraphRouteConfig(num_blocks=4, normalize="binary")
    bias = np.asarray(build_domain_attention_bias(packet, config=cfg))

    assert bias[0, 2] == 1.0
    assert bias[3, 1] == 1.0
    assert bias.sum() == 2.0
    assert build_domain_block_candidates(packet, config=cfg) == [[2], [], [], [1]]


def test_domain_attention_bias_respects_edge_kind_weights() -> None:
    packet = DomainPacket.filled(
        list(range(4)),
        domain=DomainKind.COMPILER_ERROR,
        graph_edges=DomainEdgeIndex.from_triples(
            [(0, 3, DomainEdgeKind.DIAG_PRIMARY_LOCATION)]
        ),
    )

    cfg = DomainGraphRouteConfig(
        num_blocks=4,
        edge_weights={int(DomainEdgeKind.DIAG_PRIMARY_LOCATION): 3.0},
    )
    bias = np.asarray(build_domain_attention_bias(packet, config=cfg))

    assert bias[0, 3] == 3.0


def test_domain_block_candidates_can_be_capped_by_weight() -> None:
    packet = DomainPacket.filled(
        list(range(12)),
        domain=DomainKind.BUILD_ERROR,
        graph_edges=DomainEdgeIndex.from_triples(
            [
                (0, 4, DomainEdgeKind.BUILD_TARGET_DEP),
                (1, 8, DomainEdgeKind.DIAG_PRIMARY_LOCATION),
            ]
        ),
    )
    cfg = DomainGraphRouteConfig(
        num_blocks=3,
        max_candidates_per_query=1,
        edge_weights={
            int(DomainEdgeKind.BUILD_TARGET_DEP): 1.0,
            int(DomainEdgeKind.DIAG_PRIMARY_LOCATION): 5.0,
        },
    )

    assert build_domain_block_candidates(packet, config=cfg)[0] == [2]
