from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.domain_packet import (
    DomainEdgeIndex,
    DomainPacket,
    wrap_with_domain_tokens,
)
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
    delimiter_token_ids,
)


def test_domain_packet_filled_and_wrapped() -> None:
    edges = DomainEdgeIndex.from_triples(
        [(0, 2, DomainEdgeKind.BUILD_TARGET_SOURCE)]
    )
    packet = DomainPacket.filled(
        [10, 11, 12],
        domain=DomainKind.CMAKE,
        role=DomainRoleKind.COMMAND,
        confidence=ParseConfidence.EXACT,
        graph_edges=edges,
        metadata={"build_kind": "cmake"},
    )

    wrapped = wrap_with_domain_tokens(packet)
    start_id, end_id = delimiter_token_ids(DomainKind.CMAKE)

    assert np.asarray(wrapped.token_ids).tolist() == [start_id, 10, 11, 12, end_id]
    assert np.asarray(wrapped.domain_ids).tolist() == [2, 2, 2, 2, 2]
    assert np.asarray(wrapped.role_ids).tolist() == [
        int(DomainRoleKind.DELIMITER),
        int(DomainRoleKind.COMMAND),
        int(DomainRoleKind.COMMAND),
        int(DomainRoleKind.COMMAND),
        int(DomainRoleKind.DELIMITER),
    ]
    assert wrapped.graph_edges.to_triples() == [
        (1, 3, int(DomainEdgeKind.BUILD_TARGET_SOURCE))
    ]
    assert wrapped.metadata["domain_wrapped"] is True
    assert wrapped.metadata["build_kind"] == "cmake"


def test_domain_packet_rejects_misaligned_sidecar() -> None:
    with pytest.raises(ValueError, match="domain_ids.*length 2.*token_ids length 3"):
        DomainPacket(
            token_ids=mx.array([1, 2, 3], dtype=mx.int32),
            domain=DomainKind.MAKE,
            domain_ids=mx.array([int(DomainKind.MAKE), int(DomainKind.MAKE)]),
            role_ids=mx.array([0, 0, 0]),
            entity_ids=mx.array([0, 0, 0]),
            scope_ids=mx.array([0, 0, 0]),
            source_doc_ids=mx.array([0, 0, 0]),
            confidence_ids=mx.array([int(ParseConfidence.EXACT)] * 3),
        )


def test_domain_packet_rejects_out_of_range_edge() -> None:
    with pytest.raises(ValueError, match="outside token range"):
        DomainPacket.filled(
            [1, 2],
            domain=DomainKind.BASH,
            graph_edges=DomainEdgeIndex.from_triples(
                [(0, 2, DomainEdgeKind.SHELL_PIPE)]
            ),
        )
