from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.domain_schema import DomainEdgeKind
from cppmega_mlx.nn.code_graph_routes import build_token_graph_biases
from cppmega_mlx.training.objective_data import (
    empty_objective_routes,
    graph_batch_from_objective_routes,
)


def _routes() -> dict[str, list[object]]:
    routes = empty_objective_routes()
    routes.update(
        {
            "token_chunk_starts": [0, 2],
            "token_chunk_ends": [2, 4],
            "token_chunk_kinds": [1, 1],
            "token_chunk_dep_levels": [0, 0],
            "token_call_edges": [{"from": 1, "to": 0}],
            "token_domain_edges": [
                {
                    "from": 3,
                    "to": 1,
                    "kind": int(DomainEdgeKind.DIAG_PRIMARY_LOCATION),
                }
            ],
        }
    )
    return routes


def test_objective_routes_build_typed_batch_and_dense_bias() -> None:
    graph_batch = graph_batch_from_objective_routes(
        _routes(), input_length=4, where="adapter-test"
    )
    relation_bias, edge_kind_bias = build_token_graph_biases(
        graph_batch,
        batch_size=1,
        seq_length=4,
        document_ids=mx.ones((1, 4), dtype=mx.int32),
    )
    mx.eval(relation_bias, edge_kind_bias)

    assert graph_batch.batch_size == 1
    assert tuple(relation_bias.shape) == (1, 4, 4)
    assert float(relation_bias[0, 2, 0].item()) > 0.0
    assert float(relation_bias[0, 3, 1].item()) > 0.0
    assert np.isfinite(np.asarray(relation_bias)).all()
    assert np.isfinite(np.asarray(edge_kind_bias)).all()


def test_objective_route_adapter_rejects_missing_columns_and_bad_endpoints() -> None:
    missing = _routes()
    del missing["token_chunk_ends"]
    with pytest.raises(ValueError, match="token_chunk_ends must be a sequence"):
        graph_batch_from_objective_routes(
            missing, input_length=4, where="missing-route-column"
        )

    malformed = _routes()
    malformed["token_call_edges"] = [{"from": 2, "to": 0}]
    with pytest.raises(ValueError, match=r"call edge \(2, 0\) is outside 2 chunks"):
        graph_batch_from_objective_routes(
            malformed, input_length=4, where="bad-route-endpoint"
        )
