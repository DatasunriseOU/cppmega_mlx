from __future__ import annotations

import copy
import hashlib
import json

import pytest

from cppmega_mlx.data.graph_recipe import (
    STAGE1_GRAPH_RECIPE_SHA256,
    stage1_graph_config_kwargs,
    stage1_graph_recipe_binding,
    stage1_graph_recipe_payload,
    validate_stage1_graph_config,
    validate_stage1_graph_contract,
)
from cppmega_mlx.training.objective_mixer import GraphAuxLossConfig


def _graph_contract() -> dict[str, object]:
    return {
        **stage1_graph_recipe_payload(),
        "recipe": stage1_graph_recipe_binding(),
    }


def test_stage1_graph_recipe_literal_sha_matches_canonical_payload() -> None:
    payload = json.dumps(
        stage1_graph_recipe_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert hashlib.sha256(payload).hexdigest() == STAGE1_GRAPH_RECIPE_SHA256


def test_stage1_graph_contract_rejects_stale_recipe_sha() -> None:
    graph = _graph_contract()
    graph["recipe"] = {
        **stage1_graph_recipe_binding(),
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="recipe binding is missing or stale"):
        validate_stage1_graph_contract(graph)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("relations", ["call", "type"]),
        ("topk", 8),
        ("indexer_weight", "1/500"),
    ),
)
def test_stage1_graph_contract_rejects_recipe_field_drift(
    field: str,
    bad_value: object,
) -> None:
    graph = copy.deepcopy(_graph_contract())
    graph[field] = bad_value

    with pytest.raises(ValueError, match=rf"graph_auxiliary\.{field}"):
        validate_stage1_graph_contract(graph)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("relations", ("call", "type")),
        ("topk", 8),
        ("indexer_weight", 0.002),
    ),
)
def test_stage1_graph_config_rejects_runtime_drift(
    field: str,
    bad_value: object,
) -> None:
    values = stage1_graph_config_kwargs()
    values[field] = bad_value
    config = GraphAuxLossConfig(**values)

    with pytest.raises(ValueError, match=rf"Stage-1 graph {field}"):
        validate_stage1_graph_config(config)
