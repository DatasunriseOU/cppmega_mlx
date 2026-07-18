from __future__ import annotations

from pathlib import Path

import pytest

from cppmega_mlx.data.domain_prompt_graph import (
    build_domain_prompt_graph,
    domain_prompt_graph_from_frozen,
    render_domain_prompt_graph_spec,
)
from cppmega_mlx.tokenizer.cpp_tokenizer import load_cppmega_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def _tokenizer():
    return load_cppmega_tokenizer(ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json")


def _route_spec() -> dict:
    return {
        "parts": [
            {
                "domain": "KSH",
                "path": "scripts/run.ksh",
                "text": "print input.txt | tee out.txt\n",
            },
            {
                "domain": "BUILD_ERROR",
                "path": "build.log",
                "text": "ninja: build stopped: subcommand failed with exit code 1\n",
            },
        ],
        "cross_domain_edges": [
            {
                "from_part": 0,
                "to_part": 1,
                "from_role": "COMMAND",
                "to_role": "COMMAND",
                "kind": "EMBEDDED_DOMAIN",
            }
        ],
    }


def test_domain_prompt_graph_builds_typed_sidecars_and_cross_route() -> None:
    graph = build_domain_prompt_graph(_tokenizer(), _route_spec())

    graph.validate()
    assert graph.receipt["schema"] == "cppmega_domain_prompt_graph_v1"
    assert graph.edge_counts == {
        "domain": 0,
        "build": 0,
        "shell": 3,
        "diagnostic": 2,
        "cross_domain": 1,
    }
    assert graph.eval_sidecars["token_cross_domain_edges"] == (
        {"from": 2, "to": 21, "kind": 100},
    )
    assert graph.side_channels["domain_ids"][2] == 24
    assert graph.side_channels["domain_ids"][21] == 43

    window = graph.model_inputs(
        total_token_count=len(graph.token_ids) + 1,
        window_start=0,
        window_end=len(graph.token_ids),
    )
    assert window.edge_kind_route_counts() == {40: 1, 44: 2, 64: 2, 100: 1}
    assert sum(sum(row) for row in window.dense_relation_attention_bias()) > 0.0
    assert sum(sum(row) for row in window.dense_edge_kind_attention_bias()) > 0.0


def test_domain_prompt_graph_rendering_is_explicitly_bound_to_parts() -> None:
    rendered = render_domain_prompt_graph_spec(_route_spec())

    assert rendered.startswith("<KSH_START>print input.txt")
    assert "<BUILD_ERROR_START>ninja: build stopped" in rendered
    assert rendered.endswith("exit code 1\n<BUILD_ERROR_END>")


def test_domain_prompt_graph_rejects_cross_edge_without_typed_anchor() -> None:
    spec = _route_spec()
    spec["cross_domain_edges"][0]["from_role"] = "TARGET"

    with pytest.raises(ValueError, match="no token with role TARGET"):
        build_domain_prompt_graph(_tokenizer(), spec)


def test_domain_prompt_graph_frozen_round_trip_preserves_edge_families() -> None:
    graph = build_domain_prompt_graph(_tokenizer(), _route_spec())
    frozen = domain_prompt_graph_from_frozen(
        graph.token_ids,
        graph.eval_sidecars,
        part_ranges=graph.part_ranges,
    )

    assert frozen.token_ids == graph.token_ids
    assert frozen.eval_sidecars == graph.eval_sidecars
    assert frozen.part_ranges == graph.part_ranges
