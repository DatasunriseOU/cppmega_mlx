"""E7-4 tests: suggest_optim_groups RPC + group_inference heuristic."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec.group_inference import (
    AutoGroupingResult, suggest_groups,
)
from cppmega_v4.buildspec.optim_spec import OptimKind
from cppmega_v4.jsonrpc import dispatch
from cppmega_v4.jsonrpc.schema import METHOD_REGISTRY


def _graph_for(preset: str, hidden: int = 128):
    specs = build_preset_specs(preset, hidden_size=hidden)
    return {
        "nodes": [
            {"id": s.get("name"), "kind": s["kind"],
             "params": s.get("params", {})}
            for s in specs
        ],
        "edges": [
            {"src": specs[i].get("name"), "dst": specs[i + 1].get("name")}
            for i in range(len(specs) - 1)
        ],
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_method_registered():
    assert "suggest_optim_groups" in METHOD_REGISTRY


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_dispatch_llama_hybrid_yields_two_groups():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": _graph_for("llama3_8b"),
                   "optim_kind": "muon_adamw_hybrid",
                   "hidden_size": 128},
    })
    assert resp.error is None
    props = resp.result["proposals"]
    # llama3_8b at H=128 = attention+mlp with 1D biases + 2D weights
    optim_kinds = {p["optim_kind"] for p in props}
    assert "muon" in optim_kinds
    assert "adamw" in optim_kinds
    assert resp.result["uncovered_params"] == 0
    assert resp.result["total_params"] > 0


def test_dispatch_pure_adamw_yields_adamw_groups():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": _graph_for("llama3_8b"),
                   "optim_kind": "adamw",
                   "hidden_size": 128},
    })
    assert resp.error is None
    props = resp.result["proposals"]
    optim_kinds = {p["optim_kind"] for p in props}
    assert optim_kinds == {"adamw"}


def test_dispatch_pure_lion_yields_lion_groups():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": _graph_for("llama3_8b"),
                   "optim_kind": "lion",
                   "hidden_size": 128},
    })
    assert resp.error is None
    props = resp.result["proposals"]
    optim_kinds = {p["optim_kind"] for p in props}
    # Lion route: 1D goes to AdamW (no Lion-1D split implemented), 2D
    # backbone goes to Lion
    assert "lion" in optim_kinds


def test_dispatch_rejects_unknown_optim_kind():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": _graph_for("llama3_8b"),
                   "optim_kind": "sophia",
                   "hidden_size": 128},
    })
    assert resp.error is not None
    assert "sophia" in str(resp.error.data).lower()


def test_dispatch_handles_empty_graph_gracefully():
    """Empty graph → empty proposals, no crash."""
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": {"nodes": [], "edges": []},
                   "optim_kind": "adamw",
                   "hidden_size": 128},
    })
    assert resp.error is None
    assert resp.result["proposals"] == []
    assert resp.result["total_params"] == 0


# ---------------------------------------------------------------------------
# Coverage across multiple family-reps (real-world parity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", [
    "llama3_8b", "deepseek_v3", "gemma3_27b", "mistral4",
    "glm_45", "granite_4_1",
])
def test_hybrid_zero_uncovered_for_family_reps(preset):
    """Every parameter must end up in some group — guard against
    silently-dropped tensors."""
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": _graph_for(preset),
                   "optim_kind": "muon_adamw_hybrid",
                   "hidden_size": 128},
    })
    assert resp.error is None
    assert resp.result["uncovered_params"] == 0, (
        f"{preset}: {resp.result['uncovered_params']} uncovered "
        f"out of {resp.result['total_params']}"
    )


# ---------------------------------------------------------------------------
# Direct heuristic tests
# ---------------------------------------------------------------------------


def test_suggest_groups_classifies_synthetic_params():
    """Pass a synthetic params dict and verify bucketing."""
    import mlx.core as mx
    params = {
        "block1": {
            "weight": mx.zeros((128, 128)),         # backbone 2D
            "bias": mx.zeros((128,)),                # 1D bias
            "norm": {"weight": mx.zeros((128,))},   # 1D norm
        },
        "lm_head": {"weight": mx.zeros((50257, 128))},  # embedding-like
    }
    res = suggest_groups(params, OptimKind.MUON_ADAMW_HYBRID)
    assert isinstance(res, AutoGroupingResult)
    assert res.uncovered_params == 0
    matchers = {p.matcher for p in res.proposals}
    # backbone weight → muon group
    assert any("weight" in m for m in matchers)
    # at least one bias/norm AdamW group + lm_head embeddings group
    assert "embeddings" in matchers


def test_rationale_strings_are_non_empty():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "suggest_optim_groups",
        "params": {"graph": _graph_for("llama3_8b"),
                   "optim_kind": "muon_adamw_hybrid",
                   "hidden_size": 128},
    })
    for p in resp.result["proposals"]:
        assert p["rationale"], f"empty rationale for {p['matcher']}"
        assert p["param_count"] > 0
