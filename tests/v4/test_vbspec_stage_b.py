"""VBSpec Stage B tests — resolve_shapes + ResolvedBrickGraph."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.spec import (
    DiagnosticSeverity,
    ResolveError,
    ResolvedBrickGraph,
    ResolvedEdge,
    ShapeDiagnostic,
    ShapeExpr,
    register_contract,
    resolve_shapes,
)
from cppmega_v4.spec.shape_contract import BrickShapeContract


_QWEN3_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


_LING_ENV = {
    **_QWEN3_ENV,
    "S": 8192,
    "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
}


# ---------------------------------------------------------------------------
# Empty / size-1 graphs
# ---------------------------------------------------------------------------


def test_resolve_empty_graph_returns_no_diagnostics():
    g = BrickGraph(nodes=(), edges=())
    out = resolve_shapes(g, _QWEN3_ENV)
    assert isinstance(out, ResolvedBrickGraph)
    assert out.edges == ()
    assert out.diagnostics == ()
    assert out.has_errors is False


def test_resolve_size_one_graph_warns_on_unsupplied_side_channel():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="g0"),))
    out = resolve_shapes(g, _QWEN3_ENV, strict=False)
    # gdn needs doc_ids; we didn't supply it → WARNING
    assert any(
        d.severity is DiagnosticSeverity.WARNING and "doc_ids" in d.message
        for d in out.diagnostics
    )
    assert out.has_errors is False


def test_resolve_size_one_with_side_channel_supplied_has_no_warnings():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="g0"),))
    out = resolve_shapes(
        g, _QWEN3_ENV,
        available_side_channels=frozenset({"doc_ids"}),
    )
    assert out.warnings == ()


# ---------------------------------------------------------------------------
# Linear chain resolves cleanly
# ---------------------------------------------------------------------------


def test_resolve_qwen3_next_chain_under_qwen_env_has_no_errors():
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    out = resolve_shapes(
        g, _QWEN3_ENV,
        available_side_channels=frozenset({"doc_ids"}),
    )
    assert out.has_errors is False
    # Edges have shape (B, S, H) throughout.
    for edge in out.edges:
        assert isinstance(edge, ResolvedEdge)
        assert edge.matched is True
        assert edge.shape == (1, 4096, 4096)


def test_resolve_ling26_chain_under_ling_env_has_no_errors():
    specs = build_preset_specs("ling26", hidden_size=4096)
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    out = resolve_shapes(
        g, _LING_ENV,
        available_side_channels=frozenset({"doc_ids"}),
    )
    assert out.has_errors is False


# ---------------------------------------------------------------------------
# Synthetic mismatch — strict vs lenient
# ---------------------------------------------------------------------------


def _install_mismatch_contracts():
    """Register two synthetic kinds whose shapes never line up."""
    big = BrickShapeContract(
        inputs={"x": ShapeExpr(("B", "S", "H"))},
        outputs={"y": ShapeExpr(("B", "S", "H * 2"))},
        params_elems=ShapeExpr(("0",)),
        activations_elems=ShapeExpr(("0",)),
        kv_cache_elems=ShapeExpr(("0",)),
        description="synthetic; doubles H on output",
    )
    small = BrickShapeContract(
        inputs={"x": ShapeExpr(("B", "S", "H"))},
        outputs={"y": ShapeExpr(("B", "S", "H"))},
        params_elems=ShapeExpr(("0",)),
        activations_elems=ShapeExpr(("0",)),
        kv_cache_elems=ShapeExpr(("0",)),
        description="synthetic identity",
    )
    register_contract("__test_doubles", big)
    register_contract("__test_identity", small)


def _patched_block_builders():
    """Allow BrickNode to accept our synthetic kinds without registering
    full BLOCK_BUILDERS entries."""
    from cppmega_v4.fusion import brick_graph as bg_module
    # Use the _ADDITIONAL_FUSION_KINDS escape hatch.
    bg_module._ADDITIONAL_FUSION_KINDS = bg_module._ADDITIONAL_FUSION_KINDS | {
        "__test_doubles", "__test_identity",
    }


def test_resolve_strict_mode_raises_on_first_error():
    _install_mismatch_contracts()
    _patched_block_builders()
    a = BrickNode(kind="__test_doubles", name="a")
    b = BrickNode(kind="__test_identity", name="b")
    g = BrickGraph(nodes=(a, b), edges=(("a", "b"),))
    with pytest.raises(ResolveError, match="shape mismatch"):
        resolve_shapes(g, _QWEN3_ENV, strict=True)


def test_resolve_lenient_mode_collects_diagnostic_instead_of_raising():
    _install_mismatch_contracts()
    _patched_block_builders()
    a = BrickNode(kind="__test_doubles", name="a")
    b = BrickNode(kind="__test_identity", name="b")
    g = BrickGraph(nodes=(a, b), edges=(("a", "b"),))
    out = resolve_shapes(g, _QWEN3_ENV, strict=False)
    assert out.has_errors is True
    errs = out.errors
    assert len(errs) == 1
    assert errs[0].producer == "a"
    assert errs[0].consumer == "b"
    assert "shape mismatch" in errs[0].message
    # Suggestion: last-dim mismatch → Linear bridge.
    assert errs[0].suggested_fix is not None
    assert "Linear(" in errs[0].suggested_fix
    # Edge is still recorded so callers can render it red.
    edge = out.edge("a", "b")
    assert edge.matched is False
    assert edge.producer_shape != edge.consumer_shape


def test_resolve_lenient_collects_multiple_diagnostics():
    _install_mismatch_contracts()
    _patched_block_builders()
    a = BrickNode(kind="__test_doubles", name="a")
    b = BrickNode(kind="__test_identity", name="b")
    c = BrickNode(kind="__test_doubles", name="c")
    d = BrickNode(kind="__test_identity", name="d")
    g = BrickGraph(
        nodes=(a, b, c, d),
        edges=(("a", "b"), ("b", "c"), ("c", "d")),
    )
    out = resolve_shapes(g, _QWEN3_ENV, strict=False)
    # Two error edges: a->b (H*2 vs H) and c->d (H*2 vs H). b->c matches.
    assert len(out.errors) == 2
    assert {(e.producer, e.consumer) for e in out.errors} == {
        ("a", "b"), ("c", "d"),
    }


# ---------------------------------------------------------------------------
# Suggested-fix heuristics
# ---------------------------------------------------------------------------


def test_resolve_suggests_merge_heads_when_rank4_to_rank3():
    """A brick whose declared output is (B,nh,S,head_dim) feeding a brick
    expecting (B,S,nh*head_dim) gets a merge_heads suggestion."""
    bnh_sd = BrickShapeContract(
        inputs={"x": ShapeExpr(("B", "S", "H"))},
        outputs={"y": ShapeExpr(("B", "nh", "S", "head_dim"))},
        params_elems=ShapeExpr(("0",)),
        activations_elems=ShapeExpr(("0",)),
        kv_cache_elems=ShapeExpr(("0",)),
        description="rank-4 heads-major output",
    )
    bsh = BrickShapeContract(
        inputs={"x": ShapeExpr(("B", "S", "nh * head_dim"))},
        outputs={"y": ShapeExpr(("B", "S", "H"))},
        params_elems=ShapeExpr(("0",)),
        activations_elems=ShapeExpr(("0",)),
        kv_cache_elems=ShapeExpr(("0",)),
        description="rank-3 heads-flat input",
    )
    register_contract("__test_heads_major", bnh_sd)
    register_contract("__test_heads_flat", bsh)
    _patched_block_builders()
    from cppmega_v4.fusion import brick_graph as bg_module
    bg_module._ADDITIONAL_FUSION_KINDS = bg_module._ADDITIONAL_FUSION_KINDS | {
        "__test_heads_major", "__test_heads_flat",
    }
    g = BrickGraph(
        nodes=(
            BrickNode(kind="__test_heads_major", name="a"),
            BrickNode(kind="__test_heads_flat", name="b"),
        ),
        edges=(("a", "b"),),
    )
    out = resolve_shapes(g, _QWEN3_ENV, strict=False)
    errs = out.errors
    assert len(errs) == 1
    assert "merge_heads" in (errs[0].suggested_fix or "")


# ---------------------------------------------------------------------------
# Opaque contracts downgrade ERROR to WARNING
# ---------------------------------------------------------------------------


def test_resolve_opaque_brick_boundary_emits_warning_not_error():
    """A brick whose contract is marked opaque_shape preserves the
    rank-only invariant — declared shape mismatch becomes a warning."""
    opaque_out = BrickShapeContract(
        inputs={"x": ShapeExpr(("B", "S", "H"))},
        outputs={"y": ShapeExpr(("B", "S", "H * 2"))},
        params_elems=ShapeExpr(("0",)),
        activations_elems=ShapeExpr(("0",)),
        kv_cache_elems=ShapeExpr(("0",)),
        opaque_shape=True,
        description="opaque doubler — trust me, B/S/H preserved",
    )
    identity = BrickShapeContract(
        inputs={"x": ShapeExpr(("B", "S", "H"))},
        outputs={"y": ShapeExpr(("B", "S", "H"))},
        params_elems=ShapeExpr(("0",)),
        activations_elems=ShapeExpr(("0",)),
        kv_cache_elems=ShapeExpr(("0",)),
        description="identity",
    )
    register_contract("__test_opaque_doubler", opaque_out)
    register_contract("__test_identity2", identity)
    from cppmega_v4.fusion import brick_graph as bg_module
    bg_module._ADDITIONAL_FUSION_KINDS = bg_module._ADDITIONAL_FUSION_KINDS | {
        "__test_opaque_doubler", "__test_identity2",
    }
    g = BrickGraph(
        nodes=(
            BrickNode(kind="__test_opaque_doubler", name="a"),
            BrickNode(kind="__test_identity2", name="b"),
        ),
        edges=(("a", "b"),),
    )
    out = resolve_shapes(g, _QWEN3_ENV, strict=False)
    assert out.has_errors is False
    assert len(out.warnings) >= 1
    assert any("opaque" in w.message for w in out.warnings)


# ---------------------------------------------------------------------------
# Missing dim_env raises with a useful message
# ---------------------------------------------------------------------------


def test_resolve_missing_dim_in_env_strict_raises():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="g0"),))
    env = {"B": 1, "S": 16}  # no H, nh, head_dim
    with pytest.raises(ResolveError, match="missing dim_env"):
        resolve_shapes(g, env, strict=True)


def test_resolve_missing_dim_in_env_lenient_records_error():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="g0"),))
    env = {"B": 1, "S": 16}
    out = resolve_shapes(g, env, strict=False)
    assert out.has_errors is True
    assert any("failed to resolve" in e.message for e in out.errors)


# ---------------------------------------------------------------------------
# System: every preset resolves cleanly under preset-shaped envs
# ---------------------------------------------------------------------------


_PRESET_DEFAULT_ENVS = {
    "qwen3_next":      {**_QWEN3_ENV},
    "kimi_linear":     {**_LING_ENV},
    "kimi_k2":         {**_LING_ENV},
    "deepseek_v3":     {**_LING_ENV},
    "deepseek_v4_flash": {**_QWEN3_ENV},
    "gemma4":          {**_QWEN3_ENV, "sliding_window_size": 1024},
    "mistral4":        {**_LING_ENV},
    "ling26":          {**_LING_ENV},
    "longcat":         {**_LING_ENV},
    "nemotron3":       {**_QWEN3_ENV, "d_state": 64},
    "zaya1":           {**_QWEN3_ENV, "fine_window": 256, "coarse_block_size": 16},
    "arcee_trinity":   {**_QWEN3_ENV, "sliding_window_size": 1024},
}


@pytest.mark.parametrize("preset_name", sorted(_PRESET_DEFAULT_ENVS.keys()))
def test_system_every_preset_resolves_cleanly_under_default_env(preset_name):
    specs = build_preset_specs(preset_name, hidden_size=4096)
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    env = _PRESET_DEFAULT_ENVS[preset_name]
    out = resolve_shapes(
        g, env, strict=False,
        available_side_channels=frozenset({"doc_ids", "token_ids"}),
    )
    # Opaque-brick boundaries (nsa, csa_hca, dsv4) may emit warnings but
    # no preset should produce an ERROR at preset-shaped envs.
    assert out.has_errors is False, (
        f"{preset_name!r} resolution errors: "
        f"{[e.message for e in out.errors]}"
    )
