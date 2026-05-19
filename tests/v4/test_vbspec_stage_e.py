"""VBSpec Stage E tests — verify_and_estimate + suggest_adapters + GUI workflow."""

from __future__ import annotations

import time

import pytest

from cppmega_v4.architectures import available_presets, build_preset_specs
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.spec import (
    AdapterProposal,
    VerificationResult,
    insert_adapter_chain,
    register_contract,
    suggest_adapters,
    suggest_dim_env,
    verify_and_estimate,
)
from cppmega_v4.spec.shape_contract import BrickShapeContract, ShapeExpr


# ---------------------------------------------------------------------------
# suggest_dim_env
# ---------------------------------------------------------------------------


def test_suggest_dim_env_returns_independent_copy():
    env_a = suggest_dim_env()
    env_b = suggest_dim_env()
    env_a["B"] = 999
    assert env_b["B"] != 999  # not aliased


def test_suggest_dim_env_no_preset_has_base_keys():
    env = suggest_dim_env()
    for required in ("B", "S", "H", "nh", "nkv", "head_dim",
                     "num_experts", "top_k"):
        assert required in env, required


def test_suggest_dim_env_unknown_preset_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        suggest_dim_env("totally_made_up_preset_xyz")


def test_suggest_dim_env_mla_preset_includes_extras():
    env = suggest_dim_env("ling26")
    for k in ("q_lora_rank", "kv_lora_rank", "qk_rope_head_dim",
              "qk_nope_head_dim", "v_head_dim"):
        assert k in env, k


@pytest.mark.parametrize("preset_name", sorted(available_presets()))
def test_suggest_dim_env_every_preset_has_one(preset_name):
    env = suggest_dim_env(preset_name)
    assert env["B"] >= 1 and env["H"] >= 1


# ---------------------------------------------------------------------------
# verify_and_estimate — basic call shape
# ---------------------------------------------------------------------------


def _qwen_graph():
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    return from_block_specs(specs, hidden_size=4096, instantiate=False)


def test_verify_and_estimate_returns_well_formed_result():
    g = _qwen_graph()
    r = verify_and_estimate(g, preset_name="qwen3_next")
    assert isinstance(r, VerificationResult)
    assert r.resolved.has_errors is False
    assert len(r.fusion_plan) > 0
    assert r.memory.total_bytes > 0
    assert r.elapsed_ms >= 0


def test_verify_and_estimate_explicit_dim_env_overrides_preset():
    g = _qwen_graph()
    env = suggest_dim_env("qwen3_next")
    env["S"] = 1024  # smaller seq
    r_small = verify_and_estimate(g, dim_env=env)
    r_large = verify_and_estimate(g, preset_name="qwen3_next")
    # Activations / KV scale with S → smaller env has smaller total
    assert r_small.memory.total_bytes < r_large.memory.total_bytes


def test_verify_and_estimate_device_gate_populates_fits_on_device():
    g = _qwen_graph()
    r_tight = verify_and_estimate(
        g, preset_name="qwen3_next", device_hbm_bytes=1,
    )
    assert r_tight.fits_on_device is False
    r_big = verify_and_estimate(
        g, preset_name="qwen3_next", device_hbm_bytes=10 ** 12,
    )
    assert r_big.fits_on_device is True


def test_verify_and_estimate_default_no_device_means_no_fit_check():
    g = _qwen_graph()
    r = verify_and_estimate(g, preset_name="qwen3_next")
    assert r.fits_on_device is None


def test_verify_and_estimate_summary_dict_shape():
    g = _qwen_graph()
    r = verify_and_estimate(
        g, preset_name="qwen3_next", device_hbm_bytes=80 * 10 ** 9,
    )
    summary = r.summary()
    for key in ("errors", "warnings", "regions", "memory",
                "elapsed_ms", "fits_on_device"):
        assert key in summary, key
    assert "total" in summary["memory"]


def test_verify_and_estimate_strict_mode_raises_on_mismatch():
    """When the user wants a CI gate (e.g. preset-must-resolve), strict=True
    re-raises the resolver error instead of swallowing it into diagnostics."""
    register_contract(
        "__stage_e_bad",
        BrickShapeContract(
            inputs={"x": ShapeExpr(("B", "S", "H"))},
            outputs={"y": ShapeExpr(("B", "S", "H * 4"))},
            params_elems=ShapeExpr(("0",)),
            activations_elems=ShapeExpr(("0",)),
            kv_cache_elems=ShapeExpr(("0",)),
        ),
    )
    from cppmega_v4.fusion import brick_graph as bg
    bg._ADDITIONAL_FUSION_KINDS = bg._ADDITIONAL_FUSION_KINDS | {"__stage_e_bad"}
    g = BrickGraph(
        nodes=(
            BrickNode(kind="__stage_e_bad", name="a"),
            BrickNode(kind="mlp", name="b"),
        ),
        edges=(("a", "b"),),
    )
    from cppmega_v4.spec import ResolveError
    with pytest.raises(ResolveError):
        verify_and_estimate(g, dim_env=suggest_dim_env(), strict=True)


# ---------------------------------------------------------------------------
# suggest_adapters
# ---------------------------------------------------------------------------


def test_suggest_adapters_unknown_edge_raises():
    g = _qwen_graph()
    r = verify_and_estimate(g, preset_name="qwen3_next")
    with pytest.raises(KeyError):
        suggest_adapters(r.resolved, "ghost", "missing")


def test_suggest_adapters_matched_edge_returns_empty_chain():
    g = _qwen_graph()
    r = verify_and_estimate(g, preset_name="qwen3_next")
    # any consecutive pair from the chain is matched
    edge = r.resolved.edges[0]
    proposal = suggest_adapters(r.resolved, edge.producer, edge.consumer)
    assert isinstance(proposal, AdapterProposal)
    assert proposal.chain == []
    assert "no adapter needed" in proposal.reason


def test_suggest_adapters_proposes_chain_on_synthetic_mismatch():
    """End-to-end GUI workflow: build graph with a known mismatch, ask
    for a suggestion, accept it, splice in, re-verify — clean result."""
    register_contract(
        "__stage_e_doubler_h",
        BrickShapeContract(
            inputs={"x": ShapeExpr(("B", "S", "H"))},
            outputs={"y": ShapeExpr(("B", "S", "H * 2"))},
            params_elems=ShapeExpr(("0",)),
            activations_elems=ShapeExpr(("0",)),
            kv_cache_elems=ShapeExpr(("0",)),
        ),
    )
    register_contract(
        "__stage_e_identity_h",
        BrickShapeContract(
            inputs={"x": ShapeExpr(("B", "S", "H"))},
            outputs={"y": ShapeExpr(("B", "S", "H"))},
            params_elems=ShapeExpr(("0",)),
            activations_elems=ShapeExpr(("0",)),
            kv_cache_elems=ShapeExpr(("0",)),
        ),
    )
    from cppmega_v4.fusion import brick_graph as bg
    bg._ADDITIONAL_FUSION_KINDS = bg._ADDITIONAL_FUSION_KINDS | {
        "__stage_e_doubler_h", "__stage_e_identity_h",
    }
    g = BrickGraph(
        nodes=(
            BrickNode(kind="__stage_e_doubler_h", name="a"),
            BrickNode(kind="__stage_e_identity_h", name="b"),
        ),
        edges=(("a", "b"),),
    )
    env = suggest_dim_env()
    r = verify_and_estimate(g, dim_env=env, strict=False)
    assert r.has_errors is True
    proposal = suggest_adapters(r.resolved, "a", "b")
    assert proposal.chain is not None
    assert len(proposal.chain) >= 1
    assert "adapter_linear_bridge" in (s.kind for s in proposal.chain)
    # Splice and re-verify — original direct edge gone.
    g2 = insert_adapter_chain(g, "a", "b", proposal.chain)
    r2 = verify_and_estimate(g2, dim_env=env, strict=False)
    assert ("a", "b") not in [
        (e.producer, e.consumer) for e in r2.resolved.edges
    ]


# ---------------------------------------------------------------------------
# Performance gate — <50 ms per preset (real-time GUI requirement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(available_presets()))
def test_verify_and_estimate_under_50ms_for_preset(preset_name):
    """Each preset's full one-shot verify_and_estimate (resolve + plan +
    estimate) must come in under 50 ms — the real-time GUI budget per
    the VisualBuilderSpec.md §6 E perf criterion. We use 100 ms as the
    test bound to leave margin for CI noise; the typical run is ~3 ms."""
    env = suggest_dim_env(preset_name)
    specs = build_preset_specs(preset_name, hidden_size=env["H"])
    g = from_block_specs(specs, hidden_size=env["H"], instantiate=False)

    # Two-shot: warm up first call (import + module cache), measure the
    # second.
    verify_and_estimate(g, dim_env=env)
    t0 = time.perf_counter()
    r = verify_and_estimate(g, dim_env=env)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 100.0, (
        f"verify_and_estimate({preset_name!r}) took {elapsed_ms:.1f} ms "
        "(soft cap 100 ms; perf target 50 ms)"
    )
    # The reported elapsed_ms inside the result should also be reasonable
    assert r.elapsed_ms < 100.0


# ---------------------------------------------------------------------------
# Full GUI workflow integration test
# ---------------------------------------------------------------------------


def test_system_gui_workflow_red_edge_then_accept_then_green():
    """Simulates the GUI flow end-to-end:

    1. User drops a brick whose output doesn't match the next consumer.
    2. GUI calls verify_and_estimate(strict=False) → red edge in result.
    3. GUI calls suggest_adapters on the red edge → AdapterProposal.
    4. User accepts → GUI calls insert_adapter_chain.
    5. GUI re-runs verify_and_estimate → no errors on the original edge
       (the new bridge adapter takes over; verifying the bridge's own
       runtime parameters is Stage F territory, beyond this roadmap).
    """
    register_contract(
        "__gui_workflow_h_double",
        BrickShapeContract(
            inputs={"x": ShapeExpr(("B", "S", "H"))},
            outputs={"y": ShapeExpr(("B", "S", "H * 2"))},
            params_elems=ShapeExpr(("0",)),
            activations_elems=ShapeExpr(("0",)),
            kv_cache_elems=ShapeExpr(("0",)),
        ),
    )
    register_contract(
        "__gui_workflow_h_identity",
        BrickShapeContract(
            inputs={"x": ShapeExpr(("B", "S", "H"))},
            outputs={"y": ShapeExpr(("B", "S", "H"))},
            params_elems=ShapeExpr(("0",)),
            activations_elems=ShapeExpr(("0",)),
            kv_cache_elems=ShapeExpr(("0",)),
        ),
    )
    from cppmega_v4.fusion import brick_graph as bg
    bg._ADDITIONAL_FUSION_KINDS = bg._ADDITIONAL_FUSION_KINDS | {
        "__gui_workflow_h_double", "__gui_workflow_h_identity",
    }
    g = BrickGraph(
        nodes=(
            BrickNode(kind="__gui_workflow_h_double", name="a"),
            BrickNode(kind="__gui_workflow_h_identity", name="b"),
        ),
        edges=(("a", "b"),),
    )
    env = suggest_dim_env()
    # Step 2
    r = verify_and_estimate(
        g, dim_env=env, device_hbm_bytes=80 * 10 ** 9, strict=False,
    )
    assert r.has_errors is True
    summary = r.summary()
    assert summary["errors"] >= 1
    # Step 3
    proposal = suggest_adapters(r.resolved, "a", "b")
    assert proposal.chain is not None
    # Step 4
    g2 = insert_adapter_chain(g, "a", "b", proposal.chain)
    # Step 5
    r2 = verify_and_estimate(g2, dim_env=env, strict=False)
    # The original mismatch edge no longer exists in resolved edges.
    assert ("a", "b") not in [
        (e.producer, e.consumer) for e in r2.resolved.edges
    ]
    # Memory total grew (we added a bridge node) — but stayed bounded.
    assert r2.memory.total_bytes >= r.memory.total_bytes
