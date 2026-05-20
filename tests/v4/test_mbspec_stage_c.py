"""MBSpec Stage C tests — MTPRewriter (head materialisation + loss/optim rewrite)."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec import (
    HeadDetectionError,
    LossRewriteError,
    LossKind,
    LossSpec,
    MTPRewriter,
    ModelBuildSpec,
    OptimSpec,
    Rewriter,
    adamw,
    cross_entropy_loss,
    ifim_shaped_loss,
    muon,
    sgd,
    verify_build_spec,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


_QWEN3_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


def _qwen_graph_with_head() -> BrickGraph:
    """Qwen3-Next preset + an mlp head appended at the end."""
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    specs.append({"kind": "mlp", "name": "logits"})
    return from_block_specs(specs, hidden_size=4096, instantiate=False)


def _minimal_graph_with_head() -> BrickGraph:
    """gdn -> mlp (the mlp is the head)."""
    return BrickGraph(
        nodes=(
            BrickNode(kind="gdn", name="backbone"),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("backbone", "logits"),),
    )


# ---------------------------------------------------------------------------
# Construction + protocol invariants
# ---------------------------------------------------------------------------


def test_mtp_rewriter_implements_rewriter_protocol():
    r = MTPRewriter(k=2)
    assert isinstance(r, Rewriter)
    assert r.name == "MTPRewriter(k=2)"
    assert "single_head" in r.required_preconditions
    assert "mtp_k_heads" in r.provided_postconditions


def test_mtp_rewriter_k1_provides_no_new_postcondition():
    """K=1 is a no-op — shouldn't advertise mtp_k_heads."""
    r = MTPRewriter(k=1)
    assert r.provided_postconditions == frozenset()


def test_mtp_rewriter_rejects_k_lt_1():
    with pytest.raises(ValueError, match="k must be ≥ 1"):
        MTPRewriter(k=0)


def test_mtp_rewriter_rejects_mismatched_beta_length():
    with pytest.raises(ValueError, match="beta length"):
        MTPRewriter(k=3, beta=(1.0, 0.5))


def test_mtp_rewriter_is_frozen_dataclass():
    r = MTPRewriter(k=2)
    with pytest.raises((AttributeError, TypeError)):
        r.k = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# K=1 fast-path
# ---------------------------------------------------------------------------


def test_mtp_k1_returns_input_spec_unchanged():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=1)(spec)
    assert out is spec or (
        out.graph is spec.graph and out.loss is spec.loss
    )


# ---------------------------------------------------------------------------
# K=2 graph rewrite
# ---------------------------------------------------------------------------


def test_mtp_k2_creates_two_head_nodes():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2)(spec)
    head_names = [n.name for n in out.graph.nodes if n.kind == "mlp"]
    assert "logits_0" in head_names
    assert "logits_1" in head_names
    assert "logits" not in head_names  # original was renamed


def test_mtp_k3_creates_three_head_nodes():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=3)(spec)
    head_names = sorted(
        n.name for n in out.graph.nodes if n.kind == "mlp"
    )
    assert head_names == ["logits_0", "logits_1", "logits_2"]


def test_mtp_wires_producer_into_every_head_clone():
    """The producer of the original head must feed every new head_i."""
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=3)(spec)
    edges = set(out.graph.edges)
    assert ("backbone", "logits_0") in edges
    assert ("backbone", "logits_1") in edges
    assert ("backbone", "logits_2") in edges


def test_mtp_preserves_backbone_untouched():
    g = _qwen_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2)(spec)
    backbone_names = {
        n.name for n in g.nodes if n.kind not in {"mlp"}
        or n.name != "logits"
    }
    out_names = {n.name for n in out.graph.nodes}
    assert backbone_names <= out_names


# ---------------------------------------------------------------------------
# Loss rewrite
# ---------------------------------------------------------------------------


def test_mtp_rewrites_ce_to_mtp_weighted():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2)(spec)
    assert out.loss.kind is LossKind.MTP_WEIGHTED
    assert int(out.loss.params["k"]) == 2
    assert out.loss.head_outputs == ("logits_0", "logits_1")
    assert out.loss.label_source == "next_k_tokens"


def test_mtp_honours_custom_beta():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=3, beta=(1.0, 0.5, 0.25))(spec)
    assert out.loss.params["beta_0"] == 1.0
    assert out.loss.params["beta_1"] == 0.5
    assert out.loss.params["beta_2"] == 0.25


def test_mtp_uses_original_head_output_name_as_prefix():
    g = BrickGraph(
        nodes=(
            BrickNode(kind="gdn", name="bb"),
            BrickNode(kind="mlp", name="vocab_proj"),
        ),
        edges=(("bb", "vocab_proj"),),
    )
    spec = ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(head_output_name="vocab_proj"),
        optim=adamw(),
    )
    out = MTPRewriter(k=2)(spec)
    assert out.loss.head_outputs == ("vocab_proj_0", "vocab_proj_1")


def test_mtp_rejects_non_ce_loss():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(
        graph=g, loss=ifim_shaped_loss(), optim=adamw(),
    )
    with pytest.raises(LossRewriteError, match="CROSS_ENTROPY"):
        MTPRewriter(k=2)(spec)


# ---------------------------------------------------------------------------
# Optim rewrite (head-only param group)
# ---------------------------------------------------------------------------


def test_mtp_adds_head_param_group_to_optim_by_default():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2)(spec)
    matchers = [g.matcher for g in out.optim.groups]
    assert matchers[0].startswith("regex:")
    assert "logits" in matchers[0]


def test_mtp_can_opt_out_of_param_group_addition():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2, add_head_param_group=False)(spec)
    assert len(out.optim.groups) == len(spec.optim.groups)


def test_mtp_head_group_inherits_betas_for_adamw():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2)(spec)
    assert out.optim.groups[0].betas == spec.optim.groups[0].betas


def test_mtp_head_group_inherits_ns_steps_for_muon():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=muon())
    out = MTPRewriter(k=2)(spec)
    assert out.optim.groups[0].ns_steps == spec.optim.groups[0].ns_steps


def test_mtp_head_group_for_sgd_has_neither_betas_nor_ns_steps():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=sgd())
    out = MTPRewriter(k=2)(spec)
    assert out.optim.groups[0].betas is None
    assert out.optim.groups[0].ns_steps is None


# ---------------------------------------------------------------------------
# Head-detection
# ---------------------------------------------------------------------------


def test_head_detection_explicit_marker_wins_over_kind():
    """When two candidates exist, an explicit is_head=True annotation
    takes priority over the heuristic kind check."""
    g = BrickGraph(
        nodes=(
            BrickNode(kind="mlp", name="penultimate"),
            BrickNode(kind="mlp", name="actual_head",
                      params={"is_head": True}),
        ),
        edges=(("penultimate", "actual_head"),),
    )
    spec = ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(head_output_name="actual_head"),
        optim=adamw(),
    )
    out = MTPRewriter(k=2)(spec)
    assert "actual_head_0" in {n.name for n in out.graph.nodes}
    assert "penultimate" in {n.name for n in out.graph.nodes}


def test_head_detection_raises_on_no_candidate():
    """A graph with no mlp/attention brick AND no is_head marker can't
    be MTP-rewritten — must raise."""
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="only"),))
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    with pytest.raises(HeadDetectionError, match="no head brick found"):
        MTPRewriter(k=2)(spec)


# ---------------------------------------------------------------------------
# Composition with ModelBuildSpec.apply_rewrites
# ---------------------------------------------------------------------------


def test_mtp_via_apply_rewrites_adds_state_token():
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(MTPRewriter(k=2),),
    )
    out = spec.apply_rewrites()
    assert "mtp_k_heads" in out.state_tokens
    assert out.rewrites == ()


def test_mtp_via_apply_rewrites_chains_with_verify_clean():
    """End-to-end happy path: spec with MTPRewriter → apply → verify clean."""
    g = _qwen_graph_with_head()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(MTPRewriter(k=2),),
        dim_env=_QWEN3_ENV,
    )
    applied = spec.apply_rewrites()
    diag = verify_build_spec(applied, check_shapes=False)
    # After apply, the graph really does have logits_0 / logits_1 so the
    # loss head_outputs resolve cleanly.
    loss_errors = [d for d in diag.errors if d.component == "loss"]
    assert loss_errors == []


# ---------------------------------------------------------------------------
# System: applied to Qwen3-Next preset — memory / shape sanity
# ---------------------------------------------------------------------------


def test_system_mtp_k2_doubles_head_count_only():
    """Memory grows only at the head — backbone untouched."""
    g = _qwen_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=2)(spec)
    # Backbone size = original - 1 head
    backbone_original = len(g.nodes) - 1
    head_after = sum(1 for n in out.graph.nodes if n.kind == "mlp")
    assert head_after == 2
    # Total brick count grows by exactly (k-1)
    assert len(out.graph.nodes) == len(g.nodes) + 1


def test_system_mtp_k3_share_backbone_false_keeps_chain_intact():
    """share_backbone=False does NOT clone backbone bricks (Stage C scope
    only handles the head); current default behaves the same for now.
    Document this so future Stage D-rewriter knows where the split is."""
    g = _minimal_graph_with_head()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MTPRewriter(k=3, share_backbone=False)(spec)
    # Backbone brick still appears once
    backbone_count = sum(1 for n in out.graph.nodes if n.kind == "gdn")
    assert backbone_count == 1
