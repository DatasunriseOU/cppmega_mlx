"""MBSpec Stage D tests — IFIMRewriter + MHCRewriter + composition rules."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec import (
    IFIMCompositionError,
    IFIMRewriter,
    LossKind,
    MHCCompositionError,
    MHCRewriter,
    MTPRewriter,
    ModelBuildSpec,
    Rewriter,
    adamw,
    cross_entropy_loss,
    verify_build_spec,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


_QWEN3_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


def _minimal_graph() -> BrickGraph:
    return BrickGraph(
        nodes=(
            BrickNode(kind="gdn", name="backbone"),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("backbone", "logits"),),
    )


def _attn_graph() -> BrickGraph:
    return BrickGraph(
        nodes=(
            BrickNode(kind="mlp", name="pre"),
            BrickNode(kind="gated_attention", name="attn0",
                      params={"num_attention_heads": 4,
                              "num_key_value_heads": 2, "head_dim": 16}),
            BrickNode(kind="gated_attention", name="attn1",
                      params={"num_attention_heads": 4,
                              "num_key_value_heads": 2, "head_dim": 16}),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("pre", "attn0"), ("attn0", "attn1"), ("attn1", "logits")),
    )


# ---------------------------------------------------------------------------
# IFIMRewriter — construction + protocol
# ---------------------------------------------------------------------------


def test_ifim_implements_rewriter_protocol():
    r = IFIMRewriter()
    assert isinstance(r, Rewriter)
    assert "ifim_added" in r.provided_postconditions


def test_ifim_rejects_negative_lambda():
    with pytest.raises(ValueError, match="lambda_fim"):
        IFIMRewriter(lambda_fim=-0.01)


def test_ifim_rejects_blank_aux_name():
    with pytest.raises(ValueError, match="aux_node_name"):
        IFIMRewriter(aux_node_name="   ")


def test_ifim_no_preconditions_means_works_after_anything():
    r = IFIMRewriter()
    assert r.required_preconditions == frozenset()


# ---------------------------------------------------------------------------
# IFIMRewriter — application
# ---------------------------------------------------------------------------


def test_ifim_adds_aux_node_and_edge_from_head():
    g = _minimal_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = IFIMRewriter(lambda_fim=0.1)(spec)
    names = {n.name for n in out.graph.nodes}
    assert "ifim_aux" in names
    assert ("logits", "ifim_aux") in out.graph.edges


def test_ifim_synthesises_unique_aux_name_on_collision():
    g = BrickGraph(
        nodes=(
            BrickNode(kind="gdn", name="bb"),
            BrickNode(kind="mlp", name="logits"),
            BrickNode(kind="mlp", name="ifim_aux"),  # collision
        ),
        edges=(("bb", "logits"), ("logits", "ifim_aux")),
    )
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = IFIMRewriter()(spec)
    new_names = [n.name for n in out.graph.nodes if "ifim" in n.name]
    assert "ifim_aux" in new_names
    assert "ifim_aux_1" in new_names


def test_ifim_rewrites_loss_to_ifim_shaped():
    g = _minimal_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = IFIMRewriter(lambda_fim=0.05)(spec)
    assert out.loss.kind is LossKind.IFIM_SHAPED
    assert out.loss.params["lambda_fim"] == 0.05


def test_ifim_keeps_mtp_head_outputs_after_mtp_rewrite():
    """IFIM after MTP keeps the K head names — combined loss."""
    g = _minimal_graph()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(MTPRewriter(k=2), IFIMRewriter(lambda_fim=0.1)),
    )
    out = spec.apply_rewrites()
    assert out.loss.kind is LossKind.IFIM_SHAPED
    assert out.loss.head_outputs == ("logits_0", "logits_1")
    assert "mtp_k_heads" in out.state_tokens
    assert "ifim_added" in out.state_tokens


def test_ifim_double_apply_raises():
    g = _minimal_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    with pytest.raises(IFIMCompositionError):
        IFIMRewriter()(IFIMRewriter()(spec).replace(
            state_tokens=frozenset({"ifim_added"}),
        ))


def test_ifim_aux_node_carries_lambda_metadata():
    g = _minimal_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = IFIMRewriter(lambda_fim=0.07)(spec)
    aux = next(n for n in out.graph.nodes if n.name == "ifim_aux")
    assert aux.params["is_ifim_aux"] is True
    assert aux.params["lambda_fim"] == 0.07


# ---------------------------------------------------------------------------
# MHCRewriter — construction + protocol
# ---------------------------------------------------------------------------


def test_mhc_implements_rewriter_protocol():
    r = MHCRewriter(num_copies=2)
    assert isinstance(r, Rewriter)
    assert "mhc_copies_added" in r.provided_postconditions


def test_mhc_rejects_num_copies_lt_1():
    with pytest.raises(ValueError, match="num_copies"):
        MHCRewriter(num_copies=0)


def test_mhc_rejects_negative_lambda():
    with pytest.raises(ValueError, match="lambda_mhc"):
        MHCRewriter(lambda_mhc=-0.01)


def test_mhc_num_copies_1_advertises_no_postcondition():
    """num_copies=1 is a no-op; shouldn't advertise the state token."""
    r = MHCRewriter(num_copies=1)
    assert r.provided_postconditions == frozenset()


# ---------------------------------------------------------------------------
# MHCRewriter — application
# ---------------------------------------------------------------------------


def test_mhc_num_copies_1_returns_unchanged():
    g = _attn_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MHCRewriter(num_copies=1)(spec)
    assert out is spec or out.graph is spec.graph


def test_mhc_clones_every_attention_brick():
    g = _attn_graph()  # 2 attention bricks
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MHCRewriter(num_copies=2)(spec)
    attn_nodes = [
        n for n in out.graph.nodes if n.kind == "gated_attention"
    ]
    # 2 original + 2 copies (one per original) = 4
    assert len(attn_nodes) == 4
    clone_names = sorted(n.name for n in attn_nodes if "_mhc_" in n.name)
    assert clone_names == ["attn0_mhc_1", "attn1_mhc_1"]


def test_mhc_higher_num_copies_proportional():
    g = _attn_graph()  # 2 attention bricks
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MHCRewriter(num_copies=4)(spec)
    attn_nodes = [
        n for n in out.graph.nodes if n.kind == "gated_attention"
    ]
    # 2 originals + 2 * (4-1) clones = 8
    assert len(attn_nodes) == 8


def test_mhc_clones_inherit_module_for_weight_sharing():
    """Each clone references the SAME module instance as its source —
    weight sharing happens at build time without code duplication."""
    g = BrickGraph(
        nodes=(
            BrickNode(kind="mlp", name="pre"),
            BrickNode(kind="gated_attention", name="attn",
                      params={"num_attention_heads": 4,
                              "num_key_value_heads": 2, "head_dim": 16},
                      module=None),  # in tests module is None; sharing checked via params
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("pre", "attn"), ("attn", "logits")),
    )
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MHCRewriter(num_copies=2)(spec)
    clones = [
        n for n in out.graph.nodes
        if n.params.get("is_mhc_copy") is True
    ]
    assert len(clones) == 1
    assert clones[0].params["mhc_source"] == "attn"


def test_mhc_rewrites_loss_to_mhc_attn_bias():
    g = _attn_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MHCRewriter(num_copies=2, lambda_mhc=0.02)(spec)
    assert out.loss.kind is LossKind.MHC_ATTN_BIAS
    assert out.loss.params["lambda_mhc"] == 0.02


def test_mhc_noop_on_graph_with_no_attention():
    g = _minimal_graph()  # gdn + mlp — no attention
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = MHCRewriter(num_copies=2)(spec)
    assert {n.name for n in out.graph.nodes} == {n.name for n in g.nodes}


def test_mhc_double_apply_raises():
    g = _attn_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    first = MHCRewriter(num_copies=2)(spec).replace(
        state_tokens=frozenset({"mhc_copies_added"}),
    )
    with pytest.raises(MHCCompositionError):
        MHCRewriter(num_copies=2)(first)


# ---------------------------------------------------------------------------
# Composition rules — multi-rewriter chains
# ---------------------------------------------------------------------------


def test_compose_mtp_then_ifim_then_verify_clean():
    """End-to-end happy chain: MTP → IFIM → verify."""
    g = _minimal_graph()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(MTPRewriter(k=2), IFIMRewriter(lambda_fim=0.1)),
    )
    out = spec.apply_rewrites()
    diag = verify_build_spec(out, check_shapes=False)
    rewrite_errors = [d for d in diag.errors if d.component == "rewrites"]
    assert rewrite_errors == []


def test_compose_mtp_then_mhc_chains_both_state_tokens():
    g = BrickGraph(
        nodes=(
            BrickNode(kind="gated_attention", name="attn",
                      params={"num_attention_heads": 4,
                              "num_key_value_heads": 2, "head_dim": 16}),
            BrickNode(kind="mlp", name="logits"),
        ),
        edges=(("attn", "logits"),),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(MTPRewriter(k=2), MHCRewriter(num_copies=2)),
    )
    out = spec.apply_rewrites()
    assert {"mtp_k_heads", "mhc_copies_added"} <= out.state_tokens


def test_compose_double_ifim_in_chain_raises_at_apply():
    g = _minimal_graph()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(IFIMRewriter(), IFIMRewriter()),
    )
    with pytest.raises(IFIMCompositionError):
        spec.apply_rewrites()


def test_compose_qwen_preset_with_mtp_plus_mhc_via_apply_rewrites():
    """System-level: Qwen3-Next + head + MTP + MHC chain, end-to-end clean."""
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=4096, instantiate=False)
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(MTPRewriter(k=2), MHCRewriter(num_copies=2, lambda_mhc=0.03)),
        dim_env=_QWEN3_ENV,
    )
    out = spec.apply_rewrites()
    # All MTP head clones present
    assert {"logits_0", "logits_1"} <= {n.name for n in out.graph.nodes}
    # MHC clones for the attention brick present
    mhc_clones = [n for n in out.graph.nodes if "_mhc_" in n.name]
    assert len(mhc_clones) >= 1
    # State tokens reflect both rewrites
    assert {"mtp_k_heads", "mhc_copies_added"} <= out.state_tokens
    # Verify reports no rewrite-ordering errors
    diag = verify_build_spec(out, check_shapes=False)
    assert [d for d in diag.errors if d.component == "rewrites"] == []
