"""MBSpec Stage B tests — ModelBuildSpec + verify_build_spec."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.buildspec import (
    BuildDiagnosticSeverity,
    BuildDiagnostics,
    LossSpec,
    LossKind,
    ModelBuildSpec,
    OptimSpec,
    ParamGroup,
    RewriteOrderError,
    Rewriter,
    adamw,
    cross_entropy_loss,
    mtp_weighted_loss,
    muon_adamw_hybrid,
    verify_build_spec,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


_QWEN3_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


def _qwen_graph() -> BrickGraph:
    specs = build_preset_specs("qwen3_next", hidden_size=4096)
    return from_block_specs(specs, hidden_size=4096, instantiate=False)


# ---------------------------------------------------------------------------
# Fake rewriters used by the tests (no actual code mutation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRewriter:
    """Identity rewriter that just advertises pre/post conditions."""

    name: str
    required_preconditions: frozenset[str] = field(default_factory=frozenset)
    provided_postconditions: frozenset[str] = field(default_factory=frozenset)

    def __call__(self, spec: ModelBuildSpec) -> ModelBuildSpec:
        return spec


# ---------------------------------------------------------------------------
# Unit: ModelBuildSpec construction + validation
# ---------------------------------------------------------------------------


def test_model_build_spec_rejects_non_brickgraph():
    with pytest.raises(TypeError, match="BrickGraph"):
        ModelBuildSpec(
            graph="not a graph",  # type: ignore[arg-type]
            loss=cross_entropy_loss(),
            optim=adamw(),
        )


def test_model_build_spec_rejects_non_lossspec():
    g = _qwen_graph()
    with pytest.raises(TypeError, match="LossSpec"):
        ModelBuildSpec(graph=g, loss="ce", optim=adamw())  # type: ignore[arg-type]


def test_model_build_spec_rejects_non_optimspec():
    g = _qwen_graph()
    with pytest.raises(TypeError, match="OptimSpec"):
        ModelBuildSpec(
            graph=g, loss=cross_entropy_loss(), optim="adamw",  # type: ignore[arg-type]
        )


def test_model_build_spec_rejects_rewriter_missing_attributes():
    g = _qwen_graph()
    bad = "not a rewriter"
    with pytest.raises(TypeError, match="missing attribute"):
        ModelBuildSpec(
            graph=g, loss=cross_entropy_loss(), optim=adamw(),
            rewrites=(bad,),  # type: ignore[arg-type]
        )


def test_model_build_spec_default_state_token_is_single_head():
    g = _qwen_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    assert spec.state_tokens == frozenset({"single_head"})


def test_model_build_spec_replace_returns_new_spec_and_does_not_mutate():
    g = _qwen_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    new = spec.replace(optim=muon_adamw_hybrid())
    assert spec is not new
    assert spec.optim != new.optim
    assert new.graph is spec.graph


# ---------------------------------------------------------------------------
# apply_rewrites — ordering semantics
# ---------------------------------------------------------------------------


def test_apply_rewrites_empty_chain_is_identity():
    g = _qwen_graph()
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    out = spec.apply_rewrites()
    assert out.rewrites == ()
    assert out.state_tokens == spec.state_tokens


def test_apply_rewrites_adds_postcondition_token():
    g = _qwen_graph()
    r = _FakeRewriter(
        name="add_token_r",
        required_preconditions=frozenset({"single_head"}),
        provided_postconditions=frozenset({"mtp_k_heads"}),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(r,),
    )
    out = spec.apply_rewrites()
    assert "mtp_k_heads" in out.state_tokens
    assert "single_head" in out.state_tokens
    assert out.rewrites == ()


def test_apply_rewrites_raises_on_missing_precondition():
    g = _qwen_graph()
    # This rewriter needs "ifim_added" but the initial spec only has
    # "single_head" — must raise.
    r = _FakeRewriter(
        name="needs_ifim",
        required_preconditions=frozenset({"ifim_added"}),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(r,),
    )
    with pytest.raises(RewriteOrderError, match="needs_ifim"):
        spec.apply_rewrites()


def test_apply_rewrites_chains_postcondition_into_precondition():
    """A rewriter providing X can satisfy a later rewriter requiring X."""
    g = _qwen_graph()
    r_a = _FakeRewriter(
        name="adds_x",
        required_preconditions=frozenset({"single_head"}),
        provided_postconditions=frozenset({"x"}),
    )
    r_b = _FakeRewriter(
        name="needs_x",
        required_preconditions=frozenset({"x"}),
        provided_postconditions=frozenset({"y"}),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(r_a, r_b),
    )
    out = spec.apply_rewrites()
    assert {"x", "y"} <= out.state_tokens


# ---------------------------------------------------------------------------
# verify_build_spec — happy paths
# ---------------------------------------------------------------------------


def test_verify_qwen_preset_with_ce_adamw_no_rewrites_clean():
    """Without rewrites, the loss head_output 'logits' won't match any
    graph brick name → we expect ONE error (head_output mismatch). This
    is the GUI signal "you need to add a head brick OR an MTPRewriter"."""
    g = _qwen_graph()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_QWEN3_ENV,
    )
    diag = verify_build_spec(spec)
    assert isinstance(diag, BuildDiagnostics)
    # Head-output mismatch is the only expected error here.
    loss_errors = [d for d in diag.errors if d.component == "loss"]
    assert len(loss_errors) >= 1
    assert "logits" in loss_errors[0].message


def test_verify_with_matching_head_brick_has_no_errors():
    """When a brick named 'logits' exists in the graph, the loss
    head_output resolves cleanly → no errors."""
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="logits"),))
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_QWEN3_ENV,
    )
    diag = verify_build_spec(spec)
    assert diag.has_errors is False


def test_verify_summary_dict_shape():
    g = _qwen_graph()
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=_QWEN3_ENV,
    )
    diag = verify_build_spec(spec)
    summary = diag.summary()
    assert set(summary.keys()) == {"errors", "warnings", "total"}


# ---------------------------------------------------------------------------
# verify_build_spec — error / warning surfaces
# ---------------------------------------------------------------------------


def test_verify_dead_optim_matcher_warns():
    """Optim group with matcher that selects nothing → WARNING."""
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="logits"),))
    optim = OptimSpec(
        kind=adamw().kind,
        groups=(
            ParamGroup(
                matcher="regex:totally_made_up_name",
                lr=1e-4,
                betas=(0.9, 0.95),
            ),
        ),
    )
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=optim)
    diag = verify_build_spec(spec, check_shapes=False)
    optim_warnings = [
        d for d in diag.warnings if d.component == "optim"
    ]
    assert len(optim_warnings) == 1
    assert "matches no brick names" in optim_warnings[0].message


def test_verify_matcher_all_is_never_flagged():
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="x"),))
    spec = ModelBuildSpec(graph=g, loss=cross_entropy_loss(), optim=adamw())
    diag = verify_build_spec(spec, check_shapes=False)
    optim_warnings = [d for d in diag.warnings if d.component == "optim"]
    assert optim_warnings == []


def test_verify_rewrite_chain_bad_order_errors():
    g = _qwen_graph()
    needs_x = _FakeRewriter(
        name="needs_x_first",
        required_preconditions=frozenset({"x"}),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(needs_x,),
    )
    diag = verify_build_spec(spec, check_shapes=False)
    rewrite_errors = [d for d in diag.errors if d.component == "rewrites"]
    assert len(rewrite_errors) == 1
    assert "needs_x_first" in rewrite_errors[0].message


def test_verify_rewrite_chain_with_satisfied_precondition_clean():
    g = _qwen_graph()
    adds_x = _FakeRewriter(
        name="adds_x",
        required_preconditions=frozenset({"single_head"}),
        provided_postconditions=frozenset({"x"}),
    )
    consumes_x = _FakeRewriter(
        name="consumes_x",
        required_preconditions=frozenset({"x"}),
    )
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(adds_x, consumes_x),
    )
    diag = verify_build_spec(spec, check_shapes=False)
    rewrite_errors = [d for d in diag.errors if d.component == "rewrites"]
    assert rewrite_errors == []


def test_verify_rewrite_chain_duplicate_name_warns():
    g = _qwen_graph()
    dup = _FakeRewriter(name="dup_r")
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        rewrites=(dup, dup),
    )
    diag = verify_build_spec(spec, check_shapes=False)
    rewrite_warnings = [d for d in diag.warnings if d.component == "rewrites"]
    assert len(rewrite_warnings) >= 1


def test_verify_skips_shape_check_when_dim_env_empty():
    g = _qwen_graph()
    # No dim_env → shape check is a no-op (no ERRORs from missing dims).
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env={},
    )
    diag = verify_build_spec(spec)
    shape_diags = [d for d in diag.diagnostics if d.component == "shape"]
    assert shape_diags == []


def test_verify_shape_check_surfaces_resolver_errors():
    """An impossible dim_env triggers resolver errors → component=shape."""
    g = _qwen_graph()
    bad_env = {"B": 1, "S": 16}  # missing H/nh/...
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=bad_env,
    )
    diag = verify_build_spec(spec)
    shape_errors = [d for d in diag.errors if d.component == "shape"]
    assert len(shape_errors) >= 1


def test_verify_check_shapes_false_skips_resolver():
    g = _qwen_graph()
    bad_env = {"B": 1, "S": 16}
    spec = ModelBuildSpec(
        graph=g, loss=cross_entropy_loss(), optim=adamw(),
        dim_env=bad_env,
    )
    diag = verify_build_spec(spec, check_shapes=False)
    shape_diags = [d for d in diag.diagnostics if d.component == "shape"]
    assert shape_diags == []


# ---------------------------------------------------------------------------
# System: MTP head naming convention recognised
# ---------------------------------------------------------------------------


def test_verify_mtp_loss_with_logits_brick_doesnt_error_due_to_rewrites_flag():
    """When the spec carries an MTP rewriter (we use a fake here that
    advertises the right postcondition) and the loss is mtp_weighted
    (which names heads logits_0 / logits_1), the verifier accepts the
    head names as 'future-valid' even though the graph doesn't yet have
    them."""
    g = _qwen_graph()
    mtp_loss = mtp_weighted_loss(k=2)
    fake_mtp = _FakeRewriter(
        name="MTPRewriter_k2",
        required_preconditions=frozenset({"single_head"}),
        provided_postconditions=frozenset({"mtp_k_heads"}),
    )
    spec = ModelBuildSpec(
        graph=g, loss=mtp_loss, optim=adamw(),
        rewrites=(fake_mtp,),
        dim_env=_QWEN3_ENV,
    )
    diag = verify_build_spec(spec)
    loss_errors = [d for d in diag.errors if d.component == "loss"]
    # We don't error on logits_0 / logits_1 because rewrites are present
    # — the verifier defers final-name checks to apply-time.
    assert loss_errors == []
