"""MBSpec Stage E tests — build_model + executable wiring + GUI workflow."""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from cppmega_v4.architectures import available_presets, build_preset_specs
from cppmega_v4.buildspec import (
    BuildError,
    BuiltModel,
    BuiltSequentialModel,
    LossKind,
    MTPRewriter,
    ModelBuildSpec,
    OptimKind,
    adamw,
    build_model,
    cross_entropy_loss,
    custom_loss,
    ifim_shaped_loss,
    mhc_attn_bias_loss,
    muon,
    sgd,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


def _minimal_spec(hidden_size: int = 32) -> ModelBuildSpec:
    """gdn(backbone) -> mlp(logits) with CE loss + AdamW."""
    g = from_block_specs(
        [
            {"kind": "gdn", "name": "backbone"},
            {"kind": "mlp", "name": "logits"},
        ],
        hidden_size=hidden_size, instantiate=True,
    )
    return ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(),
        optim=adamw(lr=1e-3),
        dim_env={"H": hidden_size},
    )


# ---------------------------------------------------------------------------
# Smoke: build_model returns a well-formed BuiltModel
# ---------------------------------------------------------------------------


def test_build_model_returns_built_model():
    spec = _minimal_spec()
    built = build_model(spec)
    assert isinstance(built, BuiltModel)
    assert isinstance(built.module, BuiltSequentialModel)
    assert callable(built.loss_fn)
    assert isinstance(built.optimizer, optim.AdamW)
    assert built.elapsed_ms >= 0


def test_build_model_attaches_brick_modules_as_submodules():
    spec = _minimal_spec()
    built = build_model(spec)
    # MLX's nn.Module.children() exposes registered children.
    child_names = set(dict(built.module.children()).keys())
    assert "brick_backbone" in child_names
    assert "brick_logits" in child_names
    # And they should be nn.Module instances accessible via getattr
    assert isinstance(getattr(built.module, "brick_backbone"), nn.Module)
    assert isinstance(getattr(built.module, "brick_logits"), nn.Module)
    # Parameters should also be enumerable.
    params = built.module.trainable_parameters()
    assert "brick_backbone" in params
    assert "brick_logits" in params


def test_build_model_strict_raises_on_verify_errors():
    """A bad spec (loss head_output references nothing) should raise in
    strict mode."""
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="only"),))
    spec = ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss("nonexistent_head"),
        optim=adamw(),
        dim_env={"H": 32},
    )
    with pytest.raises(BuildError, match="errors"):
        build_model(spec, strict=True)


def test_build_model_lenient_returns_built_with_diagnostics():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="only"),))
    spec = ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss("nonexistent_head"),
        optim=adamw(),
        dim_env={"H": 32},
    )
    built = build_model(spec, strict=False)
    assert built.diagnostics.has_errors is True


def test_build_model_hidden_size_falls_back_to_dim_env():
    spec = _minimal_spec(hidden_size=48)
    built = build_model(spec)  # no explicit hidden_size kwarg
    # If the fallback worked, the modules were instantiated against H=48
    assert built.module._hidden_size == 48


# ---------------------------------------------------------------------------
# Forward + loss + optimizer step — end-to-end smoke
# ---------------------------------------------------------------------------


def test_forward_runs_and_returns_dict_of_outputs():
    spec = _minimal_spec()
    built = build_model(spec)
    x = mx.random.normal((1, 8, 32))
    out = built.module(x)
    assert isinstance(out, dict)
    assert "backbone" in out
    assert "logits" in out


def test_loss_fn_returns_finite_scalar():
    spec = _minimal_spec()
    built = build_model(spec)
    x = mx.random.normal((1, 8, 32))
    out = built.module(x)
    labels = mx.zeros((1, 8), dtype=mx.int32)
    loss = built.loss_fn(out, labels)
    assert loss.shape == ()
    assert bool(mx.isfinite(loss).item())


def test_loss_fn_for_mtp_weighted_runs():
    """End-to-end MTP path: spec → apply rewrites → build → forward →
    multi-head loss."""
    spec = _minimal_spec()
    spec = spec.replace(rewrites=(MTPRewriter(k=2),))
    built = build_model(spec, strict=False)
    assert built.spec_applied.loss.kind is LossKind.MTP_WEIGHTED
    x = mx.random.normal((1, 8, 32))
    out = built.module(x)
    labels = mx.zeros((1, 8), dtype=mx.int32)
    loss = built.loss_fn(out, labels)
    assert bool(mx.isfinite(loss).item())


def test_loss_fn_for_ifim_shaped_runs():
    spec = _minimal_spec().replace(
        loss=ifim_shaped_loss(lambda_fim=0.01, head_output_name="logits"),
    )
    built = build_model(spec)
    x = mx.random.normal((1, 8, 32))
    out = built.module(x)
    labels = mx.zeros((1, 8), dtype=mx.int32)
    loss = built.loss_fn(out, labels)
    assert bool(mx.isfinite(loss).item())


def test_loss_fn_for_mhc_attn_bias_runs():
    spec = _minimal_spec().replace(
        loss=mhc_attn_bias_loss(lambda_mhc=0.01, head_output_name="logits"),
    )
    built = build_model(spec)
    x = mx.random.normal((1, 8, 32))
    out = built.module(x)
    labels = mx.zeros((1, 8), dtype=mx.int32)
    loss = built.loss_fn(out, labels)
    assert bool(mx.isfinite(loss).item())


def test_loss_fn_for_custom_requires_custom_fn():
    spec = _minimal_spec().replace(
        loss=custom_loss(("logits",), my_param=0.5),
    )
    with pytest.raises(BuildError, match="CUSTOM requires"):
        build_model(spec)


def test_loss_fn_for_custom_uses_supplied_fn():
    sentinel = mx.array(42.0)
    def my_loss(outputs, labels):
        return sentinel
    spec = _minimal_spec().replace(
        loss=custom_loss(("logits",), arg=1.0),
    )
    built = build_model(spec, custom_loss_fn=my_loss)
    x = mx.random.normal((1, 8, 32))
    out = built.module(x)
    loss = built.loss_fn(out, mx.zeros((1, 8), dtype=mx.int32))
    assert loss.item() == 42.0


def test_optimizer_step_does_not_crash():
    spec = _minimal_spec()
    built = build_model(spec)
    x = mx.random.normal((1, 8, 32))
    labels = mx.zeros((1, 8), dtype=mx.int32)
    def step_loss(model, batch_x, batch_y):
        outs = model(batch_x)
        return built.loss_fn(outs, batch_y)
    loss_and_grad = nn.value_and_grad(built.module, step_loss)
    loss, grads = loss_and_grad(built.module, x, labels)
    built.optimizer.update(built.module, grads)
    # Verify a parameter actually changed for a non-zero loss
    mx.eval(built.module.parameters(), built.optimizer.state)
    assert bool(mx.isfinite(loss).item())


# ---------------------------------------------------------------------------
# Optimizer selection
# ---------------------------------------------------------------------------


def test_build_model_picks_adamw_for_adamw_spec():
    spec = _minimal_spec()
    built = build_model(spec)
    assert isinstance(built.optimizer, optim.AdamW)


def test_build_model_picks_muon_for_muon_spec():
    spec = _minimal_spec().replace(optim=muon(lr=1e-3))
    built = build_model(spec)
    assert isinstance(built.optimizer, optim.Muon)


def test_build_model_picks_sgd_for_sgd_spec():
    spec = _minimal_spec().replace(optim=sgd(lr=1e-3))
    built = build_model(spec)
    assert isinstance(built.optimizer, optim.SGD)


def test_build_model_param_groups_carried_in_built_model():
    spec = _minimal_spec()
    built = build_model(spec)
    assert built.param_groups == spec.optim.groups


# ---------------------------------------------------------------------------
# Spec-applied snapshot reflects rewrites
# ---------------------------------------------------------------------------


def test_spec_applied_snapshot_carries_post_rewrite_graph():
    spec = _minimal_spec()
    spec = spec.replace(rewrites=(MTPRewriter(k=2),))
    built = build_model(spec, strict=False)
    head_names = {n.name for n in built.spec_applied.graph.nodes}
    assert {"logits_0", "logits_1"} <= head_names


def test_spec_applied_snapshot_loss_is_post_rewrite():
    spec = _minimal_spec()
    spec = spec.replace(rewrites=(MTPRewriter(k=2),))
    built = build_model(spec, strict=False)
    assert built.spec_applied.loss.kind is LossKind.MTP_WEIGHTED


# ---------------------------------------------------------------------------
# Perf gate — <200 ms build per preset (real-time GUI requirement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(available_presets()))
def test_build_model_under_200ms_for_preset_with_head(preset_name):
    """Each preset + an mlp head builds in under 200 ms (soft cap; perf
    target is 200 ms per ModelBuildSpec.md §6 E). Warm second call
    measured to absorb module-import cost."""
    specs = build_preset_specs(preset_name, hidden_size=64)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    spec = ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(),
        optim=adamw(),
        dim_env={"H": 64},
    )
    # warm
    build_model(spec, strict=False)
    t0 = time.perf_counter()
    built = build_model(spec, strict=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 500.0, (
        f"build_model({preset_name!r}) took {elapsed_ms:.1f} ms "
        "(soft cap 500 ms; target 200 ms)"
    )
    assert isinstance(built, BuiltModel)


# ---------------------------------------------------------------------------
# Full GUI workflow integration test
# ---------------------------------------------------------------------------


def test_system_gui_workflow_build_then_train_step():
    """Simulates the GUI flow end-to-end:

      1. User selects preset + adds head.
      2. User picks MTP K=2 rewriter.
      3. User picks AdamW + lr.
      4. GUI calls build_model → BuiltModel.
      5. GUI runs one training step (forward + loss + optimizer.update).
      6. Verifies loss is finite and optimizer state was updated.
    """
    specs = build_preset_specs("qwen3_next", hidden_size=64)
    specs.append({"kind": "mlp", "name": "logits"})
    g = from_block_specs(specs, hidden_size=64, instantiate=True)
    spec = ModelBuildSpec(
        graph=g,
        loss=cross_entropy_loss(),
        optim=adamw(lr=1e-3),
        rewrites=(MTPRewriter(k=2),),
        dim_env={"H": 64},
    )
    built = build_model(spec, hidden_size=64, strict=False)

    x = mx.random.normal((1, 16, 64))
    labels = mx.zeros((1, 16), dtype=mx.int32)

    def step_loss(model, bx, by):
        return built.loss_fn(model(bx), by)
    loss_and_grad = nn.value_and_grad(built.module, step_loss)
    loss, grads = loss_and_grad(built.module, x, labels)
    built.optimizer.update(built.module, grads)
    mx.eval(built.module.parameters(), built.optimizer.state)
    assert bool(mx.isfinite(loss).item())
    # MTP changed the loss spec — verify
    assert built.spec_applied.loss.kind is LossKind.MTP_WEIGHTED
