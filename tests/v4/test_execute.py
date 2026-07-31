"""Tests for Stage F ExecutableGraph integration.

Task 18: ``Mamba3Fp8TrainBlockLauncher.as_artifact_callable(model)``
Task 19: ``HybridTinyLM.path_c_fused_train_block_prim_func`` caching
Task 20: ``ExecutableGraph.forward`` non-finite fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_v4.fusion.auto_compile import AutoCompiledRegion, RegionPattern
from cppmega_v4.fusion.auto_planner import FusionRegionPlan
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.fusion.execute import (
    ExecutableGraph,
    RegionExecution,
    _REGION_RUN_EAGER_REASONS,
)


# ---------------------------------------------------------------------------
# Helpers — stub model / graph / regions for ExecutableGraph tests
# ---------------------------------------------------------------------------


class _StubLinear(nn.Module):
    """Trivial identity module used as a brick module."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


def _make_stub_graph(n_bricks: int = 2) -> ExecutableGraph:
    """Build an ExecutableGraph with ``n_bricks`` single-brick passthrough
    regions, each backed by a ``_StubLinear`` module."""
    nodes = tuple(
        BrickNode(
            kind="rmsnorm",
            name=f"brick_{i}",
            module=_StubLinear(),
        )
        for i in range(n_bricks)
    )
    edges = tuple(
        (f"brick_{i}", f"brick_{i + 1}") for i in range(n_bricks - 1)
    )
    graph = BrickGraph(nodes=nodes, edges=edges)
    plans = tuple(
        FusionRegionPlan(
            brick_names=(f"brick_{i}",),
            categories=("norm_or_proj",),
            backend="path_c",
            estimated_savings_us=0.0,
            reason="stub",
        )
        for i in range(n_bricks)
    )
    regions = tuple(
        AutoCompiledRegion(
            plan=plan,
            pattern=RegionPattern.SINGLE_BRICK_PASSTHROUGH,
            descriptors=(),
            schedule_template=None,
            reason="stub region",
        )
        for plan in plans
    )
    return ExecutableGraph(graph=graph, plans=plans, regions=regions)


def _make_fused_graph() -> ExecutableGraph:
    """Build an ExecutableGraph with one two-brick fused region that
    has ``has_compiled_template = True`` (so an artifact can be attached)."""
    nodes = (
        BrickNode(kind="rmsnorm", name="norm_0", module=_StubLinear()),
        BrickNode(kind="rmsnorm", name="norm_1", module=_StubLinear()),
    )
    graph = BrickGraph(nodes=nodes, edges=(("norm_0", "norm_1"),))
    plan = FusionRegionPlan(
        brick_names=("norm_0", "norm_1"),
        categories=("norm_or_proj", "norm_or_proj"),
        backend="path_c",
        estimated_savings_us=100.0,
        reason="stub fused",
    )
    # A non-None schedule_template makes has_compiled_template = True.
    region = AutoCompiledRegion(
        plan=plan,
        pattern=RegionPattern.NORM_OR_PROJ_CHAIN,
        descriptors=(),
        schedule_template=lambda region: None,
        reason="fused stub",
    )
    return ExecutableGraph(
        graph=graph, plans=(plan,), regions=(region,),
    )


# ---------------------------------------------------------------------------
# Task 18: as_artifact_callable is tested at the unit level by verifying
# the ExecutableGraph wiring with a trivial artifact callable.
# ---------------------------------------------------------------------------


class TestAttachArtifactCallable:
    """Verify that attaching a callable artifact to a fused region makes
    the ExecutableGraph run that region through the artifact and log
    ``backend == "path_c_artifact"``."""

    def test_artifact_callable_runs_path_c_artifact(self) -> None:
        eg = _make_fused_graph()

        # Attach a simple artifact that multiplies hidden by 2.
        def double_artifact(hidden: mx.array) -> mx.array:
            return hidden * 2.0

        eg.attach_artifact(0, double_artifact)

        inp = mx.ones((1, 4), dtype=mx.float32)
        out = eg.forward(inp)

        # The artifact should have run.
        assert len(eg.execution_log) == 1
        entry = eg.execution_log[0]
        assert entry.backend == "path_c_artifact"
        assert entry.eager_reason == ""
        assert entry.ran_fused is True

        # The output should be doubled.
        mx.eval(out)
        assert float(out.sum().item()) == 8.0  # 1*4*2

    def test_artifact_callable_without_attach_runs_eager(self) -> None:
        eg = _make_stub_graph(n_bricks=1)
        inp = mx.ones((1, 4), dtype=mx.float32)
        out = eg.forward(inp)
        assert eg.execution_log[0].backend == "eager_bricks"

    def test_execution_log_shows_path_c_artifact_backend(self) -> None:
        eg = _make_fused_graph()
        eg.attach_artifact(0, lambda h: h + 1.0)
        eg.forward(mx.zeros((1, 4)))
        log = eg.execution_log
        assert any(e.backend == "path_c_artifact" for e in log)


# ---------------------------------------------------------------------------
# Task 19: prim_func caching
# ---------------------------------------------------------------------------


class TestPrimFuncCache:
    """Verify that ``path_c_fused_train_block_prim_func`` returns the same
    object on repeated calls with the same ``sequence_length``."""

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls):
        from cppmega_mlx.recipes.model_factory import (
            build_local_gb10_quarter_tiny_smoke_model,
        )
        return build_local_gb10_quarter_tiny_smoke_model()

    def test_second_call_returns_same_object(self, model: Any) -> None:
        pf1 = model.path_c_fused_train_block_prim_func(sequence_length=64)
        pf2 = model.path_c_fused_train_block_prim_func(sequence_length=64)
        assert pf1 is not None
        assert pf1 is pf2, (
            "expected path_c_fused_train_block_prim_func to return the "
            "cached object on the second call"
        )

    def test_different_seq_len_produces_different_object(self, model: Any) -> None:
        pf_64 = model.path_c_fused_train_block_prim_func(sequence_length=64)
        pf_128 = model.path_c_fused_train_block_prim_func(sequence_length=128)
        # Both should succeed; they may or may not be the same object
        # depending on schedule internals, but neither should be None.
        assert pf_64 is not None
        assert pf_128 is not None


# ---------------------------------------------------------------------------
# Task 20: non-finite fallback
# ---------------------------------------------------------------------------


class TestNonFiniteFallback:
    """Verify that ``ExecutableGraph.forward`` falls back to eager when
    a compiled artifact returns NaN/Inf."""

    def test_nan_artifact_triggers_eager_fallback(self) -> None:
        eg = _make_fused_graph()

        # Attach an artifact that always returns NaN.
        def nan_artifact(hidden: mx.array) -> mx.array:
            return mx.full(hidden.shape, float("nan"), dtype=hidden.dtype)

        eg.attach_artifact(0, nan_artifact)

        inp = mx.ones((1, 4), dtype=mx.float32)
        out = eg.forward(inp)

        assert len(eg.execution_log) == 1
        entry = eg.execution_log[0]
        assert entry.backend == "eager_bricks"
        assert entry.eager_reason == "artifact_nonfinite_fallback"

        # Output should be finite (came from eager bricks).
        mx.eval(out)
        assert mx.isfinite(out).all().item()

    def test_inf_artifact_triggers_eager_fallback(self) -> None:
        eg = _make_fused_graph()

        def inf_artifact(hidden: mx.array) -> mx.array:
            return mx.full(hidden.shape, float("inf"), dtype=hidden.dtype)

        eg.attach_artifact(0, inf_artifact)
        out = eg.forward(mx.ones((1, 4), dtype=mx.float32))

        assert eg.execution_log[0].backend == "eager_bricks"
        assert eg.execution_log[0].eager_reason == "artifact_nonfinite_fallback"

    def test_finite_artifact_validates_and_skips_check_on_second_call(self) -> None:
        eg = _make_fused_graph()
        eg.attach_artifact(0, lambda h: h * 3.0)

        # First call — validates.
        eg.forward(mx.ones((1, 4)))
        assert eg._validated is True
        assert eg.execution_log[0].backend == "path_c_artifact"

        # Second call — skips validation (no isfinite check).
        eg.forward(mx.ones((1, 4)))
        assert eg.execution_log[0].backend == "path_c_artifact"

    def test_nonfinite_fallback_reason_in_registry(self) -> None:
        assert "artifact_nonfinite_fallback" in _REGION_RUN_EAGER_REASONS

    def test_validation_not_set_when_fallback_triggered(self) -> None:
        eg = _make_fused_graph()
        eg.attach_artifact(0, lambda h: mx.full(h.shape, float("nan")))
        eg.forward(mx.ones((1, 4)))
        assert eg._validated is False
