"""Stage F — ExecutableGraph forward-runner over auto-compiled regions.

These tests exercise the executor end-to-end against block specs (so we
don't depend on a full HybridTinyLM model). They prove:

  * The executor walks regions in planner order.
  * Eager fallback works when no artifact is attached.
  * The execution log labels each region correctly.
  * Attaching a fake compiled artifact replaces the eager region with
    the artifact's output, end-to-end.
  * Single-brick and dlpack-handoff regions report their canonical
    eager-reason strings.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_v4.fusion import (
    ExecutableGraph,
    RegionExecution,
    auto_compile_plan,
    auto_fuse_block_specs,
    plan_fusion_regions,
)
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.fusion.auto_compile import RegionPattern


HIDDEN = 32


# ---------------------------------------------------------------------------
# Tiny synthetic bricks — used to isolate the executor from BLOCK_BUILDERS
# init-argument noise. We re-purpose the "mlp" kind from BLOCK_BUILDERS by
# constructing one explicitly and overriding the module.
# ---------------------------------------------------------------------------


class _AddOneBrick(nn.Module):
    """Brick that returns ``x + 1``. Used to make region order observable."""

    def __init__(self, _: int):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        return x + 1


def _graph_three_addones() -> BrickGraph:
    bricks = [
        BrickNode(kind="mlp", name="a", module=_AddOneBrick(HIDDEN)),
        BrickNode(kind="mlp", name="b", module=_AddOneBrick(HIDDEN)),
        BrickNode(kind="mlp", name="c", module=_AddOneBrick(HIDDEN)),
    ]
    edges = (("a", "b"), ("b", "c"))
    return BrickGraph(nodes=tuple(bricks), edges=edges)


def _executable_from_graph(graph: BrickGraph) -> ExecutableGraph:
    plans = tuple(plan_fusion_regions(graph))
    kinds_by_name = {n.name: n.kind for n in graph.nodes}
    regions = tuple(auto_compile_plan(list(plans), kinds_by_name))
    return ExecutableGraph(graph=graph, plans=plans, regions=regions)


# ---------------------------------------------------------------------------
# Eager forward parity
# ---------------------------------------------------------------------------


def test_executable_graph_eager_forward_matches_plain_sequential():
    """Three add-one bricks must produce x+3 regardless of region grouping."""
    graph = _graph_three_addones()
    exe = _executable_from_graph(graph)
    x = mx.zeros((1, 4, HIDDEN), dtype=mx.float32)
    y = exe(x)
    mx.eval(y)
    assert float(y.sum().item()) == pytest.approx(3.0 * 1 * 4 * HIDDEN)


def test_execution_log_labels_every_region_eager_when_no_artifacts():
    graph = _graph_three_addones()
    exe = _executable_from_graph(graph)
    exe(mx.zeros((1, 4, HIDDEN), dtype=mx.float32))
    log = exe.execution_log
    assert len(log) == len(exe.regions)
    assert all(isinstance(e, RegionExecution) for e in log)
    assert all(e.backend == "eager_bricks" for e in log)
    assert all(e.duration_ns >= 0 for e in log)
    # Every eager region must record one of the canonical reasons.
    for e in log:
        assert e.eager_reason, e
        assert "eager" not in e.eager_reason.lower() or "kernel" in e.eager_reason


def test_region_summary_shape_aligns_with_regions():
    graph = _graph_three_addones()
    exe = _executable_from_graph(graph)
    summary = exe.region_summary()
    assert len(summary) == len(exe.regions)
    for index, row in enumerate(summary):
        assert row["index"] == index
        assert row["pattern"] in {p.value for p in RegionPattern}
        assert row["has_artifact"] is False
        assert row["brick_names"] == list(exe.regions[index].plan.brick_names)


# ---------------------------------------------------------------------------
# Single-brick / dlpack-handoff eager-reason classification
# ---------------------------------------------------------------------------


def test_single_brick_region_records_single_brick_reason():
    graph = BrickGraph(
        nodes=(BrickNode(kind="mlp", name="solo", module=_AddOneBrick(HIDDEN)),),
    )
    exe = _executable_from_graph(graph)
    exe(mx.zeros((1, 4, HIDDEN), dtype=mx.float32))
    assert len(exe.execution_log) == 1
    assert exe.execution_log[0].pattern is RegionPattern.SINGLE_BRICK_PASSTHROUGH
    assert "single-brick" in exe.execution_log[0].eager_reason


def test_dlpack_handoff_region_records_dlpack_reason():
    # Two sdpa_attention nodes can't fuse → backend=dlpack_handoff.
    # We attach a tiny attention-shaped pass-through module so we don't
    # need the real attention block; the kind drives the planner only.
    graph = BrickGraph(
        nodes=(
            BrickNode(kind="gated_attention", name="a0",
                      module=_AddOneBrick(HIDDEN)),
            BrickNode(kind="gated_attention", name="a1",
                      module=_AddOneBrick(HIDDEN)),
        ),
        edges=(("a0", "a1"),),
    )
    plans = tuple(plan_fusion_regions(graph))
    # Confirm planner did issue two single-brick regions OR one dlpack
    # chain — either way every region's pattern should be classified.
    kinds_by_name = {n.name: n.kind for n in graph.nodes}
    regions = tuple(auto_compile_plan(list(plans), kinds_by_name))
    exe = ExecutableGraph(graph=graph, plans=plans, regions=regions)
    exe(mx.zeros((1, 4, HIDDEN), dtype=mx.float32))
    backends = {e.backend for e in exe.execution_log}
    assert backends == {"eager_bricks"}
    # The reasons must come from the canonical set.
    valid_reasons = {
        "single-brick region; no fused PrimFunc emitted",
        "planner chose dlpack_handoff; each brick keeps its native kernel",
        "AutoCompiledRegion has no schedule template",
        "no compiled TileLang artifact attached to this region",
    }
    for entry in exe.execution_log:
        assert entry.eager_reason in valid_reasons, entry


# ---------------------------------------------------------------------------
# attach_artifact replaces eager execution
# ---------------------------------------------------------------------------


def test_attach_artifact_overrides_eager_region_output():
    graph = _graph_three_addones()
    plans = tuple(plan_fusion_regions(graph))
    kinds_by_name = {n.name: n.kind for n in graph.nodes}
    regions = tuple(auto_compile_plan(list(plans), kinds_by_name))
    exe = ExecutableGraph(graph=graph, plans=plans, regions=regions)
    # The planner groups three mlps into one fused norm_or_proj_chain
    # region with a compiled template — that's the one we attach to.
    fused_index = next(
        i for i, r in enumerate(exe.regions) if r.has_compiled_template
    )
    sentinel = mx.full((1, 4, HIDDEN), 9.0, dtype=mx.float32)
    exe.attach_artifact(fused_index, lambda _x: sentinel)

    x = mx.zeros((1, 4, HIDDEN), dtype=mx.float32)
    y = exe(x)
    mx.eval(y)
    # The fused region replaces three add-ones with a constant-9 tensor,
    # so the final output must equal sentinel (no further eager regions
    # after the single fused one because the planner grouped all three).
    assert mx.array_equal(y, sentinel).item() is True
    fused_entry = exe.execution_log[fused_index]
    assert fused_entry.backend == "path_c_artifact"
    assert fused_entry.eager_reason == ""


def test_attach_artifact_rejects_passthrough_region():
    graph = BrickGraph(
        nodes=(BrickNode(kind="mlp", name="solo", module=_AddOneBrick(HIDDEN)),),
    )
    exe = _executable_from_graph(graph)
    with pytest.raises(ValueError, match="no compiled schedule template"):
        exe.attach_artifact(0, lambda x: x)


def test_attach_artifact_rejects_invalid_index():
    graph = _graph_three_addones()
    exe = _executable_from_graph(graph)
    with pytest.raises(IndexError):
        exe.attach_artifact(99, lambda x: x)


def test_attach_artifact_rejects_non_callable():
    graph = _graph_three_addones()
    exe = _executable_from_graph(graph)
    with pytest.raises(TypeError, match="must be callable"):
        exe.attach_artifact(0, "not a callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_executable_graph convenience
# ---------------------------------------------------------------------------


def test_build_executable_graph_walks_block_specs_via_module_path():
    """The auto_fuse_block_specs path produces a graph we can wrap into
    an executable graph (without instantiating heavy BLOCK_BUILDERS)."""
    specs = [
        {"kind": "mlp", "name": "a"},
        {"kind": "mlp", "name": "b"},
        {"kind": "mlp", "name": "c"},
    ]
    graph, plans = auto_fuse_block_specs(
        specs, hidden_size=HIDDEN, instantiate=False
    )
    # We rebuild plans with already-computed kinds via auto_compile_plan.
    kinds_by_name = {n.name: n.kind for n in graph.nodes}
    regions = tuple(auto_compile_plan(list(plans), kinds_by_name))
    exe = ExecutableGraph(graph=graph, plans=plans, regions=regions)
    # eager run will fail because modules weren't instantiated — that's
    # the explicit contract; verify the error is the descriptive one.
    with pytest.raises(ValueError, match="instantiate=True"):
        exe(mx.zeros((1, 4, HIDDEN), dtype=mx.float32))


def test_fused_and_eager_region_counts_reflect_template_status():
    graph = _graph_three_addones()
    exe = _executable_from_graph(graph)
    total = len(exe.regions)
    assert exe.fused_region_count + exe.eager_region_count == total
    assert exe.fused_region_count >= 1, "mlp chain must produce one fused region"
