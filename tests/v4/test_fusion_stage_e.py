"""Stage E tests — auto_compile pattern matcher over FusionRegionPlans."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.fusion import (
    AutoCompiledRegion,
    FusionRegionPlan,
    RegionPattern,
    auto_compile_plan,
    auto_compile_region,
    detect_region_pattern,
    from_block_specs,
    plan_fusion_regions,
)
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


def _plan_from_kinds(*kinds_and_names: tuple[str, str]) -> BrickGraph:
    nodes = tuple(BrickNode(kind=k, name=n) for k, n in kinds_and_names)
    edges = tuple((nodes[i].name, nodes[i + 1].name) for i in range(len(nodes) - 1))
    return BrickGraph(nodes=nodes, edges=edges)


def _plan(kinds: list[str]) -> tuple[FusionRegionPlan, list[str]]:
    """Build one BrickGraph + one fused FusionRegionPlan covering all kinds.

    Returns (plan, kinds-in-order). Used by pattern-detection unit tests
    where the planner's grouping decision is not what we're testing.
    """
    g = BrickGraph(
        nodes=tuple(BrickNode(kind=k, name=f"{k}_{i}") for i, k in enumerate(kinds)),
    )
    plans = plan_fusion_regions(g)
    # If the planner split, we still synthesize a single-plan view for
    # pure pattern-detection testing.
    if len(plans) == 1:
        return plans[0], kinds
    # Force-fuse for the test: build a synthetic plan that pretends all
    # bricks landed in one path_c region.
    from cppmega_v4.fusion.auto_planner import _shared_mem_estimate  # noqa
    from cppmega_v4.fusion.compatibility import _CATEGORY_BY_KIND
    cats = tuple(_CATEGORY_BY_KIND[k] for k in kinds)
    return (
        FusionRegionPlan(
            brick_names=tuple(n.name for n in g.nodes),
            categories=cats,
            backend="path_c",
            estimated_savings_us=0.0,
            reason="synthetic for pattern-detection test",
        ),
        kinds,
    )


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_single_brick_passthrough():
    p, _ = _plan(["gated_attention"])
    assert detect_region_pattern(p) is RegionPattern.SINGLE_BRICK_PASSTHROUGH


def test_pattern_dlpack_handoff_chain():
    # 2 sdpa attentions: planner can't fuse them; backend stays handoff.
    p = FusionRegionPlan(
        brick_names=("a", "b"),
        categories=("sdpa_attention", "sdpa_attention"),
        backend="dlpack_handoff",
        estimated_savings_us=0.0,
        reason="cant fuse two sdpas",
    )
    assert detect_region_pattern(p) is RegionPattern.DLPACK_HANDOFF_CHAIN


def test_pattern_linear_attn_scan_with_norm():
    p, _ = _plan(["gdn", "gdn", "mlp"])
    assert detect_region_pattern(p) is RegionPattern.LINEAR_ATTN_SCAN_WITH_NORM


def test_pattern_pure_linear_attn_scan():
    p, _ = _plan(["gdn", "gdn", "gdn"])
    assert detect_region_pattern(p) is RegionPattern.LINEAR_ATTN_SCAN


def test_pattern_ssm_chunkwise_scan():
    p, _ = _plan(["mamba3", "mamba3"])
    assert detect_region_pattern(p) is RegionPattern.SSM_CHUNKWISE_SCAN


def test_pattern_sdpa_with_output_proj():
    p, _ = _plan(["gated_attention", "mlp"])
    assert detect_region_pattern(p) is RegionPattern.SDPA_WITH_OUTPUT_PROJ


def test_pattern_moe_route_combine():
    p, _ = _plan(["moe", "mlp"])
    assert detect_region_pattern(p) is RegionPattern.MOE_ROUTE_COMBINE


def test_pattern_norm_or_proj_chain():
    p, _ = _plan(["mlp", "mlp"])
    assert detect_region_pattern(p) is RegionPattern.NORM_OR_PROJ_CHAIN


def test_pattern_generic_descriptor_template():
    # ssm + linear_attn fuses per compatibility, but the cat set is neither
    # all-ssm nor all-linear-attn nor includes norm_or_proj.
    p, _ = _plan(["mamba3", "gdn"])
    assert detect_region_pattern(p) is RegionPattern.GENERIC_DESCRIPTOR_TEMPLATE


# ---------------------------------------------------------------------------
# auto_compile_region
# ---------------------------------------------------------------------------


def test_auto_compile_size_one_returns_no_template():
    p, kinds = _plan(["gdn"])
    region = auto_compile_region(p, kinds)
    assert isinstance(region, AutoCompiledRegion)
    assert region.pattern is RegionPattern.SINGLE_BRICK_PASSTHROUGH
    assert region.schedule_template is None
    assert region.has_compiled_template is False
    assert len(region.descriptors) == 1
    assert region.descriptors[0].op_name == "gdn"


def test_auto_compile_kinds_length_mismatch_raises():
    p, _ = _plan(["gdn", "gdn"])
    with pytest.raises(ValueError, match="kinds length"):
        auto_compile_region(p, kinds=["gdn"])


def test_auto_compile_unknown_kind_raises():
    p = FusionRegionPlan(
        brick_names=("x", "y"),
        categories=("linear_attn", "linear_attn"),
        backend="path_c",
        estimated_savings_us=10.0,
        reason="fused",
    )
    with pytest.raises(KeyError, match="no descriptor"):
        auto_compile_region(p, kinds=["totally_made_up_kind_xyz", "gdn"])


def test_auto_compile_linear_attn_chain_builds_template():
    p, kinds = _plan(["gdn", "gdn", "gdn"])
    region = auto_compile_region(p, kinds)
    assert region.pattern is RegionPattern.LINEAR_ATTN_SCAN
    assert region.has_compiled_template is True
    # The template carries cppmega_mlx's metadata markers.
    tmpl = region.schedule_template
    assert hasattr(tmpl, "_cppmega_path_c_brick_ops")
    assert tmpl._cppmega_path_c_brick_ops == ("gdn", "gdn", "gdn")


def test_auto_compile_sdpa_with_norm_builds_template():
    p, kinds = _plan(["gated_attention", "mlp"])
    region = auto_compile_region(p, kinds)
    assert region.pattern is RegionPattern.SDPA_WITH_OUTPUT_PROJ
    assert region.has_compiled_template is True
    assert region.schedule_template._cppmega_path_c_brick_ops == (
        "gated_attention", "mlp",
    )


def test_auto_compile_dlpack_handoff_returns_no_template():
    p = FusionRegionPlan(
        brick_names=("a", "b"),
        categories=("sdpa_attention", "sdpa_attention"),
        backend="dlpack_handoff",
        estimated_savings_us=0.0,
        reason="cant fuse",
    )
    region = auto_compile_region(
        p, kinds=["gated_attention", "gated_attention"],
    )
    assert region.pattern is RegionPattern.DLPACK_HANDOFF_CHAIN
    assert region.schedule_template is None
    # Descriptors still populated (callers may want to inspect).
    assert len(region.descriptors) == 2


# ---------------------------------------------------------------------------
# auto_compile_plan (full sequence)
# ---------------------------------------------------------------------------


def test_auto_compile_plan_threads_kinds_per_region():
    g = _plan_from_kinds(
        ("gdn", "g0"), ("gdn", "g1"), ("gdn", "g2"),
        ("gated_attention", "attn"),
        ("moe", "moe"),
    )
    plans = plan_fusion_regions(g)
    kinds_by_name = {n.name: n.kind for n in g.nodes}
    regions = auto_compile_plan(plans, kinds_by_name)
    assert len(regions) == len(plans)
    # First region: pure linear_attn scan → has a template
    assert regions[0].pattern is RegionPattern.LINEAR_ATTN_SCAN
    assert regions[0].has_compiled_template is True
    # Solo attn / solo moe: passthrough, no template
    assert regions[1].pattern is RegionPattern.SINGLE_BRICK_PASSTHROUGH
    assert regions[2].pattern is RegionPattern.SINGLE_BRICK_PASSTHROUGH
    for r in regions[1:]:
        assert r.has_compiled_template is False


# ---------------------------------------------------------------------------
# System: drive auto_compile across every preset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", [
    "qwen3_next", "kimi_linear", "kimi_k2", "deepseek_v3",
    "deepseek_v4_flash", "gemma4", "mistral4", "ling26", "longcat",
    "nemotron3", "zaya1", "arcee_trinity",
])
def test_system_every_preset_auto_compiles_without_error(preset_name):
    specs = build_preset_specs(preset_name, hidden_size=64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    kinds_by_name = {n.name: n.kind for n in g.nodes}
    regions = auto_compile_plan(plans, kinds_by_name)
    assert len(regions) == len(plans)
    for r in regions:
        # Every region carries a valid pattern label and at least one
        # descriptor per contained brick.
        assert isinstance(r.pattern, RegionPattern)
        assert len(r.descriptors) == r.plan.size
        # Patterns that should compile a template actually did.
        if r.pattern not in {
            RegionPattern.SINGLE_BRICK_PASSTHROUGH,
            RegionPattern.DLPACK_HANDOFF_CHAIN,
        }:
            assert r.has_compiled_template is True


def test_system_qwen3_next_full_auto_compile_classifies_regions_correctly():
    """End-to-end: 3 GDN fuse to a LINEAR_ATTN_SCAN region; gated_attention
    and moe land as singletons (passthrough)."""
    specs = build_preset_specs("qwen3_next", hidden_size=64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    kinds_by_name = {n.name: n.kind for n in g.nodes}
    regions = auto_compile_plan(plans, kinds_by_name)
    assert [r.pattern for r in regions] == [
        RegionPattern.LINEAR_ATTN_SCAN,
        RegionPattern.SINGLE_BRICK_PASSTHROUGH,
        RegionPattern.SINGLE_BRICK_PASSTHROUGH,
    ]


def test_system_ling26_pure_linear_region_picks_scan_pattern():
    specs = build_preset_specs("ling26", hidden_size=64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    kinds_by_name = {n.name: n.kind for n in g.nodes}
    regions = auto_compile_plan(plans, kinds_by_name)
    assert regions[0].pattern is RegionPattern.LINEAR_ATTN_SCAN
    assert regions[0].has_compiled_template is True
    assert regions[0].plan.size == 7
