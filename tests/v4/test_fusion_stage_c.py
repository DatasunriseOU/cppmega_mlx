"""Stage C tests — auto_planner greedy region grouping + cost model."""

from __future__ import annotations

import mlx.nn as nn
import pytest

from cppmega_v4.fusion import from_block_specs
from cppmega_v4.fusion.auto_planner import (
    DEFAULT_MAX_REGION_SIZE,
    DEFAULT_MAX_SHARED_MEM_BYTES,
    FusionRegionPlan,
    auto_fuse_model,
    plan_fusion_regions,
)
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


# ---------------------------------------------------------------------------
# Unit: FusionRegionPlan dataclass invariants
# ---------------------------------------------------------------------------


def test_plan_rejects_empty_region():
    with pytest.raises(ValueError, match="≥1 brick"):
        FusionRegionPlan(
            brick_names=(),
            categories=(),
            backend="path_c",
            estimated_savings_us=0.0,
            reason="bad",
        )


def test_plan_rejects_mismatched_categories():
    with pytest.raises(ValueError, match="must align"):
        FusionRegionPlan(
            brick_names=("a", "b"),
            categories=("linear_attn",),
            backend="path_c",
            estimated_savings_us=0.0,
            reason="bad",
        )


def test_plan_size_and_is_fused_helpers():
    p1 = FusionRegionPlan(
        brick_names=("a",), categories=("sdpa_attention",),
        backend="dlpack_handoff", estimated_savings_us=0.0, reason="solo",
    )
    p2 = FusionRegionPlan(
        brick_names=("a", "b"), categories=("linear_attn", "linear_attn"),
        backend="path_c", estimated_savings_us=5.5, reason="fused",
    )
    assert p1.size == 1
    assert p1.is_fused is False
    assert p2.size == 2
    assert p2.is_fused is True


# ---------------------------------------------------------------------------
# Unit: planner edge cases
# ---------------------------------------------------------------------------


def test_planner_returns_empty_for_empty_graph():
    g = BrickGraph(nodes=(), edges=())
    assert plan_fusion_regions(g) == []


def test_planner_size_one_graph_yields_one_passthrough_region():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="g0"),))
    plans = plan_fusion_regions(g)
    assert len(plans) == 1
    p = plans[0]
    assert p.brick_names == ("g0",)
    assert p.backend == "dlpack_handoff"
    assert p.estimated_savings_us == 0.0


def test_planner_rejects_invalid_limits():
    g = BrickGraph(nodes=(BrickNode(kind="gdn", name="g0"),))
    with pytest.raises(ValueError, match="max_region_size"):
        plan_fusion_regions(g, max_region_size=0)
    with pytest.raises(ValueError, match="max_shared_mem_bytes"):
        plan_fusion_regions(g, max_shared_mem_bytes=0)


# ---------------------------------------------------------------------------
# Unit: greedy grouping behaviour
# ---------------------------------------------------------------------------


def _chain(*kinds: str) -> BrickGraph:
    """Build a simple BrickGraph from kind names (auto-numbered)."""
    nodes = tuple(BrickNode(kind=k, name=f"{k}_{i}") for i, k in enumerate(kinds))
    edges = tuple((nodes[i].name, nodes[i + 1].name) for i in range(len(nodes) - 1))
    return BrickGraph(nodes=nodes, edges=edges)


def test_planner_groups_three_gdn_into_one_region():
    g = _chain("gdn", "gdn", "gdn")
    plans = plan_fusion_regions(g)
    assert len(plans) == 1
    p = plans[0]
    assert p.size == 3
    assert p.backend == "path_c"
    assert p.is_fused is True
    assert p.estimated_savings_us > 0


def test_planner_breaks_region_on_sdpa_boundary():
    """3 GDN + 1 gated_attention: GDN fuse, attn closes its own region."""
    g = _chain("gdn", "gdn", "gdn", "gated_attention")
    plans = plan_fusion_regions(g)
    assert [p.brick_names for p in plans] == [
        ("gdn_0", "gdn_1", "gdn_2"),
        ("gated_attention_3",),
    ]
    assert plans[0].backend == "path_c"
    assert plans[1].backend == "dlpack_handoff"
    # first region's close_reason must mention WHY (eligibility text)
    assert "linear-attn" in plans[0].reason or "SDPA" in plans[0].reason or \
           "softmax" in plans[0].reason


def test_planner_breaks_at_moe_attention_boundary():
    """MoE is a hard boundary against attention/scan categories, but fuses
    with adjacent norm_or_proj. Chain: gdn x2 + moe + gdn x2 — the linear
    scan and MoE never share registers."""
    g = _chain("gdn", "gdn", "moe", "gdn", "gdn")
    plans = plan_fusion_regions(g)
    assert [p.brick_names for p in plans] == [
        ("gdn_0", "gdn_1"),
        ("moe_2",),
        ("gdn_3", "gdn_4"),
    ]
    assert plans[0].backend == "path_c"
    assert plans[1].backend == "dlpack_handoff"
    assert plans[2].backend == "path_c"


def test_planner_size_one_region_at_sparse_attn():
    """nsa never fuses with anything; lands as a size-1 region."""
    g = _chain("gdn", "nsa", "gdn")
    plans = plan_fusion_regions(g)
    assert [p.size for p in plans] == [1, 1, 1]
    assert [p.brick_names[0] for p in plans] == ["gdn_0", "nsa_1", "gdn_2"]


# ---------------------------------------------------------------------------
# Unit: hard limits
# ---------------------------------------------------------------------------


def test_planner_enforces_max_region_size():
    """A long fusable chain is split when max_region_size is hit."""
    g = _chain(*(["gdn"] * 10))
    plans = plan_fusion_regions(g, max_region_size=3)
    assert [p.size for p in plans] == [3, 3, 3, 1]
    # the split reason must mention the cap
    assert "max_region_size" in plans[0].reason


def test_planner_enforces_shared_mem_budget():
    """Tight shared-mem cap forces splits even for fusable chains.

    linear_attn estimates 4 KiB per brick; with cap=10 KiB we should fit
    two bricks per region (8 KiB) but not three.
    """
    g = _chain(*(["gdn"] * 5))
    plans = plan_fusion_regions(g, max_shared_mem_bytes=10 * 1024)
    sizes = [p.size for p in plans]
    assert sizes == [2, 2, 1]
    assert "shared-mem" in plans[0].reason


def test_planner_default_limits_pass_a_reasonable_chain():
    g = _chain(*(["gdn"] * DEFAULT_MAX_REGION_SIZE))
    plans = plan_fusion_regions(g)
    # 8 gdn at 4 KiB each = 32 KiB exactly — fits the default budget.
    assert len(plans) == 1
    assert plans[0].size == DEFAULT_MAX_REGION_SIZE
    assert plans[0].backend == "path_c"


# ---------------------------------------------------------------------------
# Unit: cost model sanity
# ---------------------------------------------------------------------------


def test_cost_model_zero_for_size_one_regions():
    g = _chain("gated_attention")
    plans = plan_fusion_regions(g)
    assert plans[0].estimated_savings_us == 0.0


def test_cost_model_zero_for_dlpack_handoff_regions():
    """A region whose backend is dlpack_handoff reports no savings."""
    g = _chain("gated_attention", "gated_attention")
    plans = plan_fusion_regions(g)
    # SDPA-SDPA can't fuse → two size-1 regions
    assert len(plans) == 2
    assert plans[0].estimated_savings_us == 0.0
    assert plans[1].estimated_savings_us == 0.0


def test_cost_model_grows_with_region_size():
    g_small = _chain("gdn", "gdn")
    g_large = _chain("gdn", "gdn", "gdn", "gdn")
    small = plan_fusion_regions(g_small)[0].estimated_savings_us
    large = plan_fusion_regions(g_large)[0].estimated_savings_us
    assert large > small > 0


def test_cost_model_penalises_category_mix():
    """A homogeneous region saves more than a mixed-category region of
    the same size, because mixing adds a register-pressure penalty."""
    homo = plan_fusion_regions(_chain("gdn", "gdn", "gdn"))[0]
    # gated_attention + mlp + mlp: 2 distinct categories
    mixed = plan_fusion_regions(_chain("gated_attention", "mlp", "mlp"))[0]
    assert homo.size == mixed.size == 3
    assert homo.estimated_savings_us > mixed.estimated_savings_us


# ---------------------------------------------------------------------------
# System: full Qwen3-Next-style 5-brick chain
# ---------------------------------------------------------------------------


def test_system_qwen3_next_pattern_planned_regions():
    """3 GDN fuse, gated_attention separates, moe is its own boundary."""
    g = from_block_specs(
        [
            {"kind": "gdn", "name": "g0"},
            {"kind": "gdn", "name": "g1"},
            {"kind": "gdn", "name": "g2"},
            {"kind": "gated_attention", "name": "attn",
             "params": {"num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 16}},
            {"kind": "moe", "name": "moe",
             "params": {"num_experts": 2, "top_k": 1}},
        ],
        hidden_size=64,
        instantiate=True,
    )
    plans = plan_fusion_regions(g)
    assert [p.brick_names for p in plans] == [
        ("g0", "g1", "g2"),
        ("attn",),
        ("moe",),
    ]
    assert plans[0].backend == "path_c"
    assert plans[0].is_fused is True
    assert plans[1].backend == "dlpack_handoff"
    assert plans[2].backend == "dlpack_handoff"


def test_system_ling26_pattern_planned_regions():
    """Ling 2.6: 7x bailing_linear (split at max_region_size=8) + mla + moe."""
    specs = [{"kind": "bailing_linear", "name": f"la{i}",
              "params": {"num_attention_heads": 4,
                         "num_key_value_heads": 2, "head_dim": 16}}
             for i in range(7)]
    specs.append(
        {"kind": "bailing_mla", "name": "mla",
         "params": {"num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 16, "kv_lora_rank": 16,
                    "qk_rope_head_dim": 8, "qk_nope_head_dim": 16,
                    "v_head_dim": 16}}
    )
    specs.append(
        {"kind": "bailing_moe", "name": "moe",
         "params": {"num_experts": 2, "top_k": 1}}
    )
    g = from_block_specs(specs, hidden_size=64, instantiate=True)
    plans = plan_fusion_regions(g)
    # 7 bailing_linear (fits under max_region_size=8 default, 7*4KiB=28KiB
    # under 32KiB default), mla solo, moe solo
    assert [tuple(p.brick_names) for p in plans] == [
        tuple(f"la{i}" for i in range(7)),
        ("mla",),
        ("moe",),
    ]
    assert plans[0].size == 7
    assert plans[0].backend == "path_c"


def test_system_kimi_linear_3to1_pattern():
    """Kimi Linear: 3 KDA + MLA repeated. Verify boundaries within a single
    repeat-unit."""
    g = _chain("kda", "kda", "kda", "mla", "kda", "kda", "kda", "mla")
    plans = plan_fusion_regions(g)
    assert [p.size for p in plans] == [3, 1, 3, 1]
    assert [p.brick_names[0] for p in plans[1::2]] == ["mla_3", "mla_7"]
    # both linear-attn regions saved time
    for p in plans[::2]:
        assert p.is_fused is True
        assert p.estimated_savings_us > 0


# ---------------------------------------------------------------------------
# System: auto_fuse_model annotation
# ---------------------------------------------------------------------------


def test_auto_fuse_model_attaches_plan_attribute():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.a._v4_brick_kind = "gdn"
            self.b = nn.Linear(8, 8)
            self.b._v4_brick_kind = "gdn"
            self.c = nn.Linear(8, 8)
            self.c._v4_brick_kind = "gated_attention"

    m = _M()
    out = auto_fuse_model(m)
    assert out is m
    plan = getattr(m, "_v4_fusion_plan")
    assert isinstance(plan, tuple)
    assert all(isinstance(p, FusionRegionPlan) for p in plan)
    # Total nodes preserved; the two gdn children are fused into one
    # region and gated_attention is its own boundary. Walk order on
    # nn.Module is implementation-defined, so verify by set membership.
    flat = tuple(name for p in plan for name in p.brick_names)
    assert set(flat) == {"a", "b", "c"}
    fused_regions = [p for p in plan if p.is_fused]
    assert len(fused_regions) == 1
    assert set(fused_regions[0].brick_names) == {"a", "b"}
    solo = [p for p in plan if p.size == 1]
    assert len(solo) == 1
    assert solo[0].brick_names == ("c",)
    graph = getattr(m, "_v4_fusion_brick_graph")
    assert set(graph.names) == {"a", "b", "c"}


def test_auto_fuse_model_respects_custom_limits():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(4):
                lin = nn.Linear(8, 8)
                lin._v4_brick_kind = "gdn"
                setattr(self, f"g{i}", lin)

    m = _M()
    auto_fuse_model(m, max_region_size=2)
    plan = getattr(m, "_v4_fusion_plan")
    assert [p.size for p in plan] == [2, 2]


def test_auto_fuse_model_default_shared_mem_budget_is_advertised():
    """Sanity: the public constant matches the documented 32 KiB cap."""
    assert DEFAULT_MAX_SHARED_MEM_BYTES == 32 * 1024
    assert DEFAULT_MAX_REGION_SIZE == 8
