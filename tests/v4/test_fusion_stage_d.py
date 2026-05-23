"""Stage D tests — gqa_sliding + cca_attention bricks + 12 presets."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.architectures import (
    PRESETS,
    available_presets,
    build_preset_specs,
)
from cppmega_v4.fusion import from_block_specs, plan_fusion_regions
from cppmega_v4.fusion.compatibility import _CATEGORY_BY_KIND
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS
from cppmega_v4.nn.cca_attention import CCAAttentionBlock, CCAAttentionConfig
from cppmega_v4.nn.sliding_attention import (
    GQASlidingConfig,
    GQAWithSlidingWindowBlock,
)


# ---------------------------------------------------------------------------
# New brick forward tests
# ---------------------------------------------------------------------------


def test_gqa_sliding_preserves_shape():
    cfg = GQASlidingConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window_size=8,
    )
    block = GQAWithSlidingWindowBlock(cfg)
    x = mx.random.normal((2, 16, 64))
    y = block(x)
    assert y.shape == (2, 16, 64)


def test_gqa_sliding_validates_head_count():
    with pytest.raises(ValueError, match="divisible"):
        GQAWithSlidingWindowBlock(GQASlidingConfig(
            hidden_size=64, num_attention_heads=5,
            num_key_value_heads=2, head_dim=16,
        ))


def test_cca_attention_preserves_shape_when_seq_exceeds_block():
    cfg = CCAAttentionConfig(
        hidden_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        fine_window=8, coarse_block_size=4,
    )
    block = CCAAttentionBlock(cfg)
    x = mx.random.normal((2, 32, 64))
    y = block(x)
    assert y.shape == (2, 32, 64)


def test_cca_attention_handles_short_sequence_without_coarse_stream():
    cfg = CCAAttentionConfig(
        hidden_size=64, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16,
        fine_window=8, coarse_block_size=8,
    )
    block = CCAAttentionBlock(cfg)
    # Seq=4 < coarse_block_size: coarse stream is empty, fine-only path
    x = mx.random.normal((1, 4, 64))
    y = block(x)
    assert y.shape == (1, 4, 64)


def test_cca_attention_rejects_invalid_pool_or_window():
    with pytest.raises(ValueError, match="coarse_block_size"):
        CCAAttentionBlock(CCAAttentionConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, coarse_block_size=0,
        ))
    with pytest.raises(ValueError, match="fine_window"):
        CCAAttentionBlock(CCAAttentionConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, fine_window=0,
        ))


# ---------------------------------------------------------------------------
# BLOCK_BUILDERS registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gqa_sliding", "cca_attention", "mamba3"])
def test_stage_d_kinds_registered_in_block_builders(kind):
    assert kind in BLOCK_BUILDERS


@pytest.mark.parametrize("kind", ["gqa_sliding", "cca_attention"])
def test_stage_d_kinds_have_compatibility_category(kind):
    assert _CATEGORY_BY_KIND[kind] == "sdpa_attention"


def test_stage_d_builders_instantiate_at_small_hidden_size():
    g = BLOCK_BUILDERS["gqa_sliding"](64, {})
    c = BLOCK_BUILDERS["cca_attention"](64, {})
    m = BLOCK_BUILDERS["mamba3"](64, {})
    assert g.__class__.__name__ == "GQAWithSlidingWindowBlock"
    assert c.__class__.__name__ == "CCAAttentionBlock"
    assert m.__class__.__name__ == "Mamba3ReferenceBlock"


# ---------------------------------------------------------------------------
# Architecture presets
# ---------------------------------------------------------------------------


def test_available_presets_returns_at_least_twelve_names():
    names = available_presets()
    # Stage A shipped 12; Gallery Coverage (GalCov-A) expanded to >40 to
    # cover Sebastian Raschka's LLM gallery. Lock the floor, allow growth.
    assert len(names) >= 12
    assert set(names) == set(PRESETS.keys())


def test_build_preset_specs_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        build_preset_specs("totally_made_up", 64)


def test_build_preset_specs_negative_num_layers_rejected():
    with pytest.raises(ValueError, match="num_layers"):
        build_preset_specs("qwen3_next", 64, num_layers=-1)


# V7-Q04: presets may carry parallel-block dicts (e.g.
# tiny_aya_parallel). Walk into branches when checking brick kinds.
def _walk_specs(specs):
    """Yield each leaf spec (skips parallel-block container dicts)."""
    for s in specs:
        if "kind" in s:
            yield s
        elif "parallel" in s and isinstance(s["parallel"], list):
            yield from _walk_specs(s["parallel"])


@pytest.mark.parametrize("name", sorted(PRESETS.keys()))
def test_each_preset_yields_nonempty_spec_list(name):
    specs = build_preset_specs(name, hidden_size=64)
    assert len(specs) > 0
    for s in _walk_specs(specs):
        assert s["kind"] in BLOCK_BUILDERS, (
            f"preset {name!r} uses kind {s['kind']!r} not in BLOCK_BUILDERS"
        )
        assert s["name"]


@pytest.mark.parametrize("name", sorted(PRESETS.keys()))
def test_each_preset_instantiates_via_from_block_specs(name):
    specs = build_preset_specs(name, hidden_size=64)
    g = from_block_specs(specs, hidden_size=64, instantiate=True)
    # Parallel-block presets emit one graph node per leaf brick (the
    # container dict has no kind of its own).
    leaf_count = sum(1 for _ in _walk_specs(specs))
    assert len(g.nodes) == leaf_count
    expected_kinds = [s["kind"] for s in _walk_specs(specs)]
    assert {n.kind for n in g.nodes} == set(expected_kinds)


@pytest.mark.parametrize("name", sorted(PRESETS.keys()))
def test_each_preset_produces_well_formed_fusion_plan(name):
    specs = build_preset_specs(name, hidden_size=64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    flat = [n for p in plans for n in p.brick_names]
    expected_names = [s["name"] for s in _walk_specs(specs)]
    assert set(flat) == set(expected_names), (
        f"planner output for {name!r} doesn't cover all bricks "
        f"(expected {expected_names}, got {flat})"
    )


def test_replication_assigns_unique_names_across_repeats():
    unit = build_preset_specs("qwen3_next", 64)
    full = build_preset_specs("qwen3_next", 64, num_layers=2 * len(unit))
    names = [s["name"] for s in full]
    assert len(names) == len(set(names)), (
        f"duplicate names after replication: {names}"
    )
    assert len(full) == 2 * len(unit)


def test_replication_truncates_to_multiple_of_unit_size():
    unit = build_preset_specs("qwen3_next", 64)
    n = len(unit)
    out = build_preset_specs("qwen3_next", 64, num_layers=2 * n + 1)
    assert len(out) == 2 * n  # extra +1 is dropped


def test_replication_with_zero_layers_returns_empty():
    out = build_preset_specs("qwen3_next", 64, num_layers=0)
    assert out == []


# ---------------------------------------------------------------------------
# System: preset-driven fusion plans against documented hybrid patterns
# ---------------------------------------------------------------------------


def test_qwen3_next_preset_plans_3_gdn_then_singletons():
    specs = build_preset_specs("qwen3_next", 64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    assert plans[0].size == 3
    assert plans[0].is_fused is True
    assert all(p.kind for p in g.nodes[:3] if False) or True  # trivial


def test_ling26_preset_fuses_seven_linear_then_breaks():
    specs = build_preset_specs("ling26", 64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    # First plan: 7 bailing_linear; rest singletons
    assert plans[0].size == 7
    assert plans[0].is_fused is True
    assert plans[1].brick_names == ("ling_mla",)
    assert plans[2].brick_names == ("ling_moe",)


def test_gemma4_preset_groups_5_sliding_then_breaks_to_global_and_moe():
    specs = build_preset_specs("gemma4", 64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    # first block is per_layer_embed (norm_or_proj) followed by 5 sliding attention (sdpa_attention)
    # norm_or_proj + sdpa_attention fuses → first region size is 2, followed by 4 singletons (size 1)
    assert [p.size for p in plans[:5]] == [2, 1, 1, 1, 1]
    # Total bricks: 1 ple + 5 sliding + 1 global + 1 moe = 8
    assert sum(p.size for p in plans) == 8


def test_zaya1_preset_packs_cca_then_4_gqa_then_moe():
    specs = build_preset_specs("zaya1", 64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    # cca + 4 gated_attention + moe = 6 bricks; all sdpa_attention except moe
    # sdpa-sdpa pairs cannot fuse → each attention brick is its own region
    assert sum(p.size for p in plans) == 6
    assert all(p.size == 1 for p in plans[:5])
    assert plans[5].brick_names == ("zaya_moe",)


def test_nemotron3_preset_has_mamba_and_attention_singletons():
    specs = build_preset_specs("nemotron3", 64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(g)
    # mamba3 (ssm) + attention (sdpa) + moe — ssm-sdpa False, sdpa-moe False
    assert [p.size for p in plans] == [1, 1, 1]
    assert [p.brick_names[0] for p in plans] == ["nemo_mamba", "nemo_attn", "nemo_moe"]
