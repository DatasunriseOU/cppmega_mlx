"""GalCov Stage A tests — new preset factories for gallery coverage.

Adds ~25 preset factories that compose existing 22 bricks into
architectures from Sebastian Raschka's LLM gallery (LLaMA, Mistral,
Phi, Qwen3 dense, Mixtral-style, GLM, OLMo, Gemma 3 dense, sliding+
global+MoE family). ZERO new bricks — just compositions.
"""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import (
    PRESETS,
    available_presets,
    build_preset_specs,
)
from cppmega_v4.fusion import (
    from_block_specs,
    plan_fusion_regions,
)
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


# ---------------------------------------------------------------------------
# Coverage: 25+ new presets exist
# ---------------------------------------------------------------------------


_NEW_PRESETS_GALCOV_A: list[str] = [
    "llama3_8b", "llama3_2_1b", "llama3_2_3b",
    "olmo2_7b", "olmo3_7b", "olmo3_32b",
    "mistral_small_3_1", "nanbeige_4_1", "phi4", "granite_4_1",
    "qwen3_dense_0_6b", "qwen3_dense_4b", "qwen3_dense_8b",
    "qwen3_dense_32b", "qwen3_6_27b", "smollm3",
    "llama4_maverick", "qwen3_235b_a22b", "qwen3_30b_a3b", "grok25",
    "qwen3_coder_flash", "minimax_m2_5", "minimax_m2_7",
    "mimo_v2_5", "mimo_v2_5_pro", "tencent_hy3",
    "glm_45", "glm_47", "glm_45_air", "intellect_3", "sarvam_30b",
    "glm_5", "glm_51", "sarvam_105b",
    "gpt_oss_120b", "gpt_oss_20b", "minimax_m2", "mimo_v2_flash",
    "step3_5_flash", "tiny_aya", "ling25", "laguna_xs2",
    "gemma3_27b", "gemma3_270m", "gemma4_31b",
]


def test_galcov_a_total_preset_count_grew_past_40():
    assert len(available_presets()) >= 40, (
        f"expected ≥40 presets after GalCov-A; got {len(available_presets())}"
    )


@pytest.mark.parametrize("name", _NEW_PRESETS_GALCOV_A)
def test_galcov_a_preset_registered(name):
    assert name in PRESETS, f"missing GalCov-A preset: {name!r}"


# ---------------------------------------------------------------------------
# Each new preset instantiates + plans regions cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NEW_PRESETS_GALCOV_A)
def test_galcov_a_preset_emits_nonempty_spec(name):
    specs = build_preset_specs(name, hidden_size=64)
    assert len(specs) > 0
    for s in specs:
        assert s["kind"] in BLOCK_BUILDERS, (
            f"{name!r} uses kind {s['kind']!r} not in BLOCK_BUILDERS"
        )


@pytest.mark.parametrize("name", _NEW_PRESETS_GALCOV_A)
def test_galcov_a_preset_instantiates_at_hidden_64(name):
    specs = build_preset_specs(name, hidden_size=64)
    graph = from_block_specs(specs, hidden_size=64, instantiate=True)
    assert len(graph.nodes) == len(specs)
    for node, spec in zip(graph.nodes, specs):
        assert node.kind == spec["kind"]


@pytest.mark.parametrize("name", _NEW_PRESETS_GALCOV_A)
def test_galcov_a_preset_plans_fusion_regions(name):
    specs = build_preset_specs(name, hidden_size=64)
    graph = from_block_specs(specs, hidden_size=64, instantiate=False)
    plans = plan_fusion_regions(graph)
    flat = [n for p in plans for n in p.brick_names]
    assert flat == [s["name"] for s in specs], (
        f"{name!r} planner output doesn't cover all bricks in order"
    )


# ---------------------------------------------------------------------------
# Replication still works for new presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["llama3_8b", "qwen3_30b_a3b", "glm_45", "tiny_aya"],
)
def test_galcov_a_preset_replicates_with_unique_names(name):
    unit = build_preset_specs(name, hidden_size=64)
    n = len(unit)
    full = build_preset_specs(name, hidden_size=64, num_layers=2 * n)
    names = [s["name"] for s in full]
    assert len(names) == len(set(names))
    assert len(full) == 2 * n


# ---------------------------------------------------------------------------
# Category-specific sanity (composition shape)
# ---------------------------------------------------------------------------


def test_llama_family_has_attention_plus_mlp_pattern():
    """LLaMA-style: alternating attention + mlp (or just one pair per unit)."""
    specs = build_preset_specs("llama3_8b", hidden_size=64)
    kinds = [s["kind"] for s in specs]
    assert "attention" in kinds
    assert "mlp" in kinds


def test_mixtral_style_has_moe():
    for name in ("llama4_maverick", "qwen3_235b_a22b", "grok25"):
        kinds = [s["kind"] for s in build_preset_specs(name, hidden_size=64)]
        assert "moe" in kinds, f"{name!r} should contain moe brick"


def test_glm_style_has_moe_plus_shared_mlp():
    for name in ("glm_45", "glm_47", "intellect_3"):
        kinds = [s["kind"] for s in build_preset_specs(name, hidden_size=64)]
        assert "moe" in kinds
        # Shared expert modelled as trailing mlp
        assert kinds[-1] == "mlp", (
            f"{name!r} should end with shared mlp; got tail kind={kinds[-1]!r}"
        )


def test_mla_moe_style_uses_mla_brick():
    for name in ("glm_5", "glm_51", "sarvam_105b"):
        kinds = [s["kind"] for s in build_preset_specs(name, hidden_size=64)]
        assert "mla" in kinds
        assert "moe" in kinds


def test_sliding_global_moe_uses_gqa_sliding_plus_gated_attention():
    for name in ("gpt_oss_120b", "minimax_m2", "tiny_aya"):
        kinds = [s["kind"] for s in build_preset_specs(name, hidden_size=64)]
        assert "gqa_sliding" in kinds
        assert "gated_attention" in kinds
        assert "moe" in kinds


def test_gemma3_dense_pattern_has_sliding_global_mlp():
    for name in ("gemma3_27b", "gemma3_270m", "gemma4_31b"):
        kinds = [s["kind"] for s in build_preset_specs(name, hidden_size=64)]
        assert "gqa_sliding" in kinds
        assert "gated_attention" in kinds
        assert "mlp" in kinds
