"""VBSpec Stage D tests — memory_report (fusion-aware, KV-cache, AdamW)."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.fusion import from_block_specs, plan_fusion_regions
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.spec import (
    MemoryReport,
    estimate_memory,
    resolve_shapes,
)


_QWEN3_ENV = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}

_LING_ENV = {
    **_QWEN3_ENV,
    "S": 8192,
    "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
}


def _resolve(preset: str, env: dict):
    specs = build_preset_specs(preset, hidden_size=env["H"])
    g = from_block_specs(specs, hidden_size=env["H"], instantiate=False)
    return resolve_shapes(
        g, env, strict=False,
        available_side_channels=frozenset({"doc_ids", "token_ids"}),
    )


# ---------------------------------------------------------------------------
# Unit: estimate_memory invariants
# ---------------------------------------------------------------------------


def test_estimate_empty_graph_returns_zero_total():
    resolved = resolve_shapes(BrickGraph(nodes=()), _QWEN3_ENV)
    r = estimate_memory(resolved)
    assert isinstance(r, MemoryReport)
    assert r.total_bytes == 0
    assert r.summary()["total"] == 0


def test_estimate_size_one_graph_charges_one_brick():
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="m0"),))
    resolved = resolve_shapes(g, _QWEN3_ENV)
    r = estimate_memory(resolved, training=False)
    assert r.weights_bytes > 0
    assert r.grads_bytes == 0
    assert r.optimizer_bytes == 0
    # Single brick → no edges → no handoff
    assert r.edge_handoff_bytes == 0


def test_estimate_rejects_unknown_optimizer():
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="m0"),))
    resolved = resolve_shapes(g, _QWEN3_ENV)
    with pytest.raises(ValueError, match="unknown optimizer"):
        estimate_memory(resolved, optimizer="trash")


def test_estimate_rejects_invalid_dtype_bytes():
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="m0"),))
    resolved = resolve_shapes(g, _QWEN3_ENV)
    with pytest.raises(ValueError, match="dtype_bytes"):
        estimate_memory(resolved, dtype_bytes=0)


def test_estimate_rejects_negative_kv_cache_dtype():
    g = BrickGraph(nodes=(BrickNode(kind="mlp", name="m0"),))
    resolved = resolve_shapes(g, _QWEN3_ENV)
    with pytest.raises(ValueError, match="kv_cache_dtype_bytes"):
        estimate_memory(resolved, kv_cache_dtype_bytes=-1)


# ---------------------------------------------------------------------------
# Training vs inference deltas
# ---------------------------------------------------------------------------


def test_training_grows_total_vs_inference():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    train = estimate_memory(resolved, training=True, optimizer="adamw")
    infer = estimate_memory(resolved, training=False, optimizer="adamw")
    assert train.grads_bytes > 0
    assert train.optimizer_bytes > 0
    assert infer.grads_bytes == 0
    assert infer.optimizer_bytes == 0
    assert train.total_bytes > infer.total_bytes


def test_adamw_uses_more_optimizer_state_than_muon():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    adam = estimate_memory(resolved, optimizer="adamw")
    muon = estimate_memory(resolved, optimizer="muon")
    # AdamW = 8 bytes/param, Muon ≈ 4 → AdamW about 2× Muon for optimizer state
    assert adam.optimizer_bytes > muon.optimizer_bytes
    assert adam.optimizer_bytes >= 2 * muon.optimizer_bytes


def test_none_optimizer_zero_state():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    r = estimate_memory(resolved, optimizer="none")
    assert r.optimizer_bytes == 0


def test_kv_cache_only_in_inference_mode():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    train = estimate_memory(resolved, training=True, kv_cache_dtype_bytes=1)
    infer = estimate_memory(resolved, training=False, kv_cache_dtype_bytes=1)
    # In training mode, no kv-cache charged
    assert train.kv_cache_bytes == 0
    # In inference mode with kv_cache_dtype_bytes>0, attention bricks contribute
    assert infer.kv_cache_bytes > 0


def test_kv_cache_zero_when_dtype_bytes_zero():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    r = estimate_memory(resolved, training=False, kv_cache_dtype_bytes=0)
    assert r.kv_cache_bytes == 0


# ---------------------------------------------------------------------------
# Fusion-awareness — the headline feature
# ---------------------------------------------------------------------------


def test_fused_region_activations_use_max_not_sum():
    """3 GDN bricks in one fused region — activations should be MAX
    across bricks (shared registers), not SUM."""
    specs = build_preset_specs("qwen3_next", hidden_size=_QWEN3_ENV["H"])
    g = from_block_specs(
        specs, hidden_size=_QWEN3_ENV["H"], instantiate=False,
    )
    resolved = resolve_shapes(
        g, _QWEN3_ENV, strict=False,
        available_side_channels=frozenset({"doc_ids", "token_ids"}),
    )
    plans = plan_fusion_regions(g)
    fused = estimate_memory(resolved, fusion_plan=plans, training=True)
    unfused = estimate_memory(resolved, fusion_plan=None, training=True)
    assert fused.activations_bytes < unfused.activations_bytes
    # Specifically: the fused gdn region's activation slot should be MAX
    # across its 3 bricks (= one row), not sum.
    gdn_region = next(
        r for r in fused.per_region.values()
        if r.is_fused and r.brick_names == ("qwen3_gdn_0", "qwen3_gdn_1", "qwen3_gdn_2")
    )
    per_brick_acts = [
        fused.per_brick[n].activations_bytes for n in gdn_region.brick_names
    ]
    assert gdn_region.activations_bytes == max(per_brick_acts)


def test_fused_region_keeps_params_summed():
    """Weights aren't shared inside a fused region — params sum normally."""
    specs = build_preset_specs("qwen3_next", hidden_size=_QWEN3_ENV["H"])
    g = from_block_specs(
        specs, hidden_size=_QWEN3_ENV["H"], instantiate=False,
    )
    resolved = resolve_shapes(
        g, _QWEN3_ENV, strict=False,
        available_side_channels=frozenset({"doc_ids", "token_ids"}),
    )
    plans = plan_fusion_regions(g)
    r = estimate_memory(resolved, fusion_plan=plans)
    gdn_region = next(
        rr for rr in r.per_region.values()
        if rr.is_fused and rr.brick_names == ("qwen3_gdn_0", "qwen3_gdn_1", "qwen3_gdn_2")
    )
    summed = sum(r.per_brick[n].params_bytes for n in gdn_region.brick_names)
    assert gdn_region.params_bytes == summed


def test_handoff_skipped_for_fused_edges():
    """Edges inside a fused region don't materialise tensors."""
    specs = build_preset_specs("qwen3_next", hidden_size=_QWEN3_ENV["H"])
    g = from_block_specs(
        specs, hidden_size=_QWEN3_ENV["H"], instantiate=False,
    )
    resolved = resolve_shapes(
        g, _QWEN3_ENV, strict=False,
        available_side_channels=frozenset({"doc_ids", "token_ids"}),
    )
    plans = plan_fusion_regions(g)
    fused = estimate_memory(resolved, fusion_plan=plans)
    unfused = estimate_memory(resolved, fusion_plan=None)
    assert fused.edge_handoff_bytes < unfused.edge_handoff_bytes


def test_handoff_can_be_disabled_entirely():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    r = estimate_memory(resolved, include_edge_handoff=False)
    assert r.edge_handoff_bytes == 0


# ---------------------------------------------------------------------------
# fits_on / device budget gate
# ---------------------------------------------------------------------------


def test_fits_on_validates_headroom_range():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    r = estimate_memory(resolved)
    with pytest.raises(ValueError, match="headroom"):
        r.fits_on(80 * 10**9, headroom=0.0)
    with pytest.raises(ValueError, match="headroom"):
        r.fits_on(80 * 10**9, headroom=1.5)


def test_fits_on_validates_device_size():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    r = estimate_memory(resolved)
    with pytest.raises(ValueError, match="device_hbm_bytes"):
        r.fits_on(0)


def test_fits_on_returns_bool():
    resolved = _resolve("qwen3_next", _QWEN3_ENV)
    r = estimate_memory(resolved)
    assert isinstance(r.fits_on(100 * 10**9), bool)
    # 1-byte device certainly doesn't fit qwen3_next
    assert r.fits_on(1) is False


# ---------------------------------------------------------------------------
# System: per-preset memory plausibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", [
    "qwen3_next", "kimi_linear", "kimi_k2", "deepseek_v3",
    "deepseek_v4_flash", "gemma4", "mistral4", "ling26", "longcat",
    "nemotron3", "zaya1", "arcee_trinity",
])
def test_system_every_preset_produces_well_formed_memory_report(preset_name):
    env_overrides = {
        "qwen3_next": {}, "kimi_linear": {**_LING_ENV}, "kimi_k2": {**_LING_ENV},
        "deepseek_v3": {**_LING_ENV}, "deepseek_v4_flash": {},
        "gemma4": {"sliding_window_size": 1024},
        "mistral4": {**_LING_ENV}, "ling26": {**_LING_ENV},
        "longcat": {**_LING_ENV},
        "nemotron3": {"d_state": 64},
        "zaya1": {"fine_window": 256, "coarse_block_size": 16},
        "arcee_trinity": {"sliding_window_size": 1024},
    }
    env = {**_QWEN3_ENV, **env_overrides[preset_name]}
    specs = build_preset_specs(preset_name, hidden_size=env["H"])
    g = from_block_specs(specs, hidden_size=env["H"], instantiate=False)
    resolved = resolve_shapes(
        g, env, strict=False,
        available_side_channels=frozenset({"doc_ids", "token_ids"}),
    )
    plans = plan_fusion_regions(g)
    r = estimate_memory(resolved, fusion_plan=plans, training=True)
    assert r.total_bytes > 0
    assert r.weights_bytes > 0
    assert r.activations_bytes > 0
    # Roughly: any single repeat-unit must fit on an 80 GB device
    assert r.fits_on(80 * 10**9, headroom=0.95)
