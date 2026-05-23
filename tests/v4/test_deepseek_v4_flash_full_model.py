"""V7-Q13: DeepSeek V4 Flash preset is a FULL connected mini-model.

Operator complaint: opening the preset surfaced 4 disconnected bricks
that didn't match Raschka's diagram (no MTP head, no DSA compression
knobs, no engram memory slots, no positional embedding).

This test pins the corrected preset shape:
- abs_pos_embed input
- 3 backbone layers (each: lightning_indexer + csa_hca + engram +
  dsv4_attention + rmsnorm + moe)
- final rmsnorm
- nemotron_h_mtp drafter head

Plus the DSA compression contract — top_k ≈ H/128, kv_lora_rank ≈ H/4.
"""

from __future__ import annotations

from cppmega_v4.architectures.presets import (
    _deepseek_v4_flash, PRESETS, build_preset_specs,
)


def test_dsv4_flash_has_input_embedding() -> None:
    specs = _deepseek_v4_flash(hidden_size=512)
    assert specs[0]["kind"] == "abs_pos_embed"
    assert specs[0]["params"]["max_position_embeddings"] >= 4096


def test_dsv4_flash_has_three_backbone_layers() -> None:
    specs = _deepseek_v4_flash(hidden_size=512)
    layer_attns = [s for s in specs if s["kind"] == "dsv4_attention"]
    assert len(layer_attns) == 3, (
        f"expected 3 backbone layers, got {len(layer_attns)}")
    layer_moes = [s for s in specs if s["kind"] == "moe"]
    assert len(layer_moes) == 3
    layer_norms = [s for s in specs if s["kind"] == "rmsnorm"]
    # 3 post-norm + 1 final = 4
    assert len(layer_norms) == 4


def test_dsv4_flash_has_lightning_indexer_and_engram() -> None:
    specs = _deepseek_v4_flash(hidden_size=512)
    kinds = [s["kind"] for s in specs]
    assert kinds.count("lightning_indexer") == 3
    assert kinds.count("engram") == 3
    assert kinds.count("csa_hca") == 3


def test_dsv4_flash_has_mtp_drafter_tail() -> None:
    specs = _deepseek_v4_flash(hidden_size=512)
    assert specs[-1]["kind"] == "nemotron_h_mtp"
    assert specs[-1]["params"]["drafter_k"] == 2


def test_dsv4_flash_dsa_compression_contract() -> None:
    """top_k ≈ H/128, kv_lora_rank ≈ H/4 — the actual DeepSeek-V4
    Flash sparsity + LoRA ratios from the paper."""
    specs = _deepseek_v4_flash(hidden_size=512)
    idx = next(s for s in specs if s["kind"] == "lightning_indexer")
    assert idx["params"]["top_k"] == max(4, 512 // 128)  # = 4
    attn = next(s for s in specs if s["kind"] == "dsv4_attention")
    assert attn["params"]["kv_lora_rank"] == 512 // 4  # = 128


def test_dsv4_flash_via_build_preset_specs_is_consistent() -> None:
    """build_preset_specs(name, hidden) is the canonical UI entry —
    pin that it returns the same brick chain we just authored."""
    via_factory = _deepseek_v4_flash(hidden_size=512)
    via_dispatch = build_preset_specs("deepseek_v4_flash", hidden_size=512)
    assert [s["kind"] for s in via_dispatch] \
        == [s["kind"] for s in via_factory]
    assert [s["name"] for s in via_dispatch] \
        == [s["name"] for s in via_factory]


def test_dsv4_flash_emits_unique_brick_names() -> None:
    """Auto-edge in the canvas keys on brick.name; collisions break
    sequential edges. Verify every layer's bricks have unique names."""
    specs = _deepseek_v4_flash(hidden_size=512)
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)), (
        f"duplicate brick names: {names}")
