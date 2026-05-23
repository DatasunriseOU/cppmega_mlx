"""GalCov Stage D — 71-architecture coverage matrix.

Each entry from Sebastian Raschka's LLM Architecture Gallery (71 modern
open-weight models) is mapped to either:

  - a preset name from ``cppmega_v4.architectures.presets.PRESETS``, OR
  - ``None`` plus an ``xfail_reason`` for known gaps (true gaps from
    VisualBuilderPlan.md §10 where the brick exists but the production
    preset wiring is deferred — Tiny Aya, GPT-2 XL, Gemma 4 E2B/E4B,
    xLSTM 7B).

Two parametrized smoke tests run per entry:

  1. ``test_every_gallery_entry_passes_smoke_pipeline`` — runs the
     production validation cascade: build_preset_specs →
     from_block_specs → ModelBuildSpec(graph, ce, adamw) →
     verify_build_spec → verify_and_estimate. Asserts no ERROR-severity
     diagnostics and resolver/memory completes.

  2. ``test_every_gallery_entry_fits_on_canonical_topology`` — runs the
     fit check on two canonical topologies:
        * m3_ultra_solo at hidden=64 (toy: must fit)
        * h100_8x at hidden=512 (small but realistic: should fit)

Unmapped entries are marked ``xfail(strict=False)`` so accidental future
fixes flip them green automatically and trip the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cppmega_v4.architectures.presets import PRESETS, build_preset_specs
from cppmega_v4.buildspec import (
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
    verify_build_spec,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.parallelism import h100_8x, m3_ultra_solo
from cppmega_v4.spec import verify_and_estimate


@dataclass(frozen=True)
class GalleryEntry:
    """One row of the Raschka gallery coverage matrix."""

    gid: int
    name: str
    preset: str | None
    xfail_reason: str | None = None


# ---------------------------------------------------------------------------
# 71-entry Raschka gallery fixture.
#
# Sources:
#   - VisualBuilderPlan.md §10 (family breakdown)
#   - Sebastian Raschka, "The Big LLM Architecture Comparison" (2025/26 ed.)
#
# Mapping rule of thumb: pick the preset whose family matches the
# architectural class declared by Raschka, NOT the parameter count.
# Multiple gallery entries map to the same preset by design — e.g.
# llama3_8b covers all attention+mlp linear-chain families.
# ---------------------------------------------------------------------------


GALLERY: tuple[GalleryEntry, ...] = (
    # --- abs-pos / legacy ---
    GalleryEntry(1, "GPT-2 XL", "gpt2_xl"),

    # --- LLaMA family (attention + mlp linear chain) ---
    GalleryEntry(2, "Llama 3.1 70B", "llama3_8b"),
    GalleryEntry(3, "Llama 3.1 405B", "llama3_8b"),
    GalleryEntry(4, "OLMo 2 7B", "olmo2_7b"),

    # --- Gemma 3 dense (sliding/global GQA + mlp) ---
    GalleryEntry(5, "Gemma 3 4B", "gemma3_27b"),
    GalleryEntry(6, "Gemma 3 12B", "gemma3_27b"),
    GalleryEntry(7, "Gemma 3 27B", "gemma3_27b"),

    # --- Mixtral / sliding+MoE family ---
    GalleryEntry(8, "Mixtral 8x22B", "llama4_maverick"),
    GalleryEntry(9, "Llama 4 Maverick", "llama4_maverick"),
    GalleryEntry(10, "Qwen3 0.6B", "qwen3_dense_0_6b"),
    GalleryEntry(11, "Qwen3 30B-A3B", "qwen3_30b_a3b"),
    GalleryEntry(12, "Qwen3 235B-A22B", "qwen3_235b_a22b"),
    GalleryEntry(13, "Qwen3 4B", "qwen3_dense_4b"),
    GalleryEntry(14, "Qwen3 8B", "qwen3_dense_8b"),
    GalleryEntry(15, "Qwen3 32B", "qwen3_dense_32b"),

    # --- SmolLM3 (NoPE per-layer) ---
    GalleryEntry(16, "SmolLM3 3B", "smollm3"),

    # --- GLM-style (shared expert) ---
    GalleryEntry(17, "GLM 4.5 Air", "glm_45_air"),
    GalleryEntry(18, "GLM 4.5", "glm_45"),
    GalleryEntry(19, "GPT-OSS 20B", "gpt_oss_20b"),
    GalleryEntry(20, "GPT-OSS 120B", "gpt_oss_120b"),
    GalleryEntry(21, "Gemma 3 270M", "gemma3_270m"),
    GalleryEntry(22, "Grok 2.5", "grok25"),

    # --- DeepSeek family (MLA + MoE) ---
    GalleryEntry(23, "DeepSeek V3", "deepseek_v3"),
    GalleryEntry(24, "Llama Nemotron Nano 4B", "nemotron3"),
    GalleryEntry(25, "Mistral Small 3.1", "mistral_small_3_1"),
    GalleryEntry(26, "OLMo 3 7B", "olmo3_7b"),
    GalleryEntry(27, "OLMo 3 32B", "olmo3_32b"),
    GalleryEntry(28, "DeepSeek R1", "deepseek_v3"),
    GalleryEntry(29, "DeepSeek V3.2", "deepseek_v3"),
    GalleryEntry(30, "Mistral Large 3", "deepseek_v3"),
    GalleryEntry(31, "Llama Nemotron Super", "nemotron3"),
    GalleryEntry(32, "GLM 4.7", "glm_47"),

    # --- Kimi family (linear / k2 hybrid) ---
    GalleryEntry(33, "Kimi Linear", "kimi_linear"),
    GalleryEntry(34, "Kimi K2", "kimi_k2"),
    GalleryEntry(35, "Kimi K2.5", "kimi_k2"),
    GalleryEntry(36, "Gemma 3 small", "gemma3_270m"),
    GalleryEntry(37, "Kimi K2.6", "kimi_k2"),
    GalleryEntry(38, "Llama 3 8B", "llama3_8b"),
    GalleryEntry(39, "Llama 4 Scout", "llama4_maverick"),
    GalleryEntry(40, "Step3.5 Flash", "step3_5_flash"),
    GalleryEntry(41, "MiniMax M2", "minimax_m2"),
    GalleryEntry(42, "Phi-4", "phi4"),
    GalleryEntry(43, "MiniMax M2.5", "minimax_m2_5"),

    # --- V7-Q04: Tiny Aya parallel-block topology wired ---
    GalleryEntry(44, "Tiny Aya", "tiny_aya_parallel"),

    GalleryEntry(45, "MiniMax M2.7", "minimax_m2_7"),
    GalleryEntry(46, "DeepSeek V4-Flash", "deepseek_v4_flash"),
    GalleryEntry(47, "GLM 5", "glm_5"),
    GalleryEntry(48, "DeepSeek V4-Pro", "deepseek_v4_flash"),
    GalleryEntry(49, "Phi-4 mini", "phi4"),

    # --- V7-Q04: xLSTM 7B (matrix-memory LSTM, no self-attention) ---
    GalleryEntry(50, "xLSTM 7B", "xlstm_7b"),

    GalleryEntry(51, "GLM 5.1", "glm_51"),
    GalleryEntry(52, "Sarvam 30B", "sarvam_30b"),
    GalleryEntry(53, "Sarvam 105B", "sarvam_105b"),
    GalleryEntry(54, "Ling 2.5", "ling25"),
    GalleryEntry(55, "Ling 2.6", "ling26"),
    GalleryEntry(56, "Nanbeige 4.1", "nanbeige_4_1"),

    # --- V7-Q04: Gemma 4 E2B/E4B per-layer embed wired ---
    GalleryEntry(57, "Gemma 4 E2B", "gemma_4_e2b"),
    GalleryEntry(58, "Gemma 4 E4B", "gemma_4_e4b"),

    GalleryEntry(59, "Gemma 4 26B-A4B", "gemma4"),
    GalleryEntry(60, "Gemma 4 31B", "gemma4_31b"),
    GalleryEntry(61, "LongCat Flash-Lite", "longcat"),
    GalleryEntry(62, "Mistral 4 Small", "mistral4"),
    GalleryEntry(63, "Zaya1 8B", "zaya1"),
    GalleryEntry(64, "Intellect-3", "intellect_3"),
    GalleryEntry(65, "Qwen3 Next 35B", "qwen3_next"),
    GalleryEntry(66, "Qwen3 6.27B Coder", "qwen3_6_27b"),
    GalleryEntry(67, "Qwen3 Coder Flash", "qwen3_coder_flash"),
    GalleryEntry(68, "MiMo v2.5", "mimo_v2_5"),
    GalleryEntry(69, "MiMo v2.5 Pro", "mimo_v2_5_pro"),
    GalleryEntry(70, "MiMo v2 Flash", "mimo_v2_flash"),
    GalleryEntry(71, "Granite 4.1", "granite_4_1"),
)


def test_gallery_fixture_has_exactly_71_entries():
    assert len(GALLERY) == 71


def test_gallery_fixture_ids_are_unique_and_dense():
    ids = sorted(e.gid for e in GALLERY)
    assert ids == list(range(1, 72))


def test_gallery_fixture_presets_all_registered():
    """Every named preset must exist in PRESETS (gallery typo guard)."""
    for entry in GALLERY:
        if entry.preset is not None:
            assert entry.preset in PRESETS, (
                f"#{entry.gid} {entry.name!r} → unknown preset "
                f"{entry.preset!r}"
            )


def test_gallery_fixture_xfail_entries_carry_reason():
    for entry in GALLERY:
        if entry.preset is None:
            assert entry.xfail_reason, (
                f"#{entry.gid} {entry.name!r} has no preset AND no "
                f"xfail_reason — clarify intent"
            )


def test_gallery_fixture_known_gaps_are_zero():
    """V7-Q04 closure: all 5 prior gaps (GPT-2 XL, Tiny Aya, xLSTM,
    Gemma 4 E2B/E4B) now have preset factories wired. Lock at zero
    so a regression that drops a wired preset surfaces immediately."""
    gaps = tuple(e for e in GALLERY if e.preset is None)
    assert len(gaps) == 0, (
        f"Expected 0 gaps post-Q04; got {len(gaps)}: "
        f"{[(e.gid, e.name) for e in gaps]}"
    )


# ---------------------------------------------------------------------------
# Smoke pipeline parametrised per gallery entry.
# ---------------------------------------------------------------------------


_HIDDEN_SMOKE = 64


def _smoke_pipeline(preset: str) -> None:
    specs = build_preset_specs(preset, hidden_size=_HIDDEN_SMOKE)
    assert specs, f"empty spec list for preset {preset!r}"
    graph = from_block_specs(specs, hidden_size=_HIDDEN_SMOKE, instantiate=False)
    # Loss head name = last brick in the chain (presets terminate with an
    # mlp/moe/gated_attention that doubles as the logits-producer in the
    # smoke harness; production wiring threads a real lm_head via the
    # rewriters, but verification only needs name parity).
    head_name = graph.nodes[-1].name
    build_spec = ModelBuildSpec(
        graph=graph,
        loss=cross_entropy_loss(head_output_name=head_name),
        optim=adamw(),
    )
    diag = verify_build_spec(build_spec, check_shapes=False)
    assert not diag.has_errors, (
        f"verify_build_spec for preset {preset!r} returned errors: "
        f"{[d.message for d in diag.errors]}"
    )
    result = verify_and_estimate(graph, preset_name=preset)
    assert result.memory.total_bytes > 0


@pytest.mark.parametrize(
    "entry",
    [pytest.param(e, id=f"{e.gid:02d}_{e.name}") for e in GALLERY],
)
def test_every_gallery_entry_passes_smoke_pipeline(entry: GalleryEntry):
    if entry.preset is None:
        pytest.xfail(entry.xfail_reason or "no preset wired")
    _smoke_pipeline(entry.preset)


# ---------------------------------------------------------------------------
# Canonical topology fit check (toy hidden + small-prod hidden).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [pytest.param(e, id=f"{e.gid:02d}_{e.name}") for e in GALLERY],
)
def test_every_gallery_entry_fits_on_m3_ultra_solo_at_toy_hidden(
    entry: GalleryEntry,
):
    if entry.preset is None:
        pytest.xfail(entry.xfail_reason or "no preset wired")
    specs = build_preset_specs(entry.preset, hidden_size=_HIDDEN_SMOKE)
    graph = from_block_specs(specs, hidden_size=_HIDDEN_SMOKE, instantiate=False)
    topo = m3_ultra_solo()
    hbm = topo.devices[0].hbm_bytes
    result = verify_and_estimate(
        graph, preset_name=entry.preset, device_hbm_bytes=hbm,
    )
    assert result.fits_on_device is True, (
        f"#{entry.gid} {entry.name!r} (preset={entry.preset}) does NOT "
        f"fit on m3_ultra_solo at hidden={_HIDDEN_SMOKE}: "
        f"total={result.memory.total_bytes / 1024**3:.2f} GiB"
    )


@pytest.mark.parametrize(
    "entry",
    [pytest.param(e, id=f"{e.gid:02d}_{e.name}") for e in GALLERY],
)
def test_every_gallery_entry_fits_on_h100_8x_at_small_hidden(
    entry: GalleryEntry,
):
    if entry.preset is None:
        pytest.xfail(entry.xfail_reason or "no preset wired")
    hidden = 512
    specs = build_preset_specs(entry.preset, hidden_size=hidden)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    topo = h100_8x()
    # Aggregate HBM across 8 devices (single-replica budget upper bound).
    hbm = sum(d.hbm_bytes for d in topo.devices)
    result = verify_and_estimate(
        graph, preset_name=entry.preset, device_hbm_bytes=hbm,
    )
    assert result.fits_on_device is True, (
        f"#{entry.gid} {entry.name!r} (preset={entry.preset}) does NOT "
        f"fit on h100_8x at hidden={hidden}: "
        f"total={result.memory.total_bytes / 1024**3:.2f} GiB"
    )
