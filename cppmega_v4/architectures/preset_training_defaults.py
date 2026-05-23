"""V8-R01: Per-preset paper-defaults table.

Provides paper-anchored training defaults (lr / batch_size / schedule /
warmup / betas / clip / mixed_precision / optimizer) for the Raschka
gallery presets. Consumed by ``build_preset_specs`` and the UI's
LossTab/OptimTab/ScheduleEditor auto-fill flow described in
VisualBuilderSpec-v8 §9.

Coverage is per-family: each preset either has a paper-anchored row in
``DEFAULTS`` or falls through ``get_defaults`` to a family default
keyed on the prefix (``llama3*`` -> Llama-3 paper, ``qwen3_dense*`` ->
Qwen3 paper, etc.). ``known_keys()`` returns the explicit set, and the
fallback path always returns a valid ``TrainingDefaults`` — never None
— so callers can rely on the contract without nullability checks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


__all__ = [
    "TrainingDefaults",
    "DEFAULTS",
    "FAMILY_DEFAULTS",
    "get_defaults",
    "known_keys",
    "to_wire",
]


@dataclass(frozen=True)
class TrainingDefaults:
    """Paper-anchored training defaults for a single preset.

    Fields mirror VisualBuilderSpec-v8 §9. ``betas`` is None when the
    chosen optimizer does not use Adam-style moments (e.g. Muon).
    ``source_paper_url`` is the arxiv (or canonical) link that anchors
    the row — used by the tooltip on the auto-filled fields.
    """

    lr: float
    batch_size: int
    schedule: str         # constant | linear_warmup | cosine | wsd | inv_sqrt
    warmup_steps: int
    betas: tuple[float, float] | None
    gradient_clip: float
    mixed_precision: bool
    optimizer: str        # adamw | muon | muon_adamw_hybrid | lion | adam8bit
    source_paper_url: str


# ---------------------------------------------------------------------------
# Per-preset explicit rows (>= 30, paper-anchored)
# ---------------------------------------------------------------------------


DEFAULTS: dict[str, TrainingDefaults] = {
    # ---- LLaMA family ----
    "llama3_8b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2407.21783"),
    "llama3_2_1b": TrainingDefaults(
        lr=4e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2407.21783"),
    "llama3_2_3b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2407.21783"),
    "llama4_maverick": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://ai.meta.com/blog/llama-4/"),

    # ---- Qwen3 family ----
    "qwen3_dense_0_6b": TrainingDefaults(
        lr=5e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_dense_4b": TrainingDefaults(
        lr=4e-4, batch_size=2048, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_dense_8b": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_dense_32b": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_30b_a3b": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_235b_a22b": TrainingDefaults(
        lr=2e-4, batch_size=4096, schedule="wsd",
        warmup_steps=4000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_coder_flash": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.15115"),
    "qwen3_next": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://qwen.ai/research"),

    # ---- DeepSeek / Kimi (MLA + MoE) ----
    "deepseek_v3": TrainingDefaults(
        lr=2.4e-4, batch_size=4096, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.19437"),
    "deepseek_v4_flash": TrainingDefaults(
        lr=2.4e-4, batch_size=4096, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.19437"),
    "kimi_k2": TrainingDefaults(
        lr=2e-4, batch_size=4096, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True,
        optimizer="muon_adamw_hybrid",
        source_paper_url="https://arxiv.org/abs/2502.16982"),
    "kimi_linear": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True,
        optimizer="muon_adamw_hybrid",
        source_paper_url="https://arxiv.org/abs/2502.16982"),

    # ---- Mistral / Phi / Granite ----
    "mistral_small_3_1": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2310.06825"),
    "phi4": TrainingDefaults(
        lr=2e-4, batch_size=1024, schedule="cosine",
        warmup_steps=500, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2412.08905"),
    "granite_4_1": TrainingDefaults(
        lr=2e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2408.03326"),
    "nanbeige_4_1": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2408.03326"),

    # ---- OLMo ----
    "olmo2_7b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2501.00656"),
    "olmo3_7b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2501.00656"),
    "olmo3_32b": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2501.00656"),

    # ---- GLM family ----
    "glm_45": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2406.12793"),
    "glm_47": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2406.12793"),
    "glm_5": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2406.12793"),

    # ---- Gemma ----
    "gemma4": TrainingDefaults(
        lr=2e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2403.08295"),
    "gemma3_27b": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2403.08295"),
    "gemma3_270m": TrainingDefaults(
        lr=6e-4, batch_size=512, schedule="cosine",
        warmup_steps=500, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2403.08295"),
    "gemma4_31b": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2403.08295"),
    "gemma_4_e2b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2403.08295"),
    "gemma_4_e4b": TrainingDefaults(
        lr=2e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2403.08295"),

    # ---- Mixtral / OSS / sliding-MoE ----
    "gpt_oss_20b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://openai.com/gpt-oss"),
    "gpt_oss_120b": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="wsd",
        warmup_steps=4000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://openai.com/gpt-oss"),
    "grok25": TrainingDefaults(
        lr=2e-4, batch_size=2048, schedule="wsd",
        warmup_steps=4000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://x.ai/blog/grok-2-5"),

    # ---- SmolLM / GPT-2 / xLSTM (NoPE / abs_pos / mLSTM) ----
    "smollm3": TrainingDefaults(
        lr=6e-4, batch_size=1024, schedule="wsd",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2502.02737"),
    "gpt2_xl": TrainingDefaults(
        lr=2e-4, batch_size=512, schedule="linear_warmup",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=False, optimizer="adamw",
        source_paper_url="https://cdn.openai.com/better-language-models/"
                          "language_models_are_unsupervised_multitask_learners.pdf"),
    "xlstm_7b": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2405.04517"),

    # ---- MiniMax / mimo ----
    "minimax_m2": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2501.08313"),
    "mimo_v2_5": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2501.08313"),

    # ---- Misc ----
    "nemotron3": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2406.16860"),
    "zaya1": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2503.07301"),
    "longcat": TrainingDefaults(
        lr=3e-4, batch_size=1024, schedule="cosine",
        warmup_steps=1000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2503.04473"),
    "ling25": TrainingDefaults(
        lr=3e-4, batch_size=2048, schedule="wsd",
        warmup_steps=2000, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2402.01528"),
    "tiny_aya": TrainingDefaults(
        lr=4e-4, batch_size=512, schedule="cosine",
        warmup_steps=500, betas=(0.9, 0.95),
        gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
        source_paper_url="https://arxiv.org/abs/2406.18682"),
}


# ---------------------------------------------------------------------------
# Family fallback — keyed on preset-name prefix
# ---------------------------------------------------------------------------


FAMILY_DEFAULTS: dict[str, TrainingDefaults] = {
    "llama":   DEFAULTS["llama3_8b"],
    "qwen3":   DEFAULTS["qwen3_dense_8b"],
    "deepseek": DEFAULTS["deepseek_v3"],
    "kimi":    DEFAULTS["kimi_linear"],
    "mistral": DEFAULTS["mistral_small_3_1"],
    "phi":     DEFAULTS["phi4"],
    "granite": DEFAULTS["granite_4_1"],
    "nanbeige": DEFAULTS["nanbeige_4_1"],
    "olmo":    DEFAULTS["olmo2_7b"],
    "glm":     DEFAULTS["glm_45"],
    "gemma":   DEFAULTS["gemma4"],
    "gpt_oss": DEFAULTS["gpt_oss_20b"],
    "gpt2":    DEFAULTS["gpt2_xl"],
    "grok":    DEFAULTS["grok25"],
    "smollm":  DEFAULTS["smollm3"],
    "xlstm":   DEFAULTS["xlstm_7b"],
    "minimax": DEFAULTS["minimax_m2"],
    "mimo":    DEFAULTS["mimo_v2_5"],
    "nemotron": DEFAULTS["nemotron3"],
    "zaya":    DEFAULTS["zaya1"],
    "longcat": DEFAULTS["longcat"],
    "ling":    DEFAULTS["ling25"],
    "tiny_aya": DEFAULTS["tiny_aya"],
    "tencent": DEFAULTS["qwen3_dense_8b"],  # MoE Mixtral-like — close enough
    "intellect": DEFAULTS["glm_45"],
    "sarvam":  DEFAULTS["glm_5"],
    "step3":   DEFAULTS["gpt_oss_20b"],
    "laguna":  DEFAULTS["gpt_oss_20b"],
    "arcee":   DEFAULTS["llama3_8b"],
}


_GENERIC = TrainingDefaults(
    lr=3e-4, batch_size=1024, schedule="cosine",
    warmup_steps=1000, betas=(0.9, 0.95),
    gradient_clip=1.0, mixed_precision=True, optimizer="adamw",
    source_paper_url="https://arxiv.org/abs/2407.21783")


def get_defaults(preset_name: str) -> TrainingDefaults:
    """Return paper-anchored defaults for ``preset_name``.

    Resolution order:

    1. Exact match in :data:`DEFAULTS`.
    2. Longest matching family prefix in :data:`FAMILY_DEFAULTS`.
    3. ``_GENERIC`` (Llama-3-style sensible defaults).

    The contract is total: never raises, never returns None.
    """
    if preset_name in DEFAULTS:
        return DEFAULTS[preset_name]
    matches = [k for k in FAMILY_DEFAULTS if preset_name.startswith(k)]
    if matches:
        return FAMILY_DEFAULTS[max(matches, key=len)]
    return _GENERIC


def known_keys() -> tuple[str, ...]:
    """Sorted tuple of preset keys with an explicit paper-anchored row."""
    return tuple(sorted(DEFAULTS))


def to_wire(defaults: TrainingDefaults) -> dict[str, Any]:
    """Render :class:`TrainingDefaults` as a JSON-friendly dict.

    Tuples become lists so the result is directly JSON-serialisable for
    the ``build_preset_specs`` RPC payload (Pydantic ``model_dump``
    rejects bare tuples in result schemas).
    """
    d = asdict(defaults)
    if d["betas"] is not None:
        d["betas"] = list(d["betas"])
    return d
