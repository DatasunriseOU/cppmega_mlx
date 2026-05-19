"""12 architecture presets — JSON spec lists for the LLM gallery.

Each entry produces ONE repeat-unit's worth of bricks. Replicate to
build a full model. Brick params are kept generic (small head/expert
counts) so unit tests can instantiate every preset in seconds; tune for
production via overrides on the returned spec list.

References:
  Qwen3-Next / 3.5 / 3.6  — 3:1 GDN + Gated Attention + MoE
  Kimi Linear 48B-A3B     — 3:1 KDA + MLA + MoE
  Kimi K2 / K2.5          — 100% MLA-absorb + MoE
  DeepSeek V3             — MLA + MoE
  DeepSeek V4 Flash       — hash-indexed sparse MLA + MoE
  Gemma 4 26B-A4B         — 5:1 sliding/global GQA + MoE (with QK-norm)
  Mistral Small 4 119B    — MLA-absorb + INT4-cache + 128 sparse MoE
  Ling 2.6                — 7:1 Bailing Linear + Bailing MLA + Bailing MoE
  LongCat                 — MLA + Bailing MoE (shortcut routing variant)
  Nemotron 3 Super        — Mamba-2 + GQA + MoE
  ZAYA1-8B                — CCA + 4:1 GQA + top-1 MoE
  Arcee Trinity Large     — 3:1 sliding/global gated GQA + MoE
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _qwen3_next(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    nkv = max(2, nh // 8)
    return [
        {"kind": "gdn", "name": f"qwen3_gdn_{i}"} for i in range(3)
    ] + [
        {"kind": "gated_attention", "name": "qwen3_attn",
         "params": {"num_attention_heads": nh, "num_key_value_heads": nkv,
                    "head_dim": 64}},
        {"kind": "moe", "name": "qwen3_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


def _mla_params(hidden_size: int) -> dict[str, Any]:
    """Defaults sized to ``hidden_size`` for our MLA brick (num_heads /
    LoRA ranks / head-dim splits). Matches ``_build_mla``'s setdefault
    rules so small hidden sizes (used in unit tests) work."""
    return {
        "num_heads": max(2, hidden_size // 32),
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "q_lora_rank": max(32, hidden_size // 2),
        "kv_lora_rank": max(16, hidden_size // 4),
    }


def _kimi_linear(hidden_size: int) -> list[dict[str, Any]]:
    return [
        {"kind": "kda", "name": f"kimi_kda_{i}"} for i in range(3)
    ] + [
        {"kind": "mla", "name": "kimi_mla", "params": _mla_params(hidden_size)},
        {"kind": "moe", "name": "kimi_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


def _kimi_k2(hidden_size: int) -> list[dict[str, Any]]:
    return [
        {"kind": "mla_absorb", "name": "k2_mla", "params": _mla_params(hidden_size)},
        {"kind": "moe", "name": "k2_moe",
         "params": {"num_experts": 8, "top_k": 2}},
    ]


def _deepseek_v3(hidden_size: int) -> list[dict[str, Any]]:
    return [
        {"kind": "mla", "name": "dsv3_mla", "params": _mla_params(hidden_size)},
        {"kind": "moe", "name": "dsv3_moe",
         "params": {"num_experts": 6, "top_k": 2}},
    ]


def _deepseek_v4_flash(hidden_size: int) -> list[dict[str, Any]]:
    return [
        {"kind": "dsv4_attention", "name": "dsv4_attn"},
        {"kind": "moe", "name": "dsv4_moe",
         "params": {"num_experts": 6, "top_k": 2}},
    ]


def _gemma4(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    nkv = max(2, nh // 8)
    sliding_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                      "head_dim": 64, "sliding_window_size": 1024}
    global_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                     "head_dim": 64}
    return [
        {"kind": "gqa_sliding", "name": f"gemma_sw_{i}",
         "params": dict(sliding_params)} for i in range(5)
    ] + [
        {"kind": "gated_attention", "name": "gemma_global",
         "params": dict(global_params)},
        {"kind": "moe", "name": "gemma_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


def _mistral4(hidden_size: int) -> list[dict[str, Any]]:
    return [
        {"kind": "mistral4_mla", "name": "mistral4_attn"},
        {"kind": "moe", "name": "mistral4_moe",
         "params": {"num_experts": 8, "top_k": 2}},
    ]


def _ling26(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    nkv = max(2, nh // 8)
    return [
        {"kind": "bailing_linear", "name": f"ling_la_{i}",
         "params": {"num_attention_heads": nh, "num_key_value_heads": nkv,
                    "head_dim": 64}}
        for i in range(7)
    ] + [
        {"kind": "bailing_mla", "name": "ling_mla",
         "params": {"num_attention_heads": nh, "num_key_value_heads": nkv,
                    "head_dim": 64, "kv_lora_rank": 32,
                    "qk_rope_head_dim": 16, "qk_nope_head_dim": 32,
                    "v_head_dim": 32}},
        {"kind": "bailing_moe", "name": "ling_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


def _longcat(hidden_size: int) -> list[dict[str, Any]]:
    return [
        {"kind": "mla", "name": "longcat_mla", "params": _mla_params(hidden_size)},
        {"kind": "bailing_moe", "name": "longcat_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


def _nemotron3(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    return [
        {"kind": "mamba3", "name": "nemo_mamba"},
        {"kind": "attention", "name": "nemo_attn",
         "params": {"num_heads": nh, "head_dim": 64}},
        {"kind": "moe", "name": "nemo_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


def _zaya1(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    nkv = max(2, nh // 8)
    cca_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                  "head_dim": 64, "fine_window": 64, "coarse_block_size": 8}
    gqa_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                  "head_dim": 64}
    return [
        {"kind": "cca_attention", "name": "zaya_cca",
         "params": dict(cca_params)},
    ] + [
        {"kind": "gated_attention", "name": f"zaya_gqa_{i}",
         "params": dict(gqa_params)} for i in range(4)
    ] + [
        {"kind": "moe", "name": "zaya_moe",
         "params": {"num_experts": 4, "top_k": 1}},
    ]


def _arcee_trinity(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    nkv = max(2, nh // 8)
    sliding_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                      "head_dim": 64, "sliding_window_size": 1024}
    global_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                     "head_dim": 64}
    return [
        {"kind": "gqa_sliding", "name": f"arcee_sw_{i}",
         "params": dict(sliding_params)} for i in range(3)
    ] + [
        {"kind": "gated_attention", "name": "arcee_global",
         "params": dict(global_params)},
        {"kind": "moe", "name": "arcee_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


PRESETS: dict[str, Callable[[int], list[dict[str, Any]]]] = {
    "qwen3_next": _qwen3_next,
    "kimi_linear": _kimi_linear,
    "kimi_k2": _kimi_k2,
    "deepseek_v3": _deepseek_v3,
    "deepseek_v4_flash": _deepseek_v4_flash,
    "gemma4": _gemma4,
    "mistral4": _mistral4,
    "ling26": _ling26,
    "longcat": _longcat,
    "nemotron3": _nemotron3,
    "zaya1": _zaya1,
    "arcee_trinity": _arcee_trinity,
}


def available_presets() -> list[str]:
    return sorted(PRESETS.keys())


def build_preset_specs(
    name: str,
    hidden_size: int,
    *,
    num_layers: int | None = None,
) -> list[dict[str, Any]]:
    """Return the spec list for ``name``, optionally replicated.

    ``num_layers`` is interpreted as the **total** number of brick specs
    desired. If it is not a multiple of the preset's repeat-unit size,
    the result is truncated to the largest multiple ≤ ``num_layers``.
    When omitted, returns a single repeat-unit.

    Spec names are made unique across repetitions by appending a
    ``_rep{i}`` suffix on every repeat after the first.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available={available_presets()}")
    unit = PRESETS[name](hidden_size)
    if not unit:
        return []
    if num_layers is None:
        return [dict(s) for s in unit]
    if num_layers < 0:
        raise ValueError("num_layers must be ≥ 0")
    n_reps = num_layers // len(unit)
    out: list[dict[str, Any]] = []
    for r in range(n_reps):
        for s in unit:
            spec = dict(s)
            if r > 0:
                spec["name"] = f"{spec['name']}_rep{r}"
            spec["params"] = dict(spec.get("params") or {})
            out.append(spec)
    return out


__all__ = [
    "PRESETS",
    "available_presets",
    "build_preset_specs",
]
