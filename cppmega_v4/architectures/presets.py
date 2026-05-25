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
    """DeepSeek V4 Flash — full mini-model mirroring Raschka's diagram.

    Architecture (per https://sebastianraschka.com/llm-architecture-gallery
    + DeepSeek V4 Flash paper):
      input → abs_pos_embed (positional residual)
            → 3 × backbone layer:
                 lightning_indexer (DSA top-k key selector, ratio 1/128)
                 csa_hca (compressed self+history cross-attn, ratio 1/4)
                 engram (long-term memory residual)
                 dsv4_attention (hash-indexed sparse MLA)
                 rmsnorm (post-norm)
                 moe (sparse MoE FFN)
            → rmsnorm (final norm)
            → nemotron_h_mtp (multi-token prediction drafter head)

    Compression contract — DSA uses:
      • top_k = S/128 (lightning_indexer cuts key set 128× per query)
      • kv_lora_rank = H/4 (MLA LoRA compresses KV)
      • engram memory slots = 256 (long-term retention)

    Pair with mtp_weighted loss for the drafter head (k=2, beta=0.5).
    """
    hd = 64
    if hidden_size % hd != 0:
        if hidden_size % 32 == 0:
            hd = 32
        elif hidden_size % 16 == 0:
            hd = 16
        else:
            hd = hidden_size // 2
    nh = hidden_size // hd
    # DSA compression knobs — 1/128 sparse keys, 1/4 KV LoRA.
    top_k = max(4, hidden_size // 128)
    kv_lora_rank = max(16, hidden_size // 4)

    def _layer(i: int) -> list[dict[str, Any]]:
        return [
            {"kind": "lightning_indexer", "name": f"dsv4_idx_{i}",
             "params": {"top_k": top_k, "index_dim": 32}},
            {"kind": "csa_hca", "name": f"dsv4_csa_hca_{i}",
             "params": {"num_heads": nh, "head_dim": hd}},
            {"kind": "engram", "name": f"dsv4_engram_{i}",
             # V4 Engram block uses ngram-memory semantics (n=4 default,
             # 256-entry embed table). Use defaults — block is doc_id
             # aware; capacity tuning is a separate ticket.
             "params": {}},
            {"kind": "dsv4_attention", "name": f"dsv4_attn_{i}",
             "params": {"num_heads": nh, "head_dim": hd,
                        "kv_lora_rank": kv_lora_rank,
                        "top_k": top_k}},
            {"kind": "rmsnorm", "name": f"dsv4_post_norm_{i}"},
            {"kind": "moe", "name": f"dsv4_moe_{i}",
             "params": {"num_experts": 6, "top_k": 2,
                        "capacity_factor": 1.25}},
        ]

    layers: list[dict[str, Any]] = []
    for i in range(3):
        layers.extend(_layer(i))

    return (
        # Input: learned absolute positional residual.
        [{"kind": "abs_pos_embed", "name": "dsv4_pos",
          "params": {"max_position_embeddings": 4096}}]
        + layers
        + [
            {"kind": "rmsnorm", "name": "dsv4_final_norm"},
            # MTP drafter head — produces k=2 lookahead tokens.
            {"kind": "nemotron_h_mtp", "name": "dsv4_mtp",
             "params": {"drafter_k": 2}},
        ]
    )


def _gemma4(hidden_size: int) -> list[dict[str, Any]]:
    nh = max(8, hidden_size // 64)
    nkv = max(2, nh // 8)
    sliding_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                      "head_dim": 64, "sliding_window_size": 1024}
    global_params = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                     "head_dim": 64}
    return [
        {"kind": "per_layer_embed", "name": "gemma4_ple",
         "params": {"layer_index": 0, "num_layers": 26}},
    ] + [
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
        {"kind": "rmsnorm", "name": "nemo_norm1"},
        {"kind": "attention", "name": "nemo_attn",
         "params": {"num_heads": nh, "head_dim": 64}},
        {"kind": "rmsnorm", "name": "nemo_norm2"},
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
# Gallery coverage additions — composition-only, no new bricks
# ---------------------------------------------------------------------------


def _attn_params(hidden_size: int) -> dict[str, Any]:
    """Generic dense GQA attention params keyed off hidden_size."""
    return {"num_heads": max(2, hidden_size // 32), "head_dim": 64}


def _llama_like(name_prefix: str, depth: int = 1) -> Callable[[int], list[dict[str, Any]]]:
    """LLaMA-style dense decoder repeat-unit (attention + mlp).

    Used for llama3 family, mistral_small_3_1, nanbeige_4_1, phi4,
    granite_4_1, qwen3 dense, gemma3 dense, smollm3, olmo family.
    """
    def _factory(hidden_size: int) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for i in range(depth):
            suffix = f"_{i}" if depth > 1 else ""
            units.append({"kind": "attention", "name": f"{name_prefix}_attn{suffix}",
                          "params": _attn_params(hidden_size)})
            units.append({"kind": "mlp", "name": f"{name_prefix}_mlp{suffix}"})
        return units
    return _factory


def _mixtral_like(name_prefix: str, num_experts: int = 8, top_k: int = 2
                  ) -> Callable[[int], list[dict[str, Any]]]:
    """Mixtral-style attention + MoE (Llama4 Maverick, Qwen3 235B/30B-A3B,
    Grok 2.5, MiMo, Tencent Hy3, etc.)."""
    def _factory(hidden_size: int) -> list[dict[str, Any]]:
        return [
            {"kind": "attention", "name": f"{name_prefix}_attn",
             "params": _attn_params(hidden_size)},
            {"kind": "moe", "name": f"{name_prefix}_moe",
             "params": {"num_experts": num_experts, "top_k": top_k}},
        ]
    return _factory


def _glm_like(name_prefix: str, num_experts: int = 8, top_k: int = 2
              ) -> Callable[[int], list[dict[str, Any]]]:
    """GLM-style attention + MoE + shared MLP (acts as shared expert).

    We model the shared expert as an extra mlp brick after the MoE.
    GLM-4.5 / GLM-4.7 / Sarvam-30B / GLM-4.5-Air / INTELLECT-3.
    """
    def _factory(hidden_size: int) -> list[dict[str, Any]]:
        return [
            {"kind": "attention", "name": f"{name_prefix}_attn",
             "params": _attn_params(hidden_size)},
            {"kind": "moe", "name": f"{name_prefix}_moe",
             "params": {"num_experts": num_experts, "top_k": top_k}},
            {"kind": "mlp", "name": f"{name_prefix}_shared"},
        ]
    return _factory


def _mla_moe_like(name_prefix: str, num_experts: int = 8, top_k: int = 2
                  ) -> Callable[[int], list[dict[str, Any]]]:
    """DeepSeek-style MLA + MoE (GLM-5 / GLM-5.1 / Sarvam-105B)."""
    def _factory(hidden_size: int) -> list[dict[str, Any]]:
        return [
            {"kind": "mla", "name": f"{name_prefix}_mla",
             "params": _mla_params(hidden_size)},
            {"kind": "moe", "name": f"{name_prefix}_moe",
             "params": {"num_experts": num_experts, "top_k": top_k}},
        ]
    return _factory


def _sliding_global_moe(name_prefix: str, sliding_count: int = 5,
                        num_experts: int = 8, top_k: int = 2
                        ) -> Callable[[int], list[dict[str, Any]]]:
    """Sliding+global+MoE pattern (GPT-OSS / MiniMax M2 / MiMo-V2-Flash /
    Step-3.5 Flash / Tiny Aya / Ling-2.5 / Laguna XS.2)."""
    def _factory(hidden_size: int) -> list[dict[str, Any]]:
        nh = max(8, hidden_size // 64)
        nkv = max(2, nh // 8)
        sliding = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                   "head_dim": 64, "sliding_window_size": 1024}
        glob = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                "head_dim": 64}
        units = [
            {"kind": "gqa_sliding", "name": f"{name_prefix}_sw_{i}",
             "params": dict(sliding)} for i in range(sliding_count)
        ]
        units.append({"kind": "gated_attention", "name": f"{name_prefix}_glob",
                      "params": dict(glob)})
        units.append({"kind": "moe", "name": f"{name_prefix}_moe",
                      "params": {"num_experts": num_experts, "top_k": top_k}})
        return units
    return _factory


def _gemma3_dense(name_prefix: str, sliding_count: int = 5
                  ) -> Callable[[int], list[dict[str, Any]]]:
    """Gemma 3 / Gemma 4 dense — 5:1 sliding/global GQA + QK-norm + dense
    MLP. We approximate with gqa_sliding + gated_attention + mlp."""
    def _factory(hidden_size: int) -> list[dict[str, Any]]:
        nh = max(8, hidden_size // 64)
        nkv = max(2, nh // 8)
        sliding = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                   "head_dim": 64, "sliding_window_size": 1024}
        glob = {"num_attention_heads": nh, "num_key_value_heads": nkv,
                "head_dim": 64}
        units = [
            {"kind": "gqa_sliding", "name": f"{name_prefix}_sw_{i}",
             "params": dict(sliding)} for i in range(sliding_count)
        ]
        units.append({"kind": "gated_attention", "name": f"{name_prefix}_glob",
                      "params": dict(glob)})
        units.append({"kind": "mlp", "name": f"{name_prefix}_mlp"})
        return units
    return _factory


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


PRESETS: dict[str, Callable[[int], list[dict[str, Any]]]] = {
    # original 12 (shipped earlier)
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
    # ---- LLaMA family (#2, #3, #38) ----
    "llama3_8b":          _llama_like("llama3_8b"),
    "llama3_2_1b":        _llama_like("llama3_2_1b"),
    "llama3_2_3b":        _llama_like("llama3_2_3b"),
    # ---- OLMo family (#4, #26, #27) — QK-norm semantically; brick params identical ----
    "olmo2_7b":           _llama_like("olmo2_7b"),
    "olmo3_7b":           _gemma3_dense("olmo3_7b", sliding_count=3),
    "olmo3_32b":          _gemma3_dense("olmo3_32b", sliding_count=3),
    # ---- Mistral / Phi / Granite / Nanbeige dense (#8, #42, #49, #71) ----
    "mistral_small_3_1":  _llama_like("mistral_small_3_1"),
    "nanbeige_4_1":       _llama_like("nanbeige_4_1"),
    "phi4":               _llama_like("phi4"),
    "granite_4_1":        _llama_like("granite_4_1"),
    # ---- Qwen3 dense family (#10, #13, #14, #15, #66) ----
    "qwen3_dense_0_6b":   _llama_like("qwen3_0_6b"),
    "qwen3_dense_4b":     _llama_like("qwen3_4b"),
    "qwen3_dense_8b":     _llama_like("qwen3_8b"),
    "qwen3_dense_32b":    _llama_like("qwen3_32b"),
    "qwen3_6_27b":        _llama_like("qwen3_6_27b"),
    # ---- SmolLM3 (#16) — periodic NoPE is a per-layer toggle; modelled as plain attn ----
    "smollm3":            _llama_like("smollm3"),
    # ---- Mixtral-style attention + MoE (#9, #11, #12, #22, #39, #43, #56, #67, #68, #70) ----
    "llama4_maverick":    _mixtral_like("llama4_maverick", num_experts=8, top_k=2),
    "qwen3_235b_a22b":    _mixtral_like("qwen3_235b", num_experts=8, top_k=2),
    "qwen3_30b_a3b":      _mixtral_like("qwen3_30b", num_experts=4, top_k=2),
    "grok25":             _mixtral_like("grok25", num_experts=8, top_k=2),
    "qwen3_coder_flash":  _mixtral_like("qwen3_coder_flash", num_experts=4, top_k=2),
    "minimax_m2_5":       _mixtral_like("minimax_m2_5", num_experts=8, top_k=2),
    "minimax_m2_7":       _mixtral_like("minimax_m2_7", num_experts=8, top_k=2),
    "mimo_v2_5":          _mixtral_like("mimo_v2_5", num_experts=8, top_k=2),
    "mimo_v2_5_pro":      _mixtral_like("mimo_v2_5_pro", num_experts=8, top_k=2),
    "tencent_hy3":        _mixtral_like("tencent_hy3", num_experts=8, top_k=2),
    # ---- GLM-style (+ shared expert) (#18, #32, #47, #51, #52) ----
    "glm_45":             _glm_like("glm_45", num_experts=8, top_k=2),
    "glm_47":             _glm_like("glm_47", num_experts=8, top_k=2),
    "glm_45_air":         _glm_like("glm_45_air", num_experts=8, top_k=2),
    "intellect_3":        _glm_like("intellect_3", num_experts=8, top_k=2),
    "sarvam_30b":         _glm_like("sarvam_30b", num_experts=8, top_k=2),
    # ---- MLA + MoE DeepSeek-style (#34, #48, #63) ----
    "glm_5":              _mla_moe_like("glm_5", num_experts=8, top_k=2),
    "glm_51":             _mla_moe_like("glm_51", num_experts=8, top_k=2),
    "sarvam_105b":        _mla_moe_like("sarvam_105b", num_experts=8, top_k=2),
    # ---- Sliding+global+MoE (#19, #20, #24, #31, #41, #44, #45, #61) ----
    "gpt_oss_120b":       _sliding_global_moe("gpt_oss_120b", sliding_count=3, num_experts=8, top_k=2),
    "gpt_oss_20b":        _sliding_global_moe("gpt_oss_20b", sliding_count=3, num_experts=8, top_k=2),
    "minimax_m2":         _sliding_global_moe("minimax_m2", sliding_count=3, num_experts=8, top_k=2),
    "mimo_v2_flash":      _sliding_global_moe("mimo_v2_flash", sliding_count=5, num_experts=8, top_k=2),
    "step3_5_flash":      _sliding_global_moe("step3_5_flash", sliding_count=3, num_experts=8, top_k=2),
    "tiny_aya":           _sliding_global_moe("tiny_aya", sliding_count=3, num_experts=4, top_k=2),
    "ling25":             _sliding_global_moe("ling25", sliding_count=3, num_experts=8, top_k=2),
    "laguna_xs2":         _sliding_global_moe("laguna_xs2", sliding_count=3, num_experts=8, top_k=2),
    # ---- Gemma 3 dense (#7, #21, #36) ----
    "gemma3_27b":         _gemma3_dense("gemma3_27b", sliding_count=5),
    "gemma3_270m":        _gemma3_dense("gemma3_270m", sliding_count=5),
    "gemma4_31b":         _gemma3_dense("gemma4_31b", sliding_count=5),
}


# ---------------------------------------------------------------------------
# GalCov-C — parallel-block showcase specs (not in PRESETS to keep all
# parametrized tests linear-only; consumed directly via from_block_specs).
# ---------------------------------------------------------------------------


def tiny_aya_parallel_specs(hidden_size: int) -> list[dict[str, Any]]:
    """Tiny Aya-style parallel GQA + MLP between pre and post bricks.

    Demonstrates the GalCov-C parallel-block DSL. See gallery entry #44.
    """
    return [
        {"kind": "attention", "name": "tap_pre",
         "params": _attn_params(hidden_size)},
        {"parallel": [
            {"kind": "gated_attention", "name": "tap_gqa",
             "params": {"num_attention_heads": max(8, hidden_size // 64),
                        "num_key_value_heads": max(2, hidden_size // 64 // 8),
                        "head_dim": 64}},
            {"kind": "mlp", "name": "tap_mlp"},
        ]},
        {"kind": "moe", "name": "tap_moe",
         "params": {"num_experts": 4, "top_k": 2}},
    ]


# ---------------------------------------------------------------------------
# V7-Q04: gallery gap closures (raschka entries #1, #50, #57, #58).
# Each preset uses an already-existing brick kind from BLOCK_BUILDERS;
# the missing piece was the factory wiring into PRESETS.
# ---------------------------------------------------------------------------


def _gpt2_xl(hidden_size: int) -> list[dict[str, Any]]:
    """Gallery #1 GPT-2 XL: abs_pos_embed + attention + mlp.

    The abs_pos_embed brick adds a learned absolute positional residual
    before the attention stack; remainder is the standard transformer
    block. Matches Karpathy/GPT-2 architecture pre-RoPE.
    """
    return [
        {"kind": "abs_pos_embed", "name": "gpt2_xl_pos",
         "params": {"max_position_embeddings": 1024}},
        {"kind": "attention", "name": "gpt2_xl_attn",
         "params": _attn_params(hidden_size)},
        {"kind": "mlp", "name": "gpt2_xl_mlp"},
    ]


def _xlstm_7b(hidden_size: int) -> list[dict[str, Any]]:
    """Gallery #50 xLSTM 7B: matrix-LSTM stack, no self-attention.

    Hybrid of mlstm blocks (matrix-memory recurrence) + interspersed
    mlp blocks for FFN. Demonstrates non-attention sequence path.
    """
    return [
        {"kind": "mlstm", "name": "xlstm_7b_mlstm",
         "params": {"head_dim": 64}},
        {"kind": "mlp", "name": "xlstm_7b_mlp"},
    ]


def _gemma_4_e2b(hidden_size: int) -> list[dict[str, Any]]:
    """Gallery #57 Gemma 4 E2B: per-layer scaled embedding + GQA + MLP.

    Gemma 4 E2B/E4B add a per-layer embedding residual that scales
    inversely with depth. The layer_index/num_layers in params control
    which layer's scaling factor applies.
    """
    return [
        {"kind": "per_layer_embed", "name": "gemma4_e2b_ple",
         "params": {"layer_index": 0, "num_layers": 26}},
        {"kind": "gated_attention", "name": "gemma4_e2b_gqa",
         "params": {"num_attention_heads": max(8, hidden_size // 64),
                    "num_key_value_heads": max(2, hidden_size // 64 // 8),
                    "head_dim": 64}},
        {"kind": "mlp", "name": "gemma4_e2b_mlp"},
    ]


def _gemma_4_e4b(hidden_size: int) -> list[dict[str, Any]]:
    """Gallery #58 Gemma 4 E4B: deeper variant of E2B (44 layers).

    Same per_layer_embed + GQA + MLP recipe, more layers.
    """
    return [
        {"kind": "per_layer_embed", "name": "gemma4_e4b_ple",
         "params": {"layer_index": 0, "num_layers": 44}},
        {"kind": "gated_attention", "name": "gemma4_e4b_gqa",
         "params": {"num_attention_heads": max(8, hidden_size // 64),
                    "num_key_value_heads": max(2, hidden_size // 64 // 8),
                    "head_dim": 64}},
        {"kind": "mlp", "name": "gemma4_e4b_mlp"},
    ]


# V7-Q04 — gallery gap closures wired after factory definitions to
# avoid forward references in the PRESETS dict literal.
PRESETS["gpt2_xl"]           = _gpt2_xl              # gallery #1
PRESETS["tiny_aya_parallel"] = tiny_aya_parallel_specs  # gallery #44
PRESETS["xlstm_7b"]          = _xlstm_7b             # gallery #50
PRESETS["gemma_4_e2b"]       = _gemma_4_e2b          # gallery #57
PRESETS["gemma_4_e4b"]       = _gemma_4_e4b          # gallery #58


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
    if num_layers == 0:
        return []
    n_reps = max(1, num_layers // len(unit))
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
