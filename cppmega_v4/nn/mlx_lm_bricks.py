"""Direct re-exports of new model bricks from open mlx-lm PRs.

Vision: the V4 block-builder GUI composes models from heterogeneous
"bricks" (attention / MoE / cache variants). The kernel-fusion and
cross-brick optimization layer (separate work) operates on the resulting
nn.Module graph. To keep brick variety high, we vendor open mlx-lm PRs
into our editable mlx-lm checkout at ``/Users/dave/sources/mlx-lm`` and
re-export the new bricks here as V4-shaped nn.Modules.

PRs currently integrated (branch ``cppmega-integration`` in our mlx-lm
fork, each as a separate commit on top of upstream main df1d3f3):

  #1037 Mistral Small 4 — Absorbed MLA + INT4 latent cache
        → ``mistral4_mla`` brick (mlx_lm.models.mistral4.MLAAttention)
  #1057 LongCat Next                — full model class
        → loadable directly via mlx_lm.load; no per-brick wrapper
  #1227 Bailing / Ling-2.6-flash    — LinearAttention + MultiLatentAttention + MoE
        → ``bailing_linear`` / ``bailing_mla`` / ``bailing_moe`` bricks
  #1201 DeepSeek-V4 (Flash)         — attention + block + index/hash heads
        → ``dsv4_attention`` brick (mlx_lm.models.deepseek_v4.Attention)

Per the user's instruction: import directly from mlx-lm, do not vendor or
re-implement. Each wrapper constructs the minimal ModelArgs that the
upstream module's __init__ reads, instantiates the upstream class, and
exposes ``self.inner`` so weight loaders that walk upstream parameter
paths find them at the expected locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Direct upstream imports (no vendoring)
# ---------------------------------------------------------------------------

from mlx_lm.models.mistral4 import (
    MLAAttention as _Mistral4MLA,
    ModelArgs as _Mistral4Args,
)
from mlx_lm.models.deepseek_v4 import (
    Attention as _DSv4Attention,
    ModelArgs as _DSv4Args,
)
from mlx_lm.models.bailing_hybrid import (
    LinearAttention as _BailingLinear,
    MultiLatentAttention as _BailingMLA,
    SparseMoeBlock as _BailingMoE,
    ModelArgs as _BailingArgs,
)
from mlx_lm.models.turbo_cache import TurboQuantKVCache as _TurboQuantKVCache
from mlx_lm.models.gemma4_assistant import (
    AssistantDecoderLayer as _Gemma4AssistantLayer,
    ModelArgs as _Gemma4AssistantArgs,
)
from mlx_lm.models.nemotron_h import (
    NemotronHMTPBlock as _NemotronHMTPBlock,
    ModelArgs as _NemotronHArgs,
)


# ---------------------------------------------------------------------------
# Mistral Small 4 — Absorbed MLA + INT4 latent cache  (PR #1037)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mistral4MLAConfig:
    hidden_size: int
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 131_072


class Mistral4MLABlock(nn.Module):
    """Mistral Small 4 MLA — absorbed fast-path + INT4 quantised latent cache."""

    def __init__(self, cfg: Mistral4MLAConfig):
        super().__init__()
        self.cfg = cfg
        args = _Mistral4Args(
            model_type="mistral4",
            hidden_size=cfg.hidden_size,
            num_hidden_layers=1,
            intermediate_size=cfg.hidden_size * 4,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            vocab_size=1,
            max_position_embeddings=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
        )
        self.inner = _Mistral4MLA(args)

    def __call__(self, x, mask=None, cache=None):
        return self.inner(x, mask=mask, cache=cache)


# ---------------------------------------------------------------------------
# DeepSeek V4 (Flash) — Attention with index/hash routing  (PR #1201)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSv4AttentionConfig:
    hidden_size: int
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    head_dim: int = 128
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    o_groups: int = 4
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_topk: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0


class DSv4AttentionBlock(nn.Module):
    """DeepSeek V4 (Flash) attention with hash-indexed sparse routing."""

    def __init__(self, cfg: DSv4AttentionConfig):
        super().__init__()
        self.cfg = cfg
        args = _DSv4Args(
            model_type="deepseek_v4",
            vocab_size=1,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=1,
            num_hash_layers=0,
            num_nextn_predict_layers=0,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            q_lora_rank=cfg.q_lora_rank,
            o_lora_rank=cfg.o_lora_rank,
            head_dim=cfg.head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            o_groups=cfg.o_groups,
            index_n_heads=cfg.index_n_heads,
            index_head_dim=cfg.index_head_dim,
            index_topk=cfg.index_topk,
            n_routed_experts=1,
            n_shared_experts=0,
            num_experts_per_tok=1,
            moe_intermediate_size=cfg.hidden_size,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            swiglu_limit=7.0,
            norm_topk_prob=False,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            compress_ratios=[],
        )
        self.inner = _DSv4Attention(0, args)

    def __call__(self, x, mask=None, cache=None):
        return self.inner(x, mask=mask, cache=cache)


# ---------------------------------------------------------------------------
# Bailing / Ling-2.6-flash  (PR #1227)
# ---------------------------------------------------------------------------


def _bailing_args(
    *,
    hidden_size: int,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    num_shared_experts: int = 1,
    n_group: int = 1,
    topk_group: int = 1,
    moe_intermediate_size: int = 1024,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    qk_nope_head_dim: int = 128,
    v_head_dim: int = 128,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 1_000_000.0,
) -> _BailingArgs:
    return _BailingArgs(
        model_type="bailing_hybrid",
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        moe_intermediate_size=moe_intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_shared_experts=num_shared_experts,
        n_group=n_group,
        topk_group=topk_group,
        first_k_dense_replace=0,
        layer_group_size=1,
        group_norm_size=1,
        vocab_size=1,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=131_072,
        routed_scaling_factor=1.0,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )


@dataclass(frozen=True)
class BailingLinearConfig:
    hidden_size: int
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128


class BailingLinearAttnBlock(nn.Module):
    """Ling-2.6-flash linear-attention brick (per-channel gate)."""

    def __init__(self, cfg: BailingLinearConfig):
        super().__init__()
        self.cfg = cfg
        args = _bailing_args(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
        )
        self.inner = _BailingLinear(args, layer_idx=0)

    def __call__(self, x, mask=None, cache=None):
        return self.inner(x, mask=mask, cache=cache)


@dataclass(frozen=True)
class BailingMLAConfig:
    hidden_size: int
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128


class BailingMLABlock(nn.Module):
    """Ling-2.6-flash multi-latent attention brick."""

    def __init__(self, cfg: BailingMLAConfig):
        super().__init__()
        self.cfg = cfg
        args = _bailing_args(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            v_head_dim=cfg.v_head_dim,
        )
        self.inner = _BailingMLA(args)

    def __call__(self, x, mask=None, cache=None):
        return self.inner(x, mask=mask, cache=cache)


@dataclass(frozen=True)
class BailingMoEConfig:
    hidden_size: int
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1024
    n_group: int = 1
    topk_group: int = 1


class BailingMoEBlock(nn.Module):
    """Ling-2.6-flash sparse MoE brick (DeepSeek-V3-style routing)."""

    def __init__(self, cfg: BailingMoEConfig):
        super().__init__()
        self.cfg = cfg
        args = _bailing_args(
            hidden_size=cfg.hidden_size,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            num_shared_experts=cfg.num_shared_experts,
            moe_intermediate_size=cfg.moe_intermediate_size,
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
        )
        self.inner = _BailingMoE(args)

    def __call__(self, x):
        return self.inner(x)


# ---------------------------------------------------------------------------
# TurboQuantKVCache  (PR #1202)  — cross-cutting cache utility
# ---------------------------------------------------------------------------


def make_turbo_quant_kv_cache(bits: int = 3) -> _TurboQuantKVCache:
    """3-bit / 4-bit KV cache compression for generation.

    Drop-in replacement for the standard KVCache in any attention block
    at decode time. Direct construction; no V4 nn.Module wrapper because
    caches are passed as plain ``cache=...`` arguments to attention calls.
    """
    return _TurboQuantKVCache(bits=bits)


# ---------------------------------------------------------------------------
# Gemma 4 Assistant (MTP drafter)  (PR #1276)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemma4DrafterLayerConfig:
    hidden_size: int
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 128
    intermediate_size: Optional[int] = None
    layer_type: str = "sliding_attention"  # or "full_attention"


class Gemma4DrafterLayerBlock(nn.Module):
    """Gemma 4 assistant (MTP drafter) decoder layer.

    This is a CROSS-ATTENTION layer: the drafter attends to the
    backbone's already-computed K/V cache. ``__call__`` therefore takes
    ``(x, keys, values)`` rather than the usual ``(x, mask, cache)``;
    weight loaders that walk Gemma 4's MTP drafter checkpoint will find
    the upstream parameter paths on ``self.inner.*``.
    """

    def __init__(self, cfg: Gemma4DrafterLayerConfig):
        super().__init__()
        self.cfg = cfg
        text_config = dict(
            model_type="gemma4_assistant_text",
            hidden_size=cfg.hidden_size,
            num_hidden_layers=1,
            intermediate_size=cfg.intermediate_size or cfg.hidden_size * 4,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            vocab_size=1,
            rms_norm_eps=1e-6,
            sliding_window=512,
            rope_theta=10_000.0,
            rope_local_base_freq=10_000.0,
        )
        args = _Gemma4AssistantArgs(
            model_type="gemma4_assistant",
            backbone_hidden_size=cfg.hidden_size,
            num_centroids=1,
            centroid_intermediate_top_k=1,
            use_ordered_embeddings=False,
            tie_word_embeddings=False,
            text_config=text_config,
            vocab_size=1,
        )
        self.inner = _Gemma4AssistantLayer(args, layer_type=cfg.layer_type)

    def __call__(self, x, keys, values, position_ids=None, mask=None):
        return self.inner(x, keys, values, position_ids=position_ids, mask=mask)


# ---------------------------------------------------------------------------
# Nemotron-H MTP block  (PR #1161)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NemotronHMTPConfig:
    hidden_size: int
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: Optional[int] = None
    # Nemotron-H uses single-char block tags matching the upstream
    # ``hybrid_override_pattern`` schema:
    #   "*" = attention   "E" = MoE   "-" = MLP   "M" = Mamba SSM
    block_type: str = "*"


class NemotronHMTPBlockWrapper(nn.Module):
    """Nemotron-H Multi-Token-Prediction block."""

    def __init__(self, cfg: NemotronHMTPConfig):
        super().__init__()
        self.cfg = cfg
        args = _NemotronHArgs(
            model_type="nemotron_h",
            vocab_size=1,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size or cfg.hidden_size * 4,
            num_hidden_layers=1,
            max_position_embeddings=131_072,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            attention_bias=False,
            mamba_num_heads=cfg.num_attention_heads,
            mamba_head_dim=cfg.head_dim,
            mamba_proj_bias=False,
            ssm_state_size=128,
            conv_kernel=4,
            n_groups=1,
            mlp_bias=False,
            layer_norm_epsilon=1e-5,
            use_bias=False,
            use_conv_bias=True,
            hybrid_override_pattern=cfg.block_type,
            layers_block_type=[cfg.block_type],
            head_dim=cfg.head_dim,
        )
        self.inner = _NemotronHMTPBlock(args, block_type=cfg.block_type)

    def __call__(self, x, mask=None, cache=None):
        return self.inner(x, mask=mask, cache=cache)


__all__ = [
    "BailingLinearAttnBlock",
    "BailingLinearConfig",
    "BailingMLABlock",
    "BailingMLAConfig",
    "BailingMoEBlock",
    "BailingMoEConfig",
    "DSv4AttentionBlock",
    "DSv4AttentionConfig",
    "Gemma4DrafterLayerBlock",
    "Gemma4DrafterLayerConfig",
    "Mistral4MLABlock",
    "Mistral4MLAConfig",
    "NemotronHMTPBlockWrapper",
    "NemotronHMTPConfig",
    "make_turbo_quant_kv_cache",
]
