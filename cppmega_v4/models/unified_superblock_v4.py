"""UnifiedSuperblockV4 — composes V4 blocks per a RunTemplate.

Closes the loop from the v4 plugin: build a heterogeneous stack of V4
blocks declaratively from a template, run forward on (token_ids,
hidden_states), thread document_ids through to any Engram blocks.

Block kind → V4 module mapping:
    gdn               → cppmega_v4.nn.linear_attention.LinearAttention
    kda               → cppmega_v4.nn.kimi_delta_attention.KimiDeltaAttention
    mla_absorb        → cppmega_v4.nn.mla_absorb.AbsorbedMLA
    engram            → cppmega_v4.nn.engram_v4.EngramV4Block (doc-aware)
    moe               → cppmega_v4.nn.moe_v4.V4MoE
    lightning_indexer → cppmega_v4.nn.lightning_indexer_fp8.LightningIndexerFP8
    nsa               → cppmega_v4.nn.nsa_v4.NativeSparseAttentionV4
    csa_hca           → cppmega_v4.nn.csa_hca_v4.CSAHCAHybridV4
    mlp               → mlx.nn.Linear sandwich (gate/up/down + SiLU)

The superblock only knows how to compose blocks, not how to train. A
HybridTinyLM-style outer loop can wrap it for an actual training run.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn.csa_hca_v4 import CSAHCAConfig, CSAHCAHybridV4
from cppmega_v4.nn.engram_v4 import EngramV4Block, EngramV4Config
from cppmega_v4.nn.nsa_v4 import NSAConfig, NativeSparseAttentionV4
from cppmega_v4.run_template import BlockSpec, RunTemplate


# Block kind → factory. Each factory takes (hidden_size, params dict) and
# returns the block module. We keep factories tiny — full hyperparameter
# wiring happens inside the block's config dataclass.

def _build_engram(hidden_size: int, params: dict) -> EngramV4Block:
    cfg = EngramV4Config(hidden_size=hidden_size, **params)
    return EngramV4Block(cfg)


def _build_nsa(hidden_size: int, params: dict) -> NativeSparseAttentionV4:
    if "num_heads" not in params:
        params["num_heads"] = max(1, hidden_size // 32)
    if "head_dim" not in params:
        params["head_dim"] = hidden_size // params["num_heads"]
    cfg = NSAConfig(hidden_size=hidden_size, **params)
    return NativeSparseAttentionV4(cfg)


def _build_csa_hca(hidden_size: int, params: dict) -> CSAHCAHybridV4:
    if "num_heads" not in params:
        params["num_heads"] = max(1, hidden_size // 32)
    if "head_dim" not in params:
        params["head_dim"] = hidden_size // params["num_heads"]
    cfg = CSAHCAConfig(hidden_size=hidden_size, **params)
    return CSAHCAHybridV4(cfg)


def _build_mlp(hidden_size: int, params: dict) -> nn.Module:
    """MLP / Gated-MLP block.

    The ``activation`` param (E7-5) routes between gated and dense paths:
      - 'glu' (default, backwards-compat): sigmoid(gate) * up
      - 'swiglu'/'geglu'/'reglu': gated activation via apply_activation
      - 'gelu'/'silu'/'relu'/'relu2'/'sqrelu': dense — gate projection
        kept allocated for state-dict parity but unused; up→act→down.

    Validation lives in verify_build_spec, not here, so a misuse here
    raises ValueError from apply_activation.
    """
    from cppmega_mlx.nn.activations import IS_GATED, apply_activation

    intermediate = params.get("intermediate_size", 4 * hidden_size)
    activation = params.get("activation", "glu")
    norm_eps = params.get("norm_eps", 1e-6)
    pre_norm_kind = params.get("pre_norm", "none")
    post_norm_kind = params.get("post_norm", "none")

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(hidden_size, intermediate, bias=False)
            self.up = nn.Linear(hidden_size, intermediate, bias=False)
            self.down = nn.Linear(intermediate, hidden_size, bias=False)
            self.pre_norm = _make_norm(pre_norm_kind, hidden_size, norm_eps)
            self.post_norm = _make_norm(post_norm_kind, hidden_size, norm_eps)

        def __call__(self, x):
            if self.pre_norm is not None:
                x = self.pre_norm(x)
            up = self.up(x)
            if activation == "glu":
                y = self.down(mx.sigmoid(self.gate(x)) * up)
            elif activation in IS_GATED and IS_GATED[activation]:
                y = self.down(apply_activation(activation, up,
                                                gate=self.gate(x)))
            elif activation in IS_GATED:
                y = self.down(apply_activation(activation, up))
            else:
                # Unknown name — fall back to GLU to keep training alive.
                y = self.down(mx.sigmoid(self.gate(x)) * up)
            if self.post_norm is not None:
                y = self.post_norm(y)
            return y
    return _MLP()


def _build_pass_through_unsupported(kind: str):
    """Last-resort no-op for kinds with no factory yet. Residual-only."""
    class _PassThrough(nn.Module):
        def __init__(self):
            super().__init__()
            self._kind = kind
        def __call__(self, x):
            return x  # zero contribution
    return _PassThrough()


def _build_gdn(hidden_size: int, params: dict) -> nn.Module:
    """Real GatedDeltaNet block (Path A — Path B/C/E land via env override)."""
    from cppmega_v4.nn.linear_attention import (  # local import: avoids circular
        LinearAttentionBlock, LinearAttentionConfig,
    )
    # Defaults sensible for our 1B smoke configs.
    params.setdefault("num_heads", max(1, hidden_size // 64))
    params.setdefault("head_dim", hidden_size // params["num_heads"])
    cfg = LinearAttentionConfig(hidden_size=hidden_size, **params)
    return LinearAttentionBlock(cfg)


def _build_kda(hidden_size: int, params: dict) -> nn.Module:
    """Real Kimi Delta Attention block."""
    from cppmega_v4.nn.kimi_delta_attention import (
        KimiDeltaAttentionBlock, KimiDeltaAttentionConfig,
    )
    params.setdefault("num_heads", max(1, hidden_size // 64))
    params.setdefault("head_dim", hidden_size // params["num_heads"])
    cfg = KimiDeltaAttentionConfig(hidden_size=hidden_size, **params)
    return KimiDeltaAttentionBlock(cfg)


def _build_moe(hidden_size: int, params: dict) -> nn.Module:
    """Real V4 MoE. V4MoE returns MoEOutput — wrap to expose .output."""
    from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig

    params.setdefault("expert_hidden_size", hidden_size * 4)
    cfg = V4MoEConfig(d_model=hidden_size, **params)
    inner = V4MoE(cfg)

    class _MoEWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = inner

        def __call__(self, x):
            return self.moe(x).output

    return _MoEWrap()


def _build_mla(hidden_size: int, params: dict) -> nn.Module:
    """Real V3-style MLA block (LoRA Q + LoRA KV + RoPE + absorb fast-path)."""
    from cppmega_v4.nn.mla_block import MLABlock, MLABlockConfig
    params.setdefault("num_heads", max(1, hidden_size // 128))
    params.setdefault("qk_nope_head_dim", 128)
    params.setdefault("qk_rope_head_dim", 64)
    params.setdefault("v_head_dim", 128)
    params.setdefault("q_lora_rank", max(64, hidden_size // 2))
    params.setdefault("kv_lora_rank", max(32, hidden_size // 4))
    cfg = MLABlockConfig(hidden_size=hidden_size, **params)
    return MLABlock(cfg)


def _build_mistral4_mla(hidden_size: int, params: dict) -> nn.Module:
    """Mistral Small 4 absorbed MLA + INT4 latent cache (mlx-lm PR #1037)."""
    from cppmega_v4.nn.mlx_lm_bricks import Mistral4MLABlock, Mistral4MLAConfig
    cfg = Mistral4MLAConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in Mistral4MLAConfig.__dataclass_fields__
    })
    return Mistral4MLABlock(cfg)


def _build_dsv4_attention(hidden_size: int, params: dict) -> nn.Module:
    """DeepSeek-V4 (Flash) hash-indexed sparse attention (mlx-lm PR #1201)."""
    from cppmega_v4.nn.mlx_lm_bricks import DSv4AttentionBlock, DSv4AttentionConfig
    cfg = DSv4AttentionConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in DSv4AttentionConfig.__dataclass_fields__
    })
    return DSv4AttentionBlock(cfg)


def _build_bailing_linear(hidden_size: int, params: dict) -> nn.Module:
    """Ling-2.6-flash linear attention (mlx-lm PR #1227)."""
    from cppmega_v4.nn.mlx_lm_bricks import BailingLinearAttnBlock, BailingLinearConfig
    cfg = BailingLinearConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in BailingLinearConfig.__dataclass_fields__
    })
    return BailingLinearAttnBlock(cfg)


def _build_bailing_mla(hidden_size: int, params: dict) -> nn.Module:
    """Ling-2.6-flash multi-latent attention (mlx-lm PR #1227)."""
    from cppmega_v4.nn.mlx_lm_bricks import BailingMLABlock, BailingMLAConfig
    cfg = BailingMLAConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in BailingMLAConfig.__dataclass_fields__
    })
    return BailingMLABlock(cfg)


def _build_bailing_moe(hidden_size: int, params: dict) -> nn.Module:
    """Ling-2.6-flash sparse MoE block (mlx-lm PR #1227)."""
    from cppmega_v4.nn.mlx_lm_bricks import BailingMoEBlock, BailingMoEConfig
    cfg = BailingMoEConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in BailingMoEConfig.__dataclass_fields__
    })
    return BailingMoEBlock(cfg)


def _build_gemma4_drafter(hidden_size: int, params: dict) -> nn.Module:
    """Gemma 4 MTP-drafter decoder layer (mlx-lm PR #1276)."""
    from cppmega_v4.nn.mlx_lm_bricks import Gemma4DrafterLayerBlock, Gemma4DrafterLayerConfig
    cfg = Gemma4DrafterLayerConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in Gemma4DrafterLayerConfig.__dataclass_fields__
    })
    return Gemma4DrafterLayerBlock(cfg)


def _build_nemotron_h_mtp(hidden_size: int, params: dict) -> nn.Module:
    """Nemotron-H Multi-Token-Prediction block (mlx-lm PR #1161)."""
    from cppmega_v4.nn.mlx_lm_bricks import NemotronHMTPBlockWrapper, NemotronHMTPConfig
    cfg = NemotronHMTPConfig(hidden_size=hidden_size, **{
        k: v for k, v in params.items()
        if k in NemotronHMTPConfig.__dataclass_fields__
    })
    return NemotronHMTPBlockWrapper(cfg)


def _build_gated_attention(hidden_size: int, params: dict) -> nn.Module:
    """Qwen3-Next / Qwen3.5 / Qwen3.6 Gated Attention.

    Directly re-exports ``mlx_lm.models.qwen3_next.Qwen3NextAttention`` via
    ``cppmega_v4.nn.gated_attention.GatedAttentionBlock`` — no vendoring,
    no re-implementation. Under the hood: ``mx.fast.scaled_dot_product_attention``
    (Apple MPS) + sigmoid output gate + partial RoPE + asymmetric GQA +
    Q/K RMSNorm. See cppmega_v4/nn/gated_attention.py for the wrapper.
    """
    from cppmega_v4.nn.gated_attention import GatedAttentionBlock, GatedAttentionConfig
    cfg = GatedAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=params.get("num_attention_heads", max(1, hidden_size // 64)),
        num_key_value_heads=params.get(
            "num_key_value_heads",
            max(1, params.get("num_attention_heads", max(1, hidden_size // 64)) // 8),
        ),
        head_dim=params.get("head_dim", 64),
        rms_norm_eps=params.get("rms_norm_eps", 1e-6),
        rope_theta=params.get("rope_theta", 1_000_000.0),
        partial_rotary_factor=params.get("partial_rotary_factor", 0.25),
        max_position_embeddings=params.get("max_position_embeddings", 262_144),
        attention_bias=params.get("attention_bias", False),
        rope_scaling=params.get("rope_scaling"),
    )
    return GatedAttentionBlock(cfg)


def _build_gqa_sliding(hidden_size: int, params: dict) -> nn.Module:
    """Gemma 4-style GQA with sliding causal window.

    Used by the 5:1 sliding/global Gemma 4 and Arcee Trinity Large presets.
    Wraps ``cppmega_v4.nn.sliding_attention.GQAWithSlidingWindowBlock``.
    """
    from cppmega_v4.nn.sliding_attention import (
        GQASlidingConfig, GQAWithSlidingWindowBlock,
    )
    cfg = GQASlidingConfig(
        hidden_size=hidden_size,
        num_attention_heads=params.get(
            "num_attention_heads", max(1, hidden_size // 64)
        ),
        num_key_value_heads=params.get(
            "num_key_value_heads",
            max(1, params.get("num_attention_heads",
                              max(1, hidden_size // 64)) // 8),
        ),
        head_dim=params.get("head_dim", 64),
        sliding_window_size=params.get("sliding_window_size", 4096),
        rms_norm_eps=params.get("rms_norm_eps", 1e-6),
        rope_theta=params.get("rope_theta", 1_000_000.0),
        qk_norm=params.get("qk_norm", True),
    )
    return GQAWithSlidingWindowBlock(cfg)


def _build_cca_attention(hidden_size: int, params: dict) -> nn.Module:
    """ZAYA1 Coarse Causal Attention — compressed-context attention.

    Wraps ``cppmega_v4.nn.cca_attention.CCAAttentionBlock``.
    """
    from cppmega_v4.nn.cca_attention import (
        CCAAttentionBlock, CCAAttentionConfig,
    )
    cfg = CCAAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=params.get(
            "num_attention_heads", max(1, hidden_size // 64)
        ),
        num_key_value_heads=params.get(
            "num_key_value_heads",
            max(1, params.get("num_attention_heads",
                              max(1, hidden_size // 64)) // 8),
        ),
        head_dim=params.get("head_dim", 64),
        fine_window=params.get("fine_window", 256),
        coarse_block_size=params.get("coarse_block_size", 16),
        rms_norm_eps=params.get("rms_norm_eps", 1e-6),
    )
    return CCAAttentionBlock(cfg)


def _build_mlstm(hidden_size: int, params: dict) -> nn.Module:
    """xLSTM matrix-LSTM block (GalCov-B). Gallery entry #50 xLSTM 7B."""
    from cppmega_v4.nn.mlstm import MLSTMBlock, MLSTMConfig
    cfg = MLSTMConfig(
        hidden_size=hidden_size,
        head_dim=params.get("head_dim", 64),
        rms_norm_eps=params.get("rms_norm_eps", 1e-6),
    )
    return MLSTMBlock(cfg)


def _build_abs_pos_embed(hidden_size: int, params: dict) -> nn.Module:
    """Learned absolute positional embedding (GalCov-B). Gallery #1 GPT-2 XL."""
    from cppmega_v4.nn.abs_pos_embed import AbsPosEmbedBlock, AbsPosEmbedConfig
    cfg = AbsPosEmbedConfig(
        hidden_size=hidden_size,
        max_position_embeddings=params.get("max_position_embeddings", 4096),
    )
    return AbsPosEmbedBlock(cfg)


def _build_per_layer_embed(hidden_size: int, params: dict) -> nn.Module:
    """Per-layer scaled embedding (GalCov-B). Gallery #57 Gemma 4 E2B, #58 E4B."""
    from cppmega_v4.nn.per_layer_embed import PerLayerEmbedBlock, PerLayerEmbedConfig
    cfg = PerLayerEmbedConfig(
        hidden_size=hidden_size,
        layer_index=params.get("layer_index", 0),
        num_layers=params.get("num_layers", 32),
    )
    return PerLayerEmbedBlock(cfg)


def _build_mamba3(hidden_size: int, params: dict) -> nn.Module:
    """Mamba-3 SSM reference block (from ``cppmega_mlx.nn.mamba3``).

    Thin re-export so v4 architectures (Nemotron 3 Super) can compose
    a Mamba-2/3 block alongside attention/MoE bricks without modifying
    the plugin. Only ``d_model`` is required; everything else takes the
    Mamba3Config dataclass defaults unless overridden via params.
    """
    from cppmega_mlx.nn.mamba3 import Mamba3Config, Mamba3ReferenceBlock
    cfg_kwargs = {"d_model": hidden_size}
    cfg_kwargs.update(
        {k: v for k, v in params.items()
         if k in Mamba3Config.__dataclass_fields__ and k != "d_model"}
    )
    return Mamba3ReferenceBlock(Mamba3Config(**cfg_kwargs))


def _make_norm(kind: str, dim: int, eps: float):
    """E7-6/E7-18: return RMSNorm / LayerNorm / None per kind."""
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if kind == "none":
        return None
    raise ValueError(f"unknown norm kind {kind!r}; "
                     "use 'rmsnorm' / 'layernorm' / 'none'")


def _build_attention(hidden_size: int, params: dict) -> nn.Module:
    """Standard multi-head self-attention (causal). Used for `attention`.

    Honors params.pre_norm (default 'none', matches legacy behavior)
    and params.post_norm (default 'rmsnorm', also matches legacy).
    See cppmega_v4/buildspec/norm_validation.py for diagnostics.
    """
    num_heads = params.get("num_heads", max(1, hidden_size // 64))
    head_dim = params.get("head_dim", hidden_size // num_heads)
    norm_eps = params.get("norm_eps", 1e-6)
    pre_norm_kind = params.get("pre_norm", "none")
    post_norm_kind = params.get("post_norm", "rmsnorm")

    class _SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            self.pre_norm = _make_norm(pre_norm_kind, hidden_size, norm_eps)
            # 'norm' alias preserves state-dict keys from earlier
            # checkpoints where only one (post) norm existed.
            self.norm = _make_norm(post_norm_kind, hidden_size, norm_eps)
            # Zero-init out so the block is identity at init.
            self.o_proj.weight = mx.zeros_like(self.o_proj.weight)

        def __call__(self, x):
            if self.pre_norm is not None:
                x = self.pre_norm(x)
            B, S, _ = x.shape
            q = self.q_proj(x).reshape(B, S, num_heads, head_dim)
            k = self.k_proj(x).reshape(B, S, num_heads, head_dim)
            v = self.v_proj(x).reshape(B, S, num_heads, head_dim)
            q = mx.transpose(q, (0, 2, 1, 3))
            k = mx.transpose(k, (0, 2, 1, 3))
            v = mx.transpose(v, (0, 2, 1, 3))
            scale = head_dim ** -0.5
            scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
            mask = mx.tril(mx.ones((S, S), dtype=mx.bool_))
            scores = mx.where(mask, scores, mx.full(scores.shape, -1e9,
                                                     dtype=scores.dtype))
            w = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
            o = mx.matmul(w, v)
            o = mx.transpose(o, (0, 2, 1, 3)).reshape(B, S, num_heads * head_dim)
            o = self.o_proj(o)
            if self.norm is not None:
                o = self.norm(o)
            return o

    return _SelfAttn()


def _build_lightning_indexer(hidden_size: int, params: dict) -> nn.Module:
    """Lightning Indexer wired into a CSA+HCA block via the production adapter.

    The block as a residual is meaningful only when followed by sparse-KV
    attention that consumes the top-k indices. We bundle one indexer with
    one CSA+HCA inside the same nn.Module so the residual output is the
    actual sparse-attention contribution from `apply_indexer_to_csa_hca`.

    For pure top-k extraction (no CSA+HCA), call LightningIndexerFP8 directly.
    """
    from cppmega_v4.nn.csa_hca_indexer_adapter import apply_indexer_to_csa_hca
    from cppmega_v4.nn.csa_hca_v4 import CSAHCAConfig, CSAHCAHybridV4
    from cppmega_v4.nn.lightning_indexer_fp8 import (
        LightningIndexerFP8, LightningIndexerFP8Config,
    )
    n_heads = params.get("n_heads", max(1, hidden_size // 64))
    head_dim = params.get("head_dim", 32)
    rope_head_dim = params.get("rope_head_dim", 16)
    q_lora_rank = params.get("q_lora_rank", hidden_size)
    index_topk = params.get("index_topk", 64)
    fp8_blocks = params.get("fp8_blocks", True)
    m_csa = params.get("m_csa", 4)
    m_hca = params.get("m_hca", 16)
    # CSA+HCA's num_heads / head_dim use the *hidden* layout (not the
    # indexer's internal small-head_dim layout).
    csa_n_heads = params.get("csa_num_heads", max(1, hidden_size // 64))
    csa_head_dim = params.get("csa_head_dim", hidden_size // csa_n_heads)
    indexer_cfg = LightningIndexerFP8Config(
        hidden_size=hidden_size,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        index_topk=index_topk,
        fp8_blocks=fp8_blocks,
    )
    csa_hca_cfg = CSAHCAConfig(
        hidden_size=hidden_size,
        num_heads=csa_n_heads, head_dim=csa_head_dim,
        m_csa=m_csa, m_hca=m_hca,
    )

    class _IndexerCSAHCABundle(nn.Module):
        """Lightning Indexer → CSA+HCA bundle: residual = sparse-attn output."""
        def __init__(self):
            super().__init__()
            self.indexer = LightningIndexerFP8(indexer_cfg)
            self.csa_hca = CSAHCAHybridV4(csa_hca_cfg)
            # qr_proj: synthesise qr from x when the caller doesn't pre-LoRA
            # (the indexer's q_lora_rank is just the bottleneck dim — we
            # produce qr via a single linear when it's not handed in).
            self.qr_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            # Precomputed RoPE freqs cached by max-seq length.
            self._cached_freqs: dict[int, tuple] = {}

        def _freqs(self, seq: int) -> tuple:
            d = rope_head_dim // 2
            if seq not in self._cached_freqs:
                inv_freq = 1.0 / (
                    10000.0 ** (mx.arange(d, dtype=mx.float32) * 2.0 / rope_head_dim)
                )
                t = mx.arange(seq, dtype=mx.float32)
                f = t[:, None] * inv_freq[None, :]
                self._cached_freqs[seq] = (mx.cos(f), mx.sin(f))
            return self._cached_freqs[seq]

        def __call__(self, x):
            B, S, _ = x.shape
            qr = self.qr_proj(x)
            cos, sin = self._freqs(S)
            return apply_indexer_to_csa_hca(
                self.indexer, self.csa_hca, x, qr, (cos, sin),
            )

    return _IndexerCSAHCABundle()


BLOCK_BUILDERS: dict[str, Callable[[int, dict], nn.Module]] = {
    "engram": _build_engram,
    "nsa": _build_nsa,
    "csa_hca": _build_csa_hca,
    "mlp": _build_mlp,
    "gdn": _build_gdn,
    "kda": _build_kda,
    "moe": _build_moe,
    "attention": _build_attention,
    # gated_attention = Qwen3-Next / Qwen3.6 25%-softmax slot. Direct
    # re-export of mlx_lm.models.qwen3_next.Qwen3NextAttention; output gate
    # + partial RoPE + asymmetric GQA + Q/K RMSNorm. Underlying compute is
    # mx.fast.scaled_dot_product_attention (Apple MPS Metal SDPA).
    "gated_attention": _build_gated_attention,
    # mla = V3-style with LoRA Q + LoRA KV + RoPE on pe-only split.
    # mla_absorb = same block, prefers absorb fast-path at decode.
    "mla": _build_mla,
    "mla_absorb": _build_mla,
    "lightning_indexer": _build_lightning_indexer,
    # ----- bricks imported from open mlx-lm PRs (cppmega-integration branch) -----
    # mistral4_mla       = Mistral Small 4 absorbed MLA + INT4 latent cache (PR #1037)
    # dsv4_attention     = DeepSeek-V4 (Flash) hash-indexed sparse attention (PR #1201)
    # bailing_linear     = Ling-2.6-flash linear attention (PR #1227)
    # bailing_mla        = Ling-2.6-flash multi-latent attention (PR #1227)
    # bailing_moe        = Ling-2.6-flash sparse MoE block (PR #1227)
    "mistral4_mla": _build_mistral4_mla,
    "dsv4_attention": _build_dsv4_attention,
    "bailing_linear": _build_bailing_linear,
    "bailing_mla": _build_bailing_mla,
    "bailing_moe": _build_bailing_moe,
    # gemma4_drafter = Gemma 4 MTP-drafter decoder layer (cross-attn) (PR #1276)
    # nemotron_h_mtp = Nemotron-H Multi-Token-Prediction block (PR #1161)
    "gemma4_drafter": _build_gemma4_drafter,
    "nemotron_h_mtp": _build_nemotron_h_mtp,
    # ----- Stage D additions (Auto-Fusion architecture presets) -----
    # gqa_sliding   = Gemma 4 / Arcee Trinity 5:1 sliding-window GQA slot
    # cca_attention = ZAYA1 Coarse-Causal-Attention (compressed context)
    # mamba3        = Nemotron 3 Super SSM block (re-export from cppmega_mlx)
    "gqa_sliding": _build_gqa_sliding,
    "cca_attention": _build_cca_attention,
    "mamba3": _build_mamba3,
    "mlstm": _build_mlstm,
    "abs_pos_embed": _build_abs_pos_embed,
    "per_layer_embed": _build_per_layer_embed,
}


@dataclass
class _BuiltBlock:
    kind: str
    module: nn.Module
    needs_doc_ids: bool
    needs_token_ids: bool


_DOC_ID_KW_KINDS = {"gdn", "kda"}     # accept doc_ids via *kwargs
_TOKEN_ID_POS_KINDS = {"engram"}      # take token_ids positional


def _build_one(spec: BlockSpec, hidden_size: int) -> _BuiltBlock:
    builder = BLOCK_BUILDERS.get(spec.kind)
    if builder is None:
        raise ValueError(f"no builder registered for block kind {spec.kind!r}")
    mod = builder(hidden_size, dict(spec.params))
    return _BuiltBlock(
        kind=spec.kind, module=mod,
        needs_doc_ids=(spec.kind in _DOC_ID_KW_KINDS or spec.kind == "engram"),
        needs_token_ids=(spec.kind in _TOKEN_ID_POS_KINDS),
    )


class UnifiedSuperblockV4(nn.Module):
    """Composes V4 blocks per RunTemplate. Threads doc_ids to Engram blocks.

    Forward:
        (token_ids: [B, S] int32, hidden_states: [B, S, H],
         document_ids: [B, S] int32 | None)
        -> [B, S, H]

    Each block in the template is repeated ``spec.repeat`` times. Residual
    connections are applied around every block: ``h = h + block(h, ...)``.
    """

    def __init__(self, template: RunTemplate):
        super().__init__()
        self.template = template
        self.hidden_size = template.hidden_size
        # Flatten the (kind, repeat) entries into a list of built blocks.
        flat_specs: list[BlockSpec] = []
        for spec in template.blocks:
            for _ in range(spec.repeat):
                flat_specs.append(BlockSpec(kind=spec.kind, repeat=1,
                                             params=dict(spec.params)))
        self.blocks: list[_BuiltBlock] = [
            _build_one(s, template.hidden_size) for s in flat_specs
        ]
        # nn.Module needs the modules accessible as attributes for parameter
        # discovery: register under deterministic names.
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}_{b.kind}", b.module)

    @property
    def path_c_bricks(self) -> tuple[dict[str, str], ...]:
        """Return allocation-free brick descriptors for Path C auto-discovery."""

        return tuple(
            {
                "name": f"block_{index}_{block.kind}",
                "kind": block.kind,
            }
            for index, block in enumerate(self.blocks)
        )

    def path_c_fusion_regions(
        self,
        *,
        include_backward: bool = False,
        min_route_bricks: int = 2,
    ):
        """Return Path C candidate regions discovered from this brick stack."""

        from cppmega_mlx.runtime.path_c_fusion import (
            build_path_c_model_regions_from_model,
        )

        return build_path_c_model_regions_from_model(
            self,
            region_prefix=f"{self.template.name}_path_c",
            include_backward=include_backward,
            min_route_bricks=min_route_bricks,
        )

    def __call__(
        self,
        token_ids: mx.array,
        hidden_states: mx.array,
        document_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be (B, S), got {token_ids.shape}")
        if hidden_states.shape[:2] != token_ids.shape:
            raise ValueError(
                f"hidden_states {hidden_states.shape} must agree with "
                f"token_ids {token_ids.shape} on the first two axes"
            )
        if document_ids is not None and document_ids.shape != token_ids.shape:
            raise ValueError(
                f"document_ids {document_ids.shape} must match "
                f"token_ids {token_ids.shape}"
            )
        h = hidden_states
        for b in self.blocks:
            if b.needs_token_ids and b.needs_doc_ids:
                # Engram: positional token_ids + keyword document_ids.
                delta = b.module(h, token_ids, document_ids=document_ids)
            elif b.needs_token_ids:
                delta = b.module(h, token_ids)
            elif b.needs_doc_ids:
                # GDN/KDA: doc_ids keyword (block uses it for doc-reset).
                delta = b.module(h, doc_ids=document_ids)
            else:
                delta = b.module(h)
            h = h + delta
        return h

    def kinds(self) -> list[str]:
        return [b.kind for b in self.blocks]


__all__ = ["UnifiedSuperblockV4"]
