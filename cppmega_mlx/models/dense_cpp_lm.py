"""Stage-1 dense C++ foundation LM, composed from cppmega building blocks.

``DenseCppLM`` is an all-attention decoder assembled from the EXISTING cppmega
reference blocks:

  * token embedding (tied LM head) + learned position embedding;
  * the side-channel embeddings already used by ``hybrid_lm`` —
    :class:`CppMegaStructureEmbedding` (structure_ids / dep_levels / ast_depth /
    sibling_index / ast_node_type), :class:`CppMegaPlatformEmbedding` (platform),
    and :class:`NgramHashEmbedding` (ngram_hash) — added as scaled residuals;
  * a stack of N pre-norm residual blocks, each a :class:`CausalSelfAttention`
    (GQA mode: num_query_heads=20, num_kv_heads=4, head_dim=64, RoPE) followed by
    a SwiGLU FFN built from :class:`FeedForwardExpert` (the shared SwiGLU MLP);
  * :class:`mlx.nn.RMSNorm` pre-norm everywhere + a final RMSNorm.

This is deliberately NOT a from-scratch transformer: every sub-block is one of
our existing modules.  It does NOT edit or subclass ``HybridTinyLM``; it imports
and composes the same leaf modules ``hybrid_lm`` uses.

The attention pattern is all-attention (no MoE / Mamba / M2RNN / engram /
concept).  DSA is NOT used in this run: ``attention.mode`` defaults to ``"gqa"``.
The ``mode`` is threaded straight into :class:`AttentionConfig`, so flipping the
config to ``mode="dsa"`` later swaps GQA for the Sparse-MLA path without any
structural change here (the clean DSA seam).

RULE #1 (fail fast / fail loud): every shape/option mismatch RAISES with WHERE +
WHAT.  There is no silent fallback path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.platform_context import MAX_PLATFORM_IDS, PLATFORM_VOCAB_SIZE
from cppmega_mlx.nn.attention import AttentionConfig, CausalSelfAttention
from cppmega_mlx.nn.moe import FeedForwardExpert
from cppmega_mlx.nn.ngram_hash import NgramHashEmbedding
from cppmega_mlx.nn.platform_embedding import CppMegaPlatformEmbedding
from cppmega_mlx.nn.structure_embedding import CppMegaStructureEmbedding

# Reuse the same attention-mode literal the rest of the repo uses. DSA is the
# later (Sparse-MLA) replacement for GQA; "full"/"gqa" are the dense SDPA paths.
DenseAttentionMode = Literal["mla", "dsa", "full", "gqa"]


@dataclass(frozen=True)
class DenseCppLMConfig:
    """Config for the Stage-1 dense C++ foundation LM.

    Defaults match the real ~500M Stage-1 profile (pattern A, depth 24,
    d_model 1280, ffn 3456, GQA 20q/4kv, head_dim 64, vocab 65536). A tiny smoke
    profile (e.g. ``hidden_size=256, depth=4``) builds from the same code by
    overriding the dims; everything is config-driven.
    """

    # Core LM dims.
    vocab_size: int = 65536
    hidden_size: int = 1280
    depth: int = 24
    ffn_hidden_size: int = 3456
    max_seq_length: int = 4096

    # Attention (GQA for the first run; DSA seam via ``attention_mode``).
    num_query_heads: int = 20
    num_kv_heads: int = 4
    head_dim: int = 64
    attention_mode: DenseAttentionMode = "gqa"
    rope: bool = True
    rope_theta: float = 10000.0
    attention_sparse_topk: int = 256  # only consumed when attention_mode='dsa'

    # SwiGLU FFN.
    ffn_activation: Literal["swiglu"] = "swiglu"

    # Side-channel structure embedding (CppMegaStructureEmbedding).
    structure_components: str = "all"
    structure_bottleneck_dim: int = 128
    structure_num_categories: int = 9
    structure_max_dep_level: int = 64
    structure_max_ast_depth: int = 128
    structure_max_sibling_index: int = 128
    structure_num_node_types: int = 512

    # Platform embedding.
    platform_vocab_size: int = PLATFORM_VOCAB_SIZE
    platform_max_ids: int = MAX_PLATFORM_IDS

    # N-gram hash embedding.
    ngram_hash_enabled: bool = True
    ngram_hash_orders: tuple[int, ...] = (2, 3)
    ngram_hash_heads: int = 8
    ngram_hash_table_size: int = 500_000
    ngram_hash_embed_dim: int = 16
    ngram_hash_dropout: float = 0.0
    ngram_hash_seed: int | None = None

    # Side-channel residual scales (0.0 disables a family cleanly).
    structure_residual_scale: float = 1.0
    platform_residual_scale: float = 1.0
    ngram_residual_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.ffn_hidden_size <= 0:
            raise ValueError(
                f"ffn_hidden_size must be positive, got {self.ffn_hidden_size}"
            )
        if self.max_seq_length < 2:
            raise ValueError(
                f"max_seq_length must be >= 2, got {self.max_seq_length}"
            )
        if self.num_query_heads <= 0:
            raise ValueError(
                f"num_query_heads must be positive, got {self.num_query_heads}"
            )
        if self.num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be positive, got {self.num_kv_heads}"
            )
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads {self.num_query_heads} must be divisible by "
                f"num_kv_heads {self.num_kv_heads}"
            )
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.rope and self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even when rope=True, got {self.head_dim}"
            )
        if self.ffn_activation != "swiglu":
            raise ValueError(
                f"DenseCppLM Stage-1 FFN is SwiGLU-only, got {self.ffn_activation!r}"
            )
        for name in (
            "structure_residual_scale",
            "platform_residual_scale",
            "ngram_residual_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        # Validate the attention contract end-to-end at construction time. This
        # is the single source of truth for the GQA<->DSA seam: AttentionConfig
        # rejects e.g. mode='gqa' with num_kv_heads == num_query_heads.
        self.attention_config()
        # Validate structure-embedding component spec.
        CppMegaStructureEmbedding._parse_components(self.structure_components)

    def attention_config(
        self, mode: DenseAttentionMode | None = None
    ) -> AttentionConfig:
        """Build the AttentionConfig for one block.

        ``mode`` defaults to ``self.attention_mode`` (``"gqa"`` for the first
        run).  Passing ``mode="dsa"`` here is the clean seam that swaps GQA for
        the Sparse-MLA path without restructuring the model.
        """

        active_mode = mode if mode is not None else self.attention_mode
        return AttentionConfig(
            d_model=self.hidden_size,
            num_q_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mode=active_mode,
            use_rope=self.rope,
            rope_theta=self.rope_theta,
            sparse_topk=self.attention_sparse_topk,
        )

    def structure_embedding_kwargs(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "num_categories": self.structure_num_categories,
            "max_dep_level": self.structure_max_dep_level,
            "max_ast_depth": self.structure_max_ast_depth,
            "max_sibling_index": self.structure_max_sibling_index,
            "num_node_types": self.structure_num_node_types,
            "active_components": self.structure_components,
            "bottleneck_dim": self.structure_bottleneck_dim,
        }


class DenseCppBlock(nn.Module):
    """One pre-norm residual block: RMSNorm -> GQA attention -> +residual,
    then RMSNorm -> SwiGLU FFN -> +residual.

    Both sub-modules are EXISTING cppmega leaves (``CausalSelfAttention`` and the
    SwiGLU ``FeedForwardExpert``).  ``self.attention.config.mode`` is the live
    DSA seam — it is whatever ``DenseCppLMConfig.attention_config`` produced.
    """

    def __init__(self, config: DenseCppLMConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config.attention_config())
        self.ffn_norm = nn.RMSNorm(config.hidden_size)
        # FeedForwardExpert is the shared SwiGLU MLP used by our MoE experts;
        # used standalone here it is a plain SwiGLU FFN (gate/up/down).
        self.ffn = FeedForwardExpert(
            config.hidden_size,
            config.ffn_hidden_size,
            activation="swiglu",
            bias=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | Literal["causal"] | None,
    ) -> mx.array:
        hidden_states = hidden_states + self.attention(
            self.attn_norm(hidden_states), mask
        )
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class DenseCppLM(nn.Module):
    """Stage-1 dense, all-attention C++ foundation LM (tied LM head)."""

    def __init__(
        self,
        config: DenseCppLMConfig | None = None,
        *,
        dtype: mx.Dtype | None = None,
    ):
        super().__init__()
        self.config = config or DenseCppLMConfig()
        cfg = self.config

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)

        self.structure_embedding = CppMegaStructureEmbedding(
            **cfg.structure_embedding_kwargs()
        )
        self.platform_embedding = CppMegaPlatformEmbedding(
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.platform_vocab_size,
            max_ids=cfg.platform_max_ids,
        )
        self.ngram_hash_embedding: NgramHashEmbedding | None = None
        if cfg.ngram_hash_enabled:
            self.ngram_hash_embedding = NgramHashEmbedding(
                hidden_size=cfg.hidden_size,
                orders=cfg.ngram_hash_orders,
                num_heads=cfg.ngram_hash_heads,
                table_size=cfg.ngram_hash_table_size,
                embed_dim=cfg.ngram_hash_embed_dim,
                dropout=cfg.ngram_hash_dropout,
                seed=cfg.ngram_hash_seed,
            )

        self.layers = [DenseCppBlock(cfg) for _ in range(cfg.depth)]
        self.norm = nn.RMSNorm(cfg.hidden_size)
        # Tied LM head: a bias-free Linear whose weight is the token-embedding
        # table (vocab x hidden). MLX nn.Linear computes x @ weight.T, which is
        # exactly the tied projection.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        if dtype is not None and dtype != mx.float32:
            self.set_dtype(dtype)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def embed(
        self,
        input_ids: mx.array,
        *,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        platform_ids: mx.array | None = None,
    ) -> mx.array:
        """Token + position + scaled side-channel residual embeddings."""

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be shaped (B, S), got {tuple(input_ids.shape)}"
            )
        batch_size, seq_length = int(input_ids.shape[0]), int(input_ids.shape[1])
        if seq_length > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq_length} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )

        positions = mx.arange(seq_length)[None, :]
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        if self.ngram_hash_embedding is not None and self.config.ngram_residual_scale:
            ngram = self.ngram_hash_embedding(input_ids).astype(hidden.dtype)
            hidden = hidden + _scaled(ngram, self.config.ngram_residual_scale, hidden)

        if self.config.structure_residual_scale:
            structure = self.structure_embedding(
                structure_ids=_check_side_channel(
                    "structure_ids", structure_ids, batch_size, seq_length
                ),
                dep_levels=_check_side_channel(
                    "dep_levels", dep_levels, batch_size, seq_length
                ),
                ast_depth_ids=_check_side_channel(
                    "ast_depth_ids", ast_depth_ids, batch_size, seq_length
                ),
                sibling_index_ids=_check_side_channel(
                    "sibling_index_ids", sibling_index_ids, batch_size, seq_length
                ),
                node_type_ids=_check_side_channel(
                    "node_type_ids", node_type_ids, batch_size, seq_length
                ),
                target_dtype=hidden.dtype,
            )
            # When no structure channel is present the module returns a 0-D
            # scalar; only fold a true (B, S, D) residual into the stream.
            if structure.ndim == hidden.ndim:
                hidden = hidden + _scaled(
                    structure, self.config.structure_residual_scale, hidden
                )

        platform_ids = _check_platform_ids(platform_ids, batch_size, seq_length)
        if platform_ids is not None and self.config.platform_residual_scale:
            platform = self.platform_embedding(
                platform_ids, target_dtype=hidden.dtype
            )
            hidden = hidden + _scaled(
                platform, self.config.platform_residual_scale, hidden
            )

        return hidden

    def decoder_hidden_states(
        self,
        input_ids: mx.array,
        *,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        platform_ids: mx.array | None = None,
        apply_final_norm: bool = True,
    ) -> mx.array:
        hidden = self.embed(
            input_ids,
            structure_ids=structure_ids,
            dep_levels=dep_levels,
            ast_depth_ids=ast_depth_ids,
            sibling_index_ids=sibling_index_ids,
            node_type_ids=node_type_ids,
            platform_ids=platform_ids,
        )
        seq_length = int(input_ids.shape[1])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            seq_length, dtype=hidden.dtype
        )
        for layer in self.layers:
            hidden = layer(hidden, mask)
        if apply_final_norm:
            return self.norm(hidden)
        return hidden

    def logits(
        self,
        input_ids: mx.array,
        **side_channels: mx.array | None,
    ) -> mx.array:
        return self.lm_head(self.decoder_hidden_states(input_ids, **side_channels))

    def __call__(
        self,
        input_ids: mx.array,
        *,
        targets: mx.array | None = None,
        loss_mask: mx.array | None = None,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        platform_ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Return ``(logits, loss)``.

        ``loss`` is the masked next-token cross-entropy when ``targets`` is
        provided (otherwise ``None``).  ``loss_mask`` (1.0 = contributes) is
        optional; when absent every target position contributes.
        """

        logits = self.logits(
            input_ids,
            structure_ids=structure_ids,
            dep_levels=dep_levels,
            ast_depth_ids=ast_depth_ids,
            sibling_index_ids=sibling_index_ids,
            node_type_ids=node_type_ids,
            platform_ids=platform_ids,
        )
        if targets is None:
            return logits, None
        loss = self._cross_entropy(logits, targets, loss_mask)
        return logits, loss

    @staticmethod
    def _cross_entropy(
        logits: mx.array,
        targets: mx.array,
        loss_mask: mx.array | None,
    ) -> mx.array:
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"logits prefix shape {tuple(logits.shape[:2])} must match targets "
                f"{tuple(targets.shape)}"
            )
        token_losses = nn.losses.cross_entropy(
            logits.astype(mx.float32), targets, reduction="none"
        )
        if loss_mask is None:
            return token_losses.mean()
        if loss_mask.shape != targets.shape:
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} must match targets "
                f"{tuple(targets.shape)}"
            )
        mask = loss_mask.astype(mx.float32)
        ntokens = mask.sum()
        denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
        return (token_losses * mask).astype(mx.float32).sum() / denom

    # ------------------------------------------------------------------ #
    # CodePacket convenience
    # ------------------------------------------------------------------ #
    def forward_packet(
        self, packet: CodePacket
    ) -> tuple[mx.array, mx.array | None]:
        """Forward a :class:`CodePacket` (single window or batch).

        Token-aligned channels carried by the packet (structure_ids / dep_levels
        / ast_depth / sibling_index / ast_node_type) are routed to the matching
        embedding inputs.  1-D packets are promoted to a batch of 1; 2-D packets
        pass through.  Platform ids are taken from ``packet.metadata['platform_ids']``
        when present (the packed parquet stores per-document platform ids there).
        """

        input_ids = _as_batch(packet.token_ids)
        targets = _as_batch(packet.target_ids) if packet.target_ids is not None else None
        loss_mask = _as_batch(packet.loss_mask) if packet.loss_mask is not None else None
        platform_ids = packet.metadata.get("platform_ids") if packet.metadata else None
        return self(
            input_ids,
            targets=targets,
            loss_mask=loss_mask,
            structure_ids=_as_batch_opt(packet.structure_ids),
            dep_levels=_as_batch_opt(packet.dep_levels),
            ast_depth_ids=_as_batch_opt(packet.ast_depth),
            sibling_index_ids=_as_batch_opt(packet.sibling_index),
            node_type_ids=_as_batch_opt(packet.ast_node_type),
            platform_ids=platform_ids,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def num_parameters(self, *, count_tied_once: bool = True) -> int:
        """Total parameter count.

        ``count_tied_once`` reports the tied token-embedding / LM-head matrix a
        single time (the real on-device footprint), which is the standard way to
        quote model size.
        """

        from mlx.utils import tree_flatten

        seen_ids: set[int] = set()
        total = 0
        for _name, value in tree_flatten(self.parameters()):
            if not isinstance(value, mx.array):
                continue
            if count_tied_once:
                key = id(value)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
            total += int(value.size)
        return total

    def gqa_attention_shapes(self, batch_size: int, seq_length: int) -> dict[str, tuple]:
        """Return the canonical GQA attention tensor shapes for documentation.

        These are the post-projection ``(B, heads, S, D)`` shapes the SDPA path
        sees, plus the kv broadcast group size ``num_q_heads // num_kv_heads``.
        """

        cfg = self.config
        return {
            "q": (batch_size, cfg.num_query_heads, seq_length, cfg.head_dim),
            "k": (batch_size, cfg.num_kv_heads, seq_length, cfg.head_dim),
            "v": (batch_size, cfg.num_kv_heads, seq_length, cfg.head_dim),
            "kv_group_size": cfg.num_query_heads // cfg.num_kv_heads,
            "attention_mode": cfg.attention_mode,
        }


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _scaled(residual: mx.array, scale: float, like: mx.array) -> mx.array:
    if scale == 1.0:
        return residual
    return residual * mx.array(scale, dtype=like.dtype)


def _check_side_channel(
    name: str,
    tensor: mx.array | None,
    batch_size: int,
    seq_length: int,
) -> mx.array | None:
    if tensor is None:
        return None
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be shaped (B, S), got {tuple(tensor.shape)}")
    if tuple(tensor.shape) != (batch_size, seq_length):
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} must match input_ids "
            f"({batch_size}, {seq_length})"
        )
    return tensor


def _check_platform_ids(
    tensor: mx.array | None,
    batch_size: int,
    seq_length: int,
) -> mx.array | None:
    if tensor is None:
        return None
    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"platform_ids must be shaped (B, K) or (B, S, K), got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) != batch_size:
        raise ValueError(
            f"platform_ids batch dim {int(tensor.shape[0])} must match input batch "
            f"{batch_size}"
        )
    if tensor.ndim == 3 and int(tensor.shape[1]) != seq_length:
        raise ValueError(
            f"token-local platform_ids seq dim {int(tensor.shape[1])} must match "
            f"input sequence {seq_length}"
        )
    return tensor


def _as_batch(tensor: mx.array) -> mx.array:
    if tensor.ndim == 1:
        return tensor[None, :]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(
        f"expected a 1-D (S,) or 2-D (B, S) tensor, got shape {tuple(tensor.shape)}"
    )


def _as_batch_opt(tensor: mx.array | None) -> mx.array | None:
    return None if tensor is None else _as_batch(tensor)


__all__ = [
    "DenseAttentionMode",
    "DenseCppBlock",
    "DenseCppLM",
    "DenseCppLMConfig",
]
