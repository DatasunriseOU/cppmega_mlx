"""Correctness-first hybrid tiny LM assembly for local MLX smoke tests.

This module wires the existing local A/M/E/R reference blocks into one decoder
skeleton. It keeps NAM56R route intent visible, but it is not a full NAM56R
implementation and does not claim production kernel performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypedDict, cast

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.packing import mlx_document_boundary_mask
from cppmega_mlx.inference.engine import ContiguousKVCache, kv_cache_position
from cppmega_mlx.nn.attention import (
    AttentionConfig,
    CausalSelfAttention,
    sparse_mla_fp8_route_enabled,
)
from cppmega_mlx.nn.concept import ConceptBlock, ConceptBlockConfig
from cppmega_mlx.nn.engram import EngramBranch, EngramConfig
from cppmega_mlx.nn.m2rnn import M2RNNConfig, M2RNNMixer
from cppmega_mlx.nn.mamba3 import Mamba3Config, Mamba3ReferenceBlock
from cppmega_mlx.nn.mhc import ManifoldBranchMixer, ManifoldBranchMixerConfig
from cppmega_mlx.nn.moe import ActivationName, MoEConfig, ReferenceMoE
from cppmega_mlx.nn.ngram_hash import NgramHashEmbedding
from cppmega_mlx.nn.structure_embedding import CppMegaStructureEmbedding
from cppmega_mlx.recipes.pattern import ExpandedNamPattern, NamLayer, expand_nam_pattern
from cppmega_mlx.runtime.kernel_policy import KernelPath, selected_path
from cppmega_mlx.training.mtp import MinimalMTPHead, MTPLossConfig

HybridBackend = Literal["attention", "mamba3", "moe", "m2rnn", "engram", "concept"]
HybridBlockModule = (
    CausalSelfAttention
    | Mamba3ReferenceBlock
    | ReferenceMoE
    | M2RNNMixer
    | EngramBranch
    | ConceptBlock
)

_ROUTE_SYMBOL_BACKENDS: dict[str, HybridBackend] = {
    "A": "attention",
    "M": "mamba3",
    "E": "moe",
    "R": "m2rnn",
    "N": "engram",
    "C": "concept",
}

HybridAttentionMode = Literal["mla", "dsa", "full", "gqa"]


class StructureEmbeddingConfigKwargs(TypedDict):
    hidden_size: int
    num_categories: int
    max_dep_level: int
    max_ast_depth: int
    max_sibling_index: int
    num_node_types: int
    active_components: str
    bottleneck_dim: int


@dataclass(frozen=True)
class HybridTinyConfig:
    """Tiny local hybrid-LM config.

    Defaults are intentionally small and smoke-oriented. They are sized to make
    every currently implemented local block run together, not to match full
    NAM56R capacity, parallelism, caching, or custom kernel behavior.
    """

    vocab_size: int = 64
    hidden_size: int = 16
    pattern: str = "AEMR"
    depth: int = 4
    dsa_a_layer_ranks: tuple[int, ...] = ()
    num_attention_heads: int = 4
    num_attention_kv_heads: int | None = None
    attention_sparse_topk: int = 16
    max_seq_length: int = 16
    structure_vocab_size: int = 32
    structure_components: str = "core"
    structure_bottleneck_dim: int = 64
    structure_num_categories: int = 9
    structure_max_dep_level: int = 16
    structure_max_ast_depth: int = 64
    structure_max_sibling_index: int = 64
    structure_num_node_types: int = 256
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_expert_hidden_size: int = 32
    moe_shared_expert_hidden_size: int | None = 16
    moe_activation: ActivationName = "swiglu"
    mamba_expand: int = 2
    mamba_head_dim: int = 4
    mamba_state_dim: int = 4
    mamba_groups: int = 2
    mamba_mimo_rank: int = 1
    mamba_is_mimo: bool = False
    mamba_conv_kernel: int = 3
    mamba_chunk_size: int = 8
    mamba_rope_fraction: float = 0.5
    m2rnn_k_head_dim: int = 4
    m2rnn_v_head_dim: int = 4
    m2rnn_num_q_heads: int = 1
    m2rnn_num_k_heads: int = 1
    m2rnn_num_v_heads: int = 2
    m2rnn_num_f_heads: int = 2
    m2rnn_num_g_heads: int = 4
    m2rnn_num_weight_heads: int = 1
    m2rnn_conv_kernel: int = 4
    m2rnn_chunk_size: int = 8
    ngram_hash_enabled: bool = False
    ngram_hash_orders: tuple[int, ...] = (2, 3)
    ngram_hash_heads: int = 8
    ngram_hash_table_size: int = 500_000
    ngram_hash_embed_dim: int = 16
    ngram_hash_dropout: float = 0.0
    ngram_hash_seed: int | None = None
    mhc_enabled: bool = False
    grad_checkpoint: bool = False
    # Attention mode default applied to A-layers that did not get DSA routing.
    # "mla" preserves the legacy dense path. "full" and "gqa" are explicit
    # aliases for the same SDPA path with stricter num_kv_heads validation.
    attention_mode: HybridAttentionMode = "mla"
    # Engram (symbol "N") — local causal n-gram branch from cppmega_mlx.nn.engram.
    engram_ngram_orders: tuple[int, ...] = (2, 3, 4)
    engram_bottleneck_dim: int = 0
    engram_dropout: float = 0.0
    engram_gated: bool = False
    engram_conv_kernel: int = 0
    # Concept (symbol "C") — concept-retrieval cross-attention ported from
    # nanochat. ``concept_dim=None`` means use hidden_size.
    concept_num_concepts: int = 64
    concept_num_heads: int = 4
    concept_dim: int | None = None
    # MTP — Multi-Token Prediction. When enabled, an MTPHead is attached to the
    # model in HybridTinyLM.__init__ and made reachable as ``model.mtp_head``.
    mtp_enabled: bool = False
    mtp_depth: int = 2
    mtp_decay: float = 0.6
    mtp_loss_weight: float = 0.3
    mtp_ignore_index: int = -1

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_attention_kv_heads is not None and self.num_attention_kv_heads <= 0:
            raise ValueError("num_attention_kv_heads must be positive")
        if (
            self.num_attention_kv_heads is not None
            and self.num_attention_heads % self.num_attention_kv_heads != 0
        ):
            raise ValueError("num_attention_heads must be divisible by num_attention_kv_heads")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.attention_sparse_topk <= 0:
            raise ValueError("attention_sparse_topk must be positive")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        if self.structure_vocab_size < 2:
            raise ValueError("structure_vocab_size must be at least 2")
        if self.mamba_state_dim <= 0 or self.mamba_state_dim % 2:
            raise ValueError("mamba_state_dim must be a positive even integer")

        # Validate the route plan at config construction time, including DSA
        # ranks at tiny depths.
        self.expanded_pattern()
        # Validate the user-selected dense attention mode end-to-end.
        if self.attention_mode not in ("mla", "dsa", "full", "gqa"):
            raise ValueError(
                f"attention_mode must be one of 'mla', 'dsa', 'full', 'gqa', "
                f"got {self.attention_mode!r}"
            )
        self.attention_config(self.attention_mode)
        # Always also validate the dsa mode contract because dsa_a_layer_ranks
        # may pin specific A-layers to dsa regardless of attention_mode.
        if self.attention_mode != "dsa":
            self.attention_config("dsa")
        self.mamba3_config()
        self.m2rnn_config()
        self.moe_config()
        self.structure_embedding_config()
        self.ngram_hash_config()
        self.mhc_config()
        self.engram_config()
        self.concept_config()
        self.mtp_config()

    def expanded_pattern(self) -> ExpandedNamPattern:
        return expand_nam_pattern(
            self.pattern,
            self.depth,
            dsa_a_layer_ranks=self.dsa_a_layer_ranks,
        )

    def attention_config(
        self, mode: HybridAttentionMode | None = None
    ) -> AttentionConfig:
        active_mode: HybridAttentionMode = mode if mode is not None else self.attention_mode
        return AttentionConfig(
            d_model=self.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_kv_heads,
            mode=active_mode,
            sparse_topk=self.attention_sparse_topk,
        )

    def mamba3_config(self) -> Mamba3Config:
        return Mamba3Config(
            d_model=self.hidden_size,
            expand=self.mamba_expand,
            headdim=self.mamba_head_dim,
            d_state=self.mamba_state_dim,
            ngroups=self.mamba_groups,
            mimo_rank=self.mamba_mimo_rank,
            is_mimo=self.mamba_is_mimo,
            d_conv=self.mamba_conv_kernel,
            chunk_size=self.mamba_chunk_size,
            rope_fraction=self.mamba_rope_fraction,
        )

    def m2rnn_config(self) -> M2RNNConfig:
        return M2RNNConfig(
            d_model=self.hidden_size,
            k_head_dim=self.m2rnn_k_head_dim,
            v_head_dim=self.m2rnn_v_head_dim,
            num_q_heads=self.m2rnn_num_q_heads,
            num_k_heads=self.m2rnn_num_k_heads,
            num_v_heads=self.m2rnn_num_v_heads,
            num_f_heads=self.m2rnn_num_f_heads,
            num_g_heads=self.m2rnn_num_g_heads,
            num_weight_heads=self.m2rnn_num_weight_heads,
            conv_kernel=self.m2rnn_conv_kernel,
            chunk_size=self.m2rnn_chunk_size,
        )

    def moe_config(self) -> MoEConfig:
        return MoEConfig(
            d_model=self.hidden_size,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            expert_hidden_size=self.moe_expert_hidden_size,
            shared_expert_hidden_size=self.moe_shared_expert_hidden_size,
            activation=self.moe_activation,
        )

    def structure_embedding_config(self) -> StructureEmbeddingConfigKwargs:
        # Keep structure_vocab_size as legacy checkpoint/config metadata while
        # routing actual side channels through the source-equivalent module.
        if self.structure_vocab_size < 2:
            raise ValueError("structure_vocab_size must be at least 2")
        if self.structure_bottleneck_dim <= 0:
            raise ValueError("structure_bottleneck_dim must be positive")
        if self.structure_num_categories <= 0:
            raise ValueError("structure_num_categories must be positive")
        if self.structure_max_dep_level <= 0:
            raise ValueError("structure_max_dep_level must be positive")
        if self.structure_max_ast_depth <= 0:
            raise ValueError("structure_max_ast_depth must be positive")
        if self.structure_max_sibling_index <= 0:
            raise ValueError("structure_max_sibling_index must be positive")
        if self.structure_num_node_types <= 0:
            raise ValueError("structure_num_node_types must be positive")
        CppMegaStructureEmbedding._parse_components(self.structure_components)
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

    def ngram_hash_config(self) -> dict[str, object] | None:
        if not self.ngram_hash_enabled:
            return None
        if not self.ngram_hash_orders:
            raise ValueError("ngram_hash_orders must contain at least one n-gram order")
        if any(order <= 0 for order in self.ngram_hash_orders):
            raise ValueError("ngram_hash_orders must be positive")
        if self.ngram_hash_heads <= 0:
            raise ValueError("ngram_hash_heads must be positive")
        if self.ngram_hash_table_size <= 0:
            raise ValueError("ngram_hash_table_size must be positive")
        if self.ngram_hash_embed_dim <= 0:
            raise ValueError("ngram_hash_embed_dim must be positive")
        if not 0.0 <= self.ngram_hash_dropout < 1.0:
            raise ValueError("ngram_hash_dropout must be in [0, 1)")
        return {
            "hidden_size": self.hidden_size,
            "orders": self.ngram_hash_orders,
            "num_heads": self.ngram_hash_heads,
            "table_size": self.ngram_hash_table_size,
            "embed_dim": self.ngram_hash_embed_dim,
            "dropout": self.ngram_hash_dropout,
            "seed": self.ngram_hash_seed,
        }

    def mhc_config(self) -> ManifoldBranchMixerConfig | None:
        if not self.mhc_enabled:
            return None
        return ManifoldBranchMixerConfig(hidden_size=self.hidden_size, max_branches=2)

    def engram_config(self) -> EngramConfig:
        return EngramConfig(
            hidden_size=self.hidden_size,
            ngram_orders=self.engram_ngram_orders,
            bottleneck_dim=self.engram_bottleneck_dim,
            dropout=self.engram_dropout,
            gated=self.engram_gated,
            conv_kernel=self.engram_conv_kernel,
        )

    def concept_config(self) -> ConceptBlockConfig:
        return ConceptBlockConfig(
            hidden_size=self.hidden_size,
            num_concepts=self.concept_num_concepts,
            num_heads=self.concept_num_heads,
            concept_dim=self.concept_dim,
        )

    def mtp_config(self) -> MTPLossConfig | None:
        if not self.mtp_enabled:
            return None
        return MTPLossConfig(
            depth=self.mtp_depth,
            decay=self.mtp_decay,
            loss_weight=self.mtp_loss_weight,
            ignore_index=self.mtp_ignore_index,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize this config to a YAML document.

        Lists are emitted as flow-style for compactness. Use ``from_yaml`` to
        parse the output back into a ``HybridTinyConfig``.
        """
        import yaml  # PyYAML is part of the existing dev requirements.

        return yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            default_flow_style=None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> HybridTinyConfig:
        """Build a config from a plain dict (e.g., loaded YAML/JSON)."""

        coerced = dict(data)
        # YAML and JSON lose tuple-ness; coerce sequence-typed fields back.
        for field_name in (
            "dsa_a_layer_ranks",
            "ngram_hash_orders",
            "engram_ngram_orders",
        ):
            if field_name in coerced and coerced[field_name] is not None:
                coerced[field_name] = tuple(coerced[field_name])  # type: ignore[arg-type]
        return cls(**coerced)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, text: str) -> HybridTinyConfig:
        """Parse a YAML document into a ``HybridTinyConfig`` with validation."""

        import yaml

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(
                "from_yaml expects a YAML mapping at the top level, "
                f"got {type(loaded).__name__}"
            )
        return cls.from_dict(loaded)


class HybridTinyBlock(nn.Module):
    """One pre-norm residual A/M/E/R route block.

    The active sub-module is held under a single attribute (``self.block``)
    so MLX's ``tree_flatten`` walks every parameter exactly once. The legacy
    ``attention_block`` / ``mamba3_block`` / ``moe_block`` / ``m2rnn_block``
    accessors are exposed as plain Python ``@property`` so they do not
    introduce a second path into the parameter tree (which would otherwise
    double the optimizer state and gradient buffer cost — see
    ``docs/research/precision_strategy_decision.md``).
    """

    def __init__(self, layer: NamLayer, config: HybridTinyConfig):
        super().__init__()
        self.layer = layer
        self.norm = nn.RMSNorm(config.hidden_size)
        mhc_config = config.mhc_config()
        self.mhc: ManifoldBranchMixer | None = (
            ManifoldBranchMixer(mhc_config) if mhc_config is not None else None
        )
        self.block: HybridBlockModule

        if layer.symbol == "A":
            # DSA pinning via dsa_a_layer_ranks always wins. Otherwise the
            # model-wide attention_mode (default 'mla', or user-chosen
            # 'full'/'gqa') applies to this A-layer.
            mode: HybridAttentionMode = (
                "dsa" if layer.attention_route == "dsa" else config.attention_mode
            )
            self.backend: HybridBackend = "attention"
            self.block = CausalSelfAttention(config.attention_config(mode))
        elif layer.symbol == "M":
            self.backend = "mamba3"
            self.block = Mamba3ReferenceBlock(config.mamba3_config())
        elif layer.symbol == "E":
            self.backend = "moe"
            self.block = ReferenceMoE(config.moe_config())
        elif layer.symbol == "R":
            self.backend = "m2rnn"
            self.block = M2RNNMixer(config.m2rnn_config())
        elif layer.symbol == "N":
            self.backend = "engram"
            self.block = EngramBranch(config.engram_config())
        elif layer.symbol == "C":
            self.backend = "concept"
            self.block = ConceptBlock(config.concept_config())
        else:  # pragma: no cover - expand_nam_pattern rejects this first.
            raise ValueError(f"unsupported hybrid layer symbol {layer.symbol!r}")

    @property
    def attention_block(self) -> CausalSelfAttention | None:
        return self.block if self.backend == "attention" else None  # type: ignore[return-value]

    @property
    def mamba3_block(self) -> Mamba3ReferenceBlock | None:
        return self.block if self.backend == "mamba3" else None  # type: ignore[return-value]

    @property
    def moe_block(self) -> ReferenceMoE | None:
        return self.block if self.backend == "moe" else None  # type: ignore[return-value]

    @property
    def m2rnn_block(self) -> M2RNNMixer | None:
        return self.block if self.backend == "m2rnn" else None  # type: ignore[return-value]

    @property
    def engram_block(self) -> EngramBranch | None:
        return self.block if self.backend == "engram" else None  # type: ignore[return-value]

    @property
    def concept_block(self) -> ConceptBlock | None:
        return self.block if self.backend == "concept" else None  # type: ignore[return-value]

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | Literal["causal"] | None,
        *,
        kv_cache: ContiguousKVCache | None = None,
        attention_layer_idx: int | None = None,
    ) -> mx.array:
        self.validate_backend()
        residual = hidden_states
        delta = self.route_delta(
            hidden_states,
            mask,
            kv_cache=kv_cache,
            attention_layer_idx=attention_layer_idx,
        )
        updated = residual + delta
        if self.mhc is not None:
            return self.mhc([updated, residual])
        return updated

    def validate_backend(self) -> None:
        """Fail closed if route metadata and the active module diverge."""

        expected_backend = _ROUTE_SYMBOL_BACKENDS.get(self.layer.symbol)
        if expected_backend is None:
            raise ValueError(f"unsupported hybrid layer symbol {self.layer.symbol!r}")
        if self.backend != expected_backend:
            raise ValueError(
                f"hybrid layer {self.layer.number} symbol {self.layer.symbol!r} "
                f"requires backend {expected_backend!r}, got {self.backend!r}"
            )

        if self.block is None:
            raise ValueError(f"{self.backend} backend missing block instance")

        expected_cls = {
            "attention": CausalSelfAttention,
            "mamba3": Mamba3ReferenceBlock,
            "moe": ReferenceMoE,
            "m2rnn": M2RNNMixer,
            "engram": EngramBranch,
            "concept": ConceptBlock,
        }[self.backend]
        if not isinstance(self.block, expected_cls):
            raise ValueError(
                f"hybrid layer {self.layer.number} backend {self.backend!r} "
                f"has block of unexpected class {type(self.block).__name__!r}"
            )

    def route_delta(
        self,
        hidden_states: mx.array,
        mask: mx.array | Literal["causal"] | None,
        *,
        kv_cache: ContiguousKVCache | None = None,
        attention_layer_idx: int | None = None,
    ) -> mx.array:
        """Return this route's pre-residual contribution for regression tests."""

        self.validate_backend()
        if kv_cache is not None and self.backend != "attention":
            raise ValueError("kv_cache may only be passed to attention route blocks")
        x = self.norm(hidden_states)
        if self.backend == "attention":
            delta = cast(CausalSelfAttention, self.block)(
                x,
                mask,
                kv_cache=kv_cache,
                layer_idx=attention_layer_idx,
            )
        elif self.backend == "mamba3":
            delta, _ = cast(Mamba3ReferenceBlock, self.block)(x)
        elif self.backend == "moe":
            delta = cast(ReferenceMoE, self.block)(x).output
        elif self.backend == "m2rnn":
            m2rnn = cast(M2RNNMixer, self.block)
            if selected_path("m2rnn") is KernelPath.PATH_C:
                delta, _ = m2rnn(
                    x,
                    h0=m2rnn.initial_h0(x.shape[0], x.dtype),
                )
            else:
                delta, _ = m2rnn(x)
        elif self.backend == "engram":
            delta = cast(EngramBranch, self.block)(x)
        elif self.backend == "concept":
            delta = cast(ConceptBlock, self.block)(x)
        else:  # pragma: no cover - self.backend is fixed during construction.
            raise ValueError(f"unsupported hybrid backend {self.backend!r}")
        return delta


class HybridTinyLM(nn.Module):
    """Tiny decoder-only LM assembled from local NAM A/M/E/R reference blocks."""

    def __init__(
        self,
        config: HybridTinyConfig | None = None,
        *,
        dtype: mx.Dtype | None = None,
    ):
        super().__init__()
        self.config = config or HybridTinyConfig()
        cfg = self.config
        self.pattern = cfg.expanded_pattern()

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_seq_length, cfg.hidden_size)
        self.structure_embedding = CppMegaStructureEmbedding(**cfg.structure_embedding_config())
        self.ngram_hash_embedding = None
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
        self.layers = [HybridTinyBlock(layer, cfg) for layer in self.pattern.layers]
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        mtp_config = cfg.mtp_config()
        self.mtp_head: MinimalMTPHead | None = (
            MinimalMTPHead(self.token_embedding, self.lm_head, config=mtp_config)
            if mtp_config is not None
            else None
        )

        if dtype is not None and dtype != mx.float32:
            self.set_dtype(dtype)

    @property
    def route_symbols(self) -> tuple[str, ...]:
        return tuple(layer.symbol for layer in self.pattern.layers)

    @property
    def route_roles(self) -> tuple[str, ...]:
        return tuple(layer.role for layer in self.pattern.layers)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        document_ids: mx.array | None = None,
        kv_cache: ContiguousKVCache | None = None,
    ) -> mx.array:
        return self.lm_head(
            self.decoder_hidden_states(
                input_ids,
                structure_ids=structure_ids,
                dep_levels=dep_levels,
                ast_depth_ids=ast_depth_ids,
                sibling_index_ids=sibling_index_ids,
                node_type_ids=node_type_ids,
                document_ids=document_ids,
                kv_cache=kv_cache,
            )
        )

    def decoder_hidden_states(
        self,
        input_ids: mx.array,
        *,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        document_ids: mx.array | None = None,
        kv_cache: ContiguousKVCache | None = None,
    ) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be shaped (B, S), got {input_ids.shape}")

        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        position_offset = 0
        if kv_cache is not None:
            if self.config.grad_checkpoint:
                raise ValueError("kv_cache is incompatible with grad_checkpoint")
            if document_ids is not None:
                raise ValueError("document_ids are incompatible with kv_cache decode")
            if kv_cache.config.batch_size != batch_size:
                raise ValueError("kv_cache batch_size must match input_ids batch size")
            attention_layer_count = _attention_layer_count(self.layers)
            if kv_cache.config.num_layers != attention_layer_count:
                raise ValueError(
                    "kv_cache num_layers must match the number of attention layers"
                )
            position_offset = kv_cache_position(kv_cache)

        if position_offset + seq_length > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {position_offset + seq_length} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )

        positions = (position_offset + mx.arange(seq_length))[None, :]
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        if self.ngram_hash_embedding is not None:
            hidden_states = hidden_states + self.ngram_hash_embedding(input_ids)

        structure_embeddings = self.structure_embedding(
            structure_ids=_validate_side_channel_shape(
                "structure_ids", structure_ids, batch_size, seq_length
            ),
            dep_levels=_validate_side_channel_shape(
                "dep_levels", dep_levels, batch_size, seq_length
            ),
            ast_depth_ids=_validate_side_channel_shape(
                "ast_depth_ids", ast_depth_ids, batch_size, seq_length
            ),
            sibling_index_ids=_validate_side_channel_shape(
                "sibling_index_ids", sibling_index_ids, batch_size, seq_length
            ),
            node_type_ids=_validate_side_channel_shape(
                "node_type_ids", node_type_ids, batch_size, seq_length
            ),
            target_dtype=hidden_states.dtype,
        )
        if structure_embeddings.ndim == hidden_states.ndim:
            hidden_states = hidden_states + structure_embeddings

        document_ids = _validate_document_ids(
            document_ids,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        mask: mx.array | Literal["causal"] | None = None
        if any(layer.backend == "attention" for layer in self.layers):
            if document_ids is None:
                if kv_cache is None:
                    dsa_path_c = (
                        selected_path("sparse_mla") is KernelPath.PATH_C
                        and sparse_mla_fp8_route_enabled(KernelPath.PATH_C)
                        and any(
                            layer.backend == "attention"
                            and isinstance(layer.block, CausalSelfAttention)
                            and layer.block.config.mode == "dsa"
                            for layer in self.layers
                        )
                    )
                    if dsa_path_c:
                        mask = "causal"
                    else:
                        mask = nn.MultiHeadAttention.create_additive_causal_mask(
                            seq_length,
                            dtype=hidden_states.dtype,
                        )
            else:
                mask = mlx_document_boundary_mask(
                    document_ids,
                    causal=True,
                    expand_heads=True,
                )
        if self.config.grad_checkpoint:
            for layer in self.layers:
                hidden_states = mx.checkpoint(layer)(hidden_states, mask)
        else:
            attention_layer_idx = 0
            for layer in self.layers:
                if layer.backend == "attention":
                    hidden_states = layer(
                        hidden_states,
                        mask,
                        kv_cache=kv_cache,
                        attention_layer_idx=attention_layer_idx if kv_cache is not None else None,
                    )
                    attention_layer_idx += 1
                else:
                    hidden_states = layer(hidden_states, mask)
        return self.norm(hidden_states)


def _validate_side_channel_shape(
    name: str,
    tensor: mx.array | None,
    batch_size: int,
    seq_length: int,
) -> mx.array | None:
    if tensor is None:
        return None
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be shaped (B, S), got {tensor.shape}")
    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch dimension {tensor.shape[0]} must match input batch {batch_size}"
        )
    if tensor.shape[1] != seq_length:
        raise ValueError(
            f"{name} shape {tensor.shape} must exactly match input_ids shape "
            f"({batch_size}, {seq_length})"
        )
    return tensor


def _validate_document_ids(
    document_ids: mx.array | None,
    *,
    batch_size: int,
    seq_length: int,
) -> mx.array | None:
    if document_ids is None:
        return None
    if document_ids.ndim != 2:
        raise ValueError(f"document_ids must be shaped (B, S), got {document_ids.shape}")
    if document_ids.shape != (batch_size, seq_length):
        raise ValueError(
            f"document_ids shape {document_ids.shape} must exactly match input_ids shape "
            f"({batch_size}, {seq_length})"
        )
    has_negative = mx.any(document_ids.astype(mx.int32) < 0)
    mx.eval(has_negative)
    if bool(has_negative.item()):
        raise ValueError("document_ids must be non-negative for explicit packed batches")
    return document_ids.astype(mx.int32)


def _attention_layer_count(layers: list[HybridTinyBlock]) -> int:
    return sum(1 for layer in layers if layer.backend == "attention")

__all__ = [
    "HybridAttentionMode",
    "HybridBackend",
    "HybridBlockModule",
    "HybridTinyBlock",
    "HybridTinyConfig",
    "HybridTinyLM",
]
