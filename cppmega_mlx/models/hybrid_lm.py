"""Correctness-first hybrid tiny LM assembly for local MLX smoke tests.

This module wires the existing local A/M/E/R reference blocks into one decoder
skeleton. It keeps NAM56R route intent visible, but it is not a full NAM56R
implementation and does not claim production kernel performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypedDict

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.packing import mlx_document_boundary_mask
from cppmega_mlx.nn.attention import AttentionConfig, CausalSelfAttention
from cppmega_mlx.nn.m2rnn import M2RNNConfig, M2RNNMixer
from cppmega_mlx.nn.mamba3 import Mamba3Config, Mamba3ReferenceBlock
from cppmega_mlx.nn.moe import ActivationName, MoEConfig, ReferenceMoE
from cppmega_mlx.nn.ngram_hash import NgramHashEmbedding
from cppmega_mlx.nn.structure_embedding import CppMegaStructureEmbedding
from cppmega_mlx.recipes.pattern import ExpandedNamPattern, NamLayer, expand_nam_pattern

HybridBackend = Literal["attention", "mamba3", "moe", "m2rnn"]
HybridBlockModule = CausalSelfAttention | Mamba3ReferenceBlock | ReferenceMoE | M2RNNMixer

_ROUTE_SYMBOL_BACKENDS: dict[str, HybridBackend] = {
    "A": "attention",
    "M": "mamba3",
    "E": "moe",
    "R": "m2rnn",
}


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

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        if self.structure_vocab_size < 2:
            raise ValueError("structure_vocab_size must be at least 2")
        if self.mamba_state_dim <= 0 or self.mamba_state_dim % 2:
            raise ValueError("mamba_state_dim must be a positive even integer")

        # Validate the route plan at config construction time, including DSA
        # ranks at tiny depths.
        self.expanded_pattern()
        self.attention_config("mla")
        self.mamba3_config()
        self.m2rnn_config()
        self.moe_config()
        self.structure_embedding_config()
        self.ngram_hash_config()

    def expanded_pattern(self) -> ExpandedNamPattern:
        return expand_nam_pattern(
            self.pattern,
            self.depth,
            dsa_a_layer_ranks=self.dsa_a_layer_ranks,
        )

    def attention_config(self, mode: Literal["mla", "dsa"]) -> AttentionConfig:
        return AttentionConfig(
            d_model=self.hidden_size,
            num_q_heads=self.num_attention_heads,
            mode=mode,
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HybridTinyBlock(nn.Module):
    """One pre-norm residual A/M/E/R route block."""

    def __init__(self, layer: NamLayer, config: HybridTinyConfig):
        super().__init__()
        self.layer = layer
        self.norm = nn.RMSNorm(config.hidden_size)
        self.attention_block: CausalSelfAttention | None = None
        self.mamba3_block: Mamba3ReferenceBlock | None = None
        self.moe_block: ReferenceMoE | None = None
        self.m2rnn_block: M2RNNMixer | None = None
        self.block: HybridBlockModule

        if layer.symbol == "A":
            mode = layer.attention_route or "mla"
            self.backend: HybridBackend = "attention"
            self.attention_block = CausalSelfAttention(config.attention_config(mode))
            self.block = self.attention_block
        elif layer.symbol == "M":
            self.backend = "mamba3"
            self.mamba3_block = Mamba3ReferenceBlock(config.mamba3_config())
            self.block = self.mamba3_block
        elif layer.symbol == "E":
            self.backend = "moe"
            self.moe_block = ReferenceMoE(config.moe_config())
            self.block = self.moe_block
        elif layer.symbol == "R":
            self.backend = "m2rnn"
            self.m2rnn_block = M2RNNMixer(config.m2rnn_config())
            self.block = self.m2rnn_block
        else:  # pragma: no cover - expand_nam_pattern rejects this first.
            raise ValueError(f"unsupported hybrid layer symbol {layer.symbol!r}")

    def __call__(
        self,
        hidden_states: mx.array,
        mask: mx.array | Literal["causal"] | None,
    ) -> mx.array:
        self.validate_backend()
        residual = hidden_states
        delta = self.route_delta(hidden_states, mask)
        return residual + delta

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

        active_modules = {
            "attention": self.attention_block,
            "mamba3": self.mamba3_block,
            "moe": self.moe_block,
            "m2rnn": self.m2rnn_block,
        }
        if active_modules[self.backend] is None:
            raise ValueError(f"{self.backend} backend missing {self.backend} block")

        unexpected = [
            backend
            for backend, module in active_modules.items()
            if backend != self.backend and module is not None
        ]
        if unexpected:
            raise ValueError(
                f"hybrid layer {self.layer.number} backend {self.backend!r} "
                f"has unexpected route modules {tuple(unexpected)!r}"
            )

    def route_delta(
        self,
        hidden_states: mx.array,
        mask: mx.array | Literal["causal"] | None,
    ) -> mx.array:
        """Return this route's pre-residual contribution for regression tests."""

        self.validate_backend()
        x = self.norm(hidden_states)
        if self.backend == "attention":
            if self.attention_block is None:  # pragma: no cover - constructor invariant.
                raise ValueError("attention backend missing attention block")
            delta = self.attention_block(x, mask)
        elif self.backend == "mamba3":
            if self.mamba3_block is None:  # pragma: no cover - constructor invariant.
                raise ValueError("mamba3 backend missing mamba3 block")
            delta, _ = self.mamba3_block(x)
        elif self.backend == "moe":
            if self.moe_block is None:  # pragma: no cover - constructor invariant.
                raise ValueError("moe backend missing moe block")
            delta = self.moe_block(x).output
        elif self.backend == "m2rnn":
            if self.m2rnn_block is None:  # pragma: no cover - constructor invariant.
                raise ValueError("m2rnn backend missing m2rnn block")
            delta, _ = self.m2rnn_block(x)
        else:  # pragma: no cover - self.backend is fixed during construction.
            raise ValueError(f"unsupported hybrid backend {self.backend!r}")
        return delta


class HybridTinyLM(nn.Module):
    """Tiny decoder-only LM assembled from local NAM A/M/E/R reference blocks."""

    def __init__(self, config: HybridTinyConfig | None = None):
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
    ) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be shaped (B, S), got {input_ids.shape}")

        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        if seq_length > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {seq_length} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )

        positions = mx.arange(seq_length)[None, :]
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
        mask = None
        if any(layer.backend == "attention" for layer in self.layers):
            if document_ids is None:
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
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return self.lm_head(self.norm(hidden_states))


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

__all__ = [
    "HybridBackend",
    "HybridBlockModule",
    "HybridTinyBlock",
    "HybridTinyConfig",
    "HybridTinyLM",
]
