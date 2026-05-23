"""Correctness-first hybrid tiny LM assembly for local MLX smoke tests.

This module wires the existing local A/M/E/R reference blocks into one decoder
skeleton. It keeps NAM56R route intent visible, but it is not a full NAM56R
implementation and does not claim production kernel performance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    Sequence,
    TypedDict,
    cast,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cppmega_mlx.data.platform_context import MAX_PLATFORM_IDS, PLATFORM_VOCAB_SIZE
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
from cppmega_mlx.nn.platform_embedding import CppMegaPlatformEmbedding
from cppmega_mlx.nn.structure_embedding import CppMegaStructureEmbedding
from cppmega_mlx.recipes.pattern import ExpandedNamPattern, NamLayer, expand_nam_pattern
from cppmega_mlx.runtime.kernel_policy import KernelPath, selected_path
from cppmega_mlx.runtime.path_c_physical_abi import (
    PathCLogicalBufferOwner,
    PathCPhysicalAbiBankOwner,
    logical_bank_view,
    make_physical_abi_bank_owner,
    write_into_bank_slot,
)
from cppmega_mlx.runtime.path_c_taps import emit_and_tap_path_c_tensor
from cppmega_mlx.training.mtp import MinimalMTPHead, MTPLossConfig

if TYPE_CHECKING:
    from cppmega_mlx.runtime.path_c_fusion import PathCFusionRegion

HybridBackend = Literal["attention", "mamba3", "moe", "m2rnn", "engram", "concept"]
HybridBlockModule = (
    CausalSelfAttention
    | Mamba3ReferenceBlock
    | ReferenceMoE
    | M2RNNMixer
    | EngramBranch
    | ConceptBlock
)

HYBRID_SIDE_CHANNEL_RESIDUAL_SCALE_FAMILIES = ("platform", "structure", "syntax")
PathCActivationProbe = Callable[[Mapping[str, Any]], None]

_ROUTE_SYMBOL_BACKENDS: dict[str, HybridBackend] = {
    "A": "attention",
    "M": "mamba3",
    "E": "moe",
    "R": "m2rnn",
    "N": "engram",
    "C": "concept",
}

HybridAttentionMode = Literal["mla", "dsa", "full", "gqa"]

_PATH_C_BANK_DTYPES: dict[str, mx.Dtype] = {
    "bool": mx.bool_,
    "uint8": mx.uint8,
    "int8": mx.int8,
    "uint16": mx.uint16,
    "int16": mx.int16,
    "uint32": mx.uint32,
    "int32": mx.int32,
    "uint64": mx.uint64,
    "int64": mx.int64,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def _path_c_bank_dtype(name: str) -> mx.Dtype:
    """Translate a Path C physical-ABI dtype string into an ``mx.Dtype``."""
    try:
        return _PATH_C_BANK_DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Path C physical ABI bank dtype {name!r}"
        ) from exc




class StructureEmbeddingConfigKwargs(TypedDict):
    hidden_size: int
    num_categories: int
    max_dep_level: int
    max_ast_depth: int
    max_sibling_index: int
    num_node_types: int
    active_components: str
    bottleneck_dim: int


class PathCActivationBufferCapture:
    """Opt-in Path C activation owner that stores references, not copies."""

    def __init__(
        self,
        aliases: Mapping[str, str | Sequence[str]] | None = None,
        *,
        owner_name: str = "HybridTinyLM.path_c_activation_capture",
        capture_gradients: bool = True,
    ) -> None:
        self.owner_name = owner_name
        self.capture_gradients = capture_gradients
        self.aliases = {
            str(source): (str(target),)
            if isinstance(target, str)
            else tuple(str(item) for item in target)
            for source, target in (aliases or {}).items()
        }
        self.buffers: dict[str, mx.array] = {}
        self.events: list[Mapping[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        tensor = event.get("tensor")
        if not isinstance(tensor, mx.array):
            return
        logical_names = tuple(str(name) for name in event.get("logical_names", ()))
        for name in logical_names:
            self.buffers[name] = tensor
            aliases = list(self.aliases.get(name, ()))
            if name.endswith("_grad"):
                aliases.extend(
                    f"{alias}_grad"
                    for alias in self.aliases.get(name[: -len("_grad")], ())
                )
            for alias in aliases:
                self.buffers[alias] = tensor
        self.events.append(_path_c_capture_event_metadata(event))

    def clear(self) -> None:
        self.buffers.clear()
        self.events.clear()


def _path_c_capture_event_metadata(event: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in event.items():
        if isinstance(value, mx.array):
            metadata[f"{key}_shape"] = tuple(int(dim) for dim in value.shape)
            metadata[f"{key}_dtype"] = str(value.dtype)
        else:
            metadata[key] = value
    return metadata


def _normalize_side_channel_residual_scale(
    policy: Mapping[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    allowed = set(HYBRID_SIDE_CHANNEL_RESIDUAL_SCALE_FAMILIES)
    for raw_family, raw_scale in policy.items():
        family = str(raw_family).strip()
        if not family:
            raise ValueError("side_channel_residual_scale family names must be non-empty")
        if family not in allowed:
            allowed_list = ", ".join(HYBRID_SIDE_CHANNEL_RESIDUAL_SCALE_FAMILIES)
            raise ValueError(
                "side_channel_residual_scale only supports model-consumed "
                f"families: {allowed_list}; got {family!r}"
            )
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"side_channel_residual_scale for {family!r} must be finite and >= 0"
            )
        normalized[family] = scale
    if (
        "structure" in normalized
        and "syntax" in normalized
        and not math.isclose(normalized["structure"], normalized["syntax"])
    ):
        raise ValueError(
            "structure and syntax currently share a single structure residual; "
            "set one side_channel_residual_scale value or matching values"
        )
    return dict(sorted(normalized.items()))


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
    platform_vocab_size: int = PLATFORM_VOCAB_SIZE
    platform_max_ids: int = MAX_PLATFORM_IDS
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
    side_channel_residual_scale: dict[str, float] = field(default_factory=dict)
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
        object.__setattr__(
            self,
            "side_channel_residual_scale",
            _normalize_side_channel_residual_scale(self.side_channel_residual_scale),
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
        self.platform_embedding_config()
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

    def platform_embedding_config(self) -> dict[str, int]:
        if self.platform_vocab_size <= 1:
            raise ValueError("platform_vocab_size must be greater than one")
        if self.platform_max_ids <= 0:
            raise ValueError("platform_max_ids must be positive")
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.platform_vocab_size,
            "max_ids": self.platform_max_ids,
        }

    def residual_scale_for(self, family: str) -> float:
        if family == "structure":
            return self.side_channel_residual_scale.get(
                "structure",
                self.side_channel_residual_scale.get("syntax", 1.0),
            )
        return self.side_channel_residual_scale.get(family, 1.0)

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
        self.path_c_layer_index = layer.number - 1
        self.path_c_brick_name = f"layer_{self.path_c_layer_index}_{layer.symbol.lower()}"
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

    def _path_c_activation_probe(self) -> PathCActivationProbe | None:
        probe = getattr(self, "_path_c_activation_probe_callback", None)
        return cast(PathCActivationProbe, probe) if callable(probe) else None

    def _path_c_activation_logical_names(self, name: str) -> tuple[str, ...]:
        brick_name = str(getattr(self, "path_c_profile_brick_name", self.path_c_brick_name))
        if name == "hidden":
            return (f"{brick_name}_hidden",)
        if name == "normed":
            return (f"{brick_name}_residual_norm_hidden",)
        if name == "delta":
            return (f"{brick_name}_delta",)
        if name == "hidden_after":
            return (f"{brick_name}_hidden_after",)
        return (f"{brick_name}_{name}",)

    def _emit_path_c_activation(self, name: str, tensor: mx.array) -> mx.array:
        probe = self._path_c_activation_probe()
        if probe is None:
            return tensor
        return emit_and_tap_path_c_tensor(
            tensor,
            probe=probe,
            event={
                "name": name,
                "logical_names": self._path_c_activation_logical_names(name),
                "layer_number": self.layer.number,
                "layer_index": self.path_c_layer_index,
                "route_symbol": self.layer.symbol,
                "backend": self.backend,
                "brick_name": self.path_c_brick_name,
                "profile_brick_name": getattr(self, "path_c_profile_brick_name", None),
            },
        )

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
        doc_ids: mx.array | None = None,
    ) -> mx.array:
        self.validate_backend()
        residual = hidden_states
        residual = self._emit_path_c_activation("hidden", residual)
        delta = self.route_delta(
            hidden_states,
            mask,
            kv_cache=kv_cache,
            attention_layer_idx=attention_layer_idx,
            doc_ids=doc_ids,
        )
        delta = self._emit_path_c_activation("delta", delta)
        updated = residual + delta
        updated = self._emit_path_c_activation("hidden_after", updated)
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
        doc_ids: mx.array | None = None,
    ) -> mx.array:
        """Return this route's pre-residual contribution for regression tests.

        ``doc_ids`` is the raw ``(B, S)`` int32 document boundary tensor. Only
        the ``engram`` backend currently consumes it (to prevent n-gram
        aggregation from crossing packed document boundaries). The
        ``attention`` backend has its own additive-mask channel (``mask``) for
        the same purpose, and the remaining backends (``mamba3``, ``moe``,
        ``m2rnn``, ``concept``) do not yet support document-aware routing in
        this repo, so they silently ignore ``doc_ids``.
        """

        self.validate_backend()
        if kv_cache is not None and self.backend != "attention":
            raise ValueError("kv_cache may only be passed to attention route blocks")
        x = self.norm(hidden_states)
        x = self._emit_path_c_activation("normed", x)
        if self.backend == "attention":
            delta = cast(CausalSelfAttention, self.block)(
                x,
                mask,
                kv_cache=kv_cache,
                layer_idx=attention_layer_idx,
            )
        elif self.backend == "mamba3":
            mamba3 = cast(Mamba3ReferenceBlock, self.block)
            probe = self._path_c_activation_probe()
            if probe is not None:
                h0 = mamba3.initial_h0(x.shape[0], x.dtype)
                h0 = self._emit_path_c_activation("mamba3_h0", h0)
                h0 = self._emit_path_c_activation("state_in", h0)
                delta, state = mamba3(x, h0=h0)
                self._emit_path_c_activation("state", state)
            else:
                delta, _ = mamba3(x)
        elif self.backend == "moe":
            delta = cast(ReferenceMoE, self.block)(x).output
        elif self.backend == "m2rnn":
            m2rnn = cast(M2RNNMixer, self.block)
            use_explicit_state = (
                selected_path("m2rnn") is KernelPath.PATH_C
                or self._path_c_activation_probe() is not None
            )
            if use_explicit_state:
                h0 = m2rnn.initial_h0(x.shape[0], x.dtype)
                h0 = self._emit_path_c_activation("m2rnn_h0", h0)
                delta, state = m2rnn(
                    x,
                    h0=h0,
                    return_state=True,
                )
                self._emit_path_c_activation("m2rnn_conv_state", state.conv_state)
            else:
                delta, _ = m2rnn(x)
        elif self.backend == "engram":
            delta = cast(EngramBranch, self.block)(x, doc_ids=doc_ids)
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
        self.platform_embedding = CppMegaPlatformEmbedding(**cfg.platform_embedding_config())
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

    @property
    def path_c_bricks(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "name": str(
                    getattr(
                        block,
                        "path_c_profile_brick_name",
                        f"layer_{index}_{block.layer.symbol.lower()}",
                    )
                ),
                "kind": block.backend,
                "route_symbol": block.layer.symbol,
                **(
                    {
                        "attention_qkv_has_bias": str(
                            bool(block.attention_block.config.bias)
                        ).lower(),
                        "attention_out_proj_has_bias": str(
                            bool(block.attention_block.config.bias)
                        ).lower(),
                    }
                    if block.attention_block is not None
                    else {}
                ),
            }
            for index, block in enumerate(self.layers)
        )

    def attach_path_c_activation_probe(self, probe: PathCActivationProbe) -> int:
        """Install an explicit zero-copy activation probe on route blocks."""

        if not callable(probe):
            raise TypeError("probe must be callable")
        for block in self.layers:
            block._path_c_activation_probe_callback = probe  # type: ignore[attr-defined]
            attention = block.attention_block
            if attention is not None and callable(
                getattr(attention, "attach_path_c_prepared_probe", None)
            ):
                attention.attach_path_c_prepared_probe(
                    probe,
                    logical_prefix=str(
                        getattr(
                            block,
                            "path_c_profile_brick_name",
                            block.path_c_brick_name,
                        )
                    ),
                )
        return len(self.layers)

    def detach_path_c_activation_probe(self) -> None:
        """Remove the explicit Path C activation probe from all route blocks."""

        for block in self.layers:
            if hasattr(block, "_path_c_activation_probe_callback"):
                delattr(block, "_path_c_activation_probe_callback")
            attention = block.attention_block
            if attention is not None and callable(
                getattr(attention, "detach_path_c_prepared_probe", None)
            ):
                attention.detach_path_c_prepared_probe()

    def path_c_parameter_logical_aliases(self) -> dict[str, tuple[str, ...]]:
        """Map MLX parameter tree names to Path C logical input names."""

        parameter_names = {
            name
            for name, value in tree_flatten(self.trainable_parameters())
            if isinstance(value, mx.array)
        }
        aliases: dict[str, tuple[str, ...]] = {}

        def add_parameter(
            parameter_name: str,
            logical_suffix: str,
            *,
            block: HybridTinyBlock,
        ) -> None:
            if parameter_name not in parameter_names:
                return
            logical_prefixes = [block.path_c_brick_name]
            profile_name = getattr(block, "path_c_profile_brick_name", None)
            if profile_name is not None and profile_name not in logical_prefixes:
                logical_prefixes.append(str(profile_name))
            # Block A: accumulate candidates instead of overwriting so the
            # same MLX parameter (e.g. ``layers.{i}.norm.weight``) can
            # carry both its inter-brick bridge binding and its
            # per-brick entry-RMSNorm binding. The in-region resolver
            # picks the candidate that's actually present in the ABI map
            # for the given brick.
            new_candidates = tuple(
                f"{prefix}_{logical_suffix}" for prefix in logical_prefixes
            )
            existing = aliases.get(parameter_name, ())
            merged = list(existing)
            for candidate in new_candidates:
                if candidate not in merged:
                    merged.append(candidate)
            aliases[parameter_name] = tuple(merged)

        def add(
            layer_index: int,
            parameter_suffix: str,
            logical_suffix: str,
            *,
            block: HybridTinyBlock,
        ) -> None:
            add_parameter(
                f"layers.{layer_index}.{parameter_suffix}",
                logical_suffix,
                block=block,
            )

        for index, block in enumerate(self.layers[:-1]):
            add_parameter(
                f"layers.{index + 1}.norm.weight",
                "residual_norm_weight",
                block=block,
            )

        # Block A: every block also owns an "entry RMSNorm" candidate so the
        # first in-region brick (whichever layer that ends up being) has its
        # `layers.{i}.norm.weight` parameter routed into the fused region.
        # The in-region resolver picks this candidate only when the
        # corresponding `<brick>_entry_rmsnorm_weight` slot is actually
        # present in the ABI map; for non-first-in-region bricks the
        # bridge binding above wins because the bridge slot is the one
        # that lands in the ABI.
        for index, block in enumerate(self.layers):
            add_parameter(
                f"layers.{index}.norm.weight",
                "entry_rmsnorm_weight",
                block=block,
            )

        for index, block in enumerate(self.layers):
            if block.backend == "mamba3":
                for parameter_suffix, logical_suffix in (
                    ("block.in_proj.weight", "mamba3_in_proj_weight"),
                    ("block.out_proj.weight", "mamba3_out_proj_weight"),
                    ("block.conv_weight", "mamba3_conv_weight"),
                    ("block.conv_bias", "mamba3_conv_bias"),
                    ("block.dt_bias", "mamba3_dt_bias"),
                    ("block.B_norm_weight", "mamba3_B_norm_weight"),
                    ("block.C_norm_weight", "mamba3_C_norm_weight"),
                    ("block.B_bias", "mamba3_B_bias"),
                    ("block.C_bias", "mamba3_C_bias"),
                    ("block.D", "mamba3_D"),
                ):
                    add(index, parameter_suffix, logical_suffix, block=block)
            elif block.backend == "m2rnn":
                for parameter_suffix, logical_suffix in (
                    ("block.in_proj.weight", "m2rnn_in_proj_weight"),
                    ("block.out_proj.weight", "m2rnn_out_proj_weight"),
                    ("block.conv_weight", "m2rnn_conv_weight"),
                    ("block.conv_bias", "m2rnn_conv_bias"),
                    ("block.g_norm.weight", "m2rnn_g_norm_weight"),
                    ("block.state_weight", "m2rnn_state_weight"),
                    ("block.A_log", "m2rnn_A_log"),
                    ("block.dt_bias", "m2rnn_dt_bias"),
                    ("block.D", "m2rnn_D"),
                ):
                    add(index, parameter_suffix, logical_suffix, block=block)
            elif block.backend == "attention":
                for parameter_suffix, logical_suffix in (
                    (
                        "block.q_proj.weight",
                        "qkv_projection_attention_q_proj_weight",
                    ),
                    (
                        "block.q_proj.bias",
                        "qkv_projection_attention_q_proj_bias",
                    ),
                    (
                        "block.sparse_kv_proj.weight",
                        "qkv_projection_attention_sparse_kv_proj_weight",
                    ),
                    (
                        "block.sparse_kv_proj.bias",
                        "qkv_projection_attention_sparse_kv_proj_bias",
                    ),
                    (
                        "block.rope_inv_freq",
                        "qkv_projection_attention_rope_inv_freq",
                    ),
                    (
                        "block.out_proj.weight",
                        "sparse_mla_fp8_apply_attention_out_proj_weight",
                    ),
                    (
                        "block.out_proj.bias",
                        "sparse_mla_fp8_apply_attention_out_proj_bias",
                    ),
                ):
                    add(index, parameter_suffix, logical_suffix, block=block)

        if "norm.weight" in parameter_names:
            aliases["norm.weight"] = ("final_norm_weight",)
        if "lm_head.weight" in parameter_names:
            aliases["lm_head.weight"] = ("lm_head_weight",)

        return aliases

    def path_c_parameter_gradient_aliases(self) -> dict[str, tuple[str, ...]]:
        """Map MLX parameter-grad tree names to Path C logical grad names."""

        return {
            f"{parameter_name}_grad": tuple(
                f"{logical_name}_grad" for logical_name in logical_names
            )
            for parameter_name, logical_names in (
                self.path_c_parameter_logical_aliases()
            ).items()
        }

    def make_path_c_direct_fusion_chain_logical_buffer_owner(
        self,
        *,
        owner_name: str | None = None,
    ) -> PathCLogicalBufferOwner:
        """Expose model parameter tensors to direct Path C chain binding.

        The owner only records existing MLX array references. It deliberately
        does not flatten, cast, or synthesize missing constants/state buffers.
        """

        parameters = {
            name: value
            for name, value in tree_flatten(self.trainable_parameters())
            if isinstance(value, mx.array)
        }
        buffers: dict[str, mx.array] = {}
        for parameter_name, logical_names in (
            self.path_c_parameter_logical_aliases()
        ).items():
            tensor = parameters.get(parameter_name)
            if tensor is None:
                continue
            for logical_name in logical_names:
                buffers[logical_name] = tensor

        profile_name = str(
            getattr(self, "path_c_profile_name", "HybridTinyLM")
        )
        return PathCLogicalBufferOwner(
            owner_name=owner_name
            or f"{profile_name}.path_c_model_parameter_buffers",
            buffers=buffers,
        )

    def path_c_fusion_regions(
        self,
        *,
        include_backward: bool = False,
        min_route_bricks: int = 2,
        sequence_length: int | None = None,
    ) -> tuple[PathCFusionRegion, ...]:
        """Return Path C fusion candidate regions derived from this model."""

        from cppmega_mlx.runtime.path_c_fusion import (
            build_path_c_model_regions_from_model,
        )

        return build_path_c_model_regions_from_model(
            self,
            region_prefix=str(
                getattr(self, "path_c_region_prefix", "hybrid_tiny_lm_path_c")
            ),
            include_backward=include_backward,
            min_route_bricks=min_route_bricks,
            sequence_length=sequence_length,
        )

    def path_c_fused_train_block_prim_func(
        self,
        *,
        sequence_length: int | None = None,
    ) -> Any | None:
        """Materialise the fused-train-block PrimFunc for this model.

        This drives the same schedule planner the production runtime uses and
        returns the generated PrimFunc (carrying the physical-ABI attrs), or
        ``None`` when no Path C route region exists. The model never caches the
        artifact — callers that need to reuse it across calls should hold the
        returned reference themselves.
        """
        from cppmega_mlx.runtime.path_c_fusion_schedules import (
            plan_path_c_fusion_schedule_for_region,
        )

        # Discover forward-only regions and let plan_path_c_fusion_schedule_for_region
        # extend them with the autograd backward graph (matching the recipe
        # m04 uses in compile_path_c_fused_train_block_artifact_for_model);
        # this gives the same banked physical-ABI shapes the artifact wants.
        regions = self.path_c_fusion_regions(
            include_backward=False,
            sequence_length=sequence_length,
        )
        if not regions:
            return None
        selected = max(
            regions,
            key=lambda region: (
                len(region.nodes),
                len(region.edges),
                region.name,
            ),
        )
        planned = plan_path_c_fusion_schedule_for_region(
            selected,
            include_backward=True,
        )
        schedule_target = getattr(planned, "schedule_target", None)
        if schedule_target is None:
            return None
        return schedule_target.schedule_template(planned.region)

    def make_path_c_physical_abi_bank_owner(
        self,
        *,
        sequence_length: int | None = None,
    ) -> PathCPhysicalAbiBankOwner | None:
        """Allocate the physical ABI banks the generated fused PrimFunc needs.

        Returns a validated :class:`PathCPhysicalAbiBankOwner` whose ``buffers``
        are freshly zero-initialised MLX arrays sized exactly to the generated
        ``_cppmega_path_c_physical_buffer_abi_shapes`` map (dtype is taken from
        the corresponding logical-buffer placement so every bank reflects the
        generated kernel ABI). The owner never repacks or copies model tensors;
        it only owns the bank arrays so the runtime can bind kernel arguments
        in declared order.

        Returns ``None`` when no Path C route region exists for the model.
        """
        prim_func = self.path_c_fused_train_block_prim_func(
            sequence_length=sequence_length,
        )
        if prim_func is None:
            return None
        abi_map = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        abi_shapes = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes", {})
            or {}
        )
        if not abi_map or not abi_shapes:
            return None
        bank_dtypes: dict[str, str] = {}
        for placement in abi_map.values():
            bank = str(placement["bank"])
            dtype = str(placement["dtype"])
            existing = bank_dtypes.setdefault(bank, dtype)
            if existing != dtype:
                raise ValueError(
                    f"conflicting bank dtype for {bank!r}: "
                    f"{existing!r} vs {dtype!r}"
                )
        bank_buffers: dict[str, mx.array] = {}
        for bank, shape in abi_shapes.items():
            dtype_name = bank_dtypes.get(str(bank))
            if dtype_name is None:
                raise ValueError(
                    f"no logical buffer is placed inside physical bank {bank!r}"
                )
            mx_dtype = _path_c_bank_dtype(dtype_name)
            bank_buffers[str(bank)] = mx.zeros(
                tuple(int(dim) for dim in tuple(shape)),
                dtype=mx_dtype,
            )
        profile_name = str(
            getattr(self, "path_c_profile_name", "HybridTinyLM")
        )
        return make_physical_abi_bank_owner(
            f"{profile_name}.path_c_physical_abi_banks",
            abi_map,
            abi_shapes,
            bank_buffers,
        )

    def path_c_fused_in_region_parameter_bank_aliases(
        self,
        *,
        sequence_length: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return a parameter-name → bank-binding map for in-region trainables.

        For every MLX trainable parameter that maps onto a logical buffer in
        the generated fused-train-block PrimFunc AND has a matching ``*_grad``
        slot in the same physical ABI map, this method returns a dict whose
        values describe the bank residency for the parameter::

            {
                "logical_name": <logical name actually placed in the bank>,
                "logical_grad_name": <matching *_grad logical name>,
                "bank": <bank buffer name>,
                "dtype": <bank dtype>,
                "offset": <int>,
                "size": <int>,
                "logical_shape": tuple[int, ...],
            }

        Only parameters that are *both* bank-resident and gradient-producing
        are returned; parameters whose logical name lives in the bank but has
        no ``*_grad`` slot (for example pure inputs) are skipped.
        """

        prim_func = self.path_c_fused_train_block_prim_func(
            sequence_length=sequence_length,
        )
        if prim_func is None:
            return {}
        abi_map = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        if not abi_map:
            return {}
        grad_logical_names = frozenset(
            name for name in abi_map if name.endswith("_grad")
        )
        aliases = self.path_c_parameter_logical_aliases()
        out: dict[str, dict[str, Any]] = {}
        for parameter_name, logical_candidates in aliases.items():
            for logical_name in logical_candidates:
                grad_name = f"{logical_name}_grad"
                if logical_name not in abi_map:
                    continue
                if grad_name not in grad_logical_names:
                    continue
                info = abi_map[logical_name]
                if not isinstance(info, Mapping):
                    continue
                out[parameter_name] = {
                    "logical_name": str(logical_name),
                    "logical_grad_name": str(grad_name),
                    "bank": str(info.get("bank", "")),
                    "dtype": str(info.get("dtype", "")),
                    "offset": int(info.get("offset", 0) or 0),
                    "size": int(info.get("size", 1) or 1),
                    "logical_shape": tuple(
                        int(dim)
                        for dim in tuple(
                            info.get("logical_shape", info.get("shape", ()))
                        )
                    ),
                }
                break
        return out

    def _path_c_static_real_abi_input_values(
        self,
        abi_map: Mapping[str, Any],
    ) -> dict[str, mx.array]:
        """Return small non-trainable real-ABI inputs required by Path C.

        The fused train-block ABI contains a few real inputs that are neither
        trainable parameters nor per-batch runtime tensors. For Sparse-MLA
        these are the softmax scale and optional attention sinks. They must be
        written into model-owned banks along with parameter values; leaving the
        zero-initialized bank slots in place makes the forward/backward kernel
        run with ``sm_scale=0`` and kills q/kv gradients.
        """

        attention = self.config.attention_config("dsa")
        sm_scale = float(attention.q_head_dim) ** -0.5
        values: dict[str, mx.array] = {}
        for logical_name, info in abi_map.items():
            if not isinstance(info, Mapping):
                continue
            shape = tuple(
                int(dim)
                for dim in tuple(info.get("logical_shape", info.get("shape", ())))
            )
            if str(logical_name).endswith("sparse_mla_sm_scale"):
                values[str(logical_name)] = mx.array(
                    [sm_scale], dtype=mx.float32
                ).reshape(shape or (1,))
            elif str(logical_name).endswith("sparse_mla_sinks"):
                values[str(logical_name)] = mx.zeros(shape, dtype=mx.float32)
            elif str(logical_name).endswith("sparse_mla_has_sinks"):
                values[str(logical_name)] = mx.zeros(shape, dtype=mx.int32)
        return values

    def _sync_path_c_static_real_abi_inputs_into_bank(
        self,
        *,
        abi_map: Mapping[str, Any],
        buffers: Mapping[str, Any],
    ) -> tuple[list[str], list[tuple[str, str]]]:
        synced: list[str] = []
        skipped: list[tuple[str, str]] = []
        for logical_name, value in sorted(
            self._path_c_static_real_abi_input_values(abi_map).items()
        ):
            try:
                write_into_bank_slot(abi_map, buffers, logical_name, value)
                synced.append(logical_name)
            except Exception as exc:
                skipped.append((logical_name, f"{type(exc).__name__}: {exc}"))
        return synced, skipped

    def _path_c_lookup_parameter_holder(
        self,
        parameter_name: str,
    ) -> tuple[Any, str]:
        """Walk dotted attribute path; return (holder, leaf_attr_name)."""

        parts = parameter_name.split(".")
        if not parts:
            raise ValueError("parameter_name must be non-empty")
        holder: Any = self
        for part in parts[:-1]:
            if part.isdigit() and isinstance(holder, (list, tuple)):
                holder = holder[int(part)]
                continue
            inner = getattr(holder, part, None)
            if inner is None and isinstance(holder, Mapping):
                inner = holder[part]
            if inner is None:
                raise AttributeError(
                    f"path_c_bank residency cannot resolve {parameter_name!r} "
                    f"at {part!r}"
                )
            holder = inner
        return holder, parts[-1]

    def _path_c_get_parameter_tensor(
        self,
        parameter_name: str,
    ) -> mx.array | None:
        try:
            holder, leaf = self._path_c_lookup_parameter_holder(parameter_name)
        except AttributeError:
            return None
        value = getattr(holder, leaf, None)
        if not isinstance(value, mx.array):
            return None
        return value

    def _path_c_set_parameter_tensor(
        self,
        parameter_name: str,
        value: mx.array,
    ) -> None:
        holder, leaf = self._path_c_lookup_parameter_holder(parameter_name)
        setattr(holder, leaf, value)

    def sync_path_c_in_region_parameters_into_bank(
        self,
        bank_owner: Any,
        *,
        in_region_aliases: Mapping[str, Mapping[str, Any]] | None = None,
        sequence_length: int | None = None,
    ) -> dict[str, Any]:
        """Copy current model param values into pre-allocated bank slots.

        This is the explicit caller-visible bridge from the MLX parameter tree
        to the model-owned physical-ABI bank slots. It is invoked once per
        training step (typically immediately before launching the fused
        kernel) so optimizer-replaced parameter tensors propagate into the
        bank without any hidden allocation. The bank itself is mutated in
        place via slice assignment; no new bank is created.

        Returns a report describing how many parameters were synced and how
        many were skipped (and why).
        """

        if bank_owner is None:
            return {
                "status": "skipped",
                "reason": "bank_owner is None",
                "synced": [],
                "skipped": [],
            }
        buffers = bank_owner if isinstance(bank_owner, Mapping) else None
        if buffers is None:
            buffers = getattr(bank_owner, "buffers", None)
        if not isinstance(buffers, Mapping):
            return {
                "status": "blocked",
                "reason": "bank_owner has no buffers mapping",
                "synced": [],
                "skipped": [],
            }
        aliases = (
            in_region_aliases
            if in_region_aliases is not None
            else self.path_c_fused_in_region_parameter_bank_aliases(
                sequence_length=sequence_length,
            )
        )
        if not aliases:
            return {
                "status": "noop",
                "reason": "no in-region parameter bank aliases",
                "synced": [],
                "skipped": [],
            }
        prim_func = self.path_c_fused_train_block_prim_func(
            sequence_length=sequence_length,
        )
        abi_map = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        synced: list[str] = []
        skipped: list[tuple[str, str]] = []
        static_synced, static_skipped = (
            self._sync_path_c_static_real_abi_inputs_into_bank(
                abi_map=abi_map,
                buffers=buffers,
            )
        )
        skipped.extend(static_skipped)
        for parameter_name, info in sorted(aliases.items()):
            tensor = self._path_c_get_parameter_tensor(parameter_name)
            if tensor is None:
                skipped.append((parameter_name, "parameter_not_in_model_tree"))
                continue
            logical_name = str(info.get("logical_name", ""))
            try:
                write_into_bank_slot(
                    abi_map,
                    buffers,
                    logical_name,
                    tensor,
                )
                synced.append(parameter_name)
            except Exception as exc:
                skipped.append(
                    (parameter_name, f"{type(exc).__name__}: {exc}")
                )
        return {
            "status": "ok" if synced and not skipped else (
                "partial" if synced else "blocked"
            ),
            "reason": (
                "in-region parameters synced into bank slots"
                if synced and not skipped
                else (
                    "some in-region parameters could not be synced"
                    if synced
                    else "no in-region parameter was synced"
                )
            ),
            "synced": synced,
            "skipped": [
                {"parameter_name": name, "reason": reason}
                for name, reason in skipped
            ],
            "static_real_abi_inputs_synced": static_synced,
            "static_real_abi_inputs_skipped": [
                {"logical_name": name, "reason": reason}
                for name, reason in static_skipped
            ],
        }

    def bind_path_c_in_region_parameter_views_into_bank(
        self,
        bank_owner: Any,
        *,
        sequence_length: int | None = None,
    ) -> dict[str, Any]:
        """Bind in-region trainable parameters as zero-copy views into the bank.

        For every parameter discovered by
        :py:meth:`path_c_fused_in_region_parameter_bank_aliases`, this method:

        1. Writes the current parameter value into its bank slot (initial sync).
        2. Replaces the parameter attribute on the model's submodule with a
           bank-resident view (a slice of the bank reshaped to the logical
           shape). The view is the same MLX array that the fused kernel
           reads / writes, so subsequent ``model.trainable_parameters()``
           reports the view and downstream eager autograd sees the same
           storage backing.

        After ``optimizer.update(model, grads)`` replaces the parameter
        attribute with a fresh tensor, call
        :py:meth:`sync_path_c_in_region_parameters_into_bank` before the next
        forward pass so the new value lands in the bank again. The bank
        itself is mutated in place via slice assignment; no copy of the
        full bank ever happens.

        Returns a report mirroring
        :py:meth:`sync_path_c_in_region_parameters_into_bank` plus a
        ``bound`` list of parameters now backed by bank views.
        """

        if bank_owner is None:
            return {
                "status": "skipped",
                "reason": "bank_owner is None",
                "bound": [],
                "skipped": [],
                "in_region_parameter_count": 0,
            }
        buffers = bank_owner if isinstance(bank_owner, Mapping) else None
        if buffers is None:
            buffers = getattr(bank_owner, "buffers", None)
        if not isinstance(buffers, Mapping):
            return {
                "status": "blocked",
                "reason": "bank_owner has no buffers mapping",
                "bound": [],
                "skipped": [],
                "in_region_parameter_count": 0,
            }
        aliases = self.path_c_fused_in_region_parameter_bank_aliases(
            sequence_length=sequence_length,
        )
        if not aliases:
            return {
                "status": "noop",
                "reason": "no in-region parameter bank aliases",
                "bound": [],
                "skipped": [],
                "in_region_parameter_count": 0,
            }
        prim_func = self.path_c_fused_train_block_prim_func(
            sequence_length=sequence_length,
        )
        abi_map = dict(
            getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
            or {}
        )
        bound: list[str] = []
        skipped: list[tuple[str, str]] = []
        static_synced, static_skipped = (
            self._sync_path_c_static_real_abi_inputs_into_bank(
                abi_map=abi_map,
                buffers=buffers,
            )
        )
        skipped.extend(static_skipped)
        for parameter_name, info in sorted(aliases.items()):
            tensor = self._path_c_get_parameter_tensor(parameter_name)
            if tensor is None:
                skipped.append((parameter_name, "parameter_not_in_model_tree"))
                continue
            logical_name = str(info.get("logical_name", ""))
            try:
                write_into_bank_slot(abi_map, buffers, logical_name, tensor)
                view = logical_bank_view(abi_map, buffers, logical_name)
                self._path_c_set_parameter_tensor(parameter_name, view)
                bound.append(parameter_name)
            except Exception as exc:
                skipped.append(
                    (parameter_name, f"{type(exc).__name__}: {exc}")
                )
        self._path_c_in_region_parameter_bank_aliases = dict(aliases)
        self._path_c_in_region_parameter_bound_names = tuple(sorted(bound))
        return {
            "status": "ok" if bound and not skipped else (
                "partial" if bound else "blocked"
            ),
            "reason": (
                "in-region parameters bound as bank views"
                if bound and not skipped
                else (
                    "some in-region parameters could not be bank-bound"
                    if bound
                    else "no in-region parameter was bank-bound"
                )
            ),
            "bound": bound,
            "skipped": [
                {"parameter_name": name, "reason": reason}
                for name, reason in skipped
            ],
            "in_region_parameter_count": len(bound),
            "in_region_parameter_names": tuple(sorted(bound)),
            "in_region_parameter_bank_aliases": dict(aliases),
            "static_real_abi_inputs_synced": static_synced,
            "static_real_abi_inputs_skipped": [
                {"logical_name": name, "reason": reason}
                for name, reason in static_skipped
            ],
        }

    def path_c_fused_first_in_region_layer_index(
        self,
        *,
        sequence_length: int | None = None,
    ) -> int | None:
        """Return the layer index where the fused Path C region starts.

        Returned value is the smallest ``layers.<index>`` index that owns a
        bank-bound trainable parameter. ``None`` indicates the model has no
        fused-region parameters; callers fall back to the standard
        eager forward in that case.
        """

        aliases = self.path_c_fused_in_region_parameter_bank_aliases(
            sequence_length=sequence_length,
        )
        if not aliases:
            return None
        indices = [
            int(name.split(".", 2)[1])
            for name in aliases
            if name.startswith("layers.") and name.split(".", 2)[1].isdigit()
        ]
        if not indices:
            return None
        return min(indices)

    def attach_path_c_fused_suffix_custom_function(
        self,
        fused_suffix: Any,
        *,
        parameter_order: Sequence[str],
        first_in_region_layer_index: int,
    ) -> None:
        """Attach a fused-suffix custom function for the loss helper to use.

        The runtime composes the function (binding artifact + bank owner)
        once per install and registers it here so the loss helper can dispatch
        without re-discovering the wiring on every step.
        """

        if not callable(fused_suffix):
            raise TypeError("fused_suffix must be callable")
        if first_in_region_layer_index < 0:
            raise ValueError("first_in_region_layer_index must be non-negative")
        self._path_c_fused_suffix_custom_function = fused_suffix
        self._path_c_fused_suffix_parameter_order = tuple(parameter_order)
        self._path_c_fused_suffix_first_in_region_layer_index = int(
            first_in_region_layer_index
        )

    def detach_path_c_fused_suffix_custom_function(self) -> None:
        for attr in (
            "_path_c_fused_suffix_custom_function",
            "_path_c_fused_suffix_parameter_order",
            "_path_c_fused_suffix_first_in_region_layer_index",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def path_c_fused_suffix_custom_function_available(self) -> bool:
        return callable(
            getattr(self, "_path_c_fused_suffix_custom_function", None)
        )

    def path_c_fused_suffix_loss(
        self,
        batch: Mapping[str, mx.array],
    ) -> tuple[mx.array, mx.array]:
        """Compute (loss, ntokens) via the fused-suffix custom function.

        The prefix (embedding + side-channel embeddings + layers before
        the fused region) runs eagerly in MLX; layers inside the fused
        region, the final norm, the ``lm_head`` projection and the loss
        reduction are handed off to the fused TileLang artifact via the
        attached :py:meth:`attach_path_c_fused_suffix_custom_function`
        callable. MLX autograd therefore never traces through the
        in-region layers: the fused custom function's VJP returns
        bank-resident cotangents for every in-region trainable
        parameter, and the prefix continues backward from
        ``hidden_entry_grad`` to the embedding and prefix layers.

        Returns ``(loss, ntokens)`` as MLX scalars; the trainer uses
        these the same way it would the output of
        :func:`cppmega_mlx.training.loss.next_token_cross_entropy`.
        """

        fused_suffix = getattr(
            self, "_path_c_fused_suffix_custom_function", None
        )
        parameter_order: tuple[str, ...] = tuple(
            cast("Sequence[str]",
                getattr(self, "_path_c_fused_suffix_parameter_order", ()))
        )
        first_in_region_layer_index = getattr(
            self,
            "_path_c_fused_suffix_first_in_region_layer_index",
            None,
        )
        if (
            not callable(fused_suffix)
            or not parameter_order
            or first_in_region_layer_index is None
        ):
            raise ValueError(
                "path_c_fused_suffix_loss requires "
                "attach_path_c_fused_suffix_custom_function to be installed"
            )

        from cppmega_mlx.data.batch import ensure_lm_batch

        lm_batch = ensure_lm_batch(batch)
        input_ids = lm_batch.inputs
        targets = lm_batch.targets
        target_mask = lm_batch.target_mask
        # decoder_hidden_states accepts the side-channel tensors and the
        # document_ids stream; the prefix needs both because the masking
        # and structure / platform embeddings are inputs to the prefix
        # layers. The fused suffix bridge does not need a kv_cache and
        # does not consume the side-channel kwargs directly.
        side_channel_kwargs = dict(lm_batch.model_kwargs())
        document_ids = lm_batch.input_document_ids

        # Targets are (B, S); the fused kernel consumes a flat 1-D
        # target_ids buffer of length S. The custom function writes them
        # into the bank slot; we forward the per-step batch tensor.
        if targets.ndim != 2:
            raise ValueError(
                f"path_c_fused_suffix_loss expects targets shaped (B, S); "
                f"got {targets.shape}"
            )
        if targets.shape[0] != 1:
            raise NotImplementedError(
                "path_c_fused_suffix_loss currently expects batch_size=1 "
                "(matching the tiny smoke profile); broader shapes need "
                "the fused kernel ABI to expand its target_ids buffer."
            )
        if target_mask.shape != targets.shape:
            raise ValueError(
                "target_mask shape must match targets; got "
                f"{target_mask.shape} vs {targets.shape}"
            )
        flat_target_ids = mx.reshape(
            targets.astype(mx.int32), (targets.shape[1],)
        )
        flat_target_mask = mx.reshape(
            target_mask.astype(mx.float32), (target_mask.shape[1],)
        )

        hidden_entry = self.decoder_hidden_states(
            input_ids,
            structure_ids=side_channel_kwargs.get("structure_ids"),
            dep_levels=side_channel_kwargs.get("dep_levels"),
            ast_depth_ids=side_channel_kwargs.get("ast_depth_ids"),
            sibling_index_ids=side_channel_kwargs.get("sibling_index_ids"),
            node_type_ids=side_channel_kwargs.get("node_type_ids"),
            platform_ids=side_channel_kwargs.get("platform_ids"),
            document_ids=document_ids,
            kv_cache=None,
            stop_layer_index=int(first_in_region_layer_index),
            apply_final_norm=False,
        )

        params = tuple(
            self._path_c_get_parameter_tensor(name)
            for name in parameter_order
        )
        if any(p is None for p in params):
            missing = [
                name
                for name, p in zip(parameter_order, params, strict=True)
                if p is None
            ]
            raise ValueError(
                "path_c_fused_suffix_loss could not resolve in-region "
                f"parameters: {missing}"
            )
        result: Any = fused_suffix(
            hidden_entry,
            flat_target_ids,
            flat_target_mask,
            *params,
        )
        loss, ntokens = cast(tuple[mx.array, mx.array], result)
        # ntokens lives in the bank as float32; convert back to the standard
        # uint32 reporting form used by other loss helpers.
        ntokens = ntokens.astype(mx.uint32)
        return loss, ntokens

    def __call__(
        self,
        input_ids: mx.array,
        *,
        structure_ids: mx.array | None = None,
        dep_levels: mx.array | None = None,
        ast_depth_ids: mx.array | None = None,
        sibling_index_ids: mx.array | None = None,
        node_type_ids: mx.array | None = None,
        platform_ids: mx.array | None = None,
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
                platform_ids=platform_ids,
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
        platform_ids: mx.array | None = None,
        document_ids: mx.array | None = None,
        kv_cache: ContiguousKVCache | None = None,
        stop_layer_index: int | None = None,
        apply_final_norm: bool = True,
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

        structure_inputs = {
            "structure_ids": _validate_side_channel_shape(
                "structure_ids", structure_ids, batch_size, seq_length
            ),
            "dep_levels": _validate_side_channel_shape(
                "dep_levels", dep_levels, batch_size, seq_length
            ),
            "ast_depth_ids": _validate_side_channel_shape(
                "ast_depth_ids", ast_depth_ids, batch_size, seq_length
            ),
            "sibling_index_ids": _validate_side_channel_shape(
                "sibling_index_ids", sibling_index_ids, batch_size, seq_length
            ),
            "node_type_ids": _validate_side_channel_shape(
                "node_type_ids", node_type_ids, batch_size, seq_length
            ),
        }
        structure_residual_scale = self.config.residual_scale_for("structure")
        if structure_residual_scale != 0.0:
            structure_embeddings = self.structure_embedding(
                **structure_inputs,
                target_dtype=hidden_states.dtype,
            )
            if structure_embeddings.ndim == hidden_states.ndim:
                if structure_residual_scale == 1.0:
                    hidden_states = hidden_states + structure_embeddings
                else:
                    hidden_states = hidden_states + (
                        structure_embeddings
                        * mx.array(structure_residual_scale, dtype=hidden_states.dtype)
                    )

        platform_ids = _validate_platform_ids(
            platform_ids,
            batch_size=batch_size,
            seq_length=seq_length,
        )
        platform_residual_scale = self.config.residual_scale_for("platform")
        if platform_ids is not None and platform_residual_scale != 0.0:
            platform_residual = self.platform_embedding(
                platform_ids,
                target_dtype=hidden_states.dtype,
            )
            if platform_residual_scale == 1.0:
                hidden_states = hidden_states + platform_residual
            else:
                hidden_states = hidden_states + (
                    platform_residual
                    * mx.array(platform_residual_scale, dtype=hidden_states.dtype)
                )

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
            for layer_index, layer in enumerate(self.layers):
                if stop_layer_index is not None and layer_index >= stop_layer_index:
                    break
                if layer.backend == "engram" and document_ids is not None:
                    hidden_states = mx.checkpoint(layer)(
                        hidden_states, mask, doc_ids=document_ids
                    )
                else:
                    hidden_states = mx.checkpoint(layer)(hidden_states, mask)
        else:
            attention_layer_idx = 0
            for layer_index, layer in enumerate(self.layers):
                if stop_layer_index is not None and layer_index >= stop_layer_index:
                    break
                if layer.backend == "attention":
                    hidden_states = layer(
                        hidden_states,
                        mask,
                        kv_cache=kv_cache,
                        attention_layer_idx=attention_layer_idx if kv_cache is not None else None,
                    )
                    attention_layer_idx += 1
                elif layer.backend == "engram":
                    hidden_states = layer(
                        hidden_states, mask, doc_ids=document_ids
                    )
                else:
                    hidden_states = layer(hidden_states, mask)
        if apply_final_norm:
            return self.norm(hidden_states)
        return hidden_states


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


def _validate_platform_ids(
    tensor: mx.array | None,
    *,
    batch_size: int,
    seq_length: int,
) -> mx.array | None:
    if tensor is None:
        return None
    if tensor.ndim not in (2, 3):
        raise ValueError(
            f"platform_ids must be shaped (B, K) or (B, S, K), got {tensor.shape}"
        )
    if tensor.shape[0] != batch_size:
        raise ValueError(
            "platform_ids batch dimension must match input batch "
            f"{batch_size}, got {tensor.shape[0]}"
        )
    if tensor.ndim == 3 and tensor.shape[1] != seq_length:
        raise ValueError(
            "token-local platform_ids sequence dimension must match input sequence "
            f"{seq_length}, got {tensor.shape[1]}"
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
    "PathCActivationBufferCapture",
    "PathCActivationProbe",
    "HybridTinyBlock",
    "HybridTinyConfig",
    "HybridTinyLM",
]
