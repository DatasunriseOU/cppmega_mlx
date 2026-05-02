"""Named MLX model factory profiles for local cppmega milestones."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import mlx.core as mx

from cppmega_mlx.config.model import (
    DSAConfig,
    M2RNNConfig,
    Mamba3Config,
    MoeConfig,
    Nam56RModelConfig,
    VocabMetadata,
)
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.recipes.nam56r import build_hybrid_tiny_config_from_nam56r
from cppmega_mlx.recipes.pattern import ExpandedNamPattern, expand_nam_pattern

LOCAL_GB10_QUARTER_PROFILE = "local_gb10_quarter"
LOCAL_GB10_QUARTER_PATTERN = "AEMEAEMEAEMR"
LOCAL_GB10_QUARTER_DEPTH = 13
LOCAL_GB10_QUARTER_HIDDEN_SIZE = 3584
LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE = 18_944
LOCAL_GB10_QUARTER_NUM_HEADS = 28
LOCAL_GB10_QUARTER_HEAD_DIM = 128
LOCAL_GB10_QUARTER_VOCAB_SIZE = 65_536
LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS = (1, 2, 3)
LOCAL_GB10_QUARTER_MTP_DEPTH = 2
LOCAL_GB10_QUARTER_MTP_BETA = 0.6
LOCAL_GB10_QUARTER_MTP_LAMBDA = 0.3
LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH = 4096

ModelKind = Literal["hybrid_tiny"]


@dataclass(frozen=True)
class MTPProfile:
    """MTP factory metadata shared with the training-side loss milestone."""

    depth: int | None = LOCAL_GB10_QUARTER_MTP_DEPTH
    beta: float = LOCAL_GB10_QUARTER_MTP_BETA
    loss_weight: float = LOCAL_GB10_QUARTER_MTP_LAMBDA
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.enabled and self.depth is None:
            raise ValueError("MTP enabled requires mtp.depth")
        if self.enabled and self.depth is not None and self.depth <= 0:
            raise ValueError("MTP enabled requires positive depth")
        if self.depth is not None and self.depth < 0:
            raise ValueError("MTP depth must be non-negative")
        if self.beta <= 0:
            raise ValueError("MTP beta must be positive")
        if self.loss_weight < 0:
            raise ValueError("MTP loss_weight must be non-negative")


@dataclass(frozen=True)
class ModelFactoryProfile:
    """Validated construction profile for an MLX model factory entry.

    The profile is intentionally allocation-free.  It can be converted to the
    existing NAM56R/HybridTiny configs, or used as metadata for parity tests.
    """

    name: str
    pattern: str
    depth: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    head_dim: int
    vocab_size: int
    max_seq_length: int = LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH
    dsa_a_layer_ranks: tuple[int, ...] = LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS
    dsa_indexer_n_heads: int | None = 32
    dsa_indexer_head_dim: int | None = 64
    moe_num_experts: int = 16
    moe_top_k: int = 4
    moe_expert_hidden_size: int = 896
    moe_shared_expert_hidden_size: int = 1024
    mtp: MTPProfile = MTPProfile()
    model_kind: ModelKind = "hybrid_tiny"

    @property
    def expanded_pattern(self) -> ExpandedNamPattern:
        return expand_nam_pattern(
            self.pattern,
            self.depth,
            dsa_a_layer_ranks=self.dsa_a_layer_ranks,
        )

    def __post_init__(self) -> None:
        _require_positive("depth", self.depth)
        _require_positive("hidden_size", self.hidden_size)
        _require_positive("ffn_hidden_size", self.ffn_hidden_size)
        _require_positive("num_attention_heads", self.num_attention_heads)
        _require_positive("head_dim", self.head_dim)
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("max_seq_length", self.max_seq_length)
        _require_positive("moe_num_experts", self.moe_num_experts)
        _require_positive("moe_top_k", self.moe_top_k)
        _require_positive("moe_expert_hidden_size", self.moe_expert_hidden_size)
        _require_positive(
            "moe_shared_expert_hidden_size",
            self.moe_shared_expert_hidden_size,
        )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k must be <= moe_num_experts")
        if self.dsa_a_layer_ranks:
            if self.dsa_indexer_n_heads is None or self.dsa_indexer_head_dim is None:
                raise ValueError("DSA A-layer ranks require DSA indexer dimensions")
            _require_positive("dsa_indexer_n_heads", self.dsa_indexer_n_heads)
            _require_positive("dsa_indexer_head_dim", self.dsa_indexer_head_dim)
        if self.model_kind != "hybrid_tiny":
            raise ValueError(f"unsupported model_kind={self.model_kind!r}")
        self.expanded_pattern

    def nam56r_config(self) -> Nam56RModelConfig:
        """Return the existing validated NAM56R config shape for this profile."""

        return Nam56RModelConfig(
            pattern=self.pattern,
            depth=self.depth,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            seq_len=self.max_seq_length,
            max_position_embeddings=self.max_seq_length,
            vocab=VocabMetadata(
                local_profile_vocab_size=self.vocab_size,
                default_model_vocab_size=self.vocab_size,
            ),
            moe=MoeConfig(
                num_experts=self.moe_num_experts,
                top_k=self.moe_top_k,
                ffn_hidden_size=self.moe_expert_hidden_size,
                shared_expert_intermediate_size=self.moe_shared_expert_hidden_size,
            ),
            m2rnn=M2RNNConfig(d_model=self.hidden_size),
            mamba3=Mamba3Config(d_model=self.hidden_size),
            dsa=DSAConfig(
                a_layer_ranks=self.dsa_a_layer_ranks,
                indexer_n_heads=self.dsa_indexer_n_heads or 1,
                indexer_head_dim=self.dsa_indexer_head_dim or 1,
            ),
        )

    def hybrid_config(self, **overrides) -> HybridTinyConfig:
        """Return a HybridTinyConfig using existing NAM56R-to-MLX mapping."""

        return build_hybrid_tiny_config_from_nam56r(self.nam56r_config(), **overrides)

    def tiny_smoke_config(self, **overrides) -> HybridTinyConfig:
        """Return a small T=512-capable config preserving this route profile."""

        params = {
            "vocab_size": 256,
            "hidden_size": 16,
            "pattern": self.pattern,
            "depth": self.depth,
            "dsa_a_layer_ranks": self.dsa_a_layer_ranks,
            "num_attention_heads": 4,
            "max_seq_length": 512,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_expert_hidden_size": 32,
            "moe_shared_expert_hidden_size": 16,
            "mamba_expand": 1,
            "mamba_head_dim": 4,
            "mamba_state_dim": 4,
            "mamba_groups": 1,
            "mamba_mimo_rank": 1,
            "mamba_is_mimo": False,
            "mamba_chunk_size": 8,
            "m2rnn_k_head_dim": 4,
            "m2rnn_v_head_dim": 4,
            "m2rnn_num_q_heads": 1,
            "m2rnn_num_k_heads": 1,
            "m2rnn_num_v_heads": 1,
            "m2rnn_num_f_heads": 1,
            "m2rnn_num_weight_heads": 1,
            "m2rnn_chunk_size": 8,
        }
        params.update(overrides)
        return HybridTinyConfig(**params)

    def build_model(self, **hybrid_config_overrides) -> HybridTinyLM:
        """Allocate the profile's MLX model via the existing HybridTinyLM builder."""

        return HybridTinyLM(self.hybrid_config(**hybrid_config_overrides))

    def build_tiny_smoke_model(self, **hybrid_config_overrides) -> HybridTinyLM:
        """Allocate a tiny model that preserves route metadata for smoke tests."""

        return HybridTinyLM(self.tiny_smoke_config(**hybrid_config_overrides))


def local_gb10_quarter_profile(**overrides) -> ModelFactoryProfile:
    """Return the allocation-free local GB10 quarter factory profile."""

    profile = ModelFactoryProfile(
        name=LOCAL_GB10_QUARTER_PROFILE,
        pattern=LOCAL_GB10_QUARTER_PATTERN,
        depth=LOCAL_GB10_QUARTER_DEPTH,
        hidden_size=LOCAL_GB10_QUARTER_HIDDEN_SIZE,
        ffn_hidden_size=LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE,
        num_attention_heads=LOCAL_GB10_QUARTER_NUM_HEADS,
        head_dim=LOCAL_GB10_QUARTER_HEAD_DIM,
        vocab_size=LOCAL_GB10_QUARTER_VOCAB_SIZE,
        dsa_a_layer_ranks=LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS,
    )
    if not overrides:
        return profile
    return replace(profile, **overrides)


def get_model_profile(name: str) -> ModelFactoryProfile:
    if name == LOCAL_GB10_QUARTER_PROFILE:
        return local_gb10_quarter_profile()
    raise ValueError(
        f"unknown model factory profile {name!r}; supported: {LOCAL_GB10_QUARTER_PROFILE}"
    )


def local_gb10_quarter(**hybrid_config_overrides) -> HybridTinyLM:
    """Allocate the full local_gb10_quarter MLX model.

    This constructs the real profile dimensions and can allocate billions of
    parameters.  Tests should use ``local_gb10_quarter_profile`` or
    ``build_local_gb10_quarter_tiny_smoke_model`` unless they intentionally
    exercise full-profile memory behavior.
    """

    return local_gb10_quarter_profile().build_model(**hybrid_config_overrides)


def build_local_gb10_quarter_tiny_smoke_model(**hybrid_config_overrides) -> HybridTinyLM:
    return local_gb10_quarter_profile().build_tiny_smoke_model(**hybrid_config_overrides)


def forward_has_finite_logits(model: HybridTinyLM, input_ids: mx.array) -> bool:
    logits = model(input_ids)
    mx.eval(logits)
    return bool(mx.all(mx.isfinite(logits)).item())


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "LOCAL_GB10_QUARTER_DEPTH",
    "LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS",
    "LOCAL_GB10_QUARTER_FFN_HIDDEN_SIZE",
    "LOCAL_GB10_QUARTER_HEAD_DIM",
    "LOCAL_GB10_QUARTER_HIDDEN_SIZE",
    "LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH",
    "LOCAL_GB10_QUARTER_MTP_BETA",
    "LOCAL_GB10_QUARTER_MTP_DEPTH",
    "LOCAL_GB10_QUARTER_MTP_LAMBDA",
    "LOCAL_GB10_QUARTER_NUM_HEADS",
    "LOCAL_GB10_QUARTER_PATTERN",
    "LOCAL_GB10_QUARTER_PROFILE",
    "LOCAL_GB10_QUARTER_VOCAB_SIZE",
    "MTPProfile",
    "ModelFactoryProfile",
    "build_local_gb10_quarter_tiny_smoke_model",
    "forward_has_finite_logits",
    "get_model_profile",
    "local_gb10_quarter",
    "local_gb10_quarter_profile",
]
