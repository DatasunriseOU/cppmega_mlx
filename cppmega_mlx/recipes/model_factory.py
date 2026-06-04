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
from cppmega_mlx.tokenizer.cpp_tokenizer import (
    EXPECTED_SPECIAL_TOKENS,
    EXPECTED_VOCAB_SIZE,
    TokenizerContractError,
)

LOCAL_GB10_QUARTER_PROFILE = "local_gb10_quarter"
MODEL_FACTORY_UPSTREAM_RECIPE_MODULE = "cppmega.recipes.run_profiles"
LOCAL_GB10_QUARTER_UPSTREAM_RECIPE_NAME = "local_gb10_quarter"
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
LOCAL_GB10_QUARTER_TOKENIZER_BLOCKER_ID = "cppmega-mlx-t8f.1"
LOCAL_GB10_QUARTER_TOKENIZER_MILESTONE = "M0.1"
LOCAL_GB10_QUARTER_TOKENIZER_REQUIRED_SPECIALS: tuple[tuple[str, int], ...] = tuple(
    EXPECTED_SPECIAL_TOKENS.items()
)

NAM56R_FULL_PROFILE = "nam56r_full"
NAM56R_FULL_UPSTREAM_RECIPE_NAME = "h200_dsa_9_4_m"
NAM56R_FULL_PATTERN = "AEMEAEMEAEMR"
NAM56R_FULL_DEPTH = 52
NAM56R_FULL_HIDDEN_SIZE = 4096
NAM56R_FULL_FFN_HIDDEN_SIZE = 21_504
NAM56R_FULL_NUM_HEADS = 32
NAM56R_FULL_HEAD_DIM = 128
NAM56R_FULL_VOCAB_SIZE = 65_536
NAM56R_FULL_DSA_A_LAYER_RANKS = (1, 2, 3, 5, 6, 7, 9, 10, 11)
NAM56R_FULL_MAX_SEQ_LENGTH = 4096

ModelKind = Literal["hybrid_tiny"]


@dataclass(frozen=True)
class TokenizerContractStatus:
    """Tokenizer readiness metadata for fail-closed model factory allocation."""

    expected_vocab_size: int = EXPECTED_VOCAB_SIZE
    required_special_tokens: tuple[
        tuple[str, int], ...
    ] = LOCAL_GB10_QUARTER_TOKENIZER_REQUIRED_SPECIALS
    resolved: bool = True
    milestone: str = LOCAL_GB10_QUARTER_TOKENIZER_MILESTONE
    blocker_id: str = LOCAL_GB10_QUARTER_TOKENIZER_BLOCKER_ID
    reason: str = (
        "M0.1 is closed: the deployed GB10 65K tokenizer is vendored with "
        "id 7=<CODE_START>, id 8=<CODE_END>, id 11=<QUERY_TOOL>, "
        "id 19=<TOOL_RESULT>, id 45=<FIM_INSTRUCTION>, id 46=<SPACE>, "
        "and id 47=<NL>; MLX and upstream wrappers use explicit "
        "whitespace-sentinel encode/decode with Mac-vs-upstream parity receipts"
    )

    @property
    def is_resolved(self) -> bool:
        return self.resolved

    def require_resolved(self) -> None:
        if self.resolved:
            return
        raise TokenizerContractError(
            f"{self.milestone} tokenizer contract is unresolved "
            f"({self.blocker_id}): {self.reason}"
        )


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
    Upstream recipe identity is provenance only; it is not a local Mac/MLX
    acceptance target.
    """

    name: str
    pattern: str
    depth: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    head_dim: int
    vocab_size: int
    # Outer micro-batch axis for the path_c gridded step. Default 1 keeps every
    # existing profile/consumer at the historical bs1 behavior byte-for-byte; set
    # it to 4 (via ``local_gb10_quarter_profile(micro_batch_size=4)`` or a
    # ``replace``) to select the bs4 gridded path. It is threaded onto the
    # HybridTinyConfig the profile yields so the path_c shape-env reads it as the
    # single source of truth. RULE #1: a non-positive value RAISES in __post_init__.
    micro_batch_size: int = 1
    max_seq_length: int = LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH
    dsa_a_layer_ranks: tuple[int, ...] = LOCAL_GB10_QUARTER_DSA_A_LAYER_RANKS
    dsa_indexer_n_heads: int | None = 32
    dsa_indexer_head_dim: int | None = 64
    moe_num_experts: int = 16
    moe_top_k: int = 4
    moe_expert_hidden_size: int = 896
    moe_shared_expert_hidden_size: int = 1024
    mtp: MTPProfile = MTPProfile()
    tokenizer_contract: TokenizerContractStatus = TokenizerContractStatus()
    model_kind: ModelKind = "hybrid_tiny"
    upstream_recipe_module: str = MODEL_FACTORY_UPSTREAM_RECIPE_MODULE
    upstream_recipe_name: str | None = None

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
        _require_positive("micro_batch_size", self.micro_batch_size)
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
        if not self.upstream_recipe_module:
            raise ValueError("upstream_recipe_module must be non-empty")
        if self.upstream_recipe_name == "":
            raise ValueError("upstream_recipe_name must be non-empty when provided")
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
        """Return a HybridTinyConfig using existing NAM56R-to-MLX mapping.

        The profile's ``micro_batch_size`` rides onto the returned config as the
        single source of truth for the path_c gridded step's outer batch axis
        (``_path_c_model_shape_env_from_config`` reads ``config.micro_batch_size``).
        ``HybridTinyConfig`` is a frozen dataclass with no such field, so the value
        is carried directly on the instance via ``object.__setattr__`` (the
        documented "carry it on the config/region directly" path). RULE #1: an
        explicit ``micro_batch_size`` override that conflicts with the profile's is
        rejected, and the carried value is read back and asserted — a value that
        failed to stick RAISES rather than silently reverting to bs1.
        """

        requested_batch = int(self.micro_batch_size)
        if "micro_batch_size" in overrides:
            override_batch = int(overrides.pop("micro_batch_size"))
            if override_batch < 1:
                raise ValueError(
                    "hybrid_config(micro_batch_size=...) must be a positive int "
                    f"(got {override_batch!r}); RULE #1 forbids a silent clamp to 1"
                )
            requested_batch = override_batch
        config = build_hybrid_tiny_config_from_nam56r(
            self.nam56r_config(), **overrides
        )
        # Carry the micro-batch axis onto the frozen config instance (no field in
        # HybridTinyConfig) so the path_c shape-env reads ONE source of truth.
        object.__setattr__(config, "micro_batch_size", requested_batch)
        carried = int(getattr(config, "micro_batch_size", 0))
        if carried != requested_batch:
            raise RuntimeError(
                "model_factory.hybrid_config: failed to carry micro_batch_size "
                f"onto the HybridTinyConfig (requested {requested_batch}, read back "
                f"{carried!r}); RULE #1: refusing a silent bs{requested_batch}->bs1 "
                "revert"
            )
        return config

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

    @property
    def path_c_bricks(self) -> tuple[dict[str, str], ...]:
        """Return allocation-free brick descriptors for Path C auto-discovery."""

        config = self.hybrid_config()
        return tuple(
            {
                "name": f"{self.name}_brick_{index}_{layer.symbol}",
                "kind": layer.role,
                "route_symbol": layer.symbol,
                **(
                    {
                        "attention_qkv_has_bias": str(
                            bool(config.attention_config(layer.attention_route).bias)
                        ).lower(),
                        "attention_out_proj_has_bias": str(
                            bool(config.attention_config(layer.attention_route).bias)
                        ).lower(),
                    }
                    if layer.symbol == "A"
                    else {}
                ),
            }
            for index, layer in enumerate(self.expanded_pattern.layers)
        )

    def build_model(
        self,
        *,
        dtype: mx.Dtype | None = None,
        **hybrid_config_overrides,
    ) -> HybridTinyLM:
        """Allocate the profile's MLX model via the existing HybridTinyLM builder."""

        self.tokenizer_contract.require_resolved()
        return _attach_path_c_profile_metadata(
            HybridTinyLM(
                self.hybrid_config(**hybrid_config_overrides),
                dtype=dtype,
            ),
            self,
        )

    def build_tiny_smoke_model(
        self,
        *,
        dtype: mx.Dtype | None = None,
        **hybrid_config_overrides,
    ) -> HybridTinyLM:
        """Allocate a tiny model that preserves route metadata for smoke tests."""

        return _attach_path_c_profile_metadata(
            HybridTinyLM(
                self.tiny_smoke_config(**hybrid_config_overrides),
                dtype=dtype,
            ),
            self,
        )


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
        upstream_recipe_name=LOCAL_GB10_QUARTER_UPSTREAM_RECIPE_NAME,
    )
    if not overrides:
        return profile
    return replace(profile, **overrides)


def nam56r_full_profile(**overrides) -> ModelFactoryProfile:
    """Return the allocation-free full NAM56R factory profile metadata."""

    profile = ModelFactoryProfile(
        name=NAM56R_FULL_PROFILE,
        pattern=NAM56R_FULL_PATTERN,
        depth=NAM56R_FULL_DEPTH,
        hidden_size=NAM56R_FULL_HIDDEN_SIZE,
        ffn_hidden_size=NAM56R_FULL_FFN_HIDDEN_SIZE,
        num_attention_heads=NAM56R_FULL_NUM_HEADS,
        head_dim=NAM56R_FULL_HEAD_DIM,
        vocab_size=NAM56R_FULL_VOCAB_SIZE,
        max_seq_length=NAM56R_FULL_MAX_SEQ_LENGTH,
        dsa_a_layer_ranks=NAM56R_FULL_DSA_A_LAYER_RANKS,
        upstream_recipe_name=NAM56R_FULL_UPSTREAM_RECIPE_NAME,
    )
    if not overrides:
        return profile
    return replace(profile, **overrides)


def get_model_profile(name: str) -> ModelFactoryProfile:
    if name == LOCAL_GB10_QUARTER_PROFILE:
        return local_gb10_quarter_profile()
    if name == NAM56R_FULL_PROFILE:
        return nam56r_full_profile()
    raise ValueError(
        f"unknown model factory profile {name!r}; supported: "
        f"{LOCAL_GB10_QUARTER_PROFILE}, {NAM56R_FULL_PROFILE}"
    )


def _attach_path_c_profile_metadata(
    model: HybridTinyLM,
    profile: ModelFactoryProfile,
) -> HybridTinyLM:
    model.path_c_profile_name = profile.name
    model.path_c_input_model_name = f"{profile.name}.path_c_bricks"
    model.path_c_region_prefix = f"{profile.name}_path_c"
    for index, (layer, brick) in enumerate(zip(model.layers, profile.path_c_bricks)):
        layer.path_c_layer_index = index
        layer.path_c_profile_brick_name = brick["name"]
    return model


def local_gb10_quarter(
    *,
    dtype: mx.Dtype | None = mx.bfloat16,
    **hybrid_config_overrides,
) -> HybridTinyLM:
    """Allocate the full local_gb10_quarter MLX model.

    This constructs the real profile dimensions and can allocate billions of
    parameters.  Tests should use ``local_gb10_quarter_profile`` or
    ``build_local_gb10_quarter_tiny_smoke_model`` unless they intentionally
    exercise full-profile memory behavior.

    The default ``dtype`` is ``mx.bfloat16`` to match the cppmega CUDA training
    configuration (``precision="bf16"``); pass ``dtype=mx.float32`` to override
    for parity probes that need fp32 weights.
    """

    profile_overrides = {}
    tokenizer_contract = hybrid_config_overrides.pop("tokenizer_contract", None)
    if tokenizer_contract is not None:
        profile_overrides["tokenizer_contract"] = tokenizer_contract
    return local_gb10_quarter_profile(**profile_overrides).build_model(
        dtype=dtype,
        **hybrid_config_overrides,
    )


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
    "LOCAL_GB10_QUARTER_UPSTREAM_RECIPE_NAME",
    "LOCAL_GB10_QUARTER_TOKENIZER_BLOCKER_ID",
    "LOCAL_GB10_QUARTER_TOKENIZER_MILESTONE",
    "LOCAL_GB10_QUARTER_TOKENIZER_REQUIRED_SPECIALS",
    "LOCAL_GB10_QUARTER_VOCAB_SIZE",
    "MODEL_FACTORY_UPSTREAM_RECIPE_MODULE",
    "MTPProfile",
    "ModelFactoryProfile",
    "NAM56R_FULL_DEPTH",
    "NAM56R_FULL_DSA_A_LAYER_RANKS",
    "NAM56R_FULL_FFN_HIDDEN_SIZE",
    "NAM56R_FULL_HEAD_DIM",
    "NAM56R_FULL_HIDDEN_SIZE",
    "NAM56R_FULL_MAX_SEQ_LENGTH",
    "NAM56R_FULL_NUM_HEADS",
    "NAM56R_FULL_PATTERN",
    "NAM56R_FULL_PROFILE",
    "NAM56R_FULL_UPSTREAM_RECIPE_NAME",
    "NAM56R_FULL_VOCAB_SIZE",
    "TokenizerContractStatus",
    "build_local_gb10_quarter_tiny_smoke_model",
    "forward_has_finite_logits",
    "get_model_profile",
    "local_gb10_quarter",
    "local_gb10_quarter_profile",
    "nam56r_full_profile",
]
