"""Pure-Python model config dataclasses for the NAM56R MLX port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cppmega_mlx.recipes.pattern import expand_nam_pattern

DEFAULT_NAM56R_PATTERN = "AEMEAEMEAEMR"
DEFAULT_NAM56R_DEPTH = 52
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_FFN_HIDDEN_SIZE = 21_504
DEFAULT_ATTENTION_HEADS = 32
DEFAULT_SEQ_LEN = 4096
DEFAULT_DSA_A_LAYER_RANKS = (1, 2, 3, 5, 6, 7, 9, 10, 11)

LOCAL_PROFILE_VOCAB_SIZE = 65_536
MEGACPP_TOKENIZER_VOCAB_SIZE = 131_072

MoeDispatcher = Literal["alltoall", "allgather", "flex"]
MoeFlexBackend = Literal["deepep", "hybridep"]
M2RNNKernel = Literal["triton", "torch", "mlx"]
STRUCTURE_COMPONENT_NAMES = frozenset(
    {"structure", "dep_level", "ast_depth", "sibling_index", "ast_node_type"}
)


@dataclass(frozen=True)
class VocabMetadata:
    """Keep both cppmega vocab contracts visible during the port.

    Local run profiles historically used 65,536, while the megacpp tokenizer and
    full NAM56R launchers use 131,072.  The MLX port should not collapse these
    into one anonymous integer.
    """

    local_profile_vocab_size: int = LOCAL_PROFILE_VOCAB_SIZE
    megacpp_tokenizer_vocab_size: int = MEGACPP_TOKENIZER_VOCAB_SIZE
    default_model_vocab_size: int = MEGACPP_TOKENIZER_VOCAB_SIZE
    make_vocab_size_divisible_by: int = 128

    def __post_init__(self) -> None:
        _require_positive("local_profile_vocab_size", self.local_profile_vocab_size)
        _require_positive("megacpp_tokenizer_vocab_size", self.megacpp_tokenizer_vocab_size)
        _require_positive("default_model_vocab_size", self.default_model_vocab_size)
        _require_positive("make_vocab_size_divisible_by", self.make_vocab_size_divisible_by)
        if self.default_model_vocab_size not in {
            self.local_profile_vocab_size,
            self.megacpp_tokenizer_vocab_size,
        }:
            raise ValueError(
                "default_model_vocab_size must match one preserved vocab contract"
            )


@dataclass(frozen=True)
class MoeConfig:
    num_experts: int = 16
    top_k: int = 4
    ffn_hidden_size: int = 896
    shared_expert_intermediate_size: int = 1024
    expert_model_parallel_size: int = 1
    token_dispatcher_type: MoeDispatcher = "alltoall"
    flex_dispatcher_backend: MoeFlexBackend = "deepep"
    router_dtype: str | None = "fp32"
    grouped_gemm: bool = True
    router_fusion: bool = True

    def __post_init__(self) -> None:
        _require_positive("num_experts", self.num_experts)
        _require_positive("top_k", self.top_k)
        _require_positive("ffn_hidden_size", self.ffn_hidden_size)
        _require_positive(
            "shared_expert_intermediate_size", self.shared_expert_intermediate_size
        )
        _require_positive("expert_model_parallel_size", self.expert_model_parallel_size)
        if self.top_k > self.num_experts:
            raise ValueError("MoE top_k must be <= num_experts")
        if self.token_dispatcher_type not in {"alltoall", "allgather", "flex"}:
            raise ValueError(
                f"unsupported MoE token_dispatcher_type={self.token_dispatcher_type!r}"
            )
        if self.flex_dispatcher_backend not in {"deepep", "hybridep"}:
            raise ValueError(
                f"unsupported MoE flex_dispatcher_backend={self.flex_dispatcher_backend!r}"
            )
        if self.token_dispatcher_type == "flex" and self.router_dtype != "fp32":
            raise ValueError("MoE flex dispatcher requires router_dtype='fp32'")


@dataclass(frozen=True)
class M2RNNConfig:
    d_model: int = DEFAULT_HIDDEN_SIZE
    k_head_dim: int = 64
    v_head_dim: int = 16
    conv_kernel: int = 4
    gradient_clipping: float = 1.0
    use_residual: bool = True
    a_init_min: float = 0.0
    a_init_max: float = 16.0
    dt_init_min: float = 1e-3
    dt_init_max: float = 0.1
    dt_init_floor: float = 1e-4
    use_xma: bool = False
    runtime_kernel: M2RNNKernel = "triton"
    runtime_save_hnew: bool = False
    runtime_bwd_chunk_size: int = 64
    runtime_fwd_autotune: bool = False
    runtime_fwd_num_warps: int = 4
    runtime_fwd_num_stages: int = 3
    runtime_broadcast_views: bool = True
    runtime_bwd_reduce_broadcast_qk: bool = False

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        _require_positive("k_head_dim", self.k_head_dim)
        _require_positive("v_head_dim", self.v_head_dim)
        _require_positive("conv_kernel", self.conv_kernel)
        _require_positive("runtime_bwd_chunk_size", self.runtime_bwd_chunk_size)
        _require_positive("runtime_fwd_num_warps", self.runtime_fwd_num_warps)
        _require_positive("runtime_fwd_num_stages", self.runtime_fwd_num_stages)
        if self.runtime_kernel not in {"triton", "torch", "mlx"}:
            raise ValueError(f"unsupported M2RNN runtime_kernel={self.runtime_kernel!r}")
        if self.a_init_min > self.a_init_max:
            raise ValueError("a_init_min must be <= a_init_max")
        if self.dt_init_min > self.dt_init_max:
            raise ValueError("dt_init_min must be <= dt_init_max")
        if self.gradient_clipping <= 0:
            raise ValueError("gradient_clipping must be positive")


@dataclass(frozen=True)
class StructureConfig:
    """Source-equivalent cppmega structure embedding metadata."""

    active_components: str = "core"
    bottleneck_dim: int = 64
    num_categories: int = 9
    max_dep_level: int = 16
    max_ast_depth: int = 64
    max_sibling_index: int = 64
    num_node_types: int = 256

    @property
    def component_names(self) -> tuple[str, ...]:
        return _parse_structure_components(self.active_components)

    def __post_init__(self) -> None:
        _require_positive("bottleneck_dim", self.bottleneck_dim)
        _require_positive("num_categories", self.num_categories)
        _require_positive("max_dep_level", self.max_dep_level)
        _require_positive("max_ast_depth", self.max_ast_depth)
        _require_positive("max_sibling_index", self.max_sibling_index)
        _require_positive("num_node_types", self.num_node_types)
        self.component_names


@dataclass(frozen=True)
class SourceStructureEnvConfig:
    """Megatron custom_embedding.py env-gated structure defaults.

    This is intentionally separate from StructureConfig: the standalone
    source module defaults are wider, while the Megatron embedding seam fills in
    smaller env defaults when CPPMEGA_STRUCTURE_ENABLED=1.
    """

    enabled: bool = False
    active_components: str = "core"
    max_ast_depth: int = 20
    max_sibling_index: int = 10
    num_node_types: int = 64
    bottleneck_dim: int = 64

    @property
    def component_names(self) -> tuple[str, ...]:
        return _parse_structure_components(self.active_components)

    def __post_init__(self) -> None:
        _require_positive("max_ast_depth", self.max_ast_depth)
        _require_positive("max_sibling_index", self.max_sibling_index)
        _require_positive("num_node_types", self.num_node_types)
        _require_positive("bottleneck_dim", self.bottleneck_dim)
        self.component_names


@dataclass(frozen=True)
class NgramHashConfig:
    """Megatron custom_embedding.py env-gated ngram hash defaults."""

    enabled: bool = False
    orders: tuple[int, ...] = (2, 3)
    num_heads: int = 8
    table_size: int = 500_000
    embed_dim: int = 16
    dropout: float = 0.0
    offload: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.orders:
            raise ValueError("ngram orders must contain at least one order")
        if any(order <= 0 for order in self.orders):
            raise ValueError("ngram orders must be positive")
        _require_positive("num_heads", self.num_heads)
        _require_positive("table_size", self.table_size)
        _require_positive("embed_dim", self.embed_dim)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ngram dropout must be in [0, 1)")


@dataclass(frozen=True)
class Mamba3Config:
    d_model: int = DEFAULT_HIDDEN_SIZE
    state_dim: int = 64
    expand: int = 2
    head_dim: int = 64
    num_groups: int = 8
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    a_floor: float = 1e-4
    is_outproj_norm: bool = False
    is_mimo: bool = True
    mimo_rank: int = 4
    chunk_size: int = 64
    recompute: bool = True

    @property
    def native_num_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def author_num_heads(self) -> int:
        return self.d_model * self.expand // self.head_dim

    def __post_init__(self) -> None:
        _require_positive("d_model", self.d_model)
        _require_positive("state_dim", self.state_dim)
        _require_positive("expand", self.expand)
        _require_positive("head_dim", self.head_dim)
        _require_positive("num_groups", self.num_groups)
        _require_positive("mimo_rank", self.mimo_rank)
        _require_positive("chunk_size", self.chunk_size)
        if self.d_model % self.head_dim != 0:
            raise ValueError("d_model must be divisible by Mamba3 head_dim")
        if (self.d_model * self.expand) % self.head_dim != 0:
            raise ValueError("d_model * expand must be divisible by Mamba3 head_dim")
        if not 0.0 <= self.rope_fraction <= 1.0:
            raise ValueError("rope_fraction must be in [0, 1]")
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if self.dt_init_floor <= 0 or self.a_floor <= 0:
            raise ValueError("Mamba3 floors must be positive")


@dataclass(frozen=True)
class MLAConfig:
    q_lora_rank: int = 64
    kv_lora_rank: int = 64
    qk_head_dim: int = 64
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 64

    def __post_init__(self) -> None:
        _require_positive("q_lora_rank", self.q_lora_rank)
        _require_positive("kv_lora_rank", self.kv_lora_rank)
        _require_positive("qk_head_dim", self.qk_head_dim)
        _require_positive("qk_pos_emb_head_dim", self.qk_pos_emb_head_dim)
        _require_positive("v_head_dim", self.v_head_dim)


@dataclass(frozen=True)
class DSAConfig:
    a_layer_ranks: tuple[int, ...] = DEFAULT_DSA_A_LAYER_RANKS
    indexer_n_heads: int = 32
    indexer_head_dim: int = 64
    indexer_topk: int = 256
    indexer_loss_coeff: float = 0.001
    indexer_dtype: str = "bf16"

    def __post_init__(self) -> None:
        _require_positive_tuple("a_layer_ranks", self.a_layer_ranks)
        _require_positive("indexer_n_heads", self.indexer_n_heads)
        _require_positive("indexer_head_dim", self.indexer_head_dim)
        _require_positive("indexer_topk", self.indexer_topk)
        if self.indexer_loss_coeff < 0:
            raise ValueError("indexer_loss_coeff must be non-negative")
        if self.indexer_dtype != "bf16":
            raise ValueError("DSA indexer_dtype must be 'bf16'")


@dataclass(frozen=True)
class Nam56RModelConfig:
    pattern: str = DEFAULT_NAM56R_PATTERN
    depth: int = DEFAULT_NAM56R_DEPTH
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    ffn_hidden_size: int = DEFAULT_FFN_HIDDEN_SIZE
    num_attention_heads: int = DEFAULT_ATTENTION_HEADS
    seq_len: int = DEFAULT_SEQ_LEN
    max_position_embeddings: int = DEFAULT_SEQ_LEN
    vocab: VocabMetadata = field(default_factory=VocabMetadata)
    moe: MoeConfig = field(default_factory=MoeConfig)
    m2rnn: M2RNNConfig = field(default_factory=M2RNNConfig)
    mamba3: Mamba3Config = field(default_factory=Mamba3Config)
    mla: MLAConfig = field(default_factory=MLAConfig)
    dsa: DSAConfig = field(default_factory=DSAConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    source_structure_env: SourceStructureEnvConfig = field(
        default_factory=SourceStructureEnvConfig
    )
    ngram_hash: NgramHashConfig = field(default_factory=NgramHashConfig)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def vocab_size(self) -> int:
        return self.vocab.default_model_vocab_size

    def __post_init__(self) -> None:
        _require_positive("depth", self.depth)
        _require_positive("hidden_size", self.hidden_size)
        _require_positive("ffn_hidden_size", self.ffn_hidden_size)
        _require_positive("num_attention_heads", self.num_attention_heads)
        _require_positive("seq_len", self.seq_len)
        _require_positive("max_position_embeddings", self.max_position_embeddings)
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.max_position_embeddings < self.seq_len:
            raise ValueError("max_position_embeddings must be >= seq_len")
        if self.m2rnn.d_model != self.hidden_size:
            raise ValueError("M2RNN d_model must match hidden_size")
        if self.mamba3.d_model != self.hidden_size:
            raise ValueError("Mamba3 d_model must match hidden_size")
        expand_nam_pattern(
            self.pattern,
            self.depth,
            dsa_a_layer_ranks=self.dsa.a_layer_ranks,
        )


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_tuple(name: str, values: tuple[int, ...]) -> None:
    seen: set[int] = set()
    for value in values:
        if value < 0:
            raise ValueError(f"{name} values must be non-negative")
        if value in seen:
            raise ValueError(f"{name} values must be unique")
        seen.add(value)


def _parse_structure_components(spec: str) -> tuple[str, ...]:
    if not isinstance(spec, str):
        raise TypeError("active_components must be a string")
    normalized = spec.strip()
    if normalized == "core":
        return ("structure", "dep_level")
    if normalized == "all":
        return ("structure", "dep_level", "ast_depth", "sibling_index", "ast_node_type")
    requested = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not requested:
        raise ValueError("active_components must select at least one component")
    invalid = sorted(set(requested) - STRUCTURE_COMPONENT_NAMES)
    if invalid:
        raise ValueError(f"unknown structure components: {invalid!r}")
    ordered = ("structure", "dep_level", "ast_depth", "sibling_index", "ast_node_type")
    return tuple(name for name in ordered if name in requested)
