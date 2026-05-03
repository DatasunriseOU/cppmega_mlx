"""MLX neural-network building blocks."""

from cppmega_mlx.nn.attention import AttentionConfig, AttentionRouteInfo, CausalSelfAttention
from cppmega_mlx.nn.engram import (
    CppMegaEngramBranch,
    EngramBranch,
    EngramConfig,
    causal_depthwise_silu_conv1d,
    causal_local_average,
    parse_ngram_orders,
)
from cppmega_mlx.nn.m2rnn import (
    DEFAULT_CHUNK_SIZE,
    M2RNNConfig,
    M2RNNMixer,
    M2RNNMixerState,
    broadcast_m2rnn_heads,
    chunked_m2rnn_scan,
    m2rnn_scan,
    m2rnn_softplus_decay_gate,
)
from cppmega_mlx.nn.mhc import (
    CppMegaManifoldBranchMixer,
    ManifoldBranchMixer,
    ManifoldBranchMixerConfig,
    sinkhorn_normalize,
)
from cppmega_mlx.nn.mamba3 import (
    Mamba3CacheState,
    Mamba3Config,
    Mamba3InProjDims,
    Mamba3ReferenceBlock,
    causal_depthwise_conv1d,
    compute_mamba3_in_proj_dims,
    compute_num_rope_angles,
)
from cppmega_mlx.nn.moe import (
    FeedForwardExpert,
    MoEConfig,
    MoEOutput,
    ReferenceMoE,
    RouterOutput,
    TopKRouter,
)
from cppmega_mlx.nn.ngram_hash import (
    CppMegaNgramHashEmbedding,
    NgramHashEmbedding,
    pick_primes,
)
from cppmega_mlx.nn.sparse_mla import (
    SparseMLAShapes,
    sparse_mla_attention,
    sparse_mla_attention_reference,
)
from cppmega_mlx.nn.structure_embedding import (
    CppMegaStructureEmbedding,
    StructureEmbedding,
)

__all__ = [
    "AttentionConfig",
    "AttentionRouteInfo",
    "CppMegaEngramBranch",
    "CppMegaManifoldBranchMixer",
    "CppMegaNgramHashEmbedding",
    "CppMegaStructureEmbedding",
    "CausalSelfAttention",
    "DEFAULT_CHUNK_SIZE",
    "EngramBranch",
    "EngramConfig",
    "FeedForwardExpert",
    "M2RNNConfig",
    "M2RNNMixer",
    "M2RNNMixerState",
    "Mamba3CacheState",
    "Mamba3Config",
    "Mamba3InProjDims",
    "Mamba3ReferenceBlock",
    "ManifoldBranchMixer",
    "ManifoldBranchMixerConfig",
    "MoEConfig",
    "MoEOutput",
    "NgramHashEmbedding",
    "ReferenceMoE",
    "RouterOutput",
    "SparseMLAShapes",
    "StructureEmbedding",
    "TopKRouter",
    "broadcast_m2rnn_heads",
    "causal_depthwise_conv1d",
    "causal_depthwise_silu_conv1d",
    "causal_local_average",
    "chunked_m2rnn_scan",
    "compute_mamba3_in_proj_dims",
    "compute_num_rope_angles",
    "m2rnn_scan",
    "m2rnn_softplus_decay_gate",
    "parse_ngram_orders",
    "pick_primes",
    "sinkhorn_normalize",
    "sparse_mla_attention",
    "sparse_mla_attention_reference",
]
