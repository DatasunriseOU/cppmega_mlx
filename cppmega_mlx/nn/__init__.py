"""MLX neural-network building blocks."""

from cppmega_mlx.nn.attention import AttentionConfig, AttentionRouteInfo, CausalSelfAttention
from cppmega_mlx.nn.mamba3 import (
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

__all__ = [
    "AttentionConfig",
    "AttentionRouteInfo",
    "CausalSelfAttention",
    "FeedForwardExpert",
    "Mamba3Config",
    "Mamba3InProjDims",
    "Mamba3ReferenceBlock",
    "MoEConfig",
    "MoEOutput",
    "ReferenceMoE",
    "RouterOutput",
    "TopKRouter",
    "causal_depthwise_conv1d",
    "compute_mamba3_in_proj_dims",
    "compute_num_rope_angles",
]
