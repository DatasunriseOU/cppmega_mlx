"""MLX neural-network building blocks.

Public MLX-backed exports are loaded on first attribute access so portable
Torch/TileLang submodules remain importable on CUDA hosts without Apple MLX.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AttentionConfig": "cppmega_mlx.nn.attention",
    "AttentionRouteInfo": "cppmega_mlx.nn.attention",
    "CppMegaEngramBranch": "cppmega_mlx.nn.engram",
    "CppMegaManifoldBranchMixer": "cppmega_mlx.nn.mhc",
    "CppMegaNgramHashEmbedding": "cppmega_mlx.nn.ngram_hash",
    "CppMegaPlatformEmbedding": "cppmega_mlx.nn.platform_embedding",
    "CppMegaStructureEmbedding": "cppmega_mlx.nn.structure_embedding",
    "CausalSelfAttention": "cppmega_mlx.nn.attention",
    "DEFAULT_CHUNK_SIZE": "cppmega_mlx.nn.m2rnn",
    "EngramBranch": "cppmega_mlx.nn.engram",
    "EngramConfig": "cppmega_mlx.nn.engram",
    "FeedForwardExpert": "cppmega_mlx.nn.moe",
    "M2RNNConfig": "cppmega_mlx.nn.m2rnn",
    "M2RNNMixer": "cppmega_mlx.nn.m2rnn",
    "M2RNNMixerState": "cppmega_mlx.nn.m2rnn",
    "Mamba3CacheState": "cppmega_mlx.nn.mamba3",
    "Mamba3Config": "cppmega_mlx.nn.mamba3",
    "Mamba3InProjDims": "cppmega_mlx.nn.mamba3",
    "Mamba3ReferenceBlock": "cppmega_mlx.nn.mamba3",
    "ManifoldBranchMixer": "cppmega_mlx.nn.mhc",
    "ManifoldBranchMixerConfig": "cppmega_mlx.nn.mhc",
    "MoEConfig": "cppmega_mlx.nn.moe",
    "MoEOutput": "cppmega_mlx.nn.moe",
    "NgramHashEmbedding": "cppmega_mlx.nn.ngram_hash",
    "PATH_B_REFERENCE_BASELINE_KERNEL": "cppmega_mlx.nn.m2rnn",
    "PlatformEmbedding": "cppmega_mlx.nn.platform_embedding",
    "ReferenceMoE": "cppmega_mlx.nn.moe",
    "RouterOutput": "cppmega_mlx.nn.moe",
    "SparseMLAShapes": "cppmega_mlx.nn.sparse_mla",
    "StructureEmbedding": "cppmega_mlx.nn.structure_embedding",
    "TopKRouter": "cppmega_mlx.nn.moe",
    "broadcast_m2rnn_heads": "cppmega_mlx.nn.m2rnn",
    "causal_depthwise_conv1d": "cppmega_mlx.nn.mamba3",
    "causal_depthwise_silu_conv1d": "cppmega_mlx.nn.engram",
    "causal_local_average": "cppmega_mlx.nn.engram",
    "chunked_m2rnn_scan": "cppmega_mlx.nn.m2rnn",
    "compute_mamba3_in_proj_dims": "cppmega_mlx.nn.mamba3",
    "compute_num_rope_angles": "cppmega_mlx.nn.mamba3",
    "m2rnn_scan": "cppmega_mlx.nn.m2rnn",
    "m2rnn_softplus_decay_gate": "cppmega_mlx.nn.m2rnn",
    "parse_ngram_orders": "cppmega_mlx.nn.engram",
    "pick_primes": "cppmega_mlx.nn.ngram_hash",
    "sinkhorn_normalize": "cppmega_mlx.nn.mhc",
    "sparse_mla_attention": "cppmega_mlx.nn.sparse_mla",
    "sparse_mla_attention_reference": "cppmega_mlx.nn.sparse_mla",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
