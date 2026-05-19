"""Auto-fusion layer over V4 bricks.

See ``Auto-FusionLayerBricks.md`` (repo root) for the full design.

This package walks composed UnifiedSuperblock graphs of ``BLOCK_BUILDERS``
bricks, detects fusion-eligible adjacencies via a table-driven oracle, and
hands compiled regions to ``cppmega_mlx.runtime.path_c_fusion`` for
TileLang lowering. Tensors cross fusion boundaries zero-copy via DLPack.

Stage A surface (this commit):
  - brick_graph.BrickNode / BrickGraph + walkers
  - dlpack_bridge.mlx_to_tilelang / tilelang_to_mlx
  - compatibility.FusionEligibility / can_fuse_pair

Stages B-E are TBD (see roadmap).
"""

from __future__ import annotations

from cppmega_v4.fusion.auto_compile import (
    AutoCompiledRegion,
    RegionPattern,
    auto_compile_plan,
    auto_compile_region,
    detect_region_pattern,
)
from cppmega_v4.fusion.auto_planner import (
    DEFAULT_MAX_REGION_SIZE,
    DEFAULT_MAX_SHARED_MEM_BYTES,
    FusionRegionPlan,
    auto_fuse_block_specs,
    auto_fuse_model,
    plan_fusion_regions,
)
from cppmega_v4.fusion.brick_graph import (
    BrickGraph,
    BrickNode,
    from_block_specs,
    from_mlx_model,
)
from cppmega_v4.fusion.compatibility import (
    FusionEligibility,
    can_fuse_pair,
)
from cppmega_v4.fusion.descriptor_synthesizer import (
    build_v4_extended_registry,
    synthesize_descriptor_for_brick,
)
from cppmega_v4.fusion.dlpack_bridge import (
    dlpack_available,
    host_copy_fallback,
    mlx_to_tilelang,
    tilelang_to_mlx,
)

__all__ = [
    "DEFAULT_MAX_REGION_SIZE",
    "DEFAULT_MAX_SHARED_MEM_BYTES",
    "AutoCompiledRegion",
    "BrickGraph",
    "BrickNode",
    "FusionEligibility",
    "FusionRegionPlan",
    "RegionPattern",
    "auto_compile_plan",
    "auto_compile_region",
    "auto_fuse_block_specs",
    "auto_fuse_model",
    "build_v4_extended_registry",
    "can_fuse_pair",
    "detect_region_pattern",
    "dlpack_available",
    "from_block_specs",
    "from_mlx_model",
    "host_copy_fallback",
    "mlx_to_tilelang",
    "plan_fusion_regions",
    "synthesize_descriptor_for_brick",
    "tilelang_to_mlx",
]
