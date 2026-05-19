"""Stage B — auto-synthesize Path C brick descriptors for V4 bricks.

Background:
  ``cppmega_mlx.runtime.path_c_fusion_schedules.default_path_c_brick_schedule_descriptor_registry``
  ships 5 hand-written descriptors (legacy v3 train block). The 15+ V4
  bricks in ``cppmega_v4.models.unified_superblock_v4.BLOCK_BUILDERS``
  have no descriptors, so any region built from them fails the
  ``descriptors_for_signature`` lookup and Path C codegen bails.

Strategy:
  - For each V4 brick, synthesise a "fallback descriptor" that marks the
    op as an opaque kernel (already implemented in Path B / mlx-lm SDPA /
    upstream) but is structurally valid for the registry: non-empty
    op_name, well-formed required_codegen_steps, schedule_family chosen
    by the FusionEligibility category (linear_attn / sdpa_attention /
    moe / ssm / nonlinear_rnn / cross_attn / sparse_attn / norm_or_proj).
  - The fallback descriptor's ``implementation_status`` is
    ``"opaque_brick_passthrough"`` — that signals to downstream pattern
    matchers that the brick must NOT be inlined into a fused PrimFunc,
    but it CAN be grouped into a region (and the planner can lower
    surrounding bricks fused with DLPack hand-off on the brick's edges).

Public surface:
  - ``synthesize_descriptor_for_brick(kind) -> PathCBrickScheduleDescriptor``
  - ``build_v4_extended_registry() -> PathCBrickScheduleDescriptorRegistry``
    Combines the default v3 registry with v4 fallback descriptors. All
    BLOCK_BUILDERS kinds get a descriptor; existing v3 entries are kept
    as-is (no clobber).
"""

from __future__ import annotations

from cppmega_mlx.runtime.path_c_fusion_schedules import (
    PathCBrickScheduleDescriptor,
    PathCBrickScheduleDescriptorRegistry,
    default_path_c_brick_schedule_descriptor_registry,
)
from cppmega_v4.fusion.compatibility import _CATEGORY_BY_KIND
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


# Map fusion category -> schedule_family token (descriptor metadata).
# Categories that fuse via path_c get a family hint matching the kind of
# loop they're built around; opaque ones get a sentinel that downstream
# pattern matchers treat as "do not inline".
_FAMILY_BY_CATEGORY: dict[str, str] = {
    "linear_attn": "linear_attn_scan",
    "ssm": "ssm_chunkwise_scan",
    "sdpa_attention": "opaque_sdpa_with_outputs",
    "cross_attn": "opaque_cross_attn",
    "nonlinear_rnn": "opaque_recurrence",
    "moe": "moe_route_combine",
    "sparse_attn": "opaque_sparse",
    "norm_or_proj": "pointwise_or_linear",
    "unknown": "opaque_passthrough",
}


def _required_codegen_steps_for(kind: str, category: str) -> tuple[str, ...]:
    """Per-brick codegen step sequence. Used by descriptor consumers to
    decide which sub-templates to thread together. Conservative defaults
    so an unknown brick still passes the registry's structural check."""
    if category == "linear_attn":
        return (
            f"{kind}_scan_descriptor",
            f"{kind}_state_init",
        )
    if category == "ssm":
        return (
            f"{kind}_ssm_descriptor",
            f"{kind}_chunkwise_internal_buffers",
        )
    if category == "sdpa_attention":
        return (
            f"{kind}_opaque_sdpa_descriptor",
            f"{kind}_output_gate_o_proj_fragment",
        )
    if category == "cross_attn":
        return (
            f"{kind}_opaque_cross_attn_descriptor",
            f"{kind}_residual_output_fragment",
        )
    if category == "nonlinear_rnn":
        return (
            f"{kind}_opaque_recurrence_descriptor",
            f"{kind}_pre_norm_fragment",
        )
    if category == "moe":
        return (
            f"{kind}_route_descriptor",
            f"{kind}_expert_ffn_fragment",
            f"{kind}_combine_descriptor",
        )
    if category == "sparse_attn":
        return (
            f"{kind}_opaque_sparse_descriptor",
        )
    if category == "norm_or_proj":
        return (
            f"{kind}_pointwise_descriptor",
        )
    return (f"{kind}_opaque_passthrough",)


def _supports_backward(category: str) -> bool:
    # Conservative: only the categories with established gradient paths
    # in our codebase claim backward support. SDPA-backed and sparse
    # bricks have backwards via mlx-lm/MLX autodiff but not via Path C
    # AOT — the descriptor itself doesn't know how to emit a backward
    # PrimFunc fragment, so it sets supports_backward=False.
    return category in {"linear_attn", "ssm", "norm_or_proj", "nonlinear_rnn"}


def synthesize_descriptor_for_brick(kind: str) -> PathCBrickScheduleDescriptor:
    """Build a fallback descriptor for a single brick kind.

    Returns a structurally valid PathCBrickScheduleDescriptor that the
    registry accepts and that downstream consumers can treat as an
    opaque-brick handoff. Never raises for unknown kinds — falls back to
    ``opaque_passthrough`` family.
    """
    category = _CATEGORY_BY_KIND.get(kind, "unknown")
    family = _FAMILY_BY_CATEGORY.get(category, "opaque_passthrough")
    is_opaque = family.startswith("opaque_")
    return PathCBrickScheduleDescriptor(
        op_name=kind,
        implementation_status=(
            "opaque_brick_passthrough" if is_opaque else "descriptor_synth_v4"
        ),
        required_codegen_steps=_required_codegen_steps_for(kind, category),
        schedule_family=family,
        supports_backward=_supports_backward(category),
        description=(
            f"V4 fallback descriptor for {kind!r} (category={category}, "
            f"family={family}). Opaque bricks are grouped into regions "
            "via DLPack handoff; non-opaque get inline TileLang fragments."
        ),
        production_source=f"cppmega_v4.models.unified_superblock_v4:BLOCK_BUILDERS[{kind!r}]",
        production_fragment_status=(
            "opaque_brick" if is_opaque else "v4_descriptor_synth_pending"
        ),
        production_fragment_reason=(
            "auto-synthesized fallback; the brick already has a working "
            "Path B / mlx-lm SDPA implementation that the runtime calls "
            "via DLPack handoff. A future change can replace this with a "
            "real TileLang fragment when the schedule template lands."
        ),
    )


def build_v4_extended_registry() -> PathCBrickScheduleDescriptorRegistry:
    """Return a registry covering both legacy v3 bricks and all V4 bricks.

    Existing v3 descriptors win — we never clobber. Every key in
    BLOCK_BUILDERS that isn't already present gets a synthesised fallback
    descriptor. The result is a single registry that
    ``compile_path_c_region`` can use without ``None`` lookups.
    """
    registry = default_path_c_brick_schedule_descriptor_registry()
    for kind in BLOCK_BUILDERS:
        if registry.descriptor_for(kind) is not None:
            continue
        registry.register(synthesize_descriptor_for_brick(kind))
    return registry


__all__ = [
    "build_v4_extended_registry",
    "synthesize_descriptor_for_brick",
]
