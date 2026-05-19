"""Stage E — pattern-matching auto_compile over FusionRegionPlans.

Given a region plan from :func:`plan_fusion_regions`, this module:

  1. Detects which TileLang schedule template best fits the region
     (LinearAttn+Norm, SDPA+OProj, MoE route/combine, etc.).
  2. Pulls the per-brick descriptors from the V4-extended registry.
  3. Asks ``cppmega_mlx.runtime.path_c_fusion_schedules`` for a callable
     schedule template that accepts a FusionRegion and returns a TIR
     PrimFunc — the *codegen* artefact Path C lowering consumes.

The plugin (``cppmega_mlx``) is invoked **read-only** via its public
:func:`make_path_c_descriptor_schedule_template` API — nothing inside
``cppmega_mlx`` is modified. When a region's pattern is purely a
single-brick passthrough, or when the planner picked a
``dlpack_handoff`` backend, no template is built (compilation falls
through to the brick's own native kernel — Path B Metal, mlx-lm SDPA,
or upstream).

This module is the bridge between the planner (Stage C) and Path C
codegen; it does not itself emit MSL / metallib — that happens later
when the resulting template is passed to :func:`compile_path_c_region`.

Public surface:
  - :class:`RegionPattern` — enumerated pattern label
  - :class:`AutoCompiledRegion` — planner output + pattern + template
  - :func:`detect_region_pattern`
  - :func:`auto_compile_region`
  - :func:`auto_compile_plan`
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from cppmega_mlx.runtime.path_c_fusion_schedules import (
    PathCBrickScheduleDescriptor,
    PathCBrickScheduleDescriptorRegistry,
    make_path_c_descriptor_schedule_template,
)
from cppmega_v4.fusion.auto_planner import FusionRegionPlan
from cppmega_v4.fusion.descriptor_synthesizer import build_v4_extended_registry


class RegionPattern(str, Enum):
    """Recognised region shapes. Strings so they're JSON-friendly."""

    SINGLE_BRICK_PASSTHROUGH = "single_brick_passthrough"
    DLPACK_HANDOFF_CHAIN = "dlpack_handoff_chain"
    LINEAR_ATTN_SCAN_WITH_NORM = "linear_attn_scan_with_norm"
    LINEAR_ATTN_SCAN = "linear_attn_scan"
    SSM_CHUNKWISE_SCAN = "ssm_chunkwise_scan"
    SDPA_WITH_OUTPUT_PROJ = "sdpa_with_output_proj"
    MOE_ROUTE_COMBINE = "moe_route_combine"
    NORM_OR_PROJ_CHAIN = "norm_or_proj_chain"
    GENERIC_DESCRIPTOR_TEMPLATE = "generic_descriptor_template"


def detect_region_pattern(plan: FusionRegionPlan) -> RegionPattern:
    """Classify a FusionRegionPlan by category mix and backend.

    Detection rules (first match wins):

      1. size == 1                                        → SINGLE_BRICK_PASSTHROUGH
      2. backend == "dlpack_handoff"                      → DLPACK_HANDOFF_CHAIN
      3. linear_attn ∈ cats AND norm_or_proj ∈ cats       → LINEAR_ATTN_SCAN_WITH_NORM
      4. all linear_attn                                  → LINEAR_ATTN_SCAN
      5. all ssm                                          → SSM_CHUNKWISE_SCAN
      6. sdpa_attention ∈ cats AND norm_or_proj ∈ cats    → SDPA_WITH_OUTPUT_PROJ
      7. moe ∈ cats AND norm_or_proj ∈ cats               → MOE_ROUTE_COMBINE
      8. all norm_or_proj                                 → NORM_OR_PROJ_CHAIN
      9. else                                             → GENERIC_DESCRIPTOR_TEMPLATE
    """
    if plan.size == 1:
        return RegionPattern.SINGLE_BRICK_PASSTHROUGH
    if plan.backend == "dlpack_handoff":
        return RegionPattern.DLPACK_HANDOFF_CHAIN

    cats = set(plan.categories)
    if "linear_attn" in cats and "norm_or_proj" in cats:
        return RegionPattern.LINEAR_ATTN_SCAN_WITH_NORM
    if cats == {"linear_attn"}:
        return RegionPattern.LINEAR_ATTN_SCAN
    if cats == {"ssm"}:
        return RegionPattern.SSM_CHUNKWISE_SCAN
    if "sdpa_attention" in cats and "norm_or_proj" in cats:
        return RegionPattern.SDPA_WITH_OUTPUT_PROJ
    if "moe" in cats and "norm_or_proj" in cats:
        return RegionPattern.MOE_ROUTE_COMBINE
    if cats == {"norm_or_proj"}:
        return RegionPattern.NORM_OR_PROJ_CHAIN
    return RegionPattern.GENERIC_DESCRIPTOR_TEMPLATE


_PATTERNS_WITHOUT_TEMPLATE: frozenset[RegionPattern] = frozenset({
    RegionPattern.SINGLE_BRICK_PASSTHROUGH,
    RegionPattern.DLPACK_HANDOFF_CHAIN,
})


@dataclass(frozen=True)
class AutoCompiledRegion:
    """A planner region plus the pattern label, the per-brick descriptors,
    and the TileLang schedule template that Path C codegen will consume.

    For size-1 regions and dlpack-handoff regions the schedule_template
    is ``None`` — the region runs through each brick's native kernel
    and the planner annotation is enough.
    """

    plan: FusionRegionPlan
    pattern: RegionPattern
    descriptors: tuple[PathCBrickScheduleDescriptor, ...]
    schedule_template: Callable[..., object] | None
    reason: str

    @property
    def has_compiled_template(self) -> bool:
        return self.schedule_template is not None


_DEFAULT_REGISTRY_CACHE: PathCBrickScheduleDescriptorRegistry | None = None


def _default_registry() -> PathCBrickScheduleDescriptorRegistry:
    global _DEFAULT_REGISTRY_CACHE
    if _DEFAULT_REGISTRY_CACHE is None:
        _DEFAULT_REGISTRY_CACHE = build_v4_extended_registry()
    return _DEFAULT_REGISTRY_CACHE


def _collect_descriptors(
    plan: FusionRegionPlan,
    registry: PathCBrickScheduleDescriptorRegistry,
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    """Return the descriptor for each brick in plan order.

    The brick *kind* is recovered from the per-position category-to-kind
    inverse via the plan's ``brick_names`` slot — but since the planner
    doesn't carry kinds in its output, we rely on each region's
    categories list aligning with the op_names already registered. In
    practice we use ``op_name == brick_name`` only when the brick name
    happens to match a registered op; otherwise we look up by the
    canonical kind stored in the BrickGraph the planner came from.

    To keep this self-contained, the function expects callers to use
    :func:`auto_compile_region` with the same BrickGraph used to build
    the plan, or to pass kinds explicitly.
    """
    raise NotImplementedError(
        "use auto_compile_region(plan, kinds=...) — descriptors are looked "
        "up by brick kind, which the FusionRegionPlan does not carry"
    )


def _build_template(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    pattern: RegionPattern,
) -> Callable[..., object]:
    """Wrap the cppmega_mlx descriptor->template machinery with a stable
    entry symbol derived from the pattern. The pattern label becomes
    part of the generated PrimFunc's symbol so codegen logs are
    self-documenting."""
    entry = f"v4_autocompile_{pattern.value}"
    return make_path_c_descriptor_schedule_template(
        descriptors, entry_symbol=entry
    )


def auto_compile_region(
    plan: FusionRegionPlan,
    kinds: Sequence[str],
    *,
    registry: PathCBrickScheduleDescriptorRegistry | None = None,
) -> AutoCompiledRegion:
    """Build the :class:`AutoCompiledRegion` for a single planner plan.

    ``kinds`` must be the brick kind for each entry of ``plan.brick_names``
    in the same order. The descriptors are looked up in the supplied
    registry (default: the V4-extended registry from
    :func:`cppmega_v4.fusion.descriptor_synthesizer.build_v4_extended_registry`).

    For pass-through patterns the descriptors are still returned (so
    callers can inspect them) but ``schedule_template`` is ``None``.
    """
    if len(kinds) != plan.size:
        raise ValueError(
            f"kinds length ({len(kinds)}) must match plan.size ({plan.size})"
        )
    reg = registry or _default_registry()
    descriptors: list[PathCBrickScheduleDescriptor] = []
    missing: list[str] = []
    for k in kinds:
        d = reg.descriptor_for(k)
        if d is None:
            missing.append(k)
        else:
            descriptors.append(d)
    if missing:
        raise KeyError(
            f"no descriptor registered for brick kind(s): {missing}; "
            "extend the registry (cppmega_v4.fusion.descriptor_synthesizer) "
            "or pass a custom registry"
        )

    pattern = detect_region_pattern(plan)
    if pattern in _PATTERNS_WITHOUT_TEMPLATE:
        return AutoCompiledRegion(
            plan=plan,
            pattern=pattern,
            descriptors=tuple(descriptors),
            schedule_template=None,
            reason=(
                "size-1 region — falls through to native kernel"
                if pattern is RegionPattern.SINGLE_BRICK_PASSTHROUGH
                else "dlpack_handoff backend — no fused PrimFunc emitted"
            ),
        )

    template = _build_template(descriptors, pattern)
    return AutoCompiledRegion(
        plan=plan,
        pattern=pattern,
        descriptors=tuple(descriptors),
        schedule_template=template,
        reason=f"pattern={pattern.value}; descriptor-template ready for codegen",
    )


def auto_compile_plan(
    plans: Sequence[FusionRegionPlan],
    kinds_by_name: dict[str, str],
    *,
    registry: PathCBrickScheduleDescriptorRegistry | None = None,
) -> list[AutoCompiledRegion]:
    """Apply :func:`auto_compile_region` to a list of plans.

    ``kinds_by_name`` maps every brick name across all plans to its kind
    — typically built once from the BrickGraph: ``{n.name: n.kind for n
    in graph.nodes}``.
    """
    out: list[AutoCompiledRegion] = []
    for plan in plans:
        kinds = [kinds_by_name[n] for n in plan.brick_names]
        out.append(auto_compile_region(plan, kinds, registry=registry))
    return out


__all__ = [
    "AutoCompiledRegion",
    "RegionPattern",
    "auto_compile_plan",
    "auto_compile_region",
    "detect_region_pattern",
]
