"""cppmega Path D runtime adapter for Triton-frontend TileLang PrimFuncs.

This module is deliberately cppmega-side glue. The Triton frontend owns
``TTIR -> PrimFunc``. cppmega owns the recurrent public signatures, grid
specialization, output policy, kernel caching, launch eligibility, and the
multi-kernel plan needed to turn FLA chunks into ``(y, h_last)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional


ALLOW_DEGRADED_ENV = "CPPMEGA_V4_PATH_D_ALLOW_DEGRADED_PRIMFUNC"


class PathDRuntimeUnavailable(RuntimeError):
    """Raised when Path D is selected but the runtime adapter cannot run."""


@dataclass(frozen=True)
class PathDKernelPlan:
    """Compile/launch metadata for one Triton-derived TileLang kernel."""

    name: str
    out_idx: tuple[int, ...]
    grid: tuple[int, ...]
    target: str = "metal"
    execution_backend: str = "tvm_ffi"
    allow_degraded_primfunc: bool = False


@dataclass(frozen=True)
class PathDRecurrentPlan:
    """Public cppmega recurrent API plan expressed as FLA kernel stages."""

    name: str
    public_signature: str
    output_layout: str
    stages: tuple[PathDKernelPlan, ...]
    note: str


@dataclass
class PathDCompileResult:
    """Result of compiling one TileLang PrimFunc into a runtime artifact."""

    available: bool
    reason: str
    artifact: Optional[Any] = None
    plan: Optional[PathDKernelPlan] = None
    degraded_primfunc: bool = False
    error_type: Optional[str] = None

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the compiled artifact if present."""

        if not self.available or self.artifact is None:
            raise PathDRuntimeUnavailable(self.reason)
        return self.artifact(*args, **kwargs)


GDN_CHUNK_H_PLAN = PathDKernelPlan(
    name="gdn.chunk_delta_h",
    out_idx=(6,),
    grid=(1, 1),
)
GDN_CHUNK_O_PLAN = PathDKernelPlan(
    name="gdn.chunk_o",
    out_idx=(6,),
    grid=(1, 1, 1),
)
GDN_RECURRENT_PLAN = PathDRecurrentPlan(
    name="gdn",
    public_signature="gdn(q, k, v, beta, g, *, scale, initial_state, output_final_state)",
    output_layout="returns y[B,T,H,V] and optional h_last[B,H,K,V]",
    stages=(GDN_CHUNK_H_PLAN, GDN_CHUNK_O_PLAN),
    note="FLA chunk_delta_h produces recurrent state; chunk_o produces output.",
)

KDA_RECURRENT_PLAN = PathDRecurrentPlan(
    name="kda",
    public_signature="kda(q, k, v, g, beta, *, scale, initial_state, output_final_state)",
    output_layout="returns y[B,T,HV,V] and optional h_last[B,HV,K,V]",
    stages=(
        PathDKernelPlan("kda.intra_token_parallel", out_idx=(4, 5), grid=(1, 1, 1)),
        PathDKernelPlan("kda.intra_sub_chunk", out_idx=(4, 5), grid=(1, 1, 1)),
        PathDKernelPlan("kda.inter_solve", out_idx=(5, 6), grid=(1, 1, 1)),
        GDN_CHUNK_H_PLAN,
        GDN_CHUNK_O_PLAN,
    ),
    note="KDA forward is a staged FLA pipeline, not a single callable kernel.",
)


def _env_allows_degraded() -> bool:
    return os.environ.get(ALLOW_DEGRADED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def primfunc_has_degraded_markers(prim_func: Any) -> bool:
    """Return True when frontend emitted visible degraded breadcrumbs."""

    try:
        text = prim_func.script()
    except Exception:
        text = str(prim_func)
    return "# DEGRADED:" in text or "DEGRADED:" in text


def specialize_primfunc_for_grid(prim_func: Any, grid: tuple[int, ...]) -> Any:
    """Specialize ``gridDim_<axis>`` PrimFunc params to concrete ints."""

    mapping: dict[Any, int] = {}
    for param in getattr(prim_func, "params", ()):
        name = str(param)
        if not name.startswith("gridDim_"):
            continue
        try:
            axis = int(name.rsplit("_", 1)[1])
        except ValueError:
            continue
        mapping[param] = int(grid[axis]) if axis < len(grid) else 1
    if not mapping:
        return prim_func
    return prim_func.specialize(mapping)


def _default_compile_fn(
    prim_func: Any,
    *,
    plan: PathDKernelPlan,
) -> Any:
    import tilelang
    from cppmega_mlx.nn._tilelang import _msl_transform

    target = (
        _msl_transform._as_metal_target("metal")
        if plan.target == "metal"
        else plan.target
    )
    return tilelang.compile(
        prim_func,
        target=target,
        execution_backend=plan.execution_backend,
        out_idx=list(plan.out_idx),
    )


def compile_tilelang_primfunc(
    prim_func: Any,
    plan: PathDKernelPlan,
    *,
    compile_fn: Optional[Callable[..., Any]] = None,
) -> PathDCompileResult:
    """Specialize, validate, compile, and wrap one PrimFunc.

    The degraded marker check is intentionally before compile: those markers
    mean pointer reconstruction fell back to scalar placeholder addressing, so
    launching would be a correctness risk even if Metal codegen succeeds.
    """

    if prim_func is None:
        return PathDCompileResult(
            available=False,
            reason=f"runtime adapter {plan.name}: no PrimFunc to compile",
            plan=plan,
        )

    specialized = specialize_primfunc_for_grid(prim_func, plan.grid)
    degraded = primfunc_has_degraded_markers(specialized)
    allow_degraded = plan.allow_degraded_primfunc or _env_allows_degraded()
    if degraded and not allow_degraded:
        return PathDCompileResult(
            available=False,
            reason=(
                f"runtime adapter {plan.name}: PrimFunc contains DEGRADED "
                "pointer-lowering markers; refusing to compile/launch until "
                "PtrAnalysis-backed addressing is clean"
            ),
            plan=plan,
            degraded_primfunc=True,
        )

    try:
        compiler = compile_fn
        if compiler is None:
            artifact = _default_compile_fn(specialized, plan=plan)
        else:
            artifact = compiler(specialized, plan=plan)
    except Exception as exc:  # noqa: BLE001
        return PathDCompileResult(
            available=False,
            reason=(
                f"runtime adapter {plan.name}: tilelang.compile failed: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            plan=plan,
            degraded_primfunc=degraded,
            error_type=exc.__class__.__name__,
        )
    return PathDCompileResult(
        available=True,
        reason=f"runtime adapter {plan.name}: compiled TileLang artifact",
        artifact=artifact,
        plan=plan,
        degraded_primfunc=degraded,
    )


def _freeze_items(items: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(items.items()))


@lru_cache(maxsize=16)
def _compile_gdn_chunk_h_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import lower_fla_chunk_h

    constexprs = dict(constexprs_key)
    lowered = lower_fla_chunk_h(constexprs)
    if lowered.status != "LOWERED_FULL" or lowered.prim_func is None:
        return PathDCompileResult(
            available=False,
            reason=(
                "runtime adapter gdn.chunk_delta_h: frontend did not produce "
                f"a runnable PrimFunc; status={lowered.status}; "
                f"error={lowered.error_type}: {lowered.error_message}"
            ),
            plan=GDN_CHUNK_H_PLAN,
            error_type=lowered.error_type,
        )
    plan = PathDKernelPlan(
        name=GDN_CHUNK_H_PLAN.name,
        out_idx=GDN_CHUNK_H_PLAN.out_idx,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )
    return compile_tilelang_primfunc(lowered.prim_func, plan)


def compile_gdn_chunk_h_artifact(
    *,
    constexprs: Optional[dict[str, Any]] = None,
    grid: tuple[int, ...] = (1, 1),
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache the currently lowerable GDN Path D chunk-h artifact."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import DEFAULT_CONSTEXPRS

    cfg = dict(DEFAULT_CONSTEXPRS)
    if constexprs:
        cfg.update(constexprs)
    return _compile_gdn_chunk_h_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        bool(allow_degraded_primfunc),
    )


def gdn_runtime_adapter_status() -> tuple[bool, str]:
    """Return public GDN Path D runtime availability and blocker."""

    result = compile_gdn_chunk_h_artifact()
    if not result.available:
        return False, (
            f"GDN Path D runtime adapter installed for {GDN_RECURRENT_PLAN.public_signature}; "
            f"{result.reason}; planned stages="
            f"{', '.join(stage.name for stage in GDN_RECURRENT_PLAN.stages)}"
        )
    return False, (
        "GDN Path D runtime adapter compiled chunk_delta_h, but full public "
        "GDN launch still requires chunk_o lowering plus zero-copy binding "
        "of cppmega outputs/state; Path D remains disabled"
    )


def kda_runtime_adapter_status(coverage_reason: str) -> tuple[bool, str]:
    """Return public KDA Path D runtime availability and blocker."""

    return False, (
        f"KDA Path D runtime adapter installed for {KDA_RECURRENT_PLAN.public_signature}; "
        f"{coverage_reason}; planned stages="
        f"{', '.join(stage.name for stage in KDA_RECURRENT_PLAN.stages)}; "
        "multi-kernel launch is gated until every stage has a non-degraded "
        "PrimFunc and explicit output/state buffer ownership"
    )


def gdn_fwd_runtime_call(*args: Any, **kwargs: Any) -> Any:
    """Path D public call hook. Raises so dispatch can fallback cleanly."""

    ok, reason = gdn_runtime_adapter_status()
    if not ok:
        raise PathDRuntimeUnavailable(reason)
    raise PathDRuntimeUnavailable(
        "GDN Path D runtime adapter unexpectedly reported available without "
        "a public launch implementation"
    )


def kda_fwd_runtime_call(*args: Any, coverage_reason: str = "", **kwargs: Any) -> Any:
    """KDA Path D public call hook. Raises so dispatch can fallback cleanly."""

    ok, reason = kda_runtime_adapter_status(coverage_reason or "coverage not probed")
    if not ok:
        raise PathDRuntimeUnavailable(reason)
    raise PathDRuntimeUnavailable(
        "KDA Path D runtime adapter unexpectedly reported available without "
        "a public launch implementation"
    )


__all__ = [
    "ALLOW_DEGRADED_ENV",
    "GDN_RECURRENT_PLAN",
    "KDA_RECURRENT_PLAN",
    "PathDCompileResult",
    "PathDKernelPlan",
    "PathDRecurrentPlan",
    "PathDRuntimeUnavailable",
    "compile_gdn_chunk_h_artifact",
    "compile_tilelang_primfunc",
    "gdn_fwd_runtime_call",
    "gdn_runtime_adapter_status",
    "kda_fwd_runtime_call",
    "kda_runtime_adapter_status",
    "primfunc_has_degraded_markers",
    "specialize_primfunc_for_grid",
]
