"""cppmega Path D runtime adapter for Triton-frontend TileLang PrimFuncs.

This module is deliberately cppmega-side glue. The Triton frontend owns
``TTIR -> PrimFunc``. cppmega owns the recurrent public signatures, grid
specialization, output policy, kernel caching, launch eligibility, and the
multi-kernel plan needed to turn FLA chunks into ``(y, h_last)``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional


ALLOW_DEGRADED_ENV = "CPPMEGA_V4_PATH_D_ALLOW_DEGRADED_PRIMFUNC"
RCP_LN2 = 1.4426950408889634
GDN_FIXED_T = 64
GDN_FIXED_H = 1
GDN_FIXED_HV = 1
GDN_FIXED_K = 64
GDN_FIXED_V = 32
GDN_FIXED_BT = 64
GDN_FIXED_BV = 32
GDN_FIXED_CHUNK_O_BV = 16


class PathDRuntimeUnavailable(RuntimeError):
    """Raised when Path D is selected but the runtime adapter cannot run."""


@dataclass(frozen=True)
class PathDKernelPlan:
    """Compile/launch metadata for one Triton-derived TileLang kernel."""

    name: str
    out_idx: tuple[int, ...]
    grid: tuple[int, ...]
    scalar_specializations: tuple[Any, ...] = ()
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
    out_idx=(3, 6),
    grid=(1, 1),
    scalar_specializations=(64,),
)
GDN_KKT_PLAN = PathDKernelPlan(
    name="gdn.kkt_solve",
    out_idx=(3,),
    grid=(1, 1),
    scalar_specializations=(64,),
)
GDN_RECOMPUTE_W_U_PLAN = PathDKernelPlan(
    name="gdn.recompute_w_u",
    out_idx=(3, 4),
    grid=(1, 1),
    scalar_specializations=(64,),
)
GDN_CHUNK_O_PLAN = PathDKernelPlan(
    name="gdn.chunk_o",
    out_idx=(6,),
    grid=(math.ceil(GDN_FIXED_V / GDN_FIXED_CHUNK_O_BV), 1, 1),
    scalar_specializations=(1.0 / math.sqrt(64), 64),
)
GDN_RECURRENT_PLAN = PathDRecurrentPlan(
    name="gdn",
    public_signature="gdn(q, k, v, beta, g, *, scale, initial_state, output_final_state)",
    output_layout="returns y[B,T,H,V] and optional h_last[B,H,K,V]",
    stages=(
        GDN_KKT_PLAN,
        GDN_RECOMPUTE_W_U_PLAN,
        GDN_CHUNK_H_PLAN,
        GDN_CHUNK_O_PLAN,
    ),
    note=(
        "FLA KKT/recompute build intra-chunk state; chunk_delta_h produces "
        "recurrent state; chunk_o produces output."
    ),
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
    return _env_flag(ALLOW_DEGRADED_ENV)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
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


def specialize_primfunc_for_scalars(
    prim_func: Any,
    scalar_values: tuple[Any, ...],
) -> Any:
    """Specialize static scalar PrimFunc params such as FLA's ``T``."""

    if not scalar_values:
        return prim_func
    buffer_map = getattr(prim_func, "buffer_map", {}) or {}
    scalar_params = [
        param
        for param in getattr(prim_func, "params", ())
        if param not in buffer_map and not str(param).startswith("gridDim_")
    ]
    if len(scalar_values) == 1 and len(scalar_params) > 1:
        integer_params = [
            param
            for param in scalar_params
            if str(getattr(param, "dtype", "")).startswith(("int", "uint"))
        ]
        if len(integer_params) == 1:
            scalar_params = integer_params
    if len(scalar_values) > len(scalar_params):
        raise ValueError(
            f"plan requested {len(scalar_values)} scalar specializations, "
            f"but PrimFunc only has {len(scalar_params)} scalar params"
        )
    mapping = {}
    for idx, value in enumerate(scalar_values):
        param = scalar_params[idx]
        dtype = str(getattr(param, "dtype", ""))
        if dtype.startswith(("int", "uint")):
            mapping[param] = int(value)
        elif dtype.startswith(("float", "bfloat")):
            mapping[param] = float(value)
        else:
            mapping[param] = value
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

    try:
        specialized = specialize_primfunc_for_grid(prim_func, plan.grid)
        specialized = specialize_primfunc_for_scalars(
            specialized,
            plan.scalar_specializations,
        )
    except Exception as exc:  # noqa: BLE001
        return PathDCompileResult(
            available=False,
            reason=(
                f"runtime adapter {plan.name}: grid specialization failed: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            plan=plan,
            error_type=exc.__class__.__name__,
        )
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
    lowered = lower_fla_chunk_h(constexprs, grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=GDN_CHUNK_H_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )


def _compile_lowered_kernel_result(
    lowered: Any,
    *,
    base_plan: PathDKernelPlan,
    grid: tuple[int, ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    if lowered.status != "LOWERED_FULL" or lowered.prim_func is None:
        return PathDCompileResult(
            available=False,
            reason=(
                f"runtime adapter {base_plan.name}: frontend did not produce "
                f"a runnable PrimFunc; status={lowered.status}; "
                f"error={lowered.error_type}: {lowered.error_message}"
            ),
            plan=base_plan,
            error_type=lowered.error_type,
        )
    plan = PathDKernelPlan(
        name=base_plan.name,
        out_idx=base_plan.out_idx,
        grid=grid,
        scalar_specializations=base_plan.scalar_specializations,
        target=base_plan.target,
        execution_backend=base_plan.execution_backend,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )
    return compile_tilelang_primfunc(lowered.prim_func, plan)


@lru_cache(maxsize=16)
def _compile_gdn_kkt_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_gdn_kkt_solve,
    )

    lowered = lower_fla_gdn_kkt_solve(dict(constexprs_key), grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=GDN_KKT_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )


@lru_cache(maxsize=16)
def _compile_gdn_recompute_w_u_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_gdn_recompute_w_u,
    )

    lowered = lower_fla_gdn_recompute_w_u(dict(constexprs_key), grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=GDN_RECOMPUTE_W_U_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )


@lru_cache(maxsize=16)
def _compile_gdn_chunk_o_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import lower_fla_chunk_o

    constexprs = dict(constexprs_key)
    lowered = lower_fla_chunk_o(constexprs, grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=GDN_CHUNK_O_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )


def compile_gdn_kkt_artifact(
    *,
    constexprs: Optional[dict[str, Any]] = None,
    grid: tuple[int, ...] = (1, 1),
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache the GDN KKT solve TileLang artifact."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_GDN_KKT_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_GDN_KKT_CONSTEXPRS)
    if constexprs:
        cfg.update(constexprs)
    grid_tuple = tuple(int(x) for x in grid)
    return _compile_gdn_kkt_cached(
        _freeze_items(cfg),
        grid_tuple,
        bool(allow_degraded_primfunc),
    )


def compile_gdn_recompute_w_u_artifact(
    *,
    constexprs: Optional[dict[str, Any]] = None,
    grid: tuple[int, ...] = (1, 1),
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache the GDN recompute_w_u TileLang artifact."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_GDN_RECOMPUTE_W_U_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_GDN_RECOMPUTE_W_U_CONSTEXPRS)
    if constexprs:
        cfg.update(constexprs)
    return _compile_gdn_recompute_w_u_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        bool(allow_degraded_primfunc),
    )


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


def compile_gdn_chunk_o_artifact(
    *,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[tuple[int, ...]] = None,
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache the currently lowerable GDN Path D chunk-o artifact."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_CHUNK_O_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_CHUNK_O_CONSTEXPRS)
    cfg["BV"] = GDN_FIXED_CHUNK_O_BV
    if constexprs:
        cfg.update(constexprs)
    if grid is None:
        grid = (
            math.ceil(
                int(cfg.get("V", GDN_FIXED_V))
                / int(cfg.get("BV", GDN_FIXED_CHUNK_O_BV))
            ),
            1,
            1,
        )
    return _compile_gdn_chunk_o_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        bool(allow_degraded_primfunc),
    )


def _is_dtype(value: Any, dtype: Any) -> bool:
    return getattr(value, "dtype", None) == dtype


def _require_shape_dtype(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: Any,
) -> None:
    if tuple(getattr(value, "shape", ())) != shape:
        raise PathDRuntimeUnavailable(
            f"GDN Path D runtime adapter only supports {name}.shape={shape}; "
            f"got {getattr(value, 'shape', None)}"
        )
    if not _is_dtype(value, dtype):
        raise PathDRuntimeUnavailable(
            f"GDN Path D runtime adapter only supports {name}.dtype={dtype}; "
            f"got {getattr(value, 'dtype', None)}"
        )


def _flatten(value: Any) -> Any:
    import mlx.core as mx

    return mx.reshape(value, (-1,))


def _launch_stage(stage: PathDCompileResult, *args: Any) -> Any:
    if not stage.available:
        raise PathDRuntimeUnavailable(stage.reason)
    return stage.launch(*args)


def _bind_returned_outputs(
    stage: str,
    returned: Any,
    explicit_outputs: tuple[Any, ...],
) -> tuple[Any, ...]:
    if returned is None:
        return explicit_outputs
    returned_outputs = (
        tuple(returned) if isinstance(returned, (list, tuple)) else (returned,)
    )
    if len(returned_outputs) != len(explicit_outputs):
        raise PathDRuntimeUnavailable(
            f"{stage} returned {len(returned_outputs)} outputs for "
            f"{len(explicit_outputs)} explicit owner-output buffers"
        )
    for idx, (actual, expected) in enumerate(zip(returned_outputs, explicit_outputs)):
        if actual is not expected:
            raise PathDRuntimeUnavailable(
                f"{stage} did not preserve explicit output buffer ownership "
                f"for out_idx[{idx}]"
            )
    return returned_outputs


def _compile_gdn_runtime_stages(
    *,
    batch: int,
    use_initial_state: bool,
    output_final_state: bool,
) -> tuple[
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
]:
    nt = 1
    stage_grid = (nt, batch * GDN_FIXED_HV)
    chunk_v_grid = (
        math.ceil(GDN_FIXED_V / GDN_FIXED_BV),
        batch * GDN_FIXED_HV,
    )
    chunk_o_grid = (
        math.ceil(GDN_FIXED_V / GDN_FIXED_CHUNK_O_BV),
        nt,
        batch * GDN_FIXED_HV,
    )
    chunk_h_constexprs = {
        "USE_INITIAL_STATE": bool(use_initial_state),
        "STORE_FINAL_STATE": bool(output_final_state),
        "SAVE_NEW_VALUE": True,
    }
    return (
        compile_gdn_kkt_artifact(grid=stage_grid),
        compile_gdn_recompute_w_u_artifact(grid=stage_grid),
        compile_gdn_chunk_h_artifact(
            constexprs=chunk_h_constexprs,
            grid=chunk_v_grid,
        ),
        compile_gdn_chunk_o_artifact(
            constexprs={"BV": GDN_FIXED_CHUNK_O_BV},
            grid=chunk_o_grid,
        ),
    )


def _validate_default_scale(scale: Any) -> None:
    if scale is None:
        return
    default_scale = 1.0 / math.sqrt(GDN_FIXED_K)
    if not math.isclose(float(scale), default_scale, rel_tol=0.0, abs_tol=1e-7):
        raise PathDRuntimeUnavailable(
            "GDN Path D runtime adapter currently supports only the default "
            f"scale={default_scale}; got {scale}"
        )


def gdn_runtime_adapter_status() -> tuple[bool, str]:
    """Return public GDN Path D runtime availability and blocker."""

    checks = (
        compile_gdn_kkt_artifact,
        compile_gdn_recompute_w_u_artifact,
        compile_gdn_chunk_h_artifact,
        compile_gdn_chunk_o_artifact,
    )
    for compile_stage in checks:
        result = compile_stage()
        if not result.available:
            return False, (
                "GDN Path D runtime adapter installed for "
                f"{GDN_RECURRENT_PLAN.public_signature}; {result.reason}; "
                "planned stages="
                f"{', '.join(stage.name for stage in GDN_RECURRENT_PLAN.stages)}"
            )
    return True, (
        "GDN Path D runtime adapter available for fixed prefill "
        "B x 64 x H=1 x K=64 / V=32, USE_G cumulative gate, no varlen, "
        "default scale; compile/cache/launch stages="
        f"{', '.join(stage.name for stage in GDN_RECURRENT_PLAN.stages)}"
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


def gdn_fwd_runtime_call(
    q: Any,
    k: Any,
    v: Any,
    beta: Any,
    g: Any,
    *,
    scale: Any = None,
    initial_state: Any = None,
    output_final_state: bool = False,
    **kwargs: Any,
) -> Any:
    """Launch the constrained GDN Path D FLA chunk pipeline.

    This intentionally supports only the first runnable fixed-shape prefill
    slice. Unsupported signatures raise ``PathDRuntimeUnavailable`` so the
    dispatcher keeps the existing production fallback.
    """

    if kwargs:
        raise PathDRuntimeUnavailable(
            "GDN Path D runtime adapter does not support extra keyword args: "
            f"{sorted(kwargs)}"
        )

    import mlx.core as mx

    _validate_default_scale(scale)
    b = int(getattr(q, "shape", (0,))[0])
    expected_qk = (b, GDN_FIXED_T, GDN_FIXED_H, GDN_FIXED_K)
    expected_v = (b, GDN_FIXED_T, GDN_FIXED_HV, GDN_FIXED_V)
    expected_gate = (b, GDN_FIXED_T, GDN_FIXED_HV)

    _require_shape_dtype("q", q, expected_qk, mx.float16)
    _require_shape_dtype("k", k, expected_qk, mx.float16)
    _require_shape_dtype("v", v, expected_v, mx.float16)
    _require_shape_dtype("beta", beta, expected_gate, mx.float32)
    _require_shape_dtype("g", g, expected_gate, mx.float32)
    if initial_state is not None:
        _require_shape_dtype(
            "initial_state",
            initial_state,
            (b, GDN_FIXED_HV, GDN_FIXED_K, GDN_FIXED_V),
            mx.float32,
        )

    kkt, recompute, chunk_h, chunk_o = _compile_gdn_runtime_stages(
        batch=b,
        use_initial_state=initial_state is not None,
        output_final_state=bool(output_final_state),
    )
    for stage in (kkt, recompute, chunk_h, chunk_o):
        if not stage.available:
            raise PathDRuntimeUnavailable(stage.reason)

    nt = 1
    q_flat = _flatten(q)
    k_flat = _flatten(k)
    v_flat = _flatten(v)
    beta_flat = _flatten(beta)
    g_cumsum_flat = _flatten(mx.cumsum(g, axis=1) * RCP_LN2)

    a = mx.empty((b * GDN_FIXED_T * GDN_FIXED_HV * GDN_FIXED_BT,), dtype=mx.float16)
    w = mx.empty((b * GDN_FIXED_T * GDN_FIXED_HV * GDN_FIXED_K,), dtype=mx.float16)
    u = mx.empty((b * GDN_FIXED_T * GDN_FIXED_HV * GDN_FIXED_V,), dtype=mx.float16)
    v_new = mx.empty((b * GDN_FIXED_T * GDN_FIXED_HV * GDN_FIXED_V,), dtype=mx.float16)
    h = mx.empty((b * nt * GDN_FIXED_HV * GDN_FIXED_K * GDN_FIXED_V,), dtype=mx.float16)
    o = mx.empty((b * GDN_FIXED_T * GDN_FIXED_HV * GDN_FIXED_V,), dtype=mx.float16)
    h0 = (
        _flatten(initial_state)
        if initial_state is not None
        else mx.empty((b * GDN_FIXED_HV * GDN_FIXED_K * GDN_FIXED_V,), dtype=mx.float32)
    )
    ht = mx.empty((b * GDN_FIXED_HV * GDN_FIXED_K * GDN_FIXED_V,), dtype=mx.float32)
    gk = mx.empty((b * GDN_FIXED_T * GDN_FIXED_HV * GDN_FIXED_K,), dtype=mx.float32)
    g_gamma = mx.empty((GDN_FIXED_HV,), dtype=mx.float32)
    idx64 = mx.zeros((1,), dtype=mx.int64)
    idx32 = mx.zeros((1,), dtype=mx.float32)

    (a,) = _bind_returned_outputs(
        "gdn.kkt_solve",
        _launch_stage(kkt, k_flat, g_cumsum_flat, beta_flat, a, idx64, idx64),
        (a,),
    )
    w, u = _bind_returned_outputs(
        "gdn.recompute_w_u",
        _launch_stage(
            recompute,
            k_flat,
            v_flat,
            beta_flat,
            w,
            u,
            a,
            g_cumsum_flat,
            idx64,
            idx64,
        ),
        (w, u),
    )
    v_new, h = _bind_returned_outputs(
        "gdn.chunk_delta_h",
        _launch_stage(
            chunk_h,
            k_flat,
            u,
            w,
            v_new,
            g_cumsum_flat,
            gk,
            h,
            h0,
            ht,
            idx32,
            idx32,
        ),
        (v_new, h),
    )
    (o,) = _bind_returned_outputs(
        "gdn.chunk_o",
        _launch_stage(
            chunk_o,
            q_flat,
            k_flat,
            v_new,
            h,
            g_cumsum_flat,
            g_gamma,
            o,
            idx64,
            idx64,
        ),
        (o,),
    )

    y = mx.reshape(o, (b, GDN_FIXED_T, GDN_FIXED_HV, GDN_FIXED_V))
    final_state = (
        mx.reshape(ht, (b, GDN_FIXED_HV, GDN_FIXED_K, GDN_FIXED_V))
        if output_final_state
        else None
    )
    return y, final_state


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
    "compile_gdn_chunk_o_artifact",
    "compile_gdn_kkt_artifact",
    "compile_gdn_recompute_w_u_artifact",
    "compile_tilelang_primfunc",
    "gdn_fwd_runtime_call",
    "gdn_runtime_adapter_status",
    "kda_fwd_runtime_call",
    "kda_runtime_adapter_status",
    "primfunc_has_degraded_markers",
    "specialize_primfunc_for_grid",
    "specialize_primfunc_for_scalars",
]
