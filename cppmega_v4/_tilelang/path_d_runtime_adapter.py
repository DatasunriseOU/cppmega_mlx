"""cppmega Path D runtime adapter for Triton-frontend TileLang PrimFuncs.

This module is deliberately cppmega-side glue. The Triton frontend owns
``TTIR -> PrimFunc``. cppmega owns the recurrent public signatures, grid
specialization, output policy, kernel caching, launch eligibility, and the
multi-kernel plan needed to turn FLA chunks into ``(y, h_last)``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional


ALLOW_DEGRADED_ENV = "CPPMEGA_V4_PATH_D_ALLOW_DEGRADED_PRIMFUNC"
KDA_TOPOLOGY_SPECIALIZATION_THRESHOLD_ENV = (
    "CPPMEGA_V4_KDA_TOPOLOGY_SPECIALIZATION_THRESHOLD"
)
KDA_TOPOLOGY_SPECIALIZATION_MAX_ENTRIES_ENV = (
    "CPPMEGA_V4_KDA_TOPOLOGY_SPECIALIZATION_MAX_ENTRIES"
)
KDA_TOPOLOGY_CACHE_DIR_ENV = "CPPMEGA_V4_KDA_TOPOLOGY_CACHE_DIR"
KDA_TOPOLOGY_MANIFEST_VERSION = 1
RCP_LN2 = 1.4426950408889634
GDN_FIXED_T = 64
GDN_FIXED_H = 1
GDN_FIXED_HV = 1
GDN_FIXED_K = 64
GDN_FIXED_V = 32
GDN_FIXED_BT = 64
GDN_FIXED_BV = 32
GDN_FIXED_CHUNK_O_BV = 16
KDA_FIXED_T = 64
KDA_FIXED_H = 1
KDA_FIXED_HV = 1
KDA_FIXED_K = 64
KDA_FIXED_V = 32
KDA_FIXED_BT = 32
KDA_FIXED_BC = 16
KDA_FIXED_BV = 16
KDA_FIXED_CHUNK_O_BV = 16


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


@dataclass(frozen=True)
class KDATopologyInvariants:
    """Compact facts about a varlen topology, not exact runtime arrays."""

    total_tokens: int
    num_sequences: int
    num_chunks: int
    max_sequence_tokens: int
    max_chunk_id: int
    bounds_valid: bool
    cu_monotonic: bool
    chunk_offsets_monotonic: bool


@dataclass(frozen=True)
class KDATopologyDecision:
    """Decision returned by the hot-topology admission cache."""

    key: tuple[Any, ...]
    hits: int
    use_specialized: bool
    reason: str


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


_KDA_TOPOLOGY_HITS: dict[tuple[Any, ...], int] = {}
_KDA_TOPOLOGY_DISABLED: set[tuple[Any, ...]] = set()


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
KDA_INTRA_TOKEN_PLAN = PathDKernelPlan(
    name="kda.intra_token_parallel",
    out_idx=(4, 5),
    grid=(KDA_FIXED_T, 1),
    scalar_specializations=(1.0 / math.sqrt(64), 1, 64),
)
KDA_INTER_SOLVE_PLAN = PathDKernelPlan(
    name="kda.inter_solve",
    out_idx=(4, 6),
    grid=(1, 1),
    scalar_specializations=(1.0 / math.sqrt(64), 64),
)
KDA_RECOMPUTE_W_U_PLAN = PathDKernelPlan(
    name="kda.recompute_w_u",
    out_idx=(2, 3, 6, 7),
    grid=(1, 1),
    scalar_specializations=(64,),
)
KDA_CHUNK_O_PLAN = PathDKernelPlan(
    name="kda.chunk_o_gk",
    out_idx=(4,),
    grid=(math.ceil(KDA_FIXED_V / KDA_FIXED_CHUNK_O_BV), 1, 1),
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
    public_signature=(
        "kda(q, k, v, g, beta, *, scale, initial_state, "
        "output_final_state, cu_seqlens)"
    ),
    output_layout="returns y[B,T,HV,V] and optional h_last[N,HV,K,V]",
    stages=(
        KDA_INTRA_TOKEN_PLAN,
        KDA_INTER_SOLVE_PLAN,
        KDA_RECOMPUTE_W_U_PLAN,
        GDN_CHUNK_H_PLAN,
        KDA_CHUNK_O_PLAN,
    ),
    note=(
        "KDA forward builds Aqk/Akk, recomputes WY state, uses common "
        "chunk_delta_h with vector gate gk, then GLA chunk_o."
    ),
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _reset_kda_topology_cache_for_tests() -> None:
    _KDA_TOPOLOGY_HITS.clear()
    _KDA_TOPOLOGY_DISABLED.clear()


def _kda_compact_topology_invariants(
    *,
    cu_values: tuple[int, ...] | list[int],
    chunk_offsets: tuple[int, ...] | list[int],
    total_tokens: int,
    chunk_size: int,
) -> KDATopologyInvariants:
    cu = tuple(int(x) for x in cu_values)
    offsets = tuple(int(x) for x in chunk_offsets)
    total = int(total_tokens)
    num_sequences = max(len(cu) - 1, 0)
    lengths = [right - left for left, right in zip(cu, cu[1:])]
    num_chunks = int(offsets[-1]) if offsets else 0
    bounds_valid = (
        len(cu) >= 2
        and cu[0] == 0
        and cu[-1] == total
        and all(0 <= value <= total for value in cu)
        and all(length >= 0 for length in lengths)
    )
    cu_monotonic = all(left <= right for left, right in zip(cu, cu[1:]))
    chunk_offsets_monotonic = (
        len(offsets) == len(cu)
        and bool(offsets)
        and offsets[0] == 0
        and all(left <= right for left, right in zip(offsets, offsets[1:]))
    )
    return KDATopologyInvariants(
        total_tokens=total,
        num_sequences=num_sequences,
        num_chunks=num_chunks,
        max_sequence_tokens=max(lengths, default=0),
        max_chunk_id=num_chunks - 1 if num_chunks else -1,
        bounds_valid=bounds_valid,
        cu_monotonic=cu_monotonic,
        chunk_offsets_monotonic=chunk_offsets_monotonic,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _kda_varlen_topology_descriptor(
    *,
    cu_values: tuple[int, ...] | list[int],
    chunk_offsets: tuple[int, ...] | list[int],
    total_tokens: int,
    h_heads: int,
    hv_heads: int,
    k_dim: int,
    v_dim: int,
    chunk_size: int,
    scale: Any,
    use_initial_state: bool,
    output_final_state: bool,
) -> dict[str, Any]:
    cu = tuple(int(x) for x in cu_values)
    offsets = tuple(int(x) for x in chunk_offsets)
    lengths = tuple(int(right - left) for left, right in zip(cu, cu[1:]))
    return {
        "version": KDA_TOPOLOGY_MANIFEST_VERSION,
        "total_tokens": int(total_tokens),
        "num_sequences": len(lengths),
        "h_heads": int(h_heads),
        "hv_heads": int(hv_heads),
        "k_dim": int(k_dim),
        "v_dim": int(v_dim),
        "chunk_size": int(chunk_size),
        "lengths": lengths,
        "chunk_offsets": offsets,
        "use_initial_state": bool(use_initial_state),
        "output_final_state": bool(output_final_state),
        "scale": None if scale is None else float(scale),
    }


def _kda_varlen_topology_fingerprint(descriptor: dict[str, Any]) -> str:
    payload = json.dumps(
        _json_ready(descriptor),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _kda_topology_cache_dir() -> str:
    override = os.environ.get(KDA_TOPOLOGY_CACHE_DIR_ENV)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    tilelang_cache = os.environ.get("TILELANG_CACHE_DIR")
    if tilelang_cache:
        root = os.path.abspath(os.path.expanduser(tilelang_cache))
    else:
        root = os.path.expanduser("~/.tilelang/cache")
    return os.path.join(root, "cppmega_kda_topologies")


def _kda_topology_manifest_path(fingerprint: str) -> str:
    return os.path.join(_kda_topology_cache_dir(), f"{fingerprint}.json")


def _write_kda_topology_manifest(
    fingerprint: str,
    *,
    descriptor: dict[str, Any],
    status: str,
    stages: tuple[str, ...],
) -> None:
    cache_dir = _kda_topology_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path = _kda_topology_manifest_path(fingerprint)
    payload = {
        "version": KDA_TOPOLOGY_MANIFEST_VERSION,
        "fingerprint": str(fingerprint),
        "descriptor": _json_ready(descriptor),
        "status": str(status),
        "stages": list(stages),
        "tilelang_cache_dir": os.path.abspath(
            os.path.expanduser(
                os.environ.get("TILELANG_CACHE_DIR", "~/.tilelang/cache")
            )
        ),
    }
    temp_path = f"{path}.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, sort_keys=True, separators=(",", ":"))
    os.replace(temp_path, path)


def _read_kda_topology_manifest(fingerprint: str) -> Optional[dict[str, Any]]:
    path = _kda_topology_manifest_path(fingerprint)
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError:
        return None
    if payload.get("version") != KDA_TOPOLOGY_MANIFEST_VERSION:
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    return payload


def _kda_varlen_topology_key(
    *,
    cu_values: tuple[int, ...] | list[int],
    chunk_indices: tuple[int, ...] | list[int],
    chunk_offsets: tuple[int, ...] | list[int],
    total_tokens: int,
    h_heads: int,
    hv_heads: int,
    k_dim: int,
    v_dim: int,
    scale: Any,
    use_initial_state: bool,
    output_final_state: bool,
    chunk_size: int = KDA_FIXED_BT,
) -> tuple[Any, ...]:
    cu = tuple(int(x) for x in cu_values)
    offsets = tuple(int(x) for x in chunk_offsets)
    descriptor = _kda_varlen_topology_descriptor(
        cu_values=cu,
        chunk_offsets=offsets,
        total_tokens=total_tokens,
        h_heads=h_heads,
        hv_heads=hv_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        chunk_size=chunk_size,
        scale=scale,
        use_initial_state=use_initial_state,
        output_final_state=output_final_state,
    )
    fingerprint = _kda_varlen_topology_fingerprint(descriptor)
    del chunk_indices
    return (
        "kda_varlen_topology_v2",
        fingerprint,
    )


def _record_kda_topology_hit(key: tuple[Any, ...]) -> KDATopologyDecision:
    threshold = max(
        _env_int(KDA_TOPOLOGY_SPECIALIZATION_THRESHOLD_ENV, 3),
        0,
    )
    if threshold <= 0:
        return KDATopologyDecision(
            key=key,
            hits=0,
            use_specialized=False,
            reason="topology specialization disabled",
        )
    if key in _KDA_TOPOLOGY_DISABLED:
        hits = _KDA_TOPOLOGY_HITS.get(key, 0)
        return KDATopologyDecision(
            key=key,
            hits=hits,
            use_specialized=False,
            reason="topology specialization disabled after compile fallback",
        )

    max_entries = max(
        _env_int(KDA_TOPOLOGY_SPECIALIZATION_MAX_ENTRIES_ENV, 64),
        1,
    )
    if key not in _KDA_TOPOLOGY_HITS and len(_KDA_TOPOLOGY_HITS) >= max_entries:
        oldest = next(iter(_KDA_TOPOLOGY_HITS))
        _KDA_TOPOLOGY_HITS.pop(oldest, None)
        _KDA_TOPOLOGY_DISABLED.discard(oldest)
    hits = _KDA_TOPOLOGY_HITS.get(key, 0) + 1
    _KDA_TOPOLOGY_HITS[key] = hits
    use_specialized = hits >= threshold
    return KDATopologyDecision(
        key=key,
        hits=hits,
        use_specialized=use_specialized,
        reason=(
            "hot topology promoted"
            if use_specialized
            else "dynamic topology below specialization threshold"
        ),
    )


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


def _freeze_topology_constants(
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not topology_constants:
        return ()
    return tuple(
        sorted(
            (str(name), tuple(int(x) for x in values))
            for name, values in topology_constants.items()
        )
    )


def _thaw_topology_constants(
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
) -> dict[str, tuple[int, ...]]:
    return {name: tuple(values) for name, values in topology_constants_key}


def _metadata_constants_for_buffer(
    buffer_name: str,
    topology_constants: dict[str, tuple[int, ...]],
) -> Optional[tuple[int, ...]]:
    if buffer_name in topology_constants:
        return topology_constants[buffer_name]
    for name, values in topology_constants.items():
        if buffer_name.startswith(f"{name}_"):
            return values
    return None


def specialize_primfunc_for_topology_metadata(
    prim_func: Any,
    topology_constants: dict[str, tuple[int, ...] | list[int]],
) -> Any:
    """Constant-fold hot varlen topology metadata loads inside a PrimFunc.

    The public launch ABI remains unchanged: cppmega still passes cu_seqlens,
    chunk_indices, and chunk_offsets. Hot topologies simply compile a variant
    whose internal loads from those small metadata arrays are replaced by
    constants/selects, which keeps the dynamic path generic and avoids asking
    the analyzer/Z3 to reason about exact runtime arrays.
    """

    constants = {
        str(name): tuple(int(x) for x in values)
        for name, values in topology_constants.items()
        if values is not None
    }
    if not constants:
        return prim_func

    from tvm import tir
    from tvm.tir import stmt_functor

    def const_expr(dtype: str, value: int) -> Any:
        return tir.IntImm(dtype, int(value))

    def index_dtype(index: Any) -> str:
        dtype = str(getattr(index, "dtype", "int64"))
        return dtype if dtype.startswith(("int", "uint")) else "int64"

    def const_lookup(index: Any, values: tuple[int, ...], dtype: str) -> Any:
        if not values:
            return None
        immediate = getattr(index, "value", None)
        if immediate is not None:
            pos = int(immediate)
            if 0 <= pos < len(values):
                return const_expr(dtype, values[pos])
        idx_dtype = index_dtype(index)
        expr = const_expr(dtype, values[-1])
        for pos in range(len(values) - 2, -1, -1):
            expr = tir.Select(
                index == tir.IntImm(idx_dtype, pos),
                const_expr(dtype, values[pos]),
                expr,
            )
        return expr

    def post(node: Any) -> Any:
        buffer = getattr(node, "buffer", None)
        indices = getattr(node, "indices", None)
        if buffer is None or indices is None or len(indices) != 1:
            return node
        values = _metadata_constants_for_buffer(str(getattr(buffer, "name", "")), constants)
        if values is None:
            return node
        replacement = const_lookup(indices[0], values, str(getattr(node, "dtype", "int64")))
        return node if replacement is None else replacement

    body = stmt_functor.ir_transform(
        prim_func.body,
        None,
        post,
        ["tirx.BufferLoad"],
    )
    return prim_func.with_body(body)


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
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
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
        specialized = specialize_primfunc_for_topology_metadata(
            specialized,
            topology_constants or {},
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


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _kda_default_scale(k_dim: int, scale: Any = None) -> float:
    return 1.0 / math.sqrt(int(k_dim)) if scale is None else float(scale)


@lru_cache(maxsize=16)
def _compile_gdn_chunk_h_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
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
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
    )


def _compile_lowered_kernel_result(
    lowered: Any,
    *,
    base_plan: PathDKernelPlan,
    grid: tuple[int, ...],
    allow_degraded_primfunc: bool,
    scalar_specializations: Optional[tuple[Any, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...]]] = None,
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
        scalar_specializations=(
            base_plan.scalar_specializations
            if scalar_specializations is None
            else scalar_specializations
        ),
        target=base_plan.target,
        execution_backend=base_plan.execution_backend,
        allow_degraded_primfunc=allow_degraded_primfunc,
    )
    return compile_tilelang_primfunc(
        lowered.prim_func,
        plan,
        topology_constants=topology_constants,
    )


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


@lru_cache(maxsize=16)
def _compile_kda_intra_token_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_kda_intra_token_parallel,
    )

    lowered = lower_fla_kda_intra_token_parallel(dict(constexprs_key), grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=KDA_INTRA_TOKEN_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
    )


@lru_cache(maxsize=16)
def _compile_kda_inter_solve_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_kda_inter_solve,
    )

    lowered = lower_fla_kda_inter_solve(dict(constexprs_key), grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=KDA_INTER_SOLVE_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
    )


@lru_cache(maxsize=16)
def _compile_kda_recompute_w_u_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_kda_recompute_w_u,
    )

    lowered = lower_fla_kda_recompute_w_u(dict(constexprs_key), grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=KDA_RECOMPUTE_W_U_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
        topology_constants=_thaw_topology_constants(topology_constants_key),
    )


@lru_cache(maxsize=16)
def _compile_kda_chunk_o_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
    allow_degraded_primfunc: bool,
) -> PathDCompileResult:
    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        lower_fla_kda_chunk_o,
    )

    lowered = lower_fla_kda_chunk_o(dict(constexprs_key), grid=grid)
    return _compile_lowered_kernel_result(
        lowered,
        base_plan=KDA_CHUNK_O_PLAN,
        grid=grid,
        allow_degraded_primfunc=allow_degraded_primfunc,
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
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
    scalar_specializations: Optional[tuple[Any, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
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
        GDN_CHUNK_H_PLAN.scalar_specializations
        if scalar_specializations is None
        else scalar_specializations,
        _freeze_topology_constants(topology_constants),
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


def compile_kda_intra_token_artifact(
    *,
    batch: int = 1,
    num_sequences: Optional[int] = None,
    total_tokens: int = KDA_FIXED_T,
    h_heads: int = KDA_FIXED_H,
    hv_heads: int = KDA_FIXED_HV,
    k_dim: int = KDA_FIXED_K,
    bt: int = KDA_FIXED_BT,
    bc: int = KDA_FIXED_BC,
    is_varlen: bool = False,
    scale: Any = None,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[tuple[int, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache KDA's token-parallel diagonal block kernel."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_KDA_INTRA_TOKEN_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_KDA_INTRA_TOKEN_CONSTEXPRS)
    cfg.update(
        {
            "_RUNTIME_T": int(total_tokens),
            "H": int(h_heads),
            "HV": int(hv_heads),
            "K": int(k_dim),
            "BT": int(bt),
            "BC": int(bc),
            "IS_VARLEN": bool(is_varlen),
        }
    )
    if constexprs:
        cfg.update(constexprs)
    if grid is None:
        grid = (
            int(batch) * int(total_tokens),
            math.ceil(int(cfg.get("HV", KDA_FIXED_HV)) / int(cfg.get("BH", 1))),
        )
    scale_value = _kda_default_scale(int(cfg.get("K", k_dim)), scale)
    scalar_specializations = (
        scale_value,
        int(num_sequences if num_sequences is not None else batch),
        int(total_tokens),
    )
    return _compile_kda_intra_token_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        scalar_specializations,
        _freeze_topology_constants(topology_constants),
        bool(allow_degraded_primfunc),
    )


def compile_kda_inter_solve_artifact(
    *,
    batch: int = 1,
    total_tokens: int = KDA_FIXED_T,
    h_heads: int = KDA_FIXED_H,
    hv_heads: int = KDA_FIXED_HV,
    k_dim: int = KDA_FIXED_K,
    bt: int = KDA_FIXED_BT,
    bc: int = KDA_FIXED_BC,
    bk: int = 32,
    is_varlen: bool = False,
    scale: Any = None,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[tuple[int, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache KDA's fused inter-subchunk solve kernel."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_KDA_INTER_SOLVE_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_KDA_INTER_SOLVE_CONSTEXPRS)
    cfg.update(
        {
            "_RUNTIME_T": int(total_tokens),
            "H": int(h_heads),
            "HV": int(hv_heads),
            "K": int(k_dim),
            "BT": int(bt),
            "BC": int(bc),
            "NC": _ceil_div(int(bt), int(bc)),
            "BK": int(bk),
            "IS_VARLEN": bool(is_varlen),
        }
    )
    if constexprs:
        cfg.update(constexprs)
    if grid is None:
        grid = (
            _ceil_div(int(total_tokens), int(cfg.get("BT", bt))),
            int(batch) * int(cfg.get("HV", KDA_FIXED_HV)),
        )
    scale_value = _kda_default_scale(int(cfg.get("K", k_dim)), scale)
    scalar_specializations = (scale_value, int(total_tokens))
    return _compile_kda_inter_solve_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        scalar_specializations,
        _freeze_topology_constants(topology_constants),
        bool(allow_degraded_primfunc),
    )


def compile_kda_recompute_w_u_artifact(
    *,
    batch: int = 1,
    total_tokens: int = KDA_FIXED_T,
    h_heads: int = KDA_FIXED_H,
    hv_heads: int = KDA_FIXED_HV,
    k_dim: int = KDA_FIXED_K,
    v_dim: int = KDA_FIXED_V,
    bt: int = KDA_FIXED_BT,
    is_varlen: bool = False,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[tuple[int, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache KDA's WY recompute kernel."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_KDA_RECOMPUTE_W_U_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_KDA_RECOMPUTE_W_U_CONSTEXPRS)
    cfg.update(
        {
            "_RUNTIME_T": int(total_tokens),
            "H": int(h_heads),
            "HV": int(hv_heads),
            "K": int(k_dim),
            "V": int(v_dim),
            "BT": int(bt),
            "IS_VARLEN": bool(is_varlen),
        }
    )
    if constexprs:
        cfg.update(constexprs)
    if grid is None:
        grid = (
            _ceil_div(int(total_tokens), int(cfg.get("BT", bt))),
            int(batch) * int(cfg.get("HV", KDA_FIXED_HV)),
        )
    return _compile_kda_recompute_w_u_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        _freeze_topology_constants(topology_constants),
        bool(allow_degraded_primfunc),
    )


def compile_kda_chunk_o_artifact(
    *,
    batch: int = 1,
    total_tokens: int = KDA_FIXED_T,
    h_heads: int = KDA_FIXED_H,
    hv_heads: int = KDA_FIXED_HV,
    k_dim: int = KDA_FIXED_K,
    v_dim: int = KDA_FIXED_V,
    bt: int = KDA_FIXED_BT,
    bv: int = KDA_FIXED_CHUNK_O_BV,
    is_varlen: bool = False,
    scale: Any = None,
    constexprs: Optional[dict[str, Any]] = None,
    grid: Optional[tuple[int, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache the GLA output kernel variant used by KDA."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_KDA_CHUNK_O_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_KDA_CHUNK_O_CONSTEXPRS)
    cfg.update(
        {
            "_RUNTIME_T": int(total_tokens),
            "H": int(h_heads),
            "HV": int(hv_heads),
            "K": int(k_dim),
            "V": int(v_dim),
            "BT": int(bt),
            "BV": int(bv),
            "IS_VARLEN": bool(is_varlen),
        }
    )
    if constexprs:
        cfg.update(constexprs)
    if grid is None:
        grid = (
            _ceil_div(int(cfg.get("V", v_dim)), int(cfg.get("BV", bv))),
            _ceil_div(int(total_tokens), int(cfg.get("BT", bt))),
            int(batch) * int(cfg.get("HV", KDA_FIXED_HV)),
        )
    scale_value = _kda_default_scale(int(cfg.get("K", k_dim)), scale)
    scalar_specializations = (scale_value, int(total_tokens))
    return _compile_kda_chunk_o_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        scalar_specializations,
        _freeze_topology_constants(topology_constants),
        bool(allow_degraded_primfunc),
    )


def _is_dtype(value: Any, dtype: Any) -> bool:
    return getattr(value, "dtype", None) == dtype


def _require_shape_dtype(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    adapter_name: str = "GDN",
) -> None:
    if tuple(getattr(value, "shape", ())) != shape:
        raise PathDRuntimeUnavailable(
            f"{adapter_name} Path D runtime adapter only supports {name}.shape={shape}; "
            f"got {getattr(value, 'shape', None)}"
        )
    if not _is_dtype(value, dtype):
        raise PathDRuntimeUnavailable(
            f"{adapter_name} Path D runtime adapter only supports {name}.dtype={dtype}; "
            f"got {getattr(value, 'dtype', None)}"
        )


def _require_dtype(
    name: str,
    value: Any,
    dtype: Any,
    *,
    adapter_name: str = "KDA",
) -> None:
    if not _is_dtype(value, dtype):
        raise PathDRuntimeUnavailable(
            f"{adapter_name} Path D runtime adapter only supports {name}.dtype={dtype}; "
            f"got {getattr(value, 'dtype', None)}"
        )


def _as_int_list(value: Any, *, name: str) -> list[int]:
    import numpy as np

    try:
        return [int(x) for x in np.array(value).reshape(-1).tolist()]
    except Exception as exc:  # noqa: BLE001
        raise PathDRuntimeUnavailable(
            f"KDA Path D runtime adapter could not read small metadata {name}: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc


def _prepare_kda_varlen_metadata(
    *,
    cu_seqlens: Any,
    chunk_indices: Any,
    chunk_offsets: Any,
    total_tokens: int,
    chunk_size: int,
    mx: Any,
) -> tuple[Any, Any, Any, list[int], int, int]:
    if cu_seqlens is None:
        if chunk_indices is not None or chunk_offsets is not None:
            raise PathDRuntimeUnavailable(
                "KDA Path D runtime adapter requires cu_seqlens when "
                "chunk_indices/chunk_offsets are provided"
            )
        dummy = mx.zeros((1,), dtype=mx.int64)
        return dummy, dummy, dummy, [], 0, 0

    values = _as_int_list(cu_seqlens, name="cu_seqlens")
    if len(values) < 2:
        raise PathDRuntimeUnavailable(
            "KDA Path D varlen requires cu_seqlens with at least two entries"
        )
    if values[0] != 0:
        raise PathDRuntimeUnavailable(
            f"KDA Path D varlen requires cu_seqlens[0] == 0; got {values[0]}"
        )
    if values[-1] != int(total_tokens):
        raise PathDRuntimeUnavailable(
            "KDA Path D varlen requires packed q/k/v/g/beta length to equal "
            f"cu_seqlens[-1]; got T={total_tokens}, cu_seqlens[-1]={values[-1]}"
        )
    lengths = []
    for left, right in zip(values, values[1:]):
        if right < left:
            raise PathDRuntimeUnavailable(
                "KDA Path D varlen requires nondecreasing cu_seqlens"
            )
        lengths.append(right - left)

    pairs: list[int] = []
    offsets = [0]
    total_chunks = 0
    for seq_idx, seq_len in enumerate(lengths):
        n_chunks = _ceil_div(seq_len, chunk_size) if seq_len else 0
        for chunk_idx in range(n_chunks):
            pairs.extend((seq_idx, chunk_idx))
        total_chunks += n_chunks
        offsets.append(total_chunks)
    if total_chunks == 0:
        raise PathDRuntimeUnavailable(
            "KDA Path D varlen requires at least one non-empty chunk"
        )

    if chunk_indices is None:
        chunk_indices = mx.array(pairs, dtype=mx.int64)
    else:
        actual_pairs = _as_int_list(chunk_indices, name="chunk_indices")
        if actual_pairs != pairs:
            raise PathDRuntimeUnavailable(
                "KDA Path D varlen chunk_indices do not match cu_seqlens "
                f"for chunk_size={chunk_size}"
            )
        if getattr(chunk_indices, "dtype", None) != mx.int64:
            chunk_indices = mx.array(actual_pairs, dtype=mx.int64)

    if chunk_offsets is None:
        chunk_offsets = mx.array(offsets, dtype=mx.int64)
    else:
        actual_offsets = _as_int_list(chunk_offsets, name="chunk_offsets")
        if actual_offsets != offsets:
            raise PathDRuntimeUnavailable(
                "KDA Path D varlen chunk_offsets do not match cu_seqlens "
                f"for chunk_size={chunk_size}"
            )
        if getattr(chunk_offsets, "dtype", None) != mx.int64:
            chunk_offsets = mx.array(actual_offsets, dtype=mx.int64)

    if getattr(cu_seqlens, "dtype", None) != mx.int64:
        cu_seqlens = mx.array(values, dtype=mx.int64)

    return (
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        values,
        len(values) - 1,
        total_chunks,
    )


def _kda_chunk_local_cumsum(
    g: Any,
    *,
    chunk_size: int,
    cu_seqlens_values: Optional[list[int]],
) -> Any:
    import mlx.core as mx

    parts = []
    if cu_seqlens_values is None:
        total_tokens = int(g.shape[1])
        spans = [(0, total_tokens)]
    else:
        spans = list(zip(cu_seqlens_values, cu_seqlens_values[1:]))
    for start, end in spans:
        for chunk_start in range(int(start), int(end), int(chunk_size)):
            chunk_end = min(chunk_start + int(chunk_size), int(end))
            if chunk_end > chunk_start:
                parts.append(mx.cumsum(g[:, chunk_start:chunk_end, :, :], axis=1))
    if not parts:
        return mx.zeros_like(g)
    return mx.concatenate(parts, axis=1) * RCP_LN2


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


def _compile_kda_runtime_stages(
    *,
    batch: int,
    total_tokens: int,
    num_sequences: int,
    num_chunks: int,
    h_heads: int,
    hv_heads: int,
    k_dim: int,
    v_dim: int,
    is_varlen: bool,
    scale: Any,
    use_initial_state: bool,
    output_final_state: bool,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
) -> tuple[
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
]:
    chunks_per_batch = _ceil_div(total_tokens, KDA_FIXED_BT)
    stage_chunks = int(num_chunks) if is_varlen else chunks_per_batch
    total_h_chunks = int(num_chunks) if is_varlen else int(batch) * chunks_per_batch
    stage_grid = (stage_chunks, batch * hv_heads)
    chunk_v_grid = (
        _ceil_div(v_dim, KDA_FIXED_BV),
        (num_sequences if is_varlen else batch) * hv_heads,
    )
    chunk_o_grid = (
        _ceil_div(v_dim, KDA_FIXED_CHUNK_O_BV),
        stage_chunks,
        batch * hv_heads,
    )
    runtime_private = {
        "_RUNTIME_BATCH": int(batch),
        "_RUNTIME_T": int(total_tokens),
        "_RUNTIME_N": int(num_sequences),
        "_RUNTIME_NT": int(total_h_chunks),
        "_RUNTIME_CU_LEN": int(num_sequences) + 1,
    }
    chunk_h_constexprs = {
        **runtime_private,
        "H": int(h_heads),
        "HV": int(hv_heads),
        "K": int(k_dim),
        "V": int(v_dim),
        "BT": KDA_FIXED_BT,
        "BV": KDA_FIXED_BV,
        "USE_G": False,
        "USE_GK": True,
        "USE_INITIAL_STATE": bool(use_initial_state),
        "STORE_FINAL_STATE": bool(output_final_state),
        "SAVE_NEW_VALUE": True,
        "IS_VARLEN": bool(is_varlen),
    }
    return (
        compile_kda_intra_token_artifact(
            batch=batch,
            num_sequences=num_sequences,
            total_tokens=total_tokens,
            h_heads=h_heads,
            hv_heads=hv_heads,
            k_dim=k_dim,
            bt=KDA_FIXED_BT,
            bc=KDA_FIXED_BC,
            is_varlen=is_varlen,
            scale=scale,
            constexprs=runtime_private,
            topology_constants=topology_constants,
        ),
        compile_kda_inter_solve_artifact(
            batch=batch,
            total_tokens=total_tokens,
            h_heads=h_heads,
            hv_heads=hv_heads,
            k_dim=k_dim,
            bt=KDA_FIXED_BT,
            bc=KDA_FIXED_BC,
            is_varlen=is_varlen,
            scale=scale,
            constexprs=runtime_private,
            grid=stage_grid,
            topology_constants=topology_constants,
        ),
        compile_kda_recompute_w_u_artifact(
            batch=batch,
            total_tokens=total_tokens,
            h_heads=h_heads,
            hv_heads=hv_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            bt=KDA_FIXED_BT,
            is_varlen=is_varlen,
            constexprs=runtime_private,
            grid=stage_grid,
            topology_constants=topology_constants,
        ),
        compile_gdn_chunk_h_artifact(
            constexprs=chunk_h_constexprs,
            grid=chunk_v_grid,
            scalar_specializations=(int(total_tokens),),
            topology_constants=topology_constants,
        ),
        compile_kda_chunk_o_artifact(
            batch=batch,
            total_tokens=total_tokens,
            h_heads=h_heads,
            hv_heads=hv_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            bt=KDA_FIXED_BT,
            bv=KDA_FIXED_CHUNK_O_BV,
            is_varlen=is_varlen,
            scale=scale,
            constexprs=runtime_private,
            grid=chunk_o_grid,
            topology_constants=topology_constants,
        ),
    )


def _validate_default_scale(
    scale: Any,
    *,
    adapter_name: str = "GDN",
    k_dim: int = GDN_FIXED_K,
) -> None:
    if scale is None:
        return
    default_scale = 1.0 / math.sqrt(k_dim)
    if not math.isclose(float(scale), default_scale, rel_tol=0.0, abs_tol=1e-7):
        raise PathDRuntimeUnavailable(
            f"{adapter_name} Path D runtime adapter currently supports only the default "
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

    stages = _compile_kda_runtime_stages(
        batch=1,
        total_tokens=KDA_FIXED_T,
        num_sequences=1,
        num_chunks=_ceil_div(KDA_FIXED_T, KDA_FIXED_BT),
        h_heads=KDA_FIXED_H,
        hv_heads=KDA_FIXED_HV,
        k_dim=KDA_FIXED_K,
        v_dim=KDA_FIXED_V,
        is_varlen=False,
        scale=None,
        use_initial_state=False,
        output_final_state=True,
    )
    for stage in stages:
        if not stage.available:
            return False, (
                f"KDA Path D runtime adapter installed for {KDA_RECURRENT_PLAN.public_signature}; "
                f"{coverage_reason}; {stage.reason}; planned stages="
                f"{', '.join(plan.name for plan in KDA_RECURRENT_PLAN.stages)}"
            )
    return True, (
        "KDA Path D runtime adapter available for shape-specialized fp16 "
        "prefill and packed varlen, vector gate gk, custom scale; "
        "compile/cache/launch stages="
        f"{', '.join(stage.name for stage in KDA_RECURRENT_PLAN.stages)}"
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
            idx64,
            idx64,
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


def kda_fwd_runtime_call(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    *,
    scale: Any = None,
    initial_state: Any = None,
    output_final_state: bool = False,
    cu_seqlens: Any = None,
    chunk_indices: Any = None,
    chunk_offsets: Any = None,
    coverage_reason: str = "",
    **kwargs: Any,
) -> Any:
    """Launch the KDA Path D FLA multi-kernel pipeline."""

    if kwargs:
        raise PathDRuntimeUnavailable(
            "KDA Path D runtime adapter does not support extra keyword args: "
            f"{sorted(kwargs)}"
        )

    import mlx.core as mx

    q_shape = tuple(int(x) for x in getattr(q, "shape", ()))
    if len(q_shape) != 4:
        raise PathDRuntimeUnavailable(
            f"KDA Path D runtime adapter expects q as [B,T,H,K]; got {q_shape}"
        )
    b, total_tokens, h_heads, k_dim = q_shape
    if min(b, total_tokens, h_heads, k_dim) <= 0:
        raise PathDRuntimeUnavailable(
            f"KDA Path D runtime adapter got invalid q shape {q_shape}"
        )
    v_shape = tuple(int(x) for x in getattr(v, "shape", ()))
    if len(v_shape) != 4:
        raise PathDRuntimeUnavailable(
            f"KDA Path D runtime adapter expects v as [B,T,HV,V]; got {v_shape}"
        )
    if v_shape[0] != b or v_shape[1] != total_tokens:
        raise PathDRuntimeUnavailable(
            "KDA Path D runtime adapter requires q/k/v to share B and T; "
            f"got q={q_shape}, v={v_shape}"
        )
    hv_heads = int(v_shape[2])
    v_dim = int(v_shape[3])
    if hv_heads <= 0 or v_dim <= 0 or hv_heads % h_heads != 0:
        raise PathDRuntimeUnavailable(
            "KDA Path D runtime adapter requires HV > 0, V > 0, and HV % H == 0; "
            f"got H={h_heads}, HV={hv_heads}, V={v_dim}"
        )
    if k_dim > 256:
        raise PathDRuntimeUnavailable(
            "KDA Path D common chunk_delta_h supports K <= 256; "
            f"got K={k_dim}"
        )

    expected_qk = (b, total_tokens, h_heads, k_dim)
    expected_v = (b, total_tokens, hv_heads, v_dim)
    expected_g = (b, total_tokens, hv_heads, k_dim)
    expected_beta = (b, total_tokens, hv_heads)

    _require_shape_dtype("q", q, expected_qk, mx.float16, adapter_name="KDA")
    _require_shape_dtype("k", k, expected_qk, mx.float16, adapter_name="KDA")
    _require_shape_dtype("v", v, expected_v, mx.float16, adapter_name="KDA")
    _require_shape_dtype("g", g, expected_g, mx.float32, adapter_name="KDA")
    _require_shape_dtype("beta", beta, expected_beta, mx.float32, adapter_name="KDA")

    is_varlen = cu_seqlens is not None
    if is_varlen and b != 1:
        raise PathDRuntimeUnavailable(
            "KDA Path D varlen expects packed inputs with B=1; "
            f"got B={b}"
        )
    (
        cu_seqlens_arg,
        chunk_indices_arg,
        chunk_offsets_arg,
        cu_values,
        varlen_sequences,
        varlen_chunks,
    ) = _prepare_kda_varlen_metadata(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        total_tokens=total_tokens,
        chunk_size=KDA_FIXED_BT,
        mx=mx,
    )
    num_sequences = varlen_sequences if is_varlen else b
    num_chunks = (
        varlen_chunks
        if is_varlen
        else b * _ceil_div(total_tokens, KDA_FIXED_BT)
    )

    if initial_state is not None:
        _require_shape_dtype(
            "initial_state",
            initial_state,
            (num_sequences, hv_heads, k_dim, v_dim),
            mx.float32,
            adapter_name="KDA",
        )

    topology_key: Optional[tuple[Any, ...]] = None
    topology_descriptor: Optional[dict[str, Any]] = None
    topology_fingerprint: Optional[str] = None
    topology_constants: Optional[dict[str, tuple[int, ...]]] = None
    if is_varlen:
        chunk_indices_values = tuple(
            _as_int_list(chunk_indices_arg, name="chunk_indices")
        )
        chunk_offsets_values = tuple(
            _as_int_list(chunk_offsets_arg, name="chunk_offsets")
        )
        topology_descriptor = _kda_varlen_topology_descriptor(
            cu_values=tuple(cu_values),
            chunk_offsets=chunk_offsets_values,
            total_tokens=total_tokens,
            h_heads=h_heads,
            hv_heads=hv_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            chunk_size=KDA_FIXED_BT,
            scale=scale,
            use_initial_state=initial_state is not None,
            output_final_state=bool(output_final_state),
        )
        topology_fingerprint = _kda_varlen_topology_fingerprint(topology_descriptor)
        topology_key = ("kda_varlen_topology_v2", topology_fingerprint)
        topology_decision = _record_kda_topology_hit(topology_key)
        if topology_decision.use_specialized:
            topology_constants = {
                "cu_seqlens": tuple(cu_values),
                "chunk_indices": chunk_indices_values,
                "chunk_offsets": chunk_offsets_values,
            }

    token, inter, recompute, chunk_h, chunk_o = _compile_kda_runtime_stages(
        batch=b,
        total_tokens=total_tokens,
        num_sequences=num_sequences,
        num_chunks=num_chunks,
        h_heads=h_heads,
        hv_heads=hv_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        is_varlen=is_varlen,
        scale=scale,
        use_initial_state=initial_state is not None,
        output_final_state=bool(output_final_state),
        topology_constants=topology_constants,
    )
    if topology_constants is not None and any(
        not stage.available for stage in (token, inter, recompute, chunk_h, chunk_o)
    ):
        if topology_key is not None:
            _KDA_TOPOLOGY_DISABLED.add(topology_key)
        if topology_fingerprint is not None and topology_descriptor is not None:
            try:
                _write_kda_topology_manifest(
                    topology_fingerprint,
                    descriptor=topology_descriptor,
                    status="disabled",
                    stages=tuple(
                        stage.plan.name if stage.plan is not None else "unknown"
                        for stage in (token, inter, recompute, chunk_h, chunk_o)
                    ),
                )
            except OSError:
                pass
        token, inter, recompute, chunk_h, chunk_o = _compile_kda_runtime_stages(
            batch=b,
            total_tokens=total_tokens,
            num_sequences=num_sequences,
            num_chunks=num_chunks,
            h_heads=h_heads,
            hv_heads=hv_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            is_varlen=is_varlen,
            scale=scale,
            use_initial_state=initial_state is not None,
            output_final_state=bool(output_final_state),
        )
    elif topology_constants is not None and topology_fingerprint is not None:
        if topology_descriptor is not None:
            try:
                _write_kda_topology_manifest(
                    topology_fingerprint,
                    descriptor=topology_descriptor,
                    status="compiled",
                    stages=tuple(
                        stage.plan.name if stage.plan is not None else "unknown"
                        for stage in (token, inter, recompute, chunk_h, chunk_o)
                    ),
                )
            except OSError:
                pass
    for stage in (token, inter, recompute, chunk_h, chunk_o):
        if not stage.available:
            detail = f"; {coverage_reason}" if coverage_reason else ""
            raise PathDRuntimeUnavailable(f"{stage.reason}{detail}")

    h_chunks = num_chunks
    q_flat = _flatten(q)
    k_flat = _flatten(k)
    v_flat = _flatten(v)
    beta_flat = _flatten(beta)
    g_cumsum_flat = _flatten(
        _kda_chunk_local_cumsum(
            g,
            chunk_size=KDA_FIXED_BT,
            cu_seqlens_values=cu_values if is_varlen else None,
        )
    )

    aqk = mx.empty((b * total_tokens * hv_heads * KDA_FIXED_BT,), dtype=mx.float16)
    akkd = mx.empty((b * total_tokens * hv_heads * KDA_FIXED_BC,), dtype=mx.float32)
    akk = mx.zeros((b * total_tokens * hv_heads * KDA_FIXED_BT,), dtype=mx.float16)
    qg = mx.empty((b * total_tokens * hv_heads * k_dim,), dtype=mx.float16)
    kg = mx.empty((b * total_tokens * hv_heads * k_dim,), dtype=mx.float16)
    w = mx.empty((b * total_tokens * hv_heads * k_dim,), dtype=mx.float16)
    u = mx.empty((b * total_tokens * hv_heads * v_dim,), dtype=mx.float16)
    v_new = mx.empty((b * total_tokens * hv_heads * v_dim,), dtype=mx.float16)
    h = mx.empty((h_chunks * hv_heads * k_dim * v_dim,), dtype=mx.float16)
    o = mx.empty((b * total_tokens * hv_heads * v_dim,), dtype=mx.float16)
    h0 = (
        _flatten(initial_state)
        if initial_state is not None
        else mx.empty((num_sequences * hv_heads * k_dim * v_dim,), dtype=mx.float32)
    )
    ht = mx.empty((num_sequences * hv_heads * k_dim * v_dim,), dtype=mx.float32)
    g_scalar_unused = mx.empty((b * total_tokens * hv_heads,), dtype=mx.float32)

    aqk, akkd = _bind_returned_outputs(
        "kda.intra_token_parallel",
        _launch_stage(
            token,
            q_flat,
            k_flat,
            g_cumsum_flat,
            beta_flat,
            aqk,
            akkd,
            cu_seqlens_arg,
        ),
        (aqk, akkd),
    )
    aqk, akk = _bind_returned_outputs(
        "kda.inter_solve",
        _launch_stage(
            inter,
            q_flat,
            k_flat,
            g_cumsum_flat,
            beta_flat,
            aqk,
            akkd,
            akk,
            cu_seqlens_arg,
            chunk_indices_arg,
        ),
        (aqk, akk),
    )
    qg, kg, w, u = _bind_returned_outputs(
        "kda.recompute_w_u",
        _launch_stage(
            recompute,
            q_flat,
            k_flat,
            qg,
            kg,
            v_flat,
            beta_flat,
            w,
            u,
            akk,
            g_cumsum_flat,
            cu_seqlens_arg,
            chunk_indices_arg,
        ),
        (qg, kg, w, u),
    )
    v_new, h = _bind_returned_outputs(
        "kda.chunk_delta_h",
        _launch_stage(
            chunk_h,
            kg,
            u,
            w,
            v_new,
            g_scalar_unused,
            g_cumsum_flat,
            h,
            h0,
            ht,
            cu_seqlens_arg,
            chunk_offsets_arg,
        ),
        (v_new, h),
    )
    (o,) = _bind_returned_outputs(
        "kda.chunk_o_gk",
        _launch_stage(
            chunk_o,
            q_flat,
            v_new,
            g_cumsum_flat,
            h,
            o,
            aqk,
            cu_seqlens_arg,
            chunk_indices_arg,
        ),
        (o,),
    )

    y = mx.reshape(o, (b, total_tokens, hv_heads, v_dim))
    final_state = (
        mx.reshape(ht, (num_sequences, hv_heads, k_dim, v_dim))
        if output_final_state
        else None
    )
    return y, final_state


__all__ = [
    "ALLOW_DEGRADED_ENV",
    "GDN_RECURRENT_PLAN",
    "KDA_TOPOLOGY_CACHE_DIR_ENV",
    "KDA_TOPOLOGY_SPECIALIZATION_MAX_ENTRIES_ENV",
    "KDA_TOPOLOGY_SPECIALIZATION_THRESHOLD_ENV",
    "KDA_RECURRENT_PLAN",
    "KDATopologyDecision",
    "KDATopologyInvariants",
    "PathDCompileResult",
    "PathDKernelPlan",
    "PathDRecurrentPlan",
    "PathDRuntimeUnavailable",
    "compile_gdn_chunk_h_artifact",
    "compile_gdn_chunk_o_artifact",
    "compile_gdn_kkt_artifact",
    "compile_gdn_recompute_w_u_artifact",
    "compile_kda_chunk_o_artifact",
    "compile_kda_inter_solve_artifact",
    "compile_kda_intra_token_artifact",
    "compile_kda_recompute_w_u_artifact",
    "compile_tilelang_primfunc",
    "gdn_fwd_runtime_call",
    "gdn_runtime_adapter_status",
    "kda_fwd_runtime_call",
    "kda_runtime_adapter_status",
    "primfunc_has_degraded_markers",
    "specialize_primfunc_for_grid",
    "specialize_primfunc_for_scalars",
    "specialize_primfunc_for_topology_metadata",
]
