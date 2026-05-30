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
GDN_RUNTIME_BT = 32
GDN_CHUNK_H_METAL_SAFE_BV = 16
GDN_FIXED_CHUNK_O_BV = 16
# recompute_w_u (shared by GDN and KDA) materializes several BT*BK + BT*BV
# THREAD-PRIVATE accumulators (b_w, b_u, b_kb, b_vb, b_A, ...). With the FLA
# default BK=BV=64 these are ~4096 elements/thread (~27K total), which makes
# newComputePipelineStateWithFunction fail with "Compute function exceeds
# available stack space" on Apple Metal. Tiling the K/V reduction through
# smaller blocks shrinks each thread-private tile so the pipeline-state
# creation fits Metal's per-thread stack budget. This is an occupancy/tiling
# limit (it would fail on M5 too), independent of the Metal-4 language gate.
GDN_RECOMPUTE_W_U_METAL_SAFE_BK = 32
GDN_RECOMPUTE_W_U_METAL_SAFE_BV = 16
KDA_FIXED_T = 64
KDA_FIXED_H = 1
KDA_FIXED_HV = 1
KDA_FIXED_K = 64
KDA_FIXED_V = 32
KDA_FIXED_BT = 32
KDA_FIXED_BC = 16
KDA_FIXED_BV = 16
KDA_CHUNK_H_METAL_SAFE_BV = 8
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


def _kda_topology_cache_dir(cache_dir: Optional[str] = None) -> str:
    if cache_dir:
        return os.path.abspath(os.path.expanduser(cache_dir))
    override = os.environ.get(KDA_TOPOLOGY_CACHE_DIR_ENV)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    tilelang_cache = os.environ.get("TILELANG_CACHE_DIR")
    if tilelang_cache:
        root = os.path.abspath(os.path.expanduser(tilelang_cache))
    else:
        root = os.path.expanduser("~/.tilelang/cache")
    return os.path.join(root, "cppmega_kda_topologies")


def _kda_topology_manifest_path(
    fingerprint: str,
    cache_dir: Optional[str] = None,
) -> str:
    return os.path.join(_kda_topology_cache_dir(cache_dir), f"{fingerprint}.json")


def _write_kda_topology_manifest(
    fingerprint: str,
    *,
    descriptor: dict[str, Any],
    status: str,
    stages: tuple[str, ...],
    cache_dir: Optional[str] = None,
) -> None:
    manifest_dir = _kda_topology_cache_dir(cache_dir)
    os.makedirs(manifest_dir, exist_ok=True)
    path = _kda_topology_manifest_path(fingerprint, cache_dir=manifest_dir)
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


def _read_kda_topology_manifest(
    fingerprint: str,
    cache_dir: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    path = _kda_topology_manifest_path(fingerprint, cache_dir=cache_dir)
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


def _record_kda_topology_hit(
    key: tuple[Any, ...],
    *,
    threshold: Optional[int] = None,
    max_entries: Optional[int] = None,
) -> KDATopologyDecision:
    if threshold is None:
        threshold = _env_int(KDA_TOPOLOGY_SPECIALIZATION_THRESHOLD_ENV, 3)
    threshold = max(int(threshold), 0)
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

    if max_entries is None:
        max_entries = _env_int(KDA_TOPOLOGY_SPECIALIZATION_MAX_ENTRIES_ENV, 64)
    max_entries = max(int(max_entries), 1)
    if key not in _KDA_TOPOLOGY_HITS and len(_KDA_TOPOLOGY_HITS) >= max_entries:
        oldest = next(iter(_KDA_TOPOLOGY_HITS))
        _KDA_TOPOLOGY_HITS.pop(oldest, None)
        _KDA_TOPOLOGY_DISABLED.discard(oldest)
    hits = _KDA_TOPOLOGY_HITS.pop(key, 0) + 1
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


def _with_topology_arg_aliases(
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]],
    aliases: dict[str, str],
) -> Optional[dict[str, tuple[int, ...] | list[int]]]:
    if not topology_constants:
        return topology_constants
    out: dict[str, tuple[int, ...] | list[int]] = dict(topology_constants)
    for semantic_name, arg_name in aliases.items():
        values = topology_constants.get(semantic_name)
        if values is not None:
            out[arg_name] = values
    return out


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


def canonicalize_kda_intra_token_static_launch(prim_func: Any) -> Any:
    """Shrink KDA intra-token static-launch control expressions before compile.

    Triton emits the diagonal-token loop as ``range(i_ts, min(i_t + 1, ...))``
    and PtrAnalysis lowers block-pointer boundary checks into nested
    ``min/max`` masks. Once cppmega has specialized the fixed KDA launch
    grid and scalar ``T`` value, those expressions are exact but too large for
    TileLang's interval prover. This pass rewrites the algebraic equivalent
    into the small loop the schedule meant all along.
    """

    from tvm import tir
    from tvm.tir import stmt_functor

    thread_extents: dict[str, int] = {}

    def gather_thread_extent(node: Any) -> None:
        if type(node).__name__ != "AttrStmt":
            return
        if str(getattr(node, "attr_key", "")) != "thread_extent":
            return
        iter_var = getattr(node, "node", None)
        var = getattr(iter_var, "var", None)
        value = getattr(getattr(node, "value", None), "value", None)
        if var is not None and value is not None:
            thread_extents[str(var)] = int(value)

    stmt_functor.post_order_visit(prim_func.body, gather_thread_extent)
    if not thread_extents:
        return prim_func

    def int_value(expr: Any) -> Optional[int]:
        if type(expr).__name__ != "IntImm":
            return None
        value = getattr(expr, "value", None)
        return None if value is None else int(value)

    def dtype_of(expr: Any) -> str:
        return str(getattr(expr, "dtype", "int32"))

    def is_var(expr: Any) -> bool:
        return type(expr).__name__ == "Var"

    def same_var(lhs: Any, rhs: Any) -> bool:
        return is_var(lhs) and is_var(rhs) and str(lhs) == str(rhs)

    def make_int(dtype: str, value: int) -> Any:
        return tir.IntImm(dtype, int(value))

    def mul_div_var(expr: Any) -> Optional[tuple[Any, int]]:
        """Match ``floordiv(var, c) * c``."""

        if type(expr).__name__ != "Mul":
            return None
        for lhs, rhs in ((expr.a, expr.b), (expr.b, expr.a)):
            factor = int_value(rhs)
            if factor is None or type(lhs).__name__ != "Div":
                continue
            if int_value(lhs.b) == factor and is_var(lhs.a):
                return lhs.a, factor
        return None

    def mul_div_mod(expr: Any) -> Optional[tuple[Any, int, int]]:
        """Match ``floordiv(truncmod(var, outer), inner) * inner``."""

        if type(expr).__name__ != "Mul":
            return None
        for lhs, rhs in ((expr.a, expr.b), (expr.b, expr.a)):
            inner = int_value(rhs)
            if inner is None or type(lhs).__name__ != "Div":
                continue
            if int_value(lhs.b) != inner:
                continue
            mod = lhs.a
            if (
                type(mod).__name__ == "Mod"
                and is_var(mod.a)
                and int_value(mod.b) is not None
            ):
                return mod.a, int(int_value(mod.b)), inner
        return None

    def match_two_level_aligned_floor(expr: Any) -> Optional[tuple[Any, int]]:
        """Match ``(x//A)*A + ((x%A)//B)*B`` where ``A`` is a multiple of ``B``."""

        if type(expr).__name__ != "Add":
            return None
        for lhs, rhs in ((expr.a, expr.b), (expr.b, expr.a)):
            outer = mul_div_var(lhs)
            inner = mul_div_mod(rhs)
            if outer is None or inner is None:
                continue
            outer_var, outer_block = outer
            inner_var, modulo, inner_block = inner
            if (
                same_var(outer_var, inner_var)
                and outer_block == modulo
                and inner_block > 0
                and outer_block % inner_block == 0
            ):
                return outer_var, inner_block
        return None

    def match_aligned_floor(expr: Any) -> Optional[tuple[Any, int]]:
        return mul_div_var(expr) or match_two_level_aligned_floor(expr)

    def rebuild_aligned_floor(var: Any, block: int) -> Any:
        dtype = dtype_of(var)
        return tir.Mul(
            tir.Div(var, make_int(dtype, block)),
            make_int(dtype, block),
        )

    def canonicalize_expr(node: Any) -> Any:
        node_type = type(node).__name__
        if node_type == "Mul":
            lhs_zero = int_value(node.a) == 0
            rhs_zero = int_value(node.b) == 0
            if lhs_zero or rhs_zero:
                return make_int(dtype_of(node), 0)
            if int_value(node.a) == 1:
                return node.b
            if int_value(node.b) == 1:
                return node.a
        if node_type == "Add":
            if int_value(node.a) == 0:
                return node.b
            if int_value(node.b) == 0:
                return node.a
        if node_type == "Call":
            op_name = str(getattr(getattr(node, "op", None), "name", ""))
            if op_name == "tirx.if_then_else":
                args = list(getattr(node, "args", ()))
                if (
                    len(args) == 3
                    and "T.min" in str(args[0])
                    and type(args[1]).__name__ == "BufferLoad"
                    and str(getattr(args[1].buffer, "name", "")) == "arg2"
                ):
                    return args[1]
        if node_type == "Mod":
            rhs = int_value(node.b)
            if (
                rhs is not None
                and is_var(node.a)
                and thread_extents.get(str(node.a), rhs + 1) <= rhs
            ):
                return node.a
        if node_type == "Div":
            rhs = int_value(node.b)
            if (
                rhs is not None
                and is_var(node.a)
                and thread_extents.get(str(node.a), rhs + 1) <= rhs
            ):
                return make_int(dtype_of(node.a), 0)
        if node_type == "Add":
            matched = match_two_level_aligned_floor(node)
            if matched is not None:
                var, block = matched
                return rebuild_aligned_floor(var, block)
        return node

    body = stmt_functor.ir_transform(
        prim_func.body,
        None,
        canonicalize_expr,
        ["tirx.Call", "tirx.Mod", "tirx.Div", "tirx.Mul", "tirx.Add"],
    )

    def canonicalize_for(node: Any) -> Any:
        if type(node).__name__ != "For":
            return node
        matched = match_aligned_floor(node.min)
        if matched is None or "T.min" not in str(node.extent):
            return node
        var, block = matched
        total_extent = thread_extents.get(str(var))
        if total_extent is None or block <= 0 or total_extent % block != 0:
            return node
        dtype = dtype_of(var)
        new_min = rebuild_aligned_floor(var, block)
        new_extent = tir.Add(
            tir.Mod(var, make_int(dtype, block)),
            make_int(dtype, 1),
        )
        return tir.For(
            node.loop_var,
            new_min,
            new_extent,
            node.kind,
            node.body,
            node.thread_binding,
            node.annotations,
        )

    body = stmt_functor.ir_transform(body, None, canonicalize_for, ["tirx.For"])
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
        if plan.name == KDA_INTRA_TOKEN_PLAN.name:
            specialized = canonicalize_kda_intra_token_static_launch(specialized)
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
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
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
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
    )


@lru_cache(maxsize=16)
def _compile_gdn_recompute_w_u_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
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
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
    )


@lru_cache(maxsize=16)
def _compile_gdn_chunk_o_cached(
    constexprs_key: tuple[tuple[str, Any], ...],
    grid: tuple[int, ...],
    scalar_specializations: tuple[Any, ...],
    topology_constants_key: tuple[tuple[str, tuple[int, ...]], ...],
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
        scalar_specializations=scalar_specializations,
        topology_constants=_thaw_topology_constants(topology_constants_key),
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
    scalar_specializations: Optional[tuple[Any, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
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
        GDN_KKT_PLAN.scalar_specializations
        if scalar_specializations is None
        else scalar_specializations,
        _freeze_topology_constants(topology_constants),
        bool(allow_degraded_primfunc),
    )


def compile_gdn_recompute_w_u_artifact(
    *,
    constexprs: Optional[dict[str, Any]] = None,
    grid: tuple[int, ...] = (1, 1),
    scalar_specializations: Optional[tuple[Any, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
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
        GDN_RECOMPUTE_W_U_PLAN.scalar_specializations
        if scalar_specializations is None
        else scalar_specializations,
        _freeze_topology_constants(topology_constants),
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
    scalar_specializations: Optional[tuple[Any, ...]] = None,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
    allow_degraded_primfunc: bool = False,
) -> PathDCompileResult:
    """Compile/cache the currently lowerable GDN Path D chunk-o artifact."""

    from cppmega_v4._tilelang.linear_attention_path_d_real import (
        DEFAULT_CHUNK_O_CONSTEXPRS,
    )

    cfg = dict(DEFAULT_CHUNK_O_CONSTEXPRS)
    # The original FLA BT=64 chunk_o lowering exceeds Apple's 32 KiB
    # threadgroup-memory limit. Use the same safe runtime chunk size here
    # as the multi-stage GDN launcher uses below.
    cfg["BT"] = GDN_RUNTIME_BT
    cfg["BV"] = GDN_FIXED_CHUNK_O_BV
    if constexprs:
        cfg.update(constexprs)
    if grid is None:
        t = int(cfg.get("_RUNTIME_T", GDN_FIXED_T))
        bt = int(cfg.get("BT", GDN_RUNTIME_BT))
        grid = (
            math.ceil(
                int(cfg.get("V", GDN_FIXED_V))
                / int(cfg.get("BV", GDN_FIXED_CHUNK_O_BV))
            ),
            math.ceil(t / bt),
            1,
        )
    return _compile_gdn_chunk_o_cached(
        _freeze_items(cfg),
        tuple(int(x) for x in grid),
        GDN_CHUNK_O_PLAN.scalar_specializations
        if scalar_specializations is None
        else scalar_specializations,
        _freeze_topology_constants(topology_constants),
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
            # Tile K/V through Metal-safe blocks so the per-thread WY-recompute
            # accumulators fit the pipeline-state stack budget (shared root
            # cause with GDN recompute_w_u -> "exceeds available stack space").
            "BK": min(int(k_dim), GDN_RECOMPUTE_W_U_METAL_SAFE_BK),
            "BV": min(int(v_dim), GDN_RECOMPUTE_W_U_METAL_SAFE_BV),
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


def _gdn_chunk_local_cumsum(
    g: Any,
    *,
    chunk_size: int,
    cu_seqlens_values: Optional[list[int]],
) -> Any:
    """Chunk-local cumulative sum of the GDN scalar gate ``g[B,T,HV]``.

    FLA's chunk_delta_h consumes a chunk-local cumulative gate; the Path A
    reference applies ``exp(cumsum(g))`` per recurrence step. This mirrors
    ``_kda_chunk_local_cumsum`` but for the rank-3 scalar gate (no K axis),
    and matches FLA's ``BT``-aligned chunking and varlen sequence spans.
    """

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
                parts.append(mx.cumsum(g[:, chunk_start:chunk_end, :], axis=1))
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
        if actual is expected:
            continue
        actual_shape = tuple(int(x) for x in getattr(actual, "shape", ()))
        expected_shape = tuple(int(x) for x in getattr(expected, "shape", ()))
        actual_dtype = str(getattr(actual, "dtype", ""))
        expected_dtype = str(getattr(expected, "dtype", ""))
        if actual_shape != expected_shape or actual_dtype != expected_dtype:
            raise PathDRuntimeUnavailable(
                f"{stage} returned incompatible owner-output alias for "
                f"out_idx[{idx}]: got shape={actual_shape} dtype={actual_dtype}, "
                f"expected shape={expected_shape} dtype={expected_dtype}"
            )
    return returned_outputs


def _compile_gdn_runtime_stages(
    *,
    batch: int,
    total_tokens: int = GDN_FIXED_T,
    num_sequences: Optional[int] = None,
    num_chunks: Optional[int] = None,
    h_heads: int = GDN_FIXED_H,
    hv_heads: int = GDN_FIXED_HV,
    k_dim: int = GDN_FIXED_K,
    v_dim: int = GDN_FIXED_V,
    is_varlen: bool = False,
    scale: Any = None,
    use_initial_state: bool = False,
    output_final_state: bool = False,
    topology_constants: Optional[dict[str, tuple[int, ...] | list[int]]] = None,
) -> tuple[
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
    PathDCompileResult,
]:
    """Compile the GDN forward FLA pipeline for an arbitrary prefill shape.

    Mirrors ``_compile_kda_runtime_stages``: H/HV/K/V/T, varlen topology, and
    custom scale all flow through the same Triton-frontend lowering. The GDN
    forward uses the scalar cumulative gate (``USE_G``), so ``chunk_delta_h``
    runs with ``USE_G=True``/``USE_GK=False`` while KDA uses the vector gate.
    """

    nt = _ceil_div(total_tokens, GDN_RUNTIME_BT)
    chunks_per_batch = nt
    stage_chunks = int(num_chunks) if (is_varlen and num_chunks is not None) else chunks_per_batch
    total_h_chunks = (
        int(num_chunks)
        if (is_varlen and num_chunks is not None)
        else int(batch) * chunks_per_batch
    )
    n_seq = int(num_sequences) if num_sequences is not None else int(batch)
    chunk_scale = _kda_default_scale(k_dim, scale)
    stage_grid = (stage_chunks, batch * hv_heads)
    chunk_v_grid = (
        math.ceil(v_dim / GDN_CHUNK_H_METAL_SAFE_BV),
        (n_seq if is_varlen else batch) * hv_heads,
    )
    chunk_o_grid = (
        math.ceil(v_dim / GDN_FIXED_CHUNK_O_BV),
        stage_chunks,
        batch * hv_heads,
    )
    runtime_private = {
        "_RUNTIME_BATCH": int(batch),
        "_RUNTIME_T": int(total_tokens),
        "_RUNTIME_N": n_seq,
        "_RUNTIME_NT": total_h_chunks,
        "_RUNTIME_CU_LEN": n_seq + 1,
    }
    shared_dims = {
        "H": int(h_heads),
        "HV": int(hv_heads),
        "K": int(k_dim),
        "BT": GDN_RUNTIME_BT,
        "IS_VARLEN": bool(is_varlen),
    }
    kkt_constexprs = {**runtime_private, **shared_dims}
    recompute_constexprs = {
        **runtime_private,
        **shared_dims,
        "V": int(v_dim),
        # Tile the K/V reduction through Metal-safe blocks so the per-thread
        # recompute_w_u accumulators fit the pipeline-state stack budget.
        "BK": min(int(k_dim), GDN_RECOMPUTE_W_U_METAL_SAFE_BK),
        "BV": min(int(v_dim), GDN_RECOMPUTE_W_U_METAL_SAFE_BV),
    }
    chunk_h_constexprs = {
        **runtime_private,
        "H": int(h_heads),
        "HV": int(hv_heads),
        "K": int(k_dim),
        "V": int(v_dim),
        "BT": GDN_RUNTIME_BT,
        "BV": GDN_CHUNK_H_METAL_SAFE_BV,
        "USE_G": True,
        "USE_GK": False,
        "USE_INITIAL_STATE": bool(use_initial_state),
        "STORE_FINAL_STATE": bool(output_final_state),
        "SAVE_NEW_VALUE": True,
        "IS_VARLEN": bool(is_varlen),
    }
    chunk_o_constexprs = {
        **runtime_private,
        "H": int(h_heads),
        "HV": int(hv_heads),
        "K": int(k_dim),
        "V": int(v_dim),
        "BT": GDN_RUNTIME_BT,
        "BV": GDN_FIXED_CHUNK_O_BV,
        "IS_VARLEN": bool(is_varlen),
    }
    # Topology constant-folding is a hot-path optimization that is currently
    # only wired for the common chunk_delta_h kernel (shared with KDA). The
    # kkt/recompute/chunk_o stages keep the generic dynamic varlen path: they
    # still receive cu_seqlens/chunk_indices as runtime launch args, so the
    # result is identical — only the metadata loads are not constant-folded.
    return (
        compile_gdn_kkt_artifact(
            constexprs=kkt_constexprs,
            grid=stage_grid,
            scalar_specializations=(int(total_tokens),),
        ),
        compile_gdn_recompute_w_u_artifact(
            constexprs=recompute_constexprs,
            grid=stage_grid,
            scalar_specializations=(int(total_tokens),),
        ),
        compile_gdn_chunk_h_artifact(
            constexprs=chunk_h_constexprs,
            grid=chunk_v_grid,
            scalar_specializations=(int(total_tokens),),
            topology_constants=_with_topology_arg_aliases(
                topology_constants,
                {"cu_seqlens": "arg9", "chunk_offsets": "arg10"},
            ),
        ),
        compile_gdn_chunk_o_artifact(
            constexprs=chunk_o_constexprs,
            grid=chunk_o_grid,
            scalar_specializations=(float(chunk_scale), int(total_tokens)),
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
        _ceil_div(v_dim, KDA_CHUNK_H_METAL_SAFE_BV),
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
        "BV": KDA_CHUNK_H_METAL_SAFE_BV,
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
            topology_constants=_with_topology_arg_aliases(
                topology_constants,
                {"cu_seqlens": "arg7"},
            ),
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
            topology_constants=_with_topology_arg_aliases(
                topology_constants,
                {"cu_seqlens": "arg8", "chunk_indices": "arg9"},
            ),
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
            topology_constants=_with_topology_arg_aliases(
                topology_constants,
                {"cu_seqlens": "arg10", "chunk_indices": "arg11"},
            ),
        ),
        compile_gdn_chunk_h_artifact(
            constexprs=chunk_h_constexprs,
            grid=chunk_v_grid,
            scalar_specializations=(int(total_tokens),),
            topology_constants=_with_topology_arg_aliases(
                topology_constants,
                {"cu_seqlens": "arg9", "chunk_offsets": "arg10"},
            ),
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
            topology_constants=_with_topology_arg_aliases(
                topology_constants,
                {"cu_seqlens": "arg6", "chunk_indices": "arg7"},
            ),
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
        "GDN Path D runtime adapter available for shape-specialized fp16 "
        "prefill and packed varlen, USE_G cumulative gate, custom scale, "
        "initial/final state, fused Path B backward; compile/cache/launch "
        f"stages={', '.join(stage.name for stage in GDN_RECURRENT_PLAN.stages)}"
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
    cu_seqlens: Any = None,
    chunk_indices: Any = None,
    chunk_offsets: Any = None,
    compile_stages_fn: Optional[Callable[..., Any]] = None,
    launch_stage_fn: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> Any:
    """Launch the GDN Path D FLA multi-kernel pipeline.

    Generalized from the original fixed B x 64 x H=1 x K=64 / V=32 prefill
    slice to arbitrary H/HV/K/V/T, custom scale, initial/final recurrent
    state, and packed varlen metadata — mirroring ``kda_fwd_runtime_call``.
    The GDN forward uses the scalar cumulative gate (``USE_G``); the per-token
    gate is broadcast/cumsumed exactly as the Path A reference and FLA expect.
    Unsupported signatures raise ``PathDRuntimeUnavailable`` so the dispatcher
    keeps the existing production fallback.
    """

    if kwargs:
        raise PathDRuntimeUnavailable(
            "GDN Path D runtime adapter does not support extra keyword args: "
            f"{sorted(kwargs)}"
        )

    import mlx.core as mx

    compile_stages = (
        _compile_gdn_runtime_stages
        if compile_stages_fn is None
        else compile_stages_fn
    )
    launch_stage = _launch_stage if launch_stage_fn is None else launch_stage_fn

    q_shape = tuple(int(x) for x in getattr(q, "shape", ()))
    if len(q_shape) != 4:
        raise PathDRuntimeUnavailable(
            f"GDN Path D runtime adapter expects q as [B,T,H,K]; got {q_shape}"
        )
    b, total_tokens, h_heads, k_dim = q_shape
    if min(b, total_tokens, h_heads, k_dim) <= 0:
        raise PathDRuntimeUnavailable(
            f"GDN Path D runtime adapter got invalid q shape {q_shape}"
        )
    v_shape = tuple(int(x) for x in getattr(v, "shape", ()))
    if len(v_shape) != 4:
        raise PathDRuntimeUnavailable(
            f"GDN Path D runtime adapter expects v as [B,T,HV,V]; got {v_shape}"
        )
    if v_shape[0] != b or v_shape[1] != total_tokens:
        raise PathDRuntimeUnavailable(
            "GDN Path D runtime adapter requires q/k/v to share B and T; "
            f"got q={q_shape}, v={v_shape}"
        )
    hv_heads = int(v_shape[2])
    v_dim = int(v_shape[3])
    if hv_heads <= 0 or v_dim <= 0 or hv_heads % h_heads != 0:
        raise PathDRuntimeUnavailable(
            "GDN Path D runtime adapter requires HV > 0, V > 0, and HV % H == 0; "
            f"got H={h_heads}, HV={hv_heads}, V={v_dim}"
        )
    if k_dim > 256:
        raise PathDRuntimeUnavailable(
            "GDN Path D common chunk_delta_h supports K <= 256; "
            f"got K={k_dim}"
        )

    expected_qk = (b, total_tokens, h_heads, k_dim)
    expected_v = (b, total_tokens, hv_heads, v_dim)
    expected_gate = (b, total_tokens, hv_heads)

    _require_shape_dtype("q", q, expected_qk, mx.float16)
    _require_shape_dtype("k", k, expected_qk, mx.float16)
    _require_shape_dtype("v", v, expected_v, mx.float16)
    _require_shape_dtype("beta", beta, expected_gate, mx.float32)
    _require_shape_dtype("g", g, expected_gate, mx.float32)

    is_varlen = cu_seqlens is not None
    if is_varlen and b != 1:
        raise PathDRuntimeUnavailable(
            "GDN Path D varlen expects packed inputs with B=1; "
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
        chunk_size=GDN_RUNTIME_BT,
        mx=mx,
    )
    num_sequences = varlen_sequences if is_varlen else b
    num_chunks = (
        varlen_chunks
        if is_varlen
        else b * _ceil_div(total_tokens, GDN_RUNTIME_BT)
    )

    if initial_state is not None:
        _require_shape_dtype(
            "initial_state",
            initial_state,
            (num_sequences, hv_heads, k_dim, v_dim),
            mx.float32,
        )

    topology_constants: Optional[dict[str, tuple[int, ...]]] = None
    if is_varlen:
        topology_constants = {
            "cu_seqlens": tuple(cu_values),
            "chunk_indices": tuple(
                _as_int_list(chunk_indices_arg, name="chunk_indices")
            ),
            "chunk_offsets": tuple(
                _as_int_list(chunk_offsets_arg, name="chunk_offsets")
            ),
        }

    stage_kwargs = {
        "batch": b,
        "total_tokens": total_tokens,
        "num_sequences": num_sequences,
        "num_chunks": num_chunks,
        "h_heads": h_heads,
        "hv_heads": hv_heads,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "is_varlen": is_varlen,
        "scale": scale,
        "use_initial_state": initial_state is not None,
        "output_final_state": bool(output_final_state),
    }
    stages = compile_stages(**stage_kwargs)
    if topology_constants is not None and all(stage.available for stage in stages):
        specialized = compile_stages(
            **stage_kwargs,
            topology_constants=topology_constants,
        )
        if all(stage.available for stage in specialized):
            stages = specialized
    kkt, recompute, chunk_h, chunk_o = stages
    for stage in (kkt, recompute, chunk_h, chunk_o):
        if not stage.available:
            raise PathDRuntimeUnavailable(stage.reason)

    nt = num_chunks
    q_flat = _flatten(q)
    k_flat = _flatten(k)
    v_flat = _flatten(v)
    beta_flat = _flatten(beta)
    g_cumsum_flat = _flatten(
        _gdn_chunk_local_cumsum(
            g,
            chunk_size=GDN_RUNTIME_BT,
            cu_seqlens_values=cu_values if is_varlen else None,
        )
    )

    a = mx.empty((b * total_tokens * hv_heads * GDN_RUNTIME_BT,), dtype=mx.float16)
    w = mx.empty((b * total_tokens * hv_heads * k_dim,), dtype=mx.float16)
    u = mx.empty((b * total_tokens * hv_heads * v_dim,), dtype=mx.float16)
    v_new = mx.empty((b * total_tokens * hv_heads * v_dim,), dtype=mx.float16)
    h = mx.empty((nt * hv_heads * k_dim * v_dim,), dtype=mx.float16)
    o = mx.empty((b * total_tokens * hv_heads * v_dim,), dtype=mx.float16)
    h0 = (
        _flatten(initial_state)
        if initial_state is not None
        else mx.empty((num_sequences * hv_heads * k_dim * v_dim,), dtype=mx.float32)
    )
    ht = mx.empty((num_sequences * hv_heads * k_dim * v_dim,), dtype=mx.float32)
    gk = mx.empty((b * total_tokens * hv_heads * k_dim,), dtype=mx.float32)
    g_gamma = mx.empty((hv_heads,), dtype=mx.float32)

    (a,) = _bind_returned_outputs(
        "gdn.kkt_solve",
        launch_stage(
            kkt, k_flat, g_cumsum_flat, beta_flat, a, cu_seqlens_arg, chunk_indices_arg
        ),
        (a,),
    )
    w, u = _bind_returned_outputs(
        "gdn.recompute_w_u",
        launch_stage(
            recompute,
            k_flat,
            v_flat,
            beta_flat,
            w,
            u,
            a,
            g_cumsum_flat,
            cu_seqlens_arg,
            chunk_indices_arg,
        ),
        (w, u),
    )
    v_new, h = _bind_returned_outputs(
        "gdn.chunk_delta_h",
        launch_stage(
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
            cu_seqlens_arg,
            chunk_offsets_arg,
        ),
        (v_new, h),
    )
    (o,) = _bind_returned_outputs(
        "gdn.chunk_o",
        launch_stage(
            chunk_o,
            q_flat,
            k_flat,
            v_new,
            h,
            g_cumsum_flat,
            g_gamma,
            o,
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


def gdn_bwd_runtime_call(
    q: Any,
    k: Any,
    v: Any,
    beta: Any,
    g: Any,
    cotangent: Any,
    *,
    scale: Any = None,
    **kwargs: Any,
) -> Any:
    """Backward for the GDN Path D forward, reusing Path B's fused Metal VJP.

    Lowering a separate FLA ``chunk_gated_delta_rule_bwd`` TTIR would hit the
    same Metal cooperative-tensor GEMM path and need new OP_TABLE coverage;
    Path B already ships a real, fused, recurrent Metal backward kernel with
    the identical ``(q, k, v, beta, g)`` recurrent signature that matches the
    Path A reference forward to ~1e-8. Reusing it keeps Path D differentiable
    on every Apple-Silicon GPU (M1+) while the forward exercises the
    Triton-frontend pipeline. Returns the per-input cotangents
    ``(dq, dk, dv, dbeta, dg)``.
    """

    if kwargs:
        raise PathDRuntimeUnavailable(
            "GDN Path D backward does not support extra keyword args: "
            f"{sorted(kwargs)}"
        )
    if scale is not None:
        _validate_default_scale(
            scale, k_dim=int(getattr(q, "shape", (0, 0, 0, GDN_FIXED_K))[-1])
        )

    from cppmega_v4._tilelang.linear_attention_path_b_bwd import (
        _MAX_DIM,
        _gdn_backward_kernel,
        _path_a_grad_fallback,
    )

    primals = (q, k, v, beta, g)
    kdim = int(q.shape[-1])
    vdim = int(v.shape[-1])
    bwd_ok = (
        tuple(k.shape) == tuple(q.shape)
        and tuple(v.shape[:3]) == tuple(q.shape[:3])
        and max(kdim, vdim) <= _MAX_DIM
    )
    if not bwd_ok:
        return _path_a_grad_fallback(primals, cotangent)
    return _gdn_backward_kernel(q, k, v, beta, g, cotangent)


def gdn_apply_path_d(
    q: Any,
    k: Any,
    v: Any,
    beta: Any,
    g: Any,
    *,
    scale: Any = None,
    initial_state: Any = None,
    cu_seqlens: Any = None,
):
    """Differentiable GDN Path D forward returning ``y`` only.

    Forward runs the Triton-frontend FLA chunk pipeline; backward reuses the
    Path B fused Metal VJP (see :func:`gdn_bwd_runtime_call`). Mirrors
    ``gdn_apply_path_b`` so callers get a drop-in autograd-traced op.
    """

    import mlx.core as mx

    @mx.custom_function
    def _apply(q, k, v, beta, g):
        y, _ = gdn_fwd_runtime_call(
            q,
            k,
            v,
            beta,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        return y

    @_apply.vjp
    def _apply_vjp(primals, cotangent, output):
        del output
        pq, pk, pv, pbeta, pg = primals
        return gdn_bwd_runtime_call(pq, pk, pv, pbeta, pg, cotangent, scale=scale)

    return _apply(q, k, v, beta, g)


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
    topology_specialization_threshold: Optional[int] = None,
    topology_specialization_max_entries: Optional[int] = None,
    topology_cache_dir: Optional[str] = None,
    compile_stages_fn: Optional[Callable[..., Any]] = None,
    launch_stage_fn: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> Any:
    """Launch the KDA Path D FLA multi-kernel pipeline."""

    if kwargs:
        raise PathDRuntimeUnavailable(
            "KDA Path D runtime adapter does not support extra keyword args: "
            f"{sorted(kwargs)}"
        )

    import mlx.core as mx

    compile_stages = (
        _compile_kda_runtime_stages
        if compile_stages_fn is None
        else compile_stages_fn
    )
    launch_stage = _launch_stage if launch_stage_fn is None else launch_stage_fn

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
        if topology_specialization_threshold is None:
            topology_specialization_threshold = 1
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
        topology_decision = _record_kda_topology_hit(
            topology_key,
            threshold=topology_specialization_threshold,
            max_entries=topology_specialization_max_entries,
        )
        if topology_decision.use_specialized:
            topology_constants = {
                "cu_seqlens": tuple(cu_values),
                "chunk_indices": chunk_indices_values,
                "chunk_offsets": chunk_offsets_values,
            }

    stage_kwargs = {
        "batch": b,
        "total_tokens": total_tokens,
        "num_sequences": num_sequences,
        "num_chunks": num_chunks,
        "h_heads": h_heads,
        "hv_heads": hv_heads,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "is_varlen": is_varlen,
        "scale": scale,
        "use_initial_state": initial_state is not None,
        "output_final_state": bool(output_final_state),
    }
    stages = compile_stages(**stage_kwargs)
    if topology_constants is not None and all(stage.available for stage in stages):
        specialized_stages = compile_stages(
            **stage_kwargs,
            topology_constants=topology_constants,
        )
        if all(stage.available for stage in specialized_stages):
            stages = specialized_stages
            if topology_fingerprint is not None and topology_descriptor is not None:
                try:
                    _write_kda_topology_manifest(
                        topology_fingerprint,
                        descriptor=topology_descriptor,
                        status="compiled",
                        stages=tuple(
                            stage.plan.name if stage.plan is not None else "unknown"
                            for stage in stages
                        ),
                        cache_dir=topology_cache_dir,
                    )
                except OSError:
                    pass
        else:
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
                            for stage in specialized_stages
                        ),
                        cache_dir=topology_cache_dir,
                    )
                except OSError:
                    pass
    token, inter, recompute, chunk_h, chunk_o = stages
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
        launch_stage(
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
        launch_stage(
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
        launch_stage(
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
        launch_stage(
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
        launch_stage(
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
    "gdn_apply_path_d",
    "gdn_bwd_runtime_call",
    "gdn_fwd_runtime_call",
    "gdn_runtime_adapter_status",
    "kda_fwd_runtime_call",
    "kda_runtime_adapter_status",
    "primfunc_has_degraded_markers",
    "specialize_primfunc_for_grid",
    "specialize_primfunc_for_scalars",
    "specialize_primfunc_for_topology_metadata",
]
