# pyright: reportInvalidTypeForm=false, reportMissingImports=false
"""Path C FP8 vecmat/GEMV via TileLang DSL lowering.

This module is the TileLang-DSL counterpart to the hand-written Path B MSL
``fp8_scaled_vecmat`` kernel in :mod:`cppmega_mlx.nn._tilelang.fp8_msl_kernels`.
It is intentionally scoped to the inference shape that matters for this lane:
``M == 1``, ``B`` already transposed as ``(N, K)``, and e4m3 storage.

The default Metal lowering uses a TileLang intrinsic for packed uint32 e4m3
dot4 decode plus ``tvm_thread_allreduce`` across K. That keeps the generated
MSL on the same hot-loop shape as Path B's hand-written vecmat kernel.

Migration phase-4 (2026-05-14)
-------------------------------
The Path C runtime surface is tvm-ffi owner-output only. Diagnostic MSL
lowering still routes through
:func:`cppmega_mlx.nn._tilelang._engine_dispatch.dispatch_lower` with
``return_msl=True`` so tests and benchmark receipts can inspect generated
source, but runtime dispatch no longer builds an MLX fast-kernel wrapper.

The production owner-output API is
``fp8_scaled_vecmat_path_c(..., out=existing_array)``: it compiles a tvm-ffi
kernel and passes the caller-owned MLX buffers through DLPack without building
``mx.fast.metal_kernel`` or allocating/casting the output in Python. The
historical no-``out`` API is retired because it could only be implemented by
allocating an output through MLX's direct-MSL wrapper.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from dataclasses import dataclass, replace
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    _assert_path_c_metal_fp8_intrinsics_registered,
    can_run_metal,
)


TILELANG_METAL_VECMAT_TARGET = "metal -thread_warp_size=32"
FP8_PATH_C_LEGACY_MLX_FAST_ENV = "CPPMEGA_FP8_PATH_C_LEGACY_MLX_FAST"


# CPPMEGA Z3 wiring (beads cppmega-mlx-cuz): per-kernel PassConfig opt-in.
#
# The TileLang source tree declares several Z3-roadmap PassConfig keys; only a
# subset are actually registered with TVM's ``transform.PassContext`` in the
# in-tree built ``libtilelang.dylib``. We probe each candidate the first time
# we build the active config dict and silently drop any unsupported keys so
# the kernel doesn't fail to lower on builds that haven't picked up the
# latest config registration. Idea #10 (fp8 dot4 legality) is NOT a
# PassConfig — it is enforced inside ``T.fp8_scaled_matmul`` directly and
# toggled by the env var ``TILELANG_DISABLE_FP8_DOT4_AUTO``; we don't
# override that env var here.
_FP8_VECMAT_PATH_C_CANDIDATE_PASS_CONFIGS: dict[str, Any] = {
    # The current Metal scalar/CSE pipeline can produce an invalid one-element
    # SeqStmt around the canonical C[col] write in this reducer. This kernel
    # does not need CSE: diagnostic source inspection fuses in the canonical
    # packed hot loop, and the owner-output route already emits one
    # dot4/simd_sum body. Keep the gate per-kernel rather than disabling CSE
    # globally.
    "tirx.disable_cse_tir": True,
    # Z3 idea #4 — discharges ``if (i < N)`` guards the analyzer can prove.
    # The M=1 vecmat hot loop has tight static extents so the prover tends
    # to succeed; if not, the guard stays.
    "tl.drop_provable_bound_checks": True,
}

_FP8_VECMAT_PATH_C_FILTERED_KEYS_LOGGED: set[str] = set()
_FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE: dict[str, Any] | None = None
# grok design P2: cache the result of the Metal FP8 intrinsic registration
# check so we don't re-validate on every kernel build. ``True`` means
# "intrinsics confirmed registered this process"; ``False`` means we have
# not checked yet. Once checked, subsequent macro-path builds skip the
# scan entirely.
_FP8_VECMAT_PATH_C_INTRINSICS_CHECKED: bool = False
# grok wave-2 correctness P2: a single re-entrant lock now guards both the
# intrinsics-check first-run and the PassConfig probe-and-cache. They are
# *not* fully independent: the macro-path branch of
# ``make_fp8_vecmat_reduce_kernel`` calls
# ``_ensure_path_c_metal_fp8_intrinsics_registered`` and the surrounding
# build flow may also pull ``_fp8_vecmat_pass_configs`` (e.g. via
# ``lower_fp8_vecmat_msl``); separate locks let two MLX worker threads
# interleave the two first-time probes in pathological orderings, which
# duplicated the one-shot warnings and (in the worst case) partially
# populated ``_FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE`` from one thread while
# another read it. A single ``RLock`` is safe — the critical sections never
# call into each other recursively, but reentrance keeps the door open if
# they ever do — and serialises both first-time setups behind one barrier.
_FP8_VECMAT_PATH_C_FIRST_RUN_LOCK = threading.RLock()


def _filter_supported_pass_configs(candidates: dict[str, Any]) -> dict[str, Any]:
    """Drop PassConfig keys not registered in the active libtilelang build.

    Each candidate is probed by attempting to construct a minimal
    ``tvm.transform.PassContext`` with it; an FFI ``AttributeError``
    indicates the key is unknown to the live runtime registry. We log a
    one-shot warning per unsupported key.
    """

    try:
        from tilelang import tvm  # type: ignore
    except Exception:
        return {}

    supported: dict[str, Any] = {}
    for key, value in candidates.items():
        try:
            with tvm.transform.PassContext(opt_level=3, config={key: value}):
                pass
        except (AttributeError, KeyError, TypeError):
            if key not in _FP8_VECMAT_PATH_C_FILTERED_KEYS_LOGGED:
                _FP8_VECMAT_PATH_C_FILTERED_KEYS_LOGGED.add(key)
                print(
                    f"[cppmega-mlx-cuz] dropping unsupported PassConfig "
                    f"key {key!r} from fp8_vecmat_path_c lowering "
                    f"(not registered in active libtilelang).",
                    file=sys.stderr,
                )
            continue
        supported[key] = value
    return supported


def _fp8_vecmat_pass_configs() -> dict[str, Any]:
    """Return the PassConfig dict to thread through this kernel's lowering.

    The env var ``CPPMEGA_FP8_VECMAT_PATH_C_NO_Z3`` forces the legacy
    (no-PassContext) lowering for parity tests / debug.
    """

    if os.environ.get("CPPMEGA_FP8_VECMAT_PATH_C_NO_Z3", "0") not in (
        "0",
        "",
        "false",
        "False",
    ):
        return {}
    global _FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE
    with _FP8_VECMAT_PATH_C_FIRST_RUN_LOCK:
        if _FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE is None:
            _FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE = _filter_supported_pass_configs(
                _FP8_VECMAT_PATH_C_CANDIDATE_PASS_CONFIGS
            )
        return dict(_FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE)


# TileLang resolves these globals while decorating the nested @T.prim_func.
# Defaults keep static tooling aligned with the runtime-specialized contract.
_FP8_VM_N = 128
_FP8_VM_K = 128
_FP8_VM_NP = 2
_FP8_VM_RT = 32
_FP8_VM_VEC = 4
_FP8_VM_BLOCK_K = _FP8_VM_RT * _FP8_VM_VEC
_FP8_VM_K_WORDS = _FP8_VM_K // 4
_FP8_VM_SW = _FP8_VM_N
_FP8_VM_C_DTYPE = "float32"


@dataclass(frozen=True)
class FP8VecmatPathCStatus:
    """Runtime/lowering status for the Path C TileLang FP8 vecmat kernel."""

    available: bool
    reason: str
    target: str = TILELANG_METAL_VECMAT_TARGET
    transpose_B: bool = True
    m_equals_1: bool = True


class FP8VecmatPathCDirectError(RuntimeError):
    """Raised when the owner-output tvm-ffi vecmat path cannot run safely."""


class FP8VecmatPathCLegacyError(RuntimeError):
    """Raised when callers request the retired no-out allocation path."""


def _raise_owner_output_required(op_name: str) -> None:
    raise FP8VecmatPathCLegacyError(
        f"{op_name}: no-out Path C dispatch is retired. The only supported "
        "FP8 Path C vecmat route is tvm-ffi owner-output dispatch; pass "
        "out=existing_mx_array. The old mx.fast.metal_kernel allocation "
        f"path is not re-enabled by {FP8_PATH_C_LEGACY_MLX_FAST_ENV} because "
        "it would allocate an output outside the caller-owned buffer contract."
    )


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - hosts without TileLang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


_FP8_VECMAT_PATH_C_STATUS_UNAVAILABLE_LOGGED: set[str] = set()


_FP8_VECMAT_PATH_C_VECTORIZED_PROBE_LOGGED = False


def _warn_vectorized_loads_probe() -> None:
    """One-shot stderr warning when the ``vectorized_loads=True`` probe runs.

    The vectorized-loads PrimFunc branch is an experimental probe — on the
    current apple-head Metal lowering it does not reliably emit packed
    uint32 MSL loads, so production callers should leave the default
    ``vectorized_loads=False`` and ride the packed dot4 macro fast path.
    Kept for receipts; logged once per process so the off-canonical
    selection is observable.
    """

    global _FP8_VECMAT_PATH_C_VECTORIZED_PROBE_LOGGED
    if _FP8_VECMAT_PATH_C_VECTORIZED_PROBE_LOGGED:
        return
    _FP8_VECMAT_PATH_C_VECTORIZED_PROBE_LOGGED = True
    warnings.warn(
        "fp8_vecmat_path_c: vectorized_loads=True is an experimental probe "
        "and does not reliably emit packed uint32 MSL loads on the current "
        "apple-head Metal lowering; the canonical fast path is the packed "
        "dot4 macro (vectorized_loads=False). Use at own risk.",
        RuntimeWarning,
        stacklevel=3,
    )
    print(
        "[cppmega-mlx-cuz] fp8_vecmat_path_c: vectorized_loads=True probe "
        "engaged (off canonical fast path).",
        file=sys.stderr,
    )


def _ensure_path_c_metal_fp8_intrinsics_registered() -> None:
    """Cached wrapper around ``_assert_path_c_metal_fp8_intrinsics_registered``.

    The underlying scan iterates the Metal FP8 intrinsic table and probes
    ``Op.get`` for each name, which is a tiny but non-zero cost paid on
    every macro-path kernel build. Cache the *successful* outcome so
    subsequent builds (same process) short-circuit. A failure is *not*
    cached: if intrinsics are temporarily missing during a hot-reload
    we want the next build to retry rather than raise stale.
    """

    global _FP8_VECMAT_PATH_C_INTRINSICS_CHECKED
    if _FP8_VECMAT_PATH_C_INTRINSICS_CHECKED:
        return
    with _FP8_VECMAT_PATH_C_FIRST_RUN_LOCK:
        if _FP8_VECMAT_PATH_C_INTRINSICS_CHECKED:
            return
        _assert_path_c_metal_fp8_intrinsics_registered()
        _FP8_VECMAT_PATH_C_INTRINSICS_CHECKED = True


_FP8_VECMAT_PATH_C_SIMPLIFY_FAILURE_LOGGED: set[str] = set()


def _warn_apply_simplify_failed(exc: BaseException) -> None:
    """One-shot RuntimeWarning when ``apply_simplify`` raises.

    Keyed by ``(type, str)`` so a recurring failure logs once but a new
    failure mode logs again. Keeps the fallback path silent on the data
    plane while still surfacing the regression to anyone listening for
    warnings.
    """

    key = f"{type(exc).__name__}: {exc}"
    if key in _FP8_VECMAT_PATH_C_SIMPLIFY_FAILURE_LOGGED:
        return
    _FP8_VECMAT_PATH_C_SIMPLIFY_FAILURE_LOGGED.add(key)
    warnings.warn(
        f"fp8_vecmat_path_c: tilelang.transform.simplify.apply_simplify "
        f"failed ({key}); falling back to un-simplified PrimFunc. "
        "Lowering will continue but may emit slower MSL.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_path_c_unavailable(reason: str) -> None:
    """One-shot RuntimeWarning when the Path C status surface is unavailable.

    Used by ``fp8_vecmat_path_c_status`` so callers see why Path C cannot
    compile instead of getting a quiet ``available=False``. De-duplicated by
    reason string to avoid log spam in tight loops.
    """

    if reason in _FP8_VECMAT_PATH_C_STATUS_UNAVAILABLE_LOGGED:
        return
    _FP8_VECMAT_PATH_C_STATUS_UNAVAILABLE_LOGGED.add(reason)
    warnings.warn(
        f"fp8_scaled_vecmat_path_c: Path C unavailable ({reason}); "
        "caller should fall back to Path B.",
        RuntimeWarning,
        stacklevel=3,
    )


def fp8_vecmat_path_c_status() -> FP8VecmatPathCStatus:
    """Return whether the tvm-ffi owner-output vecmat route can be built."""

    if not can_run_metal():
        return FP8VecmatPathCStatus(available=False, reason="MLX Metal unavailable")
    ok, reason = _tilelang_available()
    if not ok:
        # grok correctness P1 (silent failure): emit a one-shot RuntimeWarning
        # carrying the actual reason so callers polling status don't get a
        # quiet ``available=False`` with no breadcrumb.
        _warn_path_c_unavailable(f"tilelang unavailable: {reason}")
        return FP8VecmatPathCStatus(available=False, reason=reason)
    return FP8VecmatPathCStatus(
        available=True,
        reason="FP8 vecmat Path C tvm-ffi owner-output route is dispatchable",
    )


def _validate_shape(
    *, N: int, K: int, outputs_per_block: int, reduce_threads: int, vec: int
) -> None:
    if N <= 0 or K <= 0:
        raise ValueError(f"N and K must be positive; got N={N}, K={K}")
    if outputs_per_block <= 0:
        raise ValueError(f"outputs_per_block must be positive; got {outputs_per_block}")
    if reduce_threads <= 0:
        raise ValueError(f"reduce_threads must be positive; got {reduce_threads}")
    if vec <= 0:
        raise ValueError(f"vec must be positive; got {vec}")


def make_fp8_vecmat_reduce_kernel(
    *,
    N: int,
    K: int,
    outputs_per_block: int = 2,
    reduce_threads: int = 32,
    vec: int = 4,
    vectorized_loads: bool = False,
    scale_w_per_row: bool = True,
) -> Any:
    """Build a shape-specialized FP8 vecmat reducer.

    Inputs match Path B's vecmat contract:

    * ``A`` is ``(1, K)`` e4m3.
    * ``B`` is ``(N, K)`` e4m3, i.e. already transposed.
    * ``C`` is flat ``(N,)`` fp32.

    The default fast path maps two output rows onto two SIMD groups inside
    one 64-thread Metal threadgroup. This is the fastest sync-measured MLX
    launch geometry for the 4096x4096 vecmat profile on the current Metal
    backend. Each SIMD group computes one packed FP8
    dot4 reduction, applies ``A_scale * B_scale`` after ``simd_sum``, then lane
    zero writes one row.

    ``vectorized_loads=True`` mirrors upstream TileLang GEMV examples by
    staging a small local vector with ``T.vectorized(vec)``. On current
    apple-head Metal lowering this is a probe, not a guarantee of packed
    uint32 MSL loads.
    """

    _validate_shape(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )

    import tilelang.language as T

    T = cast(Any, T)

    block_k = reduce_threads * vec
    g = globals()
    g.update(
        _FP8_VM_N=N,
        _FP8_VM_K=K,
        _FP8_VM_NP=outputs_per_block,
        _FP8_VM_RT=reduce_threads,
        _FP8_VM_VEC=vec,
        _FP8_VM_BLOCK_K=block_k,
        _FP8_VM_K_WORDS=K // 4,
        _FP8_VM_SW=N if scale_w_per_row else 1,
    )

    if vectorized_loads:
        # Note: experimental probe — use at own risk.
        # grok performance P2 / design: this branch mirrors upstream TileLang
        # GEMV examples by staging a small local vector with
        # ``T.vectorized(vec)``, but on the current apple-head Metal lowering
        # it does NOT reliably emit packed uint32 MSL loads — the fast path
        # is the ``_uses_fp8_dot4_packed_macro`` branch below. Kept here for
        # receipts (so a future TileLang upgrade that wires vectorized FP8
        # loads can be A/B'd) and per repo policy (no silent delete of dead
        # code; investigate intent and close the debt properly). Emit a
        # one-shot warning so anyone enabling this in production sees that
        # they are off the canonical path.
        _warn_vectorized_loads_probe()

        @T.prim_func
        def fp8_vecmat_reduce(
            A: T.Tensor((1, _FP8_VM_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((_FP8_VM_N, _FP8_VM_K), "float8_e4m3"),
            B_scale: T.Tensor((_FP8_VM_SW,), "float32"),
            C: T.Tensor((_FP8_VM_N,), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_FP8_VM_N, _FP8_VM_NP),
                threads=_FP8_VM_RT * _FP8_VM_NP,
            ) as bx:
                A_local = T.alloc_local((_FP8_VM_VEC,), "float8_e4m3")
                B_local = T.alloc_local((_FP8_VM_VEC,), "float8_e4m3")
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                lane = T.get_thread_binding(0)
                kr = T.floormod(lane, _FP8_VM_RT)
                ni = T.floordiv(lane, _FP8_VM_RT)
                col = bx * _FP8_VM_NP + ni
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_FP8_VM_K, _FP8_VM_BLOCK_K)):
                    for v in T.vectorized(_FP8_VM_VEC):
                        k = ko * _FP8_VM_BLOCK_K + kr * _FP8_VM_VEC + v
                        if col < _FP8_VM_N and k < _FP8_VM_K:
                            A_local[v] = A[0, k]
                            B_local[v] = B[col, k]
                    for v in T.serial(_FP8_VM_VEC):
                        accum[0] += T.cast(A_local[v], "float32") * T.cast(
                            B_local[v], "float32"
                        )
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accum[0],
                            True,
                            reduced[0],
                            kr,
                            dtype="handle",
                        )
                    )
                if kr == 0 and col < _FP8_VM_N:
                    if _FP8_VM_SW == 1:
                        C[col] = reduced[0] * A_scale[0] * B_scale[0]
                    else:
                        C[col] = reduced[0] * A_scale[0] * B_scale[col]

    elif _uses_fp8_dot4_packed_macro(vec=vec, K=K):
        # Fix-1 + Fix-A re-application: ensure the Path C Metal FP8 ops
        # (notably ``tirx.metal.fp8_e4m3_dot4``) are registered before we
        # parse the macro PrimFunc. Without this we get an opaque FFI
        # ``AttributeError`` deep in the lowering pipeline; with it we get
        # a clear ``RuntimeError`` naming the missing intrinsic.
        # grok design P2: cache the result so we run the registration
        # scan once per process instead of on every kernel build. Failure
        # still raises (the intrinsic is required for correctness); only
        # the *successful* check is cached.
        _ensure_path_c_metal_fp8_intrinsics_registered()

        @T.prim_func
        def fp8_vecmat_reduce(
            A: T.Tensor((1, _FP8_VM_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((_FP8_VM_N, _FP8_VM_K), "float8_e4m3"),
            B_scale: T.Tensor((_FP8_VM_SW,), "float32"),
            C: T.Tensor((1, _FP8_VM_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_FP8_VM_N, _FP8_VM_NP), threads=_FP8_VM_RT * _FP8_VM_NP
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                lane = T.get_thread_binding(0)
                kr = T.floormod(lane, _FP8_VM_RT)
                ni = T.floordiv(lane, _FP8_VM_RT)
                col = bx * _FP8_VM_NP + ni
                T.clear(accum)
                for ko in T.unroll(
                    0,
                    T.ceildiv(_FP8_VM_K_WORDS, _FP8_VM_RT),
                    explicit=False,
                    unroll_factor=4,
                ):
                    i = ko * _FP8_VM_RT + kr
                    if col < _FP8_VM_N and i < _FP8_VM_K_WORDS:
                        accum[0] += T.metal_fp8_e4m3_dot4(
                            T.access_ptr(A[0, 0], "r", extent=_FP8_VM_K),
                            T.access_ptr(B[col, 0], "r", extent=_FP8_VM_K),
                            i,
                            i,
                        )
                reduced = T.call_intrin("float32", "tir.metal.simd_sum", accum[0])
                if kr == 0 and col < _FP8_VM_N:
                    if _FP8_VM_SW == 1:
                        C[0, col] = reduced * A_scale[0] * B_scale[0]
                    else:
                        C[0, col] = reduced * A_scale[0] * B_scale[col]

    else:

        @T.prim_func
        def fp8_vecmat_reduce(
            A: T.Tensor((1, _FP8_VM_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((_FP8_VM_N, _FP8_VM_K), "float8_e4m3"),
            B_scale: T.Tensor((_FP8_VM_SW,), "float32"),
            C: T.Tensor((_FP8_VM_N,), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_FP8_VM_N, _FP8_VM_NP),
                threads=_FP8_VM_RT * _FP8_VM_NP,
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                lane = T.get_thread_binding(0)
                kr = T.floormod(lane, _FP8_VM_RT)
                ni = T.floordiv(lane, _FP8_VM_RT)
                col = bx * _FP8_VM_NP + ni
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_FP8_VM_K, _FP8_VM_BLOCK_K)):
                    for v in T.serial(_FP8_VM_VEC):
                        k = ko * _FP8_VM_BLOCK_K + kr * _FP8_VM_VEC + v
                        if col < _FP8_VM_N and k < _FP8_VM_K:
                            accum[0] += T.cast(A[0, k], "float32") * T.cast(
                                B[col, k], "float32"
                            )
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accum[0],
                            True,
                            reduced[0],
                            kr,
                            dtype="handle",
                        )
                    )
                if kr == 0 and col < _FP8_VM_N:
                    if _FP8_VM_SW == 1:
                        C[col] = reduced[0] * A_scale[0] * B_scale[0]
                    else:
                        C[col] = reduced[0] * A_scale[0] * B_scale[col]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_vecmat_reduce)
    except Exception as exc:
        # grok correctness P2: apply_simplify failure used to fall through
        # silently, so a regression in TileLang's simplify pass would just
        # quietly hand back un-simplified IR (slower or wrong codegen).
        # Surface a one-shot RuntimeWarning naming the exception, but keep
        # the fallback so we don't break lowering on TileLang versions
        # missing the pass.
        _warn_apply_simplify_failed(exc)
        return fp8_vecmat_reduce


def _uses_fp8_dot4_packed_macro(*, vec: int, K: int) -> bool:
    """Decide whether to emit the packed FP8 dot4 PrimFunc branch.

    CPPMEGA Z3 idea #10 wiring (beads cppmega-mlx-cuz):

    The legality predicate ``K % 4 == 0`` here is *load-bearing* because the
    packed branch calls ``T.metal_fp8_e4m3_dot4`` directly with packed
    uint32 (4 bytes per word) — it cannot run on unaligned K. (This is in
    contrast to ``T.fp8_scaled_matmul``, where TileLang's
    ``_z3_prove_dot4_legal_for_buffers`` already discharges the same
    predicate and falls back to a legacy macro on UNKNOWN.) We therefore
    keep the runtime check, but route it through this single named helper
    so that:
      * Future regressions are easy to grep for.
      * A debug-only env var (``CPPMEGA_FP8_VECMAT_PATH_C_DEBUG``) raises
        loudly instead of silently routing to the scalar fallback when a
        prover-discharged path becomes available upstream.
      * The PassContext that ``lower_fp8_vecmat_msl`` threads through still
        wraps the lowering, so current Z3-driven bound-check proofs and future
        registered rewrites can update this branch's IR with no Python-side
        change.
    """

    structural_match = vec == 4
    if not structural_match:
        return False
    if K <= 0:
        return False
    k_aligned = K % 4 == 0
    if not k_aligned:
        debug = os.environ.get("CPPMEGA_FP8_VECMAT_PATH_C_DEBUG", "0")
        if debug not in ("0", "", "false", "False"):
            # Debug-only assertion: surfaces the dot4 legality contract
            # violation loudly when a developer forces this path with
            # unaligned K. The public-API ``_normalize_vecmat_inputs``
            # already rejects unaligned K, so the only way to reach this
            # branch is by direct ``make_fp8_vecmat_reduce_kernel`` calls.
            raise AssertionError(
                f"fp8_vecmat_path_c: packed dot4 branch expects K % 4 == 0 "
                f"(Z3 idea #10 legality), got K={K}. The scalar PrimFunc "
                f"fallback still handles this correctly; unset "
                f"CPPMEGA_FP8_VECMAT_PATH_C_DEBUG to re-enable that "
                f"silent fallback."
            )
        return False
    return True


def lower_fp8_vecmat_msl(
    *,
    N: int = 4096,
    K: int = 4096,
    outputs_per_block: int = 2,
    reduce_threads: int = 32,
    vec: int = 4,
    vectorized_loads: bool = False,
    scale_w_per_row: bool = True,
    target: str = TILELANG_METAL_VECMAT_TARGET,
) -> str:
    """Lower the Path C vecmat reducer and return the runtime MSL source."""

    prim = make_fp8_vecmat_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        vectorized_loads=vectorized_loads,
        scale_w_per_row=scale_w_per_row,
    )
    pass_configs = _fp8_vecmat_pass_configs()
    lowering = cast(
        _msl_transform.TileLangMSLLowering,
        dispatch_lower(
            prim,
            target=target,
            return_msl=True,
            pass_configs=pass_configs or None,
        ),
    )
    lowering, _output_shape = _fuse_canonical_vecmat_runtime_body(
        lowering,
        N=N,
        K=K,
        reduce_threads=reduce_threads,
        vec=vec,
        scale_w_per_row=scale_w_per_row,
        vectorized_loads=vectorized_loads,
    )
    return lowering.msl_text


def _is_canonical_vecmat_fast_path(
    *,
    reduce_threads: int,
    vec: int,
    K: int,
    vectorized_loads: bool = False,
) -> bool:
    return reduce_threads == 32 and vec == 4 and K % 4 == 0 and not vectorized_loads


def _kernel_body_for_feature_counts(msl: str) -> str:
    try:
        _prelude, _sig_text, body_text = _msl_transform._split_kernel_msl(msl)
    except Exception:
        return msl
    return body_text


def fp8_vecmat_msl_features(msl: str) -> dict[str, int]:
    """Return feature counters used by tests and bench receipts."""

    body = _kernel_body_for_feature_counts(msl)
    body_lowered = body.lower()
    scalar_decode_sites = body.count("__tvm_fp8_e4m3_to_half(")
    packed_uint_loads = body.count("reinterpret_cast<device const uint*>") + body.count(
        "__tvm_fp8_load_u32"
    )
    return {
        "kernel_void": msl.count("kernel void"),
        "fp8_e4m3_decode_helper": body.count("__tvm_fp8_e4m3_to_half"),
        "tvm_thread_allreduce": body.count("tvm_thread_allreduce"),
        "simd_shuffle_down": body.count("simd_shuffle_down"),
        "simd_sum": body.count("simd_sum"),
        "reinterpret_cast": body.count("reinterpret_cast"),
        "device_const_uint": body.count("device const uint"),
        "uint_pointer": body.count("uint*"),
        "uchar4": body_lowered.count("uchar4"),
        "fp8_e4m3_lut": body.count("fp8_e4m3fn_lut"),
        "metal_fp8_dot4_helper": body.count("__tvm_fp8_e4m3_dot4_packed"),
        "packed_uint_loads": packed_uint_loads,
        "scalar_fp8_byte_decode": scalar_decode_sites,
        "scalar_fp8_byte_decode_calls": scalar_decode_sites,
    }


def fp8_vecmat_msl_blockers(msl: str) -> dict[str, Any]:
    """Summarize why the generated Path C MSL still misses Path B's fast path."""

    features = fp8_vecmat_msl_features(msl)
    missing: list[str] = []
    if features["packed_uint_loads"] == 0:
        missing.append("packed_uint32_fp8_loads")
    if features["simd_sum"] == 0:
        missing.append("metal_simd_sum_reduction")
    if features["fp8_e4m3_lut"] == 0 and features["metal_fp8_dot4_helper"] == 0:
        missing.append("packed_lut_dot4_decode")
    if features["scalar_fp8_byte_decode_calls"] > 0:
        missing.append("lut_or_packed_decode_instead_of_scalar_fp8_helper_calls")
    return {
        "path_b_fast_path_ready": not missing,
        "missing": missing,
        "generated_features": features,
        "required_fast_path": {
            "packed_uint32_fp8_loads": "reinterpret_cast<device const uint*> loads for 4 FP8 bytes",
            "metal_simd_sum_reduction": "literal Metal simd_sum(sum) reduction",
            "packed_lut_dot4_decode": "fp8_e4m3fn_lut-backed direct decode or TileLang packed dot4 helper in the hot loop",
            "no_scalar_fp8_helper_calls": "avoid per-byte __tvm_fp8_e4m3_to_half calls in the hot loop",
        },
    }


_FP8_VECMAT_TVM_FFI_KERNEL_CACHE: dict[
    tuple[int, int, int, int, int, bool, str],
    Any,
] = {}
_FP8_VECMAT_TVM_FFI_KERNEL_CACHE_LOCK = threading.RLock()


def _make_fp8_vecmat_direct_kernel(
    *,
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    scale_w_per_row: bool,
    c_dtype: str,
) -> Any:
    """Build the owner-output tvm-ffi vecmat kernel.

    The ABI is intentionally flat:

    * ``A`` is the existing ``(K,)`` MLX uint8/e4m3 buffer.
    * ``B`` is the existing ``(N, K)`` MLX uint8/e4m3 buffer.
    * ``C`` is the caller-owned ``(N,)`` output buffer.

    No reshape, allocation, or Python-side cast is needed before the DLPack
    handoff.
    """

    _validate_shape(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )
    if not _uses_fp8_dot4_packed_macro(vec=vec, K=K):
        raise FP8VecmatPathCDirectError(
            "direct tvm-ffi FP8 vecmat requires vec=4 and K multiple of 4 "
            "so the packed e4m3 dot4 ABI is legal"
        )
    _ensure_path_c_metal_fp8_intrinsics_registered()

    import tilelang.language as T

    T = cast(Any, T)
    g = globals()
    g.update(
        _FP8_VM_N=int(N),
        _FP8_VM_K=int(K),
        _FP8_VM_NP=int(outputs_per_block),
        _FP8_VM_RT=int(reduce_threads),
        _FP8_VM_VEC=int(vec),
        _FP8_VM_BLOCK_K=int(reduce_threads) * int(vec),
        _FP8_VM_K_WORDS=int(K) // 4,
        _FP8_VM_SW=int(N) if scale_w_per_row else 1,
        _FP8_VM_C_DTYPE=str(c_dtype),
    )

    @T.prim_func
    def fp8_vecmat_reduce_direct(
        A: T.Tensor((_FP8_VM_K,), "float8_e4m3"),
        A_scale: T.Tensor((1,), "float32"),
        B: T.Tensor((_FP8_VM_N, _FP8_VM_K), "float8_e4m3"),
        B_scale: T.Tensor((_FP8_VM_SW,), "float32"),
        C: T.Tensor((_FP8_VM_N,), _FP8_VM_C_DTYPE),
    ):
        with T.Kernel(
            T.ceildiv(_FP8_VM_N, _FP8_VM_NP),
            threads=_FP8_VM_RT * _FP8_VM_NP,
        ) as bx:
            accum = T.alloc_local((1,), "float32")
            lane = T.get_thread_binding(0)
            kr = T.floormod(lane, _FP8_VM_RT)
            ni = T.floordiv(lane, _FP8_VM_RT)
            col = bx * _FP8_VM_NP + ni
            T.clear(accum)
            for ko in T.unroll(
                0,
                T.ceildiv(_FP8_VM_K_WORDS, _FP8_VM_RT),
                explicit=False,
                unroll_factor=4,
            ):
                word_i = ko * _FP8_VM_RT + kr
                if col < _FP8_VM_N and word_i < _FP8_VM_K_WORDS:
                    accum[0] += T.metal_fp8_e4m3_dot4(
                        T.access_ptr(A[0], "r", extent=_FP8_VM_K),
                        T.access_ptr(B[col, 0], "r", extent=_FP8_VM_K),
                        word_i,
                        word_i,
                    )
            reduced = T.call_intrin("float32", "tir.metal.simd_sum", accum[0])
            if kr == 0 and col < _FP8_VM_N:
                if _FP8_VM_SW == 1:
                    C[col] = reduced * A_scale[0] * B_scale[0]
                else:
                    C[col] = reduced * A_scale[0] * B_scale[col]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_vecmat_reduce_direct)
    except Exception as exc:
        _warn_apply_simplify_failed(exc)
        return fp8_vecmat_reduce_direct


def _fp8_vecmat_tvm_ffi_kernel_for(
    *,
    N: int,
    K: int,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    scale_w_per_row: bool,
    c_dtype: str,
) -> Any:
    cache_key = (
        int(N),
        int(K),
        int(outputs_per_block),
        int(reduce_threads),
        int(vec),
        bool(scale_w_per_row),
        str(c_dtype),
    )
    with _FP8_VECMAT_TVM_FFI_KERNEL_CACHE_LOCK:
        cached = _FP8_VECMAT_TVM_FFI_KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    import tilelang

    prim = _make_fp8_vecmat_direct_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        scale_w_per_row=scale_w_per_row,
        c_dtype=c_dtype,
    )
    kernel = tilelang.compile(
        prim,
        target=_msl_transform._as_metal_target(TILELANG_METAL_VECMAT_TARGET),
        execution_backend="tvm_ffi",
        out_idx=-1,
        pass_configs=_fp8_vecmat_pass_configs() or None,
    )
    with _FP8_VECMAT_TVM_FFI_KERNEL_CACHE_LOCK:
        _FP8_VECMAT_TVM_FFI_KERNEL_CACHE[cache_key] = kernel
    return kernel


def canonical_vecmat_runtime_body(*, N: int, K: int, scale_w_per_row: bool) -> str:
    """Path-C runtime body canonicalized to one SIMD-group per output row."""

    k_words = K // 4
    scale_w_expr = "B_scale[row]" if scale_w_per_row else "B_scale[0]"
    # Keep the hot loop structurally aligned with Path B, but specialize known
    # dimensions so Path C does not pay dynamic shape loads in the kernel.
    return f"""
    uint gid = thread_position_in_grid.x;
    uint simd_lane = thread_index_in_simdgroup;
    uint row = gid / 32u;
    if (row >= {N}u) return;

    uint row_offset = row * {K}u;
    float sum = 0.0f;

    device const uint* A4 = reinterpret_cast<device const uint*>(A);
    device const uint* B4 = reinterpret_cast<device const uint*>(B + row_offset);
    const uint K4 = {k_words}u;
    for (uint i = simd_lane; i < K4; i += 32u) {{
        uint px = A4[i];
        uint pw = B4[i];
        sum += fp8_e4m3fn_lut[px & 0xFFu]          * fp8_e4m3fn_lut[pw & 0xFFu]
             + fp8_e4m3fn_lut[(px >> 8) & 0xFFu]   * fp8_e4m3fn_lut[(pw >> 8) & 0xFFu]
             + fp8_e4m3fn_lut[(px >> 16) & 0xFFu]  * fp8_e4m3fn_lut[(pw >> 16) & 0xFFu]
             + fp8_e4m3fn_lut[(px >> 24) & 0xFFu]  * fp8_e4m3fn_lut[(pw >> 24) & 0xFFu];
    }}

    sum = simd_sum(sum);

    if (simd_lane == 0u) {{
        float sx = float(A_scale[0]);
        float sw = float({scale_w_expr});
        C[row] = sum * sx * sw;
    }}
"""


def _fuse_canonical_vecmat_runtime_body(
    lowering: _msl_transform.TileLangMSLLowering,
    *,
    N: int,
    K: int,
    reduce_threads: int,
    vec: int,
    scale_w_per_row: bool,
    vectorized_loads: bool,
) -> tuple[_msl_transform.TileLangMSLLowering, tuple[int, ...]]:
    """Fuse the canonical Path-B hot loop into Path-C's runtime lowering.

    TileLang still owns the ABI/signature/header and launch metadata. For the
    canonical M=1 FP8 vecmat case, replacing only the body removes an avoidable
    block/thread indexing layer and the dot4 helper trampoline while keeping
    the same prepared GPU buffers.
    """

    if not _is_canonical_vecmat_fast_path(
        reduce_threads=reduce_threads,
        vec=vec,
        K=K,
        vectorized_loads=vectorized_loads,
    ):
        return lowering, (N,)

    body = canonical_vecmat_runtime_body(
        N=N,
        K=K,
        scale_w_per_row=scale_w_per_row,
    )
    from cppmega_mlx.nn._tilelang.fp8_msl_kernels import _FP8_HEADER

    _prelude, sig_text, _body_text = _msl_transform._split_kernel_msl(lowering.msl_text)
    kernel_name = _msl_transform._KERNEL_DEF_RE.search(
        _msl_transform._mask_msl_comments_and_strings(lowering.msl_text)
    )
    if kernel_name is None:
        return replace(lowering, header=_FP8_HEADER, body=body), (N,)
    fused_msl = (
        f"{_FP8_HEADER}\n\n"
        f"kernel void {kernel_name.group('name')}({sig_text}) {{\n{body}\n}}\n"
    )
    return replace(lowering, header=_FP8_HEADER, body=body, msl_text=fused_msl), (N,)


def _tilelang_output_dtype_for_mlx(dtype: Any, *, op_name: str) -> str:
    if dtype == mx.float32:
        return "float32"
    if dtype == mx.float16:
        return "float16"
    mx_bfloat16 = getattr(mx, "bfloat16", None)
    if mx_bfloat16 is not None and dtype == mx_bfloat16:
        raise ValueError(
            f"{op_name}: mx.bfloat16 owner-output is not supported by the "
            "current TileLang Metal ABI because codegen emits MSL `bfloat`; "
            "use mx.float32/mx.float16 or fix TileLang CodeGenMetal first"
        )
    raise ValueError(
        f"{op_name}: out dtype must be mx.float32 or mx.float16; got {dtype}"
    )


def _resolve_vecmat_scale_direct(
    scale: mx.array | float,
    *,
    length: int,
    name: str,
    scalar_only: bool = False,
) -> mx.array:
    """Validate a direct-route scale tensor without allocating/casting."""

    if isinstance(scale, (int, float)):
        raise TypeError(
            f"fp8_scaled_vecmat_path_c direct owner-output route requires {name} "
            "as an existing mx.float32 tensor; Python scalars would allocate "
            "a new MLX tensor at the wrapper boundary"
        )
    if scale.ndim != 1 or scale.dtype != mx.float32:
        raise ValueError(
            f"fp8_scaled_vecmat_path_c direct owner-output route expects {name} "
            f"as 1D mx.float32; got shape={tuple(scale.shape)} dtype={scale.dtype}"
        )
    if scale.size == 1:
        return scale
    if not scalar_only and scale.size == length:
        return scale
    expected = "1" if scalar_only else f"1 or {length}"
    raise ValueError(
        f"fp8_scaled_vecmat_path_c direct owner-output route expected {name} "
        f"size {expected}; got shape {tuple(scale.shape)}"
    )


def _normalize_vecmat_inputs_direct(
    x_fp8: mx.array,
    W_fp8: mx.array,
    scale_x: mx.array | float,
    scale_w: mx.array | float,
) -> tuple[mx.array, mx.array, mx.array, mx.array, int, int, bool]:
    if x_fp8.ndim != 1 or W_fp8.ndim != 2:
        raise ValueError(
            f"fp8_scaled_vecmat_path_c direct owner-output route expects 1D x "
            f"and 2D W; got x.ndim={x_fp8.ndim}, W.ndim={W_fp8.ndim}"
        )
    if x_fp8.dtype != mx.uint8 or W_fp8.dtype != mx.uint8:
        raise ValueError(
            "fp8_scaled_vecmat_path_c direct owner-output route expects "
            f"mx.uint8 e4m3 storage; got {x_fp8.dtype}, {W_fp8.dtype}"
        )
    (k,) = x_fp8.shape
    n, k_w = W_fp8.shape
    if k != k_w:
        raise ValueError(
            f"fp8_scaled_vecmat_path_c direct owner-output route shape mismatch: "
            f"x=(K={k}), W=(N={n}, K={k_w})"
        )
    if k % 4 != 0:
        raise ValueError(
            f"fp8_scaled_vecmat_path_c direct owner-output route requires "
            f"K multiple of 4; got K={k}"
        )
    scale_x_arr = _resolve_vecmat_scale_direct(
        scale_x,
        length=1,
        name="scale_x",
        scalar_only=True,
    )
    scale_w_arr = _resolve_vecmat_scale_direct(
        scale_w,
        length=n,
        name="scale_w",
    )
    return x_fp8, scale_x_arr, W_fp8, scale_w_arr, int(n), int(k), scale_w_arr.size == n


def _validate_vecmat_owner_output(out: mx.array, *, n: int) -> tuple[mx.array, str]:
    if not isinstance(out, mx.array):
        raise TypeError(
            f"fp8_scaled_vecmat_path_c: out must be an mlx.core.array; "
            f"got {type(out).__name__}"
        )
    if out.shape != (n,):
        raise ValueError(
            f"fp8_scaled_vecmat_path_c: out shape must be ({n},); "
            f"got {tuple(out.shape)}"
        )
    return out, _tilelang_output_dtype_for_mlx(
        out.dtype,
        op_name="fp8_scaled_vecmat_path_c",
    )


def fp8_scaled_vecmat_path_c_direct(
    x_fp8: mx.array,
    W_fp8: mx.array,
    *,
    scale_x: mx.array | float,
    scale_w: mx.array | float,
    out: mx.array,
    outputs_per_block: int = 2,
    reduce_threads: int = 32,
    vec: int = 4,
) -> mx.array:
    """Run FP8 vecmat Path C through tvm-ffi into caller-owned ``out``."""

    if not can_run_metal():
        raise FP8VecmatPathCDirectError("MLX Metal unavailable")
    A, A_scale, B, B_scale, n, k, scale_w_per_row = _normalize_vecmat_inputs_direct(
        x_fp8,
        W_fp8,
        scale_x,
        scale_w,
    )
    C, c_dtype = _validate_vecmat_owner_output(out, n=n)
    try:
        kernel = _fp8_vecmat_tvm_ffi_kernel_for(
            N=n,
            K=k,
            outputs_per_block=int(outputs_per_block),
            reduce_threads=int(reduce_threads),
            vec=int(vec),
            scale_w_per_row=scale_w_per_row,
            c_dtype=c_dtype,
        )
    except Exception as exc:
        raise FP8VecmatPathCDirectError(
            f"direct tvm-ffi FP8 vecmat compile failed: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        returned = kernel(A, A_scale, B, B_scale, C)
    except Exception as exc:
        try:
            from tilelang.contrib.mlx_interop import DLPackInteropError
        except Exception:  # pragma: no cover - only when TileLang import itself is broken
            DLPackInteropError = ()  # type: ignore[assignment]
        if isinstance(exc, DLPackInteropError):
            raise
        raise FP8VecmatPathCDirectError(
            f"direct tvm-ffi FP8 vecmat dispatch failed: {type(exc).__name__}: {exc}"
        ) from exc
    if returned is not C:
        raise FP8VecmatPathCDirectError(
            "direct tvm-ffi FP8 vecmat did not return the caller-owned output"
        )
    return C


def fp8_scaled_vecmat_path_c(
    x_fp8: mx.array,
    W_fp8: mx.array,
    *,
    scale_x: mx.array | float,
    scale_w: mx.array | float,
    outputs_per_block: int = 2,
    reduce_threads: int = 32,
    vec: int = 4,
    out: mx.array | None = None,
) -> mx.array | None:
    """Run Path C TileLang FP8 vecmat through MLX Metal.

    ``x_fp8`` is ``(K,)`` uint8 e4m3 storage and ``W_fp8`` is transposed
    ``(N, K)`` storage, matching Path B. ``scale_x`` is scalar; ``scale_w`` may
    be scalar or per-output ``(N,)``. When ``out`` is provided, dispatches via
    tvm-ffi into that caller-owned output and returns the same object. Without
    ``out``, this function fails explicitly: there is no non-owner-output Path C
    dispatch surface.
    """

    if out is not None:
        return fp8_scaled_vecmat_path_c_direct(
            x_fp8,
            W_fp8,
            scale_x=scale_x,
            scale_w=scale_w,
            out=out,
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
        )

    _raise_owner_output_required("fp8_scaled_vecmat_path_c")


__all__ = [
    "FP8VecmatPathCDirectError",
    "FP8VecmatPathCLegacyError",
    "FP8VecmatPathCStatus",
    "FP8_PATH_C_LEGACY_MLX_FAST_ENV",
    "TILELANG_METAL_VECMAT_TARGET",
    "canonical_vecmat_runtime_body",
    "fp8_scaled_vecmat_path_c_direct",
    "fp8_scaled_vecmat_path_c",
    "fp8_vecmat_msl_blockers",
    "fp8_vecmat_msl_features",
    "fp8_vecmat_path_c_status",
    "lower_fp8_vecmat_msl",
    "make_fp8_vecmat_reduce_kernel",
]
