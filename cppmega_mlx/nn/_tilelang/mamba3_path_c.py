"""Path C port of Mamba3 MIMO fwd+bwd via TileLang DSL ``@T.prim_func`` lowering.

This module is the Path C counterpart to :mod:`cppmega_mlx.nn._tilelang.mamba3`
(Path B). Path B writes MSL by hand and dispatches it via
``mx.fast.metal_kernel``. Path C writes the *same* selective-scan kernel as
TileLang DSL and dispatches it through ``tilelang.compile(...,
execution_backend="tvm_ffi", out_idx=...)`` into caller-owned MLX buffers.

Why ship both?
--------------

Path B is the shipped hand-written Metal baseline. Path C is intentionally the
*same algorithm expressed in the high-level DSL* so we can:

1. prove the TileLang Metal backend can lower a non-trivial selective-scan
   kernel end to end (against the patched apple-head TileLang
   ``cppmega/gemm-mixed-dtype-metal``);
2. side-by-side bench the lowered MSL against the hand-written MSL — if Path C
   is within 10 percent of Path B the DSL becomes the maintainable entry point;
3. provide a documentation/reproducibility artifact for the upstream PR thread
   (the lowered MSL is captured at
   ``docs/tilelang_ports/mamba3_path_c_lowered.metal``).

Numerical contract
------------------

The kernels operate on FP32 carriers. Non-FP32 callers fail closed instead of
silently materializing large cast buffers. At FP32 the Path C and Path B kernels
are *bit identical* on the tested shapes. The parity budget retained in tests is
the conservative atol=1e-4 / rtol=1e-3.

Public surface
--------------

* :func:`mamba3_mimo_fwd_path_c` — fwd lane scan returning ``(y, h_last)``.
* :func:`mamba3_mimo_bwd_path_c` — bwd lane scan returning grads w.r.t.
  ``(x, B, C, z, A, dt, D, h0)`` after host-side P-axis reductions.
* :func:`mamba3_mimo_apply_path_c` — convenience fwd surface returning ``y``.
* :func:`mamba3_mimo_apply_with_state_path_c` — returns ``(y, h_last)`` so
  model dispatch does not re-run forward just to assemble the inference cache.
* :func:`mamba3_mimo_path_c_status` — preflight check for the lowered TileLang
  DSL kernel; explicit Path C dispatch fails closed when TileLang cannot lower.

Threadgroup tuning
------------------

The Path B grid uses one thread per (b, h, p) lane with up to 256 threads per
threadgroup, matching the Apple Metal target's 1024-thread / 32 KB-shared
ceilings. Path C uses the same one-thread-per-lane algorithm and the same
256-thread cap; keeping this aligned is part of the Path C >= Path B contract.
The TileLang-lowered scan keeps ``h_state[STATE]`` and backward ``dh[STATE]`` in
per-thread registers, so the entire scan stays per-lane and avoids shared-memory
traffic.

Apple/M4 Max threadgroup limits (from ``tilelang.target.Target("metal")``):
  ``-max_num_threads=256 -max_shared_memory_per_block=32768``

The DSL never exceeds these because the entire scan is per-thread register
work; the only memory traffic is global loads/stores.

PEP-563 caveat
--------------

This module deliberately does *not* use ``from __future__ import annotations``.
TileLang's eager builder reads the inner ``@T.prim_func``'s annotations through
``typing.get_type_hints``, which walks ``__closure__`` to find non-local names
like the dimension constants we close over. When PEP-563 is active the
annotations are strings and ``get_type_hints`` only sees variables that are
also referenced from the function body's own bytecode (the ``co_freevars``
list). The Mamba3 PrimFunc body computes derived strides from ``HEADDIM`` /
``HEADS`` rather than naming ``BATCH`` directly; under PEP-563 that turns into
a ``NameError: BATCH`` from the lowering step. Disabling PEP-563 keeps the
annotations as live ``T.Tensor`` objects whose shape ints were already
captured as Python integers when the inner function was defined, sidestepping
the closure-walk path entirely.
"""

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, cast

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
)
from cppmega_mlx.nn._tilelang.mamba3 import _validate_inputs


_REUSABLE_SCALAR_BINDINGS = frozenset(
    {
        "decay",
        "decay_1",
        "sig_z",
        "silu_z",
        "silu_dz",
        "d_silu",
        "d_y_skipped",
    }
)
_FLOAT_BINDING_RE = re.compile(r"\bfloat (?P<name>[A-Za-z_]\w*) = (?P<expr>.*);")
_FWD_OUTPUT_NAMES = ("y", "h_last")
_FWD_OUTPUT_IDX = (8, 9)
_BWD_OUTPUT_NAMES = (
    "h_steps",
    "dx",
    "dz",
    "dB_partial",
    "dC_partial",
    "dA_partial",
    "ddt_partial",
    "dD_partial",
    "dh0",
)
_BWD_OUTPUT_IDX = (9, 10, 11, 12, 13, 14, 15, 16, 17)

Mamba3FwdOwnerOutputs = tuple[mx.array, mx.array]
Mamba3BwdOwnerOutputs = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]


def _reuse_tilelang_scalar_bindings(body: str) -> str:
    """Reuse scalar bindings that TileLang already emitted in the lowered body."""

    out: list[str] = []
    replacements: list[tuple[str, str]] = []

    for raw_line in body.splitlines():
        line = raw_line
        for expr, name in replacements:
            line = line.replace(expr, name)

        match = _FLOAT_BINDING_RE.search(line)
        if match is not None and match.group("name") in _REUSABLE_SCALAR_BINDINGS:
            expr = match.group("expr").strip()
            if expr and expr != match.group("name"):
                replacements.append((expr, match.group("name")))

        out.append(line)

    suffix = "\n" if body.endswith("\n") else ""
    return "\n".join(out) + suffix


def _source_with_reused_scalar_bindings(
    lowering: _msl_transform.TileLangMSLLowering,
) -> str:
    """Return the full lowered MSL string matching the dispatched source body."""

    prelude, signature, body_text = _msl_transform._split_kernel_msl(lowering.msl_text)
    body = _reuse_tilelang_scalar_bindings(body_text[1:-1])
    return (
        f"{prelude}\n"
        f"kernel void {lowering.kernel_name}({signature}) "
        f"{{{body}}}\n"
    )


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mamba3PathCStatus:
    """Runtime status for the Path C TileLang DSL Mamba3 kernel."""

    available: bool
    reason: str


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        from tilelang.engine.lower import lower as _lower  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - macOS without tilelang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def mamba3_mimo_path_c_status() -> Mamba3PathCStatus:
    """Return whether the Path C TileLang DSL kernel can dispatch on this host."""

    if not can_run_metal():
        return Mamba3PathCStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    ok, reason = _tilelang_available()
    if not ok:
        return Mamba3PathCStatus(available=False, reason=reason)
    try:
        fwd_kernel, fwd_lowering = _fwd_kernel_for(1, 4, 1, 2, 4, return_msl=True)
        bwd_kernel, bwd_lowering = _bwd_kernel_for(1, 4, 1, 2, 4, return_msl=True)
        del fwd_kernel, bwd_kernel
    except Exception as exc:
        return Mamba3PathCStatus(
            available=False,
            reason=f"TileLang/MLX lowering failed for Mamba3 Path C: {type(exc).__name__}: {exc}",
        )
    if "kernel void" not in fwd_lowering.msl_text:
        return Mamba3PathCStatus(False, "lowered Mamba3 Path C fwd source has no kernel")
    if "kernel void" not in bwd_lowering.msl_text:
        return Mamba3PathCStatus(False, "lowered Mamba3 Path C bwd source has no kernel")
    return Mamba3PathCStatus(available=True, reason="Path C TileLang DSL ready")


# ---------------------------------------------------------------------------
# TileLang PrimFunc factories (cached on shape signature)
# ---------------------------------------------------------------------------


def _threads_for(lanes: int) -> int:
    """Return the threadgroup size for a per-lane kernel.

    Keep this identical to Path B. A previous 32-thread cap multiplied the
    number of threadgroups by 8 on the real ``H*P=3584`` shape and made Path C
    lose to the baseline before scheduler-level optimizations even had a chance
    to matter.
    """

    if lanes <= 0:
        return 1
    return min(256, lanes)


@lru_cache(maxsize=128)
def _fwd_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build & cache the Path C TileLang fwd kernel for a given (B, T, H, P, N)."""

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)

    @T.prim_func
    def fwd(
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), "float32"),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), "float32"),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        A: T.Tensor((BATCH, SEQ, HEADS), "float32"),
        dt: T.Tensor((BATCH, SEQ, HEADS), "float32"),
        D: T.Tensor((HEADS,), "float32"),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), "float32"),
        y: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        h_last: T.Tensor((BATCH, HEADS, HEADDIM, STATE), "float32"),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid_in_block = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid_in_block
            # Per-lane state lives in registers (size N).
            h_state = T.alloc_local((STATE,), "float32")
            if global_lane < LANES:
                p = global_lane % HEADDIM
                h = (global_lane // HEADDIM) % HEADS
                b = global_lane // (HEADDIM * HEADS)
                for n in T.serial(STATE):
                    h_state[n] = h0[b, h, p, n]
                for t in T.serial(SEQ):
                    A_val = A[b, t, h]
                    dt_val = dt[b, t, h]
                    decay = T.exp(A_val * dt_val)
                    x_val = x[b, t, h, p]
                    z_val = z[b, t, h, p]
                    y_acc = T.alloc_local((1,), "float32")
                    y_acc[0] = 0.0
                    for n in T.serial(STATE):
                        new_h = decay * h_state[n] + x_val * B[b, t, h, n]
                        h_state[n] = new_h
                        y_acc[0] = y_acc[0] + new_h * C[b, t, h, n]
                    D_h = D[h]
                    y_skipped = y_acc[0] + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    y[b, t, h, p] = z_val * sig_z * y_skipped
                for n in T.serial(STATE):
                    h_last[b, h, p, n] = h_state[n]

    artifact = dispatch_lower(fwd, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _FWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C fwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        fwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_FWD_OUTPUT_IDX),
    )
    return kernel, lowering


@lru_cache(maxsize=128)
def _bwd_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering]:
    """Build & cache the Path C TileLang bwd kernel for a given (B, T, H, P, N).

    Mirrors the Path B MSL bwd kernel: rematerialises ``h[t]`` per lane on the
    forward pass (writing into ``h_steps``) then walks the reverse pass to
    accumulate per-lane partials. The host reduces partials to the final
    gradient shapes (no atomics needed).
    """

    import tilelang.language as T

    LANES = BATCH * HEADS * HEADDIM
    THREADS = _threads_for(LANES)

    @T.prim_func
    def bwd(
        dy: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        x: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        B: T.Tensor((BATCH, SEQ, HEADS, STATE), "float32"),
        C: T.Tensor((BATCH, SEQ, HEADS, STATE), "float32"),
        z: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        A: T.Tensor((BATCH, SEQ, HEADS), "float32"),
        dt: T.Tensor((BATCH, SEQ, HEADS), "float32"),
        D: T.Tensor((HEADS,), "float32"),
        h0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), "float32"),
        h_steps: T.Tensor((BATCH, HEADS, HEADDIM, SEQ, STATE), "float32"),
        dx: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        dz: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        dB_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM, STATE), "float32"),
        dC_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM, STATE), "float32"),
        dA_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        ddt_partial: T.Tensor((BATCH, SEQ, HEADS, HEADDIM), "float32"),
        dD_partial: T.Tensor((BATCH, HEADS, HEADDIM), "float32"),
        dh0: T.Tensor((BATCH, HEADS, HEADDIM, STATE), "float32"),
    ):
        with T.Kernel(T.ceildiv(LANES, THREADS), threads=THREADS) as bx:
            tid = T.get_thread_binding(0)
            global_lane = bx * THREADS + tid
            h_state = T.alloc_local((STATE,), "float32")
            dh = T.alloc_local((STATE,), "float32")
            dD_acc = T.alloc_local((1,), "float32")
            if global_lane < LANES:
                p = global_lane % HEADDIM
                h = (global_lane // HEADDIM) % HEADS
                b = global_lane // (HEADDIM * HEADS)

                # Forward pass: rematerialise h[t] for this lane into h_steps.
                for n in T.serial(STATE):
                    h_state[n] = h0[b, h, p, n]
                for t in T.serial(SEQ):
                    A_val = A[b, t, h]
                    dt_val = dt[b, t, h]
                    decay = T.exp(A_val * dt_val)
                    x_val = x[b, t, h, p]
                    for n in T.serial(STATE):
                        new_h = decay * h_state[n] + x_val * B[b, t, h, n]
                        h_state[n] = new_h
                        h_steps[b, h, p, t, n] = new_h

                # Reverse pass.
                for n in T.serial(STATE):
                    dh[n] = 0.0
                dD_acc[0] = 0.0
                D_h = D[h]
                for r in T.serial(SEQ):
                    t = SEQ - 1 - r
                    A_val = A[b, t, h]
                    dt_val = dt[b, t, h]
                    decay = T.exp(A_val * dt_val)
                    x_val = x[b, t, h, p]
                    z_val = z[b, t, h, p]
                    dY = dy[b, t, h, p]

                    # y_state[t] = sum_n h_steps[t,n] * C[t,n]
                    y_state = T.alloc_local((1,), "float32")
                    y_state[0] = 0.0
                    for n in T.serial(STATE):
                        y_state[0] = y_state[0] + h_steps[b, h, p, t, n] * C[b, t, h, n]
                    y_skipped = y_state[0] + D_h * x_val
                    sig_z = 1.0 / (1.0 + T.exp(-z_val))
                    silu_z = z_val * sig_z
                    silu_dz = sig_z * (1.0 + z_val * (1.0 - sig_z))

                    d_silu = dY * y_skipped
                    d_y_skipped = dY * silu_z

                    dz[b, t, h, p] = d_silu * silu_dz
                    dD_acc[0] = dD_acc[0] + d_y_skipped * x_val

                    # dh accumulates the y_state contribution.
                    for n in T.serial(STATE):
                        dh[n] = dh[n] + d_y_skipped * C[b, t, h, n]

                    # Per-lane partials for B and C; host sums over P later.
                    for n in T.serial(STATE):
                        dC_partial[b, t, h, p, n] = d_y_skipped * h_steps[b, h, p, t, n]
                        dB_partial[b, t, h, p, n] = dh[n] * x_val

                    # dx contribution comes from the input branch + skip.
                    dx_inp = T.alloc_local((1,), "float32")
                    dx_inp[0] = 0.0
                    for n in T.serial(STATE):
                        dx_inp[0] = dx_inp[0] + dh[n] * B[b, t, h, n]
                    dx_skip = d_y_skipped * D_h
                    dx[b, t, h, p] = dx_skip + dx_inp[0]

                    # Decay backward: pulls from h_prev (or h0 at t == 0).
                    d_decay = T.alloc_local((1,), "float32")
                    d_decay[0] = 0.0
                    if t == 0:
                        for n in T.serial(STATE):
                            d_decay[0] = d_decay[0] + dh[n] * h0[b, h, p, n]
                    else:
                        for n in T.serial(STATE):
                            d_decay[0] = d_decay[0] + dh[n] * h_steps[b, h, p, t - 1, n]
                    d_logdecay = d_decay[0] * decay
                    dA_partial[b, t, h, p] = d_logdecay * dt_val
                    ddt_partial[b, t, h, p] = d_logdecay * A_val

                    # Propagate dh through decay for the next reverse step.
                    for n in T.serial(STATE):
                        dh[n] = dh[n] * decay

                # After the backward sweep, dh is dh0 for this lane.
                for n in T.serial(STATE):
                    dh0[b, h, p, n] = dh[n]
                dD_partial[b, h, p] = dD_acc[0]

    artifact = dispatch_lower(bwd, target="metal", return_msl=True)
    if hasattr(artifact, "_tilelang_engine_target"):
        raise MSLDispatchUnsupported("Mamba3 Path C requires TileLang MSL extraction metadata")
    lowering = cast(_msl_transform.TileLangMSLLowering, artifact)
    input_names = [
        name for name in lowering.buffer_param_names if name not in _BWD_OUTPUT_NAMES
    ]
    if set(input_names) != {"dy", "x", "B", "C", "z", "A", "dt", "D", "h0"}:
        raise MSLDispatchUnsupported(
            "unexpected Mamba3 Path C bwd buffer signature: "
            + ", ".join(lowering.buffer_param_names)
        )
    import tilelang

    kernel = tilelang.compile(
        bwd,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=list(_BWD_OUTPUT_IDX),
    )
    return kernel, lowering


# ---------------------------------------------------------------------------
# Public dispatch entry points
# ---------------------------------------------------------------------------


def _require_fp32_no_hidden_casts(op_name: str, *arrays: mx.array) -> None:
    bad = [str(array.dtype) for array in arrays if array.dtype != mx.float32]
    if bad:
        raise RuntimeError(
            f"{op_name} direct tvm-ffi owner-output route supports mx.float32 "
            f"inputs without hidden casts; got non-fp32 dtypes {bad}"
        )


def _require_owner_array(
    op_name: str,
    name: str,
    array: mx.array,
    *,
    shape: tuple[int, ...],
) -> mx.array:
    if not isinstance(array, mx.array):
        raise TypeError(
            f"{op_name}: owner output {name} must be an mlx.core.array; "
            f"got {type(array).__name__}"
        )
    if tuple(array.shape) != shape:
        raise ValueError(
            f"{op_name}: owner output {name} must have shape {shape}; "
            f"got {tuple(array.shape)}"
        )
    if array.dtype != mx.float32:
        raise TypeError(
            f"{op_name}: owner output {name} must be mx.float32; got {array.dtype}"
        )
    return array


def _mamba3_fwd_owner_outputs(
    out: Mamba3FwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
) -> Mamba3FwdOwnerOutputs | None:
    op_name = "mamba3_mimo_fwd_path_c"
    if out is None:
        return None
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(
            f"{op_name}: out must be a (y, h_last) owner-output tuple"
        )
    y, h_last = out
    return (
        _require_owner_array(
            op_name,
            "y",
            y,
            shape=(batch, seq, heads, headdim),
        ),
        _require_owner_array(
            op_name,
            "h_last",
            h_last,
            shape=(batch, heads, headdim, state),
        ),
    )


def _mamba3_bwd_owner_outputs(
    out: Mamba3BwdOwnerOutputs | None,
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
) -> Mamba3BwdOwnerOutputs | None:
    op_name = "mamba3_mimo_bwd_path_c"
    if out is None:
        return None
    if not isinstance(out, tuple) or len(out) != len(_BWD_OUTPUT_NAMES):
        raise TypeError(
            f"{op_name}: out must be a tuple matching {_BWD_OUTPUT_NAMES!r}"
        )
    expected_shapes = (
        (batch, heads, headdim, seq, state),
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim, state),
        (batch, seq, heads, headdim, state),
        (batch, seq, heads, headdim),
        (batch, seq, heads, headdim),
        (batch, heads, headdim),
        (batch, heads, headdim, state),
    )
    return cast(
        Mamba3BwdOwnerOutputs,
        tuple(
            _require_owner_array(op_name, name, array, shape=shape)
            for name, array, shape in zip(
                _BWD_OUTPUT_NAMES,
                out,
                expected_shapes,
                strict=True,
            )
        ),
    )


def _raise_if_dlpack_boundary_failure(op_name: str, exc: Exception) -> None:
    try:
        from tilelang.contrib.mlx_interop import DLPackConversionError
    except Exception:  # pragma: no cover - only when TileLang import itself is broken
        DLPackConversionError = ()  # type: ignore[assignment]
    if isinstance(exc, DLPackConversionError):
        raise RuntimeError(
            f"{op_name} requires DLPack-exportable, contiguous caller-owned MLX "
            "input/output buffers; Path C will not copy, cast, or materialize "
            "broadcast/slice views implicitly. If this fires inside an MLX "
            "graph transform, the producer has to expose a graph-safe DLPack "
            "view or stay in the existing fused graph path."
        ) from exc


def mamba3_mimo_fwd_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    out: Mamba3FwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array]:
    """Path C forward; fail closed when TileLang metadata or dispatch is unavailable."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_fwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    _require_fp32_no_hidden_casts(
        "mamba3_mimo_fwd_path_c",
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
    )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "mamba3_mimo_fwd_path_c owner-output route is not dispatchable "
                "for seq=0; return h0 directly instead of copying it"
            )
        return mx.zeros((batch, 0, heads, headdim), dtype=mx.float32), h0

    try:
        kernel, lowering = _fwd_kernel_for(batch, seq, heads, headdim, state)
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_fwd_path_c lowering failed") from exc

    owner_outputs = _mamba3_fwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
    )
    try:
        if owner_outputs is None:
            out_list = kernel(x, B, C, z, A, dt, D, h0)
        else:
            y, h_last = owner_outputs
            out_list = kernel(
                x,
                B,
                C,
                z,
                A,
                dt,
                D,
                h0,
                out=(y, h_last),
            )
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_fwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_fwd_path_c dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != 2:
        raise RuntimeError("Mamba3 Path C fwd tvm-ffi returned an invalid output tuple")
    if owner_outputs is not None:
        y, h_last = owner_outputs
        if not all(
            got is expected
            for got, expected in zip(out_list, (y, h_last), strict=True)
        ):
            raise RuntimeError(
                "Mamba3 Path C fwd tvm-ffi did not return caller-owned outputs"
            )
    y, h_last = cast(tuple[mx.array, mx.array], tuple(out_list))
    del lowering
    return y, h_last


def _mamba3_mimo_bwd_path_c_kernel(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    out: Mamba3BwdOwnerOutputs | None = None,
) -> tuple[mx.array, ...]:
    """Run the lowered Path C bwd kernel."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        raise RuntimeError(f"mamba3_mimo_bwd_path_c unavailable: {status.reason}")

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    _require_fp32_no_hidden_casts(
        "mamba3_mimo_bwd_path_c",
        dy,
        x,
        B,
        C,
        z,
        A,
        dt,
        D,
        h0,
    )
    if seq == 0:
        if out is not None:
            raise RuntimeError(
                "mamba3_mimo_bwd_path_c owner-output route is not dispatchable "
                "for seq=0 because no TileLang kernel runs to initialize buffers"
            )
        return (
            mx.zeros_like(x),
            mx.zeros_like(B),
            mx.zeros_like(C),
            mx.zeros_like(z),
            mx.zeros_like(A),
            mx.zeros_like(dt),
            mx.zeros_like(D),
            mx.zeros_like(h0),
        )

    try:
        kernel, lowering = _bwd_kernel_for(batch, seq, heads, headdim, state)
    except (MSLDispatchUnsupported, RuntimeError, ValueError) as exc:
        raise RuntimeError("mamba3_mimo_bwd_path_c lowering failed") from exc

    owner_outputs = _mamba3_bwd_owner_outputs(
        out,
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
    )
    try:
        if owner_outputs is None:
            out_list = kernel(dy, x, B, C, z, A, dt, D, h0)
        else:
            (
                h_steps,
                dx,
                dz,
                dB_partial,
                dC_partial,
                dA_partial,
                ddt_partial,
                dD_partial,
                dh0,
            ) = owner_outputs
            out_list = kernel(
                dy,
                x,
                B,
                C,
                z,
                A,
                dt,
                D,
                h0,
                out=(
                    h_steps,
                    dx,
                    dz,
                    dB_partial,
                    dC_partial,
                    dA_partial,
                    ddt_partial,
                    dD_partial,
                    dh0,
                ),
            )
    except Exception as exc:
        _raise_if_dlpack_boundary_failure("mamba3_mimo_bwd_path_c", exc)
        raise RuntimeError("mamba3_mimo_bwd_path_c dispatch failed") from exc

    if not isinstance(out_list, (list, tuple)) or len(out_list) != len(_BWD_OUTPUT_NAMES):
        raise RuntimeError("Mamba3 Path C bwd tvm-ffi returned an invalid output tuple")
    if owner_outputs is not None:
        if not all(
            got is expected
            for got, expected in zip(out_list, owner_outputs, strict=True)
        ):
            raise RuntimeError(
                "Mamba3 Path C bwd tvm-ffi did not return caller-owned outputs"
            )
    _h_scratch, dx_pc, dz_pc, dB_p, dC_p, dA_p, ddt_p, dD_p, dh0_pc = out_list
    del lowering, _h_scratch
    # Reduce P-dim partials into final shapes.
    dB_pc = mx.sum(dB_p, axis=3)         # -> (B, T, H, N)
    dC_pc = mx.sum(dC_p, axis=3)         # -> (B, T, H, N)
    dA_pc = mx.sum(dA_p, axis=3)         # -> (B, T, H)
    ddt_pc = mx.sum(ddt_p, axis=3)       # -> (B, T, H)
    dD_pc = mx.sum(dD_p, axis=(0, 2))    # -> (H,)
    return (
        dx_pc,
        dB_pc,
        dC_pc,
        dz_pc,
        dA_pc,
        ddt_pc,
        dD_pc,
        dh0_pc,
    )


def mamba3_mimo_bwd_path_c(
    dy: mx.array,
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    out: Mamba3BwdOwnerOutputs | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Backward pass via the lowered TileLang DSL kernel."""

    return _mamba3_mimo_bwd_path_c_kernel(dy, x, B, C, z, A, dt, D, h0, out=out)


@mx.custom_function
def mamba3_mimo_apply_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> mx.array:
    """Path C forward wrapper exposing only ``y``.

    Note (Path B vs Path C):
        The Path B analogue is ``mamba3_mimo_apply`` in ``mamba3.py``.
        Neither apply accepts a ``force_metal`` / ``force_path_c`` kwarg, so
        there is no kwarg rename to migrate. This entrypoint is **not**
        re-exported from ``cppmega_mlx.nn._tilelang.__init__`` — Path C
        Mamba3 is a proof / override path; Path B is the production
        entrypoint. See ``docs/production_kernel_routing.md``.

        The direct tvm-ffi path is graph-transform callable when MLX exposes
        graph-safe DLPack export. TileLang owns output allocation through
        ``out_idx`` metadata; explicit ``out=`` remains the full-ABI
        caller-owned route.
    """

    y, _ = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
    return y


@mamba3_mimo_apply_path_c.vjp
def _mamba3_mimo_apply_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    output: mx.array,
) -> tuple[mx.array, ...]:
    del output
    x, B, C, z, A, dt, D, h0 = primals
    return mamba3_mimo_bwd_path_c(cotangent, x, B, C, z, A, dt, D, h0)


@mx.custom_function
def mamba3_mimo_apply_with_state_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Path C forward returning ``(y, h_last)``.

    The VJP delegates to the TileLang backward kernel and uses the same
    ``out_idx`` output policy as the y-only surface.
    """

    return mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)


@mamba3_mimo_apply_with_state_path_c.vjp
def _mamba3_mimo_apply_with_state_path_c_vjp(
    primals: tuple[mx.array, ...],
    cotangent: tuple[mx.array, mx.array],
    output: tuple[mx.array, mx.array],
) -> tuple[mx.array, ...]:
    del output
    x, B, C, z, A, dt, D, h0 = primals
    dy = cotangent[0]
    return mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)


# Convenience: dump the lowered MSL for the bench shape so reviewers can diff
# Path B's hand-written MSL against Path C's machine-emitted MSL without
# having to re-run the lowering pipeline.
def dump_lowered_fwd_msl(
    *, batch: int, seq: int, heads: int, headdim: int, state: int
) -> str:
    """Return the raw lowered MSL for the Path C forward kernel.

    Used by ``scripts/bench_tilelang_mamba3_path_c.py`` to write the
    ``docs/tilelang_ports/mamba3_path_c_lowered.metal`` artifact.
    """

    kernel, lowering = _fwd_kernel_for(
        batch, seq, heads, headdim, state, return_msl=True
    )
    del kernel
    return _source_with_reused_scalar_bindings(lowering)


def dump_lowered_bwd_msl(
    *, batch: int, seq: int, heads: int, headdim: int, state: int
) -> str:
    """Return the raw lowered MSL for the Path C backward kernel."""

    kernel, lowering = _bwd_kernel_for(
        batch, seq, heads, headdim, state, return_msl=True
    )
    del kernel
    return _source_with_reused_scalar_bindings(lowering)


__all__ = [
    "Mamba3PathCStatus",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "mamba3_mimo_apply_path_c",
    "mamba3_mimo_apply_with_state_path_c",
    "mamba3_mimo_bwd_path_c",
    "mamba3_mimo_fwd_path_c",
    "mamba3_mimo_path_c_status",
]
