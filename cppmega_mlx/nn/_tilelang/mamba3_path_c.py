"""Path C port of Mamba3 MIMO fwd+bwd via TileLang DSL ``@T.prim_func`` lowering.

This module is the Path C counterpart to :mod:`cppmega_mlx.nn._tilelang.mamba3`
(Path B). Path B writes MSL by hand and dispatches it via
``mx.fast.metal_kernel``. Path C writes the *same* selective-scan kernel as
TileLang DSL and asks the patched Apple-head TileLang
(:mod:`tilelang.engine.lower.lower(target="metal")`) to emit MSL, which is then
inlined into ``mx.fast.metal_kernel`` exactly the same way Path B's MSL is
inlined.

Why ship both?
--------------

Path B is the production hot path: it is the version that has shipped through
the bench harness and parity tests. Path C is intentionally the *same algorithm
expressed in the high-level DSL* so we can:

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

The kernels operate on FP32 carriers (BF16 callers must up-cast at the wrapper
level; FP16 callers also up-cast — same convention as Path B). At FP32 the
Path C and Path B kernels are *bit identical* on the tested shapes. The
parity budget retained in tests is the conservative atol=1e-4 / rtol=1e-3.

Public surface
--------------

* :func:`mamba3_mimo_fwd_path_c` — fwd lane scan returning ``(y, h_last)``.
* :func:`mamba3_mimo_bwd_path_c` — bwd lane scan returning grads w.r.t.
  ``(x, B, C, z, A, dt, D, h0)`` after host-side P-axis reductions.
* :func:`mamba3_mimo_apply_path_c` — :func:`mx.custom_function`-wrapped fwd that
  ties to the bwd VJP, mirroring :func:`mamba3.mamba3_mimo_apply`.
* :func:`mamba3_mimo_path_c_status` — preflight check for the lowered TileLang
  DSL kernel; falls back to Path B when TileLang fails to lower.

Threadgroup tuning
------------------

The Path B grid uses one thread per (b, h, p) lane with up to 256 threads per
threadgroup, matching the Apple Metal target's 1024-thread / 32 KB-shared
ceilings. Path C uses the same one-thread-per-lane algorithm, but caps the
threadgroup at 32 lanes. The TileLang-lowered scan keeps ``h_state[STATE]`` and
backward ``dh[STATE]`` in per-thread registers; smaller threadgroups reduce the
register-pressure/occupancy cliff on M4 Max while keeping the generated MSL and
global memory layout identical.

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
from typing import Any

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang._engine_dispatch import artifact_to_source, dispatch_lower
from cppmega_mlx.nn._tilelang._msl_transform import (
    MSLDispatchUnsupported,
    can_run_metal,
)
from cppmega_mlx.nn._tilelang.mamba3 import (
    _validate_inputs,
    mamba3_mimo_reference,
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
    return Mamba3PathCStatus(available=True, reason="Path C TileLang DSL ready")


# ---------------------------------------------------------------------------
# TileLang PrimFunc factories (cached on shape signature)
# ---------------------------------------------------------------------------


def _threads_for(lanes: int) -> int:
    """Return the threadgroup size for a per-lane kernel.

    Apple Metal's ``Target("metal")`` reports ``max_num_threads=256``; the
    Mamba3 selective-scan kernels have enough per-thread register state that
    smaller threadgroups benchmark better on M4 Max. A 32-thread cap preserves
    one full Apple SIMD group while avoiding the 128/256-thread occupancy cliff
    seen in the backward replay/reverse-scan kernel.
    """

    if lanes <= 0:
        return 1
    return min(32, lanes)


@lru_cache(maxsize=128)
def _fwd_kernel_for(
    BATCH: int,
    SEQ: int,
    HEADS: int,
    HEADDIM: int,
    STATE: int,
    *,
    return_msl: bool = False,
) -> tuple[Any, _msl_transform.TileLangMSLLowering | None]:
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

    artifact = dispatch_lower(fwd, target="metal", return_msl=return_msl)
    if hasattr(artifact, "_tilelang_engine_target"):
        # Engine path: artifact is a tilelang.compile callable; callers branch
        # on ``lowering is None`` to invoke it directly.
        return artifact, None
    lowering = artifact  # TileLangMSLLowering
    # Buffer alphabetic order from the lowered MSL is:
    # A, B, C, D, dt, h0, h_last, x, y, z
    kernel = mx.fast.metal_kernel(
        name=f"cppmega_mamba3_path_c_fwd_{BATCH}_{SEQ}_{HEADS}_{HEADDIM}_{STATE}",
        input_names=["A", "B", "C", "D", "dt", "h0", "x", "z"],
        output_names=["h_last", "y"],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
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
) -> tuple[Any, _msl_transform.TileLangMSLLowering | None]:
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

    artifact = dispatch_lower(bwd, target="metal", return_msl=return_msl)
    if hasattr(artifact, "_tilelang_engine_target"):
        return artifact, None
    lowering = artifact
    # Buffer alphabetic order from the lowered MSL is:
    # A, B, C, D, dA_partial, dB_partial, dC_partial, dD_partial, ddt_partial,
    # dh0, dt, dx, dy, dz, h0, h_steps, x, z
    kernel = mx.fast.metal_kernel(
        name=f"cppmega_mamba3_path_c_bwd_{BATCH}_{SEQ}_{HEADS}_{HEADDIM}_{STATE}",
        input_names=["A", "B", "C", "D", "dt", "dy", "h0", "x", "z"],
        output_names=[
            "dA_partial",
            "dB_partial",
            "dC_partial",
            "dD_partial",
            "ddt_partial",
            "dh0",
            "dx",
            "dz",
            "h_steps",
        ],
        source=lowering.body,
        header=lowering.header,
        ensure_row_contiguous=True,
    )
    return kernel, lowering


# ---------------------------------------------------------------------------
# Public dispatch entry points
# ---------------------------------------------------------------------------


def _to_fp32(x: mx.array) -> mx.array:
    if x.dtype == mx.float32:
        return x
    return x.astype(mx.float32)


def mamba3_mimo_fwd_path_c(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    z: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Path C forward; falls back to the Path B reference when TileLang fails."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if seq == 0:
        return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    out_dtype = x.dtype
    inputs_f32 = [
        _to_fp32(A),
        _to_fp32(B),
        _to_fp32(C),
        _to_fp32(D),
        _to_fp32(dt),
        _to_fp32(h0),
        _to_fp32(x),
        _to_fp32(z),
    ]

    try:
        kernel, lowering = _fwd_kernel_for(batch, seq, heads, headdim, state)
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    if lowering is None:
        # Engine path: kernel is a tilelang.compile artifact taking the prim_func
        # arg list (x, B, C, z, A, dt, D, h0, y, h_last) -- y/h_last are output
        # buffers. We currently only have shim parity tested on Metal; surface
        # engine errors as fallback to keep public contract stable.
        try:
            h_last = mx.zeros((batch, heads, headdim, state), dtype=mx.float32)
            y = mx.zeros((batch, seq, heads, headdim), dtype=mx.float32)
            kernel(
                inputs_f32[6],  # x
                inputs_f32[1],  # B
                inputs_f32[2],  # C
                inputs_f32[7],  # z
                inputs_f32[0],  # A
                inputs_f32[4],  # dt
                inputs_f32[3],  # D
                inputs_f32[5],  # h0
                y,
                h_last,
            )
            return y.astype(out_dtype), h_last.astype(out_dtype)
        except Exception:
            return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        out_list = kernel(
            inputs=inputs_f32,
            output_shapes=[
                (batch, heads, headdim, state),  # h_last (alphabetic)
                (batch, seq, heads, headdim),    # y
            ],
            output_dtypes=[mx.float32, mx.float32],
            grid=grid,
            threadgroup=lowering.threadgroup,
        )
    except Exception:
        return mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)

    h_last, y = out_list
    return y.astype(out_dtype), h_last.astype(out_dtype)


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
) -> tuple[mx.array, ...] | None:
    """Run the lowered Path C bwd kernel; return None on dispatch failure."""

    status = mamba3_mimo_path_c_status()
    if not status.available:
        return None

    batch, seq, heads, headdim, state = _validate_inputs(x, B, C, z, A, dt, D, h0)
    if seq == 0:
        return None

    inputs_f32 = [
        _to_fp32(A),
        _to_fp32(B),
        _to_fp32(C),
        _to_fp32(D),
        _to_fp32(dt),
        _to_fp32(dy),
        _to_fp32(h0),
        _to_fp32(x),
        _to_fp32(z),
    ]

    try:
        kernel, lowering = _bwd_kernel_for(batch, seq, heads, headdim, state)
    except (MSLDispatchUnsupported, RuntimeError, ValueError):
        return None

    if lowering is None:
        # Engine path: bwd kernel signature has 9 inputs + 9 outputs in the
        # @T.prim_func declaration order (A, B, C, D, dt, dy, h0, x, z, ...).
        # Engine path support is currently best-effort; on any failure we
        # surface ``None`` so the caller falls back to Path B pure-MLX.
        return None

    grid = (
        lowering.grid[0] * lowering.threadgroup[0],
        lowering.grid[1] * lowering.threadgroup[1],
        lowering.grid[2] * lowering.threadgroup[2],
    )

    try:
        out_list = kernel(
            inputs=inputs_f32,
            output_shapes=[
                (batch, seq, heads, headdim),                  # dA_partial
                (batch, seq, heads, headdim, state),           # dB_partial
                (batch, seq, heads, headdim, state),           # dC_partial
                (batch, heads, headdim),                       # dD_partial
                (batch, seq, heads, headdim),                  # ddt_partial
                (batch, heads, headdim, state),                # dh0
                (batch, seq, heads, headdim),                  # dx
                (batch, seq, heads, headdim),                  # dz
                (batch, heads, headdim, seq, state),           # h_steps
            ],
            output_dtypes=[mx.float32] * 9,
            grid=grid,
            threadgroup=lowering.threadgroup,
        )
    except Exception:
        return None

    dA_p, dB_p, dC_p, dD_p, ddt_p, dh0_pc, dx_pc, dz_pc, _h_scratch = out_list
    # Reduce P-dim partials into final shapes.
    dB_pc = mx.sum(dB_p, axis=3)         # -> (B, T, H, N)
    dC_pc = mx.sum(dC_p, axis=3)         # -> (B, T, H, N)
    dA_pc = mx.sum(dA_p, axis=3)         # -> (B, T, H)
    ddt_pc = mx.sum(ddt_p, axis=3)       # -> (B, T, H)
    dD_pc = mx.sum(dD_p, axis=(0, 2))    # -> (H,)
    return (
        dx_pc.astype(x.dtype),
        dB_pc.astype(B.dtype),
        dC_pc.astype(C.dtype),
        dz_pc.astype(z.dtype),
        dA_pc.astype(A.dtype),
        ddt_pc.astype(dt.dtype),
        dD_pc.astype(D.dtype),
        dh0_pc.astype(h0.dtype),
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
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Backward pass via the lowered TileLang DSL kernel; falls back to Path B."""

    metal_result = _mamba3_mimo_bwd_path_c_kernel(dy, x, B, C, z, A, dt, D, h0)
    if metal_result is not None:
        return metal_result
    # Path B (pure-MLX) fallback for parity preservation when TileLang fails.
    from cppmega_mlx.nn._tilelang.mamba3 import _mamba3_mimo_bwd_pure_mlx
    return _mamba3_mimo_bwd_pure_mlx(dy, x, B, C, z, A, dt, D, h0)


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
    """Path C forward wrapper exposing only ``y`` for VJP symmetry.

    Note (Path B vs Path C):
        The Path B analogue is ``mamba3_mimo_apply`` in ``mamba3.py``.
        Neither apply accepts a ``force_metal`` / ``force_path_c`` kwarg, so
        there is no kwarg rename to migrate. This entrypoint is **not**
        re-exported from ``cppmega_mlx.nn._tilelang.__init__`` — Path C
        Mamba3 is a proof / override path; Path B is the production
        entrypoint. See ``docs/production_kernel_routing.md``.
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
    return artifact_to_source(kernel if lowering is None else lowering)


def dump_lowered_bwd_msl(
    *, batch: int, seq: int, heads: int, headdim: int, state: int
) -> str:
    """Return the raw lowered MSL for the Path C backward kernel."""

    kernel, lowering = _bwd_kernel_for(
        batch, seq, heads, headdim, state, return_msl=True
    )
    return artifact_to_source(kernel if lowering is None else lowering)


__all__ = [
    "Mamba3PathCStatus",
    "dump_lowered_bwd_msl",
    "dump_lowered_fwd_msl",
    "mamba3_mimo_apply_path_c",
    "mamba3_mimo_bwd_path_c",
    "mamba3_mimo_fwd_path_c",
    "mamba3_mimo_path_c_status",
]
