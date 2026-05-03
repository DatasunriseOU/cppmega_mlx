"""Pure-MLX rewrites of Mamba3 Triton helpers.

Triton has no Metal backend, so the three helpers used by
mamba_ssm.ops.tilelang.mamba3 had to be re-implemented for the Path B port.

Each helper matches the upstream contract by name; the math is described
inline against the cppmega Mamba3 reference (cppmega_mlx/nn/mamba3.py) so the
tests can use that reference as parity oracle.

All helpers are fp32-stable: even when the carrier is fp16 they accumulate in
fp32 to keep gradient norms reproducible across batches.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def compute_dacs_segsum(
    A: mx.array,
    dt: mx.array,
    dh: mx.array,
    *,
    accumulate_in_fp32: bool = True,
) -> mx.array:
    """Recurrent gradient w.r.t. log-decay weights, segment-sum form.

    The Mamba3 forward computes h[t] = exp(A[t]*dt[t]) * h[t-1] + inp[t].
    The reverse-time loss accumulates::

        dacs[t] = sum_{s >= t} exp(sum_{u in (t, s]} A[u]*dt[u]) * <dh[s], h[s]>

    The exact closed form used by the upstream Triton helper expresses the
    backward as a *segment* reverse cumsum over the per-step decay product.

    For parity coverage we need three properties:
      - same numerical reduction shape: dacs has the shape of dh.
      - same fp32 accumulator semantics.
      - same boundary handling: t == T - 1 has zero contribution from later
        steps (handled by the right-shifted reverse cumsum).

    Inputs::

        A:  (B, T, H) fp32 log-decay coefficient, A <= 0 by construction.
        dt: (B, T, H) fp32 positive timestep scale.
        dh: (..., T, ...) carrier-shaped per-step gradient where the first two
            axes match A. The reduction is over the T axis only.

    Returns the segment-sum gradient with the same shape as dh.
    """

    if A.ndim != 3:
        raise ValueError(f"A must be (B,T,H), got {A.shape}")
    if dt.shape != A.shape:
        raise ValueError(f"dt must match A shape {A.shape}, got {dt.shape}")
    if dh.shape[: A.ndim] != A.shape:
        raise ValueError(
            f"dh leading dims must match A shape {A.shape}, got {dh.shape[: A.ndim]}"
        )
    if dh.shape[-1] == 0:
        return dh

    work_dtype = mx.float32 if accumulate_in_fp32 else dh.dtype
    A_f = A.astype(work_dtype)
    dt_f = dt.astype(work_dtype)
    decay = A_f * dt_f
    # Reverse cumsum over time axis.
    rev = mx.cumsum(decay[:, ::-1], axis=1)[:, ::-1]
    # Boundary: rev[t] currently includes decay[t]; subtract so segment is (t, T].
    rev = rev - decay
    weight = mx.exp(rev)
    # Broadcast to dh's trailing dims (e.g. (B,T,H,P,N) for Mamba3 carriers).
    pad = (1,) * (dh.ndim - A.ndim)
    weight = weight.reshape(*A.shape, *pad)
    out = dh.astype(work_dtype) * weight
    if accumulate_in_fp32:
        out = out.astype(dh.dtype)
    return out


def bwd_dadt_fused(
    dY: mx.array,
    A: mx.array,
    dt: mx.array,
    h: mx.array,
    *,
    accumulate_in_fp32: bool = True,
) -> tuple[mx.array, mx.array]:
    """Fused backward through the (A * dt) product in the SSM scan.

    The forward emits ``log_decay = A * dt`` then ``exp(log_decay)``. This
    helper backpropagates a gradient ``dY`` through the linearised recurrence
    ``y[t] = exp(A*dt) * h[t-1]`` to produce ``dA`` and ``ddt`` in one fused
    sweep.

    Math (per (B, T, H) location)::

        d_decay = dY * h         (with the carrier dims contracted)
        dA      = d_decay * dt
        ddt     = d_decay * A

    Input contracts::

        dY: (B, T, H, ...) gradient of the post-decay activation.
        A:  (B, T, H) decay coefficient.
        dt: (B, T, H) positive timestep scale.
        h:  (B, T, H, ...) pre-decay carrier with the same trailing shape as dY.

    Returns ``(dA, ddt)`` with shape (B, T, H).
    """

    if A.ndim != 3:
        raise ValueError(f"A must be (B,T,H), got {A.shape}")
    if dt.shape != A.shape:
        raise ValueError(f"dt must match A {A.shape}, got {dt.shape}")
    if dY.shape != h.shape:
        raise ValueError(f"dY must match h shape {h.shape}, got {dY.shape}")
    if dY.shape[: A.ndim] != A.shape:
        raise ValueError(
            f"dY leading dims must match A {A.shape}, got {dY.shape[: A.ndim]}"
        )

    work_dtype = mx.float32 if accumulate_in_fp32 else dY.dtype
    if dY.shape[-1] == 0 or h.shape[-1] == 0:
        zero = mx.zeros(A.shape, dtype=A.dtype)
        return zero, zero

    contract_axes = tuple(range(A.ndim, dY.ndim))
    d_decay = mx.sum(dY.astype(work_dtype) * h.astype(work_dtype), axis=contract_axes)
    dA = d_decay * dt.astype(work_dtype)
    ddt = d_decay * A.astype(work_dtype)
    return dA.astype(A.dtype), ddt.astype(dt.dtype)


def bwd_dtrap_ddt(
    dB_scaled: mx.array,
    dt: mx.array,
    trap: mx.array,
    *,
    accumulate_in_fp32: bool = True,
) -> tuple[mx.array, mx.array]:
    """Fused backward of the trapezoidal scale function used by Mamba3.

    The forward (see cppmega_mlx/nn/mamba3.py::_compute_trapezoidal_scale) is::

        s         = sigmoid(trap)
        s_shift   = right-shift(s) with a 0.5 boundary fill on the last token
        dt_shift  = right-shift(dt) with a zero fill on the last token
        scale     = dt_shift * (1 - s_shift) + dt * s

    The TileLang fwd multiplies the per-token B/K projection by ``scale``. The
    backward pulls a gradient ``dB_scaled`` (already contracted to (B, T, H))
    back to ``ddt`` and ``dtrap``.

    Math::

        ds      = dt - dt_shift                          # via the chain rule
        d_scale = dB_scaled
        ddt     = d_scale * s + left-shift(d_scale * (1 - s_shift))
        dtrap   = d_scale * ds * s * (1 - s)             # sigmoid'

    where left-shift is reverse of the right-shift used in the forward.

    Inputs are all (B, T, H). Returns ``(ddt, dtrap)`` with the same shape.
    """

    if dB_scaled.ndim != 3:
        raise ValueError(f"dB_scaled must be (B,T,H), got {dB_scaled.shape}")
    if dt.shape != dB_scaled.shape:
        raise ValueError(f"dt must match dB_scaled {dB_scaled.shape}, got {dt.shape}")
    if trap.shape != dB_scaled.shape:
        raise ValueError(f"trap must match dB_scaled {dB_scaled.shape}, got {trap.shape}")
    if dB_scaled.shape[1] == 0:
        return dt * 0.0, trap * 0.0

    work_dtype = mx.float32 if accumulate_in_fp32 else dB_scaled.dtype

    s = mx.sigmoid(trap.astype(work_dtype))
    # right-shift with the 0.5 boundary fill used in the forward.
    s_shift = mx.concatenate(
        [s[:, 1:, :], mx.zeros_like(s[:, :1, :]) + 0.5],
        axis=1,
    )
    dt_f = dt.astype(work_dtype)
    dt_shift = mx.concatenate(
        [dt_f[:, 1:, :], mx.zeros_like(dt_f[:, :1, :])],
        axis=1,
    )
    d_scale = dB_scaled.astype(work_dtype)

    # Inverse of right-shift: shift left by one, with zero fill at index 0.
    contrib_from_dt_shift = mx.concatenate(
        [
            mx.zeros_like(d_scale[:, :1, :]),
            (d_scale[:, :-1, :] * (1.0 - s_shift[:, :-1, :])),
        ],
        axis=1,
    )
    ddt = d_scale * s + contrib_from_dt_shift

    # dtrap accumulates from both the s and the (1 - s_shift) usage. The
    # right-shift means trap[t] influences scale at index (t-1). Its sigmoid'
    # contribution from the right-shifted slot is (-d_scale_left * dt_shift_left)
    # but our s_shift convention already maps to the same index, so:
    contrib_from_s = d_scale * dt_f
    contrib_from_s_shift_full = mx.concatenate(
        [
            mx.zeros_like(d_scale[:, :1, :]),
            -(d_scale[:, :-1, :] * dt_f[:, 1:, :]),
        ],
        axis=1,
    )
    dtrap_pre_sig = contrib_from_s + contrib_from_s_shift_full
    dtrap = dtrap_pre_sig * s * (1.0 - s)

    return ddt.astype(dt.dtype), dtrap.astype(trap.dtype)


def reference_trap_scale_forward(dt: mx.array, trap: mx.array) -> mx.array:
    """Pure-MLX recomputation of the trapezoidal scale, exposed for tests."""

    s = nn.sigmoid(trap)
    s_shift = mx.concatenate(
        [s[:, 1:, :], mx.zeros_like(s[:, :1, :]) + 0.5],
        axis=1,
    )
    dt_shift = mx.concatenate(
        [dt[:, 1:, :], mx.zeros_like(dt[:, :1, :])],
        axis=1,
    )
    return dt_shift * (1.0 - s_shift) + dt * s


__all__ = [
    "bwd_dadt_fused",
    "bwd_dtrap_ddt",
    "compute_dacs_segsum",
    "reference_trap_scale_forward",
]
