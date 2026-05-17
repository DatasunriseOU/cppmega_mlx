"""Retired direct-MSL compatibility surface for cppmega's M2RNN recurrence.

The old Path B implementation in this module built hand-written Metal source
strings and dispatched them through ``mx.fast.metal_kernel``. P2 cleanup retires
that direct-MSL runtime surface. The exported names remain so older tests and
callers keep resolving, but they now route through the pure-MLX reference math.
Production dispatch in :mod:`cppmega_mlx.nn.m2rnn` selects the TileLang/tvm-ffi
Path C route in ``m2rnn_path_c.py`` when eligible, or the reference fallback
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


_RETIRED_REASON = (
    "M2RNN direct-MSL Path B is retired; use m2rnn_path_c.py "
    "or the pure-MLX reference"
)


@dataclass(frozen=True)
class M2RNNMetalStatus:
    """Capability probe result for the retired Path B M2RNN surface."""

    available: bool
    reason: str


def _validate_inputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None,
) -> tuple[int, int, int, int, int]:
    """Return (B, S, H, K_DIM, V_DIM) and validate post-broadcast shapes."""

    if q.ndim != 4:
        raise ValueError(f"q must be (B,S,H,K), got {q.shape}")
    batch, seq, heads, k_dim = q.shape
    if k.shape != (batch, seq, heads, k_dim):
        raise ValueError(f"k must be {(batch, seq, heads, k_dim)}, got {k.shape}")
    if v.ndim != 4:
        raise ValueError(f"v must be 4D, got {v.shape}")
    v_dim = v.shape[-1]
    if v.shape != (batch, seq, heads, v_dim):
        raise ValueError(f"v must be {(batch, seq, heads, v_dim)}, got {v.shape}")
    if W.shape != (heads, v_dim, v_dim):
        raise ValueError(f"W must be {(heads, v_dim, v_dim)}, got {W.shape}")
    if xf.shape != (batch, seq, heads):
        raise ValueError(f"xf must be {(batch, seq, heads)}, got {xf.shape}")
    if h0 is not None and h0.shape != (batch, heads, k_dim, v_dim):
        raise ValueError(
            f"h0 must be {(batch, heads, k_dim, v_dim)}, got {h0.shape}"
        )
    return batch, seq, heads, k_dim, v_dim


def _broadcast_inputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    from cppmega_mlx.nn.m2rnn import broadcast_m2rnn_heads

    return broadcast_m2rnn_heads(q, k, v, W, xf)


def _materialize_h0(
    h0: mx.array | None,
    *,
    batch: int,
    heads: int,
    k_dim: int,
    v_dim: int,
    dtype: mx.Dtype,
) -> mx.array:
    if h0 is None:
        return mx.zeros((batch, heads, k_dim, v_dim), dtype=dtype)
    return h0.astype(dtype)


def m2rnn_metal_status(*_arrays: mx.array) -> M2RNNMetalStatus:
    """Report the retired direct-MSL Path B status."""

    return M2RNNMetalStatus(False, _RETIRED_REASON)


def m2rnn_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    *,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Pure-MLX reference, shared with the parent M2RNN module."""

    from cppmega_mlx.nn.m2rnn import m2rnn_scan

    return m2rnn_scan(q, k, v, W, xf, h0=h0)


def m2rnn_fwd_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Compatibility forward returning (y, h_last, tanh_cache).

    Despite the historical name, this no longer launches direct MSL. It uses the
    reference recurrence and keeps the saved ``tanh(z_t)`` cache for callers
    that still exercise the old backward surface.
    """

    q, k, v, W, xf = _broadcast_inputs(q, k, v, W, xf)
    batch, seq, heads, k_dim, v_dim = _validate_inputs(q, k, v, W, xf, h0)
    h0_full = _materialize_h0(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )
    return _m2rnn_fwd_reference_with_cache(q, k, v, W, xf, h0_full)


def _m2rnn_fwd_reference_with_cache(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    if seq == 0:
        return (
            mx.zeros((batch, 0, heads, v_dim), dtype=q.dtype),
            h0,
            mx.zeros((batch, 0, heads, k_dim, v_dim), dtype=q.dtype),
        )

    work_dtype = mx.float32
    h = h0.astype(work_dtype)
    W_e = W.astype(work_dtype)[None, :, :, :]
    x_all = (
        mx.expand_dims(k.astype(work_dtype), -1)
        * mx.expand_dims(v.astype(work_dtype), -2)
    )
    xf_5d = xf.astype(work_dtype)[:, :, :, None, None]
    out_steps: list[mx.array] = []
    tanh_steps: list[mx.array] = []
    for s in range(seq):
        f = xf_5d[:, s]
        z = mx.matmul(h, W_e[0]) + x_all[:, s]
        tz = mx.tanh(z)
        tanh_steps.append(tz)
        h = f * h + (1.0 - f) * tz
        out_steps.append(mx.einsum("bhk,bhkv->bhv", q[:, s].astype(work_dtype), h))
    y = mx.stack(out_steps, axis=1).astype(q.dtype)
    h_final = h.astype(q.dtype)
    tanh_cache = mx.stack(tanh_steps, axis=1).astype(q.dtype)
    return y, h_final, tanh_cache


def m2rnn_bwd_metal(
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    tanh_cache: mx.array,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compatibility backward returning gradients for (q, k, v, W, xf, h0)."""

    q, k, v, W, xf = _broadcast_inputs(q, k, v, W, xf)
    batch, seq, heads, k_dim, v_dim = _validate_inputs(q, k, v, W, xf, h0)
    if dy.shape != (batch, seq, heads, v_dim):
        raise ValueError(f"dy must be {(batch, seq, heads, v_dim)}, got {dy.shape}")
    if tanh_cache.shape != (batch, seq, heads, k_dim, v_dim):
        raise ValueError(
            "tanh_cache must be "
            f"{(batch, seq, heads, k_dim, v_dim)}, got {tanh_cache.shape}"
        )

    h0_dtype = h0.dtype if h0 is not None else q.dtype
    h0_full = _materialize_h0(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )
    if seq == 0:
        return (
            mx.zeros_like(q),
            mx.zeros_like(k),
            mx.zeros_like(v),
            mx.zeros_like(W),
            mx.zeros_like(xf),
            h0_full.astype(h0_dtype),
        )
    return _m2rnn_bwd_reference(
        dy=dy,
        q=q,
        k=k,
        v=v,
        W=W,
        xf=xf,
        h0_full=h0_full,
        tanh_cache=tanh_cache,
        out_dtypes=(q.dtype, k.dtype, v.dtype, W.dtype, xf.dtype),
        h0_dtype=h0_dtype,
    )


def _m2rnn_bwd_reference(
    *,
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0_full: mx.array,
    tanh_cache: mx.array,
    out_dtypes: tuple[mx.Dtype, ...],
    h0_dtype: mx.Dtype,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    batch, seq, heads, _k_dim = q.shape
    v_dim = v.shape[-1]
    work_dtype = mx.float32

    q_f = q.astype(work_dtype)
    k_f = k.astype(work_dtype)
    v_f = v.astype(work_dtype)
    W_f = W.astype(work_dtype)
    xf_f = xf.astype(work_dtype)
    h0_f = h0_full.astype(work_dtype)
    dy_f = dy.astype(work_dtype)
    tanh_f = tanh_cache.astype(work_dtype)

    h_steps: list[mx.array] = []
    h = h0_f
    for t in range(seq):
        h_steps.append(h)
        f = xf_f[:, t, :, None, None]
        h = f * h + (1.0 - f) * tanh_f[:, t]

    dh = mx.zeros_like(h0_f)
    dW_acc = mx.zeros((heads, v_dim, v_dim), dtype=work_dtype)
    dq_steps: list[mx.array] = []
    dk_steps: list[mx.array] = []
    dv_steps: list[mx.array] = []
    dxf_steps: list[mx.array] = []

    for t in range(seq - 1, -1, -1):
        f = xf_f[:, t, :, None, None]
        one_minus_f = 1.0 - f
        h_prev = h_steps[t]
        tz = tanh_f[:, t]
        h_t = f * h_prev + one_minus_f * tz
        dY_t = dy_f[:, t]

        dq_t = mx.einsum("bhv,bhkv->bhk", dY_t, h_t)
        dh = dh + q_f[:, t, :, :, None] * dY_t[:, :, None, :]
        dxf_t = mx.sum(dh * (h_prev - tz), axis=(-1, -2))
        dz = one_minus_f * dh * (1.0 - tz * tz)
        dk_t = mx.sum(dz * v_f[:, t, :, None, :], axis=-1)
        dv_t = mx.sum(dz * k_f[:, t, :, :, None], axis=-2)
        dW_acc = dW_acc + mx.sum(
            mx.einsum("bhkv,bhku->bhvu", h_prev, dz),
            axis=0,
        )
        dh = f * dh + mx.einsum("bhku,hvu->bhkv", dz, W_f)

        dq_steps.append(dq_t)
        dk_steps.append(dk_t)
        dv_steps.append(dv_t)
        dxf_steps.append(dxf_t)

    dq = mx.stack(list(reversed(dq_steps)), axis=1).astype(out_dtypes[0])
    dk = mx.stack(list(reversed(dk_steps)), axis=1).astype(out_dtypes[1])
    dv = mx.stack(list(reversed(dv_steps)), axis=1).astype(out_dtypes[2])
    dxf_out = mx.stack(list(reversed(dxf_steps)), axis=1).astype(out_dtypes[4])
    return (
        dq,
        dk,
        dv,
        dW_acc.astype(out_dtypes[3]),
        dxf_out,
        dh.astype(h0_dtype),
    )


def m2rnn_apply(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> mx.array:
    """Reference compatibility wrapper returning the gated output y."""

    y, _h = m2rnn_reference(q, k, v, W, xf, h0=h0)
    return y


def m2rnn_apply_with_state(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> tuple[mx.array, mx.array]:
    """Reference compatibility wrapper returning (y, h_last)."""

    return m2rnn_reference(q, k, v, W, xf, h0=h0)


__all__ = [
    "M2RNNMetalStatus",
    "m2rnn_apply",
    "m2rnn_apply_with_state",
    "m2rnn_bwd_metal",
    "m2rnn_fwd_metal",
    "m2rnn_metal_status",
    "m2rnn_reference",
]
