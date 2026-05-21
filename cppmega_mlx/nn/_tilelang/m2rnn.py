"""Retired direct-MSL compatibility surface for M2RNN.

The hand-written Path B Metal kernels were intentionally removed from this
module. Production acceleration for M2RNN must go through the TileLang/TVM-FFI
Path C route in ``m2rnn_path_c.py``; this module only keeps the legacy public
API as a pure-MLX correctness oracle for tests and fallback comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


_RETIRED_REASON = "direct-MSL Path B is retired; use m2rnn_path_c.py for native TileLang/TVM-FFI"


@dataclass(frozen=True)
class M2RNNMetalStatus:
    """Capability probe result for the retired Path B M2RNN surface."""

    available: bool
    reason: str


def m2rnn_metal_status(*_arrays: mx.array) -> M2RNNMetalStatus:
    """Report that the direct-MSL M2RNN kernel is no longer available."""

    return M2RNNMetalStatus(False, _RETIRED_REASON)


def _broadcast_inputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    from cppmega_mlx.nn.m2rnn import broadcast_m2rnn_heads

    return broadcast_m2rnn_heads(q, k, v, W, xf)


def _initial_state(
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
    expected = (batch, heads, k_dim, v_dim)
    if h0.shape != expected:
        raise ValueError(f"h0 must have shape {expected}, got {h0.shape}")
    if h0.dtype != dtype:
        raise TypeError(f"h0 dtype {h0.dtype} must match q dtype {dtype}")
    return h0


def _validate_inputs(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
) -> tuple[int, int, int, int, int]:
    """Validate the legacy M2RNN tensor contract used by Path C probes."""

    q_b, k_b, v_b, W_b, xf_b = _broadcast_inputs(q, k, v, W, xf)
    batch, seq, heads, k_dim = q_b.shape
    v_dim = v_b.shape[-1]
    _initial_state(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q_b.dtype,
    )
    if k_b.shape != q_b.shape:
        raise ValueError(f"k broadcast shape {k_b.shape} must match q shape {q_b.shape}")
    if W_b.shape != (heads, v_dim, v_dim):
        raise ValueError(
            f"W broadcast shape {W_b.shape} must be {(heads, v_dim, v_dim)}"
        )
    if xf_b.shape != (batch, seq, heads):
        raise ValueError(
            f"xf broadcast shape {xf_b.shape} must be {(batch, seq, heads)}"
        )
    return batch, seq, heads, k_dim, v_dim


def _reference_scan_with_tanh_cache(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    q, k, v, W, xf = _broadcast_inputs(q, k, v, W, xf)
    batch, seq, heads, k_dim = q.shape
    v_dim = v.shape[-1]
    h = _initial_state(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_dim,
        dtype=q.dtype,
    )

    x_all = mx.expand_dims(k, -1) * mx.expand_dims(v, -2)
    xf_5d = xf[:, :, :, None, None]
    W_expanded = mx.expand_dims(W, 0)
    outputs: list[mx.array] = []
    tanh_cache: list[mx.array] = []
    for s in range(seq):
        z = mx.matmul(h, W_expanded) + x_all[:, s]
        h_new = mx.tanh(z)
        tanh_cache.append(h_new)
        f = xf_5d[:, s]
        h = f * h + (1.0 - f) * h_new
        outputs.append(mx.einsum("bhk,bhkv->bhv", q[:, s], h))

    if outputs:
        out = mx.stack(outputs, axis=1)
        cache = mx.stack(tanh_cache, axis=1)
    else:
        out = mx.zeros((batch, 0, heads, v_dim), dtype=q.dtype)
        cache = mx.zeros((batch, 0, heads, k_dim, v_dim), dtype=q.dtype)
    return out, h, cache


def m2rnn_reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    *,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Delegate to the parent MLX M2RNN reference implementation."""

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
    """Compatibility forward wrapper backed by the pure-MLX reference."""

    return _reference_scan_with_tanh_cache(q, k, v, W, xf, h0)


def m2rnn_bwd_metal(
    dy: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    _tanh_cache: mx.array,
    h0: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Compatibility backward wrapper computed by autograd through the reference."""

    q_b, _k_b, v_b, _W_b, _xf_b = _broadcast_inputs(q, k, v, W, xf)
    batch, _seq, heads, k_dim = q_b.shape
    h0_full = _initial_state(
        h0,
        batch=batch,
        heads=heads,
        k_dim=k_dim,
        v_dim=v_b.shape[-1],
        dtype=q_b.dtype,
    )

    def loss(
        q_arg: mx.array,
        k_arg: mx.array,
        v_arg: mx.array,
        W_arg: mx.array,
        xf_arg: mx.array,
        h0_arg: mx.array,
    ) -> mx.array:
        y, _h = m2rnn_reference(q_arg, k_arg, v_arg, W_arg, xf_arg, h0=h0_arg)
        return mx.sum(y * dy)

    return mx.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, W, xf, h0_full)


def m2rnn_apply(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    W: mx.array,
    xf: mx.array,
    h0: mx.array,
) -> mx.array:
    """Differentiable compatibility wrapper returning only ``y``."""

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
    """Differentiable compatibility wrapper returning ``(y, h_last)``."""

    return m2rnn_reference(q, k, v, W, xf, h0=h0)


__all__ = [
    "M2RNNMetalStatus",
    "_validate_inputs",
    "m2rnn_apply",
    "m2rnn_apply_with_state",
    "m2rnn_bwd_metal",
    "m2rnn_fwd_metal",
    "m2rnn_metal_status",
    "m2rnn_reference",
]
