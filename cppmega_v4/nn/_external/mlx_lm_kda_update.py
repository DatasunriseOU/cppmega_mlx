"""KDA Path E adapter — wraps mlx-lm's gated_delta kernel (vectorised gate).

The same upstream Metal kernel (``mlx_lm/models/gated_delta.py``,
``_make_gated_delta_kernel(vectorized=True)``) that powers GDN Path E also
handles the per-K vectorised gate used by KDA-style attention — see
``mlx_lm/models/kimi_linear.py::KimiDeltaAttention`` which calls
``gated_delta_update`` with ``a_logits`` shaped ``[B, T, num_heads, head_dim]``.

The kernel selection happens automatically inside ``gated_delta_kernel``:
when ``g.ndim == 4`` it picks ``_gated_delta_kernel_vec``; when ``g.ndim == 3``
it picks the scalar variant used by GDN. Group expansion (Hv > Hk) is also
handled inside the kernel (``hk_idx = hv_idx / (Hv / Hk)``).

Our naive_recurrent_kda signature (FLA convention):
    naive_recurrent_kda(q, k, v, g, beta, scale=None,
                        initial_state=None, output_final_state=False)
                        -> (o, S)
    where:
      - q, k: [B, T, H,  K]                 (un-repeated; HV/H expansion is kernel-side)
      - v:    [B, T, HV, V]
      - g:    [B, T, HV, K]  (per-K log-decay; state is multiplied by exp(g))
      - beta: [B, T, HV]
      - S:    [B, HV, K, V]  (FLA layout — K rows, V cols)
    FLA pre-scales q by 1/sqrt(K).

Upstream gated_delta_kernel signature:
    gated_delta_kernel(q, k, v, g, beta, state, mask=None)
      - q, k: [B, T, Hk, Dk]
      - v:    [B, T, Hv, Dv]
      - g:    [B, T, Hv, Dk]  (vectorized)  OR  [B, T, Hv] (scalar)
      - beta: [B, T, Hv]      (kernel applies it as-is, no sigmoid)
      - state: [B, Hv, Dv, Dk]  (upstream layout — V rows, K cols; transposed vs FLA)
    Upstream does NOT pre-scale q. Decay is plain exp(g) where g is the value
    we pass in directly (kernel does ``state[i] = state[i] * g_[s_idx]``),
    so we pass ``g_decay = exp(our_g)`` directly here.

Constraints:
    - Upstream Metal kernel requires Dk % 32 == 0 and Dv % 4 == 0; smaller
      dims fall back to the pure-ops reference path.
"""

from __future__ import annotations

import math

import mlx.core as mx

from cppmega_v4.nn._external._mlx_lm_gated_delta_vendored import (
    gated_delta_kernel as _upstream_kernel,
    gated_delta_ops as _upstream_ops,
)


def kda_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    *,
    scale: float | None = None,
    initial_state: mx.array | None = None,
    output_final_state: bool = False,
):
    """Path E entry — same signature as ``naive_recurrent_kda``.

    Args:
        q, k:          [B, T, H,  K]  (FLA convention; HV/H expansion is kernel-side)
        v:             [B, T, HV, V]
        g:             [B, T, HV, K]  per-K log-decay (state *= exp(g) per step)
        beta:          [B, T, HV]
        scale:         optional — defaults to 1/sqrt(K)
        initial_state: optional [B, HV, K, V]
        output_final_state: if True, return (o, S_final[B, HV, K, V])

    Returns:
        (o[B, T, HV, V], S_final or None)
    """
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError(
            f"q/k must be [B, T, H, K]; got q={q.shape}, k={k.shape}"
        )
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError(f"v must be [B, T, HV, V]; got v={v.shape}")
    if g.shape != (*v.shape[:3], k.shape[-1]):
        raise ValueError(
            f"g must be [B, T, HV, K]; got g={g.shape}, expected {(*v.shape[:3], k.shape[-1])}"
        )
    if beta.shape != v.shape[:3]:
        raise ValueError(f"beta must be [B, T, HV]; got beta={beta.shape}")

    b_size, t_size, h_size, kdim = q.shape
    hv_size, vdim = v.shape[2], v.shape[-1]
    if hv_size % h_size != 0:
        raise ValueError(f"HV ({hv_size}) must be divisible by H ({h_size})")

    # FLA pre-scales q by 1/sqrt(K); upstream does not.
    fla_scale = scale if scale is not None else 1.0 / math.sqrt(kdim)
    q_scaled = (q.astype(mx.float32) * fla_scale).astype(q.dtype)

    # Upstream kernel multiplies state by `g` as-is each step. FLA convention:
    # state is multiplied by `exp(g)`. Pre-exponentiate.
    g_decay = mx.exp(g.astype(mx.float32))

    # Upstream state layout: [B, Hv, Dv, Dk]. FLA: [B, HV, K, V]. Transpose
    # last two axes when handing off.
    if initial_state is None:
        state = mx.zeros((b_size, hv_size, vdim, kdim), dtype=mx.float32)
    else:
        state = mx.transpose(initial_state.astype(mx.float32), (0, 1, 3, 2))

    beta_f = beta.astype(mx.float32)

    # Upstream Metal kernel needs Dk % 32 == 0 and Dv % 4 == 0; otherwise
    # fall through to the pure-ops reference (which also handles vector-gate).
    use_kernel = (kdim % 32 == 0) and (vdim % 4 == 0) and mx.metal.is_available()

    if use_kernel:
        y, new_state = _upstream_kernel(
            q_scaled, k, v, g_decay, beta_f, state, mask=None,
        )
    else:
        y, new_state = _upstream_ops(
            q_scaled, k, v, g_decay, beta_f, state, mask=None,
        )

    final = None
    if output_final_state:
        # Convert back to FLA layout [B, HV, K, V].
        final = mx.transpose(new_state, (0, 1, 3, 2))
    return y, final


__all__ = ["kda_update"]
