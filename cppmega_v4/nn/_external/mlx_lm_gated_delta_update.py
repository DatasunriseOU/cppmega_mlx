"""Path E adapter — wraps mlx-lm PR #1217's gated_delta_update for our API.

Upstream signature (vendored):
    gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                       use_kernel=True, training=False) -> (y, state)
    where ``g_decay = exp(-exp(A_log) * softplus(a + dt_bias))`` and
          ``beta = sigmoid(b)`` are computed inside.
    Upstream does NOT pre-scale q.

Our naive_recurrent_gated_delta_rule signature (FLA convention):
    naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None,
                                     initial_state=None,
                                     output_final_state=False) -> (o, h)
    where ``g`` is the per-step LOG-decay (state is multiplied by ``exp(g)``
    each step) and FLA pre-scales ``q`` by ``1/sqrt(K)``.

Adapter mapping:
1. Decay: upstream needs decay in (0,1]; FLA uses exp(g). Set A_log=0, dt_bias=0,
   then compute_g(a) = exp(-softplus(a)). We need exp(-softplus(a)) = exp(g),
   i.e. ``a = softplus_inverse(-g)``. Requires ``g <= 0`` (decay <= 1) — values
   above 0 are clamped (Path E cannot represent amplifying gates because the
   upstream parameterization is monotonically bounded by 1).
2. Beta: upstream beta = sigmoid(b). Pass ``b = logit(our_beta)``.
3. Scale: pre-multiply ``q`` by the FLA scale ``1/sqrt(K)`` before calling
   upstream (it doesn't scale internally).
"""

from __future__ import annotations

import math

import mlx.core as mx

from cppmega_v4.nn._external._mlx_lm_gated_delta_vendored import gated_delta_update as _upstream


def _softplus_inverse(y: mx.array) -> mx.array:
    """Inverse of softplus: log(exp(y) - 1). Requires y > 0; clamped to small +."""
    y_pos = mx.maximum(y, 1e-30)
    return mx.log(mx.maximum(mx.exp(y_pos) - 1.0, 1e-30))


def _logit(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Inverse of sigmoid: log(x / (1 - x)). Clamps to (eps, 1-eps)."""
    x = mx.clip(x, eps, 1.0 - eps)
    return mx.log(x) - mx.log(1.0 - x)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    g: mx.array,
    *,
    scale: float | None = None,
    initial_state: mx.array | None = None,
    output_final_state: bool = False,
):
    """Path E entry — same signature as naive_recurrent_gated_delta_rule."""
    # q/k: [B, T, H, K]; v: [B, T, H, V]; beta: [B, T, H]; g: [B, T, H] (log-decay)
    b_size, t_size, h_size = beta.shape
    dk = q.shape[-1]
    dv = v.shape[-1]
    # FLA convention: q is scaled by 1/sqrt(K) internally; upstream is not.
    fla_scale = scale if scale is not None else 1.0 / math.sqrt(dk)
    q_scaled = (q.astype(mx.float32) * fla_scale).astype(q.dtype)
    # Decay: g is log-decay; upstream representable range is decay <= 1, so g <= 0.
    g_clamped = mx.minimum(g.astype(mx.float32), -1e-6)
    a_synth = _softplus_inverse(-g_clamped)  # [B, T, H]
    # Beta: upstream applies sigmoid(b); pre-invert with logit.
    b_synth = _logit(beta.astype(mx.float32))  # [B, T, H]
    A_log = mx.zeros((h_size,), dtype=mx.float32)
    dt_bias = mx.zeros((h_size,), dtype=mx.float32)
    # Upstream state shape: [B, Hv, Dv, Dk]. FLA's "initial_state" is
    # [B, H, K, V] — transpose last two axes if provided.
    if initial_state is None:
        state = mx.zeros((b_size, h_size, dv, dk), dtype=mx.float32)
    else:
        state = mx.transpose(initial_state.astype(mx.float32), (0, 1, 3, 2))
    # Upstream Metal kernel needs Dk % 32 == 0 and Dv % 4 == 0. Fall back to
    # the ops path for smaller dims so the test suite can exercise Path E.
    use_kernel = (dk % 32 == 0) and (dv % 4 == 0)
    y, new_state = _upstream(
        q_scaled, k, v, a_synth, b_synth, A_log, dt_bias,
        state=state, mask=None, use_kernel=use_kernel, training=False,
    )
    final = None
    if output_final_state:
        # Convert back to FLA shape [B, H, K, V].
        final = mx.transpose(new_state, (0, 1, 3, 2))
    return y, final


__all__ = ["gated_delta_update"]
