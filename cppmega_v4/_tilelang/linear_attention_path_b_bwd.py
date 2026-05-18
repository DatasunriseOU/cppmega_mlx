"""GDN Path B forward + backward — fwd via fast Metal kernel, bwd via
mx.grad through Path A reference.

This is the *fwd-fast / bwd-correct* path, named ``path_b_fwd_path_a_bwd`` in
the spirit of mamba3's ``mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd``.
The fast hand-MSL kernel handles forward (3-10× over Path A), while
backward delegates to ``mx.grad`` through the FLA-naive reference so
training is at least *correct* even if not yet fully accelerated.

A future revision (tracked: cppmega_v4._tilelang.linear_attention_path_b_bwd_metal)
replaces the backward with a fused Metal kernel; until then this wrapper
unlocks full V4-stack training with measurable forward speedup.

API:
    forward_with_grad(q, k, v, beta, g) -> (o, h_last)
        Implemented as ``mx.custom_function`` so ``mx.grad`` traces through.
"""

from typing import Optional

import mlx.core as mx

from cppmega_v4._tilelang.linear_attention_path_b import gdn_forward_path_b
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)


@mx.custom_function
def gdn_apply_path_b(
    q: mx.array, k: mx.array, v: mx.array, beta: mx.array, g: mx.array,
) -> mx.array:
    """Forward-only call returning y (no state). Differentiable via custom VJP.

    The fast Path B Metal kernel produces y; the VJP differentiates the
    Path A reference (algebraically identical), so gradients are correct.
    """
    y, _ = gdn_forward_path_b(q, k, v, beta, g, output_final_state=False)
    return y


@gdn_apply_path_b.vjp
def _gdn_apply_path_b_vjp(
    primals: tuple,
    cotangent: mx.array,
    output: mx.array,
) -> tuple:
    """Backward via mx.grad through the FLA naive reference.

    Path A reference is algebraically identical to Path B (Path B is just
    a faster MSL implementation of the same recurrence), so grads from
    Path A are correct grads for Path B. This is the "path_b_fwd_path_a_bwd"
    pattern — see mamba3_path_c's analogous ``path_c_fwd_path_b_bwd`` mode.
    """
    del output  # unused — VJP re-runs forward through Path A
    q, k, v, beta, g = primals

    def _loss_proxy(q_, k_, v_, beta_, g_):
        y, _ = naive_recurrent_gated_delta_rule(q_, k_, v_, beta_, g_)
        return (y * cotangent).sum()

    grad_fn = mx.grad(_loss_proxy, argnums=(0, 1, 2, 3, 4))
    return grad_fn(q, k, v, beta, g)


__all__ = ["gdn_apply_path_b"]
