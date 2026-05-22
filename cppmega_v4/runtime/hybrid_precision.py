"""V7-D04: hybrid master-fp32-grad + bf16-forward helpers.

Best-practice mixed precision: params live in fp32 (master copy);
forward materialises a bf16 view (cast on the fly); gradients are
cast back to fp32 for optimiser apply.

  cast_for_forward(param, fwd_dtype) → fwd_dtype view
  cast_grads_to_master(grads, master_dtype=fp32) → fp32 grads
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def cast_for_forward(param: mx.array, fwd_dtype: mx.Dtype) -> mx.array:
    """Return a non-master view of `param` in fwd_dtype. The original
    fp32 master is preserved by the caller."""
    return param.astype(fwd_dtype)


def cast_grads_to_master(grads, master_dtype: mx.Dtype = mx.float32):
    """Walk a grad tree and cast every leaf to master_dtype."""
    return nn.utils.tree_map(
        lambda g: g.astype(master_dtype) if hasattr(g, "shape") else g,
        grads,
    )


def hybrid_step(*, master_params: dict, fwd_dtype: mx.Dtype,
                 loss_and_grad,
                 apply_gradients) -> dict:
    """One end-to-end hybrid mixed-precision step.

    Args:
        master_params: dict of name → fp32 mx.array master params
            (mutated in place by apply_gradients).
        fwd_dtype: e.g. mx.bfloat16.
        loss_and_grad: callable (fwd_params_dict) → (loss, grads).
        apply_gradients: callable (master_params, fp32_grads) → updated
            master_params dict.

    Returns the updated master params dict.
    """
    fwd_params = {k: cast_for_forward(v, fwd_dtype)
                  for k, v in master_params.items()}
    _, grads = loss_and_grad(fwd_params)
    grads_fp32 = cast_grads_to_master(grads, master_dtype=mx.float32)
    return apply_gradients(master_params, grads_fp32)


__all__ = [
    "cast_for_forward", "cast_grads_to_master", "hybrid_step",
]
