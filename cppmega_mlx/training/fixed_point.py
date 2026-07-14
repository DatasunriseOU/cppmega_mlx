"""Truncated-BPTT + deep-supervision training helpers for the FPRM loop.

These helpers drive :class:`cppmega_mlx.nn.stable_loop.StableFixedPointLoop`
the way arXiv:2606.18206 prescribes for training:

* **Truncated-BPTT (window K).** The unrolled loop is split into windows of
  ``K`` iterations. Gradients flow only inside a window; between windows the
  state is detached (``mx.stop_gradient``) so the BPTT horizon stays at ``K``
  regardless of total unroll depth.
* **Deep supervision.** A loss is computed at the END of every window (each a
  full forward through the head), and the per-window losses are summed. This
  supervises intermediate fixed-point iterates, not just the final one.

The :func:`fpopt_step` helper exposes a single damped FPOPT inference update
for callers that want to drive the fixed-point iteration step-by-step.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from cppmega_mlx.nn.stable_loop import StableFixedPointLoop

# Head maps the loop state ``z`` -> logits (or any tensor the loss consumes).
Head = Callable[[mx.array], mx.array]
# Loss maps ``(logits, targets) -> scalar``.
LossFn = Callable[[mx.array, mx.array], mx.array]


def deep_supervision_loss(
    loop: StableFixedPointLoop,
    z0: mx.array,
    x: mx.array,
    head: Head,
    targets: mx.array,
    loss_fn: LossFn,
    ctx: object = None,
    *,
    window: int = 4,
    num_windows: int = 1,
) -> mx.array:
    """Summed deep-supervision loss over ``num_windows`` BPTT windows.

    Each window advances the loop by ``window`` iterations, computes the head +
    loss on the window-final state, and the per-window losses are summed. The
    state is detached between windows so back-prop within ``loss`` only spans
    one window (truncated-BPTT horizon == ``window``).

    Returns a scalar suitable for ``mx.value_and_grad``. NOTE: detaching with
    ``mx.stop_gradient`` happens INSIDE this function, so when differentiated
    the gradient only flows through the most recent window unless the caller
    re-enters per window (see :func:`truncated_bptt_step`, which differentiates
    each window separately and is the recommended training entry point).
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")

    z = z0
    total = mx.array(0.0)
    for w in range(num_windows):
        z = loop.forward(z, x, ctx, training_loops=window)
        logits = head(z)
        total = total + loss_fn(logits, targets)
        if w != num_windows - 1:
            z = mx.stop_gradient(z)
    return total


def truncated_bptt_step(
    loop: StableFixedPointLoop,
    params_module: mx.array | object,
    z0: mx.array,
    x: mx.array,
    head: Head,
    targets: mx.array,
    loss_fn: LossFn,
    optimizer: object,
    ctx: object = None,
    *,
    window: int = 4,
    num_windows: int = 1,
) -> dict[str, object]:
    """Run an online truncated-BPTT + deep-supervision update schedule.

    Strategy (matches the paper's truncated-BPTT-with-deep-supervision recipe):
    for each of ``num_windows`` windows we

    1. detach the incoming state (``mx.stop_gradient``) so BPTT spans only this
       window,
    2. differentiate ``loss(head(loop(z, K)), targets)`` w.r.t. the trainable
       parameters of ``params_module`` via ``mx.value_and_grad``,
    3. apply the optimizer update,
    4. carry the (detached) window-final state into the next window.

    ``params_module`` must be an ``nn.Module`` whose ``trainable_parameters()``
    are the things being optimised (typically the wrapping model so the head +
    loop + embeddings all update). ``optimizer`` is an
    ``mlx.optimizers.Optimizer``.

    ``x`` (and ``z0``) may each be either an ``mx.array`` or a zero-arg
    callable. The callable form rebuilds the tensor INSIDE the differentiated
    closure, which is required when the injected input is produced by trainable
    upstream modules (embedding/prelude) — passing a precomputed tensor would
    detach those modules from the gradient graph.

    This is an online schedule: each window contributes exactly one optimizer
    update, and the pre-update window endpoint is detached and carried into the
    next window. The endpoint is not recomputed after the update. Returns the
    summed observed loss, exact optimizer/loop counts, and final carried state.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if not hasattr(params_module, "trainable_parameters"):
        raise TypeError(
            "params_module must expose trainable_parameters() (an nn.Module)"
        )

    import mlx.nn as nn  # local import to avoid a hard module-level dependency

    if not isinstance(params_module, nn.Module):
        raise TypeError("params_module must be an mlx.nn.Module")

    window_losses: list[float] = []

    # ``x`` may be a tensor OR a zero-arg callable that REBUILDS the injected
    # input inside the differentiated closure. The callable form is what lets
    # embedding/prelude parameters receive gradients (otherwise a precomputed
    # tensor is a detached constant and those upstream weights never learn).
    x_is_builder = callable(x)

    def current_x() -> mx.array:
        return x() if x_is_builder else x  # type: ignore[operator]

    z = z0() if callable(z0) else z0

    def window_loss(state: mx.array) -> tuple[mx.array, mx.array]:
        # value_and_grad differentiates the parameters captured by its first
        # arg. ``state`` is a detached constant; ``x`` (when a builder) is
        # rebuilt here so its upstream parameters are inside the graph too.
        x_now = current_x()
        z_out = loop.forward(state, x_now, ctx, training_loops=window)
        logits = head(z_out)
        return loss_fn(logits, targets), z_out

    loss_and_grad = nn.value_and_grad(params_module, window_loss)

    total = 0.0
    for _ in range(num_windows):
        z = mx.stop_gradient(z)
        (loss_val, z_next), grads = loss_and_grad(z)
        optimizer.update(params_module, grads)
        mx.eval(params_module.parameters(), optimizer.state, loss_val, z_next)
        loss_float = float(loss_val.item())
        window_losses.append(loss_float)
        total += loss_float
        z = z_next

    final_state = mx.stop_gradient(z)

    return {
        "loss": total,
        "window_losses": window_losses,
        "num_windows": num_windows,
        "window": window,
        "bptt_horizon": window,
        "optimizer_updates": num_windows,
        "loop_iterations": window * num_windows,
        "final_state": final_state,
        "schedule": "online_per_window",
    }


def fpopt_step(
    loop: StableFixedPointLoop,
    z: mx.array,
    x: mx.array,
    ctx: object,
    *,
    eta: float,
) -> tuple[mx.array, float]:
    """Single FPOPT damped fixed-point update.

    Computes ``ftilde = f(z; x)`` then the damped update
    ``z_next = eta * ftilde + (1 - eta) * z`` and returns
    ``(z_next, relative_residual)`` where the residual is measured between the
    pre-update ``z`` and ``ftilde`` (the same quantity halting uses).
    """
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    f_tilde = loop.residual_map(z, x, ctx)
    residual = float(loop.relative_residual(z, f_tilde).item())
    z_next = eta * f_tilde + (1.0 - eta) * z
    if not bool(mx.all(mx.isfinite(z_next)).item()):
        raise FloatingPointError("fpopt_step.updated_state contains non-finite values")
    return z_next, residual


__all__ = [
    "deep_supervision_loss",
    "fpopt_step",
    "truncated_bptt_step",
]
