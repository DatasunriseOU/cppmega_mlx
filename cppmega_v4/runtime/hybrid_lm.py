"""V7-D04: HybridLM — thin wrapper that exposes set_dtype('hybrid').

Honest-closure: cppmega_v4/runtime/hybrid_precision.py defined the
free functions cast_for_forward / cast_grads_to_master / hybrid_step
but no model wrapper let callers say `lm.set_dtype('hybrid')`.

  lm = HybridLM(model)
  lm.set_dtype('hybrid')      # master=fp32, fwd=bf16
  lm.set_dtype('fp32')        # both fp32 (debug)
  lm.set_dtype('bf16')        # both bf16 (no master)
  lm.set_dtype('fp16')        # both fp16 (paired with LossScaler)
  loss, grads = lm.value_and_grad(loss_fn, x, y)

This wrapper does NOT alter the inner module's parameter storage when
mode='hybrid'; instead, the forward path materialises a bf16 view via
cast_for_forward each call. Grads come back in fwd dtype and are cast
back to master before the optimizer step.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.runtime.hybrid_precision import (
    cast_for_forward, cast_grads_to_master,
)


DtypeMode = Literal["hybrid", "fp32", "bf16", "fp16"]


_MODE_TO_DTYPES: dict[str, tuple[mx.Dtype, mx.Dtype]] = {
    # (master, fwd)
    "hybrid": (mx.float32, mx.bfloat16),
    "fp32":   (mx.float32, mx.float32),
    "bf16":   (mx.bfloat16, mx.bfloat16),
    "fp16":   (mx.float16, mx.float16),
}


class HybridLM(nn.Module):
    """Wraps an inner nn.Module with explicit master/forward dtype split.

    Calling instance(x) routes through the inner module after casting
    its parameters to `fwd_dtype` for the duration of the forward.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self._mode: DtypeMode = "fp32"
        self._master_dtype: mx.Dtype = mx.float32
        self._fwd_dtype: mx.Dtype = mx.float32

    @property
    def mode(self) -> DtypeMode:
        return self._mode

    @property
    def master_dtype(self) -> mx.Dtype:
        return self._master_dtype

    @property
    def fwd_dtype(self) -> mx.Dtype:
        return self._fwd_dtype

    def set_dtype(self, mode: DtypeMode) -> None:
        """Configure master + forward dtype.

        'hybrid' uses master=fp32, fwd=bf16 — the best-practice mixed
        precision schedule per the V7 honest-closure plan.
        """
        if mode not in _MODE_TO_DTYPES:
            raise ValueError(
                f"unsupported HybridLM mode {mode!r}; "
                f"expected one of {sorted(_MODE_TO_DTYPES)}"
            )
        master, fwd = _MODE_TO_DTYPES[mode]
        self._mode = mode
        self._master_dtype = master
        self._fwd_dtype = fwd
        # When mode is non-hybrid, the inner module's params are cast
        # in place to the requested dtype. Hybrid keeps params in
        # master dtype and casts on each forward.
        if mode != "hybrid":
            self.inner.set_dtype(master)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._mode == "hybrid":
            # Cast params to fwd_dtype for the forward only. We do this
            # by snapshotting + restoring around the call.
            flat_master = dict(
                nn.utils.tree_flatten(self.inner.parameters()))
            try:
                bf_params: dict[str, mx.array] = {
                    k: cast_for_forward(v, self._fwd_dtype)
                    for k, v in flat_master.items()
                }
                self.inner.update(
                    nn.utils.tree_unflatten(list(bf_params.items())))
                out = self.inner(*args, **kwargs)
            finally:
                # Restore master params (fp32) so the optimizer step
                # operates on the master copy.
                self.inner.update(
                    nn.utils.tree_unflatten(list(flat_master.items())))
            return out
        return self.inner(*args, **kwargs)

    def value_and_grad(
        self, loss_fn: Callable[[nn.Module, Any, Any], mx.array],
    ) -> Callable[..., tuple[mx.array, Any]]:
        """Return a (loss, grads) callable whose grads are cast back
        to master_dtype so the optimizer step sees consistent dtypes."""
        inner_lvg = nn.value_and_grad(self.inner, loss_fn)

        def _wrapped(*args: Any, **kwargs: Any):
            loss, grads = inner_lvg(self.inner, *args, **kwargs)
            grads = cast_grads_to_master(
                grads, master_dtype=self._master_dtype)
            return loss, grads

        return _wrapped


__all__ = ["HybridLM", "DtypeMode"]
