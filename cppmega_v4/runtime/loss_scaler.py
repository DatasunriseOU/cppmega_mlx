"""V7-D03: loss scaling + overflow detection for fp16 training.

fp16 has a narrow dynamic range (~6e-5 to 6e4). Without a loss
scaler, gradients underflow silently or overflow into inf/nan
within a few steps. This module provides:

  * `LossScaler` with two modes:
      - static (`mode='static'`, `init_scale=2**16`): multiplies the
        loss by a fixed factor before the backward pass; on
        gradient overflow it just increments a counter without
        adjusting the scale.
      - dynamic (`mode='dynamic'`): halves the scale on overflow,
        doubles after `growth_interval` consecutive clean steps,
        bounded by `[min_scale, max_scale]`.
  * `unscale_and_check(grads)` divides each grad by the active scale
    and returns (unscaled_grads, overflow_detected).
  * Caller is responsible for skipping the optimizer.update() on
    overflow and calling `update(overflow=...)` after each step so
    the dynamic scaler tracks state.

Usage in a training loop:

    scaler = LossScaler(mode="dynamic")
    for step in range(N):
        loss, grads = lvg(model, x, y)
        loss = loss * scaler.scale            # scale the loss
        grads = scaler.scale_grads(grads)     # equivalent: scale grads
        unscaled, overflow = scaler.unscale_and_check(grads)
        if not overflow:
            opt.update(model, unscaled)
        scaler.update(overflow)               # adjust dynamic scale
"""

from __future__ import annotations

from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn


class LossScaler:
    """Loss / gradient scaler for fp16 training.

    Attributes:
        scale: current scale factor (mx scalar).
        overflow_count: total overflow events observed.
        clean_steps_since_overflow: monotonic counter used by dynamic
            mode to decide when to grow the scale.
    """

    def __init__(
        self,
        *,
        mode: Literal["static", "dynamic"] = "dynamic",
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 200,
        min_scale: float = 1.0,
        max_scale: float = 2.0 ** 24,
    ) -> None:
        if mode not in ("static", "dynamic"):
            raise ValueError(f"mode must be static|dynamic, got {mode}")
        if init_scale <= 0:
            raise ValueError("init_scale must be > 0")
        if growth_factor <= 1.0 or backoff_factor >= 1.0:
            raise ValueError(
                "growth_factor must be > 1 and backoff_factor must be < 1"
            )
        self.mode = mode
        self._scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = max(1, int(growth_interval))
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.overflow_count = 0
        self.clean_steps_since_overflow = 0

    @property
    def scale(self) -> float:
        return self._scale

    def scale_grads(self, grads: Any) -> Any:
        """Multiply each grad tensor by self.scale (in-place equivalent
        for tree). Used when scaling AFTER the backward pass — the
        more common pattern is to multiply the loss directly."""
        return nn.utils.tree_map(
            lambda g: g * self._scale if hasattr(g, "shape") else g,
            grads,
        )

    def unscale_and_check(self, grads: Any) -> tuple[Any, bool]:
        """Divide each grad by self.scale and check for inf/nan.

        Returns (unscaled_grads, overflow). When overflow is True the
        caller MUST skip the optimizer update for this step.
        """
        flat = dict(nn.utils.tree_flatten(grads))
        overflow = False
        unscaled: dict[str, Any] = {}
        for k, g in flat.items():
            if not hasattr(g, "shape"):
                unscaled[k] = g
                continue
            ug = g / self._scale
            # mlx has no isfinite — use the abs<inf trick.
            if bool(mx.any(mx.isnan(ug)).item()) or bool(
                    mx.any(mx.isinf(ug)).item()):
                overflow = True
            unscaled[k] = ug
        unscaled_tree = nn.utils.tree_unflatten(list(unscaled.items()))
        return unscaled_tree, overflow

    def update(self, overflow: bool) -> None:
        """Adjust the scale based on this step's overflow outcome."""
        if overflow:
            self.overflow_count += 1
            self.clean_steps_since_overflow = 0
            if self.mode == "dynamic":
                self._scale = max(
                    self.min_scale, self._scale * self.backoff_factor)
            return
        self.clean_steps_since_overflow += 1
        if (self.mode == "dynamic"
                and self.clean_steps_since_overflow
                >= self.growth_interval):
            self._scale = min(
                self.max_scale, self._scale * self.growth_factor)
            self.clean_steps_since_overflow = 0

    def snapshot(self) -> dict[str, Any]:
        """Reportable state — meant for extras / UI overlay."""
        return {
            "mode": self.mode,
            "scale": float(self._scale),
            "overflow_count": int(self.overflow_count),
            "clean_steps_since_overflow":
                int(self.clean_steps_since_overflow),
        }


__all__ = ["LossScaler"]
