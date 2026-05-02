"""Repo-local optimizer helpers for MLX training."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.optimizers as optim


ADAMW_FP32_MOMENTS_CLASS = "cppmega_mlx.training.optimizers.AdamWFP32Moments"
ADAMW_FP32_MOMENTS_SOURCE = "cppmega_mlx.training.optimizers.make_adamw"
ADAMW_BASE_CLASS = "mlx.optimizers.AdamW"
ADAMW_MOMENT_STATE_KEYS = ("m", "v")


class AdamWFP32Moments(optim.AdamW):
    """AdamW that keeps moment state in fp32 while preserving parameter dtype."""

    def init_single(self, parameter: mx.array, state: dict[str, Any]) -> None:
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, Any],
    ) -> mx.array:
        lr = self.learning_rate.astype(mx.float32)
        decayed_parameter = parameter.astype(mx.float32) * (1 - lr * self.weight_decay)
        updated = optim.Adam.apply_single(
            self,
            gradient.astype(mx.float32),
            decayed_parameter,
            state,
        )
        return updated.astype(parameter.dtype)


def make_adamw(
    *,
    learning_rate: float | Callable[[mx.array], mx.array] = 1e-3,
    weight_decay: float = 0.01,
    betas: list[float] | None = None,
    eps: float = 1e-8,
    bias_correction: bool = False,
) -> AdamWFP32Moments:
    """Construct the repo default AdamW with fp32 moments for bf16 training."""

    return AdamWFP32Moments(
        learning_rate=learning_rate,
        betas=[0.9, 0.999] if betas is None else betas,
        eps=eps,
        weight_decay=weight_decay,
        bias_correction=bias_correction,
    )


def collect_adamw_moment_dtypes(state: Any) -> dict[str, str]:
    moment_dtypes: dict[str, str] = {}

    def walk(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk((*path, str(key)), item)
            return
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk((*path, str(index)), item)
            return
        if isinstance(value, mx.array) and path and path[-1] in ADAMW_MOMENT_STATE_KEYS:
            moment_dtypes["/".join(path)] = dtype_name(value)

    walk((), state)
    return moment_dtypes


def adamw_moment_dtypes_ok(
    state: Any,
    *,
    required_dtype: str = "float32",
) -> bool:
    moment_dtypes = collect_adamw_moment_dtypes(state)
    return bool(moment_dtypes) and all(
        dtype == required_dtype for dtype in moment_dtypes.values()
    )


def dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", value)
    return str(dtype).removeprefix("mlx.core.")


__all__ = [
    "ADAMW_BASE_CLASS",
    "ADAMW_FP32_MOMENTS_CLASS",
    "ADAMW_FP32_MOMENTS_SOURCE",
    "ADAMW_MOMENT_STATE_KEYS",
    "AdamWFP32Moments",
    "adamw_moment_dtypes_ok",
    "collect_adamw_moment_dtypes",
    "dtype_name",
    "make_adamw",
]
