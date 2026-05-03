"""Minimal MLX train-step utilities."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.training.loss import next_token_cross_entropy
from cppmega_mlx.training.optimizers import make_adamw


LossFn = Callable[[nn.Module, LMTokenBatch | Mapping[str, mx.array] | mx.array], tuple[mx.array, mx.array]]

_STRICT_DTYPE_CONTRACT_ENV = "STRICT_DTYPE_CONTRACT"


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    ntokens: int
    seconds: float
    tokens_per_second: float


def _strict_dtype_contract_enabled() -> bool:
    return os.environ.get(_STRICT_DTYPE_CONTRACT_ENV) == "1"


def assert_grad_dtype_matches_param_dtype(grads, params) -> None:
    """Raise ``AssertionError`` if any grad leaf dtype differs from its param.

    Mirrors cppmega CUDA's ``--accumulate-allreduce-grads-in-fp32 = false``
    policy from ``cppmega/docs/gb10_local_memory_perf_2026_04_25.md:47-51``:
    bf16 weights must be paired with bf16 grads, never an fp32 grad shadow.
    Defensive only — disabled unless ``STRICT_DTYPE_CONTRACT=1`` is set.
    """

    grad_leaves = {
        path: leaf for path, leaf in tree_flatten(grads) if isinstance(leaf, mx.array)
    }
    param_leaves = {
        path: leaf for path, leaf in tree_flatten(params) if isinstance(leaf, mx.array)
    }
    mismatched: list[tuple[str, mx.Dtype, mx.Dtype]] = []
    for path, grad in grad_leaves.items():
        param = param_leaves.get(path)
        if param is None:
            continue
        if grad.dtype != param.dtype:
            mismatched.append((path, param.dtype, grad.dtype))
    if mismatched:
        details = "; ".join(
            f"{path}: param={param_dt}, grad={grad_dt}"
            for path, param_dt, grad_dt in mismatched
        )
        raise AssertionError(
            "STRICT_DTYPE_CONTRACT: grad dtype must match param dtype "
            "(cppmega CUDA --accumulate-allreduce-grads-in-fp32=false policy); "
            f"mismatches: {details}"
        )


def one_step_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    *,
    loss_fn: LossFn = next_token_cross_entropy,
) -> TrainStepResult:
    """Run one eager MLX AdamW-compatible training step."""

    model.train()
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    start = time.perf_counter()
    (loss, ntokens), grads = loss_and_grad(model, batch)
    if _strict_dtype_contract_enabled():
        assert_grad_dtype_matches_param_dtype(grads, model.parameters())
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss, ntokens)
    elapsed = time.perf_counter() - start

    tokens = int(ntokens.item())
    return TrainStepResult(
        loss=float(loss.item()),
        ntokens=tokens,
        seconds=elapsed,
        tokens_per_second=tokens / elapsed if elapsed > 0 else float("inf"),
    )


__all__ = [
    "TrainStepResult",
    "assert_grad_dtype_matches_param_dtype",
    "make_adamw",
    "one_step_train",
]
