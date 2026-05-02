"""Minimal MLX train-step utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.training.loss import next_token_cross_entropy
from cppmega_mlx.training.optimizers import make_adamw


LossFn = Callable[[nn.Module, LMTokenBatch | Mapping[str, mx.array] | mx.array], tuple[mx.array, mx.array]]


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    ntokens: int
    seconds: float
    tokens_per_second: float


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


__all__ = ["TrainStepResult", "make_adamw", "one_step_train"]
