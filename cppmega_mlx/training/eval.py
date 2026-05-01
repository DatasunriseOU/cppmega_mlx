"""Evaluation helpers for local MLX language-model smoke runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.training.loss import next_token_cross_entropy


LossFn = Callable[
    [nn.Module, LMTokenBatch | Mapping[str, mx.array] | mx.array],
    tuple[mx.array, mx.array],
]


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    ntokens: int
    batches: int
    seconds: float
    tokens_per_second: float


def evaluate_batches(
    model: nn.Module,
    batches: Iterable[LMTokenBatch | Mapping[str, mx.array] | mx.array],
    *,
    loss_fn: LossFn = next_token_cross_entropy,
) -> EvalMetrics:
    """Evaluate average next-token loss over an iterable of MLX batches."""

    model.eval()
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = mx.array(0.0, dtype=mx.float32)
    batch_count = 0

    start = time.perf_counter()
    for batch in batches:
        loss, ntokens = loss_fn(model, batch)
        ntokens = ntokens.astype(mx.float32)
        total_loss = total_loss + loss.astype(mx.float32) * ntokens
        total_tokens = total_tokens + ntokens
        batch_count += 1
        mx.eval(total_loss, total_tokens)

    if batch_count == 0:
        raise ValueError("evaluation requires at least one batch")

    mx.eval(total_tokens)
    tokens = int(total_tokens.item())
    if tokens <= 0:
        raise ValueError("evaluation produced zero tokens")

    avg_loss = total_loss / total_tokens
    mx.eval(avg_loss)
    elapsed = time.perf_counter() - start

    return EvalMetrics(
        loss=float(avg_loss.item()),
        ntokens=tokens,
        batches=batch_count,
        seconds=elapsed,
        tokens_per_second=tokens / elapsed if elapsed > 0 else float("inf"),
    )


__all__ = ["EvalMetrics", "evaluate_batches"]
