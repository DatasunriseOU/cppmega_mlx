"""Minimal MLX train-step utilities."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch
from cppmega_mlx.training.loss import next_token_cross_entropy
from cppmega_mlx.training.optimizers import make_adamw


LossFn = Callable[[nn.Module, LMTokenBatch | Mapping[str, mx.array] | mx.array], tuple[mx.array, mx.array]]

_STRICT_DTYPE_CONTRACT_ENV = "STRICT_DTYPE_CONTRACT"
# PR-2 (FUSED-PIPELINE-ROADMAP §4/§5): gradient accumulation in the loop.
# Default OFF (grad_accum_steps=1) so behavior is byte-identical to the prior
# single-eval step unless explicitly opted in. ``GRAD_ACCUM_STEPS`` env var
# provides a process-wide default for callers that do not pass the kwarg.
_GRAD_ACCUM_STEPS_ENV = "GRAD_ACCUM_STEPS"


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


def _resolve_grad_accum_steps(grad_accum_steps: int | None) -> int:
    """Resolve the microbatch count from kwarg then env, default 1 (OFF)."""

    if grad_accum_steps is None:
        env_value = os.environ.get(_GRAD_ACCUM_STEPS_ENV)
        grad_accum_steps = int(env_value) if env_value not in (None, "") else 1
    if not isinstance(grad_accum_steps, int) or grad_accum_steps < 1:
        raise ValueError(
            f"grad_accum_steps must be a positive integer, got {grad_accum_steps!r}"
        )
    return grad_accum_steps


def _microbatch_slices(batch_size: int, num_micro: int) -> list[tuple[int, int]]:
    """Split ``batch_size`` rows into ``num_micro`` contiguous, near-equal spans.

    Raises if the batch is too small to provide one row per microbatch — a
    silent no-op split would mask a misconfigured ``grad_accum_steps`` (RULE #1).
    """

    if num_micro > batch_size:
        raise ValueError(
            f"grad_accum_steps={num_micro} exceeds batch size {batch_size}; "
            "each microbatch needs at least one row"
        )
    base, extra = divmod(batch_size, num_micro)
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(num_micro):
        length = base + (1 if i < extra else 0)
        spans.append((start, start + length))
        start += length
    return spans


def _slice_lm_batch(batch: LMTokenBatch, start: int, stop: int) -> LMTokenBatch:
    """Return a row-slice ``[start:stop)`` of every per-example array field."""

    updates: dict[str, mx.array] = {}
    for name, value in vars(batch).items():
        if isinstance(value, mx.array) and value.shape and value.shape[0] == batch.tokens.shape[0]:
            updates[name] = value[start:stop]
    return replace(batch, **updates)


def _accumulate_grads_token_weighted(
    accumulator: dict | None,
    grads: dict,
    weight: mx.array,
) -> dict:
    """Add ``weight * grads`` into ``accumulator`` (token-weighted running sum).

    Token weighting makes the accumulated gradient numerically equal to the
    full-batch gradient: the per-microbatch loss is a token-mean, so the
    full-batch grad is ``sum_i (n_i / N) * grad_i`` by linearity.
    """

    weighted = tree_map(
        lambda g: (g * weight).astype(g.dtype) if isinstance(g, mx.array) else g,
        grads,
    )
    if accumulator is None:
        return weighted
    return tree_map(
        lambda acc, g: acc + g if isinstance(acc, mx.array) else acc,
        accumulator,
        weighted,
    )


def one_step_train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    *,
    loss_fn: LossFn = next_token_cross_entropy,
    grad_accum_steps: int | None = None,
    clear_cache: bool = False,
) -> TrainStepResult:
    """Run one eager MLX AdamW-compatible training step.

    ``grad_accum_steps`` (default 1, also settable via the ``GRAD_ACCUM_STEPS``
    env var) splits the batch into N microbatches and accumulates token-weighted
    grads with an ``mx.eval(grads)`` free-barrier between microbatches, dropping
    each microbatch's forward/backward graph before the next is built. This
    collapses peak activation memory from full-batch to ~one microbatch while
    remaining numerically equivalent to the full-batch step (FUSED-PIPELINE
    ROADMAP §4/§5, PR-2). With N=1 the path is byte-identical to the prior step.
    """

    model.train()
    num_micro = _resolve_grad_accum_steps(grad_accum_steps)

    start = time.perf_counter()
    if num_micro == 1:
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        (loss, ntokens), grads = loss_and_grad(model, batch)
        if _strict_dtype_contract_enabled():
            assert_grad_dtype_matches_param_dtype(grads, model.parameters())
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss, ntokens)
        elapsed = time.perf_counter() - start
        tokens = int(ntokens.item())
        if clear_cache:
            mx.clear_cache()
        return TrainStepResult(
            loss=float(loss.item()),
            ntokens=tokens,
            seconds=elapsed,
            tokens_per_second=tokens / elapsed if elapsed > 0 else float("inf"),
        )

    lm_batch = ensure_lm_batch(batch)
    batch_size = int(lm_batch.tokens.shape[0])
    spans = _microbatch_slices(batch_size, num_micro)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Pass 1: total live-token count across the whole batch, used to
    # token-weight each microbatch grad so the accumulated grad equals the
    # full-batch grad. ``target_mask`` is pure data (no forward graph), so this
    # costs nothing in activation memory.
    total_ntokens = lm_batch.target_mask.sum().astype(mx.float32)
    mx.eval(total_ntokens)
    total_tokens_int = int(total_ntokens.item())
    if total_tokens_int <= 0:
        raise ValueError("grad accumulation requires at least one live target token")

    accumulator: dict | None = None
    weighted_loss = mx.array(0.0, dtype=mx.float32)
    for span_start, span_stop in spans:
        micro = _slice_lm_batch(lm_batch, span_start, span_stop)
        (micro_loss, micro_ntok), micro_grads = loss_and_grad(model, micro)
        weight = (micro_ntok.astype(mx.float32) / total_ntokens)
        accumulator = _accumulate_grads_token_weighted(accumulator, micro_grads, weight)
        weighted_loss = weighted_loss + micro_loss.astype(mx.float32) * weight
        # Free-barrier: force the accumulator + running loss, then drop the
        # microbatch's forward/backward graph before building the next one.
        mx.eval(accumulator, weighted_loss)
        del micro_grads, micro, micro_loss, micro_ntok

    if accumulator is None:
        raise RuntimeError("grad accumulation produced no gradients")
    if _strict_dtype_contract_enabled():
        assert_grad_dtype_matches_param_dtype(accumulator, model.parameters())
    optimizer.update(model, accumulator)
    mx.eval(model.parameters(), optimizer.state, weighted_loss)
    elapsed = time.perf_counter() - start

    if clear_cache:
        mx.clear_cache()
    return TrainStepResult(
        loss=float(weighted_loss.item()),
        ntokens=total_tokens_int,
        seconds=elapsed,
        tokens_per_second=total_tokens_int / elapsed if elapsed > 0 else float("inf"),
    )


__all__ = [
    "TrainStepResult",
    "assert_grad_dtype_matches_param_dtype",
    "make_adamw",
    "one_step_train",
]
