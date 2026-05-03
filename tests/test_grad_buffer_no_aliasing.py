"""Regression test: gradient buffers are 1:1 with trainable parameters.

After commit ``df80703`` removed param aliasing in :class:`HybridTinyBlock`
(``self.attention_block`` / ``mamba3_block`` / ``moe_block`` /
``m2rnn_block`` are now ``@property`` rather than module attributes),
``model.trainable_parameters()`` walks every parameter exactly once. The
parallel tree built by ``nn.value_and_grad`` should follow the same shape:
each parameter must have exactly one corresponding gradient buffer, and no
two paths in the gradient tree should reference the same underlying
``mx.array``.

This test exercises the smoke-sized profile so it runs in a few seconds;
the production receipt at the full ``local_gb10_quarter`` profile lives in
``scripts/audit_grad_buffer_reuse.py`` /
``bench/baselines/grad_buffer_audit.json``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.recipes.model_factory import build_local_gb10_quarter_tiny_smoke_model
from cppmega_mlx.training.loss import next_token_cut_cross_entropy


def _array_nbytes(value: mx.array) -> int:
    return int(value.size * value.dtype.size)


def _group_by_id(flat: list[tuple[str, mx.array]]) -> dict[int, list[str]]:
    by_id: dict[int, list[str]] = {}
    for name, arr in flat:
        by_id.setdefault(id(arr), []).append(name)
    return by_id


def test_grad_buffer_is_one_to_one_with_trainable_params_on_smoke_model() -> None:
    mx.random.seed(17)
    model = build_local_gb10_quarter_tiny_smoke_model()
    mx.eval(model.parameters())

    batch_size, seq_len = 1, 16
    rng = np.random.default_rng(17)
    tokens = mx.array(
        rng.integers(low=0, high=model.config.vocab_size, size=(batch_size, seq_len)).astype(
            np.int32
        )
    )
    batch = {"tokens": tokens, "attention_mask": mx.ones((batch_size, seq_len), dtype=mx.float32)}

    def loss_fn(m: nn.Module, b: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
        return next_token_cut_cross_entropy(m, b, chunk_rows=8)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    mx.eval(loss, ntokens, grads)

    flat_params = tree_flatten(model.trainable_parameters())
    flat_grads = tree_flatten(grads)

    # Param tree: no aliasing — each path must point at a unique mx.array.
    param_by_id = _group_by_id(flat_params)
    param_aliased = {pid: names for pid, names in param_by_id.items() if len(names) > 1}
    assert not param_aliased, (
        "trainable_parameters tree has aliased buffers (regression of df80703 fix); "
        f"aliased entries: {param_aliased}"
    )

    # Grad tree: no aliasing — value_and_grad must not return the same array
    # under multiple paths either.
    grad_by_id = _group_by_id(flat_grads)
    grad_aliased = {gid: names for gid, names in grad_by_id.items() if len(names) > 1}
    assert not grad_aliased, (
        "gradient tree has aliased buffers (regression of df80703 fix); "
        f"aliased entries: {grad_aliased}"
    )

    # 1:1 cardinality + matching numel/bytes (grads share param dtype after
    # the bf16 default landed in df80703).
    assert len(flat_grads) == len(flat_params), (
        f"grad tree has {len(flat_grads)} entries vs {len(flat_params)} param entries"
    )
    assert {n for n, _ in flat_grads} == {n for n, _ in flat_params}, (
        "grad tree paths diverge from trainable_parameters paths"
    )

    param_total_numel = sum(int(arr.size) for _, arr in flat_params)
    grad_total_numel = sum(int(arr.size) for _, arr in flat_grads)
    assert grad_total_numel == param_total_numel, (
        f"grad numel {grad_total_numel:,} != param numel {param_total_numel:,}"
    )

    param_total_bytes = sum(_array_nbytes(arr) for _, arr in flat_params)
    grad_total_bytes = sum(_array_nbytes(arr) for _, arr in flat_grads)
    assert grad_total_bytes == param_total_bytes, (
        f"grad bytes {grad_total_bytes:,} != param bytes {param_total_bytes:,}"
    )

    # Sanity: at least one grad is finite and non-zero so we know the
    # backward actually populated the tree (rather than returning zeros).
    nonzero = False
    for _, grad in flat_grads:
        if grad.size == 0:
            continue
        max_abs = float(mx.max(mx.abs(grad)).item())
        if max_abs > 0.0:
            nonzero = True
            break
    assert nonzero, "every grad is exactly zero — backward did not populate"
