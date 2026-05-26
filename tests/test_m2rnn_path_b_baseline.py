from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import os

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cppmega_mlx.recipes.model_factory import build_local_gb10_quarter_tiny_smoke_model
from cppmega_mlx.runtime.kernel_policy import clear_dispatch_log, get_dispatch_log


@contextmanager
def kernel_policy_env(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_forced_path_b_local_gb10_quarter_m2rnn_training_route_runs_baseline() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model(
        pattern="R",
        depth=1,
        dsa_a_layer_ranks=(),
        vocab_size=64,
    )
    inputs = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5, 6]], dtype=mx.int32)

    def loss_fn(params):
        model.update(params)
        logits = model(inputs)
        return mx.mean(
            nn.losses.cross_entropy(
                logits.reshape((-1, logits.shape[-1])),
                targets.reshape((-1,)),
                reduction="none",
            )
        )

    clear_dispatch_log()
    with kernel_policy_env(
        {
            "CPPMEGA_KERNEL_PATH": "path_b",
            "CPPMEGA_KERNEL_PATH__M2RNN": "path_b",
        }
    ):
        loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        mx.eval(loss, grads)

    assert np.isfinite(float(loss))
    flat_grads = tree_flatten(grads)
    assert flat_grads
    assert all(np.isfinite(np.array(grad)).all() for _, grad in flat_grads)
    m2rnn_dispatches = [
        entry for entry in get_dispatch_log() if entry["op_name"] == "m2rnn"
    ]
    assert m2rnn_dispatches
    assert m2rnn_dispatches[-1] == {
        "op_name": "m2rnn",
        "path": "path_b",
        "kernel_used": "reference_path_b_baseline",
    }
