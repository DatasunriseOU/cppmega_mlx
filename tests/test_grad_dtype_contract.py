"""Contract tests: training-loop gradients must match parameter dtype.

cppmega CUDA explicitly disables ``--accumulate-allreduce-grads-in-fp32`` in
``cppmega/docs/gb10_local_memory_perf_2026_04_25.md:47-51`` so the bf16 weight
buffer is never paired with an fp32 grad shadow. cppmega.mlx mirrors that
policy: under MLX, ``nn.value_and_grad`` produces gradients in the same dtype
as the parameter being differentiated. These tests lock that behaviour in by
running one forward+backward pass through the ``local_gb10_quarter`` route at
three model dtypes (bf16, fp32, fp16) and asserting every leaf in
``tree_flatten(grads)`` matches the corresponding parameter dtype.

The fast tests use ``build_local_gb10_quarter_tiny_smoke_model`` to keep peak
memory under a few MiB. A separate production-shape test allocates the full
1.797B-parameter ``local_gb10_quarter`` model and runs the actual
``next_token_cut_cross_entropy`` train-path loss. It is opt-in via
``CPPMEGA_GRAD_DTYPE_PRODUCTION_SHAPE=1`` because the run requires roughly
3.3 GiB params + 3.3 GiB grads + activations on a Mac Studio M4 Max 128 GB.

Defensive helper :func:`assert_grad_dtype_matches_param_dtype` is exposed for
optional opt-in inside the production training loop via the
``STRICT_DTYPE_CONTRACT=1`` environment variable.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.recipes.model_factory import (
    build_local_gb10_quarter_tiny_smoke_model,
    local_gb10_quarter,
)
from cppmega_mlx.training.loss import next_token_cut_cross_entropy


_PRODUCTION_SKIP_ENV = "CPPMEGA_GRAD_DTYPE_PRODUCTION_SHAPE"


def _make_smoke_batch(model, *, seed: int = 0, batch_size: int = 2, seq_length: int = 16) -> LMTokenBatch:
    rng = np.random.default_rng(seed)
    tokens = mx.array(
        rng.integers(0, model.config.vocab_size, size=(batch_size, seq_length), dtype=np.int32)
    )
    attention_mask = mx.ones((batch_size, seq_length), dtype=mx.float32)
    return LMTokenBatch(tokens=tokens, attention_mask=attention_mask)


def _flatten_grad_dtypes(grads) -> dict[str, mx.Dtype]:
    """Return ``{path: dtype}`` for every mx.array leaf in ``grads``."""

    return {path: leaf.dtype for path, leaf in tree_flatten(grads) if isinstance(leaf, mx.array)}


def _flatten_param_dtypes(params) -> dict[str, mx.Dtype]:
    return {path: leaf.dtype for path, leaf in tree_flatten(params) if isinstance(leaf, mx.array)}


def assert_grad_dtype_matches_param_dtype(grads, params) -> None:
    """Defensive helper: every grad leaf must match the corresponding param dtype.

    Walks ``grads`` and the model's parameter pytree in parallel and raises
    ``AssertionError`` if any grad leaf is promoted to a different dtype than
    the parameter it is the gradient of. This mirrors cppmega CUDA's
    ``--accumulate-allreduce-grads-in-fp32 = false`` policy.
    """

    grad_dtypes = _flatten_grad_dtypes(grads)
    param_dtypes = _flatten_param_dtypes(params)

    missing = set(grad_dtypes) - set(param_dtypes)
    if missing:
        raise AssertionError(
            f"grad leaves {sorted(missing)} have no matching parameter; "
            "tree_flatten(grads) must match tree_flatten(params)"
        )

    mismatched: list[tuple[str, mx.Dtype, mx.Dtype]] = []
    for path, grad_dtype in grad_dtypes.items():
        param_dtype = param_dtypes[path]
        if grad_dtype != param_dtype:
            mismatched.append((path, param_dtype, grad_dtype))
    if mismatched:
        details = "; ".join(
            f"{path}: param={param_dt}, grad={grad_dt}"
            for path, param_dt, grad_dt in mismatched
        )
        raise AssertionError(
            "grad dtype must match param dtype (cppmega CUDA "
            "--accumulate-allreduce-grads-in-fp32=false policy); "
            f"mismatches: {details}"
        )


def _loss_fn(model, batch: LMTokenBatch) -> tuple[mx.array, mx.array]:
    return next_token_cut_cross_entropy(model, batch, chunk_rows=4)


def _build_smoke_model_at_dtype(dtype: mx.Dtype, *, seed: int = 0):
    mx.random.seed(seed)
    model = build_local_gb10_quarter_tiny_smoke_model()
    if dtype != mx.float32:
        model.set_dtype(dtype)
    return model


@pytest.mark.training
@pytest.mark.parametrize(
    "model_dtype",
    [
        pytest.param(mx.bfloat16, id="bfloat16"),
        pytest.param(mx.float32, id="float32"),
        pytest.param(mx.float16, id="float16"),
    ],
)
def test_grads_match_param_dtype_on_smoke_model(model_dtype: mx.Dtype) -> None:
    """``nn.value_and_grad`` must yield grads matching parameter dtype.

    For each supported model dtype we (a) build the tiny smoke variant of
    ``local_gb10_quarter``, (b) run one forward+backward via the train-path
    CCE loss, and (c) walk every leaf in ``tree_flatten(grads)`` to confirm
    the grad dtype equals ``model_dtype``. This is the contract that mirrors
    cppmega CUDA's ``--accumulate-allreduce-grads-in-fp32 = false`` policy.
    """

    model = _build_smoke_model_at_dtype(model_dtype)
    batch = _make_smoke_batch(model)

    (loss, ntokens), grads = nn.value_and_grad(model, _loss_fn)(model, batch)
    mx.eval(loss, ntokens, grads)

    grad_dtypes = _flatten_grad_dtypes(grads)
    assert grad_dtypes, "expected at least one grad leaf for the smoke model"
    mismatched = {path: dt for path, dt in grad_dtypes.items() if dt != model_dtype}
    assert not mismatched, (
        f"grads must be {model_dtype} when params are {model_dtype}; "
        f"mismatched leaves: {mismatched}"
    )

    # Defensive cross-check: the helper must accept this exact pair.
    assert_grad_dtype_matches_param_dtype(grads, model.parameters())


@pytest.mark.training
def test_assert_grad_dtype_matches_param_dtype_rejects_promoted_grads() -> None:
    """The helper must raise when any grad leaf is promoted off the param dtype.

    We build a bf16 smoke model, capture its real grads, then synthesize a
    promoted (fp32) copy of one leaf to confirm the helper detects it. This
    pins the helper's failure path, not just the happy path.
    """

    model = _build_smoke_model_at_dtype(mx.bfloat16)
    batch = _make_smoke_batch(model)

    (_loss, _ntokens), grads = nn.value_and_grad(model, _loss_fn)(model, batch)
    mx.eval(grads)

    flat = tree_flatten(grads)
    promoted_path, promoted_leaf = next(
        (path, leaf) for path, leaf in flat if isinstance(leaf, mx.array)
    )
    promoted = dict(flat)
    promoted[promoted_path] = promoted_leaf.astype(mx.float32)

    from mlx.utils import tree_unflatten

    promoted_grads = tree_unflatten(list(promoted.items()))
    with pytest.raises(AssertionError, match="grad dtype must match param dtype"):
        assert_grad_dtype_matches_param_dtype(promoted_grads, model.parameters())


@pytest.mark.training
def test_grad_buffer_size_equals_param_buffer_size_for_bf16() -> None:
    """A bf16 model's grad pytree must have the same byte count as its params.

    If grads were accidentally promoted to fp32 the gradient buffer would be
    2x the parameter buffer, which is exactly what cppmega CUDA's
    ``--accumulate-allreduce-grads-in-fp32 = false`` patch disables. We check
    this on the smoke model so the regression test runs in <5s; the
    production-shape variant lives below behind the
    ``CPPMEGA_GRAD_DTYPE_PRODUCTION_SHAPE`` env var.
    """

    model = _build_smoke_model_at_dtype(mx.bfloat16)
    batch = _make_smoke_batch(model)

    (_loss, _ntokens), grads = nn.value_and_grad(model, _loss_fn)(model, batch)
    mx.eval(grads)

    grad_bytes = sum(
        leaf.nbytes for _, leaf in tree_flatten(grads) if isinstance(leaf, mx.array)
    )
    param_bytes = sum(
        leaf.nbytes for _, leaf in tree_flatten(model.parameters()) if isinstance(leaf, mx.array)
    )
    assert grad_bytes == param_bytes, (
        f"grad buffer ({grad_bytes} B) must equal param buffer ({param_bytes} B) "
        f"under bf16; ratio={grad_bytes / max(param_bytes, 1):.3f} (expected 1.0)"
    )


@pytest.mark.training
@pytest.mark.skipif(
    os.environ.get(_PRODUCTION_SKIP_ENV) != "1",
    reason=(
        f"set {_PRODUCTION_SKIP_ENV}=1 to opt into the 1.797B-parameter "
        "local_gb10_quarter grad-dtype audit (peak ~7 GiB on Mac Studio M4 Max)"
    ),
)
def test_grads_match_bf16_on_full_local_gb10_quarter() -> None:
    """Full ``local_gb10_quarter`` model: bf16 weights must produce bf16 grads.

    Builds the production-shape model in bf16, runs one forward+backward via
    the same ``next_token_cut_cross_entropy`` loss the training script uses,
    and confirms every grad leaf is bf16. Reports the grad-buffer-vs-param-
    buffer GiB ratio so an accidental fp32 grad promotion (which would
    double the buffer) shows up as a 2x ratio.
    """

    mx.random.seed(2026)
    model = local_gb10_quarter()  # defaults to bf16 since df80703

    batch = LMTokenBatch(
        tokens=mx.array(
            np.random.default_rng(2026).integers(
                0, model.config.vocab_size, size=(1, 32), dtype=np.int32
            )
        ),
        attention_mask=mx.ones((1, 32), dtype=mx.float32),
    )

    (loss, ntokens), grads = nn.value_and_grad(model, _loss_fn)(model, batch)
    mx.eval(loss, ntokens, grads)

    grad_dtypes = _flatten_grad_dtypes(grads)
    mismatched = {path: dt for path, dt in grad_dtypes.items() if dt != mx.bfloat16}
    assert not mismatched, (
        f"production-shape grads must be bfloat16; mismatched leaves: {mismatched}"
    )

    grad_bytes = sum(
        leaf.nbytes for _, leaf in tree_flatten(grads) if isinstance(leaf, mx.array)
    )
    param_bytes = sum(
        leaf.nbytes for _, leaf in tree_flatten(model.parameters()) if isinstance(leaf, mx.array)
    )
    ratio = grad_bytes / max(param_bytes, 1)
    gib = 1024.0 ** 3
    print(
        f"\n[grad-dtype-contract] local_gb10_quarter bf16: "
        f"params={param_bytes / gib:.3f} GiB, grads={grad_bytes / gib:.3f} GiB, "
        f"ratio={ratio:.3f} (expected 1.000)"
    )
    assert ratio == pytest.approx(1.0, abs=1e-9), (
        f"grad-buffer / param-buffer ratio is {ratio:.3f}; bf16 grads should "
        "match bf16 params 1:1. A 2x ratio implies accidental fp32 grad "
        "promotion (cppmega CUDA --accumulate-allreduce-grads-in-fp32 violation)."
    )

    assert_grad_dtype_matches_param_dtype(grads, model.parameters())


__all__ = ["assert_grad_dtype_matches_param_dtype"]
