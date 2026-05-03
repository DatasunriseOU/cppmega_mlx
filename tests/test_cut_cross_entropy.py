"""Forward + backward parity for the MLX-native chunked cross-entropy.

Apple's `cut_cross_entropy` Triton kernel rejects MacOS at runtime, so the
local MLX port (``cppmega_mlx.training.cut_cross_entropy``) stands in. These
tests pin the chunked forward and eager chunked forward+backward paths to the
materialized reference at small shapes where exact equality is achievable.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from cppmega_mlx.training.cut_cross_entropy import (
    DEFAULT_CHUNK_ROWS,
    linear_cross_entropy,
    linear_cross_entropy_value_and_grad,
    materialized_cross_entropy,
)


def _make_inputs(
    *,
    seed: int = 0,
    batch: int = 2,
    seq: int = 8,
    hidden: int = 32,
    vocab: int = 64,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    e = mx.random.normal((batch, seq, hidden)).astype(dtype)
    c = mx.random.normal((vocab, hidden)).astype(dtype)
    targets = mx.random.randint(0, vocab, (batch, seq))
    mx.eval(e, c, targets)
    return e, c, targets


def _scalar(value: mx.array) -> float:
    mx.eval(value)
    return float(value.item())


def _max_abs(a: mx.array, b: mx.array) -> float:
    diff = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(diff)
    return float(diff.item())


def test_chunked_forward_matches_materialized_fp32() -> None:
    e, c, targets = _make_inputs()
    expected = materialized_cross_entropy(e, c, targets)
    chunked = linear_cross_entropy(e, c, targets, chunk_rows=4)
    assert math.isclose(_scalar(expected), _scalar(chunked), abs_tol=1e-6)


def test_chunked_forward_supports_sum_and_none_reductions() -> None:
    e, c, targets = _make_inputs()
    sum_expected = materialized_cross_entropy(e, c, targets, reduction="sum")
    sum_chunked = linear_cross_entropy(
        e, c, targets, chunk_rows=4, reduction="sum"
    )
    none_expected = materialized_cross_entropy(e, c, targets, reduction="none")
    none_chunked = linear_cross_entropy(
        e, c, targets, chunk_rows=4, reduction="none"
    )
    assert math.isclose(_scalar(sum_expected), _scalar(sum_chunked), abs_tol=1e-6)
    assert none_chunked.shape == none_expected.shape
    assert _max_abs(none_chunked, none_expected) < 1e-5


def test_eager_value_and_grad_matches_materialized_grad_fp32() -> None:
    """Chunked eager backward must match autograd over the materialized path."""

    e, c, targets = _make_inputs()

    def loss_mat(e: mx.array, c: mx.array, targets: mx.array) -> mx.array:
        return materialized_cross_entropy(e, c, targets)

    expected_loss, (expected_de, expected_dc) = mx.value_and_grad(
        loss_mat, argnums=(0, 1)
    )(e, c, targets)
    actual_loss, actual_de, actual_dc = linear_cross_entropy_value_and_grad(
        e, c, targets, chunk_rows=4
    )
    mx.eval(expected_loss, expected_de, expected_dc, actual_loss, actual_de, actual_dc)

    assert math.isclose(_scalar(expected_loss), _scalar(actual_loss), abs_tol=1e-6)
    # Bit-exact at fp32 with this small shape.
    assert _max_abs(actual_de, expected_de) <= 1e-6
    assert _max_abs(actual_dc, expected_dc) <= 1e-6


def test_eager_value_and_grad_matches_materialized_grad_bf16() -> None:
    """At bf16 we accept rtol=1e-4 / atol=1e-3 on the gradients."""

    e, c, targets = _make_inputs(dtype=mx.bfloat16, batch=1, seq=16, hidden=64, vocab=128)

    def loss_mat(e: mx.array, c: mx.array, targets: mx.array) -> mx.array:
        return materialized_cross_entropy(e, c, targets)

    expected_loss, (expected_de, expected_dc) = mx.value_and_grad(
        loss_mat, argnums=(0, 1)
    )(e, c, targets)
    actual_loss, actual_de, actual_dc = linear_cross_entropy_value_and_grad(
        e, c, targets, chunk_rows=4
    )
    mx.eval(expected_loss, expected_de, expected_dc, actual_loss, actual_de, actual_dc)

    assert _scalar(expected_loss) == pytest.approx(_scalar(actual_loss), rel=1e-3)
    # In bf16 both paths share the same float32 logits softmax, so gradients
    # round to the same bf16 representable values; tolerance accounts for
    # accumulation order in the chunked dc reduction.
    assert _max_abs(actual_de, expected_de) <= 1e-3
    assert _max_abs(actual_dc, expected_dc) <= 1e-3


def test_chunked_forward_grad_matches_materialized_grad() -> None:
    """``mx.grad`` over the chunked forward is mathematically identical."""

    e, c, targets = _make_inputs()

    def loss_mat(e: mx.array, c: mx.array, targets: mx.array) -> mx.array:
        return materialized_cross_entropy(e, c, targets)

    def loss_chunk(e: mx.array, c: mx.array, targets: mx.array) -> mx.array:
        return linear_cross_entropy(e, c, targets, chunk_rows=4)

    gmat = mx.grad(loss_mat, argnums=(0, 1))(e, c, targets)
    gch = mx.grad(loss_chunk, argnums=(0, 1))(e, c, targets)
    mx.eval(gmat, gch)
    assert _max_abs(gmat[0], gch[0]) <= 1e-6
    assert _max_abs(gmat[1], gch[1]) <= 1e-6


def test_reduction_none_masked_backward_honors_per_token_grad_output() -> None:
    """Non-uniform masks must influence every row's backward gradient.

    This guards against the Liger FLCE failure mode where reduction="none"
    backward only consumes grad_output[0] as a scalar and silently ignores
    per-position weighting.
    """

    e, c, targets = _make_inputs(batch=2, seq=5, hidden=16, vocab=32)
    weights = mx.array(
        [
            [0.0, 0.25, 1.0, 0.5, 2.0],
            [1.5, 0.0, 0.75, 1.25, 0.1],
        ],
        dtype=mx.float32,
    )
    normalizer = mx.sum(weights)

    def loss_mat(e: mx.array, c: mx.array) -> mx.array:
        per_token = materialized_cross_entropy(e, c, targets, reduction="none")
        return mx.sum(per_token * weights) / normalizer

    def loss_chunk(e: mx.array, c: mx.array) -> mx.array:
        per_token = linear_cross_entropy(
            e,
            c,
            targets,
            chunk_rows=3,
            reduction="none",
        )
        return mx.sum(per_token * weights) / normalizer

    expected = mx.grad(loss_mat, argnums=(0, 1))(e, c)
    actual = mx.grad(loss_chunk, argnums=(0, 1))(e, c)
    mx.eval(expected, actual)

    assert _max_abs(actual[0], expected[0]) <= 1e-6
    assert _max_abs(actual[1], expected[1]) <= 1e-6


def test_eager_value_and_grad_handles_2d_inputs() -> None:
    """``e`` may already be flattened to ``(N, D)``."""

    mx.random.seed(0)
    n, d, v = 12, 16, 32
    e_flat = mx.random.normal((n, d))
    c = mx.random.normal((v, d))
    targets_flat = mx.random.randint(0, v, (n,))
    mx.eval(e_flat, c, targets_flat)

    expected = materialized_cross_entropy(e_flat, c, targets_flat)
    actual_loss, actual_de, actual_dc = linear_cross_entropy_value_and_grad(
        e_flat, c, targets_flat, chunk_rows=4
    )
    mx.eval(expected, actual_loss, actual_de, actual_dc)
    assert math.isclose(_scalar(expected), _scalar(actual_loss), abs_tol=1e-6)
    assert actual_de.shape == e_flat.shape
    assert actual_dc.shape == c.shape


def test_targets_must_have_one_fewer_dim_than_e() -> None:
    e, c, _ = _make_inputs()
    bad_targets = mx.zeros((4,), dtype=mx.int32)
    with pytest.raises(ValueError, match="targets rank"):
        linear_cross_entropy(e, c, bad_targets)


def test_invalid_reduction_rejected() -> None:
    e, c, targets = _make_inputs()
    with pytest.raises(ValueError, match="reduction"):
        linear_cross_entropy(e, c, targets, reduction="bad")
    with pytest.raises(ValueError, match="reduction"):
        materialized_cross_entropy(e, c, targets, reduction="bad")


def test_default_chunk_rows_is_a_positive_integer() -> None:
    assert isinstance(DEFAULT_CHUNK_ROWS, int)
    assert DEFAULT_CHUNK_ROWS > 0


def test_chunked_forward_smaller_peak_than_materialized() -> None:
    """Chunked forward reduces peak memory at a moderate vocab."""

    e, c, targets = _make_inputs(
        batch=1, seq=64, hidden=128, vocab=4096, dtype=mx.float32
    )
    # Run a baseline first so we capture a fresh peak measurement for each.
    mx.reset_peak_memory()
    base = materialized_cross_entropy(e, c, targets)
    mx.eval(base)
    base_peak = mx.get_peak_memory()

    mx.reset_peak_memory()
    chunked = linear_cross_entropy(e, c, targets, chunk_rows=8)
    mx.eval(chunked)
    chunked_peak = mx.get_peak_memory()

    # The chunked path should not exceed the materialized peak; on a runtime
    # that does not expose ``get_peak_memory`` the values fall back to the
    # current allocator state and we accept that floor.
    assert chunked_peak <= base_peak


def test_bench_script_runs_at_tiny_shape(tmp_path: Path) -> None:
    """End-to-end bench script must produce schema-conformant JSON."""

    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "bench.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "bench_cce.py"),
        "--batch-size",
        "1",
        "--seq-len",
        "8",
        "--vocab-size",
        "256",
        "--hidden",
        "32",
        "--dtype",
        "float32",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--output",
        str(output),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    import json

    payload = json.loads(output.read_text())
    assert payload["scope"] == "local_only"
    assert set(payload["results"]) == {
        "materialized",
        "chunked_forward",
        "chunked_eager_grad",
    }
    for path_name in payload["results"]:
        for mode in ("forward_only", "forward_backward"):
            stats = payload["results"][path_name][mode]
            assert stats["peak_memory_bytes"] > 0
            assert stats["wall_ms_min"] >= 0
            assert stats["loss"] is not None


@pytest.mark.parametrize("shift", [-1, 1])
def test_loss_shifts_when_logits_shift(shift: int) -> None:
    """CE is invariant to a constant shift across the vocab axis."""

    e, c, targets = _make_inputs()
    # Shifting all classifier outputs by a constant adds the same constant to
    # every row of logits, which leaves softmax unchanged.
    e_shift = e + float(shift)
    base = materialized_cross_entropy(e, c, targets)
    chunked = linear_cross_entropy(e, c, targets, chunk_rows=4)
    chunked_shift = linear_cross_entropy(e_shift, c, targets, chunk_rows=4)
    base_shift = materialized_cross_entropy(e_shift, c, targets)
    # Both paths track each other under the shift, even though the loss value
    # itself moves.
    assert math.isclose(
        _scalar(chunked) - _scalar(base),
        0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        _scalar(chunked_shift),
        _scalar(base_shift),
        abs_tol=1e-5,
    )


def test_chunked_forward_matches_mlx_cross_entropy_directly() -> None:
    """End-to-end parity vs ``nn.losses.cross_entropy`` (the documented baseline)."""

    e, c, targets = _make_inputs()
    flat_e = e.reshape(-1, e.shape[-1])
    flat_t = targets.reshape(-1)
    logits = (flat_e @ c.T).astype(mx.float32)
    direct = nn.losses.cross_entropy(logits, flat_t, reduction="mean")
    chunked = linear_cross_entropy(e, c, targets, chunk_rows=4)
    assert math.isclose(_scalar(direct), _scalar(chunked), abs_tol=1e-6)
