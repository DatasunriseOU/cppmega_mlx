from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cppmega_mlx.nn.engram import (
    CppMegaEngramBranch,
    EngramConfig,
    causal_depthwise_silu_conv1d,
    causal_local_average,
    parse_ngram_orders,
)


def to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


def _rand(shape: tuple[int, ...], seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _flat_grads(grads: dict[str, object]) -> dict[str, np.ndarray]:
    mx.eval(grads)
    return {name: np.array(value) for name, value in tree_flatten(grads)}


def test_parse_ngram_orders_matches_nanochat_default_and_deduping() -> None:
    assert parse_ngram_orders("2, 3, 2, 0, -1, 4") == (2, 3, 4)
    assert parse_ngram_orders(()) == (2, 3, 4)
    assert parse_ngram_orders([1, 3, 1]) == (1, 3)


def test_causal_local_average_matches_manual_padded_ngram_windows() -> None:
    x = mx.array([[[1.0], [3.0], [5.0], [7.0]]], dtype=mx.float32)

    out = causal_local_average(x, 3)

    expected = np.array([[[1.0 / 3.0], [4.0 / 3.0], [3.0], [5.0]]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_causal_local_average_respects_document_boundaries() -> None:
    x = mx.array([[[2.0], [4.0], [8.0], [16.0]]], dtype=mx.float32)
    doc_ids = mx.array([[0, 0, 1, 1]], dtype=mx.int32)

    out = causal_local_average(x, 2, doc_ids=doc_ids)

    expected = np.array([[[1.0], [3.0], [4.0], [12.0]]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_engram_branch_returns_hidden_shape_and_zero_init_output() -> None:
    mx.random.seed(11)
    branch = CppMegaEngramBranch(
        EngramConfig(hidden_size=16, ngram_orders=(2, 3), bottleneck_dim=4, gated=False)
    )
    x = _rand((2, 5, 16), seed=12)

    out = branch(x)

    assert out.shape == x.shape
    assert np.count_nonzero(to_numpy(out)) == 0


def test_engram_gated_branch_exposes_sigmoid_gate_and_zero_init_output() -> None:
    mx.random.seed(21)
    branch = CppMegaEngramBranch(
        EngramConfig(
            hidden_size=12,
            ngram_orders="2,3",
            bottleneck_dim=3,
            gated=True,
            gate_sqrt_compress=True,
        )
    )
    x = _rand((2, 4, 12), seed=22)

    features = branch.ngram_features(x)
    gate = branch.gate_values(x, features)
    out = branch(x)

    gate_np = to_numpy(gate)
    assert features.shape == (2, 4, 3)
    assert gate.shape == (2, 4, 1)
    assert np.all(gate_np >= 0.0)
    assert np.all(gate_np <= 1.0)
    assert out.shape == x.shape
    assert np.count_nonzero(to_numpy(out)) == 0


def test_causal_depthwise_silu_conv_path_is_finite_and_doc_aware() -> None:
    x = mx.array([[[1.0, -1.0], [2.0, -2.0], [100.0, -100.0]]], dtype=mx.float32)
    doc_ids = mx.array([[0, 1, 1]], dtype=mx.int32)
    weight = mx.ones((2, 2, 1), dtype=mx.float32)

    doc_aware = causal_depthwise_silu_conv1d(x, weight, doc_ids=doc_ids)
    plain = causal_depthwise_silu_conv1d(x, weight)
    mx.eval(doc_aware, plain)

    assert doc_aware.shape == x.shape
    assert np.isfinite(to_numpy(doc_aware)).all()
    assert not np.allclose(to_numpy(doc_aware[:, 1, :]), to_numpy(plain[:, 1, :]))


def test_engram_conv_branch_becomes_nonzero_when_projection_is_enabled() -> None:
    mx.random.seed(31)
    branch = CppMegaEngramBranch(
        EngramConfig(
            hidden_size=8,
            ngram_orders=(2,),
            bottleneck_dim=4,
            gated=False,
            conv_kernel=3,
        )
    )
    branch.out_proj.weight = mx.ones_like(branch.out_proj.weight) * 0.25
    x = _rand((2, 6, 8), seed=32)

    out = branch(x)

    assert out.shape == x.shape
    assert np.isfinite(to_numpy(out)).all()
    assert float(mx.sum(mx.abs(out)).item()) > 0.0


def test_engram_forward_backward_gradients_are_finite_and_reach_branch_weights() -> None:
    mx.random.seed(41)
    branch = CppMegaEngramBranch(
        EngramConfig(
            hidden_size=10,
            ngram_orders=(2, 3),
            bottleneck_dim=5,
            gated=True,
            conv_kernel=2,
        )
    )
    branch.value_proj.weight = mx.ones_like(branch.value_proj.weight) * 0.1
    x = _rand((2, 5, 10), seed=42)

    def loss_fn(module: CppMegaEngramBranch) -> mx.array:
        out = module(x)
        return mx.mean(mx.square(out))

    loss, grads = nn.value_and_grad(branch, loss_fn)(branch)
    grad_arrays = _flat_grads(grads)
    mx.eval(loss)

    assert np.isfinite(np.array(loss)).all()
    for name in ("in_proj.weight", "gate_key_proj.weight", "value_proj.weight", "conv_weight"):
        assert name in grad_arrays
        assert np.isfinite(grad_arrays[name]).all()
        assert np.max(np.abs(grad_arrays[name])) > 0.0


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"bottleneck_dim": -1}, "bottleneck_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"conv_kernel": -1}, "conv_kernel"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_engram_config_validates_constructor_args(kwargs: dict[str, object], error: str) -> None:
    config_kwargs = {
        "hidden_size": 8,
        "ngram_orders": "2,3,4",
        "bottleneck_dim": 0,
        "dropout": 0.0,
        "gated": False,
        "gate_sqrt_compress": False,
        "conv_kernel": 0,
        "eps": 1e-6,
    }
    config_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=error):
        EngramConfig(
            hidden_size=int(config_kwargs["hidden_size"]),
            ngram_orders=config_kwargs["ngram_orders"],
            bottleneck_dim=int(config_kwargs["bottleneck_dim"]),
            dropout=float(config_kwargs["dropout"]),
            gated=bool(config_kwargs["gated"]),
            gate_sqrt_compress=bool(config_kwargs["gate_sqrt_compress"]),
            conv_kernel=int(config_kwargs["conv_kernel"]),
            eps=float(config_kwargs["eps"]),
        )
