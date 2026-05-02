from __future__ import annotations

import math

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cppmega_mlx.nn.mhc import (
    CppMegaManifoldBranchMixer,
    ManifoldBranchMixerConfig,
    sinkhorn_normalize,
)


def _rand(shape: tuple[int, ...], seed: int, *, dtype: mx.Dtype = mx.float32) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape, dtype=np.float32), dtype=dtype)


def _to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


def _flat_grads(grads: dict[str, object]) -> dict[str, np.ndarray]:
    mx.eval(grads)
    return {name: np.array(value) for name, value in tree_flatten(grads)}


def test_sinkhorn_normalize_produces_doubly_stochastic_transport() -> None:
    raw = _rand((2, 4, 4), seed=1) * 3.0

    transport = sinkhorn_normalize(raw, iters=20, epsilon=1e-6)
    row_sums = mx.sum(transport, axis=-1)
    col_sums = mx.sum(transport, axis=-2)
    mx.eval(transport, row_sums, col_sums)

    assert transport.dtype == mx.float32
    np.testing.assert_allclose(_to_numpy(row_sums), np.ones((2, 4)), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(_to_numpy(col_sums), np.ones((2, 4)), atol=1e-5, rtol=1e-5)
    assert np.isfinite(_to_numpy(transport)).all()


def test_manifold_branch_mixer_returns_weighted_hidden_shape() -> None:
    mx.random.seed(2)
    mixer = CppMegaManifoldBranchMixer(
        ManifoldBranchMixerConfig(hidden_size=8, sinkhorn_iters=10, temperature=0.7)
    )
    branches = [_rand((2, 5, 8), seed=3 + idx) for idx in range(3)]

    out = mixer(branches)
    weights = mixer.routing_weights(branches)
    mx.eval(out, weights)

    assert out.shape == (2, 5, 8)
    assert weights.shape == (2, 3)
    np.testing.assert_allclose(_to_numpy(mx.sum(weights, axis=-1)), np.ones(2), atol=1e-5)
    assert np.isfinite(_to_numpy(out)).all()
    assert np.isfinite(_to_numpy(weights)).all()


def test_two_branch_path_uses_nonuniform_softmax_weights() -> None:
    mx.random.seed(4)
    mixer = CppMegaManifoldBranchMixer(ManifoldBranchMixerConfig(hidden_size=6, temperature=0.5))
    branch_a = mx.ones((1, 3, 6), dtype=mx.float32)
    branch_b = mx.zeros((1, 3, 6), dtype=mx.float32)

    weights = mixer.routing_weights([branch_a, branch_b])
    out = mixer([branch_a, branch_b])
    mx.eval(weights, out)

    assert weights.shape == (1, 2)
    np.testing.assert_allclose(_to_numpy(mx.sum(weights, axis=-1)), np.ones(1), atol=1e-6)
    assert np.isfinite(_to_numpy(weights)).all()
    assert not np.allclose(_to_numpy(weights), np.array([[0.5, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(_to_numpy(out), np.broadcast_to(_to_numpy(weights)[0, 0], (1, 3, 6)))


def test_single_branch_returns_input_but_still_validates_shape() -> None:
    mixer = CppMegaManifoldBranchMixer(ManifoldBranchMixerConfig(hidden_size=4))
    x = _rand((2, 3, 4), seed=5)

    assert mixer([x]) is x

    with pytest.raises(ValueError, match="hidden size"):
        mixer([_rand((2, 3, 5), seed=6)])


def test_mhc_forward_backward_gradients_are_finite() -> None:
    mx.random.seed(7)
    mixer = CppMegaManifoldBranchMixer(
        ManifoldBranchMixerConfig(hidden_size=10, sinkhorn_iters=12, temperature=0.9)
    )
    branches = [_rand((2, 4, 10), seed=8 + idx) for idx in range(3)]
    target = _rand((2, 4, 10), seed=11)

    def loss_fn(module: CppMegaManifoldBranchMixer) -> mx.array:
        out = module(branches)
        return mx.mean(mx.square(out - target))

    loss, grads = nn.value_and_grad(mixer, loss_fn)(mixer)
    grad_arrays = _flat_grads(grads)
    mx.eval(loss)

    assert math.isfinite(float(loss.item()))
    for name in ("score_proj.weight", "score_out.weight"):
        assert name in grad_arrays
        assert np.isfinite(grad_arrays[name]).all(), name
        assert np.max(np.abs(grad_arrays[name])) > 0.0


def test_mhc_input_validation_fails_closed() -> None:
    mixer = CppMegaManifoldBranchMixer(ManifoldBranchMixerConfig(hidden_size=8, max_branches=3))
    valid = _rand((2, 4, 8), seed=12)

    with pytest.raises(ValueError, match="at least one branch"):
        mixer([])
    with pytest.raises(ValueError, match="too many branches"):
        mixer([valid, valid, valid, valid])
    with pytest.raises(ValueError, match="rank 3"):
        mixer([mx.ones((2, 8), dtype=mx.float32)])
    with pytest.raises(ValueError, match="shape mismatch"):
        mixer([valid, _rand((2, 5, 8), seed=13)])
    with pytest.raises(TypeError, match="floating dtype"):
        mixer([mx.ones((2, 4, 8), dtype=mx.int32)])
    with pytest.raises(TypeError, match="dtype"):
        mixer([valid, valid.astype(mx.float16)])


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"hidden_size": 0}, "hidden_size"),
        ({"sinkhorn_iters": -1}, "sinkhorn_iters"),
        ({"temperature": 0.0}, "temperature"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"blend_alpha": -0.1}, "blend_alpha"),
        ({"max_branches": -1}, "max_branches"),
    ],
)
def test_mhc_config_validates_constructor_args(kwargs: dict[str, object], error: str) -> None:
    config_kwargs = {
        "hidden_size": 8,
        "sinkhorn_iters": 5,
        "temperature": 1.0,
        "epsilon": 1e-6,
        "blend_alpha": 1.0,
        "max_branches": 0,
    }
    config_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=error):
        ManifoldBranchMixerConfig(
            hidden_size=int(config_kwargs["hidden_size"]),
            sinkhorn_iters=int(config_kwargs["sinkhorn_iters"]),
            temperature=float(config_kwargs["temperature"]),
            epsilon=float(config_kwargs["epsilon"]),
            blend_alpha=float(config_kwargs["blend_alpha"]),
            max_branches=int(config_kwargs["max_branches"]),
        )


def test_sinkhorn_normalize_validates_square_float_matrices() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        sinkhorn_normalize(mx.ones((3, 3), dtype=mx.float32))
    with pytest.raises(ValueError, match="square"):
        sinkhorn_normalize(mx.ones((1, 2, 3), dtype=mx.float32))
    with pytest.raises(TypeError, match="floating dtype"):
        sinkhorn_normalize(mx.ones((1, 3, 3), dtype=mx.int32))
