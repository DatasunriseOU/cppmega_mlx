"""Path E training=True smoke + parity tests (vendored mlx-lm VJP modules)."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)
from cppmega_v4.nn._external.mlx_lm_gated_delta_update import gated_delta_update


def _inputs(B=1, T=8, H=2, K=32, V=32, seed=5):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.5, (B, T, H)).astype(np.float32))
    return q, k, v, beta, g


def test_path_e_training_returns_finite_output():
    q, k, v, beta, g = _inputs()
    y, _ = gated_delta_update(q, k, v, beta, g, training=True)
    assert y.shape == v.shape
    assert np.all(np.isfinite(np.array(y)))


def test_path_e_training_matches_inference_within_chunk_tolerance():
    """training=True uses chunked VJP path; output should match training=False
    closely (chunking introduces small numerical drift)."""
    q, k, v, beta, g = _inputs(seed=11)
    y_train, _ = gated_delta_update(q, k, v, beta, g, training=True)
    y_infer, _ = gated_delta_update(q, k, v, beta, g, training=False)
    np.testing.assert_allclose(np.array(y_train), np.array(y_infer),
                                atol=1e-2, rtol=1e-2)


def test_path_e_training_propagates_grads():
    q, k, v, beta, g = _inputs(seed=23)
    cotangent = mx.random.normal(v.shape)

    def loss(q_, k_, v_, beta_, g_):
        y, _ = gated_delta_update(q_, k_, v_, beta_, g_, training=True)
        return (y * cotangent).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, g)
    for name, gr in zip(["dq", "dk", "dv", "dbeta", "dg"], grads):
        assert np.all(np.isfinite(np.array(gr))), f"{name} non-finite"
