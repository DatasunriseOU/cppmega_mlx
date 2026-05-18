"""Forward + backward tests for KDA Path B fwd / Path A bwd wrapper."""

import mlx.core as mx
import numpy as np

from cppmega_v4._tilelang.kda_path_b_bwd import kda_apply_path_b
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda


def _inputs(B=1, T=5, H=2, HV=4, K=8, V=8, seed=17):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    return q, k, v, g, beta


def test_kda_forward_matches_path_a():
    q, k, v, g, beta = _inputs()
    y_b = kda_apply_path_b(q, k, v, g, beta)
    y_a, _ = naive_recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.array(y_b), np.array(y_a), atol=1e-4, rtol=1e-3)


def test_kda_backward_matches_path_a_grads():
    q, k, v, g, beta = _inputs(seed=29)
    cotangent = mx.random.normal((q.shape[0], q.shape[1], v.shape[2], v.shape[-1]))

    def loss_b(q_, k_, v_, g_, beta_):
        y = kda_apply_path_b(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    def loss_a(q_, k_, v_, g_, beta_):
        y, _ = naive_recurrent_kda(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    grad_b = mx.grad(loss_b, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    grad_a = mx.grad(loss_a, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for name, gb, ga in zip(["dq", "dk", "dv", "dg", "dbeta"], grad_b, grad_a):
        np.testing.assert_allclose(
            np.array(gb), np.array(ga), atol=1e-4, rtol=1e-3,
            err_msg=f"{name} mismatch",
        )


def test_kda_fwd_bwd_finite():
    q, k, v, g, beta = _inputs(B=1, T=6, H=4, HV=8, K=16, V=16, seed=33)
    cotangent = mx.random.normal((q.shape[0], q.shape[1], v.shape[2], v.shape[-1]))

    def loss(q_, k_, v_, g_, beta_):
        y = kda_apply_path_b(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for name, gr in zip(["dq", "dk", "dv", "dg", "dbeta"], grads):
        assert np.all(np.isfinite(np.array(gr))), f"{name} non-finite"


# ----- Multi-simdgroup tests (V > 32) -----


def test_kda_bwd_v64_matches_path_a():
    """V=64 → 2 simdgroups, shared-mem cross-simdgroup reductions."""
    q, k, v, g, beta = _inputs(B=1, T=3, H=2, HV=4, K=16, V=64, seed=64)
    cotangent = mx.random.normal((q.shape[0], q.shape[1], v.shape[2], v.shape[-1]))

    def loss_b(q_, k_, v_, g_, beta_):
        y = kda_apply_path_b(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    def loss_a(q_, k_, v_, g_, beta_):
        y, _ = naive_recurrent_kda(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    grad_b = mx.grad(loss_b, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    grad_a = mx.grad(loss_a, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for name, gb, ga in zip(["dq", "dk", "dv", "dg", "dbeta"], grad_b, grad_a):
        np.testing.assert_allclose(
            np.array(gb), np.array(ga), atol=3e-4, rtol=3e-3,
            err_msg=f"{name} mismatch at V=64",
        )


def test_kda_bwd_v128_runs_and_grads_finite():
    """V=128 → 4 simdgroups; verify no NaN/inf and shape correctness."""
    q, k, v, g, beta = _inputs(B=1, T=2, H=2, HV=4, K=16, V=128, seed=128)
    cotangent = mx.random.normal((q.shape[0], q.shape[1], v.shape[2], v.shape[-1]))

    def loss(q_, k_, v_, g_, beta_):
        y = kda_apply_path_b(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    grads = mx.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    expected = [q.shape, k.shape, v.shape, g.shape, beta.shape]
    for name, gr, exp in zip(["dq", "dk", "dv", "dg", "dbeta"], grads, expected):
        arr = np.array(gr)
        assert tuple(arr.shape) == tuple(exp), f"{name} shape {arr.shape} != {exp}"
        assert np.all(np.isfinite(arr)), f"{name} non-finite at V=128"
