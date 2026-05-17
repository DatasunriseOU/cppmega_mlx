"""Path E (vendored mlx-lm gated_delta_update) tests.

Upstream Metal kernel requires Dk % 32 == 0 and Dv % 4 == 0, so all shape
fixtures use Dk=Dv=32 (matches realistic GDN head dims like 128).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.linear_attention_paths import (
    ENV_VAR as GDN_ENV,
    _path_e_status,
    gated_delta_recurrent_dispatch,
)
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)
from cppmega_v4.nn._external.mlx_lm_gated_delta_update import gated_delta_update


def test_path_e_status_available():
    st = _path_e_status()
    assert st.available
    assert "vendored" in st.reason.lower()


def test_path_e_output_shape():
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)  # log-decay ≤ 0
    o, _ = gated_delta_update(q, k, v, beta, g)
    assert o.shape == (B, T, H, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_path_e_parity_with_path_a():
    """Path E must produce numerically close output to Path A.

    Path E goes through upstream compute_g(A_log=0, dt_bias=0, a); we
    synthesize ``a = softplus_inverse(-log(g))`` so the round-trip recovers
    our gate. Float32 rounding gives ~1e-4 atol.
    """
    B, T, H, K, V = 1, 5, 2, 32, 32
    rng = np.random.default_rng(123)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32))
    # g is log-decay (FLA convention); must be ≤ 0 for Path E to represent it.
    g = mx.array(-rng.uniform(0.01, 0.5, (B, T, H)).astype(np.float32))
    o_e, _ = gated_delta_update(q, k, v, beta, g)
    o_a, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    np.testing.assert_allclose(np.array(o_e), np.array(o_a), atol=1e-3, rtol=1e-2)


def test_path_e_dispatch_via_env(monkeypatch):
    monkeypatch.setenv(GDN_ENV, "path_e")
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)  # log-decay ≤ 0
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    assert o.shape == (B, T, H, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_path_e_output_final_state():
    B, T, H, K, V = 1, 3, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)  # log-decay ≤ 0
    _, state = gated_delta_update(q, k, v, beta, g, output_final_state=True)
    assert state is not None
    # Upstream state shape: [B, Hv, Dv, Dk]
    assert state.shape == (B, H, V, K)
