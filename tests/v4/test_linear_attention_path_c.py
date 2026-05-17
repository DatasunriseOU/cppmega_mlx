"""GDN Path C (TileLang DSL → Metal via tvm_ffi) tests.

Path C is environment-conditional: it only runs the real kernel when
``tilelang`` is importable AND the host ``cppmega_mlx`` TileLang→MSL
infrastructure is reachable. In CI/dev envs without tilelang (typical on
Apple Silicon without the TVM toolchain), the dispatch transparently falls
back to Path A — so tests cover both modes:

1. Module imports cleanly without tilelang installed.
2. Status reports a precise reason when unavailable.
3. Dispatch with env forced to ``path_c`` still returns Path-A-equivalent
   output (the fallback path) — no NaN, correct shape.
4. When tilelang IS importable, the @T.prim_func builds and produces
   parity output (skipped if tilelang missing).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.linear_attention_path_c import (
    _path_c_runtime_status,
    _tilelang_importable,
)
from cppmega_v4._tilelang.linear_attention_paths import (
    ENV_VAR as GDN_ENV,
    _path_c_status,
    gated_delta_recurrent_dispatch,
)
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)


def test_path_c_module_imports():
    """Module must be importable even when tilelang isn't installed."""
    from cppmega_v4._tilelang import linear_attention_path_c  # noqa: F401


def test_path_c_status_reason_is_precise():
    st = _path_c_status()
    # The reason string must mention TileLang and the lowering pipeline,
    # regardless of whether the backend is actually wired up.
    assert "TileLang" in st.reason or "tilelang" in st.reason
    assert "metal" in st.reason.lower()
    assert "tvm_ffi" in st.reason


def test_path_c_runtime_status_matches_dispatch_status():
    ok, reason = _path_c_runtime_status()
    st = _path_c_status()
    assert st.available == ok
    # The runtime reason should be embedded in the dispatch reason.
    assert reason in st.reason


def test_path_c_forced_via_env_returns_valid_output(monkeypatch):
    """Forcing path_c must always produce a valid output (real or fallback)."""
    monkeypatch.setenv(GDN_ENV, "path_c")
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    assert o.shape == (B, T, H, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_path_c_fallback_matches_path_a_when_unavailable(monkeypatch):
    """When the real kernel isn't available, dispatch must equal Path A bit-for-bit."""
    ok, _ = _tilelang_importable()
    if ok:
        pytest.skip("tilelang available — fallback path not exercised here")
    monkeypatch.setenv(GDN_ENV, "path_c")
    B, T, H, K, V = 1, 5, 2, 8, 8
    rng = np.random.default_rng(7)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.5, (B, T, H)).astype(np.float32))
    o_disp, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    o_ref, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    np.testing.assert_array_equal(np.array(o_disp), np.array(o_ref))


@pytest.mark.skipif(
    not _tilelang_importable()[0],
    reason="tilelang not importable in this env",
)
def test_path_c_real_kernel_parity_with_path_a():
    """Only runs when tilelang is reachable — actual prim_func parity check."""
    from cppmega_v4._tilelang.linear_attention_path_c import _gdn_fwd_path_c_call

    B, T, H, K, V = 1, 4, 2, 32, 32
    rng = np.random.default_rng(11)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.5, (B, T, H)).astype(np.float32))
    o_c, _ = _gdn_fwd_path_c_call(q, k, v, beta, g)
    o_a, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    np.testing.assert_allclose(np.array(o_c), np.array(o_a), atol=1e-4, rtol=1e-3)
