"""GDN Path D (Triton frontend → TileLang) tests.

Path D wraps ``poc.triton_frontend.from_triton_kernel`` over FLA's
``chunk_gated_delta_rule`` kernels. The frontend now routes the real
chunk-delta-h TTIR op surface through OP_TABLE on dev hosts, but Path D
is still unavailable as a runtime backend until cppmega_v4 wires a
runnable PrimFunc/compiled artifact to the recurrent call signature.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.linear_attention_path_d import (
    _fla_chunk_kernel_importable,
    _path_d_runtime_status,
    _triton_frontend_importable,
    _try_lower_fla_chunk_kernel,
)
from cppmega_v4._tilelang.linear_attention_paths import (
    ENV_VAR as GDN_ENV,
    _path_d_status,
    gated_delta_recurrent_dispatch,
)
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)


def test_path_d_module_imports():
    from cppmega_v4._tilelang import linear_attention_path_d  # noqa: F401


def test_path_d_status_names_concrete_blocker():
    st = _path_d_status()
    # Status must mention either the missing dep or the op_mapping coverage
    # blocker, so the next contributor knows what to fix.
    assert "triton" in st.reason.lower() or "fla" in st.reason.lower() \
        or "op_mapping" in st.reason.lower()


def test_path_d_runtime_status_matches_dispatch_status():
    ok, reason = _path_d_runtime_status()
    st = _path_d_status()
    assert st.available == ok
    assert reason in st.reason


def test_triton_frontend_probe_returns_tuple():
    ok, reason = _triton_frontend_importable()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason


def test_fla_chunk_kernel_probe_returns_tuple():
    ok, reason = _fla_chunk_kernel_importable()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason


def test_path_d_forced_via_env_falls_back_cleanly(monkeypatch):
    """Forcing path_d must always return valid output (via Path A fallback)."""
    monkeypatch.setenv(GDN_ENV, "path_d")
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    assert o.shape == (B, T, H, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_path_d_fallback_matches_path_a_when_unavailable(monkeypatch):
    ok, _ = _path_d_runtime_status()
    if ok:
        pytest.skip("Path D actually available — fallback path not exercised")
    monkeypatch.setenv(GDN_ENV, "path_d")
    B, T, H, K, V = 1, 5, 2, 8, 8
    rng = np.random.default_rng(13)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.5, (B, T, H)).astype(np.float32))
    o_disp, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    o_ref, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    np.testing.assert_array_equal(np.array(o_disp), np.array(o_ref))


def test_try_lower_reports_real_coverage_or_concrete_blocker():
    result, msg = _try_lower_fla_chunk_kernel(target="metal")
    assert result is None or hasattr(result, "with_attr")
    assert isinstance(msg, str) and msg
    assert (
        "status=LOWERED_DEGRADED" in msg
        or "status=LOWERED_FULL" in msg
        or "not importable" in msg
        or "failed" in msg.lower()
        or "missing" in msg.lower()
    )
