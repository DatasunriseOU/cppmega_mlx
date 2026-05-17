"""KDA Path B/C/D tests — mirror the GDN suite for the KDA backend."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.kda_path_b import kda_forward_path_b
from cppmega_v4._tilelang.kda_path_c import (
    _path_c_runtime_status as kda_path_c_runtime,
    _tilelang_importable as kda_tilelang_importable,
)
from cppmega_v4._tilelang.kda_path_d import (
    _fla_kda_chunk_importable,
    _path_d_runtime_status as kda_path_d_runtime,
    _triton_frontend_importable,
    _try_lower_fla_kda_kernel,
)
from cppmega_v4._tilelang.kda_paths import (
    ENV_VAR as KDA_ENV,
    _path_b_status,
    _path_c_status,
    _path_d_status,
    kda_path_statuses,
    kda_recurrent_dispatch,
)
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda


# ----- Path B (hand-MSL) -----


def test_kda_path_b_status_available():
    st = _path_b_status()
    assert st.available
    assert "metal_kernel" in st.reason


def test_kda_path_b_parity_with_path_a():
    B, T, H, HV, K, V = 1, 5, 2, 4, 8, 8
    rng = np.random.default_rng(31)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    o_b, _ = kda_forward_path_b(q, k, v, g, beta)
    o_a, _ = naive_recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-3)


def test_kda_path_b_output_final_state():
    B, T, H, HV, K, V = 1, 3, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    _, S = kda_forward_path_b(q, k, v, g, beta, output_final_state=True)
    assert S is not None
    assert S.shape == (B, HV, K, V)


def test_kda_path_b_dispatch(monkeypatch):
    monkeypatch.setenv(KDA_ENV, "path_b")
    B, T, H, HV, K, V = 1, 4, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    assert o.shape == (B, T, HV, V)
    assert not bool(mx.any(mx.isnan(o)).item())


# ----- Path C (TileLang DSL) -----


def test_kda_path_c_module_imports():
    from cppmega_v4._tilelang import kda_path_c  # noqa: F401


def test_kda_path_c_status_names_pipeline():
    st = _path_c_status()
    assert "TileLang" in st.reason
    assert "tvm_ffi" in st.reason
    assert "metal" in st.reason.lower()


def test_kda_path_c_runtime_matches_dispatch_status():
    ok, reason = kda_path_c_runtime()
    st = _path_c_status()
    assert st.available == ok
    assert reason in st.reason


def test_kda_path_c_forced_via_env_returns_valid_output(monkeypatch):
    monkeypatch.setenv(KDA_ENV, "path_c")
    B, T, H, HV, K, V = 1, 4, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    assert o.shape == (B, T, HV, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_kda_path_c_fallback_matches_path_a(monkeypatch):
    ok, _ = kda_tilelang_importable()
    if ok:
        pytest.skip("tilelang available — fallback not exercised here")
    monkeypatch.setenv(KDA_ENV, "path_c")
    B, T, H, HV, K, V = 1, 4, 2, 4, 6, 6
    rng = np.random.default_rng(41)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    o_disp, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    o_ref, _ = naive_recurrent_kda(q, k, v, g, beta)
    np.testing.assert_array_equal(np.array(o_disp), np.array(o_ref))


# ----- Path D (Triton frontend) -----


def test_kda_path_d_module_imports():
    from cppmega_v4._tilelang import kda_path_d  # noqa: F401


def test_kda_path_d_status_names_concrete_blocker():
    st = _path_d_status()
    reason = st.reason.lower()
    assert "triton" in reason or "fla" in reason or "op_mapping" in reason


def test_kda_path_d_runtime_matches_dispatch_status():
    ok, reason = kda_path_d_runtime()
    st = _path_d_status()
    assert st.available == ok
    assert reason in st.reason


def test_kda_path_d_probes_return_tuples():
    ok_fe, r_fe = _triton_frontend_importable()
    ok_src, r_src = _fla_kda_chunk_importable()
    assert isinstance(ok_fe, bool) and r_fe
    assert isinstance(ok_src, bool) and r_src


def test_kda_path_d_forced_falls_back_cleanly(monkeypatch):
    monkeypatch.setenv(KDA_ENV, "path_d")
    B, T, H, HV, K, V = 1, 4, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    assert o.shape == (B, T, HV, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_kda_path_d_try_lower_returns_seam_message():
    result, msg = _try_lower_fla_kda_kernel(target="metal")
    assert result is None
    assert isinstance(msg, str) and msg


# ----- Dispatch sanity -----


def test_kda_statuses_keys_unchanged():
    assert set(kda_path_statuses().keys()) == {"path_a", "path_b", "path_c", "path_d"}
