"""KDA Path B/C/D tests — mirror the GDN suite for the KDA backend."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang._path_d_deps import TRITON_FRONTEND_UNSAFE_IMPORT_ENV
from cppmega_v4._tilelang.kda_path_b import kda_forward_path_b
from cppmega_v4._tilelang.kda_path_c import (
    _path_c_runtime_status as kda_path_c_runtime,
)
from cppmega_v4._tilelang.kda_path_d import (
    _fla_kda_chunk_importable,
    _path_d_runtime_status as kda_path_d_runtime,
    _triton_frontend_importable,
    _try_lower_fla_kda_kernel,
)
from cppmega_v4._tilelang._dispatch import PathStatus
from cppmega_v4._tilelang.kda_paths import (
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


def test_kda_path_b_initial_state_parity():
    """Streaming-decode: feeding nonzero h0 must match Path A."""
    B, T, H, HV, K, V = 1, 4, 2, 4, 6, 8
    rng = np.random.default_rng(51)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    h0 = mx.array(rng.standard_normal((B, HV, K, V)).astype(np.float32) * 0.5)
    o_b, sf_b = kda_forward_path_b(
        q, k, v, g, beta, initial_state=h0, output_final_state=True,
    )
    o_a, sf_a = naive_recurrent_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True,
    )
    mx.eval(o_b, o_a, sf_b, sf_a)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(sf_b), np.array(sf_a), atol=1e-4, rtol=1e-4)


def test_kda_path_b_streaming_chunks_match_full_run():
    B, H, HV, K, V = 1, 2, 4, 6, 6
    T1, T2 = 3, 4
    T = T1 + T2
    rng = np.random.default_rng(57)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))

    o_full, sf_full = kda_forward_path_b(q, k, v, g, beta, output_final_state=True)

    o1, sf_mid = kda_forward_path_b(
        q[:, :T1], k[:, :T1], v[:, :T1], g[:, :T1], beta[:, :T1],
        output_final_state=True,
    )
    o2, sf_end = kda_forward_path_b(
        q[:, T1:], k[:, T1:], v[:, T1:], g[:, T1:], beta[:, T1:],
        initial_state=sf_mid, output_final_state=True,
    )
    o_stream = mx.concatenate([o1, o2], axis=1)
    mx.eval(o_full, o_stream, sf_full, sf_end)
    np.testing.assert_allclose(np.array(o_stream), np.array(o_full), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(sf_end), np.array(sf_full), atol=1e-4, rtol=1e-4)


def test_kda_path_b_custom_scale_parity():
    B, T, H, HV, K, V = 1, 3, 2, 4, 8, 8
    rng = np.random.default_rng(61)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    custom_scale = 0.125
    o_b, _ = kda_forward_path_b(q, k, v, g, beta, scale=custom_scale)
    o_a, _ = naive_recurrent_kda(q, k, v, g, beta, scale=custom_scale)
    np.testing.assert_allclose(np.array(o_b), np.array(o_a), atol=1e-4, rtol=1e-4)


def test_kda_path_b_dispatch():
    B, T, H, HV, K, V = 1, 4, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta, path="path_b")
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


def test_kda_path_c_forced_returns_valid_output():
    B, T, H, HV, K, V = 1, 4, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta, path="path_c")
    assert o.shape == (B, T, HV, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_kda_path_c_fallback_matches_path_a():
    """Path C fallback parity uses an explicit status seam, not patching."""
    B, T, H, HV, K, V = 1, 4, 2, 4, 6, 6
    rng = np.random.default_rng(41)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    o_disp, _ = kda_recurrent_dispatch(
        q,
        k,
        v,
        g,
        beta,
        path="path_c",
        status_overrides={
            "path_c": PathStatus(
                path="path_c",
                available=False,
                reason="forced unavailable for fallback parity test",
            )
        },
    )
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


def test_kda_path_d_triton_probe_fails_closed_by_default():
    if os.environ.get(TRITON_FRONTEND_UNSAFE_IMPORT_ENV):
        pytest.skip(f"{TRITON_FRONTEND_UNSAFE_IMPORT_ENV} explicitly enabled")
    ok, reason = _triton_frontend_importable()
    assert ok is False
    assert "unsafe import disabled" in reason
    assert "runtime adapter not reached" in reason


def test_kda_path_d_fla_probe_fails_closed_by_default():
    if os.environ.get(TRITON_FRONTEND_UNSAFE_IMPORT_ENV):
        pytest.skip(f"{TRITON_FRONTEND_UNSAFE_IMPORT_ENV} explicitly enabled")
    ok, reason = _fla_kda_chunk_importable()
    assert ok is False
    assert "unsafe Path D import disabled" in reason
    assert "runtime adapter not reached" in reason


def test_kda_path_d_forced_falls_back_cleanly():
    B, T, H, HV, K, V = 1, 4, 2, 4, 8, 8
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, HV, V))
    g = -mx.abs(mx.random.normal((B, T, HV, K)) * 0.1)
    beta = mx.sigmoid(mx.random.normal((B, T, HV)))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta, path="path_d")
    assert o.shape == (B, T, HV, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_kda_path_d_forced_fixed_prefill_uses_runtime_adapter():
    pytest.importorskip("tilelang")
    ok, reason = kda_path_d_runtime()
    if not ok:
        pytest.skip(reason)

    from cppmega_v4._tilelang import kda_paths as kda_paths_mod

    q = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    k = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    v = mx.zeros((1, 64, 1, 32), dtype=mx.float16)
    g = mx.zeros((1, 64, 1, 64), dtype=mx.float32)
    beta = mx.zeros((1, 64, 1), dtype=mx.float32)

    y, final_state = kda_paths_mod.kda_recurrent_dispatch(
        q,
        k,
        v,
        g,
        beta,
        path="path_d",
        allow_fallback=False,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 64, 1, 32)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (1, 1, 64, 32)


def test_kda_path_d_forced_varlen_uses_runtime_adapter():
    pytest.importorskip("tilelang")
    ok, reason = kda_path_d_runtime()
    if not ok:
        pytest.skip(reason)

    from cppmega_v4._tilelang import kda_paths as kda_paths_mod

    q = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    k = mx.zeros((1, 64, 1, 64), dtype=mx.float16)
    v = mx.zeros((1, 64, 1, 32), dtype=mx.float16)
    g = mx.zeros((1, 64, 1, 64), dtype=mx.float32)
    beta = mx.zeros((1, 64, 1), dtype=mx.float32)
    cu_seqlens = mx.array([0, 16, 64], dtype=mx.int64)
    h0 = mx.zeros((2, 1, 64, 32), dtype=mx.float32)

    y, final_state = kda_paths_mod.kda_recurrent_dispatch(
        q,
        k,
        v,
        g,
        beta,
        path="path_d",
        allow_fallback=False,
        cu_seqlens=cu_seqlens,
        initial_state=h0,
        output_final_state=True,
    )
    mx.eval(y, final_state)

    assert y.shape == (1, 64, 1, 32)
    assert y.dtype == mx.float16
    assert final_state is not None
    assert final_state.shape == (2, 1, 64, 32)


def test_kda_path_d_try_lower_returns_seam_message():
    result, msg = _try_lower_fla_kda_kernel(target="metal")
    assert result is None
    assert isinstance(msg, str) and msg
    assert (
        "coverage complete" in msg
        or "not importable" in msg
        or "failed" in msg.lower()
        or "coverage gap" in msg.lower()
    )


# ----- Dispatch sanity -----


# ----- Path E (vendored mlx-lm gated_delta vec-gate kernel) -----


def test_kda_path_e_status_available():
    from cppmega_v4._tilelang.kda_paths import _path_e_status
    st = _path_e_status()
    assert st.available, st.reason
    assert "gated_delta" in st.reason


def test_kda_path_e_parity_with_path_a_kernel_path():
    """Dk%32==0 + Dv%4==0 hits the Metal kernel — must match Path A."""
    from cppmega_v4.nn._external.mlx_lm_kda_update import kda_update
    B, T, H, HV, K, V = 1, 4, 2, 4, 32, 32
    rng = np.random.default_rng(123)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    o_e, sf_e = kda_update(q, k, v, g, beta, output_final_state=True)
    o_a, sf_a = naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    mx.eval(o_e, o_a, sf_e, sf_a)
    np.testing.assert_allclose(np.array(o_e), np.array(o_a), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(sf_e), np.array(sf_a), atol=1e-4, rtol=1e-4)


def test_kda_path_e_parity_ops_fallback():
    """Small dims (Dk%32!=0) fall through to the pure-ops reference."""
    from cppmega_v4.nn._external.mlx_lm_kda_update import kda_update
    B, T, H, HV, K, V = 1, 3, 2, 4, 8, 8
    rng = np.random.default_rng(131)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    o_e, _ = kda_update(q, k, v, g, beta)
    o_a, _ = naive_recurrent_kda(q, k, v, g, beta)
    np.testing.assert_allclose(np.array(o_e), np.array(o_a), atol=1e-4, rtol=1e-4)


def test_kda_path_e_forced():
    B, T, H, HV, K, V = 1, 4, 2, 4, 32, 32
    rng = np.random.default_rng(141)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(-rng.uniform(0.01, 0.2, (B, T, HV, K)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, HV)).astype(np.float32))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta, path="path_e")
    assert o.shape == (B, T, HV, V)
    assert not bool(mx.any(mx.isnan(o)).item())


def test_kda_statuses_keys_unchanged():
    assert set(kda_path_statuses().keys()) == {
        "path_a", "path_b", "path_c", "path_d", "path_e",
    }
