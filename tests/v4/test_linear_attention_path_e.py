"""Path E (vendored mlx-lm gated_delta_update) tests.

Upstream Metal kernel requires Dk % 32 == 0 and Dv % 4 == 0, so all shape
fixtures use Dk=Dv=32 (matches realistic GDN head dims like 128).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.linear_attention_paths import (
    _path_e_status,
    _path_e_status_for_inputs,
    gated_delta_recurrent_dispatch,
    linear_attention_auto_mode_for_inputs,
)
from cppmega_v4.nn._external._path_e_eligibility import PathEUnavailable
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


def test_path_e_dispatch_forced():
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)  # log-decay ≤ 0
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g, path="path_e")
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


# ----- Hardening: fail-close on amplifying gate (g>0) -----


def test_path_e_fails_closed_on_amplifying_gate():
    """g>0 (decay>1) cannot be represented; the adapter must fail closed.

    The historic adapter silently clamped g to 0 and lost information. The
    hardened adapter raises PathEUnavailable so the dispatcher falls back.
    """
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = mx.abs(mx.random.normal((B, T, H)) * 0.1) + 0.05  # amplifying: g>0
    with pytest.raises(PathEUnavailable, match="amplifying"):
        gated_delta_update(q, k, v, beta, g)


def test_path_e_status_unavailable_for_amplifying_gate():
    """Auto-mode status must mark Path E unavailable for an amplifying gate."""
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g_amp = mx.ones((B, T, H)) * 0.2
    st = _path_e_status_for_inputs(q, k, v, beta, g_amp)
    assert not st.available
    assert "amplifying" in st.reason


def test_path_e_status_available_for_odd_dk_shape():
    """After the in-MSL remainder-mask, Dk%32!=0 forward shapes are AVAILABLE.

    (Previously this asserted unavailability; the fast forward kernel now
    handles any Dk correctly, so an odd Dk no longer fails closed.)
    """
    B, T, H, K, V = 1, 4, 2, 40, 17  # Dk%32!=0 and Dv%4!=0
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)
    st = _path_e_status_for_inputs(q, k, v, beta, g)
    assert st.available


def test_path_e_status_available_for_eligible_inputs():
    """Eligible gate + shape keeps Path E available in auto-mode."""
    B, T, H, K, V = 1, 4, 2, 32, 32
    q = mx.random.normal((B, T, H, K))
    k = mx.random.normal((B, T, H, K))
    v = mx.random.normal((B, T, H, V))
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    g = -mx.abs(mx.random.normal((B, T, H)) * 0.1)
    assert _path_e_status_for_inputs(q, k, v, beta, g).available


def test_dispatch_auto_skips_path_e_for_amplifying_gate():
    """auto-mode dispatch must NOT silently clamp an amplifying gate.

    With Path C/B/D unavailable here (only B may be available on Metal), the
    dispatch must still produce output equal to Path A's representation of the
    amplifying gate — never Path E's clamped output.
    """
    B, T, H, K, V = 1, 5, 2, 32, 32
    rng = np.random.default_rng(777)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.uniform(0.1, 0.9, (B, T, H)).astype(np.float32))
    g = mx.array(rng.uniform(0.05, 0.4, (B, T, H)).astype(np.float32))  # g>0
    # Force E out of auto by making only A available; dispatch auto.
    o_disp, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    o_a, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    # The correct (un-clamped) result matches Path A exactly within Metal eps.
    np.testing.assert_allclose(np.array(o_disp), np.array(o_a), atol=1e-4, rtol=1e-3)


def test_auto_mode_skips_path_e_when_input_status_unavailable():
    """When an input-aware override marks E unavailable, auto_pick skips it."""
    from cppmega_v4._tilelang._dispatch import PathStatus

    chosen = linear_attention_auto_mode_for_inputs(
        env_var="CPPMEGA_V4_TEST_UNSET",
        status_overrides={
            "path_b": PathStatus("path_b", False, "test unavailable"),
            "path_c": PathStatus("path_c", False, "test unavailable"),
            "path_d": PathStatus("path_d", False, "test unavailable"),
            "path_e": PathStatus("path_e", False, "amplifying gate (test)"),
        },
    )
    assert chosen == "path_a"
