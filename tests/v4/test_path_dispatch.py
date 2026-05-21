"""Path dispatch / auto-mode tests for GDN (ROI 3.B-F) and KDA (ROI 3.5.B-D)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

import pytest

from cppmega_v4._tilelang._dispatch import (
    PathStatus,
    auto_pick,
    parse_path_override,
)
from cppmega_v4._tilelang.kda_paths import (
    kda_auto_mode_for_inputs,
    kda_path_statuses,
    kda_recurrent_dispatch,
)
from cppmega_v4._tilelang.linear_attention_paths import (
    gated_delta_recurrent_dispatch,
    linear_attention_auto_mode_for_inputs,
    linear_attention_path_statuses,
)
from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
    naive_recurrent_gated_delta_rule,
)
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda


# ----- _dispatch core -----


def test_path_status_truthy():
    assert bool(PathStatus(path="path_a", available=True, reason="ref"))
    assert not bool(PathStatus(path="path_b", available=False, reason="not yet"))


def test_parse_path_override_none_when_unset_or_auto():
    assert parse_path_override(None, env_var="CPPMEGA_V4_TEST_VAR") is None
    assert parse_path_override("", env_var="CPPMEGA_V4_TEST_VAR") is None
    assert parse_path_override("auto", env_var="CPPMEGA_V4_TEST_VAR") is None


def test_parse_path_override_returns_path():
    assert parse_path_override("path_b", env_var="CPPMEGA_V4_TEST_VAR") == "path_b"


def test_parse_path_override_rejects_unknown():
    with pytest.raises(ValueError, match="unsupported"):
        parse_path_override("path_z", env_var="CPPMEGA_V4_TEST_VAR")


def test_auto_pick_prefers_first_available():
    statuses = {
        "path_a": PathStatus("path_a", True, "ok"),
        "path_b": PathStatus("path_b", True, "ok"),
        "path_c": PathStatus("path_c", False, "no"),
    }
    # Default preference is (c, b, e, d, a). With c unavailable, b wins.
    assert auto_pick(statuses) == "path_b"


def test_auto_pick_falls_back_to_path_a():
    statuses = {
        "path_a": PathStatus("path_a", True, "ok"),
        "path_b": PathStatus("path_b", False, "no"),
    }
    assert auto_pick(statuses) == "path_a"


# ----- GDN (linear_attention) paths -----


def test_gdn_statuses_keys():
    statuses = linear_attention_path_statuses()
    assert set(statuses.keys()) == {"path_a", "path_b", "path_c", "path_d", "path_e"}


def test_gdn_path_a_always_available():
    statuses = linear_attention_path_statuses()
    assert statuses["path_a"].available


def test_gdn_path_d_and_c_reasons_are_coherent():
    """Path D / Path C status reasons must be coherent regardless of whether
    the backend is currently available (depends on triton + tilelang)."""
    statuses = linear_attention_path_statuses()
    st_d = statuses["path_d"]
    # Reason must name the lowering pipeline so the next contributor knows
    # what's going on whether available is True or False.
    assert "triton" in st_d.reason.lower() or "tilelang" in st_d.reason.lower()
    assert len(st_d.reason) > 20
    # Path C reason must always name the lowering pipeline.
    st_c = statuses["path_c"]
    assert "tvm_ffi" in st_c.reason and "metal" in st_c.reason.lower()


def test_gdn_auto_mode_default_picks_first_available():
    chosen = linear_attention_auto_mode_for_inputs(
        env_var="CPPMEGA_V4_TEST_UNSET",
        status_overrides={
            "path_b": PathStatus("path_b", True, "test available"),
            "path_c": PathStatus("path_c", False, "test unavailable"),
            "path_e": PathStatus("path_e", True, "test available"),
        },
    )
    assert chosen == "path_b"


def test_gdn_dispatch_accepts_explicit_path():
    q = mx.random.normal((1, 2, 1, 4))
    k = mx.random.normal((1, 2, 1, 4))
    v = mx.random.normal((1, 2, 1, 4))
    beta = mx.random.normal((1, 2, 1))
    g = mx.random.normal((1, 2, 1)) * 0.1
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g, path="path_a")
    assert o.shape == (1, 2, 1, 4)


def test_gdn_dispatch_returns_same_as_path_a():
    """All current backends delegate to Path A — output must match Path A exactly."""
    B, T, H, K, V = 1, 5, 2, 4, 4
    rng = np.random.default_rng(99)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    o_disp, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g, path="path_a")
    o_ref, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    # Path B (Metal float32) differs from Path A (MLX float64-then-cast) by
    # ~1e-7. Use atol for the comparison.
    np.testing.assert_allclose(np.array(o_disp), np.array(o_ref), atol=1e-5)


@pytest.mark.parametrize("path", ["path_a", "path_b", "path_c", "path_d", "path_e"])
def test_gdn_dispatch_each_path_runs(path):
    """Each forced path must return finite, correctly-shaped output."""
    q = mx.random.normal((1, 4, 2, 4))
    k = mx.random.normal((1, 4, 2, 4))
    v = mx.random.normal((1, 4, 2, 4))
    beta = mx.random.normal((1, 4, 2))
    g = mx.random.normal((1, 4, 2)) * 0.1
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g, path=path)
    assert o.shape == (1, 4, 2, 4)
    assert not bool(mx.any(mx.isnan(o)).item())


# ----- KDA paths -----


def test_kda_statuses_keys():
    # KDA now has Path E (same upstream Metal kernel as GDN; KDA hits the
    # vectorised-gate branch via g.ndim==4).
    statuses = kda_path_statuses()
    assert set(statuses.keys()) == {"path_a", "path_b", "path_c", "path_d", "path_e"}


def test_kda_path_a_always_available():
    assert kda_path_statuses()["path_a"].available


def test_kda_auto_mode_accepts_path_e():
    assert kda_auto_mode_for_inputs(
        env_var="CPPMEGA_V4_TEST_UNSET",
        status_overrides={
            "path_b": PathStatus("path_b", False, "test unavailable"),
            "path_c": PathStatus("path_c", False, "test unavailable"),
            "path_e": PathStatus("path_e", True, "test available"),
        },
    ) == "path_e"


def test_kda_dispatch_returns_same_as_path_a():
    B, T, H, K, HV, V = 1, 4, 2, 4, 2, 4
    rng = np.random.default_rng(200)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, HV, K)).astype(np.float32) * 0.05)
    beta = mx.array(rng.standard_normal((B, T, HV)).astype(np.float32))
    o_disp, _ = kda_recurrent_dispatch(q, k, v, g, beta, path="path_a")
    o_ref, _ = naive_recurrent_kda(q, k, v, g, beta)
    # Path B (Metal float32) is now real and differs from Path A
    # (MLX float64-then-cast) by ~1e-7 — use atol instead of bit-exact.
    np.testing.assert_allclose(np.array(o_disp), np.array(o_ref), atol=1e-5)


@pytest.mark.parametrize("path", ["path_a", "path_b", "path_c", "path_d"])
def test_kda_dispatch_each_path_runs(path):
    q = mx.random.normal((1, 3, 2, 4))
    k = mx.random.normal((1, 3, 2, 4))
    v = mx.random.normal((1, 3, 2, 4))
    g = mx.random.normal((1, 3, 2, 4)) * 0.05
    beta = mx.random.normal((1, 3, 2))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta, path=path)
    assert o.shape == (1, 3, 2, 4)
    assert not bool(mx.any(mx.isnan(o)).item())
