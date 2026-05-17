"""Path dispatch / auto-mode tests for GDN (ROI 3.B-F) and KDA (ROI 3.5.B-D)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang._dispatch import PathStatus, auto_pick, env_override
from cppmega_v4._tilelang.kda_paths import (
    ENV_VAR as KDA_ENV,
    kda_auto_mode_for_inputs,
    kda_path_statuses,
    kda_recurrent_dispatch,
)
from cppmega_v4._tilelang.linear_attention_paths import (
    ENV_VAR as GDN_ENV,
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


def test_env_override_none_when_unset_or_auto(monkeypatch):
    monkeypatch.delenv("CPPMEGA_V4_TEST_VAR", raising=False)
    assert env_override("CPPMEGA_V4_TEST_VAR") is None
    monkeypatch.setenv("CPPMEGA_V4_TEST_VAR", "auto")
    assert env_override("CPPMEGA_V4_TEST_VAR") is None


def test_env_override_returns_path(monkeypatch):
    monkeypatch.setenv("CPPMEGA_V4_TEST_VAR", "path_b")
    assert env_override("CPPMEGA_V4_TEST_VAR") == "path_b"


def test_env_override_rejects_unknown(monkeypatch):
    monkeypatch.setenv("CPPMEGA_V4_TEST_VAR", "path_z")
    with pytest.raises(ValueError, match="unsupported"):
        env_override("CPPMEGA_V4_TEST_VAR")


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


def test_gdn_path_d_still_deferred_without_triton():
    """Path D needs triton — unavailable on Apple Silicon. Path C is now
    real-or-fallback depending on whether tilelang is importable."""
    statuses = linear_attention_path_statuses()
    st_d = statuses["path_d"]
    assert not st_d.available
    assert len(st_d.reason) > 20
    # Path C reason must always name the lowering pipeline, regardless of
    # whether tilelang is reachable.
    st_c = statuses["path_c"]
    assert "tvm_ffi" in st_c.reason and "metal" in st_c.reason.lower()


def test_gdn_auto_mode_default_picks_first_available(monkeypatch):
    monkeypatch.delenv(GDN_ENV, raising=False)
    chosen = linear_attention_auto_mode_for_inputs()
    # Preference order: c > b > e > d > a. Path C wins when tilelang is
    # available; Path E wins next; Path B (real Metal) wins on Apple Silicon
    # without tilelang; Path A is the universal floor.
    assert chosen in ("path_a", "path_b", "path_c", "path_e")


def test_gdn_auto_mode_respects_env(monkeypatch):
    monkeypatch.setenv(GDN_ENV, "path_b")
    assert linear_attention_auto_mode_for_inputs() == "path_b"


def test_gdn_dispatch_returns_same_as_path_a(monkeypatch):
    """All current backends delegate to Path A — output must match Path A exactly."""
    monkeypatch.delenv(GDN_ENV, raising=False)
    B, T, H, K, V = 1, 5, 2, 4, 4
    rng = np.random.default_rng(99)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, H, V)).astype(np.float32))
    beta = mx.array(rng.standard_normal((B, T, H)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, H)).astype(np.float32) * 0.1)
    o_disp, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    o_ref, _ = naive_recurrent_gated_delta_rule(q, k, v, beta, g)
    # Path B (Metal float32) differs from Path A (MLX float64-then-cast) by
    # ~1e-7. Use atol for the comparison.
    np.testing.assert_allclose(np.array(o_disp), np.array(o_ref), atol=1e-5)


@pytest.mark.parametrize("path", ["path_a", "path_b", "path_c", "path_d", "path_e"])
def test_gdn_dispatch_each_path_runs(monkeypatch, path):
    """Each forced path must return finite, correctly-shaped output."""
    monkeypatch.setenv(GDN_ENV, path)
    q = mx.random.normal((1, 4, 2, 4))
    k = mx.random.normal((1, 4, 2, 4))
    v = mx.random.normal((1, 4, 2, 4))
    beta = mx.random.normal((1, 4, 2))
    g = mx.random.normal((1, 4, 2)) * 0.1
    o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    assert o.shape == (1, 4, 2, 4)
    assert not bool(mx.any(mx.isnan(o)).item())


# ----- KDA paths -----


def test_kda_statuses_keys():
    # KDA has no Path E.
    statuses = kda_path_statuses()
    assert set(statuses.keys()) == {"path_a", "path_b", "path_c", "path_d"}


def test_kda_path_a_always_available():
    assert kda_path_statuses()["path_a"].available


def test_kda_auto_mode_rejects_path_e(monkeypatch):
    monkeypatch.setenv(KDA_ENV, "path_e")
    with pytest.raises(ValueError, match="no Path E"):
        kda_auto_mode_for_inputs()


def test_kda_dispatch_returns_same_as_path_a(monkeypatch):
    monkeypatch.delenv(KDA_ENV, raising=False)
    B, T, H, K, HV, V = 1, 4, 2, 4, 2, 4
    rng = np.random.default_rng(200)
    q = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, T, H, K)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, T, HV, V)).astype(np.float32))
    g = mx.array(rng.standard_normal((B, T, HV, K)).astype(np.float32) * 0.05)
    beta = mx.array(rng.standard_normal((B, T, HV)).astype(np.float32))
    o_disp, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    o_ref, _ = naive_recurrent_kda(q, k, v, g, beta)
    # Path B (Metal float32) is now real and differs from Path A
    # (MLX float64-then-cast) by ~1e-7 — use atol instead of bit-exact.
    np.testing.assert_allclose(np.array(o_disp), np.array(o_ref), atol=1e-5)


@pytest.mark.parametrize("path", ["path_a", "path_b", "path_c", "path_d"])
def test_kda_dispatch_each_path_runs(monkeypatch, path):
    monkeypatch.setenv(KDA_ENV, path)
    q = mx.random.normal((1, 3, 2, 4))
    k = mx.random.normal((1, 3, 2, 4))
    v = mx.random.normal((1, 3, 2, 4))
    g = mx.random.normal((1, 3, 2, 4)) * 0.05
    beta = mx.random.normal((1, 3, 2))
    o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    assert o.shape == (1, 3, 2, 4)
    assert not bool(mx.any(mx.isnan(o)).item())
