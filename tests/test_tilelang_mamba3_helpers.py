"""Parity tests for the Path B TileLang Mamba3 backward helpers.

These tests verify that the TileLang/Metal rewrites of the three Mamba3 Triton
helpers in ``cppmega_mlx.nn._tilelang._mamba3_helpers_tilelang`` agree with
their pure-MLX siblings (``_mamba3_helpers``) at fp16 carrier tolerance. The
sibling is the parity oracle on macOS because the upstream Triton reference is
CUDA-only.

Coverage:

  * ``compute_dacs_segsum`` -- segment reverse cumsum over the time axis.
  * ``bwd_dadt_fused``      -- fused dA/ddt computation.
  * ``bwd_dtrap_ddt``       -- fused ddt/dtrap from the trapezoidal scale.

Each test compares fp16 output between the TileLang kernel and the pure-MLX
sibling at multiple shapes. Tolerances are rtol=1e-4 atol=1e-3 to absorb the
fp16 carrier rounding documented in the module docstring.

Skips:

  * ``tilelang`` not importable (typical on macOS without the Apple branch).
  * No Metal-backed default device (CPU-only MLX).
  * The TileLang status helper reports ``available=False`` for any reason.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import mlx.core as mx


tilelang = pytest.importorskip("tilelang")  # noqa: F841


from cppmega_mlx.nn._tilelang import _mamba3_helpers as _pure_helpers  # noqa: E402
from cppmega_mlx.nn._tilelang._mamba3_helpers_tilelang import (  # noqa: E402
    bwd_dadt_fused as tl_bwd_dadt_fused,
    bwd_dtrap_ddt as tl_bwd_dtrap_ddt,
    compute_dacs_segsum as tl_compute_dacs_segsum,
    helpers_metal_status,
)


def _skip_if_unavailable() -> None:
    status = helpers_metal_status()
    if not status.available:
        pytest.skip(f"TileLang Metal helpers not available: {status.reason}")


# ---------------------------------------------------------------------------
# Inputs / oracle helpers
# ---------------------------------------------------------------------------


def _make_segsum_inputs(B: int, T_: int, H: int, P: int, N: int, *, seed: int):
    rng = np.random.default_rng(seed)
    A_np = (rng.standard_normal((B, T_, H)) * 0.1 - 0.5).astype(np.float32)
    dt_np = (np.abs(rng.standard_normal((B, T_, H))) * 0.1 + 1e-3).astype(np.float32)
    dh_np = rng.standard_normal((B, T_, H, P, N)).astype(np.float16)
    return mx.array(A_np), mx.array(dt_np), mx.array(dh_np)


def _make_dadt_inputs(B: int, T_: int, H: int, P: int, N: int, *, seed: int):
    rng = np.random.default_rng(seed)
    A_np = (rng.standard_normal((B, T_, H)) * 0.1 - 0.5).astype(np.float32)
    dt_np = (np.abs(rng.standard_normal((B, T_, H))) * 0.1 + 1e-3).astype(np.float32)
    dY_np = rng.standard_normal((B, T_, H, P, N)).astype(np.float16)
    h_np = rng.standard_normal((B, T_, H, P, N)).astype(np.float16)
    return (
        mx.array(dY_np),
        mx.array(A_np),
        mx.array(dt_np),
        mx.array(h_np),
    )


def _make_dtrap_inputs(B: int, T_: int, H: int, *, seed: int):
    rng = np.random.default_rng(seed)
    dB_np = rng.standard_normal((B, T_, H)).astype(np.float16)
    dt_np = (np.abs(rng.standard_normal((B, T_, H))) * 0.1 + 1e-3).astype(np.float16)
    trap_np = rng.standard_normal((B, T_, H)).astype(np.float16)
    return mx.array(dB_np), mx.array(dt_np), mx.array(trap_np)


def _to_fp32_np(x: mx.array) -> np.ndarray:
    return np.array(x).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,T_,H,P,N,seed",
    [
        (1, 16, 4, 4, 8, 0),
        (2, 32, 4, 4, 8, 1),
        (4, 64, 8, 4, 16, 2),
        (1, 8, 4, 1, 1, 3),
        (2, 128, 4, 64, 16, 4),  # mamba3-default chunk
    ],
)
def test_compute_dacs_segsum_parity_with_pure_mlx(B, T_, H, P, N, seed) -> None:
    _skip_if_unavailable()
    A, dt, dh = _make_segsum_inputs(B, T_, H, P, N, seed=seed)
    mx.eval(A, dt, dh)
    out_pure = _pure_helpers.compute_dacs_segsum(A, dt, dh)
    out_tl = tl_compute_dacs_segsum(A, dt, dh)
    mx.eval(out_pure, out_tl)
    assert out_tl.shape == out_pure.shape
    assert out_tl.dtype == out_pure.dtype
    np.testing.assert_allclose(
        _to_fp32_np(out_tl),
        _to_fp32_np(out_pure),
        rtol=1e-4,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "B,T_,H,P,N,seed",
    [
        (1, 16, 4, 4, 8, 10),
        (2, 32, 4, 4, 8, 11),
        (4, 64, 8, 4, 16, 12),
        (1, 8, 4, 1, 1, 13),
        (2, 128, 4, 64, 16, 14),
    ],
)
def test_bwd_dadt_fused_parity_with_pure_mlx(B, T_, H, P, N, seed) -> None:
    _skip_if_unavailable()
    dY, A, dt, h = _make_dadt_inputs(B, T_, H, P, N, seed=seed)
    mx.eval(dY, A, dt, h)
    dA_p, ddt_p = _pure_helpers.bwd_dadt_fused(dY, A, dt, h)
    dA_t, ddt_t = tl_bwd_dadt_fused(dY, A, dt, h)
    mx.eval(dA_p, ddt_p, dA_t, ddt_t)
    assert dA_t.shape == dA_p.shape
    assert ddt_t.shape == ddt_p.shape
    np.testing.assert_allclose(_to_fp32_np(dA_t), _to_fp32_np(dA_p), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(_to_fp32_np(ddt_t), _to_fp32_np(ddt_p), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize(
    "B,T_,H,seed",
    [
        (1, 16, 4, 20),
        (2, 32, 4, 21),
        (4, 64, 8, 22),
        (1, 8, 4, 23),
        (2, 1, 4, 24),
        (2, 128, 4, 25),
    ],
)
def test_bwd_dtrap_ddt_parity_with_pure_mlx(B, T_, H, seed) -> None:
    _skip_if_unavailable()
    dB, dt, trap = _make_dtrap_inputs(B, T_, H, seed=seed)
    mx.eval(dB, dt, trap)
    ddt_p, dtrap_p = _pure_helpers.bwd_dtrap_ddt(dB, dt, trap)
    ddt_t, dtrap_t = tl_bwd_dtrap_ddt(dB, dt, trap)
    mx.eval(ddt_p, dtrap_p, ddt_t, dtrap_t)
    assert ddt_t.shape == ddt_p.shape
    assert dtrap_t.shape == dtrap_p.shape
    np.testing.assert_allclose(_to_fp32_np(ddt_t), _to_fp32_np(ddt_p), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(_to_fp32_np(dtrap_t), _to_fp32_np(dtrap_p), rtol=1e-4, atol=1e-3)


def test_helpers_metal_status_reports_available_when_metal_ready() -> None:
    status = helpers_metal_status()
    # The status object is always defined; the boolean depends on the host.
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason
    assert status.fp16_carrier is True


def test_force_fallback_routes_through_pure_mlx() -> None:
    """force_fallback=True must bypass the Metal kernel entirely."""

    A, dt, dh = _make_segsum_inputs(1, 16, 4, 4, 8, seed=99)
    mx.eval(A, dt, dh)
    expected = _pure_helpers.compute_dacs_segsum(A, dt, dh)
    fallback = tl_compute_dacs_segsum(A, dt, dh, force_fallback=True)
    mx.eval(expected, fallback)
    np.testing.assert_array_equal(_to_fp32_np(fallback), _to_fp32_np(expected))


def test_module_docstring_references_attribution() -> None:
    """Make sure the source-attribution paragraph stays in the module docstring."""

    mod = importlib.import_module(
        "cppmega_mlx.nn._tilelang._mamba3_helpers_tilelang"
    )
    assert mod.__doc__ is not None
    assert "compute_dacs_segsum_triton" in mod.__doc__
    assert "bwd_dadt_fused_triton" in mod.__doc__
    assert "bwd_dtrap_ddt_triton" in mod.__doc__
    assert "fp16" in mod.__doc__.lower()
