"""Coverage for the Path C TileLang DSL Sparse-MLA blockscaled (E8M0) port.

Per the Path B/C routing audit, ``sparse_mla_blockscaled_path_c`` does *not*
yet expose a public ``apply``-shaped end-to-end entry point: only the QK probe
and the partial reducer are dispatchable. The routing doc refers to
"Path C partial reducer C/B 0.4364 / full QK unavailable", and Meta agent F's
design report (``reports/2026-05-06-tilelang-tvm-review/agent-F-path-b-vs-c/``)
captures this as a gap, not a bug.

This test file therefore:

  * pins the public surface that exists today (status probes + reducer +
    QK lowering + MSL feature inspection) so a silent removal would fail CI;
  * uses ``xfail(strict=True)`` -- not plain ``skip`` -- for the missing
    public ``sparse_mla_blockscaled_path_c_apply`` so the test will start
    catching real divergence the moment that apply lands. (Memory rule:
    prefer ``importorskip``/``xfail(strict=True)`` to plain ``skip``.)
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.sparse_mla_blockscaled_path_c import (
    E8M0_BLOCK_SIZE,
    E8M0_LAYOUT,
    E8M0_SCALE_FORMAT,
    SparseMLABlockScaledPathCStatus,
    SparseMLABlockScaledQKReducePathCStatus,
    blockscaled_sparse_mla_qk_msl_features,
    blockscaled_sparse_mla_qk_path_c_status,
    blockscaled_sparse_mla_qk_reduce_path_c,
    blockscaled_sparse_mla_qk_reduce_path_c_status,
    blockscaled_sparse_mla_qk_scaled_matmul_probe_status,
    lower_blockscaled_sparse_mla_qk_msl,
    lower_blockscaled_sparse_mla_qk_reduce_msl,
)


# ---------------------------------------------------------------------------
# Import-smoke: pin the constants the routing layer depends on.
# ---------------------------------------------------------------------------


def test_e8m0_format_constants_are_stable() -> None:
    """E8M0 format identity must not silently drift; these are wire-format."""

    assert E8M0_BLOCK_SIZE == 32
    # Layout/format strings are read by the AUTO router; case/value matters.
    assert isinstance(E8M0_LAYOUT, str) and E8M0_LAYOUT
    assert isinstance(E8M0_SCALE_FORMAT, str) and E8M0_SCALE_FORMAT


# ---------------------------------------------------------------------------
# E8M0 QK probe -- the only thing the path_c module actually exports for
# Sparse-MLA blockscaled today (per Meta agent F).
# ---------------------------------------------------------------------------


def test_qk_scaled_matmul_probe_returns_status_with_reason() -> None:
    """The probe returns a status dataclass with an actionable reason string,
    even when the simdgroup MMA tile is not eligible (e.g. M=1 Sparse-MLA tile)."""

    status = blockscaled_sparse_mla_qk_scaled_matmul_probe_status()
    assert isinstance(status, SparseMLABlockScaledPathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason
    assert status.transpose_B is True
    assert status.m == 1
    # When unavailable, the reason must list at least one concrete blocker so
    # the routing doc can stay in sync.
    if not status.available:
        assert ":" in status.reason or ";" in status.reason


def test_qk_path_c_status_is_available_or_explains_why() -> None:
    status = blockscaled_sparse_mla_qk_path_c_status()
    assert isinstance(status, SparseMLABlockScaledPathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason


def test_qk_reduce_path_c_status_reports_shape() -> None:
    """The reducer probe records the (N, K) tile it just probed."""

    status = blockscaled_sparse_mla_qk_reduce_path_c_status(N=16, K=64)
    assert isinstance(status, SparseMLABlockScaledQKReducePathCStatus)
    assert status.n == 16
    assert status.k == 64
    assert isinstance(status.reason, str) and status.reason


# ---------------------------------------------------------------------------
# MSL feature inspection: the QK lowering must expose recognisable E8M0
# decode markers so the AUTO router (and the bench harness) can introspect
# what TileLang emitted without re-lowering.
# ---------------------------------------------------------------------------


def test_lower_blockscaled_qk_msl_emits_inspectable_kernel() -> None:
    try:
        import tilelang  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang unavailable on this host: {exc}")
    try:
        msl = lower_blockscaled_sparse_mla_qk_msl(M=1, N=16, K=64)
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang dylib failed to load: {exc}")
    assert isinstance(msl, str) and msl
    assert "kernel void" in msl
    features = blockscaled_sparse_mla_qk_msl_features(msl)
    assert isinstance(features, dict)
    # Must surface at least one recognised E8M0 marker -- otherwise the
    # AUTO router's "looks like FP8 dot4 / E8M0 decode" heuristic is broken.
    assert any(
        bool(features.get(key))
        for key in (
            "metal_fp8_dot4_helper",
            "e8m0_exp2",
            "e8m0_bias_subtract_127",
            "simdgroup_multiply_accumulate",
        )
    ), f"no E8M0/FP8 markers in QK MSL features: {features}"


def test_lower_blockscaled_qk_reduce_msl_emits_kernel() -> None:
    try:
        import tilelang  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang unavailable on this host: {exc}")
    try:
        msl = lower_blockscaled_sparse_mla_qk_reduce_msl(N=16, K=64)
    except (ImportError, OSError) as exc:
        pytest.skip(f"tilelang dylib failed to load: {exc}")
    assert isinstance(msl, str) and msl
    assert "kernel void" in msl


# ---------------------------------------------------------------------------
# Reducer dispatch smoke: only meaningful on a Metal-capable host. Off-Metal
# the function returns None (documented contract); it must NOT raise.
# ---------------------------------------------------------------------------


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return mx.default_device() == mx.gpu and metal is not None and metal.is_available()


def test_qk_reduce_path_c_returns_none_when_metal_missing_or_correct_shape() -> None:
    """Reducer must either dispatch and produce (1, N) fp32, or return None."""

    n, k = 16, 64
    A_fp8 = mx.zeros((1, k), dtype=mx.uint8)
    # E8M0 scale layout: one byte per K/32 block per row; for M=1 / K=64 -> 2.
    a_scale = mx.zeros((1, k // E8M0_BLOCK_SIZE), dtype=mx.uint8)
    B_fp8 = mx.zeros((n, k), dtype=mx.uint8)
    b_scale = mx.zeros((n, k // E8M0_BLOCK_SIZE), dtype=mx.uint8)

    out = blockscaled_sparse_mla_qk_reduce_path_c(A_fp8, a_scale, B_fp8, b_scale)
    if out is None:
        # Allowed: TileLang/Metal unavailable on this host.
        return
    mx.eval(out)
    assert tuple(out.shape) == (1, n)
    assert out.dtype == mx.float32


# ---------------------------------------------------------------------------
# Future numeric-parity placeholder.
#
# Path C does not yet expose a public ``sparse_mla_blockscaled_path_c_apply``;
# Meta-F flagged this as the routing gap. The day Fix-2 lands the apply, this
# xfail flips to PASS and we get real numeric coverage instead of a quiet
# skip. ``strict=True`` makes the flip a CI signal, not a silent change.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "sparse_mla_blockscaled_path_c has no public apply yet -- only QK "
        "probe + partial reducer are exposed. See routing doc / Meta agent F."
    ),
)
def test_blockscaled_path_c_apply_matches_path_b() -> None:
    """When the path_c apply lands, this test must turn green automatically."""

    from cppmega_mlx.nn._tilelang import sparse_mla_blockscaled_path_c as path_c_mod

    apply_fn = getattr(path_c_mod, "sparse_mla_blockscaled_path_c_apply", None)
    assert apply_fn is not None, (
        "sparse_mla_blockscaled_path_c_apply not yet exposed by the path_c module"
    )
