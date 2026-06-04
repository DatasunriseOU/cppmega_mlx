"""Tests for the z3 GEMM-rewrite prover (track C).

These lock in TWO things:
  1. A known-CORRECT serial-reduction -> T.gemm rewrite is PROVED (z3_proved).
  2. A known-WRONG rewrite (transposed operand, off-by-one tile, dropped k-tile,
     wrong mask, off-by-one scale) is REFUSED with a concrete counter-witness.

(2) is the NON-VACUITY guarantee: the proof can actually fail. A tautological
encoding would pass (1) AND (2), so a test that asserts the broken cases FAIL is
what proves the prover is non-vacuous.

z3 4.15.4 is reached via the same interpreter that runs the suite. If z3 is
unavailable the prover reports z3_used=False / z3_proved=False (fail-closed),
and the import-guarded tests skip — they do not silently pass.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang._gemm_rewrite_proof import (
    GemmContraction,
    GemmRewriteNotProven,
    GemmTiling,
    b2_dinp_contraction,
    f0_cb_contraction,
    f0_summary_contraction,
    prove_gemm_rewrite,
    require_gemm_rewrite_proof,
)


def _z3_or_skip():
    try:
        import z3  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - environment without z3
        pytest.skip("z3 not importable in this environment")
    return z3


# A standard correct dense tiling for a 64x64x64 contraction: 16x32x16 tiles.
def _dense_tiling() -> GemmTiling:
    return GemmTiling(tile_m=16, tile_n=32, tile_k=16, m_blocks=4, n_blocks=2, k_steps=4)


# --------------------------------------------------------------------------- #
# CORRECT rewrites PASS.
# --------------------------------------------------------------------------- #


def test_f0_cb_rewrite_is_proved() -> None:
    z3 = _z3_or_skip()
    contraction = f0_cb_contraction(z3, chunk_size=64, dstate=64)
    proof = prove_gemm_rewrite(contraction, _dense_tiling())
    assert proof.z3_used is True
    assert proof.z3_proved is True, proof.reason
    assert proof.operand_maps_match is True
    assert proof.tiling_injective is True
    assert proof.k_covered is True
    assert proof.single_writer is True
    # Feature dict flows the proof into the same receipt plumbing.
    feats = proof.as_feature_dict()
    assert feats["gemm_proof_z3_proved"] is True
    assert feats["gemm_proof_name"] == "F0_cb=C@B^T"


def test_b2_dinp_lowertri_mask_rewrite_is_proved() -> None:
    z3 = _z3_or_skip()
    contraction = b2_dinp_contraction(z3, chunk_size=64, headdim=64)
    proof = prove_gemm_rewrite(contraction, _dense_tiling())
    assert proof.z3_proved is True, proof.reason
    assert proof.mask_equiv is True


def test_f0_summary_scale_fold_rewrite_is_proved() -> None:
    z3 = _z3_or_skip()
    contraction = f0_summary_contraction(z3, chunk_size=64, headdim=64, dstate=64)
    proof = prove_gemm_rewrite(contraction, _dense_tiling())
    assert proof.z3_proved is True, proof.reason
    assert proof.scale_equiv is True


def test_require_gate_returns_proof_for_correct_rewrite() -> None:
    z3 = _z3_or_skip()
    contraction = f0_cb_contraction(z3, chunk_size=64, dstate=64)
    proof = require_gemm_rewrite_proof(contraction, _dense_tiling())
    assert proof.z3_proved is True


# --------------------------------------------------------------------------- #
# WRONG rewrites FAIL (non-vacuity). Each builds a deliberately broken variant
# and asserts z3 refuses it AND surfaces a counter-witness in the reason.
# --------------------------------------------------------------------------- #


def test_transposed_operand_rewrite_is_refused() -> None:
    z3 = _z3_or_skip()
    # GEMM reads A transposed (k*M+i) while the serial loop reads A[i,k]=i*K+k.
    broken = GemmContraction(
        name="F0_cb_TRANSPOSED_A",
        m_extent=64,
        n_extent=64,
        k_extent=64,
        a_addr_serial=lambda i, k: i * 64 + k,
        a_addr_gemm=lambda i, k: k * 64 + i,  # transpose_A bug
        b_addr_serial=lambda k, j: j * 64 + k,
        b_addr_gemm=lambda k, j: j * 64 + k,
    )
    proof = prove_gemm_rewrite(broken, _dense_tiling())
    assert proof.z3_used is True
    assert proof.z3_proved is False
    assert proof.operand_maps_match is False
    assert "COUNTER-WITNESS" in proof.reason


def test_off_by_one_k_tile_is_refused() -> None:
    z3 = _z3_or_skip()
    contraction = f0_cb_contraction(z3, chunk_size=64, dstate=64)
    # Drop the last k-tile: k_steps=3 only covers k in [0,48), leaving [48,64).
    broken_tiling = GemmTiling(
        tile_m=16, tile_n=32, tile_k=16, m_blocks=4, n_blocks=2, k_steps=3
    )
    proof = prove_gemm_rewrite(contraction, broken_tiling)
    assert proof.z3_proved is False
    assert proof.k_covered is False
    assert "COUNTER-WITNESS" in proof.reason


def test_overlapping_output_tiles_are_refused() -> None:
    z3 = _z3_or_skip()
    contraction = f0_cb_contraction(z3, chunk_size=64, dstate=64)
    # Each block WRITES tile_m=20 rows but STEPS only m_stride=16: blocks
    # overlap, so rows 16..19 are owned by both block 0 and block 1.
    broken_tiling = GemmTiling(
        tile_m=20, tile_n=32, tile_k=16, m_blocks=4, n_blocks=2, k_steps=4, m_stride=16
    )
    proof = prove_gemm_rewrite(contraction, broken_tiling)
    assert proof.z3_proved is False
    assert proof.single_writer is False
    assert "COUNTER-WITNESS" in proof.reason


def test_wrong_causal_mask_is_refused() -> None:
    z3 = _z3_or_skip()
    # Serial keeps i>=k (inclusive diagonal); the GEMM applies strict i>k,
    # dropping the diagonal — a real off-by-one in the causal mask.
    broken = GemmContraction(
        name="B2_dinp_WRONG_MASK",
        m_extent=64,
        n_extent=64,
        k_extent=64,
        a_addr_serial=lambda i, k: i * 64 + k,
        a_addr_gemm=lambda i, k: i * 64 + k,
        b_addr_serial=lambda k, j: k * 64 + j,
        b_addr_gemm=lambda k, j: k * 64 + j,
        mask_serial=lambda i, j, k: i >= k,
        mask_gemm=lambda i, j, k: i > k,  # strict: drops the diagonal
    )
    proof = prove_gemm_rewrite(broken, _dense_tiling())
    assert proof.z3_proved is False
    assert proof.mask_equiv is False
    assert "COUNTER-WITNESS" in proof.reason


def test_off_by_one_scale_fold_is_refused() -> None:
    z3 = _z3_or_skip()
    scale = z3.Function("s", z3.IntSort(), z3.RealSort())
    # GEMM folds scale(k+1) instead of scale(k) — an off-by-one in the decay
    # fold. The prover only needs ONE index where the base function differs.
    broken = GemmContraction(
        name="F0_summary_WRONG_SCALE",
        m_extent=64,
        n_extent=64,
        k_extent=64,
        a_addr_serial=lambda i, k: k * 64 + i,
        a_addr_gemm=lambda i, k: k * 64 + i,
        b_addr_serial=lambda k, j: k * 64 + j,
        b_addr_gemm=lambda k, j: k * 64 + j,
        scale_serial=lambda i, k, j: scale(k),
        scale_gemm=lambda i, k, j: scale(k + 1),  # off-by-one fold
    )
    proof = prove_gemm_rewrite(broken, _dense_tiling())
    assert proof.z3_proved is False
    assert proof.scale_equiv is False
    assert "COUNTER-WITNESS" in proof.reason


def test_require_gate_raises_on_wrong_rewrite() -> None:
    """RULE #1: a forced rewrite that cannot be proven RAISES, never returns."""
    z3 = _z3_or_skip()
    broken = GemmContraction(
        name="F0_cb_TRANSPOSED_A",
        m_extent=64,
        n_extent=64,
        k_extent=64,
        a_addr_serial=lambda i, k: i * 64 + k,
        a_addr_gemm=lambda i, k: k * 64 + i,
        b_addr_serial=lambda k, j: j * 64 + k,
        b_addr_gemm=lambda k, j: j * 64 + k,
    )
    with pytest.raises(GemmRewriteNotProven) as excinfo:
        require_gemm_rewrite_proof(broken, _dense_tiling())
    assert "refused" in str(excinfo.value)
    assert "F0_cb_TRANSPOSED_A" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Policy / fail-closed behavior.
# --------------------------------------------------------------------------- #


def test_disabled_policy_keeps_serial_path() -> None:
    z3 = _z3_or_skip()
    contraction = f0_cb_contraction(z3, chunk_size=64, dstate=64)
    proof = prove_gemm_rewrite(contraction, _dense_tiling(), z3_policy="disabled")
    # z3 not run => not used, not proved => caller keeps the serial loop.
    assert proof.z3_used is False
    assert proof.z3_proved is False
    assert "disabled" in proof.reason


def test_disabled_policy_makes_require_gate_raise() -> None:
    """Even a CORRECT rewrite is refused when z3 is disabled (fail-closed)."""
    z3 = _z3_or_skip()
    contraction = f0_cb_contraction(z3, chunk_size=64, dstate=64)
    with pytest.raises(GemmRewriteNotProven):
        require_gemm_rewrite_proof(contraction, _dense_tiling(), z3_policy="disabled")


def test_non_positive_shape_is_not_proved() -> None:
    z3 = _z3_or_skip()
    contraction = GemmContraction(
        name="degenerate",
        m_extent=0,
        n_extent=64,
        k_extent=64,
        a_addr_serial=lambda i, k: i,
        a_addr_gemm=lambda i, k: i,
        b_addr_serial=lambda k, j: j,
        b_addr_gemm=lambda k, j: j,
    )
    proof = prove_gemm_rewrite(contraction, _dense_tiling())
    assert proof.z3_used is False
    assert proof.z3_proved is False
    assert "non-positive" in proof.reason
