"""Tests for the algebraic-equivalence escalation (track B).

These lock THREE things for each of the two algebraic layers
(z3-concrete-tiles bijection + egglog multiset):

  1. A known-CORRECT tiling (exact ``k_steps*tile_k == k_extent``, matching
     scale/mask) is PROVED, with an HONEST bounded label (``bounded=True``,
     scope names the concrete tiles, NEVER forall-N).
  2. A known-WRONG rewrite (gap k-tile, overshoot k-tile, non-injective
     overlapping tile, mismatched scale, mismatched causal mask, transposed-
     operand-style index disagreement) is REFUTED with a concrete witness and
     ``proved_by="unproven"``. This is the NON-VACUITY guarantee.
  3. A missing backend FAILS LOUD (``AlgebraicBackendUnavailable`` carrying the
     exact install command), never a fabricated proof.

Backends are reached via the same interpreter that runs the suite. If z3 /
egglog are unavailable the relevant tests SKIP (they do not silently pass).
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang.verify_algebraic import (
    AlgebraicBackendUnavailable,
    AlgebraicObligation,
    AlgebraicResult,
    EGGLOG_INSTALL_CMD,
    b2_dinp_obligation,
    egglog_available,
    escalate_algebraic,
    f0_cb_obligation,
    f0_summary_obligation,
    lean_bijection_lemma_sketch,
    prove_bijection_z3_concrete,
    prove_multiset_equiv_egglog,
    z3_available,
)


def _z3_or_skip():
    present, reason = z3_available()
    if not present:
        pytest.skip(reason)


def _egglog_or_skip():
    present, reason = egglog_available()
    if not present:
        pytest.skip(reason)


# --------------------------------------------------------------------------- #
# Layer 1: z3-concrete-tiles bijection prover.                                 #
# --------------------------------------------------------------------------- #


def test_z3_bijection_proves_f0_cb_concrete():
    _z3_or_skip()
    res = prove_bijection_z3_concrete(f0_cb_obligation(dstate=64, tile_k=16))
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"
    assert res.bounded is True
    # RULE #1: scope must name the concrete tiles, never claim forall-N.
    assert "K=64" in res.scope and "tile_k=16" in res.scope
    assert "forall" not in res.detail.lower() or "not forall" in res.detail.lower()


def test_z3_bijection_proves_b2_dinp_masked():
    _z3_or_skip()
    res = prove_bijection_z3_concrete(b2_dinp_obligation(chunk_size=64, tile_k=16))
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"


def test_z3_bijection_proves_f0_summary_scaled():
    _z3_or_skip()
    res = prove_bijection_z3_concrete(f0_summary_obligation(chunk_size=64, tile_k=16))
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"


def test_z3_bijection_refutes_gap_ktile():
    """k_steps too small -> a reduction index is uncovered (dropped product)."""
    _z3_or_skip()
    bad = AlgebraicObligation(name="F0_cb_GAP", k_extent=64, tile_k=16, k_steps=3)
    res = prove_bijection_z3_concrete(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.bounded is True
    assert "uncovered" in res.counter_witness.lower() or "GAP" in res.detail


def test_z3_bijection_refutes_overshoot_ktile():
    """k_steps too large -> tiling reads k >= K (overshoot)."""
    _z3_or_skip()
    bad = AlgebraicObligation(name="F0_cb_OVERSHOOT", k_extent=64, tile_k=16, k_steps=5)
    res = prove_bijection_z3_concrete(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert "overshoot" in res.detail.lower() or "overshoot" in res.counter_witness.lower()


def test_z3_bijection_refutes_mismatched_scale():
    """Serial folds decay*dt; gemm folds a DIFFERENT (transposed-index) scale."""
    _z3_or_skip()
    bad = AlgebraicObligation(
        name="F0_summary_BADSCALE",
        k_extent=64,
        tile_k=16,
        k_steps=4,
        scale_serial=lambda l: f"decay*dt[{l}]",
        scale_gemm=lambda l: f"decay*dt[{(l + 1) % 64}]",  # off-by-one index
    )
    res = prove_bijection_z3_concrete(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert "scale" in res.detail.lower()


def test_z3_bijection_refutes_mismatched_mask():
    """Serial causal keep(k)=(k<=i); gemm uses the WRONG (strict / off-by-one) mask."""
    _z3_or_skip()
    bad = AlgebraicObligation(
        name="B2_dinp_BADMASK",
        k_extent=64,
        tile_k=16,
        k_steps=4,
        mask_serial=lambda k: k <= 32,
        mask_gemm=lambda k: k < 32,  # drops the diagonal -> wrong
    )
    res = prove_bijection_z3_concrete(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert "mask" in res.detail.lower()


# --------------------------------------------------------------------------- #
# Layer 2: egglog multiset equivalence.                                        #
# --------------------------------------------------------------------------- #


def test_egglog_proves_f0_cb_multiset():
    _egglog_or_skip()
    res = prove_multiset_equiv_egglog(f0_cb_obligation(dstate=64, tile_k=16))
    assert res.proved is True
    assert res.proved_by == "egglog"
    assert res.bounded is True
    assert "THIS index domain only" in res.scope


def test_egglog_proves_b2_dinp_masked_multiset():
    _egglog_or_skip()
    res = prove_multiset_equiv_egglog(b2_dinp_obligation(chunk_size=64, tile_k=16))
    assert res.proved is True
    assert res.proved_by == "egglog"


def test_egglog_refutes_dropped_tile():
    """Tiled visits fewer leaves than the serial multiset -> not merged."""
    _egglog_or_skip()
    bad = AlgebraicObligation(name="F0_cb_DROP", k_extent=64, tile_k=16, k_steps=3)
    res = prove_multiset_equiv_egglog(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"


def test_egglog_refutes_overshoot_tile():
    """Tiled visits an extra (out-of-range) leaf -> different multiset."""
    _egglog_or_skip()
    bad = AlgebraicObligation(name="F0_cb_OVER", k_extent=64, tile_k=16, k_steps=5)
    res = prove_multiset_equiv_egglog(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"


def test_egglog_refutes_mismatched_mask():
    _egglog_or_skip()
    bad = AlgebraicObligation(
        name="B2_dinp_BADMASK",
        k_extent=32,
        tile_k=8,
        k_steps=4,
        mask_serial=lambda k: k <= 16,
        mask_gemm=lambda k: k < 16,
    )
    res = prove_multiset_equiv_egglog(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert "mask" in res.detail.lower()


# --------------------------------------------------------------------------- #
# Dispatcher.                                                                  #
# --------------------------------------------------------------------------- #


def test_escalate_prefers_z3_then_egglog():
    _z3_or_skip()
    res = escalate_algebraic(f0_cb_obligation())
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"  # primary


def test_escalate_egglog_first_when_preferred():
    _egglog_or_skip()
    res = escalate_algebraic(f0_cb_obligation(), prefer_egglog=True)
    assert res.proved is True
    assert res.proved_by == "egglog"


def test_escalate_reports_unproven_not_silent_pass():
    """A wrong rewrite no layer can prove returns unproven, never a silent pass."""
    _z3_or_skip()
    _egglog_or_skip()
    bad = AlgebraicObligation(name="ALL_WRONG", k_extent=64, tile_k=16, k_steps=3)
    res = escalate_algebraic(bad)
    assert res.proved is False
    assert res.proved_by == "unproven"


# --------------------------------------------------------------------------- #
# Fail-loud + honesty + design-only Lean sketch.                              #
# --------------------------------------------------------------------------- #


def test_missing_egglog_fails_loud(monkeypatch):
    """If egglog reports unavailable, prove_multiset_equiv_egglog RAISES (no fake proof)."""
    import cppmega_mlx.nn._tilelang.verify_algebraic as va

    monkeypatch.setattr(
        va, "egglog_available", lambda: (False, f"egglog missing; {EGGLOG_INSTALL_CMD}")
    )
    with pytest.raises(AlgebraicBackendUnavailable) as exc:
        va.prove_multiset_equiv_egglog(f0_cb_obligation())
    assert "pip install" in str(exc.value)


def test_egglog_available_reports_install_cmd_when_absent(monkeypatch):
    import cppmega_mlx.nn._tilelang.verify_algebraic as va

    monkeypatch.setattr(va.importlib.util, "find_spec", lambda name: None)
    present, reason = va.egglog_available()
    assert present is False
    assert EGGLOG_INSTALL_CMD in reason


def test_all_results_are_bounded_never_forall():
    """RULE #1: no algebraic result ever claims forall-N."""
    _z3_or_skip()
    for ob in (f0_cb_obligation(), b2_dinp_obligation(), f0_summary_obligation()):
        res = prove_bijection_z3_concrete(ob)
        assert res.bounded is True
        assert "forall" not in res.scope.lower()


def test_as_feature_dict_shape():
    _z3_or_skip()
    res = prove_bijection_z3_concrete(f0_cb_obligation())
    d = res.as_feature_dict()
    assert d["algebraic_proved"] is True
    assert d["algebraic_proved_by"] == "z3-concrete-tiles"
    assert d["algebraic_bounded"] is True


def test_lean_sketch_is_design_only_text():
    sketch = lean_bijection_lemma_sketch()
    assert isinstance(sketch, str)
    assert "finProdFinEquiv" in sketch
    assert "DESIGN ONLY" in sketch
    # design-only: it must NOT pretend to be an executed proof
    assert "theorem tiled_reduction_eq_serial" in sketch
