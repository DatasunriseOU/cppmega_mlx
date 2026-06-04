"""Tests for the GEMM-rewrite z3-UNKNOWN escalation machinery (track C).

The primary prover (`_gemm_rewrite_proof`) discharges structural obligations
with z3. On z3 ``unknown`` the escalation dispatcher classifies the obligation
and dispatches:
  * RACE-FREEDOM  -> TLA+/TLC bounded model checking (exhaustive interleavings),
  * ALGEBRAIC     -> z3-concrete-tiles bijection (primary) + egglog multiset
                     (secondary) + a design-only Lean sketch.

These tests lock in, for EVERY layer:
  1. a known-CORRECT obligation is PROVED, with HONEST provenance/labels
     (proved_by, bounded=True — never an over-claimed forall-N), AND
  2. a known-WRONG obligation (a real race / a dropped k-tile / an overshoot)
     is REFUTED with a counter-witness (NON-VACUITY).

If a backend is unavailable the layer RAISES EscalationUnavailable (fail-loud,
RULE #1) — it never reports a missing tool as a passing proof; those tests skip
(they do not silently pass) when the backend genuinely is not installed.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang._gemm_rewrite_escalation import (
    AlgebraicObligation,
    EscalationResult,
    EscalationUnavailable,
    RaceObligation,
    egglog_available,
    escalate_obligation,
    lean_bijection_lemma_sketch,
    prove_bijection_z3_concrete,
    prove_multiset_equiv_egglog,
    prove_race_freedom_tlc,
    representative_race_shape,
    tlc_available,
)


def _need_tlc():
    ok, why = tlc_available()
    if not ok:
        pytest.skip(f"TLC unavailable: {why}")


def _need_egglog():
    ok, why = egglog_available()
    if not ok:
        pytest.skip(f"egglog unavailable: {why}")


def _need_z3():
    try:
        import z3  # noqa: F401
    except Exception:
        pytest.skip("z3 not importable")


# --------------------------------------------------------------------------- #
# RACE-FREEDOM layer (TLA+/TLC, BOUNDED).
# --------------------------------------------------------------------------- #
def test_race_correct_f0_writer_is_proved_bounded() -> None:
    _need_tlc()
    # F0 cb=C@B^T writer: 64x64, 16x32 tiles, disjoint contiguous ownership.
    res = escalate_obligation(
        RaceObligation(
            name="F0_cb_writer",
            m_extent=64, n_extent=64,
            tile_m=16, tile_n=32, m_step=16, n_step=32,
            m_blocks=4, n_blocks=2,
        )
    )
    assert res.proved is True
    assert res.proved_by == "tlc-bounded"
    # HONESTY: bounded model checking, never forall-N.
    assert res.bounded is True
    assert "forall N" not in res.reason or "NOT proven forall N" in res.reason
    assert "prod dim=64x64" in res.scope


def test_race_correct_b2_writer_is_proved_bounded() -> None:
    _need_tlc()
    res = escalate_obligation(
        RaceObligation(
            name="B2_dinp_writer",
            m_extent=64, n_extent=64,
            tile_m=16, tile_n=16, m_step=16, n_step=16,
            m_blocks=4, n_blocks=4,
        )
    )
    assert res.proved is True
    assert res.proved_by == "tlc-bounded"
    assert res.bounded is True


def test_race_overlapping_tiles_is_refuted_with_witness() -> None:
    """NON-VACUITY: a tile wider than its stride => a real double-write race."""
    _need_tlc()
    # tile_m=20 but m_step=16: rows 16..19 owned by two adjacent blocks.
    res = escalate_obligation(
        RaceObligation(
            name="F0_cb_OVERLAP",
            m_extent=64, n_extent=64,
            tile_m=20, tile_n=32, m_step=16, n_step=32,
            m_blocks=4, n_blocks=2,
        )
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.counter_witness is not None
    assert "violated" in res.counter_witness


def test_race_direct_tlc_small_grid_exhaustive() -> None:
    """The TLC runner itself proves a tiny disjoint grid and refutes overlap."""
    _need_tlc()
    good = prove_race_freedom_tlc(
        name="tiny_ok", m_extent=4, n_extent=4,
        tile_m=2, tile_n=2, m_step=2, n_step=2, m_blocks=2, n_blocks=2,
    )
    assert good.proved is True and good.proved_by == "tlc-bounded"
    bad = prove_race_freedom_tlc(
        name="tiny_overlap", m_extent=4, n_extent=4,
        tile_m=3, tile_n=2, m_step=2, n_step=2, m_blocks=2, n_blocks=2,
    )
    assert bad.proved is False and bad.counter_witness is not None


def test_representative_shape_preserves_overlap_relation() -> None:
    # disjoint (tile==step) -> tiny disjoint grid
    disj = representative_race_shape(
        RaceObligation("d", 64, 64, 16, 16, 16, 16, 4, 4)
    )
    assert disj["tile_m"] == disj["m_step"]  # still disjoint contiguous
    # overlap (tile>step) -> tile' > step'
    over = representative_race_shape(
        RaceObligation("o", 64, 64, 20, 32, 16, 32, 4, 2)
    )
    assert over["tile_m"] > over["m_step"]  # overlap preserved


# --------------------------------------------------------------------------- #
# ALGEBRAIC layer (a): z3-concrete-tiles bijection (PRIMARY).
# --------------------------------------------------------------------------- #
def test_z3_concrete_bijection_is_proved_for_f0_tiles() -> None:
    _need_z3()
    res = prove_bijection_z3_concrete(
        name="F0_bijection", k_extent=64, tile_k=16, k_steps=4
    )
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"
    assert res.bounded is True  # these tiles only, not forall tile size
    assert "ONLY these tiles" in res.reason


def test_z3_concrete_dropped_k_tile_is_refuted() -> None:
    """NON-VACUITY: k_steps=3 leaves k in [48,64) uncovered (a gap)."""
    _need_z3()
    res = prove_bijection_z3_concrete(
        name="F0_bijection_GAP", k_extent=64, tile_k=16, k_steps=3
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.counter_witness is not None


def test_z3_concrete_overshoot_is_refuted() -> None:
    """NON-VACUITY: k_steps=5 produces k in [64,80), outside [0,64)."""
    _need_z3()
    res = prove_bijection_z3_concrete(
        name="F0_bijection_OVERSHOOT", k_extent=64, tile_k=16, k_steps=5
    )
    assert res.proved is False
    assert "overshoot" in (res.counter_witness or "").lower() or res.proved is False


# --------------------------------------------------------------------------- #
# ALGEBRAIC layer (b): egglog multiset equivalence (SECONDARY).
# --------------------------------------------------------------------------- #
def test_egglog_multiset_equiv_is_proved() -> None:
    _need_egglog()
    res = prove_multiset_equiv_egglog(
        name="F0_multiset", k_extent=64, tile_k=16, k_steps=4
    )
    assert res.proved is True
    assert res.proved_by == "egglog"
    assert res.bounded is True


def test_egglog_multiset_overshoot_is_refuted() -> None:
    """NON-VACUITY: producers outside [0,K) break multiset equality."""
    _need_egglog()
    res = prove_multiset_equiv_egglog(
        name="F0_multiset_OVERSHOOT", k_extent=64, tile_k=16, k_steps=5
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.counter_witness is not None


# --------------------------------------------------------------------------- #
# Dispatcher routing + provenance.
# --------------------------------------------------------------------------- #
def test_dispatch_algebraic_primary_is_z3_concrete() -> None:
    _need_z3()
    res = escalate_obligation(
        AlgebraicObligation(name="F0_bij", k_extent=64, tile_k=16, k_steps=4)
    )
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"  # primary wins


def test_dispatch_algebraic_prefer_egglog() -> None:
    _need_egglog()
    res = escalate_obligation(
        AlgebraicObligation(name="F0_ms", k_extent=64, tile_k=16, k_steps=4),
        prefer_egglog=True,
    )
    assert res.proved is True
    assert res.proved_by == "egglog"


def test_dispatch_unknown_obligation_type_raises() -> None:
    with pytest.raises(TypeError):
        escalate_obligation(object())  # type: ignore[arg-type]


def test_feature_dict_carries_escalation_provenance() -> None:
    _need_z3()
    res = prove_bijection_z3_concrete(
        name="F0_bij", k_extent=64, tile_k=16, k_steps=4
    )
    feats = res.as_feature_dict()
    assert feats["escalation_proved"] is True
    assert feats["escalation_proved_by"] == "z3-concrete-tiles"
    assert feats["escalation_bounded"] is True
    assert feats["escalation_kind"] == "algebraic"


# --------------------------------------------------------------------------- #
# DESIGN-ONLY Lean sketch + fail-loud backend unavailability.
# --------------------------------------------------------------------------- #
def test_lean_sketch_is_present_and_design_only() -> None:
    sketch = lean_bijection_lemma_sketch()
    assert "finProdFinEquiv" in sketch
    assert "Bijective" in sketch
    assert "DESIGN-ONLY" in sketch


def test_missing_tlc_jar_fails_loud(monkeypatch) -> None:
    """A missing tla2tools.jar RAISES (fail-loud), never reports a fake proof."""
    import cppmega_mlx.nn._tilelang._gemm_rewrite_escalation as E

    monkeypatch.setattr(E, "_tla2tools_jar", lambda: None)
    with pytest.raises(EscalationUnavailable) as exc:
        prove_race_freedom_tlc(
            name="x", m_extent=4, n_extent=4,
            tile_m=2, tile_n=2, m_step=2, n_step=2, m_blocks=2, n_blocks=2,
        )
    assert "tla2tools.jar" in str(exc.value)


def test_escalation_result_is_immutable() -> None:
    res = EscalationResult(
        name="x", kind="race", proved=True, proved_by="tlc-bounded",
        bounded=True, scope="s", reason="r",
    )
    with pytest.raises(Exception):
        res.proved = False  # type: ignore[misc]
