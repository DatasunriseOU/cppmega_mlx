"""Tests for the layered z3-first verification dispatcher (track tla-tlc).

The dispatcher in ``verify_escalation`` runs z3 FIRST (consuming the existing
``_async_barrier_plan`` / ``_gemm_rewrite_proof`` z3-probe tuple shape) and, on
z3 ``unknown``, classifies the obligation and escalates:
  * RACE-FREEDOM  -> TLA+/TLC bounded model checking,
  * ALGEBRAIC     -> z3-concrete-tiles (primary) + egglog (secondary).

These tests lock in the WHOLE layered contract:
  1. z3 ``unsat`` -> proved_by="z3" (no escalation), bounded=False.
  2. z3 ``sat`` (counter-witness) -> refuted, proved_by="unproven", NOT escalated
     (a real bug must surface, never be papered over — RULE #1 non-vacuity).
  3. z3 ``unknown`` -> ESCALATE; a CORRECT obligation is proved bounded with the
     honest provenance/label, a WRONG one (a real RACE / a dropped k-tile) is
     REFUTED by the fallback layer (NON-VACUITY).
  4. a missing escalation backend -> proved_by="unproven" (fail-loud), never a
     fake pass.

The CRITICAL non-vacuity test for this track: an INJECTED race (two writers,
same cell, tile wider than its stride) on z3-unknown MUST be REFUTED by TLC.
"""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang import _gemm_rewrite_escalation as _esc
from cppmega_mlx.nn._tilelang.verify_escalation import (
    AlgebraicObligation,
    RaceObligation,
    VerificationResult,
    algebraic_obligation_for_k_tiling,
    classify_z3_result,
    race_obligation_for_dense_tiling,
    verify_with_escalation,
)


def _need_tlc():
    ok, why = _esc.tlc_available()
    if not ok:
        pytest.skip(f"TLC unavailable: {why}")


def _need_z3():
    try:
        import z3  # noqa: F401
    except Exception:
        pytest.skip("z3 not importable")


# Z3 probe stubs mimicking the EXACT _async_barrier_plan tuple shape.
def _z3_proved(reason="z3 proved each simdgroup is output-isolated"):
    return lambda: (True, True, reason)


def _z3_unknown(reason="z3 returned unknown for simdgroup isolation"):
    return lambda: (True, False, reason)


def _z3_refuted(reason="z3 found a cross-output simdgroup witness"):
    return lambda: (True, False, reason)


def _z3_disabled():
    return lambda: (False, False, "z3 disabled by environment")


# --------------------------------------------------------------------------- #
# classify_z3_result — the proved / refuted / unknown discriminator.
# --------------------------------------------------------------------------- #
def test_classify_proved():
    assert classify_z3_result(True, True, "z3 proved ...") == "proved"


def test_classify_refuted_by_witness_text():
    assert classify_z3_result(True, False, "z3 found a cross-output witness") == "refuted"


def test_classify_unknown_by_text():
    assert classify_z3_result(True, False, "z3 returned unknown for isolation") == "unknown"


def test_classify_disabled_is_unknown_not_proved():
    # z3 never ran -> escalate (never a fake pass).
    assert classify_z3_result(False, False, "z3 disabled") == "unknown"


def test_classify_ambiguous_false_is_unknown():
    # An unrecognized z3_proved=False reason fails closed to unknown (escalate).
    assert classify_z3_result(True, False, "something inscrutable") == "unknown"


# --------------------------------------------------------------------------- #
# Layer 1: z3 settles it WITHOUT escalation.
# --------------------------------------------------------------------------- #
def test_z3_unsat_is_proved_by_z3_no_escalation():
    res = verify_with_escalation(
        name="ob_proved",
        z3_probe=_z3_proved(),
        obligation=race_obligation_for_dense_tiling(
            name="ob_proved", m_extent=64, n_extent=64, tile_m=16, tile_n=16
        ),
    )
    assert res.proved is True
    assert res.proved_by == "z3"
    assert res.z3_resolved is True
    assert res.escalated is False
    # z3 obligations are quantified -> NOT a bounded claim.
    assert res.bounded is False


def test_z3_sat_is_refuted_and_not_escalated():
    """NON-VACUITY: a z3 counter-witness refutes; escalation must NOT mask it."""
    res = verify_with_escalation(
        name="ob_refuted",
        z3_probe=_z3_refuted(),
        obligation=race_obligation_for_dense_tiling(
            name="ob_refuted", m_extent=64, n_extent=64, tile_m=16, tile_n=16
        ),
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.z3_resolved is True
    assert res.escalated is False  # the bug is surfaced, not escalated away
    assert res.counter_witness is not None


# --------------------------------------------------------------------------- #
# Layer 2 (RACE): z3 unknown -> TLC. Correct PROVED bounded; race REFUTED.
# --------------------------------------------------------------------------- #
def test_race_unknown_escalates_to_tlc_proved_bounded():
    _need_tlc()
    res = verify_with_escalation(
        name="F0_cb_writer",
        z3_probe=_z3_unknown(),
        obligation=race_obligation_for_dense_tiling(
            name="F0_cb_writer", m_extent=64, n_extent=64,
            tile_m=16, tile_n=32,
        ),
    )
    assert res.proved is True
    assert res.proved_by == "tlc-bounded"
    assert res.escalated is True
    assert res.z3_proved is False
    # HONESTY: bounded model checking, NEVER "proven forall N".
    assert res.bounded is True
    assert "NOT proven forall N" in res.reason or "forall N" not in res.reason


def test_injected_race_on_z3_unknown_is_refuted_by_tlc():
    """THE non-vacuity test for this track: an INJECTED race (two writers, same
    cell, tile wider than its stride) reaching the dispatcher on z3-UNKNOWN MUST
    be REFUTED by TLC — never silently passed."""
    _need_tlc()
    racy = race_obligation_for_dense_tiling(
        name="F0_cb_OVERLAP",
        m_extent=64, n_extent=64,
        tile_m=20,           # tile wider than the stride below
        tile_n=32,
        m_step=16,           # rows 16..19 owned by TWO adjacent blocks => race
        n_step=32,
        m_blocks=4, n_blocks=2,
    )
    res = verify_with_escalation(
        name="F0_cb_OVERLAP", z3_probe=_z3_unknown(), obligation=racy
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.escalated is True
    assert res.counter_witness is not None
    assert "violated" in res.counter_witness


# --------------------------------------------------------------------------- #
# Layer 2 (ALGEBRAIC): z3 unknown -> z3-concrete-tiles (primary) / egglog.
# --------------------------------------------------------------------------- #
def test_algebraic_unknown_escalates_to_z3_concrete_proved_bounded():
    _need_z3()
    res = verify_with_escalation(
        name="F0_bijection",
        z3_probe=_z3_unknown("z3 returned unknown (symbolic tile size)"),
        obligation=algebraic_obligation_for_k_tiling(
            name="F0_bijection", k_extent=64, tile_k=16
        ),
    )
    assert res.proved is True
    assert res.proved_by == "z3-concrete-tiles"
    assert res.escalated is True
    assert res.bounded is True
    assert "ONLY these tiles" in res.reason


def test_algebraic_dropped_k_tile_on_unknown_is_refuted():
    """NON-VACUITY (algebraic): a dropped k-tile (gap) is refuted, not passed."""
    _need_z3()
    res = verify_with_escalation(
        name="F0_bijection_GAP",
        z3_probe=_z3_unknown(),
        obligation=AlgebraicObligation(
            name="F0_bijection_GAP", k_extent=64, tile_k=16, k_steps=3
        ),
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.escalated is True
    assert res.counter_witness is not None


# --------------------------------------------------------------------------- #
# Fail-loud: missing escalation backend never reports a fake proof.
# --------------------------------------------------------------------------- #
def test_missing_tlc_backend_is_unproven_not_fake_pass(monkeypatch):
    monkeypatch.setattr(_esc, "_tla2tools_jar", lambda: None)
    res = verify_with_escalation(
        name="x",
        z3_probe=_z3_unknown(),
        obligation=race_obligation_for_dense_tiling(
            name="x", m_extent=4, n_extent=4, tile_m=2, tile_n=2
        ),
    )
    assert res.proved is False
    assert res.proved_by == "unproven"
    assert res.escalated is True
    assert "tla2tools.jar" in res.reason


# --------------------------------------------------------------------------- #
# Dispatcher hygiene + receipt plumbing.
# --------------------------------------------------------------------------- #
def test_unknown_obligation_type_raises():
    with pytest.raises(TypeError):
        verify_with_escalation(
            name="x", z3_probe=_z3_unknown(), obligation=object()  # type: ignore[arg-type]
        )


def test_feature_dict_carries_layered_provenance():
    res = verify_with_escalation(
        name="ob",
        z3_probe=_z3_proved(),
        obligation=race_obligation_for_dense_tiling(
            name="ob", m_extent=64, n_extent=64, tile_m=16, tile_n=16
        ),
    )
    feats = res.as_feature_dict()
    assert feats["verify_proved"] is True
    assert feats["verify_proved_by"] == "z3"
    assert feats["verify_z3_resolved"] is True
    assert feats["verify_escalated"] is False
    assert feats["verify_kind"] == "race"


def test_result_is_immutable():
    res = VerificationResult(
        name="x", kind="race", proved=True, proved_by="z3",
        z3_used=True, z3_proved=True, z3_resolved=True, escalated=False,
        bounded=False, scope="s", reason="r",
    )
    with pytest.raises(Exception):
        res.proved = False  # type: ignore[misc]


def test_disabled_z3_escalates_when_backend_present():
    """z3 disabled -> treated as unknown -> escalate (TLC resolves it bounded)."""
    _need_tlc()
    res = verify_with_escalation(
        name="disabled_then_tlc",
        z3_probe=_z3_disabled(),
        obligation=race_obligation_for_dense_tiling(
            name="disabled_then_tlc", m_extent=64, n_extent=64,
            tile_m=16, tile_n=16,
        ),
    )
    assert res.escalated is True
    assert res.proved is True
    assert res.proved_by == "tlc-bounded"
