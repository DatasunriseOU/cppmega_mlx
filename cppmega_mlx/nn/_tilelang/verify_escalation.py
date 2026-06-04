"""Layered verification dispatcher: z3 FIRST, escalate on ``unknown``.

This is the top-level escalation MACHINERY the user's layered strategy calls
for. It plugs into the EXISTING z3 interface and never rewrites it; it wraps the
SAME ``(proved, refuted, unknown)`` pattern the in-repo z3 provers expose and,
when z3 returns UNKNOWN, classifies the obligation and dispatches:

  1. z3 FIRST (fast, symbolic). The probe is a python callable returning the
     ``_async_barrier_plan._z3_proves_simdgroup_output_isolated`` shape
     ``(z3_used: bool, z3_proved: bool, reason: str)`` — exactly the tuple that
     prover returns, including the literal ``"z3 returned unknown ..."`` string
     on ``z3.unknown``. That string IS the escalation trigger.

  2. On z3 UNKNOWN, classify by category and escalate:

       RACE-FREEDOM  -> a TLA+ spec of the tiled-GEMM writer state machine + a
                        single-writer safety invariant, model-checked with TLC
                        over BOUNDED representative sizes (exhaustive
                        interleavings). Delegates to the validated
                        ``_gemm_rewrite_escalation.escalate_obligation`` /
                        ``prove_race_freedom_tlc`` layer. CRITICAL HONESTY: TLC
                        is BOUNDED model checking -> labelled
                        ``"verified bounded @ tile=X dim=Y"``, provenance
                        ``"tlc-bounded"``, NEVER "proven forall N".

       ALGEBRAIC     -> the tiling-bijection / multiset-of-products equivalence
                        that z3's nonlinear-integer-arith returns unknown on.
                        Delegates to the ``_gemm_rewrite_escalation`` algebraic
                        layer: z3-concrete-tiles (PRIMARY, provenance
                        ``"z3-concrete-tiles"``) then egglog (SECONDARY,
                        provenance ``"egglog"``). Proves ONLY the concrete tiles
                        / index domain checked -> bounded.

  3. Everything is backstopped by the all-elements numeric parity test (the fp
     ground-truth gate), which lives elsewhere and is NOT a substitute for the
     structural proof claim recorded here.

RULE #1 (no silent overclaim, fail-fast/fail-loud):
  * A z3 ``unsat`` is recorded ``proved_by="z3"`` (the only ∀-over-the-encoded-
    domain claim, since the z3 obligations are quantified).
  * A z3 ``sat`` (counter-witness) is recorded refuted, ``proved_by="unproven"``
    — the rewrite is WRONG, surfaced loudly, NOT escalated to paper over it.
  * A TLC pass is ``proved_by="tlc-bounded"`` with ``bounded=True`` and a scope
    string; the label NEVER reads "proven forall N".
  * z3-concrete-tiles / egglog are bounded to the checked shapes -> labelled.
  * An obligation NO layer resolves is ``proved_by="unproven"`` (NOT silently
    passed). A missing backend (TLC jar, egglog) RAISES from the underlying
    layer; this dispatcher surfaces that as ``unproven`` with the fetch command
    in the reason — never a fake proof.

Provenance is carried on every result via ``proved_by``:
    "z3" | "tlc-bounded" | "z3-concrete-tiles" | "egglog" | "unproven"

This module is standalone against the ``_async_barrier_plan`` /
``_gemm_rewrite_proof`` / ``_gemm_rewrite_escalation`` interfaces. Wiring it into
the running track-C prover (w8ctouyfx) is a drop-in: call
:func:`verify_with_escalation` wherever that prover gets a z3 result, handing it
the z3 probe + the race / algebraic descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from . import _gemm_rewrite_escalation as _esc

# Re-export so callers wire against ONE module.
AlgebraicObligation = _esc.AlgebraicObligation
RaceObligation = _esc.RaceObligation
EscalationResult = _esc.EscalationResult
EscalationUnavailable = _esc.EscalationUnavailable

ProvedBy = _esc.ProvedBy  # "z3" | "tlc-bounded" | "z3-concrete-tiles" | "egglog" | "unproven"
ObligationKind = _esc.ObligationKind  # "race" | "algebraic"

# The z3 probe contract: a zero-arg callable returning the EXACT tuple shape
# ``_async_barrier_plan._z3_proves_simdgroup_output_isolated`` returns:
#   (z3_used: bool, z3_proved: bool, reason: str)
# On z3.unknown the prover returns (True, False, "z3 returned unknown ...") and
# THAT is the escalation trigger; on z3.unsat (True, True, "...proved...") it is
# done; on z3.sat (True, False, "...witness...") the rewrite is refuted.
Z3Probe = Callable[[], "tuple[bool, bool, str]"]

# The substring the in-repo z3 provers put in their reason on ``z3.unknown``
# (see ``_async_barrier_plan`` line 111 and ``_gemm_rewrite_proof`` _discharge).
# Used to DISTINGUISH unknown (escalate) from a sat counter-witness (refute):
# both return z3_proved=False, so we classify by the reason text.
_Z3_UNKNOWN_MARKERS = ("returned unknown", "z3 unknown", "unknown for")
_Z3_REFUTED_MARKERS = ("witness", "counter-witness", "sat ", "found a")


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of the layered (z3-first, escalate-on-unknown) verification.

    Mirrors / extends ``MetalReductionSyncPlan`` and ``GemmRewriteProof``:
    carries the ``(z3_used, z3_proved)`` pair PLUS the escalation provenance
    (``proved_by``, ``bounded``, ``scope``) the task asked for, and flows into
    the same receipt plumbing via :meth:`as_feature_dict`.

    ``proved`` is True iff SOME layer discharged the obligation. ``z3_resolved``
    flags whether z3 alone settled it (proved OR refuted) without escalating.
    ``escalated`` flags that z3 returned unknown and a fallback layer ran.
    ``bounded`` is True when the proof holds only at the checked sizes/tiles
    (TLC bounded, z3-concrete-tiles, egglog) so callers never over-claim ∀N.
    """

    name: str
    kind: ObligationKind
    proved: bool
    proved_by: ProvedBy
    z3_used: bool
    z3_proved: bool
    z3_resolved: bool
    escalated: bool
    bounded: bool
    scope: str
    reason: str
    counter_witness: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_feature_dict(self) -> dict[str, object]:
        return {
            "verify_name": self.name,
            "verify_kind": self.kind,
            "verify_proved": self.proved,
            "verify_proved_by": self.proved_by,
            "verify_z3_used": self.z3_used,
            "verify_z3_proved": self.z3_proved,
            "verify_z3_resolved": self.z3_resolved,
            "verify_escalated": self.escalated,
            "verify_bounded": self.bounded,
            "verify_scope": self.scope,
            "verify_reason": self.reason,
            "verify_counter_witness": self.counter_witness or "",
        }


def classify_z3_result(z3_used: bool, z3_proved: bool, reason: str) -> str:
    """Classify a z3 probe tuple into one of: proved | refuted | unknown.

    z3's ``check()`` is collapsed by the in-repo provers to a 3-tuple where both
    ``sat`` (counter-witness, rewrite WRONG) and ``unknown`` (not decided) report
    ``z3_proved=False``. We MUST distinguish them: a ``sat`` is a real bug to
    surface (do NOT escalate to hide it), an ``unknown`` is the escalation
    trigger. We classify by the reason text the provers emit (RULE #1: a probe
    whose reason is ambiguous is treated as ``unknown`` and escalated, never
    silently called proved).
    """
    if z3_proved:
        return "proved"
    if not z3_used:
        # z3 never ran (disabled / unavailable) -> treat as unknown -> escalate
        # (the fallback either resolves it or reports unproven; never a fake pass).
        return "unknown"
    low = reason.lower()
    if any(m in low for m in _Z3_UNKNOWN_MARKERS):
        return "unknown"
    if any(m in low for m in _Z3_REFUTED_MARKERS):
        return "refuted"
    # Ambiguous z3_proved=False reason: fail-closed to unknown (escalate), never
    # assume proved.
    return "unknown"


def _result_from_escalation(
    esc: EscalationResult,
    *,
    z3_used: bool,
    z3_reason: str,
) -> VerificationResult:
    """Lift an ``EscalationResult`` (TLC / z3-concrete / egglog) to the top-level
    ``VerificationResult``, stamping that z3 ran first and escalated."""
    return VerificationResult(
        name=esc.name,
        kind=esc.kind,
        proved=esc.proved,
        proved_by=esc.proved_by,
        z3_used=z3_used,
        z3_proved=False,  # z3 itself did not prove it (it returned unknown)
        z3_resolved=False,
        escalated=True,
        bounded=esc.bounded,
        scope=esc.scope,
        reason=f"z3 unknown ({z3_reason}); escalated -> {esc.reason}",
        counter_witness=esc.counter_witness,
        notes=esc.notes,
    )


def verify_with_escalation(
    *,
    name: str,
    z3_probe: Z3Probe,
    obligation: "AlgebraicObligation | RaceObligation",
    prefer_egglog: bool = False,
) -> VerificationResult:
    """Run the layered strategy for ONE obligation: z3 first, escalate on unknown.

    Parameters
    ----------
    name:
        Human label for the obligation (flows into receipts).
    z3_probe:
        Zero-arg callable returning ``(z3_used, z3_proved, reason)`` — the EXACT
        shape ``_async_barrier_plan._z3_proves_simdgroup_output_isolated`` and
        the ``_gemm_rewrite_proof`` obligations return. The dispatcher does NOT
        build the z3 query itself; it consumes the existing prover's verdict so
        there is ONE z3 path (RULE #1: one clear path, no shadow re-encoding).
    obligation:
        The descriptor used for escalation if z3 is unknown. A
        :class:`RaceObligation` escalates to TLC (bounded); an
        :class:`AlgebraicObligation` escalates to z3-concrete-tiles then egglog.
        Its ``kind`` ALSO classifies the obligation, so the category dispatch is
        explicit, not guessed.
    prefer_egglog:
        For algebraic obligations, run egglog as the primary instead of
        z3-concrete-tiles (the two are cross-checks).

    Returns
    -------
    VerificationResult
        ``proved_by`` records which layer settled it:
          * ``"z3"`` — z3 proved it (unsat). The only claim that is ∀ over the
            z3-encoded (quantified) domain; ``bounded=False``.
          * ``"unproven"`` + ``z3_resolved`` — z3 REFUTED it (sat witness). The
            rewrite is WRONG; surfaced loudly, NOT escalated to mask it.
          * ``"tlc-bounded"`` / ``"z3-concrete-tiles"`` / ``"egglog"`` — a
            fallback layer discharged it; ``bounded=True``, scope says exactly
            what was checked.
          * ``"unproven"`` + ``escalated`` — z3 unknown AND no fallback layer
            resolved it (or the backend was unavailable). NOT a pass.
    """
    if isinstance(obligation, RaceObligation):
        kind: ObligationKind = "race"
    elif isinstance(obligation, AlgebraicObligation):
        kind = "algebraic"
    else:
        raise TypeError(
            f"obligation must be RaceObligation or AlgebraicObligation, got "
            f"{type(obligation).__name__}"
        )

    # --- Layer 1: z3 FIRST (consume the existing prover's verdict). ---
    z3_used, z3_proved, z3_reason = z3_probe()
    verdict = classify_z3_result(z3_used, z3_proved, z3_reason)

    if verdict == "proved":
        return VerificationResult(
            name=name,
            kind=kind,
            proved=True,
            proved_by="z3",
            z3_used=True,
            z3_proved=True,
            z3_resolved=True,
            escalated=False,
            bounded=False,  # z3 obligations are quantified over the index domain
            scope="z3 symbolic (quantified over the encoded index domain)",
            reason=f"z3 proved obligation '{name}' (unsat): {z3_reason}",
        )

    if verdict == "refuted":
        # z3 found a concrete counter-witness: the rewrite is WRONG. Do NOT
        # escalate — escalation must never paper over a real bug (RULE #1).
        return VerificationResult(
            name=name,
            kind=kind,
            proved=False,
            proved_by="unproven",
            z3_used=True,
            z3_proved=False,
            z3_resolved=True,
            escalated=False,
            bounded=False,
            scope="z3 symbolic",
            reason=(
                f"z3 REFUTED obligation '{name}': a concrete counter-witness "
                f"shows the rewrite is WRONG (NOT escalated): {z3_reason}"
            ),
            counter_witness=z3_reason,
        )

    # verdict == "unknown" -> Layer 2: classify + escalate.
    try:
        esc = _esc.escalate_obligation(obligation, prefer_egglog=prefer_egglog)
    except EscalationUnavailable as exc:
        # A required backend (TLC jar / egglog) is missing. Fail-LOUD: record
        # unproven WITH the fetch command from the backend, never a fake pass.
        return VerificationResult(
            name=name,
            kind=kind,
            proved=False,
            proved_by="unproven",
            z3_used=z3_used,
            z3_proved=False,
            z3_resolved=False,
            escalated=True,
            bounded=False,
            scope="escalation backend unavailable",
            reason=(
                f"z3 unknown ({z3_reason}); escalation backend UNAVAILABLE -> "
                f"obligation '{name}' is UNPROVEN (NOT silently passed): {exc}"
            ),
        )

    return _result_from_escalation(esc, z3_used=z3_used, z3_reason=z3_reason)


# --------------------------------------------------------------------------- #
# Convenience constructors mirroring the real F0 / B2 obligations, so the
# track-C prover (and the test) escalate the obligation it actually emits.
# --------------------------------------------------------------------------- #
def race_obligation_for_dense_tiling(
    *,
    name: str,
    m_extent: int,
    n_extent: int,
    tile_m: int,
    tile_n: int,
    m_step: Optional[int] = None,
    n_step: Optional[int] = None,
    m_blocks: Optional[int] = None,
    n_blocks: Optional[int] = None,
) -> RaceObligation:
    """Build the single-writer RaceObligation for a dense tiled-GEMM writer.

    Defaults encode the CORRECT disjoint-contiguous tiling (step == tile,
    blocks == ceil(extent/step)); pass a larger ``tile`` than ``step`` to model
    a real overlap (the non-vacuity case the TLC layer must refute).
    """
    ms = tile_m if m_step is None else m_step
    ns = tile_n if n_step is None else n_step
    mb = ((m_extent + ms - 1) // ms) if m_blocks is None else m_blocks
    nb = ((n_extent + ns - 1) // ns) if n_blocks is None else n_blocks
    return RaceObligation(
        name=name,
        m_extent=m_extent,
        n_extent=n_extent,
        tile_m=tile_m,
        tile_n=tile_n,
        m_step=ms,
        n_step=ns,
        m_blocks=mb,
        n_blocks=nb,
    )


def algebraic_obligation_for_k_tiling(
    *,
    name: str,
    k_extent: int,
    tile_k: int,
    k_steps: Optional[int] = None,
) -> AlgebraicObligation:
    """Build the tiling-bijection AlgebraicObligation over the reduction axis k.

    Default ``k_steps = ceil(k_extent / tile_k)`` is the CORRECT covering;
    pass a smaller ``k_steps`` to model a dropped k-tile (gap) or a larger one
    to model overshoot — the non-vacuity cases the algebraic layer must refute.
    """
    ks = ((k_extent + tile_k - 1) // tile_k) if k_steps is None else k_steps
    return AlgebraicObligation(
        name=name, k_extent=k_extent, tile_k=tile_k, k_steps=ks
    )
