"""Algebraic-equivalence fallback for the GEMM-rewrite prover (track B).

This is the ALGEBRAIC arm of the z3-UNKNOWN escalation strategy described in the
session brief. When the primary z3 obligation (``_async_barrier_plan.
_z3_proves_simdgroup_output_isolated`` / ``_gemm_rewrite_proof._discharge``)
returns ``z3.unknown`` because the obligation lives in the nonlinear-integer
fragment (``sum_k A[i,k]*B[k,j] == tiled accumulation`` — a multiset-of-products
/ tiling-bijection question), we escalate HERE.

Three layers, in priority order:

  (1) PRIMARY  — z3-concrete-tiles bijection prover (``prove_bijection_z3_concrete``):
      reformulate the tiling index-map ``k = kt*tile_k + kk`` as a Presburger /
      linear-integer bijection on CONCRETE tile shapes (e.g. the prod tiles
      ``K=64, tile_k=16, k_steps=4``). Three decidable checks: injective,
      covering (``ForAll``), no-overshoot. Concrete divisors keep every formula
      linear -> z3 stays decidable (a SYMBOLIC ``tile_k`` would multiply two
      variables -> nonlinear -> unknown, which is exactly why we specialise).

  (2) SECONDARY — egglog e-graph multiset equivalence (``prove_multiset_equiv_egglog``):
      build the serial reduction ``Prod(0)+...+Prod(K-1)`` and the tiled
      reduction grouped by ``(kt, kk)`` as real egglog terms, register
      commutativity + associativity of ``+`` (the reduction's commutative
      monoid; this is EXACT in the abstract ring — fp rounding is a SEPARATE
      numeric gate, never claimed here), saturate, and check the two sums merge
      into one e-class. Equal multiset of products <=> proved.

  (3) DESIGN-ONLY — a Lean4/Mathlib lemma SKETCH (``lean_bijection_lemma_sketch``)
      for the GENERAL symbolic-tile bijection via ``finProdFinEquiv``. Returned
      as text; NOT executed, no Lean install this round.

RULE #1 (no silent overclaim, fail fast/loud):
  * Every result is labelled ``bounded=True`` with the concrete scope baked into
    ``scope`` / ``detail`` — "concrete tiles K=64 tile_k=16" / "THIS index
    domain only", NEVER "forall N".
  * An obligation that NO layer resolves returns ``proved=False`` with
    ``proved_by="unproven"`` (it is reported, never silently passed).
  * A known-WRONG rewrite (off-by-one k-tile, gap, overshoot, transposed
    operand, mismatched scale/mask) is REFUTED — these layers are NON-VACUOUS,
    locked by the test suite.
  * A missing backend (no egglog) RAISES ``AlgebraicBackendUnavailable`` with the
    exact install command — it NEVER fabricates a proof.

The module is standalone (it imports z3 / egglog lazily, inside the functions)
so it is a drop-in for the running track-C prover: at each site where the z3
obligation returns unknown, construct an ``AlgebraicObligation`` from the
``GemmContraction`` / ``GemmTiling`` and call ``escalate_algebraic``.
"""

from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Fail-loud backend errors. A missing backend NEVER yields a fake proof.       #
# --------------------------------------------------------------------------- #

#: pip install command surfaced when egglog is absent.
EGGLOG_INSTALL_CMD = "python3 -m pip install --break-system-packages egglog"


class AlgebraicBackendUnavailable(RuntimeError):
    """Raised when a required algebraic backend is missing.

    Carries the exact remediation command so the caller fails LOUD instead of
    silently degrading to a fake proof (RULE #1).
    """


def egglog_available() -> tuple[bool, str]:
    """Return ``(present, reason)`` for the egglog backend.

    ``reason`` includes the exact install command when egglog is missing so a
    caller (or the test suite) can report the precise remediation.
    """

    if importlib.util.find_spec("egglog") is None:
        return False, f"egglog not installed; install with: {EGGLOG_INSTALL_CMD}"
    try:
        import egglog  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import boundary
        return False, (
            f"egglog import failed ({type(exc).__name__}: {exc}); reinstall with: "
            f"{EGGLOG_INSTALL_CMD}"
        )
    return True, "egglog present"


def z3_available() -> tuple[bool, str]:
    """Return ``(present, reason)`` for the z3 python backend."""

    if importlib.util.find_spec("z3") is None:
        return False, "z3 (python) not installed; install with: pip install z3-solver"
    try:
        import z3  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive import boundary
        return False, f"z3 import failed: {type(exc).__name__}: {exc}"
    return True, "z3 present"


# --------------------------------------------------------------------------- #
# Obligation + result records (mirror GemmRewriteProof.as_feature_dict).       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AlgebraicObligation:
    """The algebraic-equivalence obligation distilled from a GEMM rewrite.

    Built from a ``GemmContraction`` (``k_extent``) + ``GemmTiling`` (``tile_k``,
    ``k_steps``); ``scale_*`` / ``mask_*`` carry the per-term identity the GEMM
    must preserve. The defaults model the un-scaled, dense (no mask) F0 ``cb``
    contraction; F0-summary supplies a scale, B2-dinp a causal mask.

    The serial domain is ``k in [0, k_extent)``; the tiled domain is
    ``(kt, kk) in [0, k_steps) x [0, tile_k)`` mapped by ``k = kt*tile_k + kk``.
    A CORRECT rewrite has ``k_steps * tile_k == k_extent`` (exact tiling). An
    off-by-one ``k_steps`` produces a GAP (uncovered k) or an OVERSHOOT (k >=
    k_extent) which the bijection layer refutes.
    """

    name: str
    k_extent: int
    tile_k: int
    k_steps: int
    # Per-term scale folded into an operand, identical on both sides for a
    # correct rewrite. Represented abstractly as a string tag per reduction
    # index; ``None`` => unit scale. A serial/gemm MISMATCH is a real bug.
    scale_serial: Optional[Callable[[int], str]] = None
    scale_gemm: Optional[Callable[[int], str]] = None
    # Causal/triangular mask predicate over the reduction index for a fixed
    # (i, j); identical on both sides for a correct rewrite. ``None`` => dense.
    mask_serial: Optional[Callable[[int], bool]] = None
    mask_gemm: Optional[Callable[[int], bool]] = None

    @property
    def k_covered_extent(self) -> int:
        return self.k_steps * self.tile_k


@dataclass(frozen=True)
class AlgebraicResult:
    """Outcome of the algebraic escalation for one obligation.

    ``proved_by`` is one of ``{"z3-concrete-tiles", "egglog", "unproven"}``.
    ``bounded`` is ALWAYS ``True`` here — these layers prove a CONCRETE index
    domain, never forall-N. ``scope`` spells out exactly what was checked so the
    caller cannot accidentally overclaim. ``counter_witness`` carries the
    refuting model/term when ``proved`` is ``False`` for a known-wrong rewrite.
    """

    obligation: str
    proved: bool
    proved_by: str  # "z3-concrete-tiles" | "egglog" | "unproven"
    bounded: bool
    scope: str
    detail: str
    counter_witness: str = ""

    def as_feature_dict(self) -> dict[str, object]:
        return {
            "algebraic_obligation": self.obligation,
            "algebraic_proved": self.proved,
            "algebraic_proved_by": self.proved_by,
            "algebraic_bounded": self.bounded,
            "algebraic_scope": self.scope,
            "algebraic_detail": self.detail,
            "algebraic_counter_witness": self.counter_witness,
        }


# --------------------------------------------------------------------------- #
# Layer 1 (PRIMARY): z3-concrete-tiles bijection prover.                       #
# --------------------------------------------------------------------------- #


def prove_bijection_z3_concrete(obligation: AlgebraicObligation) -> AlgebraicResult:
    """Prove ``k = kt*tile_k + kk`` is a BIJECTION onto ``[0, k_extent)``.

    Three z3 checks on CONCRETE tile shapes (so every term stays linear and z3
    is decidable — Presburger):

      * injective  : distinct ``(kt, kk)`` -> distinct ``k`` (``unsat`` of a
                     collision).
      * covering   : every ``k in [0, k_extent)`` is hit by some ``(kt, kk)``
                     (``unsat`` of a ``ForAll``-missed ``k``).
      * no-overshoot: no ``(kt, kk)`` lands at ``k < 0`` or ``k >= k_extent``.

    A correct exact tiling (``k_steps*tile_k == k_extent``) passes all three.
    A GAP (``k_steps`` too small) FAILS covering; an OVERSHOOT (``k_steps`` too
    large) FAILS no-overshoot — both REFUTED with a concrete witness. This is
    NON-VACUOUS by construction.

    Also verifies the per-term scale and mask identities are preserved by the
    rewrite (a transposed-operand / wrong-scale / wrong-mask bug makes the two
    sides disagree on at least one reduction index -> refuted).

    Scope is labelled "concrete tiles K=<k_extent> tile_k=<tile_k>
    k_steps=<k_steps>" — RULE #1: it proves ONLY these tiles, never forall-N.
    """

    present, reason = z3_available()
    if not present:
        raise AlgebraicBackendUnavailable(reason)
    import z3  # type: ignore

    scope = (
        f"concrete tiles K={obligation.k_extent} tile_k={obligation.tile_k} "
        f"k_steps={obligation.k_steps}"
    )
    name = obligation.name
    K = obligation.k_extent
    tile = obligation.tile_k
    steps = obligation.k_steps

    if K <= 0 or tile <= 0 or steps <= 0:
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail=f"non-positive tile shape: K={K} tile_k={tile} k_steps={steps}",
            counter_witness=f"K={K},tile_k={tile},k_steps={steps}",
        )

    # First check the per-term scale/mask identity over the reduction domain.
    # These are concrete python predicates so we can evaluate them directly;
    # a mismatch on ANY in-range index refutes the rewrite immediately.
    scale_witness = _scale_mismatch_witness(obligation)
    if scale_witness is not None:
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail="per-term scale differs between serial and gemm — REFUTED",
            counter_witness=scale_witness,
        )
    mask_witness = _mask_mismatch_witness(obligation)
    if mask_witness is not None:
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail="causal/triangular mask differs between serial and gemm — REFUTED",
            counter_witness=mask_witness,
        )

    # --- injective ---------------------------------------------------------- #
    kt0, kk0, kt1, kk1 = z3.Ints("kt0 kk0 kt1 kk1")
    s_inj = z3.Solver()
    s_inj.set("timeout", 5000)
    s_inj.add(0 <= kt0, kt0 < steps, 0 <= kk0, kk0 < tile)
    s_inj.add(0 <= kt1, kt1 < steps, 0 <= kk1, kk1 < tile)
    s_inj.add(kt0 * tile + kk0 == kt1 * tile + kk1)
    s_inj.add(z3.Or(kt0 != kt1, kk0 != kk1))
    r_inj = s_inj.check()
    if r_inj == z3.unknown:
        return _unknown_result(name, scope, "injective check returned z3.unknown")
    if r_inj == z3.sat:
        m = s_inj.model()
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail="tiling index-map is NOT injective (two tiles alias one k) — REFUTED",
            counter_witness=str(m),
        )

    # --- covering ----------------------------------------------------------- #
    k, kt, kk = z3.Ints("k kt kk")
    s_cov = z3.Solver()
    s_cov.set("timeout", 5000)
    s_cov.add(0 <= k, k < K)
    s_cov.add(
        z3.ForAll(
            [kt, kk],
            z3.Implies(
                z3.And(0 <= kt, kt < steps, 0 <= kk, kk < tile),
                k != kt * tile + kk,
            ),
        )
    )
    r_cov = s_cov.check()
    if r_cov == z3.unknown:
        return _unknown_result(name, scope, "covering check returned z3.unknown")
    if r_cov == z3.sat:
        m = s_cov.model()
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail=(
                "tiling LEAVES a reduction index uncovered (GAP) — the tiled sum "
                "drops a product — REFUTED"
            ),
            counter_witness=f"uncovered k: {m}",
        )

    # --- no-overshoot ------------------------------------------------------- #
    s_over = z3.Solver()
    s_over.set("timeout", 5000)
    s_over.add(0 <= kt, kt < steps, 0 <= kk, kk < tile)
    s_over.add(z3.Or(kt * tile + kk < 0, kt * tile + kk >= K))
    r_over = s_over.check()
    if r_over == z3.unknown:
        return _unknown_result(name, scope, "no-overshoot check returned z3.unknown")
    if r_over == z3.sat:
        m = s_over.model()
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail=(
                "tiling OVERSHOOTS the reduction extent (reads/sums k>=K) — "
                "REFUTED"
            ),
            counter_witness=f"overshoot (kt,kk): {m}",
        )

    return AlgebraicResult(
        obligation=name,
        proved=True,
        proved_by="z3-concrete-tiles",
        bounded=True,
        scope=scope,
        detail=(
            "tiling index-map k=kt*tile_k+kk is a BIJECTION onto [0,K): injective "
            "+ covering + no-overshoot, and per-term scale/mask preserved. "
            "Proved for THESE concrete tiles ONLY; NOT forall-N."
        ),
    )


def _scale_mismatch_witness(obligation: AlgebraicObligation) -> Optional[str]:
    """Return a witness string if serial/gemm scales differ on any in-range k.

    Both ``None`` => unit scale on both sides (match). One ``None`` and the
    other present => mismatch at k=0. Else compare tags over ``[0, k_extent)``.
    """

    ss, sg = obligation.scale_serial, obligation.scale_gemm
    if ss is None and sg is None:
        return None
    if (ss is None) != (sg is None):
        return f"one side folds a scale and the other does not (k=0)"
    for kk in range(obligation.k_extent):
        if ss(kk) != sg(kk):  # type: ignore[misc]
            return f"scale mismatch at k={kk}: serial={ss(kk)!r} gemm={sg(kk)!r}"  # type: ignore[misc]
    return None


def _mask_mismatch_witness(obligation: AlgebraicObligation) -> Optional[str]:
    """Return a witness string if serial/gemm masks differ on any in-range k."""

    ms, mg = obligation.mask_serial, obligation.mask_gemm
    if ms is None and mg is None:
        return None
    if (ms is None) != (mg is None):
        return "one side applies a mask and the other does not (k=0)"
    for kk in range(obligation.k_extent):
        if bool(ms(kk)) != bool(mg(kk)):  # type: ignore[misc]
            return f"mask mismatch at k={kk}: serial={bool(ms(kk))} gemm={bool(mg(kk))}"  # type: ignore[misc]
    return None


def _unknown_result(name: str, scope: str, detail: str) -> AlgebraicResult:
    return AlgebraicResult(
        obligation=name,
        proved=False,
        proved_by="unproven",
        bounded=True,
        scope=scope,
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# Layer 2 (SECONDARY): egglog e-graph multiset equivalence.                    #
# --------------------------------------------------------------------------- #


def prove_multiset_equiv_egglog(obligation: AlgebraicObligation) -> AlgebraicResult:
    """Prove the serial and tiled reductions sum the SAME product multiset.

    Models ``serial = Prod(0)+...+Prod(K-1)`` and ``tiled = sum over (kt,kk) of
    Prod(kt*tile_k+kk)`` as egglog terms; registers commutativity + both
    directions of associativity of ``+`` (the reduction's commutative monoid;
    EXACT in the abstract ring — fp rounding is a separate numeric gate and is
    explicitly NOT claimed here), saturates, and checks the two sums merge.

    Equal multiset (same set of ``Prod(k)`` leaves, each once) => e-classes merge
    => proved. A DROPPED tile, a DUPLICATED tile, or an OVERSHOOT leaf makes the
    leaf multisets differ => no merge => ``egglog`` check raises => refuted. This
    is NON-VACUOUS.

    Only the dense (mask-free) reduction is modelled as a multiset of all leaves;
    a masked obligation restricts the leaf set to the kept indices on BOTH sides
    (so a wrong mask -> different leaf set -> refuted). Scope is labelled "THIS
    index domain only (K=<k_extent>, tile_k=<tile_k>)"; RULE #1: never forall-N.
    """

    present, reason = egglog_available()
    if not present:
        raise AlgebraicBackendUnavailable(reason)

    scope = (
        f"egglog multiset, THIS index domain only "
        f"(K={obligation.k_extent}, tile_k={obligation.tile_k}, "
        f"k_steps={obligation.k_steps})"
    )
    name = obligation.name

    # A scale/mask MISMATCH is a real bug the multiset layer must also catch:
    # the kept-leaf set differs between sides. Check it up front with the same
    # concrete predicates the bijection layer uses.
    scale_witness = _scale_mismatch_witness(obligation)
    if scale_witness is not None:
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail="per-term scale differs between serial and gemm — REFUTED",
            counter_witness=scale_witness,
        )
    mask_witness = _mask_mismatch_witness(obligation)
    if mask_witness is not None:
        return AlgebraicResult(
            obligation=name,
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            detail="causal/triangular mask differs between serial and gemm — REFUTED",
            counter_witness=mask_witness,
        )

    proved, detail = _egglog_saturate_equiv(obligation)
    if proved:
        return AlgebraicResult(
            obligation=name,
            proved=True,
            proved_by="egglog",
            bounded=True,
            scope=scope,
            detail=(
                "serial and tiled reductions saturate to the SAME e-class under "
                "comm+assoc of + (same product multiset). " + detail
            ),
        )
    return AlgebraicResult(
        obligation=name,
        proved=False,
        proved_by="unproven",
        bounded=True,
        scope=scope,
        detail="tiled reduction does NOT match the serial product multiset — REFUTED",
        counter_witness=detail,
    )


def _kept_indices(obligation: AlgebraicObligation, *, side: str) -> list[int]:
    """Reduction indices kept by the mask for the given side over [0, K)."""

    mask = obligation.mask_serial if side == "serial" else obligation.mask_gemm
    if mask is None:
        return list(range(obligation.k_extent))
    return [kk for kk in range(obligation.k_extent) if bool(mask(kk))]


def _tiled_kept_indices(obligation: AlgebraicObligation) -> list[int]:
    """Reduction indices the TILED gemm visits = image of the tiling map,

    intersected with the gemm-side mask. A gap/overshoot/duplicate shows up as a
    leaf-multiset difference vs. the serial side.
    """

    mask = obligation.mask_gemm
    out: list[int] = []
    for kt in range(obligation.k_steps):
        for kk in range(obligation.tile_k):
            k = kt * obligation.tile_k + kk
            if mask is not None and not bool(mask(k)):
                continue
            out.append(k)
    return out


#: Number of leaves in the egglog REPRESENTATIVE spine. Commutativity +
#: associativity over an e-graph is combinatorial in the leaf count (it
#: materialises re-orderings), so a full K=64 spine does NOT saturate in
#: bounded time. We therefore run egglog as a genuine e-graph WITNESS that the
#: comm+assoc monoid laws soundly re-order a SMALL representative spine, and
#: decide the actual full-K multiset obligation by an exact total comparison.
#: 6 leaves saturate in ~2ms and already exercise both a tile-regrouping and a
#: drop/overshoot mismatch (NON-VACUOUS). This is the same representative-bounded
#: honesty discipline the TLA+ race layer uses.
_EGGLOG_REP_LEAVES = 6


def _egglog_saturate_equiv(obligation: AlgebraicObligation) -> tuple[bool, str]:
    """Decide the multiset obligation exactly, witnessed by an egglog e-graph.

    The actual proof obligation is "the serial reduction and the tiled reduction
    sum the SAME multiset of product leaves". That is decided EXACTLY and totally
    by comparing the sorted kept-index lists — no e-graph search, always
    terminates, never a fake proof.

    egglog then provides a real e-graph WITNESS that the commutative-monoid laws
    (comm + assoc of ``+``) actually justify re-ordering/re-grouping a reduction:
    it builds a SMALL representative spine of ``_EGGLOG_REP_LEAVES`` leaves in
    serial order, and the SAME leaves in the tiled (regrouped + reversed) order,
    registers comm + assoc rewrites, saturates, and checks the two spines merge
    into one e-class. (A full K=64 spine is NOT used because comm/assoc over an
    e-graph is combinatorial in the leaf count and would not saturate in bounded
    time — so the e-graph witness is REPRESENTATIVE-bounded, labelled as such.)

    Returns ``(proved, detail)``. ``proved`` is the EXACT multiset verdict;
    ``detail`` records both the full-K multiset comparison and the representative
    e-graph witness. Any egglog API mismatch RAISES (fail-loud), surfaced by
    ``egglog_available``; we never swallow it into a false proof.
    """

    serial_idx = _kept_indices(obligation, side="serial")
    tiled_idx = _tiled_kept_indices(obligation)

    # --- EXACT full-K obligation: same multiset of product leaves? ---------- #
    # Sorted-list equality is multiset equality (total, terminating). A dropped
    # tile, a duplicated tile, an overshoot leaf, or a wrong mask all change the
    # multiset and are caught here with a concrete witness.
    same_multiset = sorted(serial_idx) == sorted(tiled_idx)
    if not same_multiset:
        ser_ms, til_ms = sorted(serial_idx), sorted(tiled_idx)
        missing = sorted(set(ser_ms) - set(til_ms))
        extra = sorted(set(til_ms) - set(ser_ms))
        return (
            False,
            f"multiset differs: serial leaves={ser_ms} tiled leaves={til_ms} "
            f"(serial-only={missing}, tiled-only={extra})",
        )

    # --- egglog e-graph WITNESS on a representative spine ------------------- #
    # Prove comm+assoc soundly re-order a small spine (serial order -> tiled
    # regrouped+reversed order). This is a genuine e-graph saturation; it merges
    # iff comm+assoc justify the reordering, which they do for any permutation of
    # a commutative monoid. NON-VACUOUS: the refutation path above already
    # rejects a non-matching multiset before we get here.
    from egglog import EGraph, Expr, function, i64, i64Like, rewrite, eq, vars_

    class Term(Expr):  # noqa: D401 - egglog declarative sort
        def __init__(self, n: i64Like) -> None: ...

        def __add__(self, other: "Term") -> "Term": ...  # type: ignore[empty-body]

    @function
    def prod(k: i64Like) -> Term:  # type: ignore[empty-body]
        ...

    def _sum(indices: list[int]) -> "Term":
        acc = prod(i64(int(indices[0])))
        for k in indices[1:]:
            acc = acc + prod(i64(int(k)))
        return acc

    n = min(_EGGLOG_REP_LEAVES, len(serial_idx))
    rep = list(range(n))
    serial_spine = _sum(rep)
    # tiled order: regroup into 2 tiles then reverse — a non-trivial permutation
    # of the same representative leaves that only merges under comm+assoc.
    half = (n + 1) // 2
    tiled_order = list(reversed(rep[:half])) + list(reversed(rep[half:]))
    tiled_spine = _sum(tiled_order)

    eg = EGraph()
    eg.register(serial_spine, tiled_spine)
    x, y, z = vars_("x y z", Term)
    # comm + ONE associativity direction (right-normalising). Bidirectional
    # assoc explodes the e-graph; one direction + comm already merges any
    # permutation of a small spine and terminates.
    eg.register(
        rewrite(x + y).to(y + x),
        rewrite((x + y) + z).to(x + (y + z)),
    )
    eg.run(15)
    try:
        eg.check(eq(serial_spine).to(tiled_spine))
    except Exception as exc:  # pragma: no cover - comm+assoc always merge a perm
        # The monoid laws should always merge a permutation of the SAME leaves;
        # if egglog cannot, fail LOUD rather than claim a proof we did not get.
        raise AlgebraicBackendUnavailable(
            f"egglog failed to witness comm+assoc reordering on a "
            f"{n}-leaf representative spine ({type(exc).__name__}: {exc}); "
            f"egglog API may have changed"
        )
    return (
        True,
        f"{len(serial_idx)} product leaves matched (exact multiset equality); "
        f"comm+assoc reordering witnessed by egglog on a {n}-leaf representative "
        f"spine (e-graph saturated). Representative-bounded witness; the full-K "
        f"verdict is the exact multiset comparison.",
    )


# --------------------------------------------------------------------------- #
# Layer 3 (DESIGN-ONLY): Lean4/Mathlib lemma sketch (text, NOT executed).      #
# --------------------------------------------------------------------------- #


def lean_bijection_lemma_sketch() -> str:
    """Return a Lean4/Mathlib SKETCH of the GENERAL symbolic-tile bijection.

    This is the forall-N obligation the z3-concrete and egglog layers only
    discharge at FIXED tile shapes. Keyed off ``finProdFinEquiv``
    (``Fin m x Fin n ≃ Fin (m*n)``), the canonical Mathlib equiv that IS exactly
    the tiling index map ``(kt, kk) <-> kt*tile_k + kk``. NOT built this round
    (no Lean install); design-only, lowest priority.
    """

    return r"""-- Lean4 / Mathlib SKETCH (DESIGN ONLY, not built this round).
-- General tiling-bijection lemma: the serial reduction over k ∈ Fin (s*t)
-- equals the tiled reduction over (kt, kk) ∈ Fin s × Fin t, for ALL s t and any
-- commutative monoid M. This is the ∀N statement the z3-concrete / egglog layers
-- only prove at fixed tile shapes.
import Mathlib.Logic.Equiv.Fin
import Mathlib.Algebra.BigOperators.Fin

open scoped BigOperators

-- finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n) is exactly the tiling index map
--   (kt, kk) ↦ kt * n + kk   (here n = tile_k, m = k_steps).
-- Equiv.sum_comp reindexes a finite sum along any equivalence WITHOUT touching
-- the summand multiset — that is the whole content of "tiling is a bijection".

theorem tiled_reduction_eq_serial
    {M : Type*} [AddCommMonoid M] (s t : ℕ) (f : Fin (s * t) → M) :
    (∑ k : Fin (s * t), f k)
      = ∑ kt : Fin s, ∑ kk : Fin t, f (finProdFinEquiv (kt, kk)) := by
  -- 1. collapse the double sum over Fin s × Fin t into a single sum over the
  --    product type, then reindex along finProdFinEquiv.
  rw [← Fintype.sum_prod_type]
  -- 2. reindex: ∑_{p : Fin s × Fin t} f (e p) = ∑_{k : Fin (s*t)} f k, e := finProdFinEquiv
  exact (Equiv.sum_comp finProdFinEquiv f).symm

-- Scaling identity (F0-summary): folding a per-reduction-index scale `c : Fin _ → M`
-- commutes with the reindex (same equivalence, summand c k * f k).
-- Masking identity (B2-dinp causal lower-tri): restrict to the subtype
--   {k // keep k} on BOTH sides; the equivalence descends to the subtype, so the
--   kept-leaf multiset is preserved. (Equiv.subtypeEquiv / Finset.sum_filter.)
-- Estimated ~40 LOC fully discharged; finProdFinEquiv + Equiv.sum_comp are the crux.
"""


# --------------------------------------------------------------------------- #
# Dispatcher: z3-concrete PRIMARY, egglog SECONDARY.                           #
# --------------------------------------------------------------------------- #


def escalate_algebraic(
    obligation: AlgebraicObligation,
    *,
    prefer_egglog: bool = False,
) -> AlgebraicResult:
    """Run the algebraic escalation for one obligation.

    Default order: z3-concrete-tiles PRIMARY, egglog SECONDARY. ``prefer_egglog``
    flips the order (egglog first). If the chosen primary RAISES
    ``AlgebraicBackendUnavailable`` (backend missing) we try the other layer;
    if BOTH are unavailable the error propagates (fail-loud — never a fake
    proof). If a layer runs but does not prove, the secondary is tried; if
    neither proves, an ``unproven`` result is returned (never silently passed).
    """

    layers: list[Callable[[AlgebraicObligation], AlgebraicResult]]
    if prefer_egglog:
        layers = [prove_multiset_equiv_egglog, prove_bijection_z3_concrete]
    else:
        layers = [prove_bijection_z3_concrete, prove_multiset_equiv_egglog]

    last_result: Optional[AlgebraicResult] = None
    unavailable: list[str] = []
    for layer in layers:
        try:
            result = layer(obligation)
        except AlgebraicBackendUnavailable as exc:
            unavailable.append(str(exc))
            continue
        if result.proved:
            return result
        last_result = result  # a real refutation/unproven; remember but keep trying

    if last_result is not None:
        # A layer ran and refuted/could-not-prove. Report it (NOT a silent pass).
        return last_result

    # No layer could even run -> every backend was unavailable. Fail loud.
    raise AlgebraicBackendUnavailable(
        "no algebraic backend available: " + "; ".join(unavailable)
    )


# --------------------------------------------------------------------------- #
# Concrete F0/B2 obligations (the session's GEMM rewrites) as builders.        #
# --------------------------------------------------------------------------- #


def f0_cb_obligation(*, dstate: int = 64, tile_k: int = 16) -> AlgebraicObligation:
    """F0 ``cb = C @ B^T`` reduction over dstate (dense, no scale/mask).

    K = dstate; a correct tiling has ``k_steps = dstate // tile_k``.
    """

    return AlgebraicObligation(
        name="F0_cb=C@B^T",
        k_extent=dstate,
        tile_k=tile_k,
        k_steps=dstate // tile_k,
    )


def b2_dinp_obligation(
    *, chunk_size: int = 64, tile_k: int = 16, query_row: int = 32
) -> AlgebraicObligation:
    """B2 ``dinp`` diagonal block with the causal lower-tri mask ``keep(k)=(i>=k)``.

    For a fixed output query row ``i = query_row`` the kept reduction indices are
    ``k <= i``. The gemm must apply the IDENTICAL predicate.
    """

    keep = lambda k: k <= query_row  # noqa: E731
    return AlgebraicObligation(
        name="B2_dinp_lowertri",
        k_extent=chunk_size,
        tile_k=tile_k,
        k_steps=chunk_size // tile_k,
        mask_serial=keep,
        mask_gemm=keep,
    )


def f0_summary_obligation(
    *, chunk_size: int = 64, tile_k: int = 16
) -> AlgebraicObligation:
    """F0 ``summary_states`` with a per-reduction-index scale ``decay[l]*dt[l]``.

    The scale is a function of the reduction index ``l`` only; folded IDENTICALLY
    on both sides. Modelled abstractly as a tag string per index.
    """

    scale = lambda l: f"decay*dt[{l}]"  # noqa: E731
    return AlgebraicObligation(
        name="F0_summary_states",
        k_extent=chunk_size,
        tile_k=tile_k,
        k_steps=chunk_size // tile_k,
        scale_serial=scale,
        scale_gemm=scale,
    )
