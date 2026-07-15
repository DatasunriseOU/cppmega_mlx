"""Escalation machinery for the GEMM-rewrite z3 prover (track C, standalone).

The primary prover lives in ``_gemm_rewrite_proof.prove_gemm_rewrite``: it
discharges each structural obligation (operand maps, tiling bijection, mask,
scale, single-writer) with z3, returning unsat=proved / sat=refuted /
unknown=not-proved (fail-closed, RULE #1). z3 can return ``unknown`` on:

  * ALGEBRAIC EQUIVALENCE with SYMBOLIC tile sizes (the tiling bijection /
    multiset-of-products equivalence becomes nonlinear integer arithmetic when
    tile_k is a variable, e.g. ``k == kt*TILE + kk`` with TILE symbolic), or
  * RACE-FREEDOM phrased as a concurrency interleaving rather than a static
    ownership predicate.

This module is the ESCALATION dispatcher the user specified. It does NOT
rewrite the existing prover; it wraps the SAME (proved, refuted, unknown)
pattern and, on UNKNOWN, classifies the obligation and dispatches:

  RACE-FREEDOM  -> generate a TLA+ spec of the tiled-GEMM writer state machine
                   + a single-writer safety invariant, model-check with TLC
                   over BOUNDED representative sizes (exhaustive interleavings).
                   HONESTY: TLC is BOUNDED model checking -> "verified bounded
                   @ tile=X dim=Y", NEVER "proven forall N".

  ALGEBRAIC     -> (a) PRIMARY z3-concrete-tiles: reformulate the bijection on
                       CONCRETE tile shapes (L=P=N=64, tile=16) with LINEAR /
                       Presburger constraints (concrete divisors are decidable)
                       -> proves ONLY those tiles.
                   (b) SECONDARY egglog: build the serial reduction and the
                       tiled form as e-graph terms, saturate with comm / assoc /
                       distrib rewrites, check the product MULTISET is equal.
                   (c) DESIGN-ONLY Lean sketch (string): the one-time general
                       symbolic-tile bijection lemma. NOT executed here.

Provenance is carried on every result via ``proved_by``:
    "z3" | "tlc-bounded" | "z3-concrete-tiles" | "egglog" | "unproven"

An obligation NO layer resolves is reported "unproven" — never silently passed.
Every layer is NON-VACUOUS: a known-wrong rewrite (a real race, a transposed
operand, an off-by-one tile) MUST be refuted by its layer; the sibling test
locks that in.

This is built standalone against the ``_async_barrier_plan`` /
``_gemm_rewrite_proof`` interfaces; wiring it into the running prover is a
drop-in follow-up (call :func:`escalate_obligation` wherever that prover gets
z3 ``unknown``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Literal, Optional

ProvedBy = Literal[
    "z3",
    "tlc-bounded",
    "z3-concrete-tiles",
    "egglog",
    "unproven",
]

ObligationKind = Literal["race", "algebraic"]

# Where the TLA+ jar is expected. Overridable for CI / other machines.
_TLA2TOOLS_ENV = "CPPMEGA_TLA2TOOLS_JAR"
_DEFAULT_TLA2TOOLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "tools",
    "tla2tools.jar",
)

_TLC_TIMEOUT_S = 120


# --------------------------------------------------------------------------- #
# Result type — mirrors MetalReductionSyncPlan / GemmRewriteProof so it flows
# into the same receipt plumbing, extended with escalation provenance.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EscalationResult:
    """Outcome of escalating ONE proof obligation past z3 ``unknown``.

    ``proved`` is True iff some layer discharged the obligation. ``proved_by``
    records WHICH layer (or "unproven"). ``bounded`` flags a result that holds
    only at the checked sizes/tiles (TLC bounded, z3-concrete-tiles) so callers
    never over-claim forall-N. ``scope`` is a human label of exactly what was
    checked (e.g. "tile=16 dim=64").
    """

    name: str
    kind: ObligationKind
    proved: bool
    proved_by: ProvedBy
    bounded: bool
    scope: str
    reason: str
    counter_witness: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_feature_dict(self) -> dict[str, object]:
        return {
            "escalation_name": self.name,
            "escalation_kind": self.kind,
            "escalation_proved": self.proved,
            "escalation_proved_by": self.proved_by,
            "escalation_bounded": self.bounded,
            "escalation_scope": self.scope,
            "escalation_reason": self.reason,
            "escalation_counter_witness": self.counter_witness or "",
        }


class EscalationUnavailable(RuntimeError):
    """Raised when a required escalation backend (TLC, egglog) is unavailable.

    RULE #1 fail-loud: a missing backend is surfaced with the exact fetch
    command, not silently treated as a passing proof.
    """


# --------------------------------------------------------------------------- #
# Layer A — RACE-FREEDOM via TLA+ / TLC (bounded model checking).
#
# We generate a TLA+ spec of the tiled-GEMM writer: a set of writer "blocks",
# each owning a rectangle of output cells (i in [bm*m_step, +tile_m),
# j in [bn*n_step, +tile_n)); the state machine lets blocks Write their owned
# cells in any interleaving, recording a writer id per cell. The safety
# invariant NoDoubleWrite says no cell is ever claimed by two distinct blocks.
# TLC explores ALL interleavings exhaustively over the bounded grid.
# --------------------------------------------------------------------------- #
def _tla2tools_jar() -> Optional[str]:
    cand = os.environ.get(_TLA2TOOLS_ENV) or _DEFAULT_TLA2TOOLS
    return cand if os.path.isfile(cand) else None


def tlc_available() -> tuple[bool, str]:
    """Return (available, reason). Fail-loud-friendly diagnostics."""
    if shutil.which("java") is None:
        return False, "java not on PATH (TLC needs a JRE >= 11)"
    jar = _tla2tools_jar()
    if jar is None:
        return False, (
            "tla2tools.jar not found; fetch with: curl -sSL -o "
            f"{_DEFAULT_TLA2TOOLS} "
            "https://github.com/tlaplus/tlaplus/releases/latest/download/"
            "tla2tools.jar  (or set $%s)" % _TLA2TOOLS_ENV
        )
    return True, f"TLC ready: java + {jar}"


def render_writer_tla(
    *,
    module: str,
    m_extent: int,
    n_extent: int,
    tile_m: int,
    tile_n: int,
    m_step: int,
    n_step: int,
    m_blocks: int,
    n_blocks: int,
) -> str:
    """Render a TLA+ module of the tiled-GEMM writer + NoDoubleWrite invariant.

    The model is deliberately concrete (all extents are constants) so TLC
    exhaustively checks every interleaving of block writes over the grid. A
    correct (disjoint, contiguous) tiling -> invariant holds at all states; an
    overlapping tiling -> TLC finds a state where two blocks wrote one cell and
    reports the trace (the race witness).
    """
    return f"""---------------------------- MODULE {module} ----------------------------
EXTENDS Integers, FiniteSets, TLC

\\* Tiled-GEMM writer race-freedom, BOUNDED model.
\\* Output grid is [0,M) x [0,N). Block (bm,bn) owns the rectangle
\\* [bm*MStep, bm*MStep+TileM) x [bn*NStep, bn*NStep+TileN).
\\* A correct rewrite has disjoint, covering ownership; TLC checks NoDoubleWrite
\\* over EVERY interleaving of block writes. The constants are bound in the .cfg.

CONSTANTS M, N, TileM, TileN, MStep, NStep, MBlocks, NBlocks

Cells == (0 .. (M - 1)) \\X (0 .. (N - 1))
Blocks == (0 .. (MBlocks - 1)) \\X (0 .. (NBlocks - 1))

Owns(blk, c) ==
    /\\ c[1] >= blk[1] * MStep
    /\\ c[1] <  blk[1] * MStep + TileM
    /\\ c[2] >= blk[2] * NStep
    /\\ c[2] <  blk[2] * NStep + TileN

\\* wrote[c] accumulates the SET of distinct block-ids that have written cell c.
\\* Each (block,cell) it owns fires at most once (write-once guard), so the state
\\* space is finite (subsets of the owned (block,cell) pairs). TLC explores every
\\* interleaving. A race = some cell ends up with two distinct writers in wrote.
VARIABLES wrote

BlkId(blk) == blk[1] * NBlocks + blk[2]

Init == wrote = [ c \\in Cells |-> {{}} ]

Write(blk, c) ==
    /\\ Owns(blk, c)
    /\\ BlkId(blk) \\notin wrote[c]
    /\\ wrote' = [ wrote EXCEPT ![c] = wrote[c] \\cup {{BlkId(blk)}} ]

Next == \\E blk \\in Blocks, c \\in Cells : Write(blk, c)

\\* Safety (race-freedom): no output cell is ever written by two distinct blocks.
\\* A correct disjoint tiling makes every reachable state satisfy this; an
\\* overlapping tiling lets two blocks both write a shared cell -> TLC reports
\\* the interleaving that produced the double write.
NoDoubleWrite == \\A c \\in Cells : Cardinality(wrote[c]) <= 1

Spec == Init /\\ [][Next]_wrote
=============================================================================
"""


def _render_tlc_cfg(
    *,
    m_extent: int,
    n_extent: int,
    tile_m: int,
    tile_n: int,
    m_step: int,
    n_step: int,
    m_blocks: int,
    n_blocks: int,
) -> str:
    return (
        "SPECIFICATION Spec\n"
        "INVARIANT NoDoubleWrite\n"
        "CONSTANTS\n"
        f"    M = {m_extent}\n"
        f"    N = {n_extent}\n"
        f"    TileM = {tile_m}\n"
        f"    TileN = {tile_n}\n"
        f"    MStep = {m_step}\n"
        f"    NStep = {n_step}\n"
        f"    MBlocks = {m_blocks}\n"
        f"    NBlocks = {n_blocks}\n"
    )


def prove_race_freedom_tlc(
    *,
    name: str,
    m_extent: int,
    n_extent: int,
    tile_m: int,
    tile_n: int,
    m_step: int,
    n_step: int,
    m_blocks: int,
    n_blocks: int,
) -> EscalationResult:
    """Model-check tiled-GEMM single-writer race-freedom with TLC (BOUNDED).

    Returns proved=True / proved_by="tlc-bounded" / bounded=True when the
    NoDoubleWrite invariant holds across ALL interleavings at the given
    concrete sizes; proved=False with a counter_witness trace when TLC finds an
    overlap (a real race). Honesty: this is bounded @ the checked grid, NOT
    forall N — ``scope`` and ``bounded`` say so.
    """
    avail, why = tlc_available()
    scope = (
        f"dim={m_extent}x{n_extent} tile={tile_m}x{tile_n} "
        f"step={m_step}x{n_step} blocks={m_blocks}x{n_blocks}"
    )
    if not avail:
        raise EscalationUnavailable(
            f"TLC race-freedom escalation for '{name}' unavailable: {why}"
        )

    module = "GemmWriter"
    spec = render_writer_tla(
        module=module,
        m_extent=m_extent,
        n_extent=n_extent,
        tile_m=tile_m,
        tile_n=tile_n,
        m_step=m_step,
        n_step=n_step,
        m_blocks=m_blocks,
        n_blocks=n_blocks,
    )
    cfg = _render_tlc_cfg(
        m_extent=m_extent,
        n_extent=n_extent,
        tile_m=tile_m,
        tile_n=tile_n,
        m_step=m_step,
        n_step=n_step,
        m_blocks=m_blocks,
        n_blocks=n_blocks,
    )
    jar = _tla2tools_jar()

    with tempfile.TemporaryDirectory(prefix="gemm_race_tlc_") as tmp:
        tla_path = os.path.join(tmp, f"{module}.tla")
        cfg_path = os.path.join(tmp, f"{module}.cfg")
        with open(tla_path, "w") as fh:
            fh.write(spec)
        with open(cfg_path, "w") as fh:
            fh.write(cfg)
        # ``-deadlock`` turns OFF deadlock detection: the terminal state where
        # every owned cell has been written is the INTENDED completion of the
        # writer (the write-once guard disables all further actions), not a bug.
        # We only care about the NoDoubleWrite SAFETY invariant here.
        proc = subprocess.run(
            ["java", "-cp", jar, "tlc2.TLC", "-deadlock",
             "-config", cfg_path, tla_path],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=_TLC_TIMEOUT_S,
        )
        out = proc.stdout + "\n" + proc.stderr

    invariant_violated = "Invariant NoDoubleWrite is violated" in out
    # A clean run: exploration finished with state stats and NO "Error:" line.
    explored = "states generated" in out
    any_error = "Error:" in out
    ok = explored and not any_error
    if ok and not invariant_violated:
        return EscalationResult(
            name=name,
            kind="race",
            proved=True,
            proved_by="tlc-bounded",
            bounded=True,
            scope=scope,
            reason=(
                "TLC exhaustively checked all interleavings: NoDoubleWrite holds "
                f"(verified BOUNDED @ {scope}; NOT proven forall N)"
            ),
            notes=tuple(_tlc_stats(out)),
        )
    if invariant_violated:
        witness = _extract_tlc_trace(out)
        return EscalationResult(
            name=name,
            kind="race",
            proved=False,
            proved_by="unproven",
            bounded=True,
            scope=scope,
            reason=(
                "TLC found a NoDoubleWrite violation: two distinct blocks own "
                f"the same output cell (a real race) @ {scope}"
            ),
            counter_witness=witness,
            notes=tuple(_tlc_stats(out)),
        )
    # Neither clean pass nor a clear invariant violation: do NOT claim a proof.
    return EscalationResult(
        name=name,
        kind="race",
        proved=False,
        proved_by="unproven",
        bounded=True,
        scope=scope,
        reason=f"TLC did not return a clean result; raw tail: {out[-600:]}",
    )


def _tlc_stats(out: str) -> list[str]:
    lines = []
    for line in out.splitlines():
        if "states generated" in line and "distinct states found" in line:
            lines.append(line.strip())
    return lines


def _extract_tlc_trace(out: str) -> str:
    # Capture from the error line through the first few state dumps.
    idx = out.find("Invariant NoDoubleWrite is violated")
    if idx < 0:
        return out[-800:]
    return out[idx: idx + 900]


# --------------------------------------------------------------------------- #
# Layer B(a) — ALGEBRAIC bijection on CONCRETE tiles via z3 (PRIMARY).
#
# The symbolic-tile bijection k == kt*TILE + kk (TILE a variable) is nonlinear
# and z3 returns unknown. With a CONCRETE divisor the encoding is linear /
# Presburger and z3 decides it instantly. We prove BOTH directions:
#   * injective: distinct (kt,kk) tile coords never map to the same k, and
#   * surjective/coverage: every k in [0,K) has exactly one in-range producer,
# i.e. the tile map is a BIJECTION onto the reduction domain [0,K). This is the
# F0/B2 tiling bijection obligation, proven for the SPECIFIC tiles only.
# --------------------------------------------------------------------------- #
def prove_bijection_z3_concrete(
    *,
    name: str,
    k_extent: int,
    tile_k: int,
    k_steps: int,
) -> EscalationResult:
    """Prove the k-tiling is a bijection [0,k_steps)x[0,tile_k) <-> [0,K).

    Concrete (numeric) tile_k/k_steps keep this in z3's decidable linear
    fragment. proved_by="z3-concrete-tiles", bounded=True (these tiles only).
    A wrong tiling (gap or overlap, e.g. k_steps*tile_k != k_extent) yields a
    sat counter-witness and proved=False.
    """
    scope = f"K={k_extent} tile_k={tile_k} k_steps={k_steps}"
    try:
        import z3  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise EscalationUnavailable(
            f"z3-concrete bijection for '{name}' unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    notes: list[str] = []

    # --- injective: two distinct tile coords -> same in-range k. (negation) ---
    s_inj = z3.Solver()
    s_inj.set("timeout", 1000)
    kt0, kk0, kt1, kk1 = (z3.Int(n) for n in ("kt0", "kk0", "kt1", "kk1"))
    for kt in (kt0, kt1):
        s_inj.add(0 <= kt, kt < k_steps)
    for kk in (kk0, kk1):
        s_inj.add(0 <= kk, kk < tile_k)
    k0 = kt0 * tile_k + kk0
    k1 = kt1 * tile_k + kk1
    s_inj.add(k0 < k_extent, k1 < k_extent)
    s_inj.add(k0 == k1)
    s_inj.add(z3.Or(kt0 != kt1, kk0 != kk1))
    r_inj = s_inj.check()
    injective = r_inj == z3.unsat
    if r_inj == z3.sat:
        notes.append(f"injective FAILED, witness {s_inj.model()}")
    elif r_inj == z3.unknown:
        notes.append("injective: z3 unknown (concrete divisor should decide)")

    # --- coverage: a k with NO in-range producer. (negation, quantified) ---
    s_cov = z3.Solver()
    s_cov.set("timeout", 1000)
    k = z3.Int("k")
    kt, kk = z3.Int("kt"), z3.Int("kk")
    s_cov.add(0 <= k, k < k_extent)
    s_cov.add(
        z3.ForAll(
            [kt, kk],
            z3.Implies(
                z3.And(0 <= kt, kt < k_steps, 0 <= kk, kk < tile_k),
                k != kt * tile_k + kk,
            ),
        )
    )
    r_cov = s_cov.check()
    covered = r_cov == z3.unsat
    if r_cov == z3.sat:
        notes.append(f"coverage FAILED, uncovered k = {s_cov.model()}")
    elif r_cov == z3.unknown:
        notes.append("coverage: z3 unknown")

    # --- no over-coverage: a producer landing OUTSIDE [0,K) (gap-the-other-way) ---
    s_over = z3.Solver()
    s_over.set("timeout", 1000)
    kt2, kk2 = z3.Int("kt2"), z3.Int("kk2")
    s_over.add(0 <= kt2, kt2 < k_steps, 0 <= kk2, kk2 < tile_k)
    s_over.add(kt2 * tile_k + kk2 >= k_extent)
    r_over = s_over.check()
    no_overshoot = r_over == z3.unsat
    if r_over == z3.sat:
        notes.append(f"overshoot: producer outside [0,K): {s_over.model()}")

    proved = injective and covered and no_overshoot
    if proved:
        return EscalationResult(
            name=name,
            kind="algebraic",
            proved=True,
            proved_by="z3-concrete-tiles",
            bounded=True,
            scope=scope,
            reason=(
                "z3 proved the k-tiling is a BIJECTION onto [0,K) (injective + "
                f"covering, no overshoot) for CONCRETE tiles @ {scope}; proves "
                "ONLY these tiles, not forall tile size"
            ),
            notes=tuple(notes),
        )
    cw = "; ".join(notes) if notes else None
    return EscalationResult(
        name=name,
        kind="algebraic",
        proved=False,
        proved_by="unproven",
        bounded=True,
        scope=scope,
        reason=(
            "z3 did NOT prove the concrete-tile bijection (gap/overlap/overshoot "
            f"or unknown) @ {scope}"
        ),
        counter_witness=cw,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Layer B(b) — ALGEBRAIC multiset-of-products equivalence via egglog (SECONDARY).
#
# Models the serial reduction's product index multiset and the tiled form's, as
# e-graph terms, saturates with the tiling rewrite (k = kt*tile + kk), and
# checks the two enumerated product multisets are EQUAL as sets of (i,j,k)
# triples. Because the reduction is a SUM over a commutative+associative monoid,
# equal product multisets => equal sums (the reassociation is exact in the
# abstract ring; fp rounding is the separate numeric gate). We enumerate the
# concrete index domain and assert set-equality inside an egglog program.
# --------------------------------------------------------------------------- #
def egglog_available() -> tuple[bool, str]:
    try:
        import egglog  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        return False, (
            f"egglog not importable ({type(exc).__name__}); install with: "
            "python3 -m pip install --break-system-packages egglog"
        )
    return True, "egglog ready"


def prove_multiset_equiv_egglog(
    *,
    name: str,
    k_extent: int,
    tile_k: int,
    k_steps: int,
) -> EscalationResult:
    """Prove serial vs tiled product MULTISETS are equal via an e-graph.

    We use egglog's e-graph to model the tiling rewrite as an equality
    saturation: the tiled producer ``prod(kt*tile_k + kk)`` is rewritten to the
    serial producer ``prod(k)`` and we check that the e-class of the tiled
    reduction term equals the e-class of the serial reduction term over the
    concrete index set. proved_by="egglog", bounded=True (this domain only).

    Implementation note: rather than rely on a specific egglog Python DSL
    version, we drive equality saturation on a small integer term algebra:
    register, for every concrete (kt,kk), the fact ``Tile(kt,kk) = Serial(k)``
    where ``k = kt*tile_k+kk`` (when in range), then assert the multiset of
    Serial(k) reached from tiles equals {Serial(0..K-1)}. If egglog's API shape
    differs, we fall back to a pure-Python multiset check that reproduces the
    e-graph's set-equality conclusion (documented, not a silent pass).
    """
    scope = f"K={k_extent} tile_k={tile_k} k_steps={k_steps}"
    avail, why = egglog_available()
    if not avail:
        raise EscalationUnavailable(
            f"egglog multiset escalation for '{name}' unavailable: {why}"
        )

    # Build the two multisets the e-graph would equate after saturation:
    #   serial  = { k : 0 <= k < K }
    #   tiled   = { kt*tile_k + kk : 0<=kt<k_steps, 0<=kk<tile_k, in range }
    serial = {}
    for kk_ in range(k_extent):
        serial[kk_] = serial.get(kk_, 0) + 1
    tiled = {}
    overshoot = []
    for kt in range(k_steps):
        for kk in range(tile_k):
            kv = kt * tile_k + kk
            if kv >= k_extent:
                overshoot.append((kt, kk, kv))
                continue
            tiled[kv] = tiled.get(kv, 0) + 1

    # Drive egglog as the authoritative engine: model each producer as a term
    # and saturate the tiling equality, then read back equivalence classes.
    egglog_ok, egglog_note = _egglog_saturate_equiv(serial, tiled, tile_k, k_steps, k_extent)

    notes: list[str] = [egglog_note]
    missing = {k: serial[k] for k in serial if tiled.get(k, 0) != serial[k]}
    extra = {k: tiled[k] for k in tiled if serial.get(k, 0) != tiled[k]}
    equal_multiset = (not missing) and (not extra) and (not overshoot)

    proved = equal_multiset and egglog_ok
    if proved:
        return EscalationResult(
            name=name,
            kind="algebraic",
            proved=True,
            proved_by="egglog",
            bounded=True,
            scope=scope,
            reason=(
                "egglog equality saturation: tiled product multiset == serial "
                f"product multiset (sum over comm/assoc monoid) @ {scope}; the "
                "abstract-ring reassociation is exact (fp rounding is the "
                "separate numeric gate); proves THIS index domain only"
            ),
            notes=tuple(notes),
        )
    cw_parts = []
    if missing:
        cw_parts.append(f"k present in serial but not tiled: {sorted(missing)[:8]}")
    if extra:
        cw_parts.append(f"k over-produced by tiling: {sorted(extra)[:8]}")
    if overshoot:
        cw_parts.append(f"producers outside [0,K): {overshoot[:8]}")
    return EscalationResult(
        name=name,
        kind="algebraic",
        proved=False,
        proved_by="unproven",
        bounded=True,
        scope=scope,
        reason=(
            "egglog multiset equivalence FAILED: tiled and serial product "
            f"multisets differ @ {scope}"
        ),
        counter_witness="; ".join(cw_parts) or egglog_note,
        notes=tuple(notes),
    )


def _egglog_saturate_equiv(serial, tiled, tile_k, k_steps, k_extent) -> tuple[bool, str]:
    """Use egglog's e-graph to prove the serial SUM == the tiled SUM.

    Models the reduction as a real e-graph term over a commutative + associative
    ``Add`` of per-k product leaves ``Prod(k)``:

        serial_sum = Prod(0) + Prod(1) + ... + Prod(K-1)
        tiled_sum  = sum over (kt,kk) of Prod(kt*tile_k+kk)   [in range]

    We register COMMUTATIVITY and ASSOCIATIVITY rewrites for ``Add`` (the
    reduction is over a commutative monoid; this is exactly the reassociation a
    tiled GEMM performs — exact in the abstract ring, fp rounding is the
    separate numeric gate), saturate, and ask the e-graph whether the two sum
    terms are in the SAME e-class. When the tiled producer multiset equals the
    serial multiset, equality saturation merges them and ``check(eq)`` succeeds;
    when they differ (a missing/extra/overshoot producer), no rewrite sequence
    equates them and ``check`` raises -> equiv False. On any egglog API mismatch
    we RAISE (fail-loud), never silently degrade.
    """
    from egglog import EGraph, Expr, i64, i64Like, rewrite, eq  # type: ignore

    class R(Expr):  # reduction term algebra
        @classmethod
        def prod(cls, k: i64Like) -> "R":  # a single product A[i,k]*B[k,j]
            ...

        def __add__(self, other: "R") -> "R":  # commutative+associative monoid
            ...

    def sum_terms(keys: list[int]) -> "R":
        acc = R.prod(i64(keys[0]))
        for k in keys[1:]:
            acc = acc + R.prod(i64(k))
        return acc

    serial_keys = sorted(serial.keys())
    tiled_keys_list = []
    for kt in range(k_steps):
        for kk in range(tile_k):
            kv = kt * tile_k + kk
            tiled_keys_list.append(kv)  # keep overshoot so a bad tiling differs

    if not serial_keys or not tiled_keys_list:
        return False, "egglog: empty reduction (degenerate)"

    egraph = EGraph()
    a, b, c = R.prod(i64(0)), R.prod(i64(1)), R.prod(i64(2))
    x, y, z = a, b, c
    # commutativity + associativity of the reduction's Add.
    egraph.register(rewrite(x + y).to(y + x))
    egraph.register(rewrite((x + y) + z).to(x + (y + z)))
    egraph.register(rewrite(x + (y + z)).to((x + y) + z))

    serial_sum = sum_terms(serial_keys)
    tiled_sum = sum_terms(tiled_keys_list)
    egraph.register(serial_sum)
    egraph.register(tiled_sum)
    egraph.saturate()

    equiv = True
    try:
        egraph.check(eq(serial_sum).to(tiled_sum))
    except Exception:
        equiv = False
    note = (
        f"egglog: saturated comm/assoc over serial sum of {len(serial_keys)} "
        f"products and tiled sum of {len(tiled_keys_list)} products; "
        f"e-graph proves serial_sum == tiled_sum: {equiv}"
    )
    return equiv, note


# --------------------------------------------------------------------------- #
# Layer B(c) — DESIGN-ONLY Lean lemma sketch for the GENERAL (symbolic) bijection.
# Not installed, not executed. Returned as a string for the follow-up.
# --------------------------------------------------------------------------- #
LEAN_BIJECTION_LEMMA_SKETCH = r"""
-- DESIGN-ONLY (Lean 4 / Mathlib). The general symbolic-tile bijection that the
-- z3-concrete / egglog layers only establish for fixed tiles. One-time lemma:
-- the tile map (kt, kk) ↦ kt*tile + kk is a bijection
--   Fin k_steps × Fin tile  ≃  Fin (k_steps * tile)
-- and, restricting to k_steps*tile = K, onto the reduction domain Fin K.
--
-- import Mathlib.Logic.Equiv.Fin
-- import Mathlib.Data.Fin.Basic
--
-- theorem tiling_bijection (tile k_steps : ℕ) (htile : 0 < tile) :
--     Function.Bijective
--       (fun p : Fin k_steps × Fin tile => (p.1.val * tile + p.2.val)) := by
--   -- This is exactly `finProdFinEquiv` (Mathlib): Fin m × Fin n ≃ Fin (m*n)
--   -- via a*n + b, which is a bijection for n > 0. Use its forward map and the
--   -- divmod inverse (euclidean division by `tile`):
--   constructor
--   · -- injective: from k0 = k1 with kk0,kk1 < tile, divmod by tile is unique
--     intro p q hpq
--     -- Nat.div_add_mod / Nat.mod_lt give kt = k / tile, kk = k % tile uniquely.
--     exact (finProdFinEquiv).injective (by simpa using hpq)
--   · -- surjective: every k < k_steps*tile has k = (k/tile)*tile + (k%tile)
--     intro k
--     exact (finProdFinEquiv).surjective k
--
-- Corollary (the GEMM obligation): for K = k_steps * tile, the multiset of
-- serial products {A[i,k]*B[k,j] : k ∈ Fin K} equals the multiset produced by
-- the tiled loop {A[i, kt*tile+kk]*B[kt*tile+kk, j]}, hence (summing over a
-- commutative monoid) the serial reduction equals the tiled accumulation in the
-- abstract ring. fp rounding is out of scope (separate numeric parity gate).
--
-- Effort: ~1 file, ~40 LOC, relies on Mathlib `finProdFinEquiv`. Build this
-- ONCE to discharge ALL tile sizes ∀; until then z3-concrete + egglog give the
-- per-shape bounded results this module returns.
""".strip()


def lean_bijection_lemma_sketch() -> str:
    """Return the design-only Lean lemma text (NOT executed; no Lean install)."""
    return LEAN_BIJECTION_LEMMA_SKETCH


# --------------------------------------------------------------------------- #
# The dispatcher — classify a z3-unknown obligation and escalate.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlgebraicObligation:
    """An algebraic-equivalence obligation (tiling bijection over k)."""

    name: str
    k_extent: int
    tile_k: int
    k_steps: int


@dataclass(frozen=True)
class RaceObligation:
    """A race-freedom obligation (tiled writer single-writer).

    ``representative`` (default True) shrinks the grid to a TLC-tractable
    REPRESENTATIVE size that preserves the tiling's overlap structure: the
    overlap relation between adjacent blocks depends only on (tile - step), not
    on the absolute extent, so checking a few blocks in each axis exhaustively
    covers every overlap pattern the production grid exhibits. The full
    production grid (e.g. 64x64) is exponential for TLC and times out; the
    representative grid keeps the BOUNDED guarantee honest while being fast.
    Set ``representative=False`` to force the literal production grid (may time
    out -> reported "unproven", never a silent pass).
    """

    name: str
    m_extent: int
    n_extent: int
    tile_m: int
    tile_n: int
    m_step: int
    n_step: int
    m_blocks: int
    n_blocks: int
    representative: bool = True
    # 2 blocks per axis = a block plus ONE neighbor, which exhibits the full
    # pairwise overlap relation NoDoubleWrite quantifies over (the invariant is
    # over PAIRS of blocks). Each axis contributes <= tile'+step' cells, keeping
    # the per-cell write-set state space tractable (16 cells -> ~65k states).
    rep_blocks: int = 2


def _rep_tile_step(tile: int, step: int) -> tuple[int, int]:
    """Shrink one axis's (tile, step) preserving the OVERLAP RELATION.

    Whether two adjacent blocks (at offsets b*step and (b+1)*step, each writing
    ``tile`` cells) overlap depends ONLY on the sign/amount of ``tile - step``,
    not on the absolute magnitudes:
      * disjoint & contiguous  <=> tile == step
      * overlapping            <=> tile  > step  (by ``tile-step`` cells)
      * gap (uncovered)        <=> tile  < step
    We map to the SMALLEST (tile', step') with the same overlap delta so TLC
    sees the identical race structure on a tiny grid:
      step' = max(1, min(step, 2)); overlap = tile - step; tile' = step'+overlap
    clamped to >= 1. This collapses tile=16/step=16 to 2/2, tile=20/step=16 to
    6/2 (overlap 4 preserved... but we only need overlap>0), tile=12/step=16 to
    a gap. To keep cells minimal while preserving the three cases we cap overlap
    representation at 1 cell (presence of overlap is what the invariant detects).
    """
    if tile == step:
        return 2, 2  # disjoint contiguous
    if tile > step:
        # overlapping: represent with a 1-cell overlap (step'=2, tile'=3).
        return 3, 2
    # gap (tile < step): represent with a 1-cell gap (step'=3, tile'=2).
    return 2, 3


def representative_race_shape(ob: "RaceObligation") -> dict[str, int]:
    """Shrink a production race obligation to a TLC-tractable representative.

    Two independent reductions, each preserving the exact race structure:
      1. (tile, step) per axis are collapsed to the smallest pair with the same
         OVERLAP RELATION (disjoint / overlap / gap) — see :func:`_rep_tile_step`.
         The interior of a large tile contains no race information; only the
         boundary between adjacent blocks does.
      2. the block count per axis is capped at ``rep_blocks`` (default 2 = a
         block plus one neighbor); since NoDoubleWrite quantifies over PAIRS of
         blocks and all blocks/steps are identical, two adjacent blocks exhibit
         every overlap pattern an arbitrarily long row of identical blocks can.
    The resulting grid is a handful of cells, so TLC's exhaustive interleaving
    search runs in well under a second while the BOUNDED guarantee stays honest:
    it proves race-freedom for the tiling's STRUCTURE, labelled as bounded.
    """
    mb = max(1, min(ob.m_blocks, ob.rep_blocks))
    nb = max(1, min(ob.n_blocks, ob.rep_blocks))
    tm, ms = _rep_tile_step(ob.tile_m, ob.m_step)
    tn, ns = _rep_tile_step(ob.tile_n, ob.n_step)
    m_ext = (mb - 1) * ms + tm
    n_ext = (nb - 1) * ns + tn
    return {
        "m_extent": m_ext,
        "n_extent": n_ext,
        "tile_m": tm,
        "tile_n": tn,
        "m_step": ms,
        "n_step": ns,
        "m_blocks": mb,
        "n_blocks": nb,
    }


def escalate_obligation(
    obligation: AlgebraicObligation | RaceObligation,
    *,
    prefer_egglog: bool = False,
) -> EscalationResult:
    """Dispatch a z3-UNKNOWN obligation to the right escalation layer.

    RACE obligation -> TLC bounded model checking.
    ALGEBRAIC obligation -> z3-concrete-tiles (primary); egglog if z3-concrete
        cannot decide OR ``prefer_egglog``. The two are cross-checks: a result
        is reported proved only if its layer actually discharged it; otherwise
        "unproven" (never silently passed). Either raises EscalationUnavailable
        if its backend is missing (fail-loud), so a caller can choose to keep
        the serial path rather than mistake a missing tool for a proof.
    """
    if isinstance(obligation, RaceObligation):
        if obligation.representative:
            shape = representative_race_shape(obligation)
            base = prove_race_freedom_tlc(
                name=obligation.name,
                m_extent=shape["m_extent"],
                n_extent=shape["n_extent"],
                tile_m=shape["tile_m"],
                tile_n=shape["tile_n"],
                m_step=shape["m_step"],
                n_step=shape["n_step"],
                m_blocks=shape["m_blocks"],
                n_blocks=shape["n_blocks"],
            )
            prod = (
                f"prod dim={obligation.m_extent}x{obligation.n_extent} "
                f"tile={obligation.tile_m}x{obligation.tile_n} "
                f"step={obligation.m_step}x{obligation.n_step} "
                f"blocks={obligation.m_blocks}x{obligation.n_blocks}"
            )
            # Re-stamp scope/reason so callers see the production shape this
            # representative grid stands in for (overlap-structure-preserving).
            return EscalationResult(
                name=base.name,
                kind=base.kind,
                proved=base.proved,
                proved_by=base.proved_by,
                bounded=True,
                scope=f"{prod}  via representative {base.scope}",
                reason=(
                    f"{base.reason} [representative grid preserves the "
                    f"overlap relation of the {prod}]"
                ),
                counter_witness=base.counter_witness,
                notes=base.notes,
            )
        return prove_race_freedom_tlc(
            name=obligation.name,
            m_extent=obligation.m_extent,
            n_extent=obligation.n_extent,
            tile_m=obligation.tile_m,
            tile_n=obligation.tile_n,
            m_step=obligation.m_step,
            n_step=obligation.n_step,
            m_blocks=obligation.m_blocks,
            n_blocks=obligation.n_blocks,
        )

    if isinstance(obligation, AlgebraicObligation):
        if prefer_egglog:
            return prove_multiset_equiv_egglog(
                name=obligation.name,
                k_extent=obligation.k_extent,
                tile_k=obligation.tile_k,
                k_steps=obligation.k_steps,
            )
        primary = prove_bijection_z3_concrete(
            name=obligation.name,
            k_extent=obligation.k_extent,
            tile_k=obligation.tile_k,
            k_steps=obligation.k_steps,
        )
        # A concrete counterexample is already a definitive refutation.  Do
        # not ask the optional secondary backend to re-prove a known failure:
        # that would turn a useful witness into an unrelated "backend missing"
        # result and hide the actual production defect.
        if primary.counter_witness is not None:
            return primary
        if primary.proved:
            return primary
        # z3-concrete could not decide -> SECONDARY egglog cross-check.
        secondary = prove_multiset_equiv_egglog(
            name=obligation.name,
            k_extent=obligation.k_extent,
            tile_k=obligation.tile_k,
            k_steps=obligation.k_steps,
        )
        return secondary

    raise TypeError(f"unknown obligation type: {type(obligation).__name__}")
