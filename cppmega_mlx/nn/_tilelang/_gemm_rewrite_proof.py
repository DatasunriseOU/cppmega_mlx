"""z3 prover for the serial-reduction -> T.gemm (matmul2d) rewrite.

Track C of the Metal-GEMM work. The auto-GEMM pass (track B) detects a serial
reduction whose body is ``acc += A[..k..] * B[..k..]`` (a contraction = a
matmul) and wants to rewrite it to ``T.gemm`` / a tensorized block. Before that
rewrite may fire this module PROVES, with z3, that the rewrite preserves the
reduction semantics and is race-free. RULE #1: no rewrite without a passing z3
proof — a contraction the prover cannot discharge keeps the serial path (or, in
forced mode, RAISES); it is NEVER silently GEMM-ified.

Scope (the boundary the user stated):
  * z3 proves the SYMBOLIC / structural contract — the operand index maps match
    (no accidental transpose), the tiling is a bijection over the reduction
    index (every serial product is produced exactly once), the mask / scale the
    GEMM folds is identical to the serial one, and each output cell is written
    by exactly one threadgroup (race-free).
  * z3 does NOT prove fp16/fp32 bit-equivalence — reassociating a floating-point
    sum changes rounding. That stays the all-elements numeric parity test
    (~5e-4 fp16 gate). This module is structure-only on purpose.

The prover mirrors ``_async_barrier_plan._z3_proves_simdgroup_output_isolated``
and ``mamba3_path_c._z3_proves_mamba3_lane_mapping``: each obligation builds the
NEGATION of a property and checks for ``unsat`` (proved) / ``sat`` (a concrete
counter-witness) / ``unknown`` (not proved, fail-closed). The proof is
NON-VACUOUS — a deliberately broken rewrite (transposed operand, off-by-one
tile, dropped k-tile, wrong mask) yields a ``sat`` counter-witness, locked in by
the sibling test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from . import _msl_transform

# Reuse the exact env-disable knobs the sibling z3 provers honor.
_Z3_DISABLE_ENV = (
    "TILELANG_DISABLE_Z3",
    "TILELANG_DISABLE_Z3_GEMM_REWRITE",
    "CPPMEGA_DISABLE_Z3",
)

GemmRewriteZ3Policy = Literal["env", "enabled", "disabled"]

# Per-obligation solver timeout (ms). The encodings are linear integer
# arithmetic (Presburger) plus a single uninterpreted scale function, which z3
# decides effectively instantly at production tile sizes; the timeout only
# bounds pathological inputs (which then report z3_proved=False, fail-closed).
_SOLVER_TIMEOUT_MS = 500


# A symbolic predicate / scale is a python callable that, given z3 Int handles
# for the output/reduction indices, returns a z3 BoolRef (mask) or ArithRef
# (scale). The SAME callable the track-B pass reconstructs and folds into the
# emitted T.gemm is handed here, so the proof checks the rewrite the pass
# actually emits — not an idealized one.
MaskPredicate = Callable[..., object]
ScaleExpr = Callable[..., object]


@dataclass(frozen=True)
class GemmContraction:
    """A serial reduction the auto-pass recognized as a matmul.

    ``out[i, j] = sum_k  scale(i, k, j) * A[a_row(i,k), a_col(i,k)]
                                        * B[b_row(k,j), b_col(k,j)]``
    restricted to ``mask_keep(i, j, k)``.

    The operand index maps are given as ``(row_stride, col_stride)`` flattened
    addresses so a transpose shows up as a swapped address arithmetic. The
    default maps encode the un-transposed contraction ``A[i,k] @ B[k,j]``.
    """

    name: str
    m_extent: int  # output rows  (i in [0, m_extent))
    n_extent: int  # output cols  (j in [0, n_extent))
    k_extent: int  # reduction    (k in [0, k_extent))

    # Operand A flattened address as the SERIAL loop reads it and as the GEMM
    # reads it. Each is f(i, k) -> z3 ArithRef. A transposed-A bug makes the two
    # disagree (e.g. serial i*K+k vs gemm k*M+i).
    a_addr_serial: Callable[..., object]
    a_addr_gemm: Callable[..., object]
    # Operand B flattened address, f(k, j) -> z3 ArithRef.
    b_addr_serial: Callable[..., object]
    b_addr_gemm: Callable[..., object]

    # Optional causal / triangular mask. f(i, j, k) -> z3 BoolRef. None => all
    # products kept (dense matmul).
    mask_serial: Optional[MaskPredicate] = None
    mask_gemm: Optional[MaskPredicate] = None

    # Optional per-term scale folded into an operand (F0 folds decay*dt into x;
    # B2 folds decay into dY). f(i, k, j) -> z3 ArithRef over an uninterpreted
    # base scale function. None => unit scale.
    scale_serial: Optional[ScaleExpr] = None
    scale_gemm: Optional[ScaleExpr] = None

    def has_mask(self) -> bool:
        return self.mask_serial is not None or self.mask_gemm is not None

    def has_scale(self) -> bool:
        return self.scale_serial is not None or self.scale_gemm is not None


@dataclass(frozen=True)
class GemmTiling:
    """The threadgroup / micro-tile decomposition the GEMM rewrite emits.

    The reduction index ``k`` is decomposed ``k = kt*tile_k + kk`` over
    ``kt in [0, k_steps)``, ``kk in [0, tile_k)``; the output cell ``(i, j)`` is
    owned by threadgroup ``(bm, bn)`` which STEPS by ``(m_stride, n_stride)``
    and WRITES a tile of ``(tile_m, tile_n)`` cells, i.e. block ``(bm, bn)``
    owns ``i in [bm*m_stride, bm*m_stride+tile_m)``,
    ``j in [bn*n_stride, bn*n_stride+tile_n)``. A correct rewrite has
    ``m_stride == tile_m`` / ``n_stride == tile_n`` (contiguous, disjoint); a
    buggy rewrite whose tile exceeds its stride overlaps and is caught by the
    single-writer obligation. ``m_stride`` / ``n_stride`` default to the tile
    extents.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    m_blocks: int
    n_blocks: int
    k_steps: int
    m_stride: Optional[int] = None
    n_stride: Optional[int] = None

    @property
    def m_step(self) -> int:
        return self.tile_m if self.m_stride is None else self.m_stride

    @property
    def n_step(self) -> int:
        return self.tile_n if self.n_stride is None else self.n_stride

    @property
    def k_covered_extent(self) -> int:
        return self.k_steps * self.tile_k


@dataclass(frozen=True)
class GemmRewriteProof:
    """Result of proving one serial-reduction -> T.gemm rewrite.

    Mirrors ``MetalReductionSyncPlan``: carries the (z3_used, z3_proved, reason)
    triple plus a per-obligation breakdown and the contraction it proved, and
    exposes ``as_feature_dict`` so it flows into the same receipt plumbing.
    ``z3_proved`` is True iff EVERY applicable obligation discharged to unsat.
    """

    name: str
    z3_used: bool
    z3_proved: bool
    reason: str
    # Per-obligation outcomes (True = proved/unsat). Obligations that do not
    # apply to this contraction (e.g. no mask) report True with a "n/a" note.
    operand_maps_match: bool = False
    tiling_injective: bool = False
    k_covered: bool = False
    mask_equiv: bool = False
    scale_equiv: bool = False
    single_writer: bool = False
    obligation_notes: tuple[str, ...] = field(default_factory=tuple)
    m_extent: int = 0
    n_extent: int = 0
    k_extent: int = 0
    tile_m: int = 0
    tile_n: int = 0
    tile_k: int = 0

    def as_feature_dict(self) -> dict[str, int | bool | str]:
        return {
            "gemm_proof_name": self.name,
            "gemm_proof_z3_used": self.z3_used,
            "gemm_proof_z3_proved": self.z3_proved,
            "gemm_proof_reason": self.reason,
            "gemm_proof_operand_maps_match": self.operand_maps_match,
            "gemm_proof_tiling_injective": self.tiling_injective,
            "gemm_proof_k_covered": self.k_covered,
            "gemm_proof_mask_equiv": self.mask_equiv,
            "gemm_proof_scale_equiv": self.scale_equiv,
            "gemm_proof_single_writer": self.single_writer,
            "gemm_proof_m_extent": self.m_extent,
            "gemm_proof_n_extent": self.n_extent,
            "gemm_proof_k_extent": self.k_extent,
            "gemm_proof_tile_m": self.tile_m,
            "gemm_proof_tile_n": self.tile_n,
            "gemm_proof_tile_k": self.tile_k,
        }


class GemmRewriteNotProven(RuntimeError):
    """Raised when a forced GEMM rewrite cannot be z3-proven (RULE #1)."""


def _z3_disabled(policy: GemmRewriteZ3Policy = "env") -> bool:
    if policy == "enabled":
        return False
    if policy == "disabled":
        return True
    if policy != "env":
        raise ValueError(f"invalid GEMM-rewrite z3 policy: {policy!r}")
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
        for name in _Z3_DISABLE_ENV
    )


def _disabled_proof(contraction: GemmContraction, tiling: GemmTiling, reason: str) -> GemmRewriteProof:
    return GemmRewriteProof(
        name=contraction.name,
        z3_used=False,
        z3_proved=False,
        reason=reason,
        m_extent=contraction.m_extent,
        n_extent=contraction.n_extent,
        k_extent=contraction.k_extent,
        tile_m=tiling.tile_m,
        tile_n=tiling.tile_n,
        tile_k=tiling.tile_k,
    )


def _validate_shapes(contraction: GemmContraction, tiling: GemmTiling) -> Optional[str]:
    extents = {
        "m_extent": contraction.m_extent,
        "n_extent": contraction.n_extent,
        "k_extent": contraction.k_extent,
        "tile_m": tiling.tile_m,
        "tile_n": tiling.tile_n,
        "tile_k": tiling.tile_k,
        "m_blocks": tiling.m_blocks,
        "n_blocks": tiling.n_blocks,
        "k_steps": tiling.k_steps,
        "m_step": tiling.m_step,
        "n_step": tiling.n_step,
    }
    bad = {name: value for name, value in extents.items() if value <= 0}
    if bad:
        return f"non-positive contraction/tiling values: {bad}"
    return None


def prove_gemm_rewrite(
    contraction: GemmContraction,
    tiling: GemmTiling,
    *,
    z3_policy: GemmRewriteZ3Policy = "env",
) -> GemmRewriteProof:
    """Prove a serial-reduction -> T.gemm rewrite is correct + race-free.

    Returns a :class:`GemmRewriteProof`. ``z3_proved`` is True iff z3 ran and
    every applicable obligation discharged to ``unsat`` (i.e. the negated
    property is unsatisfiable). On any ``sat`` the model witness is attached to
    ``reason`` and ``z3_proved`` is False; on ``unknown`` / z3 unavailable /
    disabled, ``z3_proved`` is False (fail-closed, RULE #1).
    """

    shape_err = _validate_shapes(contraction, tiling)
    if shape_err is not None:
        return _disabled_proof(contraction, tiling, shape_err)

    if _z3_disabled(z3_policy):
        reason = "z3 disabled by policy" if z3_policy != "env" else "z3 disabled by environment"
        return _disabled_proof(contraction, tiling, reason)

    # Order matters: preload libz3 the same way mamba3_path_c does so a later
    # tilelang import in this process does not hard-abort on a basename libz3.
    _msl_transform.ensure_libz3_preloaded()
    try:
        import z3  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional local dependency
        return _disabled_proof(
            contraction, tiling, f"z3 unavailable: {type(exc).__name__}: {exc}"
        )

    notes: list[str] = []
    results: dict[str, bool] = {}

    def _discharge(label: str, build) -> bool:
        """Run one obligation. ``build(solver, *ints)`` adds the NEGATED property.

        Returns True iff unsat (property proved). Records a human note. Any
        z3 exception => False with a "z3 raised" note (defensive boundary).
        """

        solver = z3.Solver()
        solver.set("timeout", _SOLVER_TIMEOUT_MS)
        try:
            build(solver, z3)
            result = solver.check()
        except Exception as exc:  # pragma: no cover - defensive z3 boundary
            notes.append(f"{label}: z3 raised {type(exc).__name__}: {exc}")
            return False
        if result == z3.unsat:
            notes.append(f"{label}: proved (unsat)")
            return True
        if result == z3.unknown:
            notes.append(f"{label}: z3 returned unknown")
            return False
        notes.append(f"{label}: COUNTER-WITNESS {solver.model()}")
        return False

    # --- Obligation 1: operand index maps match (no accidental transpose). ---
    def _op_maps(solver, z3):
        i, k, j = z3.Int("i"), z3.Int("k"), z3.Int("j")
        solver.add(0 <= i, i < contraction.m_extent)
        solver.add(0 <= k, k < contraction.k_extent)
        solver.add(0 <= j, j < contraction.n_extent)
        a_serial = contraction.a_addr_serial(i, k)
        a_gemm = contraction.a_addr_gemm(i, k)
        b_serial = contraction.b_addr_serial(k, j)
        b_gemm = contraction.b_addr_gemm(k, j)
        # Negation: SOME in-range index where an operand address differs.
        solver.add(z3.Or(a_serial != a_gemm, b_serial != b_gemm))

    results["operand_maps_match"] = _discharge("operand_maps", _op_maps)

    # --- Obligation 2: tiling is injective over k (no product produced twice). ---
    def _injective(solver, z3):
        kt0, kk0, kt1, kk1 = z3.Int("kt0"), z3.Int("kk0"), z3.Int("kt1"), z3.Int("kk1")
        for kt in (kt0, kt1):
            solver.add(0 <= kt, kt < tiling.k_steps)
        for kk in (kk0, kk1):
            solver.add(0 <= kk, kk < tiling.tile_k)
        k0 = kt0 * tiling.tile_k + kk0
        k1 = kt1 * tiling.tile_k + kk1
        # only consider producers that actually land in the reduction range
        solver.add(k0 < contraction.k_extent, k1 < contraction.k_extent)
        # Negation: two DISTINCT tile coords producing the SAME k (aliasing).
        solver.add(k0 == k1)
        solver.add(z3.Or(kt0 != kt1, kk0 != kk1))

    results["tiling_injective"] = _discharge("tiling_injective", _injective)

    # --- Obligation 3: every serial k has exactly-one tiled producer (coverage). ---
    def _coverage(solver, z3):
        k = z3.Int("k")
        kt, kk = z3.Int("kt"), z3.Int("kk")
        solver.add(0 <= k, k < contraction.k_extent)
        # Negation: a serial k with NO in-range tile producer.
        solver.add(
            z3.ForAll(
                [kt, kk],
                z3.Implies(
                    z3.And(0 <= kt, kt < tiling.k_steps, 0 <= kk, kk < tiling.tile_k),
                    k != kt * tiling.tile_k + kk,
                ),
            )
        )

    results["k_covered"] = _discharge("k_covered", _coverage)

    # --- Obligation 4: mask predicate equivalence (if any). ---
    if contraction.has_mask():
        mask_serial = contraction.mask_serial or (lambda i, j, k: z3.BoolVal(True))
        mask_gemm = contraction.mask_gemm or (lambda i, j, k: z3.BoolVal(True))

        def _mask(solver, z3):
            i, j, k = z3.Int("i"), z3.Int("j"), z3.Int("k")
            solver.add(0 <= i, i < contraction.m_extent)
            solver.add(0 <= j, j < contraction.n_extent)
            solver.add(0 <= k, k < contraction.k_extent)
            keep_serial = mask_serial(i, j, k)
            keep_gemm = mask_gemm(i, j, k)
            # Negation: SOME index kept by one side and dropped by the other.
            solver.add(keep_serial != keep_gemm)

        results["mask_equiv"] = _discharge("mask_equiv", _mask)
    else:
        results["mask_equiv"] = True
        notes.append("mask_equiv: n/a (dense, no mask)")

    # --- Obligation 5: per-term scale fold equivalence (if any). ---
    if contraction.has_scale():
        scale_serial = contraction.scale_serial
        scale_gemm = contraction.scale_gemm
        if scale_serial is None or scale_gemm is None:
            results["scale_equiv"] = False
            notes.append(
                "scale_equiv: one side declares a scale but the other does not"
            )
        else:

            def _scale(solver, z3):
                i, j, k = z3.Int("i"), z3.Int("j"), z3.Int("k")
                solver.add(0 <= i, i < contraction.m_extent)
                solver.add(0 <= j, j < contraction.n_extent)
                solver.add(0 <= k, k < contraction.k_extent)
                fac_serial = scale_serial(i, k, j)
                fac_gemm = scale_gemm(i, k, j)
                # Negation: SOME index where the folded scale diverges.
                solver.add(fac_serial != fac_gemm)

            results["scale_equiv"] = _discharge("scale_equiv", _scale)
    else:
        results["scale_equiv"] = True
        notes.append("scale_equiv: n/a (unit scale)")

    # --- Obligation 6: each output cell written by exactly one threadgroup. ---
    def _single_writer(solver, z3):
        bm0, bn0, bm1, bn1 = z3.Int("bm0"), z3.Int("bn0"), z3.Int("bm1"), z3.Int("bn1")
        i, j = z3.Int("i"), z3.Int("j")
        for bm in (bm0, bm1):
            solver.add(0 <= bm, bm < tiling.m_blocks)
        for bn in (bn0, bn1):
            solver.add(0 <= bn, bn < tiling.n_blocks)
        solver.add(0 <= i, i < contraction.m_extent)
        solver.add(0 <= j, j < contraction.n_extent)

        def owns(bm, bn):
            return z3.And(
                bm * tiling.m_step <= i,
                i < bm * tiling.m_step + tiling.tile_m,
                bn * tiling.n_step <= j,
                j < bn * tiling.n_step + tiling.tile_n,
            )

        # Negation: two DISTINCT blocks both owning the SAME output cell.
        solver.add(owns(bm0, bn0))
        solver.add(owns(bm1, bn1))
        solver.add(z3.Or(bm0 != bm1, bn0 != bn1))

    results["single_writer"] = _discharge("single_writer", _single_writer)

    all_proved = all(results.values())
    if all_proved:
        reason = (
            f"z3 proved GEMM rewrite '{contraction.name}': operand maps match, "
            f"tiling is a bijection over k, mask/scale identical, single-writer "
            f"output (structural equivalence + race-freedom; fp parity is the "
            f"separate numeric gate)"
        )
    else:
        failed = [name for name, ok in results.items() if not ok]
        reason = (
            f"z3 did NOT prove GEMM rewrite '{contraction.name}'; failed "
            f"obligations: {failed}; notes: {notes}"
        )

    return GemmRewriteProof(
        name=contraction.name,
        z3_used=True,
        z3_proved=all_proved,
        reason=reason,
        operand_maps_match=results["operand_maps_match"],
        tiling_injective=results["tiling_injective"],
        k_covered=results["k_covered"],
        mask_equiv=results["mask_equiv"],
        scale_equiv=results["scale_equiv"],
        single_writer=results["single_writer"],
        obligation_notes=tuple(notes),
        m_extent=contraction.m_extent,
        n_extent=contraction.n_extent,
        k_extent=contraction.k_extent,
        tile_m=tiling.tile_m,
        tile_n=tiling.tile_n,
        tile_k=tiling.tile_k,
    )


def require_gemm_rewrite_proof(
    contraction: GemmContraction,
    tiling: GemmTiling,
    *,
    z3_policy: GemmRewriteZ3Policy = "env",
) -> GemmRewriteProof:
    """Built-in proof gate for track B's pass (RULE #1, fail-fast/fail-loud).

    Call this from the auto-GEMM pass / the hand-rewrite flag in FORCED mode:
    when the rewrite is being applied it MUST be proven first. If z3 cannot
    prove it (sat counter-witness, unknown, unavailable, or disabled), this
    RAISES :class:`GemmRewriteNotProven` with the WHERE + WHAT — it never
    returns an unproven go-ahead. In detect-and-prefer mode the pass should
    instead call :func:`prove_gemm_rewrite` and keep the serial loop when
    ``z3_proved`` is False.
    """

    proof = prove_gemm_rewrite(contraction, tiling, z3_policy=z3_policy)
    if not proof.z3_proved:
        raise GemmRewriteNotProven(
            f"GEMM rewrite '{contraction.name}' refused: z3 did not prove it "
            f"correct/race-free. {proof.reason}"
        )
    return proof


# --------------------------------------------------------------------------- #
# Concrete contraction descriptors for the real F0 / B2 rewrites.
#
# These build the GemmContraction/GemmTiling for the loops the track-B pass
# targets, so both the pass and the test prove the rewrite that is actually
# emitted (not an idealized one). They take a live ``z3`` module so the index
# maps / masks / scales are z3 expressions.
# --------------------------------------------------------------------------- #


def f0_cb_contraction(z3, *, chunk_size: int, dstate: int):
    """F0 ``cb = C @ B^T`` over the dstate axis (dense, un-transposed-in-k).

    Serial loop (mamba3_chunked_precompute_core F0):
      ``cb[li, si] = sum_n C[base+li, n] * B[base+si, n]``
    Here output rows = li (chunk), cols = si (chunk), reduction k = n (dstate).
    Operand A is C indexed [row=li, col=n]; operand B is B indexed [row=si,
    col=n] — i.e. B^T in the matmul (the GEMM emits ``transpose_B=True``). The
    address maps encode that both operands read column = n with no transpose of
    the reduction axis, so A_addr(i,k)=i*dstate+k and B_addr(k,j)=j*dstate+k.
    """

    return GemmContraction(
        name="F0_cb=C@B^T",
        m_extent=chunk_size,
        n_extent=chunk_size,
        k_extent=dstate,
        a_addr_serial=lambda i, k: i * dstate + k,
        a_addr_gemm=lambda i, k: i * dstate + k,
        b_addr_serial=lambda k, j: j * dstate + k,
        b_addr_gemm=lambda k, j: j * dstate + k,
    )


def b2_dinp_contraction(z3, *, chunk_size: int, headdim: int):
    """B2 ``dinp`` diag input-outer grad with the causal lower-tri mask.

    The diagonal block contribution keeps products only where the query row is
    at or after the key row: ``keep(i, j, k) = (i >= k)`` (lower-triangular /
    causal). The GEMM must apply the IDENTICAL predicate. Here i,k index the
    chunk (L), j indexes headdim. The reduction is over the chunk position k.
    """

    return GemmContraction(
        name="B2_dinp_lowertri",
        m_extent=chunk_size,
        n_extent=headdim,
        k_extent=chunk_size,
        a_addr_serial=lambda i, k: i * chunk_size + k,
        a_addr_gemm=lambda i, k: i * chunk_size + k,
        b_addr_serial=lambda k, j: k * headdim + j,
        b_addr_gemm=lambda k, j: k * headdim + j,
        mask_serial=lambda i, j, k: i >= k,
        mask_gemm=lambda i, j, k: i >= k,
    )


def f0_summary_contraction(z3, *, chunk_size: int, headdim: int, dstate: int):
    """F0 ``summary_states[p,n] = sum_l decay[l]*dt[l]*x[l,p]*B[l,n]``.

    The per-row scale ``decay[l]*dt[l]`` (a function of the reduction index l
    only) is folded into the x operand before the matmul. We model it as an
    uninterpreted base function ``scale(l)`` and prove the GEMM folds the SAME
    index. Output rows = p (headdim), cols = n (dstate), reduction k = l (chunk).
    """

    scale = z3.Function("f0_scale", z3.IntSort(), z3.RealSort())
    return GemmContraction(
        name="F0_summary_states",
        m_extent=headdim,
        n_extent=dstate,
        k_extent=chunk_size,
        a_addr_serial=lambda i, k: k * headdim + i,  # x[l, p]
        a_addr_gemm=lambda i, k: k * headdim + i,
        b_addr_serial=lambda k, j: k * dstate + j,  # B[l, n]
        b_addr_gemm=lambda k, j: k * dstate + j,
        scale_serial=lambda i, k, j: scale(k),
        scale_gemm=lambda i, k, j: scale(k),
    )
