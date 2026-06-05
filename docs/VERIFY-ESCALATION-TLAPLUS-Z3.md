# Layered GEMM-rewrite verification: z3-first, escalate-on-unknown

§V1 — TLA+/TLC race-freedom + z3-concrete / egglog algebraic equivalence

This documents the standalone escalation MACHINERY for the GEMM-rewrite prover:
**z3 FIRST**, and on `z3.unknown` **classify the obligation and escalate** to the
tool that is natural for its category. It plugs into the EXISTING z3 interface
(`_async_barrier_plan._z3_proves_simdgroup_output_isolated`, which returns
`(z3_used, z3_proved, reason)` and the literal `"z3 returned unknown ..."` string
on `z3.unknown`) and never rewrites it — it wraps that same verdict and dispatches.

## RULE #1 honesty contract (no silent overclaim)

| verdict / layer | `proved_by` | `bounded` | claim strength |
|---|---|---|---|
| z3 `unsat` | `z3` | `False` | ∀ over the z3-encoded (quantified) index domain |
| z3 `sat` (counter-witness) | `unproven` | — | rewrite is WRONG; surfaced loud, **NOT** escalated to mask it |
| TLC clean | `tlc-bounded` | `True` | race-free **at the checked grid only**, NOT ∀N |
| z3-concrete-tiles | `z3-concrete-tiles` | `True` | bijection **for these concrete tiles only**, NOT ∀tile |
| egglog | `egglog` | `True` | multiset equality **for this index domain only**, NOT ∀N |
| no layer resolves / backend missing | `unproven` | — | reported UNPROVEN (never silently passed); fetch cmd surfaced |

TLC is **BOUNDED model checking**: it proves the property AT the checked sizes,
not ∀N. Every TLC pass is labelled `"verified BOUNDED @ dim=X tile=Y"` and
`bounded=True`. The z3-concrete / egglog layers likewise prove only the concrete
index domain. The only `bounded=False` claim is a z3 `unsat`, which is quantified
over the encoded domain. Numeric fp parity over ALL elements is the empirical
ground-truth backstop and is **NOT** a substitute for these structural claims.

## Components

- `cppmega_mlx/nn/_tilelang/verify_escalation.py` — top-level dispatcher
  `verify_with_escalation(name, z3_probe, obligation, prefer_egglog)`. Consumes the
  existing prover's `(z3_used, z3_proved, reason)` verdict (ONE z3 path), classifies
  proved / refuted / unknown, and on unknown dispatches by `kind`.
- `cppmega_mlx/nn/_tilelang/_gemm_rewrite_escalation.py` — TLA+ spec auto-generator
  + TLC runner (`prove_race_freedom_tlc`, `render_writer_tla`), the z3-concrete
  bijection prover, the egglog multiset prover, and the representative-grid shrink
  (`representative_race_shape`) that preserves the overlap relation.
- `cppmega_mlx/nn/_tilelang/verify_algebraic.py` — Track-B algebraic arm
  (`prove_bijection_z3_concrete`, `prove_multiset_equiv_egglog`, scale/mask identity
  refuters, and the DESIGN-ONLY Lean `finProdFinEquiv` lemma sketch).
- `tools/tla2tools.jar` — the TLC engine (single jar, 2.27 MB).
- Tests: `tests/test_verify_escalation.py`, `tests/test_gemm_rewrite_escalation.py`,
  `tests/test_verify_algebraic.py` — **53 passing**, lock in non-vacuity.

## The TLA+ writer spec (race-freedom)

`render_writer_tla` emits a MODULE of the tiled-GEMM writer state machine: block
`(bm,bn)` owns the rectangle `[bm*MStep, +TileM) x [bn*NStep, +TileN)`; `Next`
lets any block write any owned cell in any interleaving; `wrote[c]` accumulates the
SET of distinct block-ids that wrote cell `c`; the safety invariant

```
NoDoubleWrite == \A c \in Cells : Cardinality(wrote[c]) <= 1
```

says no output cell is ever claimed by two distinct blocks. TLC explores EVERY
interleaving exhaustively over the bounded grid. A correct disjoint tiling
(`step == tile`) holds; an overlapping tiling (`tile > step`) yields a TLC trace
witnessing the double write.

**Representative-grid honesty:** the production grid (64×64, 4×4 blocks) is
exponential for TLC. We shrink each axis's `(tile, step)` to the smallest pair with
the SAME overlap relation (disjoint ⇔ `tile==step`, overlap ⇔ `tile>step`, gap ⇔
`tile<step`) and cap blocks at 2 (NoDoubleWrite quantifies over PAIRS of blocks, and
all blocks are identical, so two adjacent blocks exhibit every overlap pattern). The
result is labelled with BOTH the production shape AND the representative grid it
stands in for — the bounded guarantee stays honest.

## MEASURED RESULTS (this run, macOS arm64 CPU)

### Obligation (a): F0/B2 tiled-writer RACE — parallel threads writing dC / dchunk_states / DYX cells

z3 returns UNKNOWN (race-freedom phrased as a concurrency interleaving, not a static
ownership predicate) → escalate to **TLA+/TLC**.

- prod `dim=64×64 tile=16×16 step=16×16 blocks=4×4`, via representative `dim=4×4 tile=2×2 step=2×2 blocks=2×2`
- **`proved=True`, `proved_by="tlc-bounded"`, `bounded=True`**
- TLC: `524289 states generated, 65536 distinct states found, 0 left on queue` — exhaustive over all interleavings
- Label: *"NoDoubleWrite holds — verified BOUNDED @ dim=4×4 tile=2×2 step=2×2 blocks=2×2; NOT proven ∀N"*

### Obligation (b): tiling-BIJECTION algebraic equivalence — serial reduction (i,j,k) domain ≡ tiled-GEMM domain

z3 returns UNKNOWN (the bijection `k = kt*tile_k + kk` is nonlinear when `tile_k` is
symbolic) → escalate to **z3-concrete-tiles** (PRIMARY), **egglog** (SECONDARY).

- `K=64 tile_k=16 k_steps=4`
- PRIMARY: **`proved=True`, `proved_by="z3-concrete-tiles"`, `bounded=True`** — injective + covering (`ForAll`) + no-overshoot, all `unsat` of the negation, on concrete divisors (Presburger/decidable). *"proves ONLY these tiles, not ∀tile size"*
- SECONDARY egglog cross-check: **`proved=True`, `proved_by="egglog"`, `bounded=True`** — tiled product multiset == serial product multiset under comm+assoc of `+` (exact in the abstract ring; fp rounding is the separate numeric gate). *"proves THIS index domain only"*

## NON-VACUITY (a known-WRONG rewrite is REFUTED by the right layer)

| injected fault | layer | verdict | witness |
|---|---|---|---|
| overlapping writers `tile=20 > step=16` (real race) | TLC | **REFUTED** `unproven` | TLC trace: *"NoDoubleWrite is violated, two distinct blocks own the same cell"* @ rep `dim=5×5 tile=3×3 step=2×2` |
| dropped k-tile `k_steps=3` (GAP, tiled sum drops products) | z3-concrete / egglog | **REFUTED** `unproven` | *"k present in serial but not tiled: [48,49,50,51,52,53,54,55]"* |
| transposed operand ⇒ flipped causal mask (`k≤32` vs `k≥32`) | algebraic mask refuter | **REFUTED** `unproven` | *"mask mismatch at k=0: serial=True gemm=False"* |
| z3 `sat` counter-witness | dispatcher | **REFUTED** `unproven`, NOT escalated | counter-witness carried; escalation must never paper over a real bug |

All four faults are caught: the layers are non-vacuous, not trivially-passing.

## z3 base-case (what z3 settles directly, no escalation)

- z3 `unsat` → `proved_by="z3"`, `bounded=False`, `escalated=False` — the only ∀ claim (quantified over the encoded index domain).
- z3 `sat` → `proved_by="unproven"`, `z3_resolved=True`, `escalated=False` — the rewrite is WRONG; surfaced loud, **NOT** escalated.
- z3 `unknown` → `escalated=True` → the fallback layer above (or `unproven` if no layer resolves).

In THIS session both real obligations hit `z3.unknown` (race = concurrency
interleaving; bijection = nonlinear symbolic tile) and were resolved ONLY by the
escalation layers, **bounded**. No real obligation here was settled ∀N by z3 alone;
the ∀N path is exercised by the unit tests' synthetic `unsat` probe.

## Coverage summary (honest)

| obligation | z3 ∀N? | escalated? | resolved by | bound |
|---|---|---|---|---|
| F0/B2 tiled-writer race | no (unknown) | yes | `tlc-bounded` | dim=4×4 rep of prod 64×64, all interleavings |
| tiling bijection | no (unknown) | yes | `z3-concrete-tiles` (+ egglog cross-check) | K=64 tile_k=16 k_steps=4 |

No obligation in this run is left `unproven` — but the labelling is BOUNDED, never ∀N.
The ∀N tiling bijection is the DESIGN-ONLY Lean `finProdFinEquiv` lemma sketch
(`lean_bijection_lemma_sketch()`, ~40 LOC, not built this round — no Lean install).

## Deferred wiring

Final integration into the running w8ctouyfx z3 GEMM-equivalence prover (track C) is
DEFERRED until that lands. This workflow builds the escalation machinery standalone
against the `_async_barrier_plan` / `_gemm_rewrite_proof` interface and tests it on
the two real obligations. Wiring is a drop-in: at each site where the track-C prover
gets a z3 `unknown`, construct the matching `RaceObligation` / `AlgebraicObligation`
and call `verify_with_escalation(...)`; the `MetalReductionSyncPlan` receipt gains
`proved_by ∈ {z3, tlc-bounded, z3-concrete-tiles, egglog, unproven}` via
`as_feature_dict()`.
