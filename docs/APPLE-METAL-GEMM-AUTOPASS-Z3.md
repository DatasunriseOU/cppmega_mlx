# Apple-Metal GEMM rewrite + auto-GEMMify pass + built-in z3 proof

§M1 — three tracks, MEASURED on this Apple-Silicon Mac (M4 Max, macOS arm64,
MLX 0.31.1 metal=True, tilelang built+loadable from
`/Volumes/external/sources/tilelang/build`). This work ran ENTIRELY LOCALLY on
the Apple GPU — no gb10, no extrapolation for Track A timings.

Environment of record: `/Volumes/external/sources/cppmega.mlx/.venv/bin/python`
(CPython 3.13.12), z3 4.15.4, tilelang `0.1.9+git5b30bb10`.

---

## Track A — Metal-GEMM rewrite of F0/B2/B0 Metal prims (hand A)

Flag: `CPPMEGA_PATH_C_METAL_GEMM`. The serial Metal prim is kept BYTE-IDENTICAL
as the parity reference (flag OFF). RULE #1: when the flag is ON, GEMM is the ONE
path; a tile/divisibility/compile failure RAISES (no silent serial fallback).

### F0 (`chunk_precompute_fwd`) — FULLY GEMM-ified, MEASURED on the Apple GPU

The two head-independent serial reductions become `T.gemm`:
1. `cb = C @ B^T` (transpose_B)
2. `summary_states = (decay·dt-weighted x)^T @ B` (transpose_A)

MEASURED (`scratch/probe_f0b2b0_metal_gemm_parity.py`, dims B=1 S=128 chunk=64
G=1 H=2 P=N=64, fp16 inputs, mps device, 50-iter timing, gate 5e-4):

| output           | max\|abs diff\| serial vs gemm | nan |
|------------------|-------------------------------:|-----|
| cb               | 0.000e+00 (bit-exact)          | no  |
| dA_cumsum        | 0.000e+00 (bit-exact)          | no  |
| summary_states   | 2.128e-06                      | no  |

- serial = **0.1574 ms**, gemm = **0.0192 ms** → **8.20x faster**, worst diff
  2.128e-06 ≪ 5e-4 gate. Parity checked over EVERY output element (no subset).

### Codegen is REAL tensor-op matmul, not serial relabeled

`get_kernel_source()` on the F0 GEMM prim (flag ON):
- `simdgroup_multiply_accumulate(...)` × **2** — the two T.gemm calls:
  `simdgroup_multiply_accumulate(cb_frag[...], A_local[...], B_local[...], cb_frag[...])`
  and `simdgroup_multiply_accumulate(states_frag[...], ...)`.
- `make_filled_simdgroup` × 128 (the register fragment accumulators),
  `simdgroup` token × 206.

The SERIAL F0 prim (flag OFF): `simdgroup_multiply` × **0**,
`make_filled_simdgroup` × **0** (its 34 `simdgroup` tokens are `simdgroup_barrier`
sync only). So the GEMM prim emits Apple's `metal.simdgroup` cooperative tensor
matmul (the matmul2d lowering target on M1-M4 where the register-fragment route
is forced), and the serial prim does scalar loops. The two are genuinely
different code, not a relabel.

> Metal vs the CUDA twin: C accumulators are register FRAGMENTS, which forces the
> stable 8x8 `metal.simdgroup` path (shared-C with N≥32/K%16==0 would route to
> the M5-only `metal.cooperative_tensor` path that fails on M1-M4). fp16 operands
> / fp32 accum; summary_states stored fp32; dacs reloaded from fp16 dA_cumsum to
> fit Apple's hard 32 KB threadgroup limit.

### B2 / B0 — honest RAISE (RULE #1), serial stays the working path

With the flag ON, `build_chunk_scan_combine_bwd_metal` (B2) and
`build_chunk_precompute_bwd_metal` (B0) RAISE a clear `NotImplementedError`
(verified by the probe). With the flag OFF, both serial prims compile to a
`JITKernel` and run. Reasons (honest scope, not a bug):
- B2's 4-contraction fragment-staging set exceeds Apple's 32 KB threadgroup
  limit (the gb10 CUDA twin needs 72.5 KB).
- B0 is a reverse-cumsum scan + per-l atomic scatters, not a 2D matmul.

The `chunk_scan_combine_bwd_metal_gemm_prim` (DYX + dchunk_states, two clean
GEMMs) is defined for reference / larger-budget backends.

**Track A GO/NO-GO: GO for F0** (8.20x, bit-exact cb + 2.1e-6 summary, real
simdgroup matmul). **NO-GO (honest) for B2/B0 on Apple** — over Apple's 32 KB
threadgroup budget / not a matmul; the serial prim is the path and the GEMM flag
RAISES rather than launching wrong.

---

## Track B — automatic compiler pass `AutoGemmifyReductions`

`/Volumes/external/sources/tilelang/tilelang/transform/auto_gemmify_reductions.py`
— a Python-registered `tvm` `prim_func_pass` (NO C++ rebuild; the live
`build/` is untouched). Default OFF; opt-in via env
`TILELANG_ENABLE_AUTO_GEMMIFY` or PassConfig key `tl.auto_gemmify_reductions`.

### What it DOES auto-detect + auto-prove (no hand rewrite)

Given a raw `@T.prim_func` written in the canonical serial-reduction shape
```
for ls in range(M*N):
    i = ls // N ; j = ls % N
    acc = local[1]; acc[0] = 0
    for k in range(K):
        acc[0] = acc[0] + A[..i..k..] * B[..k..j..]
    out[..i..j..] = acc[0]
```
the pass, WITHOUT any annotation:
1. structurally recognizes the contraction,
2. INFERS transpose flags from the actual TIR index positions
   (cb → transpose_A=False/transpose_B=True; summary → transpose_A=True), and
3. runs the built-in z3 prover (Track C machinery) which proves algebraic
   equivalence + race-freedom NON-vacuously, then stamps `tl.auto_gemmify_proved`.

Demonstrated end-to-end on the F0 `cb` serial kernel (`match_contraction` +
`prove_contraction` + `rewrite_contractions`):
`MATCH M,N,K=32x32x16 transpose_A=False transpose_B=True; z3 algebra_proved=True
race_proved=True PROVED=True`.

Test suite: **14/14 PASS**
(`testing/python/transform/test_auto_gemmify_reductions.py`) — including the
correct cb/summary/scaled matches, and the DECLINE cases (k-dependent scale,
3-operand product, plain 1D reduction, wrong-transpose z3 counterexample,
z3-disabled env).

### The honest limit (prototype scope)

`_emit_gemm_for_match` returns `None` — the pass does NOT splice a `T.gemm`
tile-op into the raw TIR in-place. Synthesizing the exact shared-buffer staging
+ `tl.region` operands that `LayoutInference` + the Metal `gemm.cc` selector
demand from raw TIR is the brittle part flagged as risk #1 in the design; rather
than fabricate shared allocations that would make `LowerTileOp` raise, the
in-place splice is DEFERRED to the frontend builder path (the hand A F0 prim).
So on the test kernel: `rewritten=0`, `decline_reason=gemm_splice_deferred_to_builder`,
`gemm calls after=0`. The detector + z3 prover (the load-bearing, demonstrable
machinery) run and prove regardless.

**Track B GO/NO-GO: GO as a detector+prover prototype** — it AUTO-detects the
serial-reduction-is-a-GEMM pattern from raw TIR and AUTO-proves the rewrite,
with transpose inference, with no hand annotation. **NO-GO for fully-automatic
in-place T.gemm emission** — that final IR splice is honestly deferred to the
builder, so the auto-pass recognizes+proves the rewrite but does not yet emit it
itself. This is a working PROTOTYPE, scoped honestly, not a production pass.

---

## Track C — built-in z3 proof of the rewrite

`/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/_gemm_rewrite_proof.py`
— mirrors `MetalReductionSyncPlan` / `mamba3_path_c` provers. Each obligation
checks the NEGATION of a property for `unsat` (proved) / `sat` (concrete
counter-witness) / `unknown` (fail-closed). `z3_proved` is True iff EVERY
applicable obligation discharges to `unsat`.

Obligations: operand-map match (no accidental transpose), tiling injectivity
over k, k-coverage/surjectivity, mask-predicate equivalence, scale-fold
equivalence (uninterpreted base fn), single-writer output disjointness.

This is the built-in gate for Track B: the auto-pass calls
`prove_contraction` / `require_gemm_rewrite_proof`; an unprovable contraction
RAISES `GemmRewriteNotProven` (forced mode) or keeps the serial path
(detect-and-prefer mode). NO rewrite without a passing z3 proof (RULE #1).

### Proof results — MEASURED (`tests/test_gemm_rewrite_proof.py`, 13/13 PASS)

CORRECT rewrites PROVE:
- F0 `cb=C@B^T` → z3_proved=True (operand maps match, tiling injective, k
  covered, single writer).
- B2 `dinp` lower-tri causal mask → z3_proved=True (mask_equiv=True).
- F0 `summary` k-independent scale fold → z3_proved=True (scale_equiv=True).

WRONG rewrites are REFUSED with a concrete COUNTER-WITNESS (the NON-VACUITY
proof — a tautological encoding would pass these too):
- transposed A operand (`k*M+i` vs `i*K+k`) → operand_maps_match=False.
- dropped last k-tile (k_steps=3 covers [0,48) of 64) → k_covered=False.
- overlapping output tiles (tile_m=20, stride=16 → rows 16..19 double-owned) →
  single_writer=False.
- strict `i>k` vs inclusive `i>=k` causal mask (drops the diagonal) →
  mask_equiv=False.
- `scale(k+1)` vs `scale(k)` off-by-one decay fold → scale_equiv=False.

Track B's own z3 (`prove_contraction`) is also non-vacuous: flipping a transpose
flag yields a z3 algebra counterexample (`z3_algebra_proved=False`,
`decline_reason=z3_algebra_counterexample`) even on a square M==K==N kernel, and
disabling z3 (`TILELANG_DISABLE_Z3`) makes every candidate DECLINE
(`decline_reason=z3_disabled`) — it never rewrites without a proof.

**Track C GO/NO-GO: GO** — the proof runs, proves correct rewrites, and FAILS on
deliberately-broken rewrites with a counter-witness. It proves SYMBOLIC/
structural equivalence + race-freedom (NOT fp bit-equivalence — that is the
numeric parity test in Track A). Non-vacuity is locked in by the wrong-rewrite
tests.

---

## Honest attribution / scope summary

| Track | Claim | Status |
|-------|-------|--------|
| A F0  | serial→T.gemm(simdgroup), measured on Apple GPU | **REAL** — 8.20x, bit-exact cb, 2.1e-6 summary, real `simdgroup_multiply_accumulate` codegen |
| A B2/B0 | GEMM on Apple | honest **RAISE** (over 32 KB threadgroup / not a matmul); serial is the path |
| B     | auto-detect + auto-prove serial-reduction contraction from raw TIR | **REAL** prototype — detects + infers transpose + z3-proves, 14/14 tests |
| B     | auto in-place T.gemm emission | **DEFERRED** — `_emit_gemm_for_match` returns None; splice deferred to builder |
| C     | z3 equivalence + race-free proof, non-vacuous, built into the pass | **REAL** — 13/13 tests, wrong rewrites refused with counter-witness |

All Track A timings are MEASURED on this Apple GPU (no extrapolation). Track B/C
are tested on this machine with the live tilelang build + z3 4.15.4. No C++
rebuild; the live `/Volumes/external/sources/tilelang/build` is untouched.
