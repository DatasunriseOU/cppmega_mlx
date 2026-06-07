# Triton-frontend iteration 4 — real 2D-strided `T.copy` CopyNode for the K-loop producer (TMA / UTMALDG)

Status: **HONEST PARTIAL — mechanism PROVEN, default §P1 path NO-GO on UTMALDG.**
The iteration-4 CopyNode conversion is committed on the routed-triton async path.
The CopyNode IS recognized by `PipelinePlanning` as a proven copy and, when the
innermost global stride is provably contiguous, the C-tile lowers to genuine
**UTMALDG** (TMA bulk async load) on sm_121 — proven by SASS from HEAD. BUT on the
real §P1 de-monomorphized kernel **neither tile is statically TMA-eligible**
(dout's innermost axis is the K-reduction/seq axis, stride 7168, genuinely
non-contiguous; C's innermost is runtime-contiguous but passed as an opaque
symbolic arg so `is_one()` cannot prove it). Both therefore correctly fall through
to the existing predicated-`For` path — the generated `.cu` is **byte-identical**
to iteration-3, UTMALDG stays 0, and there is **no ms drop** from the 1102 ms
iteration-2/-3 baseline.

All numbers below are MEASURED-FROM-COMMITTED-HEAD on gb10 (sm_121a), regenerating
the `_chunk_scan_bwd_dstates` artifacts from HEAD, interleaved OFF/OPT, parity
re-verified under the GPU mutex.

- tilelang committed HEAD: `1378c1737b0dda677a24b3808801ed24745b610a`
  (`merge/upstream-codegen-reorg`). gb10 working tree byte-identical to HEAD for
  all 4 routed-frontend files (md5-verified: `memory.py 5d6f289d…`,
  `control.py 5c7f9cb9…`, `__init__.py 31ebe2f4…`, `op_mapping.py 3648f32f…`).

## What landed (committed, iteration 4)

`poc/triton_frontend/op_emitters/memory.py`:

- `_ptrstate_block_base_and_strides` — splits the collapsed PtrState flat address
  into a per-block `elem_offset` (loop-independent offsets + carry advance) and
  per-axis strides, preserving the 2D axis-contiguity that the 1D flat-index
  lowering had discarded.
- `_emit_ptrstate_tile_load_copynode` — declares the global source (dout/C) as a
  2D `tir.Buffer` aliasing the flat arg `data` Var with `strides=[stride0,stride1]`
  + `elem_offset=base`, emits a REAL `T.copy(src2d -> shared_tile)` CopyNode, and
  SPLITS the bounds mask OFF into a separate masked epilogue (not inside the
  `if_then_else` producer).
- A **TMA-eligibility gate (RULE #1)**: the CUDA bulk-copy lowering hard-requires
  a statically-provable contiguous innermost global stride
  (`ICHECK(is_one(desc.global_stride[0]))` in `src/backend/cuda/op/copy.cc`). If
  the innermost stride is provably 1 (or `routed_contiguous_innermost` supplies
  ground-truth contiguity) the CopyNode is emitted; otherwise the function
  RETURNS FALSE and the caller keeps the load on the predicated-`For` path
  WITHOUT stamping `num_stages` — no predicated-For dressed up as a coalesced copy.

Wired into `_emit_ptrstate_tile_load_tir`, gated on
`routed_triton_async_loads and rank>=2`. The iteration-3 infra
(`async_loads` gate, `_gmem_shared_copies` counter, `map_scf_for._maybe_pipeline`
`num_stages` stamp) is the committed prerequisite.

## Measured — default §P1 path (gb10, `_chunk_scan_bwd_dstates`)

§P1 config: b=1 nh=112 hd=64 ds=64 nc=64 cs=64, BM/BN/BK = 64/64/32.

| metric | async OFF | async ON (HEAD default) | iter-3 baseline | Triton ref |
|---|---|---|---|---|
| generated `.cu` md5 | `99c232418b10c2793928ed63776b76c2` | `99c232418b10c2793928ed63776b76c2` | `99c23241…` | — |
| `.cu` byte-identical OFF==ON | — | **True** | True | — |
| SASS UTMALDG | 0 | **0** | 0 | — |
| SASS LDGSTS | 0 | 0 | 0 | (Ampere-only) |
| SASS LDG | 38 | 38 | 38 | — |
| SASS spill (STL+LDL) | 306 (168+138) | 306 (168+138) | 306 | 0 |
| SASS HMMA | 32 | 32 | 32 | 256 |
| §P1 parity MAXDIFF | — | **4.882812e-04** | 4.88e-04 | — |
| ALLCLOSE 1e-3 | — | **PASS** | PASS | — |
| EXEC ms/kernel (CUDA events, N=50, ×4 interleaved) | median **1101.79** | median **1101.41** | ~1102 | ~1.12 |
| Δ ms (OFF−OPT) | — | **0.38** (noise) | 0 | — |

OFF per-rep: 1101.59 / 1101.99 / 1102.38 / 1101.25.
OPT per-rep: 1101.19 / 1101.35 / 1102.03 / 1101.47.
The `.cu` is byte-identical OFF==ON → the 0.38 ms is run-to-run noise. **No real
drop from 1102.** Small real-strided multi-K-trip parity (parity_all7 `small`:
b1 nh8 hd64 ds64 nc4 cs64 ng2 s256, 2 K-trips, real strides): MAXDIFF=**3.051758e-05**,
ALLCLOSE 1e-3 PASS — no regression from baseline.

## Mechanism PROOF — CopyNode → UTMALDG fires when the innermost is contiguous

Reproducible from HEAD by supplying ground-truth contiguity
(`ctx.routed_contiguous_innermost=True`, the flag the route would set once it has
verified the real tensor's innermost stride == 1) and dropping `disable_tma`. The
C-tile (`[k, ds]`, innermost = ds) then pins a literal-1 innermost and the CopyNode
takes the TMA bulk path:

| metric | default §P1 | ground-truth-contiguous C-tile |
|---|---|---|
| `.cu` `CUtensorMap` | 0 | **2** |
| `.cu` `mbarrier` | 0 | **22** |
| `.cu` `swizzle` | 0 | **1** |
| SASS UTMALDG | 0 | **6** |
| SASS UTMACCTL | 0 | **1** |
| SASS LDG | 38 | **32** (the 6 C-tile loads moved to TMA) |
| SASS HMMA | 32 | 32 (intact) |

This proves the CopyNode→`cuda::Copy::LowerBulk`→UTMALDG path is real and live on
sm_121 — the missing piece was a statically-provable contiguous innermost, not the
pipeline plumbing.

## RULE #1 — honest TMA-eligibility of the §P1 tiles

- **dout tile** is `[hd, k]` → innermost = the **K-reduction (seq) axis**, runtime
  stride `nheads*headdim = 112*64 = 7168`. **Genuinely non-contiguous → TMA-INELIGIBLE.**
  TMA needs the innermost axis contiguous; dout does not qualify. Correct
  fall-through to predicated-`For` (NOT counted as an async copy).
- **C tile** is `[k, ds]` → innermost = ds, runtime stride 1 (contiguous), BUT the
  de-monomorphized kernel passes it as opaque `arg19`, so `is_one()` cannot prove
  `== 1` at compile time → TMA-INELIGIBLE on the default path. The mechanism proof
  above shows it DOES take UTMALDG once contiguity is ground-truthed.

No silent fallback was introduced: when a tile is not provably TMA-eligible the
CopyNode emitter RETURNS FALSE and the load stays on the correct predicated-`For`
path, and the K-loop is NOT mis-annotated with `num_stages`. Parity holds at §P1
(MAXDIFF 4.88e-04, unchanged) and small multi-K-trip (3.05e-05). The GEMM
cooperative path is byte-intact.

## Other paths (no-regression)

`route_all7`: 5/7 route with mma=2 intact (dstates / dc / state_db / state_dx /
ddAcs). The 2 fails (`dcb`: PtrState unresolved SSA `%1044`; `dx`: undefined
`arg4_numel_147`) are PRE-EXISTING and IDENTICAL to baseline — not a regression.
GEMM HMMA=32 intact in the routed kernels.

## GO / NO-GO

- **GO** on the CopyNode mechanism: the real 2D-strided `T.copy` is recognized as a
  proven copy and lowers to genuine UTMALDG (SASS UTMALDG=6, LDG 38→32) when the
  innermost is contiguous — proven from HEAD.
- **NO-GO** on a §P1 ms drop this iteration: both §P1 tiles are statically
  TMA-ineligible (dout genuinely non-contiguous innermost; C contiguous but
  opaque-symbolic), so the default path correctly falls through, `.cu` is
  byte-identical, UTMALDG=0, EXEC stays 1101 ms.

Remaining gap to native ~1.12 ms is ~983x, now precisely localized to TWO
prerequisites the frontend cannot satisfy alone: (1) the de-monomorphized launch
must surface a literal-1 innermost stride for C (monomorphization / a
route-supplied contiguity contract), and (2) dout needs a transposed tile so its
innermost becomes the contiguous headdim axis rather than the seq axis. With both,
C+dout take UTMALDG; HMMA (32 vs Triton 256) and spill (306) remain the next
fronts after the loads coalesce.
