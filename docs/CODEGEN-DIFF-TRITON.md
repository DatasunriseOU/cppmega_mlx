# CODEGEN-DIFF-TRITON — line-level diff of `_chunk_scan_bwd_dstates` (ours vs native Triton)

Verifiable, MEASURED diff (gb10, sm_121a, CUDA 13.2). All counts cite the exact
`.ttir`/`.cu`/`.sass` artifacts; every ms is strip-and-timed (CUDA-event, median-of-3,
parity-gated MAXDIFF 4.88e-04), not estimated. MEASURED vs EXTRAPOLATION called out per line.

## Artifacts (all on gb10, regen at tilelang HEAD `87dcce72`, `TL_FORCE_CP_ASYNC=1`)
- OURS routed CUDA-C: `/tmp/ttir7/dstates_OPT.cu` (336 lines, §P1 cp.async), `/tmp/ttir7/dstates_OFF.cu` (serial)
- OURS routed SASS:   `/tmp/em_sass_OPT.sass` (cp.async, canonical executed), `/tmp/ttir7/dstates_OFF.sass`
- OURS input TTIR:    `/tmp/ttir7/_chunk_scan_bwd_dstates.ttir` (895 lines)
- NATIVE §P1 cubin:   `~/.triton/cache/7J3654N5QPZLTONFWHMU6ZNERPPDBJUHP5FWSUOFEUFLUGSSGOVA/` (num_warps=8, num_stages=3, shared=197120)
- NATIVE SASS:        `/tmp/ttir7/native_P1.sass` ; NATIVE PTX/TTGIR/TTIR in same cache dir
- Build+time harness: `poc/triton_frontend/_test_harness/tridao_parity/em_dstates_cpasync.py`; strips `/tmp/ttir7/three_point.py`

## Top-line MEASURED ms (interleaved CUDA-event, N=60×4 reps, median-of-medians)
| kernel                                   | ms       | HMMA | LDGSTS         | LDG | spill (STL+LDL) |
|------------------------------------------|----------|------|----------------|-----|-----------------|
| OURS OFF (serial prologue, plain LDG)    | 3135.2   | 32   | 0              | 128 | 2494            |
| OURS OPT (prologue folded, cp.async)     | 742.4    | 32   | 16 (.E 32-bit) | 96  | 272             |
| NATIVE §P1                               | **1.2008** | 256 | 75 (.128 BYPASS) | 0 | **0**           |

GAP_TO_NATIVE = **618× (OPT/NATIVE)**. SPEEDUP OFF→OPT = 4.22×.

## ⚠ CRITICAL HONEST CORRECTION (proven from the diff, not asserted)
The two routes do **NOT** consume the same TTIR for §P1. VERIFIED:
- OURS `tt.dot`: `tensor<64x32xf32> * tensor<32x64xf32> -> tensor<64x64xf32>` (make_range ends 64,64,32) → **BLOCK_M=64, BLOCK_N=64, BLOCK_K=32**
- NATIVE `tt.dot`: `tensor<128x64xf32> * tensor<64x256xf32> -> tensor<128x256xf32>` (ends 128,256,64) → **BLOCK_M=128, BLOCK_N=256, BLOCK_K=64**

So the 32-vs-256 HMMA gap is the input TTIR's tile config (8× = (128·256)/(64·64)), NOT something our
codegen chose. Both emit the IDENTICAL MMA shape `HMMA.1688.F32.TF32` / `tl::mma_sync<...,16,8,8>`.
**The HMMA gap does NOT explain the 600×: native does 8× MORE MMA work (256 vs 32) yet is 618× FASTER.**
Native's 128×256 tile even over-covers the hd=64,ds=64 problem (masked); the speed driver is the
prologue + loads + spills + pipelining, NOT the GEMM.

---

## DIFF 1 — SERIAL SCALAR PROLOGUE + LOCAL SPILLS  ·  MEASURED 2041.8 ms (the dominant cost)
**Evidence (`dstates_OPT.cu`):** 64 serial `for` loops (lines 71–150, 241–333), **zero** thread-distributed
(`grep -c 'for.*threadIdx' = 0`). Every one of the 128 threads runs all 64 loops identically. They
materialize Triton's `tt.make_range`/`tt.broadcast`/mask address tensors as dense per-thread local arrays:
- `tile_binop_7/12/17/22[64]` — `offs_m`/`offs_n` aranges, computed BOTH `int64_t` AND `int` (lines 71–82) = the i32→i64 overflow-guard duplication
- `bcast_54/58[2048]`, `tile_binop_64[2048]` — 64×32 broadcast of A-addresses (lines 98–112); `bcast_103/107/113[2048]` for B; `bcast_236/240/246/252[4096]` + `bool bcast_269/273/277[4096]` in the epilogue
- Total **~155 KB/thread local arrays** (70.4 KB int64 + 84.5 KB int) → register file overflow → spill.

**SASS (`/tmp/em_sass_OPT.sass`):** `STL=136 LDL=136` (spill, OFF=2494), `IMAD=803 ISETP=796 MOV=449 IABS=209 I2F=104 F2I=104 BRA=310` vs NATIVE `IMAD=298 ISETP=69 MOV=112 IABS=5 I2F=1 F2I=1 BRA=3`. The `IABS`/`I2F`/`F2I` come from the signed-div + i32→i64 overflow-assert chain.

**Input-TTIR cause:** OURS `_chunk_scan_bwd_dstates.ttir` (895 lines): `arith.extsi=102`, `2147483647=66`, `arith.cmpi=109`. NATIVE `.ttir` (299 lines): `arith.extsi=0`, `2147483647=0`, `arith.cmpi=7`. The native pre-opt `.source` HAS the same `cmpi sle 2147483647`/`extsi` chain (lines 223–240) — Triton's own MLIR canonicalize/CSE/int-range pipeline folds it away; the TTIR our route receives is a LESS-OPTIMIZED snapshot that still carries it.

**STRIP-AND-TIME (MEASURED, parity PASS both):** `prologue_opt=False` (A_OFF) 3144.5 ms → `prologue_opt=True` (B_SIMT, plain LDG) 1102.7 ms ⇒ **A−B = 2041.8 ms** attributable to the serial prologue + spill removal (65% of OFF).

---

## DIFF 2 — GEMM TILE  ·  32 vs 256 HMMA  ·  tile-size only (NOT the bottleneck)
**Evidence:** OURS accumulator `dot_c_frag_176[32]` (32 floats/thread × 128 = 4096 = 64×64 output). MMA loop (`dstates_OPT.cu` 218–233): `ki∈[0,4) × i_5∈[0,2) × j_1∈[0,2) × 2 = 32` `tl::mma_sync<kTensorFloat32,...,16,8,8>` HMMA. NATIVE: accumulator `tensor<128x256xf32>` → 256 `HMMA.1688.F32.TF32`.
- Per-block output tile: OURS **64×64**, NATIVE **128×256** (8× = HMMA ratio).
- SAME MMA instruction shape (m16n8k8 TF32) in both — confirmed `HMMA.1688.F32.TF32` in both SASS.

**Cause:** the BLOCK dims are baked into the input TTIR (DIFF preamble). NOT a tilelang/tvm retile —
`from_ttir` cannot change them; feeding native's 128×256 TTIR through our route fails (`unknown dtype 'i32 {tt.divisibility=16}'`), proving the two TTIRs are distinct Triton lowering artifacts.

**STRIP-AND-TIME:** cannot strip the in-kernel GEMM via source-doctoring (tilelang compiles from the TIR module; the `get_kernel_source` monkeypatch is read-back only and does not reach nvcc — verified: NOGEMM SASS still HMMA=32). A standalone-matmul proxy would be EXTRAPOLATION, so no fabricated number is given. Bound: the GEMM is inside the 740.4 ms residual (DIFF 3/4). Since native does 8× more HMMA and is 618× faster, the GEMM tile is NOT the bottleneck — flagged EXTRAPOLATION-only, do not rank it high.

---

## DIFF 3 — LDGSTS COALESCING/VECTORIZATION  ·  16×32-bit vs 75×128-bit  ·  MEASURED 362.3 ms
**Evidence (SASS widths):**
- NATIVE: `LDGSTS.E.BYPASS.128 ×72` + `LDGSTS.E ×3` = 75. PTX: `cp.async.cg.shared.global ×72` (16-byte, L2-bypass) + `cp.async.ca ×3`, `cp.async.commit ×9`, `cp.async.wait ×2`.
- OURS OPT: `LDGSTS.E ×16` (32-bit = `tl::cp_async_gs<4>`, 1 float, **4× narrower**) + `LDG.E.CONSTANT ×96` (plain synchronous scalar).

**Operand-level (`dstates_OPT.cu`):** ONLY operand C (`arg1`, line 188) uses cp.async (→16 LDGSTS). Operand dout (`arg0`, lines 152–161) loads via a **64×32 = 2048-iter scalar loop run identically by all 128 threads** (no `threadIdx` in the index) and operand dA (`arg3`, 163–170) similarly → the 96 redundant `LDG.E.CONSTANT`. Native moves ALL operands via thread-distributed `LDGSTS.128`.

**Pipelining:** OURS `cp_async_commit(); cp_async_wait<0>()` back-to-back (lines 190–191) = **depth-1, no overlap**, 2 LDGDEPBAR. NATIVE = 9 LDGDEPBAR + 9 commit-groups (num_stages=3 deep pipeline).

**STRIP-AND-TIME (MEASURED, parity PASS):** B_SIMT (plain coalesced LDG) 1102.7 ms → C_CPA (cp.async overlap) 740.4 ms ⇒ **B−C = 362.3 ms** for the cp.async/load-overlap delta. (The remaining redundant-load + narrow-width cost is inside the 740.4 ms residual.)

---

## DIFF 4 — REDUNDANT PER-THREAD WORK + residual spill  ·  inside the 740.4 ms residual
**Evidence:** dout/dA loads (DIFF 3), the masked-OOB epilogue (`dstates_OPT.cu` 193–200: 32×64 loop all threads), the `__expf` + broadcast (172–179), and the carry add (241–244) are all full-tile scalar loops with NO threadIdx distribution → 128× redundant. SASS still carries 272 spills + `I2F/F2I=104` overflow-math in C_CPA. Native uses uniform-datapath address math (`UF2I/UI2F/ULOP3/USEL/R2UR` — present in native SASS, ABSENT in ours) so address arithmetic is computed once per warp, not per thread.

**Bound:** residual after DIFF 1+3 stripped = C_CPA = 740.4 ms = 617× native. This residual is the
redundant-per-thread loads/epilogue + narrow LDGSTS + 272 spills + depth-1 pipeline + (some) under-tiled GEMM.
No clean single-knob strip isolates each sub-component (they share the same redundant-loop emission), so
per-component ms inside the residual is EXTRAPOLATION — reported honestly as a bound, not fabricated.

---

## OPS PRESENT IN ONE, ABSENT IN OTHER (verified by per-mnemonic count)
- OURS only: `LDG` (96 — native=0, all-async), `VIMNMX`.
- NATIVE only: `UF2I/UI2F/ULOP3/USEL/R2UR/UVIMNMX` (uniform-datapath address math), `CS2R` (×128, vectorized accumulator zero-init — ours uses scalar loops), `SGXT`, `PLOP3`.

---

## RANKED ACTIONABLE TO-DO (by MEASURED ms; converge our codegen to Triton-equivalent)

1. **[2041.8 ms — MEASURED] Fold the i32→i64 overflow-guard + materialized address/broadcast tensors.**
   Cause: input TTIR carries the un-canonicalized `arith.extsi`/`cmpi sle 2147483647` chain (extsi=102 vs native 0), and our emitters materialize `tt.make_range`/`tt.broadcast` as dense per-thread `int64_t[2048]/[4096]` local arrays in 64 non-distributed serial loops (`op_emitters/*` tile_binop/bcast emission).
   Fix (two independent levers, both verified-needed): (a) run Triton's int-range/canonicalize/CSE on the TTIR BEFORE our route consumes it (or strip the overflow-guard given 32-bit-safe strides) so `extsi/cmpi/2147483647` collapse → kills IABS=209, I2F/F2I=104, most ISETP=796; (b) make the address/broadcast emission lazy/fused into the load addressing (per-lane scalar, thread-distributed) instead of materializing `bcast_*[2048]/[4096]` arrays → kills the 272→0 spill and the 64 redundant loops. `prologue_opt=True` already captures part of this (2041.8 ms recovered); the residual `int64_t[*]` arrays + 272 spills remain.

2. **[≤362.3 ms cp.async-overlap MEASURED + redundant-load residual] Distribute & widen ALL operand loads; deepen the pipeline.**
   Cause: only operand C is cp.async (and at 32-bit); dout/dA load via 64×32 scalar loops run by all 128 threads (96 LDG); `cp_async_commit; wait<0>` is depth-1.
   Fix: route dout(arg0) and dA(arg3) through the same coalesced async CopyNode as C (memory.py copy emitter), vectorize `cp_async_gs<4>`→`<16>` for `LDGSTS.128`, and emit a num_stages≥3 software pipeline (LowerHopper/pipeline pass) so commit/wait overlap. Target: 75× `LDGSTS.128`, 0 plain LDG, 9 commit-groups — matching native.

3. **[EXTRAPOLATION — not the bottleneck] GEMM tile 64×64×32 → 128×256×64.**
   Cause: input TTIR is traced at the small tile. Fix: trace/route the §P1-autotuned 128×256×64 config (or retile in `from_ttir`). Rank LOW: native does 8× more HMMA and is still 618× faster — fixing the tile alone will not close the gap; do this only after #1 and #2.

## METHOD NOTE (RULE #1)
MEASURED = strip-and-time deltas via the route's real `prologue_opt`/`async_loads` knobs, parity-gated.
EXTRAPOLATION = GEMM-tile ms (source-doctoring strip does not reach nvcc through this adapter; standalone-matmul proxy rejected as not a strip of the actual kernel) and per-sub-component split of the 740.4 ms residual. Both flagged explicitly above; no estimated number is presented as measured.
