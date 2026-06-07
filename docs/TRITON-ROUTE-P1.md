# TRITON-ROUTE-P1 — Routed Tri-Dao bwd: parity + exec-ms + prologue breakdown

MEASURED from committed HEAD on gb10 (NVIDIA GB10 sm_121, aarch64-linux).
- gb10 tilelang HEAD: `db5fe57c117460ce84d72cd7b07f153f51c4ef0a` (de-monomorph: symbolic flat-arg buffer extents)
- local mirror HEAD: `48b9236a86ef0ebd0b22ec3625b374b7d2fbf874` (same logical fix, separate repo)
- Working tree clean (only `3rdparty/tvm` submodule pointer = built state). No uncommitted source.
- All 7 PrimFunc JSONs regenerated FRESH from HEAD via `emit_pf_json.py` before measuring.

RULE#1: real-strided / §P1 numbers ONLY. No degenerate (single-K-trip / cs==BK) or
tiny-config number reported as a real-strided/§P1 result. Routed exec ms reported ONLY
for the parity-PASS kernel. Every failing kernel reports its EXACT defect; no fabricated PASS.

---

## 1. Per-kernel parity — REAL strided multi-K-trip (cs=64 != BLOCK_K=32, 2 K-trips/chunk)

### 1a. SMALL real-strided config (b1 nh8 hd64 ds64 nc4 cs64 ng2 s256, grid (1,4,8))

| Kernel                         | Status @ HEAD          | MAXDIFF      | ALLCLOSE 1e-3 | Notes |
|--------------------------------|------------------------|--------------|---------------|-------|
| `_chunk_scan_bwd_dstates`      | **PASS**               | 3.051758e-05 | True          | nz 131072/131072 full; cs=64,BK=32 = 2 K-trips |
| `_chunk_scan_bwd_dc`           | SEGFAULT (routed launch) | n/a        | —             | dumps core at kernel launch |
| `_chunk_scan_bwd_dcb`          | NO-ROUTE / compile fail | n/a         | —             | emit RAISES `PtrState references unresolved SSA value '%1044'`; compiled variant fails nvcc (`mma.h` error limit) |
| `_chunk_scan_bwd_dx`           | NO-LAUNCH (de-mono incomplete) | n/a  | —             | `tvm InternalError: arg4_numel_243 used but not passed as API argument` |
| `_chunk_state_bwd_db`          | SEGFAULT (routed launch) | n/a        | —             | dumps core at kernel launch |
| `_chunk_state_bwd_dx`          | SEGFAULT (routed launch) | n/a        | —             | dumps core at kernel launch |
| `_chunk_state_bwd_ddAcs_stable`| NO-COMPARE             | n/a          | —             | atomic/return-only output; harness pin_kernel also KeyError BLOCK_SIZE_M |

**SMALL real-strided result: 1/7 numerically correct (dstates).**

### 1b. PRODUCTION §P1 config (b1 nh112 hd64 ds64 nc64 cs64 ng8 s4096, grid (1,64,112), numel 29,360,128)

Only `_chunk_scan_bwd_dstates` routes+launches; only it has a §P1 parity number:

| Kernel                    | Status | MAXDIFF      | ALLCLOSE 1e-3 | nonzero        |
|---------------------------|--------|--------------|---------------|----------------|
| `_chunk_scan_bwd_dstates` | **PASS** | 4.882812e-04 | True        | 29360128/29360128 (native 29360127) |

The SAME compiled kernel that passed at small (1,4,8) launches+passes at §P1 (1,64,112) with no
segfault — the de-monomorph (symbolic flat-arg extents) fix is CONFIRMED for dstates: numel grew
from ~131K to 29.36M with the same baked PrimFunc.

**§P1 prod result: 1/7 numerically correct (dstates).**

---

## 2. Routed EXEC ms/kernel @ §P1 (parity-PASS kernel only)

Authoritative on-device kernel time, 3 independent methods, all agree:

| Method                              | dstates §P1 routed ms/kernel |
|-------------------------------------|------------------------------|
| nsys `cuda_gpu_kern_sum` (1 launch) | **1473.43 ms** (avg of 4 kern instances 1465–1478ms) |
| CUDA events (N=100)                 | 1470.52 ms                   |
| Python wall (N=50)                  | 1476.32 / 1472.43 ms         |

Device time ≈ wall time → the cost is **genuine on-GPU kernel runtime**, NOT host/launch overhead.

**vs native: native `_chunk_scan_bwd_dstates` @ §P1 = 1.14 ms/kernel.**
Routed = ~1473 ms → **≈1292x SLOWER than native.**

---

## 3. Prologue vs GEMM breakdown — strip-and-time (MEASURED)

The routed kernel body is: (a) a ~133-`for`-loop **prologue** that materializes the Triton
pointer arithmetic (`tl.arange`, broadcasts, int32-overflow guards) as serial per-thread
element-wise loops over `bool[4096]` / `int[4096]` arrays, followed by (b) a tiny **GEMM**
(2× `mma.sync` in a 4-iter K-loop with `ldmatrix`), then (c) carry/epilogue.

Strip-and-time (inject `return;` immediately before the GEMM block, recompile through the
identical TileLang adapter path, same §P1 args, CUDA-event timed):

| Variant                          | DEVICE ms/kernel @ §P1 |
|----------------------------------|------------------------|
| FULL kernel                      | 1473.29 ms             |
| PROLOGUE-ONLY (return pre-GEMM)  | 1474.99 ms             |
| ⇒ GEMM + epilogue contribution   | ~0 ms (within noise)   |

**~100% of the 1473ms is the prologue. The actual GEMM is negligible.**

### Is this inherent to the per-block model?
YES — as currently lowered. The routed PrimFunc emits each Triton pointer-offset computation
as a literal element-wise loop (e.g. `tile_binop[i] = (val <= INT_MAX) & (INT_MIN <= val)`)
over 4096-element arrays, run **serially by every one of the 128 threads in every one of the
7168 blocks**, instead of folding the offsets into the load addressing the way native Triton
does. Roughly ~20 such `[4096]` loops × 128 threads × 7168 blocks of redundant scalar work
dwarf the 2-instruction GEMM. This is a structural property of the current whole-block
walker lowering (pointer math not fused into addressing, not thread-parallelized, not
vectorized) — not a launch artifact and not fixable by tuning; it needs the frontend to
fold pointer arithmetic into load/store addressing.

---

## 4. GO / NO-GO

**NO-GO for production routing of the Tri-Dao bwd as a whole.**

Honest scorecard, measured from committed HEAD:
- Correctness: **1/7** kernels numerically correct at real-strided multi-K-trip (small AND §P1):
  only `_chunk_scan_bwd_dstates`. The other 6 fail at HEAD: 3 SEGFAULT at launch (dc, db, state_dx),
  1 NO-ROUTE emit-raise (dcb %1044), 1 NO-LAUNCH de-mono-incomplete (dx arg4_numel_243),
  1 NO-COMPARE atomic-output (ddAcs).
- Runnability: dstates IS de-monomorphized — same baked kernel runs small→§P1 with no segfault. GOOD.
- Performance: even the one correct kernel is **~1292x slower than native** (1473ms vs 1.14ms),
  and the strip-and-time proves ~100% of that is the un-fused pointer-arithmetic prologue.

### What is solid / reproducible
- `_chunk_scan_bwd_dstates` multi-K-trip (cs!=BK) correctness: bit-close at small (3.05e-05)
  AND §P1 (4.88e-04), full output blocks written. The original BLOCKER-1 (multi-K-trip
  K-accumulation) and BLOCKER-2 (monomorphization segfault) are RESOLVED for dstates.

### Exact residuals (RAISED, not papered over)
1. **6/7 kernels do not route/run at HEAD** — each with the exact defect above. The §P1 GO
   requires fixing the dc/db/state_dx launch segfaults, the dx de-mono gap (`arg4_numel_243`
   not threaded as an API arg — same class of fix as the dstates extent symbolization, but
   for an index/numel symbol), and the dcb emit defect (`%1044` unresolved PtrState).
2. **Performance is non-viable** even for the passing kernel: the prologue must fold Triton
   pointer arithmetic into addressing (eliminate the serial `[4096]` bool/int offset loops)
   before any routed kernel can compete with native.

---

## 5. Reproduce (from committed HEAD on gb10)

```
ssh gb10
cd /home/dave/source/tilelang && source /home/dave/cppmega-venv/bin/activate
# 1. regen PrimFunc JSONs from HEAD
for k in _chunk_scan_bwd_dstates _chunk_scan_bwd_dc _chunk_scan_bwd_dcb _chunk_scan_bwd_dx \
         _chunk_state_bwd_db _chunk_state_bwd_dx _chunk_state_bwd_ddAcs_stable; do
  python poc/triton_frontend/_test_harness/tridao_parity/emit_pf_json.py $k; done
# 2. small real-strided parity (per kernel, isolated so one segfault doesn't abort the rest)
python poc/triton_frontend/_test_harness/tridao_parity/parity_all7.py small _chunk_scan_bwd_dstates
# 3. §P1 production dstates parity + timing
python poc/triton_frontend/_test_harness/tridao_parity/parity_prod_dstates.py
# 4. prologue strip-and-time
python /tmp/strip_time_final.py full
python /tmp/strip_time_final.py prologue
```
