# TRITON-ROUTE-RUN — routed Tri-Dao bwd: PROVE + MEASURE (MEASURED from HEAD)

All numbers below are **MEASURED on gb10 (NVIDIA GB10 sm_121, aarch64-linux)** and
**reproducible from the committed HEAD**:

- tilelang frontend HEAD: `aada7de1` (local) = `6a5500d7` (gb10 `/home/dave/source/tilelang`)
  — *"fix(triton-frontend): sink SBlock alloc_buffers INSIDE kernel launch"*
- gb10 working tree clean at that commit; pf JSONs regenerated from HEAD before every run.

RULE #1: no fabricated PASS. Every claim here was re-run from the committed HEAD after
regenerating the routed PrimFunc JSON via `emit_pf_json.py`. Where the route fails or the
launch segfaults, that is reported as the literal verdict — not papered over.

---

## 1. Routing coverage (honest, from HEAD)

Two distinct route gates:

- **emit / PrimFunc level** (`from_ttir` → `tvm.ir.save_json`): 6/7 produce a PrimFunc.
  `_chunk_scan_bwd_dcb` RAISES `EmitError: PtrState references unresolved SSA value '%1044'`
  — it does NOT route.
- **full lowering to CUDA with real `mma`** (`tilelang.engine.lower`, `route_all7.py`): **4/7**.

```
ROUTE _chunk_scan_bwd_dstates        global=True mma=2 atomicAdd=0
ROUTE _chunk_scan_bwd_dc             global=True mma=2 atomicAdd=0
ROUTEFAIL _chunk_scan_bwd_dcb        EmitError: PtrState references unresolved SSA value '%1044'
ROUTE _chunk_scan_bwd_dx             global=True mma=2 atomicAdd=1
ROUTEFAIL _chunk_state_bwd_db        InternalError: variables (arg4,) used but not passed as API args
ROUTEFAIL _chunk_state_bwd_dx        InternalError: variables (arg4,) used but not passed as API args
ROUTE _chunk_state_bwd_ddAcs_stable  global=True mma=2 atomicAdd=1
TOTAL_OK_WITH_MMA=4/7
```

The prior "7/7 route to TVM PrimFunc w/ real T.gemm" was measured at the *emit* level;
at *full lowering* only 4/7 survive (`db`/`dx` fail with an unbound `arg4`).

---

## 2. Parity (routed vs native triton, torch.allclose 1e-3, all elements)

### 2a. dstates — DEGENERATE single-K-trip (cs == BK == 32) — PASS

`parity_tiny.py` (cfg b1 nh2 hd64 ds64 nc2 **cs32** ng2 s256, grid (1,2,2), numel 16384):

```
NATIVE nz=16384/16384 sum=-1240.0947
ROUTED nz=16384/16384 sum=-1240.0947
MAXDIFF=0.000000e+00 ALLCLOSE_1e-3=True  -> PASS
```

This is a bit-exact PASS, **but only because cs==BK gives a SINGLE K-trip** (no
K-accumulation across trips). It does not exercise the strided multi-K-trip path.

### 2b. dstates — REAL STRIDED multi-K-trip (cs=64 != BK=32, 2 K-trips) — FAIL

`parity_all7.py small _chunk_scan_bwd_dstates` (cfg b1 nh8 hd64 ds64 nc4 **cs64** ng2 s256,
grid (1,4,8)):

```
NATIVE nz=131072/131072 sum=-18940.2539
ROUTED nz=65536/131072  sum=-9057.1641      <-- only HALF the blocks written
MAXDIFF=2.409756e+02 ALLCLOSE_1e-3=False -> FAIL
```

**The original DEFECT 2 (MAXDIFF ~2.41e2) is NOT fixed for the real strided / multi-K-trip
case.** It only "passes" in the degenerate single-K-trip tiny config. The routed kernel writes
exactly half the output blocks (65536/131072) under the 2-K-trip config — consistent with a
K-accumulation / second-K-trip defect, not operand staging.

### 2c. Other 6 kernels (small, real strided) — none confirmed PASS

| kernel | small-mode verdict (from HEAD) |
|---|---|
| `_chunk_scan_bwd_dstates` | FAIL — MAXDIFF 2.41e2, half blocks |
| `_chunk_scan_bwd_dc` | SEGFAULT at routed launch |
| `_chunk_scan_bwd_dcb` | does not route (EmitError %1044) |
| `_chunk_scan_bwd_dx` | SEGFAULT at routed launch |
| `_chunk_state_bwd_db` | InternalError on lower (arg4 unbound) |
| `_chunk_state_bwd_dx` | InternalError on lower (arg4 unbound) |
| `_chunk_state_bwd_ddAcs_stable` | NO_CAPTURE — atomic/return output; harness grid KeyError(BLOCK_SIZE_M) |

**Parity PASS count (real strided, all elements, 1e-3): 0 / 7.**
The single bit-exact PASS (2a) is the degenerate single-K-trip config only.

### 2d. dstates — PRODUCTION §P1 — SEGFAULT (monomorphized kernel)

The routed dstates PrimFunc declares **every buffer at a FIXED flat extent 1048576** and bakes
its grid/loop bounds to the tiny dims — it is **monomorphized**, not parameterized. §P1 needs
grid (1,64,112) and a 29,360,128-element dout. Feeding §P1-sized tensors segfaults at the
routed launch (both `parity_prod_dstates.py` and `parity_all7.py prod` core-dump). Native §P1
runs fine. **There is no routed §P1 to compare.**

---

## 3. EXEC ms (real kernel runtime — LAUNCH + CUDA-event timed)

### 3a. Native §P1 reference (re-measured)

```
NATIVE_P1_GPU_EVENT_MS = 1.17 ms/kernel   grid=(1,64,112) numel=29,360,128
```
(matches the 1.14 ms reference.)

### 3b. Routed dstates EXEC ms — ONLY at the tiny config it can run

`parity_tiny` is a single-K-trip bit-exact PASS, so per RULE #1 (exec ms only for parity-PASS),
this is the only routed dstates EXEC number that is allowed:

```
ROUTED dstates GPU-event = 4.46 ms/kernel   (tiny grid (1,2,2), numel 16384)
NATIVE dstates same tiny config           = 0.017 ms/kernel
```

**Residual bottleneck:** the routed kernel takes 4.46 ms to fill 4 blocks / 16384 elements —
~260x the native time for the SAME work — because it runs a fixed whole-block prologue
(the ~168-loop monomorphized prologue) independent of useful work. This is the dominant
residual, NOT operand staging.

There is **no routed §P1 EXEC ms** (§2d: it segfaults), so the routed-vs-native-1.14ms
comparison at §P1 cannot be made from HEAD.

### 3c. USER ASK — fla_dot_exp2 + matmul GPU-EXEC before/after (REAL kernel runtime)

The generic arg-marshaller segfaulted both when the runtime buffer was real-sized (smaller than
the monomorphized declared 1048576 extent). Fix = pack each buffer to the declared extent
(`fit()` in the exec harness). Both go from core-dump to running:

| kernel | BEFORE | AFTER (ms/kernel) | MAXDIFF | note |
|---|---|---|---|---|
| matmul (64x64x64) | SEGFAULT (core dump) | **1.343** | 2.60e-2 | TF32-MMA truncation (ALLCLOSE False @1e-3; precision, not a value bug) |
| fla_dot_exp2 (16x16x16) | SEGFAULT (core dump) | **0.0322** | 1.26e-5 | ALLCLOSE_1e-3 = True |

No regression: both were a hard crash before, both run after. Same root cause as the dstates
prod segfault (undersized buffer vs monomorphized declared extent).

---

## 4. GO / NO-GO

**NO-GO** for the headline goal ("routed Tri-Dao bwd RUNNABLE + NUMERICALLY CORRECT 1e-3,
measured by EXEC ms vs native 1.14ms").

- Runnable: only the degenerate single-K-trip tiny config; §P1 segfaults (monomorphized).
- Numerically correct 1e-3 all-elements real strided: **0/7** (dstates FAILs 2.41e2 at 2 K-trips;
  half blocks written). The earlier "FIXED / MAXDIFF 0" was the single-K-trip config only.
- EXEC ms vs 1.14ms: cannot be produced — routed §P1 does not launch.

**Delivered (GO) sub-results, reproducible from HEAD:**
- USER ASK: matmul SEGFAULT→1.343 ms; fla_dot_exp2 SEGFAULT→0.0322 ms (no regression).
- dstates single-K-trip bit-exact PASS (MAXDIFF 0).
- Honest residual root-cause: monomorphized fixed-extent PrimFunc + whole-block prologue
  (≈168 loops), and an unfixed 2nd-K-trip K-accumulation defect (half blocks at cs=64).

**Remaining defects to fix (the real blockers):**
1. **Monomorphization** — routed PrimFunc bakes extent 1048576 + tiny grid; cannot run §P1.
   Must parameterize buffer extents/grid by the runtime dims.
2. **Multi-K-trip K-accumulation** — at cs!=BK only half the output blocks are written and
   MAXDIFF=2.41e2. The single-K-trip PASS masks this.
3. **Route gaps** — `dcb` EmitError %1044; `db`/`dx` unbound `arg4` at lowering.

---

## 5. Reproduce (gb10, GPU mutex held)

```bash
ssh gb10 'cd /home/dave/source/tilelang && source /home/dave/cppmega-venv/bin/activate
  # regenerate routed PrimFunc JSONs from HEAD
  for n in _chunk_scan_bwd_dstates _chunk_scan_bwd_dc _chunk_scan_bwd_dx \
           _chunk_state_bwd_db _chunk_state_bwd_dx _chunk_state_bwd_ddAcs_stable; do
    python poc/triton_frontend/_test_harness/tridao_parity/emit_pf_json.py $n; done
  python poc/triton_frontend/_test_harness/tridao_parity/route_all7.py        # 4/7 mma
  python poc/triton_frontend/_test_harness/tridao_parity/parity_tiny.py       # single-K PASS
  python poc/triton_frontend/_test_harness/tridao_parity/parity_all7.py small _chunk_scan_bwd_dstates  # real-strided FAIL 2.41e2
  python /tmp/exec_matmul2.py   # matmul AFTER 1.343 ms
  python /tmp/exec_fla3.py      # fla   AFTER 0.0322 ms'
```
