# Triton-route loop-carried PtrState forwarding (`carry_index` gather elimination)

Status as of commit `15949e5a` (branch `merge/upstream-codegen-reorg`, repo
`/Volumes/external/sources/tilelang`). All GPU execution measured on **gb10**
(NVIDIA GB10, sm_121, aarch64-linux), source `/home/dave/source/tilelang`,
venv `/home/dave/cppmega-venv`. NO local Metal / Apple-GPU dispatch was used.

---

## 1. What changed this run

Three frontend files (parent `40201225` -> HEAD `15949e5a`, commits `54b72972`
+ `15949e5a`):

| File | Change |
|---|---|
| `poc/triton_frontend/op_emitters/control.py` | Forward loop-carried PtrState across `scf.for`: each iter_arg/block-arg inherits the PtrState of its INIT operand; the per-iteration `tt.addptr` advance is folded so in-loop `tt.load`s on the carried block-args emit strided `make_tptr` tile-loads instead of the scalar `carry_index` gather. Plus FIX 1 (canonical per-axis `blockIdx` cache; sorted x,y,z thread_extent). |
| `poc/triton_frontend/op_emitters/memory.py` | FIX 2: `_ptrstate_flat_index` no longer re-multiplies the already-flat per-axis `offsets` by the axis stride (`stride^2` double-count removed); `flat += off; flat += lv*stride`. |
| `poc/triton_frontend/op_mapping.py` | `map_tt_program_id` caches one Var per program-id axis (`ctx.program_id_axis_var`) so a repeated axis read returns the SAME binding -> one `launch_thread` (kills the duplicate `gridDim_0_1`). |

---

## 2. carry_index gather elimination (the headline)

The `carry_index` scalar masked-GATHER (one repeated base address per GEMM
operand tile -> structural permutation) is the defect being killed.

| Kernel (Tri-Dao bwd) | carry_index BEFORE | carry_index AFTER |
|---|---|---|
| dstates, dc, dx, state_db, state_dx, ddAcs_stable | 6 each | **0 each** |

Verified on the regenerated routed PrimFunc `/tmp/pf__chunk_scan_bwd_dstates.json`
(`grep -c carry_index` -> 0). The in-loop tile_loads carry the real
per-program 2D strided `make_tptr` recovered by the aarch64 C++ PtrAnalysis shim.

Additionally FIX 1 dropped the duplicate launch dim: routed dstates kernel
`nparams` went **35 -> 34** (`gridDim_0_1` gone), and FIX 2 made the placement
correct (every grid block writes its native `dprev` slot).

---

## 3. Parity (HONEST — NO-GO on §P1 prod)

* **Placement permutation: ELIMINATED.** With FIX 1 + FIX 2 the grid no longer
  collapses to block (0,0,0); the prior-run io.pt config (nc8 nh8 = 64 grid
  blocks) wrote 64/64 distinct 4096-tiles, total nz 262144/262144 (vs
  4096/131072 = one block before).
* **In-tile value parity: STILL FAILS.** dstates small config MAXDIFF
  ≈ 2.41e2, `allclose(1e-3)=False`. A SEPARATE in-tile GEMM value error remains
  (distinct from the now-fixed placement permutation and the now-zero gather).
* **§P1 production (S4096, c64, g8, H112, P64, N64, bs1, grid (1,64,112)):
  routed kernel SEGFAULTS at CUDA launch** (measured this run with the correct
  34-arg call via `/tmp/time_routed34.py` and `/tmp/parity_prod_dstates34.py`).
  This is the residual NO-GO blocker; native ref = 1.14 ms/kernel.

Per RULE #1 no routed ms is reported as a PASS for dstates: the kernel does not
yet produce a correct result at the prod config, so a timing number there would
be a degenerate/garbage-output number. Reported as NO-GO, not fabricated.

---

## 4. Routed ms (parity-PASS kernels only)

**None.** No bwd kernel reaches `allclose(1e-3)` parity at the prod config this
run (dstates in-tile values wrong + §P1 segfault; the other 6 unverified
downstream of the same in-tile GEMM defect). Therefore, per the rule "routed ms
only for parity-PASS", there is no routed-ms table to report. Native baseline
for reference: `_chunk_scan_bwd_dstates` = **1.14 ms/kernel** @ §P1.

Residual bottleneck quantified: §P1 routed launch segfault in the multi-K-trip
in-tile GEMM (cs=64, BK=32 -> 2 K-trips), plus the redundant whole-block
prologue (168 loops) noted previously.

---

## 5. Other-paths GEMM perf — BEFORE vs AFTER (MANDATORY, delivered)

UNCONDITIONAL user ask: confirm this run's frontend edits did NOT regress the
OTHER frontend GEMM paths. Measured on gb10 by capturing TTIR (subprocess),
routing via `from_ttir(target="cuda")`, in fresh isolated processes.

* **BEFORE** = the 3 frontend files checked out at parent `40201225`
  (`git checkout 40201225 -- control.py memory.py op_mapping.py`, pyc purged).
* **AFTER** = HEAD `15949e5a`.
* `route_ms` = min over 8 reps, then min over 3 isolated-process trials (the
  routing path is exactly where the edited code executes; `route_ms` is the
  honest frontend-cost metric for these paths).

| Kernel | scf.for | carry_index B / A | route_ms BEFORE (min) | route_ms AFTER (min) | Verdict |
|---|---|---|---|---|---|
| `fla_dot_exp2` (dot + exp2) | 0 | 0 / 0 | 73.86 | 75.68 | NO REGRESSION (~+2.5%, machine noise; route 73.8/74.3/99.7 B vs 75.7/76.2/97.6 A) |
| `matmul` (K-loop GEMM) | 1 | 14 / 14 | 112.30 | 115.24 | NO REGRESSION (~+2.6%, noise; route 116/112/113 B vs 119/115/115 A) |
| `dot_reduce_atomic` (dot+rowsum+atomic) | 0 | n/a | EmitError `arith.constant true` | StructuralEqual `(32,)` (later stage) | NO REGRESSION — AFTER fails LATER (the `arith.constant true` emit gap was fixed; now reaches a shape check) |
| `dot_reduce_atomic_trans_b` | 0 | n/a | EmitError `arith.constant true` | StructuralEqual `(32,)` (later stage) | NO REGRESSION — same as above |

GPU-exec of these 4 standalone kernels is blocked by PRE-EXISTING tilelang
codegen issues present identically BEFORE and AFTER (matmul = CUDA segfault at
launch on the 32-tile config; fla_dot_exp2 = `m_warp*n_warp==num_warps` check
`1*1 != 4`). These are not introduced by this run's edits (verified by running
the identical BEFORE/AFTER call). Hence exec_ms is not reportable for these
paths in either state; the comparable frontend metric is `route_ms` + the
`carry_index` count, both of which show no regression.

**Honest attribution note on matmul:** the loop-carried PtrState forwarding did
NOT eliminate the `carry_index` gather on the simple `matmul` K-loop
(`carry_index=14` BEFORE and AFTER). The forwarding implemented this run is
scoped to the Tri-Dao bwd carried-ptr pattern; the generic
`a_ptrs += BK*stride` / `b_ptrs += BK*stride` carry in `matmul` still falls to
the scalar gather. This is a real, disclosed limitation — not a regression, and
not claimed as fixed.

---

## 6. Full-7 routed-bwd sum / vs native / vs path_c

Not computable as a parity-PASS aggregate: the routed bwd is NO-GO this run
(§P1 segfault + in-tile value error). path_c v1 reference chain = 905 ms;
native full-bwd reference per-kernel = 1.14 ms/kernel for dstates. No fabricated
routed sum is reported.

---

## 7. GO / NO-GO

**NO-GO for routed Tri-Dao bwd parity.** Achieved this run: carry_index gather
6 -> 0 on all 6 routable kernels; placement permutation eliminated; duplicate
launch-dim fixed (nparams 35->34); `arith.constant true` emit gap fixed.
Remaining: a SEPARATE in-tile GEMM value error and a §P1 prod launch segfault —
the routed bwd is not yet numerically correct or runnable at prod scale.

**GO on the mandatory other-paths regression check:** fla_dot_exp2 and matmul
route with NO regression (route_ms within machine noise, carry_index unchanged);
dot_reduce paths fail later (improved), not earlier. No GEMM path regressed.
