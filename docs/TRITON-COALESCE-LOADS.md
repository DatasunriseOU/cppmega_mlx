# Triton-frontend iteration 3 — coalesced async K-loop loads (LDGSTS)

Status: **HONEST PARTIAL / NO-GO on LDGSTS this iteration.** Pipeline-annotation
infrastructure landed and committed on the routed-triton path; the generated `.cu`
is byte-identical OFF==ON, so cp.async/LDGSTS stays 0 and there is **no ms drop**
from the 1102 ms iteration-2 baseline. The blocker is precisely localized in
TVM's `pipeline_planning.cc` and verified by reading the source.

All numbers below are MEASURED-FROM-COMMITTED-HEAD on gb10 (sm_121a), regenerating
the `_chunk_scan_bwd_dstates` `.cu` from HEAD with `async_loads` OFF vs ON,
interleaved, CUDA-event timed.

- tilelang committed HEAD: `53be06b321b78844ab12873980cfbf5a0bbbf742`
  (gb10 working tree byte-identical to HEAD for the 4 iteration-3 files; md5-verified).

## What landed (committed, iteration 3)

- `WalkerCtx.routed_triton_async_loads` gate (ON by default on the routed
  prologue-opt path) + a shared `_gmem_shared_copies` counter list.
- `_emit_load_copy` / `_emit_ptrstate_tile_load_tir` append to the counter when
  they emit a global->shared shared-scope copy.
- `control.map_scf_for._maybe_pipeline()` stamps the serial K-loop `For` with the
  `num_stages` software-pipeline annotation (mirroring `T.Pipelined`) when the
  body emitted >=1 shared copy and the body is a `SeqStmt`.

## Measured (gb10, `_chunk_scan_bwd_dstates`, §P1)

§P1 config: b=1 nh=112 hd=64 ds=64 nc=64 cs=64, BM/BN/BK = 64/64/32.

| metric | async OFF | async ON | Triton ref |
|---|---|---|---|
| generated `.cu` md5 | `99c232418b10c2793928ed63776b76c2` | `99c232418b10c2793928ed63776b76c2` | — |
| cp.async in `.cu` | 0 | 0 | — |
| SASS LDGSTS | 0 | 0 | 75 |
| SASS spill (STL+LDL) | 306 (168+138) | 306 (168+138) | 0 |
| SASS HMMA | 32 | 32 | 256 |
| plain LDG | 38 | 38 | — |
| §P1 parity MAXDIFF | 4.882812e-04 | 4.882812e-04 | — |
| ALLCLOSE 1e-3 | PASS | PASS | — |
| EXEC ms/kernel (CUDA events, N=50) | ~1102.3 | ~1102.3 (byte-identical) | — |
| native ms/kernel | — | — | ~1.12 |

The `.cu` is byte-identical OFF==ON, so timing is identical: rep medians 1102.14 /
1102.31 / 1102.43 / 1102.64 ms over interleaved reps. **No real drop from 1102.**
Remaining gap to native ~1.12 ms is ~984x and is dominated by LDGSTS=0
(memory-bound plain LDG) plus HMMA 32 vs 256 tensor-core under-util.

Other paths: route_all7 = 5/7 build with mma intact
(dstates/dc/db/state_dx/ddAcs, mma=2 each); dcb/dx fail IDENTICALLY to baseline
(pre-existing PtrState/undefined-arg failures, not a regression). GEMM HMMA=32
intact in the routed kernels.

## Why LDGSTS did not rise — precisely localized blocker (source-verified)

The K-loop producer for the dout/C/dA tiles is a **predicated manual `tir.For`
grid**:

```
shared[i,j] = T.if_then_else(mask, global[flat_idx], other)
```

TVM's async-pipeline planner only creates a cp.async stage for a *proven copy*
stage. In `src/transform/pipeline_planning.cc`:

- line ~973: `pure_copy_stage = collector.GetGlobalCopyPattern() && IsPureCopyStmt(block->body);`
- `IsPureCopyStmt` (line ~652) requires the store VALUE to be a *pure raw copy
  value* — `is_pure_raw_copy_value` accepts only a `BufferLoad` of a global-like
  buffer, optionally wrapped in a `Cast`. A `tir.if_then_else` is a `CallNode`,
  so it returns `false`, which sets `saw_non_copy_buffer_store = true` and makes
  `IsPureCopyStmt` return `false`.

Result: `copy_stage = false` -> no async stage created -> no cp.async emitted
even though the `num_stages` annotation is present. The byte-identical `.cu`
confirms the annotation alone is inert.

Converting the predicated `For` to a clean `T.copy` `CopyNode` is further blocked
because the structured 2D tile was lowered to a 1D `BufferLoad(src, [flat_idx])`
with SYMBOLIC per-axis strides: the dout/C last-axis stride is
`stride_dout_hdim` / `stride_c_dstate` (runtime args), NOT a literal 1, so the
block-level axis-contiguity Triton's `Coalesce.cpp` keys on was discarded in the
flat-index lowering.

## RULE #1 compliance

No silent uncoalesced fallback was introduced. The kernel still emits correct
(plain LDG) code and parity holds at §P1 (MAXDIFF 4.88e-04). The honest reason
LDGSTS stayed 0 is stated above and verified against the TVM source. The real fix
(next iteration): preserve a 2D strided source region, split the predicate off the
copy, emit a clean `T.copy` `CopyNode` global->shared so `IsPureCopyStmt` passes,
then re-verify §P1 parity and re-measure ms/LDGSTS.

## GO / NO-GO

NO-GO on LDGSTS for iteration 3. Infrastructure (num_stages annotation, gate,
counter) is landed and committed with full no-regression. The next iteration must
attack the `IsPureCopyStmt` gate by producing a CopyNode, not a predicated
`if_then_else` store. Remaining gap to native ~1.12 ms is ~984x, dominated by the
uncoalesced loads and tensor-core under-utilization.
