# TRITON ROUTE — FRAMEFIX measured result (Tri-Dao bwd through tilelang→tvm)

ALL numbers below are **MEASURED on gb10 (NVIDIA GB10, sm_121a)** with the
current FRAMEFIX code (`poc/triton_frontend/frame_register.py`, byte-identical
local ↔ gb10), under the GPU mutex (`/tmp/gb10_gpu_owner.lock`,
`fragfix-measure`). gb10-only for all GPU exec; no local Metal dispatch.
GPU util 0% throughout; SIGTERM-only.

Honest verdict up front: **NO-GO. Parity 0/7.** The 7 Tri-Dao bwd kernels route
end-to-end to real TF32 tensor-core MMA and FRAMEFIX fixes the partial-tile
C-carry write to a FULL-tile write, but the routed output VALUES are
structurally wrong (random, not transpose/scale/precision) and the routed
kernel is ~2000x slower than native. Routed ms is therefore reported ONLY as a
recorded fact — under RULE #1 it is NOT a valid speed claim because the kernel
fails parity.

## 1. Route + real MMA (PASS, re-confirmed)

`route_all7.py` (from_ttir → tilelang.engine.lower, target=cuda):

| kernel                          | __global__ | mma | atomicAdd |
|---------------------------------|-----------|-----|-----------|
| _chunk_scan_bwd_dstates         | yes       | 2   | 0         |
| _chunk_scan_bwd_dc              | yes       | 2   | 0         |
| _chunk_scan_bwd_dcb             | yes       | 2   | 0         |
| _chunk_scan_bwd_dx              | yes       | 2   | 1         |
| _chunk_state_bwd_db             | yes       | 2   | 0         |
| _chunk_state_bwd_dx             | yes       | 2   | 3         |
| _chunk_state_bwd_ddAcs_stable   | yes       | 2   | 1         |

**TOTAL_OK_WITH_MMA = 7/7.**

SASS (`sass_all7.py`, real `HMMA.1688.F32.TF32` per cubin):

| kernel                          | HMMA.1688.F32.TF32 |
|---------------------------------|--------------------|
| _chunk_scan_bwd_dstates         | 32                 |
| _chunk_scan_bwd_dc              | 32                 |
| _chunk_scan_bwd_dcb             | 32                 |
| _chunk_scan_bwd_dx              | 32                 |
| _chunk_state_bwd_db             | 32                 |
| _chunk_state_bwd_ddAcs_stable   | 8                  |
| _chunk_state_bwd_dx             | 8                  |
| **TOTAL**                       | **176**            |

7/7 emit real TF32 tensor-core MMA in SASS; atomicAdd preserved on the reduce
kernels. (The "288 HMMA" in earlier state notes was a different tile config; the
current FRAMEFIX build emits 176.)

## 2. Parity (FAIL — 0/7)

Only `_chunk_scan_bwd_dstates` was exercised at parity (the lead kernel and the
one FRAMEFIX targets). Real strided layout (cs=64 != BLOCK_K=32 → 2 K-trips,
NON-degenerate, strides ≠ 0). Native = mamba_ssm Triton `_chunk_scan_bwd_dstates_kernel`.

### Structural probe (small, b1 nh8 hd64 ds64 nc8 cs64, numel 262144)
`probe_dstates_struct.py`, routed-vs-native-Triton reference:

```
ROUTED vs ref(1trip) maxdiff=1.7166e+02
ROUTED vs ref(2trip) maxdiff=2.4306e+02
tile[0,0,0] matched(<1e-2)=1/4096
routed/ref2 ratio mean=0.0156 std=2.8486
```

Ratio is random (mean ~0, std ~2.8) → NOT a transpose, NOT a scale, NOT a
precision rounding. The values are structurally permuted.

### Production §P1 (b1 nh112 hd64 ds64 nc64 cs64, numel 29360128, grid (1,64,112))
`parity_prod_dstates.py`, routed-vs-native-Triton (same process, pf via
load_json to dodge the from_ttir↔torch LLVM clash):

```
NATIVE nonzero=29360127/29360128 sum=-203350.2188
ROUTED nonzero=29360128/29360128 sum=-573166.2500   <- FULL tile written (FRAMEFIX fixed the partial write)
MAXDIFF=1.141630e+03
ALLCLOSE_1e-3=False
FAIL
```

FRAMEFIX DID fix the partial-tile write (routed nz 29360128/29360128 = full,
was 2048/262144 / 32-per-tile before FRAMEFIX). But values are wrong.

### Per-kernel parity table

| kernel                          | routed nz (prod) | MAXDIFF vs native | allclose 1e-3 | PASS |
|---------------------------------|------------------|-------------------|---------------|------|
| _chunk_scan_bwd_dstates         | 29360128/29360128 (FULL) | 1.14e+03 | False | **FAIL** |
| _chunk_scan_bwd_dc              | not run (lead kernel fails first) | — | — | FAIL (blocked) |
| _chunk_scan_bwd_dcb             | not run | — | — | FAIL (blocked) |
| _chunk_scan_bwd_dx              | not run | — | — | FAIL (blocked) |
| _chunk_state_bwd_db             | not run | — | — | FAIL (blocked) |
| _chunk_state_bwd_dx             | not run | — | — | FAIL (blocked) |
| _chunk_state_bwd_ddAcs_stable   | not run | — | — | FAIL (blocked) |

**Parity = 0/7 PASS.** The lead kernel fails structurally; the remaining 6 share
the same operand-staging path and are not separately PASS-able, so no kernel is
parity-PASS.

## 3. Root cause (precisely isolated, reproduced)

FRAMEFIX (commit `ff350d21` / gb10 `92fd35d2`, post-walk SBlock layout_map
re-registration) correctly pins the **C accumulator** fragment's
`make_mma_store_layout` so the C carry write becomes full-tile. It does NOT fix
the **A/B operand source** staging:

- The A operand (dout*exp fragment) is restaged to shared via
  `reduction.py::_stage_operand_to_shared` → `_emit_copy_stmt`, which lowers to a
  SERIAL linear-index-swizzled fill (`for i_j_fused in 0..512`).
- That fill is INCONSISTENT with the cooperative tid-parametrized `ptx_ldmatrix`
  read the gemm emits → the gemm consumes a permuted A. A native
  `T.copy(A,As)+T.gemm` with identical M/N/K/trans flags is numerically CORRECT
  (maxdiff 0.018, 4096/4096), proving the defect is the frontend operand staging,
  not the gemm.
- Secondary: the FRAMEFIX C-layout pin is FLAKY — LayoutInference intermittently
  raises `Get different layout for dot_c_frag_*` (+1-offset conflict) under
  hash-map iteration nondeterminism; some runs fail to compile entirely.

## 4. Timing (recorded fact — NOT a valid speed claim, parity FAIL)

| path                                                | ms / kernel @ §P1 | source |
|-----------------------------------------------------|-------------------|--------|
| native mamba_ssm Triton `_chunk_scan_bwd_dstates`   | **1.16 ms**       | MEASURED `native_prod.py` |
| routed-through-our-stack (FRAMEFIX)                 | 2419.43 ms        | MEASURED `parity_prod_dstates.py` — parity FAIL |

Routed is ~2080x slower than native (serial linear-index operand fill, MVP
scalar path, PtrAnalysis C++ shim not loaded in-process). Under RULE #1 this ms
is reported as a measured fact only — it is NOT a speed result because the
kernel fails parity. No full-7 routed-bwd sum is reported: 0/7 PASS parity, so
there is nothing valid to sum.

## 5. Path-vs-path summary

| path                                  | status | §P1 dstates ms | full-bwd |
|---------------------------------------|--------|----------------|----------|
| native mamba_ssm full bwd (7 kernels + non-Triton) | reference | 1.16 ms/kernel | ~10 ms-class (full chain) |
| routed-through-our-stack (FRAMEFIX)   | **route 7/7, parity 0/7 → NO-GO** | 2419 ms (wrong values) | n/a (parity FAIL) |
| path_c v1 (cppmega.mlx)               | reference | — | 905 ms (full chain) |

## 6. GO / NO-GO

**NO-GO.** Achieved: 7/7 route to real TF32 MMA (176 HMMA), FRAMEFIX converts
the partial-tile C-carry write to a full-tile write (measured nz 29360128/29360128).
Blocked: routed output VALUES are structurally wrong (MAXDIFF 1.14e+03 prod /
2.43e+02 small, ratio random) and the routed kernel is ~2000x slower than native.

Remaining defect (RAISED, not papered over, per RULE #1): the A/B GEMM operand
staging (`reduction.py::_stage_operand_to_shared`) emits a serial
linear-index-swizzled shared fill that is layout-inconsistent with the gemm's
cooperative `ptx_ldmatrix` read; plus the FRAMEFIX C-layout pin is flaky under
LayoutInference hash-map nondeterminism. FRAMEFIX fixed the C carry; the next fix
must frame-register / layout-route the operand restage the same way (or use a
layout-aware `T.copy` for A/B as proven correct in the native isolation test).

Mutex released and cleaned up at end of run.
