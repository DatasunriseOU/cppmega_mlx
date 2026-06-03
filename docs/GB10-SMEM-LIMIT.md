# GB10 / sm_121 Dynamic Shared-Memory Ceiling — Verdict + Fix

Status: DECIDED. Synthesizes three independent web-research reports (Exa, Brave, Perplexity)
cross-checked against our actual source tree at
`/Volumes/external/sources/tilelang/3rdparty/tvm` and `/Volumes/external/sources/tilelang/tilelang`.
Date: 2026-06-02.

---

## 1. VERDICT (lead)

GB10 / sm_121 per-block usable dynamic shared memory is **~99 KB (101,376 bytes), reachable** —
it is **NOT** a hard ~48 KB cap. All three reports agree, and the device itself advertises
`cudaDevAttrMaxSharedMemoryPerBlockOptin = 101376`. The ~48 KB you see is the universal
"static-shared-memory-without-opt-in" boundary (49,152 B), which every CUDA arch keeps for
back-compat; crossing it requires the per-function dynamic opt-in. **The bisect's 48–50 KB wall is
NOT a missing-carveout problem.** The original hypothesis — "real limit is ~99 KB, TVM just forgets
the `cudaFuncAttributePreferredSharedMemoryCarveout` opt-in" — is **HALF RIGHT**: the 99 KB half is
correct, but the carveout half is **REFUTED**. The carveout is "only a hint, and the driver can
choose a different ratio if required" (NVIDIA Runtime API), and the proven positive controls —
CUTLASS production GEMM, Triton — reach full smem on this exact silicon while setting **only**
`MAX_DYNAMIC_SHARED_SIZE_BYTES` and **no carveout**. TVM already issues that same single opt-in call
(verified at `cuda_module.cc:215-216`). So a *plain* opt-in failure means the request side is wrong:
either (a) TVM/dlight is budgeting tiles against a **hardcoded 49,152 B** instead of the real 99 KB
cap, or (b) the kernel genuinely asks for **> ~99 KB** (a B200/sm_90-sized tile that simply cannot
run on GB10 and must be re-tiled). The fix is on the request/budget side, not a carveout call.

---

## 2. The exact numbers (sm_120 / sm_121), with NVIDIA-doc citations

Per the **NVIDIA Blackwell Tuning Guide §1.4.1.1**
(https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html):

| Quantity | cc 12.0 / 12.1 (RTX 50, RTX PRO, **GB10/Spark**) | cc 10.0 (B200/datacenter) |
|---|---|---|
| Shared memory capacity per SM | **128 KB** | 256 KB |
| Max shared memory **per thread block** | **99 KB** | 227 KB |
| Max static shared (no opt-in) | **48 KB (49,152 B)** | 48 KB |
| Max **dynamic** opt-in per block | **99 KB = 101,376 B** | 227 KB |
| Warps/SM, regs/SM, blocks/SM | 48 warps, 64K regs, 32 blocks | — |

- The 48 KB no-opt-in boundary is explicit: *"static shared memory allocations remain limited to
  48 KB, and an explicit opt-in is also required to enable dynamic allocations above this limit"*
  (Blackwell Tuning Guide).
- The opt-in hard constraint: *"The sum of this value and the function attribute `sharedSizeBytes`
  cannot exceed the device attribute `cudaDevAttrMaxSharedMemoryPerBlockOptin`"*
  (CUDA Runtime API, Execution Control:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html). Request above it →
  `cudaErrorInvalidValue`.
- Device-reported value on GB10 confirmed identically by two independent sources:
  CUTLASS #3144 — *"DGX Spark GB10 (SM121, cudaDevAttrMaxSharedMemoryPerBlockOptin = 101376)"*,
  NVIDIA staff: *"SM120 (RTX 6000, 5090) and SM121 (Spark) only support 99 KiB smem. B200 is SM100,
  which has 228 KiB."* (https://github.com/NVIDIA/cutlass/issues/3144); and FlashInfer SM121 audit
  #3170 — *"SM120 and SM121 share a single 12.x spec column: 99 KB/block, 100 KB/SM"*
  (https://github.com/flashinfer-ai/flashinfer/issues/3170).
- **There is NO hidden 228 KB on GB10. 228 KB is SM100-only.** (Per-SM figure: Tuning Guide says
  128 KB/SM, FlashInfer says ~100 KB/SM usable — minor source discrepancy, immaterial to the 99 KB
  per-block ceiling.)
- Why the bisect's accept/reject landed near ~50,000 rather than exactly 49,152: the **sum** of
  static `sharedSizeBytes` + dynamic request must stay ≤ optin, and there is a ~1 KB CUDA-runtime
  reservation. A few KB of static smem in the kernel explains the 1–2 KB offset. (Mechanism
  documented; the exact 50000/51200 cut is UNVERIFIED but brackets 49,152 precisely.)

---

## 3. Why TE / CUTLASS / Triton get the full smem — exact API sequence

The opt-in is a **single attribute**, set once per function before launch:

```cpp
// CUTLASS production GEMM — include/cutlass/gemm/device/gemm_universal_adapter.h (~line 504/558)
if (smem_size >= (48 << 10)) {
  result = cudaFuncSetAttribute(device_kernel<GemmKernel>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);   // <-- the ONLY opt-in
}
```
(https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/gemm_universal_adapter.h)

```c
// Triton — driver API, same single attribute (works on Blackwell)
cuFuncSetAttribute(cufunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, n_shared_bytes);
```
(https://github.com/triton-lang/triton/issues/2775)

The carveout is a **separate, percentage L1/shared-split occupancy hint**, NOT a gate to reach the
opt-in max:

```cpp
// CuTe tutorial — the ONE example that sets carveout, and only to force L1->SMEM=100%, not to cross 48KB
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100); // "Set L1 to be SMEM only"
```
(https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)

The carveout contract: *"the shared memory carveout preference, in percent of the maximum shared
memory ... This is only a hint, and the driver can choose a different ratio if required to execute
the function."* (https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html). On
Volta+, the driver auto-grants the carveout needed to satisfy a successful `MaxDynamicSharedMemorySize`
opt-in. **CUTLASS's own changelog frames it as a bugfix that needed only the MaxDynamic attribute**,
not a carveout (https://docs.nvidia.com/cutlass/4.4.0/CHANGELOG.html). Net: no mainstream framework
needs the carveout to exceed 48 KB.

A second, real dimension some sources raise (Exa) is the **ptxas compile target**: bare
`sm_120`/`compute_120` ptxas may assume a ~49 KB smem ceiling, while the arch-specific `sm_121a`/
`sm_120a` (or family `120f`/`121f`) targets assume ~102 KB. If GB10 cubins are built for a baseline
target, ptxas itself can refuse the larger size before the driver is even asked. Whether bare
`sm_120` hard-rejects in every CUDA version is UNVERIFIED, but ensuring the GB10 build uses
`sm_121a`/`120f` (requires CUDA ≥ 12.9) is the safe target. (charleschen GB10 wiki:
https://wiki.charleschen.ai/Review/Research/gb10-moe-smem-101kb-tile-budget-constraint ; CUDA
Programming Guide §5.1.2.3.)

---

## 4. The concrete fix for OUR stack

TVM's launch-side opt-in is **already correct and identical to CUTLASS/Triton** — verified in our
tree, `cuda_module.cc:215-216`:

```cpp
CUresult result = cuFuncSetAttribute(
    fcache_[device_id], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, wl.dyn_shmem_size);
if (result != CUDA_SUCCESS)
  TVM_FFI_THROW(InternalError) << "Failed to set the allowed dynamic shared memory size to " << wl.dyn_shmem_size;
```

Adding a carveout call here would NOT raise the ceiling (CUTLASS proves it's not required). **The
real fix is the smem BUDGET used by lowering/scheduling, which is hardcoded to 49,152 B in two
places — both confirmed present in our tree:**

1. **PRIMARY (highest-confidence) — replace the hardcoded 49152 budget so the scheduler targets the
   true 99 KB cap and never under-budgets:**
   - `/Volumes/external/sources/tilelang/3rdparty/tvm/python/tvm/target/tag_registry/cuda.py`
     — line 22 default `shared_mem=49152` and line 42 `"max_shared_memory_per_block": 49152`.
   - `/Volumes/external/sources/tilelang/3rdparty/tvm/python/tvm/s_tir/dlight/analysis/common_analysis.py:369`
     — `"cuda": 49152`.
   For cc ≥ 7.0, these must reflect the device's real `cudaDevAttrMaxSharedMemoryPerBlockOptin`
   (101,376 on GB10), not 49,152. Note TileLang's carver already reads it correctly:
   `/Volumes/external/sources/tilelang/tilelang/carver/arch/cuda.py:150`
   `self.smem_cap = cuda_driver.get_shared_memory_per_block()` — so the bug is the **TVM/dlight**
   path mis-budgeting, which can either under-allocate (kernels that want 50–99 KB get capped to
   48 KB at schedule time) or be inconsistent with carver.

2. **Build target — ensure GB10 compiles for `sm_121a` / `compute_121a` (or family `120f`/`121f`),
   not bare `sm_120`/`sm_90`**, so ptxas permits the full ~99–102 KB. (Requires CUDA ≥ 12.9.)

3. **DEFENSIVE / SECONDARY (matches the original hypothesis, cheap, harmless, but NOT the proven
   fix):** after the existing opt-in in `cuda_module.cc:216`, optionally add
   `cuFuncSetAttribute(fcache_[device_id], CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
   CU_SHAREDMEM_CARVEOUT_MAX_SHARED /* 100 */);`. The JIT wrappers
   (`/Volumes/external/sources/tilelang/tilelang/jit/adapter/wrapper.py:27`, which today emits only
   `cudaFuncAttributeMaxDynamicSharedMemorySize`) could mirror it. Treat as belt-and-suspenders only.

4. **If a fused kernel genuinely needs > 99 KB (e.g. sparse-MLA tiled for sm_90/228 KB), NO API call
   fixes it — it MUST be re-tiled to ≤ 99 KB on GB10.** This is the dominant real-world GB10 failure
   mode across the ecosystem (over-requesting by assuming SM100's 227 KB). Direct precedents:
   QwenLM/FlashQLA #4 — *"Failed to set the allowed dynamic shared memory size to 196608"* (192 KB,
   kernel dimensioned for sm_90); fix = re-tile (num_stages 2→1, block_DV 128→64) to ~80–85 KB
   (https://github.com/QwenLM/FlashQLA/issues/4). TileLang #2201 — *"Failed to set ... to 168128"*
   (164 KB on RTX 5090/sm_120); fixed by re-tiling under 99 KB in TileLang 0.1.9
   (https://github.com/tile-ai/tilelang/issues/2201). Modular #5859 — used the 228 KB Blackwell
   constant on GB10 → `CUDA_ERROR_INVALID_VALUE`; fix = clamp to the real 102,400 B/SM
   (https://github.com/modular/modular/issues/5859). TensorRT-LLM #11368 — GB10 runs CUTLASS FP4
   GEMM at 41.6 TFLOPS with GB10-sized tiles; only B200-class tiles overflow
   (https://github.com/NVIDIA/TensorRT-LLM/issues/11368).

**Our Metal kernels already fit ≤ 32 KB threadgroup memory, so any re-tile-to-fit path (step 4) is
feasible for us** — the worst case is bounded and we have a working reference budget.

---

## 5. Honest unknowns

- **Exact 50000-accept / 51200-reject cut.** No public report shows GB10 rejecting at ~50 KB; this
  is *anomalously low* for a pure opt-in failure (that would trip near ~99 KB). Most likely the
  kernel's **static `sharedSizeBytes` is being summed** with the dynamic request (combined-limit), or
  the schedule under-budgeted to 49,152. Confirm locally by logging, at the failing launch: the
  device `cudaDevAttrMaxSharedMemoryPerBlockOptin`, the kernel's static `sharedSizeBytes`, and the
  exact `wl.dyn_shmem_size` TVM passes — then compare against a minimal CUTLASS kernel requesting the
  same total. This single repro decides between "config under-budget" and "over-request".
- **Per-SM smem figure**: Tuning Guide 128 KB/SM vs FlashInfer ~100 KB/SM. Immaterial to the 99 KB
  per-block ceiling, but the canonical per-SM number is unresolved.
- **ptxas baseline-target hard-reject**: that bare `sm_120` ptxas rejects > ~49 KB in *every* CUDA
  version is empirically reported (charleschen wiki) but not verified across versions in our build.
- **Whether our GB10 cubins are currently built `sm_121a`/`120f` vs bare** — needs a check of our
  actual NVCC/ptxas arch flags; not inspected here.

---

## 6. Sources

- Blackwell Tuning Guide — https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- CUDA Runtime API, Execution Control — https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
- cudaFuncAttributes (carveout "only a hint") — https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaFuncAttributes.html
- CUDA Driver API, Execution Control — https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
- CUTLASS #3144 (GB10 = 101376) — https://github.com/NVIDIA/cutlass/issues/3144
- CUTLASS gemm_universal_adapter.h — https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/device/gemm_universal_adapter.h
- CUTLASS CuTe tutorial (carveout=100) — https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu
- CUTLASS 4.4 changelog (opt-in bugfix) — https://docs.nvidia.com/cutlass/4.4.0/CHANGELOG.html
- FlashInfer SM121 audit #3170 — https://github.com/flashinfer-ai/flashinfer/issues/3170
- Triton #2775 (driver opt-in) — https://github.com/triton-lang/triton/issues/2775
- TVM cuda_module.cc — https://github.com/apache/tvm/blob/main/src/runtime/cuda/cuda_module.cc
- TVM PR #11478 (added >48K opt-in) — https://github.com/apache/tvm/pull/11478
- QwenLM/FlashQLA #4 (192 KB overflow on GB10) — https://github.com/QwenLM/FlashQLA/issues/4
- TileLang #2201 (164 KB overflow, re-tile fix) — https://github.com/tile-ai/tilelang/issues/2201
- Modular #5859 (228 KB constant bug) — https://github.com/modular/modular/issues/5859
- TensorRT-LLM #11368 (GB10 99 KiB, works with GB10 tiles) — https://github.com/NVIDIA/TensorRT-LLM/issues/11368
- Crovella opt-in answer — https://stackoverflow.com/questions/63757245/using-maximum-shared-memory-in-cuda
- charleschen GB10 smem wiki — https://wiki.charleschen.ai/Review/Research/gb10-moe-smem-101kb-tile-budget-constraint
- ptxas Blackwell target reference — https://gh.evko.io/nvopen-tools/ptxas/targets/blackwell.html
- default dynamic smem forum — https://forums.developer.nvidia.com/t/default-value-of-max-dynamic-shared-memory/317700
