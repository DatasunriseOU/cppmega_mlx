# T.fp8_scaled_matmul macro baseline for TileLang (Metal + CUDA)

## Status

**Correctness baseline, not the scheduler/MMA optimization.** This patch
exposes `T.fp8_scaled_matmul(...)` as a hygienic `@T.macro` that lowers on
Metal and CUDA by emitting the audiohacking/fp8-mps-metal scalar
scaled-matmul algorithm directly into the `@T.prim_func` AST. It does **not**
register a real TIR op in `src/op/builtin.cc`, does **not** add a Metal
scheduler pass, and does **not** implement the proposed fast path
`dequant FP8 -> FP16 threadgroup tile -> existing FP16 simdgroup MMA`.

## What ships

| File                                                  | Lines   | Role                                                                          |
| ----------------------------------------------------- | ------- | ----------------------------------------------------------------------------- |
| tilelang/language/fp8_op.py                           | 379     | New file. The fp8_scaled_matmul macro + validators + dispatch.                |
| tilelang/language/__init__.py                         | +1      | Re-export T.fp8_scaled_matmul.                                                |
| testing/python/cpu/test_fp8_scaled_matmul_lowering.py | 195     | New file. IR-level lowering tests (no GPU required).                          |
| testing/python/metal/test_fp8_scaled_matmul_metal.py  | 879     | New file. Metal codegen + xcrun compile + e2e parity vs audiohacking + bench. |

Total: 1454 insertions across 4 files.

The e4m3 subnormal decode (the `e+7` biased-exponent fix in
__tvm_fp8_e4m3_to_half) is a **prerequisite** of this patch but
is not part of it: it ships with the
docs/upstream/tilelang_metal_fp8 storage-only patch, which now
bakes the corrected constant into the helper from the start. Apply
patch #5 (Agent C / metal_fp8) before this one and no codegen edit
is needed here -- the macro just calls into the already-correct
helper. See the "e4m3 subnormal correctness" section in
docs/upstream/tilelang_metal_fp8/README.md for the byte-level
analysis.

## Apple Silicon / MLA motivation

For Apple Silicon M4 / M4 Max MLA work, this macro gives TileLang programs a
testable FP8 scaled-matmul surface with Metal lowering, xcrun compile checks,
and parity against an independent audiohacking-style reference. That is useful
for validating FP8 storage/decode and scale semantics before wiring a faster
Metal scheduler path. It should not be presented as a measured FP8 simdgroup
speedup.

## Algorithm

The macro is a line-for-line port of the fp8_scaled_matmul_kernel body
from [audiohacking/fp8-mps-metal](https://github.com/audiohacking/fp8-mps-metal)
(commit d4fbd40c, MIT). The expansion inside a @T.prim_func is:

```python
for i, j in T.Parallel(M, N):
    for k in T.serial(K):
        a = T.cast(A_fp8[i, k], "float32")        # FP8 byte -> fp32
        b = T.cast(B_fp8[k, j], "float32")        # FP8 byte -> fp32
        sa = A_scale[0] if A_scale.shape == (1,) else A_scale[i]
        sb = B_scale[0] if B_scale.shape == (1,) else B_scale[j]
        C_local[i, j] += a * b * sa * sb

```

Per-tensor vs per-row dispatch happens at macro-expansion time based on
the static shape of the scale operand; the resulting MSL contains **no
runtime predicate** for the scale layout.

## Lowering paths

The macro emits the same TIR on every target. Output codegen differs only
in the FP8-to-fp32 cast helper:

* **Metal** -- T.cast(fp8 byte, fp32) lowers via
  __tvm_fp8_e4m3_to_half / __tvm_fp8_e5m2_to_half from the Agent C
  storage-only patch. The TileLang Gemm dispatcher at
  tilelang/tileop/gemm/__init__.py already routes FP8 inputs through
  GemmMetalScalar (Agent E), so the loop body for T.gemm(fp8, fp8,
  fp32) and T.fp8_scaled_matmul(...) is structurally identical except
  for the extra * sa * sb multiplications. The resulting MSL is
  functionally identical to the audiohacking kernel (one branch + a few
  shifts per byte per dequantization + fp32 fma).

* **CUDA / ROCm** -- T.cast uses TVM's native FP8 path
  (__nv_fp8_e4m3_to_half etc.). For Hopper / Blackwell, callers who
  want the tensor-core FP8 FMA path should use
  T.tcgen05_gemm_blockscaled(...) directly (PRs #202 / #1600). Those
  GEMMs ingest the e8m0fnu block-scale operand explicitly and don't
  fit this op's per-tensor / per-row scale signature -- we keep this
  intrinsic limited to the audiohacking-style scalar layout.

* **CPU** -- same scalar TIR; T.cast(fp8, fp32) lowers via TVM's CPU
  FP8 helpers.

We chose the macro form for this baseline (rather than registering a new TIR
op via src/op/builtin.cc) because:

1. The lowering is identical to T.gemm(fp8, fp8, fp32) plus a fused
   per-element scale; a registered op would just call back into the
   existing GemmMetalScalar machinery with a wrapped body. The macro
   skips the round-trip.
2. This patch does not implement the separate Metal scheduler path that would
   dequantize FP8 into half threadgroup tiles and feed the existing FP16
   simdgroup MMA emitter. Until that exists, the validated lowering path here
   is scalar dequant + fp32 fma, and the tests explicitly assert that no
   `simdgroup_multiply_accumulate` is emitted for this macro.
3. Macro form means **no C++ rebuild** for the surface API. This patch
   is pure Python; the e4m3 decode helper already ships with the
   correct biased exponent in patch #5
   (docs/upstream/tilelang_metal_fp8), so nothing in
   codegen_metal.cc moves here.

## Prerequisite: correct e4m3 subnormal decode

This intrinsic depends on the Agent C storage-only patch
(docs/upstream/tilelang_metal_fp8/) shipping the corrected
__tvm_fp8_e4m3_to_half helper, where the subnormal path uses
(e + 7) for the half biased exponent rather than the earlier
(e + 8). With the corrected helper, byte-by-byte e4m3 -> half
decode matches torch.float8_e4m3fn / mlx.from_fp8 / the
audiohacking LUT reference for the entire finite range. Without
that fix (i.e. against a stale build of patch #5) the per-tensor
32x32x64 e2e test fails with about 7% relative error on outputs
that depend on small-magnitude operands (bytes 0x01-0x07 and
0x81-0x87, the seven subnormal magnitudes per sign). Apply
patch #5 first and rebuild the C++ extension once; this patch
itself does not require a rebuild.

## Test results


$ cd /tmp/tilelang_apple_head/tilelang
$ pytest testing/python/cpu/test_fp8_scaled_matmul_lowering.py \
         testing/python/metal/test_fp8_scaled_matmul_metal.py -v



testing/python/cpu/test_fp8_scaled_matmul_lowering.py
  test_macro_expands_to_scalar_kloop_metal              PASSED
  test_per_tensor_scale_lowering_shape                  PASSED
  test_per_row_scale_lowering_shape                     PASSED
  test_e5m2_lowering_uses_e5m2_helper                   PASSED
  test_validation_rejects_non_fp8_inputs                PASSED
  test_validation_rejects_bad_scale_size                PASSED
  test_validation_rejects_k_mismatch                    PASSED
  test_intrinsic_in_pre_lowering_ir                     PASSED

testing/python/metal/test_fp8_scaled_matmul_metal.py
  test_per_tensor_scale_lowers_on_metal                 PASSED
  test_per_row_scale_lowers_on_metal                    PASSED
  test_per_col_scale_lowers_on_metal                    PASSED
  test_e5m2_lowers_on_metal                             PASSED
  test_mixed_e4m3_e5m2_lowers_on_metal                  PASSED
  test_xcrun_compile_per_tensor_scale                   PASSED
  test_xcrun_compile_per_row_scale                      PASSED
  test_xcrun_compile_mixed_dtype                        PASSED
  test_e2e_per_tensor_scale_parity                      PASSED
  test_e2e_per_row_scale_parity                         PASSED
  test_rejects_non_fp8_a                                PASSED
  test_rejects_bad_scale_shape                          PASSED
  test_e2e_audiohacking_parity_per_tensor_128           PASSED
  test_e2e_audiohacking_parity_per_row_singleblock      PASSED
  test_e2e_audiohacking_parity_vecmat_4096              PASSED
  test_bench_matmul_vs_audiohacking                     PASSED
  test_bench_vecmat_vs_audiohacking                     PASSED

25 passed in 4.71s


The 8 CPU lowering tests don't need a GPU; they assert that the macro
expansion produces the audiohacking-shaped TIR (Cast(fp8 -> fp32) *
Cast(fp8 -> fp32) * sa * sb accumulation) and that the per-tensor /
per-row branch picks the right buffer index pattern.

The 12 Metal codegen + xcrun tests verify that:

* The lowered MSL contains the __tvm_fp8_e4m3_to_half /
  __tvm_fp8_e5m2_to_half calls in the kernel body (not just the
  prelude).
* The MSL is accepted by xcrun --sdk macosx metal -c (offline
  compile) on per-tensor, per-row, and mixed-dtype operands.
* The MSL contains no simdgroup_multiply_accumulate -- FP8 inputs go
  through the scalar fallback.

The 5 e2e + bench tests (gated on tilelang.testing.requires_metal,
torch.mps, and the cppmega_mlx audiohacking MSL kernels):

* test_e2e_per_tensor_scale_parity -- 32x32x64, max abs err 0.0
  vs torch reference.
* test_e2e_per_row_scale_parity -- 32x32x64 with per-row A scale,
  max abs err < 1e-3 vs torch reference.
* test_e2e_audiohacking_parity_per_tensor_128 -- 128x128x128
  per-tensor vs the audiohacking LUT-decode kernel via mlx.core,
  abs err < 1e-3.
* test_e2e_audiohacking_parity_per_row_singleblock -- single-block
  per-row vs audiohacking, abs err < 1e-3.
* test_e2e_audiohacking_parity_vecmat_4096 -- M=1, K=N=4096 matmul
  vs audiohacking matmul kernel, abs err < 5e-2 (FP32 FMA reordering
  across 4096-K contraction).

## Bench results

Run on M-series Apple Silicon via pytest -s (the bench tests print to
stdout). Median of 9-of-10 iterations after 3 warm-up calls.

### 128x128x128 e4m3 per-tensor scaled matmul


[bench] 128x128x128 per-tensor e4m3 FP8 scaled matmul:
  TileLang  :   0.555 +/- 0.016 ms  (0.008 TFLOPS)
  audiohack :   0.172 +/- 0.031 ms  (0.024 TFLOPS)
  ratio TileLang / audio = 3.16x


The audiohacking kernel uses a single-pass dispatch with 256-entry LUT
decode and 4-element K-axis unrolling; TileLang's scalar fallback runs
the same arithmetic but goes through the GemmMetalScalar replicated
fragment layout. The scalar fallback is correctness-equivalent at
1e-3 absolute tolerance.

### M=1 N=K=4096 e4m3 vecmat


[bench] M=1 N=4096 K=4096 e4m3 FP8 vecmat:
  TileLang scalar  :   1.098 +/- 0.032 ms  ( 0.031 TFLOPS)
  audiohack simdg  :   0.182 +/- 0.012 ms  ( 0.184 TFLOPS)
  ratio TileLang / audio = 6.01x (audiohacking wins; TileLang has no
                                  simdgroup reduction yet)


The audiohacking project ships a dedicated
fp8_scaled_vecmat_kernel for M=1 with simdgroup reduction (one SIMD
group per output row, simd_sum reduction). TileLang's scalar fallback
distributes the M*N=N output cells across threads, so for M=1 each
thread does K/threads of the work in scalar form -- no cross-thread
reduction. Closing this gap is a follow-up; the necessary scaffolding
(metal_simdgroup tile-op) already exists for fp16/fp32 in
tilelang/tileop/metal_simdgroup.py but doesn't yet plumb through for
FP8 storage.

## Limitations and follow-ups

These items are explicitly out of scope for this PR but tracked here:

1. **Per-row scale at multi-block tile granularity.** The macro
   currently indexes A_scale[i] where i is the **block-local** row
   in the K-tile loop. To support a full (M,) per-row scale with
   BM < M, the user must either pass a sliced view at the call site
   (A_scale[by * BM:(by + 1) * BM]) or wait for a follow-up macro
   extension that threads by * BM into the scale index. Per-tensor
   (1,) scales work transparently for any tile size. Single-block
   per-row (BM == M) is fully covered by tests.

2. **Performance: simdgroup reduction for vecmat.** Audiohacking is
   6x faster on M=1 because of simd_sum. Plumbing this through
   T.fp8_scaled_matmul requires a simdgroup-reduction tile-op for
   FP8-storage inputs; the TileLang Metal codegen rejects allocating
   metal.simdgroup buffers with FP8 dtype, so the path would be
   "dequant FP8 -> half on load, run simdgroup_matrix_multiply, scale
   post-loop." That's a separate Agent task.

3. **Performance: K-axis vectorisation.** The audiohacking kernel
   reads 4 FP8 bytes at a time via uint reinterpretation
   (reinterpret_cast<device const uint*>(W + row_offset)). TileLang
   has the vector FP8 cast helpers from Agent F-1
   (__tvm_fp8_e4m3_to_half_v4) but the macro's K-loop uses scalar
   T.cast. A vectorised inner loop is a clean follow-up that would
   close most of the 3.2x matmul performance gap.

4. **CUDA scheduler dispatch.** On Hopper / Blackwell, the
   T.tcgen05_gemm_blockscaled op (PRs #202 / #1600) is a tensor-core
   FP8 path. Adding an automatic dispatch from T.fp8_scaled_matmul
   to T.tcgen05_gemm_blockscaled when (a) the target architecture
   supports it and (b) the scale layout matches e8m0fnu would buy
   substantial CUDA performance. The current CUDA path uses scalar
   T.cast plus standard fp32 fma -- correct but slow.

## How to apply

```bash
cd /tmp/tilelang_apple_head/tilelang
# Pre-req: storage-only FP8 patch + auto-dequant dispatcher + vector cast.
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch
# Rebuild C++ once for the storage-only/vector FP8 prerequisites.
cd build && cmake --build . -j 8

```

The patch applies cleanly with git apply --check against the post-Agent-F-1
state of the tilelang tree (commit a69d6df7 in the
cppmega/gemm-mixed-dtype-metal branch). Re-run that clean-apply check before
filing; this README is not evidence that the artifact applies to public
`tile-ai/tilelang` `main`.

## Attribution

* **audiohacking/fp8-mps-metal** (commit
  d4fbd40c48aa2a243e600d06627c7dd818150636, MIT). Algorithm: scalar
  dequant, fp32 fma, per-tensor / per-row scale broadcast.
* **AppMana/mps-fp8-for-torch-and-comfyui-python-package** (commit
  a902571eca5362f5e2496cf33dcce52c8bac6a15, Apache 2.0). The
  cppmega.mlx cppmega_mlx.nn._tilelang.fp8_msl_kernels module ports
  this fork's 256-entry LUT decode + integer-bit encoder via
  mx.fast.metal_kernel; it serves as the ground-truth oracle in our
  e2e parity tests.
* **tilelang/tilelang** -- this patch builds on Agent C's storage-only
  FP8 (docs/upstream/tilelang_metal_fp8/), Agent E's auto-dequant
  Gemm dispatcher (docs/upstream/tilelang_metal_fp8_gemm/), and
  Agent F-1's vector FP8 cast (docs/upstream/tilelang_metal_fp8_vector/).
