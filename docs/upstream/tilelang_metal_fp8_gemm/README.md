# TileLang Metal FP8 GEMM — software dequant-and-multiply path

Status: regenerated 2026-05-03; round-trip verified. The change
makes T.gemm(fp8_A, fp8_B, fp32_C) lower cleanly on the Metal target and emit
MSL that compiles with xcrun metal -c. The stored
0001-metal-fp8-gemm-software-path.patch applies cleanly on apple-head@7f4a5cb8
after the mixed-dtype patch (#3) is applied. The patch is independent of the
storage-only FP8 patch (#5) at the file level (touches dispatcher Python files
only), but the runtime correctness of the resulting GEMM depends on #5 being in
place so the per-element T.cast invocations expand to the inline FP8 decode
helpers.

## Stack-apply order

Apply on a clean apple-head@7f4a5cb8 in this order:

1. tilelang_gemm_mixed_dtype (#3) — required prereq; this patch's first hunk
   extends the `Special-case` docstring and the `_select_gemm_instruction`
   metal branch added by #3.
2. tilelang_metal_pipelined (#4) — independent; either order is fine.
3. tilelang_metal_fp8 (#5) — required for runtime correctness (the `T.cast`
   in the scalar fallback expects the FP8 prelude helpers); not required for
   patch-level apply.
4. tilelang_metal_fp8_gemm (#6) — this patch. Verified with
   `git apply --check` and `git apply --reverse --check`.

## Round-trip status

```
git apply               docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch  -> OK
git apply --reverse     docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch  -> OK
```

## Blocker


LOG(FATAL) << "Only float16, float32, and bfloat16 are supported, but got "
           << op->dtype;


Hit at tilelang/src/target/codegen_metal.cc:454 (the metal.simdgroup
allocation check inside VisitStmt_(AllocateNode)). The TileLang Metal
GEMM emitter (tilelang/tileop/gemm/gemm_metal.py) calls
T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")
with in_dtype = "float8_e4m3" because the dispatcher in
tilelang/tileop/gemm/__init__.py::Gemm._select_gemm_instruction returns
GemmInst.METAL_SIMDGROUP whenever A and B share a dtype, even when that
dtype is FP8.

The simdgroup MMA path requires simdgroup_dtypeNxN allocations and the
codegen rejects FP8 there because Metal has no simdgroup_uchar8x8
intrinsic — even Apple M5 NAX cooperative tensors expose only FP16 and
INT8 at the matmul level (per WWDC 2025).

## Apple FP8 reality (May 2026)

Reconfirmed from the storage-only FP8 patch in
docs/upstream/tilelang_metal_fp8/README.md:

| GPU family            | Native FP8 ALU        | Native FP8 simdgroup matmul |
| --------------------- | --------------------- | --------------------------- |
| M1–M3 (Apple7–Apple8) | No                    | No                          |
| M4 / M4 Max (Apple9)  | No                    | No                          |
| M5 NAX (Apple10)      | No (FP16 / INT8 only) | No                          |
| MSL 4.0 / 4.1 / 5.0   | No float8 scalar type | n/a                         |

Therefore any FP8 GEMM on Metal must dequant in software. The correctness
fallback mirrors the audiohacking
[fp8-mps-metal](https://github.com/audiohacking/fp8-mps-metal)
fp8_scaled_matmul_kernel: per-element decode of FP8 to half / float
inside the inner loop, accumulate in float32, store. No simdgroup_matrix
intrinsics involved. The performance path is different: avoid decoding inside
the multiply loop, dequantize a tile once, then feed FP16 data to the existing
Metal simdgroup MMA path.

## Patch design

Two pure-Python changes; no C++ rebuild required.

### Layer 1 — codegen-level FP8 GEMM stub

**Not needed** for the simple (SS-with-fragment-C) case! The TileLang
scalar fallback GemmMetalScalar (PR #2118) already emits per-element
reads with T.cast(value, accum_dtype) for both A and B operands. With
Agent C's FP8 storage-only patch (docs/upstream/tilelang_metal_fp8/)
already in place, those T.cast calls expand at codegen time to
__tvm_fp8_e4m3_to_half(...) / __tvm_fp8_e5m2_to_half(...) helper
calls in MSL. The resulting kernel is the audiohacking software
dequant-and-accumulate pattern.

The only missing wiring is **dispatcher routing** (Layer 2) — the FP8
case must take the scalar path, not the simdgroup path.

### Layer 2 — dispatcher routing

Two pure-Python changes:

1. **tilelang/tileop/gemm/__init__.py** — extend
   Gemm._select_gemm_instruction to route FP8 inputs through
   GemmInst.Scalar (which the existing
   _get_implementation_class then maps to GemmMetalScalar on the
   Metal target). Adds a new helper _has_fp8_input_dtype() that
   detects FP8 by string-prefix match on buffer.dtype (the only
   reliable cross-version signal; tvm.DataType.is_float8 doesn't
   exist as a Python attribute on the vendored TileLang/tvm
   0e15b274b).

2. **tilelang/transform/metal_fragment_to_simdgroup.py** — extend the
   accumulator-rewrite exclusion to FP8 cases. The pass converts
   local.fragment accumulators to metal.simdgroup for GEMMs that
   will use the simdgroup_matrix intrinsic. Mixed-dtype GEMMs were
   already excluded (so the scalar fallback can do its per-element
   casts); FP8 GEMMs need the same exclusion so the C accumulator
   stays in local.fragment (thread float[N]) instead of
   metal.simdgroup (which rejects FP8 — but C is fp32 here, so the
   accumulator promotion would otherwise still happen and trigger
   downstream simdgroup load/store calls that we don't want).

This mirrors the pattern from the existing mixed-dtype patch
(docs/upstream/tilelang_gemm_mixed_dtype/), which is the natural
companion: that patch handles Q@Kt -> S, S@V chains; this patch
handles FP8 in either operand.

## Test results

/tmp/test_fp8_gemm_metal.py (the canonical probe from the task spec):


FP8 GEMM on metal: OK
result type: CompiledArtifact
got source via attribute: kernel_source (len=4403)
MSL contains __tvm_fp8_e4m3_to_half: True
MSL contains simdgroup_multiply_accumulate: False


Sample emitted MSL (just the inner loop):

msl
for (int i_1 = 0; i_1 < 32; ++i_1) {
  for (int j_1 = 0; j_1 < 32; ++j_1) {
    for (int k = 0; k < 64; ++k) {
      float a_val = ((float)(__tvm_fp8_e4m3_to_half(A_shared[((i_1 * 64) + k)])));
      float b_val = ((float)(__tvm_fp8_e4m3_to_half(B_shared[((k * 32) + j_1)])));
      C_local[((i_1 * 32) + j_1)] = (C_local[((i_1 * 32) + j_1)] + (a_val * b_val));
    }
  }
}


This is the audiohacking pattern almost verbatim: load uchar, decode
to half (then promote to float for the accumulator), multiply,
accumulate. No simdgroup_multiply_accumulate. No FP8 simdgroup load.

The kernel **compiles cleanly with xcrun --sdk macosx metal -c**
(exit code 0).

Runtime receipt on MPS for the same 128x64x128 probe:


compile_ms=113.559
adapter_type=MetalKernelAdapter
source_len=4403
source_has_fp8_helper=True
source_has_simdgroup_mma=False
max_abs_vs_cpu_dequant_matmul=0.000000
wall_ms_samples=12.758037,12.773256,12.786633,12.778948,12.814483
wall_ms_median=12.778948


The generated MSL uses device uint4* vectorized global loads into
threadgroup uchar tiles, then decodes both operands in the innermost multiply
loop and accumulates into thread float C_local[1024]. That proves the fallback
is correct and Metal-compilable, but it also proves why it is slow.

## Variants tested

| A dtype     | B dtype     | Lowering       | Helpers emitted          |
| ----------- | ----------- | -------------- | ------------------------ |
| float8_e4m3 | float8_e4m3 | OK             | __tvm_fp8_e4m3_to_half   |
| float8_e5m2 | float8_e5m2 | OK             | __tvm_fp8_e5m2_to_half   |
| float8_e4m3 | float8_e5m2 | OK (mixed FP8) | both e4m3 + e5m2 helpers |

## Upstream test impact

testing/python/metal/: 51 pass / 6 fail (was 46/11 baseline pre-Agent-C +
mixed-dtype patches). The 6 remaining failures are pre-existing and
unrelated to FP8:
- 5 tests fail on float32x2 vector dtype in metal.simdgroup allocation
  (a separate issue in the vectorisation passes), and
- 1 test (test_native_fp8_fp4_metal_storage_fail_closed_in_subprocess)
  is a *negative* test that asserts FP8 lowering fails — now stale because
  Agent C's storage-only patch made it succeed.

testing/python/cpu/test_tilelang_cpu_tgemm.py: 11 pass (unchanged).
cppmega.mlx tests/test_tilelang_*.py: 134 pass (unchanged).

## Diff stat


 tilelang/tileop/gemm/__init__.py                  | 81 +++++++++++++++++++++--
 tilelang/transform/metal_fragment_to_simdgroup.py | 26 ++++++--
 2 files changed, 99 insertions(+), 8 deletions(-)
 (140 lines total, regenerated 2026-05-03)


## Why pure Python (no codegen change needed)

Agent C's VisitExpr_(CastNode) in codegen_metal.cc already lowers
scalar T.cast(fp8 -> half) into __tvm_fp8_e4m3_to_half(...) calls.
The TileLang scalar gemm prim_func emits exactly that scalar cast for
each loaded operand. So the entire FP8 GEMM body becomes a software
dequant-multiply-accumulate loop with zero new codegen helpers. The
prelude (the helper functions) is already injected by Agent C's
Finish() override when enable_fp8_ is set during PrintType /
VisitExpr_(CastNode).

The only thing the pre-existing infrastructure was missing was a
**routing decision**: when do we use simdgroup MMA vs. scalar fallback?
That's the dispatcher's job and it's pure Python.

## Performance and profiler audit

This scalar FP8 fallback is not performance-competitive with the existing FP16
Metal path. The matched FP16 128x64x128 probe emits
simdgroup_multiply_accumulate, has max_abs_vs_cpu_fp16_matmul=0.000000, and
measured:


compile_ms=85.063
source_has_fp8_helper=False
source_has_simdgroup_mma=True
wall_ms_samples=0.011233,0.011771,0.012467,0.012221,0.011419
wall_ms_median=0.011771
profiler_do_bench_error=AssertionError: Torch not compiled with CUDA enabled


On this probe the FP8 software fallback is about 1086x slower than the FP16
simdgroup baseline (12.778948 / 0.011771). Treat the fallback as a correctness
and lowering unblocker for tiny-K or diagnostic paths, not as a large-GEMM
optimization.

The repository-local profiler cannot replace the wall-clock MPS harness today:
tilelang/profiler/bench.py synchronizes CUDA, allocates its cache flush tensor
on device="cuda", and uses torch.cuda.Event. On this Mac it fails with
AssertionError: Torch not compiled with CUDA enabled. The timings above use
time.perf_counter() over repeated kernel launches with torch.mps.synchronize().

There is also a nonfatal cache persistence bug in this local TileLang stack:
both FP8 and FP16 compiles log
AttributeError: 'MetalKernelAdapter' object has no attribute 'libpath' during
atomic cache save. Execution continues and correctness checks pass, but upstream
should fix the cache path separately so Metal kernels can persist like CUDA
artifacts.

For large GEMMs the right shape on Metal is to:

1. Pre-dequantize FP8 to FP16 in a fused load kernel (mx.fast.metal_kernel
   or a separate TileLang prim_func), preferably using packed uchar4 / uint
   loads and grouped conversion to half4 so each FP8 byte is decoded once per
   tile, and
2. Run the actual GEMM in FP16 with the simdgroup path.

That's the same pattern audiohacking uses for their high-throughput
fp8_scaled_vecmat_kernel (which uses simd_sum reduction across 32
lanes; not directly applicable to general GEMM but instructive). For any
sparse-MLA or blockscaled inference use case, the burden of proof is now clear:
measure the real K and tile shape. If K is tiny, scalar fallback may be
acceptable; if K is material, use dequant-then-simdgroup or an MPS/MLX
quantized path instead.

Concrete follow-up optimization target: split the current scalar loop into a
software dequant prologue that writes FP16 tiles into shared memory, then reuse
standard FP16 simdgroup MMA. A smaller improvement would vectorize the current
decode path so loads and helper calls operate on packed groups instead of
independent scalar bytes, but this still cannot close a ~1000x gap while the
kernel remains a scalar triply nested GEMM.

## Upstream PR readiness

This is an intended Python-only change on top of:
- PR #2118 ("Metal scalar fallback for T.gemm")
- docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
  (the codegen FP8 prelude / cast-node patch)
- docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch
  (the mixed-dtype dispatcher patch — its _has_mixed_input_dtype
  helper served as the template for _has_fp8_input_dtype)

May 3 2026 apply matrix (after regeneration):

| Base                                             | Prereqs applied first                          | GEMM patch result                                                   |
| ------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------- |
| public tile-ai/tilelang@2eec5f0, TVM 0e15b274b   | none                                           | FAIL: __init__.py hunk drift; metal_fragment_to_simdgroup.py absent |
| clean apple-head@7f4a5cb8, TVM 0e15b274b         | tilelang_gemm_mixed_dtype                      | OK (apply + reverse)                                                |
| clean apple-head@7f4a5cb8, TVM 0e15b274b         | tilelang_gemm_mixed_dtype + tilelang_metal_fp8 | OK (apply + reverse)                                                |

For tile-ai/tilelang:main, expect public-main drift until the Metal branch
files and PR #2118 stack are reconciled.

## Files

- 0001-metal-fp8-gemm-software-path.patch — the dispatcher patch
- README.md — this document

## How to apply

Passing local receipt:

bash
cd /tmp/cppmega-lane5-recheck-submodule/mixed-head
git checkout a69d6df7
git submodule update --init 3rdparty/tvm
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch


Expected result: both commands pass on a69d6df7. On public main or clean
apple-head@7f4a5cb8, do not file this artifact as-is; regenerate/rebase it
against the intended branch stack first.

## Attribution

The audiohacking
[fp8-mps-metal](https://github.com/audiohacking/fp8-mps-metal) project
provided the reference MSL pattern (fp8_scaled_matmul_kernel). Their
license (MIT) and the inline IEEE-754 decode helpers from their
fp8_matmul.metal informed both the storage-only codegen patch
(docs/upstream/tilelang_metal_fp8/) and this dispatcher routing patch.
