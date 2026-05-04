# TileLang Metal FP8 — storage-only emulation patch

Status: storage-only partial fix shipped. With this patch alone, T.Cast between
float8_e4m3 / float8_e5m2 and float16 (or any fp/int via half) lowers cleanly
on the Metal target. T.gemm(fp8_A, fp8_B, fp32_C) still needs the companion
tilelang_metal_fp8_gemm software fallback patch, or the caller must
explicitly dequantize FP8 to half/float before the gemm.

Packaging note: this storage-only patch is replayable on the local Metal branch
apple-head@7f4a5cb8 with TVM submodule 0e15b274b, but it is not replayable
on public tile-ai/tilelang@2eec5f0 because that public snapshot lacks
TileLang's src/target/codegen_metal.{cc,h} specialization. The companion
tilelang_metal_fp8_gemm README records a local-stack receipt: that GEMM patch
applies cleanly on the mixed-dtype branch head a69d6df7 after this patch, but
not on public main or clean apple-head@7f4a5cb8.

## Blocker


LOG(FATAL) << "Cannot convert type " << t << " to Metal type";


Hit at 3rdparty/tvm/src/target/source/codegen_metal.cc:271 (and the same
spot in TileLang's specialised codegen at src/target/codegen_metal.cc:279)
the moment any float8_e4m3 / float8_e5m2 / float8_e8m0fnu dtype reaches
PrintType. Sparse-MLA FP8, blockscaled, and mxfp8 lowering on Metal target
were unreachable as a result.

CUDA codegen at 3rdparty/tvm/src/target/source/codegen_cuda.cc:410-417
already handles FP8 by emitting __nv_fp8_e4m3 / __nv_fp8_e5m2 types
backed by cuda_fp8.h. Metal had no equivalent path.

## Apple FP8 reality findings (May 2026)

Researched via Apple developer documentation, MSL feature set tables, and
WWDC 2025 sessions on cooperative tensors:

| GPU family            | Native FP8 ALU            | Native FP8 simdgroup matmul |
| --------------------- | ------------------------- | --------------------------- |
| M1–M3 (Apple7–Apple8) | No                        | No                          |
| M4 / M4 Max (Apple9)  | **No**                    | No                          |
| M5 NAX (Apple10)      | **No** — FP16 / INT8 only | No                          |
| MSL 4.0 / 4.1 / 5.0   | No float8 scalar type     | n/a                         |

Apple's MetalPerformancePrimitives matmul2d and the M5 NAX cooperative
tensor primitives announced at WWDC 2025 expose **FP16 and INT8** matmul
intrinsics; they do not expose FP8. Local xcrun --sdk macosx metal probes
confirm the same practical boundary: device uchar* storage compiles, but
float8_t is unknown, float8 is a reserved incomplete type, and
simdgroup_matrix<uchar, ...> fails the Metal stdlib element-type assertion.
The local SDK headers also expose no MTLTensorDataTypeFloat8E4M3 macro.

**Conclusion:** any FP8 path on Metal is purely software emulation. The
right strategy is "storage-only FP8" — pack 8-bit values in uchar /
ucharN buffers, dequant to half on load, do the math in half (or
float accumulator), and quant back on store. This is exactly what
mx.quantized_matmul does for mxfp8 / nvfp4 in MLX core, and what the
TVM stock CUDA codegen does for pre-sm89 hardware.

## Patch design

Two files in two repositories, all changes are additive:

- 3rdparty/tvm/src/target/source/codegen_metal.{cc,h} — TVM's stock Metal
  codegen used by target.build.metal.
- src/target/codegen_metal.{cc,h} — TileLang's CodeGenTileLangMetal
  specialisation used by target.build.tilelang_metal.

For each codegen we add:

1. **PrintType FP8 case.** When t.is_float8() is true, emit uchar for
   lanes==1, ucharN for lanes∈[2,4], uint2 for lanes==8,
   uint4 for lanes==16. Sets enable_fp8_=true so the prelude is
   emitted. Mirrors the CUDA codegen's behaviour where FP8 vectors >4 are
   packed into wider integer storage.

2. **PrintFP8Prelude helper emission.** Inline MSL functions:
    - __tvm_fp8_e4m3_to_half(uchar) -> half
    - __tvm_fp8_e5m2_to_half(uchar) -> half
    - __tvm_half_to_fp8_e4m3(half) -> uchar
    - __tvm_half_to_fp8_e5m2(half) -> uchar

   Encodings follow the OCP "OFP8 Formats for Deep Learning" v1.0 spec.
   E4M3 uses the finite-only encoding (S.1111.111 is NaN, no Inf).
   E5M2 uses IEEE-style with NaN/Inf. Both directions implement
   round-to-nearest-even on the discarded mantissa bits.

3. **VisitExpr_(CastNode) override.** When either side of a cast is an
   FP8 dtype, scalar casts route through the helpers
   (fp8 -> half -> target or from_ty -> half -> fp8). Vector casts
   raise a clear LOG(FATAL) directing the caller to scalarise — the
   TVM lower pipeline already scalarises most user FP8 casts via the
   tir.transform.legalize_fp8 pass, so this branch is rarely hit.

4. **Finish() override.** If enable_fp8_ is set, splice the prelude
   right after using namespace metal; so the helpers see the MSL
   namespace without further qualification.

Arithmetic (binary ops, gemms, reductions) on FP8 operands is **out of
scope.** The TVM legalize_fp8 pass typically expands FP8 arithmetic
into Cast(fp8, op + op + ...) with op = Cast(half, fp8_load) chains,
which our VisitExpr_(CastNode) covers cleanly. For paths that bypass
that pass (e.g. T.gemm with FP8 buffers in metal.simdgroup storage)
the existing simdgroup-allocation check in VisitStmt_(AllocateNode)
fails fast: "Only float16, float32, and bfloat16 are supported".

## Diff stat

Split into two companion patches (May 4 2026):

TileLang half (0001-tilelang-metal-fp8-storage-only.patch):

 src/target/codegen_metal.cc                     | 177 ++++++++++++++++
 src/target/codegen_metal.h                      |  10 +
 2 files changed, 187 insertions(+)

TileLang/tvm half (0002-tvm-metal-fp8-storage-only.patch):

 src/target/source/codegen_metal.cc              | 203 +++++++++++++++++++
 src/target/source/codegen_metal.h               |  10 +
 2 files changed, 213 insertions(+)

Combined: 4 files changed, 400 insertions(+).

Both halves apply cleanly via git apply --check:
- TileLang half against jorgecurious/tilelang:metal-gemm-upstream-rebase.
- TileLang/tvm half against tile-ai/tvm:tilelang_main @ 0e15b274.

May 3 2026 replay audit:

| Base                              | TVM submodule | storage-only patch                                      |
| --------------------------------- | ------------- | ------------------------------------------------------- |
| public tile-ai/tilelang@2eec5f0   | 0e15b274b     | FAIL: public main lacks src/target/codegen_metal.{cc,h} |
| local apple-head@7f4a5cb8         | 0e15b274b     | OK                                                      |
| local mixed-dtype branch a69d6df7 | 0e15b274b     | OK                                                      |

## Test results

All four scalar cast directions lower successfully through TileLang's
lower(target=tvm.target.Target("metal")) and produce well-formed MSL
that **compiles with xcrun metal -c**:


[OK] fp8_e4m3 -> half
[OK] half -> fp8_e4m3
[OK] fp8_e5m2 -> half
[OK] half -> fp8_e5m2


### e4m3 subnormal correctness

The __tvm_fp8_e4m3_to_half helper ships with the corrected biased
exponent for the subnormal path (h = sign | ((e + 7) << 10) | (m << 8))
from the start. After the mantissa-realignment loop normalises the
leading 1 to bit 2 (0x4), the half biased exponent is (e - 9 + 1) +
15 = e + 7, not e + 8. An earlier draft of this helper used (e + 8),
which decoded all seven e4m3 subnormal magnitudes (bytes 0x01-0x07
and 0x81-0x87) at 2x their true value and showed up as ~7% relative
error in T.fp8_scaled_matmul parity tests against PyTorch
torch.float8_e4m3fn / mlx.from_fp8 / the audiohacking reference
kernel. The shipped patch is correct on byte-by-byte parity for the
full e4m3 finite range; the docs/upstream/tilelang_metal_fp8_scaled_matmul
test suite (25/25 passed in 4.14s) covers this end-to-end.

Sample MSL (TileLang codegen, e4m3 round-trip kernel):

msl
#include <metal_stdlib>
using namespace metal;

// FP8 storage-only emulation helpers (MSL has no native float8 type).
inline half __tvm_fp8_e4m3_to_half(uchar x) { ... }
inline half __tvm_fp8_e5m2_to_half(uchar x) { ... }
inline uchar __tvm_half_to_fp8_e4m3(half v) { ... }
inline uchar __tvm_half_to_fp8_e5m2(half v) { ... }

kernel void fp8_round_trip_kernel(
    device uchar* A [[ buffer(0) ]],
    device uchar* B [[ buffer(1) ]],
    device half*  C [[ buffer(2) ]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]]) {
  half x = __tvm_fp8_e4m3_to_half(A[((int)threadIdx.x)]);
  B[((int)threadIdx.x)] = __tvm_half_to_fp8_e4m3((x + 1.000000e+00h));
  C[((int)threadIdx.x)] = x;
}


The stock TVM Metal codegen path (target.build.metal) goes through the
legalize_fp8 pass first, which expands FP8 ops into bit-shuffle code
inline. With this patch its PrintType no longer faults, so the legalised
output also compiles with xcrun metal -c.

With this storage-only patch alone, T.gemm(fp8_A, fp8_B, fp32_C) still fails
at the metal.simdgroup allocation check (TileLang's existing assert at
src/target/codegen_metal.cc:454):
> Only float16, float32, and bfloat16 are supported, but got float8_e4m3

Caller must either apply the companion tilelang_metal_fp8_gemm software
fallback patch or dequantize FP8 to half/float in shared memory before T.gemm.
This matches the FP8 gemm pattern used in
tilelang/examples/deepseek_v32/fp8_lighting_indexer.py.

mxfp8 (float8_e8m0fnu scale storage) also lowers correctly: it's a
device uchar* buffer with no helper calls (just pass-through).

## Performance and profiler audit (May 3 2026)

This patch is a representation and lowering unblocker, not a speedup by
itself. It makes FP8 buffers printable as integer storage and makes scalar
casts explicit in generated MSL. Any arithmetic that reaches Metal still pays
software decode/encode cost unless a higher-level kernel dequantizes once and
then uses the existing FP16/INT8 hardware paths.

Native-FP8 probe commands used:

bash
xcrun --sdk macosx metal -c uchar_storage.metal
xcrun --sdk macosx metal -c float8_t.metal
xcrun --sdk macosx metal -c float8_scalar.metal
xcrun --sdk macosx metal -c simdgroup_uchar.metal
clang++ -fobjc-arc -framework Metal -framework Foundation mtl_tensor_dtype.mm -o mtl_tensor_dtype
rg -n 'MTLTensorDataType.*(Float8|FP8|E4M3|E5M2)|Float8|FP8|E4M3|E5M2' \
  "$(xcrun --sdk macosx --show-sdk-path)/System/Library/Frameworks/Metal.framework/Headers"


Results:

- uchar_storage.metal: OK.
- float8_t.metal: FAIL, unknown type name float8_t.
- float8_scalar.metal: FAIL, float8 is reserved/incomplete.
- simdgroup_uchar.metal: FAIL, invalid simdgroup_matrix element type.
- MTLTensorDataTypeFloat8E4M3: absent from the local Metal headers.

The same SDK capability boundary is now locked by
test_metal_fp8_capability_probe.py; run it from this repository with:

bash
./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8/test_metal_fp8_capability_probe.py -q


TileLang's current profiler path is not usable for MPS timing. The source in
tilelang/profiler/bench.py calls torch.cuda.synchronize(), allocates its
cache flush tensor on device="cuda", and records torch.cuda.Event timings.
On this Mac the README probe reports AssertionError: Torch not compiled with
CUDA enabled. The valid local timing receipt is therefore a wall-clock harness
around the Metal execution backend with torch.mps.synchronize(), not
get_profiler().do_bench(...).

See docs/upstream/tilelang_metal_fp8_gemm/README.md for the GEMM fallback
numbers. The important conclusion for this storage-only patch is narrow: it is
required to make FP8 visible to Metal codegen, but performance work must happen
above it by avoiding repeated per-element conversion.

## Upstream PR readiness for the TileLang/tvm fork

This patch is targeted at the TileLang/tvm fork at
https://github.com/tile-ai/tvm because the TileLang specialisation
(tilelang/src/target/codegen_metal.cc) duplicates the same PrintType
logic. Two separate PRs are appropriate:

- **Upstream apache/tvm PR:** the changes to
  3rdparty/tvm/src/target/source/codegen_metal.{cc,h} are vanilla TVM
  and should land in apache/tvm first. They mirror the existing CUDA
  FP8 storage path and are storage-only (no Apple-specific intrinsic
  use). Suggested PR title: "Metal codegen: add storage-only FP8
  emulation (e4m3 / e5m2 / e8m0fnu)".

- **Downstream tile-ai/tvm cherry-pick:** trivial; the TileLang fork
  carries an extra CodeGenTileLangMetal class with the same
  PrintType body, so the patch must be duplicated there.

- **TileLang PR:** the tilelang/src/target/codegen_metal.{cc,h}
  half can land in tile-ai/tilelang independently of the TVM
  changes, because the TileLang codegen carries its own copy.

Open questions for upstream review:

- The OFP8 e4m3 implementation uses the spec's "finite + NaN" encoding
  (S.1111.111 == NaN, no Inf). Some HuggingFace ml_dtypes libraries
  use float8_e4m3fn for this. We assume kFloat8_e4m3 and
  kFloat8_e4m3fn map to the same MSL helper; if a code path
  differentiates, route the appropriate variant in
  VisitExpr_(CastNode).

- We do not yet handle vector casts (lanes>1) inline; the TVM
  legalize_fp8 pass scalarises these for us today. If any future
  pass emits vector FP8 casts directly to codegen, we need to extend
  the lambda in VisitExpr_(CastNode) to fan out per-lane.

- Helpers are emitted unconditionally if any FP8 dtype is referenced.
  Dead-stripping by Apple's Metal compiler removes unused ones; we
  could narrow the prelude to only the helpers actually called by
  tracking calls inside VisitExpr_, but the cost is ~80 lines of
  inlined IR per kernel which is negligible.

## Files

- 0001-tilelang-metal-fp8-storage-only.patch — TileLang half. Touches
  src/target/codegen_metal.{cc,h} (the CodeGenTileLangMetal
  specialisation). Applies cleanly at the tile-ai/tilelang repo root
  on top of jorgecurious/tilelang:metal-gemm-upstream-rebase.
- 0002-tvm-metal-fp8-storage-only.patch — TileLang/tvm submodule half.
  Touches src/target/source/codegen_metal.{cc,h} in the vendored TVM
  fork. Applies cleanly at the tile-ai/tvm:tilelang_main repo root at
  HEAD 0e15b274 (the SHA TileLang's 3rdparty/tvm submodule pins).

## Filed PRs

- TileLang half: https://github.com/tile-ai/tilelang/pull/2144 (stacks on tile-ai/tilelang#2130)
- TileLang/tvm half: https://github.com/tile-ai/tvm/pull/38 (targets tilelang_main @ 0e15b274)

Both PRs must land for full FP8 storage-only coverage: the tilelang
PR unblocks target.build.tilelang_metal, the tvm PR unblocks
target.build.metal (vanilla TVM Metal codegen used by tilelang's
external_runtime path and any code that lowers via stock TVM).

## How to apply

The patches are companion patches; apply each in its respective repo.

TileLang half (against the supermodule):

bash
cd /tmp && git clone --branch metal-gemm-upstream-rebase https://github.com/jorgecurious/tilelang.git
cd tilelang
git submodule update --init --depth 1 3rdparty/tvm
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0001-tilelang-metal-fp8-storage-only.patch

TileLang/tvm half (against the vendored TVM fork):

bash
cd /tmp && git clone https://github.com/tile-ai/tvm.git tilelang_tvm
cd tilelang_tvm && git checkout 0e15b274bce8b46f971abf5ac390e844aa6acee5
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8/0002-tvm-metal-fp8-storage-only.patch

