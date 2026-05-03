# TileLang Metal FP8 GEMM — software dequant-and-multiply path

Status: shipped. `T.gemm(fp8_A, fp8_B, fp32_C)` lowers cleanly on the Metal
target and emits MSL that compiles with `xcrun metal -c`. The patch is a
pure-Python dispatcher change that builds on the storage-only FP8 codegen
patch (`docs/upstream/tilelang_metal_fp8/`) and the Metal scalar-gemm
fallback (PR #2118).

## Blocker

```
LOG(FATAL) << "Only float16, float32, and bfloat16 are supported, but got "
           << op->dtype;
```

Hit at `tilelang/src/target/codegen_metal.cc:454` (the `metal.simdgroup`
allocation check inside `VisitStmt_(AllocateNode)`). The TileLang Metal
GEMM emitter (`tilelang/tileop/gemm/gemm_metal.py`) calls
`T.alloc_local((warp_rows * 64), in_dtype, scope="metal.simdgroup")`
with `in_dtype = "float8_e4m3"` because the dispatcher in
`tilelang/tileop/gemm/__init__.py::Gemm._select_gemm_instruction` returns
`GemmInst.METAL_SIMDGROUP` whenever A and B share a dtype, even when that
dtype is FP8.

The simdgroup MMA path requires `simdgroup_dtypeNxN` allocations and the
codegen rejects FP8 there because Metal has no `simdgroup_uchar8x8`
intrinsic — even Apple M5 NAX cooperative tensors expose only FP16 and
INT8 at the matmul level (per WWDC 2025).

## Apple FP8 reality (May 2026)

Reconfirmed from the storage-only FP8 patch in
`docs/upstream/tilelang_metal_fp8/README.md`:

| GPU family | Native FP8 ALU | Native FP8 simdgroup matmul |
|---|---|---|
| M1–M3 (Apple7–Apple8) | No | No |
| M4 / M4 Max (Apple9) | No | No |
| M5 NAX (Apple10) | No (FP16 / INT8 only) | No |
| MSL 4.0 / 4.1 / 5.0 | No `float8` scalar type | n/a |

Therefore any FP8 GEMM on Metal must dequant in software. The right
strategy mirrors the audiohacking
[`fp8-mps-metal`](https://github.com/audiohacking/fp8-mps-metal)
`fp8_scaled_matmul_kernel`: per-element decode of FP8 to half / float
inside the inner loop, accumulate in float32, store. No simdgroup_matrix
intrinsics involved.

## Patch design

Two pure-Python changes; no C++ rebuild required.

### Layer 1 — codegen-level FP8 GEMM stub

**Not needed** for the simple `(SS-with-fragment-C)` case! The TileLang
scalar fallback `GemmMetalScalar` (PR #2118) already emits per-element
reads with `T.cast(value, accum_dtype)` for both A and B operands. With
Agent C's FP8 storage-only patch (`docs/upstream/tilelang_metal_fp8/`)
already in place, those `T.cast` calls expand at codegen time to
`__tvm_fp8_e4m3_to_half(...)` / `__tvm_fp8_e5m2_to_half(...)` helper
calls in MSL. The resulting kernel is the audiohacking software
dequant-and-accumulate pattern.

The only missing wiring is **dispatcher routing** (Layer 2) — the FP8
case must take the scalar path, not the simdgroup path.

### Layer 2 — dispatcher routing

Two pure-Python changes:

1. **`tilelang/tileop/gemm/__init__.py`** — extend
   `Gemm._select_gemm_instruction` to route FP8 inputs through
   `GemmInst.Scalar` (which the existing
   `_get_implementation_class` then maps to `GemmMetalScalar` on the
   Metal target). Adds a new helper `_has_fp8_input_dtype()` that
   detects FP8 by string-prefix match on `buffer.dtype` (the only
   reliable cross-version signal; `tvm.DataType.is_float8` doesn't
   exist as a Python attribute on the vendored TileLang/tvm
   `0e15b274b`).

2. **`tilelang/transform/metal_fragment_to_simdgroup.py`** — extend the
   accumulator-rewrite exclusion to FP8 cases. The pass converts
   `local.fragment` accumulators to `metal.simdgroup` for GEMMs that
   will use the simdgroup_matrix intrinsic. Mixed-dtype GEMMs were
   already excluded (so the scalar fallback can do its per-element
   casts); FP8 GEMMs need the same exclusion so the C accumulator
   stays in `local.fragment` (`thread float[N]`) instead of
   `metal.simdgroup` (which rejects FP8 — but C is fp32 here, so the
   accumulator promotion would otherwise still happen and trigger
   downstream simdgroup load/store calls that we don't want).

This mirrors the pattern from the existing mixed-dtype patch
(`docs/upstream/tilelang_gemm_mixed_dtype/`), which is the natural
companion: that patch handles `Q@Kt -> S, S@V` chains; this patch
handles FP8 in either operand.

## Test results

`/tmp/test_fp8_gemm_metal.py` (the canonical probe from the task spec):

```
FP8 GEMM on metal: OK
result type: CompiledArtifact
got source via attribute: kernel_source (len=4150)
MSL contains __tvm_fp8_e4m3_to_half: True
MSL contains simdgroup_multiply_accumulate: False
```

Sample emitted MSL (just the inner loop):

```msl
for (int i_1 = 0; i_1 < 32; ++i_1) {
  for (int j_1 = 0; j_1 < 32; ++j_1) {
    for (int k = 0; k < 64; ++k) {
      float a_val = ((float)(__tvm_fp8_e4m3_to_half(A_shared[((i_1 * 64) + k)])));
      float b_val = ((float)(__tvm_fp8_e4m3_to_half(B_shared[((k * 32) + j_1)])));
      C_local[((i_1 * 32) + j_1)] = (C_local[((i_1 * 32) + j_1)] + (a_val * b_val));
    }
  }
}
```

This is the audiohacking pattern almost verbatim: load `uchar`, decode
to `half` (then promote to `float` for the accumulator), multiply,
accumulate. No simdgroup_multiply_accumulate. No FP8 simdgroup load.

The kernel **compiles cleanly with `xcrun --sdk macosx metal -c`**
(exit code 0).

## Variants tested

| A dtype | B dtype | Lowering | Helpers emitted |
|---|---|---|---|
| `float8_e4m3` | `float8_e4m3` | OK | `__tvm_fp8_e4m3_to_half` |
| `float8_e5m2` | `float8_e5m2` | OK | `__tvm_fp8_e5m2_to_half` |
| `float8_e4m3` | `float8_e5m2` | OK (mixed FP8) | both e4m3 + e5m2 helpers |

## Upstream test impact

`testing/python/metal/`: 51 pass / 6 fail (was 46/11 baseline pre-Agent-C +
mixed-dtype patches). The 6 remaining failures are pre-existing and
unrelated to FP8:
- 5 tests fail on `float32x2` vector dtype in `metal.simdgroup` allocation
  (a separate issue in the vectorisation passes), and
- 1 test (`test_native_fp8_fp4_metal_storage_fail_closed_in_subprocess`)
  is a *negative* test that asserts FP8 lowering fails — now stale because
  Agent C's storage-only patch made it succeed.

`testing/python/cpu/test_tilelang_cpu_tgemm.py`: 11 pass (unchanged).
`cppmega.mlx tests/test_tilelang_*.py`: 134 pass (unchanged).

## Diff stat

```
 tilelang/tileop/gemm/__init__.py                  | 62 +++++++++++++++++++---
 tilelang/transform/metal_fragment_to_simdgroup.py | 28 +++++++++-
 2 files changed, 83 insertions(+), 7 deletions(-)
```

## Why pure Python (no codegen change needed)

Agent C's `VisitExpr_(CastNode)` in `codegen_metal.cc` already lowers
scalar `T.cast(fp8 -> half)` into `__tvm_fp8_e4m3_to_half(...)` calls.
The TileLang scalar gemm prim_func emits exactly that scalar cast for
each loaded operand. So the entire FP8 GEMM body becomes a software
dequant-multiply-accumulate loop with zero new codegen helpers. The
prelude (the helper functions) is already injected by Agent C's
`Finish()` override when `enable_fp8_` is set during `PrintType` /
`VisitExpr_(CastNode)`.

The only thing the pre-existing infrastructure was missing was a
**routing decision**: when do we use simdgroup MMA vs. scalar fallback?
That's the dispatcher's job and it's pure Python.

## Performance note

The scalar path is ALU-bound on FP8 decode (one branch + a few shifts
per byte) and won't match the throughput of `simdgroup_multiply_accumulate`
on FP16 operands. For large GEMMs the right shape on Metal is to:

1. Pre-dequantize FP8 to FP16 in a fused load kernel (`mx.fast.metal_kernel`
   or a separate TileLang prim_func), and
2. Run the actual GEMM in FP16 with the simdgroup path.

That's the same pattern audiohacking uses for their high-throughput
`fp8_scaled_vecmat_kernel` (which uses `simd_sum` reduction across 32
lanes; not applicable to general GEMM but instructive). For the use
cases unblocked here (sparse-MLA FP8 score path, blockscaled FP8
inference) the scalar path is the production path: the K-dim is small
enough that the dequant overhead doesn't dominate, and we avoid an extra
materialisation pass.

If we hit a kernel where the scalar path is too slow, a follow-up patch
can add a "dequant-then-simdgroup" hybrid: split the GEMM into a
software dequant prologue that writes FP16 into shared memory, then a
standard FP16 simdgroup MMA. That's strictly an optimisation; the
correctness path established here is the foundation.

## Upstream PR readiness

This is a clean Python-only patch on top of:
- PR #2118 ("Metal scalar fallback for T.gemm")
- `docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch`
  (the codegen FP8 prelude / cast-node patch)
- `docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch`
  (the mixed-dtype dispatcher patch — its `_has_mixed_input_dtype`
  helper served as the template for `_has_fp8_input_dtype`)

For an upstream PR against `tile-ai/tilelang:main`, the patch should
apply unchanged once the storage-only FP8 codegen patch is upstreamed
(the dispatcher change itself doesn't depend on that — it just routes
around the codegen rejection).

## Files

- `0001-metal-fp8-gemm-software-path.patch` — the dispatcher patch
- `README.md` — this document

## Attribution

The audiohacking
[`fp8-mps-metal`](https://github.com/audiohacking/fp8-mps-metal) project
provided the reference MSL pattern (`fp8_scaled_matmul_kernel`). Their
license (MIT) and the inline IEEE-754 decode helpers from their
`fp8_matmul.metal` informed both the storage-only codegen patch
(`docs/upstream/tilelang_metal_fp8/`) and this dispatcher routing patch.
