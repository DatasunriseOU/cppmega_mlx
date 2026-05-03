# Local TVM+TileLang build state (2026-05-03)

## Decision: Option 3 (defer storage-mode patch on TileLang's vendored TVM)

*Why*: TileLang vendors github.com/TileLang/tvm@0e15b274b — a TileLang-maintained fork — not apache/tvm. Our TVM_METAL_STORAGE_MODE patch was authored against apache/tvm@8873a4c and uses TVM_FFI_ICHECK macros that don't exist in TileLang's older vendored copy. Storage-mode patch stays in /Volumes/external/sources/tvm as upstream-PR artifact only.

*The actual TileLang HEAD unlock = PR #2118 "Metal scalar fallback for T.gemm"* is already cherry-picked into apple-head and needs no patching. Path B kernels go through mx.fast.metal_kernel directly — they don't use TVM runtime, so the storage-mode patch is orthogonal.

## TileLang fork state

- *Path*: /tmp/tilelang_apple_head/tilelang
- *Branch*: apple-head
- *Cherry-picked PRs from main*:
  - 9f1297bf PR #2118 "Add Metal scalar fallback for T.gemm"
  - 7f4a5cb8 "Preserve Metal reduce thread range"
  - e75164b2 "fix(metal): harden simdgroup store lowering"
  - 31ca6726 "fix(metal): address swarm eval review followups"
  - e7cbbea0 "fix(metal): harden simdgroup review paths"
  - d5d3c3a6 "style(metal): apply pre-commit formatting"
  - 93f16ee1 "test(metal): tolerate split local.var initialization"
  - 20bb32cf "docs(metal): document internal runtime coverage"
  - b9cd29d5 "test(metal): add internal runtime coverage probes"
  - b5082cce "fix(jit): select mps when cuda is unavailable"
- *3rdparty/tvm*: clean, no local modifications (TileLang/tvm @ 0e15b274b)
- *Build*: cmake + ninja on M-series, ~10 min wall time after resume
- *Artifacts in build/lib/*:
  - libtilelang.dylib (the TileLang runtime)
  - libtvm.dylib (vendored, monolithic — no separate libtvm_runtime)
  - libtvm_ffi.dylib
  - libz3.dylib
  - tilelang_cython_wrapper.{abi3,cpython-313-darwin}.so
- *Workaround applied*: ln -sf libtvm.dylib libtvm_runtime.dylib because tilelang's Python loader looks for libtvm_runtime.dylib but the vendored TVM build only emits libtvm.dylib as a monolithic library.

## Standalone TVM fork

- *Path*: /Volumes/external/sources/tvm
- *Branch*: cppmega/metal-shared-storage-opt-in @ 7cc4ce1
- *Status*: NOT BUILT here. Reserved as upstream-PR artifact only.
- *Patch artifact*: docs/upstream/tvm_shared_storage/0001-metal-shared-storage-opt-in.patch (181 lines, +102/−13).

## cppmega.mlx venv state

- pip install -e tilelang from /tmp/tilelang_apple_head/tilelang — version 0.1.9+git7f4a5cb8
- apache_tvm_ffi-0.1.7 was pulled in as a tilelang dep but is the wheel matching tilelang's vendored build (NOT the previously crashing 0.1.10).
- Vendored libtvm.dylib (with the libtvm_runtime.dylib symlink) is what gets loaded.

## Verification

### Import smoke

$ .venv/bin/python -c "import tilelang; from tilelang import tvm; print(tvm.metal().exist)"
Loading tilelang libs from dev root: /private/tmp/tilelang_apple_head/tilelang/build
True

*No crash.* The earlier EXC_BAD_ACCESS in libtvm_ffi_testing.dylib is gone — that was a PyPI tilelang+apache-tvm-ffi version mismatch which we cleared via uninstall + editable rebuild.

### T.gemm on metal target lowering (the key unlock)

/tmp/test_metal_gemm.py (single T.gemm with shared+fragment):

METAL T.GEMM LOWERING: OK
type: CompiledArtifact


*PR #2118 unlock confirmed.* This was the blocker for sparse-MLA / topk_selector / sparse-MLA FP8.

### Progressive probe (/tmp/test_sparse_mla_pipeline.py)

| Kernel pattern                                    | Status   | Notes                                                                                                   |
| ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------- |
| k1_simple_gemm (1× T.gemm + shared + fragment)    | ✅ OK     | basic case, the unlock                                                                                  |
| k2_pipelined_gemm (T.Pipelined with num_stages=2) | ❌ FAILED | Buffer A_shared is 3-dimensional — multi-stage pipelining creates 3D shared buffers, separate issue     |
| k3_multi_gemm (Q·Kᵀ then ·V chain)                | ❌ FAILED | T.gemm A and B must have the same dtype — dtype mismatch when fp32 accumulator feeds back as fp16 input |

*Verdict*: PR #2118 unblocks the simple case but *does not yet unblock the production sparse-MLA fwd kernel* which uses BOTH:
- T.Pipelined with num_stages > 1 (the 3D buffer issue)
- Chained gemms with fp32 accumulator → fp16 input (dtype constraint)

These are separate upstream issues. Need additional PRs or workarounds.

### cppmega.mlx test status (post-build)


$ .venv/bin/python -m pytest tests/test_tilelang_*.py -q --no-header
103 passed, 3 skipped, 82 warnings in 2.80s


The 3 skipped tests are still gated on hardcoded *_metal_status strings inside our _tilelang/*.py modules — those status strings are stale (not live probes). The kernels themselves never get the chance to attempt lowering because the dispatcher checks the cached status before trying.

To make the unlocked T.gemm path observable in our tests, either:
- Refactor *_metal_status() to do a live lowering probe instead of returning a hardcoded string, OR
- Re-attempt the actual sparse-MLA TileLang lowering and surface the new (different) blocker.

## Path B blocker status — honest update

| Kernel                            | Pre-build status                                                                                     | Post-build status                                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| topk_selector                     | "Unknown storage scope shared.dyn" + injective layout                                                | *Same* — those issues are independent of T.gemm fix                                                       |
| sparse-MLA BF16                   | "Unsupported target for gemm"                                                                        | *Partially lifted*: simple T.gemm works, but the kernel uses Pipelined and chained gemms which still fail |
| sparse-MLA FP8                    | "Unsupported target for gemm" + "float8_e4m3 not Metal type"                                         | *Partially lifted (gemm)*, FP8 dtype lowering still missing in codegen_metal.cc:271                       |
| sparse-MLA blockscaled            | same as FP8                                                                                          | same                                                                                                      |
| Mamba3 main                       | (was unblocked all along — bypasses TileLang lowering, writes MSL directly via mx.fast.metal_kernel) | unchanged ✓                                                                                               |
| Mamba3 helpers (TileLang variant) | (worked at build-time; uses the bypass approach where possible)                                      | unchanged ✓                                                                                               |

## Known issues / next steps

1. *libtvm_runtime.dylib symlink is fragile* — the libtvm.dylib → libtvm_runtime.dylib symlink lives in the build directory and survives only across the editable install. Document this for anyone re-cloning the build.
2. ~~*T.Pipelined num_stages > 1* creates 3D shared buffers that lower fails on~~ — **fixed** (2026-05-02) by *docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch*. The InjectSoftwarePipeline pass prepends a num_stages "version" dim to shared buffers (2D → 3D), which the CUDA macro generators handle via leading-region indices but the Metal MPSIntrinEmitter did not. Patched `tilelang/intrinsics/metal_macro_generator.py` to mirror CUDA's pattern: pull leading dim mins from `BufferRegion.region[:-2]` and prefix them into `T.access_ptr(buffer[...])`. Verified with `test_pipelined_probe.py` — pipelined kernels at num_stages 2, 3, and a Q*K^T attention pattern all lower successfully on Metal target.
3. *T.gemm dtype constraint*: A.dtype == B.dtype enforced. fp32 accumulator pattern needs an explicit cast back to fp16 before the next gemm. Easy fix in our scaffolds when we re-attempt.
4. *FP8 dtype* (float8_e4m3, float8_e5m2) still rejected by codegen_metal.cc:271 — no PR in flight upstream.
5. ~~*shared.dyn storage scope* still rejected~~ — **already fixed in apple-head** (no patch needed). The codegen at `src/target/codegen_metal.cc:493` accepts `shared.dyn` and treats it as `threadgroup`. The remaining limitation is symbolic-extent dynamic shmem (`ICHECK constant_size > 0` at line 494), which is **not on cppmega's Path B path** — topk_selector was rewritten as direct `mx.fast.metal_kernel` MSL. See `docs/upstream/tilelang_metal_shared_dyn/README.md` for full investigation outcome.
6. *float32x4 simdgroup ramp-load* (newly surfaced after item 2 fix): an explicit-K-loop or pipelined kernel with a 32x32 fragment + half output produces `T.Cast("float16x4", BufferLoad(C_local, ramp))`. PointerValueTypeRewrite then vectorises the metal.simdgroup AllocateNode element dtype to float32x4, which `codegen_metal.cc:454` ICHECK-rejects (only scalar float16/float32/bfloat16 allowed in the simdgroup branch). Workaround: keep the output fragment ≤ 16x16 so codegen falls back to `simdgroup_store`. A clean upstream fix needs the codegen to either reject the ramp-load on metal.simdgroup buffers earlier or handle vectorised simdgroup access via lane shuffle. Documented in `docs/upstream/tilelang_metal_pipelined/README.md`.
7. *Status helpers in cppmega_mlx/nn/_tilelang/*.py* return hardcoded strings — they should live-probe or be marked @pytest.fixture style for refresh.

## Files modified

- (this file) docs/upstream/local_build_status.md
- `tilelang/intrinsics/metal_macro_generator.py` (in `/tmp/tilelang_apple_head/tilelang/`)
- `docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch`
- `docs/upstream/tilelang_metal_pipelined/README.md`
- `docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py`
- `docs/upstream/tilelang_metal_shared_dyn/0001-metal-shared-dyn-storage-scope.patch` (no-op artefact)
- `docs/upstream/tilelang_metal_shared_dyn/README.md`
- `docs/upstream/tilelang_metal_shared_dyn/test_shared_dyn_probe.py`

No code changes in cppmega_mlx/. The build state is documented; refactoring our status helpers + re-attempting the BF16 sparse-MLA Path B port (with workarounds for the K-loop and dtype constraints) is the natural follow-up beads issue.

## Update 2026-05-03 — TileLang mixed-dtype T.gemm patch (k3_multi_gemm unblocked)

*Patch artifact*: docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch (170 lines, +169 / -9).
*Local TileLang branch*: cppmega/gemm-mixed-dtype-metal @ a69d6df7 (built off apple-head).
*Files touched* (Python only; no C++ rebuild):
- tilelang/tileop/gemm/gemm_base.py — drop assert A.dtype == B.dtype, add has_mixed_input_dtype property.
- tilelang/tileop/gemm/__init__.py — Metal dispatcher routes mixed-dtype to GemmMetalScalar.
- tilelang/transform/metal_fragment_to_simdgroup.py — exclude mixed-dtype-gemm accumulators (and fragment A operand) from the simdgroup rewrite.
- testing/python/metal/test_metal_codegen_linux.py — new test test_attention_chain_mixed_dtype_metal_codegen.

### Probe re-run after patch

| Kernel pattern                               | Pre-patch                                               | Post-patch                                         |
| -------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- |
| k1_simple_gemm                               | OK                                                      | OK                                                 |
| k2_pipelined_gemm                            | FAILED (3-D buffer from T.Pipelined num_stages=2)       | FAILED — same blocker, not addressed by this patch |
| k3_multi_gemm (Q·Kᵀ → ·V chain, mixed dtype) | FAILED AssertionError: A and B must have the same dtype | *OK*                                               |

### Upstream test impact

- testing/python/metal/: 50 pass / 6 fail (was 49/6) — *+1 net pass*; the previously-failing test_t_gemm_metal_codegen_pipelined_float32 flipped to passing as a side effect of the more conservative simdgroup-rewrite criterion.
- testing/python/cpu/test_tilelang_cpu_tgemm.py: 11 pass (unchanged).
- cppmega.mlx local suite (tests/test_tilelang_*.py): 130 pass (unchanged).

### Path B blocker status — updated

| Kernel                 | Pre-patch status                                      | Post-patch status                                              |
| ---------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| topk_selector          | "Unknown storage scope shared.dyn" + injective layout | unchanged (independent issue)                                  |
| sparse-MLA BF16        | gated on chained-gemm dtype constraint + Pipelined    | *dtype constraint lifted*; Pipelined num_stages>1 still blocks |
| sparse-MLA FP8         | dtype + FP8 codegen                                   | dtype lifted; FP8 codegen still missing                        |
| sparse-MLA blockscaled | same as FP8                                           | same                                                           |

### Why this didn't need a TVM patch

GemmMetalScalar (PR #2118) already emits per-element T.cast(value, accum_dtype) for both A and B reads inside its scalar gemm prim_func. The only missing wiring was (a) the dispatcher pre-check, and (b) the simdgroup-rewrite exclusion. Both are pure-Python changes in TileLang.

### Upstream-PR readiness

This patch is independent from the local FP8 prelude / metal_macro_generator.py work and from the TVM storage-mode patch. Cleanly applies on top of apple-head (which has PR #2118 cherry-picked). For a PR against tile-ai/tilelang:main the patch should apply unchanged once PR #2118 lands upstream.

Backend symmetry note: CUDA / ROCm / Hopper / Blackwell still use the C++ GemmGetGemmInst selection. If they hit a (A.dtype != B.dtype) chain, they'll currently fail downstream in their MMA emitters rather than in GemmBase.in_dtype. Mirroring the dispatcher route to scalar for those backends is out-of-scope for this patch and left for the relevant backend PRs.

## Update 2026-05-02 — TileLang Metal FP8 T.gemm software path (sparse-MLA FP8 unblocked)

Patch artifact: `docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch` (140 lines, +90 / -7).
Local TileLang branch: still on `cppmega/gemm-mixed-dtype-metal` (uncommitted). Pure-Python patch, no C++ rebuild.
Files touched (Python only):
- `tilelang/tileop/gemm/__init__.py` — extend `_select_gemm_instruction` to route FP8 inputs to `GemmInst.Scalar` (mirrors mixed-dtype routing); add `_has_fp8_input_dtype` helper.
- `tilelang/transform/metal_fragment_to_simdgroup.py` — exclude FP8-input GEMMs from the local.fragment → metal.simdgroup rewrite, mirroring the existing mixed-dtype exclusion.

### Test outcome

The canonical FP8 GEMM probe (`/tmp/test_fp8_gemm_metal.py`):

```
FP8 GEMM on metal: OK
MSL contains __tvm_fp8_e4m3_to_half: True
MSL contains simdgroup_multiply_accumulate: False  (correctly: scalar fallback path)
xcrun --sdk macosx metal -c /tmp/test_fp8_gemm.metal: exit 0
```

Emitted MSL inner loop (audiohacking `fp8_scaled_matmul_kernel` pattern):

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

Variants verified to lower OK:
- `T.gemm(e4m3 A, e4m3 B, fp32 C)` — uses `__tvm_fp8_e4m3_to_half` only
- `T.gemm(e5m2 A, e5m2 B, fp32 C)` — uses `__tvm_fp8_e5m2_to_half` only
- `T.gemm(e4m3 A, e5m2 B, fp32 C)` — mixed FP8 (also routes to scalar via the existing mixed-dtype check); uses both helpers

### Upstream test impact

`testing/python/metal/`: 51 pass / 6 fail (was 46/11 baseline pre-patches; +5 net pass).
The 6 remaining failures are pre-existing and unrelated to FP8 routing — 5 are `float32x2` simdgroup-vector allocation bugs, 1 is a stale negative test that asserts FP8 lowering fails (made obsolete by Agent C's storage-only patch).
`testing/python/cpu/test_tilelang_cpu_tgemm.py`: 11 pass (unchanged).
cppmega.mlx local suite (`tests/test_tilelang_*.py`): 134 pass (unchanged).

### Path B blocker status — updated

| Kernel | Pre-patch status | Post-patch status |
|---|---|---|
| topk_selector | "Unknown storage scope shared.dyn" + injective layout | unchanged (independent issue) |
| sparse-MLA BF16 | dtype constraint lifted; Pipelined num_stages>1 still blocks | unchanged |
| sparse-MLA FP8 | dtype lifted; FP8 codegen still missing | **FP8 codegen lifted** (storage-only via Agent C + scalar gemm via this patch); Pipelined still blocks |
| sparse-MLA blockscaled | same as FP8 | **same as FP8** (scalefactor `e8m0fnu` storage works; runtime scaling needs T.cast(scale, fp32) wiring) |

### Why pure Python (Layer 1 not needed)

Agent C's `VisitExpr_(CastNode)` in `codegen_metal.cc` already lowers scalar `T.cast(fp8 -> wider)` into `__tvm_fp8_e4m3_to_half(...)` calls. The TileLang scalar gemm prim_func (`GemmMetalScalar.lower`) emits exactly that scalar cast for each loaded operand. So with the dispatcher routing fix the entire FP8 GEMM body becomes a software dequant-multiply-accumulate loop that mirrors audiohacking `fp8_scaled_matmul_kernel`. Codegen layer 1 (the suggested `EmitFP8GemmSoftware`) was not necessary — the existing pieces compose.

### Performance / production path note

The scalar path is ALU-bound on FP8 decode (one branch + a few shifts per byte) and won't match the throughput of `simdgroup_multiply_accumulate` on FP16. For sparse-MLA FP8 score paths the K-dim is small and the dequant overhead doesn't dominate. For larger GEMMs the production pattern is to pre-dequantise FP8 to FP16 in a fused load kernel and then run the matmul in FP16 with the simdgroup path — this is what audiohacking does for their high-throughput vec-mat case. We can revisit if a kernel becomes performance-critical.

## Update 2026-05-03 — TileLang Metal FP8 vector-cast + scaled-matmul stub

Two new patches landed on top of Agent C's `tilelang_metal_fp8/0001-metal-fp8-storage-only.patch`:

### Patch F-1: vector FP8 cast lowering

- **Path**: `docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch`
- **README**: `docs/upstream/tilelang_metal_fp8_vector/README.md`
- **Files touched** (C++; rebuild required):
  - `src/target/codegen_metal.cc` — add `PrintFP8VectorPrelude` + vector cast routing in `VisitExpr_(CastNode)` for lanes 2/3/4; mirror the prelude splice in `Finish()` via `enable_fp8_vector_`.
  - `src/target/codegen_metal.h` — declare `PrintFP8VectorPrelude` and `enable_fp8_vector_`.
  - `3rdparty/tvm/src/target/source/codegen_metal.{cc,h}` — same change for the parallel `CodeGenMetal` path (apache/tvm half).
- **Effect**: `T.Cast("float16x4", fp8_x4)` (and v2/v3) now lowers to MSL containing `__tvm_fp8_e4m3_to_half_v4` instead of raising `LOG(FATAL)`. Wider widths (lanes 8/16) still FATAL with a clearer message.
- **Rebuild status**: Built incrementally in `/tmp/tilelang_apple_head/tilelang/build/` (~12s, 4 ninja units rebuilt).
- **Test impact** (`testing/python/metal/test_metal_codegen_linux.py`): pre-patch 8 pass / 4 fail → post-patch 9 pass / 3 fail. Net **+1**. The pre-existing 3 failures are unrelated (`metal.simdgroup` vector dtype check). cppmega.mlx tilelang suite: 134 pass (unchanged).
- **Probe**: `/tmp/test_fp8_vector_cast.py` — both phases pass (scalar baseline + vectorized lanes=4).

### Patch F-2: T.fp8_scaled_matmul frontend stub

- **Path**: `docs/upstream/tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch`
- **README**: `docs/upstream/tilelang_metal_fp8_scaled_matmul/README.md`
- **Files touched** (Python only; no rebuild):
  - `tilelang/language/fp8_op.py` (new) — frontend stub with target-aware dispatch.
  - `tilelang/language/__init__.py` — export `fp8_scaled_matmul`.
- **Effect**: `T.fp8_scaled_matmul(A_fp8, A_scale, B_fp8, B_scale, C_out)` exists as a Python frontend. On Metal target it raises `NotImplementedError` with a redirect message pointing at `cppmega_mlx.nn._tilelang.fp8_msl_kernels` (where the vendor agent's `mx.fast.metal_kernel` wrappers will land). On other targets it emits a placeholder `tir.call_intrin` for downstream lowering.
- **Why a stub**: full Metal lowering needs a scaled-gemm scheduler pass that fuses per-load scale into the K-loop and a `GemmMetalScalar` extension for FP8 operands — both sizeable, exceeded the budget. The stub gives users a stable API surface today.
- **Probe**: `/tmp/test_fp8_scaled_matmul.py` — all three phases pass (import surface, Metal redirect, cuda fallback).

### Path B blocker status — updated again

| Kernel | Pre-F1/F2 status | Post-F1/F2 status |
|---|---|---|
| topk_selector | "Unknown storage scope shared.dyn" | unchanged (independent issue) |
| sparse-MLA BF16 | dtype lifted; Pipelined num_stages>1 still blocks | unchanged |
| sparse-MLA FP8 | dtype lifted; FP8 codegen partial (scalar OK, vector FATAL) | **vector lifted**; scaled-gemm scheduler still missing |
| sparse-MLA blockscaled | same as FP8 | same |

### Rebuild log

```
$ cd /tmp/tilelang_apple_head/tilelang/build && ninja -j$(sysctl -n hw.ncpu)
[1/5] Building CXX object tvm/CMakeFiles/tvm_objs.dir/src/target/source/codegen_metal.cc.o
[2/5] Building CXX object CMakeFiles/tilelang_objs.dir/src/target/codegen_metal.cc.o
[3/5] Linking CXX shared library lib/libtvm.dylib
[4/5] Linking CXX shared library lib/libtilelang.dylib
```

Both patches are applied to `/tmp/tilelang_apple_head/tilelang/` (working tree, not committed). Reapply via `git apply <patch>` after rebasing onto a fresh `apple-head`.
