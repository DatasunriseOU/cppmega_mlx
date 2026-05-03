# Local TVM+TileLang build state (2026-05-03)

## Submission pack summary: 9 upstream artifacts

This `docs/upstream/` directory is a community contribution pack for the
macOS / Apple Silicon Metal path across MLX, TVM, and TileLang. The motivation
is practical M4 Max sparse-MLA / TileLang bring-up: remove avoidable Metal
lowering blockers, keep FP8 honest as software storage-only emulation because
Apple GPUs expose no native FP8 scalar or simdgroup matmul type, and make the
MLX <-> TVM DLPack bridge possible without a host copy when both sides use
Shared `MTLBuffer` storage. The pack does **not** claim full sparse-MLA
production parity or FP16 simdgroup throughput for FP8 paths.

Mamba3 note: there are no Mamba3-named upstream patches or probes under
`docs/upstream`. Mamba3 appears here only as downstream motivation for the
Metal / TileLang / TVM interop work; the reviewed submission artifacts below
are not Mamba3-specific PRs.

| # | Directory | Upstream target | Artifact | Purpose | Packaging / apply status | Proof surface | Honest limitation |
|---|---|---|---|---|---|---|---|
| 1 | `mlx_from_dlpack` | `ml-explore/mlx` | `0001-add-from_dlpack-metal-consumer.patch` | Add `mx.from_dlpack(obj)` with a fail-closed Metal Shared-buffer consumer. | Normal patch artifact; apply from the MLX checkout with an absolute path into this repo. | MLX CMake build and `python/tests/test_dlpack_consumer.py` 8/8. | Needs the TVM Shared-storage producer patch for TVM zero-copy; MLX exporter still needs a later kDLMetal phase. |
| 2 | `tvm_shared_storage` | `apache/tvm` | `0001-metal-shared-storage-opt-in.patch` | Add opt-in `TVM_METAL_STORAGE_MODE=shared/managed/private`. | Normal patch artifact; standalone TVM fork artifact, not for TileLang's older vendored TVM. | `runtime_check.mm`, `syntax_check.mm`, and `test_metal_shared_storage.py`. | Must be paired with MLX consumer for end-to-end TVM -> MLX DLPack. |
| 3 | `tilelang_gemm_mixed_dtype` | TileLang Apple-head / Metal-dev branch | `0001-tilelang-allow-mixed-gemm-dtypes.patch` | Let Metal scalar fallback lower chained `fp16 x fp16 -> fp32`, then `fp32 x fp16 -> fp32` attention GEMMs. | Normal branch-specific patch artifact; public `tile-ai/tilelang:main` has drift and needs refresh before PR filing. | `docs/upstream/test_sparse_mla_pipeline.py::k3_multi_gemm`; upstream Metal/CPU test slices documented below. | Non-Metal backend scalar routing remains follow-up work. |
| 4 | `tilelang_metal_pipelined` | TileLang Apple-head / Metal-dev branch | `0001-metal-pipeline-3d-buffer.patch` | Thread the pipeline stage dimension through Metal `T.access_ptr` for `T.Pipelined(num_stages > 1)`. | Normal branch-specific patch artifact; public main currently lacks the touched Metal macro emitter file. | `test_pipelined_probe.py` for `num_stages=2`, `num_stages=3`, and Q*K^T attention shape. | The separate `float32x4` simdgroup vector dtype failure remains open. |
| 5 | `tilelang_metal_fp8` | TileLang + TileLang/tvm or apache/tvm Metal codegen | `0001-metal-fp8-storage-only.patch` | Print FP8 as `uchar` storage and lower scalar FP8 casts through MSL helper functions. | Normal patch artifact. | Scalar FP8 cast probes and `xcrun --sdk macosx metal -c` compile. | Storage/cast only; FP8 GEMM needs the companion dispatcher path and is software, not native FP8. |
| 6 | `tilelang_metal_fp8_gemm` | TileLang | `0001-metal-fp8-gemm-software-path.patch` | Route FP8-input `T.gemm` on Metal to scalar dequant-multiply-accumulate fallback. | **Packaging blocker:** current artifact does not apply on clean `apple-head@7f4a5cb8`, even after `mixed_dtype` + storage-only FP8. Regenerate before PR. | Local branch receipt says FP8 GEMM variants lower and compile to MSL without `simdgroup_multiply_accumulate`. | Receipt is not enough for submission; the stored patch artifact is currently not replayable. |
| 7 | `tilelang_metal_fp8_vector` | TileLang + TVM Metal codegen | `0001-metal-fp8-vector-cast.patch` | Intended follow-up for vector FP8 casts with lanes 2/3/4. | **Packaging blocker:** current file is corrupt; `git apply --check` fails at line 73. Regenerate before any PR. | README records the successful local probe, but the stored patch artifact itself is not applyable. | Cannot be submitted as-is; all dependent stack commands must wait for a regenerated patch. |
| 8 | `tilelang_metal_fp8_scaled_matmul` | TileLang language surface | `0001-tilelang-fp8-scaled-matmul-intrinsic.patch` | Add `T.fp8_scaled_matmul(...)` frontend stub and explicit Metal redirect. | Normal patch artifact, but its documented prereq stack includes the vector patch, which currently needs regeneration. | Import / Metal redirect / fallback probe documented in README. | Frontend stub only; no real Metal scheduler lowering yet. |
| 9 | `tilelang_metal_shared_dyn` | TileLang investigation artifact | `0001-metal-shared-dyn-storage-scope.patch` | Document that `shared.dyn` static extents are already fixed on Apple-head. | Intentional no-op artifact; `git apply --check` reports `No valid patches in input`. | `test_shared_dyn_probe.py` covers static, merged, and symbolic regimes. | Not a code PR; symbolic dynamic shared memory still needs a separate issue/patch if required. |

MLX DLPack performance/readiness note: the `mlx_from_dlpack` artifact
apply-checks cleanly on `ml-explore/mlx@e8ebdeb` and its source has a real
copy boundary difference (`kDLCPU` imports copy into a fresh MLX allocation;
`kDLMetal` Shared-buffer imports wrap the foreign `MTLBuffer` directly). The
pack should still describe this as copy-elision readiness, not a measured
end-to-end speedup, because no checked-in foreign `kDLMetal` producer smoke or
from_dlpack-specific profiler receipt exists yet.

### Current submission order

Treat the table above as the authoritative current readiness snapshot. The
chronological sections below keep the bring-up history and may still quote
pre-patch blockers before later patches resolved them.

File as ordinary code PRs after refreshing against each project's intended PR
base: `mlx_from_dlpack`, `tvm_shared_storage`,
`tilelang_gemm_mixed_dtype`, `tilelang_metal_pipelined`, and
`tilelang_metal_fp8`.

Hold `tilelang_metal_fp8_gemm` and `tilelang_metal_fp8_vector` until their
patch artifacts are regenerated and `git apply --check` passes. Treat
`tilelang_metal_fp8_scaled_matmul` as a frontend-stub PR only; if filed
together with the FP8 vector stack, it is gated by that regenerated vector
patch. Do not file `tilelang_metal_shared_dyn` as a code PR; use it as an
upstream issue / investigation note for the remaining symbolic
dynamic-shared-memory limitation.

### Pack-level proof commands

Run documentation/package checks from the cppmega.mlx checkout:

```bash
cd /Volumes/external/sources/cppmega.mlx
git diff --check -- docs/upstream
rg -n "git apply (docs/upstream|../docs/upstream)" docs/upstream/*/README.md docs/upstream/local_build_status.md
.venv/bin/python -m py_compile \
  docs/upstream/test_metal_gemm.py \
  docs/upstream/test_sparse_mla_pipeline.py \
  docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py \
  docs/upstream/tilelang_metal_shared_dyn/test_shared_dyn_probe.py \
  docs/upstream/tvm_shared_storage/test_metal_shared_storage.py
```

Patch packaging probes:

```bash
git apply --stat docs/upstream/mlx_from_dlpack/0001-add-from_dlpack-metal-consumer.patch
git apply --stat docs/upstream/tvm_shared_storage/0001-metal-shared-storage-opt-in.patch
git apply --stat docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch
git apply --stat docs/upstream/tilelang_metal_fp8/0001-metal-fp8-storage-only.patch
git apply --stat docs/upstream/tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch
git apply --stat docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch

# Expected blockers / non-patches:
git apply --check docs/upstream/tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch
# -> patch failed: tilelang/tileop/gemm/__init__.py / metal_fragment_to_simdgroup.py
git apply --check docs/upstream/tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch
# -> error: corrupt patch at line 73
git apply --check docs/upstream/tilelang_metal_shared_dyn/0001-metal-shared-dyn-storage-scope.patch
# -> error: No valid patches in input
```

## Audit barrier 2026-05-03 — sparse MLA / FP8 upstream artifact replay

Scope: only `docs/upstream` sparse-MLA and FP8 TileLang artifacts were audited.
Clean replay base was a fresh clone from `/tmp/tilelang_apple_head/tilelang`
checked out at `apple-head@7f4a5cb8` with `3rdparty/tvm@0e15b274b`.

Apply-check matrix on that base:

| Artifact | Result | Notes |
|---|---|---|
| `tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch` | OK | Branch-specific; public `tile-ai/tilelang:main@2eec5f0` has drift and lacks `tilelang/intrinsics/metal_macro_generator.py`. |
| `tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch` | OK | Branch-specific; public main drift breaks hunks in tests, dispatcher, and `metal_fragment_to_simdgroup.py`. |
| `tilelang_metal_fp8/0001-metal-fp8-storage-only.patch` | OK | Requires initialized `3rdparty/tvm` submodule. |
| `tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch` | OK | Also applies on current public main as a standalone frontend stub. Its documented full stack still waits on regenerated vector FP8 cast artifact. |
| `tilelang_metal_fp8_vector/0001-metal-fp8-vector-cast.patch` | FAIL | `error: corrupt patch at line 73`; the file switches to prose instead of a valid second unified diff. |
| `tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch` | FAIL | Fails on clean `apple-head` and after applying both `mixed_dtype` and storage-only FP8; hunks for `tilelang/tileop/gemm/__init__.py` and `tilelang/transform/metal_fragment_to_simdgroup.py` need regeneration. |
| `tilelang_metal_shared_dyn/0001-metal-shared-dyn-storage-scope.patch` | NO-OP | `No valid patches in input`; keep as investigation artifact, not a code PR. |

Live scoped probes in the cppmega.mlx venv:

| Command | Result | Meaning |
|---|---|---|
| `.venv/bin/python docs/upstream/test_sparse_mla_pipeline.py` | exit 0; `k1_simple_gemm OK`, `k2_pipelined_gemm FAILED InternalError: Check failed: float32x4`, `k2_32x32_no_pipeline FAILED InternalError: Check failed: float32x4`, `k2_pipelined_16x16_control OK`, `k3_multi_gemm OK` | Mixed-dtype attention chain is lowered; the 32x32 sparse-MLA fragment shape still exposes the separate simdgroup vector dtype blocker even without `T.Pipelined`, while the 16x16 pipelined control proves the 3-D pipeline-buffer fix itself is not the remaining blocker. |
| `.venv/bin/python docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py` | exit 0; `k_pipe_2 OK`, `k_pipe_3 OK`, `k_attn OK` | The 3-D pipeline-region fix works for the included reduced probes. |

### Performance lane 1 — TileLang Metal GEMM runtime baseline

The `docs/upstream/test_metal_gemm.py` probe is now both a pytest lowering
check and a reproducible timing CLI. It does execute on MPS, not just lower:
the runtime path uses `tilelang.compile(..., target="metal",
execution_backend="torch")`, returns a `MetalKernelAdapter`, calls the compiled
kernel with MPS tensors, and verifies the result against `torch.matmul`.

Commands:

```bash
.venv/bin/python docs/upstream/test_metal_gemm.py
.venv/bin/python docs/upstream/test_metal_gemm.py --profile-lowering --lowering-repeats 9
.venv/bin/python docs/upstream/test_metal_gemm.py --profile-runtime --runtime-reps 300 --runtime-warmups 50 --runtime-rounds 7
.venv/bin/python docs/upstream/test_metal_gemm.py --profile-runtime --enable-tilelang-cache --runtime-reps 100 --runtime-warmups 10 --runtime-rounds 3
xctrace record --quiet --no-prompt --template 'Metal System Trace' --time-limit 3s --output /tmp/tilelang_metal_gemm_lane1_final.trace --target-stdout - --launch -- ./.venv/bin/python docs/upstream/test_metal_gemm.py --profile-runtime --runtime-reps 5000 --runtime-warmups 100 --runtime-rounds 20
```

Lowering on this local M-series TileLang dev build
(`/private/tmp/tilelang_apple_head/tilelang/build`) produced MSL with
`kernel void` and `simdgroup_multiply_accumulate`; kernel source length was
1899 bytes. The final sequential nine-repeat lowering run measured min
51.276 ms, median 54.413 ms, and mean 57.180 ms, with the same
generated-source checks.

Runtime timing uses synchronized wall-clock measurements because TileLang's
`tilelang.profiler.do_bench` still follows CUDA-only paths on this checkout
(`torch.cuda.synchronize()`, CUDA cache tensors, CUDA events / CUPTI /
CUDAGraph code paths), while a `torch.mps.Event(enable_timing=True)` smoke
test hung on this machine. The PyTorch baseline is now allocation-free:
`torch.matmul(a, b, out=torch_out)` writes into a preallocated MPS tensor,
the probe verifies that the returned pointer is the same output tensor, and
`runtime_torch_matmul_out_max_abs_vs_torch_matmul` was 0.0.

Four repeated 300-rep / 50-warmup / 7-round runtime runs with TileLang disk
cache disabled by default:

| Run | compile ms | TileLang median ms | PyTorch `matmul(out=...)` median ms | Speedup | max abs vs PyTorch |
|---|---:|---:|---:|---:|---:|
| 1 | 72.366 | 0.004734 | 0.013609 | 2.875x | 0.03125 |
| 2 | 66.549 | 0.004759 | 0.013972 | 2.936x | 0.03515625 |
| 3 | 67.649 | 0.004531 | 0.013672 | 3.018x | 0.03125 |
| 4 | 67.924 | 0.004653 | 0.013672 | 2.938x | 0.03125 |
| 5 final sequential rerun | 92.071 | 0.004745 | 0.014519 | 3.060x | 0.03125 |

A longer `xctrace`-wrapped wall-clock run with 5000 reps / 100 warmups /
20 rounds measured compile 71.789 ms, TileLang median 0.003188 ms, PyTorch
`matmul(out=...)` median 0.017245 ms, speedup 5.409x, and max abs 0.03125.
Use the shorter sequential run above as the conservative baseline; the longer
run is useful mainly because it gives `xctrace` enough time to attach while
preserving the same correctness and generated-source checks.

The short `xctrace` smoke also launched the same probe successfully under the
Metal System Trace template and wrote a 116 MB trace under `/tmp`; the trace
TOC records launched `python` exit status 0 and Metal trace tables. It is not
used as a kernel-counter receipt here because that CLI capture reported
`Counter Set: (null)` and `Shader Timeline: Disabled`.

TileLang disk cache is disabled by default in the runtime mode because the
current Metal adapter has no shared-library path to save. Running with
`--enable-tilelang-cache` reproduces the non-fatal upstream cache bug:
`AttributeError: 'MetalKernelAdapter' object has no attribute 'libpath'` from
`tilelang/cache/kernel_cache.py`, while the kernel still compiles, executes,
and verifies.

No upstream kernel speedup was made in this docs-only lane. The generated
kernel already uses the simdgroup MMA path for this 64x32x64 FP16 probe, and a
real scheduler/codegen speedup would require editing TileLang outside
`docs/upstream`. The safe improvement here is the reproducible runtime mode,
the fair preallocated PyTorch baseline, and the cache/profiler limitation
receipts.

### Performance lane 2 — sparse MLA lowering/profiling probe

The sparse-MLA probe is a source-codegen / lowering probe, not a runtime kernel
benchmark: `tilelang.engine.lower.lower(...)` returns `CompiledArtifact` with
`rt_mod=none` for the passing cases in this script, and no MLX / TVM runtime
adapter is invoked. The status output now prints the artifact detail directly:
`k1_simple_gemm`, `k2_pipelined_16x16_control`, and `k3_multi_gemm` all report
`CompiledArtifact; rt_mod=none` with generated Metal source sizes of 1897,
3101, and 3044 bytes respectively.

The new CLI modes make the compile-only boundary explicit:

```bash
.venv/bin/python docs/upstream/test_sparse_mla_pipeline.py
.venv/bin/python docs/upstream/test_sparse_mla_pipeline.py --time --repeat 3
.venv/bin/python docs/upstream/test_sparse_mla_pipeline.py --profile --kernel k2 --profile-limit 16
.venv/bin/python docs/upstream/test_sparse_mla_pipeline.py --time --repeat 2 --device-compile
```

Measured on the local M-series TileLang dev build (`/private/tmp/tilelang_apple_head/tilelang/build`):

| Kernel | Status | mean ms | median ms | min ms | max ms | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `k1_simple_gemm` | OK | 61.0 | 61.7 | 59.3 | 62.0 | Baseline scalar Metal `T.gemm` lowering; artifact detail shows `rt_mod=none`. |
| `k2_pipelined_gemm` | expected fail | 147.2 | 148.3 | 144.2 | 149.0 | Production-like 32x32/fp32 accumulator + `T.Pipelined`; fails before runtime on `float32x4`. |
| `k2_32x32_no_pipeline` | expected fail | 71.0 | 71.0 | 67.7 | 74.2 | Same 32x32/fp32 accumulator without `T.Pipelined`; proves the current blocker is the simdgroup vector dtype rewrite, not the pipeline-buffer patch. |
| `k2_pipelined_16x16_control` | OK | 110.3 | 109.4 | 108.7 | 112.7 | Pipelined control that avoids `float32x4`; confirms the software-pipeline region path still lowers. |
| `k3_multi_gemm` | OK | 140.7 | 138.7 | 136.8 | 146.4 | Chained mixed-dtype attention GEMMs lower after the mixed-dtype dispatcher patch; artifact detail shows `rt_mod=none`. |

The k2 profiler points at compile/lowering time, not Metal execution time.
For `k2_pipelined_gemm`, cumulative time was concentrated in
`lower.py:271(lower)` -> `transform.py:153(__call__)` -> `phase.py:144(LowerAndLegalize)`;
`gemm_metal.py:21(lower)` accounted for about 51 ms of the 185 ms profiled
sample. The reduced `k2_32x32_no_pipeline` failed in 81 ms and still reached
`gemm_metal.py:82(_gemm_ss_simdgroup)` before the same `float32x4` ICHECK.
Requesting TileLang's Metal device-compile path (`--device-compile`) did not
change the runtime boundary: the passing cases still reported
`CompiledArtifact; rt_mod=none`, while the 32x32 cases failed on the same
`float32x4` check.

The only safe "speedup" available in this probe is a shape workaround for
experimentation: keep the Metal TileLang path at the 16x16/fp16-fragment
control shape, or bypass TileLang for the production 32x32/fp32 sparse-MLA
kernel until the upstream Metal simdgroup allocation path accepts vectorized
`float32x{2,4}` element dtypes or avoids vectorizing that allocation. The
probe intentionally keeps both 32x32 cases as pytest `xfail` so the expected
failure remains visible without breaking scoped CI.

Submission guidance from this audit:

- Submit or refresh branch-specific PRs from the real TileLang Metal branch,
  not from public main, unless rebased first.
- Do not include `tilelang_metal_fp8_vector` or `tilelang_metal_fp8_gemm` in a
  PR pack as-is. Their README motivation is useful, but their patch artifacts
  are not replayable.
- Treat `tilelang_metal_fp8_scaled_matmul` as a frontend-stub PR only. It is
  not proof of a real scaled-FP8 Metal scheduler lowering.
- Treat `tilelang_metal_shared_dyn` as a documented no-op / issue note unless a
  future symbolic dynamic-shared-memory patch is produced.

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
