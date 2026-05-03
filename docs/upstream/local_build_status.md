# Local TVM+TileLang build state (2026-05-03)

## Decision: Option 3 (defer storage-mode patch on TileLang's vendored TVM)

**Why**: TileLang vendors `github.com/TileLang/tvm@0e15b274b` — a TileLang-maintained fork — not apache/tvm. Our `TVM_METAL_STORAGE_MODE` patch was authored against `apache/tvm@8873a4c` and uses `TVM_FFI_ICHECK` macros that don't exist in TileLang's older vendored copy. Storage-mode patch stays in `/Volumes/external/sources/tvm` as upstream-PR artifact only.

**The actual TileLang HEAD unlock = PR #2118 "Metal scalar fallback for T.gemm"** is already cherry-picked into `apple-head` and needs no patching. Path B kernels go through `mx.fast.metal_kernel` directly — they don't use TVM runtime, so the storage-mode patch is orthogonal.

## TileLang fork state

- **Path**: `/tmp/tilelang_apple_head/tilelang`
- **Branch**: `apple-head`
- **Cherry-picked PRs from main**:
  - `9f1297bf` PR #2118 "Add Metal scalar fallback for T.gemm"
  - `7f4a5cb8` "Preserve Metal reduce thread range"
  - `e75164b2` "fix(metal): harden simdgroup store lowering"
  - `31ca6726` "fix(metal): address swarm eval review followups"
  - `e7cbbea0` "fix(metal): harden simdgroup review paths"
  - `d5d3c3a6` "style(metal): apply pre-commit formatting"
  - `93f16ee1` "test(metal): tolerate split local.var initialization"
  - `20bb32cf` "docs(metal): document internal runtime coverage"
  - `b9cd29d5` "test(metal): add internal runtime coverage probes"
  - `b5082cce` "fix(jit): select mps when cuda is unavailable"
- **3rdparty/tvm**: clean, no local modifications (TileLang/tvm @ 0e15b274b)
- **Build**: cmake + ninja on M-series, ~10 min wall time after resume
- **Artifacts in build/lib/**:
  - `libtilelang.dylib` (the TileLang runtime)
  - `libtvm.dylib` (vendored, monolithic — no separate libtvm_runtime)
  - `libtvm_ffi.dylib`
  - `libz3.dylib`
  - `tilelang_cython_wrapper.{abi3,cpython-313-darwin}.so`
- **Workaround applied**: `ln -sf libtvm.dylib libtvm_runtime.dylib` because tilelang's Python loader looks for `libtvm_runtime.dylib` but the vendored TVM build only emits `libtvm.dylib` as a monolithic library.

## Standalone TVM fork

- **Path**: `/Volumes/external/sources/tvm`
- **Branch**: `cppmega/metal-shared-storage-opt-in` @ `7cc4ce1`
- **Status**: NOT BUILT here. Reserved as upstream-PR artifact only.
- **Patch artifact**: `docs/upstream/tvm_shared_storage/0001-metal-shared-storage-opt-in.patch` (181 lines, +102/−13).

## cppmega.mlx venv state

- `pip install -e tilelang` from `/tmp/tilelang_apple_head/tilelang` — version `0.1.9+git7f4a5cb8`
- `apache_tvm_ffi-0.1.7` was pulled in as a tilelang dep but is the wheel matching tilelang's vendored build (NOT the previously crashing `0.1.10`).
- Vendored `libtvm.dylib` (with the `libtvm_runtime.dylib` symlink) is what gets loaded.

## Verification

### Import smoke
```
$ .venv/bin/python -c "import tilelang; from tilelang import tvm; print(tvm.metal().exist)"
Loading tilelang libs from dev root: /private/tmp/tilelang_apple_head/tilelang/build
True
```
**No crash.** The earlier `EXC_BAD_ACCESS` in `libtvm_ffi_testing.dylib` is gone — that was a PyPI tilelang+apache-tvm-ffi version mismatch which we cleared via uninstall + editable rebuild.

### T.gemm on metal target lowering (the key unlock)

`/tmp/test_metal_gemm.py` (single T.gemm with shared+fragment):
```
METAL T.GEMM LOWERING: OK
type: CompiledArtifact
```

**PR #2118 unlock confirmed.** This was the blocker for sparse-MLA / topk_selector / sparse-MLA FP8.

### Progressive probe (`/tmp/test_sparse_mla_pipeline.py`)

| Kernel pattern | Status | Notes |
|---|---|---|
| `k1_simple_gemm` (1× T.gemm + shared + fragment) | ✅ OK | basic case, the unlock |
| `k2_pipelined_gemm` (T.Pipelined with num_stages=2) | ❌ FAILED | `Buffer A_shared is 3-dimensional` — multi-stage pipelining creates 3D shared buffers, separate issue |
| `k3_multi_gemm` (Q·Kᵀ then ·V chain) | ❌ FAILED | `T.gemm A and B must have the same dtype` — dtype mismatch when fp32 accumulator feeds back as fp16 input |

**Verdict**: PR #2118 unblocks the simple case but **does not yet unblock the production sparse-MLA fwd kernel** which uses BOTH:
- T.Pipelined with num_stages > 1 (the 3D buffer issue)
- Chained gemms with fp32 accumulator → fp16 input (dtype constraint)

These are separate upstream issues. Need additional PRs or workarounds.

### cppmega.mlx test status (post-build)

```
$ .venv/bin/python -m pytest tests/test_tilelang_*.py -q --no-header
103 passed, 3 skipped, 82 warnings in 2.80s
```

The 3 skipped tests are still gated on hardcoded `*_metal_status` strings inside our `_tilelang/*.py` modules — those status strings are stale (not live probes). The kernels themselves never get the chance to attempt lowering because the dispatcher checks the cached status before trying.

To make the unlocked T.gemm path observable in our tests, either:
- Refactor `*_metal_status()` to do a live lowering probe instead of returning a hardcoded string, OR
- Re-attempt the actual sparse-MLA TileLang lowering and surface the new (different) blocker.

## Path B blocker status — honest update

| Kernel | Pre-build status | Post-build status |
|---|---|---|
| topk_selector | "Unknown storage scope shared.dyn" + injective layout | **Same** — those issues are independent of T.gemm fix |
| sparse-MLA BF16 | "Unsupported target for gemm" | **Partially lifted**: simple T.gemm works, but the kernel uses Pipelined and chained gemms which still fail |
| sparse-MLA FP8 | "Unsupported target for gemm" + "float8_e4m3 not Metal type" | **Partially lifted (gemm)**, FP8 dtype lowering still missing in `codegen_metal.cc:271` |
| sparse-MLA blockscaled | same as FP8 | same |
| Mamba3 main | (was unblocked all along — bypasses TileLang lowering, writes MSL directly via `mx.fast.metal_kernel`) | unchanged ✓ |
| Mamba3 helpers (TileLang variant) | (worked at build-time; uses the bypass approach where possible) | unchanged ✓ |

## Known issues / next steps

1. **`libtvm_runtime.dylib` symlink is fragile** — the `libtvm.dylib → libtvm_runtime.dylib` symlink lives in the build directory and survives only across the editable install. Document this for anyone re-cloning the build.
2. **T.Pipelined num_stages > 1** creates 3D shared buffers that lower fails on. Unsupported pattern in TileLang Metal target as of `apple-head`. Workaround for sparse-MLA: rewrite the K-loop without `T.Pipelined`, accept the perf hit, or wait for upstream fix.
3. **T.gemm dtype constraint**: `A.dtype == B.dtype` enforced. fp32 accumulator pattern needs an explicit cast back to fp16 before the next gemm. Easy fix in our scaffolds when we re-attempt.
4. **FP8 dtype** (`float8_e4m3`, `float8_e5m2`) still rejected by `codegen_metal.cc:271` — no PR in flight upstream.
5. **`shared.dyn` storage scope** still rejected (topk_selector blocker) — separate issue from T.gemm.
6. **Status helpers in `cppmega_mlx/nn/_tilelang/*.py`** return hardcoded strings — they should live-probe or be marked `@pytest.fixture` style for refresh.

## Files modified

- (this file) `docs/upstream/local_build_status.md`

No code changes in `cppmega_mlx/`. The build state is documented; refactoring our status helpers + re-attempting the BF16 sparse-MLA Path B port (with workarounds for the K-loop and dtype constraints) is the natural follow-up beads issue.
