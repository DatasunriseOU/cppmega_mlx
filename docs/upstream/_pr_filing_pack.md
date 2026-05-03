# PR Filing Pack - Four Ready Upstream Artifacts

This pack contains everything needed to file the four PR-ready upstream
artifacts listed below manually with `gh pr create`. **Nothing here pushes or
opens PRs**; it only consolidates the materials so the user can paste them into
their PR flow with their own credentials. Other `docs/upstream` artifacts are
not certified by this pack.

## Apple Silicon / MLA motivation

The common motivation is Apple Silicon M4 / M4 Max development of MLA-style
Metal kernels: remove CPU-copy interop friction between TVM and MLX, unblock
mixed-dtype chained GEMM patterns used by attention/MLA probes, and fix
software-pipelined threadgroup indexing so generated MSL can be inspected and
compiled locally before larger accelerator runs. This is a correctness and
developer-loop story, not a claim of FP8 simdgroup speedup.

## Suggested filing order

1. **PR #2 (apache/tvm shared-storage opt-in) FIRST.** It is independent and
   strictly opt-in, has the lowest review-cycle risk (LOW), and is the
   producer half of the zero-copy MLX↔TVM DLPack story. PR #1 references
   it as the paired patch.
2. **PR #1 (ml-explore/mlx from_dlpack)** SECOND. Reference the apache/tvm
   PR # in the body once PR #2 has a number.
3. **PR #3 (tile-ai/tilelang mixed-dtype gemm)** and **PR #4 (tile-ai/tilelang
   pipelined 3D buffer)** THIRD/FOURTH (independent of each other; either
   order works).

---

## PR #2 — apache/tvm: TVM_METAL_STORAGE_MODE env opt-in

### 1. Target repo + base branch
- Repo: `apache/tvm`
- Base branch: `main`

### 2. Source branch (suggested name on the user's fork)
- Local branch: `cppmega/metal-shared-storage-opt-in` in
  `/Volumes/external/sources/tvm` (HEAD `7cc4ce1`)
- Suggested fork-side branch: `cppmega/metal-shared-storage-opt-in`
  (i.e. push the local branch unchanged to the user's `apache/tvm` fork)
- Branched from: `apache/tvm@8873a4c`

### 3. Suggested commit message
Subject (already on the local commit):

```
[Runtime][Metal] add TVM_METAL_STORAGE_MODE env opt-in for Shared/Managed buffers
```

The existing local commit message body is fine to keep verbatim — it covers
motivation, default-preserving behaviour, the cache-once mechanic, and the
new `metal.GetStorageMode` FFI helper. (Subject in the patch artifact reads
"[Metal] Opt-in shared-storage allocation via TVM_METAL_STORAGE_MODE";
prefer the fork's actual subject line `[Runtime][Metal] add
TVM_METAL_STORAGE_MODE env opt-in for Shared/Managed buffers` since that is
what's in `cppmega/metal-shared-storage-opt-in`.)

### 4. PR title (under 70 chars)

```
[Runtime][Metal] Add TVM_METAL_STORAGE_MODE env opt-in
```

(63 chars.)

### 5. PR body

```markdown
## Summary

Adds an opt-in environment variable `TVM_METAL_STORAGE_MODE` that lets users
allocate device data buffers as `MTLResourceStorageModeShared` (or `Managed`)
instead of the default `MTLResourceStorageModePrivate`. Default behaviour is
unchanged.

| value             | mode                                    | semantics                                              |
| ----------------- | --------------------------------------- | ------------------------------------------------------ |
| unset / `private` | `MTLResourceStorageModePrivate`         | default, GPU-only, preserves historical behaviour      |
| `shared`          | `MTLResourceStorageModeShared`          | CPU+GPU mapped — required for zero-copy DLPack to MLX  |
| `managed`         | `MTLResourceStorageModeManaged`         | macOS-only intermediate (driver tracks dirty pages)    |
| anything else     | `MTLResourceStorageModePrivate` + warn  | safe fall-back                                         |

The env var is read once on first `MetalWorkspace::AllocDataSpace` and cached
for the lifetime of the process; no per-allocation overhead. A new FFI helper
`metal.GetStorageMode` is registered alongside the existing
`metal.GetProfileCounters` / `metal.ResetProfileCounters` helpers so tests
can verify the resolved mode without an ObjC bridge.

The staging-buffer pool (`metal_common.h:383`) and temp-buffer pool
(`metal_device_api.mm:374`) already use `MTLStorageModeShared` and are
intentionally untouched — they're host-staging by design and don't fall
under the data-space allocator.

## Why

TVM's Metal device API has always allocated `MTLBuffer` with
`MTLResourceStorageModePrivate`. This is the right choice for pure-GPU
workloads (no CPU page mapping), but it blocks zero-copy DLPack interop with
other Metal-using frameworks that allocate Shared/Managed buffers — notably
`ml-explore/mlx`, which uses `MTLResourceStorageModeShared` everywhere. Two
allocators on the same `MTLDevice` produce buffers with different
page-mapping semantics; DLPack capsules from TVM cannot be consumed by
`mx.array` (live-tested: `std::bad_cast` on `mx.array(tvm_metal_capsule)`).

This change unblocks the bridge from TVM-NDArray to `mlx.array` (both wrap
`MTLBuffer`; require matching storage mode for the same foreign capsule to
be consumable). It is the producer half of a pair; the consumer half is a
parallel ml-explore/mlx PR that adds `mx.from_dlpack(obj)`.

## Test plan

- [ ] `xcrun --sdk macosx clang++ -std=c++17 -framework Metal syntax_check.mm
      -o syntax_check && ./syntax_check` — exercises env-var parsing for all
      6 cases (unset, shared, mixed-case Shared, invalid, managed, private).
- [ ] Build runtime: `mkdir build && cd build && cmake -DUSE_METAL=ON
      -DUSE_LLVM=ON -DCMAKE_BUILD_TYPE=Release .. && make -j tvm_runtime`
- [ ] `./runtime_check` (TVM-linked probe) — validates that the env var
      flows to a real `MTLBuffer.storageMode`. Live captured 2026-05-03 on
      Apple M4 Max for unset/shared/managed/private.
- [ ] `TVM_METAL_STORAGE_MODE=shared python -c "import tvm; arr =
      tvm.nd.empty((4,), dtype='float32', device=tvm.metal()); print(arr.shape)"`
- [ ] CI: macos-arm64 runner in apache/tvm should exercise the existing
      Metal tests; default behaviour (env unset) is unchanged.

## Caveats / non-goals

- This is a **copy-elision interop patch**, not a kernel-speed patch. Default
  Private mode remains the right choice for TVM-only workloads.
- The patch artifact only changes `src/runtime/metal/metal_device_api.mm`;
  it does not yet add an upstream `tests/python/runtime/...` file. A
  subprocess-isolated Python test for the env-cache behaviour can be folded
  in if maintainers want it in tree (downstream
  `test_metal_shared_storage.py` is available and ready to upstream).
- Local Metal microbenchmarks on Apple M4 Max show Shared buffers remove
  the staging-buffer + blit/wait cost at CPU↔Metal transfer boundaries
  (e.g., 1 MiB CPU→Metal median 138.375 µs Private vs 12.750 µs Shared in
  the downstream probe). These numbers are local-health checks, not
  in-tree benchmarks.

## Attribution

Co-developed with `cppmega.mlx` for Apple-Silicon Metal interop with MLX.
```

### 6. Required pre-filing steps

```bash
cd /Volumes/external/sources/tvm
git fetch origin
git checkout cppmega/metal-shared-storage-opt-in
git rebase origin/main          # branched from 8873a4c, drift expected to be small
# If drift breaks the patch:
git format-patch -1 HEAD --output=/tmp/0001-metal-shared-storage-opt-in.patch
diff /tmp/0001-metal-shared-storage-opt-in.patch \
  /Volumes/external/sources/cppmega.mlx/docs/upstream/tvm_shared_storage/0001-metal-shared-storage-opt-in.patch
# (do NOT overwrite the authoritative patch under docs/upstream/)
```

Then push the rebased branch to the user's apache/tvm fork:

```bash
git push <user-fork-remote> cppmega/metal-shared-storage-opt-in
```

### 7. Test command for reviewers

```bash
# Build runtime with Metal:
mkdir build && cd build
cmake -DUSE_METAL=ON -DUSE_LLVM=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu) tvm_runtime

# Verify env var flows to MTLBuffer.storageMode:
TVM_METAL_STORAGE_MODE=shared python -c "
import tvm
print('mode:', tvm.get_global_func('metal.GetStorageMode')())
arr = tvm.nd.empty((4,), dtype='float32', device=tvm.metal())
print('alloc OK:', arr.shape)
"
# Expect: mode: shared, alloc OK: (4,)
```

### 8. Reviewer-targeting hints

- TVM uses a non-standard `.github/CODEOWNERSHIP` (intentionally renamed so
  GitHub does not auto-request review) — anyone can review.
- Recent `src/runtime/metal/` committers (last ~6 months):
  - `Akaash Parthasarathy` — `[Metal] Include logging headers for metal (#19493)`
  - `Tianqi Chen` (`@tqchen`) — refactors across runtime
  - `Miti` — `[Metal] Batched command dispatch and staging buffer pool (#18877)`
- Suggest @-mention: `@tqchen` (apache/tvm-committers; broad runtime ownership).
- Mention the paired `ml-explore/mlx` PR # once it exists.

### Patch artifact (absolute path)

`/Volumes/external/sources/cppmega.mlx/docs/upstream/tvm_shared_storage/0001-metal-shared-storage-opt-in.patch`

---

## PR #1 — ml-explore/mlx: from_dlpack Metal-aware consumer

### 1. Target repo + base branch
- Repo: `ml-explore/mlx`
- Base branch: `main`

### 2. Source branch (suggested name on the user's fork)
- Local branch: `cppmega/from-dlpack-metal-consumer` in
  `/Volumes/external/sources/mlx` (HEAD `22fc6b2`)
- Suggested fork-side branch: `cppmega/from-dlpack-metal-consumer`
- Branched from: `ml-explore/mlx@e8ebdeb`

### 3. Suggested commit message

Subject (already on the local commit, matches the patch artifact):

```
[Python] add mx.from_dlpack(obj) Metal-aware consumer
```

Body — reuse the patch's existing message verbatim. It already covers the
dispatch table, used-capsule rejection, strided-view rejection, motivation
(zero-copy interop with TVM-NDArray and mlx-mfa), and references the paired
TVM patch.

### 4. PR title (under 70 chars)

```
[Python] Add mx.from_dlpack(obj) Metal-aware consumer
```

(53 chars.)

### 5. PR body

```markdown
## Summary

Adds a top-level `mx.from_dlpack(obj)` that consumes either a raw `PyCapsule`
(named `dltensor` or `dltensor_versioned`) or any object whose `__dlpack__()`
chain yields one (up to 4 unwrap iterations). Dispatch is by
`DLDevice.device_type`:

| device_type   | behavior                                                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kDLCPU (1)    | Copy into a fresh MLX allocation; producer's deleter runs immediately after copy                                                                                                    |
| kDLMetal (8)  | Wrap the foreign `MTLBuffer` as `mx.array` via the existing `array(allocator::Buffer, Shape, Dtype, Deleter)` constructor. Storage mode validated to `MTLResourceStorageModeShared` |
| Any other     | Rejected explicitly (kDLCUDA, kDLROCM, ...)                                                                                                                                         |

Used capsules detected and rejected on second consume. Non-contiguous strided
views rejected with a clear error.

8 unit tests pass on a clean cmake build (`cmake -DMLX_BUILD_METAL=ON
-DMLX_BUILD_PYTHON_BINDINGS=ON`). Total ~452 LOC C++ + 90 LOC tests.

## Why

MLX 0.31.x has `mx.array.__dlpack__()` and `__dlpack_device__()` — export
works, but:

- `mx.array.__dlpack_device__()` advertises `(8, 0)` (`kDLMetal`) when Metal
  is available.
- The exporter at `python/src/convert.cpp:100-159` emits the actual capsule
  with `device_type=1` (`kDLCPU`) — uses `a.data<T>()` (host pointer)
  without a `nb::device::metal` annotation.
- There is **no** `mx.from_dlpack(obj)` at all.
- `mx.array(tvm_metal_tensor)` falls into `create_array` →
  `to_array_with_accessor` and fails with `bad_cast`.

So MLX is currently a one-way `kDLCPU` producer despite advertising
`kDLMetal`. Any zero-copy interop with TVM-NDArray, mlx-mfa, or other Apple
Silicon Metal producers requires a host roundtrip — defeating the whole
point of DLPack.

This PR adds the consumer half. With the paired `apache/tvm`
`TVM_METAL_STORAGE_MODE=shared` opt-in (linked below), two allocators on
the same `MTLDevice` can finally share an `MTLBuffer` without `memcpy`.

## Storage-mode design choice

We reject anything other than `MTLStorageModeShared`. The MLX runtime
expects `Buffer::raw_ptr()` to return a host-readable pointer; relaxing
this would silently break kernel preludes that call `MTL::Buffer::contents()`
(which only returns non-null for shared/managed storage per
`mlx/backend/metal/allocator.cpp:14-15, 23-28, 196-207`).

The error message tells the producer exactly what to fix — they need to
allocate with `MTLResourceStorageModeShared`. For TVM, the paired patch
provides `TVM_METAL_STORAGE_MODE=shared`.

## Test plan

- [ ] `cd /path/to/mlx && mkdir build && cd build && cmake
      -DMLX_BUILD_METAL=ON -DMLX_BUILD_PYTHON_BINDINGS=ON .. &&
      make -j$(nproc) core` (~3-4 min on M-series)
- [ ] `cd python && pip install -e .`
- [ ] `python -m pytest python/tests/test_dlpack_consumer.py -v` — expect
      8/8 pass:
  1. `test_function_exists` — `mx.from_dlpack` is bound
  2. `test_round_trip_via_numpy` — kDLCPU path: `np.ndarray` →
     `mx.from_dlpack` → `mx.array`
  3. `test_round_trip_via_capsule` — raw PyCapsule (named `dltensor`)
  4. `test_self_round_trip` — `mx.array` → `__dlpack__` → `from_dlpack`
  5. `test_dtypes` — 13 dtype cases (bool, int8/16/32/64, uint8/16/32/64,
     float16/32/64, complex64)
  6. `test_rejects_non_dlpack_object`
  7. `test_rejects_used_capsule`
  8. `test_strided_view_rejected`
- [ ] Pre-existing `test_dlpack` and `test_dlpack_device_type` in
      `python/tests/test_array.py` still pass (no regression).

## Caveats

- The current tests exercise the CPU path, raw capsule handling, self
  round-trip, dtype mapping, used-capsule rejection, and strided
  rejection. They **do not yet** construct a foreign `kDLMetal` capsule
  backed by a non-MLX `MTL::Buffer`, so the zero-copy Metal wrapping path
  and the non-Shared-storage rejection path are covered by code review
  only. We can add either a small ObjC++ test producer or a TVM
  integration smoke as a follow-up.
- Treat the Metal path as a **copy-elision interop patch**, not as a
  measured kernel speedup. The honest performance claim is narrower:
  once paired with a Shared-`MTLBuffer` producer such as the TVM
  storage-mode opt-in, this consumer can avoid the host-copy import path;
  timing that bridge still needs the foreign producer test.
- DLPack v1 follow-ups left for cycle 2: `flag_bitmask_read_only`,
  `stream=` keyword, `byte_offset`, strided import.

## Follow-up (not in this PR)

A separate **Phase 2** PR will fix MLX's exporter
(`python/src/convert.cpp:100-159`) to actually emit `kDLMetal` capsules
when running on Metal — currently the device_type advertised by
`__dlpack_device__()` and the device_type of the emitted capsule disagree.
Estimated ~30-50 LOC follow-up to complete the bidirectional zero-copy
story. **Not included here.**

## Pairing

Paired upstream patch: apache/tvm `TVM_METAL_STORAGE_MODE` env opt-in
(linked above). Both patches must land for the zero-copy MLX↔TVM use case
to work end-to-end. mlx-mfa (which inherits MLX's allocator) is
Shared-by-default and works out of the box with this patch alone.

## Attribution

Co-developed with `cppmega.mlx` for Apple-Silicon Metal interop.
```

### 6. Required pre-filing steps

```bash
cd /Volumes/external/sources/mlx
git fetch origin
git checkout cppmega/from-dlpack-metal-consumer
git rebase origin/main          # branched from e8ebdeb; drift expected to be small
# If drift breaks the patch:
git apply --check /Volumes/external/sources/cppmega.mlx/docs/upstream/mlx_from_dlpack/0001-add-from_dlpack-metal-consumer.patch
# Regenerate locally (do NOT touch the authoritative artifact):
git format-patch -1 HEAD --output=/tmp/0001-add-from_dlpack-metal-consumer.patch
```

Then push to the user's `ml-explore/mlx` fork:

```bash
git push <user-fork-remote> cppmega/from-dlpack-metal-consumer
```

### 7. Test command for reviewers

```bash
cd /path/to/mlx
mkdir -p build && cd build
cmake -DMLX_BUILD_METAL=ON -DMLX_BUILD_PYTHON_BINDINGS=ON ..
make -j$(sysctl -n hw.ncpu) core
cd ../python
pip install -e .
python -m pytest python/tests/test_dlpack_consumer.py -v
# Expect 8/8 pass.

# Sanity check binding:
python -c "
import mlx.core as mx
import numpy as np
a = mx.from_dlpack(np.arange(12).reshape(3, 4))
print('from_dlpack OK:', a.shape, a.dtype)
"
```

### 8. Reviewer-targeting hints

- ml-explore/mlx has no `CODEOWNERS` file in `.github/`.
- Recent committers around `python/src/convert.cpp` and DLPack code:
  - `@awni` (Awni Hannun) — primary maintainer, broad ownership across
    `python/src` and array creation. **Top suggestion to @-mention.**
  - `@angeloskath` (Angelos Katharopoulos) — recent fixes in array creation
    (`Fix regression in array creation (#3353)`).
  - `@cheng-jl` / `@frost-intel` — Windows CI / infra; less relevant here.
- Suggest @-mention: `@awni`, `@angeloskath`.
- In the PR body, reference the paired apache/tvm PR # once it exists.

### Patch artifact (absolute path)

`/Volumes/external/sources/cppmega.mlx/docs/upstream/mlx_from_dlpack/0001-add-from_dlpack-metal-consumer.patch`

---

## PR #3 — tile-ai/tilelang: allow mixed-dtype T.gemm via Metal scalar fallback

### 1. Target repo + base branch
- Repo: `tile-ai/tilelang`
- **Base branch: `apple-head` (preferred). The patch DOES NOT apply
  cleanly to public `main`** — `tilelang/transform/metal_fragment_to_simdgroup.py`
  is absent from public main, and several test/__init__ hunks no longer
  match.
- Note for the user: `apple-head` is the local Metal-dev branch in the
  user's `cppmega/gemm-mixed-dtype-metal` fork. If `tile-ai/tilelang`
  upstream does not accept PRs against a non-default branch (most likely
  it will), the PR must be **rebased onto public main**, which requires
  also bringing in the `GemmMetalScalar` lowering work (PR #2118) and the
  `MetalFragmentToSimdgroup` transform — significantly larger scope.
  **Recommendation: file against `apple-head` and ask maintainers to merge
  it forward; or carry it as a fork-only patch until the Metal track lands
  on main.**

### 2. Source branch (suggested name on the user's fork)
- Local branch: `cppmega/gemm-mixed-dtype-metal` in
  `/tmp/tilelang_apple_head/tilelang` (HEAD `a69d6df7`)
- Suggested fork-side branch: `cppmega/gemm-mixed-dtype-metal`
- Branched from: `tile-ai/tilelang@7f4a5cb8` (apple-head HEAD: "Preserve
  Metal reduce thread range")

### 3. Suggested commit message

Subject (already on the local commit):

```
tilelang: allow mixed-dtype T.gemm via Metal scalar fallback
```

The existing local commit body is fine to keep verbatim — covers the
3-step fix (drop assert + dispatch to scalar + skip simdgroup rewrite),
the chained-attention motivation, and the conservative-by-default design.

### 4. PR title (under 70 chars)

```
fix(metal): allow mixed-dtype T.gemm via scalar fallback
```

(56 chars.)

### 5. PR body

```markdown
## Summary

Drops the unconditional `A.dtype == B.dtype` assert in
`GemmBase.in_dtype` and routes mixed-dtype Metal GEMMs to the existing
`GemmMetalScalar` fallback added by PR #2118. Updates the
`MetalFragmentToSimdgroup` transform to skip accumulator vars produced
by mixed-dtype GEMMs and the fragment operand A of a mixed-dtype GEMM
(otherwise the scalar fallback would dereference a `simdgroup_floatNN`
register elementwise, which Metal codegen rejects).

| File                                              | +/-           |
| ------------------------------------------------- | ------------- |
| tilelang/tileop/gemm/gemm_base.py                 | +22 / -1      |
| tilelang/tileop/gemm/__init__.py                  | +30 / -0      |
| tilelang/transform/metal_fragment_to_simdgroup.py | +63 / -7      |
| testing/python/metal/test_metal_codegen_linux.py  | +55 / -0      |
| **Total**                                         | **+170 / -8** |

Python-only, zero C++ changes.

## Why

`tilelang/tileop/gemm/gemm_base.py:88` asserts `self.A.dtype ==
self.B.dtype`, which blocks chained mixed-precision patterns. The
canonical case is the two-step attention path:

```python
T.gemm(Q_shared, K_shared, S_local, transpose_B=True)   # fp16 × fp16 → fp32
T.gemm(S_local, V_shared, O_local)                       # fp32 × fp16 → fp32
```

The second call has `A.dtype = float32` (the accumulator from the first
GEMM) and `B.dtype = float16`, tripping the assert before any backend
dispatch could choose how to handle it.

This is overly conservative. cuBLAS, CUTLASS, MPS BNNS, and MSL
`simdgroup_matrix_multiply` accept different precisions for A/B (or for
the accumulator C). The fp16-input/fp32-accumulator case is canonical.

## Test plan

- [ ] `cd /path/to/tilelang && git apply
      /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch`
- [ ] `pytest testing/python/metal/test_metal_codegen_linux.py::test_attention_chain_mixed_dtype_metal_codegen -v`
      (new test added by this patch).
- [ ] `pytest testing/python/metal/` — expect 50 passed, 6 failed,
      3 skipped (vs. baseline 49 passed, 6 failed, 3 skipped — net +1
      passing, no regressions). The 6 baseline metal-test failures are
      pre-existing; this patch does not touch their failure paths.
- [ ] `pytest testing/python/cpu/test_tilelang_cpu_tgemm.py` — 11 passed
      (no regressions; CPU scalar already handles mixed dtypes).
- [ ] Probe: `python /Volumes/external/sources/cppmega.mlx/docs/upstream/test_sparse_mla_pipeline.py`
      — `k3_multi_gemm` lowers OK (was: AssertionError "A and B must have
      the same dtype").

## Caveats

- **Targets `apple-head`.** A fresh `tile-ai/tilelang` `main` checkout at
  `2eec5f0` does not contain
  `tilelang/transform/metal_fragment_to_simdgroup.py` and several other
  hunks fail. Refresh the patch against the actual upstream PR base
  before submitting.
- **Profiler/runtime not green for the chained probe.** The current MSL
  generated for `k3_multi_gemm` fails `xcrun --sdk macosx metal -c`
  because the first same-dtype GEMM still attempts the simdgroup path
  while its consumer (the mixed-dtype scalar GEMM) leaves `S_local` as
  `thread float`. A safe upstream fix has to handle that
  producer/consumer conflict explicitly (e.g., route every GEMM that
  touches a mixed-consumed fragment through the scalar path, or
  materialize a separate scalar/cast copy for the second GEMM). This PR
  is the **frontend-unblock** part; the producer/consumer co-design is
  a follow-up.
- One follow-up could mirror this on CUDA / ROCm: route to a scalar
  fallback when `A.dtype != B.dtype` and the chosen MMA emitter would
  refuse same-dtype inputs. Out of scope here.

## Backwards compatibility

When `A` and `B` share a dtype, the dispatch and rewrite both produce the
same IR as before — strict no-op for the existing same-dtype hot path.

## Attribution

Builds on PR #2118 (`Add Metal scalar fallback for T.gemm`,
`@chenkailun.c`). This PR routes mixed-dtype inputs through that
fallback.
```

### 6. Required pre-filing steps

```bash
cd /tmp/tilelang_apple_head/tilelang
git fetch origin
git checkout cppmega/gemm-mixed-dtype-metal
# Verify still applies cleanly on apple-head:
git rebase origin/apple-head 2>/dev/null || \
  echo "drift on apple-head — refresh patch against the actual PR base"

# Public-main drift check (informational only, do NOT rebase onto main
# unless you also bring in PR #2118 and the simdgroup transform):
git checkout -B /tmp/check-public-main origin/main 2>/dev/null || true
git apply --check \
  /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch
# Expected: error: "tilelang/transform/metal_fragment_to_simdgroup.py: No such file or directory".
```

Push to fork:

```bash
git checkout cppmega/gemm-mixed-dtype-metal
git push <user-fork-remote> cppmega/gemm-mixed-dtype-metal
```

### 7. Test command for reviewers

```bash
cd /path/to/tilelang   # apple-head checkout
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch
# Python-only patch; no rebuild needed:
ninja -C build tilelang_objs   # only if a C++ rebuild was already pending

# New test:
.venv/bin/python -m pytest testing/python/metal/test_metal_codegen_linux.py::test_attention_chain_mixed_dtype_metal_codegen -v

# Regression suite:
.venv/bin/python -m pytest testing/python/metal/ -q
# Expect: 50 passed, 6 failed (pre-existing), 3 skipped.

# Probe:
.venv/bin/python /Volumes/external/sources/cppmega.mlx/docs/upstream/test_sparse_mla_pipeline.py
# Expect: k1_simple_gemm OK, k3_multi_gemm OK, k2_pipelined_gemm FAILED (separate baseline issue).
```

### 8. Reviewer-targeting hints

- tile-ai/tilelang has no `CODEOWNERS` file.
- Recent committers around `tilelang/intrinsics/metal_*` and
  `tilelang/tileop/gemm/`:
  - `Jorge C` — `fix(metal): harden simdgroup store lowering`,
    `harden simdgroup review paths`. **Top suggestion to @-mention.**
  - `chenkailun.c` — author of PR #2118 (`Add Metal scalar fallback for
    T.gemm`); this PR builds directly on their work.
  - `Yichen Yan` — `[Metal] Add Metal GEMM support with simdgroup_matrix MMA`.
- Suggest @-mention: the author of PR #2118 plus the `Jorge C` committer
  for the simdgroup transform. (GitHub handles unknown — the user should
  look up handles in the tile-ai/tilelang PR list before tagging.)

### Patch artifact (absolute path)

`/Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_gemm_mixed_dtype/0001-tilelang-allow-mixed-gemm-dtypes.patch`

---

## PR #4 — tile-ai/tilelang: Metal pipeline 3D-buffer indexing fix

### 1. Target repo + base branch
- Repo: `tile-ai/tilelang`
- **Base branch: `apple-head` (required). The patch CANNOT apply to public
  `main`** — `tilelang/intrinsics/metal_macro_generator.py` does not exist
  on public main (`git apply --check` reports `No such file or directory`).
  Either land on the apple-head fork, or rebase onto main only after the
  Metal macro emitter has landed there.

### 2. Source branch (suggested name on the user's fork)
- Local branch: not yet committed to the user's fork as a separate
  branch — the patch lives only as
  `/Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch`.
  Apply it on top of `apple-head`:
  ```bash
  cd /tmp/tilelang_apple_head/tilelang
  git checkout -b cppmega/metal-pipeline-3d-buffer apple-head
  git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch
  git add -A
  git commit -m "fix(metal): propagate pipelined version index in MPSIntrinEmitter access_ptr"
  ```
- Suggested fork-side branch: `cppmega/metal-pipeline-3d-buffer`

### 3. Suggested commit message

Subject:

```
fix(metal): propagate pipelined version index in MPSIntrinEmitter access_ptr
```

Body:

```
T.Pipelined(..., num_stages=N) with N > 1 lowers fine on the CUDA target
but fails to lower on the Metal target with:

    IndexError: Buffer A_shared is 3-dimensional (shape=[2, 32, 32]),
    but 2 index(es) were provided: (row_idx, col_idx).

InjectSoftwarePipeline (inject_pipeline.cc::RewriteAllocBuffer) prepends
a "version" dim of size num_stages to every shared buffer participating
in a multi-stage pipeline; RewritePipelineBufferRegion inserts the
per-iteration version index at region[0]. The CUDA macro generators
handle this by collecting [r.min for r in region[:-2]] and threading
those leading indices through. The Metal macro generator
(tilelang/intrinsics/metal_macro_generator.py::MPSIntrinEmitter) was
written for the 2D-only case.

This patch updates _parse_buffer_2d to extract a leading_indices tuple
and propagates it through ldmatrix_a, ldmatrix_b, and simdgroup_copy
into T.access_ptr(buffer[leading + (row, col)]). Mirrors the existing
CUDA pattern in mma_macro_generator.py:265.

When region has length 2, leading == () and the existing 2D fast path
is bit-identical.

Diff stat:
    tilelang/intrinsics/metal_macro_generator.py | 29 ++++++++++++++++++-----
    1 file changed, 22 insertions(+), 7 deletions(-)
```

### 4. PR title (under 70 chars)

```
fix(metal): propagate pipelined version index in access_ptr
```

(58 chars.)

### 5. PR body

```markdown
## Summary

Single-file Python edit (`tilelang/intrinsics/metal_macro_generator.py`,
+22/-7) that threads the software-pipeline version index through
`MPSIntrinEmitter.ldmatrix_a` / `ldmatrix_b` / `simdgroup_copy` into
`T.access_ptr(buffer[...])`. Mirrors the CUDA pattern already in
`mma_macro_generator.py:265`. Zero behaviour change for 2D buffers (the
pre-patch hot path).

## Why

`T.Pipelined(..., num_stages=N)` with N > 1 lowers fine on the CUDA
target but fails on the Metal target with:

```
IndexError: Buffer A_shared is 3-dimensional (shape=[2, 32, 32]),
but 2 index(es) were provided: (row_idx, col_idx).
Please provide exactly 3 index/indices or slice(s).
```

`tilelang/src/transform/inject_pipeline.cc::RewriteAllocBuffer` prepends
a "version" dimension of size `num_stages` to every shared buffer
participating in a multi-stage pipeline (this is how double / triple
buffering is realised). A 2D shared buffer `[M, N]` becomes a 3D buffer
`[num_stages, M, N]`, and `RewritePipelineBufferRegion` inserts the
per-iteration version index at `region[0]`.

The CUDA macro generators (`mma_macro_generator.py`,
`wmma_macro_generator.py`, etc.) handle this correctly: they collect the
leading region dims via `[r.min for r in region[:-2]]` and pass them as
prefix indices into `T.access_ptr`. The Metal macro generator was
written for the 2D-only case and trips TileLang's `Buffer.__getitem__`
arity check.

## Test plan

- [ ] `cd /path/to/tilelang && git apply
      /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch`
- [ ] `python /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py`
      — three pipelined kernels at varying num_stages, all lower:

| Kernel                                   | Status |
| ---------------------------------------- | ------ |
| k_pipe_2 (num_stages=2 + 16x16 fragment) | OK     |
| k_pipe_3 (num_stages=3 + 16x16 fragment) | OK     |
| k_attn (pipelined Q*K^T)                 | OK     |

- [ ] Generated MSL also compiles with: `xcrun --sdk macosx metal -c
      <generated>.metal -o <generated>.air`
- [ ] Optional manual perf:
      `TILELANG_PIPELINED_RUNTIME_PROFILE=1 TILELANG_PIPELINED_RUNTIME_REPS=50
      python test_pipelined_probe.py`
- [ ] No regressions in the Metal regression suite.

Generated-source metrics (deterministic, asserted in the probe):

| Kernel   | Source lines | Threadgroup buffers            | simdgroup_load | simdgroup_multiply_accumulate | simdgroup_store |
| -------- | -----------: | ------------------------------ | -------------: | ----------------------------: | --------------: |
| k_pipe_2 |           56 | A_shared[1024], B_shared[1024] |              4 |                             2 |               2 |
| k_pipe_3 |           69 | A_shared[1536], B_shared[1536] |              6 |                             3 |               2 |
| k_attn   |           55 | K_shared[512], Q_shared[256]   |              4 |                             2 |               2 |

## Caveats

- **`apple-head`-only.** Public `main` at `2eec5f0` does not contain
  `tilelang/intrinsics/metal_macro_generator.py`. Refresh against the
  branch that owns the Metal macro emitter before opening a PR there.
- The unrelated `k2_pipelined_gemm` (32×32 fragment) baseline failure
  (`StorageRewrite::PointerValueTypeRewrite` vectorising a
  `metal.simdgroup` buffer to `float32x4`) is a separate pre-existing
  bug and is **not** addressed here. Reproducible without pipelining
  using an explicit `for ko in range(...)` K-loop.
- Torch/MPS launch smoke completed for `k_pipe_2`, `k_pipe_3`, `k_attn`,
  but the Metal adapter logs a non-fatal cache-save error
  (`MetalKernelAdapter has no libpath`); manual launch timing is
  available behind `TILELANG_PIPELINED_RUNTIME_PROFILE=1`. Treat
  numbers as local-health checks, not in-tree benchmarks.

## Backwards compatibility

For 2D regions, `leading == ()` and `(leading + (row, col)) == (row, col)`
— bit-identical to the pre-patch hot path. The 1D fast path is
unaffected.

## Attribution

Mirrors the CUDA pattern in
`tilelang/intrinsics/mma_macro_generator.py:265` (existing).
```

### 6. Required pre-filing steps

```bash
cd /tmp/tilelang_apple_head/tilelang
git fetch origin
git checkout apple-head
git pull --ff-only
# Stage the patch as a real commit on a feature branch:
git checkout -b cppmega/metal-pipeline-3d-buffer
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch
git add tilelang/intrinsics/metal_macro_generator.py
git commit -m "fix(metal): propagate pipelined version index in MPSIntrinEmitter access_ptr"

# Sanity check:
git apply --check /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch
# (Should be a no-op; confirms the patch matches what is now committed.)

# Push to user's fork:
git push <user-fork-remote> cppmega/metal-pipeline-3d-buffer
```

### 7. Test command for reviewers

```bash
cd /path/to/tilelang   # apple-head checkout
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch
# Python-only — no rebuild required.
.venv/bin/python /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py
# Expect: k_pipe_2 OK, k_pipe_3 OK, k_attn OK.
# Then verify generated MSL compiles:
xcrun --sdk macosx metal -c <generated>.metal -o <generated>.air
```

### 8. Reviewer-targeting hints

- Same as PR #3:
  - `Jorge C` — recent simdgroup-related fixes
    (`fix(metal): harden simdgroup store lowering`,
    `harden simdgroup review paths`).
  - `Yichen Yan` — original Metal GEMM author
    (`[Metal] Add Metal GEMM support with simdgroup_matrix MMA`).
- Suggest @-mention: the author of the original Metal GEMM PR plus
  `Jorge C` for the simdgroup transform area. The user should look up
  GitHub handles in the tile-ai/tilelang PR list before tagging.

### Patch artifact (absolute path)

`/Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch`

---

## Notes on what is **not** in this pack

- This pack intentionally certifies only the four PR entries above. It does
  not certify the remaining `docs/upstream` patch artifacts as PR-ready.
- The FP8 TileLang lanes (`tilelang_metal_fp8`,
  `tilelang_metal_fp8_gemm`, `tilelang_metal_fp8_vector`, and
  `tilelang_metal_fp8_scaled_matmul`) need their own clean target-branch
  `git apply --check` receipts before filing. Keep the e4m3 subnormal decode
  fix in the storage-only FP8 patch so later FP8 patches do not carry the same
  codegen fix again.
- `tilelang_metal_fp8_vector` is conditional until revalidated on a fresh
  intended TileLang base with the storage-only FP8 prerequisite applied. If PR
  prep still reports a corrupt-patch / line-73 apply failure, regenerate from
  a clean checkout rather than filing from this pack.
- `tilelang_metal_fp8_scaled_matmul` should not be described as real
  `src/op/builtin.cc` registration, a Metal scheduler pass, an FP8->FP16
  threadgroup tile feeding existing FP16 simdgroup MMA, or a measured speedup
  unless the patch and tests actually contain that implementation. The current
  docs artifact describes a macro scalar correctness baseline and explicitly
  checks that FP8 Metal codegen does not use `simdgroup_multiply_accumulate`.
- PR #9 is documentation-only and has no upstream PR.
- The MLX exporter side of the DLPack story (fixing
  `python/src/convert.cpp:100-159` to emit `kDLMetal` capsules — Phase 2)
  is a follow-up to PR #1 and is **not in this pack**.
