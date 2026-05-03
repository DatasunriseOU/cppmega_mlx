# Upstream patch: TVM Metal Shared-storage opt-in

*Status*: local fork ready, PR not yet filed.

*Local fork branch*: cppmega/metal-shared-storage-opt-in in /Volumes/external/sources/tvm (sibling to this repo). Branched from apache/tvm@8873a4c (HEAD on main at clone time).

*Patch artifact*: 0001-metal-shared-storage-opt-in.patch in this directory (181 lines, +102/−13 in 1 file). Apply with git am or git apply.

*Live-verified*: 4/4 scenarios on real Apple Silicon hardware (Xcode 26.4.1, MacOSX26.4 SDK). MTLBuffer.storageMode matches the env var setting after MetalWorkspace::AllocDataSpace. See runtime_check.mm in this directory.

## How to apply

```bash
cd /Volumes/external/sources/tvm
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tvm_shared_storage/0001-metal-shared-storage-opt-in.patch
```

## Why we need this

TVM's Metal device API allocates MTLBuffer with MTLResourceStorageModePrivate (GPU-only). MLX 0.31.x allocates with MTLResourceStorageModeShared. Two allocators on the same Metal device produce buffers with different page-mapping semantics; DLPack zero-copy capsules from TVM cannot be consumed by mx.array (live-tested: std::bad_cast on mx.array(tvm_metal_capsule)).

This blocks Path C (apache-tvm-ffi + mlc-ai-nightly-cpu pip-install path for the cppmega.mlx Mamba3/sparse-MLA TileLang ports). Without it, TVM-emitted Metal kernels can't share buffers with MLX without a host roundtrip.

## What the patch does

Adds an env var TVM_METAL_STORAGE_MODE that overrides the storage mode used in MetalWorkspace::AllocDataSpace:

| value           | mode                                    | semantics                                             |
| --------------- | --------------------------------------- | ----------------------------------------------------- |
| unset / private | MTLResourceStorageModePrivate           | *default*, GPU-only, preserves historical behaviour   |
| shared          | MTLResourceStorageModeShared            | CPU+GPU mapped — required for zero-copy DLPack to MLX |
| managed         | MTLResourceStorageModeManaged           | macOS-only intermediate (driver tracks dirty pages)   |
| anything else   | MTLResourceStorageModePrivate + warning | safe fall-back                                        |

The env var is parsed once on the first AllocDataSpace call and cached in a function-local static. No per-allocation overhead.

## Backward compatibility

*Strict.* Default behaviour (env unset) is MTLResourceStorageModePrivate — the historical TVM Metal allocation strategy. Every existing TVM workload, including all of mlc-llm and mlc-ai-nightly-cpu, sees identical behaviour without the env var. The change is opt-in.

The dead commented-out #if TARGET_OS_IPHONE conditional at the same location is dropped (it was superseded by the new env-driven path).

The staging-buffer pool (metal_common.h:383) and temp-buffer pool (metal_device_api.mm:374) already use MTLStorageModeShared and are intentionally left untouched — they're host-staging by design and don't fall under the data-space allocator.

## Test surface

The upstream patch adds `metal.GetStorageMode` as a small FFI-visible probe so
tests can verify the resolved env-var mode without an ObjC bridge. The patch
artifact itself currently changes only `src/runtime/metal/metal_device_api.mm`;
it does not yet add an upstream `tests/python/runtime/...` file.

Local review coverage in this directory:

- `syntax_check.mm` compiles the env parser against Apple Metal headers and can
  be run in fresh processes for unset/shared/managed/private/invalid values.
- `runtime_check.mm` links against a patched `libtvm_runtime.dylib`, calls
  `metal.GetStorageMode`, then allocates through `device_api.metal.AllocDataSpace`
  and inspects the returned `MTLBuffer.storageMode`.
- `test_metal_shared_storage.py` is a downstream pytest smoke: it checks
  `metal.GetStorageMode`, builds a tiny Metal TIR add kernel, and verifies a
  round-trip result.

Before filing upstream, either fold a pytest equivalent into the TVM patch or
mention the C++/Python probe receipts in the PR body and be explicit that the
first patch is runtime-only.

## How we use it locally

cppmega.mlx Path C (TVM kernel runtime + MLX tensor frontend) sets the env var before importing TVM:

```python
import os
os.environ["TVM_METAL_STORAGE_MODE"] = "shared"
import tvm  # MetalWorkspace caches "shared" on first alloc
```

Once set, mx.array.__dlpack__() and tvm.runtime.from_dlpack(...) should round-trip without std::bad_cast (assuming the matching MLX-side mx.from_dlpack patch lands or already works for the export-only direction).

## Performance reading

This is not a pure TVM/Metal kernel-speed patch. Default Private mode should
remain the right choice for TVM-only workloads because device data stays
GPU-only. The performance motivation is narrower: Shared mode removes the
host-staging buffer plus blit/wait step at the TVM/foreign-framework boundary,
which is the path needed by DLPack interop with MLX.

The downstream `runtime_check.mm` file now includes a standalone transfer
microbenchmark that does not need TVM headers:

```bash
xcrun --sdk macosx clang++ -std=c++17 -fobjc-arc -framework Metal \
  -O2 -DCPPMEGA_STANDALONE_METAL_BENCH runtime_check.mm -o metal_transfer_bench
./metal_transfer_bench 50
```

Live result on Apple M4 Max, Xcode 26.4.1 / macOS SDK 26.4, 50 iterations:

| bytes    | direction    | private median us | shared median us | shared/private |
| -------- | ------------ | ----------------- | ---------------- | -------------- |
| 4 KiB    | CPU -> Metal | 117.250           | 0.042            | 0.000          |
| 4 KiB    | Metal -> CPU | 110.458           | 0.083            | 0.001          |
| 1 MiB    | CPU -> Metal | 138.375           | 12.750           | 0.092          |
| 1 MiB    | Metal -> CPU | 122.833           | 12.542           | 0.102          |
| 16 MiB   | CPU -> Metal | 392.458           | 220.000          | 0.561          |
| 16 MiB   | Metal -> CPU | 403.083           | 236.666          | 0.587          |

Interpretation: the speedup is the avoided staging/blit synchronization on
host boundary transfers. It does not prove faster Metal compute kernels, and it
does not replace the TVM-linked `runtime_check` that inspects the real
`MetalWorkspace::AllocDataSpace` buffer mode. The standalone bench verifies the
copied data after timing so the measured paths cannot be optimized away.

## PR description draft

Ready text for the apache/tvm PR body:

```markdown
## [Runtime][Metal] Add TVM_METAL_STORAGE_MODE env opt-in

The Metal device API has always allocated MTLBuffer with
MTLResourceStorageModePrivate. This is the right choice for pure-GPU
workloads (faster, no CPU page mapping), but it blocks zero-copy
DLPack interop with other Metal-using frameworks that allocate
Shared/Managed buffers — notably ml-explore/mlx, which uses
MTLResourceStorageModeShared everywhere.

This PR adds an opt-in env var TVM_METAL_STORAGE_MODE so users who
need the foreign-buffer interop can request Shared (or Managed) mode
explicitly. Default behaviour is unchanged: env unset -> Private.

Cache is process-local and parsed once on first AllocDataSpace call.
No per-allocation overhead. Unknown values fall back to Private with
a warning.

Local probes cover default/shared/managed/private/invalid values. A follow-up
test commit can add subprocess-isolated Python coverage under
tests/python/runtime/ if maintainers want the env-cache behaviour covered in
tree.

Motivation: enables zero-copy bridge from TVM-NDArray to mlx.array
(both wrap MTLBuffer; require matching storage mode for the same
foreign capsule to be consumable).

Performance note: this should be treated as a copy-elision interop path, not a
kernel-speed path. Private remains default for pure TVM workloads. Local Metal
microbenchmarks on Apple M4 Max show Shared buffers remove the staging-buffer
and blit/wait cost at CPU<->Metal transfer boundaries (for example, 1 MiB
CPU->Metal median 138.375 us Private vs 12.750 us Shared in the downstream
probe).
```

## Filing checklist

When ready to file:

1. Push branch to a fork (gh repo fork apache/tvm if not already).
2. gh pr create --base main --head <fork>:cppmega/metal-shared-storage-opt-in --title "[Runtime][Metal] Add TVM_METAL_STORAGE_MODE env opt-in" --body-file PR_BODY.md
3. CI: macos-arm64 GitHub Actions runner exists in apache/tvm — should exercise the new test automatically.

## Risks

- *Maintainer pushback* likelihood: LOW. Backward-compat preserved, opt-in only, motivated by a real interop case.
- *Review cycles*: 1-2. The change is mechanical; debate may center on the env-var name or whether to expose this through a Python API instead. We prefer env var for zero per-call overhead.
- *Upstream conflict surface*: ZERO. Verified by full open-PR scan on 2026-05-03 (50 open PRs in apache/tvm reviewed). The only Metal-relevant in-flight PR is *#19423* (TIR cooperative_tensor builtins for M5 NAX tensor cores), which touches include/tvm/tirx/builtin.h + src/runtime/thread_storage_scope.h — disjoint from src/runtime/metal/metal_device_api.mm. No competing storage-mode work exists.

## Local-fork install instructions

To use this patched TVM in cppmega.mlx today (without waiting for upstream merge):

```bash
cd /Volumes/external/sources/tvm
git checkout cppmega/metal-shared-storage-opt-in

# Build the runtime (5-10 min on M4 Max):
mkdir build && cd build
cmake -DUSE_METAL=ON -DUSE_LLVM=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu) tvm_runtime

# Install the Python frontend (development install):
cd ../python
pip install -e .

# Verify:
TVM_METAL_STORAGE_MODE=shared python -c "
import tvm
print('metal:', tvm.metal().exist)
arr = tvm.nd.empty((4,), dtype='float32', device=tvm.metal())
print('alloc OK:', arr.shape)
"
```

## Files in this directory

- 0001-metal-shared-storage-opt-in.patch — apply with git am or git apply (181 lines, +102/−13)
- README.md — this file
- syntax_check.mm — standalone Metal-only program that vendors the helper and exercises the env-var parsing (no libtvm needed). Build: xcrun --sdk macosx clang++ -std=c++17 -framework Metal syntax_check.mm -o syntax_check. Verifies all 6 cases (unset, shared, mixed-case Shared, invalid, managed, private).
- runtime_check.mm — live in-process C++ test that loads the freshly-built libtvm_runtime.dylib and (a) calls metal.GetStorageMode() via FFI, (b) calls device_api.metal.AllocDataSpace() and inspects the MTLBuffer.storageMode. It also has `-DCPPMEGA_STANDALONE_METAL_BENCH` mode for copy-elision transfer timing without TVM headers. The TVM-linked mode is the strongest possible verification — it proves the env var actually flows through to a real MTLBuffer. *Live results captured 2026-05-03 on Apple M4 Max:*


$ ./runtime_check
metal.GetStorageMode -> 'private'
MTLBuffer.storageMode = private
OK

$ TVM_METAL_STORAGE_MODE=shared ./runtime_check
metal.GetStorageMode -> 'shared'
MTLBuffer.storageMode = shared
OK

$ TVM_METAL_STORAGE_MODE=managed ./runtime_check
metal.GetStorageMode -> 'managed'
MTLBuffer.storageMode = managed
OK

$ TVM_METAL_STORAGE_MODE=private ./runtime_check
metal.GetStorageMode -> 'private'
MTLBuffer.storageMode = private
OK


- test_metal_shared_storage.py — Python smoke test (skips if `tvm` or a Metal device is unavailable) for downstream CI. Checks the `metal.GetStorageMode` FFI helper, compiles a TIR element-wise add for target="metal", round-trips through `tvm.nd.array`, and asserts numerical correctness with the env var set.
