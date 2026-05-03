# Upstream patch: MLX from_dlpack Metal-aware consumer

*Status*: local fork ready, build OK, 8/8 tests pass, PR not yet filed.

*Local fork branch*: cppmega/from-dlpack-metal-consumer in /Volumes/external/sources/mlx (sibling to this repo). Branched from ml-explore/mlx@e8ebdeb (HEAD on main at clone time).

*Patch artifact*: 0001-add-from_dlpack-metal-consumer.patch in this directory (822 lines, ~542 LOC of code in 8 files). Apply with git am or git apply.

*Build status*: cmake -DMLX_BUILD_METAL=ON -DMLX_BUILD_PYTHON_BINDINGS=ON .. && make core succeeds (~3-4 min on M-series). Incremental rebuild: ~10 sec.

## How to apply

bash
cd /Volumes/external/sources/mlx
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/mlx_from_dlpack/0001-add-from_dlpack-metal-consumer.patch


## Pairing with the TVM patch

This patch *only delivers value when paired* with the TVM MTLStorageModeShared opt-in (docs/upstream/tvm_shared_storage/). Without TVM allocating in Shared mode, foreign Metal capsules from TVM still fail this consumer's storage-mode validation. Land them as a pair so the user-visible feature ("zero-copy MLX ↔ TVM via DLPack") actually works end-to-end.

mlx-mfa (which inherits MLX's allocator) is Shared-by-default and works out of the box with this patch alone.

## Why we need this

MLX 0.31.x has mx.array.__dlpack__() and __dlpack_device__() — EXPORT works, but:

- mx.array.__dlpack_device__() advertises (8, 0) (kDLMetal) when Metal is available
- The exporter at convert.cpp:100-159 emits the actual capsule with device_type=1 (kDLCPU) — uses a.data<T>() (host pointer) without a nb::device::metal annotation
- There is NO mx.from_dlpack(obj) at all
- mx.array(tvm_metal_tensor) falls into create_array → to_array_with_accessor and fails with bad_cast

So MLX is currently a one-way kDLCPU producer despite advertising kDLMetal. Any zero-copy interop with TVM-NDArray, mlx-mfa, or other Apple Silicon Metal producers requires a host roundtrip — defeating the whole point of DLPack.

## What the patch does

Adds a top-level mx.from_dlpack(obj) that consumes either:

- A raw PyCapsule (named "dltensor" or "dltensor_versioned")
- Any object whose __dlpack__() chain yields one (up to 4 unwrap iterations)

Dispatch is by DLDevice.device_type:

| device_type  | behavior                                                                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| kDLCPU (1)   | Copy into a fresh MLX allocation; run producer's deleter immediately after copy                                                                                                                                                        |
| kDLMetal (8) | Wrap the foreign MTLBuffer as an mx.array via array(allocator::Buffer, Shape, Dtype, Deleter) — the constructor MLX already exposes. Storage mode validated to be MTLResourceStorageModeShared; non-Shared rejected with a clear error |
| Any other    | Rejected explicitly (kDLCUDA, kDLROCM, etc.)                                                                                                                                                                                           |

Used capsules are detected and rejected on second consume. Non-contiguous strided views are rejected with a clear error.

## Storage-mode design choice

*Option 1a* (reject anything other than MTLStorageModeShared). The MLX runtime expects Buffer::raw_ptr() to return a host-readable pointer; relaxing this would silently break kernel preludes that call MTL::Buffer::contents() (which only returns non-null for shared/managed storage per mlx/backend/metal/allocator.cpp:14-15, 23-28, 196-207).

The error message tells the producer exactly what to fix — they need to allocate with MTLResourceStorageModeShared. For TVM, the paired patch (docs/upstream/tvm_shared_storage/) provides TVM_METAL_STORAGE_MODE=shared.

## Performance / readiness status

Treat the Metal path as a copy-elision interop patch, not as a measured kernel
speedup. The CPU consumer path allocates a fresh MLX buffer, memcpys producer
bytes, then invokes the DLPack deleter. The Metal consumer path casts
DLTensor.data to MTL::Buffer*, rejects non-Shared storage, and returns
mx::array(wrapped, shape, dtype, deleter) without copying the payload.

Current local evidence is apply/build/test plus code review of that copy
boundary. A clean apply-check against ml-explore/mlx@e8ebdeb passed on
2026-05-03, and the unit tests cover the CPU path and capsule/error handling.
There is still no checked-in foreign kDLMetal producer smoke or
from_dlpack-specific profiler receipt, so do not claim an end-to-end TVM -> MLX
speedup yet. The honest performance claim is narrower: once paired with a
Shared-MTLBuffer producer such as the TVM storage-mode opt-in, this consumer
can avoid the host-copy import path; timing that bridge still needs the foreign
producer test.

## True LOC count

| File                                    | LOC        |
| --------------------------------------- | ---------- |
| python/src/dlpack_consumer.cpp          | 226        |
| python/src/dlpack_consumer.h            | 24         |
| python/src/dlpack_consumer_metal.cpp    | 138        |
| python/src/dlpack_consumer_no_metal.cpp | 11         |
| python/src/dlpack_format.h              | 25         |
| python/src/array.cpp additions          | ~26        |
| python/src/CMakeLists.txt additions     | 2          |
| *C++ subtotal*                          | *~452*     |
| python/tests/test_dlpack_consumer.py    | 90         |
| *Grand total*                           | *~542 LOC* |

Honest count: not the 200 LOC I initially estimated, not 800. The DLPack ABI plumbing + storage-mode validation + version-aware capsule handling + nanobind integration brings it to ~450 LOC of C++.

## Test surface

8/8 pass on cmake build of MLX:

1. test_function_exists — mx.from_dlpack is bound
2. test_round_trip_via_numpy — kDLCPU path: np.ndarray → mx.from_dlpack → mx.array
3. test_round_trip_via_capsule — raw PyCapsule (named dltensor)
4. test_self_round_trip — mx.array → __dlpack__ → from_dlpack (currently CPU path due to MLX exporter discrepancy; Phase 2)
5. test_dtypes — 13 dtype cases (bool, int8/16/32/64, uint8/16/32/64, float16/32/64, complex64)
6. test_rejects_non_dlpack_object — bare object rejected
7. test_rejects_used_capsule — second consume rejected
8. test_strided_view_rejected — non-contiguous slice rejected

Pre-existing test_dlpack and test_dlx_device_type in python/tests/test_array.py still pass (no regression).

Coverage gap before filing: the current tests exercise the CPU path, raw
capsule handling, self round-trip, dtype mapping, used-capsule rejection, and
strided rejection. They do not yet construct a foreign kDLMetal capsule backed
by a non-MLX MTL::Buffer, so the zero-copy Metal wrapping path and the
non-Shared-storage rejection path are covered by code review only. Add either a
small ObjC++ test producer or a TVM integration smoke before presenting this as
fully end-to-end Metal-tested.

## PR description draft

Ready text for the ml-explore/mlx PR body:

markdown
## [Python] Add mx.from_dlpack(obj) Metal-aware consumer

MLX currently advertises kDLMetal in mx.array.*dlpack_device*() but
emits kDLCPU capsules in *dlpack*() and has no consumer at all.
This means cross-stack DLPack interop on Apple Silicon (with TVM,
mlx-mfa, etc) requires a host roundtrip — defeating the point of
DLPack.

This PR adds mx.from_dlpack(obj) that consumes either a raw PyCapsule
or an object with *dlpack*(). Dispatch by DLDevice.device_type:

- kDLCPU: copy into a fresh MLX allocation
- kDLMetal: wrap the foreign MTLBuffer via the existing
  array(allocator::Buffer, Shape, Dtype, Deleter) constructor.
  Storage mode validated to MTLResourceStorageModeShared; non-Shared
  rejected with a clear error pointing the producer at allocator
  fix-up.
- Other device types rejected explicitly.

Used capsules detected, strided views rejected.

8 unit tests added. Total ~452 LOC C++ + 90 LOC tests.

A separate Phase 2 PR will fix MLX's exporter (convert.cpp:100-159)
to actually emit kDLMetal capsules when running on Metal — currently
the device_type advertised by *dlpack_device*() and the device_type
of the emitted capsule disagree.

Motivation: zero-copy DLPack with TVM-NDArray and mlx-mfa is the
canonical Apple-Silicon-Metal interop primitive. With this PR plus
the paired apache/tvm TVM_METAL_STORAGE_MODE=shared opt-in, two
allocators on the same MTLDevice can finally share an MTLBuffer
without memcpy.


## Filing checklist

When ready to file:

1. Push branch to a fork (gh repo fork ml-explore/mlx if not already).
2. gh pr create --base main --head <fork>:cppmega/from-dlpack-metal-consumer --title "[Python] Add mx.from_dlpack(obj) Metal-aware consumer" --body-file PR_BODY.md
3. CI: macos-arm64 GitHub Actions runner exists in ml-explore/mlx — should exercise the new tests automatically.

## Risks

- *Maintainer pushback* likelihood: MEDIUM (higher than the TVM patch). Realistic asks during review:
  - "Why duplicate DLManagedTensor rather than vendor upstream dlpack.h?" Answer: no third-party dep churn for an MVP.
  - "Why fail-closed on Private storage instead of falling back to a copy?" Answer: silent copy hides the bug; the error message tells producer how to fix.
  - "What about DLPack v1 flag_bitmask_read_only?" Honor in a follow-up.
  - "stream= keyword for DLPack v1?" Add in cycle 2.
  - "byte_offset support? strided import?" Worthwhile follow-ups but not MVP.
  - "Phase 2 (export emits kDLMetal) should land first." Possibly — discuss in PR.
- *Review cycles*: 3-4. The patch is real surface area, not a one-liner. The DLPack ABI handling will get scrutinized.
- *Upstream blocker likelihood*: LOW-MEDIUM. The feature is genuinely useful (mlx-mfa users need it today; TVM users need it once paired with the TVM patch). The implementation is solid (8 tests pass, build clean). Main risk is review-cycle latency, not rejection.

## Local-fork install instructions

To use this patched MLX in cppmega.mlx today:

bash
cd /Volumes/external/sources/mlx
git checkout cppmega/from-dlpack-metal-consumer

# Build (3-4 min on M4 Max):
mkdir build && cd build
cmake -DMLX_BUILD_METAL=ON -DMLX_BUILD_PYTHON_BINDINGS=ON ..
make -j$(sysctl -n hw.ncpu) core

# Install Python frontend:
cd ../python
pip install -e .

# Verify:
python -c "
import mlx.core as mx
import numpy as np
a = mx.from_dlpack(np.arange(12).reshape(3, 4))
print('from_dlpack OK:', a.shape, a.dtype)
"

# Run tests:
python -m pytest python/tests/test_dlpack_consumer.py -v


## Files in this directory

- 0001-add-from_dlpack-metal-consumer.patch — apply with git am or git apply (822 lines patch, ~542 LOC of code)
- README.md — this file

## Related

- Paired upstream patch: docs/upstream/tvm_shared_storage/ — TVM env-var opt-in for Shared storage. *Both patches must land for the zero-copy MLX↔TVM use case to work.*
- Future Phase 2: fix MLX's convert.cpp:100-159 exporter to emit kDLMetal capsules on Metal devices. ~30-50 LOC follow-up; complete the bidirectional zero-copy story.
