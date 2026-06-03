# DLPack ↔ CUDA interchange — findings & cherry-pick plan (MLX ↔ tvm-ffi ↔ TileLang)

Status: synthesized 2026-06-02. Goal: let MLX-CUDA arrays be consumed **zero-copy** by TileLang
`target="cuda"` kernels so Path-C drops its eager host-roundtrip bridge.

Provenance note (per our branch policy): tilelang is on `merge/upstream-codegen-reorg`
(local `tir`→`tirx` migration); `3rdparty/tvm` is pinned to `9b0a1667`. The `tir`→`tirx`
rename is a **local merge concern only** — no upstream issue ties it to DLPack/CUDA device
kernels (UNVERIFIED that it has any DLPack interaction; nothing found).

---

## 1. What our error actually was, vs. what real zero-copy needs

### 1a. The error we hit (already fixed by fe32cb1) — a target mismatch, not a deep DLPack bug
Two local guards reject any MLX array fed to a non-`metal` kernel:

- `tilelang/jit/adapter/tvm_ffi.py:599-603` — runtime guard raises `DLPackDeviceError`:
  ```python
  uses_mlx_runtime = has_mlx_arrays(dlpack_args)
  if uses_mlx_runtime and target_kind != "metal":
      raise DLPackDeviceError(
          f"MLX arrays export Metal DLPack buffers, but this kernel targets {target_kind!r}.")
  ```
- `tilelang/contrib/mlx_interop.py` — validation machinery hardcodes Metal as the expected device:
  `DLPACK_DEVICE_CUDA=2` / `DLPACK_DEVICE_METAL=8` (lines 32-33), `validate_dlpack_device()`
  (line 511) emits the "is on {X}, but this path requires {Y}" message, and
  `validate_dlpack_inputs_for_target()` (line 535, `expected_device_type = DLPACK_DEVICE_METAL`
  at 540) + `first_mlx_array_device()` (line 553, METAL at 560) **hardcode METAL** for MLX arrays.
  The `_DLPACK_DEVICE_NAMES` map (lines 38-39) only knows `kDLCUDA=2`/`kDLMetal=8`, so the
  `DLDeviceType(13)` in our original error printed as a raw number — that 13 is `kDLCUDAManaged`,
  emitted by MLX itself (see §1b).

**fe32cb1** ("GDN/KDA Path-C device-aware target") fixed the *symptom* by switching the kernel
target `metal`→`cuda` AND adding a host-roundtrip data bridge. The target switch alone is
insufficient because MLX cannot export a CUDA DLPack capsule at all (§1b) — so fe32cb1 is a
**real workaround for a real gap**, not merely a target flip.

### 1b. The real gap — MLX has NO zero-copy CUDA DLPack import/export
Verified from MLX source in our tree (`/Volumes/external/sources/mlx`):

- `python/src/array.cpp:522-534` — `__dlpack_device__` returns **13 (`kDLCUDAManaged`)** on a
  CUDA host, not `kDLCUDA=2`; and **`device_id` is hardcoded to 0** in all branches (multi-GPU
  round-trip broken):
  ```cpp
  if (mx::metal::is_available())      return nb::make_tuple(8, 0);   // kDLMetal
  else if (mx::cu::is_available())    return nb::make_tuple(13, 0);  // kDLCUDAManaged (!=2)
  else                                return nb::make_tuple(1, 0);   // kDLCPU
  ```
- `python/src/convert.cpp:300-318` (`mlx_to_dlpack`) — **export throws** for CUDA:
  ```cpp
  if (device_type == nb::device::cuda::value ||
      device_type == nb::device::cuda_managed::value)
    throw nb::buffer_error("CUDA DLPack export is not supported.");   // line ~311
  ```
- Import side (upstream PR #3495, merged) **rejects all non-CPU DLPack inputs** by design.

So `__dlpack_device__` advertises CUDA(-managed) but `__dlpack__()` then refuses to produce a
capsule. This is the hard blocker for Path C.

### 1c. How fe32cb1's bridge moves data today (copy, NOT zero-copy)
- `cppmega.mlx/cppmega_mlx/nn/_tilelang/_cuda_eager.py:122-156` —
  `_mlx_to_torch_cuda` (122-141): MLX-GPU → `np.array` host → `torch.from_numpy(...).to("cuda")`
  (H2D); bf16 widened through fp32 (no numpy bf16). `_torch_cuda_to_mlx` (144-156): CUDA → D2H →
  numpy → `mx.array`. Stream sync is an explicit `torch.cuda.synchronize()` per call
  (e.g. lines 267, 441, 706, 974, 1135) — not DLPack stream handoff.
- The TileLang-CUDA kernels themselves are genuine: `tilelang.compile(prim, target="cuda", ...)`
  (lines 200, 381, 624, 925, 1028). Only the *interchange* is an eager copy.

### 1d. Where a zero-copy path would live (the bridge call sites to collapse)
- Fused production path: `cppmega.mlx/scripts/m04_train_step.py:7166-7173` dispatches to
  `_path_c_call_cuda_artifact_with_mlx_bridge` (body at **6805-6894**): walks positional args
  (6840), swaps each MLX buffer for `_mlx_to_torch_cuda` (6843), calls `artifact(*torch_args)`
  (6849), `torch.cuda.synchronize()` (6859), writes results back via `_torch_cuda_to_mlx`
  (6868-6894). **This is exactly the code that zero-copy would delete.**
- Per-op switches: `cppmega_v4/_tilelang/kda_path_c.py:137-199` (`_device_can_run_metal()` 137-151;
  CUDA-eager branch otherwise; metal compile 123-129) and
  `cppmega_v4/_tilelang/linear_attention_path_c.py:178,206`.

---

## 2. PR / issue map across all 5 repos

Legend: **HELP** = directly enables zero-copy MLX-CUDA→TileLang; **CTX** = context/foundation;
**BLOCK** = a blocker we must work around or patch.

### MLX (ml-explore/mlx) — the blocker side
| # | state | what it does | helps? |
|---|---|---|---|
| [#1120](https://github.com/ml-explore/mlx/pull/1120) | MERGED 2024-05-16 | `array.__dlpack__()` export (CPU/Metal, predates CUDA) | CTX |
| [#1159](https://github.com/ml-explore/mlx/issues/1159) | CLOSED | req `__dlpack_device__` (Metal-or-CPU; no CUDA) | CTX |
| [#1165](https://github.com/ml-explore/mlx/pull/1165) | MERGED 2024-05-31 | `__dlpack_device__` → `(device_type, device_id)`; Metal-or-CPU | CTX (no CUDA branch) |
| [#1983](https://github.com/ml-explore/mlx/pull/1983) | CLOSED (WIP, superseded) | umbrella CUDA backend effort | CTX |
| [#2075](https://github.com/ml-explore/mlx/pull/2075) | MERGED 2025-05-07 | CUDA backend backbone (CUDA streams, `cudaEvent` sync) | CTX (why `mx::cu::is_available()` exists) |
| [#2391](https://github.com/ml-explore/mlx/issues/2391) | CLOSED | CUDA sync model doc (`cudaEvent`, `cuda::atomic` busy-wait); **no DLPack stream interop** | CTX (defines stream model we'd interop with) |
| [#2099](https://github.com/ml-explore/mlx/issues/2099) | OPEN | deadlocks: busy-wait kernel + `cudaFree` | BLOCK (free-imported-tensor risk) |
| [#3476](https://github.com/ml-explore/mlx/pull/3476) | CLOSED, not merged | earlier consumer: CPU copy + Metal MTLBuffer; CUDA never handled | CTX (superseded) |
| [#3495](https://github.com/ml-explore/mlx/pull/3495) | MERGED 2026-05-13 | DLPack import path **rejects all non-CPU** (CUDA+MPS) by design; tests | **BLOCK** (hard import reject) |
| [#3531](https://github.com/ml-explore/mlx/pull/3531) | OPEN draft | zero-copy **Metal** MTLBuffer import/export; `from_dlpack(copy=...)`; needs nanobind #1338; review notes `set_data(offset,deleter)` broke CUDA call sites (compile-only) | CTX — **Metal-only, no kDLCUDA** |
| [#3548](https://github.com/ml-explore/mlx/issues/3548) | CLOSED "not planned" | RFC consumer (CPU+Metal+TileLang/TVM-FFI Metal); **kDLCUDA explicitly rejected; stream sync unsolved** | **BLOCK** (maintainers won't do CUDA) |
| [#2848](https://github.com/ml-explore/mlx/issues/2848) | OPEN | "construct mx.array from mps/cuda arrays from other frameworks" | tracking issue for our need — unfulfilled |
| [#3342](https://github.com/ml-explore/mlx/pull/3342) | MERGED | `array.data_ptr()` low-level binding | **HELP** (escape hatch: build our own DLManagedTensor from device ptr) |
| [#3579](https://github.com/ml-explore/mlx/pull/3579) | MERGED | expose `from_dlpack` on `__array_namespace__()` | CTX (no device change) |

### tvm-ffi (apache/tvm-ffi) — the side that already works on CUDA
| # | state | what it does | helps? |
|---|---|---|---|
| [#96](https://github.com/apache/tvm-ffi/pull/96) | MERGED 2025-10-11 | unified `DLPackExchangeAPI` + `current_work_stream`; conversions → `_no_sync` | **HELP** (core import/export + stream query) |
| [#109](https://github.com/apache/tvm-ffi/pull/109) | MERGED 2025-10-13 | Cython ABI cuda-core stream protocol; stream as `void*` | **HELP** |
| [#236](https://github.com/apache/tvm-ffi/pull/236) | MERGED 2025-11-07 | accept `cuda.bindings` `CUstream` as `void_p` | **HELP** (raw CUstream passthrough) |
| [#260](https://github.com/apache/tvm-ffi/pull/260) | MERGED 2025-11-12 | `from_dlpack` uses exchange API | **HELP** (fast zero-copy import) |
| [#288](https://github.com/apache/tvm-ffi/pull/288) | MERGED 2025-11-26 | exchange API passed by PyCapsule (back-compat int) | **HELP** (ABI) |
| [#301](https://github.com/apache/tvm-ffi/pull/301) | MERGED 2025-12-02 | fix `from_dlpack` fallback after #288 | **HELP** (import robustness) |
| [#517](https://github.com/apache/tvm-ffi/pull/517) | MERGED 2026-04-02 | recursive container→tensor w/ stream propagation | CTX |
| [#521](https://github.com/apache/tvm-ffi/pull/521) | MERGED 2026-04-02 | `ffi.ContainerFindFirstNonCPUDevice` (stream-capture scan in C++) | CTX |
| [#466](https://github.com/apache/tvm-ffi/pull/466) | MERGED 2026-02-20 | ROCm `current_work_stream` correctness + GPU stream-identity test | CTX (same machinery as CUDA) |
| [#585](https://github.com/apache/tvm-ffi/pull/585) | MERGED 2026-05-13 | prefer torch `__dlpack_c_exchange_api__` on CPU/CUDA/ROCm; backend-aware lib select | **HELP** (CUDA metadata correctness; real fix for tilelang#2178) |
| [#578](https://github.com/apache/tvm-ffi/issues/578) | OPEN issue | req `current_work_stream`+MTLBuffer for **kDLMetal** | CTX (Metal-only gap) |

UNVERIFIED whether tvm-ffi has any hard `device_type==kDLCUDA` *rejection*; merged paths support
kDLCUDA, so any `DLPackDeviceError` from the tvm-ffi side is a device-id/target mismatch, not a CUDA block.

### TileLang (tile-ai/tilelang)
| # | state | what it does | helps? |
|---|---|---|---|
| [#1108](https://github.com/tile-ai/tilelang/pull/1108) | MERGED 2025-10-31 | rebase to TVM v0.22.0 / tvm-ffi; object-system modernization | CTX (enables tvm_ffi backend) |
| [#1259](https://github.com/tile-ai/tilelang/pull/1259) | MERGED 2025-11-18 | tvm-ffi default backend; relax dtype in `ArgBinder::BindBuffer`/`BindDLTensor` | **HELP** (DLTensor binding relax for external arrays) |
| [#1289](https://github.com/tile-ai/tilelang/pull/1289) | MERGED | enable tvm-ffi for Metal | CTX (Metal) |
| [#2095](https://github.com/tile-ai/tilelang/pull/2095) | OPEN | convert torch.Tensor args → `tvm.runtime.Tensor` before `Executable`; regression test | CTX (host-side handoff MLX would traverse) |
| [#2178](https://github.com/tile-ai/tilelang/pull/2178) / [#2179](https://github.com/tile-ai/tilelang/pull/2179) | #2179 MERGED (CI workaround); real fix in tvm-ffi #585 | ROCm bad C DLPack exchange corrupted DLTensor metadata (`ndim expected 2, got 512`) | CTX (concrete cross-FFI metadata corruption example) |

### TVM (apache/tvm) — foundational, mostly pre-tvm-ffi-split
| # | state | what it does | helps? |
|---|---|---|---|
| [#8032](https://github.com/apache/tvm/pull/8032) | MERGED | rename `gpu`→`cuda`, DLPack→v0.5 | CTX (device naming baseline) |
| [#5301](https://github.com/apache/tvm/pull/5301) | MERGED | set `Container.shape_` in `NDArray::FromDLPack` | CTX (correct shape on import) |
| [#13678](https://github.com/apache/tvm/issues/13678) | CLOSED | memory-safety in `NDArray::FromExternalDLTensor` | CTX (lifetime of MLX-owned DLTensors) |
| [#17529](https://github.com/apache/tvm/issues/17529) | OPEN | `IsContiguous(dl_tensor)` fails → "DLManagedTensor must be contiguous" | BLOCK-risk (non-contiguous MLX array trips this) |

DLPack handling moved into tvm-ffi post-split; no apache/tvm PR fixes a `kDLCUDA` import bug
post-split (UNVERIFIED).

---

## 3. Cherry-pick recommendation

**Bottom line: there is nothing to cherry-pick that makes this work end-to-end, because the
missing piece does not exist upstream.** The TVM/tvm-ffi/TileLang side already imports kDLCUDA
DLPack with stream sync in the versions we (transitively) pin; the blocker is 100% MLX, and the
only MLX work in flight (#3531) is **Metal-only** and the CUDA RFC (#3548) was closed
"not planned." So:

### Already in our pinned versions (assume present — verify before relying)
- tvm-ffi: #96, #109, #236, #260, #288, #301, #585 (CUDA import + stream + capsule ABI).
- TileLang: #1108, #1259 (tvm-ffi default backend + relaxed DLTensor binding).
These are the entire consumer-side enablement; **no cherry-pick needed** if our pins are ≥ #585
(2026-05-13). **Action: verify** the effective tvm-ffi rev under `3rdparty/tvm` (pinned
`9b0a1667`) includes #585 and #466; if older, cherry-pick `#585` (CUDA metadata correctness)
and `#260`/`#288`/`#301` (import path) — these are self-contained and unrelated to the
`tir`→`tirx` rename, so **low conflict risk** on `merge/upstream-codegen-reorg`.

### Nothing upstream to cherry-pick for the MLX side
- #3531 is **Metal-only** and draft — cherry-picking it does NOT give CUDA export; it only
  touches CUDA call sites for compile correctness. Not useful for Path C zero-copy.
- #3495 is a *blocker* (rejects CUDA import); we'd be **reverting/extending** it, not adopting it.
- #3548 (the only CUDA-consumer design) was closed "not planned" — no PR to pull.

➡️ **We must carry a local MLX patch.** See §4.

### Conflict risk on our branches
- MLX patch lands in `python/src/convert.cpp` + `python/src/array.cpp` — **independent of the
  tilelang `tir`→`tirx` migration** (different repo). Low risk; rebases cleanly over upstream
  `main` except where #3531 also edits `set_data(...offset,deleter)` — coordinate with our
  existing Metal zero-copy fork commits.
- TileLang patch lands in `tilelang/jit/adapter/tvm_ffi.py` + `tilelang/contrib/mlx_interop.py`
  — these are pure-Python adapter files, **untouched by the codegen `tir`→`tirx` reorg**, so
  near-zero conflict on `merge/upstream-codegen-reorg`.

---

## 4. What we implement (the real gap) — scoped

Because no upstream fix exists, build a local zero-copy bridge. Three coordinated changes:

### (A) MLX: export a real kDLCUDA capsule + fix device_id  [`/Volumes/external/sources/mlx`]

> **STATUS: ✅ IMPLEMENTED, BUILT, VERIFIED, COMMITTED (2026-06-02).**
> Commit `0e4bdac3f` on `DatasunriseOU/mlx` branch `upstream-integration` (pushed).
> The device_type that works = **kDLCUDA(2)** with the real device ordinal.
> (tvm-ffi `from_dlpack` accepts BOTH kDLCUDA(2)→`cuda:N` and kDLCUDAManaged(13)→
> `cuda_managed:N`, verified empirically; we emit **kDLCUDA(2)** because a
> `cuda_managed` device does not cleanly match a TileLang `target="cuda"` kernel,
> and MLX CUDA buffers are `cudaMallocManaged` unified memory that is directly
> addressable from CUDA kernels — so kDLCUDA(2) is valid and is what torch/tvm-ffi
> report as plain `cuda:0`.)
>
> **The MLX C++ diff (5 files):**
> - `python/src/array.cpp` `__dlpack_device__`: return `(2, cu::current_device())`
>   instead of `(13, 0)`.
> - `python/src/convert.cpp` `mlx_to_dlpack`: default CUDA backend to kDLCUDA(2),
>   resolve real `device_id`, remove the `throw "CUDA DLPack export is not
>   supported."`, thread `device_id` into the nanobind ndarray.
> - `python/src/convert.cpp` `mlx_to_dlpack_impl`: **THE LOAD-BEARING FIX** — use
>   `a.buffer().raw_ptr()` (NOT `a.buffer().ptr()`) for the CUDA data pointer.
>   `ptr()` returns the opaque `CudaBuffer*` wrapper struct; `raw_ptr()` unwraps
>   `CudaBuffer.data` AND calls `move_to_unified_memory()` so the pointer is real,
>   coherent, device-addressable unified memory. With the wrong `ptr()` the
>   export "succeeded" (correct device/shape/strides) but every read returned
>   garbage/NaN. `byte_offset = a.offset()` is already in bytes. Plus a
>   `cu::synchronize_device()` before handoff (producer-readiness; MLX exposes no
>   public stream handle for the versioned `__dlpack__(stream=)` arg, so a full
>   device sync is the conservative correct choice).
> - `mlx/backend/cuda/cuda.h` + `device_info.cpp` + `no_cuda.cpp`: new public
>   `cu::current_device()` (cudaGetDevice) and `cu::synchronize_device()`
>   (cudaDeviceSynchronize), with no-op stubs for non-CUDA/Metal builds.
>
> **Build (gb10):** editable MLX-CUDA at `/home/dave/source/mlx` (origin
> `DatasunriseOU/mlx`, same `upstream-integration` + this commit). Built via direct
> cmake/ninja `core` target into `/home/dave/source/mlx/build-dlpack` with
> `MLX_CUDA_ARCHITECTURES=121a-real` (GB10 is compute cap 12.1 / sm_121; the
> family-specific `121a` real arch is required — see caveats below), then both
> fresh `core.cpython-313-aarch64-linux-gnu.so` and `libmlx.so` copied into the
> editable `python/mlx/`. Old `.so` backed up at `/tmp/mlx-*-OLD-*.so` and
> `/tmp/mlx-core-backup-*.so` (reversible).
>
> **Verified (gb10, GPU idle):**
> - `__dlpack_device__` → `(2, 0)` (kDLCUDA), no throw on `__dlpack__()`.
> - `tvm_ffi.from_dlpack(mlx_cuda_array)` → zero-copy `cuda:0` tensor.
> - Zero-copy DATA integrity: torch view of the same buffer == MLX source,
>   `max abs err 0.0` (incl. bf16 and a non-contiguous slice view).
> - MLX-CUDA array → tvm-ffi → **TileLang `target="cuda"` kernel** → numeric
>   parity **PASS, max abs err 0.0** (feeding the tvm-ffi tensor directly, which
>   bypasses the stale TileLang "MLX is Metal" guard that the Python-path agent
>   owns — see (C)).
> - Env-safety: existing MLX forward (Linear/relu/softmax/matmul, bf16 reductions)
>   still PASS. Metal path unchanged (Linux/CUDA host, N/A here, code path
>   untouched).
>
> **Build caveats (honest, gb10-specific — pre-existing upstream issues, NOT this patch):**
> - The full `setup.py` stage-2 multi-arch list (`75;80;90a;100a;120a;120-virtual`)
>   **does not compile** on this tree/toolchain: `mlx/backend/cuda/quantized/
>   fp_quantize.cuh` (from upstream commit `1fecc115`, PR #3157) guards its
>   `ptx::` TMA calls with `#if __CUDA_ARCH__ >= 1000`, but `ptx.cuh` defines
>   those members with the stricter `__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000` — so
>   any **virtual** PTX pass (`compute_120`/`compute_121`) or any non-family-
>   specific arch fails with `namespace "mlx::core::ptx" has no member ...`. This
>   is an upstream MLX bug unrelated to DLPack. We sidestep it by building the
>   single GB10-native family-specific real arch `121a-real` (no virtual pass).
> - `121a` SASS is family-locked: a `120a-real` build links but fails at RUNTIME
>   on sm_121 (`no kernel image is available` / `cudaGraphAddKernelNode invalid
>   argument`). Must build `121a-real` to match the device.
> - `pip install -e .` itself fails at the editable-wheel COPY step
>   (`can't copy '.../core.cpython-313...so': doesn't exist` — a setuptools
>   build-lib path bug); the C++ compiles+links fine. We therefore install via
>   direct cmake `core` build + manual copy of the two `.so` into `python/mlx/`.

5. **Import** (optional, for results path): NOT implemented. `from_dlpack` still
   rejects foreign CUDA buffers (allocator/lifetime mismatch with MLX's own
   managed allocator + #2099 deadlock risk). The results path (TileLang→MLX) is
   handled by the Python adapters, not MLX C++.

### (B) Or the lower-risk escape hatch (no MLX C++ patch)
Build the `DLManagedTensor` ourselves in Python from `mx.array.data_ptr()` (#3342) +
`DLDevice{kDLCUDA, id}` + shape/strides/dtype, hand the capsule straight to tvm-ffi
`from_dlpack` (which imports zero-copy via #260/#288). This avoids forking MLX C++ but we own
the deleter/stream-sync correctness in Python. **Preferred first step** — smallest blast radius,
testable, and it sidesteps the closed-as-not-planned upstream stance.

### (C) TileLang: stop hardcoding Metal  [`/Volumes/external/sources/tilelang`]
1. `tilelang/jit/adapter/tvm_ffi.py:599-603` — replace the `target_kind != "metal"` reject with
   a device-vs-target check: allow MLX arrays whose DLPack device matches `target_kind`
   (cuda↔kDLCUDA, metal↔kDLMetal).
2. `tilelang/contrib/mlx_interop.py:535-563` — make `validate_dlpack_inputs_for_target` /
   `first_mlx_array_device` select `expected_device_type` from `target_kind` instead of
   hardcoding `DLPACK_DEVICE_METAL` (lines 540, 560). Add `kDLCUDAManaged=13` to
   `_DLPACK_DEVICE_NAMES` (lines 38-39) so errors are legible, and thread the CUDA stream through
   the `__dlpack__(stream=...)` arg.

### (D) cppmega.mlx: delete the eager bridge once (A/B)+(C) land
- `scripts/m04_train_step.py:6805-6894` — collapse `_path_c_call_cuda_artifact_with_mlx_bridge`
  to pass MLX arrays directly to `artifact(...)`; drop `_mlx_to_torch_cuda`/`_torch_cuda_to_mlx`,
  the numpy bf16 widening, and the writeback-stitch.
- `cppmega_mlx/nn/_tilelang/_cuda_eager.py:122-156` — remove the copy bridge + torch dependency,
  keep the `tilelang.compile(target="cuda")` kernels.

**Scope estimate:** (B)+(C) is the minimum viable zero-copy path and touches only Python in two
repos. (A) is the upstream-quality version but requires C++ MLX changes + stream-sync we'd carry
indefinitely (upstream won't take it per #3548).

---

## 5. Honest unknowns / UNVERIFIED
- Whether our effective `3rdparty/tvm`-bundled tvm-ffi rev (pin `9b0a1667`) actually contains
  #585/#466/#260/#288/#301 — **must check** before assuming the consumer side is ready.
- Whether tvm-ffi's `from_dlpack` accepts **kDLCUDAManaged(13)** (vs only kDLCUDA(2)). If not,
  MLX must emit kDLCUDA(2) (§4A.1), which may be wrong if the buffer is truly managed memory.
- Whether MLX's CUDA stream export is stream-aware at all — no MLX PR wires the versioned
  `stream` arg into the CUDA backend; appears nonexistent (#3548 closed not-planned).
- #2099 deadlock surface (busy-wait kernel + `cudaFree`) under cross-framework handoff is real
  but unquantified for our usage.
- #17529 (`IsContiguous` reject) could bite if an MLX view is non-contiguous; not yet hit.
- No PR/issue numbers were re-confirmed against the live trackers in this pass beyond the
  source-level facts checked in our tree (convert.cpp throw, array.cpp `13`/`0`, the two
  TileLang guards). The MLX/tvm-ffi/tilelang PR numbers come from the prior verified research
  inputs A/B and were not independently re-fetched here (no web access used).
- The local `tir`→`tirx` migration has no known DLPack/CUDA interaction (UNVERIFIED there is none;
  nothing found tying them).
