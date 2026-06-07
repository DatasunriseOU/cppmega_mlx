# Triton C-tile executed TMA — iteration 6 (EXECUTEMEASURE)

Date: 2026-06-07. Host: gb10 (NVIDIA GB10, sm_121 / cap 12.1, aarch64-linux, CUDA 13.2).
Repro source: `/home/dave/source/tilelang` @ tilelang HEAD `a82f63ea` + tvm submodule
commit `2a4f9754` (codegen-llvm alloca alignment fix landed this iteration).

## Executive result — NO-GO (C-tile TMA launches but FAULTS at execution)

The C-tile TMA now **launches** (the mod64 tensorMap-create abort is fixed) and the
UTMALDG instruction is **present and reached at runtime**, but it **faults with
`CUDA_ERROR_ILLEGAL_INSTRUCTION` inside `tl::tma_load` (copy_sm90.h:96)**. The kernel
does not run to completion, so NO valid §P1 execution-ms delta can be measured. The key
question ("does one real coalesced TMA load measurably drop §P1 ms?") is **not yet
answerable** — the load does not execute correctly.

## What this iteration discovered and fixed (executed evidence)

### 1. gb10 was missing the linux PtrAnalysis C++ shim (root cause of "UTMALDG=0")
The prior REBUILDGROUND result reporting "tma_load=2 ... LDG drop to 38" was NOT
reproducible on gb10. On gb10 the only built shim artifacts were **Mach-O darwin** `.so`
files (rsync'd from the Mac). With no loadable `_triton_frontend_cxx` on linux, the
frontend fell back to the **MVP scalar path**: `ctx.ptr_states` empty ->
`_emit_ptrstate_tile_load_copynode` **never fires** -> the C-tile grounding code at
`memory.py:1391-1404` is **dead** -> generated CUDA had **zero** tma/cp.async/mbarrier,
UTMALDG=0, LDG=128.

Fix: built the **real linux shim** (not the stub) on gb10 against the existing Triton
aarch64 install:
- `TRITON_INSTALL_DIR=/home/dave/source/triton-aarch64-install` (has `libTritonIR.a` etc.)
- MLIR/LLVM from `/home/dave/.triton/llvm/llvm-ac5dc54d-almalinux-arm64`
- pybind11 from the venv
- output: `_triton_frontend_cxx.cpython-313-aarch64-linux-gnu.so` (build dir
  `poc/triton_frontend/_cxx/build_linux`, also copied into `_cxx/build/` for
  auto-discovery; stale darwin `.so` removed).

With the real shim, the **codegen** SASS shows the grounding firing:
| build | UTMALDG | LDG | spill(STL+LDL) | HMMA |
|-------|---------|-----|----------------|------|
| OFF (prologue_opt=False) | 0 | 66 | 658 | 32 |
| OPT (prologue_opt=True)  | **10** | 96 | 272 | 32 |

So the C-tile grounding DOES emit real TMA in codegen (UTMALDG 0 -> 10). Note: LDG did
**not** drop to 38 (it rose 66 -> 96); the prior "38" figure is not reproducible here.

### 2. tensorMap 64-byte alignment was NOT actually fixed for this kernel (now fixed)
The aligned(64) fix at `codegen_c_host.cc:438` only covers the **C-host** backend. This
kernel is lowered via the **LLVM backend** (`codegen_cpu.cc`), where the TMA descriptor is
materialized through a `tvm_ffi_any` stack alloca emitted as `align 8`:
```
%stack_ffi_any539 = alloca [35 x %0], align 8     <- BEFORE
```
At align 8 the descriptor address is mod64 != 0 and the runtime validator FATALs:
`tensorMap address must be 64-byte aligned, but got ... mod64=16` (also seen mod64=48).

Fix (committed, tvm `2a4f9754`): in `codegen_cpu.cc` `CreateIntrinsic` stack-alloca, pin
the `tvm_ffi_any` branch to `setAlignment(Align(64))` — mirroring the existing
`tensormap` branch. After rebuilding `libtvm_compiler.so` (ninja, no GPU needed):
```
%stack_ffi_any539 = alloca [35 x %0], align 64    <- AFTER
```
Result: the mod64 abort is **gone**; the kernel **launches**.

### 3. Remaining blocker — UTMALDG faults at execution (NO-GO)
With alignment fixed, the §P1 OPT kernel launches (grid=(1,64,112) block=(256,1,1)) and
then crashes. `compute-sanitizer --tool memcheck` pinpoints it exactly:
```
Illegal instruction
  at void tl::tma_load<...>(const CUtensorMap_st&, ...)+0x18f0 in copy_sm90.h:96
  by thread (128,0,0) in block (0,0,0)
  Device Frame: _chunk_scan_bwd_dstates_kernel_kernel+0x1050 in tvm_kernels.cu:171
```
i.e. the `cp.async.bulk.tensor.2d` (UTMALDG) instruction itself faults. The descriptor is
now statically valid (passes `copy.cc:988 is_one(global_stride[0])` via the grounding)
and 64-byte aligned, but the descriptor encoding is **invalid at execution** on sm_121 for
the real C-tensor layout (grounded innermost ds-stride=1, with the outer k-axis stride and
global dims coming from the symbolic runtime args). The TMA load is genuinely reached and
genuinely wrong — it is not codegen-only and not a silent fallback.

Note: a **plain TileLang matmul** (TMA and even non-TMA) also raised
`CUDA_ERROR_ILLEGAL_INSTRUCTION` on this same gb10 config in spot checks, suggesting a
broader sm_121/CUDA-13.2 codegen-arch interaction may compound the descriptor issue. The
OFF dstates path (HMMA-only, no TMA) runs cleanly and passes parity, so the dstates
illegal-instruction is specifically tied to the TMA/async pipelined path (sanitizer frame =
`tl::tma_load`), not the GEMM.

## Parity (unaffected paths still correct)
- §P1 dstates OFF path: **MAXDIFF = 4.882812e-04, allclose@1e-3 PASS** (unchanged; the
  alignment fix does not touch the OFF/scalar path).
- §P1 dstates OPT path: cannot be parity-checked — kernel crashes before producing output.
- Native baseline (mamba_ssm triton): **~1.13 ms** median (min 1.130), N=60 — matches the
  prior 1.12 ms reference.

## GO / NO-GO
**NO-GO for "one real executed C-tile TMA load with measured ms delta."** Two real
blockers were cleared this iteration (missing linux shim; LLVM-backend descriptor
alignment), advancing the state from "UTMALDG=0, grounding dead" to "UTMALDG launches".
The exact remaining reason (RULE#1) is: **the grounded C-tile TMA descriptor faults with
illegal instruction inside `tl::tma_load` (copy_sm90.h:96) on sm_121** — the descriptor
encoding / sm90 bulk-tensor path is not valid for this de-monomorphized layout on this GPU.
Resolving it requires a correct sm_121 TMA descriptor (correct global dims/box/strides for
the partially-grounded C layout, and likely an sm_100/sm_120-class bulk-tensor path rather
than the sm90 template) — a deeper fix than this iteration's grounding+alignment scope.

## Reproduce from HEAD
```
# build linux shim (no GPU):
cmake -S poc/triton_frontend/_cxx -B poc/triton_frontend/_cxx/build_linux -GNinja \
  -DMLIR_DIR=/home/dave/.triton/llvm/llvm-ac5dc54d-almalinux-arm64/lib/cmake/mlir \
  -DLLVM_DIR=/home/dave/.triton/llvm/llvm-ac5dc54d-almalinux-arm64/lib/cmake/llvm \
  -DTRITON_INSTALL_DIR=/home/dave/source/triton-aarch64-install \
  -Dpybind11_DIR=<venv>/lib/python3.13/site-packages/pybind11/share/cmake/pybind11 \
  -DPython3_EXECUTABLE=<venv>/bin/python
ninja -C poc/triton_frontend/_cxx/build_linux
cp poc/triton_frontend/_cxx/build_linux/_triton_frontend_cxx.*.so poc/triton_frontend/_cxx/build/
# rebuild tvm with the alignment fix (no GPU):
ninja -C build lib/libtvm_compiler.so
# codegen (UTMALDG 0->10):
PYTHONPATH=.../build:. python poc/triton_frontend/_test_harness/tridao_parity/sass_dstates.py
# exec (GPU): launches, then illegal instruction in tl::tma_load:
CUDA_LAUNCH_BLOCKING=1 .../compute-sanitizer python /tmp/run_opt_only.py
```
