# TRITON cp.async / LDGSTS — EXECUTED measurement (§P1 dstates, GB10 sm_121a)

**Date:** 2026-06-09
**Repo / commit:** `tilelang` @ `0f9dd051` (merge/upstream-codegen-reorg) — AsyncImpl `is_async_copy` route.
**GPU:** NVIDIA GB10, sm_121a, CUDA 13.2. **GPU-mutex held for all exec.**
**Kernel:** routed `_chunk_scan_bwd_dstates` §P1 = `b1 nh112 hd64 ds64 nc64 cs64`, grid **(1,64,112)**.
**Disassembler:** `/usr/local/cuda-13.2/bin/cuobjdump -sass` on the **actually-executed** JIT cubin
(`/tmp/tvm-debug-mode-tempdirs/<ts>/00000/tvm_kernels.cubin`).

---

## TL;DR — HONEST GO/NO-GO: **NO-GO** for "cp.async measurably drops §P1 ms"

cp.async/LDGSTS emission is **genuine and verified in isolation** (LDGSTS=16, UTMALDG=0, real
`LDGSTS.E [R15], desc[UR10][R24.64]` opcodes), but it **does NOT emit on the executed §P1 runtime
path**, so it was **never the thing that ran** at §P1 and therefore **did not** drop §P1 ms. The
OFF→OPT speedup that IS measured (3147→2625 ms, ~16.6%) comes from `prologue_opt`, **not** cp.async —
both executed cubins ran **plain LDG (LDGSTS=0, LDG=128)**.

The key question — *does cp.async/LDGSTS measurably drop §P1 ms?* — is answered **NO on executed
evidence**: the §P1 cp.async path was not reached at runtime.

---

## EXECUTED numbers (CUDA-events, N=60 median, interleaved OFF then OPT, GPU-mutex)

| Mode | Build | Ran to completion | Parity MAXDIFF | EXEC ms (median/mean/min) | Executed SASS |
|------|-------|-------------------|----------------|---------------------------|---------------|
| OFF  | `prologue_opt=False` (default) | YES, no fault | **4.882812e-04** PASS | **3146.89 / 3145.88 / 3130.35** | LDGSTS=0 UTMALDG=0 LDG=128 HMMA=32 |
| OPT  | `prologue_opt=True` + `TL_FORCE_CP_ASYNC=1` | YES, no fault | **4.882812e-04** PASS | **2625.15 / 2624.53 / 2614.99** | LDGSTS=0 UTMALDG=0 LDG=128 HMMA=32 |

- §P1 grid confirmed **(1,64,112)** in both runs; native nonzero 29,360,127.
- Parity is the spec target **4.88e-04** in BOTH modes (nc=64 = real multi-K-trip; small-strided real args).
- **ms_delta_vs_1102:** the prior "1102 ms" was against a **stale frozen JSON**; the honest
  fresh-build OFF baseline is **3146.89 ms**. OPT is **2625.15 ms** (Δ = -521.74 ms, -16.6%) — but that
  delta is `prologue_opt`, NOT cp.async (both ran plain LDG).
- **vs native 1.12 ms:** OPT 2625 ms is still **~2344x** off native. Gap to native is dominated by the
  addressing/prologue fold and the fact the load still coalesces as plain LDG, not async.

## cp.async/LDGSTS — GENUINE but only in the no-triton compile path

Verified genuine async copy (compile-only, **triton NOT imported**), executed-cubin SASS:

```
LDGSTS.E [R15],      desc[UR10][R24.64]      ;
LDGSTS.E [R15+0x4],  desc[UR10][R24.64+0x4]  ;
... x16 total ;   UTMALDG=0 ;   HMMA=32 (GEMM intact)
CUDA: tl::cp_async_gs<4>(...) + tl::cp_async_commit()   (cp_async_gs=1)
```

These are real `LDGSTS` opcodes (not plain LDG relabeled, not TMA). dstates_parity holds (4.88e-04).

## ROOT CAUSE — why §P1 exec does not reach cp.async (RULE #1, no silent fallback)

The cp.async emission is **path-dependent on the TTIR parse provider**, and the realistic §P1
execution path takes the provider that does NOT hit the routed copynode emitter:

- **triton NOT imported** → text-TTIR falls through to the `mlir.ir` provider → routed copynode
  emitter runs → `is_async_copy` set → `cp_async_gs=1`, **LDGSTS=16**.
- **triton imported** (required for the native reference + real arg strides at §P1) → the
  Triton-native MLIR provider succeeds → different walker result → routed copynode emitter NOT hit →
  `cp_async_gs=0`, **LDGSTS=0** (plain LDG).

The §P1 measure run imports triton/mamba_ssm (to build the parity reference and supply real strides),
so it lands on the **cp_async_gs=0** branch. Reproduced deterministically:

```
TL_FORCE_CP_ASYNC=1, prologue_opt=True, NO torch/triton import  -> cp_async_gs=1 (LDGSTS=16)
TL_FORCE_CP_ASYNC=1, prologue_opt=True, WITH torch/triton import -> cp_async_gs=0 (plain LDG)
```

There is **no silent fallback**: the executed default is the bit-correct synchronous LDG, and the
cp.async path is honestly opt-in. But on the executed §P1 path the opt-in flag is **inert** because the
emitter that consumes it is bypassed by the triton-native TTIR provider.

## Other honest notes

- `prologue_opt=True` WITHOUT `TL_FORCE_CP_ASYNC=1` is a **PRE-EXISTING break at HEAD**: fresh build
  fails `MakePackedAPI` with `B_local` undefined (num_stages pipeline mis-schedules the masked-OOB
  epilogue). `TL_FORCE_CP_ASYNC=1` only compiles because it DROPS the num_stages annotation
  (control.py) so PipelinePlanning skips the loop.
- A bare explicit cp.async emits `commit_group` with **no inserted `cp.async.wait`** before the GEMM
  consumer → racy if it ever executed. It did not execute at §P1.

## No-regression (EXECUTED)

- **Standard GEMM (serial copy + T.gemm):** `ALLCLOSE=True`, HMMA path intact — PASS.
  (`T.Pipelined`/num_stages hand-matmul faults `illegal instruction` on GB10 — that is the
  pre-existing GB10 TMA limitation under the software pipeline, unrelated to the gated dstates route.)
- **OFF/OPT dstates cubins:** HMMA=32 unchanged — the gated route does NOT touch GEMM codegen.
- **path_c F0 (`tests/test_mamba3_path_c_engine.py`):** 4 skipped, **0 failed / 0 errored** — no regression.
- **prod default sm_121a:** untouched (OFF default = synchronous LDG, parity 4.88e-04).

## Reproduce from HEAD

```bash
ssh gb10
cd /home/dave/source/tilelang && git checkout 0f9dd051   # merge/upstream-codegen-reorg
source /home/dave/cppmega-venv/bin/activate
# OFF (default plain LDG):
P1_MODE=OFF                       python /tmp/p1_ldgsts_exec.py
# OPT (prologue_opt + force cp.async flag; still plain LDG at exec due to triton provider):
P1_MODE=OPT  TL_FORCE_CP_ASYNC=1  python /tmp/p1_ldgsts_exec.py
# Genuine LDGSTS (compile-only, NO triton import):
TL_FORCE_CP_ASYNC=1 python -c 'import tilelang;from poc.triton_frontend import from_ttir;\
k=tilelang.compile(from_ttir(open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read(),\
name="_chunk_scan_bwd_dstates_kernel",target="cuda",_allow_text_ttir=True,prologue_opt=True),target="cuda");\
print(k.get_kernel_source().count("cp_async_gs"))'  # -> 1
# Disasm executed cubin:
/usr/local/cuda-13.2/bin/cuobjdump -sass <newest tvm_kernels.cubin> | grep -c LDGSTS
```

## Remaining gap to native 1.12 ms

OPT executed 2625 ms vs native 1.12 ms (~2344x). To close it the §P1 path must (1) actually reach the
cp.async emitter under the triton-loaded runtime provider (route the copynode emitter on the
triton-native TTIR path, not only the `mlir.ir` fallback), (2) insert a real `cp.async.wait` so the
async load is correct without the broken num_stages pipeline, and (3) fold the addressing/prologue so
the load coalesces. Until (1) lands, cp.async/LDGSTS provides **zero** §P1 speedup on executed evidence.
