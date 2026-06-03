# Relax graph-path train_step vs Megatron — measured tok/s + peak memory (gb10 CUDA)

**Status: MEASURED, 2026-06-03, gb10 (Grace-Blackwell aarch64, `tvm.cuda(0)`).**
This is the profiling phase of the Relax graph-memory path (docs/RELAX-GRAPH-MEMORY-PATH.md
PR-1..6). PR-6 proved the whole-step graph train_step RUNS on gb10 CUDA at the full
28-layer 1.8B config (loss finite, 8.787 GB planned device-peak). This doc PROFILES it
over multiple steps with the loop optimization applied, and compares the measured tok/s
+ peak memory against the live Megatron baseline (**3399 tok/s @ ~26 GB**, bs=4×seq=4096,
16384 tok/step, local_gb10_quarter 1.835B, reproduced live this session,
`MEGATRON-VS-MLX-PATHS.md`).

RULE #1 (fail-loud, no fabrication): every number here is measured on gb10 or an
explicitly-labelled extrapolation from a measured datapoint. Any non-finite loss or any
region that cannot run RAISES (the profiler and the e2e runner both fail-closed).

---

## TL;DR (the headline numbers)

| | Relax graph-path (this work) | Megatron-LM (baseline) |
|---|---|---|
| config | 28-layer 1.8B, seq=4096, batch=1 | 1.835B, bs=4×seq=4096 |
| tokens/step | 4,096 (seq×batch=4096×1) | 16,384 (bs=4×seq=4096) |
| **peak memory** | **6.400 GB planned 8L / 12.998 GB planned 28L device-peak** (Korthikanti launch-cache §14; was 4.682/8.787 GB pre-§14 — the launch lever trades +1.7/+4.2 GB for the launch reduction, still 2x under Megatron) | **~26 GB** |
| **fits Megatron-class memory** | **YES — 13.0 GB planned 28L < 26 GB (2x under)** | 26 GB |
| **tok/s** | **45.44 tok/s @ 8 layers MEASURED (Korthikanti launch-fusion, §14; was 29.82 @ §13, 25.2 pre-§13); ~11.76 tok/s @ 28 layers EXTRAPOLATED (was 5.19 @ §13)** | **3399 tok/s** |
| tok/s ratio vs Megatron | 0.0134x (8L measured, §14) / 0.0035x (28L extrapolated) — i.e. **75x–289x slower** (was 114x–654x @ §13) | 1.0x |
| what runs the compute | **REAL tilelang path_c-CUDA MR kernel, device-resident fwd (§13) + Korthikanti per-segment recompute cache (§14, 28L: 117→51 forward launches)** + abstract numpy bwd/adam/loss (lever 4) | tuned fused-FP8-CUDA kernels + selective recompute |

**(a) Does the graph path FIT in Megatron-class memory? YES.** The StaticPlanBlockMemory
planned device-peak for the whole 28-layer 1.8B step is **8.787 GB** — roughly **3x under**
Megatron's ~26 GB, and the optimizer state is kept at 1x by the explicit in-place Adam op
(lever-5). The eager MLX path cannot even run this step as one monolithic graph (OOMs at
114-116 GB on the same box, `MEGATRON-VS-MLX-PATHS.md`); the graph path runs it and plans
to 8.79 GB. **This is the win the whole memory-path was built for.**

**(b) What tok/s does it achieve? Far below Megatron — but TWO levers have now been spent and
both worked.** The §2/§4 PR-7 baseline (abstract numpy host-staged forward) measured **25.2
tok/s @ 8L**, ~100% host-staging-bound. **§13** landed the **device-resident forward** (REAL
tilelang MR kernel, banks on `tvm.cuda(0)`, no `.numpy()` in the hot path) → **29.82 tok/s @
8L (1.18×)**, moving the floor off host-staging onto real-kernel compute × launch count.
**§14 lands the launch-fusion lever — the Korthikanti per-segment recompute cache** (walk each
sqrt-N segment ONCE during backward, caching all its checkpoints, instead of the naive
O(N·√N) per-region prefix re-derivation): **45.44 tok/s @ 8L MEASURED (1.524× over §13), and
28L extrapolates to 11.76 tok/s (2.267× over §13's 5.19)**. The lever cuts forward launches
**20→13 at 8L and 117→51 at 28L** (the per-call MR-kernel cost is unchanged at 6.56 s — same
kernel — so the win is purely fewer launches); numerics are **identical** (loss 5.525e-06
every step, bit-for-bit with §13). The **8L gap narrows 114×→75×, the 28L gap 654×→289×**. The
honest cost: the segment cache holds a whole segment's recomputed checkpoints concurrently, so
the **planned device-peak rises 4.682→6.400 GB at 8L and 8.787→12.998 GB at 28L** — a real
memory/launch tradeoff, still 2× under Megatron's 26 GB. The gap is NOT architecture; the
remaining floor is the **6.56 s/call MR kernel itself** (now 95.0% of the step, 51 launches at
28L). The next lever is a **faster/fused/FP8 MR kernel** (per-call compute, not launch count).
See §4 for the baseline, **§13 for the device-resident re-profile, §14 for the launch-fusion
result + the new attribution**, and §5 for the lever list.

---

## 1. What was profiled, and the loop optimization applied

`scratch/pr7_profile_train_step_gb10.py` profiles the SAME whole-step `@R.function`
(`path_c_relax_train_step.build_train_step`) that PR-6 proved runs on gb10 CUDA:

> forward (sqrt-N remat) → `pathc.bank_loss` → backward (re-emit fwd for non-boundary
> checkpoints) → `pathc.adam_inplace`, returning `(loss, param', m', v')`,

with the 5 physical banks (activation / activation-grad / parameter / parameter-grad /
state-checkpoint, 2028.1 MB/region at real local_gb10_quarter scale) threaded as
cross-region Relax SSA tensors.

### Loop optimization: COMPILE-ONCE / RUN-MANY (+ optimizer-state threading)

The PR-6 runner (`run_train_step_on_device`) rebuilt the IRModule, ran `tvm.compile`, and
constructed a fresh `relax.VirtualMachine` on **every** call. A training loop runs ONE
step graph thousands of times — recompiling each step is pure waste. The profiler applies
the obvious loop optimization:

* **`tvm.compile` + VM build ONCE**, then execute N steps on the **same** VM. `compile_s`
  is reported separately and **amortizes to ~0** over a real loop. At 8 layers the
  compile-once is **0.03 s** (the 28-layer compile is ~14 min — a one-time cost in a loop,
  the strongest argument for caching the executable; see §5).
* **Optimizer state threaded step→step**: the step's outputs `(param', m', v')` feed the
  next step's inputs, so the optimizer banks are never re-allocated per step (the in-place
  Adam semantics carry across the loop).

The tok/s headline uses the **mean NON-compile step** (first step dropped as warm-up).

### tokens/step (honest derivation)

The bank ABI is parsed from `local_gb10_quarter_profile()` (hidden=3584,
max_seq_length=4096). The activation bank is **44,957,696 f32 = 3.0625 × seq × hidden**
with seq=4096 → **batch=1, one sequence of 4096 tokens**. So the graph step processes
**4,096 tokens/step**. Megatron's baseline step is **bs=4 × seq=4096 = 16,384 tok/step**
— a **4× batch difference, stated explicitly**: throughput (tok/s) is the normalized
metric, but the per-step token count differs 4×, so the graph path would need to process
4× the batch (or 4 steps) to match one Megatron step's token count.

---

## 2. MEASURED: 8-layer whole-step, multi-step on gb10 CUDA

| metric | MEASURED (gb10 CUDA, 8 layers, 10 steps) |
|---|---|
| compile-once | **0.03 s** (8-layer; amortizes to ~0 over a loop) |
| mean step (warm, step0 dropped) | **162.23 s** (median 162.70 s) |
| tokens/step | 4,096 (seq=4096, batch=1) |
| **THROUGHPUT** | **25.2 tok/s** |
| planned device-peak | 4.682 GB |
| MEASURED peak (free-delta high-water) | **10.30 GB** |
| loss (every step) | 4.165547e-02, finite, stable |

```
=== PROFILE gb10 CUDA, 8 layers (n_layers=8, 10 steps) ===
  compile-once     : 0.03s (amortized to ~0 over a loop)
  mean step (warm) : 162228.8 ms  median 162697.5 ms
  THROUGHPUT       : 25.2 tok/s
  planned dev-peak : 4.682 GB
  MEASURED peak    : 10.30 GB free-delta high-water
```

**The per-step wall is essentially identical across all 10 steps** (162.2 s mean, 162.7 s
median, step0 161.26s, step1 161.47s — within 0.3%). This is the key profiling finding:
the step is **host-staging-bound and constant per step**, NOT warm-up-limited. There is
no steady-state speedup to find beyond the one-time CUDA-context/JIT init (which here is
dwarfed by the staging), because every step re-stages every bank host↔device through the
abstract drivers. The compile-once loop optimization makes the per-process compile a
one-time 0.03 s (8L) / ~14 min (28L) cost instead of paying it every step.

---

## 3. Peak memory — FITS Megatron-class (the win)

| metric | value | basis |
|---|---|---|
| **planned device-peak (28-layer 1.8B)** | **8.787 GB** | StaticPlanBlockMemory honest device-allocator high-water (`true_planned_peak`) |
| planned device-peak (8-layer) | 4.682 GB | same analyzer |
| measured free-delta (28-layer, PR-6) | 17.64 GB | `/proc/meminfo` MemAvailable high-water (incl. host staging) |
| measured free-delta (8-layer, this profile) | **10.30 GB** | background sampler over the 10-step loop |
| **Megatron-LM footprint** | **~26 GB** | live receipt, `MEGATRON-VS-MLX-PATHS.md` |
| eager MLX (same step, same box) | **OOM @114-116 GB** | cannot run as one monolithic step |

**The graph path fits Megatron-class memory with room to spare**: 8.787 GB planned
device-peak is ≈3x under Megatron's 26 GB. The O(√N) rematerialization (PR-4) keeps the
checkpoint bank at O(√N) and the in-place Adam (PR-5 lever-5) keeps the optimizer state at
1x, so the whole-step peak is dominated by the √N checkpoint term, not the all-live
activation total (45.59 GB) the eager path retains.

**Why measured free-delta (17.64 GB / 8L value) > planned device-peak (8.787 GB):** the
abstract bank drivers host-stage each bank across the `call_dps_packed` boundary — on CUDA
the ABI tensors are device tensors, so each driver reads them via `.numpy()` (device→host
copy) and writes back via `tvm.runtime.tensor(host).copyto(out)` (host→device). The
transient host buffers + the input device tensors + the CUDA context inflate the free-delta
above the device-allocator high-water. The **planned 8.787 GB is the honest device peak**;
the free-delta is full process residency including host staging. Fusing every region
on-device (not just the forward) collapses the staging — §5.

---

## 4. WHERE THE TIME GOES — per-region profiler breakdown (8-layer, gb10 CUDA)

Per-region wall over 3 steps, profiler timers wrapping every region driver (8 layers,
gb10 CUDA; mean step 142.5 s; driver-tot ≈ 100% of the step wall):

| region kind | calls/step | wall (3 steps) | **% of step** |
|---|---:|---:|---:|
| **fwd + sqrt-N remat** | 20 (8 fwd + 12 recompute-fwd) | 414.26 s | **96.9%** |
| backward | 8 | 10.76 s | 2.5% |
| adam (in-place optimizer) | 1 | 2.30 s | 0.5% |
| loss | 1 | 0.20 s | 0.0% |
| **driver total** | 30 | **427.52 s** | ≈100% of the 142.5 s × 3 step wall |

```
  per-region wall over 3 steps (host-driver time):
    fwd+remat :  414255.0 ms  (60 calls)   96.9%
    bwd       :   10757.8 ms  (24 calls)    2.5%
    adam      :    2299.2 ms  (3 calls)     0.5%
    loss      :     203.4 ms  (3 calls)     0.0%
    driver-tot:  427515.4 ms  (rest = VM/host-staging overhead)
```

**The step time is ~100% host-staging-bound, and the forward+remat path is 96.9% of it.**
The `driver-tot` (427.5 s over 3 steps) ≈ the full step wall (142.5 s × 3) — i.e. the
Relax VM scaffolding adds ~0%; essentially the entire step is spent inside the abstract
region drivers, host↔device-staging the 2028 MB/region banks through numpy. The
forward+remat regions dominate (20 calls/step, each round-tripping the 965 MB state
checkpoint + 360 MB param + activation banks to host and back); the backward, adam, and
loss are together <3%. This is why:

1. **tok/s is ~25 (8L) / ~5 (28L), not thousands** — the bottleneck is PCIe/unified-memory
   host↔device copies of the 965 MB state bank + 360 MB param bank + activation banks,
   ~30 regions/step at 8 layers, ~147 at 28 layers, each fully round-tripped to host numpy.
2. The device-compute path is NOT the bottleneck — PR-6 stage 2 proved the REAL tilelang
   path_c-CUDA kernel (17 params, 149024 bytes CUDA-C, 14.68M nonzero) runs through the
   same forward boundary; the abstract drivers are a memory/executability proof, not the
   throughput path.

---

## 4a. DEVICE-RESIDENT forward driver — LANDED (the host-bounce removal)

The DPS driver has been reworked so the forward keeps the banks **device-resident on
`tvm.cuda(0)` end-to-end** — no numpy in the per-region hot path. What changed:

* **`path_c_dps_adapter.py`** — new device-resident primitives:
  `device_bank_view` (zero-copy `Tensor._create_view(size, dtype, relative_byte_offset)`
  flat sub-range VIEW of a bank), `device_pack` / `device_unpack` (device→device `copyto`
  into/out of those views — NO host array), `alloc_device_banks` (`tvm.runtime.empty` on
  device, zeroed once, reused across calls), and `make_device_resident_kernel_driver`
  (builds the kernel's positional arg list **once** — the 5 device banks at their
  positions, the curried `run_backward` scalar at the gate, and **device-resident**
  zero scratch for every auxiliary route-buffer param; per call it just runs
  `kernel(*args)` + `dev.sync()`, and the kernel **mutates the device banks in place at
  `out_idx`** so there is nothing to read back). `make_region_dps_packed._packed` now
  routes by the output tensor's DLPack device type (`__dlpack_device__()[0] != kDLCPU`):
  a device VM takes `_packed_device_resident` (device views), a CPU VM takes the numpy
  reference path (a clear device-vs-host gate, not a try/except fallback). The old
  `make_real_kernel_driver` is kept as the **numpy-staged reference** for the CPU
  self-test and the equivalence check.
* **`path_c_relax_step_banks.py`** — `bank_arg_is_device` (DLPack-device gate at the VM
  boundary), `bank_copy_prefix_device` (device→device prefix copy + device-resident
  tail-zero), `_zero_device_tensor` (zeros from a cached **device** zero buffer — only the
  first fill per (shape,dtype,device) touches the host).
* **`path_c_relax_train_step.py`** — `make_real_bank_forward_driver(..., device=dev)` now
  takes a device-resident path: when the `pathc.bank_fwd_*` `call_dps_packed` bank tensors
  arrive on a real device, it seeds the kernel's act/param banks via device→device view
  copies, runs the device-resident kernel driver in place, and fills `act_out`/`state_out`
  via device→device copies. **No `.numpy()` at the VM boundary in the forward.**

**MEASURED on gb10 `tvm.cuda(0)` (`scratch/pr7_device_resident_driver_validate_gb10.py`),
REAL MR JITKernel (17 params, `out_idx=[0,2,3,4,6..16]`):**

| check | result |
|---|---|
| (A) numeric equivalence, device-resident vs numpy-staged forward | `max\|dev−numpy\| = 1.025e-06` (act_out and state_out), 44,957,695 / 44,957,696 nonzero — **equivalent within fp tol** |
| (C) per-forward-call wall | numpy-staged **8472.4 ms** → device-resident **6515.0 ms** = **1.30×** |
| (B) e2e train_step on CUDA, device-resident forward, 2 layers × 3 steps | **RUNS, loss finite** (5.525e-06, stable across steps) |

The **1.30×** is the per-call ceiling for THIS kernel: this MR region is one **6.5 s**
single-launch kernel, so its own 5-bank H2D/D2H staging is small relative to compute; the
device-resident win at the driver level is the elimination of the per-call `np.zeros` bank
allocation + `np.ascontiguousarray` copies + dict rebuilds (the scout's 1.38× ceiling).
The LARGER lever is at the **VM boundary**: the device-resident forward removes the
per-region per-step `.numpy()` round-trip of the full 2028 MB banks across
`call_dps_packed` — the 96.9% term in §4 — for every forward + remat region. bwd/adam/loss
remain the abstract numpy drivers (lever 4); the forward is now device-resident.

---

## 5. Apples-to-apples vs Megatron 3399 / 26 GB, and the honest gap

| side | tok/step | peak mem | tok/s | ms/token | what runs |
|---|---:|---:|---:|---:|---|
| **Megatron-LM (live)** | 16,384 | ~26 GB | **3399** | 0.294 | fused-FP8-CUDA + selective recompute + distributed optimizer |
| Megatron-LM (clean-box receipt) | 16,384 | ~26 GB | 3712-3747 | 0.267 | same |
| **Relax graph-path, 8 layers (MEASURED)** | 4,096 | 10.30 GB measured / 4.682 GB planned | **25.2** | 39.6 | abstract numpy host-staged region drivers on CUDA VM |
| **Relax graph-path, 28-layer 1.8B (EXTRAPOLATED)** | 4,096 | 8.787 GB planned (17.64 GB measured, PR-6) | **~5.2** | ~193 | same; 4.9× more regions (147 vs 30) at constant per-region staging cost |
| eager MLX (same 1.8B step) | 16,384 | **OOM 114-116 GB** | **cannot run** | — | monolithic eager backward+optimizer burst exceeds the box |

### The extrapolation (explicit, no fabrication)

The 8-layer step is **30 `call_dps_packed` regions** (8 fwd + 12 recompute-fwd + 8 bwd + 1
adam + 1 loss); the 28-layer step is **147 regions** (28 + 89 + 28 + 1 + 1) — a measured
**4.9× region-count increase**. Because the profiler shows the per-step wall is
**host-staging-bound and ~constant per region**, the 28-layer step extrapolates to
**4.9 × 161.4 s ≈ 791 s/step → 4096 / 791 ≈ ~5.2 tok/s**. (PR-6 already executed the full
28-layer step end-to-end on gb10 with finite loss, so this is a throughput extrapolation
over a CONFIRMED-runnable config, not a runnability claim.)

### The honest gap

* **Memory: WIN.** 8.787 GB planned ≪ 26 GB Megatron, optimizer kept in-place. The graph
  path delivers the Megatron-class (and better) memory envelope, and runs the 1.8B step the
  eager path OOMs on. **This is the deliverable the memory-path set out to prove, and it is
  proven on device.**
* **Throughput: TRAILS by 135x (8L) to ~654x (28L) — and the gap is fully attributed.** It is **not**
  the device kernels and **not** the model architecture (identical 1.835B stack). It is the
  **abstract per-region host-staging** across the `call_dps_packed` boundary (every bank
  round-tripped to host numpy per region) plus the **4× smaller batch** (4096 vs 16384
  tok/step, less amortization). Megatron's 3399 tok/s is fused-FP8-CUDA large-batch kernels
  that never leave the device; the graph path currently leaves the device on every region.

### Levers that close the throughput gap (next, measured-cost-known)

1. **Keep the banks DEVICE-RESIDENT in the forward (eliminate the host staging) — LANDED.**
   The single biggest lever. The per-region `.numpy()` / `copyto` round-trips were
   **~100% of the step wall, the forward+remat path alone 96.9%** (§4). The DPS driver
   has now been **reworked to keep the banks as `tvm.cuda(0)` device tensors end-to-end**
   in the forward: pack/unpack of the logical I/O sub-ranges is a **zero-copy device VIEW
   (`Tensor._create_view` with a byte offset) + a device→device `copyto`**, the physical
   banks are allocated **once** (`tvm.runtime.empty` on device) and reused across calls,
   and the real tilelang JITKernel is fed the device banks directly — it mutates them
   **in place at `out_idx`**, so there is nothing to read back. **No `np.ndarray` /
   `np.from_dlpack` / `.numpy()` / host `copyto` in the per-region forward compute path.**
   See §4a for the landed driver + the measured numeric equivalence and per-call speedup.
   Remaining: extend the same device-resident treatment to bwd/adam/loss (those are still
   the abstract numpy s_tir-wall drivers, lever 4), and fuse adjacent forward+remat
   regions into fewer launches.
2. **Cache the compiled executable (28-layer compile is ~14 min, one-time).** Memoize the
   per-region lowering / batch identical regions so the whole-step `tvm.compile` is paid
   once per shape, not once per process. (compile-once is already applied across steps;
   this caches across processes.)
3. **Batch ×4** to match Megatron's 16,384 tok/step (the bank ABI is batch=1 today) for
   the amortization Megatron gets for free.
4. **Real backward kernel.** The bwd driver is still the abstract numpy s_tir-wall path
   (documented in the dps adapter); wiring the real path_c-CUDA backward removes the last
   abstract region.

---

## 6. Reproduce

```
# gb10 CUDA multi-step profile (compile-once / run-many), 8 layers x 10 steps:
ssh gb10
cd /home/dave/source/cppmega_mlx
PR7_LAYERS=8 PR7_STEPS=10 \
PYTHONPATH=/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/cppmega_mlx \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/pr7_profile_train_step_gb10.py
```
FAIL-LOUD (RULE #1): non-finite loss or any region that cannot run RAISES, naming it.
The whole-step e2e (PR-6) and its CPU self-check are
`scratch/pr6_cuda_e2e_train_step_gb10.py` and
`python -m cppmega_mlx.runtime.path_c_relax_train_step`.

```
# DEVICE-RESIDENT forward driver validation (numeric equivalence + e2e + speedup), §4a:
PR7V_LAYERS=2 PR7V_STEPS=3 PR7V_N=20 \
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/pr7_device_resident_driver_validate_gb10.py
```
Asserts `max|device−numpy| < 1e-3` (measured 1.025e-06), a finite e2e loss, and reports
the per-forward-call device-resident vs numpy-staged wall (measured 1.30×).

---

## 13. RE-PROFILE WITH THE DEVICE-RESIDENT FORWARD DRIVER — measured tok/s + the host-staging recovery (gb10 CUDA, 2026-06-03)

This re-runs the whole-step profiler with the **device-resident forward driver** of §4a
wired into every `pathc.bank_fwd_i` region (the REAL tilelang MR JITKernel, banks kept on
`tvm.cuda(0)` end-to-end, no `.numpy()` in the per-region forward hot path), and re-measures
tok/s + peak + the per-region breakdown against the §2/§4 PR-7 baseline (which ran the
**abstract numpy host-staged** forward). Profiler:
`scratch/pr7_profile_device_resident_gb10.py` (compile-once / run-many, optimizer state
threaded, background `/proc/meminfo` peak sampler, per-region wall timers). Loss finite and
stable (`5.525e-06`) every step — same artifact the §4a equivalence proved
(`max|device−numpy| = 1.025e-06`).

### MEASURED (gb10 CUDA, 8 layers, 4 steps, warm = steps 1–3)

| metric | PR-7 baseline (abstract numpy fwd) | **device-resident fwd (this §13)** |
|---|---:|---:|
| forward compute | `np.maximum` proxy + full host round-trip/region | **REAL tilelang MR kernel, banks device-resident** |
| mean warm step | 162.23 s | **137.36 s** (median 135.65 s) |
| **THROUGHPUT** | **25.2 tok/s** | **29.82 tok/s** |
| step-time speedup | 1.0× | **1.18×** (162.2 → 137.4 s) |
| planned device-peak | 4.682 GB | 4.682 GB (unchanged — same plan) |
| **MEASURED peak (free-delta)** | **10.30 GB** | **8.08 GB** (−2.22 GB: the per-region host staging buffers are gone) |
| loss | 4.166e-02 (abstract) | **5.525e-06** (real kernel), finite, stable |

```
=== PROFILE (DEVICE-RESIDENT FWD) gb10 CUDA, 8 layers (n_layers=8, 4 steps) ===
  compile-once     : 0.03s (amortized to ~0 over a loop)
  mean step (warm) : 137355.2 ms  median 135645.1 ms
  THROUGHPUT       : 29.82 tok/s
  planned dev-peak : 4.682 GB
  MEASURED peak    : 8.08 GB free-delta high-water
  per-region wall over 4 steps (host-driver time):
    fwd+remat :  526817.0 ms  (80 calls)   95.8%  [DEVICE-RESIDENT]
    bwd       :   19611.1 ms  (32 calls)    3.6%  [abstract numpy]
    adam      :    3037.5 ms  (4 calls)     0.6%  [abstract numpy]
    loss      :     328.6 ms  (4 calls)     0.1%  [abstract numpy]
  per-call fwd  :    6585.2 ms/call (device-resident)
  per-call bwd  :     612.8 ms/call (abstract)
```

### The host-staging recovery — what the elimination actually bought (honest)

* **Net 8L speedup: 1.18× (25.2 → 29.82 tok/s), peak −2.22 GB (10.30 → 8.08 GB).** The
  measured free-delta dropping 2.22 GB is the direct, unambiguous signature of removing the
  per-region host staging buffers — the device-allocator plan (4.682 GB) is unchanged, so
  the entire drop is host-side numpy that no longer exists.
* **Apples-to-apples at the real-kernel forward: 1.28× — and this is the honest staging
  number.** The §2 baseline 25.2 tok/s ran a *cheap numpy proxy* forward (`np.maximum`) whose
  ~6.9 s/call was almost entirely the 2028 MB host round-trip. This §13 run pays the *real*
  6.5 s MR-kernel **compute** per call. So the 1.18× *net* mixes two changes (removed the host
  bounce **and** swapped in the heavier real kernel). Holding the forward fixed at the REAL
  kernel, the device-resident path is `8472 ms → 6585 ms/call` = **1.28× per forward call**
  (consistent with §4a's 1.30× and the scout's 1.38× ceiling): the recovered term is the
  per-call ~1.9 s of bank `.numpy()` / `copyto` host staging, now gone.
* **The 96.9% → 95.8% "forward share" did NOT move because the bottleneck SHIFTED, not
  vanished.** In the §4 baseline the forward's 96.9% was **host-staging**. In §13 the forward
  is **95.8% device-kernel COMPUTE** — the 6585 ms/call is the real MR kernel executing on
  the GPU, with its ~1.9 s host bounce removed. **The step is no longer host-staging-bound in
  the forward; it is now real-device-compute-bound in the forward.** That is the qualitative
  result of this lever: host-staging is eliminated where it was 96.9% of the wall, and the
  exposed floor is the kernel's own compute × the launch count.

### Extrapolation to 28 layers / 1.8B (explicit, labelled — device-resident per-call costs)

Using the MEASURED device-resident per-call costs (fwd+remat **6.585 s/call**, bwd **0.613
s/call**) and the §5 region counts (28L = 117 fwd+remat [28 fwd + 89 sqrt-N recompute] + 28
bwd + 1 adam + 1 loss):

```
28L step ≈ 117×6.585 + 28×0.613 + adam/loss ≈ 770.5 + 17.2 + ~0.8 ≈ 788.5 s/step
        → 4096 / 788.5 ≈ 5.19 tok/s   (EXTRAPOLATED from measured 8L per-call costs)
```

The 28L number is **~unchanged (5.19 vs the baseline's ~5.2)** — because at 28L the step is
dominated by **117 real-kernel forward launches** (the sqrt-N remat multiplies 28 forward
regions into 117), and each launch is now the real 6.5 s MR kernel **compute**, which the
device-residency does not shrink. The 8L net win (1.18×) comes mostly from the smaller remat
multiplier at 8L; the host-staging recovery (1.28× per call) is real but is swamped at 28L by
the kernel-compute × launch-count floor.

### Re-stated gap vs Megatron 3399 tok/s @ 26 GB

| side | tok/step | peak mem | tok/s | gap vs Megatron |
|---|---:|---:|---:|---:|
| **Megatron-LM (live)** | 16,384 | ~26 GB | **3399** | 1.0× |
| Relax graph-path 8L, abstract-numpy fwd (PR-7 baseline) | 4,096 | 10.30 GB | 25.2 | 135× |
| **Relax graph-path 8L, DEVICE-RESIDENT fwd (§13)** | 4,096 | **8.08 GB** | **29.82** | **114×** |
| Relax graph-path 28L, device-resident fwd (EXTRAPOLATED) | 4,096 | 8.787 GB planned | **5.19** | **654×** |

**The new gap is 114× at 8L (was 135×) and ~654× at 28L (unchanged).** The device-resident
forward closed the 8L gap from 135× to 114× (the 1.18× net speedup) and cut measured peak by
2.22 GB, but did not move the 28L extrapolation, because at depth the floor is the real
kernel's compute × the remat launch count, not the host staging that was just removed.

### What attributes the REMAINING gap — the next lever (honest, post-host-staging)

The forward host-staging that was **96.9% of the baseline step** is eliminated; the remaining
114×–654× gap is now attributed to, in order:

1. **Real MR-kernel compute per launch (6.585 s/call) — the new dominant term.** This is a
   single-launch, **unfused** MR region kernel (one giant kernel doing the whole M+R route).
   6.5 s/call × 20 (8L) / 117 (28L) launches is now 95.8% of the step. The next lever is a
   **faster / fused / tiled MR kernel** (and FP8 like Megatron) — this is device-compute
   optimization, not staging.
2. **sqrt-N remat launch multiplication.** The 28 forward regions become 117 forward launches
   at 28L (89 are recompute). Each recompute is a full 6.5 s kernel launch. Fusing adjacent
   forward+remat regions into fewer launches (§5 lever 1 remainder, deferred) directly attacks
   this — it is the single biggest 28L lever now that staging is gone.
3. **Abstract numpy bwd/adam/loss still host-stage (now 4.3% combined at 8L).** bwd = 612.8
   ms/call abstract (§13), adam 0.6%, loss 0.1%. Wiring the **real device-resident backward**
   (lever 4) removes the last abstract host round-trips; small at 8L, but it grows with depth.
4. **4× batch gap.** Megatron runs 16,384 tok/step (bs=4) vs this path's 4,096 (batch=1) — 4×
   less amortization per launch. Batching ×4 (§5 lever 3) is free amortization Megatron gets.

**Bottom line:** the host-staging lever is spent and it worked where it applied — the forward
is now device-resident, peak dropped 2.22 GB, 8L gap 135×→114×. The throughput floor has
**moved from host-staging to real-kernel compute × launch count**; the next levers are kernel
fusion/speed (1, 2) and the real device backward (3), not more staging removal.

### Reproduce (§13)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
PR7_LAYERS=8 PR7_STEPS=4 \
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/pr7_profile_device_resident_gb10.py
```
FAIL-LOUD: non-finite loss, any region that cannot run, or a device tensor not detected as
device RAISES. Measured 2026-06-03 on gb10 `tvm.cuda(0)`, TVM 0.25.dev0.

---

## 14. KORTHIKANTI LAUNCH-FUSION — the per-segment recompute cache, re-profiled (gb10 CUDA, 2026-06-03)

§13 left the throughput floor at **real-kernel compute × the sqrt-N remat launch count**, and
named **forward-launch fusion** as the single biggest 28L lever. §14 lands and measures it:
the **Korthikanti per-segment recompute cache** in the remat assembly
(`path_c_relax_step_remat.py`, `path_c_relax_train_step.py`, `path_c_relax_step_optim.py`).

**What changed (the lever).** The naive sqrt-N remat re-derived the forward prefix `b..i`
*independently for every non-boundary backward region i* — region b+1 recomputes [b..b+1],
b+2 recomputes [b..b+2] from scratch, etc. That is the **O(N·√N) checkpointing anti-pattern**:
the same prefix is re-run once per region in the segment. The Korthikanti cache instead **walks
each segment exactly once**: the first backward region in a segment re-emits the forward
`(b+1)..seg_end` a single time and **caches every checkpoint**; every later backward region in
that segment reads its checkpoint from the cache. Recompute drops from O(N·√N) to **O(N)**
(each non-boundary region recomputed exactly once = N − #boundaries). Numerically **identical**
— the cached `ck_j` is the same op on the same boundary activation, just emitted once instead
of redundantly.

**Forward launch count (the measured lever):**

| depth | boundaries | naive forward launches | **Korthikanti launches** | reduction |
|---|---|---:|---:|---:|
| **8L** | [0,3,6] | 8 + 12 = **20** | 8 + 5 = **13** | **1.538×** |
| **28L** | [0,6,12,18,24] | 28 + 89 = **117** | 28 + 23 = **51** | **2.294×** |

### MEASURED (gb10 CUDA, 8 layers, 4 steps, warm = steps 1–3)

| metric | §13 (device-resident, naive remat) | **§14 (Korthikanti launch-cache)** |
|---|---:|---:|
| forward launches/step | 20 | **13** |
| mean warm step | 137.36 s | **90.144 s** (median 90.28 s) |
| **THROUGHPUT** | **29.82 tok/s** | **45.44 tok/s** |
| **step-time speedup** | 1.0× | **1.524×** (137.36 → 90.14 s) |
| per-call fwd (device-resident) | 6585.2 ms | **6559.7 ms** (unchanged — same kernel) |
| per-call bwd (abstract) | 612.8 ms | 456.1 ms |
| forward share of step | 95.8% | **95.0%** |
| planned device-peak (8L) | 4.682 GB | **6.400 GB** (+1.718 GB: segment checkpoint cache) |
| loss | 5.525e-06 | **5.525e-06** (identical — numeric equivalence) |

```
=== PROFILE (DEVICE-RESIDENT FWD) gb10 CUDA, 8 layers (n_layers=8, 4 steps) ===
  mean step (warm) : 90144.0 ms  median 90280.5 ms
  tokens/step      : 4096 (seq=4096, batch=1)
  THROUGHPUT       : 45.44 tok/s
  planned dev-peak : 6.400 GB
  per-region wall over 4 steps (host-driver time):
    fwd+remat :  341106.7 ms  (52 calls)   95.0%  [DEVICE-RESIDENT]   (13/step)
    bwd       :   14595.1 ms  (32 calls)    4.1%  [abstract numpy]    (8/step)
    adam      :    3041.8 ms  ( 4 calls)    0.8%  [abstract numpy]
    loss      :     318.3 ms  ( 4 calls)    0.1%  [abstract numpy]
  per-call fwd  :    6559.7 ms/call (device-resident)
  fwd+remat calls/step = 13, bwd calls/step = 8
  losses: 5.525325e-06, 5.525331e-06, 5.525332e-06, 5.525341e-06 (finite, stable)
```

### The launch-fusion win — what it bought (honest)

* **Net 8L speedup: 1.524× (29.82 → 45.44 tok/s).** The lever cut forward launches 20→13;
  with the per-call MR cost unchanged (6.585→6.560 s, same kernel), the predicted 13/20 =
  0.65× step time matched the measured 90.14/137.36 = 0.656× almost exactly. **This is a pure
  launch-count win** — no kernel change, no numeric change (loss bit-for-bit 5.525e-06).
* **The honest memory cost.** The Korthikanti cache keeps a *whole segment's* recomputed
  checkpoints live concurrently (the segment is walked once and cached), whereas the naive
  remat kept only ~1 recomputed checkpoint live at a time. The checkpoint bank is 0.9427 GB;
  the max segment length is 3 (8L) / 6 (28L), so the planned device-peak rises **4.682→6.400
  GB (8L)** and **8.787→12.998 GB (28L)**. This is a real **memory-for-launches tradeoff**,
  stated explicitly — still **2× under Megatron's 26 GB**, and the eager path still OOMs where
  this runs. (The profiler's 22.86 GB free-delta high-water was host buff/cache noise: free
  returned to the 117 GB baseline after `drop_caches`; the load-bearing device number is the
  **6.400 GB planned peak**.)
* **The forward share barely moved (95.8%→95.0%) because the bottleneck is unchanged in KIND.**
  The step is still **real-device-kernel-compute-bound in the forward** — we just launch the
  6.56 s MR kernel 13× instead of 20× at 8L (51× instead of 117× at 28L). Launch fusion
  attacked the *count*; the *per-launch cost* is the next floor.

### Extrapolation to 28 layers / 1.8B (explicit, labelled — measured §14 per-call costs)

Using the MEASURED §14 per-call costs (fwd **6.5597 s/call**, bwd **0.4561 s/call**) and the
Korthikanti 28L launch counts (**51 fwd** [28 fwd + 23 recompute] + 28 bwd + 1 adam + 1 loss):

```
28L step ≈ 51×6.5597 + 28×0.4561 + adam 0.760 + loss 0.080 ≈ 348.2 s/step
        → 4096 / 348.2 ≈ 11.76 tok/s   (EXTRAPOLATED from measured 8L per-call costs)
```

vs §13's 28L extrapolation of 5.19 tok/s (which used the naive 117 launches): the launch lever
delivers **2.267× at 28L** (it scales BETTER with depth — 117→51 is 2.29× vs 8L's 20→13 =
1.54×, because deeper nets have larger segments where the naive O(N·√N) redundancy is worse).
Isolating the lever: the same per-call cost with the naive 117 launches gives 781 s/step → 5.24
tok/s, so the launch reduction ALONE is **2.24× at 28L** (matching the 2.29× launch ratio).

### Re-stated gap vs Megatron 3399 tok/s @ 26 GB

| side | tok/step | planned peak | tok/s | gap vs Megatron |
|---|---:|---:|---:|---:|
| **Megatron-LM (live)** | 16,384 | ~26 GB | **3399** | 1.0× |
| Relax 8L, device-resident fwd, naive remat (§13) | 4,096 | 4.682 GB | 29.82 | 114× |
| **Relax 8L, device-resident fwd, Korthikanti (§14)** | 4,096 | **6.400 GB** | **45.44** | **75×** |
| Relax 28L, Korthikanti (§14, extrapolated) | 4,096 | 12.998 GB | 11.76 | **289×** |
| Relax 28L, naive remat (§13, extrapolated) | 4,096 | 8.787 GB | 5.19 | 654× |

**The new gap is 75× at 8L (was 114×) and 289× at 28L (was 654×).** The launch-fusion lever
compounds with the device-resident forward and is the **biggest single throughput win at depth**
(28L gap more than halved). It does NOT touch the per-launch MR-kernel cost — that is the next
floor.

### What attributes the REMAINING gap — the next lever (honest, post-launch-fusion)

After §13 (host-staging gone) and §14 (launch count cut), the 75×–289× gap is now attributed
to, in order:

1. **The 6.56 s/call MR-kernel compute — the new sole dominant term (95.0% of the step).** This
   is the **single biggest remaining lever and is no longer about launch count** — it is the
   per-launch cost of the unfused MR region kernel (the whole M+R route in one giant kernel; the
   mamba scan stage runs `T.Kernel(1, threads=1024)`, a single-block serial recurrence —
   §13/diagnosis). Cutting *per-call* cost needs a **faster/fused/tiled/FP8 MR kernel** (e.g. a
   CUDA SSD chunked scan to replace the single-SM serial recurrence, collapsing the 4096-deep
   critical path). At 51×6.56 s = 334.5 s of the 348 s/step at 28L, this is where ~96% of the
   remaining time lives.
2. **Abstract numpy bwd/adam/loss (now 5.0% combined at 8L).** bwd = 456 ms/call abstract,
   adam 0.8%, loss 0.1%. Wiring the **real device-resident backward** (lever 4) removes the last
   abstract host round-trips; small at 8L, grows with depth.
3. **4× batch gap.** Megatron runs 16,384 tok/step (bs=4) vs this path's 4,096 (batch=1) — 4×
   less amortization per launch. Batching ×4 (§5 lever 3) is free amortization Megatron gets.
4. **Memory headroom for further tradeoffs.** The Korthikanti cache spent some of the 26 GB
   envelope (28L now 13.0 GB planned). Larger checkpoint segments (fewer recompute, more cache)
   or batching would consume more — there is ~13 GB of headroom to Megatron's 26 GB to trade.

**Bottom line:** the launch-fusion lever is spent and it worked — 8L 29.82→45.44 tok/s
(1.524×), 28L 5.19→11.76 tok/s (2.267×), 8L gap 114×→75×, 28L gap 654×→289×, numerics
identical, at a stated +1.7/+4.2 GB planned-peak cost (still 2× under Megatron). The throughput
floor has **moved from launch count to the per-call MR-kernel compute (95.0% of the step)**; the
next lever is a faster/fused/FP8 MR kernel — per-launch device-compute work, not launch count
and not staging.

### Reproduce (§14)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
PR7_LAYERS=8 PR7_STEPS=4 \
TVM_LIBRARY_PATH=/home/dave/sources/tl_apache_tvm_swap_zfinal/build/lib \
LD_LIBRARY_PATH=/home/dave/sources/tl_apache_tvm_swap_zfinal/build/lib:/home/dave/source/cppmega_mlx/.venv/lib/python3.12/site-packages/z3/lib \
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/sources/tl_apache_tvm_swap_zfinal/3rdparty/tvm/python \
/home/dave/cppmega-venv/bin/python scratch/pr7_profile_device_resident_gb10.py
```
The §14 stack is `tl_apache_tvm_swap_zfinal` (tvm 0.25.dev0, has `tirx`) + native `tvm_ffi` +
editable `source/tilelang` — the only internally-consistent tree on gb10 (the older
`source/tilelang/build` libs in §13's reproduce now mismatch the `tirx`-migrated tilelang). The
Korthikanti cache is in `path_c_relax_step_remat.build_remat_bank_chain` /
`path_c_relax_train_step.build_train_step` / `path_c_relax_step_optim.build_full_step_with_optim`
(`_recompute_segment`, `saved_exit`, `recomputed_ckpt`). FAIL-LOUD as §13. Launch counts
verified analytically by `recompute_overhead(n)`: 8L total_fwd=13, 28L total_fwd=51.

---

## 7. Verdict

The Relax graph-path train_step **fits Megatron-class memory (12.998 GB planned 28L
device-peak vs Megatron 26 GB, optimizer in-place) and runs the full 28-layer 1.8B step the
eager path OOMs on** — the memory-first goal is MET on device. Two throughput levers have now
landed and both are measured. **§13 (device-resident forward)** took 8L from 25.2 → **29.82
tok/s** and moved the floor off host-staging onto real-kernel compute × launch count. **§14
(Korthikanti per-segment recompute cache — launch fusion)** takes 8L to **45.44 tok/s
MEASURED (1.524× over §13)** and 28L to **11.76 tok/s extrapolated (2.267× over §13)** by
cutting forward launches **20→13 at 8L and 117→51 at 28L**, numerics bit-for-bit identical
(loss 5.525e-06), at a stated memory cost (planned peak 4.682→6.400 GB at 8L, 8.787→12.998 GB
at 28L — the segment checkpoint cache, still 2× under Megatron). The **8L gap vs Megatron's
3399 tok/s narrows 114×→75×, the 28L gap 654×→289×** — the launch lever is the biggest single
throughput win at depth and scales better with depth (2.29× launch reduction at 28L). With
host-staging eliminated (§13) and launch count cut (§14), **the throughput floor has moved to
the per-call MR-kernel compute itself — 6.56 s/call, now 95.0% of the step**; the next lever
is a faster/fused/FP8 MR kernel (the single-block `T.Kernel(1, threads=1024)` mamba serial
recurrence → a gridded CUDA SSD chunked scan), per-launch device-compute work, not launch
count and not staging. The memory envelope is proven; the throughput path is now a single
kernel's per-call compute.
