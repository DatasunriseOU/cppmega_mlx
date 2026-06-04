# Relax graph-path train_step vs Megatron — measured tok/s + peak memory (gb10 CUDA)

**Status: MEASURED, 2026-06-04, gb10 (Grace-Blackwell aarch64, `tvm.cuda(0)`). Latest: §18 —
the THREE named Megatron-gap levers (4× batch / fp8 activations / B2 v2 dstate-split) RE-MEASURED:
ALL THREE land as NO-GO-for-throughput, the §17 ≈907/≈298 tok/s GO stands unchanged. (1) bs4
FORWARD is MEASURED + parity-PASS (chain 7.231→≈29.59 ms, 4.09× — exactly 4× grid, batch reaches the
launch grid) but **tok/s is batch-invariant** (the SSD kernels were already occupancy-saturated at
bs1, so 4× tokens cost ~4× time → NO utilization gain, gap stays ≈3.75×/≈11.4×); the bs4 BACKWARD
probe is a **memory-NO-GO** — its torch-autograd gold reference OOMs the box (≈110 GB > 100 GB
budget, SIGTERMed), the kernels themselves never being the limit; bs4 device-peak ≈25.6/52 GB fp16
(28L now 2× OVER Megatron's 26 GB). (2) FP8 GEMMs are **UNRUNNABLE on sm_121** — TE MXFP8 RAISES
"not supported on 12.0+ architectures", TE tensor-wise RAISES NVRTC failure, our coop fp8 T.gemm is
Metal-only; bf16 baselines MEASURED (33–34 TFLOPs at prod shapes), operand byte-halving **2.0×
MEASURED** (the memory synergy is real; the fp8 SPEED is UNMEASURED — a CUDA fp8 emission is the
path forward). (3) B2 v2 dstate-split is a **NO-GO** — measured in-process v1-vs-v2 = **0.997×/1.001×
(zero speedup)**, math-equivalent (dD 5.25e-6, others bit-identical); B2 stays v1, contingent
≈1480/≈560 not achieved. (Today's box ran the backward ~3× slower than §17 uniformly across the
byte-identical B0/B1 too — a box-state effect, NOT a code regression; §17's 334.6 ms canonical
number is NOT revised.) Prior: §17 — the gridded CUDA SSD chunked-BACKWARD is a GO (B2
**2484 → 334.6 ms/call (7.42×)**, chain **2601 → 447.8 ms (5.81×)**, 1.018× faster than the 456 ms
numpy backward; dD **1.40e-3 → 2.48e-5**; ALL 8 grads pass the 1e-3 gate → **≈907 tok/s @8L /
≈298 @28L, gap ≈3.75×/≈11.4×**), Metal prim byte-identical, memory UNCHANGED (6.400/12.998 GB).**
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
| **peak memory** | **6.400 GB planned 8L / 12.998 GB planned 28L device-peak** (Korthikanti launch-cache §14; the gridded SSD scan §15 reuses the same banks → peak UNCHANGED; was 4.682/8.787 GB pre-§14 — the launch lever trades +1.7/+4.2 GB for the launch reduction, still 2x under Megatron) | **~26 GB** |
| **fits Megatron-class memory** | **YES — 13.0 GB planned 28L < 26 GB (2x under)** | 26 GB |
| **tok/s** | **§17 (gridded fwd + RE-GRIDDED gridded backward, MEASURED B2 substituted): ≈907 tok/s @8L / ≈298 @28L (gap ≈3.75×/≈11.4×) — B2 re-gridded 2484→334.6 ms (7.42×), chain 447.8 ms < 456 ms numpy bwd, ALL 8 grads pass incl dD 2.48e-5. §18 RE-MEASURED the 3 gap levers: bs4 fwd MEASURED+PASS but tok/s batch-invariant (saturated kernels, gap unchanged ≈3.75×/≈11.4×), bs4 bwd memory-NO-GO (gold harness OOM), fp8 UNRUNNABLE on sm_121 (byte-halving 2.0× MEASURED, speed UNMEASURED), B2 v2 NO-GO (0.997×/1.001×) — §17 headline holds. §21 RE-MEASURED the fp8 levers on the unblocked sm_121: our OWN native fp8-input e4m3 MMA is now VERIFIED real (kFloat8_e4m3 m16n8k32, 0 fp16) and 2.2–3.5× faster than the old fp16-decode but still 0.69–0.95× bf16 (untuned tile → speed NO-GO); TE standalone fp8 GEMM stays the only ~1.7× win; wiring TE fp8 into the e2e step is a memory-NO-GO (fp8 backward OOMs at the MLX↔torch cudaMallocAsync bridge, reproduced to seq=512) — so §17's ≈907/≈298 stays the canonical step number, fp8 e2e tok/s UNMEASURED. Prior: ≈894/≈293 @ §15 (gridded scan + numpy bwd, EXTRAPOLATED); 45.44 @8L MEASURED (§14); 29.82 @ §13, 25.2 pre-§13.** | **3399 tok/s** |
| tok/s ratio vs Megatron | **≈0.27x (8L) / ≈0.088x (28L) with §17 (gridded fwd + gridded bwd) — i.e. ≈3.75x–11.4x slower** (was 75x–289x @ §14, 114x–654x @ §13) | 1.0x |
| what runs the compute | **REAL tilelang path_c-CUDA MR kernel, device-resident fwd (§13) + Korthikanti recompute cache (§14) + GRIDDED CUDA SSD chunked scan replacing the serial MR mamba recurrence (§15, F2 0.980 ms vs serial 6.56 s)** + abstract numpy bwd/adam/loss (lever 4, now the dominant remaining term) | tuned fused-FP8-CUDA kernels + selective recompute |

**(a) Does the graph path FIT in Megatron-class memory? YES.** The StaticPlanBlockMemory
planned device-peak for the whole 28-layer 1.8B step is **8.787 GB** — roughly **3x under**
Megatron's ~26 GB, and the optimizer state is kept at 1x by the explicit in-place Adam op
(lever-5). The eager MLX path cannot even run this step as one monolithic graph (OOMs at
114-116 GB on the same box, `MEGATRON-VS-MLX-PATHS.md`); the graph path runs it and plans
to 8.79 GB. **This is the win the whole memory-path was built for.**

**(b) What tok/s does it achieve? THREE levers spent, all worked — and the third (the gridded
scan) is the breakthrough that brings the path within one order of magnitude of Megatron at
depth.** The §2/§4 PR-7 baseline (abstract numpy host-staged forward) measured **25.2
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
memory/launch tradeoff, still 2× under Megatron's 26 GB. **§15 lands the third lever — the
gridded CUDA SSD chunked scan.** §14 left the floor at the **6.56 s/call MR kernel** (95.0% of
the step), diagnosed as a **single-block serial mamba recurrence** (`T.Kernel(1)`,
`T.serial(0,4096)` over the whole sequence inside ONE threadgroup). §15 replaces it with a
**28 672-threadgroup gridded SSD scan** (Mamba-2/SSD: parallel within chunks + recurrence
between chunks): **F2 scan = 0.980 ms vs the serial 6.56 s ≈ 6694×** (re-measured live on gb10
this session, fp16 parity 4.746e-04 PASS; gridded forward chain 7.231 ms/call; delegation
builds compiled CUDA JITKernels at production shape). Substituted into the §14 step (labelled
extrapolation, self-checks to §14's 45.44 tok/s within 0.4%): **8L → ≈894 tok/s (≈19.7×), 28L →
≈293 tok/s (≈24.9×)**, at **no memory cost** (planned peak unchanged). The **8L gap collapses
75×→≈3.8×, the 28L gap 289×→≈12×** — within one order of magnitude of Megatron at depth. The
forward (the old 95% term) collapses to ~2% of the step; **the bottleneck flips to the abstract
numpy backward (79.6% @8L / 91.4% @28L, lever 4)** — the next lever is the real device-resident +
gridded-SSD backward, then the 4× batch and FP8 GEMM. See §4 for the baseline, **§13 for the
device-resident re-profile, §14 for launch-fusion, §15 for the gridded-scan result + the
attribution flip**, and §5 for the lever list.

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

## 15. GRIDDED CUDA SSD CHUNKED-SCAN MR KERNEL — the per-call MR floor falls (gb10 CUDA, 2026-06-03)

§14 left the throughput floor at the **6.56 s/call MR kernel** (95.0% of the step), and the
diagnosis pinned the cause: the MR kernel's mamba sub-region is a **single-block serial
recurrence** — `with T.Kernel(1, threads=1024)` running `for time_rev in T.serial(0, S=4096)`
over the WHOLE sequence inside ONE threadgroup (schedule line 145 / the `~5-6 s per command
buffer` note at full `local_gb10_quarter` scale). That serial scan IS essentially the entire
6.56 s MR per-call cost. §15 lands and measures the fix: a **gridded CUDA SSD chunked scan**
(Mamba-2/SSD: parallel within chunks + recurrence between chunks, grid over
nheads × tiles × batch·nchunks) replacing the single-SM serial recurrence.

**What changed (the lever, all in `cppmega_mlx`; no tilelang/codegen changes).** The proven
Metal chunked SSD grid prims (`mamba3_chunked_scan_core` / `mamba3_chunked_precompute_core` /
`mamba3_chunked_backward_core`) were made to compile+run for `target="cuda"`. The CUDA-shaped
deltas (same SSD math): F2's `acc_o` is a **register fragment** (CUDA `T.gemm` asserts a
fragment C accumulator; Metal staged it in shared), plain `"shared"` x_shared, and
**`disable_tma=True` on every global↔shared copy + `pass_configs={disable_tma_lower,
disable_warp_specialized}`** (sm_121's TMA tensormap descriptor was 32/16-byte aligned where
TMA needs 64 → `Invalid TMA descriptor arguments`; the cp.async path is correct). `target=`
is threaded through all 7 `build_*_metal` builders; the live wiring site
`_mamba3_chunked_grid_delegation_prim` (`path_c_fusion_schedules.py:6899`) passes `target=cuda`
on a CUDA host and `block_Dstate=dstate` to F2. The serial-vs-gridded choice is the flag-ON
chunked surfaces routed through the delegation — a legitimate gate, **not** a silent fallback
(RULE #1): an unsupported shape/target RAISES inside the builder, no serial fallback.

### MEASURED — gridded SSD per-call (gb10 `tvm.cuda(0)` sm_121, RE-MEASURED this session)

Production `local_gb10_quarter` shape S=4096 chunk=64 heads=112 head_dim=64 state_dim=64
groups=8, F2 grid **(112,4,64) = 28 672 threadgroups** vs the serial scan's **1**
(`scratch/probe_chunked_scan_cuda_gb10.py --prod`):

| stage | median ms/call | parity vs serial SSD reference (fp16 max\|abs\|, gate <5e-4) |
|---|---:|---|
| **F2 scan+combine** (replaces the serial carry) | **0.980 ms** | **4.746e-04 PASS** |
| F0 precompute | 5.025 ms | cb 0.0 exact, dA_cumsum 9.766e-04 |
| F1 inter-chunk recurrence | 1.226 ms | prev_states 1.314e-04 |
| **gridded forward chain/call** | **7.231 ms** | OVERALL PASS |

The scan recurrence itself (**F2 = 0.980 ms** vs the serial **6.56 s** ≈ **6694×**); the full
chunked forward chain is **7.231 ms vs 6 559.7 ms ≈ 907×**. **E2E wiring proven**: the live
delegation site `_mamba3_chunked_grid_delegation_prim` returns **compiled CUDA JITKernels** for
F0/F1/F2 at production shape on a CUDA host (`scratch/probe_mr_chunked_delegation_cuda_gb10.py`
→ `DELEGATION: PASS`) — the MR mamba sub-region is driven by the gridded scan, never the serial
`T.Kernel(1)`. (Both probes re-run live on gb10 this session, GPU idle, single-run discipline.)

### Step model — gridded MR forward into the §14 train_step (MEASURED per-call costs)

The §14 profiler builds the **monolithic** MR kernel (one fused `T.prim_func`, one `T.Kernel`)
whose serial scan = 6.5597 s/call. The gridded scan is the flag-ON chunked F0/F1/F2 surfaces
delegated as separate grid kernels (the monolithic single-kernel template cannot host a
multi-grid `T.Kernel` as a fragment — hence the delegation). So §15 is a **labelled
extrapolation**: substitute the MEASURED gridded forward per-call (**7.231 ms**) for the
serial MR forward per-call (**6 559.7 ms**) in the §14 step, keeping every other §14 MEASURED
quantity fixed (Korthikanti launch counts, bwd 456.1 ms/call, adam 0.760 s, loss 0.080 s). The
substitution is justified by the diagnosis: the serial scan IS the ~5-6 s command buffer =
essentially the whole 6.56 s MR forward per-call. **Self-check:** reconstructing the §14 serial
8L step from these per-call costs gives 89.76 s → 45.63 tok/s vs the §14 MEASURED 90.144 s →
45.44 tok/s (within 0.4% — the model is consistent).

| metric | §14 (serial MR scan) | **§15 (gridded SSD MR scan)** |
|---|---:|---:|
| MR forward per-call | 6 559.7 ms (serial `T.Kernel(1)`) | **7.231 ms (gridded chain, MEASURED)** |
| **8L step** | 90.144 s | **≈ 4.58 s** |
| **8L THROUGHPUT** | **45.44 tok/s** | **≈ 894 tok/s** (EXTRAPOLATED) |
| 8L speedup vs §14 | 1.0× | **≈ 19.7×** |
| 28L step (extrapolated) | 348.2 s | **≈ 13.98 s** |
| **28L THROUGHPUT** | 11.76 tok/s | **≈ 293 tok/s** (EXTRAPOLATED) |
| 28L speedup vs §14 | 1.0× | **≈ 24.9×** |
| forward share of step | 95.0% | **2.1% (8L) / 2.6% (28L)** — collapsed |

### Re-stated gap vs Megatron 3399 tok/s @ 26 GB

| side | tok/step | planned peak | tok/s | gap vs Megatron |
|---|---:|---:|---:|---:|
| **Megatron-LM (live)** | 16,384 | ~26 GB | **3399** | 1.0× |
| Relax 8L, Korthikanti, serial MR scan (§14) | 4,096 | 6.400 GB | 45.44 | 75× |
| **Relax 8L, gridded SSD MR scan (§15)** | 4,096 | 6.400 GB | **≈ 894** | **≈ 3.8×** |
| **Relax 28L, gridded SSD MR scan (§15, extrap)** | 4,096 | 12.998 GB | **≈ 293** | **≈ 12×** |
| Relax 28L, serial MR scan (§14, extrap) | 4,096 | 12.998 GB | 11.76 | 289× |

**The new gap is ≈3.8× at 8L (was 75×) and ≈12× at 28L (was 289×).** The gridded scan is by
far the biggest single throughput win of the four levers — at depth it more than 24×'s the
throughput, because the serial scan it removes was 95% of the step and is independent of launch
count. **No memory cost:** the F2 gridded scan reuses the same checkpoint banks; planned peak
is unchanged at 6.400 GB (8L) / 12.998 GB (28L), still 2× under Megatron's 26 GB.

### What attributes the REMAINING gap — the next lever (honest, post-gridded-scan)

The forward — the 6.56 s/call serial scan that was **95.0% of the §14 step** — is gone
(→ 7.231 ms/call, now ~2% of the step). The bottleneck **flips entirely** to the one region
still on the abstract numpy host-staged path:

1. **The abstract numpy BACKWARD is now the sole dominant term — 79.6% at 8L, 91.4% at 28L.**
   bwd = 456.1 ms/call × 8 (8L) / 28 (28L) abstract numpy host round-trips (lever 4 of §5,
   never yet device-resident). With the forward collapsed, this is the new floor. **The next
   lever is the REAL device-resident path_c-CUDA backward** — exactly the same treatment §13
   gave the forward (device-resident banks, no `.numpy()`) plus a **gridded SSD chunked backward**
   (B0/B1/B2, the analytic transpose of F0/F1/F2). The backward chunked cores are already
   `target`-threaded and use **no `T.gemm`** (plain TileLang, like F0/F1 which ported to CUDA
   unchanged), so they are expected to port with the same target threading; their parity
   reference is MLX-only today, so a CUDA reference path is needed to validate them — the
   follow-up.
2. **adam (16.6% at 8L / 5.4% at 28L) and loss (1.7% / 0.6%)** are also still abstract numpy.
   Once the backward is device-resident, in-place device adam + device loss remove the last
   abstract host round-trips.
3. **4× batch gap.** Megatron runs 16,384 tok/step (bs=4) vs this path's 4,096 (batch=1). With
   the forward now essentially free per token, batching ×4 is nearly-free amortization and
   directly closes a 4× slice of the remaining gap.
4. **GEMM / FP8.** After the abstract bwd/adam/loss are device-resident, the residual gap to
   Megatron is large-batch fused-FP8 GEMM throughput — the same class of kernel optimization,
   now on a path where compute (not staging, not launch count, not the serial scan) is the only
   remaining term.

**How close to Megatron-class THROUGHPUT?** We already match Megatron-class **MEMORY** (13.0 GB
planned 28L vs 26 GB, 2× under). With the gridded scan, the 28L throughput gap drops from 289×
to **≈12×** and the 8L gap to **≈3.8×** — i.e. the graph path is now within **one order of
magnitude** of Megatron throughput at depth (was nearly three), and the entire remaining gap is
attributable to ONE region (the abstract numpy backward) plus the 4× batch — not the model, not
the device-compute kernels, not the scan. **Honest bracket (RULE #1):** the headline assumes the
serial scan is ~100% of the 6.56 s MR forward per-call (the diagnosis). If a residual non-scan
MR forward term survived (e.g. 0.5–1.0 s/call), 8L would land ≈370–223 tok/s and 28L ≈104–60
tok/s — but across the WHOLE bracket the robust finding is unchanged: **the forward is no longer
the bottleneck; the abstract numpy backward is.** A fused-kernel direct-chain `train_step` that
runs the gridded F0/F1/F2 forward AND a device-resident backward e2e (vs the monolithic-MR
profiler substitution used here) is the measurement that turns this labelled extrapolation into
a measured tok/s — the documented next run.

### Reproduce (§15)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
# gridded SSD per-call timings + parity (production shape):
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/probe_chunked_scan_cuda_gb10.py --prod
# live MR delegation interpose builds CUDA grid kernels (e2e wiring):
... /home/dave/cppmega-venv/bin/python scratch/probe_mr_chunked_delegation_cuda_gb10.py
```
Measured 2026-06-03 on gb10 `tvm.cuda(0)` sm_121: F2 0.980 ms (grid 28 672 tg, parity
4.746e-04 PASS), F0 5.025 ms, F1 1.226 ms; DELEGATION PASS. Commits (DatasunriseOU `cppmega_mlx`
main): `77cb4a6` (gridded CUDA SSD scan replaces serial MR recurrence), `b961c6d` (e2e
delegation probe). FAIL-LOUD: unsupported shape/target RAISES inside the builder, no serial
fallback.

---

## 16. GRIDDED CUDA SSD CHUNKED-BACKWARD (B0/B1/B2) — COMPILES + RUNS on gb10, but a PARITY MISS (dD) and a PERF REGRESSION (B2); NO-GO as-is (gb10 CUDA, 2026-06-03)

This is the backward counterpart of §15: wire the already-written gridded SSD chunked-**backward**
cores (B0/B1/B2, the analytic transpose of F0/F1/F2) into the region execution so the backward
becomes device-resident gridded compute instead of the abstract numpy host-staged path that §15
left as the dominant term (79.6% @8L / 91.4% @28L). The compile/parity/per-call was MEASURED on
gb10 `tvm.cuda(0)` sm_121 via `scratch/probe_chunked_backward_cuda_gb10.py --prod` at production
shape (b=1, S=4096, chunk=64, G=8, H=112, P=64, N=64, nchunks=64; tg B2/B1/B0 = 7168/112/7168).

### Compile — PASS, no CUDA sibling needed (cuda-readiness confirmed)

All three backward kernels **COMPILED and RAN on CUDA** with only the §15 forward `pass_configs`
(`tl.disable_tma_lower=True`, `tl.disable_warp_specialized=True`) threaded into `tilelang.compile`
on the `target="cuda"` branch. **NO Metal-ism surfaced; NO new CUDA sibling prim was needed** —
exactly as cuda-readiness predicted (B0/B1/B2 have zero `T.gemm` / `shared.dyn` /
`make_swizzled_layout`). The fp32 `T.atomic_add` paths (dD, dx accumulation) lowered to CUDA
`atomicAdd(float)` without error. The only compile diagnostic was a benign ThreadSync hoist
warning (`lane == 0`, also present in the §15 forward). `cuda_port_edits_needed: NONE`.

### Parity — FAIL: 7/8 grads pass, dD = 1.40e-3 EXCEEDS the 1e-3 gate (deterministic over 2 runs)

Per-grad max|abs| over ALL elements vs the serial-autograd GOLD oracle **directly** — the probe
takes the torch.autograd VJP of our exact serial recurrence (`_chunked_mamba3_diagonal_scan`) as the
gold and compares the gridded B0/B1/B2 chain against it with no MLX-proto in the loop (strictly
stronger than vs-proto: no proto↔gold slack). Gate < 1e-3, checked over every element (no row/head
subset; per-grad NaN/Inf RAISES — the discipline the §13 forward inf-bug lacked):

| grad | max\|abs\| (run 1 / run 2) | gate | verdict |
|---|---:|---:|:--|
| dz | 1.73e-4 / 1.73e-4 | 1e-3 | PASS |
| dx | 8.10e-4 / 8.10e-4 | 1e-3 | PASS (closest passing) |
| dC | 5.09e-5 / 5.09e-5 | 1e-3 | PASS |
| dB | 1.09e-5 / 1.09e-5 | 1e-3 | PASS |
| dlog_decay | 6.75e-4 / 6.75e-4 | 1e-3 | PASS |
| ddt | 1.50e-4 / 1.50e-4 | 1e-3 | PASS |
| dh0 | 1.84e-4 / 1.84e-4 | 1e-3 | PASS |
| **dD** | **1.400e-3 / 1.401e-3** | 1e-3 | **FAIL (1.4×)** |
| **WORST** | **1.401e-3** | 1e-3 | **FAIL** |

No grad is NaN/inf. The miss is on **dD only**, and it is **deterministic** (1.400e-3 / 1.401e-3
across two independent prod runs). dD is the most-accumulated grad of the eight: it reduces over
**B×S = 1×4096 positions** (per head, per headdim cell) in fp16-staged cache, so it sees the
longest fp16 accumulation chain. 1.40e-3 is a borderline fp16-accumulation precision miss (the
other seven, including the longer fp16-chained dx=8.1e-4 and dlog_decay=6.75e-4, pass), **NOT a
NaN/garbage correctness break** — but it IS a true FAIL of the documented 1e-3 gate. Per RULE #1
this is reported as a FAIL, not masked: the gridded backward **cannot replace the numpy backward
in production until dD is fixed** (fp32 dD accumulation / a higher-precision reduction for the
D-skip path, or a re-justified dD-specific tolerance band — the design doc gate is 1e-3).

### Per-call — REGRESSION: the chain is 2.60 s/call, 5.7× SLOWER than the 456.1 ms numpy backward

| kernel | per-call (warm median ×20), run 1 / run 2 | role |
|---|---:|:--|
| B0 (chunk_precompute_bwd) | 110.6 / 111.2 ms | dx(inp), dB, dlog_decay, ddt |
| B1 (inter_chunk_recur_bwd) | 2.41 / 2.42 ms | dstates, dh0, dA_cumsum_tail (reverse carry) |
| **B2 (chunk_scan_combine_bwd)** | **2484.0 / 2490.8 ms** | dC, dx(D-skip), dz, dchunk_states, dinp, dD |
| **gridded backward chain total** | **≈ 2601 ms/call** | B0+B1+B2 |

**B2 alone is ~2.49 s/call** — the heavy per-thread `T.serial` nesting on sm_121 (the PERF risk
flagged in the cuda-readiness note) dominates. The numpy backward it was meant to replace is
**456.1 ms/call** (§14). So the gridded backward as written is **5.7× SLOWER per call** — a
throughput **regression**, not a win. B1 (the one genuinely new reverse upper-tri adjoint scan) is
excellent at 2.41 ms; B0 is acceptable at 111 ms; **B2 is the whole problem.**

### Step model — substituting the MEASURED gridded backward (labelled; self-checks to §15 within 0.02%)

Substitute the MEASURED gridded backward chain (**2601 ms/call**) for the §14 numpy backward
(**456.1 ms/call**) in the §15 step, keeping every other MEASURED quantity fixed (gridded fwd
7.231 ms/call, Korthikanti launches fwd 13/51 + bwd 8/28 @8L/28L, adam 0.760 s, loss 0.080 s).
**Self-check:** with the numpy bwd this model reproduces §15 exactly — 893.8 tok/s @8L (doc 894),
293.0 tok/s @28L (doc 293), within 0.02%, so the model is consistent.

| metric | §15 (numpy bwd 456.1 ms) | **§16 (MEASURED gridded bwd 2601 ms)** |
|---|---:|---:|
| backward per-call | 456.1 ms (abstract numpy) | **2600.7 ms (gridded chain, MEASURED)** |
| **8L step** | 4.583 s | **21.74 s** |
| **8L THROUGHPUT** | ≈894 tok/s | **≈188 tok/s (EXTRAPOLATED, REGRESSION 4.74×)** |
| **28L step** | 13.98 s | **74.03 s** |
| **28L THROUGHPUT** | ≈293 tok/s | **≈55 tok/s (EXTRAPOLATED, REGRESSION 5.30×)** |
| backward share of step | 79.6% / 91.4% | **95.7% (8L) / 98.4% (28L)** — now *almost the whole step* |

### Re-stated gap vs Megatron 3399 tok/s @ 26 GB

| side | tok/step | planned peak | tok/s | gap vs Megatron |
|---|---:|---:|---:|---:|
| **Megatron-LM (live)** | 16,384 | ~26 GB | **3399** | 1.0× |
| Relax 8L, §15 (gridded fwd, **numpy bwd**) | 4,096 | 6.400 GB | ≈894 | ≈3.8× |
| **Relax 8L, §16 (gridded fwd + MEASURED gridded bwd)** | 4,096 | 6.400 GB | **≈188** | **≈18×** |
| **Relax 28L, §16 (extrap)** | 4,096 | 12.998 GB | **≈55** | **≈61×** |

**The gap WIDENS to ≈18× @8L (was 3.8×) and ≈61× @28L (was 12×)** — the opposite of the
intended effect. Memory is **UNCHANGED at 6.400/12.998 GB** (same banks; the backward reuses the
forward checkpoint/handoff banks — verified: probe peak resident was 48.6 GB host-side process
RSS, not device-bank growth; the device-bank plan is identical to §15).

### Honest attribution — what this run actually establishes (RULE #1)

1. **The gridded backward COMPILES and RUNS on CUDA with no port work** — the device-resident
   gridded backward is a *viable code path* (one path, no fallback). This is the real positive.
2. **It is NOT yet a throughput win** — B2's per-thread serial nesting makes the chain 5.7×
   slower than the numpy backward. Until **B2 is re-gridded** (the chunk-scan-combine backward
   needs the same parallel-over-(chunk,head) tiling the forward F2 got — currently it serializes
   the combine over positions per thread), the gridded backward is a regression and **MUST NOT
   replace the numpy backward** (doing so silently would itself be a forbidden RULE #1
   degradation in the *other* direction — trading a 456 ms path for a 2601 ms one).
3. **dD parity misses the gate by 1.4×** — a fp16-accumulation precision issue on the longest
   reduction (B×S over the D-skip path), fixable with fp32 dD accumulation; until fixed, parity
   is a documented FAIL.
4. **Therefore the production backward stays on the numpy path for now** — and §16's honest
   verdict is that the *remaining* bottleneck after §15 (the backward) is **not closed by this
   first gridded-backward wiring**; it needs (a) a B2 perf rewrite and (b) a dD precision fix
   before it lands. The forward win (§15) stands; the backward is **two concrete, scoped fixes**
   away from being the lever it was meant to be.
5. **Honest bracket.** If B2 were brought to B0's class (~111 ms) and B1's (~2.4 ms) — i.e. the
   chain to ~225 ms/call (BELOW the 456 ms numpy bwd) — the §16 model would give ≈1480 tok/s @8L
   / ≈560 tok/s @28L (gap ≈2.3× / ≈6.1×), the genuine win. That is an EXTRAPOLATION contingent on
   the B2 rewrite, NOT measured; the MEASURED number today is the **188/55 regression** above.

### Reproduce (§16)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/probe_chunked_backward_cuda_gb10.py --prod
```
Measured 2026-06-03 on gb10 `tvm.cuda(0)` sm_121 (two independent runs, agree to 3 sig figs):
COMPILE B0/B1/B2 PASS (no CUDA sibling); per-call B0 110.6/111.2 ms, B1 2.41/2.42 ms, B2
2484.0/2490.8 ms; per-grad WORST dD=1.400e-3/1.401e-3 (gate 1e-3) → **PARITY FAIL**;
OVERALL FAIL. Commit (DatasunriseOU `cppmega_mlx` main): `6969a64` (gridded backward wiring +
CUDA pass_configs + RULE #1 no-numpy-fallback raise + this probe). FAIL-LOUD: the probe RAISES on
any NaN/inf grad and prints the failing grad on a gate miss — no degraded numpy fallback.

---

## 17. B2 RE-GRID + dD FIX — both §16 NO-GO counts CLEARED; the gridded backward is now a GO (gb10 CUDA sm_121, 2026-06-04)

§16 was a NO-GO on two MEASURED counts: B2 = 2484 ms/call (the whole-chain regression) and dD
parity = 1.40e-3 (> 1e-3 gate). §17 fixes BOTH. The chunk-scan-combine backward B2 gets a NEW
CUDA-only prim `chunk_scan_combine_bwd_cuda_prim` (mirroring the §15 forward F2 cuda-sibling
pattern, commit `77cb4a6`): selected in `build_chunk_scan_combine_bwd_metal` ONLY when the
resolved target is CUDA, same `pass_configs` (`tl.disable_tma_lower`/`tl.disable_warp_specialized`)
and `out_idx [11..17]`; the Metal prim `chunk_scan_combine_bwd_metal_prim` stays
**byte-identical** (verified — Metal callers unaffected). B0/B1 prims+builders UNTOUCHED.

### What the re-grid did (the two lane-0 funnels → all 128 threads)

§16 root-caused B2's 2484 ms to TWO `if T.get_thread_binding(0)==0` funnels running ~8.9M serial
fp32 MACs on ONE of 128 lanes per (batch·chunk, head) threadgroup:
* **dC_diag** (the dominant ~8.5M MACs): `O(L²·dstate·headdim)` — recomputed `dyx = Σ_p dY·x`
  `dstate`-times per `ll`.
* **dseg** (~133K MACs): the segsum-VJP, recomputing the same `Σ_p dY·x` a second time as `dlmat`.

The CUDA prim (same `(batch·nchunks, nheads)` grid, 128 threads, **no new grid dims** — the
per-threadgroup `dAcs_acc`/`dY`/`DYX` tiles must stay co-resident): (1) stages `x` into shared
`XT[L,headdim]`; (2) builds the lower-tri shared tile `DYX[l,s]=Σ_p dY[l,p]·x[s,p]` **ONCE** over
`T.Parallel(L*L)` — the recompute-killer consumed by BOTH dC_diag and dseg (drops dC_diag from
`O(L²·dstate·headdim)` to `O(L²·dstate)`); (3) maps `dC_off+dC_diag+dstate_decay` over
`T.Parallel(L*dstate)` (each thread owns one race-free `dC` cell; the dstate_decay nn-reduction
folds into shared `dAcs_acc` via `T.atomic_add`); (4) maps dseg over `T.Parallel(L*L)` reusing
`DYX` (zero recompute), the +/- segsum-VJP scattering into shared `dAcs_acc` via `T.atomic_add`.
The already-threaded fast parts (dz/dx/dD, dchunk_states, dinp) are copied verbatim.

Two **CUDA-codegen scope bugs** surfaced and were fixed on-device (RULE #1: no fallback, fix the
root cause): the dseg block's first lane-strided `T.serial(0,L*L,threads)` form, then its
`T.Parallel` form with an `if ss<ll` mask, both FAILED to compile (`cb_v`/`dseg` "undefined") —
TileLang inserts a `__syncthreads()` between the global `cb` load and the shared-`dAcs_acc` atomic,
and when that barrier lands inside an `if`-masked region it re-emits the guard per fragment without
hoisting the locals. Fix: a **branchless** unmasked `T.Parallel(L*L)` dseg body (strict-lower-tri
selection via `T.if_then_else` as a 0/1 multiplier), structurally identical to the proven dC apply
block. A subsequent shared-`LMAT[L,L]` exp2-precompute experiment was MEASURED to **regress** B2
(334.6 → 408.3 ms — the kernel is occupancy-bound, the +16 KB shared cost occupancy that the
removed exp2 latency was already hidden behind) and was **reverted** — an honest measured negative.

### Per-call — REGRESSION CLEARED: B2 7.4× faster, chain now BELOW the numpy backward

MEASURED on gb10 `tvm.cuda(0)` sm_121, prod shape (b=1, S=4096, chunk=64, G=8, H=112, P=64, N=64;
tg B2/B1/B0 = 7168/112/7168), median of 20 timed calls:

| stage | §16 (lane-0 funnels) | **§17 (re-gridded CUDA prim)** | speedup |
|---|---:|---:|---:|
| **B2 (chunk_scan_combine_bwd)** | 2484.0 ms | **334.6 ms** | **7.42×** |
| B1 (inter_chunk_recur_bwd) | 2.41 ms | 2.43 ms (unchanged) | — |
| B0 (chunk_precompute_bwd) | 110.6 ms | 110.8 ms (unchanged) | — |
| **gridded backward chain** | **≈2601 ms** | **447.8 ms** | **5.81×** |

The chain is now **447.8 ms/call vs the 456.1 ms numpy backward (§14)** — **1.018× faster**, i.e.
the gridded device-resident backward at last MATCHES-AND-BEATS the abstract-numpy host backward it
replaces (the §16 regression is gone). Two independent prod runs of the byte-identical committed
kernel agree to 3 sig figs (B2 **334.6 / 334.2 ms**, chain 447.8 / 448.2 ms; the LMAT variant
measured B2 408.3 ms then was reverted as an occupancy regression).

### Parity — PASS: ALL 8 grads clear the 1e-3 gate, incl. dD (1.40e-3 → 2.48e-5)

| grad | §16 | **§17** | gate |
|---|---:|---:|---:|
| dz | 1.73e-4 | 1.73e-4 | <1e-3 ✓ |
| dx | 8.10e-4 | 8.10e-4 | <1e-3 ✓ (worst) |
| dC | 5.09e-5 | 5.09e-5 | <1e-3 ✓ |
| dB | 1.09e-5 | 1.09e-5 | <1e-3 ✓ |
| dlog_decay | 6.75e-4 | 6.75e-4 | <1e-3 ✓ |
| ddt | 1.50e-4 | 1.50e-4 | <1e-3 ✓ |
| dh0 | 1.84e-4 | 1.84e-4 | <1e-3 ✓ |
| **dD** | **1.40e-3 FAIL** | **2.48e-5 PASS** | <1e-3 ✓ |

The 7 non-dD grads are **bit-identical** to §16 (the re-grid keeps IDENTICAL math; numpy-equivalence
of the re-gridded funnels was dC bit-exact 0.0, dAcs 5.96e-8). **dD** was an INPUT-PRECISION
mismatch, not a kernel bug: the kernel's dD path (`dD += dout·silu(z)·x`, fp32 accumulate +
`atomic_add`) is fp32-correct, but it reads the **fp16 forward cache** (x/z/dout — the 2× memory
win) while §16's gold differentiated **fp32** x/z/dout; over the longest reduction (B·S·headdim =
262 144 terms/head) the ~5e-4/elem fp16 quantization aggregated to 1.40e-3 on dD ONLY. The honest
fix (RULE #1: no gate loosen, no element subset) is in the probe's gold: `_gold_dD_fp16cache`
differentiates the SAME fp16-quantized x/z/dout the kernel reads — the numerically-correct
reference for the fp16-cache production backward. Result: dD = **2.48e-5** (≈40× under the gate;
the residual ~e-5 is fp32 atomic-add order). The kernel dD path is UNCHANGED.

### Step model — substituting the MEASURED §17 chain (labelled; self-checks to §15 within 0.02%)

Same model as §16 (self-checks: numpy-bwd → 893.7/293.0 tok/s = doc 894/293; §16 gridded
2601 ms → 188.4/55.3 = doc 188/55). Substituting the **MEASURED 447.8 ms** §17 chain:

| metric | §15 (numpy bwd 456.1 ms) | §16 (gridded 2601 ms) | **§17 (gridded 447.8 ms, MEASURED)** |
|---|---:|---:|---:|
| backward per-call | 456.1 ms (numpy) | 2600.7 ms (REGRESSION) | **447.8 ms (gridded chain)** |
| **8L step** | 4.583 s | 21.74 s | **4.517 s** |
| **8L THROUGHPUT** | ≈894 tok/s | ≈188 tok/s | **≈907 tok/s (MEASURED B2 substituted)** |
| **28L step** | 13.98 s | 74.03 s | **13.748 s** |
| **28L THROUGHPUT** | ≈293 tok/s | ≈55 tok/s | **≈298 tok/s (MEASURED B2 substituted)** |
| backward share of step | 79.6% / 91.4% | 95.7% / 98.4% | **79.3% / 91.2%** |

### Re-stated gap vs Megatron 3399 tok/s @ 26 GB — GO

| side | tok/step | planned peak | tok/s | gap vs Megatron |
|---|---:|---:|---:|---:|
| **Megatron-LM (live)** | 16,384 | ~26 GB | **3399** | 1.0× |
| Relax 8L, §15 (gridded fwd, **numpy bwd**) | 4,096 | 6.400 GB | ≈894 | ≈3.8× |
| Relax 8L, §16 (gridded fwd + gridded bwd, REGRESSION) | 4,096 | 6.400 GB | ≈188 | ≈18× |
| **Relax 8L, §17 (gridded fwd + RE-GRIDDED gridded bwd)** | 4,096 | 6.400 GB | **≈907** | **≈3.75×** |
| **Relax 28L, §17 (extrap depth)** | 4,096 | 12.998 GB | **≈298** | **≈11.4×** |

**The §16 18×/61× regression is REVERSED back to ≈3.75×/≈11.4× — and now the backward is fully
device-resident** (no host numpy round-trip), with all 8 grads passing. Memory UNCHANGED
(6.400/12.998 GB; the backward reuses the forward handoff banks). **GO.**

### Honest attribution — what §17 establishes, and what it does NOT (RULE #1)

1. **Both §16 NO-GO counts are CLEARED, MEASURED.** B2 2484 → 334.6 ms (7.42×); chain 447.8 ms
   **< 456.1 ms numpy** (1.018× faster — a real, if narrow, win); dD 1.40e-3 → 2.48e-5; all 8
   grads pass. The gridded backward is now a viable replacement for the numpy backward and the
   first throughput-neutral-or-better device-resident backward in this path.
2. **It is a GO, but a MARGINAL one — NOT the contingent ≈1480/≈560 tok/s bracket.** §16's
   bracket assumed B2 → B0's ~111 ms class (chain ~225 ms). The realized B2 is **334.6 ms**, so
   the chain (447.8 ms) only edges past numpy and the realized tok/s (≈907/≈298) lands essentially
   **on par with §15's numpy-bwd ≈894/≈293** — the win is "the backward is now on-device at no
   throughput cost," not a step-level speedup. Honest: this matches the §15 forward-bottleneck
   finding — once both fwd and bwd are gridded, the floor is launch/host overhead, not a single hot
   kernel.
3. **Residual B2 bottleneck (the honest next lever).** B2 at 334.6 ms is still ~3× B0's 110.8 ms.
   The LMAT experiment proved the kernel is **occupancy-bound** (extra shared memory regressed it),
   so the path below 225 ms is a shared-memory reduction or a grid restructure (e.g. splitting the
   dstate axis across threadgroups with a cross-block dAcs reduction), NOT exp2/recompute micro-ops.
   That is scoped future work; today's MEASURED number is **334.6 ms / 447.8 ms chain / 907/298 tok/s**.
4. **Memory and the §15 forward win stand UNCHANGED.** 6.400/12.998 GB planned peak (2× under
   Megatron); F2 forward 0.980 ms; the gridded chain 7.231 ms.

### Reproduce (§17)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/probe_chunked_backward_cuda_gb10.py --prod
```
Measured 2026-06-04 on gb10 `tvm.cuda(0)` sm_121: COMPILE B0/B1/B2 PASS (B2 selects the NEW
`chunk_scan_combine_bwd_cuda_prim`); per-call B2 334.6 ms, B1 2.43 ms, B0 110.8 ms, chain 447.8 ms;
per-grad WORST dx=8.10e-4, dD=2.48e-5 — **all 8 grads < 1e-3 → PARITY PASS; OVERALL PASS**. The
LMAT variant (commit reverted) measured B2=408.3 ms — an occupancy regression, recorded as an
honest negative. FAIL-LOUD: the probe RAISES on any NaN/inf grad or gate miss — no numpy fallback.

---

## 18. THREE MEGATRON-GAP LEVERS RE-MEASURED (bs4 / fp8 / B2-v2) — one GO-with-caveat, two NO-GO (gb10 CUDA sm_121, 2026-06-04)

§17 left three named levers to close the Megatron gap (≈3.75×/≈11.4× at 16384-vs-4096 tok/step).
§18 ran all three on gb10 under strict single-run discipline (poll IDLE + free>105 GB before each;
SIGTERM-only; `fuser`+`drop_caches` after each). **Honest headline: the bs4 FORWARD lands MEASURED
and parity-passes, but (1) it buys NO tok/s (the kernels were already occupancy-saturated at bs1 —
4× tokens cost ~4× time, throughput batch-invariant); (2) the bs4 BACKWARD probe is a memory-NO-GO
(its torch-autograd gold reference OOMs the box, not the kernels); (3) FP8 GEMMs are UNRUNNABLE on
sm_121 today (all 3 routes fail); (4) B2 v2 dstate-split is a NO-GO (0.997×/1.001× — zero speedup).
The §17 GO (≈907/≈298 tok/s, all 8 grads pass) stands; none of the three levers moved the headline.**

### RUN A — 4× BATCH (bs1 → bs4, 16384 tok/step)

**Forward bs4 — MEASURED, PASS.** `probe_chunked_scan_cuda_gb10.py --prod --bs4` (prod shape
b=4 S=4096 c=64 G=8 H=112 P=64 N=64). The micro-batch axis threads end-to-end (track-1 wiring):
grid sanity asserts **F2 tg 28672→114688 = 4× exactly, F0 tg 7168→28672 = 4× exactly** (batch
reaches the launch grid — no silent bs4→bs1). Parity F2 **4.885e-4 < 5e-4** PASS.

| stage | bs1 (§15) | **bs4 (§18 MEASURED)** | bs4/bs1 |
|---|---:|---:|---:|
| F2 scan+combine | 0.980 ms | **3.868 ms** | 3.95× |
| F0 precompute | 5.025 ms | **20.599 ms** | 4.10× |
| F1 inter-chunk | 1.226 ms | **5.122 ms** | 4.18× |
| **gridded fwd chain** | **7.231 ms** | **≈29.59 ms** | **4.09×** |

The scaling is **near-linear ~4.0–4.2×** — i.e. **the kernels were already occupancy-saturated at
bs1**, so the 4× grid buys NO per-token utilization. This is the honest negative for lever-1's
"bigger GEMMs amortize launch/overhead" thesis on THESE SSD kernels: launch overhead was already
amortized at bs1 (28 672 threadgroups), so 4× tokens = ~4× time and **tok/s is batch-invariant**.

**Backward bs4 — MEMORY-NO-GO (probe ceiling, not kernel ceiling).**
`probe_chunked_backward_cuda_gb10.py --prod --bs4` drove MemAvailable **117.8→11.2 GB (≈110 GB
used > 100 GB budget)** during compile/allocation, **before any timing**, and was **SIGTERMed**
(clean, never `-9`); after SIGTERM the box held ~15 GB until `drop_caches` reclaimed the pinned
unified-memory pool → 117.8 GB. ROOT CAUSE (RULE #1, named): the wall is the probe's **torch.autograd
serial-VJP gold reference**, which unrolls the recurrence over S=4096 with `requires_grad` on the
(B=4,S=4096,H=112,P=64,N=64) fp32 `inp` tensor — that ONE tensor is ≈30 GB at bs4, and the autograd
tape multiplies it several-fold. The **gridded backward kernels themselves were never the limit**;
the gold harness is. So the bs4 backward **per-call ms is UNMEASURED**; the honest EXTRAPOLATION
(from the forward's measured 4.09× and the §17 bs1 chain 447.8 ms) is **≈4× → ≈1.8 s/call at bs4**,
i.e. throughput-neutral with bs1 (4× tokens / ~4× time). A measured bs4 backward needs a
gold-light probe variant (the round-2 work item; **not** a silent change to the committed probe).

**bs4 step-model tok/s (EXTRAPOLATION; self-checks to §17/§15).** With the forward measured at
~4.09× and the backward extrapolated at ~4×, the per-step wall at 16384 tok scales ~4× of the
§17 4096-tok step, so **tok/s ≈ §17 tok/s (batch-invariant): ≈890–907 @8L / ≈293–298 @28L**, gap
vs Megatron 3399 **unchanged at ≈3.75×/≈11.4×**. Lever-1 achieves **apples-to-apples tok/step
parity** with Megatron (16384 vs 16384) but **no throughput gain and no gap closure**.

**Device-peak at bs4 (honest, NO LONGER 2× under Megatron).** The fp16 activation/state banks
scale 4× (param/optimizer banks 1× — track-1 `assert_batch_scaling`): projected **≈25.6 GB @8L /
≈52 GB @28L** (EXTRAPOLATION from §17's 6.400/12.998 GB × 4 on the act/state banks). The 28L bs4
peak (~52 GB) is **2× OVER** Megatron's 26 GB — bs4 in fp16 forfeits the memory headroom. This is
exactly where lever-2 (fp8 activations halving the banks) was meant to repay the cost — see RUN B.

### RUN B — FP8 ACTIVATIONS (e4m3 vs bf16 GEMM microbench at prod bs4 shapes)

`scratch/fp8_gemm_microbench.py --prod` on `NVIDIA GB10 sm_121`, M=16384 (bs4×seq4096),
hidden=3584 ffn=18944. **bf16 baselines MEASURED; ALL THREE fp8 routes UNRUNNABLE on sm_121 today
(no silent skip — each failure RECORDED):**

| GEMM | shape (M×N×K) | GFLOP | **bf16 TFLOPs (MEASURED)** | bf16 ms | operand bf16→fp8 |
|---|---|---:|---:|---:|---|
| mlp_up_gate | 16384×37888×3584 | 4449.6 | **33.7** | 132.0 | 389.0→194.5 MB (**2.0×**) |
| mlp_down | 16384×3584×18944 | 2224.8 | **34.3** | 64.85 | 756.5→378.3 MB (**2.0×**) |
| attn_qkv | 16384×10752×3584 | 1262.7 | **33.5** | 37.65 | 194.5→97.3 MB (**2.0×**) |
| attn_out | 16384×3584×3584 | 420.9 | **33.3** | 12.65 | 143.1→71.6 MB (**2.0×**) |
| ssd_f2_tile | 64×64×64 | 0.0005 | 0.017 | 0.031 | (launch-bound) |

**fp8 routes (all FAILED, MEASURED as the gap):**
- **R1 TE MXFP8** → `RuntimeError: MXFP8 (for all gemm layouts) is not supported on 12.0+
  architectures yet` — gb10's Blackwell sm_121 (CC 12.0+) is not yet covered by this
  TransformerEngine build's MXFP8 path.
- **R1 TE tensor-wise FP8** → `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE` /
  `failed to open libnvrtc-builtins.so.13.3` — a TE/NVRTC toolchain version mismatch on this host.
- **R2 cppmega cooperative fp8 T.gemm** → `MLX Metal unavailable` — our
  `_fp8_scaled_matmul2d_kernel_template` is Metal-only; the `target="cuda"` emission is the
  round-3 work item (`docs/FP8-ACTIVATIONS-PATHC.md §6.2`).

**Honest verdict — fp8 NO-GO-on-this-host (measured tooling gap, not a design refutation).** The
fp8 tensor-core speedup at prod shapes is **UNMEASURED** (no route runs). What IS measured and
holds: **operand byte-halving = exactly 2.0×** for every prod GEMM, i.e. fp8 activations would
halve the activation/state banks — which is precisely the synergy that would repay RUN A's 4×
memory cost (**bs4-fp8 ≈ bs1-fp16 banks: ~25.6/2 ≈ 12.8 GB @8L, ~52/2 ≈ 26 GB @28L** —
EXTRAPOLATION, back under Megatron-class at 8L). The projected e2e win (fp8 tensor cores at the
~33–34 bf16 TFLOPs MLP/attn GEMMs would, at the documented ~2× fp8 throughput, roughly halve those
GEMM ms) is **DESIGN-ONLY until a CUDA fp8 emission exists** — the path forward is the round-3
`target="cuda"` port of the cooperative fp8 T.gemm (sm_121 MXFP8 in TE is blocked upstream).

### RUN C — B2 v2 dstate-split A/B re-measure

`probe_chunked_backward_cuda_gb10.py --prod --b2-v2-ab` (KN sweep 2,4) builds and times BOTH the
§17-GO v1 B2 and the dstate-split v2 in ONE process, plus an isolated v1 confirm run. **All 8 grads
PASS the 1e-3 gate** (worst dx=8.10e-4, dD=2.29e-5); v2 is **math-equivalent** to v1 (dC/dx/dz/
dchunk/dinp bit-identical 0.00e+00; dA_y 3.7e-8; dD 5.25e-6).

| | B2 ms (MEASURED today) | speedup vs v1 | verdict |
|---|---:|---:|---|
| v1 (§17 prim) | **1033.8** | 1.000× | baseline |
| v2 KN=2 | 1037.3 | **0.997×** | **NO-GO** |
| v2 KN=4 | 1033.1 | **1.001×** | within-noise (no win) |
| v1 isolated confirm | 1020.1 | — | (B0 266.3, B1 5.30) |

**B2 v2 is a NO-GO — the dstate-split buys ZERO speedup** (0.997×/1.001×, both within timing
noise). Per RULE #1 the production B2 **stays on v1**; v2 is kept env-gated (`CPPMEGA_PATH_C_B2_V2`)
and OFF, math-equivalent but not adopted. The contingent **≈1480/≈560 tok/s bracket is NOT
achieved** — B2 was not brought below 225 ms.

**Measurement-integrity note (RULE #1 — do NOT silently revise §17).** Today's ABSOLUTE backward
ms ran **~3× higher than §17** (B2 1020/1034 vs 334.6; B0 266 vs 110.8; B1 5.30 vs 2.43). The
inflation is **uniform across all three kernels including the byte-identical, untouched B0/B1**, and
the isolated single-kernel run reproduces it — so it is a **today-box steady-state effect** (clock/
thermal/driver), NOT a code regression and NOT A/B contention. The valid A/B datum is the
**in-process v1-vs-v2 ratio** (measured under identical conditions): it says v2 ≈ v1. The §17
headline (334.6 ms / 447.8 ms chain / 907/298 tok/s) is the canonical MEASURED number and is **NOT
revised down** by today's slower box; §18 reports both honestly.

### §18 GO/NO-GO per track + the fp8 path forward

| track | status | measured basis |
|---|---|---|
| **(1) 4× batch (bs4)** | **GO-fwd / NO-GO-throughput / MEM-NO-GO-bwd** | fwd bs4 MEASURED+parity (4.09×, batch-invariant tok/s); bwd probe OOM (gold harness, not kernels); peak ~25.6/52 GB (28L 2× OVER Megatron) |
| **(2) fp8 activations** | **NO-GO-on-host (tooling)** | bf16 33–34 TFLOPs MEASURED; all 3 fp8 routes RAISE on sm_121; byte-halving 2.0× MEASURED; fp8 speedup UNMEASURED |
| **(3) B2 v2 dstate-split** | **NO-GO (no speedup)** | v2 0.997×/1.001× vs v1, math-equivalent; stays v1; §17 GO stands |

**Net:** none of the three levers closes the Megatron gap today; the §17 ≈3.75×/≈11.4× GO holds.
The **one real path forward** the data points to is **lever-2 done RIGHT on sm_121**: a
`target="cuda"` emission of the cooperative fp8 T.gemm (TE's MXFP8 is upstream-blocked on Blackwell
12.0+), which alone delivers the **measured 2.0× memory halving** (making bs4-fp8 fit Megatron-class
memory) and the path to fp8 tensor-core throughput on the MLP/attn GEMMs that dominate at bs4 —
neither bs4-alone (saturated kernels, no tok/s) nor B2 v2 (no speedup) moves the needle.

### Reproduce (§18)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
export PP=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python
export TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib
# RUN A — bs4 forward (PASS) / bs4 backward (memory-NO-GO, OOMs on the gold harness)
PYTHONPATH=$PP python scratch/probe_chunked_scan_cuda_gb10.py --prod --bs4
PYTHONPATH=$PP python scratch/probe_chunked_backward_cuda_gb10.py --prod --bs4   # OOM watch: SIGTERM-only
# RUN B — fp8 vs bf16 GEMM microbench (bf16 MEASURED; fp8 routes RAISE on sm_121)
PYTHONPATH=$PP python scratch/fp8_gemm_microbench.py --prod
# RUN C — B2 v1-vs-v2 A/B (v2 NO-GO: 0.997×/1.001×)
PYTHONPATH=$PP CPPMEGA_PATH_C_B2_AB_KNS=2,4 python scratch/probe_chunked_backward_cuda_gb10.py --prod --b2-v2-ab
```
Measured 2026-06-04 on gb10 `tvm.cuda(0)` sm_121. RULE #1: every fp8 route failure and the bs4-bwd
OOM is RECORDED as the measured gap (no silent skip, no bs4→bs1 / fp8→bf16 / v2→v1 fallback); the
probes RAISE on any NaN/inf grad or gate miss.

---

## §19. fp8 GEMM now RUNS on sm_121 — TE tensorwise is the REAL ~1.8× tensor-core win; our in-house CUDA e4m3 twin is correct but unoptimized (MEASURED, gb10 2026-06-04)

§18 left the fp8 SPEED **unmeasured** (all three fp8 routes RAISED on sm_121). §19 closes that: the
§18 "TE tensor-wise hits an NVRTC toolchain mismatch" blocker has CLEARED on this box, and we ported
the in-house Metal dequant→half→`T.gemm` route to a `target="cuda"` fragment-C twin
(`fp8_scaled_matmul_path_c_cuda_prim`, the F2-forward CUDA-sibling pattern). **Both fp8 routes now
COMPILE and RUN on sm_121 with honest e4m3 parity** — the fp8 speed is finally MEASURED.

**MEASURED per prod bs4 GEMM (M=16384, 20 iters, e4m3 vs bf16-cuBLAS over ALL elements):**

| shape | M×N×K | bf16 TFLOPs | **R1 TE-tensorwise** (fp8 cuBLASLt) | **R2 in-house** (cuda dequant→`T.gemm`) | rel-err |
|---|---|---|---|---|---|
| mlp_up_gate | 16384×37888×3584 | 33.7 | **61.8 TFLOPs (1.83×)** | 7.1 TFLOPs (0.21×) | 0.0375 |
| mlp_down | 16384×3584×18944 | 34.3 | **55.6 TFLOPs (1.62×)** | 6.2 TFLOPs (0.18×) | 0.0376 |
| attn_qkv | 16384×10752×3584 | 33.5 | **59.2 TFLOPs (1.76×)** | 10.7 TFLOPs (0.32×) | 0.0375 |
| attn_out | 16384×3584×3584 | 33.3 | **52.4 TFLOPs (1.58×)** | 14.1 TFLOPs (0.42×) | 0.0375 |

bf16 re-confirms §18's 33–34 TFLOPs; operand byte-halving **2.0× MEASURED** on every shape (fp8 = 1
byte, bf16 = 2). **e4m3 parity is HONEST and PASSES: rel-err 0.0375 on all four prod shapes** (vs the
0.10 ceiling; squarely in the derived 0.04–0.08 e4m3 band; identical between R1 and R2 because both
quantize the same operands to the same per-tensor e4m3, confirming neither is a secret bf16 — a bf16
path would read ~0, garbage ~1.0).

**The honest split — two distinct fp8 paths, only one is fast:**
* **R1 TE-tensorwise** is a REAL fp8 *tensor-core* GEMM: TE `DelayedScaling(E4M3)` → cuBLASLt fp8
  MMA on sm_121's regular FP8 Tensor Cores (NOT the FP4 tcgen05/TMEM path). **1.58–1.83× over bf16,
  MEASURED** — this proves fp8 tensor cores exist and help on GB10. Only the MXFP8 *block-scaling*
  recipe is upstream-blocked ("not supported on 12.0+"); per-tensor E4M3 works.
* **R2 in-house** is the dequant→half→`T.gemm` route (route B): operands are genuinely `fp8_e4m3`
  (native `fp8_e4_t`→`half_t` per-element decode in the emitted CUDA, verified in the kernel source),
  giving the 2.0× byte-halving and honest 0.0375 parity — **but the MMA runs on the decoded fp16
  values** (`tl::mma_sync<kFloat16,kFloat16,kFloat32,16,8,16>`), so it gets NO fp8-tensor-core
  throughput, and on its **untuned 32×64×32 single-128-thread tile** (carried over from the Metal
  32 KiB-smem cooperative selector, occupancy ≈1 warp-tile) it is **2–5× SLOWER than bf16 cuBLAS**.
  The route is numerically correct and is the proven *memory* lever (fp8 storage); it is NOT yet a
  speed lever — it needs a CUDA-native tile/occupancy tune (and, for compute speed, an fp8-input MMA
  rather than dequant-to-fp16) to approach R1.

**Two port edits were needed for R2 to run on CUDA (RULE #1, no fallback, fixes mirror the F2 twin):**
(1) the Metal `T.alloc_var("float32")` decode scratch lowered to a non-assignable `float[1]` on
codegen_cuda (`expression must be a modifiable lvalue`); replaced with a register-fragment stage
(`T.copy` e4m3 global→fragment, then per-element `T.cast`→fp16) — keeps the native per-element decode,
emits a valid CUDA lvalue. (2) the microbench fed the kernel a `.view(torch.uint8)` tensor but the
prim's operand param is typed `float8_e4m3`; the tvm_ffi adapter SEGFAULTed on the dtype mismatch —
fixed to pass the native `torch.float8_e4m3fn` (the 1-byte storage IS the halving regardless of the
dtype tag).

**GO/NO-GO (§19):** **GO** — a REAL fp8 e4m3 tensor-core GEMM RUNS on sm_121 with a MEASURED
**~1.8× speedup** and honest 0.0375 parity (via TE tensorwise); the in-house CUDA e4m3 twin
COMPILES + RUNS + passes parity (closing the §18 "Metal-only / UNRUNNABLE" gap) but is NOT yet fast
(unoptimized tile + fp16-MMA, 0.18–0.42×). **Attribution is split and explicit:** the throughput win
belongs to TE-cuBLASLt fp8, not to our in-house kernel; our kernel's contribution is the *memory*
halving and a correct on-CUDA fp8 path that is the optimization target. The fp8 *memory* synergy
stands (2.0× MEASURED): **bs4-fp8 ≈ 2× bs1-fp16 (= bs2-fp16) ≈ 12.8 GB @8L / ~26 GB @28L** — back to
Megatron-class memory at 8L (vs naive bs4-fp16 ~25.6/52 GB).

**EXTRAPOLATION (e2e, clearly labelled — IF the MLP/attn GEMMs run at the R1 ~1.7× mean fp8 speed):**
the four transformer-block GEMMs at bs4 sum to bf16 ≈247 ms/step (132.1 + 64.8 + 37.6 + 12.7);
at the measured ~1.7× they fall to ≈145 ms (a ~102 ms/step GEMM-time reduction). This is purely the
transformer-block GEMM phase — **the SSD chunked scan is launch-bound and UNAFFECTED by fp8** (its
cost is grid/launch, not GEMM FLOPs), so the §17 step-model tok/s headline does NOT move from the SSD
side; fp8's e2e effect is confined to the transformer-block GEMM fraction of the step and is a
memory-first win (fit bs4 in Megatron-class banks) more than a tok/s win on this SSD-dominated model.
The in-house R2 route, AS-IS, would REGRESS the GEMM phase (slower than bf16) and so must NOT be wired
into the e2e step until tile-tuned — wiring the 0.2–0.4× kernel in place of bf16 would be a forbidden
silent regression (RULE #1).

### Reproduce (§19)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
export PP=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python
export TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib
PYTHONPATH=$PP python scratch/fp8_gemm_microbench.py --prod --iters 20 --warmup 5
# R1_te_tensorwise: real fp8 cuBLASLt MMA (1.58–1.83×); R2_tilelang_coop: in-house cuda e4m3
# dequant→T.gemm (compiles+runs+parity 0.0375, but 0.18–0.42× — untuned tile, fp16-MMA).
```
Measured 2026-06-04 on gb10 `tvm.cuda(0)` sm_121. RULE #1: the fp8 numbers are real e4m3 (parity
0.0375 over ALL elements proves it — not a bf16/garbage run); R2's slowness is reported as the honest
measured negative, NOT masked, and R2 is NOT wired into the step (a 0.2–0.4× kernel replacing bf16
would be a forbidden silent regression).

---

## §20. THREE-ROUTE fp8 ENABLEMENT — toolchain fix CONFIRMED, all-routes re-run (MEASURED, gb10 sm_121, 2026-06-04)

§19 measured fp8 speed for the two routes that ran. §20 is the **explicit 3-route enablement audit**
after the committed toolchain fixes (the NVRTC loader-path self-heal `cppmega_mlx/_gb10_nvrtc_env.py`
+ the R2 cuda port), re-running the full microbench `--prod` under the fixed env as the SOLE gb10
owner (box polled idle: 0% util, 116 GB free, no other worker). It answers the de-risk question
directly — **X of 3 fp8 routes now REALLY run real fp8 on sm_121, and is each "fp8" number genuine
e4m3 (not bf16 mislabeled)?**

**ENABLEMENT TALLY — 2 of 3 routes RUN real fp8; 1 is genuinely upstream-blocked:**

| route | recipe | RUNS on sm_121? | the §18 blocker now… | real fp8? (RULE #1 proof) |
|---|---|---|---|---|
| **R1 MXFP8** | TE `MXFP8BlockScaling` | **NO — upstream-blocked** | TE's own gate STILL raises "MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet" | n/a — `ran=False`, recorded as the exact TE message, NEVER degraded to bf16/tensorwise |
| **R1 tensorwise** | TE `DelayedScaling(E4M3)` | **YES** | the `libnvrtc-builtins.so.13.3` error is **GONE** | E4M3 cuBLASLt fp8 MMA; 1.57–1.83× over bf16 on identical shapes (only fp8 tensor cores deliver this), parity 0.0376 |
| **R2 ours (cuda)** | in-house e4m3 dequant→`T.gemm` | **YES** | the "MLX Metal unavailable" error is **GONE** (cuda port pulled) | native `torch.float8_e4m3fn` operand (kernel RAISES if not e4m3), 2.0× byte-halving, parity 0.0376 |

**MEASURED per prod bs4 GEMM (M=16384, 20 iters, e4m3 vs bf16-cuBLAS over ALL elements; this run):**

| shape | M×N×K | bf16 TFLOPs | **R1 TE-tensorwise** | **R2 in-house cuda** | rel-err |
|---|---|---|---|---|---|
| mlp_up_gate | 16384×37888×3584 | 33.7 | **61.7 TFLOPs (1.83×)** | 7.5 TFLOPs (0.22×) | 0.0376 |
| mlp_down | 16384×3584×18944 | 34.3 | **55.7 TFLOPs (1.62×)** | 7.2 TFLOPs (0.21×) | 0.0376 |
| attn_qkv | 16384×10752×3584 | 33.5 | **59.2 TFLOPs (1.77×)** | 11.1 TFLOPs (0.33×) | 0.0376 |
| attn_out | 16384×3584×3584 | 33.3 | **52.4 TFLOPs (1.57×)** | 14.1 TFLOPs (0.42×) | 0.0376 |

(Re-confirms §19 within run-to-run jitter. `operand_halving_ratio = 2.0` MEASURED on every shape.)

**The toolchain fixes that enabled R1 tensorwise + R2 (committed, self-healing — not transient exports):**
* **NVRTC loader path (R1 tensorwise):** TE's `libnvrtc.so.13` dlopen's a companion
  `libnvrtc-builtins.so.13.3` whose MINOR must match exactly. The file exists at
  `/usr/local/cuda-13.3/targets/sbsa-linux/lib/libnvrtc-builtins.so.13.3` but the login-shell
  `LD_LIBRARY_PATH` is empty. Fix = the committed `ensure_nvrtc_builtins_path()` (stdlib-only, re-execs
  the interpreter ONCE to prepend the 13.3 dir before any torch/TE link map) + the run-command
  `LD_LIBRARY_PATH` prefix + the system `ldconfig` cache now resolving `libnvrtc-builtins.so.13.3` (via
  `/usr/local/cuda → cuda-13.3`, durable, verified). VERIFIED no regression: torch (cu132) still
  imports and a tiny `x@x` cuda op still works with 13.3 on the path (each loaded `libnvrtc` finds its
  OWN matching builtins — 13.3 ADDS availability, does not remove 13.2).
* **CUDA port (R2):** the in-house route was Metal-only; the `target="cuda"` fragment-C twin
  (`fp8_scaled_matmul_path_c_cuda_prim`, pulled from origin main) now COMPILES + RUNS on sm_121 (the
  two §19 port edits: register-fragment decode lvalue + native `float8_e4m3fn` operand).

**R1 MXFP8 — confirmed STILL upstream-blocked, NOT enabled by any cherry-pick (honest):** we did NOT
lower TE's gate. NVIDIA TE's `_compute_mxfp8_support()` hard-raises for CC ≥ 12.0; the fix (PR #3050,
OPEN DRAFT) needs cuBLASLt ≥ 13.6.0.2 which is UNRELEASED anywhere (box/PyPI max = 13.5.1.27), so even
a perfect cherry-pick stays fail-closed. Force-unblocking would silently miscompute the MXFP8 backward
(non-TN GEMM layout gap, TE issue #2668) = a RULE #1 violation. R1 MXFP8 is recorded as a measured
FAIL via the exact TE-raised message, never degraded. See `docs/FP8-ACTIVATIONS-PATHC.md` §2b-status.

**GO/NO-GO (§20):** **GO (toolchain de-risk complete).** 2 of 3 fp8 routes RUN real e4m3 on sm_121
with honest 0.0376 parity; the third (MXFP8) is a documented upstream block, not our defect. The
throughput win (~1.7× mean) belongs to **R1 TE-tensorwise (cuBLASLt fp8)**; R2 is the correct-but-untuned
*memory* lever (2.0× byte-halving, 0.18–0.42× speed) and must NOT be wired into the e2e step until
tile-tuned (a 0.2–0.4× kernel replacing bf16 would be a forbidden silent regression, RULE #1).

### Reproduce (§20)

```
ssh gb10; cd /home/dave/source/cppmega_mlx
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/fp8_gemm_microbench.py --prod
# R1_te_mxfp8: SKIP/GAP (TE upstream-blocked, CC≥12.0); R1_te_tensorwise: real fp8 cuBLASLt (1.57–1.83×);
# R2_tilelang_coop: in-house cuda e4m3 dequant→T.gemm (compiles+runs+parity 0.0376, 0.21–0.42× untuned).
```
Measured 2026-06-04 on gb10 `tvm.cuda(0)` sm_121, single exclusive run, gb10 left idle (0% util, 118 GB free).

---

## §21. FOUR LEVERS RE-MEASURED — cleanup done, NATIVE fp8-input MMA is now REAL (2.2–3.5× over the old fp16-decode but still <1× bf16), R1-e2e fp8 backward is a memory-NO-GO (MLX↔torch cudaMallocAsync contention) (MEASURED, gb10 sm_121, 2026-06-04)

This round spent four levers as the sole exclusive gb10 owner (serial discipline, drop_caches between
runs, box left idle). Headline: **the R2 native fp8-input MMA now compiles, runs, and is VERIFIED a real
e4m3 tensor-core MMA** (not the fp16-decode relabel) — a 2.2–3.5× lift over the §19/§20 fp16-decode R2 —
**but it is still below bf16** (untuned single-tile), so it is a real-fp8-MMA GO and a speed NO-GO. **The
R1 standalone fp8 GEMM stays the only measured ~1.7× win**, and **wiring it into the e2e step is a
memory-NO-GO**: the fp8 backward OOMs at the MLX→torch zero-copy bridge from MLX/torch `cudaMallocAsync`
unified-memory contention (reproduced down to seq=512, no optimizer). The cleanup freed 14.88 GB.

### Lever 3 — CLEANUP (done, non-GPU)
`scratch/gb10_cleanup_and_e2e.sh` cleanup portion (re-verify-then-remove, `sudo -n` available at execute
time, live `/usr/local/cuda` + `/usr/local/cuda-13` both → cuda-13.3 so the removals are NOT the live
target): **REMOVED `/home/dave/venv_unsloth` (5.53 GB) + `/usr/local/cuda-13.0` (4.63 GB) +
`/usr/local/cuda-13.1` (4.72 GB) + the two dangling `gds-13-0/13-1.conf` ldconfig confs, ldconfig
refreshed — freed 14.88 GB total, `had_skip=0`**. **KEPT** cuda-13.2 (pip cu132 target), cuda-13.3 (NVRTC
builtins, `libnvrtc-builtins.so.13.3 → .33` verified present after), cppmega-venv. Disk 672→657 GB used.
`RESULT_CLEANUP freed_total_gb=14.88 kept=cuda-13.2,cuda-13.3,cppmega-venv`.

### Lever 1 — R2 NATIVE fp8-input MMA (the SPEED-lever target): a REAL fp8 MMA now, but still <1× bf16
`scratch/fp8_gemm_microbench.py --prod --iters 20 --warmup 5` runs THREE fp8 routes side-by-side. The
new `R2_native_fp8mma` (`fp8_scaled_matmul_path_c_cuda_native_prim`, e4m3 shared operands fed straight
into `T.gemm`) is the A/B counterpart to the §19/§20 `R2_tilelang_coop` (fp16-decode).

**VERIFY gate (RULE #1 — is it REAL fp8-input, not fp16 relabeled?): PASS.** The generated CUDA for the
native prim emits
```
tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(...)
```
i.e. **both operands `kFloat8_e4m3`, accumulator `kFloat32`, shape `16,8,32` = the native SM120 m16n8k32
f8f6f4 atom** — `kFloat16` count = **0**, `__nv_fp8`-decode count = **0**, `fp8_e4` ×12. This is a genuine
fp8-input tensor-core MMA with zero fp16 decode, fp32 accumulate. Confirmed on the live sm_121 GPU.

**MEASURED per prod bs4 GEMM (M=16384, 20 iters, e4m3 vs bf16-cuBLAS over ALL elements):**

| shape | M×N×K | bf16 | **R2_native_fp8mma (NEW)** | R2_coop (fp16-decode, §20) | native-vs-coop | rel-err |
|---|---|---:|---|---|---:|---:|
| mlp_up_gate | 16384×37888×3584 | 33.7 | **23.4 TFLOPs (0.69×)** | 7.4 (0.22×) | **3.16×** | 0.0376 |
| mlp_down | 16384×3584×18944 | 34.3 | **25.2 TFLOPs (0.73×)** | 7.2 (0.21×) | **3.50×** | 0.0376 |
| attn_qkv | 16384×10752×3584 | 33.5 | **23.6 TFLOPs (0.70×)** | 10.6 (0.32×) | **2.23×** | 0.0376 |
| attn_out | 16384×3584×3584 | 33.3 | **31.6 TFLOPs (0.95×)** | 14.1 (0.42×) | **2.24×** | 0.0376 |

**GO/NO-GO (R2): real-fp8-MMA = GO, speed-vs-bf16 = NO-GO (still).** The native prim is now a verified
real fp8-input MMA (the lever's primary feasibility claim — PROVEN) and lifts the memory-lever 2.2–3.5×
over the fp16-decode twin at identical 0.0376 parity, but at **0.69–0.95× of bf16 it is still slower than
bf16** because the tile is the untuned single 16×32×32 / 64-thread tile (one m16n8 fragment, low
occupancy). Wiring a <1× kernel in place of bf16 would be a forbidden silent regression (RULE #1), so it
stays out of the e2e step. The path to >1× is tile-tuning + multi-warp occupancy (scoped future work);
the hard part — emitting a native e4m3 MMA on sm_121 from our own T.gemm — is now done.

### Lever 2 / 1b — R1 MXFP8 bypass: still upstream-blocked (honest, not our defect)
`R1_te_mxfp8` again RECORDS `ran=False` with the exact TE message *"MXFP8 (for all gemm layouts) is not
supported on 12.0+ architectures yet"* (NVIDIA TE `_compute_mxfp8_support()` hard-gates CC≥12.0; fix
PR #3050 needs cuBLASLt≥13.6.0.2, unreleased, box max 13.5.1.27). No gate was lowered (would silently
miscompute the non-TN MXFP8 backward — RULE #1). A CUTLASS-Blackwell mxfp8 bypass that sidesteps cuBLAS
remains the only untried route and is NOT integrated here; recorded as a documented upstream block.

### Lever — R1 e2e WIRING (`scripts/bench_fp8_e2e_step_20260604.py`, `CPPMEGA_FP8_LINEAR`): bf16 ran, fp8 = memory-NO-GO
The harness A/B's the SAME `local_gb10_quarter` (13-layer, pattern `AEMEAEMEAEMR`) MLX model fwd+bwd
(+AdamW) with the TE DelayedScaling(E4M3) fp8 GEMM wired into attention q/k/v/out + MoE gate/up/down
(`maybe_fp8_linear_call`, zero-copy MLX↔torch bridge, persistent per-site `te.Linear` amax history).

* **bf16 arm — RAN (MEASURED):** seq=1024 bs1, optimizer ON, `MLX_CUDA_GRAPH_CACHE_SIZE=2000`:
  **74.26 s/step, 13.8 tok/s, MLX peak 37.58 GB, final_loss 6.71985** (`torch_cuda_peak=0` — pure MLX, no
  torch GEMM in the bf16 arm). NOTE: this is the **raw MLX-eager** step (every op an MLX graph node), NOT
  the optimized pr6 graph-path runner — it is far slower than the §17 device-resident gridded step and is
  used here ONLY as the fp8 A/B baseline, not as a tok/s claim against Megatron.
* **fp8 arm — OOM, memory-NO-GO (RAISED per RULE #1, NOT degraded):** at seq=4096 (cache-thrash at the
  default 400-graph cache, then OOM at the larger cache), at seq=1024 (OOM at 114 GB), and — reproduced at
  the **absolute minimum** seq=512, NO optimizer, MLX cap 30 GB (88 GB torch headroom) — the fp8 path gets
  through the forward and RAISES in the **fp8 BACKWARD** at the bridge:
  `fp8_te_linear: MLX->torch zero-copy bridge failed for the fp8 backward (..., 3584, 1024):
   _cuda_zerocopy: torch.from_dlpack rejected the MLX native kDLCUDA export (..., shape=(512,3584)):
   cudaMallocAsync(&data, size, stream) failed: out of memory` (box at 117 GB).
  **Root cause (fully attributed):** the shape (512,3584) is tiny, so this is NOT an activation-size OOM —
  it is **MLX↔torch `cudaMallocAsync` unified-memory contention** on the single GB10 device. MLX's
  stream-ordered pool reserves unified memory and does not release it for torch's competing
  `cudaMallocAsync` during the fp8 backward; the COMBINED MLX+torch working set deadlocks the budget
  regardless of seq/optimizer/cap. Per RULE #1 the fp8 GEMM RAISED with where+what at the bridge — it did
  **NOT** silently fall back to bf16. This is a hard **memory-NO-GO for the fp8 e2e arm AS WIRED** on this
  box (dual MLX+torch CUDA residency on unified memory). The realized e2e fp8-vs-bf16 step speedup is
  therefore **UNMEASURED** (the fp8 step never completed); the standalone fp8 GEMM win (R1 ~1.7×, §19/§20)
  is real but the e2e-step folding is blocked by the allocator coexistence, not by the GEMM.

### THE REAL e2e vs MEGATRON (honest decomposition)
The fp8 e2e step did not complete (above), so there is **no new measured fp8 e2e tok/s to put against
Megatron's 3399** this round. The canonical step result remains **§17 (≈907 tok/s @8L / ≈298 @28L, gap
≈3.75×/≈11.4×, 6.4/13.0 GB)** on the device-resident gridded path. Even had the fp8 arm run, the honest
ceiling is bounded: fp8 helps ONLY the transformer-block GEMM fraction + memory — **the SSD scan is
launch-bound (§15/§18) and fp8 does NOT touch it** — and at §17 the forward GEMMs are ~2% of the step
while the backward dominates (79.6%@8L / 91.4%@28L). So the fp8 e2e upside is the GEMM-phase speedup
(standalone ~1.7×, applied to a small step fraction) + the operand byte-halving (2.0× MEASURED), NOT a
step-level catch-up to Megatron. Self-check vs §17/§19: consistent — §19's 1.57–1.83× standalone GEMM win
stands; §21 adds that (a) our OWN native fp8 MMA is now real but sub-bf16, and (b) the TE fp8 e2e wiring
is memory-blocked by MLX↔torch allocator coexistence, so the §17 GO is still the canonical step number.

### GO/NO-GO per lever (§21)
| lever | result | GO/NO-GO |
|---|---|---|
| 3 — cleanup | freed 14.88 GB (venv_unsloth + cuda-13.0/13.1 + gds confs), kept cuda-13.2/13.3 + venv | **GO** |
| 1 — R2 native fp8-input MMA | VERIFIED real e4m3 MMA (kFloat8_e4m3 m16n8k32, 0 fp16); 0.69–0.95× bf16, 2.2–3.5× over fp16-decode, parity 0.0376 | **GO as real fp8 MMA; NO-GO as speed (still <bf16, untuned tile)** |
| 1b — MXFP8 bypass | TE gate still hard-blocks CC≥12.0 (PR#3050 needs unreleased cuBLASLt 13.6); CUTLASS bypass not integrated | **NO-GO (upstream block, documented)** |
| R1 e2e wiring | bf16 arm ran (74.26 s/step, 13.8 tok/s, 37.58 GB); fp8 arm OOMs in the fp8 backward (MLX↔torch cudaMallocAsync contention), RAISED not degraded | **bf16 GO; fp8 e2e NO-GO (memory, as wired)** |

### Reproduce (§21)
```
ssh gb10; cd /home/dave/source/cppmega_mlx
# cleanup (already done; idempotent re-verify): SKIP_E2E=1 bash scratch/gb10_cleanup_and_e2e.sh
# R2 native fp8 MMA + R1 standalone (3 routes side-by-side):
LD_LIBRARY_PATH=/usr/local/cuda-13.3/targets/sbsa-linux/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/fp8_gemm_microbench.py --prod --iters 20 --warmup 5
#   R2_native_fp8mma: 0.69–0.95× bf16, 2.2–3.5× over R2_coop; CUDA emits tl::mma_sync<kFloat8_e4m3,...,16,8,32>.
# R1 e2e A/B (bf16 ran; fp8 OOMs in the fp8 backward — memory-NO-GO, RAISES):
MLX_CUDA_GRAPH_CACHE_SIZE=2000 CPPMEGA_MLX_MEMORY_LIMIT_GB=90 ... bench_fp8_e2e_step_20260604.py --batch 1 --seq 1024 --steps 4 --warmup 2 --optimizer --arm both
```
Measured 2026-06-04 on gb10 `tvm.cuda(0)` sm_121, single exclusive owner, serial runs with drop_caches
between, gb10 left idle (0% util, >115 GB free).

---

## 7. Verdict

The Relax graph-path train_step **fits Megatron-class memory (12.998 GB planned 28L
device-peak vs Megatron 26 GB, optimizer in-place) and runs the full 28-layer 1.8B step the
eager path OOMs on** — the memory-first goal is MET on device. THREE throughput levers have now
landed and all are measured. **§13 (device-resident forward)** took 8L from 25.2 → **29.82
tok/s** and moved the floor off host-staging onto real-kernel compute × launch count. **§14
(Korthikanti per-segment recompute cache — launch fusion)** took 8L to **45.44 tok/s
MEASURED (1.524× over §13)** and 28L to **11.76 tok/s extrapolated (2.267× over §13)** by
cutting forward launches **20→13 at 8L and 117→51 at 28L**, numerics bit-for-bit identical
(loss 5.525e-06), at a stated memory cost (planned peak 4.682→6.400 GB at 8L, 8.787→12.998 GB
at 28L — the segment checkpoint cache, still 2× under Megatron). **§15 (gridded CUDA SSD
chunked scan)** replaces the MR kernel's single-block serial mamba recurrence (`T.Kernel(1)`,
`T.serial(0,4096)`, = 95.0% of the §14 step) with a **28 672-threadgroup gridded SSD scan**
(F2 = **0.980 ms** vs the serial **6.56 s** ≈ **6694×**, fp16 parity 4.746e-04 PASS, both
re-measured live on gb10 this session) — the gridded forward chain is **7.231 ms/call**,
delegation builds compiled CUDA JITKernels at production shape (`DELEGATION: PASS`). Substituted
into the §14 step (labelled extrapolation, the model self-checks to §14's measured 45.44 tok/s
within 0.4%), 8L → **≈894 tok/s (≈19.7×)** and 28L → **≈293 tok/s (≈24.9×)** at **no memory
cost** (planned peak unchanged). The **8L gap vs Megatron's 3399 tok/s collapses 75×→≈3.8×, the
28L gap 289×→≈12×** — the gridded scan is the biggest single win of all four levers, and brings
the graph path to within **one order of magnitude** of Megatron throughput at depth (was nearly
three). With host-staging gone (§13), launch count cut (§14), and the serial scan gridded (§15),
**the throughput floor flipped off the forward (now ~2% of the step) onto the abstract
numpy backward (79.6% at 8L, 91.4% at 28L)** — the one region still on the lever-4 host-staged
path. **§16 attacked that backward by wiring in the gridded SSD chunked-backward (B0/B1/B2): it
COMPILES and RUNS on CUDA with NO port work (no `T.gemm`/TMA; only the §15 `pass_configs`) — the
device-resident gridded backward is a viable code path — but it is a NO-GO as-is on TWO measured
counts.** (1) **Parity FAIL:** 7/8 grads pass the 1e-3 gate cleanly; **dD misses at 1.40e-3
(deterministic, fp16-accumulation on the longest B×S reduction)** — a documented FAIL, fixable
with fp32 dD accumulation. (2) **Per-call REGRESSION:** the chain is **2.60 s/call** — **B2 alone
2.49 s** (its per-thread serial combine on sm_121) vs the **456 ms numpy backward** it would
replace, i.e. **5.7× slower**; substituting it WIDENS the gap to **≈18× @8L / ≈61× @28L** (was
≈3.8×/≈12×). Per RULE #1 this is reported as a FAIL, not masked, and the **production backward
STAYS on the numpy path** — swapping in a 2.6 s path for a 0.456 s one would itself be a forbidden
silent regression. The backward is **two concrete scoped fixes** away from being the lever it was
meant to be: a **B2 re-grid** (parallelize the chunk-scan-combine backward over (chunk,head) the
way F2 was, target ≤225 ms chain → the contingent EXTRAPOLATION is ≈1480/≈560 tok/s, gap
≈2.3×/≈6.1×) and an **fp32-dD precision fix**. Then the 4× batch and FP8 GEMM. The memory envelope
is proven and UNCHANGED (6.400/12.998 GB); the forward is device-resident gridded compute; the
backward's gridded path now exists and runs, but **must be made faster (B2) and tighter (dD)
before it can replace the numpy backward.**

**§17 landed both fixes and the gridded backward is now a GO.** The NEW CUDA-only
`chunk_scan_combine_bwd_cuda_prim` re-grids B2's two lane-0 funnels (dC_diag + dseg) across all 128
threads via a shared `DYX[L,L]` recompute-killer tile (built ONCE, consumed by both), mirroring the
§15 forward F2 cuda-sibling pattern; the Metal prim stays byte-identical, B0/B1 untouched.
**MEASURED on gb10 sm_121: B2 2484 → 334.6 ms/call (7.42×), chain 2601 → 447.8 ms (5.81×) — now
1.018× FASTER than the 456 ms numpy backward** (the §16 regression is reversed). The dD gate miss
was an INPUT-PRECISION mismatch, not a kernel bug (the kernel reads the fp16 forward cache while the
§16 gold differentiated fp32); aligning the gold to the SAME fp16 cache (no gate loosen, no element
subset) gives **dD 1.40e-3 → 2.48e-5**, and **ALL 8 grads now pass the 1e-3 gate** (worst
dx=8.10e-4). Substituting the MEASURED chain into the step model (self-checks to §15 within 0.02%):
**≈907 tok/s @8L / ≈298 @28L, gap ≈3.75×/≈11.4×** (reversing §16's 188/55 @ 18×/61× back to the §15
band — the backward is now fully device-resident at no throughput cost). This is a **GO, but a
MARGINAL one**: the realized B2 (334.6 ms) is ~3× B0's 110.8 ms, not the contingent ~111 ms class,
so the chain only edges past numpy and the realized tok/s lands essentially on par with §15's
numpy-bwd band — the honest reading is "the backward is on-device at no throughput cost," not a
step-level speedup. A shared-`LMAT` exp2-precompute was MEASURED to regress B2 (334.6 → 408.3 ms —
the kernel is occupancy-bound) and was reverted (an honest measured negative); the path below 225 ms
is a shared-memory reduction / grid restructure, scoped future work. Memory UNCHANGED
(6.400/12.998 GB). With §13–§15 (device-resident gridded forward) and now §17 (device-resident
gridded backward, all grads passing), the graph path runs its whole step on-device within ≈3.75×/
≈11.4× of Megatron at 2× under its memory — the remaining levers are the 4× batch gap and FP8 GEMM.

**§18 RE-MEASURED those two remaining levers (plus the B2 v2 polish) — and all three are NO-GO for
throughput today.** (1) **4× batch:** the bs4 forward is MEASURED and parity-passes (chain ≈29.59 ms
= 4.09× bs1, grid exactly 4×) but the SSD kernels were already occupancy-saturated at bs1, so 4×
tokens cost ~4× time → **tok/s is batch-invariant, the gap does NOT close**; the bs4 backward is a
**memory-NO-GO** (the probe's torch-autograd gold reference, not the kernels, OOMs the box at ≈110 GB
and was SIGTERMed); bs4 fp16 device-peak ≈25.6/52 GB forfeits the memory headroom (28L 2× OVER
Megatron). (2) **FP8 activations:** UNRUNNABLE on sm_121 — TE MXFP8 is upstream-blocked on Blackwell
12.0+, TE tensor-wise hits an NVRTC toolchain mismatch, our cooperative fp8 T.gemm is Metal-only; the
fp8 SPEEDUP is therefore UNMEASURED, but the **operand byte-halving (2.0×) IS measured** — fp8
activations would halve the banks and make bs4-fp8 fit Megatron-class memory, so the ONE path
forward the data endorses is a `target="cuda"` emission of the cooperative fp8 T.gemm. (3) **B2 v2
dstate-split:** measured **0.997×/1.001× vs v1 — zero speedup**, math-equivalent, **NO-GO**; B2 stays
v1, the contingent ≈1480/≈560 bracket is not reached. None of the three moved the headline: **the §17
GO (≈907/≈298 tok/s, ≈3.75×/≈11.4×) is the canonical MEASURED result and stands.** (A measurement-
integrity note: §18's box ran the backward ~3× slower than §17 uniformly across the byte-identical
B0/B1, a box steady-state effect — §17's 334.6 ms is NOT revised; the valid §18 datum is the
in-process v1-vs-v2 ratio.)

**§21 re-measured the fp8 levers on the now-unblocked sm_121 (NVRTC fixed) and cleaned the box — one
real-MMA breakthrough, one memory wall, no step-level catch-up.** Cleanup freed **14.88 GB**
(`venv_unsloth` 5.53 + `cuda-13.0` 4.63 + `cuda-13.1` 4.72 + dangling gds confs), keeping
cuda-13.2/13.3 + the venv. The **R2 native fp8-input e4m3 MMA now compiles, runs, and is VERIFIED a real
fp8 tensor-core MMA** — the generated CUDA emits `tl::mma_sync<kFloat8_e4m3,kFloat8_e4m3,kFloat32,16,8,32>`
(zero `kFloat16`, zero `__nv_fp8`-decode, the native SM120 m16n8k32 f8f6f4 atom), fp32 accumulate, parity
0.0376. It is **2.2–3.5× faster than the §19/§20 fp16-decode R2** (23.4/25.2/23.6/31.6 vs 7.4/7.2/10.6/14.1
TFLOPs) — the lever's hard part (emitting a native e4m3 MMA from our own `T.gemm` on sm_121) is **DONE** —
**but at 0.69–0.95× of bf16 it is still slower than bf16** (untuned single 16×32×32 / 64-thread tile, low
occupancy), so it is a **real-fp8-MMA GO and a speed NO-GO**; a <1× kernel will not replace bf16 (RULE #1).
The **R1 standalone TE DelayedScaling(E4M3) GEMM stays the only measured ~1.7× win** (1.57–1.83×, §19/§20).
**Wiring R1 into the e2e step is a memory-NO-GO:** the new `bench_fp8_e2e_step_20260604.py` A/B ran the
**bf16 arm (MEASURED: 74.26 s/step, 13.8 tok/s, 37.58 GB MLX peak, seq=1024 bs1 13-layer, raw MLX-eager —
not the §17 graph-path)**, but the **fp8 arm RAISED in the fp8 BACKWARD** at the MLX→torch zero-copy bridge
with `cudaMallocAsync ... out of memory` — reproduced down to the absolute minimum (seq=512, NO optimizer,
30 GB MLX cap, 88 GB torch headroom), so it is **NOT** an activation-size OOM but **MLX↔torch
`cudaMallocAsync` unified-memory contention** on the single GB10 (MLX's pool reserves unified memory and
won't yield it to torch's competing alloc during the fp8 backward). Per RULE #1 the fp8 GEMM RAISED with
where+what — it did **NOT** silently degrade to bf16 — so the realized fp8 e2e step speedup is **UNMEASURED**
(blocked by allocator coexistence, not by the GEMM). Honest e2e-vs-Megatron: with the fp8 step blocked,
**§17's ≈907/≈298 tok/s (gap ≈3.75×/≈11.4×) remains the canonical step result**; even unblocked, fp8 helps
only the transformer-block GEMM fraction + memory (the SSD scan is launch-bound and fp8 does NOT touch it,
and at §17 the GEMMs are ~2% of the step vs the backward's ~80–91%), so fp8 is a memory + GEMM-phase lever,
not a step-level catch-up to Megatron. Net: the box is cleaner, our own native fp8 MMA is real (and the
tuning target is now occupancy, not feasibility), and the TE-fp8 e2e path needs an MLX/torch allocator
co-residency fix (shared pool / unified allocator) before it can be measured end-to-end on this box.

## §22. SIX LEVERS RE-MEASURED — fp8 MMA tile-tuning now CROSSES bf16 (1.17–1.55×), DLPack reject FIXED (bridge works), the mamba gap is UN-FUSED KERNELS not launch-staging, CUTLASS 4.5.1 builds but SF-layout NO-GO (MEASURED, gb10 sm_121, 2026-06-04)

Single exclusive GB10 owner, serial. Each GPU run: ownership poll + drop_caches before/after,
SIGTERM-only. Six levers measured; headline numbers below are MEASURED unless tagged EXTRAPOLATION.

### Lever: MLX CUDA-graph-cap sweep (the cheap high-value test) — REFUTES launch-bound for the MLX path
MLX `mlx/backend/cuda/device.cpp` `get_graph_limits()` hard-codes `case 1210 (DGX Spark): ops=20,
mb=25` — far below H100/Blackwell `ops=100, mb=1000`. Swept `MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER`
on the runnable bf16 full-model step (`scripts/bench_fp8_e2e_step_20260604.py --arm bf16`, seq=512 bs1 13L):

| ops/mb | tok/s | step s | MLX peak GB |
| --- | --- | --- | --- |
| 20/25 (GB10 default) | **31.7** | 16.17 | 13.2 |
| 64/200 | 31.4 | 16.30 | 13.3 |
| 200/2000 | 26.7 (WORSE) | 19.15 | 32.6 |
| 400/4000 | 26.4 (WORSE) | 19.42 | 52.0 |

**Raising the caps did NOT raise tok/s — it stayed flat then REGRESSED while peak memory ballooned
13→52 GB. This MLX-eager full-model step is NOT launch/graph-cap-bound; it is kernel/compute-bound.**
The ONE real win was `MLX_CUDA_GRAPH_CACHE_SIZE=4000`: the GB10 default (400) **crashes** the bf16 step
with `"Cache thrashing... set MLX_CUDA_GRAPH_CACHE_SIZE to a larger value than 400"` + a hard
`cudaGraphAddDependencies ... invalid argument` — raising the cache size to 4000 fixes that crash and is
the actionable knob (the cap sweep is not).

### Lever 1 (dlpack-fix): the §21 "torch.from_dlpack REJECTED the MLX native kDLCUDA export" is FIXED
On the current MLX build (`0.32.0.dev20260602+6680d75a`) the native CUDA DLPack export landed:
`mx.array(bf16).__dlpack_device__() == (kDLCUDA=2, 0)` and **`torch.from_dlpack(arr)` now ACCEPTS the
MLX native kDLCUDA capsule** → a `cuda:0` torch view (MEASURED probe). The isolated single-Linear fp8
fwd+bwd through the bridge (`_fp8_linear_impl` + `_fp8_linear_bwd`, prod MoE-up shape K=3584 N=18944,
M=128) **fully SUCCEEDS** (dgrad (128,3584)+wgrad (18944,3584), MLX peak 0.38 GB, torch 0.5 GB) — the
bridge, TE fp8 backward, and zero-copy writeback all work. So the DLPack-bridge half of lever 1 is GO.

**BUT the fp8 E2E step still OOMs — at a DIFFERENT spot than §21 claimed.** The OOM is now precisely
`fp8_te_linear.py:321 mx.eval(x2, weight, g2)` inside the FIRST `_fp8_linear_bwd`, i.e. MLX's OWN
`cudaMallocAsync(...) failed: out of memory` while materializing the full 13-layer reverse-mode graph in
one eval — NOT a DLPack reject and NOT TE memory. Diagnosis (MEASURED): bf16-only full step peaks at just
**7.63 GB @ seq128 / 9.49 @ seq256**, with 110+ GB physically free; yet the fp8 arm's first deep-backward
`mx.eval` triggers a cudaMallocAsync pool-growth failure regardless of `CPPMEGA_TORCH_MEM_FRAC` (tried
0.30/0.45/0.95) or `CPPMEGA_MLX_MEMORY_LIMIT_GB=70`. Switching torch off cudaMallocAsync to the native
caching allocator makes it WORSE (kernel OOM-kill, total-vm 161 GB). Root cause = **two cudaMallocAsync
pools (MLX's + TE's) cannot both grow on the GB10 UVM device**: the per-Linear bridge crossing forces a
mid-backward full-graph materialization that the coexisting pools cannot satisfy. This needs the MLX↔torch
shared/unified-pool C++ rebuild (the dlpack-fix STEP 0, deferred this round as box-risk) OR restructuring
the fp8 seam to not force a full mid-backward eval. **fp8 e2e tok/s remains UNMEASURED; bf16 reference =
97.4 tok/s @seq128 / 55.9 @seq256 / 31.7 @seq512 (bs1 13L).**

### Lever 2 (fp8-mma-tune): native SM120 e4m3 MMA tile-tuning CROSSES bf16 on ALL prod shapes — GO
`scratch/fp8_mma_autotune.py --prod` drove the pipelined tunable native prim through TileLang AutoTuner.
The §21 "0.69–0.95× untuned single tile" residual is RESOLVED — raising `threads` 128→256 and
`num_stages` 0→2/3 (real pipelining) crosses >1× on every prod GEMM:

| shape | bf16 TFLOPs | best fp8 tile (M,N,K,stages,threads) | fp8 TFLOPs | ×bf16 | rel_err |
| --- | --- | --- | --- | --- | --- |
| mlp_up_gate 16384×37888×3584 | 33.7 | (128,64,64,**3**,**256**) | **39.6** | **1.17×** | 2.1e-7 |
| mlp_down 16384×3584×18944 | 34.3 | (128,128,64,2,256) | **51.9** | **1.51×** | 1.4e-6 |
| attn_qkv 16384×10752×3584 | 33.5 | (128,128,64,2,256) | **41.8** | **1.25×** | 2.1e-7 |
| attn_out 16384×3584×3584 | 33.3 | (128,128,64,2,256) | **51.7** | **1.55×** | 2.1e-7 |

All parity-passing (rel_err ≤ 1.4e-6, essentially exact). `BEST_CONFIG_JSON` printed for pinning into
`_native_fp8_tile_for`. **Native fp8 MMA is now a real GEMM-phase win (1.17–1.55×) — pin the tuned tiles.**

### Lever 5 (mamba3-compare, THE KEY): the gap is the UN-FUSED multi-kernel SSD, NOT launch-staging
At the SAME §17 prod config (S=4096, c=64, g=8, H=112, P=64, N=64, bs1) — see
`docs/MAMBA3-PATHC-VS-CPPMEGA.md` for the full table:
- **cppmega** (Megatron SSD Triton `mamba_chunk_scan_combined`, the EXACT op cppmega drives): **fwd 3.11 ms
  in ONE fused kernel**, bwd ≈10.0 ms (state resident in smem).
- **OURS (path_c)**: **F0 precompute ALONE = 16.37 ms (≥5.7× cppmega's WHOLE 3.11 ms fused fwd)**, F1 1.24
  ms; **F2 scan-combine SEGFAULTS at launch** at the prod state N=64 (only legal tile block_Dstate=64
  exceeds the GB10 ~99 KB dynamic-smem cap; the code RAISES on block_Dstate<64 rather than truncate state
  — RULE #1). cppmega's m2rnn Triton also could not run (smem OutOfResources, 101 KB > 99 KB cap); ours
  m2rnn is Metal-only (NOT-RUNNABLE-on-CUDA, not fabricated).
- **VERDICT: the loser is the KERNEL DECOMPOSITION, not host/launch staging** (consistent with the MLX
  graph-knob refutation above). path_c splits the SSD forward into F0/F1/F2 separate gridded kernels that
  round chunk tensors through global memory; cppmega fuses it into one smem-resident kernel. **The lever is
  to FUSE F0/F1/F2 (and B0/B1/B2) into a single smem-resident SSD kernel and make the N=64 scan tile fit
  the GB10 smem cap** — not to tune launch overhead.

### Lever 4 (tn-adapter + nvfp4): NVFP4 RtN fwd+bwd is a GEMM-level GO via path_c; tok/s memory-blocked
- **NVFP4 RtN backward (TE standalone, `nvfp4_te_backward_rtn_probe`)** RUNS fwd+dgrad+wgrad at 3 shapes
  (256×512×256, 1024×3584×3584, 2048×3584×10752), all finite: fwd_rel ≈ 0.146, dgrad_rel ≈ 0.146,
  wgrad_rel ≈ 0.134 vs bf16 (matches §21 prior 0.1465/0.1343 — the FP4 RtN quant floor).
- **`nvfp4_path_c_gemm` (MLX operands, lifted into path_c via the lever-1 bridge)** runs **fwd AND the
  mx.grad VJP backward** at 1024×3584×3584 (rel_vs_bf16 0.146, dgrad finite) — the path_c FP4 GEMM route
  is wired through the working DLPack bridge.
- **mxfp8 TN-adapter**: structurally present (lifted from cppmega's WORKING shim — retargets cuBLAS-
  unsupported NN/NT dgrad/wgrad to TN); as a standalone it correctly reports not-active — it only fires
  inside TE's `general_gemm` during a full mxfp8 backward, which is the same MLX-graph-eval OOM as lever 1.
- **tok/s effect UNMEASURED** (wiring fp4/mxfp8 into the full model hits the lever-1 OOM); GEMM-level parity
  + runs are the GO evidence.

### Lever 3 (cutlass-mxfp8, heaviest/last): CUTLASS 4.5.1 BUILDS + dispatches on sm_121a, but SF-layout NO-GO
`_cutlass_mxfp8_sm120_build.sh` cloned **CUTLASS v4.5.1** (host/CPU-only side-checkout, did NOT clobber the
live 4.1.0) and **nvcc-13.3 compiled `_cutlass_mxfp8_sm120.so` clean for `sm_121a`** (launchers
`cppmega_mxfp8_gemm_sm121` / `_last_error` / `_sf_sizes` exported). The kernel LOADS and DISPATCHES, BUT the
smoke parity **FAILS the gate: rel_err 0.41 > 0.12** → the code RAISES (RULE #1: refuses to report a
passing/TFLOPs route for a numerically-wrong result). Cause: the E8M0 scale-factor (SFA/SFB) atom layout the
Python driver packs does NOT match CUTLASS's swizzled `Sm1xxBlkScaledConfig::tile_atom_to_shape_SF` ordering
(scales land on the wrong 32-element blocks). The .cu side correctly uses `tile_atom_to_shape_SFA/SFB`; the
fix is a host-side SF repack to the atom-swizzled order. **TFLOPs NOT reported (the result is garbage). NO-GO
this round; build/dispatch/launch CONFIRMED, only the SF-packing layout remains. Target stays 188 TFLOPs / 4×.**

### tok/s vs Megatron 3399 (honest)
**This round did NOT raise the realized e2e step tok/s past §17 (≈907 @8L / ≈298 @28L).** The fp8/fp4 e2e
arm is memory-blocked (lever 1 OOM), so its tok/s is UNMEASURED, not a regression. The bf16 reference at
small configs is 31.7–97.4 tok/s (seq512→seq128, bs1, 13L). The real, MEASURED forward progress is at the
GEMM phase (lever 2: fp8 MMA 1.17–1.55× bf16, tuned, parity-clean) and the DLPack bridge (lever 1: native
kDLCUDA export works, single-GEMM fp8 bwd works). **§17's ≈907/≈298 stays the canonical step number.** The
KEY structural finding (lever 5) is that path_c's mamba forward is ≥5.7× cppmega's because it is un-fused
(F0 alone 16.37 vs 3.11 ms fused) and F2 won't launch at N=64 (smem cap) — the throughput lever is SSD
kernel FUSION + smem-fit, not launch/graph tuning (both the graph-knob sweep and the per-region timing
refute launch-bound).

### GO / NO-GO per lever
- MLX graph-knob: **NO-GO as a tok/s lever** (flat→worse); `MLX_CUDA_GRAPH_CACHE_SIZE=4000` is the only
  actionable knob (fixes the default-400 thrash crash). Confirms NOT launch-bound.
- Lever 1 dlpack-fix: **PARTIAL GO** — DLPack reject FIXED, single-GEMM fp8 bwd works; **e2e NO-GO** (MLX+TE
  cudaMallocAsync dual-pool OOM at the mid-backward full-graph eval; needs the shared-pool MLX rebuild).
- Lever 2 fp8-mma-tune: **GO** — 1.17–1.55× bf16, parity-clean, tiles found.
- Lever 5 mamba-compare: **KEY FINDING delivered** — gap = un-fused SSD kernels + F2 smem-cap segfault,
  NOT launch-staging.
- Lever 4 tn-adapter/nvfp4: **GEMM-level GO** (nvfp4 RtN fwd+bwd runs via path_c, parity ~0.146); e2e
  tok/s memory-blocked.
- Lever 3 cutlass-mxfp8: **NO-GO this round** (builds + dispatches on 4.5.1/sm_121a, SF atom-layout
  parity 0.41 → defer the host-side SF repack). **[RESOLVED in §25: the SF repack is fixed, parity now
  0.0377, the kernel CROSSES bf16 at 1.38–1.78× — but does NOT reach the 4–4.5× / 188 TFLOPs target.]**


## §25. CUTLASS MXFP8 SF-LAYOUT BUG FIXED — parity 0.41 → 0.0377 (REAL E4M3+E8M0 on sm_121a), kernel CROSSES bf16 at 1.38–1.78× / 46–59 TFLOPs but does NOT reach the 188 TFLOPs / 4–4.5× target (MEASURED, gb10 sm_121, 2026-06-04)

Single exclusive GB10 owner, serial. Rebuilt the side-checkout `.so` (CUTLASS v4.5.1 header-only at
`/home/dave/source/cutlass-451`, the live tilelang `3rdparty/cutlass` 4.1.0 untouched), nvcc-13.3,
`-gencode arch=compute_121a,code=sm_121a`. ALL numbers MEASURED.

### The root-cause fix (host-side SF repack to match `Sm1xxBlkScaledConfig::tile_atom_to_shape_SF`)
§22 left lever 3 as a NO-GO: the kernel built + dispatched but smoke parity FAILED the 0.12 gate at
**rel_err 0.41** because the Python driver packed the E8M0 SFA/SFB scale bytes as a *contiguous*
`(rows, nblk)` stream, while CUTLASS reads the scale-factor tensor through the **interleaved (swizzled)
atom layout** `Sm1xxBlkScaledConfig<32>::SfAtom = ((_32,_4),(_32,_4)):((_16,_4),(_0,_1))` tiled
row-major into 512-byte (128-row × 4-block) atoms. Every per-32-block scale landed on the WRONG block.

The fix is purely host-side (`cppmega_mlx/nn/_tilelang/cutlass_mxfp8_sm120.py`): `_sf_scatter_index`
builds the `(rows*nblk,)` int64 byte-offset index from the machine-verified closed form
`n_katom=ceil(nblk/4); within=(r%32)*16+((r//32)%4)*4+(kb%4); tile=(r//128)*n_katom+(kb//4);
offset=within+tile*512`, then `mxfp8_gemm_from_hp` SCATTERS each ue8m0 byte to its swizzled offset
into a zero-filled atom buffer (sized by the kernel's own `cppmega_mxfp8_sf_sizes`). The `.cu` gained
a `cppmega_mxfp8_sf_offset` export that calls CUTLASS's actual `tile_atom_to_shape_SFA/SFB` so the
Python closed-form is **cross-checked byte-for-byte against CUTLASS** on the first drive of each shape
(RULE #1: a future CUTLASS swizzle change RAISES instead of silently mis-packing).

**Cross-check: PASS, 0 mismatches** vs CUTLASS over a spread of (row, kb) probes incl. the swizzle
boundaries (row 33/127, kb 4) and the edge shape M=130 / nblk=5. E.g. (SFA, row=127, kb=3) → kernel
511 == python 511; (SFB, row=129, kb=0) → kernel 1040 == python 1040.

### Smoke parity — PASS honestly (gate kept at 0.12, ALL elements, no subset)
`scratch/cutlass_mxfp8_bench.py --smoke` (rel_err over the FULL output, the §22 0.41 → now):

| shape | rel_err | gate | verdict |
| --- | --- | --- | --- |
| smoke_256 256×256×256 | **3.776e-02** | 0.12 | PASS |
| smoke_512 512×512×512 | **3.787e-02** | 0.12 | PASS |
| smoke_1024 1024×1024×1024 | **3.778e-02** | 0.12 | PASS |

rel_err lands in the expected E4M3 mxfp8 band (~0.0376 like §21 R2 / §19–§20 per-tensor), confirming
this is REAL block-scaled MXFP8 (E4M3 data + E8M0 per-32 scales), not a secret bf16 path. The gate was
NOT loosened; no elements were subset.

### Prod bs4 TFLOPs — REAL mxfp8, parity-clean, but the host quant+scatter overhead must be isolated
The `--prod` bench times `mxfp8_gemm_from_hp` (quantize bf16→e4m3 + build+scatter E8M0 scales + GEMM)
on EVERY iter, so it conflates host prep with the kernel. Isolating the **pure CUTLASS kernel** (operands
pre-quantized once, only `mxfp8_gemm` timed) vs the `from_hp` path (MEASURED, all rel_err 0.0377):

| shape | bf16 TFLOPs | mxfp8 KERNEL TFLOPs | ×bf16 | from_hp TFLOPs (incl host) | host quant+scatter |
| --- | --- | --- | --- | --- | --- |
| mlp_up_gate 16384×37888×3584 | 33.6 | **46.3** | **1.38×** | 25.8 | 76.0 ms (44%) |
| mlp_down 16384×3584×18944 | 34.3 | (not isolated)¹ | — | 12.1 | dominated by K=18944 scatter |
| attn_qkv 16384×10752×3584 | 33.5 | **46.0** | **1.37×** | 19.7 | 36.8 ms (57%) |
| attn_out 16384×3584×3584 | 33.3 | **59.4** | **1.78×** | 12.2 | 27.3 ms (79%) |

¹ mlp_down's huge K=18944 → nblk=592 makes the per-call scatter the bottleneck; the kernel itself is in
the same 46–59 TFLOPs band as the others.

### Verdict — GO on correctness, NO-GO on the 188 TFLOPs / 4–4.5× target
**The SF-layout bug is FIXED and the route is now REAL, parity-clean MXFP8 that CROSSES bf16 (kernel
1.38–1.78×, 46–59 TFLOPs).** This is the same direction as lever 2's native fp8 MMA (1.17–1.55×) — a
real GEMM-phase win — but **it does NOT approach the ~188 TFLOPs / 4–4.5× DGX-Spark figure** from the
NVIDIA Dev Forum thread. On this GB10 sm_121a part the CUTLASS block-scaled MXFP8 config peaks at
~46–59 TFLOPs, ~1.4–1.8× bf16, NOT 4×. Two honest residuals before this is pipeline-usable:
1. **The 4× gap is real, not a measurement artifact** — even the pure kernel (host overhead excluded)
   tops out at 1.78×. The 188 TFLOPs figure is not reproduced here; report it as NOT MET.
2. **Host quant+scatter is 44–79% of the per-call `from_hp` wall time** — in a real pipeline the weights
   are quantized ONCE (amortized) and only activations re-quantize per step, so the *effective* speedup
   is closer to the 1.4–1.8× kernel number than to the from_hp number. The current per-call scatter
   (especially for large-K mlp_down) needs a fused quant+swizzle kernel to not dominate.

So lever 3 moves from §22 NO-GO to **partial GO**: correct, real MXFP8, crosses bf16 — but the headline
4–4.5× / 188 TFLOPs target is **NOT MET** on GB10 sm_121a (honest MEASURED, no fabrication).

## §23. FUSED smem-resident SSD FORWARD — N=64 smem fit SOLVED + parity PASS, but the naïve serial-reduction fusion is a PERF NO-GO (126.7 ms, 40.7× SLOWER than cppmega's 3.11 ms) (MEASURED, gb10 sm_121, 2026-06-04)

The §22 lever-5 prescription was: fuse F0/F1/F2 into ONE smem-resident SSD kernel (mirror cppmega's
Triton `mamba_chunk_scan_combined`, state resident in smem) AND make the N=64 scan tile fit the GB10
~99 KB cap. This round built `cppmega_mlx/nn/_tilelang/mamba3_ssd_fused_fwd.py`
(`ssd_fused_fwd_cuda_prim`): a persistent per-(batch,head) threadgroup that loops the 64 chunks
serially, carrying `state[headdim,dstate]` fp32 RESIDENT in shared memory — cb/dA_cumsum/summary/
prev_states NEVER round-trip to global (only Output + final_state are written). Env-gated
`CPPMEGA_MAMBA3_SSD_FUSED_FWD` (default OFF); the un-fused F0/F1/F2 + Metal path stay byte-identical.

**ALL numbers MEASURED at the §17 prod cfg (S=4096 c=64 g=8 H=112 P=64 N=64 bs1), gb10 sm_121,
`scratch/probe_ssd_fused_fwd_gb10.py --prod`, warm median over 20 timed device-resident dispatches.**

### What WORKED (two real wins, both MEASURED)
- **N=64 smem fit SOLVED — the fused kernel LAUNCHES at N=64.** Per-threadgroup static smem budget =
  **82,688 B (80.75 KiB) < the 101,376 B (~99 KB) GB10 opt-in cap**, 18,688 B headroom (fp16 operand
  tiles + fp32 accumulation; an all-fp32 staging would be 112.75 KiB and overflow). Grid `(1,112)` =
  112 persistent threadgroups. COMPILE ok (out_idx=[7,8]), RUN ok, NaN=False.
- **The un-fused F2 ALSO now LAUNCHES at N=64.** The §22 "F2 SEGFAULTS at launch at N=64" was NOT a
  block_Dstate-vs-99KB-smem problem (F2's default `block_Dstate=128 ≥ dstate=64` already passes its
  own guard); the actual root cause was a **dtype contract break**: `chunk_scan_fwd_cuda_prim` declared
  `prev_states` fp16 while F1 writes it fp32 (§3.3) — a 2-byte-typed read of a 4-byte buffer walked the
  dstate stride OOB → launch crash. The Phase-0 fix (declare `prev_states` accum_dtype) makes the
  **whole un-fused F0→F1→F2 chain run end-to-end at prod N=64** (F2 RUN ok, NaN=False). This corrects
  the §22 attribution of the segfault.
- **Parity PASS over ALL elements (fp16 gate 5e-4).** fused-vs-un-fused: out max|abs| **4.883e-04**,
  final_state **3.159e-06**. fused-vs-serial: out **4.750e-04**, final_state **1.818e-06**. un-fused-vs-serial:
  out **4.750e-04**, final_state **4.977e-06**. The fused SSD math is numerically faithful.

### What FAILED — the fused kernel is 40.7× SLOWER than the target (PERF NO-GO)
| kernel | MEASURED fwd ms (prod) |
|---|---|
| **fused F0+F1+F2 (this round)** | **126.69 ms** (min 123.65) |
| un-fused F0 | 24.57 ms |
| un-fused F1 | 1.23 ms |
| un-fused F2 | 4.22 ms |
| **un-fused SUM (F0+F1+F2)** | **30.02 ms** |
| cppmega Megatron fused (ref) | 3.11 ms |

- fused vs un-fused = **0.24× (≈4.2× SLOWER)**. fused vs cppmega 3.11 ms = **40.74× SLOWER**.
- **Root cause (honest): the fusion eliminated the global round-trips but replaced the GEMMs with
  serial scalar reductions.** The un-fused F0/F2 use `T.gemm` (tensor cores) for cb=C@B^T, the diagonal
  scan, and the off-diagonal C@state; the fused prim does every one of those as an explicit
  `T.serial` scalar FMA loop over shared tiles, on a grid of only 112 threadgroups × 128 threads. The
  per-chunk work (cb: L·L·N, Y_diag: L·P·L, summary: P·N·L) is now scalar, serial, and tensor-core-idle,
  looped 64× per threadgroup. Killing the 16–24 ms of F0 global traffic was real, but it was dwarfed by
  the ~100 ms of lost GEMM throughput. **Residency-without-tensor-cores is a net loss here.**
- Note F0 measured **24.57 ms** this run (vs the §22-cited 16.37 ms) — same un-fused kernel, re-measured;
  it remains the dominant un-fused cost and the ≥5× cppmega gap stands either way.

### tok/s effect if wired into the §17 step
NOT wired (env gate default OFF — nothing regresses). Wiring this fused fwd in would make the mamba
forward **WORSE** (126.7 ms replacing the 30.0 ms un-fused chain, +96.7 ms/mamba-block/step), so the
step tok/s would DROP, not rise. No tok/s gain to report — reporting a drop honestly rather than a
fabricated speedup. EXTRAPOLATION only (not benchmarked end-to-end) because it is a clear regression and
running it in the model would only confirm a slowdown.

### GO / NO-GO (honest)
- **GO: the N=64 smem-fit + launch goal of the "minimum viable win".** The prod config now runs
  end-to-end at N=64 — both the new fused kernel AND the un-fused F2 (via the prev_states dtype fix).
  Parity is clean over all elements.
- **NO-GO: the perf goal (approach cppmega's 3.11 ms).** The naïve smem-resident scalar-reduction
  fusion is 40.7× too slow. Smem-residency alone is not the lever; the lever is **GEMM/tensor-core
  utilization**. The next iteration must keep the chunk state resident BUT do cb / Y_diag / Y_off /
  summary as `T.gemm` (block-tiled, tensor-core) inside the persistent threadgroup — i.e. the cppmega
  Triton structure (smem-resident block-MMA), not a per-(batch,head) serial scan. Until then the
  un-fused F0/F1/F2 chain (30.0 ms, now N=64-clean) remains the production path. RULE #1: the fused path
  stays env-gated OFF and RAISES on tile/smem/compile failure — it never silently replaces the faster
  un-fused chain.


## §24. fp8 E2E BACKWARD OOM CLEARED — the §22 "dual-pool" premise was WRONG (MLX+torch already share ONE cudaMallocAsync pool); the real blocker was torch's `set_per_process_memory_fraction` POISONING that shared pool. fp8 e2e step now COMPLETES (MEASURED, gb10 sm_121, 2026-06-04)

Single exclusive GB10 owner, serial; ownership poll + drop_caches before/after, SIGTERM-only.
This closes the §21/§22 lever-1 **e2e NO-GO** ("R1-e2e fp8 backward is a memory-NO-GO … needs the
shared-pool MLX rebuild"). It needed **no MLX rebuild** — the fix is Python-only, and the §22
root-cause diagnosis was partly wrong. All numbers below are MEASURED.

### MEASURED root cause (the §22 "two pools double-reserve" premise does NOT hold on this build)
Direct ctypes probe of the CUDA runtime (`libcudart.so`, MLX `0.32.0.dev20260602+6680d75a` + torch
`cudaMallocAsync` backend):
- The device's **CURRENT** cudaMallocAsync pool **already EQUALS the device DEFAULT pool**
  (`cudaDeviceGetMemPool == cudaDeviceGetDefaultMemPool`), so `cudaDeviceSetMemPool(dev, mlx_default)`
  is a **no-op repoint** — there was never a second device-current pool to merge.
- **Both** allocators already draw from that ONE pool: a 2 GB torch alloc then a 2 GB MLX alloc grow
  the SAME pool's `ReservedMemCurrent` 2→4 GB; a 40 GB MLX + 10 GB torch co-allocation reaches **50 GB
  on the one pool with NO double-count**. So the "MLX pool + torch pool each reserve-and-retain" model
  from §21/§22 is **refuted** for this MLX+torch combination — there is structurally already one shared
  pool.
- The ACTUAL blocker: **`torch.cuda.set_per_process_memory_fraction(F)` POISONS the shared pool.**
  MEASURED, that call flips the shared default pool's `cudaMemPoolAttrReleaseThreshold` `0 →
  UINT64_MAX` (retain-everything) AND installs a hard per-process ceiling. Because MLX's `mx.eval`
  draws from the SAME pool, MLX's deep-backward growth trips torch's ceiling and OOMs at
  `fp8_te_linear.py:321 mx.eval(x2, weight, g2)` — **even at seq128 with frac 0.95**. Worse, torch's
  cudaMallocAsync backend **RE-SETS that threshold to UINT64_MAX on its FIRST allocation** after we
  zero it, so a one-time `ReleaseThreshold=0` is silently clobbered during the fp8 forward.

### THE FIX (Python-only, no MLX rebuild)
1. **Drop the `set_per_process_memory_fraction` cap when the shared pool is active**
   (`scripts/bench_fp8_e2e_step_20260604.py` `_budget_mlx_torch`): the cap only makes sense in the
   genuine dual-pool fallback (`CPPMEGA_FP8_SHARED_POOL=0`); with the one shared pool it starves MLX.
2. **Re-assert `ReleaseThreshold=0` on EVERY bridge-boundary trim, not once**
   (`cppmega_mlx/nn/_tilelang/_cuda_zerocopy.py` `trim_cuda_mempool` / `trim_device_current_mempool`):
   torch keeps re-poisoning the shared pool to retain-everything, so the threshold must be re-zeroed
   each backward so `cudaMemPoolTrimTo(0)` can actually reclaim idle reserved memory.
3. `install_shared_mempool()` stays as the up-front (idempotent) **ReleaseThreshold=0 re-assert** after
   torch's CUDA init — the `cudaDeviceSetMemPool` part is a confirmed no-op here, but the threshold
   re-assert is load-bearing.

### MEASURED result — the fp8 backward COMPLETES (no `cudaMallocAsync … out of memory`)
`scripts/bench_fp8_e2e_step_20260604.py`, bs1, `--fp8-floor 0` (fp8 forced on EVERY wired Linear — the
stress case the §22 OOM hit), `MLX_CUDA_GRAPH_CACHE_SIZE=4000`, `CPPMEGA_FP8_SHARED_POOL=1`:

| arm | seq | optimizer | step s (median) | tok/s | MLX peak GB | torch peak / reserved GB | final loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **bf16 ref** | 64 | no | 0.544 | **117.6** | 6.878 | 0.0 | 11.345 |
| **fp8 e2e** | 64 | no | 8.66 | **7.4** | 40.754 | 40.656 / 40.75 | 11.25562 |
| **fp8 e2e** | 64 | **yes (AdamW)** | 11.75 | **5.4** | 54.43 | 54.332 / 54.469 | 7.398 |

- **OOM CLEARED:** all 271 per-Linear fp8 backward crossings + the final `mx.eval(loss, grads)` (and
  the full AdamW update) complete with **EXIT 0**. Traced memory grows monotonically to ~43.7 GB
  (no-opt) / 54.4 GB (opt) and **stays under the 117 GB part** at seq64. The §22 NO-GO is now a **GO**
  at a safe config.
- **ONE shared pool, no double-count (the proof):** at the fp8 peak, `mlx_active ≈ torch_reserved ≈
  pool_reserved ≈ 40.7 GB` (no-opt) / `≈ 54.4 GB` (opt) — all three counters track the SAME number, i.e.
  a single reservation feeding both allocators, not two summed pools. The release threshold stayed
  `thr=0` across every backward (the re-assert holds against torch's re-poisoning).
- **Parity:** fp8 final loss 11.25562 vs bf16 11.345 (Δ −0.08938) at the un-optimized step; with the
  AdamW update the fp8 loss drops 11.25 → 7.398, confirming grads flowed end-to-end (finite, applied).
- **Zero-copy PRESERVED (RULE #1):** the bridge took the **native kDLCUDA export** path
  (`_mlx_native_cuda_dlpack_available == True` → `torch.from_dlpack(arr)`), a real `cuda:0` device view.
  Proof of SHARED storage: writing 7.0 through the torch view changed the source MLX array's sum to
  exactly 224.0 (4×8×7). No host copy, no `.cpu()`, no fallback — the path RAISES on a genuinely
  uncrossable tensor.

### Residual (honest)
The fp8 e2e step is **16× SLOWER than bf16** at this config (7.4 vs 117.6 tok/s) — NOT because of the
GEMM (lever 2 showed the tuned fp8 MMA is 1.17–1.55× bf16) but because `--fp8-floor 0` forces fp8 on
EVERY tiny Linear, so 271 per-Linear MLX↔torch bridge crossings + TE fp8 quant/dequant per backward
dominate the step. The realistic config (a sane floor that only fp8s the big GEMMs) was not the goal
here — the goal was to **clear the OOM and prove the backward completes**, which it does. The memory
footprint also grows ~linearly with seq and with the number of fp8 sites (seq128 floor0 still peaks at
~119 GB and OOMs — just over the part); seq64 floor0 is the MEASURED safe config. The next lever is to
(a) raise the fp8 floor so only the cost-effective GEMMs cross the bridge, and (b) batch the per-Linear
backward crossings so the deep graph materializes incrementally instead of accumulating to tens of GB.

### GO / NO-GO
- **GO: the §22 e2e fp8-backward OOM is FIXED, Python-only, no MLX rebuild.** Backward + optimizer
  complete; shared pool confirmed single-reservation; zero-copy preserved.
- **NO-GO (unchanged): fp8 e2e as a tok/s WIN at floor0.** 16× slower (bridge-crossing-bound, not
  GEMM-bound). The realized e2e tok/s win still awaits floor-tuning + crossing-batching; §17's
  ≈907/≈298 stays the canonical bf16 step number.


## §26. F0 PRECOMPUTE RE-EXPRESSED AS TENSOR-CORE T.gemm — the §22/§23 MEASURED 24.57 ms lever is GONE: new F0 = 1.13 ms (REAL sm_121 HMMA, parity PASS all elements), at Tri-Dao chunk_state+bmm parity. F0 is no longer the forward bottleneck; but the §17 STEP is backward-bound so the headline tok/s is essentially unchanged (MEASURED, gb10 sm_121, 2026-06-04)

**What changed.** §22/§23 named the path_c → Megatron 3399 gap root cause: OUR F0 precompute was
pathologically slow because `chunk_precompute_fwd_metal_prim` computed both its GEMM-shaped
reductions as SERIAL scalar loops — `cb[L,L] = C@B^T` as a per-`(li,si)` dot over dstate, and
`summary_states[P,N]` as a `P*N=4096`-cell serial-`l` accumulate — emitting NO tensor-core MMA. This
round adds `chunk_precompute_fwd_cuda_prim` (the sm_121/gb10 twin), re-expressing those two
head-independent loops as `T.gemm` and keeping `dA_cumsum` as the single-lane serial scan (it is a
scan, not a GEMM). `build_chunk_precompute_metal` selects the CUDA prim when the resolved target is
CUDA (mirrors F2's per-target branch); the serial Metal prim is **byte-identical** and F1 is
**untouched**.

  - **(1) cb** = `T.gemm(C_shared[L,N], B_shared[L,N], cb_frag[L,L], transpose_B=True)` per
    `(chunk, group)`, written once/group via the `head_idx % heads_per_group == 0` guard
    (Tri-Dao `_bmm_chunk_fwd`).
  - **(2) summary_states** = `T.gemm(x_dec_shared[L,P], B_shared[L,N], states_frag[P,N],
    transpose_A=True)` per `(chunk, head)` — the dominant `P*N` term. The per-row
    `decay*dt = exp2((dacs[L-1]-dacs[l])*p)*dt[l]` is folded into the **x operand** (Tri-Dao
    `_chunk_state_fwd` order; B kept native fp16; no `minimum(.,0)` clamp, matching the serial sign
    convention). Output stays fp32 (= accum_dtype; the §23 N=64 dtype contract).

DTYPE contract unchanged (§23): fp16 operands, fp32 register-fragment C accumulate, `cb` stored fp16,
`summary_states` stored fp32. Idioms mirror the proven F2 `chunk_scan_fwd_cuda_prim`: plain
`scope="shared"` operands (NO `make_swizzled_layout`), `disable_tma=True` on every copy, and
`pass_configs={tl.disable_tma_lower, tl.disable_warp_specialized}` on the CUDA branch. A smem-budget
gate (48.5 KB at prod L=P=N=64 « 99 KB) and an m16n8k16 divisibility gate **RAISE** on violation —
RULE #1, no serial fallback.

### MEASURED on gb10 sm_121 (prod cfg b=1 S=4096 c=64 g=8 H=112 P=64 N=64)

| F0 variant (same gb10 box, same session) | per-call ms (median / min) |
|---|---:|
| **NEW tensor-core CUDA F0** | **1.129 / 1.113 ms** |
| serial F0 prim re-built on CUDA (the §22/§23 kernel, this box) | 16.36 / 16.35 ms |
| §23-headline serial F0 (slower box state, same kernel) | 24.57 ms |
| Tri-Dao `_bmm_chunk_fwd` + `_chunk_state_fwd` (cb+state) | 1.061 ms |
| Tri-Dao cb + state + `_chunk_cumsum_fwd` (≈ all F0 does) | 1.133 ms |

- **vs serial F0:** **14.5×** faster than the same-box serial F0 (16.36 ms), **21.8×** faster than the
  §23-headline 24.57 ms. The §22/§23 "F0 is the 24.57 ms lever" finding is now CLOSED.
- **vs cppmega / Tri-Dao:** new F0 (1.129 ms, which ALSO does the `dA_cumsum` scan) is **1.06×**
  Tri-Dao's cb+state (1.061 ms) and **0.996×** their cb+state+cumsum (1.133 ms) — i.e. at Tri-Dao
  tensor-core parity for the same work.

### REAL sm_121 tensor-core MMA — verified at codegen AND SASS (not relabeled serial)

- **Codegen** (`/tmp/f0_cuda_src.cu`, dumped via `JITKernel.get_kernel_source()`): the cb and
  summary_states regions emit `tl::mma_sync<kFloat16,kFloat16,kFloat32,16,8,16,...>` **×4** (2 per
  GEMM) fed by `ptx_ldmatrix_x4` / `ptx_ldmatrix_x4_trans` **×4**; **zero** leftover
  `for (int n = 0; n < 64 …)` serial-N reduction dots. `tl::mma_sync` routes to CuTe
  `SM80_16x8x16_F32F16F16F32_TN`, whose inline asm is
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
- **SASS** (`nvcc -arch=sm_121a -cubin` of the dumped `.cu`, `cuobjdump -sass`): **64×
  `HMMA.16816.F32`** (the machine tensor-core op) + **32× `LDSM`** (ldmatrix). A relabeled serial
  loop cannot produce HMMA in the binary. CONFIRMED real.

### Parity — ALL elements, fp16 gate 5e-4 (PASS)

| output | new F0 vs serial F0 reference (fp32 einsum) | new F0 vs serial Metal prim (rebuilt on CUDA) |
|---|---:|---:|
| `cb` | 1.22e-4 | 2.44e-4 |
| `summary_states` | 3.74e-6 | 3.74e-6 |
| `dA_cumsum` (fp16-stored) | 4.89e-4 | 0.0 (verbatim scan) |

NaN=False. Both GEMM outputs pass the 5e-4 fp16 gate over the FULL prod tensors (no subset).

### F0 → F1 → F2 chain with the new F0 (MEASURED)

| stage | ms (median) |
|---|---:|
| **F0 (new tensor-core)** | **1.129** |
| F1 inter-chunk recur | 1.230 |
| F2 scan+combine | 2.885 |
| **chain SUM** | **5.244** |

The un-fused forward chain drops from **≈30.0 ms** (§23, serial F0) to **5.24 ms** — now **1.69×**
cppmega's whole fused mamba fwd (3.11 ms), down from ≈9.6×. F0 is no longer the forward bottleneck;
F2 (2.88 ms) is now the largest forward term.

### tok/s effect — EXTRAPOLATION (clearly labelled; NOT a measured step win)

The §17 canonical step is **backward-bound**: the gridded backward chain is **447.8 ms** while the
WHOLE forward chain is **7.231 ms** (§15), of which the §17/§15 step already ran F0 as the **5.025 ms**
chunked precompute (NOT the 24.57 ms serial — that figure is the §23 *fused-probe* re-measure of the
same kernel). Dropping F0 to **1.13 ms** saves **≈3.9 ms** off the forward chain, i.e. **≈0.86 % of the
≈455 ms step**. So the §17 headline **tok/s is essentially UNCHANGED**: **≈907 @8L / ≈298 @28L**, gap
vs Megatron 3399 **unchanged at ≈3.75×/≈11.4×**. (Step-level tok/s was not re-measured this round; the
above is the honest step-model EXTRAPOLATION.) The win is **forward-chain-level, not step-level**: F0
is removed as a hot kernel and the forward chain now sits at near-cppmega (1.69×), closing the
specific §22/§23 lever. The next step-level lever remains the **backward** (B2 at 334.6 ms, §17 honest
attribution #3), not F0.

### GO / NO-GO

- **GO (kernel-level): F0 is now a REAL tensor-core kernel at Tri-Dao parity.** 1.13 ms (14.5–21.8×
  over serial), HMMA-verified at SASS, parity PASS all elements, RULE #1 raises (no serial fallback).
  The §22/§23 "F0 is the 24.57 ms gap lever" finding is CLOSED — the forward chain is now 5.24 ms
  (1.69× cppmega's 3.11 ms whole fwd).
- **NO-GO (honest, step-level): this does NOT move the headline tok/s.** The §17 step is
  backward-bound (447.8 ms bwd vs 7.2 ms fwd), so a 3.9 ms forward saving is ≈0.86 % of the step.
  §17's ≈907/≈298 stays the canonical number; the next step-level lever is the backward (B2), not F0.
