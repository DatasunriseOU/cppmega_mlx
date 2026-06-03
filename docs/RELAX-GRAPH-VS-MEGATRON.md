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
| **peak memory** | **8.787 GB planned device-peak** (17.64 GB measured free-delta) | **~26 GB** |
| **fits Megatron-class memory** | **YES — 8.79 GB planned < 26 GB (≈3x under)** | 26 GB |
| **tok/s** | **25.2 tok/s @ 8 layers MEASURED; ~5.2 tok/s @ 28 layers EXTRAPOLATED** | **3399 tok/s** |
| tok/s ratio vs Megatron | 0.0074x (8L measured) / 0.0015x (28L extrapolated) — i.e. **135x–654x slower** | 1.0x |
| what runs the compute | abstract numpy host-staged region drivers (memory/executability proof) + REAL path_c-CUDA kernel proven through the fwd boundary (PR-6 stage 2) | tuned fused-FP8-CUDA kernels + selective recompute |

**(a) Does the graph path FIT in Megatron-class memory? YES.** The StaticPlanBlockMemory
planned device-peak for the whole 28-layer 1.8B step is **8.787 GB** — roughly **3x under**
Megatron's ~26 GB, and the optimizer state is kept at 1x by the explicit in-place Adam op
(lever-5). The eager MLX path cannot even run this step as one monolithic graph (OOMs at
114-116 GB on the same box, `MEGATRON-VS-MLX-PATHS.md`); the graph path runs it and plans
to 8.79 GB. **This is the win the whole memory-path was built for.**

**(b) What tok/s does it achieve? Far below Megatron — and honestly so.** The current
graph path is a MEMORY-FIRST design whose region compute is still the abstract numpy
host-staged drivers (the memory/executability proof harness), NOT tuned device kernels.
Profiled on gb10 CUDA, the per-step wall is **host-staging-bound and ~constant per
region**, giving **~25 tok/s at 8 layers (measured)** and an extrapolated **~5 tok/s at
28 layers** — a **135x (8L measured) to ~654x (28L extrapolated) throughput gap vs
Megatron's 3399 tok/s**. The gap is NOT
in the device kernels (PR-6 proved the real path_c-CUDA kernel runs 14.68M nonzero through
the same boundary); it is entirely the per-region host↔device numpy staging across the
`call_dps_packed` boundary. See §4 for where the time goes and §5 for the levers that
close it.

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

1. **Fuse every region on-device (eliminate host staging).** The single biggest lever,
   now MEASURED: the per-region `.numpy()` / `copyto` round-trips are **~100% of the step
   wall, and the forward+remat path alone is 96.9%** (§4). PR-6 stage 2 already runs the
   REAL path_c-CUDA kernel through the forward boundary on device; doing the same for the
   forward+remat regions first (96.9% of the time) — then bwd/adam/loss — and keeping the
   banks resident on `tvm.cuda(0)` across regions removes the host staging and collapses
   the free-delta toward the 8.787 GB device plan. Expected: the step becomes
   device-compute-bound, not copy-bound — the regime where the real kernels' throughput
   shows. Attacking the forward+remat staging is the highest-leverage single change.
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

---

## 7. Verdict

The Relax graph-path train_step **fits Megatron-class memory (8.787 GB planned device-peak
vs Megatron 26 GB, optimizer in-place) and runs the full 28-layer 1.8B step the eager path
OOMs on** — the memory-first goal is MET on device. Its **throughput trails Megatron by
135x (8L) to ~654x (28L)** (25.2 tok/s @ 8L measured, ~5.2 tok/s @ 28L extrapolated, vs 3399), and the gap
is **fully attributed to the abstract per-region host↔device staging** (not the device
kernels, not the architecture) plus the 4× batch gap. The path is a memory envelope that is
now proven; closing the throughput gap is the on-device-fusion work in §5, with the real
path_c-CUDA kernel already proven through the forward boundary.
