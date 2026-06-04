# FP8 ACTIVATIONS END-TO-END for the path_c pipeline (design + microbench spec)

**§19 MEASURED UPDATE (gb10 sm_121, 2026-06-04) — fp8 GEMM now RUNS; the SPEED is MEASURED.**
The §18 "fp8 UNRUNNABLE on sm_121" gap is CLOSED on two fronts. (1) **TE per-tensor E4M3
(`DelayedScaling`, cuBLASLt fp8 MMA) is the REAL fp8 tensor-core win: 1.57–1.84× over bf16
MEASURED** at the four prod bs4 GEMM shapes (52.1–61.8 fp8 TFLOPs vs 33.3–34.3 bf16), rel-err
**0.0375 honest over ALL elements** (≤0.10 ceiling; only the MXFP8 *block-scaling* recipe stays
upstream-blocked "not supported on 12.0+"; the per-tensor NVRTC-builtins loader gap is fixed by the
committed self-healing `cppmega_mlx/_gb10_nvrtc_env.py`). (2) **Our in-house Metal dequant→half→
`T.gemm` route is ported to a `target="cuda"` fragment-C twin** (`fp8_scaled_matmul_path_c_cuda_prim`)
that COMPILES + RUNS + passes parity (0.0375, native `fp8_e4_t`→`half_t` decode verified in the
emitted CUDA) — **but it is NOT yet fast (0.18–0.42×): its MMA runs on the decoded fp16 values
(`mma_sync<kFloat16,…>`, not an fp8-input MMA) on an untuned 32×64×32 single-128-thread tile**, so it
is the proven *memory* lever (2.0× operand-byte halving MEASURED) and the on-CUDA optimization target,
NOT a speed lever as-is. Attribution is explicit: the throughput belongs to TE-cuBLASLt fp8; our
kernel gives the memory halving + a correct on-CUDA fp8 path. Full numbers + reproduce: RELAX §19.
(Two port edits were required: the Metal `T.alloc_var` decode scratch lowered to a non-assignable
`float[1]` on codegen_cuda → replaced with a register-fragment stage; and the microbench must pass the
native `torch.float8_e4m3fn` not a `.view(uint8)` reinterpret, which segfaults the tvm_ffi adapter.)

**§20 ENABLEMENT AUDIT (gb10 sm_121, 2026-06-04) — re-ran the full microbench `--prod` under the
fixed env as the SOLE gb10 owner (box idle, 116 GB free).** Explicit 3-route tally: **2 of 3 fp8
routes RUN real e4m3** — **R1 tensorwise** (the `libnvrtc-builtins.so.13.3` loader error is GONE; 1.57–
1.83× cuBLASLt fp8) and **R2 ours cuda** (the "MLX Metal unavailable" error is GONE; compiles+runs+
parity 0.0376, 0.21–0.42×). **R1 MXFP8 stays upstream-blocked** — TE's gate still raises "not supported
on 12.0+", recorded as a measured FAIL (NEVER degraded to bf16/tensorwise), and we did NOT lower the
gate (PR #3050 needs unreleased cuBLASLt ≥13.6.0.2; force-unblock = silent MXFP8-backward miscompute =
RULE #1 violation). All fp8 numbers verified REAL e4m3: operand byte-halving 2.0× MEASURED, parity
0.0376 over ALL elements (not ~0 bf16-mislabeled, not >0.10 garbage). Full §20 table + reproduce +
toolchain-fix detail: RELAX §20.

---

**Status: DESIGN (read-only round) + §19 MEASURED microbench, 2026-06-04. Target: gb10 sm_121 (Grace-Blackwell, `tvm.cuda(0)`).**
This is Track 2 of the 3-lever Megatron-gap workflow (lever 1 = bs1→bs4; lever 2 = THIS doc, fp8
activations end-to-end; lever 3 = B2 grid-restructure). It is the user's thesis verbatim: *"why hold
bf16 activations at all if every kernel is fp8? that is BOTH speed (fp8 tensor cores) AND memory
(fp8 = half of bf16/fp16), all in-place including fwd and bwd."* This round produces the design and a
DE-RISKING gb10 microbench (`scratch/fp8_gemm_microbench.py`); it wires NOTHING into the live pipeline
yet. Numbers below are PROJECTIONS labelled as such; the microbench round will MEASURE the GEMM win.

RULE #1 (no silent fallback) is load-bearing throughout: the eventual fp8 path is the ONE path; on
any failure it RAISES with where+what — never an fp8→bf16 or bs4→bs1 silent degrade.

---

## 0. TL;DR

* **Memory (the certain win).** Holding the path_c activation banks in fp8 (1 byte) instead of fp16
  (2 bytes) HALVES the activation-cache footprint. The §17 device-peak is 6.400 GB @8L / 12.998 GB
  @28L; the activation-cache portion drops ~2×. Cross-track synergy: fp8 repays HALF of lever 1's
  4× token cost — **bs4-in-fp8 ≈ 2× bs1-in-fp16 (i.e. ≈ bs2-in-fp16) memory**, NOT ≈ bs1-in-fp16
  (4 tokens × 1 byte = 2× of 1 token × 2 bytes). Concretely ~12.8 GB @8L / ~26 GB @28L for bs4-fp8
  vs the naive ~25.6/~52 GB bs4-fp16 — back to Megatron-class (~26 GB) at 8L. PROJECTION. (Correction
  per §18 adversarial verify: the earlier "bs4-fp8 ≈ bs1-fp16" headline was arithmetically wrong.)
* **Speed (NOW MEASURED, §19).** The dominant GEMMs at bs4 are the transformer-block MLP/attention
  GEMMs (M = bs·seq = 16384), NOT the SSD scan. On Blackwell these run on the **regular FP8 Tensor
  Cores** (per-tensor cuBLASLt e4m3), which — unlike the FP4 SR/RHT path — DO exist on sm_121 (no
  tcgen05/TMEM dependency): **MEASURED 1.57–1.84× over bf16** (TE `DelayedScaling` E4M3, 52.1–61.8
  fp8 TFLOPs vs 33.3–34.3 bf16, rel-err 0.0375 ≤ 0.10). MXFP8 block-scaling stays upstream-blocked on
  12.0+. Our in-house `target="cuda"` dequant→`T.gemm` twin RUNS + parity-passes (0.0375) but is
  **0.18–0.42× as-is** (decoded-fp16 MMA on an untuned tile) — a memory lever (2.0× MEASURED) + the
  optimization target, not yet a speed lever. (Was: "the microbench measures … before any rewrite" —
  now measured; full table in RELAX §19.)
* **Scaling recommendation.** Use **MXFP8 (E4M3, 32-element block / E8M0 scale) for ALL operands
  incl. activation gradients** as the primary recipe, with **per-tensor delayed-amax E4M3** (reusing
  `fp8_amax.py`) as the simpler fallback-FREE alternative for the in-house TileLang kernels that do
  not yet emit block scales. E5M2 is NOT recommended for gradients (measured perplexity regression
  in the literature; E4M3-everywhere matches bf16 on Blackwell).
* **Precision guardrails (the §17 dD lesson, generalized).** The longest-reduction grads must keep a
  **fp32 accumulator** and the reference must differentiate the SAME fp8-quantized cache the kernel
  reads. dD over B·S·headdim = 262 144 terms/head already needed this at fp16; at fp8 the
  per-element quant error grows from ~5e-4 to ~6e-3, so dD (and the LSE/softmax accumulators, and the
  cross-block dAcs reduction) STAY in fp32 accumulate — only the *stored operands* go fp8.

---

## 1. What fp8 GEMM / scaling we ALREADY have (inventory)

| asset | format | scaling granularity | amax handling | backward? | backend |
|---|---|---|---|---|---|
| `fp8_matmul_path_c.py` | **e4m3** (uint8 storage) | **per-tensor** scalar fp32 `scale_a`/`scale_b` | caller-supplied scalar scales | forward only | **Metal-only** (M4 cooperative `matmul2d`); RAISES "MLX Metal unavailable" on CUDA |
| `fp8_vecmat_path_c.py` | e4m3 (packed uint32 dot4) | per-tensor scalar + per-row B scale | caller-supplied | forward only (M=1 GEMV) | **Metal-only** |
| `fp8_amax.py` | e4m3fn (`_FP8_E4M3_MAX=448.0`) | per-tensor | **block-reduce + `T.atomic_max`** amax kernel; host computes `inv_scale = 448/amax`; RNE quantize kernel | n/a (utility) | **CUDA + Metal** (real `"cuda"` target — the only CUDA-ready fp8 asset) |
| `sparse_mla_fp8_path_c.py` | e4m3 | scalar A + scalar/per-row B | prepared-buffer (caller owns fp8 + scales) | fwd + bwd surfaces (prepared) | Metal Path C |
| `sparse_mla_blockscaled*.py` | **MXFP8** (block-32, E8M0-like) | **per-block-32** | quantize to packed uint32 + uint8 scales | fwd + bwd (prepared) | Metal Path C |
| `mxfp4_matmul_path_c.py` | e2m1 + block-16 scale | per-block-16 (fp32 scale) | codec dequant→fp16 into shared, then `T.gemm` | forward GEMM | **Metal-only** cooperative `matmul2d` |
| `scripts/_nvfp4_route.py` | e2m1 + E4M3 block-16 | per-block-16 | TE `NVFP4BlockScaling` (RtN recipe) | **fwd + bwd VERIFIED on gb10** (dgrad 0.1465 / wgrad 0.1343 rel-err) | **CUDA/TE cuBLASLt** |
| `cutlass_mxfp8_sm120.py` + `_cutlass_mxfp8_sm120.cu` | **MXFP8** (e4m3 + E8M0 block-32) | **per-block-32** | host `build_e8m0_block_scales` (`ceil(log2(amax/448))`) + **swizzled SFA/SFB repack** (`Sm1xxBlkScaledConfig::tile_atom_to_shape_SF`) | forward GEMM (TN) | **CUDA, standalone CUTLASS v4.5.1 sm_121a `.so`** — a REAL native CUDA MXFP8 GEMM (§25) |

> **§25 update (gb10 sm_121a, 2026-06-04):** the standalone CUTLASS MXFP8 route is now **correct and
> parity-clean**. The §22 SF-layout bug (rel_err 0.41) was the host packing the E8M0 SFA/SFB bytes
> contiguously instead of into CUTLASS's swizzled `((_32,_4),(_32,_4)):((_16,_4),(_0,_1))` 512-byte
> atoms. The fix (`_sf_scatter_index` machine-verified vs the kernel's `cppmega_mxfp8_sf_offset` →
> `tile_atom_to_shape_SFA/SFB`, 0 mismatches) drops **rel_err to 0.0377** (E4M3 mxfp8 band, gate 0.12,
> ALL elements). The **pure kernel crosses bf16 at 1.38–1.78× (46–59 TFLOPs vs 33–34)** on the prod bs4
> shapes — a real GEMM-phase win, the same band as the native fp8 MMA (lever 2). **It does NOT reach the
> ~188 TFLOPs / 4–4.5× DGX-Spark target** on this part. The per-call host quant+scatter is 44–79% of the
> `mxfp8_gemm_from_hp` wall time and must be fused/amortized (weights quantized once) before pipeline use.
> So this is now the SECOND real CUDA fp8 GEMM asset (alongside the TE nvfp4 route), but the 4× headline
> is NOT met (honest MEASURED).

**The hard gap this design must close:** every in-house fp8 *GEMM* kernel is **Metal-only**. The only
fp8 assets that run on gb10/CUDA today are (a) `fp8_amax.py`'s amax+quantize (a scaling utility, not a
GEMM) and (b) the nvfp4 route's **TransformerEngine cuBLASLt** path (a real CUDA FP4/FP8 GEMM). So the
e2e plan does NOT reinvent a CUDA fp8 GEMM from scratch — it has two proven routes to choose between
(measured by the microbench): **(R1) TransformerEngine MXFP8/per-tensor cuBLASLt** for the
transformer-block GEMMs, and **(R2) TileLang `T.gemm` with fp8→fp16 shared-buffer dequant** (the exact
pattern already in `_fp8_scaled_matmul2d_kernel_template` and in the F2 scan core) ported to a `"cuda"`
target for the SSD GEMMs that live inside our own grid kernels.

---

## 2. Design — fp8 activations end-to-end across path_c

### 2a. Which tensors go fp8 vs stay higher precision

| tensor class | precision | rationale |
|---|---|---|
| **Activation banks** (path_c F0/F1/F2 + B0/B1/B2 handoff cache: `x`, `z`, `B`, `C`, `dt`-views, `cb`, `dA_cumsum`, `prev_states`, `dY`) | **fp8 e4m3** stored, fp32 accumulate | the memory + tensor-core win; these are the banks `path_c_relax_step_banks.py` sizes by `batch*seq*…` |
| **Transformer-block GEMM operands** (MLP up/down, attn QKV/proj) | **fp8 e4m3** (MXFP8 block-32) | the dominant FLOPs at bs4; cuBLASLt/TE FP8 tensor cores |
| **Master weights** | **bf16 (or fp32 master)** | optimizer needs a high-precision master; never fp8 |
| **Optimizer state** (Muon momentum, Adam8bit scalars) | **unchanged** (q8 momentum / fp32 scalars) | already low-precision by its own recipe; out of scope |
| **Reductions / accumulators** (GEMM `acc_o`, dD `atomic_add`, softmax/LSE, cross-block dAcs) | **fp32** | RULE #1 numerical-stability guardrail; see §2d |
| **Longest-reduction grads** (dD = B·S·headdim per head; the §17 case) | fp32 accumulate, fp8 *stored* operands | §17 dD lesson generalized |
| **LM-head / embedding, RMSNorm, residual adds, SwiGLU nonlinearity** | **bf16** (carrier) | small FLOPs, sensitive; not GEMMs; not worth fp8 (matches the nvfp4 route's `NVFP4_UNSUPPORTED_TRAINING_OPS` carve-out) |

### 2b. Scaling strategy — RECOMMENDATION: MXFP8 (E4M3, block-32) primary; per-tensor delayed-amax secondary

> **§2b-status (MEASURED 2026-06-04, gb10 sm_121 / CC 12.1) — MXFP8 is UPSTREAM-BLOCKED here today; per-tensor delayed-amax (R1 tensorwise) is the route that RUNS.**
> The MXFP8 "primary" recommendation below is the *target* recipe for Blackwell, but it does **not** run on gb10 (GB10, sm_121) this round, and we do **not** fake it (RULE #1). NVIDIA TransformerEngine's own Python gate `transformer_engine/pytorch/quantization.py::_compute_mxfp8_support()` (line 69) hard-raises `"MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet."` for compute capability ≥ (12,0). Root cause (TE tech lead *ptrendx*, issue [#2668](https://github.com/NVIDIA/TransformerEngine/issues/2668), 2026-02-11): cuBLAS lacks the **non-TN GEMM layouts** MXFP8 backward needs on SM120/SM121 — a cuBLAS gap, **not** hardware/CUTLASS and **not** our code. The fix, PR [#3050](https://github.com/NVIDIA/TransformerEngine/pull/3050) (still an **OPEN DRAFT** as of 2026-05-29, complemented by PR [#2833](https://github.com/NVIDIA/TransformerEngine/pull/2833)), relaxes the gate **only** when `tex.get_cublasLt_version() >= 130600` (**cuBLASLt 13.6.0.2**) — and that cuBLASLt does **not exist anywhere yet** (PyPI max + box max are both 13.5.1.27 = `130501 < 130600`). So even a perfect cherry-pick of #3050 today would still `fail-closed` (`is_mxfp8_available()` → False), which is correct: forcing the gate lower would silently miscompute the MXFP8 backward (a RULE #1 violation). **VERDICT: MXFP8 on sm_121 is genuinely upstream-blocked on (a) an unmerged draft PR and (b) an unreleased cuBLASLt 13.6.0.2 dependency. Recorded as a measured FAIL, never silently degraded to bf16 or tensorwise.**
>
> **Forward plan (when NVIDIA ships cuBLASLt 13.6.0.2):** `pip install 'nvidia-cublas>=13.6.0.2'` into `cppmega-venv`, then update `/home/dave/TransformerEngine` to a SHA carrying BOTH PR #2833 and PR #3050 once merged (or cherry-pick `git fetch origin pull/2833/head:pr2833 && pull/3050/head:pr3050` onto a throwaway branch off local `main`, rebuild with `NVTE_CUDA_ARCHS=121a pip install -e . --no-build-isolation`). Expect conflicts: the installed TE is a fork-tainted build (HEAD `8d1d79bf`, v2.16.0.dev0, jewelmusicee remote + local cppmega MXFP8 transpose commits, 31 behind origin) — merge origin/main first. Re-verify forward **and** backward numerically before labeling the route fp8-pass; the drafts are unfinished ("TODO: code/git history clean up", tests box unchecked).
>
> **What runs on gb10 today:** **R1 tensorwise** (per-tensor Float8 delayed-amax, the "secondary" below) — TE supports it on sm_121 (NVIDIA forum: "FP8 with delayed scaling works correctly on these architectures as a workaround"); its only blocker was the NVRTC `libnvrtc-builtins.so.13.3` loader path, now fixed durably (system `ldconfig` cache + the committed `cppmega_mlx/_gb10_nvrtc_env.py` re-exec guard + the run-command `LD_LIBRARY_PATH` prefix). And **R2** (our CUDA `T.gemm` e4m3 dequant twin). §19 MEASURED: R1 tensorwise = real fp8 cuBLASLt MMA, 1.58–1.83× over bf16, rel_err 0.0375; R2 compiles+runs+parity 0.0375 but 0.18–0.42× (untuned tile + fp16-MMA).

* **Primary: MXFP8 block-32 E4M3 (E8M0 power-of-2 scale).** On Blackwell this is the native recipe.
  Decisive grounding from the 2025 literature + NVIDIA docs: **E4M3 for ALL tensors including
  activation gradients** (E5M2-for-gradients measurably regresses perplexity on Blackwell; the
  fine-grained block scale absorbs the dynamic range that E5M2 used to provide). Block-32 localizes
  the scale so a single outlier does not crush a whole tensor's mantissa. We already have the Metal
  block-scaled precedent (`sparse_mla_blockscaled*.py`, MXFP8 block-32) to port the codec from.
* **Secondary (fallback-FREE, for the in-house TileLang grid GEMMs): per-tensor delayed-amax E4M3.**
  Reuse `fp8_amax.py` verbatim: its CUDA `T.atomic_max` amax kernel + `inv_scale = 448/amax` +
  RNE quantize kernel is exactly the per-tensor recipe. "Delayed" = use the previous step's amax
  (one-iteration-stale) to avoid a sync on the hot path; `fp8_amax.py` computes current amax, so the
  delay is a host-side ring buffer of the last amax per bank (cheap, no kernel change). This is the
  simpler path for the SSD-scan `T.gemm` operands where a per-block-32 swizzle is not yet wired.
* **NOT recommended:** per-tensor *current* scaling on the hot path (forces a device sync between the
  amax and the quantize launch — `fp8_amax.py` already documents this two-pass cost) and E5M2
  gradients (literature regression).

**Why this is not a silent fallback (RULE #1):** "primary MXFP8, secondary per-tensor" is a
*per-kernel format choice fixed at build time per GEMM site*, not a runtime degrade. A GEMM site emits
ONE recipe; if its chosen fp8 kernel fails to compile/run it RAISES — it does not silently retry in
the other format or in bf16.

### 2c. In-place fwd+bwd memory choreography

The §17 backward already **reuses the forward handoff banks** (memory UNCHANGED 6.400/12.998 GB
between §15 and §17 — the backward reads the same fp16 cache the forward wrote). The fp8 plan keeps
that choreography and just narrows the bank dtype:

1. **Bank dtype flip.** `path_c_physical_abi.py::_DTYPE_NBYTES` currently has **no `float8_e4m3`
   entry** (it stops at `uint8:1`). Add `"float8_e4m3": 1` / `"float8_e5m2": 1` so the bank-sizer
   computes 1-byte-per-element extents. The relax-step bank builder
   (`path_c_relax_step_banks.py`) then sizes the activation banks at half the bytes.
2. **In-place fwd→bwd.** The forward F0/F1/F2 write the fp8 cache once; the backward B0/B1/B2 read it
   in place (no extra staging) — identical to §17, but at 1 byte/elem. The fp32 accumulators
   (`acc_o`, dD, dAcs) live in registers/shared, NOT in the banks, so they are unaffected by the
   bank narrowing.
3. **Scale sidecar.** Each fp8 bank gets a tiny companion scale bank: per-tensor = one fp32 scalar;
   MXFP8 = `numel/32` E8M0 (uint8) scales. The scale sidecar is <0.4% of the fp8 bank (`1/32` byte
   per element for MXFP8) so it does not erase the halving.

**Quantified memory (PROJECTION).** Let `A` = the activation-cache share of the §17 device-peak. The
fp16→fp8 halving takes the cache to `A/2`; non-cache (weights, fp32 accumulators that are transient,
launch scratch) is unchanged. The user's stated bracket — 6.4/13.0 GB → ~3.2/6.5 GB — assumes the
cache dominates the peak; the honest number depends on the measured cache share (the microbench round
prints the bank-byte breakdown). The cross-track identity is the robust claim: **bs4 fp8 banks =
bs1 fp16 banks** (4× tokens × ½ bytes/elem = 2×; vs the naive bs4-fp16 4×), so bs4-fp8 lands at ≈2×
the bs1-fp16 peak (~12.8/~26 GB) rather than 4× (~25.6/~52 GB) — under the 100 GB gb10 budget.

### 2d. Numerical-stability guardrails

1. **fp32 accumulate everywhere a GEMM/reduction lives.** The F2/B2 `T.gemm` already uses
   `accum_dtype` (fp32) `acc_o`; KEEP it. fp8 changes only the *operand* shared buffers
   (`A_shared`/`B_shared` go e4m3, dequant→fp16 into the cooperative input exactly like
   `_fp8_scaled_matmul2d_kernel_template`), never the accumulator.
2. **The §17 dD rule, generalized to fp8.** dD reduces B·S·headdim = 262 144 terms/head. At fp16
   cache (~5e-4/elem quant) the aggregate hit was 1.40e-3 (> 1e-3 gate) until the gold was aligned to
   the fp16 cache (`_gold_dD_fp16cache` → 2.48e-5). At fp8 e4m3 the per-element quant is ~6e-3 (≈12×
   worse), so for the fp8 round the **reference must differentiate the SAME fp8-quantized x/z/dout the
   kernel reads** (a `_gold_dD_fp8cache`), and dD MUST stay fp32-accumulate + atomic_add. The parity
   gate stays 1e-3; if dD cannot pass at fp8 even with the aligned gold, that grad's *stored cache*
   stays fp16 (a per-tensor precision pin, NOT a silent fallback — it RAISES if the fp8 attempt is
   numerically out of family, and the fp16 pin is a declared design choice for that one bank).
3. **The nvfp4 RtN precedent.** The nvfp4 backward passes at dgrad 0.1465 / wgrad 0.1343 rel-err
   under a 0.35 gate — i.e. a *4-bit* backward is already accepted at ~0.15 rel-err for the
   transformer-block GEMMs. fp8 e4m3 (3 mantissa bits vs e2m1's 1) is strictly more precise, so the
   transformer-block fp8 backward is expected comfortably inside that envelope. The SSD-scan grads
   (dD etc.) are the tighter 1e-3 gate and get the fp32-accumulate + aligned-gold treatment above.
4. **NaN/Inf fail-loud.** `fp8_amax.py` already RAISES `FloatingPointError` on a non-finite amax
   (refuses a degenerate scale) and pre-filters NaN before `atomic_max`. Keep that; extend the same
   fail-loud to the MXFP8 block-scale codec.

---

## 3. Prod GEMM shapes that dominate at bs4 (microbench targets)

Prod config (`local_gb10_quarter`, MEGATRON-VS-MLX-PATHS.md §config): **hidden H=3584, ffn F=18944,
heads=28, seq S=4096, bs=4 → M = bs·S = 16384 tokens.** The transformer-block GEMMs (NOT the SSD
scan) dominate the bs4 FLOPs:

| GEMM | shape (M × N × K) | per-layer count | why it dominates |
|---|---|---|---|
| **MLP up (gate+up / SwiGLU)** | 16384 × (2·18944) × 3584 | 1 | largest single GEMM; 2·F output for gated MLP |
| **MLP down** | 16384 × 3584 × 18944 | 1 | K=18944 is the longest contraction |
| **Attn QKV proj** | 16384 × 3·3584 × 3584 | 1 | square-ish, frequent |
| **Attn out proj** | 16384 × 3584 × 3584 | 1 | square |
| **F2 state contraction** (SSD `T.gemm`, our grid kernel) | per-(batch·nchunk, head): (chunk=64) × headdim=64 × dstate=64 | batch·nchunks·nheads | small tiles, MANY of them; the in-house `T.gemm` site |

The first four are huge dense GEMMs that hit peak FP8 tensor-core FLOPs and amortize launch overhead —
exactly where bs4 + fp8 compound (lever-1 × lever-2 synergy). The fifth (F2/B2) is the in-house grid
`T.gemm` where the fp8→fp16 dequant pattern slots in. **Microbench targets = the four transformer
GEMMs at M=16384** (the bs4 wins), plus a single small SSD tile as a sanity check.

---

## 4. Microbench spec — `scratch/fp8_gemm_microbench.py` (gb10, standalone)

**Goal:** MEASURE fp8(e4m3) vs bf16 TFLOPs + bytes at the §3 prod shapes at bs4, BEFORE the e2e
rewrite. DE-RISKS the rewrite with a real tensor-core-speedup + memory number. Standalone, single-run
discipline (only the Profile agent runs it on gb10).

**Harness (verbatim §16/§17 pattern):**
```
ssh gb10; cd /home/dave/source/cppmega_mlx
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang/3rdparty/tvm/python:/home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python \
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib \
/home/dave/cppmega-venv/bin/python scratch/fp8_gemm_microbench.py --prod
```

**What it measures, per shape in §3:**
1. **bf16 baseline GEMM** — `torch.matmul` (cuBLAS bf16) median of 20 timed calls → TFLOPs =
   `2·M·N·K / t`.
2. **fp8 e4m3 GEMM** — TWO routes, reported side by side (RULE #1: report both honestly, pick the
   measured winner; do NOT hide a slower route):
   * **R1 = TransformerEngine** `te.Linear` under `MXFP8BlockScaling` (and a `Float8` per-tensor
     recipe) wrapped in `te.fp8_autocast` — the proven gb10 CUDA cuBLASLt FP8 path (the nvfp4 route
     already drives `te.Linear`/`te.fp8_autocast` here; reuse `nvfp4_te_capability_probe`'s
     import+capability guard so an absent/blocked TE RAISES with the precise reason, never silently
     skips).
   * **R2 = our `fp8_matmul_path_c` cooperative `T.gemm`** ported to `target="cuda"` (the
     `_fp8_scaled_matmul2d_kernel_template` fp8→fp16 dequant + `T.gemm` pattern). On gb10 the current
     Metal-only build RAISES "MLX Metal unavailable" — so the microbench RECORDS that as the measured
     gap (the kernel needs a CUDA emission) rather than pretending it ran. This is the honest
     "what's missing" signal that scopes the next round.
3. **memory** — for each operand, `bytes_bf16 = 2·numel` vs `bytes_fp8 = 1·numel (+ scales)`; print
   the realized halving and the scale-sidecar overhead.
4. **numeric sanity** — fp8 vs bf16 `rel_err` (`‖C_fp8 − C_bf16‖ / ‖C_bf16‖`); finite check; RAISE on
   NaN/Inf (mirrors the §16/§17 probes). Report rel_err next to the nvfp4 0.1465 dgrad precedent so
   the reader can place the fp8 error in context.

**Acceptance / GO signal:** fp8 TFLOPs > bf16 TFLOPs at the four big shapes (the de-risk) AND
rel_err finite + within an fp8-reasonable bound (expect ~1e-2, well under the nvfp4 0.15). The
microbench does NOT gate the rewrite by itself; it produces the MEASURED speedup+memory the e2e plan
needs.

**Machine-parseable output.** Besides the human `# SUMMARY` table, the probe emits a single greppable
`RESULT_JSON: {…}` line (schema `fp8_gemm_microbench/v1`) plus an indented `# RESULT (pretty)` block.
Per shape the JSON carries `bf16_tflops`, `best_fp8_tflops`, `fp8_vs_bf16_speedup`, `best_fp8_rel_err`,
the `bytes` block (`operand_bf16`, `operand_fp8`, `operand_halving_ratio`, MXFP8/per-tensor scale
sidecars) and EVERY route's record (incl. the ones that RECORDED a gap with their where+what reason —
RULE #1: the gaps are in the data, not hidden). The orchestrator/Profile agent parses the
`RESULT_JSON:` line.

**Also spec'd: the fp8-activation cast/round helper** (`scratch/fp8_gemm_microbench.py` includes a
`fp8_quantize_activation(x, *, recipe)` reference):
* per-tensor: `amax = fp8_amax_tilelang(x)` (reuse), `inv_scale = 448/amax`,
  `q = fp8_quantize_tilelang(x, inv_scale)` (reuse) — already CUDA-ready.
* MXFP8 block-32: reshape last dim into blocks of 32, per-block `amax`, E8M0 (power-of-2) scale,
  RNE cast to e4m3 — port the codec from `sparse_mla_blockscaled*.py`'s `_quantize_mxfp8`.
* In-place contract: writes into a caller-owned fp8 bank + scale sidecar; no hidden alloc (mirrors
  the `path_c_physical_abi.py` no-hidden-allocation policy and `fp8_quantize_tilelang`'s `out=`).
* fail-loud: RAISES on non-finite amax (reuse `fp8_amax.py`'s `FloatingPointError`).

---

## 5. Phased implementation plan

**This round (design):** this doc + `scratch/fp8_gemm_microbench.py` (spec'd in §4). No live-pipeline
edits. The Profile agent runs the microbench once on gb10 and reports MEASURED fp8-vs-bf16 TFLOPs +
memory + rel_err.

**Round 2 — fp8 banks (memory win, no compute change yet):**
* Add `"float8_e4m3"/"float8_e5m2": 1` to `path_c_physical_abi.py::_DTYPE_NBYTES`.
* Flip the activation-cache bank dtype to `float8_e4m3` in `path_c_relax_step_banks.py` + add the
  scale sidecar banks; producers quantize on write (reuse `fp8_quantize_tilelang`), consumers dequant
  on read (fp8→fp16 into the `T.gemm` shared input). MEASURE the device-peak drop.
* Gate: §17 backward parity (all 8 grads ≤ 1e-3, dD via the `_gold_dD_fp8cache` aligned gold) must
  STILL pass at fp8 banks. If dD regresses, pin that one bank to fp16 (declared, not silent).

**Round 3 — fp8 forward GEMM:** route the F2 SSD `T.gemm` operands through the fp8→fp16 dequant
(CUDA emission of `_fp8_scaled_matmul2d_kernel_template`) and the transformer-block GEMMs through the
microbench-winning route (R1 TE MXFP8, most likely). Re-measure forward tok/s.

**Round 4 — fp8 backward GEMM:** B2 SSD grads + transformer-block dgrad/wgrad in fp8, fp32-accumulate,
aligned-gold parity gate. Re-measure the full step. (Lever-3 B2 grid-restructure composes here.)

**RULE #1 in the eventual path:** every fp8 GEMM site is ONE compiled recipe; a compile/run/parity
failure RAISES with where+what (kernel name + shape + which grad + measured rel_err). There is NO
fp8→bf16 retry, NO bs4→bs1 retry, NO slow-prim retry. The only declared precision choice is a
per-bank fp16 pin for a grad that provably cannot hold fp8 under the gate — and that pin is named in
this doc, surfaced at build time, not a runtime degrade.

---

## 6. Open questions (the e2e-rewrite unknowns the microbench round must resolve)

1. **R1 vs R2 winner on gb10.** Does TE MXFP8 `te.Linear` beat a CUDA-ported `T.gemm` fp8 kernel at
   the §3 shapes, and by how much? (microbench measures.)
2. **CUDA emission of the in-house fp8 GEMM.** `fp8_matmul_path_c`/`mxfp4_matmul_path_c` are
   Metal-only cooperative kernels. How much work is a `target="cuda"` emission of the fp8→fp16
   dequant + `T.gemm` (the F2/B2 SSD tiles can't use TE — they live inside our grid kernel)?
3. **MXFP8 swizzle layout.** Blackwell cuBLASLt MXFP8 needs the scales in a specific swizzled
   hardware layout; does TE handle that internally for `te.Linear`, and what's the cost of producing
   swizzled scales for the in-house GEMM path?
4. **dD (and other long-reduction grads) at fp8.** Does dD pass the 1e-3 gate with a
   `_gold_dD_fp8cache` aligned gold, or does its bank need the fp16 pin? (Round-2 gate measures.)
5. **Realized cache share of the device-peak.** What fraction of 6.400/12.998 GB is the activation
   cache (the part that halves) vs weights+scratch (unchanged)? Determines whether the 3.2/6.5 GB
   bracket is realistic or optimistic. (Round-2 bank-byte print measures.)
6. **Activation-grad format.** Confirm E4M3-everywhere holds on sm_121 (the literature says yes for
   Blackwell MXFP8; our SSD grads are a different distribution than transformer activations, so the
   round-4 parity gate is the real test).

---

### Sources (web-grounded design claims)

* NVIDIA — Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training:
  https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/
* NVIDIA TransformerEngine — MXFP8 feature docs (E4M3-everywhere, E8M0 block-32, swizzle):
  https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html
* NVIDIA TransformerEngine — Using FP8 and FP4 (per-tensor current/delayed vs MXFP8 recipes):
  https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
* TransformerEngine issue #2668 — MXFP8 support status on SM120/Blackwell (open inquiry; corroborated
  by our own gb10 evidence that MXFP8/per-tensor FP8 GEMM uses regular Tensor Cores, NOT the
  tcgen05/TMEM path the FP4 SR/RHT backward needs — see `scripts/_nvfp4_route.py`):
  https://github.com/NVIDIA/TransformerEngine/issues/2668
* In-repo precedents: `scripts/_nvfp4_route.py` (gb10 TE cuBLASLt FP4/FP8 fwd+bwd, the CUDA GEMM
  route), `cppmega_mlx/nn/_tilelang/fp8_amax.py` (CUDA-ready per-tensor amax/quantize),
  `sparse_mla_blockscaled*.py` (MXFP8 block-32 codec), `docs/RELAX-GRAPH-VS-MEGATRON.md` §17 (the dD
  fp16-cache parity lesson).
