# NVFP4 BACKWARD on GB10 / sm_121 — the MIXED non-standard approach (DECISION + plan)

**Scope:** Decide whether a *mixed, non-standard* NVFP4 backward path is feasible on
GB10 (sm_121) using the user's specific recipe — **HW round-to-nearest (RtN) e2m1
quant + the working block-scaled FP4 GEMM for dgrad/wgrad, deliberately SKIPPING the
datacenter-only RHT (Random-Hadamard fused quant) and SR (stochastic-rounding
`cvt.rs` cast)** — and give a concrete, ranked implementation plan to REPLACE the
current `scripts/_nvfp4_route.py` `Nvfp4CudaKernelMiscompiled` backward raise with a
real, runnable backward.

This doc synthesizes four field reports (A=Brave/CUTLASS-layouts, B=Exa/accuracy,
C=Perplexity/HW-cvt, D=Tavily/exact-APIs) **plus** two pre-existing in-repo receipts
(`docs/NVFP4-NANOCHAT-RECIPE.md`, `docs/NVFP4-TRAINING-KERNELS.md`). Where a claim is
load-bearing it is cited to its source; nothing here is invented beyond those.

---

## 1. VERDICT (lead)

**FEASIBLE — YES, WITH CAVEATS. The user's mixed approach overturns the earlier
"datacenter-only, can't be done" conclusion in `_nvfp4_route.py`.**

The earlier conclusion was *correctly scoped but over-generalized*. It is true that
the **default** `NVFP4BlockScaling()` recipe's two pre-GEMM quantization kernels — the
SR FP4 cast (`cvt.rs.satfinite.e2m1x4`, TE-source-gated to `ArchSpecific<100>/<103>`
= sm_100a/sm_103a) and the RHT fused kernel (built on the SM100 TMEM/`tcgen05`
blockscaled pipeline absent on sm_12x) — are genuinely datacenter-only and **cannot**
be enabled by a TE rebuild on sm_121. That part of the existing analysis stands
(report C confirms `tcgen05` absent on sm_120/sm_121; report D confirms the SR `.rs`
cvt and the RHT shared-mem cap crash). **But those kernels are SEPARABLE from the
backward GEMM itself.** Every field report agrees the backward *matmul* is the
identical TN block-scaled FP4 GEMM as the forward (report A: "the backward matmuls are
the same kernel as your working forward… RHT/SR are *pre-GEMM quantization*, both
optional knobs"; report C/D: `disable_rht=True, disable_stochastic_rounding=True` is
the *sanctioned* SM120 workaround, TE issues #2372/#3062). Stripping RHT+SR leaves a
plain RtN E2M1 + UE4M3-block-scale GEMM — which our forward already proves runs on
gb10. **And we already have a local receipt that the full mixed path TRAINS on gb10:**
`docs/NVFP4-NANOCHAT-RECIPE.md` documents a real GB10 base-train run (`report.log`,
560 M params) under `NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True,
override_linear_precision=(False, False, True))` with **loss descending 11.09 → 6.58
over 22 steps** and `loss.backward()` succeeding in eager mode (`debug_nvfp4_crash.py:51`;
the only B=32 failure was OOM, not an arch crash). The cost of skipping RHT/SR is an
**accuracy/convergence** cost (report B quantifies it; see §4), not a runnability wall.
**Decision: build the mixed RtN backward. The current blanket backward raise must be
narrowed to fail loud ONLY when the default SR/RHT path is hit, not when the
RtN-no-RHT path is used.**

---

## 2. What the HW actually exposes on sm_121 (RtN-yes, SR-no, block-scale-yes)

From report C (Perplexity), corroborated by report D and the in-repo `_nvfp4_route.py`
source audit:

| Primitive | On sm_121? | Evidence |
|-----------|-----------|----------|
| **RtN e2m1 cast** (`cvt.rn.satfinite.e2m1x2` / `.e2m1x4`) | **YES** | PTX `cvt` to `.e2m1x2/.e2m1x4` (`.rn`, `.satfinite`) + dequant `e2m1x2.to.f16x2.rn`/`.bf16x2…ue8m0` are the SM120-family packed FP4 quant/dequant primitives (LLVM NVPTX intrinsic list; flashinfer SM121 audit lists FP4 among sm_12.x tensor-core dtypes). Report C. |
| **Block-scaled FP4 MMA** (`mma.sync…m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X…e2m1.e2m1.f32.ue4m3`) | **YES — via arch-specific (`a`) target, NOT tcgen05** | The hand-written PTX compiles cleanly on `sm_120a`/`sm_120f`/`sm_121a` (cutlass#3227, report A). flashinfer audit: NVFP4 MMA (`.kind::mxf4nvf4`) = "Yes" for `120a`/`121a`. Report C. **Caveat: `.a`-target only** — an AOT cubin for family `sm_120f` does NOT contain it; runtime remaps 121→120f on CUDA ≥12.9, silently losing native FP4 MMA. Compile arch-specific (`-arch=sm_121a` / `TORCH_CUDA_ARCH_LIST=12.1a`) or accept JIT-only. |
| **SR e2m1 cast** (`cvt.rs.satfinite.e2m1x4`, 32-bit rnd-bits operand) | **NO** | Introduced PTX ISA 8.7 / CUDA 12.8 for `sm_100a` (B200); LLVM intrinsic `llvm.nvvm.f32x4.to.e2m1x4.rs…`. TE source-gates it to `ArchSpecific<100>/<103>`. Report C/D. (Open citation, §7: the verbatim PTX-table cell stating `.rs` excludes `sm_120a`.) |
| **`tcgen05` / TMEM** (the datacenter 5th-gen blockscaled MMA + the RHT kernel's pipeline) | **NO** | PTXAS target audit: `sm_120/120a/120f: No`, `sm_121/121a/121f: No`; only `sm_100/sm_103` have it. This is why the **RHT** fused kernel (`row_cast_col_hadamard_transform_cast_fusion.cu`, CUTLASS SM100 TMEM pipeline) raises `CUDA error: invalid argument` on gb10. Report C; matches `_nvfp4_route.py` lines 50–58. |
| **RHT fused-quant kernel** | **NO** (shared-mem cap, not a missing FP4 unit) | TE #3062/#2372: `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` fails because the kernel is sized for sm_100's ~232 KB opt-in smem; sm_120/121 opt-in cap is **101376 bytes** (99 KB) → `cudaErrorInvalidValue`. Report C/D. |

**Net:** the two primitives the mixed approach NEEDS — RtN e2m1 cvt and the block-scaled
FP4 MMA — are present on sm_121. The two it deliberately SKIPS — SR cvt and the RHT
fused kernel — are the only datacenter-gated pieces. **The HW supports the mixed path.**

> **PREMISE TO VERIFY (report D, load-bearing):** `_nvfp4_route.py` and
> `NVFP4-TRAINING-KERNELS.md` both call the working forward "cuBLASLt FP4 GEMM." But
> cuBLASLt's NVFP4 path is officially scoped to **Compute Capability 10.x / 11.0**
> (CUDA 13.3 release notes), and multiple reports show `cublasLtMatmul` returns *no
> heuristic* for FP4 on sm_120/121. **TE's working sm_121 forward is therefore almost
> certainly cuDNN-frontend (block-scaled FP4 Gemm) or CUTLASS, not raw cuBLASLt.** This
> does not change the verdict (TE dispatches the right backend internally; nanochat's
> receipt proves fwd+bwd run), but it means **if we ever drop below TE to raw library
> calls, the reuse target is cuDNN/CUTLASS, not raw cuBLASLt.** Confirm with
> `CUBLASLT_LOG_LEVEL=5` / TE logs on gb10 before architecting a raw-library backward.

---

## 3. The backward as 2 GEMMs (dgrad, wgrad) — layout & transpose cost

For `y = x·Wᵀ` (a `te.Linear(K→N)`), the two backward GEMMs are:

- **dgrad** `dx = dy · W`  (contracts over N, the output-feature dim)
- **wgrad** `dW = dyᵀ · x` (contracts over M, the token/batch dim)

**The hard layout law: FP4 block-scaled GEMM is TN-ONLY on the entire Blackwell family
(sm_100 AND sm_120/121).** This is hardware-level, baked into the MMA instruction
(`…row.col…` = A row-major/K-major, B column-major/K-major). Sources (report A):
CUTLASS `sm120_blockscaled_mma_builder.inl` `static_assert("Only TN layout is
supported.")`; blackwell_functionality.html TN="Y", others "N"; cuBLASLt enforces
`transa=OP_T, transb=OP_N` exactly as FP8 does (report D, `LtFp8Matmul` "trans_a can
not be false").

**TN-only does NOT block the backward — it dictates HOW you quantize.** Because the MMA
only contracts over K with both operands K-major, *every* GEMM (fprop, dgrad, wgrad) is
already expressed as TN in real NVFP4 training. The standard solution (TE, vuiseng9) is
to keep **both a rowwise- and a columnwise-quantized copy** of each tensor and feed the
correctly-oriented copy into each GEMM — both operands block-quantized **along their
contraction (K) axis** with 1×16 UE4M3 scales (report A/B/C). Concretely:

| GEMM | Contraction K | Operands quantized along K | TN form | Repack cost |
|------|---------------|----------------------------|---------|-------------|
| dgrad `dy·W` | N (out-features) | `dy` columnwise, `W` columnwise | A=OP_T, B=OP_N | reuse W's **2D (16×16)** scale so dgrad sees the *same* weight rep as fprop (this is exactly what 2D weight scaling exists for — it makes rowwise and columnwise W numerically identical, report B §angle3). `dy` needs a 1×16-along-N quant. |
| wgrad `dyᵀ·x` | M (tokens) | `dy` rowwise, `x` rowwise | A=OP_T, B=OP_N | both `dy` and `x` need a 1×16-along-M (columnwise vs the fprop helper) quant — a **transposed-operand** quantization. |

**The real implementation risk is the transposed-operand quant + swizzle, not the MMA**
(report C §4): the forward quant helper produces scales along the *forward* contraction
axis; the backward GEMMs need columnwise (transposed) quantization on the other axis.
Reusing the forward helper naively puts scales on the wrong axis. The UE4M3 scale tensor
is also stored in a **swizzled/interleaved layout** (`SfKMajorAtom`; cuBLAS "1D block
scaling factors layout"), NOT plain row-major `(M, K/16)` — getting the transposed
operand's scale stride/swizzle wrong is the most likely `CUBLAS_STATUS_INVALID_VALUE`
(report C/D). **TE already solves all of this internally** (rowwise+columnwise copies +
swizzle + transpose handling), which is exactly why the TE-recipe path (§5a) is
least-effort: you do not re-derive any of it.

**dgrad is cheap, wgrad is the risk.** This maps onto the literature (§4) and is why the
proven nanochat recipe puts **fprop+dgrad in FP4 and keeps wgrad in BF16**.

---

## 4. Accuracy: RtN-instead-of-SR and no-RHT backward — the convergence cost

From report B (Exa), with numbers. **The cost lands almost entirely on wgrad.**

- **Cim et al. (PSU+AMD), arXiv:2605.09825** (native FP4 MI355X, Llama-3.1-8B): token
  overhead to reach target ppl vs FP8 — **Fprop-only FP4 = 8–9%; Fprop+Dgrad = 10–11%;
  +Wgrad = jumps to 26–27%.** "Wgrad quantization is the dominant contributor to
  convergence degradation." Their headline directly validates skipping SR+RHT:
  "stochastic rounding and randomized Hadamard rotations **fail to stabilize** training
  once Wgrad is quantized, whereas **deterministic Hadamard rotations consistently
  restore** stable optimization." A **deterministic 16×16 Hadamard** (a fixed ±1 bf16
  batched matmul — no RNG, no `cvt.rs`, no gated fused kernel) on the wgrad inputs alone
  brought the full-pipeline overhead 26% → **8–9%.**
- **NVIDIA NVFP4 pretraining, arXiv:2509.25149** (the gated 12B/10T recipe): RHT applied
  to **Wgrad inputs only** ("worse when applying to Fprop or Dgrad"); SR on **gradients
  only**; weights/activations already use RtN. So even the gold recipe uses **RtN for
  the forward** — our working forward is already RtN. The 4 techniques are framed as
  non-optional *for trillion-token-scale convergence*, not for runnability (report A §3).
- **Fishman/Banner, "FP4 All the Way"** (openreview kuzye4EPLR): NVFP4 with **no
  Hadamard at all**, SR only in backward, RtN forward → small loss gap fully closed by a
  brief QAF (FP4 fwd / BF16 bwd) phase.
- **Floor gap:** even *with* the full recipe the BF16 residual loss gap is ≈0.9–1.0%
  (Dong et al., arXiv:2602.02047). A RtN+deterministic-Hadamard backward sits modestly above.

**Budget for our mixed path:**
- RtN, **no** rotation, wgrad in FP4: runs and converges but ≈**26% token overhead** —
  acceptable for first bring-up, noticeably worse for long runs.
- RtN + **deterministic 16×16 Hadamard** on wgrad inputs (the recommended accuracy point):
  back to ≈**8–9%** overhead. Cheap, sm_121-runnable (plain bf16 matmul).
- **wgrad in BF16** (nanochat's `override_linear_precision=(False,False,True)`): forfeits
  FP4 throughput on wgrad only, keeps it on fprop+dgrad; **proven to train on gb10**
  (loss 11.09→6.58, `report.log`). This is the conservative, already-validated default.
- **Skip SR entirely** — Cim/AMD show SR does **not** help the wgrad case and can hurt;
  Quartet II independently calls SR the weakest link it replaces. With a deterministic
  Hadamard (or BF16 wgrad) you do not need SR.

> **Honest caveat (report B + nanochat doc):** no public ablation gives a single "+X%
> loss" number for removing *only* RHT or *only* SR; the receipts prove the mixed recipe
> **runs and loss descends ~22 steps**, NOT that it matches BF16 final accuracy. The
> failure mode to watch is slow wgrad-bias accumulation / outlier-underflow over many
> steps surfacing as a growing train-loss gap vs bf16 — **not a NaN**.

---

## 5. THE PLAN — least-effort path to a working backward (ranked)

### (a) [START HERE] TE non-default recipe — RtN, no RHT, no SR. EXISTS and is PROVEN on gb10.

This **is** the user's mixed backward, already wired through TE's rowwise/columnwise
quant + FP4 GEMM + correct swizzled scales. It is the sanctioned SM120 workaround
(TE #2372/#3062) and we already have a local gb10 receipt (`NVFP4-NANOCHAT-RECIPE.md`).

```python
from transformer_engine.common.recipe import NVFP4BlockScaling, Format

recipe = NVFP4BlockScaling(
    disable_rht=True,                       # RHT fused kernel OFF (the TMEM/tcgen05 crash)
    disable_stochastic_rounding=True,       # SR cvt.rs OFF (the sm_100a-gated cast)
    override_linear_precision=(False, False, True),  # (fprop, dgrad, wgrad): wgrad→BF16
    fp4_format=Format.E2M1,
)
# with te.fp8_autocast(enabled=True, fp8_recipe=recipe): y = te_linear(x); loss.backward()
```

`override_linear_precision` is TE's `(fprop, dgrad, wgrad)` tuple where `True`=force that
GEMM to **BF16**, `False`=use recipe FP4. `(False, False, True)` ⇒ **fprop FP4 + dgrad FP4
+ wgrad BF16** (per `NVFP4-NANOCHAT-RECIPE.md §2`). This keeps the single sensitive GEMM
(wgrad) in BF16 (§4) and leaves only RtN E2M1 fprop+dgrad — the identical FP4 path the
forward already proves works. **No `tcgen05`, no TMEM, no SR cvt, no Hadamard kernel is
touched.**

Knobs / env (report D):
- **Build arch-specific:** `NVTE_CUDA_ARCHS=120` (or `121`) so TE emits the `a`/`f` FP4
  cvt variants — otherwise you hit `nvfp4_transpose.cuh:234 … FP4 cvt PTX instructions
  are architecture-specific` (TE #2255). Equivalently `-arch=sm_121a` for any custom build.
- **`NVTE_BACKWARD_OVERRIDE=high_precision|dequantized`** — explicit per-GEMM escape hatch
  to force a still-misbehaving dgrad/wgrad to BF16 (use to keep the loud-fail clear path,
  not a silent fallback).
- **`NVTENVFP44Over6Mode=MinMSE`** — adaptive block-scaling accuracy knob to recover some
  of the accuracy lost by dropping RHT/SR (arXiv:2512.02010), orthogonal.
- TE's `check_recipe_support()` does NOT currently flag sm_120 for the default recipe
  (TE #3062) — it will let you build `NVFP4BlockScaling()` then crash. **We must set the
  disable flags explicitly; never rely on TE to auto-degrade.**

**Effort:** a recipe flag set + arch-specific TE build. **Failure modes:** must compile
`121`/`121a` or FP4 MMA is absent; residual sm_100-gated kernels in other TE paths. This
is the entire mechanism — same family as our forward.

### (b) Stretch: dgrad+wgrad BOTH in FP4 (RtN) + optional deterministic 16×16 Hadamard on wgrad

Once (a) is green, move wgrad into FP4: `override_linear_precision=(False, False, False)`,
keeping `disable_rht=True, disable_stochastic_rounding=True`. Per §4 expect ≈26% token
overhead. To recover to ≈8–9%, add a **deterministic** 16×16 Hadamard (a fixed ±1
bf16/fp16 batched GEMM you own — **not** TE's RHT fused kernel) on the wgrad inputs before
quant. This is the best accuracy/throughput point and is fully sm_121-runnable (no RNG, no
`cvt.rs`, no TMEM). TE does not expose a deterministic-only Hadamard knob, so this is a
small custom pre-quant matmul layered on the (a) path.

### (c) Raw library FP4 calls for the 3 GEMMs (only if escaping TE)

Most control, most work — you re-implement columnwise (transposed) quant + scale swizzle +
transpose handling TE already solves (§3, report C §4). Exact wiring (report D):
- elements `CUDA_R_4F_E2M1` (2 packed/byte; K in *elements*, buffer K/2 bytes);
- scales `CUDA_R_UE4M3`, mode `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 (=1)`;
- attrs `…_A_SCALE_POINTER/…_B_SCALE_POINTER` + `…_A_SCALE_MODE/…_B_SCALE_MODE`,
  `…_D_SCALE_POINTER` for the per-tensor FP32 out-scale (watch K≥32 alignment, pytorch#157054);
- `computeType=CUBLAS_COMPUTE_32F`; `transa=OP_T, transb=OP_N` (TN); `SfKMajorAtom`
  interleaved scale layout; build `-arch=sm_121a`.
- **BUT** per the §2 premise box: raw **cuBLASLt** FP4 is NOT dispatched on CC 12.x — you
  must go through **CUTLASS 3.8+ (Example 79a/79b, proven on sm_121)** or **cudnn-frontend
  block-scaled FP4 Gemm**, `-arch=sm_121a`. Code templates: `vuiseng9/fp4-training`
  (full 3-GEMM wiring, B200) + `VincentKaufmann/fp4-cuda-kernel` (the only standalone FP4
  GEMM proven to run on GB10/sm_121, forward-only — ships the GPU BF16→NVFP4 cast +
  `SfKMajorAtom` engine you'd reuse, add 2 transposed-operand GEMM calls). **Extra wrinkle:**
  some GB10 silicon reportedly lacks `cvt.rn.satfinite.e2m1x2.f32` — validate the RtN cvt on
  the exact device or use a software cvt substitute (report D).

### (d) Conservative baseline — FP4-storage + BF16-compute (dequant→bf16 GEMM)

Explicit, surfaced (never silent): dequant the NVFP4 operands to BF16 and run that GEMM as
a normal bf16 tensor-core matmul. Keeps the 2× memory/bandwidth win of FP4 storage,
forfeits the 2× FP4 tensor-core throughput on that GEMM only. This is exactly nanochat's
wgrad-BF16 (which IS option (a)'s default) and "FP4 All the Way"'s QAF / NVIDIA's
last-15%-in-BF16. Use as the per-GEMM downgrade you **select explicitly and report**, never
a try/except degraded path (RULE #1).

### Wiring into `scripts/_nvfp4_route.py` — REPLACE the blanket backward raise

The current `nvfp4_te_gemm_probe` does two wrong things vs this verdict:
1. It runs the **forward** with `disable_rht=True, disable_stochastic_rounding=True` (correct)
   but then deliberately constructs a **fresh** `NVFP4BlockScaling()` *default* recipe for the
   backward (lines 437–462) — i.e. it tests the SR+RHT path that we KNOW is datacenter-only,
   then raises `Nvfp4CudaKernelMiscompiled`. That raise is correct *for the default recipe* but
   **mislabels the whole backward as impossible**, which the nanochat receipt disproves.
2. `raise_if_nvfp4_training_unsupported` (line 569) hard-raises on ANY `--dtype nvfp4` training
   step, and `NVFP4_UNSUPPORTED_TRAINING_OPS` lists `nvfp4_gemm_backward_vjp_te_cuda_sm121` and
   `gemm_backward_vjp` as unconditionally unavailable.

Concrete changes (keep RULE #1 — still fail loud when the primitive genuinely isn't there):
- **Add** a `nvfp4_te_backward_mixed_probe(...)` that runs the backward under the **mixed**
  recipe `NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True,
  override_linear_precision=(False, False, True), fp4_format=Format.E2M1)` (with the
  graceful kwarg-degradation that `nanochat/gpt.py:_try_make_recipe` uses for older TE), calls
  `loss.backward()`, and asserts `dgrad`/`wgrad` rel-RMSE vs a BF16 reference is **< target**
  (§6). This is the *real* path. Keep the existing default-recipe probe but RENAME its raise to
  reflect "this is the SR/RHT **default** recipe, which is datacenter-only" — it is a negative
  control, not the verdict.
- **Move** `nvfp4_gemm_backward_vjp_te_cuda_sm121` from `NVFP4_UNSUPPORTED_TRAINING_OPS` to a new
  `NVFP4_SUPPORTED_OPS` entry `backward_nvfp4_dgrad_fp4_wgrad_bf16_te_mixed_cuda_sm121` once the
  probe passes on gb10. Keep `gemm_backward_vjp` (the *Metal* e2m1 GEMM grad) unsupported — that
  is a separate, genuinely-absent kernel.
- **Narrow** `raise_if_nvfp4_training_unsupported`: it should raise ONLY for the ops that remain
  genuinely missing (RMSNorm/SwiGLU/softmax/attention/Mamba3/M2RNN/embedding/optimizer-state in
  nvfp4 — those have no kernel and a full e2e nvfp4 step still can't run), NOT for the Linear
  backward GEMM, which now has a real path. The raise message must stop claiming the backward GEMM
  is impossible and instead point at the remaining non-GEMM blockers.
- **Keep** `Nvfp4CudaKernelMiscompiled` as the guard for the *default SR/RHT* recipe and for the
  "TE said available but gradients are garbage (rel_err>0.6)" detector — that fail-loud is still
  correct and must stay (RULE #1): if someone passes the default recipe, or if the mixed backward
  silently mis-executes, RAISE with WHERE+WHAT.

---

## 6. Test plan — verify dgrad/wgrad vs a BF16 reference on gb10

Mirror the existing forward `0.147` rel-err probe. On gb10 (sm_121), TE built with
`NVTE_CUDA_ARCHS=121` (or `120`):

1. **Build a fresh `te.Linear(K→N, bias=False, params_dtype=bf16)`**, copy a fixed weight, seed RNG.
2. **BF16 reference grads** (plain autograd, no autocast): forward `y_bf = lin(x)`, `y_bf.backward(g)`
   → capture `xg_bf = x.grad` (dgrad ref), `wg_bf = lin.weight.grad` (wgrad ref). (This is already
   built in `nvfp4_te_gemm_probe` lines 422–433 — reuse it.)
3. **Mixed FP4 grads**: under `te.fp8_autocast(enabled=True, fp8_recipe=mixed_recipe)` with the §5a
   recipe, `y = lin(x_fp4); y.backward(g)` → `xg_fp4`, `wg_fp4`. `torch.cuda.synchronize()` with
   `CUDA_LAUNCH_BLOCKING=1` so any async kernel error surfaces as a Python exception (already done,
   line 420).
4. **Targets** (rel-RMSE = `‖a−b‖/(‖b‖+eps)`):
   - **dgrad (FP4, RtN):** target **rel_err ≤ 0.20** (same order as the forward's 0.147 — dgrad is the
     same RtN E2M1 GEMM). FAIL LOUD if > 0.5 (mis-executed FP4 path).
   - **wgrad (BF16 via `override_linear_precision`):** target **rel_err ≤ 1e-2** (it IS bf16; any
     larger value means the override didn't take and FP4 leaked in — investigate).
   - For the option-(b) all-FP4 variant: **wgrad (FP4, RtN, no rotation)** target **rel_err ≤ 0.35**
     (looser; this is the sensitive GEMM); with deterministic Hadamard expect ≤ 0.20.
5. **Smoke convergence** (the nanochat receipt, reproducible): run ≥20 optimizer steps of the real
   model under the mixed recipe and assert **loss is monotonically decreasing** over a short window
   (nanochat saw 11.09→6.58 in 22 steps). A garbage backward cannot produce a clean descending loss.
6. **Negative control:** keep a probe that runs the **default** `NVFP4BlockScaling()` and asserts it
   RAISES with a `NVFP4_TE_ARCH_PTX_MARKERS` substring — proving we still fail loud on the
   datacenter-only path and that the mixed path is genuinely different.
7. **Build-arch guard:** assert the TE/torch arch list contains `sm_121`/`sm_120` and that the FP4
   forward probe is finite — if the cubin was remapped to a family target without the `a` FP4 MMA,
   the forward rel_err will blow up; catch it here before trusting the backward.

Record all rel-errs into the m04 receipt (do not just pass/fail) so a growing gap is visible.

---

## 7. Honest risks / unknowns (stated plainly)

- **No BF16-parity number exists.** The gb10 receipts prove the mixed recipe **runs and loss
  descends ~22 steps** (`report.log`, `debug_nvfp4_crash.py`), NOT that it matches BF16 final
  accuracy. We have no dgrad/wgrad rel-RMSE measurement yet — §6 step 4 is the first time we'd
  actually quantify it. Treat "trains, loss decreases" as the supported claim.
- **wgrad-in-FP4 (option b) is unmeasured by us.** The ≈26%-overhead / deterministic-Hadamard-fix
  numbers are from Cim/AMD on **MXFP4/MI355X**, not NVFP4/gb10 — directionally strong (wgrad is the
  risk) but not a gb10 measurement. nanochat sidesteps it entirely by keeping wgrad in BF16.
- **Forward-backend premise (report D, §2 box).** "Working forward = cuBLASLt" may be false; it's
  likely cuDNN/CUTLASS. Unverified until we set `CUBLASLT_LOG_LEVEL=5` on gb10. Affects only a future
  raw-library backward (option c), not the TE path.
- **`.rs`-excludes-`sm_120a` PTX-table line is the one open citation** (report C): the gating is fully
  corroborated by TE source gating + the sm_100a-only `cvt.rs` forum thread + SM120's arch-specific
  cvt path, but the single verbatim PTX-ISA §9.7.9 table cell was not retrieved (doc served paginated).
- **GB10 RtN cvt wrinkle:** one forum report claims some GB10 silicon lacks
  `cvt.rn.satfinite.e2m1x2.f32` (report D). TE's working forward suggests our device has it, but a
  raw-library path must validate the cvt on the exact silicon.
- **Slow-bias failure mode, not a crash.** Skipping SR means RtN gradient bias accumulates; the risk
  is a growing train-loss gap over a long run, surfacing late — watch the §6-step-5 loss curve over
  more than 22 steps before committing to a long training run.
- **Other non-GEMM nvfp4 ops still missing.** Even with the backward GEMM fixed, a full e2e nvfp4
  training step still lacks nvfp4 RMSNorm/SwiGLU/softmax/attention/Mamba3/M2RNN/embedding/optimizer —
  the narrowed `raise_if_nvfp4_training_unsupported` must still fail loud on those.

---

### Primary sources (from the four field reports — cited, not invented)
CUTLASS TN-only: `sm120_blockscaled_mma_builder.inl` static_assert, blackwell_functionality.html,
example 79b, cutlass#2800/#3227/#3096 (A). RtN/SR/tcgen05 HW audit: flashinfer#3170, PTXAS blackwell
target audit, `cvt.rs` forum thread, LLVM NVPTX intrinsics, PTX ISA §9.7.9 (C). Accuracy: Cim/AMD
arXiv:2605.09825, NVIDIA arXiv:2509.25149, FP4-All-the-Way openreview kuzye4EPLR, Graphcore
arXiv:2509.17791, Quartet/II, Dong arXiv:2602.02047 (B). Exact APIs/knobs: TE #2372/#3062/#2255,
`NVFP4BlockScaling(disable_rht, disable_stochastic_rounding)`, `NVTE_BACKWARD_OVERRIDE`,
`NVTE_CUDA_ARCHS=120/121`, cuBLASLt CC-10.x/11.0 scope + cuDNN/CUTLASS reality, `CUDA_R_4F_E2M1`/
`CUDA_R_UE4M3`/`VEC16_UE4M3`/`SfKMajorAtom`, vuiseng9/fp4-training, VincentKaufmann/fp4-cuda-kernel (D).
In-repo receipts: `docs/NVFP4-NANOCHAT-RECIPE.md` (gb10 fwd+bwd trained, `override_linear_precision=
(False,False,True)`, loss 11.09→6.58), `docs/NVFP4-TRAINING-KERNELS.md`, `scripts/_nvfp4_route.py`.
