# NVFP4 on GB10 / sm_121 — the nanochat WORKING recipe (fwd + backward) and a port plan to `_nvfp4_route.py`

Status: 2026-06-01. Investigator note for cppmega.mlx. Source repo: `/Volumes/external/sources/nanochat`.

**TL;DR (no overclaiming — RULE #1):** nanochat *did* solve a **full forward+backward
NVFP4 training step on GB10 (sm_121)** and has a live receipt of it (loss
11.09 → 6.58 over 22 steps). It did NOT do this by getting the *default*
production recipe (RHT + stochastic-rounding backward) to run — that crashes on
sm_121 exactly as our `_nvfp4_route.py` documents. It did it by using a
**non-standard mixed recipe** that our route never tried:

> `NVFP4BlockScaling(disable_rht=True, disable_sr=True, disable_stochastic_rounding=True, override_linear_precision=(False, False, True), fp4_format=Format.E2M1)`

i.e. **RHT off, SR off, and wgrad forced to BF16** — leaving only **fprop FP4 +
dgrad FP4 with plain round-to-nearest** (the same cuBLASLt FP4 path our forward
already proves works). This is precisely the "non-standard mixed backward
reusing the working FP4 GEMM with RtN, skipping RHT/SR" hypothesis. **It RAN and
TRAINED on gb10.** Our current `Nvfp4CudaKernelMiscompiled` backward raise is
correct *only for the default RHT+SR recipe*; it is **too broad** — it forecloses
the RtN/no-RHT/wgrad-in-BF16 path that nanochat demonstrated.

---

## 1. Files / commits found (paths + line refs)

All paths under `/Volumes/external/sources/nanochat`.

| File | Lines | What it is |
|------|-------|-----------|
| `CHANGELOG_GB10.md` | 240–268 | **The "win" doc.** "NVFP4 on GB10: FIXED! Requires `disable_rht=True`." Exact recipe block at 251–257; benchmark TFLOPS table 259–264; root cause = TE RHT (`hadamard_transform_cast_fusion.cu`) crashes on SM120/121 `CUDA Error: invalid argument` when M>32 (244–248); ref to TE issue #2372 (266). |
| `nanochat/gpt.py` | 911–1057 | `select_precision(target, disable_rht=True, disable_sr=True)` — builds the recipe. `_build_te_recipe("nvfp4")` at 940–962 constructs `NVFP4BlockScaling` with the exact kwargs (951–961). Auto-selects nvfp4 on SM≥12 when TE says nvfp4 ok (1008–1011). `_check_te_capability` at 887–908. |
| `nanochat/gpt.py` | 1060–1100, 1136–1182 | `make_autocast_ctx` / `make_block_autocast_ctx` — wraps the model in `te.autocast(enabled=True, recipe=plan.recipe)` (1096–1100). |
| `nanochat/te_linear_replacement.py` | 1–145 | `replace_all_linears_with_te(model)` — swaps every `nn.Linear` for `te.pytorch.Linear` (so the FP4 GEMM is actually used), with an exclude list (embeddings, lm_head, routers, LoRA…) at 28–44. Gated by `--use_te_all_linears`. |
| `scripts/base_train.py` | 9609–9633 | Wires CLI → `select_precision(target=args.precision, disable_rht=args.nvfp4_disable_rht, disable_sr=args.nvfp4_disable_sr)`. Prints the precision receipt (9614). NVFP4 needs `device_batch_size ≥ 2` (9627–9633). |
| `scripts/base_train.py` | 10854–10890 | Applies `replace_all_linears_with_te` before FSDP/compile when `--use_te_all_linears`. |
| `scripts/train_args.py` | 3304–3329 | CLI: `--precision auto\|nvfp4\|fp8\|bf16`, `--nvfp4_disable_rht` (default True, "Required for SM121/GB10"), `--nvfp4_disable_sr` (default True, "Required for SM121/GB10"). |
| `scripts/benchmark_nvfp4.py` | 228–296 | Raw GEMM benchmark: builds `NVFP4BlockScaling(fp4_format=Format.E2M1, disable_rht=True)` over a `te.Linear`, forward-only timing. |
| `scripts/debug_nvfp4_crash.py` | 22–53 | **Minimal fwd+bwd repro that RAN.** `select_precision('nvfp4')` then `model(x,y)` and **`loss.backward()` (line 51)**. Comment line 17: "B=32 OOMs during backward" (so B=16 backward succeeded — it OOM'd, it did not arch-crash). |
| `scripts/compare_bf16_nvfp4.py` / `compare_bf16_nvfp4_b16.py` | full | BF16-vs-NVFP4 **training** benchmark incl. `(loss/ga).backward()` (compare:121,140). |
| `tests/test_precision_selection.py` | 106–159 | Unit test pinning the recipe contract: `disable_rht`, `disable_stochastic_rounding is True`, **`override_linear_precision == (False, False, True)`** (121), `fp4_format == "E2M1"` (122); and that autocast uses `te.autocast(enabled=True, recipe=plan.recipe)` (159). |
| `report.log` | 13–54 | **LIVE gb10 RECEIPT (see §3).** |

No hand-written CUDA/Triton FP4 kernel and no CUTLASS-120f path exists in nanochat
for this — the entire mechanism is **TransformerEngine `NVFP4BlockScaling` +
`te.Linear` + cuBLASLt**, same family our forward already uses. (The MoE-expert
FP8 path `nanochat/te_experts.py` uses TE `GroupedLinear`/`tex.te_general_grouped_gemm`
with dgrad-NN + wgrad-NT layouts, but that bank is **FP8**, not nvfp4 — do not
conflate it.)

---

## 2. The exact working recipe (fwd + bwd mechanism, quant + GEMM layout)

### Recipe object (verbatim, from `nanochat/gpt.py:951–961`)

```python
recipe = NVFP4BlockScaling(
    disable_rht=True,                    # Random Hadamard Transform OFF
    disable_sr=True,                     # stochastic rounding OFF (alias)
    disable_stochastic_rounding=True,    # stochastic rounding OFF (canonical kwarg)
    override_linear_precision=(False, False, True),  # (fprop, dgrad, wgrad): wgrad→BF16
    fp4_format=Format.E2M1,
)
# applied via: with te.autocast(enabled=True, recipe=recipe): y = te_linear(x)
```

`select_precision` defaults `disable_rht=True, disable_sr=True`. `_try_make_recipe`
(gpt.py:916–938) degrades gracefully kwarg-by-kwarg if an older TE rejects a name,
so the same code path works across TE versions.

### What each GEMM actually runs (the load-bearing claim)

`override_linear_precision` is the TE tuple `(fprop, dgrad, wgrad)` where `True`
= force that GEMM to **high precision (BF16)**, `False` = use the recipe's FP4.
nanochat's `(False, False, True)` therefore yields:

| GEMM | Precision | Quant cast | RHT | SR |
|------|-----------|-----------|-----|----|
| **fprop** (y = x·Wᵀ) | **NVFP4 (E2M1)** | round-to-nearest | OFF | OFF |
| **dgrad** (dx = dy·W) | **NVFP4 (E2M1)** | round-to-nearest | OFF | OFF |
| **wgrad** (dW = dyᵀ·x) | **BF16** | — | — | — |

This is the key difference from the **default** recipe our doc analyzed
(`NVFP4-TRAINING-KERNELS.md §1`), where `fp4_quant_bwd_grad` has **RHT=True and
SR=True**. nanochat removes both the RHT fused kernel (the one that raises
`row_col_rht_gemm_ntt_w_sfc: CUDA Error: invalid argument` on sm_121) and the SR
FP4 cvt (`cvt.rs.satfinite.e2m1x4`, the one source-gated to `sm_100a`/`sm_103a`),
**and** moves the one remaining FP4 backward GEMM that would otherwise need an
NT-transposed weight-grad quant (wgrad) out to BF16. What is left in FP4 — fprop
and dgrad — is **plain round-to-nearest E2M1 + E4M3 16-element block scale**, i.e.
the identical cuBLASLt `CUDA_R_4F_E2M1 + UE4M3` path our forward already proves
works on sm_121. No `tcgen05`, no TMEM, no SR cvt, no Hadamard kernel is touched.

So the cppmega premise "the backward needs SR + RHT and those are datacenter-only"
is true **for the default recipe** but is **avoidable**: dgrad does not *require*
SR — SR only *reduces quantization bias*; with `disable_sr=True` dgrad quantizes
RtN, exactly like the forward. wgrad is the part that genuinely benefits from
SR/RHT for accuracy, and nanochat simply keeps wgrad in BF16 instead.

### Quant layout (unchanged from our §4)

E2M1 elements; one **E4M3** level-1 scale per **16 consecutive elements**; one
FP32 level-2 per-tensor scale; cuBLASLt `CUDA_R_4F_E2M1 × CUDA_R_UE4M3`,
FP32 accumulate. nanochat does not customize the block geometry; it uses TE's
default NVFP4 layout.

---

## 3. gb10 / sm_121 evidence it RAN (logs / loss)

**`report.log` (lines 11–54) — a real GB10 base-train run:**

```
Autodetected device type: cuda
HW Capability: NVFP4=True () | FP8=True            # report.log:13
Precision: NVFP4 (E2M1) + WGrad BF16              # report.log:14
...
Converted model to bfloat16 for TE training        # report.log:23
Number of parameters: 560,988,160
step 00000/21400 | loss: 11.090637 | tok/sec: 15,424 | mfu: 10.77   # report.log:32
step 00010/21400 | loss:  7.426396 ...                              # report.log:42
step 00022/21400 | loss:  6.580820 | tok/sec: 16,619 | mfu: 11.61   # report.log:54
```

This is the receipt: an NVFP4-E2M1 forward + **WGrad-BF16 backward** run where the
**loss falls monotonically 11.09 → 6.58 over 22 optimizer steps**. A broken/garbage
backward cannot produce a clean descending loss curve — the gradients are correct.
560 M params, depth-20 d1280 model, B=32×T=2048, grad-accum 8. (`report.log` header
dated 2026-01-11.)

**Supporting evidence:**
- `scripts/debug_nvfp4_crash.py:51` calls `loss.backward()` under the nvfp4
  recipe with **no torch.compile** (raw eager) — the only failure noted is OOM at
  B=32 (comment line 17), i.e. a memory limit, *not* an arch-specific PTX crash.
- `CHANGELOG_GB10.md:259–264` benchmark (RHT-disabled) — measured GB10 FP4 vs BF16:
  1.34× (4096³), 1.23× (8192×4096²), 1.36× (16384×4096²). These are real GB10
  TFLOPS, so the FP4 GEMM executed.
- `CHANGELOG_GB10.md:69–77` MFU calc keys off `"GB10" in get_device_name(0)` and
  `precision_plan.use_te` → confirms the device was a real GB10 and TE was active.

**Caveat (stated plainly, RULE #1):** the receipts prove (a) the recipe **runs**
and (b) the loss **descends** on gb10. They do **not** include a dgrad/wgrad
rel-RMSE-vs-bf16 number or a full-convergence (val-bpb-to-target) comparison
against a BF16 baseline. The strongest correctness claim the evidence supports is
"trains, loss decreases as expected for ~22 steps" — not "matches BF16 final
accuracy." If we adopt this for cppmega.mlx we should add a dgrad rel-err probe
(mirror of our forward 0.147 probe) to quantify the backward error before relying
on it for a long run.

---

## 4. Is this DATACENTER-ONLY? Reconciling with our `_nvfp4_route.py`

Our `docs/NVFP4-TRAINING-KERNELS.md §3b` conclusion — "backward is genuinely
datacenter-only, a rebuild can't fix it" — is **correct as scoped**: it is about
the **default recipe's** SR FP4 cvt + RHT kernel, both of which are real and
unavailable on sm_121. nanochat does **not contradict** that; it **avoids** that
code entirely:

- It never calls the SR cvt (`disable_sr/disable_stochastic_rounding=True`) → the
  `ptx.cuh:935 ... FP4 cvt PTX instructions are architecture-specific` path is
  never reached.
- It never calls the RHT fused kernel (`disable_rht=True`) → the
  `row_cast_col_hadamard_transform_cast_fusion.cu ... CUDA Error: invalid argument`
  path is never reached.
- The one FP4 backward GEMM that remains (dgrad) uses the **same** cuBLASLt
  block-scaled FP4 pipeline as the forward (which we already verified at rel_err
  0.147), with RtN quant.
- wgrad is BF16, so the NT-layout weight-grad FP4 quant is never needed either.

So our route's blanket `nvfp4_gemm_backward_vjp_te_cuda_sm121` →
`Nvfp4CudaKernelMiscompiled` raise is **over-broad**: it blocks a backward path
that demonstrably works. The accurate statement is: **"the default RHT+SR FP4
backward is datacenter-only; a reduced RtN/no-RHT backward with wgrad-in-BF16
runs on sm_121."**

---

## 5. Concrete port plan to `scripts/_nvfp4_route.py`

Goal: replace the unconditional backward raise with a **real, fail-loud RtN
backward path** that mirrors nanochat, and keep the *default-recipe* backward
fail-loud (it really is broken). Everything below is plain TE Python API — no new
CUDA, no CUTLASS, no kernels to write.

### 5.1 Exact APIs / enums / flags to use

- `transformer_engine.common.recipe.NVFP4BlockScaling`
- `transformer_engine.common.recipe.Format.E2M1`
- `transformer_engine.pytorch.Linear` (the GEMM carrier) and
  `transformer_engine.pytorch.autocast(enabled=True, recipe=...)`
  (older TE: `fp8_autocast(enabled=True, fp8_recipe=...)` — probe both, as
  `nanochat/gpt.py:1092` does).
- Recipe kwargs (copy nanochat exactly, with graceful per-kwarg fallback à la
  `_try_make_recipe`, `gpt.py:916–938`):
  - `disable_rht=True`
  - `disable_sr=True`
  - `disable_stochastic_rounding=True`
  - `override_linear_precision=(False, False, True)`   ← (fprop, dgrad, wgrad); wgrad→BF16
  - `fp4_format=Format.E2M1`
- Capability gate (already present in our probe): keep
  `te.pytorch.is_nvfp4_available()` / `fp8.check_nvfp4_support()`.

### 5.2 New supported-op + the code change

1. **Add a new SUPPORTED op constant** in `_nvfp4_route.py`, e.g.
   `"backward_nvfp4_rtn_dgrad_fp4_wgrad_bf16_te_cublaslt_cuda_sm121"`, and
   **move** it out of `NVFP4_UNSUPPORTED_TRAINING_OPS` into `NVFP4_SUPPORTED_OPS`.

2. **Keep** `nvfp4_gemm_backward_vjp_te_cuda_sm121` (the *default* RHT+SR backward)
   in `NVFP4_UNSUPPORTED_TRAINING_OPS` — that raise is still correct.

3. **Extend `nvfp4_te_gemm_probe`** so that when `run_backward=True` it builds the
   recipe with the **nanochat reduced kwargs** (5.1) instead of the default recipe,
   then runs `loss.backward()` on a fresh `te.Linear`. Add a backward rel-err
   measurement: compare `x.grad` (FP4 dgrad) against a BF16-reference dgrad and
   record `bwd_dgrad_rel_rmse_vs_bf16`. This gives us the missing §3 receipt.
   - If this backward *raises an arch-specific PTX/RHT error*, keep raising
     `Nvfp4CudaKernelMiscompiled` (defensive — should not happen with RHT+SR off,
     but RULE #1).
   - Only treat it as supported if it runs **and** the dgrad rel-err is within a
     declared bound (e.g. < 0.3, matching our forward acceptance threshold).

4. **Update the FAIL-LOUD raise text** so it stops claiming the *whole* nvfp4
   backward is datacenter-only. New phrasing: the **RHT+SR** backward is
   datacenter-only; the **RtN dgrad-FP4 / wgrad-BF16** backward is available via
   `override_linear_precision=(False,False,True)` + `disable_rht/disable_sr`.

5. **Mechanism to actually USE it in a training step** (mirrors
   `nanochat/te_linear_replacement.py`): to route real matmuls through FP4, every
   trainable `nn.Linear` in the step must be a `te.pytorch.Linear` constructed
   under the recipe's autocast. For cppmega.mlx's CUDA/gb10 path that means a
   `replace_linears_with_te()` helper + wrapping the fwd/bwd in
   `te.autocast(recipe=...)`. (Per-op things still without an nvfp4 kernel —
   rmsnorm, swiglu, softmax, attention, mamba scan, optimizer state — remain
   fail-loud exactly as today; this port only un-blocks the **GEMM** backward.)

### 5.3 What to copy, file-by-file

| From nanochat | To cppmega.mlx | Adapt |
|---------------|----------------|-------|
| `gpt.py:940–962` (`_build_te_recipe("nvfp4")`) | `_nvfp4_route.py` new `_build_nvfp4_rtn_recipe()` | drop nanochat plumbing; keep the 5-kwarg recipe + `_try_make_recipe` graceful fallback (`gpt.py:916–938`) |
| `gpt.py:1088–1100` (autocast probe) | `_nvfp4_route.py` probe | probe `te.autocast` then `te.fp8_autocast` |
| `te_linear_replacement.py:55–145` | new `scripts/_te_linear_replace.py` (CUDA-only) | keep the exclude list (embeddings/lm_head/router/lora); only needed if we wire a real step, not for the probe |
| `debug_nvfp4_crash.py` | extend our `nvfp4_te_gemm_probe(run_backward=True)` | use reduced recipe; add dgrad rel-err |
| `tests/test_precision_selection.py:106–159` | `tests/test_nvfp4_route.py` new case | assert recipe kwargs `(False,False,True)` + `disable_rht`/`disable_sr`; add a `test_cuda_backward_nvfp4_rtn_runs_and_descends` to run on gb10 |

### 5.4 Validation gate before trusting it (RULE #1)

Before flipping the backward to "supported" in production:
1. Run the extended probe on gb10 → record `bwd_dgrad_rel_rmse_vs_bf16` (must be
   finite and within bound).
2. Run a short real training step (≥ ~20 steps) and confirm **loss descends**
   (reproduce report.log's 11.09→6.58 shape) — this is the integration receipt.
3. If either fails, keep the raise. Do **not** ship a silent BF16 fallback.

---

## 6. Bottom line

- nanochat solved **forward + backward NVFP4 training on GB10/sm_121** and has a
  live receipt (`report.log`: NVFP4 E2M1 + WGrad-BF16, loss 11.09→6.58).
- Mechanism is **TransformerEngine only** (`NVFP4BlockScaling` + `te.Linear` +
  cuBLASLt) — no custom kernel, no CUTLASS-120f, no Triton.
- The "win" = `disable_rht=True, disable_sr=True, disable_stochastic_rounding=True,
  override_linear_precision=(False,False,True), fp4_format=E2M1`, documented in
  `CHANGELOG_GB10.md:240–268` and pinned by `tests/test_precision_selection.py`.
- Net effect: **fprop FP4 + dgrad FP4 (round-to-nearest, no RHT/SR) + wgrad BF16**
  — exactly the "reuse the working FP4 GEMM for the backward, skip RHT/SR" path.
- Our `_nvfp4_route.py` correctly blocks the **default** RHT+SR backward, but its
  raise is **too broad**: it also blocks this RtN path, which works. The port is
  small and pure-Python: add the reduced recipe, exercise it in
  `nvfp4_te_gemm_probe(run_backward=True)` with a dgrad rel-err, move the new op to
  supported, and (for a real step) add a TE-Linear replacement pass mirroring
  `te_linear_replacement.py`.
- Honesty note: nanochat's receipt proves *runs + loss-descends*, not *matches-BF16
  final accuracy*. Add the dgrad rel-err + a short convergence check before we
  rely on it for a long run.
