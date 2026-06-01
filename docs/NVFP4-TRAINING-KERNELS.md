# NVFP4 (e2m1 + block-scale) TRAINING kernels — what exists, what runs on gb10, what fails loud

Status as of 2026-06-01. Author: David Gornshtein (davidgornshtein@gmail.com).

This document is the authoritative, evidence-backed record of the NVFP4
4-bit **training** kernel situation for cppmega.mlx, covering: the production
NVFP4 training recipes/kernels that exist upstream, exactly what is available on
our gb10 (NVIDIA GB10, sm_121) box vs. what is NOT, the e2m1 + block-scale math,
which ops we wired with REAL kernels, which ops fail loud (and how to enable
them), and the test we added with its measured results.

It obeys RULE #1 (no silent fallbacks): every nvfp4 op goes through ONE clear
path; where a kernel is missing or mis-compiled the route RAISES a precise,
actionable error naming WHERE + WHAT failed and the enablement — it never
silently downcasts to bf16.

---

## 1. What NVFP4 is (format + training recipe)

NVFP4 is NVIDIA's native Blackwell 4-bit **training** format. Two-level block
scaling:

* **Elements**: `E2M1` (1 sign, 2 exponent, 1 mantissa). Value set:
  `{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
* **Level-1 block scale**: every **16 consecutive elements** share one scale of
  type `E4M3` (FP8, 4 exp / 3 mantissa) — NOT power-of-two. (Our repo's existing
  Metal codec uses fp32 block scales over the same 16-element block geometry.)
* **Level-2 scale**: a single per-tensor `FP32` scale for the whole tensor.

cuBLASLt exposes this directly as `CUDA_R_4F_E2M1` matmuls with `CUDA_R_UE4M3`
scales over 16-element blocks (cuBLAS 12.9+).

**Training recipe** (NVIDIA TransformerEngine `NVFP4BlockScaling`, verified live
on gb10 — see §3). The three GEMMs of a linear layer use NVFP4 with these
per-operand quant params (exact recipe dump from gb10 TE 2.16.0.dev0):

| Operand cast            | RHT  | Stochastic rounding | 2D block quant |
|-------------------------|------|---------------------|----------------|
| `fp4_quant_fwd_inp`     | True | False               | False          |
| `fp4_quant_fwd_weight`  | False| False               | True           |
| `fp4_quant_bwd_grad`    | True | **True**            | False          |

* **Random Hadamard Transform (RHT)** is applied to GEMM inputs (fwd input and
  bwd grad) to reshape the distribution toward Gaussian and smooth outliers.
* **Stochastic rounding (SR)** is applied when quantizing gradients to NVFP4, to
  remove the bias quantization would otherwise introduce.
* **Weights** use 2D (selective) block quantization.
* On NVIDIA datacenter Blackwell, FP4 GEMMs run ~4x BF16 (GB200) / ~6x (GB300).

Sources: §7.

---

## 2. Which NVFP4 training kernels exist upstream (with URLs/versions)

* **NVIDIA TransformerEngine** — `NVFP4BlockScaling` recipe + `te.fp8_autocast`
  over `te.Linear`/`te.LayerNormLinear`. Python API verbatim:
  `transformer_engine.common.recipe.NVFP4BlockScaling`,
  `transformer_engine.pytorch.NVFP4Quantizer`,
  `transformer_engine.pytorch.NVFP4Tensor`,
  `transformer_engine.pytorch.is_nvfp4_available()`,
  `transformer_engine.pytorch.fp8.check_nvfp4_support()`.
  C API: `kNVTEFloat4E2M1`, `nvte_quantize()`,
  `kNVTEQuantizationConfigStochasticRounding`.
  Provides fwd + bwd (dgrad/wgrad) NVFP4 GEMM via cuBLASLt. Version on gb10:
  `2.16.0.dev0+a46079cb` (built from source at `/home/dave/TransformerEngine`).
* **cuBLASLt** — native block-scaled FP4 GEMM (`CUDA_R_4F_E2M1` + `CUDA_R_UE4M3`,
  16-elem blocks) since cuBLAS 12.9. gb10 has `nvidia-cublas 13.4.0.1`.
* **CUTLASS** — `examples/72_blackwell_narrow_precision_gemm/
  72b_blackwell_nvfp4_nvfp4_gemm.cu` block-scaled FP4 GEMM. gb10 has
  `nvidia-cutlass-dsl 4.4.2`.
* **FlashInfer** — SM120 NVFP4 MoE grouped GEMM (with the SM120 patches /
  `compute_120f`). gb10 has `flashinfer-python 0.6.8`.
* **Community references**: `github.com/vuiseng9/fp4-training` (cuBLASLt +
  Microxcaling mxfp8/nvfp4 training, concept→implementation);
  NVIDIA's 12B hybrid Mamba-Transformer NVFP4 pretraining (10T tokens).

---

## 3. What is available on gb10 (sm_121) — measured, not assumed

Box: `ssh gb10` → NVIDIA GB10, **compute_cap 12.1 (sm_121)**, driver 595.71.05,
CUDA 13.2. Venv `/home/dave/cppmega-venv`: torch `2.13.0.dev20260417+cu132`
(cuda 13.2), TE `2.16.0.dev0+a46079cb`, cublas `13.4.0.1`, cutlass-dsl `4.4.2`,
flashinfer `0.6.8`.

Capability probe (`scripts/_nvfp4_route.nvfp4_te_cuda_capability()`) on gb10:

```json
{ "cuda_device": "NVIDIA GB10", "compute_capability": "sm_121",
  "te_version": "2.16.0.dev0+a46079cb",
  "te_check_nvfp4_support": [true, ""], "te_is_nvfp4_available": true }
```

### 3a. FORWARD NVFP4 GEMM — WORKS on gb10 (real, measured)

`te.Linear` under `te.fp8_autocast(NVFP4BlockScaling())` forward = cuBLASLt
`CUDA_R_4F_E2M1` + E4M3 block-scale matmul. Measured forward output rel-RMSE vs
the bf16 reference:

```
forward_nvfp4_gemm_te_cublaslt_cuda_sm121: rel_rmse_vs_bf16 = 0.147, finite=True
```

0.147 is the expected FP4 forward error (no RHT). This is a REAL nvfp4 op on
gb10 and we wire/test it (§5, §6).

### 3b. BACKWARD NVFP4 GEMM — BROKEN on gb10 (root cause identified)

Running the **default production recipe** (RHT on fwd-inp + bwd, SR on bwd)
backward on gb10 raises a hard CUDA error from the Random-Hadamard fused quant
kernel:

```
RuntimeError: .../hadamard_transform/row_cast_col_hadamard_transform_cast_fusion.cu:1200
  in function row_col_rht_gemm_ntt_w_sfc: CUDA Error: invalid argument
```

With RHT disabled, the backward stochastic-rounding FP4 cast asserts (one per
thread):

```
.../util/ptx.cuh:935 in function mul_cvt_bf16_to_fp4_8x_stochastic_rounding:
  FP4 cvt PTX instructions are architecture-specific.
  Try recompiling with sm_XXXa instead of sm_XXX.
```

**Root cause (proven via cuobjdump on the TE .so):** the installed
`libtransformer_engine.so` embeds SASS for:

```
sm_75, sm_80, sm_89, sm_90, sm_90a, sm_100, sm_100a, sm_103a, sm_120  (PLAIN)
```

i.e. TE's default CUDA-13 arch list `75;80;89;90;100;120` — `sm_120` is built
**PLAIN**, and the architecture-specific `a` variants exist only for datacenter
Blackwell (`sm_100a`, `sm_103a`). GB10 is `sm_121`, runs the `sm_120` plain
SASS, which lacks the arch-specific FP4 cvt / RHT instructions → the asserts
above.

**The `a` variant is NOT the fix on desktop/consumer Blackwell.** Per
NVIDIA/cutlass#3096 and flashinfer-ai/flashinfer#2723, forcing `compute_120a`
gencode SEGFAULTS on desktop Blackwell ("RTX PRO 6000 reports compute capability
12.0, not 12.0a; the `a`-specific instructions are not available on desktop
Blackwell"). The working target is the **family-specific `compute_120f`** added
in CUDA 13.0, which "enables the full SM120 feature set with working TMA WS
grouped-GEMM tactics" (compute_120f took a CUTLASS NVFP4 MoE from 14.6 → 39.0
tok/s vs compute_120a). So the enablement for gb10 backward is to rebuild TE/
cuBLASLt-path FP4 kernels under CUDA ≥ 13 targeting `120f`, once TE's build
emits the `f` family target.

**Danger note (why we fail loud):** the backward FP4 error is ASYNC. Without
`CUDA_LAUNCH_BLOCKING=1` AND a fresh `te.Linear`, TE quantizer-workspace state
can mask the failure and hand back gradients that *look* plausible — a silent
mis-execution. Our probe defeats this by (a) using a fresh module for the
default-recipe backward and (b) treating the arch-specific assertion as a hard
RAISE. We never accept the masked gradients.

### 3c. MXFP8 — also unavailable on gb10 (for completeness)

`check_mxfp8_support()` → `(False, 'MXFP8 (for all gemm layouts) is not supported
on 12.0+ architectures yet.')`. Not used by the nvfp4 route; recorded so the
arch picture is complete.

---

## 4. The e2m1 + block-scale math (as implemented)

For a vector `v` split into 16-element blocks `b_j`:

1. `amax_j = max(|b_j|)`; level-1 scale `s_j = amax_j / E2M1_MAX` (E2M1_MAX = 6),
   stored as `E4M3` (TE) or `fp32` (repo Metal codec).
2. Each element `e = round(b_j[i] / s_j)` snapped to the nearest E2M1 value;
   gradients additionally use **stochastic rounding** (round up/down with
   probability proportional to distance between the two representable values).
3. A per-tensor `FP32` level-2 scale captures global dynamic range.
4. Optional **RHT** is applied to fwd-input and grad tensors before step 1 to
   Gaussianize the distribution.
5. Dequant: `b_j[i] ≈ s_j * e * level2_scale`.

GEMM: NVFP4(A) × NVFP4(B) accumulates in FP32 on the tensor cores (cuBLASLt
`CUDA_R_4F_E2M1`), de-scaling per-block by the E4M3 scales.

---

## 5. What we WIRED (real kernels) vs FAIL-LOUD (and how to enable)

Route module: `scripts/_nvfp4_route.py`. Wiring into the training step:
`scripts/m04_train_step.py` (imports + RULE #1 gate at `_run_existing_training`
and a defensive guard at `run_local_gb10_quarter_training`). `--dtype nvfp4` is
an accepted route (no argparse rejection); the dry-run path emits the route
metadata; the real training step RAISES the precise blocker.

**REAL nvfp4 ops wired (`NVFP4_SUPPORTED_OPS`):**

| Op | Backend | Kernel | Status |
|----|---------|--------|--------|
| `forward_gemm_operands_e2m1_block_scale_metal_m4` | Metal/M4 | `cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c` over `quantize_mxfp4_blockwise` | real (M4); exercised by `nvfp4_gemm_smoke` |
| `forward_nvfp4_gemm_te_cublaslt_cuda_sm121` | CUDA/gb10 | TE `NVFP4BlockScaling` + cuBLASLt FP4 | **VERIFIED on gb10, rel_err 0.147**; exercised by `nvfp4_te_gemm_probe(run_backward=False)` |

**FAIL-LOUD ops (`NVFP4_UNSUPPORTED_TRAINING_OPS`)** — the route RAISES naming
each; it does NOT bf16-fallback:

| Op | Why | Enablement |
|----|-----|-----------|
| `nvfp4_gemm_backward_vjp_te_cuda_sm121` | TE FP4-cvt/RHT PTX is arch-specific; TE built `sm_120` plain (not the `f` family) → asserts / CUDA invalid-argument on gb10 | Rebuild TE under CUDA ≥ 13 with family target `compute_120f` (NOT `sm_120a`, which SEGFAULTS on desktop Blackwell; cf. cutlass#3096) |
| `cuda_blackwell_nvfp4_gemm_metal_only` | The Metal fwd GEMM is Metal-only; it RAISES on CUDA | Use the TE/cuBLASLt CUDA forward path above |
| `gemm_backward_vjp` | No nvfp4 grad kernel for the Metal e2m1 GEMM | Implement an nvfp4 VJP for `mxfp4_matmul_path_c` |
| `optimizer_state_and_update` | AdamW moments are fp32-only; no nvfp4 optimizer state | Implement nvfp4 optimizer-state kernels |
| `rmsnorm`, `swiglu_ffn`, `softmax`, `sparse_mla_attention`, `mamba3_selective_scan`, `m2rnn_recurrence`, `embedding_and_lm_head`, `residual_add` | No nvfp4 kernel for these graph ops | Implement per-op nvfp4 kernels |

Because a full HybridTinyLM step needs ALL of the above, `--dtype nvfp4`
delivers the honest partial: real forward nvfp4 GEMM, fail-loud for the rest.
For a full training step TODAY use `--dtype fp8_path_c` or `--dtype bfloat16`.

**Fail-loud guards (RULE #1):**

* `raise_if_nvfp4_training_unsupported(args)` — on the training critical path,
  RAISES `Nvfp4TrainingRouteUnavailable` before any op would run in bf16.
* `_run_existing_training` emits a `blocked_receipt(... NVFP4_E2E_TRAINING_
  BLOCKER_TYPE)` with the precise reason.
* `nvfp4_te_gemm_probe(run_backward=True)` RAISES
  `Nvfp4CudaKernelMiscompiled` on the broken gb10 backward (arch-specific PTX),
  rather than returning the masked/garbage gradients TE would otherwise hand
  back. No silent fallback exists anywhere in the route.

---

## 6. The test we added and its results

Test: `tests/test_nvfp4_route.py` (10 tests). Backend-aware: always-on metadata
+ fail-loud assertions; Metal-only assertions skip when MLX/Metal absent; CUDA
assertions run when torch+TE+CUDA+NVFP4 present.

* `test_route_constants_well_formed`, `test_route_requested_predicate`,
  `test_payload_advertises_both_forward_backends_and_no_e2e`,
  `test_reason_message_is_actionable`,
  `test_capability_probe_never_crashes_without_torch` — route metadata + probe.
* `test_fail_loud_guard_raises_for_nvfp4_and_names_missing_ops`,
  `test_fail_loud_guard_is_noop_for_other_dtypes` — RULE #1 gate.
* `test_metal_forward_nvfp4_gemm_matches_bf16` — real Metal fwd GEMM < 0.5 rel.
* `test_cuda_forward_nvfp4_gemm_matches_bf16` — **real gb10 NVFP4 fwd GEMM
  < 0.3 rel** (measured 0.147).
* `test_cuda_backward_nvfp4_gemm_fails_loud_on_miscompiled_te` — the broken gb10
  backward RAISES `Nvfp4CudaKernelMiscompiled` naming the `compute_120f`
  enablement. (Skips, expecting pass, when `NVFP4_BACKWARD_FIXED=1` once TE is
  rebuilt with `120f`.)

**Results:**

* Mac (this checkout): `7 passed, 3 skipped` — Metal skipped because the
  in-tree tilelang dev build (`merge/upstream-codegen-reorg`) has a tvm_ffi
  circular-import (pre-existing infra gap, unrelated to nvfp4); CUDA skipped (no
  CUDA on Mac). The Metal assertion is gated by a real tiny-GEMM smoke so it
  skips cleanly on a broken-tilelang host and runs where tilelang is healthy.
* gb10 (`/home/dave/cppmega-venv`): **`9 passed, 1 skipped`** — Metal skipped (no
  Apple GPU); the CUDA forward test PASSES (real nvfp4 GEMM, 0.147 rel) and the
  CUDA backward fail-loud test PASSES (broken backward RAISES as designed).

Reproduce on gb10:

```bash
# copy scripts/_nvfp4_route.py + tests/test_nvfp4_route.py to a dir with a
# scripts/ package, then:
cd <dir> && /home/dave/cppmega-venv/bin/python -m pytest tests/test_nvfp4_route.py -v
```

---

## 7. Sources

* TransformerEngine NVFP4 feature doc —
  https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
* TransformerEngine Common API (NVFP4BlockScaling, kNVTEFloat4E2M1) —
  https://nvidia.github.io/TransformerEngine/api/common.html
* "NVFP4 Trains with Precision of 16-Bit and Speed/Efficiency of 4-Bit" (NVIDIA
  blog) —
  https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/
* NVIDIA 4-bit pretraining (12B hybrid Mamba-Transformer, 10T tokens) —
  https://www.marktechpost.com/2026/05/18/nvidia-introduces-a-4-bit-pretraining-methodology-using-nvfp4-validated-on-a-12b-hybrid-mamba-transformer-at-10t-token-horizon/
* TransformerEngine repo —
  https://github.com/NVIDIA/TransformerEngine
* cuBLAS 12.9 block-scaled FP4 (CUDA_R_4F_E2M1 + CUDA_R_UE4M3, 16-elem blocks) —
  https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/
* CUTLASS SM120 NVFP4: sm_120a SEGFAULT, compute_120f fix (CUDA 13.0) —
  https://github.com/NVIDIA/cutlass/issues/3096
* FlashInfer SM120 NVFP4 grouped GEMM patching —
  https://github.com/flashinfer-ai/flashinfer/issues/2723
* CUTLASS NVFP4 GEMM example —
  https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu
* Community nvfp4 training (cuBLASLt + Microxcaling) —
  https://github.com/vuiseng9/fp4-training
