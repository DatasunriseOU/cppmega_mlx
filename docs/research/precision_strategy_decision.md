# Precision strategy for cppmega.mlx — decision based on 4-agent audit

Date: 2026-05-03

Synthesis of four parallel research streams into a single decision document on
**how to handle precision (weights / grads / moments / master)** in cppmega.mlx.

Source streams:
- Web research on canonical bf16+fp32 patterns: `docs/research/bf16_fp32_master_moments_pattern.md`
- Inspect nanochat: `/Volumes/external/sources/nanochat`
- Inspect cppmega CUDA: `/Volumes/external/sources/cppmega`
- Inspect cppmega.mlx local: `docs/research/cppmega_mlx_optimizer_precision_audit.md`

---

## TL;DR

**Three production reference stacks, three different answers**:

| Stack | Weights | Grads | Optimizer moments | Master copy of weights |
|---|---|---|---|---|
| **Megatron-LM canonical** | bf16 | fp32 | fp32 m, v | **fp32 separate buffer** |
| **DeepSpeed bf16_optimizer** | bf16 | fp32 | fp32 m, v | **fp32 separate buffer** |
| **PyTorch AMP standard** | fp32 (storage) → bf16 (autocast at fwd) | fp32 | fp32 m, v | implicit (weights ARE fp32) |
| **nanochat default path** | bf16 | bf16 | fp32 (PyTorch internal) | NO separate buffer |
| **nanochat FP8 (COAT) path** | bf16 | bf16 → fp32 cast at step | fp8 + fp32 metadata | **fp32 separate buffer** |
| **cppmega CUDA (production)** | bf16 | bf16 | **int8 quantized Muon momentum, uint8 Adam8bit** | **NO master** (`Float16NoMasterOptimizer`) |
| **cppmega.mlx today (Lion)** | fp32 (default!) or bf16 (post-cast) | bf16 | fp32 momentum | unclear (under audit) |

The "canonical" textbook recipe (Megatron-LM, 18 bytes/param non-distributed)
is **NOT** what GB10 actually runs. cppmega CUDA in production runs the
**`Float16NoMasterOptimizer` path**: bf16 weights, bf16 grads, **quantized**
optimizer state (Muon int8 momentum + per-256-block fp32 absmax; Adam8bit
uint8 states + fp32 metadata only for tensors ≥4096 elements).

This is **why GB10 fits**. NAM56R-quarter MBS=4 max_alloc on GB10 = 25 GiB
(per `cppmega/docs/memory_dtype_audit_2026_04_25.md`), not the textbook
40+ GiB.

---

## Per-stack details

### 1. Megatron-LM canonical (textbook)

Per-param byte breakdown:

| Buffer | Bytes | Source |
|---|---:|---|
| bf16 model param | 2 | `Float16Module.bfloat16()` |
| fp32 main param (master) | 4 | `DistributedOptimizer.main_params` |
| fp32 main grad (post-allreduce) | 4 | `--accumulate-allreduce-grads-in-fp32` |
| fp32 m (Adam) | 4 | `optim.AdamW.exp_avg` |
| fp32 v (Adam) | 4 | `optim.AdamW.exp_avg_sq` |
| **Total per param** | **18** | non-distributed |
| With ZeRO-1 (opt sharding) | 6 + 12/d | d = data-parallel size |

For 1.985B params: **18 × 1.985B = 35.7 GB** non-distributed. With ZeRO-1
on 4 ranks: 6 × 1.985B + 12 × 1.985B / 4 = **17.9 GB per rank**.

Source: Megatron-Core `distrib_optimizer.py` + `optimizer.py` +
docs/distrib_optimizer.md.

### 2. PyTorch AMP autocast (standard non-Megatron)

- Weights live as **fp32** in the model storage.
- `torch.amp.autocast(dtype=bfloat16)` casts on the fly during forward to bf16.
- Grads computed in bf16 inside autocast region, scattered back to fp32 at
  param.grad after the autocast exits.
- Optimizer steps fp32 weights with fp32 m, v (PyTorch's AdamW always stores
  moments in fp32 internally, regardless of param dtype).

Per-param: 4 (fp32 weight) + 4 (fp32 grad) + 4 (m) + 4 (v) = **16 bytes**,
no master because the weight IS the master.

This is the "no separate master" interpretation that Hugging Face accelerate
documents.

### 3. DeepSpeed bf16_optimizer

Mirrors Megatron: bf16 model + fp32 master + fp32 m, v + fp32 grads. Same
18 bytes/param. Differs in sharding details (ZeRO-1/2/3 fully shards the
fp32 buffers).

### 4. nanochat default path

bf16 model + bf16 grads + standard `torch.optim.AdamW` and `Muon`. PyTorch's
AdamW keeps `exp_avg` and `exp_avg_sq` in fp32 even when params are bf16
(verified in PyTorch source).

**No separate fp32 master copy of weights** for the default path.

Per-param: 2 (bf16 weight) + 2 (bf16 grad) + 4 (m fp32) + 4 (v fp32) =
**12 bytes** for AdamW path; for Muon's `momentum_buffer` = 4 (fp32) so
2 + 2 + 4 = **8 bytes/param** for 2D matrices.

Muon vs AdamW split predicate (nanochat — `nanochat/gpt.py:7937-7958`):

```python
elif (
    p.ndim == 2
    and not name.endswith(".bias")
    and not name.endswith("_bias")
    and ".router." not in name
    and not name.endswith(".router.weight")
):
    matrix_params.append(p)         # → Muon
else:
    adamw_misc_params.append(p)     # → AdamW
```

**Simpler than Megatron's `_is_nonlinear_or_embedding`** — nanochat uses pure
ndim+name check.

### 5. nanochat FP8 (COAT) path — explicit fp32 master

When `use_fp8_optimizer=True`:

- **Master copy created** (`fp8_optimizer.py:208-230`): for each non-fp32
  parameter, a fp32 master is allocated; mapping `id(model) ↔ id(master)`
  is stored.
- **Step**:
  1. Cast bf16 grad → fp32, write to master.grad
  2. Run inner optimizer (AdamW with COAT FP8 states) on master
  3. Cast fp32 master → bf16, write back to model param

Per-param cost: 2 (bf16 weight) + 4 (fp32 master) + 2 (bf16 grad) + 2.19
(COAT FP8 m, v + scales) = **~10.19 bytes/param**.

This is the **only** nanochat path with an explicit fp32 master.

### 6. cppmega CUDA production — Float16NoMasterOptimizer

The actual configuration used in cppmega/Megatron at GB10:

- `precision: "bf16"` (`nam56r_nemo_recipe.py:166`).
- `use_distributed_optimizer: True` (always for NeMo recipes —
  `nam56r_nemo_recipe.py:236`).
- `use_bf16_no_master_emerging_optimizer: True`,
  `use_bf16_no_master_emerging_fallback_optimizer: True` (`run_profiles.py:186-187`).
- `muon_scalar_optimizer: "adam8bit"` (default `bnb.optim.Adam8bit`) or
  `"lion8bit"` (`run_profiles.py:181`, A/B test in
  `docs/lion8bit_ab_2026_04_25.md`).

**No fp32 main_param**, **no fp32 main_grad**, BF16 grad accumulation
(`--accumulate-allreduce-grads-in-fp32` is **rejected** in cppmega per the
patches noted in `gb10_local_memory_perf_2026_04_25.md:47-51`).

Optimizer state breakdown (per `memory_dtype_audit_2026_04_25.md`):

- Quantized Muon: `quantized_momentum_buffer.data` = int8 (1 byte/param);
  `absmax` = fp32, one per 256-block of params (= 1/64 byte/param).
- Adam8bit: uint8 states (2 × 1 = 2 bytes/param) + fp32 metadata only for
  tensors ≥4096 elements.
- M2RNN, sparse-MLA, DSA: per-op fp32 scale/lse buffers (small).

Per-param effective for 2D matrices: 2 (bf16 weight) + 2 (bf16 grad) +
1.0156 (int8 momentum + absmax) = **~5 bytes/param** for Muon-routed.

**This is why MBS=4 NAM56R-quarter fits in 25 GiB max_alloc on GB10.**

### 7. cppmega.mlx today

Per `docs/research/cppmega_mlx_optimizer_precision_audit.md`:

- Model loads as **fp32 by default** (1.797B fp32 = 6.69 GiB) — this is wrong;
  master plan says bf16. Cast to bf16 happens **post-build** via
  `model.set_dtype()` at the call site, if at all.
- `LionFP32Moments` / `AdamWFP32Moments` override upstream MLX init to
  force fp32 moments even when params are bf16. **Confirmed correct policy
  for numerical stability.**
- Lion observed state at runtime: **11.58 GiB** for 1.797B params. Expected
  fp32 momentum alone: 1.797B × 4 = 7.19 GiB. The extra ~4.4 GiB is either
  a transient fp32 master copy (created during step), or a persistent master
  copy held in optimizer.state, **or** intermediate cast workspace. Cannot
  distinguish without inspecting the live state dict.

---

## Decision matrix — what cppmega.mlx should adopt

The user explicitly wants "то же самое тут на macOS" (same thing here on macOS)
as cppmega CUDA does. That means: **`Float16NoMasterOptimizer` analog for
MLX**.

| Element | cppmega CUDA | cppmega.mlx target | Gap |
|---|---|---|---|
| Weights | bf16 | bf16 | **fp32 default → bf16 default** (build-time, not post-cast) |
| Grads | bf16 | bf16 | already bf16 if param is bf16 (MLX autograd matches param dtype) |
| Optimizer moments | int8 / uint8 quantized + fp32 absmax/metadata | fp32 currently → ideally int8/uint8 quantized | **No bnb-equivalent quantized optimizer in MLX**. Building one is post-M0 work. |
| Master copy of weights | NO | NO | Audit suggests possible accidental master in current Lion state — fix. |

**Three paths forward, ordered by complexity:**

### Path 1 — Match canonical Megatron textbook (NOT what GB10 does)
- bf16 weights + fp32 master + fp32 m, v = 18 bytes/param = 35.7 GiB for 1.985B.
- Adds 4 bytes/param of master we don't need.
- **Don't do this.** Web research action item #1 says explicitly: don't add a master.

### Path 2 — Match nanochat default path (closest to what we have)
- bf16 weights + bf16 grads + fp32 m, v in MLX optimizer.state.
- **No master copy.**
- Per-param for AdamW: 12 bytes = 23.8 GiB for 1.985B.
- Per-param for Lion: 8 bytes = 15.9 GiB for 1.985B.
- Per-param for Muon (matrix): 8 bytes for the 2D part + AdamW for scalars.
- **This is the action item.** Web research and local audit converge.

### Path 3 — Match cppmega CUDA `Float16NoMasterOptimizer` (target)
- bf16 weights + bf16 grads + **quantized 8-bit optimizer states**.
- Per-param: ~5 bytes/param matrices = 9.9 GiB for 1.985B Muon-routed.
- **No bnb in MLX**. Need to write our own equivalent of `Adam8bit` and
  `quantized_muon_momentum`. Or vendor `bitsandbytes` semantics into MLX
  custom kernels. Significant work — likely Stream F or later.

---

## Action items (concrete, ordered)

### Now (Path 2 — match nanochat default + fix dtype default)

1. **Fix model build dtype default**: `cppmega_mlx/recipes/model_factory.py::local_gb10_quarter` should accept `dtype=mx.bfloat16` and pass it to `HybridTinyLM.__init__`. Add `dtype` parameter to `HybridLMConfig` and apply at construction time, not via `model.set_dtype()` post-hoc. **Saves 3.35 GiB on weight allocation.**

2. **Verify no master copy of weights is created in `LionFP32Moments` / `AdamWFP32Moments`**: read the optimizer.state dict at runtime, dump every key and dtype. If a `master_param` or fp32 weight buffer exists, remove it. **Saves up to 4.4 GiB on Lion.** This is the explanation for the observed 11.58 GiB Lion state.

3. **Add a runtime parity test** asserting:
   ```python
   for k, v in opt.state.items():
       if k in ("step", "lr", ...): continue
       assert v.dtype == mx.float32, f"{k}: {v.dtype}"
       # AND no key shaped like a parameter — only m/v moments allowed
   ```

4. **Update `docs/mlx_port_master_plan.md` Stream E addendum** with a precision-policy section citing this decision: "We follow nanochat's default path. No fp32 master. fp32 m, v (mandatory for numerical stability per Megatron-LM rationale). bf16 weights + bf16 grads. Path 3 (quantized states) is post-M0."

### Next (Path 3 — match cppmega CUDA over time)

5. **Vendor `bnb.optim.Adam8bit` semantics into MLX**: implement an `Adam8bit` MLX optimizer with uint8 quantized states + per-256-block fp32 absmax. Estimated effort: ~200-300 LOC plus a fast Metal kernel for the quantize/dequantize step. Target: post-M0, before Stream F (heterogeneous distributed run).

6. **Vendor `quantized_muon_momentum` semantics**: int8 quantized momentum buffer with fp32 absmax. Pair with our existing `MuonWithNSCarrier`. Target: same milestone as #5.

7. **Add `Float16NoMasterOptimizer` MLX equivalent** as the wrapping
   MultiOptimizer that combines #5 and #6, gated on
   `use_bf16_no_master_emerging_optimizer=True` flag for parity with cppmega
   CUDA configs.

### Don't do (decided against)

- **Don't** add a fp32 master copy of weights for the default path. It's
  Megatron textbook but cppmega CUDA explicitly disables it via
  `use_bf16_no_master_emerging_optimizer=True`. We match production, not
  textbook.
- **Don't** ship a `GradScaler` / loss-scaling path. bf16 doesn't need it
  (exponent range is fp32-like). nanochat doesn't have it either.
- **Don't** chase fp8 training on M4 Max for the default training run.
  Apple Silicon (M1-M4) has no native FP8 ALU; software emulation is in
  scope only for the sparse-MLA FP8 quantization path (already shipped via
  Path B), not for general bf16-replacement training. M5/M6 will revisit.

---

## Memory accounting target after Path 1+2 land

Per-param (bf16 model, fp32 m, v, no master):

| Optimizer | bytes/param | Total for 1.985B | Δ vs current |
|---|---:|---:|---:|
| Lion | 2 (w) + 2 (g) + 4 (m) | **15.9 GiB** | -8 GiB vs observed 11.58 + 6.69 fp32 = 18.27 |
| AdamW | 2 + 2 + 4 (m) + 4 (v) | **23.8 GiB** | n/a yet measured |
| MuonAdamWMulti (rough — split per-tensor) | 14-19 GiB | **27-33 GiB** | matches GB10 receipt direction |

After Path 3 (quantized states):

| Optimizer | bytes/param | Total for 1.985B |
|---|---:|---:|
| Adam8bit | 2 + 2 + 1 + 1 (8-bit m, v) | **11.9 GiB** |
| Muon int8 + Adam8bit fallback | ~5 (2D Muon) / ~6 (scalars) | **9.9-11.9 GiB** |

GB10 target: **~10 GiB optimizer** at MBS=4 NAM56R-quarter. We can match.

---

## References

- `docs/research/bf16_fp32_master_moments_pattern.md` (web research)
- `docs/research/cppmega_mlx_optimizer_precision_audit.md` (local audit)
- `cppmega/docs/memory_dtype_audit_2026_04_25.md` (CUDA receipt)
- `cppmega/docs/lion8bit_ab_2026_04_25.md` (CUDA Lion8bit A/B)
- `cppmega/cppmega/recipes/run_profiles.py` (no-master flags)
- `nanochat/nanochat/gpt.py` (default optimizer setup)
- `nanochat/nanochat/fp8_optimizer.py` (COAT master pattern)
- `bench/baselines/memory_probe_local_gb10_quarter_t1024.json` (fp32 receipt)
- `bench/baselines/memory_probe_local_gb10_quarter_bf16.json` (bf16 receipt)
