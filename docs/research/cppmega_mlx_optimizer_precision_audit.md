# CPPMega MLX Optimizer Precision Audit

## Section A: Current State of Optimizers

### AdamWFP32Moments (lines 44–66)
- **Momentum state (m)**: `mx.zeros(parameter.shape, dtype=mx.float32)`
- **Velocity state (v)**: `mx.zeros(parameter.shape, dtype=mx.float32)`
- **Weight parameter**: Passed through as-is; can be bf16 or fp32
- **Gradient handling**: Cast to fp32 in `apply_single()` before Adam math (line 61)
- **Weight return**: Cast back to `parameter.dtype` after update (line 65)

**Byte accounting per 1.797B params (local_gb10_quarter):**
- Parameters (fp32): 1.797B × 4 = 7.19 GiB
- Parameters (bf16): 1.797B × 2 = 3.59 GiB
- Moment state (m, fp32): 1.797B × 4 = 7.19 GiB
- Velocity state (v, fp32): 1.797B × 4 = 7.19 GiB
- **Total AdamW (bf16 weights + fp32 moments)**: 3.59 + 7.19 + 7.19 = **17.97 GiB**

### LionFP32Moments (lines 87–112)
- **Momentum state (m)**: `mx.zeros(parameter.shape, dtype=mx.float32)` (1× only, not 2×)
- **Weight parameter**: Passed through as-is; can be bf16 or fp32
- **Gradient handling**: Cast to fp32 in `apply_single()` (line 108)
- **Weight return**: Cast back to `parameter.dtype` (line 112)

**Byte accounting per 1.797B params:**
- Parameters (bf16): 3.59 GiB
- Momentum state (m, fp32): 1.797B × 4 = **7.19 GiB**
- **Total Lion state: 7.19 GiB**
- **Expected (bf16 params + fp32 m): 3.59 + 7.19 = 10.78 GiB**

### MuonWithNSCarrier (lines 254–328)
- **State (v)**: `mx.zeros(parameter.shape, dtype=_muon_state_dtype())` → always fp32
- **Newton-Schulz carrier**: Configurable via `ns_carrier` env var ("fp32" or "bf16")
  - Momentum accum and update always stay in fp32
  - NS iterations run in carrier dtype (line 315–320)
- **Weight return**: Cast back to `parameter.dtype` (line 328)

**Byte accounting (2-D weights only):**
- Parameters (bf16): 1.797B × 2 = 3.59 GiB (estimated; routing is selective)
- Momentum state (v, fp32): ~3.59 GiB (same shape as params, but fp32)
- NS carrier matrices (bf16 or fp32): Temporary allocation during NS iterations

### MuonAdamWMulti (lines 331–414)
- Routes 2-D linear weights → Muon
- Routes embeddings, lm_head, scalars, 3-D+ tensors → AdamW
- State dict has explicit "muon" and "adamw" buckets for auditing (lines 394–404)

---

## Section B: Comparison to Upstream MLX

### Upstream optim.Adam (line 507–510)
```python
def init_single(self, parameter: mx.array, state: dict):
    state["m"] = mx.zeros_like(parameter)
    state["v"] = mx.zeros_like(parameter)
```
- Moments created with **same dtype as parameter** (via `zeros_like`)
- For bf16 params: m, v are bf16 (2 bytes each)
- For fp32 params: m, v are fp32 (4 bytes each)

### Upstream optim.AdamW (line 580–588)
- Inherits `init_single` from Adam → moments match param dtype
- Calls parent `apply_single` with weight-decayed param
- **No explicit fp32 casting of gradients or moments**

### Upstream optim.Lion (line 689–705)
```python
def init_single(self, parameter: mx.array, state: dict):
    state["m"] = mx.zeros_like(parameter)
```
- Momentum created with **same dtype as parameter**
- For bf16 params: m is bf16
- Update computed via `sign(c)` on bf16 momentum if param is bf16

### Upstream optim.Muon (line 892–894)
```python
def init_single(self, parameter: mx.array, state: dict):
    state["v"] = mx.zeros_like(parameter)
```
- Momentum created with **same dtype as parameter**
- NS iterations happen on whatever dtype momentum has

### CPPMega Override Strategy
CPPMega's `AdamWFP32Moments`, `LionFP32Moments`, and `MuonWithNSCarrier` **force optimizer state to fp32** regardless of parameter dtype. This is intentional:
- Upstream MLX optimizers keep moments in parameter dtype for memory efficiency
- CPPMega prioritizes **numerical stability** by maintaining fp32 m, v accumulators even when parameters are bf16
- This matches Megatron-LM's mixed-precision pattern where gradients and moments stay in higher precision

---

## Section C: Memory Probe Interpretation

### Measured Data
**fp32 model + Lion (memory_probe_local_gb10_quarter_t1024.json):**
- Model: 6.692 GiB (fp32)
- Opt state: **11.579 GiB**
- After opt init: 18.272 GiB active (model + opt state)

**bf16 model + Lion (memory_probe_local_gb10_quarter_bf16.json):**
- Model: 3.346 GiB (bf16)
- Opt state: Not directly reported but inferred from "after opt init"
- After opt init: 14.926 GiB active

### Analysis of 11.579 GiB Lion State (fp32 model)

For fp32 parameters, `LionFP32Moments.init_single()` still creates fp32 momentum:
```
state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
```

**Naive expectation:** 1.797B params × 4 bytes = 7.19 GiB momentum

**Observed:** 11.579 GiB

**Calculation of the gap:**
- Observed: 11.579 GiB
- Expected (m only, fp32): 7.19 GiB
- Excess: 11.579 - 7.19 = **4.389 GiB**

**Hypothesis:**
The excess ≈4.4 GiB matches approximately **fp32 master copy of weights** (expected: 7.19 GiB, but only a portion is materialized during optimizer state serialization). However, we cannot determine the exact breakdown from probes alone.

**Possibility A (unlikely):** No separate master copy; the 11.579 comes from:
- Momentum (m): 7.19 GiB (fp32)
- Transient casting buffers: ≤4.4 GiB
- This would mean the design does NOT preserve a persistent master copy of weights

**Possibility B (likely):** A persistent fp32 master copy is stored in optimizer state:
- Momentum (m): 7.19 GiB
- Master weights (fp32): ≈4.4 GiB (partial, or metadata overhead)
- Total: 11.579 GiB (matches observation)

**Verdict:** Can't determine without running `optimizer.state` inspection at runtime. The code does not show an explicit master weight copy in `LionFP32Moments`, but MLX's state tree might embed one implicitly.

---

## Section D: HybridLM Build Dtype Issue

### Build Flow
1. `local_gb10_quarter()` → calls `local_gb10_quarter_profile().build_model()`
2. `build_model()` → returns `HybridTinyLM(self.hybrid_config())`
3. `HybridTinyLM.__init__()` → creates embeddings, blocks, lm_head **without dtype specification**
4. All parameters default to **fp32** (MLX default)

### Where bf16 is Applied (if at all)
In bench_local_gb10_quarter_throughput.py (line 536):
```python
model = local_gb10_quarter(grad_checkpoint=grad_checkpoint)
model.set_dtype(mx.bfloat16)  # <-- Applied AFTER model build
```

### The Problem
1. Model built as fp32 (0.3s per memory_probe_t1024.json)
2. Dtype cast happens **after** model allocation
3. Optimizer init called on the **now-bf16** model (0.7s per log, "after optimizer init (lion, 0.7s)")
4. Result: Optimizer sees bf16 params but the weight-decay and casting logic assumes may operate on original dtype

### Missing Fix
In `cppmega_mlx/models/hybrid_lm.py`, `HybridTinyLM.__init__()` should:
- Accept an optional `dtype` parameter (default: `mx.float32` for legacy compat)
- Apply it immediately after parameter creation, before returning

Example patch location (around line 349):
```python
def __init__(self, config: HybridTinyConfig | None = None, dtype: mx.Dtype | None = None):
    super().__init__()
    self.config = config or HybridTinyConfig()
    cfg = self.config
    self.pattern = cfg.expanded_pattern()
    
    # Build all parameters
    self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    # ... rest of module creation ...
    
    # Apply dtype uniformly if requested
    if dtype is not None and dtype != mx.float32:
        self.set_dtype(dtype)
```

### Downstream: model_factory.py
Similarly, `local_gb10_quarter()` should accept `dtype` kwarg and pass it through:
```python
def local_gb10_quarter(dtype: mx.Dtype | None = None, **hybrid_config_overrides) -> HybridTinyLM:
    profile = local_gb10_quarter_profile(**profile_overrides).build_model(
        dtype=dtype,
        **hybrid_config_overrides
    )
```

---

## Section E: Action Items

1. **Runtime Optimizer State Inspection:**
   Add diagnostic code to print `optimizer.state` structure with dtype and shape info after init on bf16 model. This will reveal whether a master weight copy exists and account for the 11.579 GiB Lion state gap.

2. **Add dtype Parameter to HybridTinyLM:**
   Modify `HybridTinyLM.__init__(dtype=None)` to accept and apply a dtype arg immediately post-construction, before returning to caller.

3. **Thread dtype Through model_factory.py:**
   Update `local_gb10_quarter()` and `ModelFactoryProfile.build_model()` to accept `dtype` kwarg and pass it to HybridTinyLM.

4. **Benchmark with Unified Build-Time dtype:**
   Re-run memory_probe with `local_gb10_quarter(dtype=mx.bfloat16)` to measure whether build-time casting vs. post-build casting changes peak memory or state sizes.

5. **Document Numerics Decision:**
   If a fp32 master copy of weights IS being stored, add a comment in `LionFP32Moments` explaining: "We maintain a persistent fp32 master copy to preserve gradient accumulation precision. MLX's fp32-weighted updates marginally outperform bf16 rounding in loss convergence tests (cite or link to internal parity report)."

