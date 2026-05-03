# bf16 + fp32 master moments — canonical pattern

Date: 2026-05-03

Scope: research lane for porting the canonical "bf16 model weights + fp32
master moments" recipe to MLX on Apple Silicon. This is documentation only;
it does not modify any source code outside `docs/research/`.

The shorthand "fp32 master moments" is overloaded in the field. The two
distinct fp32 buffers in production stacks are:

1. **fp32 master weights** — a separate fp32 copy of the parameters, owned
   by the optimizer, that the actual `param -= lr * update` step writes
   into. The bf16 model weights are then refreshed from this fp32 copy.
2. **fp32 optimizer moments** — the running averages `m`, `v` (Adam), the
   single momentum buffer `m` (Lion), or `v` + Newton-Schulz carrier (Muon).

Megatron-LM, DeepSpeed, and PyTorch AMP all keep both in fp32 by default.
"Master moment" in our codebase refers specifically to the second item —
the optimizer state — but to use the recipe correctly we must understand
how each framework decides whether the first item (master weight copy) is
also required.

## What gets stored at each precision

Per-framework dtype matrix for the canonical bf16 mixed-precision recipe.
"sep fp32 weight" means a distinct fp32 buffer separate from the model
parameter; "promotion" means an inline cast to fp32 inside the optimizer
step.

| Framework / mode                                           | Model weight (storage) | Activation (compute) | Gradient (accum) | fp32 master weight | Optimizer moments |
|------------------------------------------------------------|------------------------|----------------------|------------------|--------------------|-------------------|
| Megatron-LM `Float16OptimizerWithFloat16Params`            | bf16                   | bf16                 | fp32 (`main_grad`) | sep fp32           | fp32              |
| Megatron-LM `DistributedOptimizer`                         | bf16                   | bf16                 | fp32 (`main_grad`) | sep fp32 (sharded) | fp32 (sharded)    |
| Megatron-LM precision-aware optimizer (newer)              | bf16                   | bf16                 | bf16 or fp32     | bf16 + 16-bit fp32 remainder | configurable bf16/fp32 |
| DeepSpeed `BF16_Optimizer` (ZeRO 0/1/2/3 with bf16)        | bf16 (`bf16_groups`)   | bf16                 | configurable (`grad_acc_dtype`, default fp32) | sep fp32 (`fp32_groups_flat_partition`) | fp32 (inherited) |
| PyTorch native `torch.amp` autocast (no master)            | fp32                   | bf16 (cast)          | fp32             | not separate (params already fp32) | fp32 |
| PyTorch FSDP `MixedPrecision(param_dtype=bf16)`            | bf16 (shard)           | bf16                 | fp32 reduce      | fp32 ("full"-shard) optional | fp32 |
| HF `accelerate` `mixed_precision="bf16"`                   | fp32 (params)          | bf16 (cast)          | fp32             | n/a (param IS fp32)| fp32 |
| MLX `mlx.optimizers.AdamW` (upstream as-is)                | bf16 (if model is bf16)| bf16                 | bf16 (matches param dtype) | none             | matches param dtype (bf16!) |
| `cppmega_mlx.training.optimizers.AdamWFP32Moments` (ours)  | bf16                   | bf16                 | bf16 (cast to fp32 inside step) | none — inline promotion | fp32 |

Two big takeaways:

- The choice between **PyTorch AMP-style fp32 params + bf16 cast** and
  **Megatron/DeepSpeed-style bf16 params + sep fp32 master** is a memory
  optimization. Both are mathematically equivalent: the optimizer step
  always runs against an fp32 buffer. Megatron and DeepSpeed save 4N bytes
  of activation memory by storing the model in bf16; AMP keeps params fp32
  to avoid managing two buffers.
- `mlx.optimizers.AdamW` upstream is *not* mixed-precision-safe out of the
  box. `init_single` allocates `state["m"] = mx.zeros_like(parameter)` (see
  `optimizers.py:507-510`), so if the model is bf16 the optimizer moments
  inherit bf16 too. There is no master-weight buffer at all.

## Memory accounting

Per-parameter byte cost for a model with N params, training in bf16
mixed precision. These match the Megatron Core distributed optimizer docs:

| Pattern                                                         | bf16 weight | grad | fp32 master | fp32 m+v | total bytes/param |
|-----------------------------------------------------------------|-------------|------|-------------|----------|-------------------|
| **A.** Megatron-LM canonical (non-distrib)                      | 2           | 4    | 4           | 8        | **18**            |
| **B.** Megatron-LM canonical, distributed across `d` DP ranks   | 2           | 4    | 4/d         | 8/d      | **6 + 12/d**      |
| **C.** Megatron precision-aware (bf16 grads + bf16 moments)     | 2           | 2    | 4           | 4        | **12**            |
| **D.** DeepSpeed BF16 + fp32 grad accum + ZeRO-1 (master sharded) | 2         | 4    | 4/d         | 8/d      | **6 + 12/d**      |
| **E.** PyTorch AMP / accelerate `bf16` (params stay fp32)       | 4 (=fp32)   | 4    | 0 (param IS master) | 8 | **16**            |
| **F.** No master, fp32 moments only (textbook bf16 + Adam)      | 2           | 2    | 0           | 8        | **12**            |
| **G.** Pure bf16 (everything bf16, no master, no fp32 moments)  | 2           | 2    | 0           | 4        | **10** (unstable) |

The canonical Megatron / DeepSpeed answer for production bf16 training is
**Pattern A** (non-distributed: 18 bytes/param) or **Pattern B** (distributed:
6 + 12/d bytes/param). The "16 bytes" number you sometimes see quoted comes
from the fp16 case (Pattern with fp16 grads) or from omitting the separate
gradient buffer when grads are accumulated directly into `main_grad`.

Source for table: NVIDIA Megatron Core distributed optimizer docs at
https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/dist_optimizer.html
explicitly publishes the column "bf16 parameters, fp32 gradients: 18 bytes
non-distributed, 6 + 12/d distributed."

When is the master copy redundant?

- If params are already fp32 (Pattern E), the master copy *is* the param
  buffer. Adding a second fp32 buffer is pure cargo-culting.
- If you only need to support inference-after-train and never resume, you
  can skip the master and reload-from-bf16 (Pattern F) — but you accept
  a small drift over thousands of steps because the bf16 round-trip during
  `param.copy_(fp32_master)` is lossless only if you store the missing 16
  mantissa bits somewhere (the precision-aware optimizer trick).
- If the optimizer never crosses the bf16 boundary (e.g. you cast the bf16
  param to fp32 inside the step, do the update in fp32, cast back) you get
  the math benefit of the fp32 master without allocating a separate buffer.
  This is the path our `AdamWFP32Moments` already takes.

## Reference implementations

### Megatron-LM `Float16OptimizerWithFloat16Params`

File: `megatron/core/optimizer/optimizer.py` on `main`.

The optimizer keeps three parallel parameter group lists:

- `float16_groups`: original fp16/bf16 model parameters
- `fp32_from_float16_groups`: fp32 master copies of the bf16/fp16 params
- `fp32_from_fp32_groups`: any params that were already fp32

Master copy is created with:

```python
main_param = param.detach().clone().float()
param_group['params'][i] = main_param
param.main_param = main_param
```

After backward, fp16/bf16 grads are promoted to fp32 and routed to the
master:

```python
if hasattr(model_param, 'main_grad'):
    main_param.grad = model_param.main_grad.float()
```

Then the wrapped Adam optimizer steps against `main_param` (fp32). Finally
the bf16 model param is refreshed from the updated fp32 master.

### Megatron-LM `DistributedOptimizer`

File: `megatron/core/optimizer/distrib_optimizer.py` on `main`.

For each model param shard, a master shard is allocated as:

```python
shard_main_param = shard_model_param.clone().float()
```

Optimizer moments are allocated with explicit dtype config:

```python
tensors = {
    'exp_avg':    init_shard(self.config.exp_avg_dtype),
    'exp_avg_sq': init_shard(self.config.exp_avg_sq_dtype),
}
```

with config defaults `exp_avg_dtype=torch.float32`,
`exp_avg_sq_dtype=torch.float32`, `main_grads_dtype=torch.float32`,
`params_dtype=torch.float32` (see
`megatron/core/optimizer/optimizer_config.py`).

### DeepSpeed `BF16_Optimizer`

File: `deepspeed/runtime/bf16_optimizer.py` on `master`. No module docstring;
the design is in code comments.

```python
self.bf16_groups          # list of bf16 model param groups
self.bf16_groups_flat     # contiguous bf16 buffer per group
bf16_dp_partitions = [
    self.bf16_groups_flat[i].narrow(0, dp_index * partition_size, partition_size)
    for dp_index in range(real_dp_world_size)
]

# create fp32 params partition
self.fp32_groups_flat_partition.append(
    bf16_dp_partitions[partition_id].clone().float().detach()
)
```

The wrapped optimizer (e.g. `AdamW`) is given the fp32 partitions as its
parameters, so its internal state (`exp_avg`, `exp_avg_sq`) is created in
fp32 by virtue of operating on fp32 inputs. After step, bf16 partitions
are refreshed:

```python
bf16_partitions[partition_id].data.copy_(fp32_partition.data)
```

Gradient accumulation precision is configurable via `grad_acc_dtype` and
asserted to be one of `[torch.float32, torch.bfloat16]`. ZeRO-1 sharding
requires `grad_acc_dtype=fp32` for bf16 training.

### Upstream MLX `mlx.optimizers.AdamW`

File:
`/Volumes/external/sources/cppmega.mlx/.venv/lib/python3.13/site-packages/mlx/optimizers/optimizers.py`.

```python
def init_single(self, parameter: mx.array, state: dict):
    state["m"] = mx.zeros_like(parameter)   # inherits param dtype
    state["v"] = mx.zeros_like(parameter)   # inherits param dtype

def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
    lr = self.learning_rate.astype(gradient.dtype)
    b1, b2 = self.betas
    m = state["m"]
    v = state["v"]
    m = b1 * m + (1 - b1) * gradient
    v = b2 * v + (1 - b2) * mx.square(gradient)
    state["m"] = m
    state["v"] = v
    ...
```

Three problems for bf16 training:

1. `mx.zeros_like(parameter)` allocates moments in the param dtype. On a
   bf16 model the optimizer state is bf16 — Pattern G in the table above.
2. `lr.astype(gradient.dtype)` makes the LR bf16 too. Tiny LR values
   (`5e-5`) round to the nearest bf16, and `1 - b1 = 0.1` against a bf16
   gradient cannot represent the small per-step deltas accurately.
3. `mx.square(gradient)` is computed in bf16, so `v` accumulates squared
   bf16 gradients with bf16 precision. Loss-of-significance is fast.

There is no master-weight mechanism anywhere in `mlx.optimizers`. The only
precision-control hook is what each subclass does in `init_single`/
`apply_single`.

## What MLX gives us today

Observed behaviour from `mlx.optimizers` (verified against
`mlx==0.31.1` in the local `.venv`):

- `Optimizer.init_single` is the only hook for state allocation. Subclasses
  decide moment dtype.
- `Optimizer.apply_single` runs in whatever dtype `gradient` arrives in.
  There is no autocast.
- `MultiOptimizer` exists for filter-based group routing (used by Muon +
  AdamW splits) but exposes state as `{"states": [...]}`, which our
  `MuonAdamWMulti` deliberately does not subclass — the audit tooling needs
  named buckets.
- `clip_grad_norm` runs in fp32 if the input grads are fp32, otherwise in
  the input dtype. It does not promote.

What MLX does *not* give us:

- No fp32 master-weight buffer in any optimizer.
- No `param_dtype` / `grad_dtype` / `exp_avg_dtype` knobs analogous to
  Megatron's `OptimizerConfig`.
- No `te.fp8_autocast` or `torch.amp` analogue. mlx-lm's `lora.py` and
  `tuner/trainer.py` confirm this: the only precision touch is
  `ce.astype(mx.float32)` on the cross-entropy reduction
  (`mlx_lm/tuner/trainer.py:97`); the optimizer is created with
  `opt = opt_class(learning_rate=lr, **optimizer_config)` and inherits the
  loaded model's dtype.

So we have to write the precision policy ourselves at the optimizer
subclass level. That is exactly what
`cppmega_mlx/training/optimizers.py` does for `AdamWFP32Moments`,
`LionFP32Moments`, and `MuonWithNSCarrier`.

## Why fp32 moments are required

bf16 has an 8-bit exponent (same as fp32) but only **7 bits of mantissa**
(vs fp32's 23). For optimizer state this matters in three places:

1. **Second moment `v` underflow.** `v = b2 * v + (1 - b2) * g^2` with
   `b2 = 0.999` means the per-step delta is ~`0.001 * g^2`. When `g` is
   already small (~1e-4 typical mid-training), `g^2 ~ 1e-8` and the delta
   is `1e-11`. bf16's smallest representable delta against `v ~ 1e-3` is
   roughly `v * 2^-7 ~ 8e-6`. The update vanishes — `v` becomes stale.
2. **First moment `m` round-off.** `m` accumulates a weighted average of
   the gradient. With `b1 = 0.9`, the implicit time constant is ~10 steps,
   so `m` reaches its "asymptotic" magnitude quickly and small per-step
   updates against a stable `m` round to zero in bf16.
3. **Master-weight stale updates.** `w -= lr * m / (sqrt(v) + eps)`. With
   `lr = 1e-4` and a typical Adam update on the order of 1, the per-step
   weight delta is ~1e-4. If `w ~ 0.5` is bf16, the delta is ~`0.5 * 2^-7
   = 4e-3` to round; deltas an order of magnitude smaller silently
   disappear. Over thousands of steps the model fails to fit. This is the
   "stale weights" problem, which is precisely what the fp32 master copy
   was introduced to fix.

Empirical evidence cited by the major frameworks:

- Mixed Precision Training, Micikevicius et al. 2017
  (https://arxiv.org/abs/1710.03740) — original fp32 master weight result.
- FP8-LM, Peng et al. 2023 (https://arxiv.org/abs/2310.18313) — extends
  the master-weight argument to fp8, motivates per-tensor scaling.
- DeepSpeed `BF16_Optimizer` source asserts `grad_acc_dtype in [fp32,
  bf16]` and the ZeRO-1+bf16 path *requires* fp32 grad accumulation,
  documented in the GitHub bug-fix history (see issue #2734).
- Hugging Face `accelerate` and PyTorch FSDP both ship fp32 reductions for
  bf16 training by default. The reduction (sum of grads across DP ranks)
  is the noisiest place to lose mantissa bits.

The repo's existing `AdamWFP32Moments`/`LionFP32Moments` already do the
right thing for items (1) and (2) by allocating moments in fp32 and
casting the gradient to fp32 *inside* `apply_single`. Item (3) is handled
by the inline weight promotion `parameter.astype(mx.float32) * (1 - lr *
weight_decay)` followed by `.astype(parameter.dtype)` at the end — the
optimizer math runs in fp32, the storage stays bf16, no separate master
buffer is required.

## What we need to do for cppmega.mlx

The action items are ordered by risk to training stability.

1. **No separate fp32 master-weight buffer is required for our setup.**
   `AdamWFP32Moments.apply_single` already promotes `parameter` to fp32
   inline before the Adam step and casts the result back to `parameter.dtype`
   at the end. This is mathematically equivalent to "fp32 master + bf16
   model" for a single-rank, non-sharded optimizer (which is our case on
   M4 Max). Adding a separate fp32 weight buffer would double parameter
   memory (Pattern E in the table) for zero numerical benefit. **Decision:
   stay with inline promotion. Do not allocate a master buffer.** Document
   this as the explicit design choice in `cppmega_mlx/training/optimizers.py`.

2. **`AdamWFP32Moments` is correct, but the parity audit should verify
   moments are fp32 *after* `apply_gradients` runs at least once.**
   `init_single` allocates fp32 moments, but if a future change in MLX adds
   a tree-cast or a checkpoint round-trip that materialises moments in the
   param dtype, the audit needs to catch it. The existing
   `adamw_moment_dtypes_ok(state, required_dtype="float32")` helper in
   `cppmega_mlx/training/optimizers.py:500-508` already handles this; just
   make sure the parity test runs it on a *post-step* state, not on
   `optimizer.state` before the first call.

3. **`LionFP32Moments` is correct.** Lion only carries one momentum buffer
   (`m`), allocated as fp32 in `init_single`. The inline promotion in
   `apply_single` mirrors the AdamW path. Verify the same post-step audit
   applies; the moment-key set is `("m",)` not `("m", "v")`.

4. **`MuonAdamWMulti`/`MuonWithNSCarrier` is *almost* correct but has a
   subtle bug to verify.** `MuonWithNSCarrier.init_single` allocates
   `state["v"] = mx.zeros(parameter.shape, dtype=_muon_state_dtype())`,
   which is fp32 — good. The Newton-Schulz iteration runs in either fp32
   or bf16 according to `ns_carrier`, and the *output* of the NS step is
   cast back to `update.dtype` (fp32 momentum dtype) before the parameter
   update. This is correct: momentum stays fp32, only the NS polynomial
   uses the carrier dtype. **Action: add a parity test that asserts
   `state["v"].dtype == mx.float32` for both Muon and AdamW buckets after
   one `MuonAdamWMulti.apply_gradients` call, regardless of the
   `ns_carrier` setting.**

5. **Add a documentation block in `cppmega_mlx/training/optimizers.py` that
   names the precision pattern explicitly.** Currently the docstrings
   mention "fp32 moment state for bf16-weight training" but do not state
   "no separate master copy is allocated; fp32 promotion is inline." That
   sentence should appear once at module level so future readers don't
   re-add a master buffer they think is missing.

6. **Defensive guard: assert lr is fp32 inside our `apply_single`.** Upstream
   `Optimizer._maybe_schedule` stores `lr = mx.array(learning_rate)` which
   defaults to fp32, but `apply_single` then does `lr.astype(gradient.dtype)`
   in upstream Adam. Our wrappers already do `lr.astype(mx.float32)`
   instead, but a small `assert lr.dtype == mx.float32` in the parity tests
   would catch a future regression where someone passes a bf16 LR through
   a scheduler.

7. **Out of scope, but worth noting for a future ZeRO-style port: if/when
   we shard optimizer state across multiple Apple devices** (e.g. M4 Max +
   M4 Pro studio), Pattern B becomes attractive (6 + 12/d bytes/param) and
   we *would* need a real fp32 master buffer because the bf16 model and
   the fp32 master have to live on different devices. For single-device
   training on the M4 Max we never hit this regime.

8. **Do not adopt the precision-aware-optimizer trick (bf16 moments + 16-bit
   fp32 remainder) yet.** It's a Megatron-Bridge recent feature that drops
   memory from 18 to ~12 bytes/param at the cost of more complex state
   serialisation. We don't need it on a 128 GiB unified-memory M4 Max for
   the model sizes we run, and the savings tradeoff is brittle for Lion
   and Muon (which have different moment-count profiles than AdamW). Park
   it as a future experiment behind an env-flag if memory becomes a
   bottleneck.

Sources:

- Megatron-LM `distrib_optimizer.py` and `optimizer.py` (NVIDIA, GitHub `main`).
- Megatron-Core distributed optimizer memory table:
  https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/features/dist_optimizer.html
- Megatron-LM optimizer config fields (`exp_avg_dtype`, `exp_avg_sq_dtype`,
  `main_grads_dtype`, `params_dtype`):
  `megatron/core/optimizer/optimizer_config.py`.
- DeepSpeed `bf16_optimizer.py` (microsoft/DeepSpeed, GitHub `master`).
- DeepSpeed config docs: https://www.deepspeed.ai/docs/config-json/
- Mixed Precision Training, Micikevicius et al. 2017
  (https://arxiv.org/abs/1710.03740).
- FP8-LM, Peng et al. 2023 (https://arxiv.org/abs/2310.18313).
- MLX optimizers source: `.venv/lib/python3.13/site-packages/mlx/optimizers/optimizers.py`.
- MLX-LM tuner source: `.venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py`,
  `.venv/lib/python3.13/site-packages/mlx_lm/lora.py`.
- Local cppmega_mlx implementation:
  `cppmega_mlx/training/optimizers.py` (`AdamWFP32Moments`,
  `LionFP32Moments`, `MuonWithNSCarrier`, `MuonAdamWMulti`).
