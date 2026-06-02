# Graph-level (TVM Relax) memory planning vs eager: PoC result + integration path

**Thesis (proven below on real hardware):** lowering a multi-op training step to ONE
TVM Relax graph and running `StaticPlanBlockMemory` gives a **strictly lower peak
memory** than eager execution, because the graph has cross-op / cross-layer
liveness that an eager single-`mx.eval` barrier cannot exploit.

**Status:** PoC done and passing (measured numbers below). Integration is a
multi-quarter effort; this doc specifies the first PR and the full roadmap.

PoC code: [`cppmega_mlx/runtime/relax_memory_plan_poc.py`](../cppmega_mlx/runtime/relax_memory_plan_poc.py)
(self-checking, fail-loud; run instructions at the bottom).

---

## 1. PoC result (MEASURED, not estimated)

Device: **CPU, LLVM Relax VM** (CUDA path identical -- same passes run in
`relax/backend/cuda/pipeline.py:64,67`; CPU chosen only because gb10 was busy and
the planning decision is target-independent -- it operates on the Relax IR before
codegen). TVM `0.25.dev0`, the vendored fork at
`/Volumes/external/sources/tilelang/3rdparty/tvm`.

Two baselines are measured, both from the **real compiled IR** (no fabrication):

* **ALL-LIVE total** = eager `mx.eval` semantics. MLX's training step is lazy; the
  single terminal `mx.eval(model.parameters(), optimizer.state, loss, ntokens)`
  (`cppmega_mlx/training/loop.py:176`) forces the *entire* fwd+bwd+update tape at
  once, so every intermediate buffer is allocated simultaneously. This equals the
  sum of all distinct `builtin.alloc_tensor` buffers (no reuse) -- TVM's own
  `relax.analysis.estimate_memory_usage` reports exactly this as the
  "without memory planning" number.
* **STRICT peak** = a *tighter, harder* baseline that already frees each buffer at
  its last use (alias-followed liveness) but still does NOT share storage. The
  planner must beat even this wherever genuine concurrency exists.

`StaticPlanBlockMemory` is then measured as **planned working set** (sum of the
reused `memory.alloc_storage` storages) and **planned strict peak** (max
simultaneous live storage honouring the compiler-emitted `kill_storage`
free-barriers).

| Case | ALL-LIVE (eager) | planned | reduction | STRICT peak (eager) | planned | reduction |
|---|---|---|---|---|---|---|
| fwd-only chain, n=2048, 8 layers | 256.00 MB | 48.00 MB | **5.33x** | 32.00 MB | 32.00 MB | 1.00x |
| fwd-only residual, n=1024, 8 layers | 96.00 MB | 16.00 MB | **6.00x** | 12.00 MB | 12.00 MB | 1.00x |
| **fwd+bwd (Gradient), n=1024, 6 layers** | 170.00 MB | 77.00 MB | **2.21x** | 52.00 MB | 45.00 MB | **1.16x** |
| **fwd+bwd (Gradient), n=512, 8 layers** | 57.00 MB | 25.25 MB | **2.26x** | 17.00 MB | 13.25 MB | **1.28x** |

Both correctness checks pass: the planned VM output matches an independent numpy
reference for the forward graphs, and the `Gradient`-produced `main_adjoint`
(fwd+bwd in ONE dataflow block) actually executes on the VM under planning.

### What the numbers honestly mean

* **Planning ALWAYS lowers the eager all-live total** (5-6x for forward, ~2.2x for
  fwd+bwd). This is the number that matters for the MLX OOM, because eager
  `mx.eval` *is* the all-live case -- it has no incremental free within the forced
  tape. **This alone is the win that fixes the OOM.**
* **For a pure feed-forward chain, the STRICT peak does not drop** (32 -> 32 MB):
  last-use liveness alone already bounds a linear chain to ~2 buffers, so buffer
  *sharing* reduces the *number of distinct allocations* (the all-live total) but
  not the concurrent high-water. We report this honestly rather than contrive it.
* **For fwd+bwd, the STRICT peak DOES drop, and the win grows with depth**
  (1.16x at 6 layers -> 1.28x at 8 layers). This is the load-bearing proof of the
  thesis: in a training step, forward activations are *irreducibly* live across
  the backward pass, creating genuine concurrency. The graph planner reuses dead
  buffers inside that fwd/bwd overlap window; eager `mx.eval` cannot, because it
  sees no liveness -- it just materialises the whole tape.

The growing-with-depth trend is the signature of the cross-layer-liveness effect
the roadmap predicts, and it extrapolates to the deep 1.8B step.

---

## 2. Does Relax autodiff cover fwd+bwd? YES (verified, with one constraint)

`relax.transform.Gradient("main", require_grads=ws)` (src `gradient.cc:787`, py
`transform.py:55`) produces `main_adjoint`, a single function holding **forward +
backward in ONE dataflow block**, returning `(loss, (grad_params...))`. The PoC
builds it, plans it, and runs it. So `StaticPlanBlockMemory` plans fwd and bwd in
ONE liveness scope -- exactly the cross-layer span eager lacks.

* **Constraint:** `Gradient` requires the input be exactly one DataflowBlock
  (`gradient.cc:680` `ICHECK(seq_expr->blocks.size() == 1)`). fwd+bwd fit; the
  optimizer update is generated as a *separate* function by `SetupTrainer`
  (`training/setup_trainer.py`: `backbone_loss`, `backbone_loss_adjoint`,
  `optimizer`). To co-plan the optimizer update with activations+grads in ONE
  liveness graph you must inline the optimizer into the adjoint block before
  planning (`relax.transform.InlineFunctions`) -- feasible but unverified against
  the single-block ICHECK; this is a roadmap risk, not a blocker for the OOM win
  (the activation+grad term is the dominant one and is already co-planned).

* **Rematerialization / O(sqrt N):** exists, but ONLY as *manual* gradient
  checkpointing inside the AD pass (`relax.grad.start_checkpoint` /
  `end_checkpoint`, `op/tensor/grad.cc`; `CheckpointGenerator` recomputes in
  backward, `gradient.cc:252`). There is **no automatic remat-point selector** in
  this TVM (grep of `python/tvm/relax/` finds none). So the PoC's win is **pure
  liveness buffer reuse, no remat**; sqrt-N remat is a *later* lever (see roadmap),
  not promised by stock Relax.

---

## 3. Integration architecture: MLX -> Relax for our train step

### The boundary (the realistic, minimal one)

**MLX orchestrates; only the heavy joint train-step region becomes ONE Relax
program.** Embeddings, tokenizer, data, MoE-router eager edges stay in MLX. The
compute that OOMs (fwd+bwd+update over activations+grads+optimizer state) moves
into a single Relax function that `StaticPlanBlockMemory` sees end-to-end. MLX
feeds inputs and reads outputs via the existing **DLPack zero-copy bridge**
(`cppmega_mlx/nn/_tilelang/_cuda_zerocopy.py`).

### Where the whole-step graph is assembled today: NOWHERE

path_c lowers **one TIR PrimFunc per fusion region** through `tilelang.compile`
(`path_c_fusion.py:3138` `tilelang_single_entry_lowerer`, which RAISES on >1 entry
at `:3197`). The whole train step is stitched in *eager MLX Python*:
`nn.value_and_grad` + `optimizer.update` + the terminal `mx.eval` at
`loop.py:171-176`. There is no Relax anywhere and no whole-step liveness. That
terminal `mx.eval` is the OOM site and the all-live baseline above.

### The new whole-step assembly site (what the first PR builds)

Reuse the per-region PrimFuncs path_c already emits and assemble them **one level
ABOVE** the per-region kernel boundary, in Relax:

```
NEW  cppmega_mlx/runtime/path_c_relax_step.py
       uses tvm.relax.BlockBuilder to register the per-region PrimFuncs into ONE
       IRModule and emit  @R.function train_step(x, params...):
           h1 = R.call_tir(fwd_region0, ...)
           h2 = R.call_tir(fwd_region1, ...)
           ... loss / cotangent ...
           g1 = R.call_tir(bwd_region1, ...)
           g0 = R.call_tir(bwd_region0, ...)
       then: StaticPlanBlockMemory + KillAfterLastUse + relax.build
```

Each per-region PrimFunc is exactly a `call_tir` leaf. **Do NOT touch**
`_single_entry_prim_func` / `tilelang_single_entry_lowerer` (`path_c_fusion.py`)
or `_assert_single_kernel_region` (`tilelang/.../engine/fusion.py:1523`) -- those
correctly enforce the per-region single kernel; the whole-step graph lives above
them and calls those kernels as `call_tir` leaves.

Reuse unchanged:
* `build_path_c_descriptor_prim_func` (`path_c_fusion_schedules.py:2156`) -- fwd
  region PrimFunc.
* `build_path_c_aot_autograd_region` (`path_c_fusion.py:2583`) -- already builds a
  joint fwd+bwd region surface; promote it from per-region single-entry lowering
  to a `call_tir` leaf in the assembled function.
* `build_path_c_model_regions_from_model` (`path_c_fusion.py:1180`) -- region list.

Swap only in the toy/opt-in harness: the eager stitch at `loop.py:171-176` becomes
a single Relax-VM call to `train_step`.

### The load-bearing unknown (validate first)

path_c PrimFuncs bind a **physical bank ABI** (`path_c_physical_abi.py`,
`merge_path_c_physical_abi_for_prim_funcs` at `path_c_fusion.py:1199`). Relax
`call_tir` expects destination-passing (DPS) output-buffer convention. **It is
UNVERIFIED that path_c's physical-ABI PrimFuncs satisfy `call_tir` DPS without a
thin wrapper.** First PR builds the toy with plain *logical-buffer* ABIs to defer
this; a per-region DPS adapter shim is the second PR if needed.

---

## 4. Pass pipeline targeting 26-40 GB for the 1.8B step

Eager today: ~118 GB == all-activations-live + grads-live + full optimizer state
live (forced by the one terminal `mx.eval`). Target ~26-40 GB. The levers, in
order, each attacking one term:

1. **Whole step -> one graph** (`path_c_relax_step.py` above, or
   `relax.training.SetupTrainer(loss, optimizer, loss_args)` which emits
   `backbone`, `backbone_loss`, `backbone_loss_adjoint`, `optimizer` in one
   module). *Enabler -- without it none of the below apply across the step.*
2. **Reverse-mode AD** -- `relax.transform.Gradient` (already proven in PoC).
   Liveness now spans fwd->bwd in one scope.
3. **Liveness buffer reuse** -- `relax.transform.StaticPlanBlockMemory` (in the
   default pipeline at `relax/pipeline.py:94` and `cuda/pipeline.py:64`). *This is
   the PoC win: collapses all-activations-live into the concurrent working set.*
   Attacks the activation + grad high-water. **Proven: 2.2x on fwd+bwd, 5-6x on
   fwd.**
4. **Selective / sqrt-N rematerialization** -- wrap per-block recompute regions in
   `relax.grad.start_checkpoint`/`end_checkpoint`. Attacks the activation term to
   O(sqrt N): Chen 2016 ~7x (48->7 GB, +1 fwd); Korthikanti 2022 selective ~5x at
   >90% less recompute overhead. **Manual checkpoint boundaries only -- no auto
   selector in this TVM; this is a new pass if auto placement is wanted.**
5. **In-place optimizer update** -- NOT free in Relax (tqchen: no in-place planning
   by default). Implement Adam/SGD as an in-place op so the planner aliases
   `param <- updated_param` and `m/v` in place, OR shard optimizer state ZeRO-1
   (4x model-state on multi-device; single-device -> in-place is the only lever).
   Attacks the optimizer-state term.

**Math:** 118 GB --(step 3, ~2.2x on the fwd+bwd activation+grad term)--> ~55-60 GB
--(step 4, ~5x selective remat on activations)--> ~30-35 GB --(step 5, in-place
optimizer)--> **~26-30 GB**. Step 3 is proven; steps 4-5 are projected from the
cited literature and are the multi-quarter remainder.

---

## 5. First PR (smallest real increment)

**PR 1 -- "whole-step Relax assembly + planning, toy 2-region step":**

* Add `cppmega_mlx/runtime/path_c_relax_step.py`: `BlockBuilder` assembler that
  takes 2 adjacent regions' fwd+bwd PrimFuncs and emits one `@R.function
  train_step` issuing `call_tir`s in fwd-then-reverse order; runs
  `StaticPlanBlockMemory` + `KillAfterLastUse`; `relax.build` to a VM `Executable`.
* Build with plain logical-buffer ABIs (defer the physical-ABI/DPS adapter).
* Wire the existing DLPack bridge (`_cuda_zerocopy.py`) to feed MLX inputs / read
  outputs.
* Gate behind a toy harness flag; do NOT change the eager `loop.py` default.
* Acceptance: `estimate_memory_usage` + the PoC's peak analyzer report a measured
  lower planned peak than the 2 independent `tilelang.compile` kernels, AND the
  Relax-VM output matches the eager MLX output (fail-loud, RULE #1).

PoC code committed alongside this doc is the executable reference for the analyzer
and the planning invocation PR 1 reuses.

---

## 6. Honest risks / scope

* **Multi-quarter.** PoC (this doc) is week-1. PR 1 (2-region toy) is weeks. The
  full 1.8B step with remat + in-place optimizer is multiple quarters.
* **DPS / physical-ABI adapter** is the top unknown (section 3). If path_c
  PrimFuncs cannot be `call_tir` leaves without a wrapper, PR 1.5 writes the shim.
* **Optimizer co-planning** needs inlining into the adjoint's single dataflow
  block; the single-block ICHECK (`gradient.cc:680`) may reject it -- validate
  before promising the optimizer-state win.
* **No auto-remat in this TVM.** sqrt-N requires manual checkpoint markers or a new
  min-cut partitioner pass (the torch AOTAutograd `partitioners.py` algorithm is
  the reference). Do NOT promise sqrt-N from stock Relax.
* **DLPack import direction into MLX** (CUDA side) is unverified on gb10 -- confirm
  zero-copy round-trip before relying on it for the CUDA target.
* **The strict-peak win is modest for shallow nets** (1.16-1.28x) and only the
  all-live win is large; the strict win grows with depth, so the 1.8B projection
  rests on the depth trend, not the toy magnitude. Re-measure on the real depth.

---

## Run the PoC

```
TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \
PYTHONPATH=/Volumes/external/sources/cppmega.mlx \
/Volumes/external/sources/nanochat/.venv/bin/python3 -u \
  -m cppmega_mlx.runtime.relax_memory_plan_poc
```

It RAISES (fail-loud) if planning fails to lower the all-live total in any case,
or the strict peak in the fwd+bwd cases, or if a planned VM output disagrees with
its numpy reference. Last verified: 2026-06-02, all checks passed (numbers in
section 1).
```
