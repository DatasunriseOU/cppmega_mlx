# Fused Block-Composable Training Pipeline — Architecture + Roadmap

**triton → tilelang → tvm → ffi, targeting gb10 (sm_121) + H100 (sm_90) + B200 (sm_100), ≥ Megatron**

Author: synthesis of research passes A–G. Date: 2026-06-02.
Status: design + roadmap. Treat all `file:line` / URL citations as evidence; UNVERIFIED items flagged inline.

---

## 1. Verdict (lead)

**YES-BUT.** We can realistically build a block-composable, auto-fused training pipeline on `triton→tilelang→tvm→ffi` that matches or beats Megatron on H100 (sm_90) and B200 (sm_100), and runs (with a known weak-arch tax) on gb10 (sm_121). It is **not** greenfield: we already own ~80% of Megatron's spec/registry machinery (`tilelang/engine/fusion.py:64,295`; `cppmega.mlx/.../path_c_fusion_schedules.py` descriptor registry + greedy planner) and the hard fused kernels (`examples/deepseek_v32/sparse_mla_fwd.py` + `sparse_mla_bwd.py`, `deepgemm`, `fusedmoe`). The "BUT" is three concrete, bounded gaps: (1) our path_c **codegen emits a batch=1, literal-seq, single-threadgroup source string** — the structural shape-pin — whereas TileLang's own DeepSeek kernels already use `@tilelang.jit` closure params + `T.dynamic("seq")` + a real grid (`sparse_mla_fwd.py:35-42,76`); (2) the **training-step memory blocker (116 GB) is a loop/graph problem, not a kernel problem** — only loop-level grad-accum/selective-recompute (now) or whole-step graph liveness+remat (durable) fixes it; (3) several leverage kernels (deepgemm, fusedmoe, indexer) are **forward-only** — their training backward is net-new code.

**Why was MLX ported to CUDA — was it necessary?** *Not as a training substrate, and our own docs never claimed it as one.* MLX is scoped Apple-Silicon/Mac-local (`mlx_port_master_plan.md:4`), with CUDA explicitly a *reference/parity target, not production* (`mlx_port_master_plan.md:50`; `mlx_buy_vs_build.md:432` "CUDA receipts, not parity targets for MLX"). MLX-CUDA exists in our stack because (a) MLX is our Apple/Metal front-end that *also happens* to run on CUDA for cross-checks (`MEGATRON-VS-MLX-PATHS.md:95` "`path_b` = CUDA reference"), and (b) upstream MLX's own pitch is "prototype-on-Mac, deploy-on-NVIDIA" portability, not a Megatron replacement ([9to5Mac 2025-07-15](https://9to5mac.com/2025/07/15/apples-machine-learning-framework-is-getting-support-for-nvidia-gpus/)). MLX-CUDA crashes on missing ops (no fallback) — gather-MM/quant/FFT/LAPACK all unsupported ([ml-explore/mlx #2422](https://github.com/ml-explore/mlx/discussions/2422)), which is exactly why our path_b is forced into dense `ReferenceMoE` and a chunk-cap-32 scan (`TOKPS-DISCREPANCY.md:113`). So MLX-eager-on-CUDA was a *consequence*, not a *necessity* — the right CUDA story is the fused Path C; MLX stays the Apple front-end + eager numerical oracle.

**Is the fused/graph path the answer, not eager?** *Yes — and it is the strongest argument for path_c beyond speed.* The 116 GB peak is a direct consequence of MLX eager reverse-mode AD retaining the full forward graph via shared_ptr chains until a single terminal `mx.eval` (`loop.py:81-88`), with no execution barrier to free a checkpointed region before its recompute materializes. Megatron fits the same 16,384-tok step in ~26 GB via selective recompute (Korthikanti 5×, <4% overhead, [arXiv 2205.05198](https://arxiv.org/abs/2205.05198)) + distributed optimizer + sequence/tensor parallel. A static graph IR (TVM Relax) gives the two things eager structurally cannot: **whole-program liveness + buffer reuse**, and **compiler-chosen rematerialization with a guaranteed free-barrier** (Chen 2016 O(√N) peak, arXiv:1604.06174). The honest interim production baseline remains **PyTorch/Megatron-LM on CUDA** (already runs the real config at 3399–4182 tok/s @ ~26 GB, `MEGATRON-VS-MLX-PATHS.md:265`); path_c is the composable replacement we are building, and the loop-level eager fixes (§4) close most of the memory gap *now* without waiting for the full Relax lowering.

---

## 2. The Architecture — block-composable fused training pipeline

The winning design is **Megatron's spec system, re-skinned over machinery we already wrote**. Megatron's composability is a two-layer *declarative spec → reflective builder*: a `ModuleSpec` dataclass graph (pure data) + a single `build_module` dispatcher that substitutes hand-fused TransformerEngine classes wherever the spec names them (`megatron/core/transformer/spec_utils.py`; `gpt_layer_specs.py`). **Megatron does NOT auto-discover fusions** — fusion is pre-baked into hand-written TE classes; the spec system only *selects and wires* them. Composition = declarative; fusion = library lookup. That is the cheap, robust target.

### 2.1 The four layers (all map to existing code)

```
  [ Block spec layer ]   declarative dataclass: slots = op names           ← Megatron ModuleSpec
        |                  resolve in FusionBlockRegistry
        v
  [ Fusion registry  ]   signature-keyed fused chain templates            ← Megatron TE leaf classes
        |                  select_longest() auto-applies longest match     ← greedy auto-apply
        v
  [ Region planner   ]   greedy segmenter: device-cap-aware splits         ← path_c planner
        |                  fwd/bwd fission, ABI/smem/op-count caps
        v
  [ TVM → FFI emit   ]   LowerAndLegalize + OptimizeForTarget → IRModule   ← engine/phase.py
                           → tilelang.compile/JIT → __tvm_ffi_<step> export
```

**Layer 1 — Block spec = declarative dataclass.** Mirror `TransformerLayerSubmodules`: a dataclass whose slots (`norm`, `mixer`, `mlp`, `*_bda`) hold `op` names that resolve in `FusionBlockRegistry.register(FusionBlockDescriptor(...))` (`tilelang/engine/fusion.py:64`). `node_from_block` (`fusion.py:98`) is our `build_module` analog — it normalizes any caller block (dict/obj with `op`/`op_name`/`kind`/`route_symbol`) into a `FusionNode`. Register `mamba3`, `gqa`/`mla`, `moe` as `FusionBlockDescriptor`s with fwd `prim_func` + a paired `_bwd` descriptor in `PathCBrickScheduleDescriptorRegistry` (`path_c_fusion_schedules.py:1247`; bwd synthesis already exists ~`:1281`). This is the direct analog of naming `TELayerNormColumnParallelLinear` in a slot.

**Layer 2 — Fusion = signature lookup, NOT search.** Register fused chains in `FusionScheduleRegistry.register(op_signature, schedule_template, status=)` (`fusion.py:295`) keyed by op-signature tuples — e.g. `("rmsnorm","gqa_qkv","flash_attn","out_proj")`, `("rmsnorm","mamba3_in_proj","mamba3_chunk_scan","mamba3_out_proj")`, `("router","moe_groupgemm","moe_combine")`. `select_longest` (`fusion.py:344`) greedily matches the longest registered fused chain; unmatched nodes fall back to per-op descriptors. **Copy Megatron, reject Mirage-style search for the hot path.** No silent degraded path: if a registered chain's template raises, it must surface (RULE #1 fail-fast) — audit that `FusionScheduleRegistry` template *execution* has no try/except masking a broken chain (only the cache-key tag at `fusion.py:330` uses `suppress`, which is fine).

**Layer 3 — Region planner = the greedy segmenter we already have.** `plan_path_c_direct_fusion_chain_for_region` (`path_c_fusion_schedules.py:15931`) extends each segment as far as possible while (a) not crossing the fwd/bwd boundary, (b) staying under `max_kernel_buffers` (Metal 31-buffer ABI cap, `:129`), (c) staying under device-resolved op-count caps (Metal 2 fwd / 1 bwd; CUDA `None`=monolithic, `:15952-15977`). Infeasible segments **raise `PathCSplitInfeasible`** (`:452`) — explicit fail-fast, no dtype-bank fallback (consistent with RULE #1). This logic is sound and stays. Drive its *candidate* fusions from Layer-2 matched chains rather than the fixed `_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE` (`:535`).

**Layer 4 — Train step = fwd region + bwd region, emitted through TVM→FFI.** Use the existing `_train_step_suffix_loss_*` codegen in path_c to emit loss/cotangent/param-grad tails (the fused-CCE + grad path Megatron gets from TE+`cut_cross_entropy`). `FusionAutogradPlan` (`fusion.py:221`) pairs fwd/bwd regions per block. Emit exactly as the stack already does: TileLang region → `LowerAndLegalize` + `OptimizeForTarget` (`tilelang/engine/phase.py:265,411`) → TVM IRModule → `tilelang.compile`/JIT → FFI. To expose the fused train-step as a callable op, export the `__tvm_ffi_<train_step>` symbol (or C++ `TVM_FFI_DLL_EXPORT_TYPED_FUNC`), pull stream via `TVMFFIEnvGetStream()`, raise via `TVMFFIErrorSetRaisedFromCStr`, return 0/-1 ([tvm.apache.org/ffi compiler_integration](https://tvm.apache.org/ffi/guides/compiler_integration.html)). Targets parametrize sm_90/sm_100/sm_121 via TVM `target`.

### 2.2 Where TVM Relax graph + memory planning fits

TVM Relax `FuseOps`+`FuseTIR` is real automatic cross-op fusion (post-dominator + union-find over a dataflow block, [tvm.apache.org/docs/arch/fusion](https://tvm.apache.org/docs/arch/fusion.html)) — **but `kOutEWiseFusable` complex ops cannot chain** (matmul→attention→matmul is NOT auto-fused into one mega-kernel). So automatic `FuseOps` will *not* build our flash-attn/SSD-scan mega-kernels. The supported path is **`FuseOpsByPattern` / `MergeCompositeFunctions` + BYOC**: match a user-defined dataflow pattern (rmsnorm+qkv+sparse-MLA) and offload to our TileLang kernel as an external `call_dps_packed` ([external_library_dispatch](https://tvm.apache.org/docs/arch/external_library_dispatch.html)). **Use Relax as: (a) the optional, OFF-hot-path candidate *generator* of new `FusionScheduleRegistry` entries that you verify and pin — never a runtime fallback; (b) the whole-train-step graph that provides global liveness + remat (§4).** Schedule the residual elementwise/norm TIR with **Dlight** (`ApplyDefaultSchedule(Matmul,GEMV,Reduction,...)`, works with dynamic shapes, seconds not minutes). Reserve **MetaSchedule** for offline tuning of the *inside* of each registered TileLang chain — not a per-shape JIT path. NOTE: the in-flight `merge/upstream-codegen-reorg` branch renamed `tir→tirx` (visible in current commits); Relax doc examples now read `from tvm import relax, tirx` — keep that in mind when wiring passes.

### 2.3 Multi-arch targeting (gb10 / H100 / B200)

TileLang auto-detects the device (`target.py:276-282` reads `torch.cuda.get_device_capability` → `sm_{arch}`) and has per-arch predicates (`target_is_hopper`, `target_is_sm120`, feature predicates `target_has_bulk_copy/ldmatrix/...`, `tilelang/utils/target.py:361-395`). It does **not** auto-optimize across arch — per-arch kernels are hand-written (`gemm_sm100/` tcgen05 vs SM90 wgmma are separate files; there is no single kernel that auto-lowers WGMMA↔tcgen05). Three concrete multi-arch facts drive the plan:

- **The `a`-vs-`f` target gap (one-function patch).** `tilelang/contrib/nvcc.py:439` always appends `"a"` for `major>=9`, never `"f"`. On gb10 (CC 12.1) auto-detect emits `sm_121a` (architecture-*specific*) — per [CUDA Prog. Guide §5.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) an `a` binary "can only be run on the exact compute capability it was compiled for." A single GB10+RTX50 binary needs the **family** target `compute_120f`/`sm_120f`. Patching `get_target_arch` to emit `f` is the concrete fix.
- **smem carve-out delta.** ~228 KB on H100/B200 vs **~99 KB on consumer Blackwell (sm_120/121)** (cutlass#3144; NVIDIA Blackwell Tuning Guide). SM100 tile schedules **overflow smem on gb10** and must be re-tiled + re-autotuned. This is the one universal gb10 gotcha.
- **Two kernel tiers in our tree.** *Tier A* (generic `T.gemm`) ports to all three with autotune-only effort — `sparse_mla_fwd/bwd`, `deepgemm`, `fusedmoe`, NSA, blocksparse. *Tier B* (hand-pipelined `T.wgmma_gemm`, SM90-locked) — `sparse_mla_fwd_pipelined/_seesaw`, `mla_decode_ws` — do NOT run on sm_100/121 as-is; each has a portable generic twin, so the loss is perf-on-H100, not capability.

**Per-arch precision routing must fail loud, not degrade** (RULE #1): FP8 everywhere on sm_90/100; on gb10, NVFP4/FP4-MoE dispatch is reportedly broken (NVIDIA forum #357663; CUTLASS FP8 mis-dispatch on sm121) — **RAISE** rather than silently fall to a lower precision until verified fixed on the box.

---

## 3. What to PULL now (ranked pull-list)

### 3.1 OUR TileLang kernels → which path_c reference op they replace

All paths under `/Volumes/external/sources/tilelang/examples/`. Effort tags: Low = autotune-only; Med = smem re-tile for gb10's 99 KB; verify = smoke-test vectorized atomics on sm_121 before trusting.

| Rank | Our kernel (file) | fwd/bwd | Replaces in path_c | Port-effort gb10(121)/H100(90)/B200(100) |
|---|---|---|---|---|
| **1** | `deepseek_v32/sparse_mla_fwd.py:12` + `sparse_mla_bwd.py:83` | **fwd+bwd** (lse `:163`; atomic dKV `:224,231`) | O(n²) gather sparse-MLA fwd **AND** bwd → O(seq·topk). **Only training-complete MLA we own.** | Low / Low / Low (bwd: Low-Med — verify `atomic_addx4` lowers on sm_121) |
| **2** | `deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py:11` + `deepseek_v32/inference/kernel.py:34` act_quant | **fwd only** | FP8 GEMM for every Linear/proj (path_b has zero fused FP8) + activation quant | Med (smem re-tile block_N) / Low / Low. **bwd is net-new.** |
| **3** | `fusedmoe/example_fusedmoe_tilelang.py:12,97` | **fwd only** | dense `ReferenceMoE` (all 16 experts) → token-grouped sparse expert GEMM | Med / Low / Low. **MoE bwd is net-new.** |
| **4** | `deepseek_v32/fp8_lighting_indexer.py:100` + `topk_selector.py` | **fwd only** | produces the topk indices feeding #1 (required to wire sparse-MLA end-to-end) | Low / Low / Low |
| **5** | `deepseek_nsa/example_tilelang_nsa_fwd.py:11` + `_bwd.py:157,319` | **fwd+bwd** | the only *other* training-capable sparse attn — backup to #1 if MLA bwd atomics misbehave on gb10 | Low-Med / Low / Low-Med |
| 6–9 | `blocksparse_attention` (fwd-only), `blocksparse_gemm` (niche), `deepseek_mla/example_mla_decode.py` (inference-only), `deepseek_v4/sparse_attn_fwd_sm90.py` (illustration), `deepseek_v4/act_quant.py` FP4 (B200-only future) | mixed | lower priority / niche | varies |

**SKIP (Tier B, SM90-locked):** `sparse_mla_fwd_pipelined.py:151`, `sparse_mla_fwd_seesaw.py:297`, `deepseek_mla/example_mla_decode_ws.py:101` — use their generic twins on gb10/B200; native on H100 only.

**Critical training gap:** of leverage-1/2/3, only **v32 sparse-MLA ships a real backward**. deepgemm, fusedmoe, indexer, blocksparse are **fwd-only** — FP8-GEMM bwd and MoE bwd are **not in our tree** and are the main net-new write before path_c trains at Megatron parity.

### 3.2 EXTERNAL libs to vendor (license-cleared)

**License rule:** need MIT/Apache/BSD. **DO NOT VENDOR Unsloth core (AGPL-3.0)** — copyleft, infects the tree; use Liger/CCE/QuACK for the same ops.

| Lib | License | Pull for (gap) | Arch | Effort/notes |
|---|---|---|---|---|
| **Apple CCE** (apple/ml-cross-entropy) | Apple permissive (verify LICENSE) | **fused CCE — primary gap** (Gemma-2: loss mem 24 GB→1 MB). Copy the chunked eval-and-free pattern. | Ampere+ (Triton); `torch_compile` for Mac | Top mem ROI (~12 GiB/rank, `apply_linear_ce_patch.py:10`) |
| **Liger-Kernel** (linkedin) | BSD-2 | FusedLinearCrossEntropy, RMSNorm, RoPE, SwiGLU, alignment losses | Triton (NVIDIA+AMD) | drop-in alt to CCE |
| **QuACK** (Dao-AILab) | Apache-2.0 | CrossEntropy fwd+bwd, RMSNorm fwd+bwd, **Blackwell GEMM** (`gemm_sm90/sm100/sm120.py`) | SM90+SM100; sm120 file present (gb10 family) | CuTe-DSL; sm121 unvalidated |
| **FlashInfer** (flashinfer-ai) | Apache-2.0 | FP8+FP4 GEMM, grouped GEMM, **fused MoE (top-k, FP8/FP4 experts)**, MLA, norms/RoPE | SM75→Blackwell incl sm_120/121 | fixes dense MoE; **but** open FP8-attn is FA3=SM90-only, gb10 had open compile failures (#2252) |
| **FlashAttention-3** (Dao-AILab `hopper/`) | BSD-3 | flash-attn fwd+**bwd** | **SM90 only** | H100 turbo; not B200/gb10 |
| **FlashAttention-4** (CuTeDSL) | BSD-3 | flash-attn fwd+bwd, MLA absorbed | tested SM90–SM100; SM120 emerging/UNVERIFIED | beta (fa4-v4.0.0.beta11) |
| **ThunderKittens 2.0** (HazyResearch) | MIT | tile primitives, WGMMA+**TCGEN05**, MXFP8/NVFP4 | SM90+SM100; no sm120/121 | build-your-own fused |

**Top external picks:** Apple CCE (the CCE), FlashInfer fused MoE (fixes dense ReferenceMoE), FA3/FA4 + ThunderKittens for attention primitives, QuACK Blackwell GEMM. **No permissive fused/distributed-optimizer kernel zoo exists** — that stays Megatron-LM `--use-distributed-optimizer` (Apache-2.0) territory, our own work atop it.

---

## 4. The memory fix (highest leverage) — 116 GB → ~30–40 GB

**Root cause (grounded in our code).** MLX reverse-mode AD builds a lazy tape; every intermediate is kept alive by a shared_ptr chain until backward consumes it ([MLX DeepWiki VJP](https://deepwiki.com/ml-explore/mlx/3.2-automatic-differentiation-(vjp-and-jvp))). Our loop builds the entire fwd+bwd+update graph lazily and forces it **once** at a single terminal `mx.eval` (`loop.py:81-88`) — so peak = every seq=4096 saved activation + params + grads + full optimizer state, all live simultaneously. On gb10 unified memory there is no separate pool to spill to; the peak IS the box. Our per-layer `nn.utils.checkpoint` is wired in the model fwd (`hybrid_lm.py:2290`) but the loop has **no `mx.eval` barrier**, so the recompute working set adds roughly as much as the stored set removes — checkpoint is a no-op for peak.

**Why Megatron's recompute is actually free:** `torch.utils.checkpoint` runs fwd under `no_grad`, **discards** intermediates immediately, re-runs fwd in backward layer-by-layer so only ONE layer is live → O(L·sbh)→O(sbh). Selective (attention-only) recompute gives **5× activation reduction at <4% overhead** (Korthikanti, [arXiv 2205.05198](https://arxiv.org/abs/2205.05198)); that + sequence/tensor parallel is how 16,384 tok fits in ~26 GB (their no-recompute baseline is >80 GB, same order as our 116).

**Smallest changes now (ordered by GB-per-line, all in `one_step_train`, all using machinery we already have):**

1. **Gradient accumulation in the loop (biggest lever).** Split batch into N microbatches; per micro run `value_and_grad`, `mx.eval(grads)` into a running accumulator, **drop the micro's graph** before the next. Peak activation collapses from full-batch to 1-microbatch. Scaffolding already exists (`checkpoint.py:101-104,768-777`), just not wired into the loop. Expect the dominant ~80–100 GB activation term to drop roughly linearly with N.
2. **Per-layer selective checkpoint WITH an `mx.eval` barrier.** Add `mx.eval(hidden_states)` after each checkpointed (attention-only/Korthikanti) layer so the stored input is forced+freed before recompute materializes. Converts our no-op checkpoint into a real O(sbh)+2sbhL footprint at <4% overhead.
3. **8-bit optimizer state (already built).** Switch to `optimizers_quantized.py` / `_fused_adam8bit_kernel.py` / `native_optim/fused_8bit.cpp`. Adam m+v in fp32 is 2× param bytes; 8-bit cuts the static optimizer block ~4× — several GB off the floor grad-accum/checkpoint can't touch. (Note: ZeRO-1 `distributed_optimizer.py` is a **no-op on single gb10 node**, world_size=1, `:260-261`; classic CPU-offload is moot on unified memory.)
4. **`mx.clear_cache()` after step + memory limit (already built).** `runtime/memory.py:114,181,191`; `training/profile.py:53-56` exist but aren't called in the loop. The allocator caches freed buffers, inflating effective peak.
5. **Use the fused flash-attn / SSD-scan path_c kernels** (`deepseek_v32` sparse_mla, mamba3 chunked scan) — they store nothing and recompute in bwd internally, a free kernel-level remat of the `5as/h` attention term. (Alone they don't fix 116 GB — they shrink the attention term but leave the `34·sbhL` cross-layer term, which only loop-level or graph-level remat removes.)

**The durable fix (path_c graph, follow-on):** moving the whole train step into a TVM Relax graph gives **static liveness + buffer reuse** (reuse storage the moment a value is dead — structurally impossible in eager) and **compiler-chosen rematerialization with a guaranteed free-barrier** (O(√N) peak, provable). Caveat: our planner today builds **per-region** single-entry templates, not a whole-step program — so path_c only owns *cross-layer* liveness once the step is lowered as ONE graph with global planning. **Expected:** grad-accum (N=4–8) + real selective checkpoint barriers alone bring the activation term into 20–30 GB; 8-bit optimizer + clear_cache trims the static floor → seq=4096 step lands in 30–40 GB *without* yet needing full Relax lowering. The Relax graph makes that low peak *guaranteed and automatic* rather than hand-placed.

---

## 5. Ranked roadmap / first PRs (leverage-ordered)

### PR-1 — Highest-leverage kernel pull + path_c wire: sparse-MLA fwd+bwd
**Do first.** It is the single largest mem+compute sink and the only training-complete MLA we own.
- **First step:** wire `examples/deepseek_v32/sparse_mla_fwd.py` + `sparse_mla_bwd.py` (+ `fp8_lighting_indexer.py` + `topk_selector.py` to feed indices) as the brick fragments for the `sparse_mla_fp8_apply{,_bwd}` descriptors in `path_c_fusion_schedules.py`, replacing the string-emitted reference. Smoke-test `atomic_addx4` (dKV) on gb10 first.
- **Unblocks:** kills the O(n²) gather sparse-MLA on fwd AND bwd; makes v32 MLA training-capable; removes the biggest activation+compute term.

**STATUS (2026-06-02): WIRED + ENV-GATED; BLOCKED ON gb10/sm_121 (NOT atomics — driver smem opt-in).**
- **Wiring landed (default OFF):** new `cppmega_mlx/nn/_tilelang/_sparse_mla_v32_fused.py` wraps the real fused v32 `sparse_mla_fwd.py`/`sparse_mla_bwd.py` over bf16 `q/kv/indices`; wired into the differentiable `sparse_mla_fp8_path_c_apply_prepared_float` fwd + VJP, gated by `CPPMEGA_SPARSE_MLA_V32_FUSED=1`. Gate OFF = existing reference path, unchanged (50 structural tests still green; the one local failure is a pre-existing tvm_ffi circular-import in the Metal FP8 producer, reproduced on unmodified `main`). RULE #1: when gated ON + `force_path_c`, every fused failure RAISES with where+what — no silent reference fallback.
- **gb10 smoke-test verdict (single idle run, free>90G, cleaned after):** the fused fwd/bwd does **NOT** run on gb10/sm_121 today. The blocker is **NOT** `atomic_addx4` and **NOT** a true smem overflow — it is the TileLang/TVM runtime's dynamic-shared-memory opt-in. Three findings, in order hit:
  1. **`T.dynamic` symbolic-shape lowering crash** (gb10 tilelang @ upstream `main` `c0a6fe5a`): `LowerDeviceKernelLaunch` raises `Check failed: new_data_expr->IsInstance<VarNode>() ... backing allocation must be a tirx::Var` on the `T.dynamic("batch/seq_len")` buffers — the `tir→tirx` migration regression (roadmap §6 grid/symbolic risk, confirmed). Static shapes dodge it.
  2. **smem overflow at default tiling:** static fwd compiles for `-arch=sm_121a` then `ptxas error: Entry function 'main_kernel' uses too much shared data (0x38800=231424 bytes, 0x18c00=101376 max)` — the H100/B200-class 228 KB tile vs gb10's 99 KB carve-out.
  3. **the real wall — runtime smem opt-in rejected:** re-tiled down to ~56 KB, the kernel fails at `cuda_module.cc:218` `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 57344)` → `Failed to set the allowed dynamic shared memory size`. **Bisected with a trivial kernel: the driver accepts ≤50000 B but rejects ≥51200 B**, i.e. the >48 KB opt-in carve-out is **unusable** on this gb10 stack even though the device reports `shared_memory_per_block_optin=101376`. The fused tile's fixed `H_per_block×512` + `BI×512` bf16 buffers cannot fit under ~48 KB (min viable `block_I=32` already needs ~56 KB; `block_I=16` violates a GEMM `warp_col_tiles>8` constraint). So **no valid fused config fits the gb10 runtime ceiling.**
- **`atomic_addx4` on sm_121: UNDETERMINED** — the bwd never reached the atomic scatter because the fwd (and any dyn-smem kernel) is blocked at the smem opt-in first. NOT verified working, NOT verified broken.
- **Parity:** the fused fwd+bwd-vs-torch parity is the upstream v32 example's own `assert_tensors_similar` (passes on sm_90). End-to-end fused-vs-reference parity could **not** be measured on gb10 (kernel can't run) — honestly reported, not faked.
- **Net:** on **H100 (sm_90) / B200 (sm_100)** the 99 KB+ carve-out fits and the gated fused path should run; on **gb10 the fused v32 MLA is blocked** on the runtime dynamic-smem opt-in (a TVM `cuda_module.cc` / driver issue), not the kernel ISA. Fallback per §3.1 rank-5 (NSA `deepseek_nsa/*_bwd.py`) is the next probe — but NSA tiles likely hit the same >48 KB opt-in wall, so the durable gb10 fix is patching the TVM runtime smem-opt-in call (or re-tiling every dyn-smem kernel under 48 KB), tracked as a gb10 follow-on.

### PR-2 — The memory-loop fix (grad-accum + selective checkpoint barrier + 8-bit optimizer)
**Highest mem ROI, smallest diff.** Independent of PR-1; can land in parallel.
**STATUS (2026-06-02): LANDED + MEASURED ON gb10 (commit `05e53d8`). bs=4×seq=4096 went OOM→FITS.**
- **First step:** wire `grad_accum_steps` (scaffolding at `checkpoint.py:101-104`) into `one_step_train` (`loop.py`) with `mx.eval(grads)`-and-drop per microbatch; add `mx.eval(hidden_states)` barrier after each attention-only checkpointed layer; switch optimizer to `optimizers_quantized.py`; call `maybe_clear_cache_after_step` (`runtime/memory.py:191`) each step.
- **Unblocks:** seq=4096 step 116 GB → 30–40 GB on gb10 — makes the real production config (bs4×seq4096) *reachable* on a single box without waiting for Relax.

**What landed (`05e53d8`, all env-gated, default = prior behavior — RULE #1):**
- `one_step_train` (`loop.py`): `grad_accum_steps` kwarg / `GRAD_ACCUM_STEPS` env (default 1 = byte-identical to the prior single-`mx.eval` step). When N>1: split the batch into N microbatches, run `value_and_grad` per micro, `mx.eval` the **token-weighted** running accumulator as a free-barrier, drop each micro's fwd/bwd graph before the next. Optional `clear_cache` hook.
- `hybrid_lm.py`: opt-in `mx.eval(hidden_states)` after each grad-checkpointed layer (`CPPMEGA_MLX_CHECKPOINT_EVAL_BARRIER=1`, default OFF) — converts the previously no-op checkpoint into a real free-barrier.
- 8-bit optimizer (`make_adam8bit`) + `maybe_clear_cache_after_step` already wired in `train_hybrid_tiny.py`; probe `scripts/pr2_seq4096_memory_probe.py` exposes `--opt adam8bit`.

**Numeric equivalence (RULE #1, verified on Mac, `scratch/pr2_grad_accum_equivalence.py`):** N=4 micro vs full-batch → loss abs-diff **0.0**, worst grad rel-diff **2.8e-7**, post-step param rel-diff **1.2e-7**; checkpoint barrier ON vs OFF → grads **bitwise identical**. Token-weighting (`weight = n_i / N_total`) makes `sum_i weight_i · grad_i` exactly the full-batch grad for a token-mean loss. 165 loop/compiled/gradient/checkpoint tests pass.

**Measured peak — gb10, seq=4096, dense pattern=A, hidden=2048 / depth=24 / 542.5M params** (peak = `mx.get_peak_memory`; 100 GB box budget; `free -g` cross-check; watchdog SIGTERM @100 G):

| config | grad_accum | optimizer | active GiB | peak GiB | tok/s | fits ≤100 G? |
|---|---|---|---|---|---|---|
| bs=4×seq4096 monolithic (BEFORE) | 1 | AdamW | — | **OOM (SIGTERM @114 G)** | — | **NO** |
| bs=4×seq4096 (AFTER) | 4 | AdamW | 6.06 | **67.94** | 526 | **YES** |
| bs=4×seq4096 = Megatron config | 4 | Adam8bit | **3.05** | **64.93** | 514 | **YES** |
| bs=8×seq4096 (2× prod batch) | 8 | Adam8bit | 3.05 | 64.93 | 516 | YES |

**Result:** the real `bs=4×seq=4096` step went from **OOM (>114 GB, watchdog-killed)** to **67.94 GiB — it now fits on a single gb10**, loss finite + decreasing (10.54→10.29), ~526 tok/s. Active working set drops **linearly with N** (6.06 GiB @ N=4 → 3.05 GiB @ N=8), confirming the activation lever. The ~64–68 GiB *peak* is the **allocator high-water mark, not the live set** (active is 3–6 GiB) — `clear_cache` trims but the CUDA-MLX allocator/graph-cache holds the high-water mark. **Honest floor:** loop-level levers land the seq=4096 step at **~62–68 GiB peak — well under the 100 GB box budget, above the 30–40 GiB stretch target.** Reaching 30–40 GiB *peak* needs allocator-limit pinning (`apply_memory_limit_plan` / `MLX_CUDA_GRAPH_CACHE_SIZE`) and/or the durable Relax whole-step liveness (§4); the dominant residual term is the **allocator high-water mark / static param+optimizer block**, not activations (which grad-accum already collapses).

### PR-3 — Auto-fusion / block-registry (the Megatron-spec equivalent)
**The composability layer.** Depends on PR-1's brick descriptors existing.
- **First step:** define the `TransformerLayerSubmodules`-style block-spec dataclass; register `mamba3`/`mla`/`moe` as `FusionBlockDescriptor`s (`fusion.py:64`) with paired `_bwd`; register the fused chain signatures in `FusionScheduleRegistry` (`fusion.py:295`); drive `plan_path_c_direct_fusion_chain_for_region` candidates from `select_longest` (`fusion.py:344`) matches instead of the fixed signature. Audit template-execution for any try/except that masks a broken chain (RULE #1).
- **Unblocks:** arbitrary block stacks auto-emit the fused TileLang→TVM→FFI train step the way `get_gpt_layer_with_transformer_engine_spec` auto-wires TE. FP8-GEMM-bwd and MoE-bwd are the net-new writes gated here.

### PR-4 — Multi-arch (unpin shape + family targets + per-arch variants)
**The portability layer.** The sharpest single lever is the codegen, not the planner.
- **First step (unpins arbitrary bs/seq, no planner redesign):** port the path_c emitter from batch=1/literal-seq/`T.Kernel(1,...)` to the DeepSeek idiom — add a `batch` field to `PathCModelShapeEnv` (`path_c_fusion.py:408`, currently absent); emit `(batch, seq, hidden)` buffers; replace string-interpolated `int(sequence_length)` (`schedules:3496,3560,14683`) with `T.dynamic("seq")`; promote `T.Kernel(1,...)` → `T.Kernel(batch, ...)` (this also removes the macOS watchdog row-chunking hacks). Then patch `nvcc.py:439` to emit `compute_120f`/`sm_120f` for a single gb10+RTX50 binary; re-tile + re-autotune Tier-A kernels for gb10's 99 KB smem via Carver/`tilelang.autotuner`.
- **Unblocks:** one kernel serves all seq (kills per-shape recompile + `seqlen % chunk` hard-raise); gb10+RTX50 share one binary; ≥Megatron on H100/B200, runnable-with-tax on gb10.

---

## 6. Honest risks / unknowns

**UNVERIFIED (must test before trusting):**
- `T.atomic_addx4` (sparse_mla_bwd dKV) and `atomic_add` (NSA dQ, topk_selector) **vectorized-atomic lowering on sm_121** — smoke-test on gb10 before the bwd path is trusted. If broken, RAISE (don't degrade); NSA bwd (#5) is the backup.
- Exact 116 GB breakdown by term (activation vs optimizer vs params) — not measured this session; the ordering of PR-2 levers assumes activation-dominated, which is standard but unconfirmed for our exact config.
- gb10 smem carve-out (~99 KB) — from NVIDIA arch norms, not re-confirmed on the box; verify with `cudaDevAttrMaxSharedMemoryPerBlockOptin`. If a Tier-A kernel's tile can't fit 99 KB even re-tiled, that kernel is **blocked on gb10** (capability, not perf) — fail loud.
- `tilelang_single_entry_lowerer`'s handling of grid/symbolic dims (lives in `path_c_fusion.py`, imported at `schedules:41`) — confirm before PR-4 step 3 (grid promotion).
- FA4 SM120/121 functionality (code paths exist, tridao states only SM90–SM100 tested); FlashInfer/CUTLASS have open gb10 gaps (#2252, forum #357663) — for sm_121 lean on **our own** TileLang kernels, not pulled libs.
- Apple CCE exact license string — read LICENSE before vendoring.

**Genuinely multi-quarter / hard:**
- **FP8-GEMM backward and MoE backward** are net-new TileLang code (not in our tree) — the main blocker to Megatron-parity *training* (vs inference). deepgemm/fusedmoe/indexer are all fwd-only.
- **Whole-step Relax lowering with global liveness + remat** (the durable memory fix, §4) — our planner is per-region today; making it own the entire train-step graph is a larger architectural change than the loop-level eager fixes.
- **NVFP4 on gb10** — dispatch reportedly broken; B200 FP4 (tcgen05/TMEM) is net-new work, not a port (we emit zero tcgen05 in examples today).
- **Distributed/streamed optimizer + sequence/tensor parallel** — no permissive pullable kernel zoo; the sharding orchestration stays host-side (Megatron-LM territory); only Muon NS-5 + int8-momentum are TileLang candidates.

**Could fail:**
- If `compute_120f` family targets don't actually run sm_121 binaries on RTX50 (or vice-versa) as the CUDA guide implies, the one-binary goal for gb10+RTX50 collapses to per-arch builds (perf-neutral, build-matrix cost only).
- If the path_c emitter's grid promotion (PR-4) interacts badly with `tilelang_single_entry_lowerer`, the shape-unpin needs the lowerer touched too — larger than the "no planner redesign" estimate.

---

### Source index
Internal: `cppmega.mlx/docs/{mlx_port_master_plan.md,mlx_buy_vs_build.md,MEGATRON-VS-MLX-PATHS.md,TOKPS-DISCREPANCY.md,HW-AWARE-AUTOSPLIT-DESIGN.md}`; `cppmega.mlx/cppmega_mlx/{training/loop.py,training/checkpoint.py,training/optimizers_quantized.py,training/cut_cross_entropy.py,models/hybrid_lm.py,training/distributed_optimizer.py,runtime/memory.py,runtime/path_c_fusion.py,runtime/path_c_fusion_schedules.py}`; `cppmega/cppmega/{recipes/run_profiles.py,megatron/*}`; `tilelang/{engine/fusion.py,engine/phase.py,contrib/nvcc.py,utils/target.py}`; `tilelang/examples/deepseek_v32/*`, `deepseek_deepgemm/*`, `fusedmoe/*`, `deepseek_nsa/*`, `gemm_sm100/*`, `flash_attention_sm100/*`. External: [ml-explore/mlx #2422](https://github.com/ml-explore/mlx/discussions/2422), [9to5Mac MLX-NVIDIA](https://9to5mac.com/2025/07/15/apples-machine-learning-framework-is-getting-support-for-nvidia-gpus/), [Korthikanti 2205.05198](https://arxiv.org/abs/2205.05198), Chen 2016 arXiv:1604.06174, [Megatron fusion docs](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/nemo_megatron/mcore_customization.html), [TVM fusion arch](https://tvm.apache.org/docs/arch/fusion.html), [TVM external lib dispatch](https://tvm.apache.org/docs/arch/external_library_dispatch.html), [TVM-FFI compiler integration](https://tvm.apache.org/ffi/guides/compiler_integration.html), [CUDA Prog. Guide §5.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), apple/ml-cross-entropy, linkedin/Liger-Kernel, Dao-AILab/{flash-attention,quack}, flashinfer-ai/flashinfer, HazyResearch/ThunderKittens.
