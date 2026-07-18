# Model, Training, Generation, and MLX Conversion Review

Date: 2026-07-14

## Executive verdict

**NO-GO for a full NAM56R training claim, a graph-supervised DSA claim, a
production checkpoint publication, or a code-generation quality claim.** The
current repositories contain several real and useful pieces, but those pieces
do not compose into one verified model/training/generation system:

1. Static NAM routing is correct, but MLX `mla` is full-rank MHA/SDPA and the
   MLX Hybrid `dsa` route is not the graph-supervised learned indexer used by
   `DenseCppLM`.
2. The strongest H200 receipts prove a 625M dense GQA model, first without
   graph bias and then with dense post-scale graph bias through sequence 16K.
   They do not prove real DSA, latent MLA, the 52-layer NAM route stack, or the
   current graph auxiliary objective.
3. The MLX graph objective trains the already-prior-biased score. The fixed
   `beta * S_graph` term can lower the loss without the neural indexer learning
   the graph. Both MLX and CUDA still materialize quadratic score/prior tensors.
4. Stage-1 has a real single-forward CE plus graph loss, but the canonical
   runner is a finite fixed-LR smoke loop without scheduler, checkpoint,
   restart, validation, generation, or compile gates. Its packed-document CE
   consumer can accept cross-document targets.
5. FIM, AST-FIM, typed IFIM, commit transduction, and stable-loop components are
   implemented, but several public contracts are weaker or more synthetic than
   their names imply and none has a current mixed-objective training receipt.
6. Raw DCP to MLX conversion now has a real v4 artifact and strict reload
   parity, but the reference is a handwritten NumPy forward, not actual
   Megatron/Transformer Engine logits. The evaluator ignores the v4 runtime
   requirements and can run a graph-required checkpoint graph-off.
7. No model-generated C++ receipt passes compile and execution. Worse, the
   evaluator executes generated binaries without isolation, and the public
   `gen.run(smoke=false)` route still returns synthetic random generation.

Dense GQA is a usable historical baseline. Everything beyond that requires the
acceptance gates in this report.

## Scope and snapshots

This review was rebuilt from source, focused runtime checks, immutable receipts,
and current local execution. The previous contents of this document were not
used as evidence.

| Surface | Snapshot and treatment |
| --- | --- |
| MLX integration | Source audit was frozen at `0df99eea77ae1bcb3a41115c972753a5c03a7fe4`. The shared branch advanced to `bc6c8ed5b6f58092685ef347fc5283034f960a64` before final verification; intervening commits changed conveyor/indexer and tokenizer/domain-contract code, but none of the model/training/generation implementation files cited in the findings. The final tokenizer/domain contract was re-imported and tested separately. |
| CUDA/Megatron integration | `/Volumes/external/sources/cppmega_full_integration` at `9c4dd4b2ff257184dd9faffbc6add1a4db698608`. Its only post-freeze change was unrelated CI-runner coverage. |
| Receipt store | `/Volumes/external/sources/cppmega` at `2e799a2ee13d54413c91edb7a4adbf304ace7b858`; historical outputs were treated as immutable path-specific evidence. |
| Plain MLX sibling | `/Volumes/external/sources/cppmega.mlx` at `8a8de80ad9adae66dc1fbcc05798454582a47bac`, used as a lineage/drift cross-check. The integration checkout is the reviewed MLX implementation. |

The shared MLX worktree changed during the review. In-flight tokenizer/domain
edits briefly broke collection, then landed as `bc6c8ed`; the committed result
imports cleanly and its focused tests pass. Those files were not edited, staged,
or used to rehabilitate any model finding. Only this review document belongs to
this change.

Evidence terms used below:

- **Implemented**: current source contains the path and its local contracts.
- **Runtime-covered**: focused tests or direct probes executed the behavior.
- **Live-proven**: a real H200, DCP, local MLX, compiler, or server execution
  receipt exists and is bound tightly enough to support the stated claim.
- **Receipt-gated**: implementation exists, but no matching current execution
  receipt exists.
- **Broken**: current source, focused execution, or a receipt contradicts the
  advertised contract.

## Findings, severity first

### Critical

#### C-1. Generated C++ is compiled and executed without a sandbox

The CUDA gate concatenates candidate text into source at
`../cppmega_full_integration/scripts/cpp_generation_compile_eval.py:261`, runs
the compiler through an inherited environment at
`../cppmega_full_integration/scripts/cpp_generation_compile_eval.py:281`, and
executes the generated binary directly at
`../cppmega_full_integration/scripts/cpp_generation_compile_eval.py:426` and
`:433`. A temporary directory changes the working directory; it does not limit
filesystem, network, subprocess, credential, CPU, memory, or file access. The
MLX domain-routed harness repeats the pattern at
`scripts/eval_domain_routed_codegen.py:160` and `:189`.

**Impact:** evaluating an untrusted model completion is arbitrary local code
execution with the user's privileges. No codegen acceptance run is safe until
compile and execution happen in a scrubbed, resource-limited, network-disabled
sandbox with read-only inputs.

#### C-2. Public `gen.run(smoke=false)` is synthetic but reports non-smoke

`cppmega_v4/jsonrpc/gen_run_method.py:47` defines `smoke` as the synthetic/real
switch, but `_build_step_fn()` always constructs seeded random logits at `:114`
and a zero-valued fake KV cache at `:150`. The result echoes the caller's
`smoke=false` at `:293`. This method is registered at
`cppmega_v4/jsonrpc/dispatcher.py:279` and exposed by
`cppmega_v4/jsonrpc/server.py:96`.

A direct current-source probe returned generated tokens while reporting
`smoke=false`. Until a checkpoint-, tokenizer-, and model-backed path exists,
`smoke=false` must raise; synthetic generation needs an explicitly synthetic
method/result schema.

#### C-3. Full NAM launchers disable NaN checks and a failed run is recorded OK

Both active full-stack CUDA runners pass
`--no-check-for-nan-in-loss-and-grad` at
`../cppmega_full_integration/scripts/remote_smoke_h200_dsa_9_4_m.sh:732` and
`../cppmega_full_integration/scripts/remote_production_h200_nam56r_v1.sh:441`.
The only full-shape hybrid receipt is a one-step synthetic reachability run:
`../cppmega_full_integration/artifacts/mamba3_wave31_g8_reachability/wave31_g8_h200_reachability_20260430/result.json:12`
identifies the setup, `:154` contains `grad norm: nan` and validation/test `NAN`,
while `:195` records `returncode: 0` and `status: ok`.

**Impact:** this is only backend reachability evidence. It is not healthy
training evidence, and the current launcher can still convert numerical failure
into a nominal success.

#### C-4. Compile oracles share a translation unit with adversarial completions

The functional checks in
`../cppmega_full_integration/evals/cpp_docstring_compile_cases.jsonl:1` and
`evals/cpp_generation_cases.jsonl:2` rely on `assert(...)` in the same
translation unit as model output. An audit probe compiled an incorrect
completion after it emitted:

```cpp
#undef assert
#define assert(x) ((void)0)
```

Current negative tests only cover honest wrong values at
`../cppmega_full_integration/tests/test_cpp_generation_compile_eval.py:72`.
Candidate implementation and hidden oracle must be separate translation units,
with explicit comparison/exit codes and adversarial macro/preprocessor tests.

### P0: blocks production training or evaluation

#### P0-1. There is no integrated full NAM plus graph-supervised real DSA path

The static route expansion is correct. The 52-layer `AEMEAEMEAEMR` layout
contains 13 A, 22 E, 13 M, and 4 R layers; DSA is assigned to layers
`(5,9,13,21,25,29,37,41,45)` and MLA to `(1,17,33,49)`, as locked by
`tests/test_nam56r_pattern.py:124` and `:174`.

The execution semantics do not match that label:

- The MLX adapter at `cppmega_mlx/recipes/nam56r.py:158` does not pass MLA
  low-rank settings into the Hybrid model. `cppmega_mlx/models/hybrid_lm.py:288`
  aliases `mla`, `full`, and `gqa` to one SDPA path; KV heads default to Q heads
  in `cppmega_mlx/nn/attention.py:169`, and Q/K/V are full-rank projections at
  `:628`. This is MHA, not MLA.
- Hybrid `mode="dsa"` takes sparse Path B/C only under explicit runtime policy;
  otherwise it falls through to dense SDPA at
  `cppmega_mlx/nn/attention.py:1040`, `:1099`, and `:1167`. Its sparse indices
  are deterministic causal/mask candidates from `:816-900` and `:963-1007`,
  not a learned graph-supervised indexer.
- The separate `DenseCppLM` `GraphIndexedAttention` does implement
  `I_final = I_neural + beta * S_graph` at
  `cppmega_mlx/models/dense_cpp_lm.py:272-377` and consumes graph bias at
  `:713-781`, but it is an all-attention dense model, not Hybrid NAM.
- MLX `HybridTinyLM` accepts no relation/kind graph prior at
  `cppmega_mlx/models/hybrid_lm.py:2133`.
- CUDA selects real DSA and real Megatron MLA at
  `../cppmega_full_integration/cppmega/megatron/nam56r_full_spec.py:146-206`,
  but the dense graph-bias patch raises on MLA at
  `../cppmega_full_integration/cppmega/megatron/graph_route_attention_bias_patch.py:452-470`.

The result is three different surfaces: dense graph-biased GQA, MLX learned
top-k full-rank GQA, and CUDA sparse latent MLA. None is a proven full NAM plus
graph-supervised DSA system.

#### P0-2. MLX graph supervision can learn the fixed prior instead of the indexer

MLX builds selection scores by adding the fixed graph term at
`cppmega_mlx/nn/sparse_mla.py:515-535`. Stage-1 derives labels from the same
relation prior at `cppmega_mlx/training/stage1_production.py:221-246`, then sends
the already-biased scores into BCE/ranking loss at
`cppmega_mlx/training/objective_mixer.py:1095-1145` and `:1320-1335`. KL is
explicitly disabled at `cppmega_mlx/training/objective_mixer.py:1120-1129`.

The regression at `tests/test_indexer_losses.py:198-231` demonstrates the
shortcut: increasing `beta` to 10 reduces graph loss and produces a gradient
that encourages more prior. Neural scores are ReLU/nonnegative at
`cppmega_mlx/training/indexer_losses.py:137-169`, so negative BCE examples
cannot be driven below a log(2) floor. Top-k selection is detached at
`cppmega_mlx/nn/sparse_mla.py:617-651`; CE cannot repair the neural selector.

CUDA has the right separation concept: it subtracts `beta * S_graph` before the
auxiliary loss at
`../cppmega_full_integration/cppmega/megatron/dsa_indexer_fused_patch.py:789-803`.
MLX must expose separate `neural_scores` and `selection_scores`, train only the
former, use signed logits, and prove loss invariance to fixed `beta`.

#### P0-3. Both graph/indexer implementations remain quadratic

MLX creates full `[B, Hi, S, S]` dot products and `[B, S, S]` scores at
`cppmega_mlx/nn/sparse_mla.py:470-574`, performs full `argsort` at `:577-651`,
then calls the reference gather path at `:654-726` instead of the production
Path C dispatcher. Compiler graph priors are also dense `[B,S,S]` arrays at
`cppmega_mlx/nn/code_graph_routes.py:319-418`.

CUDA's fused indexer removes one larger intermediate but retains a final
`[B,S,S]` score at
`../cppmega_full_integration/cppmega/megatron/dsa_indexer_fused_patch.py:933-1013`.
Its dense graph-bias wrapper documents the 4 GiB BF16 allocation for
`B=8,S=16384` at
`../cppmega_full_integration/cppmega/megatron/graph_route_attention_bias_patch.py:137-153`.
The 16K receipt used microbatch 2, still about 1 GiB per dense-layer bias path.

Real sparse CUDA attention exists at
`../cppmega_full_integration/cppmega/megatron/dsa_sparse_attention.py:31-138`
and the latent SparseMLA adapter at
`../cppmega_full_integration/cppmega/megatron/sparse_mla_ops/sparse_mla.py:356-411`,
but the adapter makes full layout copies with `.permute(...).contiguous()` at
`:589-603` and `:648-685`.

No 32K/64K/128K claim is credible until candidate construction, scoring,
selection, and attention are bounded-sparse with allocation traces proving no
quadratic staging and no wrapper-sized full-tensor copies.

#### P0-4. Stage-1's declared recipe, implementation, and training mechanics disagree

`configs/stage1_cpp_foundation_dense500m.yaml:13-21` and `:80-103` describe a
first dense-GQA/CE phase followed by DSA/KL warmup. The canonical MLX production
runner rejects non-DSA execution at
`cppmega_mlx/training/stage1_production.py:451-512`. It creates a fresh model and
fixed-LR AdamW, loops for a requested step count, and exits at `:522-579`; it has
no scheduler, clipping, checkpoint/resume, held-out evaluation, generation, or
compile gate. Richer CLIs return into this path before their own persistence and
evaluation logic at `scripts/train_eval_stage1.py:630` and `:1023`.

The CE consumer also trusts `loss_mask` without enforcing same-document target
transitions at `cppmega_mlx/data/batch.py:210-240`. Stage-1 masks graph pairs by
document at `cppmega_mlx/training/stage1_production.py:221-246` but forwards `target_mask` directly to
CE. The fixture in `tests/test_stage1_combined_graph_objective.py:57-100` changes
document ID while keeping an all-one loss mask and expects five live targets at
`:187-197`, thereby locking in one cross-document CE target.

The canonical runner must reject every positive CE target where source and
target document IDs differ, and it needs one exact-restart training state that
binds model, optimizer, RNG, scheduler, data cursor, objective window, and
accumulation state.

#### P0-5. H200 receipts prove dense GQA, not full NAM, DSA, or full recompute

The 5,000-step run is a 48-position, 625,218,594-parameter model at
`../cppmega/outputs/nebius/cppmega-h200-megatron-1782697038/seq_1024_bs_192.log:839-844`.
Its attention receipt is `no_bias` at `:6261`. The graph curriculum explicitly
reports 20 Q heads, 4 KV groups, `post_scale_bias/b1ss`, FP8, and no DSA
arguments at
`../cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/stage_1_seq_1024_bs_192.log:160-198`
and `:1162-1168`.

The nominal full production launcher is not runnable under defaults: it chooses
FP8 indexer dtype at
`../cppmega_full_integration/scripts/remote_production_h200_nam56r_v1.sh:84-88`
and raises because that path is unsupported at `:227-240`. It also deletes its
checkpoint directory before using the same path for `--load` and `--save` at
`:50-51` and `:441-446`.

The Europe "full recompute" case is actually selective `transformer_layer`
recompute at
`../cppmega_full_integration/scripts/remote_train_h200_nam56r_europe_sweep.sh:61-74`
and `:288-296`; `run_config ... || true` hides failures at `:437-445`.
Historical H200 receipts prove Transformer Engine tensorwise/hybrid FP8 GEMMs
and selective MLP recompute. They do not prove FP8 attention/indexer, full
recompute, or a healthy full NAM step.

#### P0-6. DCP v4 parity is narrow and its runtime contract is advisory

`outputs/mtr005_latest/model.json:25-43` records a strict-reload FP32 eight-token
parity result with max absolute error `0.0032396913`; `:78-100` hashes the real
torch-dist source artifacts, and `:101-119` binds a 20Q/4KV GQA iteration-2391
source. This is materially stronger than the historical v1 conversion.

It is still not Megatron runtime parity. The converter reconstructs a 24-block
NumPy forward at
`scripts/convert_megatron_dense500m_torchdist_to_mlx.py:1201-1291`, compares
only eight positions at `:1294-1375`, filters non-tensor DCP metadata at `:579`,
and explicitly rejects DSA conversion at `:104-110`. Transformer Engine
`_extra_state`, FP8 runtime semantics, packed documents, long-position behavior,
Mamba/M2RNN/MoE/MLA, and optimizer/RNG state are not proved.

The manifest requires graph routes and edge kinds at
`outputs/mtr005_latest/model.json:54-75`. The evaluator reads only the `config`
object at `scripts/cpp_jsonl_generation_compile_eval.py:1086`, selects graph
behavior from CLI/case settings at `:1109-1166`, and permits non-strict loading.
It can therefore execute this graph-required checkpoint graph-off, which the
fresh local run did.

The source checkpoint also has an identically zero structure residual: the real
DCP contains zero `stacked_emb.weight` and zero `up_proj.weight`, despite
`structure_tokens: 8` in the parity receipt at
`outputs/mtr005_latest/model.json:43`. Current CUDA source has repaired
initialization, but that cannot retroactively change iteration 2391.

#### P0-7. No current codegen acceptance run passes

Across the 20 tracked `local_mlx_*` compile reports, 80 model candidates yielded
`compiled=2`, `ran=0`, `passed=0`. Historical H200 generation loaded iteration
5000 and produced four candidates, all `compiled=0`, at
`../cppmega/outputs/nebius/cppmega-h200-generation-1782753301/compile_report.json:144-151`.

Fresh current-source local runs were also executed:

- v4 checkpoint, graph-off, one source-prefix case, 16 generated tokens:
  completion `acc.set_max_capacity(clamped);\n acc\n`, length-truncated,
  `compiled=0`, `ran=0`, `passed=0`, `pass@1=0`.
- v1 checkpoint, graph-off: the same completion and `0/1` compile.
- v1 and v4 graph-on attempts: failed before inference because the current
  environment lacks Python `libclang` bindings. Existing hash-bound prompt
  caches did not avoid the indexer import/build path.

The graph-off v4 execution is useful runtime evidence but is not a valid model
acceptance receipt because the manifest requires graph routes. The codegen path
remains receipt-gated even after graph tooling is repaired.

#### P0-8. CUDA graph-objective control and weighting are inconsistent

`../cppmega_full_integration/cppmega/megatron/graph_objective_loss.py:143-159`
defines the graph-aux enable switch, but the wrapper checks only graph routes at
`../cppmega_full_integration/cppmega/megatron/dsa_indexer_fused_patch.py:816-832`.
`indexer_weight` is parsed and required positive at
`../cppmega_full_integration/cppmega/megatron/graph_objective_loss.py:15-75` but
omitted from the formula at `:221-276`.
Coverage defaults to top-k 8 at `:69-77`, while token-expanded chunk edges can
create more than eight positives per query, making zero coverage loss
mathematically impossible.

Canonical H200 launchers set graph auxiliary weight/indexer coefficient to zero
and skip indexer loss at
`../cppmega_full_integration/scripts/nebius_h200_megatron_cpp_world_sweep.py:661-668`
and
`../cppmega_full_integration/scripts/nebius_h200_megatron_cpp_world_curriculum.py:344-352`.
The real H200 graph
receipt is therefore CE plus dense graph-attention bias, not learned graph DSA.

### P1: correctness, semantics, and reproducibility gaps

#### P1-1. Objective names overstate some supervision semantics

The nine-task mixer and exact quota machinery are real at
`cppmega_mlx/training/task_mixer.py:43-70` and `:184-242`. Plain FIM and source
remapping are implemented at `cppmega_mlx/data/fim.py:50-55` and
`cppmega_mlx/training/objective_data.py:499-760`. Packed commit constituent
binding is exact at `cppmega_mlx/training/objective_data.py:264-370`.

The remaining semantic gaps are concrete:

- Public AST-FIM and AST-IFIM silently fall back to character transforms at
  `cppmega_mlx/data/ast_fim.py:324-354` and `:406-443`. The production mixer
  prechecks eligibility, but direct public callers can still misreport task
  identity.
- Typed IFIM carries a real typed instruction but samples a random synthetic
  target at `cppmega_mlx/training/objectives.py:276`, `:304`, and `:350`; it is
  not commit-change instruction following.
- `pre_to_post` is a real changed dependency-chain excerpt with only a bounded
  preamble, not full-file editing, at
  `tools/clang_indexer/process_commits.py:1679`, `:1692`, `:1730`, and `:1754`.
- Change masks expand touched lines to entire functions/classes and use stripped
  text keys that alias duplicate lines at
  `tools/clang_indexer/process_commits.py:586`, `:638`,
  `:650`, and `:706`. Stage-1 strips the misaligned assembled-document masks at
  `cppmega_mlx/training/objective_data.py:424-460`, then materializes zero change masks at
  `cppmega_mlx/training/megatron_objectives.py:1125` and `:1251`. This avoids
  corrupt labels but supplies no change-mask supervision.
- World-model "real" pre/post states are slices of one assembled token vector at
  `cppmega_mlx/data/trajectory_packet.py:227-245` and are labeled
  `is_synthetic=False` at `:319`.

There is a current live bundle publication receipt, but it proves remote object
hashes rather than a mixed-objective train:
`../cppmega_full_integration/outputs/review/publish_megatron_bundle_nebius_live_verification_20260714T094239Z.json:505-587`.

#### P1-2. Stable loop is a correctness reference, not a production training path

`cppmega_mlx/nn/stable_loop.py:1-44` defines the coupled-state semantics;
convergence fails closed at `:91-101`, scaling is explicit at `:255-274`, and
training versus inference behavior is separated at `:371-430`. This is good
reference code.

The training integration remains prototype-only. `truncated_bptt_step()` at
`cppmega_mlx/training/fixed_point.py:73-176` performs one optimizer update per
window but has no non-test caller. Its API says callable `z0` is built inside
differentiation, while the implementation evaluates it before the closure and
stop-gradients it at `:104-108` and `:140-156`. The only model integration is
the tiny smoke model at `cppmega_mlx/models/stable_loop_cpp_lm.py:1-72`; there is
no production checkpoint, generation, or H200 receipt.

#### P1-3. Packed-document isolation is incomplete in CUDA and optional MLX paths

CUDA changes the LM loss mask at
`../cppmega_full_integration/cppmega/megatron/structure_dataset_patch.py:1422`
and creates graph document IDs at `:1491`, but does not reset attention masks or
positions at `:1566`. The real stage-5 receipt records
`reset_attention_mask=False` and `reset_position_ids=False` at
`../cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/stage_5_seq_16384_gbs_8_mbs_2.log:633`
while loading packed doc sidecars at `:1060`. Later documents can attend prior
documents.

CUDA n-gram enrichment also receives no document IDs at
`../cppmega_full_integration/cppmega/megatron/custom_embedding.py:164`; shifted
tokens can cross boundaries in
`../cppmega_full_integration/cppmega/features/engram/ngram_hash.py:117`.
MLX Hybrid's optional n-gram call has the same omission at
`cppmega_mlx/models/hybrid_lm.py:2204`, although the module supports document
boundaries at `cppmega_mlx/nn/ngram_hash.py:139`.

#### P1-4. Accumulated compiled gradients are not live-token weighted

CE is a mean over live tokens at `cppmega_mlx/training/loss.py:62-69`.
`CompiledPretrainingStep` sums these mean gradients and divides by the number of
microbatches at `cppmega_mlx/training/compiled.py:2019-2033`. With unequal loss
masks this differs from the true global live-token mean. The eager loop already
has the correct `ntokens * gradient` weighting at
`cppmega_mlx/training/loop.py:121`. Current production uses accumulation one, so
this becomes active only when accumulation is enabled.

#### P1-5. Generation state and sampling are incomplete beyond eager full-prefix

The specialized evaluator rebuilds the full active prefix every token at
`scripts/cpp_jsonl_generation_compile_eval.py:1274`; `DenseCppLM` has no KV
argument at `cppmega_mlx/models/dense_cpp_lm.py:916`. Generic generation extends
structure/document fields but not complete domain/role/confidence/graph state at
`cppmega_mlx/inference/generation.py:791-888`. Hybrid cache is passed only to
attention routes at `cppmega_mlx/models/hybrid_lm.py:2335`; Mamba3, M2RNN, and
Engram have no persisted route state in that path.

Top-p sampling at `cppmega_mlx/inference/sampling.py:62` excludes the token that
crosses the cumulative threshold. For probabilities `[0.6, 0.3, 0.1]` with
`p=0.8`, the valid nucleus is the first two tokens, but current code retains
only the first.

#### P1-6. CUDA and MLX continued training use different positional architectures

The converter requires a RoPE-only source at
`scripts/convert_megatron_dense500m_torchdist_to_mlx.py:274`. MLX enables RoPE
but always instantiates and adds a learned position table at
`cppmega_mlx/models/dense_cpp_lm.py:482` and `:562`. Conversion zeros that table
at `scripts/convert_megatron_dense500m_torchdist_to_mlx.py:553`, but native MLX training
can update it. Inference starts neutral; continued-training equivalence is not
preserved without a real `rope_only` architecture mode.

#### P1-7. A current focused DSA contract test fails

With a compatible MLX/pytest/pyarrow environment, current source produced one
real focused failure:
`tests/test_dsa_indexer.py:228-256` constructs `attention_mode="dsa"`, expects a
missing graph bias to fail, then supplies `block_bias`. The model rejects the
bias because `graph_routes_enabled` remains false at
`cppmega_mlx/models/dense_cpp_lm.py:722-735`. Six tests passed before this
failure. The configuration contract and the test disagree; neither should be
treated as a green DSA gate until the intended invariant is made explicit.

### P2: hardening gaps

1. `scripts/cpp_jsonl_generation_compile_eval.py:82-84` defaults its compile
   gate to the plain `/cppmega` checkout rather than the paired integration.
   Current tests use another checkout with a matching script hash and mask the
   wiring error.
2. Generation summaries at `scripts/cpp_jsonl_generation_compile_eval.py:2054`
   do not bind checkpoint/manifest hashes, evaluator commit/worktree, argv, or
   non-strict loading state.
3. DCP publication uses two independent `os.replace` calls at
   `scripts/convert_megatron_dense500m_torchdist_to_mlx.py:1477`; it records
   source hashes but no target safetensors hash or atomic pair completion marker.
4. `scripts/train_hybrid_tiny.py:245` documents DSA ranks as one-based but
   forwards them unchanged at `:478` and `:898` into the zero-based contract at
   `cppmega_mlx/recipes/pattern.py:180`.

## Dense graph bias versus real DSA

These terms must not be conflated in future receipts:

| Path | What selects keys | Attention executed | Graph role | Complexity | Live status |
| --- | --- | --- | --- | --- | --- |
| CUDA dense GQA graph bias | No selector; every causal key remains | Dense TE GQA | Full post-scale `[B,1,S,S]` bias | Dense attention plus dense bias | Proven through 16K historically |
| MLX `DenseCppLM` DSA | Learned full score plus fixed graph prior, full sort/top-k | Sparse gather over full-rank GQA K/V | Prior affects both selection and, incorrectly, auxiliary loss | Quadratic selector and sort | Focused source/tests; no train receipt |
| MLX Hybrid `dsa` | Deterministic recent/mask-valid candidates under opt-in policy | Sparse Path B/C, otherwise dense SDPA | No learned graph prior | Sparse only under explicit policy | Not graph-supervised DSA |
| CUDA real DSA | Learned dense lightning-indexer score plus graph prior | Real sparse absorbed/latent MLA | Prior is removed before auxiliary loss | Quadratic selector, sparse final attention | Implemented; no accepted H200 DSA receipt |

## Implemented/runtime/live proof matrix

| Surface | Implemented | Focused runtime proof | Live proof | Verdict |
| --- | --- | --- | --- | --- |
| Static NAM 13/22/13/4 and 9+4 routing | Yes | Exact MLX expansion; CUDA route subset `24 passed, 1 deselected` | One synthetic one-step reachability receipt with NaN | Static only; healthy runtime unproven |
| True MLX MLA | No | Tests exercise the aliased SDPA path | None | Broken/misnamed |
| Dense GQA | Yes | Profile/launcher tests | 5,000-step H200 run | Proven historical baseline |
| Dense GQA plus graph bias | Yes | Bias construction tests | Five H200 buckets through 16K | Proven historical dense path |
| MLX learned `I_neural + beta*S_graph` | Yes | Unit/gradient tests plus current contract failure | No training/checkpoint receipt | Implemented, objective semantics blocked |
| MLX sparse Path C | Yes, separately | Kernel/attention tests exist | No graph-indexer-to-Path-C receipt | Receipt-gated and disconnected |
| CUDA real sparse DSA/MLA | Yes | Graph/sparse contract tests | No finite current H200 DSA receipt | Receipt-gated |
| Graph auxiliary objective | MLX and CUDA source exist | MLX one-forward decomposition tests; CUDA contract tests | Canonical H200 coefficients were zero | Not live-proven |
| Mixed FIM/IFIM/commit objectives | Yes, with semantic limits | Focused objective suite included in `152 passed` run | Bundle hashes only | No mixed-objective train receipt |
| Stable loop/TBPTT | Reference/helper only | Focused tests included in `152 passed` run | None | Experimental |
| TE FP8 | Yes | Launcher/profile tests | H200 tensorwise/hybrid FP8 | Proven for GEMM path, not DSA attention/indexer |
| Selective MLP recompute | Yes | Launcher tests | H200 receipts | Proven historical |
| Full recompute | No correct launcher | String path is mislabeled and fail-open | None | Broken/unproven |
| Exact optimizer/RNG restart | Current preflight source exists | Contract tests | Historical reload was model-only | Receipt-gated |
| Raw DCP to MLX v4 | Yes, dense GQA only | Tiny conversion tests plus real strict reload | Real source DCP converted locally | Serialization/layout proven; runtime parity unproven |
| v4 graph-required generation | Eager path exists | Graph-off execution; graph-on blocked by libclang environment | `0/1` compile | Failed, not acceptance evidence |
| C++ compile/link/run gate | Yes | Gold and adversarial probes | Model candidates `0` passed | Unsafe and oracle-defeatable |

## Receipt ledger

### H200 dense baseline

`../cppmega/outputs/nebius/cppmega-h200-megatron-1782697038` ran sequence 1024,
GBS/MBS 192, 5,000 steps, BF16, Transformer Engine hybrid/tensorwise FP8, and
selective MLP recompute. This is **983,040,000 nominal token slots**. Iteration
5000 had zero skipped/NaN iterations and saved a checkpoint at
`../cppmega/outputs/nebius/cppmega-h200-megatron-1782697038/seq_1024_bs_192.log:6252-6254`;
one-batch validation/test losses were
`1.453965/1.427063` at `:6327-6338`. It was dense GQA with no graph bias.

`../cppmega/outputs/nebius/cppmega-h200-megatron-1782706113` loaded model weights
from iteration 5000 and advanced to target 5001. It used `--no-load-optim` and
`--no-load-rng`; it proves weight reload, not exact continuation.

### H200 dense graph-bias curriculum

`../cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/summary.log:1-24`
records successful stages:

- 1024 / GBS 192 / MBS 192 / iteration 1421
- 2048 / GBS 96 / MBS 96 / iteration 1686
- 4096 / GBS 40 / MBS 40 / iteration 2311
- 8192 / GBS 16 / MBS 4 / iteration 2756
- 16384 / GBS 8 / MBS 2 / iteration 2391

This is **1,664,122,880 nominal token slots**. Larger 4K/8K batches OOMed before
the reduced variants. Final training was finite at
`../cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/stage_5_seq_16384_gbs_8_mbs_2.log:8453`;
validation/test losses were
`0.5113884/0.4466256` at `:8540-8551`; the stage ended OK at `:8555`. This proves
dense 20Q/4KV GQA plus graph bias, not DSA or exact continuation across stages.

### Conversion and local macOS evaluation

The real v4 artifact is `outputs/mtr005_latest/model.safetensors` with manifest
`outputs/mtr005_latest/model.json`. The manifest binds the iteration-2391 raw
DCP, hashes its source files, records graph-route runtime requirements, and
passes strict-reload eight-token FP32 parity. It does not bind actual Megatron
runtime logits or the target safetensors hash.

Fresh local MLX execution against that v4 artifact completed graph-off generation
but failed code quality (`compiled=0`, `ran=0`, `passed=0`). Graph-on execution
failed before generation because Python `libclang` bindings were unavailable.
No local or H200 model-generated candidate has a passing compile/run receipt.

### Bundle receipt

`../cppmega_full_integration/outputs/review/publish_megatron_bundle_nebius_live_verification_20260714T094239Z.json:2-681`
verifies 64 remote objects and their hashes. It is publication/integrity proof,
not proof that the current objective contract trained successfully.

## Current-source repairs that make older conclusions stale

The following are real improvements and should be preserved:

1. MLX Stage-1 now computes CE and graph loss from one decoder invocation at
   `cppmega_mlx/training/objective_mixer.py:1279-1335`; the old double-forward
   criticism is no longer valid.
2. Production AST objective selection prechecks real interior-chunk eligibility
   at `cppmega_mlx/training/objective_mixer.py:388`; the public fallback APIs remain unfixed.
3. FIM/IFIM token-source remapping and exact packed commit binding exist at
   `cppmega_mlx/training/objective_data.py:264-370` and `:499-760`.
4. Immutable production bundle validation binds artifact sets, restore receipts,
   tokenizer/data contracts, and producer cursors at
   `cppmega_mlx/data/production_bundle.py:448-590` and `:1221-1306`.
5. CUDA removes fixed graph prior from the indexer auxiliary score at
   `../cppmega_full_integration/cppmega/megatron/dsa_indexer_fused_patch.py:789-803`
   and clones active graph tensors for
   backward at `:864-884`.
6. Raw torch-dist DCP reading, grouped GQA mapping, strict reloaded safetensors
   parity, source hashes, and a v4 manifest now exist at
   `scripts/convert_megatron_dense500m_torchdist_to_mlx.py:248-430` and
   `:1326-1479`.
7. Current H200 preflight source includes finite-loss/gradient, H200 SM90,
   nonzero graph-prior, stale-root, and cold-restore checks at
   `../cppmega_full_integration/scripts/h200_megatron_preflight.py:493-588` and
   `:1040-1284`. No H200 receipt yet exercises the complete hardened preflight.
8. Repository compile/link support, candidate identity, pass@k, and failed-report
   preservation now exist. They do not resolve unsafe execution, oracle
   isolation, or the absence of passing model output.

## Repair plan

### Phase 0: stop false and unsafe proof

1. Sandbox compiler and candidate execution with no network, scrubbed
   environment, read-only inputs, CPU/RSS/process/file limits, and retained
   failure artifacts.
2. Move hidden tests into a separate translation unit; ban `assert` as the sole
   oracle and add macro/preprocessor escape regressions.
3. Make `gen.run(smoke=false)` fail until it authenticates and loads a real
   checkpoint/tokenizer/model.
4. Define one immutable receipt schema binding source commits and dirty state,
   image digest, exact argv/environment, recipe, bundle/tokenizer/objective
   hashes, checkpoint/manifest hashes, generated candidates, and gate hashes.

### Phase 1: establish a truthful dense GQA baseline

1. Fix MLX and CUDA packed-document attention, position, CE, and n-gram
   isolation. Add adversarial two-document suffix-invariance tests.
2. Add real `rope_only` MLX mode and remove/freeze unmatched learned positions.
3. Make Stage-1 scheduler/checkpoint/restart atomic and bind optimizer, RNG,
   scheduler, dataset cursor, objective window, and accumulation state.
4. Compare actual Megatron/TE BF16 logits with strict-reloaded MLX logits on
   hash-bound tokens and sidecars. Report FP8 separately.
5. Retrain/reconvert after repaired structure initialization and require a
   nonzero structure-on/off logit delta.

### Phase 2: isolate dense graph bias as its own experiment

1. Use one canonical sparse graph-sidecar schema and edge-kind mapping in CUDA
   and MLX.
2. Run graph-off versus dense-graph-bias with identical weights/data/seeds.
3. Receipt nonzero relation/kind counts and per-route logit/loss deltas. Do not
   call this DSA.
4. Replace dense `[B,S,S]` prior construction before extending context beyond
   the proven 16K path.

### Phase 3: repair and prove real DSA

1. Separate `I_neural` from `I_final`; compute BCE/coverage/KL only on signed
   neural logits. Freeze or remove `beta` from auxiliary optimization.
2. Align CUDA AUX gating, apply `indexer_weight` exactly once, and formulate
   achievable chunk/edge coverage targets.
3. Score only bounded CSR/block candidates plus local/sink candidates. Forbid
   full score matrices, full argsort, and full-tensor wrapper copies.
4. Connect MLX graph selection directly to a forced production Path C kernel;
   raise if that kernel is unavailable.
5. Prove q/k/head parameter deltas, fixed-prior loss invariance, causal/document
   isolation, MLX/CUDA numerical agreement, exact sparse-kernel dispatch, and
   zero fallback events.

### Phase 4: make full NAM real

1. Implement actual MLX MLA low-rank projections or remove the `mla` label.
2. Thread graph conditioning through CUDA MLA and MLX Hybrid routes.
3. Emit a runtime module inventory from instantiated PP/VPP modules proving
   13 A, 22 E, 13 M, 4 R, exactly 9 DSA, and 4 true MLA layers, with projection
   classes/ranks and kernel names.
4. Remove unsupported FP8 defaults, NaN bypasses, automatic sparse-kernel
   fallback, checkpoint deletion, and swallowed launcher failures.

### Phase 5: close objective and stable-loop semantics

1. Remove public AST character fallbacks. Plain character FIM must be selected
   as a separate task.
2. Bind typed IFIM to exact changed/AST target coordinates or rename it
   `typed_random_ifim`.
3. Derive commit masks from hunk/token coordinates, not whole-function ranges or
   stripped-text lookup; preserve independently aligned pre/post masks.
4. Rename excerpt transduction or produce bounded complete old/new files.
5. Either integrate stable loop/world action into the canonical persisted
   trainer, including a real next-action objective, or mark them experimental in
   all configs and receipts.

### Phase 6: scale only after contract closure

Context progression is 1K -> 4K -> 16K -> 32K -> 64K -> 128K. Every step must
retain allocation traces, peak RSS/HBM, throughput, finite loss/gradient,
selected-k distribution, graph candidate counts, document-boundary counters,
and exact checkpoint continuation. A single observed `[B,S,S]` allocation or
unexpected kernel fallback fails the scale gate.

## Training acceptance stages

| Stage | Execution | Required gate |
| --- | --- | --- |
| T0 contract freeze | Local macOS, no training | One hash-bound recipe; clean source snapshots; bundle/tokenizer/objective/sidecar validation; actual CUDA-to-MLX logits |
| T1 tiny dense GQA | Local MLX overfit, save, cold load, resume | Same loss/weights/data cursor after uninterrupted versus resumed path; packed-doc isolation |
| T2 dense H200 baseline | 20-step preflight, then 1K and 5K | Finite loss/grad; exact optimizer/RNG/scheduler restart; BF16 and FP8 reported separately |
| T3 dense graph-bias ablation | Matched graph-off/on at 1K, 4K, 16K | Nonzero relation/kind consumption; no cross-doc edges; predefined delta; no DSA claim |
| T4 learned DSA | One A layer, then all 9 DSA layers | Neural-only objective learns; no quadratic allocations; forced sparse kernel; checkpoint/restart and generation |
| T5 full NAM56R | 52-layer 9 DSA + 4 true MLA | Runtime inventory; 20+ finite real-data steps; zero fallback; cold continuation; NAM generation receipt |
| T6 mixed objectives | Causal/FIM/AST/IFIM/commit/recovery quotas | Exact realized quotas, per-objective live tokens/losses, route retention, semantic goldens, checkpoint hash |
| T7 long context | 32K/64K/128K after T0-T6 | Bounded sparse memory, measured HBM/throughput, finite continuation, no hidden dense/copy path |

## Codegen evaluation stages

1. **E0 evaluator trust:** sandbox execution, separate hidden oracle, adversarial
   preprocessor/system-call tests, immutable gate receipt, and consistent exit
   semantics.
2. **E1 oracle calibration:** gold fixtures must pass and deliberately wrong or
   malicious fixtures must fail. This validates only the harness.
3. **E2 local dense baseline:** use v4 checkpoint and manifest, strict load, both
   shipped MLX cases, nontrivial generation lengths, at least 10 samples per
   task when reporting pass@10, and the paired compile/link/run gate.
4. **E3 graph ablation:** same weights/seeds/prompts graph-off and graph-on;
   graph-required checkpoints must reject graph-off unless the run is explicitly
   labeled an ablation. Receipt graph hashes/counts for every decode window.
5. **E4 objective modes:** source-prefix, FIM, AST-FIM, IFIM, and commit prompts
   must carry their declared sidecars or fail closed. Zero-sidecar transformed
   modes are reported separately and cannot prove graph/domain conditioning.
6. **E5 current H200 checkpoint:** generate, transfer, compile, link, and run
   before deleting the instance; bind image/source/checkpoint hashes and retain
   all raw candidates and failures.
7. **E6 release benchmark:** pre-register pass@1/pass@10 and compile/run
   thresholds against the dense baseline. Process exit alone is not a quality
   score, and `0/N` is an executed failure, not missing evidence.

## Verification performed for this review

- Eight independent read-only Codex reviews were run with mandatory
  `gpt-5.6-sol` and `model_reasoning_effort="ultra"`, covering NAM/GQA,
  DSA/graph objective, objectives/FIM/commit, stable loop/training, H200,
  DCP conversion, generation/eval, and a holistic cross-check.
- Focused independent MLX files completed **152 passed in 5.89s** across
  objective/FIM, stable-loop, compiled graph contracts, and DCP conversion.
- The final `bc6c8ed` tokenizer/domain-contract suites completed **55 passed,
  1 skipped in 3.24s**.
- The critical DSA set completed six tests and then hit the genuine
  `graph_routes_enabled=false` contract failure described in P1-7.
- Another Stage-1 slice completed **56 passed, 1 deselected** before a fresh
  subprocess could not see `pyarrow`: `tests/conftest.py:102-145` deliberately
  strips `PYTHONPATH`, while the project `.venv` does not install `pyarrow`.
  No aggregate clean suite is claimed.
- CUDA pure contract subsets reported `24 passed, 1 deselected`, `14 passed`,
  and `12 passed`; broader import used a Megatron version whose
  `get_batch_on_this_tp_rank` seam did not match the checkout. This is an
  environment/version block, not CUDA runtime proof.
- Fresh local v4 and v1 graph-off generation executed and both produced `0/1`
  compile. Fresh graph-on attempts failed before inference on missing Python
  `libclang` bindings.
- No remote training was launched. No model, training, generation, test, or
  configuration source was modified.

## Remaining risk summary

The highest residual risks are arbitrary code execution from generated output,
false non-smoke API provenance, NaN-accepting full-stack launchers, an
oracle-defeatable compile gate, prior-shortcut graph supervision, quadratic DSA
memory, incomplete packed-document isolation, and the absence of any current
full NAM/DSA/mixed-objective/codegen success receipt. These are release blockers,
not documentation gaps.
