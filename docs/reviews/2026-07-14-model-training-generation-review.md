# Model, Training, Generation, and MLX Conversion Review

Date: 2026-07-14

## Scope and review snapshot

This is a code-and-receipt review, not a restatement of planning documents. It
examines the current target checkout, the current sibling CUDA/Megatron
checkout where data and run provenance live, focused tests, recent commits,
checked-in benchmark receipts, H200 logs, converted checkpoints, and local
macOS generation/compile receipts. No training or expensive benchmark was run.

| Surface | Snapshot reviewed | Worktree state while reviewed |
| --- | --- | --- |
| `cppmega_mlx_full_integration` | Review began at `c54750eb7c28da6d17867a6061933a13795202fd`; final committed source snapshot before staging was `3bd0e6d4f1d5de5cd52b901078d33e5a32527390`. | Active uncommitted FIM/objective edits appeared during review. They were read and classified separately; only this report is owned by the reviewer. |
| `cppmega_full_integration` | `ef44b42b0bf669e0ed5be17910bed8fb8ed98615` | Source worktree was clean; only untracked `outputs/review/` remained. The source-snapshot binding patch observed earlier was committed as `ef44b42`. |
| Historical run/MLX receipts | `/Volumes/external/sources/cppmega/outputs` | Read-only. Receipts are classified by the exact path they prove, not by later claims. |

The deterministic GitHub links below are pinned to those local source commits.
Because both branches were ahead of their remotes, a pinned URL may not resolve
until its local commit is pushed. Local links are the authoritative links for
this review.

Recent target commits were inspected, especially:

- `a8937dc` gates transformed objectives that ambiguously span
  multiple physical sources. It does not remap semantic sidecars after token
  permutation.
- `f81c9ad` aligns objective graph Arrow types.
- `de7e97a` labels otherwise unknown tokens enclosed by a domain
  delimiter pair.
- `95b0ad0`, `8973812`, and `c54750e` harden and
  accelerate cross-repository symbol indexing.
- `c5e8e05` makes world-model transitions action-conditioned.
- `c66ef8d` hardens conversion/eval integrity, but its conversion
  parity reference is still an MLX model loaded with already-mapped tensors,
  not an original Megatron forward.
- `8a1b594` binds materialized objectives to source shards in the
  upstream objective contract. This strengthens source provenance but does not
  make the MLX opener validate a whole immutable bundle.

Sibling commit `ef44b42` adds
`_validate_objective_source_binding()` to bind objective artifacts to
the repaired source snapshot. That is useful upstream hardening, but it does
not make the MLX dataset opener validate a published bundle.

## Executive verdict

The only substantial H200 model-training proof is dense GQA. A second H200
curriculum proves dense GQA with post-scale graph relation bias through 16k.
Neither proves DSA, latent MLA, sparse learned indexing, the hybrid route stack,
the world model, stable loops, or 128k context. The 5000-step dense run consumed
983,040,000 nominal sequence tokens and completed without skipped/NaN
iterations, but its four generation cases produced no compiling program.

The current MLX architecture contains useful and often well-tested components,
but it does not yet form one provenance-bound production training path:

1. The dense learned indexer materializes `(B,S,S)` scores and performs
   a full argsort. It is quadratic and cannot be treated as a 128k DSA path.
2. The committed non-causal materializer zeros or removes semantic, structure,
   graph, and change-mask sidecars. An active uncommitted patch now carries an
   exact token-source map and remaps many token-aligned sidecars for transformed
   code objectives, but it still clears every transformed chunk/graph sidecar,
   leaves commit semantics mostly synthetic/zero, and has no current passing
   end-to-end receipt. With the default mixer, up to 50 percent of samples use
   one of these transformed contracts.
3. The MLX Megatron opener validates local shapes and schemas but does not
   require or hash-check the immutable tokenizer/domain/objective/bundle
   descriptors produced by the sibling pipeline. It can accept a stale or
   mixed artifact set.
4. The canonical Stage-1 runner uses plain next-token cross-entropy and omits
   the graph auxiliary objective. The separate graph objective performs two
   model passes and injects an all-zero edge-kind bias.
5. The converter's current logit parity compares two MLX instances after
   mapping. It does not prove original Megatron/DCP logits match MLX logits.

These are scale blockers. A 10B-token run or a 128k DSA run should not start
until the corresponding acceptance gates in this report pass.

## Evidence vocabulary

The labels in this report are intentionally non-overlapping:

- **Implemented**: reachable current code exists and its declared contract is
  internally coherent enough to inspect.
- **Unit-tested**: focused tests exercise the component contract. This does not
  imply data-to-checkpoint or production-kernel proof.
- **End-to-end-proven**: one immutable receipt covers real data ingestion,
  forward/loss/update, checkpoint/restart, generation, and the relevant
  compile/run gate with the same path.
- **H200-proven**: an exact H200 receipt exercised that named path. H200 proof
  does not transfer to similarly named MLX code.
- **Local macOS-eval-proven**: a real checkpoint was loaded and generation plus
  compile evaluation ran locally. A zero quality score is still execution
  proof, not success.
- **Unproven**: no receipt found that satisfies the relevant claim.

## Architecture and evidence matrix

| Path | Implemented | Unit-tested | End-to-end-proven | H200-proven | Local macOS-eval-proven | Current verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Dense GQA Stage-1 | Yes | Yes | Partial historical train/eval chain, not one current immutable pipeline | Yes, 5k-step dense run and graph curriculum | Yes, load/generate/compile gate ran | Baseline path only; quality gate failed and native MLX double-position behavior differs from converted RoPE-only weights. |
| Dense learned-indexer DSA | Yes, pure-MLX reference | Yes | No | No | No | Quadratic score matrix and argsort; not viable at 128k and not latent MLA. |
| Hybrid `mla` mode | Yes by name | Yes at tiny/reference scale | No | No | No | Classified as dense SDPA and uses full-rank Q/K/V. The name does not establish latent MLA. |
| Hybrid sparse MLA Path C | Yes, deterministic causal sparse indices plus FP8 kernel route | Kernel tests and receipts exist | Tiny fused-kernel execution only | No | Kernel receipt only, not model eval | Not connected to the dense learned indexer; projection-gradient ownership and packed-document production use remain unproven. |
| Graph relation routes/bias | Yes | Yes | No current full path | Yes for dense post-scale relation bias | Yes in repo-prompt eval | Relation signal is proven; categorical edge-kind signal is zeroed in training/eval call sites. |
| Graph indexer auxiliary loss | Yes, separate function | Unit contract exists | No | No | No | Not called by canonical runner; does a second model forward when called. |
| Mamba3/MIMO | Yes, reference and Path C | Yes | Kernel receipts only | No | Kernel benchmark only | Packed documents force reference/auto; explicit Path B/C reject reset masks. |
| MoE | Yes, correctness reference | Yes | No | No | No | Default computes every expert for every token; sparse route host-syncs indices; router auxiliary loss is discarded by HybridTinyLM. |
| M2RNN | Yes, reference and Path C | Yes | Kernel receipts only | No | Kernel benchmark only | Explicit Path B/C reject packed-document resets; local Path C was slightly slower than B in the 20-step receipt. |
| Engram | Yes | Yes | No | No | No | Hidden-state causal local averaging, not token-ID hash-table memory; not in default NAM route. |
| Concept | Yes | Yes | No | No | No | Learned global prototype cross-attention; no production route or receipt. |
| Stable loop/fixed point | Yes, tiny model/helpers | Yes | No | No | No | No sidecar-rich production runner; truncated-BPTT helper updates once per window, not once on a summed multi-window loss. |
| World model/action trajectories | Yes | Yes, synthetic and contract layers | No | No | No | Typed action conditioning exists, but verified real reward is metadata-only and the loss trains independent one-step transitions. |
| Megatron sidecar ingress | Yes | Yes | No immutable bundle-open proof | Upstream data was used historically, exact current MLX contract not proved | No | Local shape/schema checks are strong; artifact identity and contract hashes are not enforced. |
| Tokenizer/domain delimiters | Yes | Yes | CASE5 upstream receipts exist, current MLX bundle identity not proved | Indirectly used | Used by local eval | START/END pairs are validated; `FILE_SEP` is contradictory between contract role 49 and real token 14. |
| FIM/AST-FIM/IFIM builders | Yes | Yes on the committed baseline; active remap patch not yet accepted | No sidecar-rich materialized objective E2E | No objective-specific proof | Prompt modes ran | Token transforms work. Active worktree code remaps token sidecars, but still drops all transformed chunk/graph routes; clang sidecars are explicitly disabled for FIM/IFIM generation. |
| Generic eager/KV inference | Yes | Partially | No | No | Dense eager path yes | Full-prefix eager works. KV tests were not runnable in this environment; generated semantic kwargs are incomplete. |
| Repo codegen/compile gate | Yes, DenseCppLM-specific | Unit/fixture coverage | Gate executes, but no successful model receipt | 0/4 compile/run | 20 local reports, all 0/4 passed | Execution-proven and quality-failing. Paged serving remains explicitly unimplemented. |
| Megatron to MLX conversion | GQA only | 17 focused tests passed | Serialization/reload yes; true source-logit parity no | Source checkpoint is H200-trained | Converted checkpoints load and generate | Current v3 converter is structurally fail-closed, but existing local checkpoints are older schema v1 and parity is MLX-to-MLX after mapping. |

## Current Stage-1 objective and data contracts

### Declared objective mix

The executable default in
[task_mixer.py](../../cppmega_mlx/training/task_mixer.py#L45-L70)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/task_mixer.py#L45-L70))
is:

| Objective | Rate |
| --- | ---: |
| Causal LM | 0.50 |
| Plain FIM | 0.05 |
| AST-FIM | 0.05 |
| Typed IFIM | 0.10 |
| Commit diff | 0.10 |
| Pre-to-post | 0.10 |
| Symbol recovery | 0.033333... |
| Type recovery | 0.033333... |
| Callee recovery | 0.033333... |

AST-FIM dispatch forces `ast_fim_rate=1.0`, so it is not the
character-fallback mixture used by the lower-level default. The committed
`3bd0e6d` builders do not return an exact token-source map. The active
uncommitted [fim.py](../../cppmega_mlx/data/fim.py) adds
`FIMPermutationWithProvenance` and
`fim_permutation_source_indices()`; there is no pinned GitHub link for
that dirty code. The active materializer consumes the map for token-aligned
fields but not for chunk/graph sidecars or typed commit sections.

### Competing stage declarations

[stage1_cpp_foundation_dense500m.yaml](../../configs/stage1_cpp_foundation_dense500m.yaml#L1-L144)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/configs/stage1_cpp_foundation_dense500m.yaml#L1-L144))
declares the first run as GQA with graph/domain sidecars required and DSA off.
It also says the existing Parquet must not be rematerialized and embeds
machine-specific `/Users/dave/sources/parquet/...` paths. That is not
a portable, immutable data contract. Its "~500M" name is a model-size profile,
not a token budget; the H200 log measured 625,218,594 parameters after n-gram
and structure modules were included.

[stage_domain_routed_foundation.yaml](../../configs/stage_domain_routed_foundation.yaml#L1-L82)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/configs/stage_domain_routed_foundation.yaml#L1-L82))
declares three richer stages and objectives such as opener classification,
graph edge prediction/BCE/coverage, repair, world transitions, reward,
next-action, and patch generation. No runner consumes this complete schema.
Its `allow_token_only_baseline: false` requirement conflicts with the
current transformed-objective materialization that deliberately zeros semantic
sidecars.

### Actual runners

The older
[scripts/train_eval_stage1.py](../../scripts/train_eval_stage1.py#L97-L799)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/scripts/train_eval_stage1.py#L97-L799))
materializes only a subset of structure/call/type signals, uses zeros for graph
and edge-kind validation/probe inputs, and its probe syntax-checks a generated
file rather than compiling/running a complete repository case. Defaults of
10,020 steps, batch 4, and length 4,096 amount to 164,167,680 nominal sequence
tokens, not 500M or 10B.

The newer canonical
[run_stage1_graph_domain_production()](../../cppmega_mlx/training/stage1_production.py#L233-L343)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/stage1_production.py#L233-L343))
opens rich indexed data and checks live route counters. However:

- it constructs `CompiledPretrainingStep` without a loss override, so
  [compiled.py](../../cppmega_mlx/training/compiled.py#L1732-L1776)
  uses plain next-token cross-entropy;
- it has no scheduler, checkpoint/restart, held-out generation, or compile/run
  gate;
- it reads model vocabulary from permissive dataset metadata;
- it records `observed_edge_kind_prior_nonzero`, but lines 317-324 do
  not include that counter in the fail-closed condition.

The separate
[production_training_loss()](../../cppmega_mlx/training/objective_mixer.py#L724-L811)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/objective_mixer.py#L724-L811))
does add a graph auxiliary loss, but it sets
`edge_kind_bias = zeros_like(block_bias)`, calls the complete model
once for LM loss, then calls `model.indexer_scores()`, which performs
another complete decoder pass. The canonical runner does not use it.

### Materialized objective contract

[materialize_megatron_document()](../../cppmega_mlx/training/megatron_objectives.py)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/megatron_objectives.py#L550-L720))
correctly proves the shifted-LM relation and preserves full sidecars for the
only aligned task, causal LM. The committed baseline at `3bd0e6d`
(whose objective code is unchanged from `8a1b594`) still uses the
earlier behavior: transformed objectives zero every token
sidecar, retain only constant physical/document provenance, clear symbols and
change masks, and remove chunk/graph metadata.

The active uncommitted worktree is materially better and must be distinguished
from that baseline. FIM/AST-FIM/IFIM/recovery builders now return
`source_token_indices` with `-1` for inserted control
tokens. The materializer validates exact one-to-one source coverage and remaps
token-aligned structure, domain, role, confidence, source, symbol, and change
vectors. Domain delimiter stacks are reconstructed and validated.

The remaining gap is still production-blocking:

- the final `if not aligned` block in the active local materializer
  clears all transformed chunk spans and every graph relation, so graph/indexer
  training is absent on transformed objectives;
- commit-diff/pre-to-post objectives still zero most token semantics and have
  no exact source map across typed commit sections;
- inserted-token provenance is copied from the nearest source for physical
  identity, while semantic inserted-token policies need explicit acceptance;
- the patch was active and uncommitted during review, with no accepted
  data-to-indexed-dataset receipt.

Complete the exact map through chunks, graph endpoints, and commit sections;
use typed sentinels for genuinely synthetic markers; fail closed for an
unmappable objective; then materialize and re-open a real immutable bundle.

## Model-path review

### Dense GQA and dense DSA

[DenseCppLMConfig](../../cppmega_mlx/models/dense_cpp_lm.py#L80-L174)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/dense_cpp_lm.py#L80-L174))
has a coherent GQA profile and explicitly rejects `attention_mode="mla"`
because latent attention is not implemented. That is an honest contract.
`GraphIndexedAttention` nevertheless calls its gathered operation
"dense MLA" while retaining full-rank Q and KV projections
([local](../../cppmega_mlx/models/dense_cpp_lm.py#L272-L377),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/dense_cpp_lm.py#L272-L377)).
The low-rank component is only the separate indexer.

The dense model also adds a learned absolute position embedding before
RoPE-enabled attention
([local](../../cppmega_mlx/models/dense_cpp_lm.py#L470-L570),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/dense_cpp_lm.py#L470-L570)).
The converter zeros this tensor because the Megatron source used RoPE only.
Therefore converted H200 checkpoints and native MLX training do not share the
same positional architecture unless native training also freezes/zeros the
absolute table or removes it.

The graph resolver itself is one of the stronger contracts:
[_resolve_graph_bias()](../../cppmega_mlx/models/dense_cpp_lm.py#L713-L781)
rejects graph payloads when routes are off, requires relation and edge-kind
biases in production, combines typed graph inputs, and removes cross-document
bias. The failures are at callers that manufacture a zero edge-kind matrix.

### DSA, sparse MLA, and the 128k claim

There are three distinct implementations that must not be conflated:

1. Dense model DSA uses a learned indexer plus full-rank Q/KV.
   [lightning_indexer_scores()](../../cppmega_mlx/nn/sparse_mla.py#L470-L574)
   emits `(B,S,S)` fp32 scores through a full einsum, and
   [indexer_topk_indices()](../../cppmega_mlx/nn/sparse_mla.py#L577-L651)
   performs `mx.argsort` over the full key axis.
2. Hybrid attention treats `mla` as a dense mode and gives
   `dsa` a dense SDPA fallback
   ([local](../../cppmega_mlx/nn/attention.py#L17-L40),
   [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/attention.py#L17-L40)).
3. The Path C sparse route uses full-rank projections and deterministic causal
   window indices, not the learned graph indexer
   ([local](../../cppmega_mlx/nn/attention.py#L783-L875),
   [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/attention.py#L783-L875)).

The FP8 producer stop-gradients Q/KV payload and scale tensors. A dedicated
fused backward may own projection gradients, but generic autograd cannot be
assumed to do so. A full model gradient receipt must show nonzero, finite
gradients for Q, sparse KV, output projection, and the learned indexer.

The checked-in sparse benchmark covers only 128, 512, and 1,024 token shapes:
[sparse_mla.json](../../bench/tilelang_ports/sparse_mla.json)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/bench/tilelang_ports/sparse_mla.json)).
It proves a local kernel path at those shapes, not a model, learned indexer, or
128k context.

The production fix is one integrated attention design: latent Q/KV
projections, a blockwise learned indexer that never allocates `S x S`,
typed graph relation/kind priors, gathered sparse attention, and one backward
contract. Until that exists, "128k DSA" is an engineering milestone, not a
configuration change.

### Graph routes, categorical kinds, and auxiliary loss

[PromptGraphModelInputs.dense_attention_bias()](../../cppmega_mlx/data/prompt_graph.py#L1418-L1505)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/data/prompt_graph.py#L1418-L1505))
collapses all graph routes into one scalar relation matrix and discards
`_kind` on triple edges. The specialized generator then passes
`zeros_like(block_bias)` as edge-kind bias
([local](../../scripts/cpp_jsonl_generation_compile_eval.py#L1200-L1242),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/scripts/cpp_jsonl_generation_compile_eval.py#L1200-L1242)).
Training does the same in `production_training_loss()`.

Add separate `relation_bias` and `edge_kind_bias` outputs to
the prompt graph contract. Preserve the categorical kind on each triple route,
map pair-route types explicitly, and require a nonzero kind-prior receipt for
every production graph run. Add the missing
`observed_edge_kind_prior_nonzero <= 0` runner gate.

The graph auxiliary objective should consume indexer scores returned by the
same decoder forward used for LM loss. One call should return logits, LM loss,
and per-layer indexer scores; graph loss should be reduced from those scores.

### Hybrid NAM routes

[hybrid_lm.py](../../cppmega_mlx/models/hybrid_lm.py#L1-L5)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/hybrid_lm.py#L1-L5))
explicitly describes itself as a correctness-first tiny assembly, not full
NAM56R. The declarative
[Nam56RModelConfig](../../cppmega_mlx/config/model.py#L294-L342)
is translated back into `HybridTinyConfig` by
[build_hybrid_tiny_config_from_nam56r()](../../cppmega_mlx/recipes/nam56r.py#L130-L197).
That preserves route intent, not production completeness.

- **Mamba3:** reference and Path C code exist, with strong kernel-level tests.
  Explicit Path B/C reject packed-document reset masks
  ([local](../../cppmega_mlx/nn/mamba3.py#L598-L648),
  [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/mamba3.py#L598-L648)).
- **M2RNN:** has the same production conflict
  ([local](../../cppmega_mlx/nn/m2rnn.py#L310-L360),
  [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/m2rnn.py#L310-L360)).
- **MoE:** the default reference runs every expert over every token, while the
  opt-in sparse route materializes routing indices through NumPy on the host
  ([local](../../cppmega_mlx/nn/moe.py#L267-L337),
  [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/moe.py#L267-L337)).
  The router computes a load-balance auxiliary loss, but
  `HybridTinyBlock` keeps only `.output`
  ([local](../../cppmega_mlx/models/hybrid_lm.py#L760-L810),
  [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/hybrid_lm.py#L760-L810)).
- **Engram:** the branch is hidden-state causal local averaging, not
  token-ID/hash-table memory, and its own docstring says it is not wired by
  default
  ([local](../../cppmega_mlx/nn/engram.py#L161-L207),
  [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/engram.py#L161-L207)).
- **Concept:** implemented as cross-attention to learned global prototypes
  ([local](../../cppmega_mlx/nn/concept.py#L60-L117),
  [GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/concept.py#L60-L117)).

HybridTinyLM also combines learned absolute positions with RoPE attention and
rejects `document_ids` during KV-cache decode
([local](../../cppmega_mlx/models/hybrid_lm.py#L2178-L2205),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/hybrid_lm.py#L2178-L2205)).

Before any hybrid scale run, require packed-document parity on every explicit
route, device-only MoE dispatch, router auxiliary loss in the differentiated
objective, and a complete checkpoint/restart/generation receipt.

### Stable loop and fixed point

[StableLoopCppLM](../../cppmega_mlx/models/stable_loop_cpp_lm.py#L1-L240)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/stable_loop_cpp_lm.py#L1-L240))
is deliberately tiny and has no production side channels, packed-document
contract, or Stage runner. The fixed-loop training and FPOPT inference behavior
are unit-tested.

[deep_supervision_loss()](../../cppmega_mlx/training/fixed_point.py#L32-L70)
documents that only the latest detached window contributes gradients unless
the caller re-enters each window.
[truncated_bptt_step()](../../cppmega_mlx/training/fixed_point.py#L73-L169)
does re-enter, but applies an optimizer update after every window and advances
state again using newly updated weights. This is a specific online schedule,
not one optimizer step over the sum of all window losses. The intended
mathematics must be decided, named in the receipt, and tested against a small
full-unroll reference before scale.

### World model and action trajectories

[CodeLoopWorldModel](../../cppmega_mlx/models/code_loop_world_model.py#L101-L354)
([GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/code_loop_world_model.py#L101-L354))
mean-pools observation and action tokens, applies an action-conditioned latent
transition, and broadcasts one latent over target positions for decode. The
multi-step rollout API carries latent state correctly.

The training loss does not use that rollout. It loops over transitions and
calls one-step prediction independently
([local](../../cppmega_mlx/training/world_model.py#L132-L208),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/world_model.py#L132-L208)).
`TrajectoryPacket` verifies element types but does not require
`next_obs[t] == obs[t+1]`
([local](../../cppmega_mlx/data/trajectory_packet.py#L179-L210),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/data/trajectory_packet.py#L179-L210)).

The real agent adapter stores `verified_reward` only in metadata and
leaves `Transition.reward` unset
([local](../../cppmega_mlx/data/agent_trajectory.py#L334-L405),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/data/agent_trajectory.py#L334-L405)).
The Transition contract permits reward/done labels only when
`is_synthetic=True`, and the loss trains control heads only on those
synthetic labels. Verified build/test outcomes therefore never train the
reward or done heads.

Replace the binary synthetic flag with explicit label provenance such as
`verified`, `synthetic`, and `none`. Permit
verified labels, forbid unproven real labels, validate trajectory continuity,
and train carried multi-step latent rollouts with a declared teacher-forcing
schedule.

## Data ingress, tokenizer, and objective integrity

### MLX ingress is schema-aware but artifact-identity-blind

The target Megatron reader validates MMIDIDX layout, sidecar dtypes/shapes,
coordinates, and cross-shard schema. It does not bind the files to the bundle
that produced them:

- [_load_sidecar()](../../cppmega_mlx/data/megatron_indexed.py#L1160-L1178)
  returns an empty object if metadata is absent;
- [_token_metadata()](../../cppmega_mlx/data/megatron_indexed.py#L1892-L1905)
  defaults to the 131,072-token MegaCPP contract;
- [_validate_multishard_schema()](../../cppmega_mlx/data/megatron_indexed.py#L1045-L1118)
  compares local schema and tokenizer labels but not tokenizer SHA, domain
  schema SHA, CASE5 receipt, objective artifact/contract hashes, source
  snapshot, or bundle artifact-set digest.

The sibling producer has stronger contracts:

- [_require_case5_schema()](/Volumes/external/sources/cppmega_full_integration/scripts/data_prep_parquet_to_megatron.py#L568-L667)
  ([GitHub](https://github.com/DatasunriseOU/cppmega/blob/ef44b42b0bf669e0ed5be17910bed8fb8ed98615/scripts/data_prep_parquet_to_megatron.py#L568-L667))
  checks exact Arrow types and frozen contract hashes.
- [_add_case5_manifest()](/Volumes/external/sources/cppmega_full_integration/scripts/data_prep_parquet_to_megatron.py#L1846-L1869)
  records the CASE5 receipt.
- [_validate_bundle()](/Volumes/external/sources/cppmega_full_integration/scripts/data/publish_megatron_bundle_to_nebius_s3.py#L1211-L1500)
  ([GitHub](https://github.com/DatasunriseOU/cppmega/blob/ef44b42b0bf669e0ed5be17910bed8fb8ed98615/scripts/data/publish_megatron_bundle_to_nebius_s3.py#L1211-L1500))
  validates tokenizer/data contracts, objective descriptors, every artifact
  size/hash, and the artifact-set digest.

Production MLX training should accept an immutable bundle root or manifest,
resolve the selected sequence bucket from that manifest, hash-check all
referenced artifacts before mmap, and reject bare prefixes. A one-byte
mutation, missing descriptor, stale contract, mixed bucket, or unlisted file
must fail before model construction.

### Tokenizer and domain delimiters

The frozen tokenizer is 65,536 tokens and the FIM IDs are coherent:
prefix/middle/suffix 4/5/6 and IFIM instruction 45. Domain START/END reserved
roles are derived and validated against `<RESERVED_N>`.

There is one direct contradiction:

- [tokenizer_contract_v1.json](../../cppmega_mlx/tokenizer/tokenizer_contract_v1.json#L26-L31)
  assigns logical `FILE_SEP` to reserved id 49 and says each role
  resolves to `<RESERVED_N>`;
- [tokenizer_contract.py](../../cppmega_mlx/data/tokenizer_contract.py#L33-L44)
  and tests define `FILE_SEP` as real token id 14;
- [tokenizer.json](../../cppmega_mlx/tokenizer/tokenizer.json) contains
  `<FILE_SEP>` at 14 and `<RESERVED_49>` at 49.

The validator only derives START/END pairs, so it misses the contradiction.
Remove the duplicate reserved role or rename it to a distinct semantic role,
then validate every special/alias/reserved assignment against the frozen
artifact and hash the resulting complete role map.

## Inference, generation, compile gates, and serving

The generic eager generator
[generate_tokens()](../../cppmega_mlx/inference/generation.py#L87-L149)
is a correct bootstrap but recomputes the entire prefix. The extension contract
only knows structure, document, and platform kwargs
([local](../../cppmega_mlx/inference/generation.py#L25-L37),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/inference/generation.py#L25-L37)).
Domain, role, confidence, source identity, and graph signals are not extended
for generated tokens. Unknown sequence-aligned kwargs pass through unchanged.
Standard output normalization accepts only plain logits or the DenseCppLM
`(logits, None)` tuple.

The specialized C++ evaluator supports causal, FIM, and IFIM prompts and has a
proper fail-closed compile subprocess
([local](../../scripts/cpp_jsonl_generation_compile_eval.py#L1371-L1395),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/scripts/cpp_jsonl_generation_compile_eval.py#L1371-L1395)).
It rejects clang sidecars for reordered FIM/IFIM prompts because alignment is
unknown, while repo graph mode owns a separate exact prompt graph. This is
honest, but it means sidecar-rich FIM generation is not proven. The evaluator
builds DenseCppLM only and reconstructs parts of config from CLI/checkpoint
metadata rather than consuming the full converter runtime requirements.

The paged scheduler is intentionally not model-integrated:
[require_model_integrated_paged_attention()](../../cppmega_mlx/inference/serving.py#L522-L525)
always raises. Paged serving is implemented only at scheduling/data-structure
level.

Acceptance requires one checkpoint-driven model factory shared by conversion,
eager generation, cached generation, and compile evaluation. It must validate
the runtime requirements manifest, extend every generated semantic sidecar by
explicit policy, preserve graph relation and kind signals, and support each
declared model output contract.

## Checkpoint conversion to MLX

[convert_megatron_dense500m_torchdist_to_mlx.py](../../scripts/convert_megatron_dense500m_torchdist_to_mlx.py#L103-L109)
supports GQA only and correctly rejects DSA. Mapping is metadata-driven,
unmapped learned tensors fail closed, grouped QKV is explicitly rearranged,
learned absolute positions/platform tables are neutralized, and emitted
safetensors are strictly reloaded.

The parity name overstates what is compared. At conversion time,
[_mapped_source_reference_model()](../../scripts/convert_megatron_dense500m_torchdist_to_mlx.py#L902-L917)
builds a DenseCppLM and loads the already mapped target arrays. Lines 1016-1047
then compare that MLX instance to another MLX instance reloaded from emitted
safetensors
([local](../../scripts/convert_megatron_dense500m_torchdist_to_mlx.py#L967-L1055),
[GitHub](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/scripts/convert_megatron_dense500m_torchdist_to_mlx.py#L967-L1055)).
This proves serialization and target reconstruction, not source-framework
mathematics.

The inspected local converted checkpoints under
`/Volumes/external/sources/cppmega/outputs/checkpoints/mlx_converted/`
use manifest schema v1, contain no current `runtime_requirements` or
`logit_parity`, and predate the v3 converter gate.

Add a source-side Megatron inference exporter that loads the exact DCP
checkpoint and emits logits for a fixed sidecar-rich test vector. Compare:

1. original Megatron logits;
2. in-memory mapped MLX logits;
3. strictly reloaded safetensors logits.

Record dtype, exact source checkpoint digest, tokenizer/domain/graph inputs,
max/mean absolute and relative error, and target model manifest. For fp32 use
predeclared `max_abs <= 1e-4` and `mean_abs <= 1e-5`;
for bf16 derive and freeze a looser tolerance from the fp32 reference before
publishing.

## Historical receipt truth

### H200 dense 5,000-step run

Receipt:
[seq_1024_bs_192.log](/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-megatron-1782697038/seq_1024_bs_192.log)
and
[summary.log](/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-megatron-1782697038/summary.log).

| Fact | Receipt value |
| --- | --- |
| Architecture | Dense GQA, 20 query heads, 4 KV groups, no graph attention bias |
| Parameters | 625,218,594 total; 128,309,472 n-gram; 83,522 structure |
| Train | 5,000/5,000 iterations, global batch 192, sequence 1,024 |
| Nominal sequence tokens | 983,040,000 |
| Stability | skipped iterations 0; NaN iterations 0 |
| Validation/test | PPL 4.280051 / 4.166446 |
| Peak allocated | 104.090 GiB |
| Final status | OK |

This is H200 proof for the dense GQA Megatron path only. It is not proof for
MLX training, DSA, graph bias, hybrid routes, or any world-model objective.

Generation from the same run:
[generation_summary.json](/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-generation-1782753301/generation_summary.json)
and
[compile_report.json](/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-generation-1782753301/compile_report.json).
Four source-prefix cases generated 128 tokens each; outputs were visibly
degenerate/reserved-token heavy. Compile summary was 0 compiled, 0 passed,
0 ran out of 4.

### H200 graph-route curriculum

Receipt root:
[cppmega-h200-graphroutes-1782831200](/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200)
with [summary.log](/Volumes/external/sources/cppmega/outputs/nebius/cppmega-h200-graphroutes-1782831200/summary.log).

| Stage | Sequence | Successful GBS/MBS | Iterations | Nominal tokens | Validation PPL | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1,024 | 192 | 1,421 | 279,379,968 | 2.459471 | OK |
| 2 | 2,048 | 96 | 1,686 | 331,481,088 | 2.033314 | OK |
| 3 | 4,096 | 40 | 2,311 | 378,634,240 | 1.984795 | OK after GBS 48 failed |
| 4 | 8,192 | 16/4 | 2,756 | 361,234,432 | 1.636913 | OK after other attempts failed |
| 5 | 16,384 | 8/2 | 2,391 | 313,393,152 | 1.667605 | OK |
| Total successful stages | | | | 1,664,122,880 | | |

This route used dense GQA plus `post_scale_bias`, not learned DSA.
Every later stage used finetune with no optimizer/RNG load and scheduler
override. Weights continue, but optimizer and RNG do not, so this is a staged
curriculum rather than one continuous 1.664B-token optimization run. PPL values
also span changing sequence lengths, batches, and datasets and are not directly
comparable.

### Local macOS model evaluation

Twenty local `local_mlx_*` compile reports were inspected under
[outputs/evals](/Volumes/external/sources/cppmega/outputs/evals). Every report
has 0/4 passed and 0/4 ran. Two reports compiled one of four cases, but neither
passed. The current graph-route Stage-5 source-prefix and docstring runs each
compiled 0/4.

This is local macOS-eval proof that converted dense checkpoints loaded,
generated, and reached the compile gate. It is also direct evidence that the
current checkpoint/generation stack does not meet a useful code-quality gate.

### Kernel and limited E2E receipts

- [mamba3.json](../../bench/tilelang_ports/mamba3.json) is a local Mac synthetic
  kernel benchmark, not a model run.
- [mamba3_path_c.json](../../bench/tilelang_ports/mamba3_path_c.json) records
  local forward/backward parity through sequence 4,096.
- [m2rnn_path_c_vs_path_b_20step.json](../../bench/tilelang_ports/m2rnn_path_c_vs_path_b_20step.json)
  records a 20-step tiny synthetic run: Path C 126,674.93 tok/s versus Path B
  129,382.75 tok/s, ratio 0.979.
- [path_c_fusion_production_smoke_receipt.json](../../reports/path_c_fusion_production_smoke_receipt.json)
  says native compilation succeeded but `actually_executed=false`,
  with zero launches and a compile-only runtime contract.
- [path_c_single_fused_kernel_e2e_exec_receipt.json](../../reports/path_c_single_fused_kernel_e2e_exec_receipt.json)
  executes two tiny fused launches with finite loss. Only 2 of 8 gradient leaves
  are nonzero. It is a kernel E2E smoke, not full model/training E2E proof.

### Historical sidecar object receipt

A prior verified receipt recorded Nebius object set
`cppmega-sidecar-20260627/cppmega-sidecar/verified-20260627`: 3,182
objects, 2,638,374,680 bytes, and 2,076,705,386 valid tokens, with code buckets
1,024/2,048/4,096 and commit/PR buckets through 8,192. The manifest was not
found in the target, sibling, or original checkout during this review, so these
numbers are historical provenance, not a currently re-opened or re-hashed
artifact. They must not authorize a new run without downloading and validating
the immutable manifest.

## Exact findings and proposed fixes

| ID | Severity | Finding | Concrete fix and acceptance |
| --- | --- | --- | --- |
| MTR-001 | P0 | Dense DSA allocates `S x S` index scores and full argsort; hybrid Path C uses a different deterministic indexer. There is no latent-MLA learned-index 128k path. | Implement one blockwise learned latent DSA path. Reject any production graph containing an `(S,S)` tensor. Prove forward/backward parity at <=2k, nonzero projection/indexer grads, and near-linear memory before long-context tests. |
| MTR-002 | P0 | Committed code strips all transformed sidecars. Active uncommitted code now remaps many token fields, but still strips every transformed chunk/graph route and leaves commit semantics mostly zero. | Finish exact source provenance for chunks, graph endpoints, and commit sections; use typed sentinels for synthetic markers; fail closed otherwise. Per-objective tests and a re-opened real bundle must show valid endpoints and nonzero eligible semantic/graph coverage. |
| MTR-003 | P0 | MLX opens bare Megatron prefixes without immutable bundle identity/hash validation. | Add a production bundle opener that verifies manifest schema, all artifact sizes/SHA-256 values, tokenizer/domain/CASE5/objective/source-snapshot bindings, then opens only manifest-listed prefixes. Tamper and mixed-bundle tests must fail before mmap/model build. |
| MTR-004 | P0 | No canonical differentiated Stage-1 objective combines LM and graph aux from one forward. Current aux path zeroes edge kinds and runs the decoder twice. | Return indexer scores from the LM forward; compute one combined loss; pass real relation/kind priors; wire it into `CompiledPretrainingStep`. Assert one decoder invocation and nonzero graph/indexer gradients. |
| MTR-005 | P0 | Conversion parity is MLX-to-MLX after mapping, not original Megatron-to-MLX. Existing evaluated checkpoints use older schema v1. | Export fixed source logits from exact DCP, compare source/in-memory/reloaded MLX, include all required sidecars and immutable hashes, and reject publication without parity. |
| MTR-006 | P1 | H200 and local generation quality is zero compile/run success on the available four-case receipts. | Build a leakage-controlled >=100-case repository eval. Require nonzero compile/run, improvement over frozen baseline, and a lower confidence bound above baseline before scaling. Keep the four cases as smoke only. |
| MTR-007 | P1 | Graph triple edge kinds are discarded in prompt bias; training and generation inject zero kind bias; production runner omits its recorded nonzero-kind gate. | Split relation and kind matrices end to end, preserve triple kinds, type pair relations, and fail if both observed kind edges and kind prior are not nonzero. |
| MTR-008 | P1 | Native dense/hybrid MLX adds learned absolute positions while attention uses RoPE; converted source checkpoints zero the learned table. | Choose one positional contract. For RoPE checkpoints, remove/freeze-zero absolute embeddings and add native-vs-converted architecture tests. |
| MTR-009 | P1 | MoE load-balance aux is computed then discarded; default compute is all-expert dense; sparse compute host-syncs routing metadata. | Return route aux through model output and differentiated loss; implement device grouped dispatch; prove no host sync, expert capacity/drop accounting, and nonzero router/expert grads. |
| MTR-010 | P1 | Mamba3 and M2RNN explicit optimized paths reject packed-document resets even though production data is packed. | Add reset-aware kernels or route every packed batch explicitly to a proven backend. Require packed-vs-unpacked output/gradient parity and receipt route counters. |
| MTR-011 | P1 | World-model verified reward remains metadata-only; real labels are forbidden; independent one-step loss does not train the rollout contract or continuity. | Add explicit verified/synthetic/none label provenance, continuity validation, carried latent rollout loss, scheduled teacher forcing, and real held-out trajectory evaluation. |
| MTR-012 | P1 | Generic generation does not extend domain/role/confidence/source/graph semantics and only normalizes Dense tuple output. | Define generated-token policy for every sequence sidecar, support model-specific typed outputs through a shared adapter, and parity-test eager/cached generation. |
| MTR-013 | P1 | Tokenizer contract assigns `FILE_SEP` to both logical id 49 and actual special id 14 under different tables; validator misses non-delimiter reserved roles. | Make role naming unambiguous and validate every role/alias against the frozen tokenizer plus complete-map SHA. |
| MTR-014 | P1 | Stable-loop truncated-BPTT applies one optimizer update per window, while prose may be read as one summed deep-supervision update. | Specify schedule semantics, test against full unroll at tiny scale, and include update count/BPTT horizon in receipts before integrating production sidecars. |
| MTR-015 | P1 | Current focused model tests expose a fusion-region ABI/test regression: expected first node `mamba3_mimo`, runtime emits `entry_rmsnorm`. | Decide whether entry RMSNorm belongs to the public fusion-region ABI, update implementation or test atomically, and add a compiled tiny forward/backward receipt. |
| MTR-016 | P2 | Canonical runner lacks scheduler, exact token accounting, checkpoint/restart, held-out generation, and compile gates. | Consolidate these into one runner and receipt schema. Count trained tokens from the loss mask, not `steps*batch*sequence`. |
| MTR-017 | P2 | Paged KV scheduling exists without model-integrated attention. | Keep fail-closed until block-table attention, preemption/recompute, and contiguous-cache parity pass. Do not advertise it as serving-ready. |

## Staged training and evaluation plan

Token budgets below mean **loss-mask training tokens**, not model parameters,
padding, or nominal sequence capacity.

### Gate 0: contract closure, no scale run

Required before any new training:

- immutable bundle opener passes positive, missing-file, one-byte-tamper,
  cross-bucket, cross-bundle, stale-contract, and unlisted-file tests;
- every objective preserves/remaps all semantic sidecars and graph endpoints;
- one canonical loss performs one model forward and produces finite nonzero LM,
  graph, edge-kind, and DSA-indexer gradients when enabled;
- source Megatron to MLX conversion parity passes;
- positional architecture is identical across native training and conversion;
- production dependency environment runs the required suites with zero
  collection skips for pyarrow/tokenizers/clang/MLX runtime;
- MTR-015 fusion ABI is reconciled.

Acceptance artifact: one signed/hash-bound "training candidate" manifest naming
code commit, data bundle ID, tokenizer/domain/objective hashes, model config,
loss config, runtime/kernel policy, and eval set IDs.

### Gate 1: tiny deterministic smoke

Run 1, 10, and 100 optimizer steps on a tiny immutable bundle at sequence 128
and 512, first GQA, then each opt-in path independently.

Acceptance:

- finite loss and all intended parameter groups receive finite gradients;
- overfit a 1-4 sample batch to near-zero masked-token loss;
- exact fp32 restart reproduces next batch, loss, optimizer state, and weights;
- bf16 restart stays within predeclared tolerance;
- trained-token count equals sum of the loss mask;
- every configured objective and sidecar route has nonzero receipt counters;
- checkpoint reload gives identical greedy tokens;
- all golden smoke completions compile and run;
- no fallback or route substitution is hidden in receipt counters.

### Gate 2: 1M and 10M token integration pilots

Use repository/time-disjoint train/validation/test splits. Run 1M tokens to
debug schedules and 10M tokens to compare:

- token-only GQA control;
- sidecar-rich GQA;
- graph relation plus edge-kind bias;
- graph auxiliary objective;
- each transformed objective family.

Acceptance:

- no NaN/skipped updates and no unexpected fallback;
- deterministic resume at multiple checkpoints;
- validation loss and compile/run metrics improve over the frozen token-only
  control with uncertainty reported;
- no objective has zero realized rate or zero semantic coverage;
- generated code passes delimiter, FIM layout, source-identity, and graph-window
  invariants;
- checkpoint converts with source-logit parity and passes local macOS eager plus
  cached generation.

### Gate 3: 500M training-token dense GQA pilot

This is a token-budget milestone, distinct from the "~500M" model label. Keep
the current dense GQA baseline and sidecar-rich candidate as paired runs or use
a predeclared ablation schedule.

Acceptance:

- exactly 500,000,000 masked training tokens, reported by domain/objective;
- no optimizer/RNG reset hidden inside a "continuous" run;
- checkpoint/restart receipts at 1M, 10M, 100M, 250M, and 500M;
- held-out PPL/loss by domain and objective, graph/indexer metrics, FIM exact
  middle recovery, and repository compile/run eval at each milestone;
- candidate is no worse than baseline on core LM metrics and materially better
  on at least one predeclared code/graph objective;
- source-to-MLX conversion and local Mac compile evaluation pass at the final
  checkpoint.

### Gate 4: continuous 10B-token dense foundation run

Start only after Gate 3 acceptance. Use one immutable dataset manifest and a
versioned sequence curriculum. If optimizer state is intentionally reset,
declare separate mid-training stages; do not report them as one continuous
10B optimization run.

Acceptance:

- exactly 10,000,000,000 masked tokens with per-bucket reuse counts;
- no train/validation/test repository or future-time leakage;
- no NaN/skipped steps; fail closed on data/kernel/model fallback;
- complete optimizer/RNG/dataloader restart continuity;
- fixed eval suite at 0, 0.5B, 1B, 2B, 5B, and 10B;
- compile-and-run lower confidence bound exceeds the frozen 500M-token
  checkpoint and token-only baseline;
- final H200 checkpoint converts through current v3+ source parity and passes
  local macOS eval.

### Gate 5: hybrid and DSA ablations

Do not combine Mamba, MoE, M2RNN, engram, concept, stable loop, and DSA in the
first ablation. Add one path at a time against the accepted dense checkpoint.

Acceptance per path:

- same data/order/token budget and predeclared parameter/FLOP adjustment;
- packed-document forward/gradient parity;
- full-model train/restart/generation receipt, not only kernel timing;
- no host synchronization in compiled hot path;
- throughput/memory and quality both reported;
- MoE includes load balance/capacity metrics; recurrent paths include state
  reset and continuation tests; stable loop includes halting/BPTT semantics.

### Gate 6: true DSA long-context progression to 128k

First replace the quadratic reference with the integrated latent learned-index
path. Progress 4k -> 16k -> 32k -> 64k -> 128k; do not jump from the 1,024-token
kernel receipt.

Acceptance at each doubling:

- instrumentation proves no tensor with both query and key dimensions equal to
  full sequence length;
- isolated indexer/attention peak memory grows by less than 2.5x when sequence
  doubles at fixed batch, heads, latent rank, and top-k;
- sparse output/gradient parity against dense reference at <=2k;
- top-k recall >=0.90 against exhaustive learned-index scores on held-out short
  sequences, with graph and no-graph slices reported;
- finite nonzero gradients for latent Q/KV, output, indexer projections/weights,
  graph beta, and every enabled side embedding;
- packed-document boundaries and graph routes never cross documents;
- 128k H200 smoke completes forward/backward/update/checkpoint/reload and then
  local Mac evaluation at a supported shorter window;
- long-context retrieval, repository dependency navigation, FIM, and compile/run
  metrics beat the accepted 16k dense baseline, not merely fit in memory.

## Verification performed for this review

No packages were installed and no expensive training was run. The project
`.venv` did not contain pytest/numpy/pyarrow; default system Python
3.14 had pytest and MLX but lacked several optional production dependencies. A
separate Homebrew Python 3.13 had pytest/pyarrow/numpy/MLX and ran the focused
materialization tests, but lacked clang/tokenizers and its spawned CLI process
did not resolve MLX.

| Command group | Result | Interpretation |
| --- | --- | --- |
| Tokenizer/domain/FIM/AST-FIM/objective unit tests | 73 passed in 0.54s on the current active worktree | Core builders and domain-preserving provenance still pass after the active remap. |
| Production objective mixer on Python 3.13 with pyarrow/numpy/MLX | 45 passed in 0.48s on the current active worktree | Eligibility, quotas, and production objective unit contracts pass. |
| Dependency materialization/provenance plus production mixer | 51 passed, 3 failed in 1.18s on the current active worktree | Two failures are environment gaps in child processes (missing MLX and clang); one live mismatch no longer raises for a manually forced multi-physical transformed document. The active remediation is not yet accepted. |
| Dense/attention/sparse/hybrid/MoE/Mamba/M2RNN/engram/stable-loop/fixed-point group | 214 passed, 5 skipped, 15 failed in 21.31s | 5 failures came from partial `tvm_ffi`, 9 from missing `mlx_lm`, and one is the real fusion-region ABI/test mismatch MTR-015. |
| Converter tests alone | 17 passed in 0.92s | Mapping/manifest/reload unit contracts pass; they do not add original Megatron forward parity. |
| World-model subset excluding real-Parquet tests | 31 passed, 6 deselected in 1.33s | Synthetic/model contracts pass. Six real-Parquet tests require missing `pyarrow`. |
| Broad world/inference/compile/conversion group | 111 passed, 1 skipped, 39 failed in 65.95s | Most failures were missing `pyarrow`, `mlx_lm`, `tokenizers`, or Python `clang`; one golden compile fixture hit its 60s timeout. |
| Initial broad data collection | Collection failed on missing `pyarrow` | Production Parquet coverage was not runnable in the current environment. |

The model group regression is:
`tests/test_hybrid_lm.py::test_hybrid_tiny_lm_exposes_path_c_fusion_regions_from_route_symbols`.
The test expects the first fusion node to be `mamba3_mimo`, while
current [path_c_fusion.py](../../cppmega_mlx/runtime/path_c_fusion.py#L1872-L1875)
adds `entry_rmsnorm` first. This is not attributable to an optional
dependency and must be reconciled.

## Provenance index

| Area | Local current source | Pinned GitHub |
| --- | --- | --- |
| Dense model/GQA/DSA/graph resolver | [dense_cpp_lm.py](../../cppmega_mlx/models/dense_cpp_lm.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/dense_cpp_lm.py) |
| Sparse indexer/reference | [sparse_mla.py](../../cppmega_mlx/nn/sparse_mla.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/sparse_mla.py) |
| Hybrid attention/Path C producer | [attention.py](../../cppmega_mlx/nn/attention.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/nn/attention.py) |
| Hybrid route assembly | [hybrid_lm.py](../../cppmega_mlx/models/hybrid_lm.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/hybrid_lm.py) |
| Objective materialization | [megatron_objectives.py](../../cppmega_mlx/training/megatron_objectives.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/megatron_objectives.py) |
| Canonical Stage-1 runner | [stage1_production.py](../../cppmega_mlx/training/stage1_production.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/stage1_production.py) |
| Megatron MLX ingress | [megatron_indexed.py](../../cppmega_mlx/data/megatron_indexed.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/data/megatron_indexed.py) |
| Prompt graph | [prompt_graph.py](../../cppmega_mlx/data/prompt_graph.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/data/prompt_graph.py) |
| World model/trajectory | [model](../../cppmega_mlx/models/code_loop_world_model.py), [loss](../../cppmega_mlx/training/world_model.py), [adapter](../../cppmega_mlx/data/agent_trajectory.py) | [model](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/models/code_loop_world_model.py), [loss](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/training/world_model.py) |
| Generation/compile eval | [generation.py](../../cppmega_mlx/inference/generation.py), [C++ eval](../../scripts/cpp_jsonl_generation_compile_eval.py) | [generation](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/cppmega_mlx/inference/generation.py), [eval](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/scripts/cpp_jsonl_generation_compile_eval.py) |
| Converter | [converter](../../scripts/convert_megatron_dense500m_torchdist_to_mlx.py) | [source](https://github.com/DatasunriseOU/cppmega_mlx/blob/3bd0e6d4f1d5de5cd52b901078d33e5a32527390/scripts/convert_megatron_dense500m_torchdist_to_mlx.py) |
| Sibling Parquet to Megatron | [converter](/Volumes/external/sources/cppmega_full_integration/scripts/data_prep_parquet_to_megatron.py) | [source](https://github.com/DatasunriseOU/cppmega/blob/ef44b42b0bf669e0ed5be17910bed8fb8ed98615/scripts/data_prep_parquet_to_megatron.py) |
| Sibling bundle publication | [publisher](/Volumes/external/sources/cppmega_full_integration/scripts/data/publish_megatron_bundle_to_nebius_s3.py) | [source](https://github.com/DatasunriseOU/cppmega/blob/ef44b42b0bf669e0ed5be17910bed8fb8ed98615/scripts/data/publish_megatron_bundle_to_nebius_s3.py) |

## Remaining risks

- Historical H200 receipts are real but do not pin the exact current target
  source tree or current MLX converter.
- The historical sidecar object manifest was not locally available for
  re-hashing.
- Complete production Parquet-to-CLI, KV-cache, tokenizer-backed generation,
  clang-indexed codegen, and some Path C tests could not pass in the available
  Python runtimes. These are explicit coverage gaps, not passing evidence.
- Sibling source-binding hardening landed as `ef44b42`, but it still
  needs to be consumed by a fail-closed MLX bundle opener.
- Concurrent target work partially remediated transformed token-sidecar
  provenance but was still active. Its final commit and tests may change line
  numbers or close part of MTR-002; graph/chunk and commit mapping remained open
  in the last inspected worktree.
- No path currently satisfies the strict end-to-end definition across immutable
  data, training, restart, conversion, local generation, and successful
  compile/run.
