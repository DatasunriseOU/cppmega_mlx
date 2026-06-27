# Graph-Routed C/C++ World Model

Status: design contract for cppmega world-model training and inference.

This document defines the unified cppmega world-model shape. It is intentionally
not a general-purpose language-model contract. A cppmega world model consumes a
domain-special-token-wrapped stream and a graph route packet on every training
and inference step. Text-only dense models can exist as diagnostics or baselines,
but they are not cppmega world models unless graph routes and domain wrappers are
present in the sample path.

## Non-Negotiable Contract

Every cppmega model sample has two mandatory surfaces:

1. A domain-special-token-wrapped stream: C/C++ source, build context, shell
   commands, tool output, and diagnostics are represented as explicit domains,
   not anonymous prose. The current tokenizer contract already reserves code,
   fill-in-middle, whitespace, and tool-use tokens in
   [`cppmega_mlx/data/tokenizer_contract.py`](../cppmega_mlx/data/tokenizer_contract.py)
   and the C++ tokenizer preserves code-aware whitespace and comments in
   [`cppmega_mlx/tokenizer/cpp_tokenizer.py`](../cppmega_mlx/tokenizer/cpp_tokenizer.py).
2. A graph route packet: token/chunk-aligned edges and route metadata are passed
   beside the token stream. [`GraphPacket`](../cppmega_mlx/data/graph_packet.py)
   carries typed relations, [`CodePacket`](../cppmega_mlx/data/code_packet.py)
   binds tokens, provenance, structure, semantics, graph edges, and chunks, and
   [`CodePacketBuilder`](../cppmega_mlx/data/code_packet_builder.py) converts
   sidecar columns into those packets without fabricating missing channels.

The mandatory rule is stricter than the existing generic side-channel plan in
[`docs/side_channel_conditioning_plan.md`](side_channel_conditioning_plan.md):
side channels may be optional for generic experiments, but graph routes and
domain wrappers are required for cppmega world models.

## Stream Domains

The token stream is a typed transcript, not a flat prompt. Current code already
provides the building blocks: stable special token IDs, C++-aware tokenization,
and a platform-context renderer in
[`cppmega_mlx/data/platform_context.py`](../cppmega_mlx/data/platform_context.py).
The world-model packer must use these surfaces consistently.

| Domain | Wrapper contract | Required payload | Live anchors |
| --- | --- | --- | --- |
| C/C++ source | `&lt;CODE_START&gt; ... &lt;CODE_END&gt;` with code-aware whitespace tokens | Source spans, file path/doc id, token offsets, chunk ids, platform context | [`tokenizer_contract.py`](../cppmega_mlx/data/tokenizer_contract.py), [`cpp_tokenizer.py`](../cppmega_mlx/tokenizer/cpp_tokenizer.py), [`platform_context.py`](../cppmega_mlx/data/platform_context.py) |
| Build | Tool query/result span, for example `&lt;QUERY_TOOL&gt; build:cxx ... &lt;TOOL_RESULT&gt; ...` | Build system, compiler, target triple, flags, exit status, stdout/stderr, produced diagnostics | [`cppmega_v4/buildspec/api.py`](../cppmega_v4/buildspec/api.py), sibling [`../cppmega/cppmega/recipes/run_profiles.py`](../../cppmega/cppmega/recipes/run_profiles.py) |
| Shell | Tool query/result span | Command, cwd, env profile id, exit status, stdout/stderr, touched artifact hints | sibling [`../cppmega/cppmega/recipes/run_profiles.py`](../../cppmega/cppmega/recipes/run_profiles.py), sibling [`docs/status/cppmega_run_profiles_and_token_flow.md`](../../cppmega/docs/status/cppmega_run_profiles_and_token_flow.md) |
| Diagnostics | Tool result or diagnostic span tied back to source tokens | Compiler/test/build-spec error code, message, source range, symbol, command/build profile | [`cppmega_v4/buildspec/diagnostics.py`](../cppmega_v4/buildspec/diagnostics.py), [`tools/clang_indexer/process_commits.py`](../tools/clang_indexer/process_commits.py) |

Exact wrapper spellings beyond the existing special tokens are a packer/schema
decision, but the invariant is fixed: every non-source domain must be explicit
inside the stream and aligned to graph metadata where possible. Free-form
untyped prose should not enter the world-model training stream.

## Graph Route Surface

Graph routing is not a retrieval hint added after attention. It is a first-class
input used before sparse token selection:

- [`GraphPacket`](../cppmega_mlx/data/graph_packet.py) stores relation-indexed
  token edges and aggregates them to block routes.
- [`CodePacket`](../cppmega_mlx/data/code_packet.py) is the structured sample
  object that carries tokens, provenance, structure channels, semantic channels,
  graph edges, and chunk metadata together.
- [`CodePacketBuilder`](../cppmega_mlx/data/code_packet_builder.py) maps parquet
  row-group columns and LM token batches into `CodePacket` without inventing
  optional metadata when source columns are absent.
- [`cppmega_v4/probe/capabilities.py`](../cppmega_v4/probe/capabilities.py)
  distinguishes token-level graph columns from source-level graph aliases; only
  token graph columns are automatically remapped.
- [`cppmega_v4/probe/requirements.py`](../cppmega_v4/probe/requirements.py)
  already makes bricks/losses such as `engram`, `csa_hca`, and MHC dependent on
  call/type graph inputs.

Current relation families are `call` and `type`, with `def_use` represented in
packet semantics and graph relation examples. The world-model schema should add
the same packet interface for build and diagnostic relations:

| Relation | Meaning | Status |
| --- | --- | --- |
| `call` | Caller/callee or use-to-definition call structure | Implemented graph route family |
| `type` | Token/symbol to type relation | Implemented graph route family |
| `def_use` | Value-origin and variable-flow relation | Represented in packet semantics; route promotion required |
| `include` | Header/include dependency and translation-unit reachability | Design target |
| `build_error` | Compiler/linker/build failure span to source span, command, and symbol | Design target |
| `diagnostic` | Test, static-analysis, or build-spec diagnostic span to source/build node | Design target |

Missing graph columns are sample/schema errors for cppmega world models. For
non-world-model baselines, missing relations may be represented as an explicit
unknown/empty route, but they must not be silently converted into text-only
training.

## Route Bias And Sparse Selection

The attention path follows a graph-biased DSA/indexer design:

```text
domain-wrapped tokens + GraphPacket
    -> CodeGraphRouter
    -> S_graph relation/block bias
    -> I_final = I_neural + beta * S_graph
    -> top-k candidate selection
    -> SparseMLA / DSA attention
```

[`cppmega_mlx/nn/code_graph_routes.py`](../cppmega_mlx/nn/code_graph_routes.py)
implements the route prior. It converts fixed graph edges into block-level bias,
uses learnable relation weights, defaults to `call` and `type`, and fails loudly
when required relations are missing. This is the local "TNO-style" rule: the
route topology is fixed by compiler/indexer evidence, while the channel mixing
is learned. The external Topological Neural Operators paper is cited for the
same design principle of fixed structural flow with learned transforms, not as a
claim that cppmega implements that paper directly.

[`cppmega_mlx/training/indexer_losses.py`](../cppmega_mlx/training/indexer_losses.py)
then supervises the indexer with:

- KL distillation from dense attention during warm-up.
- BCE edge supervision for graph-supported candidates.
- Coverage hinge loss so route-supported candidates are not starved out of
  top-k.
- `recall_at_k` and dense-attention top-k overlap metrics.
- `apply_graph_indexer_bias` and `select_graph_biased_topk`, which apply graph
  bias before top-k selection.

The sparse attention consumer is
[`cppmega_mlx/nn/sparse_mla.py`](../cppmega_mlx/nn/sparse_mla.py), whose API
expects sparse top-k indices and treats `-1` as invalid. In sibling
`../cppmega`, the Megatron path carries the same pressure into production:
[`cppmega/megatron/dsa_indexer_fused_patch.py`](../../cppmega/cppmega/megatron/dsa_indexer_fused_patch.py)
removes the full per-head index-score materialization,
[`cppmega/megatron/dsa_splitk_indexer_loss.py`](../../cppmega/cppmega/megatron/dsa_splitk_indexer_loss.py)
keeps loss computation streaming, and
[`cppmega/megatron/dsa_sparse_attention.py`](../../cppmega/cppmega/megatron/dsa_sparse_attention.py)
computes sparse entries rather than full score matrices.

## LoopWM Trajectories And The Stable Loop

Cppmega world-model training is trajectory-shaped. A sample predicts state
transitions over code and environment state, not just the next text token.
[`cppmega_mlx/data/trajectory_packet.py`](../cppmega_mlx/data/trajectory_packet.py)
defines `Transition` and `TrajectoryPacket` for ordered code-edit dynamics. A
real transition carries observation, action/edit supervision, next observation,
and optional changed-token regions; synthetic reward/done labels are explicit
and cannot be fabricated in normal real-code trajectories.

The stable loop is:

1. Observe a domain-wrapped stream plus graph route packet.
2. Predict a constrained edit, patch, build command, or diagnostic action.
3. Decode only through the active domain wrapper.
4. Validate syntax, tokenizer boundaries, graph alignment, and build/test
   outcome.
5. Convert compiler, linker, test, and build-spec failures into diagnostic
   streams and build-error graph routes.
6. Re-index the updated source/build state and feed the next transition.

The LoopWM influence is architectural rather than a claim of current parity with
the paper: cppmega should reuse a stable transition loop over typed environment
state, with graph routes refreshed at each step, instead of letting free-form
text rollouts drift away from buildable code. The trajectory packet gives the
local serialization contract for this.

## Constrained Valid-Code Generation

Generation is valid-code constrained:

- Code tokens are emitted only inside code wrappers.
- Build/shell/tool output is emitted only inside tool/diagnostic wrappers.
- The tokenizer must preserve `&lt;SPACE&gt;`, `&lt;NL&gt;`, strings, comments, and
  fill-in-middle boundaries.
- Generated edits are checked against source ranges and chunk metadata before
  they are accepted into the next state.
- Syntax/build/test failures become structured diagnostics and graph routes,
  not free-form explanations that replace the failed code.
- [`cppmega_v4/buildspec/api.py`](../cppmega_v4/buildspec/api.py) and
  [`cppmega_v4/buildspec/diagnostics.py`](../cppmega_v4/buildspec/diagnostics.py)
  show the fail-closed pattern: strict build-spec verification raises
  `BuildError` when diagnostics contain errors.

The clang indexers are the live source of compiler-derived structure:
[`tools/clang_indexer/index_project.py`](../tools/clang_indexer/index_project.py)
indexes cross-file functions, classes, calls, and dependency levels, while
[`tools/clang_indexer/process_commits.py`](../tools/clang_indexer/process_commits.py)
turns commit diffs into enriched training documents with changed functions,
call graph context, dependency levels, and edit operations.

## Diagnostics Enrichment And Build-Error Routes

Build and diagnostic feedback should be represented twice:

1. In the wrapped stream, so the model sees the exact command, output, and
   source span in temporal order.
2. In graph routes, so the indexer can bias attention from an error message to
   the responsible tokens, symbols, translation unit, build profile, and prior
   fixes.

The build-error route schema should be token/chunk aligned:

```text
diagnostic node
    -> source token span
    -> symbol/type/def-use nodes
    -> build command/profile
    -> previous fix transition when available
```

This route family lets DSA select the tokens that explain a build failure before
generation proposes the next edit. It also keeps diagnostics compatible with the
existing graph-packet and indexer-loss machinery instead of adding a separate
diagnostics-only retrieval path.

## Invariants For Implementers

- A cppmega world-model batch without graph routes is invalid.
- A cppmega world-model batch without domain wrappers is invalid.
- Graph bias is applied before top-k selection, not after attention.
- Route topology comes from compiler/indexer/build evidence; learnable weights
  may mix relation channels, but they must not hallucinate edges.
- C++/build/shell/diagnostic streams keep their domain boundaries through
  packing, batching, training, inference, and trajectory replay.
- Build/test failures are training signal. They become diagnostic streams and
  graph routes, not discarded logs.
- Baselines may omit parts of this contract only when they are explicitly named
  baselines and are not promoted as cppmega world models.

## Architecture Diagram

```text
           libclang / build / shell / test evidence
                         |
                         v
        +----------------+----------------+
        | CodePacket / TrajectoryPacket   |
        | tokens, chunks, provenance,     |
        | semantics, graph edges, edits   |
        +----------------+----------------+
                         |
             +-----------+-----------+
             |                       |
             v                       v
   domain-special-token stream   GraphPacket routes
   code/build/shell/diag         call/type/def_use/build_error
             |                       |
             +-----------+-----------+
                         v
              CodeGraphRouter fixed routes
                         |
                         v
         DSA indexer: I_neural + beta * S_graph
                         |
                         v
              graph-biased top-k SparseMLA
                         |
                         v
       LoopWM transition: constrained edit/action
                         |
                         v
        validation -> diagnostics -> enriched routes
                         |
                         +---- next transition
```

## Source List

Repo sources in `cppmega.mlx`:

- [`docs/graph_supervised_dsa_indexer.md`](graph_supervised_dsa_indexer.md)
- [`docs/data_pipeline.md`](data_pipeline.md)
- [`docs/side_channel_conditioning_plan.md`](side_channel_conditioning_plan.md)
- [`cppmega_mlx/nn/code_graph_routes.py`](../cppmega_mlx/nn/code_graph_routes.py)
- [`cppmega_mlx/data/graph_packet.py`](../cppmega_mlx/data/graph_packet.py)
- [`cppmega_mlx/data/code_packet.py`](../cppmega_mlx/data/code_packet.py)
- [`cppmega_mlx/data/code_packet_builder.py`](../cppmega_mlx/data/code_packet_builder.py)
- [`cppmega_mlx/data/trajectory_packet.py`](../cppmega_mlx/data/trajectory_packet.py)
- [`cppmega_mlx/data/platform_context.py`](../cppmega_mlx/data/platform_context.py)
- [`cppmega_mlx/data/tokenizer_contract.py`](../cppmega_mlx/data/tokenizer_contract.py)
- [`cppmega_mlx/tokenizer/cpp_tokenizer.py`](../cppmega_mlx/tokenizer/cpp_tokenizer.py)
- [`cppmega_mlx/training/indexer_losses.py`](../cppmega_mlx/training/indexer_losses.py)
- [`cppmega_mlx/nn/sparse_mla.py`](../cppmega_mlx/nn/sparse_mla.py)
- [`cppmega_v4/probe/requirements.py`](../cppmega_v4/probe/requirements.py)
- [`cppmega_v4/probe/capabilities.py`](../cppmega_v4/probe/capabilities.py)
- [`cppmega_v4/buildspec/api.py`](../cppmega_v4/buildspec/api.py)
- [`cppmega_v4/buildspec/diagnostics.py`](../cppmega_v4/buildspec/diagnostics.py)
- [`tools/clang_indexer/index_project.py`](../tools/clang_indexer/index_project.py)
- [`tools/clang_indexer/process_commits.py`](../tools/clang_indexer/process_commits.py)
- [`tests/test_code_graph_routes.py`](../tests/test_code_graph_routes.py)

Sibling sources in `../cppmega`:

- [`cppmega/megatron/structure_dataset_patch.py`](../../cppmega/cppmega/megatron/structure_dataset_patch.py)
- [`cppmega/megatron/structure_batch.py`](../../cppmega/cppmega/megatron/structure_batch.py)
- [`cppmega/megatron/dsa_indexer_fused_patch.py`](../../cppmega/cppmega/megatron/dsa_indexer_fused_patch.py)
- [`cppmega/megatron/dsa_splitk_indexer_loss.py`](../../cppmega/cppmega/megatron/dsa_splitk_indexer_loss.py)
- [`cppmega/megatron/dsa_sparse_attention.py`](../../cppmega/cppmega/megatron/dsa_sparse_attention.py)
- [`cppmega/megatron/sparse_mla_ops/sparse_mla.py`](../../cppmega/cppmega/megatron/sparse_mla_ops/sparse_mla.py)
- [`cppmega/megatron/tilelang_sparse_mla/topk_selector.py`](../../cppmega/cppmega/megatron/tilelang_sparse_mla/topk_selector.py)
- [`cppmega/megatron/dsa_local_spec.py`](../../cppmega/cppmega/megatron/dsa_local_spec.py)
- [`cppmega/remote/dsa_config_patch.py`](../../cppmega/cppmega/remote/dsa_config_patch.py)
- [`cppmega/recipes/run_profiles.py`](../../cppmega/cppmega/recipes/run_profiles.py)
- [`docs/status/cppmega_architecture_status.md`](../../cppmega/docs/status/cppmega_architecture_status.md)
- [`docs/status/cppmega_run_profiles_and_token_flow.md`](../../cppmega/docs/status/cppmega_run_profiles_and_token_flow.md)
- [`docs/data_preparation.md`](../../cppmega/docs/data_preparation.md)
- [`scripts/data/verify_tokenizer_contract.py`](../../cppmega/scripts/data/verify_tokenizer_contract.py)
- [`tests/test_structure_dataset_patch_bridge.py`](../../cppmega/tests/test_structure_dataset_patch_bridge.py)

Verified external sources:

- [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366)
- [GraphCodeBERT OpenReview PDF](https://openreview.net/pdf?id=jLoC4ez43PZ)
- [DeepSeek-V3.2 technical report PDF](https://arxiv.org/pdf/2512.02556)
- [Looped World Models](https://arxiv.org/html/2606.18208v1)
- [Topological Neural Operators](https://arxiv.org/abs/2606.09806)
- [Clang libclang tutorial](https://clang.llvm.org/docs/LibClang.html)
- [libclang C Interface to Clang](https://clang.llvm.org/doxygen/group__CINDEX.html)
- [MLX documentation](https://ml-explore.github.io/mlx/)
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/en/index)
- [TileLang documentation](https://tilelang.com/)
- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [LLVM project repository](https://github.com/llvm/llvm-project)
- [Boost C++ Libraries repository](https://github.com/boostorg/boost)
- [{fmt} repository](https://github.com/fmtlib/fmt)
- [GoogleTest repository](https://github.com/google/googletest)
