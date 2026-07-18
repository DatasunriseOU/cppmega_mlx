# Side-Channel Conditioning Plan

## Goal

Make token metadata usable as a configurable conditioning system for training
and inference, without making the implementation C++-specific. The same
contract must work when we have all metadata, a partial subset, or only plain
text tokens.

This is not an AST-to-AST objective. The primary objective remains next-token
prediction. Side channels are optional inputs that condition hidden states,
attention/routing, or losses when they are available.

## Current Anchors

Existing design docs to mirror:

- `ModelBuildSpec.md`: staged backend and GUI contract style.
- `VisualBuilderSpec-v2.md`: GUI-visible configuration, validation, JSON-RPC,
  and test matrix style.
- `docs/data_pipeline.md`: current data ingress and side-channel pipeline.
- `docs/parquet_samples.md`: current sample parquet gap report.

Existing code seams to extend:

- `cppmega_v4/probe/capabilities.py`: parquet/tokenizer capability discovery.
- `cppmega_v4/probe/requirements.py`: requirements and alternatives.
- `cppmega_v4/jsonrpc/schema.py`: GUI-facing JSON-RPC parameter schema.
- `cppmega_v4/jsonrpc/methods.py`: JSON-RPC method dispatch.
- `cppmega_mlx/data/parquet_dataset.py`: training batch materialization.
- `cppmega_mlx/nn/platform_embedding.py`: platform side-channel residual.
- `cppmega_mlx/nn/structure_embedding.py`: structure/AST side-channel residual.
- `vbgui/src/components/DataInspector.tsx`: parquet/token preview.
- `vbgui/src/components/Sidebar.tsx`: current configuration tabs.
- `vbgui/src/state/spec.ts`: GUI build-spec state.

Important current gaps:

- The GB10 sample parquet is not an enriched-tokenized packed-row artifact.
  It currently preserves some source-level metadata, but after retokenization
  the token-aligned side-channel columns are missing.
- `VerifyParams.available_side_channels` exists, but it is only a list of
  names. It does not yet express family policy, coverage, derivation, dropout,
  or inference fallback.
- Platform conditioning has a real consumer. Structure conditioning exists, but
  the zero-initialization path must be checked so gradients reach the side
  embeddings while the initial model behavior remains unchanged.
- The GUI can inspect data and configure losses/optimizers/rewriters/sharding,
  but side-channel policy is not yet a first-class configuration surface.

## Data Model

Side channels should be grouped by family, not hard-coded per language.

| Family | Examples | Training use | Inference source |
| --- | --- | --- | --- |
| universal | `input_ids`, `target_ids`, `loss_mask`, `doc_ids`, `pack_id`, `valid_token_count`, `num_docs` | batch structure and CE masking | tokenizer and packer |
| platform | `platform_ids`, `source_platform_ids`, OS, arch, compiler, accelerator, ABI, language standard | additive residual, routing tags, ablations | explicit `PlatformContext`, prompt text, repo/toolchain config |
| syntax | `token_ast_depth`, `token_sibling_index`, `token_ast_node_type` | additive residual, optional auxiliary prediction later | parser/compiler adapter when source is parseable |
| structure | `token_structure_ids`, `token_dep_levels`, chunk start/end/kind/depth | residual, chunk-aware masks, packing diagnostics | parser plus token-span mapper |
| semantic graph | symbol ids, call targets, type refs, def-use, call/type edges | graph bias, retrieval/routing hints, optional losses later | compiler/indexer adapter when project context is available |
| temporal diff | change masks, hunk ids, edit ops, commit windows | repair/edit conditioning, optional edit-head losses later | diff/patch input, VCS context, issue-agent context |

The packed-row training parquet must keep everything in token coordinates:

- Required token fields:
  `input_ids`, `target_ids`, `loss_mask`, `doc_ids`, `pack_id`,
  `valid_token_count`, `num_docs`.
- Optional token side channels:
  all family-specific per-token arrays sliced/padded to the packed row length.
- Optional graph/span channels:
  edge lists and chunk spans remapped to packed-row token coordinates.
- Required provenance:
  source file id, language id, extractor name/version, tokenizer id, and
  whether each family is original, derived, missing, or dropped.

## Configuration Contract

Add a generic `SideChannelSpec` that is serializable through Python, JSON-RPC,
and GUI state.

```python
SideChannelSpec(
    mode="auto",  # off | auto | require | if_available
    families={
        "platform": FamilySpec(mode="auto", dropout=0.10, residual_scale=1.0),
        "syntax": FamilySpec(mode="if_available", dropout=0.25),
        "structure": FamilySpec(mode="if_available", dropout=0.25),
        "semantic_graph": FamilySpec(mode="if_available", dropout=0.50),
        "temporal_diff": FamilySpec(mode="off"),
    },
    inference=InferenceEnrichmentSpec(
        source="auto",  # none | prompt_only | parse_if_possible | project_index
        fail_policy="error",  # error by default
        # drop_family/text_only are explicit opt-ins
        timeout_ms=500,
    ),
)
```

Each `FamilySpec` should include:

- `mode`: `off`, `auto`, `require`, or `if_available`.
- `columns`: expected parquet columns and aliases.
- `embedding`: `categorical`, `numeric_bucket`, `span`, `edge_bias`, or
  `none`.
- `dropout`: probability of dropping that family during training.
- `residual_scale` or `gate_init`: explicit residual/gating policy.
- `fallback`: `zeros`, `unknown_id`, `drop_family`, or `error`.
- `language_scope`: `any`, `cpp`, `rust`, `go`, `python`, or custom plugin id.

This gives three useful modes:

- Full metadata: train and infer with every configured family.
- Partial metadata: train with family dropout and infer with available families.
- No metadata: exact text-only behavior remains valid and tested.

## Training Path

Training should consume side channels as conditioning, not as required labels.

1. Materialize enriched token parquet into packed-row parquet.
   Reuse the existing sequential/best-fit packing policy and keep token
   coordinates stable inside each packed row.
2. Extend capability discovery to report side-channel family coverage,
   token alignment, graph remapping status, and provenance.
3. Extend `LMTokenBatch` and dataset loading with a `side_channels` map keyed
   by family and column name.
4. Route enabled families into model consumers:
   - platform: existing platform embedding residual;
   - syntax/structure: structure embedding residual;
   - graph: first as optional attention/routing bias, later as dedicated graph
     modules only if measurable;
   - temporal/diff: edit-conditioning residual and optional future heads.
5. Apply side-channel dropout during training so inference remains robust when
   some families are missing.
6. Keep CE next-token loss as the default objective. Add auxiliary heads only
   behind explicit config, for example AST node prediction or edit-op
   prediction, and never make them implicit.

Gradient behavior:

- Side-channel embeddings are ordinary trainable parameters optimized with the
  same optimizer as the model unless the config freezes them.
- Additive residual means the side embedding is projected to hidden size and
  added to token hidden states before the chosen block boundary.
- Zero initial behavioral impact should be achieved without killing gradients.
  If both the embedding table and projection are zero, no useful gradient reaches
  the side-channel table. The implementation should use a small trainable gate,
  non-zero projection, or another tested initialization that starts near zero
  while preserving gradient flow.

## Inference Path

Inference needs a generic enrichment builder:

```text
prompt text
  -> tokenizer
  -> optional language detector
  -> optional PlatformContext parser/renderer/encoder
  -> optional language adapter
  -> token-span mapper
  -> side-channel tensors plus provenance
  -> model forward
```

The builder should support these input levels:

- Text only: no side channels except defaults and tokenizer-derived structure.
- Text plus platform: explicit OS/compiler/arch/accelerator context becomes
  platform ids.
- Single parseable file: syntax/structure channels from the language adapter.
- Project context: semantic graph channels from compiler/indexer adapters.
- Diff/repair context: temporal/diff channels from patch/VCS metadata.

Language adapters must share one interface:

```python
class CodeMetadataAdapter:
    language: str

    def probe(self, context) -> AdapterCapabilities: ...
    def extract(self, source_or_project, options) -> CodeMetadata: ...
    def map_to_tokens(self, metadata, tokens, tokenizer) -> TokenMetadata: ...
```

Initial adapters:

- C/C++: clang compile database, clang AST/index data, existing nanochat
  generator code where it matches this contract.
- Rust: rust-analyzer or rustc JSON/HIR adapter later.
- Go: `go/packages` or `gopls` adapter later.
- Python: `ast` plus optional type/index provider later.

Fallback must be explicit. The inference builder defaults to fail-closed
`error`; a caller that intentionally runs a graphless/general-purpose model
must select one of the degraded policies and retain its receipt:

- `drop_family`: omit unavailable family and record provenance.
- `text_only`: run the model without all side channels.
- `error`: fail early when a required family cannot be produced.

## GUI Plan

Add side channels as a first-class config area, not only a data preview.

Required GUI surfaces:

- A `Side Channels` sidebar tab or a `Data` configuration tab.
- Family toggles with mode, dropout, fallback, and residual/gate settings.
- Platform context editor with canonical fields and rendered text preview.
- Data inspector coverage view:
  per-family present/missing/derived/dropped, token alignment, row coverage,
  and sample token ribbons.
- Inference playground controls:
  text-only vs enriched, language adapter, platform context, fail policy, and
  preview of generated side-channel tensors.
- Contract Probe panel:
  show which selected bricks/losses require which families and why.

The GUI should be generic. It should not contain C++-only assumptions beyond
adapter-specific labels.

## Test Plan

Backend unit tests:

- `SideChannelSpec` defaults, JSON serialization, validation, and bad config.
- Capability discovery on token-only, partial, and fully enriched parquet.
- Packed-row materializer preserves token coordinates and remaps graph/span
  metadata.
- Dataset loading returns the same CE fields with or without side channels.
- Platform and structure residual paths preserve text-only parity at init.
- Side-channel dropout actually drops families and keeps batch shapes stable.
- Inference builder supports text-only, platform-only, parseable single-file,
  and missing-adapter fallback.

GUI tests:

- Side-channel config renders from default spec.
- Toggling family modes updates JSON-RPC payloads.
- Data inspector shows family coverage and token ribbons.
- Inference playground can run text-only and enriched previews.
- Required-family errors are visible and actionable.

Smoke/E2E tests:

- Train smoke with no metadata.
- Train smoke with platform-only metadata.
- Train smoke with platform plus syntax/structure metadata.
- Inference smoke text-only vs enriched.
- Contract Probe smoke for selected model/loss/data combinations.

## Implementation Stages

### Stage A: Spec and schema

- Add `SideChannelSpec`, `FamilySpec`, `InferenceEnrichmentSpec`, and
  `DataMaterializationSpec`.
- Thread them through JSON-RPC schema and default GUI state.
- Add validation and serialization tests.

### Stage B: Data and probe

- Extend parquet capability discovery from column names to family coverage.
- Add packed-row materializer contract for all token-coordinate metadata.
- Add provenance fields and explicit missing/derived/dropped reasons.
- Add tests using current sample parquet and synthetic fully enriched parquet.

### Stage C: Model and training

- Fix side-channel residual initialization so gradients flow.
- Move side-channel tensors through `LMTokenBatch`.
- Add configurable family dropout.
- Keep CE as the default loss; add auxiliary losses only behind explicit flags.
- Add train-step smoke tests with no, partial, and full metadata.

### Stage D: Inference enrichment

- Add `InferenceSideChannelBuilder`.
- Wire canonical `PlatformContext` parser/renderer/encoder.
- Add C/C++ adapter first by reusing the existing clang/nanochat extractor code.
- Add cache keys based on content hash, tokenizer id, adapter version, and
  platform context.
- Add text-only and enriched inference tests.

### Stage E: GUI

- Add side-channel config state and JSON-RPC wiring.
- Extend Data Inspector with family coverage and alignment diagnostics.
- Add inference playground controls for enrichment.
- Add GUI tests for configuration, preview, and probe errors.

### Stage F: More languages

- Add Rust, Go, and Python adapters behind the same adapter interface.
- Keep language-specific extraction out of model code.
- Add adapter conformance tests so every language produces the same canonical
  token metadata contract.

## Acceptance Criteria

- Side-channel behavior is fully configurable from Python, JSON-RPC, and GUI.
- Training works with all metadata, partial metadata, and no metadata.
- Inference works with text only, platform context, and adapter-derived code
  metadata.
- The GUI can inspect available metadata, configure usage policy, and run
  preview/test flows.
- Contract Probe reports required, optional, missing, derived, and dropped
  side-channel families.
- No C++-specific logic leaks into generic model/training configuration.
- Rust, Go, and Python can be added by implementing the adapter interface, not
  by changing the core side-channel machinery.
