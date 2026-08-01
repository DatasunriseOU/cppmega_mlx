# Mamba3 / DSA / MoE / MTP Checkpoint Conversion Decision (CUDA → MLX)

```text
Status: active
Canonical: none
Date: 2026-08-01
Scope: Draft decision on whether and how to extend Megatron→MLX checkpoint
       conversion beyond the dense500m GQA family to the active CUDA track
       (mamba3, DSA indexer, MoE, MTP). Final decision rests with the owner.
```

## Context

`scripts/convert_megatron_dense500m_torchdist_to_mlx.py` is deliberately
narrow. `_require_supported_attention_mode` (lines 104-110) raises
`NotImplementedError` for any `attention_mode != "gqa"`, and the source
validator additionally hard-fails on MLA/MoE tensor families, untied output
weights, non-`None` `num_experts` / `mtp_num_layers`, and any hybrid layer
pattern other than the all-attention dense route (`*-` × 24). The supported
contract and parity gate are documented in
`docs/mtr005_megatron_dcp_to_mlx.md`.

Meanwhile the active CUDA track has moved past that checkpoint family:
mamba3 MIMO blocks (see
`cppmega/docs/mamba3_mimo_p2_psiv_cache_design.md` and related status notes),
the graph-supervised DSA lightning indexer, MoE experts, and MTP heads. None
of these can be converted today, so local MLX inference/training/evals cannot
consume current CUDA checkpoints.

The MLX side is not starting from zero. Reference implementations already
exist and are wired into `cppmega_mlx/models/hybrid_lm.py`:

- `cppmega_mlx/nn/mamba3.py` — `Mamba3ReferenceBlock` (correctness-first,
  with Path B/Path C backward variants).
- `cppmega_mlx/nn/sparse_mla.py` — graph-supervised lightning indexer
  (`lightning_indexer_scores`) with `block_bias` graph prior.
- `cppmega_mlx/nn/moe.py` — `ReferenceMoE`.
- `cppmega_mlx/training/mtp.py` — MTP objective on the training side.

So the open question is conversion and parity, not MLX model coverage.

## Options

### A. Full port

Extend the converter to map every active CUDA family (mamba3 state, DSA
indexer, MoE experts, MTP heads) in one effort.

- Cost: highest. Four new tensor-family mappings plus four parity gates,
  each with its own receipt format, landed while the CUDA source layouts are
  still moving. Every upstream layout change invalidates a mapping and its
  parity receipt.
- Benefit: any CUDA checkpoint becomes MLX-consumable; local evals and
  training always run on current weights.

### B. Do not port (status quo)

Keep the converter dense500m-only. MLX consumes only dense GQA checkpoints;
the CUDA track stays CUDA-only.

- Cost: zero new code.
- Benefit: none for the MLX track. Local MLX evals keep running on stale
  dense500m weights, and the mamba3/DSA reference implementations in
  `cppmega_mlx/nn` remain untestable against real trained weights.

### C. Staged partial port (per-family, parity-gated)

Extend conversion one family at a time, each behind the same strict
validation style the converter already uses (explicit allowlists, blocking
errors on unknown tensors), each landing only with its own parity receipt:

1. **mamba3 state** — map Megatron mamba3 MIMO tensors (projections, conv,
   dt/A/D state parameters, per-head `psi`, norms) onto
   `Mamba3ReferenceBlock`. Highest immediate value: unblocks local evals and
   training on the current CUDA track. Parity: NumPy-reference forward gate
   in the style of the dense500m 24-layer gate, including recurrent-state
   (`h0`) semantics.
2. **DSA indexer** — map lightning-indexer projections and graph `block_bias`
   parameters; depends on the graph-route parity front (block_bias eval
   receipt) being settled. Parity: indexer score-matrix comparison plus
   end-to-end logit gate.
3. **MoE experts** — map router + expert tensors onto `ReferenceMoE`.
   Moderate key-mapping cost; sequence after mamba3/DSA because the CUDA MoE
   configuration is still churning.
4. **MTP heads** — map MTP head weights; only matters if MLX-side training on
   converted checkpoints becomes a goal (MTP affects objectives, not plain
   inference). Lowest priority; defer until an MLX training consumer exists.

Per-family parity procedure, reused for each stage: independent NumPy
reference forward over raw DCP tensor names, fresh MLX reload of the emitted
safetensors, max/mean/p99/RMS logit error against `atol=0.004, rtol=0.001`,
`model.json` carrying SHA-256 of every source artifact and the complete
source-to-target tensor map. Packed-document n-gram parity remains out of
scope here (tracked separately as backlog P066).

## Recommendation

**Option C — staged partial port, in the order mamba3 → DSA → MoE → MTP.**

Rationale:

- The CUDA track is the source of truth and is actively changing; a
  big-bang port (A) would chase moving layouts across four families at once,
  while a staged port only ever maintains one new mapping at a time.
- Doing nothing (B) strands the MLX reference implementations and keeps
  local evals on stale weights; the mamba3 stage alone removes most of that
  cost.
- The MLX reference blocks already exist, so each stage is predominantly
  key-mapping plus a parity gate — no new MLX kernel work is required to
  start.
- MTP has no inference consumer today, so deferring it loses nothing.

This is a draft recommendation. **The final go/no-go decision rests with the
owner.** On "go", create a bd epic with per-family subtasks linked to
`cppmega-mlx-c30.1` (backlog step P065).
