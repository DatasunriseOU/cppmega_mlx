"""Static tables of data requirements per brick kind and per loss kind.

Both tables are deliberately frozen module-level dicts: probe results
must be deterministic, and additions must be visible in code review.

Coverage gate: ``BRICK_REQUIREMENTS`` MUST contain every key in
``BLOCK_BUILDERS``; the entry may be an empty tuple but the key must
be present. A test in tests/v4/test_probe_stage_b.py enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from cppmega_v4.buildspec.loss_spec import LossKind


@dataclass(frozen=True)
class DataRequirement:
    """One concrete piece of data a brick or loss expects to receive."""

    key: str
    origin: Literal["tokenizer", "parquet", "derived"]
    required: bool
    reason: str
    satisfied_by: tuple[str, ...] = ()

    def is_satisfied_by(self, provided: frozenset[str]) -> bool:
        if self.key in provided:
            return True
        return any(alt in provided for alt in self.satisfied_by)


# ---------------------------------------------------------------------------
# Per-brick data dependencies. Most bricks consume only ``input_ids`` and
# need nothing else from data — they get an empty tuple.
# ---------------------------------------------------------------------------


_ENGRAM_NEEDS_CALL_EDGES = DataRequirement(
    key="call_edges", origin="parquet", required=True,
    reason="engram brick consumes per-token call-graph edges to bias attention",
)

_CSAHCA_NEEDS_TYPE_EDGES = DataRequirement(
    key="type_edges", origin="parquet", required=True,
    reason="csa_hca brick consumes type-graph edges for hierarchical attention",
)


BRICK_REQUIREMENTS: Mapping[str, tuple[DataRequirement, ...]] = {
    # generic transformer bricks — only input_ids
    "attention":         (),
    "gated_attention":   (),
    "mla":               (),
    "mla_absorb":        (),
    "mistral4_mla":      (),
    "dsv4_attention":    (),
    "bailing_linear":    (),
    "bailing_mla":       (),
    "bailing_moe":       (),
    "gqa_sliding":       (),
    "cca_attention":     (),
    "gemma4_drafter":    (),
    "nemotron_h_mtp":    (),
    "lightning_indexer": (),
    "mlp":               (),
    "moe":               (),
    "gdn":               (),
    "kda":               (),
    "nsa":               (),
    "mamba3":            (),
    "mlstm":             (),
    "abs_pos_embed":     (),
    "per_layer_embed":   (),
    "embedding_table":   (),
    "rmsnorm":           (),
    "layernorm":         (),
    "residual":          (),
    # bricks that need real side-channels from parquet
    "engram":            (_ENGRAM_NEEDS_CALL_EDGES,),
    "csa_hca":           (_CSAHCA_NEEDS_TYPE_EDGES,),
}


# ---------------------------------------------------------------------------
# Per-loss data dependencies.
# ---------------------------------------------------------------------------


LOSS_REQUIREMENTS: Mapping[LossKind, tuple[DataRequirement, ...]] = {
    LossKind.CROSS_ENTROPY: (
        DataRequirement(
            key="labels", origin="derived", required=True,
            reason="CE loss needs next-token labels (collate-derived)",
            satisfied_by=("input_ids",),
        ),
    ),
    LossKind.MTP_WEIGHTED: (
        DataRequirement(
            key="labels_k_shifted", origin="derived", required=True,
            reason="MTP loss needs K shifted-label streams "
                   "(produced by collator from input_ids)",
            satisfied_by=("input_ids",),
        ),
    ),
    LossKind.IFIM_SHAPED: (
        DataRequirement(
            key="FIM_PREFIX", origin="tokenizer", required=True,
            reason="IFIM loss inserts <FIM_PREFIX> token at the cut point",
        ),
        DataRequirement(
            key="FIM_MIDDLE", origin="tokenizer", required=True,
            reason="IFIM loss inserts <FIM_MIDDLE> token before the suffix",
        ),
        DataRequirement(
            key="FIM_SUFFIX", origin="tokenizer", required=True,
            reason="IFIM loss inserts <FIM_SUFFIX> token before the rest",
        ),
    ),
    LossKind.MHC_ATTN_BIAS: (
        DataRequirement(
            key="type_edges", origin="parquet", required=True,
            reason="MHC loss needs type-graph edges to penalise wrong copy targets",
        ),
    ),
    LossKind.CUSTOM: (),
}
