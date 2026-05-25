"""Fusion eligibility oracle — table-driven pair classifier.

Decides, for a producer/consumer pair of bricks, whether they can be fused
into one TileLang region (single PrimFunc, shared register state) or must
stay separate kernels glued by DLPack handoff.

The decision is by-kind, not by-instance: two bricks of compatible kinds can
fuse regardless of params, provided downstream descriptor synthesis succeeds.

Categories (assigned per brick kind):
  - ``linear_attn``: GDN / KDA / Bailing linear — single fused scan over T
    is the eligibility test; can fuse with norm/residual/MoE pre/post.
  - ``sdpa_attention``: gated_attention / mla / mistral4_mla / dsv4_attention
    / bailing_mla / attention — opaque ``mx.fast.scaled_dot_product_attention``
    kernel inside; can fuse output-side (gate*output, o_proj) but not the
    attention compute itself.
  - ``nonlinear_rnn``: m2rnn — recurrent tanh inside; can fuse pre/post
    normalisation but NOT inside the recurrence.
  - ``ssm``: mamba3 — selective SSM with chunkwise scan; fuses with norm
    pre/post like linear_attn.
  - ``moe``: moe / bailing_moe — top-k routed; fuses inside one expert FFN
    but the routing step is its own fusion boundary.
  - ``cross_attn``: gemma4_drafter — needs external K/V tensors; fusion
    only on the residual-add output side.
  - ``mtp``: nemotron_h_mtp — fusion depends on the underlying block_type;
    conservatively classified as ``sdpa_attention`` here.
  - ``norm_or_proj``: mlp / residual / lightning_indexer — small ops that
    fuse with neighbours easily.
  - ``sparse_attn``: nsa / csa_hca — data-dependent control flow; cannot
    fuse with anything in the current planner.
"""

from __future__ import annotations

from dataclasses import dataclass

from cppmega_v4.fusion.brick_graph import BrickNode


# ---------------------------------------------------------------------------
# Per-brick category map
# ---------------------------------------------------------------------------

_CATEGORY_BY_KIND: dict[str, str] = {
    # linear attention family
    "gdn": "linear_attn",
    "kda": "linear_attn",
    "bailing_linear": "linear_attn",
    # SDPA-backed attention family
    "gated_attention": "sdpa_attention",
    "attention": "sdpa_attention",
    "mla": "sdpa_attention",
    "mla_absorb": "sdpa_attention",
    "mistral4_mla": "sdpa_attention",
    "dsv4_attention": "sdpa_attention",
    "bailing_mla": "sdpa_attention",
    # Stage D — new SDPA-backed bricks (sliding GQA, coarse causal attn)
    "gqa_sliding": "sdpa_attention",
    "cca_attention": "sdpa_attention",
    # GalCov-B — gallery coverage bricks
    "mlstm": "nonlinear_rnn",
    "abs_pos_embed": "norm_or_proj",
    "per_layer_embed": "norm_or_proj",
    "embedding_table": "norm_or_proj",
    # cross attention drafter
    "gemma4_drafter": "cross_attn",
    # MTP block — opaque, treat like attention for fusion purposes
    "nemotron_h_mtp": "sdpa_attention",
    # State-space family
    # (mamba3 is in cppmega_mlx, not yet in v4 BLOCK_BUILDERS; covered defensively.)
    "mamba3": "ssm",
    "ssm": "ssm",
    # M2RNN nonlinear recurrence
    "m2rnn": "nonlinear_rnn",
    # MoE family
    "moe": "moe",
    "bailing_moe": "moe",
    # Sparse attention family (data-dependent control flow)
    "nsa": "sparse_attn",
    "csa_hca": "sparse_attn",
    # Light ops / projections / norms
    "mlp": "norm_or_proj",
    "engram": "norm_or_proj",
    "lightning_indexer": "norm_or_proj",
}


def _category(kind: str) -> str:
    return _CATEGORY_BY_KIND.get(kind, "unknown")


# ---------------------------------------------------------------------------
# Pair compatibility table
# ---------------------------------------------------------------------------
#
# Cells are ``(can_fuse, backend, reason_hint)``. A pair is fusable iff there
# is a backend that can compile both as one kernel (or compose them with
# zero overhead). The backends:
#   path_c          — TileLang fusion region; full PrimFunc
#   metal_inline    — both kernels are hand-MSL and can share threadgroup
#   dlpack_handoff  — adjacent kernels separated by a zero-copy boundary
#                     (no actual fusion; planner may still group for codegen
#                     orchestration but no shared registers)
#
# Symmetric by default (a-to-b same as b-to-a). Asymmetric entries are
# listed explicitly with both orderings.

_PairKey = tuple[str, str]
_PairValue = tuple[bool, str, str]

_PAIR_RULES: dict[_PairKey, _PairValue] = {}


def _set(a: str, b: str, can: bool, backend: str, reason: str) -> None:
    _PAIR_RULES[(a, b)] = (can, backend, reason)
    _PAIR_RULES[(b, a)] = (can, backend, reason)


# Same-category fusion
_set("linear_attn", "linear_attn", True, "path_c",
     "two linear-attn passes share a T loop; descriptor scan can stack them")
_set("sdpa_attention", "sdpa_attention", False, "dlpack_handoff",
     "two SDPA kernels each launch their own Metal kernel; no register sharing")
_set("ssm", "ssm", True, "path_c",
     "two SSM chunkwise scans share chunk_size loop structure")
_set("moe", "moe", False, "dlpack_handoff",
     "MoE routing is a hard fusion boundary; only inside an expert FFN can fuse")
_set("nonlinear_rnn", "nonlinear_rnn", False, "dlpack_handoff",
     "nonlinear recurrence cannot be merged without changing semantics")
_set("sparse_attn", "sparse_attn", False, "dlpack_handoff",
     "data-dependent control flow blocks fusion")
_set("norm_or_proj", "norm_or_proj", True, "path_c",
     "norms and projections are pure pointwise/matmul, fuse trivially")
_set("cross_attn", "cross_attn", False, "dlpack_handoff",
     "cross-attn needs external K/V; cannot fuse with another cross-attn")

# Mixed-category fusion (the productive cases)
_set("linear_attn", "norm_or_proj", True, "path_c",
     "linear-attn output feeds RMSNorm/o_proj; both fuse into one PrimFunc")
_set("ssm", "norm_or_proj", True, "path_c",
     "SSM output feeds residual+RMSNorm")
_set("sdpa_attention", "norm_or_proj", True, "path_c",
     "post-SDPA o_proj/residual can fuse with the gate-multiply on output side")
_set("moe", "norm_or_proj", True, "path_c",
     "MoE-expert FFN sandwiches a Linear pair; pre/post norm fuses")
_set("nonlinear_rnn", "norm_or_proj", True, "path_c",
     "pre-norm before tanh recurrence is fine; post-residual+norm also")
_set("cross_attn", "norm_or_proj", True, "path_c",
     "drafter output o_proj/residual fuses with the post-block norm")
_set("mtp", "norm_or_proj", True, "path_c",
     "MTP block output norm/residual can fuse with the head projection")

# Hard incompatibilities across categories
_set("linear_attn", "nonlinear_rnn", False, "dlpack_handoff",
     "linear-attn scan loop semantics differ from RNN tanh recurrence")
_set("linear_attn", "sparse_attn", False, "dlpack_handoff",
     "sparse attention's data-dependent indexing breaks linear-attn vectorisation")
_set("sdpa_attention", "nonlinear_rnn", False, "dlpack_handoff",
     "RNN recurrence cannot live inside a softmax kernel")
_set("sdpa_attention", "sparse_attn", False, "dlpack_handoff",
     "different read patterns; fuse impossible")
_set("sdpa_attention", "linear_attn", False, "dlpack_handoff",
     "SDPA softmax requires explicit attn weights; linear-attn drops them")
_set("sdpa_attention", "ssm", False, "dlpack_handoff",
     "discrete-time SSM scan differs from per-token softmax")
_set("ssm", "nonlinear_rnn", False, "dlpack_handoff",
     "different recurrence forms")
_set("ssm", "sparse_attn", False, "dlpack_handoff",
     "data-dependent control flow vs continuous-time discretisation")
_set("ssm", "linear_attn", True, "path_c",
     "both reduce to a scan; descriptor template can co-schedule them")
_set("moe", "linear_attn", False, "dlpack_handoff",
     "MoE routing tensors are token-disjoint; cannot share register state")
_set("moe", "sdpa_attention", False, "dlpack_handoff",
     "MoE routing is a hard fusion boundary")
_set("moe", "ssm", False, "dlpack_handoff",
     "MoE routing is a hard fusion boundary")
_set("moe", "nonlinear_rnn", False, "dlpack_handoff",
     "MoE routing is a hard fusion boundary")
_set("moe", "sparse_attn", False, "dlpack_handoff",
     "MoE routing is a hard fusion boundary")
_set("moe", "cross_attn", False, "dlpack_handoff",
     "MoE routing is a hard fusion boundary")
_set("cross_attn", "linear_attn", False, "dlpack_handoff",
     "drafter needs external K/V; cannot share state")
_set("cross_attn", "sdpa_attention", False, "dlpack_handoff",
     "drafter needs external K/V; cannot share state")
_set("cross_attn", "ssm", False, "dlpack_handoff",
     "drafter needs external K/V; cannot share state")
_set("cross_attn", "nonlinear_rnn", False, "dlpack_handoff",
     "drafter needs external K/V; cannot share state")
_set("cross_attn", "sparse_attn", False, "dlpack_handoff",
     "drafter needs external K/V; cannot share state")
_set("cross_attn", "moe", False, "dlpack_handoff",
     "drafter needs external K/V; cannot share state")
_set("sparse_attn", "nonlinear_rnn", False, "dlpack_handoff",
     "incompatible control-flow patterns")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionEligibility:
    """Result of a pair compatibility check."""

    can_fuse: bool
    backend: str  # "path_c" | "metal_inline" | "dlpack_handoff"
    reason: str
    producer_category: str
    consumer_category: str


def can_fuse_pair(a: BrickNode, b: BrickNode) -> FusionEligibility:
    """Return a FusionEligibility for the ordered (producer, consumer) pair.

    Always returns a result — never raises. Unknown kinds yield
    ``can_fuse=False, backend='dlpack_handoff'`` with a descriptive reason.
    """
    cat_a = _category(a.kind)
    cat_b = _category(b.kind)
    if cat_a == "unknown" or cat_b == "unknown":
        return FusionEligibility(
            can_fuse=False,
            backend="dlpack_handoff",
            reason=(
                f"unknown category for kind={a.kind!r}/{b.kind!r}; "
                "add to _CATEGORY_BY_KIND to enable fusion analysis"
            ),
            producer_category=cat_a,
            consumer_category=cat_b,
        )

    can, backend, reason = _PAIR_RULES.get(
        (cat_a, cat_b),
        (False, "dlpack_handoff",
         f"no rule for ({cat_a}, {cat_b}); defaulting to handoff"),
    )
    return FusionEligibility(
        can_fuse=can,
        backend=backend,
        reason=reason,
        producer_category=cat_a,
        consumer_category=cat_b,
    )


__all__ = [
    "FusionEligibility",
    "can_fuse_pair",
]
