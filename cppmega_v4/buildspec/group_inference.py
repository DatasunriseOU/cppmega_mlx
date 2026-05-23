"""Heuristic auto-grouping of model parameters for optimizer selection.

Given an instantiated brick graph and a target ``OptimKind`` (typically
``muon_adamw_hybrid`` but also valid for adamw / lion / muon), produce
a list of :class:`ProposedGroup` records describing how the GUI should
populate ``OptimSpec.groups``. Each proposal carries:

  - ``matcher``: regex DSL matching parameter qualified names
  - ``optim_kind``: which optimizer this subset routes to
    (relevant for muon_adamw_hybrid; equals the global kind otherwise)
  - ``lr / weight_decay / betas``: paper-aligned defaults from
    optim_spec factories
  - ``param_count``: how many parameters this matcher covers
  - ``rationale``: human-readable explanation rendered as the
    Auto-group banner tooltip

The hard rules:

  * 1D shape ∧ name contains ``embedding|lm_head|wte|wpe`` →
    embeddings group (AdamW with weight_decay=0)
  * 2D shape ∧ name contains ``expert`` → moe_experts group (AdamW)
  * 1D shape ∧ name contains ``bias|norm|gain`` → 1d group (AdamW)
  * 2D+ shape, no special name match → backbone group (Muon if
    hybrid; else the global kind)

When a parameter matches no rule, it is folded into the backbone
group; if backbone group is empty and uncovered_params > 0 we add
a single ``regex:.*`` AdamW catch-all so verify always passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cppmega_v4.buildspec.optim_spec import OptimKind


@dataclass(frozen=True)
class ProposedGroup:
    matcher: str
    optim_kind: OptimKind
    lr: float
    weight_decay: float
    betas: tuple[float, float] | None
    ns_steps: int | None
    param_count: int
    rationale: str


@dataclass(frozen=True)
class AutoGroupingResult:
    proposals: list[ProposedGroup] = field(default_factory=list)
    total_params: int = 0
    uncovered_params: int = 0


# Param-name buckets (case-insensitive substrings).
_EMBED_TOKENS = ("embed_tokens", "embedding", "wte", "wpe", "lm_head")
_EXPERT_TOKENS = ("expert", "moe.experts")
_NORM_TOKENS = ("norm", "gain", "rmsnorm", "layernorm", "scale")
_BIAS_TOKENS = ("bias",)


def _matches_any(name: str, tokens: Iterable[str]) -> bool:
    lc = name.lower()
    return any(tok in lc for tok in tokens)


def _bucket_for(name: str, shape: tuple[int, ...]) -> str:
    """Classify one parameter into a logical bucket."""
    if _matches_any(name, _EMBED_TOKENS):
        return "embeddings"
    if _matches_any(name, _EXPERT_TOKENS):
        return "moe_experts"
    if len(shape) == 1 and _matches_any(name, _BIAS_TOKENS + _NORM_TOKENS):
        return "norm_or_bias"
    if len(shape) == 1:
        # 1D leaf that didn't match a known token (e.g. embedding pad
        # vector, learned positional bias) — treat as norm_or_bias too.
        return "norm_or_bias"
    if len(shape) >= 2:
        return "backbone_2d"
    return "other"


def _flat_iter(params: dict, prefix: str = "") -> Iterable[tuple[str, tuple[int, ...]]]:
    """Walk an MLX-style nested params dict and yield (qualified_name, shape).
    Treats anything with a `shape` attribute as a leaf."""
    for k, v in params.items():
        name = f"{prefix}.{k}" if prefix else k
        if hasattr(v, "shape"):
            yield name, tuple(v.shape)
        elif isinstance(v, dict):
            yield from _flat_iter(v, name)
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if hasattr(item, "shape"):
                    yield f"{name}.{i}", tuple(item.shape)
                elif isinstance(item, dict):
                    yield from _flat_iter(item, f"{name}.{i}")


def suggest_groups(
    params: dict,
    optim_kind: OptimKind,
) -> AutoGroupingResult:
    """Classify each leaf parameter, then collapse into matcher groups.

    Returns a list of ``ProposedGroup`` ordered as the GUI should
    insert them (specific first, catch-all last)."""
    buckets: dict[str, int] = {}
    bucket_names: dict[str, list[str]] = {}

    total = 0
    for name, shape in _flat_iter(params):
        b = _bucket_for(name, shape)
        n = 1
        for d in shape:
            n *= int(d)
        buckets[b] = buckets.get(b, 0) + n
        bucket_names.setdefault(b, []).append(name)
        total += n

    proposals: list[ProposedGroup] = []
    hybrid = optim_kind is OptimKind.MUON_ADAMW_HYBRID
    pure_muon = optim_kind is OptimKind.MUON
    adam_betas = (0.9, 0.95)
    adam_lr = 3e-4
    muon_lr = 2e-3
    lion_lr = 1e-4
    sgd_lr = 1e-2

    def adam_group(matcher: str, count: int, rationale: str,
                   wd: float = 0.01) -> ProposedGroup:
        return ProposedGroup(
            matcher=matcher, optim_kind=OptimKind.ADAMW, lr=adam_lr,
            weight_decay=wd, betas=adam_betas, ns_steps=None,
            param_count=count, rationale=rationale,
        )

    def lion_group(matcher: str, count: int, rationale: str) -> ProposedGroup:
        return ProposedGroup(
            matcher=matcher, optim_kind=OptimKind.LION, lr=lion_lr,
            weight_decay=0.01, betas=(0.9, 0.99), ns_steps=None,
            param_count=count, rationale=rationale,
        )

    def muon_group(matcher: str, count: int, rationale: str) -> ProposedGroup:
        return ProposedGroup(
            matcher=matcher, optim_kind=OptimKind.MUON, lr=muon_lr,
            weight_decay=0.01, betas=None, ns_steps=5,
            param_count=count, rationale=rationale,
        )

    def sgd_group(matcher: str, count: int, rationale: str) -> ProposedGroup:
        return ProposedGroup(
            matcher=matcher, optim_kind=OptimKind.SGD, lr=sgd_lr,
            weight_decay=0.0, betas=None, ns_steps=None,
            param_count=count, rationale=rationale,
        )

    def specific_group(matcher: str, count: int, rationale: str) -> ProposedGroup:
        """Pick the right specific-group optimiser. For hybrid: AdamW
        on 1D/embeddings/MoE. For pure adamw/muon/lion/sgd: same as
        backbone."""
        if hybrid:
            return adam_group(matcher, count, rationale)
        if optim_kind is OptimKind.ADAMW or optim_kind is OptimKind.ADAM_8BIT:
            return adam_group(matcher, count, rationale)
        if optim_kind is OptimKind.LION or optim_kind is OptimKind.LION_8BIT:
            return lion_group(matcher, count, rationale)
        if pure_muon:
            return muon_group(matcher, count, rationale)
        return sgd_group(matcher, count, rationale)

    if buckets.get("embeddings", 0):
        proposals.append(specific_group(
            "embeddings", buckets["embeddings"],
            f"AdamW on {len(bucket_names['embeddings'])} lookup tables "
            f"({buckets['embeddings']} params) — Muon would skip these "
            f"(1D lookup, no 2D matmul)",
        ))
    if buckets.get("moe_experts", 0):
        proposals.append(specific_group(
            "moe_experts", buckets["moe_experts"],
            f"AdamW on {len(bucket_names['moe_experts'])} MoE expert "
            f"weights ({buckets['moe_experts']} params) — small "
            f"per-expert matrices benefit from AdamW over Muon",
        ))
    if buckets.get("norm_or_bias", 0):
        proposals.append(specific_group(
            "regex:.*(bias|norm|gain|scale).*", buckets["norm_or_bias"],
            f"AdamW on {len(bucket_names['norm_or_bias'])} 1D params "
            f"({buckets['norm_or_bias']} params) — biases, layer-norm "
            f"gains, scalars",
        ))
    if buckets.get("backbone_2d", 0):
        if hybrid or pure_muon:
            proposals.append(muon_group(
                "regex:.*\\.weight$", buckets["backbone_2d"],
                f"Muon on {len(bucket_names['backbone_2d'])} 2D linear "
                f"weights ({buckets['backbone_2d']} params) — Q/K/V/O "
                f"projections, MLP gate/up/down, etc.",
            ))
        elif optim_kind in (OptimKind.LION, OptimKind.LION_8BIT):
            proposals.append(lion_group(
                "all", buckets["backbone_2d"],
                f"Lion on {len(bucket_names['backbone_2d'])} 2D weights "
                f"(lr=1e-4 is 10x smaller than AdamW because sign-based "
                f"updates ignore gradient magnitude)",
            ))
        else:
            proposals.append(specific_group(
                "all", buckets["backbone_2d"],
                f"{optim_kind.value} on {len(bucket_names['backbone_2d'])}"
                f" 2D weights",
            ))

    if not proposals and total > 0:
        # No bucket matched anything — emit a catch-all.
        proposals.append(specific_group(
            "all", total, f"{optim_kind.value} on all {total} parameters",
        ))

    covered = sum(p.param_count for p in proposals)
    return AutoGroupingResult(
        proposals=proposals,
        total_params=total,
        uncovered_params=max(0, total - covered),
    )


__all__ = [
    "AutoGroupingResult", "ProposedGroup", "suggest_groups",
]
