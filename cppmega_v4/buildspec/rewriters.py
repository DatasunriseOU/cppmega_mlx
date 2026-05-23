"""Concrete graph rewriters.

Stage C + D scope (this commit family): :class:`MTPRewriter`,
:class:`IFIMRewriter`, :class:`MHCRewriter`.

A rewriter is a callable ``(ModelBuildSpec) -> ModelBuildSpec`` that
matches the :class:`cppmega_v4.buildspec.Rewriter` Protocol. It:

  - reads the input graph + loss + optim
  - materialises a new graph (adding/removing nodes / edges)
  - rewrites the loss spec to match the new outputs (e.g. CE → MTP-weighted)
  - optionally adds a new optimizer parameter group (e.g. head-only lr)

Rewriters are PURE — they never mutate the input spec; they return a
fresh one via :meth:`ModelBuildSpec.replace`.

MTPRewriter
-----------

Multi-Token Prediction rewrite. Given a spec with a single head brick
(``state_token "single_head"``), materialises K-1 additional head copies
and rewrites the loss to :func:`mtp_weighted_loss`. The new state token
``"mtp_k_heads"`` advertises the post-condition for later rewriters.

Naming convention: the original head brick is renamed
``<head>_0`` and copies become ``<head>_1 ... <head>_{k-1}``. Loss
head_outputs match this scheme (``logits_0 / logits_1 / ...``).

Param-group integration: when ``add_head_param_group=True`` (default
True), the optimizer gains a new high-priority ``regex:.*_head_\\d+$``
group with the same lr as the base — callers can override the head-only
lr after the fact via :meth:`OptimSpec.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from cppmega_v4.buildspec.loss_spec import (
    LossKind,
    LossSpec,
    ifim_shaped_loss,
    mtp_weighted_loss,
)
from cppmega_v4.buildspec.model_build_spec import (
    ModelBuildSpec,
)
from cppmega_v4.buildspec.optim_spec import (
    OptimKind,
    OptimSpec,
    ParamGroup,
)
from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_HEAD_KINDS: Final[frozenset[str]] = frozenset({
    # bricks that produce the "logits" head — Stage C scope. Add more
    # as we wire heads explicitly. For now we recognise mlp-as-head and
    # accept an explicit ``is_head=True`` attribute on a BrickNode.
    "mlp",
    "attention",
})


def _find_head_node(graph: BrickGraph) -> BrickNode:
    """Locate the head brick.

    Resolution order:
      1. Last node in the graph whose ``params`` contains ``"is_head": True``.
      2. Last node in the graph whose kind is in :data:`_HEAD_KINDS`.
      3. Else: raises :class:`HeadDetectionError`.
    """
    explicit = [
        n for n in graph.nodes if n.params.get("is_head") is True
    ]
    if explicit:
        return explicit[-1]
    typed = [n for n in graph.nodes if n.kind in _HEAD_KINDS]
    if not typed:
        raise HeadDetectionError(
            "no head brick found — annotate a brick with "
            "`params={'is_head': True}` or end the graph with one of "
            f"{sorted(_HEAD_KINDS)}"
        )
    return typed[-1]


class HeadDetectionError(RuntimeError):
    """Raised by :class:`MTPRewriter` when no head brick can be located."""


# ---------------------------------------------------------------------------
# MTPRewriter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MTPRewriter:
    """K-head Multi-Token-Prediction rewriter.

    Fields:
      k: number of prediction heads (≥ 1). K=1 is a no-op (returns input
        spec unchanged besides the postcondition token).
      beta: optional per-head weights for the mtp_weighted loss. When
        None, falls back to the default geometric decay used by
        :func:`mtp_weighted_loss`.
      share_backbone: when True (default), the backbone bricks are kept
        as-is and only the head is replicated K times. When False, the
        original head node is renamed ``_0`` and new ``_1.._{k-1}``
        clones are added (still pointing into the backbone end-node).
      add_head_param_group: when True (default), the optimizer spec gets
        a new head-only param group with the same lr as the first
        existing group. Useful when the GUI wants to surface a separate
        slider for head lr.
    """

    k: int = 2
    beta: tuple[float, ...] | None = None
    share_backbone: bool = True
    add_head_param_group: bool = True

    # Rewriter protocol
    name: str = field(init=False)
    required_preconditions: frozenset[str] = field(init=False)
    provided_postconditions: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"MTPRewriter.k must be ≥ 1, got {self.k}")
        if self.beta is not None and len(self.beta) != self.k:
            raise ValueError(
                f"MTPRewriter.beta length ({len(self.beta)}) must equal "
                f"k ({self.k})"
            )
        # Init the Rewriter-protocol fields.
        object.__setattr__(self, "name", f"MTPRewriter(k={self.k})")
        object.__setattr__(
            self, "required_preconditions", frozenset({"single_head"}),
        )
        object.__setattr__(
            self, "provided_postconditions",
            frozenset() if self.k == 1 else frozenset({"mtp_k_heads"}),
        )

    def __call__(self, spec: ModelBuildSpec) -> ModelBuildSpec:
        if self.k == 1:
            # No-op fast path. Don't touch the loss either — caller asked
            # for K=1 explicitly.
            return spec

        if spec.loss.kind is not LossKind.CROSS_ENTROPY:
            raise LossRewriteError(
                f"MTPRewriter requires LossKind.CROSS_ENTROPY input "
                f"(got {spec.loss.kind!r}) — automatic rewrite to "
                "mtp_weighted only safe from a pure CE base"
            )

        head = _find_head_node(spec.graph)
        original_loss_name = spec.loss.head_outputs[0]

        # ---- rewrite graph: rename head -> head_0; add head_1..k-1 ----
        new_nodes: list[BrickNode] = []
        existing_names = {n.name for n in spec.graph.nodes}
        existing_names.discard(head.name)

        renamed_head = BrickNode(
            kind=head.kind,
            name=f"{head.name}_0",
            params=dict(head.params),
            module=head.module,
        )
        head_names: list[str] = [renamed_head.name]
        for n in spec.graph.nodes:
            if n.name == head.name:
                new_nodes.append(renamed_head)
            else:
                new_nodes.append(n)

        for i in range(1, self.k):
            clone_name = f"{head.name}_{i}"
            # Ensure unique
            if clone_name in existing_names:
                clone_name = f"{head.name}_mtp_{i}"
            existing_names.add(clone_name)
            new_nodes.append(
                BrickNode(
                    kind=head.kind,
                    name=clone_name,
                    params=dict(head.params),
                    module=None,  # weights re-init at build time
                )
            )
            head_names.append(clone_name)

        # ---- rewrite edges: every edge into old head_name -> head_0 ----
        producer_of_head = [
            p for p, c in spec.graph.edges if c == head.name
        ]
        new_edges: list[tuple[str, str]] = []
        for p, c in spec.graph.edges:
            if c == head.name:
                new_edges.append((p, renamed_head.name))
                # Also wire the producer into every new head clone.
                for clone in head_names[1:]:
                    new_edges.append((p, clone))
            elif p == head.name:
                # Anything that consumed the old head now consumes head_0
                # (the "real" prediction); MTP heads are leaves that
                # only feed the loss, not subsequent bricks.
                new_edges.append((renamed_head.name, c))
            else:
                new_edges.append((p, c))

        # If the head had no producers (size-1 graph), wire the clones
        # directly into the head_0 slot — they'll consume the same input
        # the head_0 brick gets at runtime.
        if not producer_of_head and self.k > 1:
            for clone in head_names[1:]:
                new_edges.append((renamed_head.name, clone))

        new_graph = BrickGraph(nodes=tuple(new_nodes), edges=tuple(new_edges))

        # ---- rewrite loss: CE -> mtp_weighted -------------------------
        new_loss = mtp_weighted_loss(
            k=self.k,
            beta=self.beta,
            head_output_prefix=original_loss_name,
        )

        # ---- rewrite optim: optionally add head-only group ------------
        new_optim = spec.optim
        if self.add_head_param_group:
            base_lr = spec.optim.groups[0].lr
            head_matcher = f"regex:.*{original_loss_name}_\\d+$"
            head_group = ParamGroup(
                matcher=head_matcher,
                lr=base_lr,
                weight_decay=spec.optim.groups[0].weight_decay,
                betas=(
                    spec.optim.groups[0].betas
                    if spec.optim.kind in {OptimKind.ADAMW, OptimKind.MUON_ADAMW_HYBRID}
                    else None
                ),
                ns_steps=(
                    spec.optim.groups[0].ns_steps
                    if spec.optim.kind is OptimKind.MUON
                    else None
                ),
            )
            # Prepend so the head matcher wins over the catch-all "all".
            new_optim = OptimSpec(
                kind=spec.optim.kind,
                groups=(head_group, *spec.optim.groups),
                gradient_clip_norm=spec.optim.gradient_clip_norm,
                mixed_precision=spec.optim.mixed_precision,
            )

        return spec.replace(
            graph=new_graph, loss=new_loss, optim=new_optim,
        )


class LossRewriteError(RuntimeError):
    """Raised by MTPRewriter when the input loss can't be safely converted."""


# ---------------------------------------------------------------------------
# IFIMRewriter
# ---------------------------------------------------------------------------


_ATTENTION_KINDS: Final[frozenset[str]] = frozenset({
    "attention", "gated_attention", "gqa_sliding", "cca_attention",
    "mla", "mla_absorb", "mistral4_mla", "bailing_mla",
    "dsv4_attention", "nsa", "csa_hca",
})


@dataclass(frozen=True)
class IFIMRewriter:
    """Inverse Fisher Information Matrix shaping rewriter.

    Adds a virtual ``ifim_aux`` brick downstream of the (post-MTP if
    present) head and rewrites the loss to :func:`ifim_shaped_loss`
    with ``lambda_fim`` weighting.

    Pre-condition: ``"single_head"`` OR ``"mtp_k_heads"`` (works either
    way — when MTP rewrote first, IFIM attaches to every head clone via
    the loss aggregator; the graph stays simple and only one aux node
    appears).

    Post-condition: adds ``"ifim_added"`` to the state-token set.
    """

    lambda_fim: float = 0.1
    aux_node_name: str = "ifim_aux"

    # Rewriter protocol
    name: str = field(init=False)
    required_preconditions: frozenset[str] = field(init=False)
    provided_postconditions: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.lambda_fim < 0:
            raise ValueError(
                f"IFIMRewriter.lambda_fim must be ≥ 0, got {self.lambda_fim}"
            )
        if not self.aux_node_name.strip():
            raise ValueError("IFIMRewriter.aux_node_name must be non-empty")
        object.__setattr__(
            self, "name", f"IFIMRewriter(λ={self.lambda_fim})",
        )
        # Accept either single-head CE base or post-MTP K-head — runtime
        # picks the right loss aggregator.
        object.__setattr__(
            self, "required_preconditions", frozenset(),
        )
        object.__setattr__(
            self, "provided_postconditions", frozenset({"ifim_added"}),
        )

    def __call__(self, spec: ModelBuildSpec) -> ModelBuildSpec:
        if "ifim_added" in spec.state_tokens:
            raise IFIMCompositionError(
                "IFIMRewriter already applied to this spec; double-apply "
                "would compound the lambda_fim weight unsafely"
            )

        # Don't touch the underlying graph topology — IFIM is a
        # loss-side rewrite. We add the aux node so downstream consumers
        # (memory_report, GUI) can see it in the brick list. Edges
        # connect it as a leaf consumer of the existing head(s).
        existing_names = {n.name for n in spec.graph.nodes}
        aux_name = self.aux_node_name
        suffix = 0
        while aux_name in existing_names:
            suffix += 1
            aux_name = f"{self.aux_node_name}_{suffix}"

        # Pick the head(s) to feed: prefer the loss spec's declared
        # head_outputs; fall back to the last node when none match.
        head_names = [
            n.name for n in spec.graph.nodes
            if n.name in spec.loss.head_outputs
        ]
        if not head_names:
            head_names = [spec.graph.nodes[-1].name] if spec.graph.nodes else []

        aux_node = BrickNode(
            kind="mlp",  # safe stand-in shape contract; real impl is loss-side
            name=aux_name,
            params={"is_ifim_aux": True, "lambda_fim": self.lambda_fim},
            module=None,
        )
        new_nodes = (*spec.graph.nodes, aux_node)
        new_edges = (
            *spec.graph.edges,
            *[(h, aux_name) for h in head_names],
        )
        new_graph = BrickGraph(nodes=new_nodes, edges=new_edges)

        # Rewrite loss: preserve original head_outputs; switch to IFIM kind.
        new_loss = ifim_shaped_loss(
            lambda_fim=self.lambda_fim,
            head_output_name=spec.loss.head_outputs[0],
        )
        # If the input was MTP-weighted, keep the multi-head structure:
        # promote IFIM into a "MTP-with-IFIM" combined spec by reusing
        # the MTP head_outputs as the IFIM heads.
        if spec.loss.kind is LossKind.MTP_WEIGHTED:
            new_loss = LossSpec(
                kind=LossKind.IFIM_SHAPED,
                params={"lambda_fim": float(self.lambda_fim)},
                head_outputs=spec.loss.head_outputs,
                label_source="next_token",
            )

        return spec.replace(graph=new_graph, loss=new_loss)


class IFIMCompositionError(RuntimeError):
    """Raised by :class:`IFIMRewriter` when applied twice to the same spec."""


# ---------------------------------------------------------------------------
# MHCRewriter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MHCRewriter:
    """Multi-Head-Copy attention bias rewriter.

    For every attention brick in the graph, materialises ``num_copies-1``
    additional weight-shared copies and adds an auxiliary
    :func:`mhc_attn_bias_loss` term with ``lambda_mhc`` weighting.

    The copies are NAMED ``<orig>_mhc_<i>`` and reuse the original
    module reference (weight sharing happens at build time — Stage E
    wires them into one nn.Linear with cached forward).

    Post-condition: adds ``"mhc_copies_added"``.
    """

    num_copies: int = 2
    lambda_mhc: float = 0.05

    # Rewriter protocol
    name: str = field(init=False)
    required_preconditions: frozenset[str] = field(init=False)
    provided_postconditions: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_copies < 1:
            raise ValueError(
                f"MHCRewriter.num_copies must be ≥ 1, got {self.num_copies}"
            )
        if self.lambda_mhc < 0:
            raise ValueError(
                f"MHCRewriter.lambda_mhc must be ≥ 0, got {self.lambda_mhc}"
            )
        object.__setattr__(
            self, "name",
            f"MHCRewriter(copies={self.num_copies}, λ={self.lambda_mhc})",
        )
        object.__setattr__(
            self, "required_preconditions", frozenset(),
        )
        object.__setattr__(
            self, "provided_postconditions",
            frozenset() if self.num_copies == 1
            else frozenset({"mhc_copies_added"}),
        )

    def __call__(self, spec: ModelBuildSpec) -> ModelBuildSpec:
        if self.num_copies == 1:
            return spec
        if "mhc_copies_added" in spec.state_tokens:
            raise MHCCompositionError(
                "MHCRewriter already applied to this spec; double-apply "
                "would multiply attention copies geometrically"
            )

        new_nodes: list[BrickNode] = list(spec.graph.nodes)
        new_edges: list[tuple[str, str]] = list(spec.graph.edges)
        existing_names = {n.name for n in spec.graph.nodes}

        attention_nodes = [
            n for n in spec.graph.nodes if n.kind in _ATTENTION_KINDS
        ]
        if not attention_nodes:
            # No-op: graph has no attention bricks; advertise the
            # postcondition anyway so chained rewriters can rely on it.
            return spec.replace(graph=spec.graph)

        for attn in attention_nodes:
            for i in range(1, self.num_copies):
                clone_name = f"{attn.name}_mhc_{i}"
                suffix = 0
                while clone_name in existing_names:
                    suffix += 1
                    clone_name = f"{attn.name}_mhc_{i}_{suffix}"
                existing_names.add(clone_name)
                clone = BrickNode(
                    kind=attn.kind,
                    name=clone_name,
                    params={**attn.params, "is_mhc_copy": True,
                            "mhc_source": attn.name},
                    module=attn.module,   # weight-shared at build time
                )
                new_nodes.append(clone)
                # Wire identically: every producer of attn also feeds
                # the copy (so it sees the same input). The output of
                # the copy is consumed by the auxiliary loss only.
                for p, c in spec.graph.edges:
                    if c == attn.name:
                        new_edges.append((p, clone_name))

        new_graph = BrickGraph(nodes=tuple(new_nodes), edges=tuple(new_edges))

        # Switch loss to MHC_ATTN_BIAS (keep head_outputs from previous loss).
        new_loss = LossSpec(
            kind=LossKind.MHC_ATTN_BIAS,
            params={"lambda_mhc": float(self.lambda_mhc)},
            head_outputs=spec.loss.head_outputs,
            label_source=spec.loss.label_source,
        )
        return spec.replace(graph=new_graph, loss=new_loss)


class MHCCompositionError(RuntimeError):
    """Raised by :class:`MHCRewriter` when applied twice to the same spec."""


__all__ = [
    "HeadDetectionError",
    "IFIMCompositionError",
    "IFIMRewriter",
    "LossRewriteError",
    "MHCCompositionError",
    "MHCRewriter",
    "MTPRewriter",
]
