"""Shape resolver — walks a BrickGraph and assigns concrete shapes to
every edge, surfacing mismatches as diagnostics (lenient mode) or
raising (strict mode).

This is the layer the GUI calls on every graph mutation: drop a brick,
draw an edge — resolver runs in <1 ms and tells you whether the edge is
green (shapes match), yellow (need an auto-adapter — see Stage C) or
red (incompatible at this dim_env).

Stage B scope (this commit):
  - Detect shape-match / shape-mismatch on every edge under a given
    ``dim_env``.
  - Classify mismatches into structured :class:`ShapeDiagnostic`
    records with severity, location, expected vs actual shape.
  - Honour ``opaque_shape=True`` contracts (skip per-byte audit;
    require only that the side-channel ``B/S/H`` invariant holds).
  - Produce :class:`ResolvedBrickGraph` — the BrickGraph plus
    per-edge resolved shapes plus the diagnostics list.
  - ``strict=True`` → raises :class:`ResolveError` on the first error
    diagnostic. ``strict=False`` → returns the graph with diagnostics
    list populated (GUI consumes this).

Stage C will plug in the adapter library — when a mismatch is "almost
matched" (same dim count, only layout differs), the resolver will
auto-insert a reshape / permute brick instead of emitting a diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode
from cppmega_v4.spec.shape_contract import (
    BrickShapeContract,
    ResolveError,
    contract_for,
)


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ShapeDiagnostic:
    """One issue surfaced by :func:`resolve_shapes`.

    Fields:
      severity: INFO | WARNING | ERROR. GUI colour-codes accordingly.
      producer / consumer: brick names. None when the diagnostic refers
        to a single node (e.g. missing side-channel input).
      message: human-readable explanation. Includes the resolved shape
        tuples on both sides so the user sees the actual numbers, not
        just the symbolic mismatch.
      suggested_fix: short hint about which adapter / param change could
        resolve it. Stage C will fill this in for concrete patterns.
    """

    severity: DiagnosticSeverity
    message: str
    producer: str | None = None
    consumer: str | None = None
    suggested_fix: str | None = None


@dataclass(frozen=True)
class ResolvedEdge:
    """One edge in a ResolvedBrickGraph.

    For matched edges: ``shape`` is the concrete tuple both sides agree
    on, ``producer_shape == consumer_shape == shape``.

    For mismatched edges (lenient mode): ``producer_shape`` and
    ``consumer_shape`` differ; ``shape`` falls back to producer_shape
    so downstream consumers can still continue.
    """

    producer: str
    consumer: str
    producer_shape: tuple[int, ...]
    consumer_shape: tuple[int, ...]
    matched: bool

    @property
    def shape(self) -> tuple[int, ...]:
        return self.producer_shape


@dataclass(frozen=True)
class ResolvedBrickGraph:
    """A BrickGraph with shape information resolved per edge."""

    original: BrickGraph
    dim_env: Mapping[str, int]
    edges: tuple[ResolvedEdge, ...]
    diagnostics: tuple[ShapeDiagnostic, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        return any(
            d.severity is DiagnosticSeverity.ERROR for d in self.diagnostics
        )

    @property
    def errors(self) -> tuple[ShapeDiagnostic, ...]:
        return tuple(
            d for d in self.diagnostics
            if d.severity is DiagnosticSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[ShapeDiagnostic, ...]:
        return tuple(
            d for d in self.diagnostics
            if d.severity is DiagnosticSeverity.WARNING
        )

    def edge(self, producer: str, consumer: str) -> ResolvedEdge:
        for e in self.edges:
            if e.producer == producer and e.consumer == consumer:
                return e
        raise KeyError(
            f"no resolved edge ({producer!r} -> {consumer!r}) "
            f"in graph with {len(self.edges)} edges"
        )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


_DEFAULT_INPUT_PORT = "x"
_DEFAULT_OUTPUT_PORT = "y"


def _resolve_node_output(
    node: BrickNode,
    contract: BrickShapeContract,
    dim_env: Mapping[str, int],
) -> tuple[int, ...]:
    return contract.outputs[_DEFAULT_OUTPUT_PORT].resolve(dim_env)


def _resolve_node_input(
    node: BrickNode,
    contract: BrickShapeContract,
    dim_env: Mapping[str, int],
) -> tuple[int, ...]:
    return contract.inputs[_DEFAULT_INPUT_PORT].resolve(dim_env)


def _check_side_channels(
    node: BrickNode,
    contract: BrickShapeContract,
    available_channels: frozenset[str],
) -> list[ShapeDiagnostic]:
    diags: list[ShapeDiagnostic] = []
    missing = contract.needs - available_channels
    for ch in sorted(missing):
        diags.append(
            ShapeDiagnostic(
                severity=DiagnosticSeverity.WARNING,
                message=(
                    f"brick {node.name!r} (kind={node.kind!r}) needs "
                    f"side-channel {ch!r}; caller must supply it"
                ),
                producer=None,
                consumer=node.name,
                suggested_fix=f"supply {ch!r} via the model's forward signature",
            )
        )
    return diags


def resolve_shapes(
    graph: BrickGraph,
    dim_env: Mapping[str, int],
    *,
    strict: bool = True,
    available_side_channels: frozenset[str] = frozenset(),
) -> ResolvedBrickGraph:
    """Resolve every edge in ``graph`` under ``dim_env``.

    Args:
      graph: the BrickGraph to verify.
      dim_env: concrete int values for every named dim the contracts
        reference. Use :func:`cppmega_v4.spec.suggest_dim_env` (Stage E)
        for a sensible default, or build one from a preset.
      strict: when True (default), the first ERROR diagnostic raises
        :class:`ResolveError`. When False, all diagnostics are
        collected and returned in the ResolvedBrickGraph.
      available_side_channels: names of side-channel inputs (``doc_ids``,
        ``token_ids``, ``kv_cache``) the caller will provide at forward
        time. Bricks that ``need`` channels not in this set get a
        WARNING (never ERROR — the caller might still supply them).

    Returns: ResolvedBrickGraph with one ResolvedEdge per graph edge
    and (in lenient mode) any diagnostics encountered.
    """
    diagnostics: list[ShapeDiagnostic] = []
    resolved_edges: list[ResolvedEdge] = []

    # Pre-resolve every node's input and output shapes once, surfacing
    # contract-lookup failures and ResolveErrors as ERROR diagnostics.
    node_in: dict[str, tuple[int, ...]] = {}
    node_out: dict[str, tuple[int, ...]] = {}
    node_contracts: dict[str, BrickShapeContract] = {}

    for node in graph.nodes:
        try:
            contract = contract_for(node.kind)
        except KeyError as exc:
            d = ShapeDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=str(exc),
                producer=None,
                consumer=node.name,
                suggested_fix=(
                    "register a BrickShapeContract for this kind in "
                    "cppmega_v4/spec/shape_contract.py"
                ),
            )
            diagnostics.append(d)
            if strict:
                raise ResolveError(d.message) from exc
            continue
        node_contracts[node.name] = contract
        try:
            node_in[node.name] = _resolve_node_input(node, contract, dim_env)
            node_out[node.name] = _resolve_node_output(node, contract, dim_env)
        except ResolveError as exc:
            d = ShapeDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                message=(
                    f"failed to resolve shape for brick {node.name!r} "
                    f"(kind={node.kind!r}): {exc}"
                ),
                producer=None,
                consumer=node.name,
                suggested_fix="add the missing dim to dim_env",
            )
            diagnostics.append(d)
            if strict:
                raise
            continue
        diagnostics.extend(
            _check_side_channels(node, contract, available_side_channels)
        )

    # Now walk edges, comparing producer.out with consumer.in.
    for producer_name, consumer_name in graph.edges:
        if producer_name not in node_out or consumer_name not in node_in:
            # We already emitted an ERROR for the failing node — skip
            # the edge to avoid cascading noise.
            continue
        p_shape = node_out[producer_name]
        c_shape = node_in[consumer_name]

        p_contract = node_contracts[producer_name]
        c_contract = node_contracts[consumer_name]
        opaque_either_side = (
            p_contract.opaque_shape or c_contract.opaque_shape
        )

        matched = p_shape == c_shape
        resolved_edges.append(
            ResolvedEdge(
                producer=producer_name,
                consumer=consumer_name,
                producer_shape=p_shape,
                consumer_shape=c_shape,
                matched=matched,
            )
        )

        if matched:
            continue

        # Opaque bricks declare "trust me, B/S/H are preserved". As long
        # as the rank matches we downgrade to WARNING — the resolver
        # can't introspect what the opaque brick actually returns.
        rank_match = len(p_shape) == len(c_shape)
        if opaque_either_side and rank_match:
            diagnostics.append(
                ShapeDiagnostic(
                    severity=DiagnosticSeverity.WARNING,
                    message=(
                        f"opaque-brick boundary {producer_name!r} -> "
                        f"{consumer_name!r}: declared shapes differ "
                        f"({p_shape} vs {c_shape}) but at least one side "
                        "is opaque — caller must verify at runtime"
                    ),
                    producer=producer_name,
                    consumer=consumer_name,
                    suggested_fix=None,
                )
            )
            continue

        d = ShapeDiagnostic(
            severity=DiagnosticSeverity.ERROR,
            message=(
                f"shape mismatch on edge {producer_name!r} -> "
                f"{consumer_name!r}: producer outputs {p_shape} "
                f"({p_contract.outputs[_DEFAULT_OUTPUT_PORT].dims}), "
                f"consumer expects {c_shape} "
                f"({c_contract.inputs[_DEFAULT_INPUT_PORT].dims})"
            ),
            producer=producer_name,
            consumer=consumer_name,
            suggested_fix=_suggest_fix(p_shape, c_shape),
        )
        diagnostics.append(d)
        if strict:
            raise ResolveError(d.message)

    return ResolvedBrickGraph(
        original=graph,
        dim_env=dict(dim_env),
        edges=tuple(resolved_edges),
        diagnostics=tuple(diagnostics),
    )


def _suggest_fix(
    p_shape: tuple[int, ...], c_shape: tuple[int, ...]
) -> str | None:
    """Heuristic adapter suggestion. Stage C will replace this with a
    proper :func:`suggest_adapters` lookup."""
    if len(p_shape) == len(c_shape):
        # Permutation guess: same multiset, different order
        if sorted(p_shape) == sorted(c_shape):
            return (
                "shapes are permutations of each other — try inserting a "
                "transpose / permute adapter (Stage C)"
            )
        # Same rank, last dim differs → linear projection
        if p_shape[:-1] == c_shape[:-1]:
            return (
                f"last-dim mismatch ({p_shape[-1]} != {c_shape[-1]}) — "
                f"insert a Linear({p_shape[-1]}, {c_shape[-1]}) bridge"
            )
    # Rank changes — likely heads merge/split
    if (
        len(p_shape) == 4 and len(c_shape) == 3
        and p_shape[0] == c_shape[0] and p_shape[2] == c_shape[1]
        and p_shape[1] * p_shape[3] == c_shape[2]
    ):
        return "merge_heads adapter: (B,nh,S,d) -> (B,S,nh*d)"
    if (
        len(p_shape) == 3 and len(c_shape) == 4
        and p_shape[0] == c_shape[0] and p_shape[1] == c_shape[2]
        and p_shape[2] == c_shape[1] * c_shape[3]
    ):
        return "split_heads adapter: (B,S,nh*d) -> (B,nh,S,d)"
    return None


__all__ = [
    "DiagnosticSeverity",
    "ResolvedBrickGraph",
    "ResolvedEdge",
    "ShapeDiagnostic",
    "resolve_shapes",
]
