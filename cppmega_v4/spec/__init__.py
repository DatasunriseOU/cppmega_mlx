"""Visual-Builder spec layer over V4 bricks.

See ``VisualBuilderSpec.md`` (repo root) for the full design.

Stage A surface (this commit):
  - shape_contract.ShapeExpr / BrickShapeContract / contract_for /
    register_contract / registered_kinds / ResolveError
"""

from __future__ import annotations

from cppmega_v4.spec.adapters import (
    ADAPTER_RULES,
    AdapterRule,
    AdapterSuggestion,
    insert_adapter_chain,
    suggest_adapter_chain,
)
from cppmega_v4.spec.memory_report import (
    BrickMemoryRow,
    MemoryReport,
    RegionMemoryRow,
    estimate_memory,
)
from cppmega_v4.spec.resolver import (
    DiagnosticSeverity,
    ResolvedBrickGraph,
    ResolvedEdge,
    ShapeDiagnostic,
    resolve_shapes,
)
from cppmega_v4.spec.shape_contract import (
    BrickShapeContract,
    ResolveError,
    ShapeExpr,
    contract_for,
    register_contract,
    registered_kinds,
)

__all__ = [
    "ADAPTER_RULES",
    "AdapterRule",
    "AdapterSuggestion",
    "BrickMemoryRow",
    "BrickShapeContract",
    "DiagnosticSeverity",
    "MemoryReport",
    "RegionMemoryRow",
    "ResolveError",
    "ResolvedBrickGraph",
    "ResolvedEdge",
    "ShapeDiagnostic",
    "ShapeExpr",
    "contract_for",
    "estimate_memory",
    "insert_adapter_chain",
    "register_contract",
    "registered_kinds",
    "resolve_shapes",
    "suggest_adapter_chain",
]
