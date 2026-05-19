"""Visual-Builder spec layer over V4 bricks.

See ``VisualBuilderSpec.md`` (repo root) for the full design.

Stage A surface (this commit):
  - shape_contract.ShapeExpr / BrickShapeContract / contract_for /
    register_contract / registered_kinds / ResolveError
"""

from __future__ import annotations

from cppmega_v4.spec.shape_contract import (
    BrickShapeContract,
    ResolveError,
    ShapeExpr,
    contract_for,
    register_contract,
    registered_kinds,
)

__all__ = [
    "BrickShapeContract",
    "ResolveError",
    "ShapeExpr",
    "contract_for",
    "register_contract",
    "registered_kinds",
]
