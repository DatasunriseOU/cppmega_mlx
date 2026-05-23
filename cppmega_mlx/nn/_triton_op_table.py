"""V7-N03 / V7-D32 (ynwz): Triton OP_TABLE coverage manifest.

The POC frontend at `tl_poc_review/poc/triton_frontend/op_table.py`
ships an `OP_TABLE` dict. This module mirrors the names cppmega_mlx
declares support for (RFC §5.5 Tier-1 surface) so a coverage test
can assert ≥ 80% mapping without importing the external POC.

Each entry is a string Triton op name; presence in OP_TABLE_KEYS
means the bridge claims it can lower the op. Coverage is the ratio
``|OP_TABLE_KEYS ∩ RFC_55_OPS| / |RFC_55_OPS|``.
"""

from __future__ import annotations


# RFC §5.5 Tier-1 surface: 50 Triton ops the frontend must cover.
RFC_55_OPS: tuple[str, ...] = (
    # Memory ops
    "tl.load", "tl.store", "tl.atomic_add", "tl.atomic_min",
    "tl.atomic_max", "tl.atomic_cas",
    # Layout / shape
    "tl.make_range", "tl.arange", "tl.expand_dims", "tl.reshape",
    "tl.broadcast_to", "tl.splat", "tl.trans",
    # Arithmetic
    "tl.dot", "tl.add", "tl.sub", "tl.mul", "tl.div", "tl.fdiv",
    "tl.maximum", "tl.minimum",
    # Reductions
    "tl.sum", "tl.max", "tl.min", "tl.argmax", "tl.argmin",
    # Math
    "tl.exp", "tl.log", "tl.sqrt", "tl.rsqrt", "tl.abs", "tl.sigmoid",
    "tl.softmax",
    # Casts
    "tl.cast", "tl.to",
    # Logic / select
    "tl.where", "tl.zeros", "tl.zeros_like", "tl.full",
    # Program / control
    "tl.program_id", "tl.num_programs",
    # Async copy / barriers
    "tl.async_copy", "tl.mbarrier",
    # TMA descriptors
    "tl.tma_descriptor", "tl.tma_load", "tl.tma_store",
    # Block / debug
    "tl.partial_barrier", "tl.print", "tl.device_assert",
    # Random
    "tl.rand",
)


# Ops cppmega_mlx's bridge actually claims to lower (matches the POC
# frontend's OP_TABLE keys at the time of writing). When the POC is
# upgraded these strings move accordingly; the coverage test pins
# the floor.
OP_TABLE_KEYS: frozenset[str] = frozenset({
    "tl.load", "tl.store", "tl.atomic_add", "tl.atomic_min",
    "tl.atomic_max",
    "tl.make_range", "tl.arange", "tl.expand_dims", "tl.reshape",
    "tl.broadcast_to", "tl.splat", "tl.trans",
    "tl.dot", "tl.add", "tl.sub", "tl.mul", "tl.div",
    "tl.maximum", "tl.minimum",
    "tl.sum", "tl.max", "tl.min", "tl.argmax", "tl.argmin",
    "tl.exp", "tl.log", "tl.sqrt", "tl.rsqrt", "tl.abs",
    "tl.sigmoid", "tl.softmax",
    "tl.cast", "tl.to",
    "tl.where", "tl.zeros", "tl.zeros_like", "tl.full",
    "tl.program_id", "tl.num_programs",
    "tl.async_copy", "tl.mbarrier",
    "tl.partial_barrier", "tl.print",
})


def op_coverage_ratio() -> float:
    """Fraction of RFC §5.5 ops covered by OP_TABLE_KEYS."""
    rfc = set(RFC_55_OPS)
    covered = rfc & OP_TABLE_KEYS
    return len(covered) / max(1, len(rfc))


def covered_ops() -> tuple[str, ...]:
    return tuple(sorted(set(RFC_55_OPS) & OP_TABLE_KEYS))


def missing_ops() -> tuple[str, ...]:
    return tuple(sorted(set(RFC_55_OPS) - OP_TABLE_KEYS))


__all__ = [
    "RFC_55_OPS", "OP_TABLE_KEYS", "op_coverage_ratio",
    "covered_ops", "missing_ops",
]
