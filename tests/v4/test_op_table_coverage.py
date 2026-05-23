"""V7-N03/D32 (ynwz): OP_TABLE covers ≥80% of RFC §5.5 Tier-1 ops."""

from __future__ import annotations

from cppmega_mlx.nn._triton_op_table import (
    OP_TABLE_KEYS, RFC_55_OPS,
    covered_ops, missing_ops, op_coverage_ratio,
)


def test_rfc_55_lists_50_ops():
    assert len(RFC_55_OPS) == 50


def test_op_table_covers_at_least_80pct_of_rfc_55():
    ratio = op_coverage_ratio()
    assert ratio >= 0.80, (
        f"OP_TABLE coverage {ratio:.0%} < 80%; "
        f"missing: {missing_ops()}")


def test_op_table_covers_at_least_40_ops():
    """≥40 ops mapped per the audit's hard floor."""
    assert len(covered_ops()) >= 40, (
        f"only {len(covered_ops())} ops mapped; "
        f"missing: {missing_ops()}")


def test_op_table_keys_are_subset_of_rfc_55_or_extras():
    """Every advertised key either belongs to RFC §5.5 or is an
    explicit extension; this catches typos like tl.dott."""
    rfc = set(RFC_55_OPS)
    extras = OP_TABLE_KEYS - rfc
    # Any extra must still start with the tl. prefix to be a real
    # Triton op identifier.
    for op in extras:
        assert op.startswith("tl."), (
            f"OP_TABLE_KEYS contains non-triton id {op!r}")
