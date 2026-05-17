"""ROI 3.7 — benchmark receipt schema tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega_v4._tilelang.benchmark_receipt import (
    CellShape,
    measure_cell,
    write_receipt,
)


def test_cell_shape_required_fields():
    s = CellShape(block="gdn", path="path_a", batch=1, seq_len=4, num_heads=2,
                  head_dim_k=4, head_dim_v=4)
    assert s.block == "gdn"
    assert s.path == "path_a"
    assert s.dtype == "float32"


def test_measure_cell_gdn_returns_receipt():
    shape = CellShape(
        block="gdn", path="path_a", batch=1, seq_len=4,
        num_heads=2, head_dim_k=4, head_dim_v=4,
    )
    receipt = measure_cell(shape)
    assert receipt.cell_shape.block == "gdn"
    assert receipt.fwd_seconds >= 0
    assert receipt.output_shape == (1, 4, 2, 4)
    assert receipt.backend_available is True  # path_a always available


def test_measure_cell_kda_returns_receipt():
    shape = CellShape(
        block="kda", path="path_a", batch=1, seq_len=3,
        num_heads=2, head_dim_k=4, head_dim_v=4, num_v_heads=4,
    )
    receipt = measure_cell(shape)
    assert receipt.cell_shape.block == "kda"
    assert receipt.output_shape == (1, 3, 4, 4)  # HV = 4
    assert receipt.backend_available is True


def test_measure_cell_deferred_path_marks_unavailable():
    """Path D (Triton frontend) is unavailable on Apple Silicon — stable fixture."""
    shape = CellShape(
        block="gdn", path="path_d", batch=1, seq_len=2,
        num_heads=2, head_dim_k=4, head_dim_v=4,
    )
    receipt = measure_cell(shape)
    assert receipt.backend_available is False
    assert receipt.backend_reason  # non-empty rationale


def test_write_receipt_produces_valid_json(tmp_path: Path):
    shape = CellShape(
        block="gdn", path="path_a", batch=1, seq_len=2,
        num_heads=2, head_dim_k=4, head_dim_v=4,
    )
    receipt = measure_cell(shape)
    out_path = write_receipt(receipt, tmp_path)
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    assert parsed["cell_shape"]["block"] == "gdn"
    assert parsed["cell_shape"]["path"] == "path_a"
    assert "fwd_seconds" in parsed
    assert "backend_available" in parsed
    assert "output_shape" in parsed
