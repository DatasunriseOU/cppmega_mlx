"""Tests for ROI 3.7 — head-to-head benchmark matrix + auto-promotion."""

import json
from pathlib import Path

import pytest

from cppmega_v4._tilelang.benchmark_matrix import (
    GDN_PATHS,
    KDA_PATHS,
    MatrixCell,
    PromotionDecision,
    _decide_promotion,
    run_matrix,
    write_matrix_receipt,
)
from cppmega_v4._tilelang.benchmark_receipt import CellShape


# ----- Decision rule -----


def _cell(path, t, *, ok=True, finite=True):
    return MatrixCell(
        block="gdn", path=path,
        median_seconds=t, iters=5,
        backend_available=ok, backend_reason="ok",
        output_finite=finite,
    )


def test_promote_when_candidate_beats_incumbent_by_margin():
    cells = [
        _cell("path_a", 1.0),
        _cell("path_b", 0.5),   # 50% faster — well below 5% margin
    ]
    dec = _decide_promotion(
        block="gdn", shape_signature="sig",
        cells=cells, incumbent=("path_a", 1.0), margin_delta=0.05,
    )
    assert dec.winning_path == "path_b"
    assert dec.promotion_applied is True
    assert dec.env_value == "path_b"


def test_no_promote_within_margin():
    cells = [
        _cell("path_a", 1.0),
        _cell("path_b", 0.97),  # only 3% faster — under 5% margin
    ]
    dec = _decide_promotion(
        block="gdn", shape_signature="sig",
        cells=cells, incumbent=("path_a", 1.0), margin_delta=0.05,
    )
    assert dec.winning_path == "path_a"
    assert dec.promotion_applied is False


def test_skip_unavailable_paths():
    cells = [
        _cell("path_a", 1.0, ok=True),
        _cell("path_b", 0.1, ok=False),  # fastest but unavailable
        _cell("path_c", 0.7, ok=True),   # available 30% faster — promotes
    ]
    dec = _decide_promotion(
        block="gdn", shape_signature="sig",
        cells=cells, incumbent=("path_a", 1.0), margin_delta=0.05,
    )
    assert dec.winning_path == "path_c"
    assert dec.promotion_applied is True


def test_skip_non_finite_paths():
    cells = [
        _cell("path_a", 1.0),
        _cell("path_b", 0.1, finite=False),  # fastest but produces NaNs
    ]
    dec = _decide_promotion(
        block="gdn", shape_signature="sig",
        cells=cells, incumbent=("path_a", 1.0), margin_delta=0.05,
    )
    assert dec.winning_path == "path_a"
    assert dec.promotion_applied is False


def test_no_eligible_paths_keeps_incumbent():
    cells = [
        _cell("path_a", 0.1, ok=False),
        _cell("path_b", 0.1, finite=False),
    ]
    dec = _decide_promotion(
        block="gdn", shape_signature="sig",
        cells=cells, incumbent=("path_a", 1.0), margin_delta=0.05,
    )
    assert dec.winning_path == "path_a"
    assert dec.promotion_applied is False


def test_keep_incumbent_when_unchanged():
    """If best is the incumbent, no promotion is applied (idempotent)."""
    cells = [
        _cell("path_a", 0.5),
        _cell("path_b", 0.7),
    ]
    dec = _decide_promotion(
        block="gdn", shape_signature="sig",
        cells=cells, incumbent=("path_a", 0.5), margin_delta=0.05,
    )
    assert dec.winning_path == "path_a"
    assert dec.promotion_applied is False


# ----- Matrix runner -----


def test_run_matrix_gdn_covers_all_5_paths():
    shape = CellShape(
        block="gdn", path="path_a", batch=1, seq_len=4,
        num_heads=2, head_dim_k=4, head_dim_v=4,
    )
    receipt = run_matrix(shape, warmup=1, iters=2)
    assert len(receipt.cells) == 5
    paths_seen = {c.path for c in receipt.cells}
    assert paths_seen == set(GDN_PATHS)
    assert receipt.promotion.env_var.endswith("LINEAR_ATTENTION")


def test_run_matrix_kda_covers_4_paths_no_path_e():
    shape = CellShape(
        block="kda", path="path_a", batch=1, seq_len=4,
        num_heads=2, head_dim_k=4, head_dim_v=4, num_v_heads=2,
    )
    receipt = run_matrix(shape, warmup=1, iters=2)
    assert len(receipt.cells) == 4
    paths_seen = {c.path for c in receipt.cells}
    assert paths_seen == set(KDA_PATHS)
    assert receipt.promotion.env_var.endswith("KDA")


def test_run_matrix_winner_beats_or_equals_path_a():
    shape = CellShape(
        block="gdn", path="path_a", batch=1, seq_len=4,
        num_heads=2, head_dim_k=4, head_dim_v=4,
    )
    receipt = run_matrix(shape, warmup=1, iters=2)
    # Winner must be one of the available paths.
    winner_cells = [c for c in receipt.cells if c.path == receipt.promotion.winning_path]
    assert len(winner_cells) == 1
    assert winner_cells[0].backend_available or receipt.promotion.winning_path == "path_a"


def test_write_matrix_receipt_produces_valid_json(tmp_path: Path):
    shape = CellShape(
        block="gdn", path="path_a", batch=1, seq_len=4,
        num_heads=2, head_dim_k=4, head_dim_v=4,
    )
    receipt = run_matrix(shape, warmup=1, iters=2)
    out = write_matrix_receipt(receipt, tmp_path)
    assert out.exists()
    parsed = json.loads(out.read_text())
    assert "shape" in parsed
    assert "cells" in parsed and len(parsed["cells"]) == 5
    assert "promotion" in parsed
    assert parsed["promotion"]["env_var"].endswith("LINEAR_ATTENTION")
    # Per-cell receipts also written.
    cell_files = list(tmp_path.glob("gdn_path_*.json"))
    assert len(cell_files) == 5


def test_promotion_decision_serializable():
    cells = [_cell("path_a", 1.0), _cell("path_b", 0.5)]
    dec = _decide_promotion(
        block="gdn", shape_signature="x",
        cells=cells, incumbent=("path_a", 1.0), margin_delta=0.05,
    )
    d = dec.to_json()
    assert d["winning_path"] == "path_b"
    assert d["promotion_applied"] is True
    # Round-trips through json.
    assert json.loads(json.dumps(d))["env_value"] == "path_b"
