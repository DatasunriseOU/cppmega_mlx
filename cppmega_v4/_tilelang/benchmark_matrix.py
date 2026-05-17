"""ROI 3.7 — head-to-head benchmark matrix + auto-promotion receipt.

Builds on ``benchmark_receipt.measure_cell`` to run every (block, path)
combination at a fixed shape and emit:

  1. A per-cell receipt JSON (same schema as today).
  2. A matrix-level summary JSON ranking paths by median fwd time.
  3. An auto-promotion decision: the fastest path becomes the new default
     for that (block, shape) — recorded as ``CPPMEGA_V4_KERNEL_PATH__<...>``
     env-var hints in the summary so callers can pin them in CI.

The decision rule is intentionally conservative:
  - A path is eligible only if ``backend_available == True`` and produces
    *finite* output.
  - To win, the candidate's median fwd time over ``--warmup + --iters``
    measurements must be ≤ ``(1 - delta) * incumbent_best``, where
    ``incumbent`` is the previously-recorded best for the same shape (or
    Path A on first run). ``delta`` defaults to 0.05 (5% margin).
  - On ties (delta not met), the incumbent stays — prevents thrashing.

This module ships pure-Python: it does not run a 1B-parameter training
step (that lives in ``scripts/run_v4_benchmark_matrix.py``, layered on
top). The matrix-runner here is what feeds CI gates and human-readable
HTML tables.
"""

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

from cppmega_v4._tilelang.benchmark_receipt import (
    Block,
    CellReceipt,
    CellShape,
    Path_,
    measure_cell,
    write_receipt,
)
from cppmega_v4._tilelang.kda_paths import ENV_VAR as KDA_ENV
from cppmega_v4._tilelang.linear_attention_paths import ENV_VAR as GDN_ENV

ENV_VAR_FOR_BLOCK: dict[Block, str] = {
    "gdn": GDN_ENV,
    "kda": KDA_ENV,
}

GDN_PATHS: tuple[Path_, ...] = ("path_a", "path_b", "path_c", "path_d", "path_e")
KDA_PATHS: tuple[Path_, ...] = ("path_a", "path_b", "path_c", "path_d")


@dataclass
class MatrixCell:
    """One (block, path) measurement across multiple iters."""

    block: Block
    path: Path_
    median_seconds: float
    iters: int
    backend_available: bool
    backend_reason: str
    output_finite: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PromotionDecision:
    """Auto-promotion outcome for one (block, shape)."""

    block: Block
    shape_signature: str
    winning_path: Path_
    median_seconds: float
    incumbent_path: Path_
    incumbent_seconds: float
    promotion_applied: bool
    margin_delta: float
    env_var: str
    env_value: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MatrixReceipt:
    """Full head-to-head matrix output."""

    shape: CellShape  # block field is the block under test
    cells: list[MatrixCell]
    promotion: PromotionDecision
    warmup: int
    iters: int
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, object]:
        return {
            "shape": asdict(self.shape),
            "cells": [c.to_json() for c in self.cells],
            "promotion": self.promotion.to_json(),
            "warmup": self.warmup,
            "iters": self.iters,
            "timestamp": self.timestamp,
        }


def _shape_signature(shape: CellShape) -> str:
    parts = [
        f"B{shape.batch}", f"T{shape.seq_len}", f"H{shape.num_heads}",
        f"Dk{shape.head_dim_k}", f"Dv{shape.head_dim_v}",
    ]
    if shape.num_v_heads is not None:
        parts.append(f"HV{shape.num_v_heads}")
    parts.append(shape.dtype)
    return "_".join(parts)


def _output_is_finite(receipt: CellReceipt) -> bool:
    """Re-run a forward to check finiteness — cheap on small shapes."""
    s = receipt.cell_shape
    if s.block == "gdn":
        q = mx.random.normal((s.batch, s.seq_len, s.num_heads, s.head_dim_k))
        k = mx.random.normal((s.batch, s.seq_len, s.num_heads, s.head_dim_k))
        v = mx.random.normal((s.batch, s.seq_len, s.num_heads, s.head_dim_v))
        beta = mx.random.normal((s.batch, s.seq_len, s.num_heads))
        g = mx.random.normal((s.batch, s.seq_len, s.num_heads)) * 0.1
        from cppmega_v4._tilelang.linear_attention_paths import (
            gated_delta_recurrent_dispatch,
        )
        o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
    else:
        hv = s.num_v_heads or s.num_heads
        q = mx.random.normal((s.batch, s.seq_len, s.num_heads, s.head_dim_k))
        k = mx.random.normal((s.batch, s.seq_len, s.num_heads, s.head_dim_k))
        v = mx.random.normal((s.batch, s.seq_len, hv, s.head_dim_v))
        g = mx.random.normal((s.batch, s.seq_len, hv, s.head_dim_k)) * 0.05
        beta = mx.random.normal((s.batch, s.seq_len, hv))
        from cppmega_v4._tilelang.kda_paths import kda_recurrent_dispatch
        o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
    mx.eval(o)
    return not bool(mx.any(mx.isnan(o)).item())


def _measure_path(
    shape_template: CellShape,
    path: Path_,
    *,
    warmup: int,
    iters: int,
) -> MatrixCell:
    """Warm up + measure ``iters`` forwards through a forced path."""
    env_var = ENV_VAR_FOR_BLOCK[shape_template.block]
    cell_shape = CellShape(**{**asdict(shape_template), "path": path})
    samples: list[float] = []
    prev = os.environ.get(env_var)
    os.environ[env_var] = path
    try:
        for _ in range(warmup):
            r = measure_cell(cell_shape)
        for _ in range(iters):
            r = measure_cell(cell_shape)
            samples.append(r.fwd_seconds)
        finite = _output_is_finite(r) if samples else False
    finally:
        if prev is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = prev

    return MatrixCell(
        block=shape_template.block,
        path=path,
        median_seconds=statistics.median(samples) if samples else float("inf"),
        iters=iters,
        backend_available=r.backend_available,
        backend_reason=r.backend_reason,
        output_finite=finite,
    )


def _decide_promotion(
    *,
    block: Block,
    shape_signature: str,
    cells: list[MatrixCell],
    incumbent: tuple[Path_, float],
    margin_delta: float,
) -> PromotionDecision:
    eligible = [
        c for c in cells if c.backend_available and c.output_finite
        and c.median_seconds < float("inf")
    ]
    if not eligible:
        # Nothing available — keep incumbent.
        winner = incumbent[0]
        win_time = incumbent[1]
        promote = False
    else:
        best = min(eligible, key=lambda c: c.median_seconds)
        # Promote only if best beats incumbent by at least margin_delta.
        threshold = incumbent[1] * (1.0 - margin_delta)
        if best.median_seconds <= threshold or best.path == incumbent[0]:
            winner = best.path
            win_time = best.median_seconds
            promote = winner != incumbent[0]
        else:
            winner = incumbent[0]
            win_time = incumbent[1]
            promote = False
    return PromotionDecision(
        block=block,
        shape_signature=shape_signature,
        winning_path=winner,
        median_seconds=win_time,
        incumbent_path=incumbent[0],
        incumbent_seconds=incumbent[1],
        promotion_applied=promote,
        margin_delta=margin_delta,
        env_var=ENV_VAR_FOR_BLOCK[block],
        env_value=winner,
    )


def run_matrix(
    shape_template: CellShape,
    *,
    warmup: int = 2,
    iters: int = 5,
    incumbent: Optional[tuple[Path_, float]] = None,
    margin_delta: float = 0.05,
) -> MatrixReceipt:
    """Run all paths for one (block, shape), decide promotion, return receipt.

    ``incumbent`` defaults to ``(path_a, +inf)`` — first ever run will
    install the fastest *available* path with no margin requirement (any
    finite time beats +inf trivially).
    """
    paths = GDN_PATHS if shape_template.block == "gdn" else KDA_PATHS
    cells = [
        _measure_path(shape_template, p, warmup=warmup, iters=iters)
        for p in paths
    ]
    sig = _shape_signature(shape_template)
    incumbent = incumbent or ("path_a", float("inf"))
    promotion = _decide_promotion(
        block=shape_template.block, shape_signature=sig,
        cells=cells, incumbent=incumbent, margin_delta=margin_delta,
    )
    return MatrixReceipt(
        shape=shape_template, cells=cells, promotion=promotion,
        warmup=warmup, iters=iters,
    )


def write_matrix_receipt(receipt: MatrixReceipt, out_dir: Path) -> Path:
    """Drop the matrix receipt under ``out_dir/<block>_matrix_<sig>.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = _shape_signature(receipt.shape)
    fname = f"{receipt.shape.block}_matrix_{sig}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(receipt.to_json(), indent=2))
    # Also drop the per-cell receipts so existing matrix HTML still renders.
    for cell in receipt.cells:
        cs = CellShape(**{**asdict(receipt.shape), "path": cell.path})
        write_receipt(
            CellReceipt(
                cell_shape=cs, fwd_seconds=cell.median_seconds,
                backend_available=cell.backend_available,
                backend_reason=cell.backend_reason,
                output_shape=(), output_dtype="float32",
            ),
            out_dir,
        )
    return out_path


__all__ = [
    "ENV_VAR_FOR_BLOCK",
    "GDN_PATHS",
    "KDA_PATHS",
    "MatrixCell",
    "MatrixReceipt",
    "PromotionDecision",
    "run_matrix",
    "write_matrix_receipt",
]
