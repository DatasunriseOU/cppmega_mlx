"""ROI 3.7 — Head-to-head benchmark receipt schema for GDN/KDA paths.

Mirrors the JSON shape used by ``reports/raw/cppmega_1b_path_matrix_cells/``
so the v4 benchmark plugs straight into the existing matrix HTML render.
The full benchmark runner (training cell via ``scripts/m04_train_step.py`` on
``data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet``) lands
when Paths B/C/D/E for GDN/KDA become non-fallback; this module ships the
schema + dispatch-aware single-cell measurement now so the harness can hook
in incrementally.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx

from cppmega_v4._tilelang.kda_paths import kda_recurrent_dispatch
from cppmega_v4._tilelang.linear_attention_paths import gated_delta_recurrent_dispatch

Block = Literal["gdn", "kda"]
Path_ = Literal["path_a", "path_b", "path_c", "path_d", "path_e"]


@dataclass(frozen=True)
class CellShape:
    block: Block
    path: Path_
    batch: int
    seq_len: int
    num_heads: int
    head_dim_k: int
    head_dim_v: int
    num_v_heads: int | None = None
    dtype: str = "float32"


@dataclass
class CellReceipt:
    """One (block, path, shape, dtype) measurement cell."""

    cell_shape: CellShape
    fwd_seconds: float
    backend_available: bool
    backend_reason: str
    output_shape: tuple[int, ...]
    output_dtype: str

    def to_json(self) -> dict[str, object]:
        return {
            "cell_shape": asdict(self.cell_shape),
            "fwd_seconds": self.fwd_seconds,
            "backend_available": self.backend_available,
            "backend_reason": self.backend_reason,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
        }


def measure_cell(shape: CellShape) -> CellReceipt:
    """Run a single forward and emit a receipt."""
    from cppmega_v4._tilelang.kda_paths import kda_path_statuses
    from cppmega_v4._tilelang.linear_attention_paths import linear_attention_path_statuses

    statuses = (
        linear_attention_path_statuses() if shape.block == "gdn" else kda_path_statuses()
    )
    st = statuses[shape.path]
    if shape.block == "gdn":
        q = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        k = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        v = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_v))
        beta = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads))
        g = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads)) * 0.1
        t0 = time.perf_counter()
        o, _ = gated_delta_recurrent_dispatch(q, k, v, beta, g)
        mx.eval(o)
        elapsed = time.perf_counter() - t0
    else:  # kda
        hv = shape.num_v_heads if shape.num_v_heads is not None else shape.num_heads
        q = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        k = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        v = mx.random.normal((shape.batch, shape.seq_len, hv, shape.head_dim_v))
        g = mx.random.normal((shape.batch, shape.seq_len, hv, shape.head_dim_k)) * 0.05
        beta = mx.random.normal((shape.batch, shape.seq_len, hv))
        t0 = time.perf_counter()
        o, _ = kda_recurrent_dispatch(q, k, v, g, beta)
        mx.eval(o)
        elapsed = time.perf_counter() - t0

    return CellReceipt(
        cell_shape=shape,
        fwd_seconds=elapsed,
        backend_available=st.available,
        backend_reason=st.reason,
        output_shape=tuple(o.shape),
        output_dtype=str(o.dtype),
    )


def write_receipt(receipt: CellReceipt, out_dir: Path) -> Path:
    """Drop the receipt JSON into ``out_dir/<block>_<path>.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{receipt.cell_shape.block}_{receipt.cell_shape.path}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(receipt.to_json(), indent=2))
    return out_path


__all__ = ["Block", "CellReceipt", "CellShape", "Path_", "measure_cell", "write_receipt"]
