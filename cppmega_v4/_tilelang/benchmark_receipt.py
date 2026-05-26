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
import os
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
    requested_path: Path_ | None = None
    measured_path: Path_ | None = None
    fallback_used: bool = False
    dispatch_error: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "cell_shape": asdict(self.cell_shape),
            "fwd_seconds": self.fwd_seconds,
            "backend_available": self.backend_available,
            "backend_reason": self.backend_reason,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "requested_path": self.requested_path or self.cell_shape.path,
            "measured_path": self.measured_path or self.cell_shape.path,
            "fallback_used": self.fallback_used,
            "dispatch_error": self.dispatch_error,
        }


def _path_e_metal_kernel_supported(shape: CellShape) -> bool:
    return shape.head_dim_k % 32 == 0 and shape.head_dim_v % 4 == 0


def measure_cell(shape: CellShape) -> CellReceipt:
    """Run a single forward and emit a receipt."""
    from cppmega_v4._tilelang.kda_paths import ENV_VAR as KDA_ENV
    from cppmega_v4._tilelang.kda_paths import kda_path_statuses
    from cppmega_v4._tilelang.linear_attention_paths import ENV_VAR as GDN_ENV
    from cppmega_v4._tilelang.linear_attention_paths import linear_attention_path_statuses

    statuses = (
        linear_attention_path_statuses() if shape.block == "gdn" else kda_path_statuses()
    )
    st = statuses[shape.path]
    backend_available = bool(st.available)
    backend_reason = st.reason
    if shape.path == "path_e" and not _path_e_metal_kernel_supported(shape):
        backend_available = False
        backend_reason = (
            f"{st.reason}; benchmark shape uses Path E ops fallback, not the "
            "vendored Metal kernel (requires head_dim_k%32==0 and head_dim_v%4==0)"
        )

    env_var = GDN_ENV if shape.block == "gdn" else KDA_ENV
    prev = os.environ.get(env_var)

    if shape.block == "gdn":
        q = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        k = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        v = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_v))
        beta = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads))
        g = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads)) * 0.1

        def run(path: Path_, *, allow_fallback: bool):
            return gated_delta_recurrent_dispatch(
                q, k, v, beta, g, path=path, allow_fallback=allow_fallback,
            )

    else:  # kda
        hv = shape.num_v_heads if shape.num_v_heads is not None else shape.num_heads
        q = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        k = mx.random.normal((shape.batch, shape.seq_len, shape.num_heads, shape.head_dim_k))
        v = mx.random.normal((shape.batch, shape.seq_len, hv, shape.head_dim_v))
        g = mx.random.normal((shape.batch, shape.seq_len, hv, shape.head_dim_k)) * 0.05
        beta = mx.random.normal((shape.batch, shape.seq_len, hv))

        def run(path: Path_, *, allow_fallback: bool):
            return kda_recurrent_dispatch(
                q, k, v, g, beta, path=path, allow_fallback=allow_fallback,
            )

    measured_path: Path_ = shape.path
    fallback_used = False
    dispatch_error: str | None = None
    os.environ[env_var] = shape.path
    try:
        try:
            if shape.path != "path_a" and not backend_available:
                measured_path = "path_a"
                fallback_used = True
                t0 = time.perf_counter()
                o, _ = run("path_a", allow_fallback=False)
            else:
                t0 = time.perf_counter()
                o, _ = run(shape.path, allow_fallback=False)
            mx.eval(o)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            if shape.path == "path_a":
                raise
            dispatch_error = f"{exc.__class__.__name__}: {exc}"
            backend_available = False
            backend_reason = (
                f"{st.reason}; dispatch failed with fallback disabled: "
                f"{dispatch_error}"
            )
            measured_path = "path_a"
            fallback_used = True
            t0 = time.perf_counter()
            o, _ = run("path_a", allow_fallback=False)
            mx.eval(o)
            elapsed = time.perf_counter() - t0
    finally:
        if prev is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = prev

    return CellReceipt(
        cell_shape=shape,
        fwd_seconds=elapsed,
        backend_available=backend_available,
        backend_reason=backend_reason,
        output_shape=tuple(o.shape),
        output_dtype=str(o.dtype),
        requested_path=shape.path,
        measured_path=measured_path,
        fallback_used=fallback_used,
        dispatch_error=dispatch_error,
    )


def write_receipt(receipt: CellReceipt, out_dir: Path) -> Path:
    """Drop the receipt JSON into ``out_dir/<block>_<path>.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{receipt.cell_shape.block}_{receipt.cell_shape.path}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(receipt.to_json(), indent=2))
    return out_path


__all__ = ["Block", "CellReceipt", "CellShape", "Path_", "measure_cell", "write_receipt"]
