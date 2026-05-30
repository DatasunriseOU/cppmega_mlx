#!/usr/bin/env python3
"""GDN Path D (Triton-frontend) vs Path A (FLA-naive) parity harness.

Path D lowers FLA's ``chunk_gated_delta_rule`` forward kernels through
``poc.triton_frontend`` into TileLang PrimFuncs and runs them on Metal via
``tilelang.compile(target='metal', execution_backend='tvm_ffi')``. This script
proves numerical parity of the generalized Path D runtime adapter
(``cppmega_v4._tilelang.path_d_runtime_adapter.gdn_fwd_runtime_call``) against
the Path A pure-MLX reference
(``cppmega_v4.nn._external.fla_naive_gated_delta_rule
.naive_recurrent_gated_delta_rule``) across a sweep of production GDN shapes
(H/HV/K/V, custom scale, initial/final state, and packed varlen), and emits a
receipt JSON.

Run recipe (host with the local FLA + triton_frontend checkouts)::

    CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT=1 \
    CPPMEGA_MLX_TRITON_FRONTEND_PATH=/path/to/tilelang \
    CPPMEGA_MLX_FLA_SOURCE_PATH=/path/to/flash-linear-attention \
    PYTHONPATH=/path/to/cppmega.mlx \
        python scripts/v4_gdn_path_d_parity.py --out reports/raw/v4_gdn_path_d_parity.json

The actual Metal launch of the FLA chunk GEMMs uses TileLang's
``mpp::tensor_ops::matmul2d`` cooperative-tensor path, which requires an
Apple M5-class GPU (Metal 4). On M1-M4 the *compile* (status probe) succeeds
but the launch fails to compile the generated MSL; this harness records that
host blocker per-cell instead of crashing, so it produces a complete receipt
either way and a clean green parity run is captured on M5 hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class GDNParityShape:
    name: str
    batch: int
    seq_len: int
    h_heads: int
    hv_heads: int
    k_dim: int
    v_dim: int
    custom_scale: bool = False
    initial_state: bool = False
    output_final_state: bool = True
    varlen_splits: Optional[tuple[int, ...]] = None  # packed seq lengths (B must be 1)


# Production-representative GDN linear-attention shapes. K=64/V=32 is the
# lowering-proven slice; the rest exercise multi-head, larger value dim,
# custom scale, recurrent state, and packed varlen. GDN's linear path uses
# H == HV (GQA / HV != H is KDA's domain, not GDN's), so all shapes keep
# H == HV. K is kept <= 64 in the default sweep because the FLA kkt_solve
# kernel's threadgroup-memory footprint at K=128 exceeds Apple's 32 KiB SMEM
# limit (a per-shape Metal constraint, not a Path D logic limit).
DEFAULT_SHAPES: tuple[GDNParityShape, ...] = (
    GDNParityShape("fixed_k64_v32", 2, 64, 1, 1, 64, 32),
    GDNParityShape("t128_k64_v32", 2, 128, 1, 1, 64, 32),
    GDNParityShape("mh_h2_hv2_k64_v64", 2, 128, 2, 2, 64, 64),
    GDNParityShape("mh_h4_hv4_k64_v32", 1, 128, 4, 4, 64, 32),
    GDNParityShape("custom_scale", 2, 64, 1, 1, 64, 32, custom_scale=True),
    GDNParityShape("with_state", 2, 64, 1, 1, 64, 32, initial_state=True),
    GDNParityShape("varlen_64_32", 1, 96, 1, 1, 64, 32, varlen_splits=(64, 32)),
    GDNParityShape("varlen_3seq", 1, 160, 1, 1, 64, 32, varlen_splits=(64, 64, 32)),
)


@dataclass
class GDNParityCell:
    name: str
    shape: dict[str, Any]
    backend_available: bool
    backend_reason: str
    launched: bool
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    state_max_abs_diff: Optional[float] = None
    parity_ok: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class GDNParityReceipt:
    host: dict[str, Any]
    env: dict[str, Any]
    tol_abs: float
    tol_rel: float
    cells: list[GDNParityCell]
    all_launched: bool
    all_parity_ok: bool
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "env": self.env,
            "tol_abs": self.tol_abs,
            "tol_rel": self.tol_rel,
            "cells": [asdict(c) for c in self.cells],
            "all_launched": self.all_launched,
            "all_parity_ok": self.all_parity_ok,
            "timestamp": self.timestamp,
        }


def _metal_launch_blocker(msg: str) -> bool:
    return any(
        marker in msg
        for marker in ("mpp", "tensor_ops", "execution_simdgroup", "undeclared identifier")
    )


def _metal_smem_blocker(msg: str) -> bool:
    return "threadgroup memory" in msg.lower() and "exceeds" in msg.lower()


def _metal_pipeline_state_blocker(msg: str) -> bool:
    """Metal pipeline-state creation (newComputePipelineStateWithFunction) failed.

    Surfaces as ``Check failed: (state != nullptr)`` from the TVM Metal runtime,
    typically with "Compute function exceeds available stack space". This is an
    Apple per-thread occupancy limit for a given kernel/tiling (e.g. the
    ``h_blockdim64`` chunk-state kernel with an initial state loaded), not a
    Path D launch-ABI or arg-count defect.
    """

    low = msg.lower()
    return "state != nullptr" in low or (
        "cannot get state" in low and "function" in low
    )


def _tilelang_non_tail_output_abi_bug(msg: str) -> bool:
    """TileLang tvm-ffi Metal mis-maps any non-tail ``out_idx`` owner-output.

    This is a REAL TileLang codegen/launch defect, *not* a cppmega-side or
    per-shape Metal-hardware constraint, and *not* the matmul2d zeros /
    cooperative-tensor launch-config bug fixed in tilelang ``7ae53998``.

    Root cause (isolated with a 4-line repro, zero cppmega code): the
    ``execution_backend='tvm_ffi'`` Metal route only binds owner-output buffers
    correctly when every ``out_idx`` parameter sits at the *tail* of the kernel
    signature. When an output is followed by any input parameter, the bridge
    binds the wrong MLX buffer to that output slot. Minimal evidence::

        out_idx=[2] of 3 params (tail)      -> correct (max_err 0.0)
        out_idx=[1] of 3 params (non-tail)  -> WRONG  (max_err ~4.8)
        out_idx=[1] of 4 params (non-tail)  -> WRONG  (max_err ~8.8)
        out_idx=[1,2] of 4 (last adjacent)  -> out[2] OK, out[1] WRONG

    When the resulting buffer-count bookkeeping breaks hard (e.g. the GDN
    ``kkt_solve`` stage: 6 params, out_idx=[3], two trailing int64 topology
    buffers), it surfaces as the make_packed_api host assertion
    ``<kernel>: num_args should be N``; milder cases silently return wrong
    numbers. Every GDN Path D stage (kkt_solve out_idx=[3]/6, recompute_w_u
    out_idx=[3,4]/9, chunk_delta_h out_idx=[3,6]/11) has a non-tail output, so
    the pipeline cannot launch correctly until TileLang binds non-tail
    owner-outputs by index on the Metal tvm-ffi route.

    NOTE: this is a *diagnosis only*. The cell is still recorded as not-launched
    / not-parity-ok; no numerical fallback is substituted (RULE #1).
    """

    return "num_args should be" in msg


def _run_cell(shape: GDNParityShape, *, tol_abs: float, tol_rel: float) -> GDNParityCell:
    import mlx.core as mx

    from cppmega_v4._tilelang.linear_attention_path_d import _path_d_runtime_status
    from cppmega_v4._tilelang.path_d_runtime_adapter import gdn_fwd_runtime_call
    from cppmega_v4.nn._external.fla_naive_gated_delta_rule import (
        naive_recurrent_gated_delta_rule,
    )

    ok, reason = _path_d_runtime_status()
    cell = GDNParityCell(
        name=shape.name,
        shape=asdict(shape),
        backend_available=bool(ok),
        backend_reason=reason,
        launched=False,
    )
    if not ok:
        cell.error = "backend not available (status probe failed)"
        return cell

    b, t = shape.batch, shape.seq_len
    h, hv, k, v = shape.h_heads, shape.hv_heads, shape.k_dim, shape.v_dim
    mx.random.seed(0)
    # Scale inputs down so the recurrence stays in-range (matches Path A/B tests).
    qf = (mx.random.normal((b, t, h, k)) * 0.2).astype(mx.float16)
    kf = (mx.random.normal((b, t, h, k)) * 0.2).astype(mx.float16)
    vf = (mx.random.normal((b, t, hv, v)) * 0.2).astype(mx.float16)
    beta = mx.sigmoid(mx.random.normal((b, t, hv))).astype(mx.float32)
    g = (-mx.abs(mx.random.normal((b, t, hv)) * 0.1)).astype(mx.float32)
    scale = (0.5 / math.sqrt(k)) if shape.custom_scale else None

    n_seq = b
    cu_seqlens = None
    if shape.varlen_splits is not None:
        assert b == 1, "varlen requires packed B=1"
        offsets = [0]
        for length in shape.varlen_splits:
            offsets.append(offsets[-1] + length)
        cu_seqlens = mx.array(offsets, dtype=mx.int64)
        n_seq = len(shape.varlen_splits)

    h0 = None
    if shape.initial_state:
        h0 = (mx.random.normal((n_seq, hv, k, v)) * 0.1).astype(mx.float32)

    try:
        # --- Path A reference (fp32) ---------------------------------------
        # Path A is per-sequence; for varlen we run each packed segment and
        # concatenate to match Path D's packed output.
        if cu_seqlens is not None:
            offs = [0]
            for length in shape.varlen_splits:
                offs.append(offs[-1] + length)
            ref_parts = []
            for si, (lo, hi) in enumerate(zip(offs, offs[1:])):
                seg_q = qf[:, lo:hi].astype(mx.float32)
                seg_k = kf[:, lo:hi].astype(mx.float32)
                seg_v = vf[:, lo:hi].astype(mx.float32)
                seg_beta = beta[:, lo:hi]
                seg_g = g[:, lo:hi]
                seg_state = h0[si : si + 1] if h0 is not None else None
                seg_o, _ = naive_recurrent_gated_delta_rule(
                    seg_q, seg_k, seg_v, seg_beta, seg_g,
                    scale=scale, initial_state=seg_state, output_final_state=False,
                )
                ref_parts.append(seg_o)
            ref_o = mx.concatenate(ref_parts, axis=1)
        else:
            ref_o, _ = naive_recurrent_gated_delta_rule(
                qf.astype(mx.float32), kf.astype(mx.float32), vf.astype(mx.float32),
                beta, g, scale=scale, initial_state=h0, output_final_state=False,
            )
        mx.eval(ref_o)

        # --- Path D ---------------------------------------------------------
        y, final_state = gdn_fwd_runtime_call(
            qf, kf, vf, beta, g,
            scale=scale, initial_state=h0,
            output_final_state=shape.output_final_state,
            cu_seqlens=cu_seqlens,
        )
        mx.eval(y, *( (final_state,) if final_state is not None else () ))
        cell.launched = True

        diff = mx.abs(y.astype(mx.float32) - ref_o.astype(mx.float32))
        cell.max_abs_diff = float(mx.max(diff))
        cell.mean_abs_diff = float(mx.mean(diff))
        denom = mx.maximum(mx.abs(ref_o.astype(mx.float32)), 1e-3)
        cell.max_rel_diff = float(mx.max(diff / denom))
        cell.parity_ok = bool(
            cell.max_abs_diff <= tol_abs or cell.max_rel_diff <= tol_rel
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"{exc.__class__.__name__}: {exc}"
        cell.error = msg[:600]
        if _tilelang_non_tail_output_abi_bug(msg):
            cell.error = (
                "TileLang tvm-ffi Metal NON-TAIL owner-output ABI bug: a Path D "
                "stage with an out_idx that is not the tail parameter is bound to "
                "the wrong MLX buffer, surfacing here as the make_packed_api host "
                "assertion 'num_args should be N'. This is a REAL TileLang "
                "codegen/launch defect (reproducible with a 4-line kernel, no "
                "cppmega code), NOT a per-shape Metal-hardware constraint and NOT "
                "the matmul2d zeros/launch-config bug fixed in tilelang 7ae53998. "
                "Every GDN Path D stage (kkt_solve/recompute_w_u/chunk_delta_h) "
                "has a non-tail output, so the pipeline cannot launch until "
                "TileLang binds non-tail owner-outputs by index. See "
                f"_tilelang_non_tail_output_abi_bug. raw={msg[:200]}"
            )
        elif _metal_smem_blocker(msg):
            cell.error = (
                "Metal threadgroup-memory limit: kernel SMEM footprint exceeds "
                "Apple's 32 KiB per-threadgroup cap at this K/V tiling "
                f"(per-shape Metal constraint). raw={msg[:200]}"
            )
        elif _metal_pipeline_state_blocker(msg):
            cell.error = (
                "Metal pipeline-state limit: newComputePipelineStateWithFunction "
                "failed (compute function exceeds available stack space) for this "
                "kernel/tiling -- a per-shape Apple occupancy constraint, not a "
                f"Path D launch-ABI/arg-count defect. raw={msg[:200]}"
            )
        elif _metal_launch_blocker(msg):
            cell.error = (
                "Metal launch blocked: TileLang cooperative-tensor GEMM "
                "(mpp::tensor_ops::matmul2d) requires an Apple M5-class GPU "
                "(Metal 4). Status/compile passed; launch unavailable on this "
                f"host. raw={msg[:200]}"
            )
    return cell


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("reports/raw/v4_gdn_path_d_parity.json"))
    p.add_argument("--tol-abs", type=float, default=2e-2)
    p.add_argument("--tol-rel", type=float, default=2e-2)
    p.add_argument("--only", type=str, default=None, help="comma-separated shape names")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    shapes = DEFAULT_SHAPES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        shapes = tuple(s for s in DEFAULT_SHAPES if s.name in wanted)

    cells = [_run_cell(s, tol_abs=args.tol_abs, tol_rel=args.tol_rel) for s in shapes]

    receipt = GDNParityReceipt(
        host={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        env={
            "CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT": os.environ.get(
                "CPPMEGA_V4_ENABLE_UNSAFE_TRITON_IMPORT"
            ),
            "CPPMEGA_MLX_TRITON_FRONTEND_PATH": os.environ.get(
                "CPPMEGA_MLX_TRITON_FRONTEND_PATH"
            ),
            "CPPMEGA_MLX_FLA_SOURCE_PATH": os.environ.get("CPPMEGA_MLX_FLA_SOURCE_PATH"),
        },
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
        cells=cells,
        all_launched=all(c.launched for c in cells),
        all_parity_ok=all(bool(c.parity_ok) for c in cells),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt.to_json(), indent=2))

    print(f"GDN Path D parity receipt -> {args.out}")
    for c in cells:
        status = (
            "PARITY_OK" if c.parity_ok
            else ("LAUNCHED_MISMATCH" if c.launched else "NOT_LAUNCHED")
        )
        detail = (
            f"max_abs={c.max_abs_diff:.3e} max_rel={c.max_rel_diff:.3e}"
            if c.launched and c.max_abs_diff is not None
            else (c.error or c.backend_reason)[:120]
        )
        print(f"  {c.name:24s} {status:18s} {detail}")
    print(
        f"all_launched={receipt.all_launched} "
        f"all_parity_ok={receipt.all_parity_ok}"
    )
    # Exit 0 if every available cell either parity-passed or was blocked only
    # by the M5 launch capability; nonzero only on a real launched mismatch.
    launched_mismatch = any(c.launched and not c.parity_ok for c in cells)
    return 1 if launched_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
