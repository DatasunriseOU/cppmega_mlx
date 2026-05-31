#!/usr/bin/env python3
"""Forward-activation parity harness for the row-chunked-independent FORWARD ops.

Runs the tiny local_gb10_quarter smoke direct-chain (same 7-op structure as full
scale, small dims so it COMPILES + RUNS on Metal) FORWARD-ONLY with RANDOM,
non-zero caller-owned input buffers, then captures every FORWARD output/activation
buffer (the attention KV-history workspace q_fp8/q_scale/kv_fp8/kv_scale, the
sparse-MLA attention output + LSE, and the rmsnorm/m2rnn/mamba3 intermediate
activations).

Used to prove the forward row-chunk fix (attention_qkv_projection /
sparse_mla_fp8_apply routed to launcher_chunks + isolated) leaves the forward
ACTIVATIONS bit-identical to the canonical maximally-fused grid_chunks dataflow.

The correct A/B (no git stash needed):
  * default run            -> the row-chunked forward (cap=2 + launcher isolation)
  * ``--greedy-reference`` -> the GREEDY all-grid_chunks forward (cap=99, forward
                              launcher routing disabled), the canonical
                              explicit-dataflow oracle.
The two MUST be bitwise-equal on the forward activations
(q_fp8/q_scale/kv_fp8/kv_scale + sparse_mla out/lse). VERIFIED bitexact (diff=0).

NOTE: do NOT use the SHIPPED cap=2 grouping ([residual,m2rnn][residual,attn]) as
the oracle: it has a PRE-EXISTING segmentation-dependent FP8-quantization
divergence from greedy that is unrelated to row-chunking (confirmed: the cap=2
baseline itself diverges from greedy by the full FP8 range even with NO
row-chunking). Only the greedy maximally-fused dataflow is a stable reference.

RULE #1: the route RAISES on failure; this harness reports loudly and never
fabricates parity.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def _stable_name_seed(name: str, base_seed: int) -> int:
    """Process-stable per-buffer RNG seed (Python's hash() is salted)."""

    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402
import m04_train_step as m  # noqa: E402
from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    build_local_gb10_quarter_tiny_smoke_model,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (  # noqa: E402
    DESCRIPTOR_EXECUTION_STAGE_FORWARD,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--force-grid-chunks",
        action="store_true",
        help=(
            "plan with forward_max_segment_nodes=None to disable the forward "
            "row-chunk isolation (greedy grid_chunks forward) -- a same-branch "
            "A/B without needing a git stash"
        ),
    )
    parser.add_argument(
        "--greedy-reference",
        action="store_true",
        help=(
            "plan the GREEDY maximally-fused all-grid_chunks forward (cap=99, "
            "forward launcher routing disabled) -- the canonical explicit-dataflow "
            "reference. The row-chunked forward (default run) must be BITWISE equal "
            "to this. NOTE: the SHIPPED cap=2 baseline has a PRE-EXISTING "
            "segmentation-dependent FP8-quantization divergence from greedy that is "
            "unrelated to row-chunking, so greedy -- not cap=2 -- is the correct "
            "parity oracle."
        ),
    )
    args = parser.parse_args()
    if args.greedy_reference:
        # Disable the forward row-chunk launcher routing so the heavy forward ops
        # stay grid_chunks even when isolated, giving the pure greedy all-grid
        # reference dataflow.
        from cppmega_mlx.runtime import path_c_fusion_schedules as _P

        _orig_route = _P._target_with_max_rows_per_launch

        def _grid_only_for_forward(target, region, mrpl, mode):
            if not any(
                str(getattr(n, "op_name", "")).endswith("_bwd")
                for n in region.nodes
            ):
                return target  # leave forward ops grid_chunks
            return _orig_route(target, region, mrpl, mode)

        _P._target_with_max_rows_per_launch = _grid_only_for_forward

    model = build_local_gb10_quarter_tiny_smoke_model()
    regions = tuple(
        model.path_c_fusion_regions(
            include_backward=False,
            min_route_bricks=2,
            sequence_length=args.seq_len,
        )
    )
    sel = m._select_path_c_model_route_region(regions)
    assert sel is not None
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    plan_kwargs = {}
    if args.force_grid_chunks:
        plan_kwargs["forward_max_segment_nodes"] = None
    if args.greedy_reference:
        plan_kwargs["forward_max_segment_nodes"] = 99
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True, **plan_kwargs,
    )
    fwd_dispatch = [
        (
            [n.op_name for n in s.region.nodes],
            getattr(getattr(s, "schedule_target", None), "row_dispatch_mode", None),
        )
        for s in chain.segments
        if s.execution_phase == "forward"
    ]
    print(
        f"[fwd-parity] chain.status={chain.status} segs={len(chain.segments)} "
        f"forward_dispatch={fwd_dispatch}",
        flush=True,
    )
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)

    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain)
    buffers: dict[str, mx.array] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        shape = tuple(int(d) for d in spec["shape"])
        # Random small inputs; output/grad buffers start at zero and get filled.
        # CRITICAL for A/B parity: seed each buffer's RNG by its NAME (not by
        # draw order) so identical-named inputs get BIT-identical values across
        # two chains whose segment count / buffer-spec ORDER differs (row-chunked
        # has 12 segments, grid_chunks 9). A single shared rng would draw in
        # different orders and give different inputs -> a false non-parity.
        is_grad = "_grad" in name or name.endswith("_grad")
        if is_grad or str(spec["dtype"]) not in ("float32", "float16", "bfloat16"):
            buffers[name] = mx.zeros(shape, dtype=dtype)
        else:
            rng = np.random.default_rng(_stable_name_seed(name, args.seed))
            arr = rng.standard_normal(shape).astype(np.float32) * 0.1
            buffers[name] = mx.array(arr).astype(dtype)
    mx.eval(*buffers.values())

    # FORWARD-ONLY: drive only the forward segments so we capture exactly the
    # forward activations (no backward grad writes muddy the comparison).
    payload = m.run_path_c_direct_fusion_chain_route(
        chain=chain,
        logical_buffers=buffers,
        artifacts=artifacts,
        execution_phases=[DESCRIPTOR_EXECUTION_STAGE_FORWARD],
    )
    print(f"[fwd-parity] route status={payload.get('status')}", flush=True)
    if payload.get("status") != "ok":
        print("[fwd-parity] ROUTE FAILED", flush=True)
        return 2

    # Capture every FORWARD activation/output buffer. These are the buffers the
    # forward ops WRITE (not the input weights, not the backward grads). We key on
    # the canonical activation names produced by the forward chain.
    forward_activation_markers = (
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
        "attention_hidden",
        "attention_out",
        "sparse_mla",
        "lse",
        "m2rnn_hidden",
        "m2rnn_delta",
        "mamba3_delta",
        "hidden_after_mamba3",
        "residual_norm",
        "entry_rmsnorm",
    )
    captured = {}
    for name, val in buffers.items():
        low = name.lower()
        if low.endswith("_grad") or "_grad_" in low:
            continue
        if any(marker in low for marker in forward_activation_markers):
            mx.eval(val)
            captured[name] = np.array(val).astype(np.float64)

    np.savez(args.out, **captured)
    nz = {k: float(np.max(np.abs(v))) for k, v in captured.items()}
    nonzero = {k: v for k, v in nz.items() if v > 0}
    print(
        f"[fwd-parity] captured {len(captured)} forward activation buffers; "
        f"{len(nonzero)} non-zero. max-abs sample: "
        f"{dict(list(nonzero.items())[:8])}",
        flush=True,
    )
    print(f"[fwd-parity] wrote -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
