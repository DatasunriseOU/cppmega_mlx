"""m04 Metal full-Path-C-backward UNLOCK verification (mamba3 chunked runtime wiring).

Drives the LIVE m04 direct-fusion-chain route end-to-end (fwd + bwd) through the
real production recipe (mirror of scripts/m04_train_step.py ~:11786):

    plan_path_c_direct_fusion_chains_for_model(..., max_segment_nodes=1)
      -> _select_path_c_direct_chain_for_region
      -> make_path_c_direct_chain_pre_step_runtime_owner   (allocates ALL buffers,
         incl. the 11 mamba3 chunked handoff buffers when the flag is ON)
      -> install_path_c_direct_chain_training_runtime_for_model (compiles segments)
      -> runtime.value_and_grad(...)  (runs run_path_c_direct_fusion_chain_route
         for forward then backward, binding handoff buffers positionally)

Run TWICE (separate processes): flag OFF (serial mamba3_mimo backward) and flag ON
(chunked F0/F1/F2 + B0/B1/B2). Emits a JSON receipt with segment counts, loss,
per-grad tensors, and fwd+bwd wall-times. A driver then diffs the two receipts for
per-grad parity (< 1e-3) and confirms end-to-end completion.

Usage:
    python verify_mamba3_chunked_runtime_unlock.py --mode off --out off.json
    CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1 python ... --mode on --out on.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np


def _import_m04():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    import m04_train_step as m04  # type: ignore

    return m04


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "on"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=128)  # divisible by chunk(64)
    args = ap.parse_args()

    m04 = _import_m04()
    from cppmega_mlx.recipes.model_factory import build_local_gb10_quarter_tiny_smoke_model
    from cppmega_mlx.runtime.path_c_fusion import (
        path_c_mamba3_chunked_scan_enabled,
    )
    from mlx.utils import tree_flatten

    # Planning fns + the loss bridge live in the m04 module namespace.
    plan_path_c_direct_fusion_chains_for_model = m04.plan_path_c_direct_fusion_chains_for_model
    build_path_c_model_regions_from_model = m04.build_path_c_model_regions_from_model
    PathCResidualSumSuffixLossCotangentBridge = m04.PathCResidualSumSuffixLossCotangentBridge

    flag_on = path_c_mamba3_chunked_scan_enabled()
    print(f"[verify:{args.mode}] chunked_scan_enabled={flag_on}", flush=True)

    # Build a smoke model that contains a mamba3 brick. The chunked SSD scan-core
    # requires mamba head_dim divisible by block_N=16 (verified config used
    # head_dim=64, d_state=16, chunk=64); the default smoke dims (head_dim=4) are
    # infeasible for the chunked grid kernel. Override to the smallest chunked-
    # feasible mamba dims while keeping the rest of the model tiny. hidden_size =
    # num_attention_heads * head_dim, and d_inner(=hidden*expand) % mamba_head_dim
    # == 0 so mamba_num_heads = d_inner // mamba_head_dim.
    mx.random.seed(0)
    model = build_local_gb10_quarter_tiny_smoke_model(
        hidden_size=64,
        num_attention_heads=1,  # attention head_dim = 64
        mamba_expand=1,         # d_inner = 64
        mamba_head_dim=64,      # divisible by block_N=16; nheads = 64/64 = 1
        mamba_state_dim=16,
        mamba_groups=1,
        mamba_chunk_size=64,
    )
    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    prefix = m04._path_c_direct_chain_region_prefix(model, profile_name)
    seq = int(args.seq)

    # batch-1 token ids
    vocab = int(getattr(model, "vocab_size", 0) or 1024)
    ids = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))
    tgt = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))
    batch = {"tokens": ids, "target_tokens": tgt}

    direct_chains = plan_path_c_direct_fusion_chains_for_model(
        model,
        region_prefix=prefix,
        include_backward=True,
        max_segment_nodes=1,
        sequence_length=seq,
    )
    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=prefix,
        include_backward=False,
        sequence_length=seq,
    )
    selected_region = m04._select_path_c_model_route_region(regions)
    chain = m04._select_path_c_direct_chain_for_region(direct_chains, selected_region)
    if chain is None:
        raise SystemExit("no direct chain selected")
    chain_status = str(getattr(chain, "status", ""))
    print(f"[verify:{args.mode}] chain status={chain_status}", flush=True)

    # Segment census: count chunked vs serial mamba3 segments.
    seg_info = []
    for seg in getattr(chain, "segments", ()):
        target = getattr(seg, "schedule_target", None)
        op = ""
        if target is not None and str(seg.status) == "ok":
            try:
                pf = target.schedule_template(seg.region)
                op = ",".join(getattr(pf, "_cppmega_path_c_brick_ops", ()) or ())
            except Exception as exc:
                op = f"<err:{type(exc).__name__}>"
        seg_info.append(
            {
                "index": int(seg.index),
                "status": str(seg.status),
                "phase": str(getattr(seg, "execution_phase", "?")),
                "region": str(getattr(seg.region, "name", "")),
                "brick_ops": op,
            }
        )
    n_chunked = sum(
        1
        for s in seg_info
        if any(
            k in s["brick_ops"]
            for k in (
                "mamba3_chunk_precompute",
                "mamba3_inter_chunk_recur",
                "mamba3_chunk_scan_combine",
            )
        )
    )
    print(f"[verify:{args.mode}] segments={len(seg_info)} chunked_mamba3_segs={n_chunked}", flush=True)

    initial_owner = m04.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain, model=model, batch=batch, batch_row=0
    )

    def pre_step_owner_factory(model_arg, batch_arg, *, batch_row=None, chain=chain):
        return m04.make_path_c_direct_chain_pre_step_runtime_owner(
            chain=chain, model=model_arg, batch=batch_arg, batch_row=batch_row
        )

    # Confirm the handoff buffers were allocated into the owner when flag ON.
    handoff_fwd = sorted(
        n for n in initial_owner.buffers if n.split("_", 1)[-1] in
        {"cb", "dA_cumsum", "summary_states", "prev_states", "final_state"}
        and "mamba3" in n
    )
    handoff_bwd = sorted(
        n for n in initial_owner.buffers
        if any(n.endswith("_" + s) for s in
               ("dh_last", "dchunk_states", "dstates", "dinp_diag",
                "dA_cumsum_y", "dA_cumsum_tail"))
    )
    print(f"[verify:{args.mode}] owner handoff_fwd={handoff_fwd}", flush=True)
    print(f"[verify:{args.mode}] owner handoff_bwd={handoff_bwd}", flush=True)

    install = m04.install_path_c_direct_chain_training_runtime_for_model(
        model=model,
        chain=chain,
        logical_owner=initial_owner,
        sequence_length=seq,
        training_critical_path=True,
        run_probe=False,
        loss_cotangent_bridge=PathCResidualSumSuffixLossCotangentBridge(chunk_rows=128),
        pre_step_owner_factory=pre_step_owner_factory,
    )
    print(f"[verify:{args.mode}] install status={install.get('status')} reason={install.get('reason')}", flush=True)
    if install.get("status") != "ok":
        raise SystemExit(f"install blocked: {json.dumps(install, default=str)[:2000]}")

    runtime = getattr(model, "path_c_direct_fusion_chain_training_runtime", None)
    if runtime is None:
        raise SystemExit("runtime not installed")

    def _forbidden(*_a, **_k):
        raise AssertionError("must not delegate to eager loss_and_grad")

    t0 = time.perf_counter()
    (loss, ntokens), grads = runtime.value_and_grad(model, batch, _forbidden)
    flat = [(str(n), v) for n, v in tree_flatten(grads) if isinstance(v, mx.array)]
    mx.eval(loss, ntokens, *(v for _, v in flat))
    elapsed = time.perf_counter() - t0

    loss_v = float(loss.item())
    finite = bool(mx.isfinite(loss).item())
    print(f"[verify:{args.mode}] loss={loss_v:.6e} finite={finite} grads={len(flat)} elapsed={elapsed:.3f}s", flush=True)

    # Persist grads (host numpy) for parity diff.
    grad_dump = {}
    for n, v in flat:
        a = np.asarray(v.astype(mx.float32))
        grad_dump[n] = {
            "shape": list(a.shape),
            "absmax": float(np.abs(a).max()) if a.size else 0.0,
            # store a flat float32 list only for small tensors to bound file size;
            # for parity we use a stable hashable summary + per-element for small.
            "sum": float(a.sum()),
            "l2": float(np.sqrt((a.astype(np.float64) ** 2).sum())),
            "data": a.reshape(-1).tolist() if a.size <= 200000 else None,
        }

    receipt = {
        "mode": args.mode,
        "flag_on": flag_on,
        "seq": seq,
        "chain_status": chain_status,
        "segment_count": len(seg_info),
        "chunked_mamba3_segments": n_chunked,
        "segments": seg_info,
        "owner_handoff_fwd": handoff_fwd,
        "owner_handoff_bwd": handoff_bwd,
        "loss": loss_v,
        "loss_finite": finite,
        "ntokens": int(ntokens.item()),
        "grad_count": len(flat),
        "elapsed_total_s": elapsed,
        "grads": grad_dump,
    }
    with open(args.out, "w") as f:
        json.dump(receipt, f)
    print(f"[verify:{args.mode}] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
