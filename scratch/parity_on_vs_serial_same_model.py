"""Definitive parity: flag-ON chunked route vs eager serial — SAME model instance.

Builds ONE model, runs the flag-ON direct-chain route (chunked mamba + projection
bridge) to get its 163 grads + loss, then runs the model's OWN eager full-model
value_and_grad (same suffix loss) on the SAME instance. Identical weights, so the
diff isolates ONLY the chunked-vs-serial numerics. The eager mamba uses whatever
the live kernel path selects (matching the route's PREFIX mamba), so prefix layers
match; the in-region layer-10 mamba is the bridge's target.

Run with CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1.
"""

from __future__ import annotations

import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


def main() -> int:
    seq = 64
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    import m04_train_step as m04  # type: ignore
    from cppmega_mlx.recipes.model_factory import (
        build_local_gb10_quarter_tiny_smoke_model,
    )
    from cppmega_mlx.training.cut_cross_entropy import linear_cross_entropy

    mx.random.seed(0)
    model = build_local_gb10_quarter_tiny_smoke_model(
        hidden_size=64, num_attention_heads=1, mamba_expand=1, mamba_head_dim=64,
        mamba_state_dim=16, mamba_groups=1, mamba_chunk_size=64,
    )
    mx.eval(model.parameters())
    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    prefix = m04._path_c_direct_chain_region_prefix(model, profile_name)
    vocab = int(getattr(model, "vocab_size", 0) or 1024)
    ids = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))
    tgt = mx.array(np.random.randint(0, max(2, vocab), size=(1, seq)).astype(np.int32))
    batch = {"tokens": ids, "target_tokens": tgt}

    # ---- Flag-ON direct-chain route ----
    from cppmega_mlx.runtime.path_c_fusion import path_c_mamba3_chunked_scan_enabled
    assert path_c_mamba3_chunked_scan_enabled(), "run with the flag ON"
    direct_chains = m04.plan_path_c_direct_fusion_chains_for_model(
        model, region_prefix=prefix, include_backward=True, max_segment_nodes=1,
        sequence_length=seq,
    )
    regions = m04.build_path_c_model_regions_from_model(
        model, region_prefix=prefix, include_backward=False, sequence_length=seq,
    )
    sel = m04._select_path_c_model_route_region(regions)
    chain = m04._select_path_c_direct_chain_for_region(direct_chains, sel)
    initial_owner = m04.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain, model=model, batch=batch, batch_row=0
    )

    def factory(model_arg, batch_arg, *, batch_row=None, chain=chain):
        return m04.make_path_c_direct_chain_pre_step_runtime_owner(
            chain=chain, model=model_arg, batch=batch_arg, batch_row=batch_row
        )

    install = m04.install_path_c_direct_chain_training_runtime_for_model(
        model=model, chain=chain, logical_owner=initial_owner, sequence_length=seq,
        training_critical_path=True, run_probe=False,
        loss_cotangent_bridge=m04.PathCResidualSumSuffixLossCotangentBridge(chunk_rows=128),
        pre_step_owner_factory=factory,
    )
    print(f"[parity] install status={install.get('status')}", flush=True)
    runtime = getattr(model, "path_c_direct_fusion_chain_training_runtime", None)
    assert runtime is not None

    def forbidden(*_a, **_k):
        raise AssertionError("must not delegate to eager loss_and_grad")

    t0 = time.perf_counter()
    (route_loss, _nt), route_grads = runtime.value_and_grad(model, batch, forbidden)
    rflat = {str(n): v for n, v in tree_flatten(route_grads) if isinstance(v, mx.array)}
    mx.eval(route_loss, *rflat.values())
    route_time = time.perf_counter() - t0
    print(f"[parity] route loss={float(route_loss.item()):.6f} grads={len(rflat)} t={route_time:.3f}s", flush=True)

    # ---- Eager serial full-model value_and_grad on the SAME model ----
    norm = model.norm
    norm_w = norm.weight
    head_w = model.lm_head.weight
    norm_eps = float(getattr(norm, "eps", 1e-5))
    mask = mx.ones(tgt.shape, dtype=mx.float32)

    def loss_fn(m):
        hidden = m.decoder_hidden_states(ids, apply_final_norm=False)
        inv_rms = mx.rsqrt(mx.mean(hidden * hidden, axis=-1, keepdims=True) + mx.array(norm_eps, dtype=hidden.dtype))
        normed = hidden * inv_rms * norm_w
        tl = linear_cross_entropy(normed, head_w, tgt, reduction="none", chunk_rows=128, eval_chunks=False)
        nt = mask.sum()
        return (tl * mask).astype(mx.float32).sum() / mx.maximum(nt, mx.array(1.0, dtype=mx.float32))

    t1 = time.perf_counter()
    serial_loss, serial_grads = nn.value_and_grad(model, loss_fn)(model)
    sflat = {str(n): v for n, v in tree_flatten(serial_grads) if isinstance(v, mx.array)}
    mx.eval(serial_loss, *sflat.values())
    serial_time = time.perf_counter() - t1
    print(f"[parity] serial loss={float(serial_loss.item()):.6f} grads={len(sflat)} t={serial_time:.3f}s", flush=True)

    # ---- Diff ----
    import re
    common = sorted(set(rflat) & set(sflat))
    only_route = sorted(set(rflat) - set(sflat))
    only_serial = sorted(set(sflat) - set(rflat))
    print(f"[parity] loss diff = {abs(float(route_loss.item())-float(serial_loss.item())):.6f}")
    print(f"[parity] common={len(common)} only_route={len(only_route)} only_serial={len(only_serial)}")
    if only_route: print("  only_route:", only_route[:6])
    if only_serial: print("  only_serial:", only_serial[:6])

    def gdiff(n):
        a = np.asarray(rflat[n].astype(mx.float32)); b = np.asarray(sflat[n].astype(mx.float32))
        if a.shape != b.shape: return float("inf")
        return float(np.abs(a - b).max()) if a.size else 0.0

    buckets = {}
    for n in common:
        m = re.match(r"layers\.(\d+)\.", n)
        li = int(m.group(1)) if m else -1
        buckets.setdefault(li, []).append((gdiff(n), n))
    worst = 0.0; wn = ""
    for li in sorted(buckets):
        ds = [d for d, _ in buckets[li]]
        tag = "MAMBA" if li in (2, 6, 10) else ("M2RNN" if li == 11 else "")
        lw = max(ds)
        if lw > worst: worst = lw; wn = max(buckets[li])[1]
        print(f"  layer {li:2d} {tag:6s}: n={len(ds):2d} worst={lw:.3e} mean={np.mean(ds):.3e}")
    print(f"[parity] OVERALL worst per-grad diff = {worst:.3e} at {wn}")
    # Mamba layer-10 (bridge target) detail
    print("[parity] layer-10 mamba (bridge target):")
    for d, n in sorted(buckets.get(10, []), reverse=True):
        flag = "OK" if d < 1e-3 else "OVER-1e-3"
        print(f"    {d:.3e}  {n}  [{flag}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
