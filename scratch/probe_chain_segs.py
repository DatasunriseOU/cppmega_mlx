"""Probe the DIRECT CHAIN segments (planner output) for flag OFF vs ON."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import mlx.core as mx
import numpy as np
import m04_train_step as m04  # type: ignore
from cppmega_mlx.recipes.model_factory import build_local_gb10_quarter_tiny_smoke_model
from cppmega_mlx.runtime.path_c_fusion import path_c_mamba3_chunked_scan_enabled

flag_on = path_c_mamba3_chunked_scan_enabled()
mode = "ON" if flag_on else "OFF"
print(f"=== flag {mode} ===")

mx.random.seed(0)
model = build_local_gb10_quarter_tiny_smoke_model(
    hidden_size=64, num_attention_heads=1, mamba_expand=1, mamba_head_dim=64,
    mamba_state_dim=16, mamba_groups=1, mamba_chunk_size=64,
)
profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
prefix = m04._path_c_direct_chain_region_prefix(model, profile_name)
seq = 128

direct_chains = m04.plan_path_c_direct_fusion_chains_for_model(
    model, region_prefix=prefix, include_backward=True, max_segment_nodes=1,
    sequence_length=seq,
)
regions = m04.build_path_c_model_regions_from_model(
    model, region_prefix=prefix, include_backward=False, sequence_length=seq,
)
sel_region = m04._select_path_c_model_route_region(regions)
chain = m04._select_path_c_direct_chain_for_region(direct_chains, sel_region)
print(f"chain status={getattr(chain,'status',None)} n_segments={len(chain.segments)}")
for seg in chain.segments:
    region = seg.region
    ops = []
    for n in getattr(region, "nodes", ()):
        ops.append(n.op_name)
    phase = str(getattr(seg, "execution_phase", "?"))
    print(f"  seg[{seg.index}] status={seg.status} phase={phase} region={region.name} ops={ops}")
