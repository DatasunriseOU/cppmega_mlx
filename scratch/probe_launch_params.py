"""Inspect tirx.kernel_launch_params + thread_extent + dyn shmem attrs for the
chunked B2/B1/B0/fused kernels, so Blocker (B) recovers the right dyn_shmem size."""
import os, sys
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/tilelang")

import mlx.core as mx  # noqa
from cppmega_mlx.nn._tilelang import mamba3_path_c as mpc

# Build the 4 chunked builders directly (they are lru_cached factories).
# Dims must match the nam56r surface used in the smoke.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64

import inspect
builders = [n for n in dir(mpc) if "chunked" in n.lower() or "inter_chunk" in n.lower()
            or n.endswith("_prim") or "b2" in n.lower() or "b1" in n.lower() or "b0" in n.lower()]
print("Candidate builder names in mamba3_path_c:")
for n in builders:
    print("  ", n)
