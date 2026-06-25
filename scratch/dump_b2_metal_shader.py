"""Dump the generated Metal shader source for B2 (chunk_scan_combine_bwd) for both
tested shapes. memguard 70. No fabrication: emits the REAL get_kernel_source()."""
from __future__ import annotations
import os, sys, threading, time

_LIM = 70 * 1024 * 1024
_PEAK = 0
def _rss():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
def _guard():
    global _PEAK
    while True:
        r = _rss()
        if r > _PEAK: _PEAK = r
        if r > _LIM:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n"); sys.stderr.flush(); os._exit(137)
        time.sleep(0.25)
threading.Thread(target=_guard, daemon=True).start()

sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
from cppmega_mlx.nn._tilelang.mamba3_chunked_backward_core import (
    build_chunk_scan_combine_bwd_metal,
)

SHAPES = {
    "tested-S512": dict(batch=1, seqlen=512, chunk_size=64, ngroups=1, nheads=2, headdim=64, dstate=16),
    "nam56r-H128": dict(batch=1, seqlen=512, chunk_size=64, ngroups=8, nheads=128, headdim=64, dstate=64),
}

for name, sh in SHAPES.items():
    k = build_chunk_scan_combine_bwd_metal(**sh)
    src = k.get_kernel_source()
    out = f"/Volumes/external/sources/cppmega.mlx/scratch/b2_metal_{name}.metal"
    with open(out, "w") as f:
        f.write(src)
    print(f"[{name}] wrote {out} ({len(src)} bytes, {src.count(chr(10))} lines)")
    # quick structural greps
    for tok in ["simdgroup_float8x8", "simdgroup_matrix", "simdgroup_multiply",
                "threadgroup ", "threadgroup_barrier", "dispatchThreadgroups",
                "thread_position_in_grid", "for (", "make_filled_simdgroup"]:
        print(f"    grep {tok!r}: {src.count(tok)}")

print(f"PEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("RC=0")
