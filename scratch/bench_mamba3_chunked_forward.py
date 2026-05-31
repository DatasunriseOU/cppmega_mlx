"""Measured forward speedup: serial scan vs chunked parallel-scan on Metal (M4 Max).

Serial = cppmega_mlx.nn.mamba3._chunked_mamba3_diagonal_scan (the OUR-variant
reference, a Python source-order loop over S -- the single-threadgroup-equivalent
serial forward the Path-C emitter replaces).
Chunked = scratch/mamba3_chunked_forward_proto.chunked_mamba3_forward (the SSD
4-step decomposition that lowers to many MLX matmul kernels -> saturates the GPU).

Both run on the SAME MLX/Metal device. Wall-time is end-to-end mx.eval. This is a
faithful proxy for "1 threadgroup serial scan" vs "grid-parallel chunked scan":
the serial form is O(S) dependent steps; the chunked form is O(S/C) + batched matmul.
"""
from __future__ import annotations
import sys, time
import mlx.core as mx

sys.path.insert(0, "/Volumes/external/sources/cppmega_mlx_chunkfwd")
sys.path.insert(0, "/Volumes/external/sources/cppmega_mlx_chunkfwd/scratch")

from cppmega_mlx.nn.mamba3 import _chunked_mamba3_diagonal_scan  # noqa
from mamba3_chunked_forward_proto import chunked_mamba3_forward  # noqa
from test_mamba3_chunked_parity import make_inputs  # noqa


def bench(fn, *, warmup=2, iters=5):
    for _ in range(warmup):
        o, h = fn()
        mx.eval(o, h)
    mx.synchronize() if hasattr(mx, "synchronize") else None
    t0 = time.perf_counter()
    for _ in range(iters):
        o, h = fn()
        mx.eval(o, h)
    dt = (time.perf_counter() - t0) / iters
    return dt


if __name__ == "__main__":
    # Realistic-but-tractable shape (serial Python loop over S is slow; keep S moderate).
    B, H, P, N = 1, 8, 64, 16
    dtype = mx.float32
    print(f"device: {mx.default_device()}  dtype={dtype}")
    print(f"{'seq':>6} {'chunk':>6} {'serial_ms':>12} {'chunked_ms':>12} {'speedup':>9}")
    for S in (256, 512, 1024):
        d = make_inputs(B, S, H, P, N, dtype)
        for cs in (64, 128):
            if S % cs:
                continue
            def serial(d=d, cs=cs):
                return _chunked_mamba3_diagonal_scan(
                    d["log_decay"], d["inp"], d["C"], d["x"], d["z"], d["D"], d["h0"], chunk_size=cs)
            def chunked(d=d, cs=cs):
                return chunked_mamba3_forward(
                    d["log_decay"], d["inp"], d["C"], d["x"], d["z"], d["D"], d["h0"], chunk_size=cs)
            ts = bench(serial)
            tc = bench(chunked)
            print(f"{S:>6} {cs:>6} {ts*1e3:>12.3f} {tc*1e3:>12.3f} {ts/tc:>8.2f}x")
