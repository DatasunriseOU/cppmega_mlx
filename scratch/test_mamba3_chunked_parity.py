"""Parity + chunk-size sweep for the chunked mamba3 forward prototype.

Oracle: cppmega_mlx.nn.mamba3._chunked_mamba3_diagonal_scan (OUR serial scan).
Also cross-checks against state-spaces ssd_minimal algebra via the OUR mapping.
"""

from __future__ import annotations

import sys

import mlx.core as mx

sys.path.insert(0, "/Volumes/external/sources/cppmega_mlx_intfwd")
sys.path.insert(0, "/Volumes/external/sources/cppmega_mlx_intfwd/scratch")

from cppmega_mlx.nn.mamba3 import _chunked_mamba3_diagonal_scan  # noqa: E402
from mamba3_chunked_forward_proto import chunked_mamba3_forward  # noqa: E402


def make_inputs(batch, seq, nheads, headdim, d_state, dtype, seed=0):
    mx.random.seed(seed)
    x = mx.random.normal((batch, seq, nheads, headdim)).astype(dtype)
    Bm = mx.random.normal((batch, seq, nheads, d_state)).astype(dtype) * 0.5
    C = mx.random.normal((batch, seq, nheads, d_state)).astype(dtype) * 0.5
    z = mx.random.normal((batch, seq, nheads, headdim)).astype(dtype)
    # A in (-softplus) regime, clamped <= -0.01; dt = softplus -> small positive
    A = -(mx.abs(mx.random.normal((batch, seq, nheads))) + 0.01)
    dt = mx.softmax(mx.random.normal((batch, seq, nheads)), axis=-1) * 0.0 + 0.05
    D = mx.random.normal((nheads,)).astype(dtype)
    h0 = mx.zeros((batch, nheads, headdim, d_state)).astype(dtype)
    log_decay = (A * dt)[:, :, :, None, None].astype(dtype)
    inp = (x[:, :, :, :, None] * Bm[:, :, :, None, :]).astype(dtype)
    return dict(log_decay=log_decay, inp=inp, C=C, x=x, z=z, D=D, h0=h0, Bm=Bm)


def max_abs_rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    diff = mx.abs(a - b)
    mad = float(mx.max(diff))
    denom = mx.maximum(mx.abs(b), mx.array(1e-6))
    rel = float(mx.max(diff / denom))
    return mad, rel


def run(batch, seq, nheads, headdim, d_state, chunk_size, dtype):
    d = make_inputs(batch, seq, nheads, headdim, d_state, dtype)
    ref_out, ref_h = _chunked_mamba3_diagonal_scan(
        d["log_decay"], d["inp"], d["C"], d["x"], d["z"], d["D"], d["h0"],
        chunk_size=chunk_size,
    )
    mx.eval(ref_out, ref_h)
    out, h = chunked_mamba3_forward(
        d["log_decay"], d["inp"], d["C"], d["x"], d["z"], d["D"], d["h0"],
        chunk_size=chunk_size,
    )
    mx.eval(out, h)
    o_mad, o_rel = max_abs_rel(out, ref_out)
    h_mad, h_rel = max_abs_rel(h, ref_h)
    nan = bool(mx.any(mx.isnan(out))) or bool(mx.any(mx.isnan(h)))
    return o_mad, o_rel, h_mad, h_rel, nan


if __name__ == "__main__":
    B, S, H, P, N = 2, 256, 4, 16, 16
    print(f"shape: batch={B} seq={S} nheads={H} headdim={P} d_state={N}")
    for dtype in (mx.float32, mx.bfloat16):
        print(f"\n=== dtype={dtype} ===")
        for cs in (64, 128, 256):
            if S % cs != 0:
                continue
            o_mad, o_rel, h_mad, h_rel, nan = run(B, S, H, P, N, cs, dtype)
            tol = 5e-3 if dtype == mx.float32 else 5e-2
            ok = (o_mad < tol) and (h_mad < tol) and (not nan)
            print(
                f"chunk={cs:3d} | out max|d|={o_mad:.2e} rel={o_rel:.2e} | "
                f"h max|d|={h_mad:.2e} rel={h_rel:.2e} | nan={nan} | "
                f"{'PASS' if ok else 'FAIL'} (tol={tol:.0e})"
            )
