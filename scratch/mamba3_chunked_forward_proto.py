"""Chunked parallel-scan FORWARD prototype for OUR mamba3 diagonal selective SSM.

Step 3 of docs/MAMBA3-PARALLEL-FEASIBILITY.md: replace the single-threadgroup
serial forward scan with the SSD/FLA 4-step chunked decomposition, proving the
speedup before the backward rewrite.

OUR exact recurrence (cppmega_mlx/nn/mamba3.py:_chunked_mamba3_diagonal_scan):

    log_decay[t] = A[t] * dt[t]                      # (B,S,H), scalar-per-head decay
    inp[t]       = x[t, :, None] * B[t, None, :]     # (B,S,H,P,N) outer product
    h[t]         = exp(log_decay[t]) * h[t-1] + inp[t]
    y[t]         = sum(h[t] * C[t, None, :], -1) + D * x[t]
    out[t]       = silu(z[t]) * y[t]

IMPORTANT mapping note vs state-spaces ssd_minimal: ssd_minimal folds dt into the
input (X <- x*dt) AND into A (A*dt). OUR recurrence folds dt ONLY into the decay
(log_decay = A*dt) and uses inp = x (X) B (no dt on the input). So the chunked
decomposition below uses:
    a[t]   = A[t]*dt[t]            (log decay exponent)
    Xin[t] = x[t]                  (NO dt multiply -- this is the OUR variant)
    B,C    as-is.

The intra-chunk / inter-chunk / state->output algebra is identical to SSD; only
the per-step input scaling differs. Cumulative decay is always formed in
LOG-SPACE (sum of A*dt, exponentiate once) per the feasibility doc's underflow
caveat -- never a product of per-step exp values.

This is the numerically-validated algorithm core. A TileLang prim_func that maps
the same einsums onto a (chunks x channels) grid is in
mamba3_chunked_forward_tilelang.py (adapted from
tilelang/examples/linear_attention/example_mamba_chunk_scan.py).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _segsum(a: mx.array) -> mx.array:
    """Stable lower-triangular segment sum.

    a: (..., L)  ->  returns (..., L, L) where out[...,i,j] = sum_{j<k<=i} a[k]
    for i>=j, and -inf for i<j. Matches state-spaces/mamba segsum().
    """
    L = a.shape[-1]
    # repeat along a new trailing axis: (..., L) -> (..., L, L)
    a_rep = mx.repeat(a[..., None], repeats=L, axis=-1)
    # strictly-lower mask (diagonal=-1): keep a[...,d] for e<d
    row = mx.arange(L).reshape(L, 1)
    col = mx.arange(L).reshape(1, L)
    strict_lower = (row > col)  # (L,L), True where d>e  (axis -2 is d, -1 is e)
    a_masked = mx.where(strict_lower, a_rep, mx.zeros_like(a_rep))
    seg = mx.cumsum(a_masked, axis=-2)
    lower = (row >= col)
    neg_inf = mx.full(seg.shape, -1e30, dtype=seg.dtype)
    return mx.where(lower, seg, neg_inf)


def chunked_mamba3_forward(
    log_decay: mx.array,  # (B,S,H,1,1)
    inp: mx.array,        # (B,S,H,P,N)  == x[...,None] * B[...,None,:]
    C: mx.array,          # (B,S,H,N)
    x: mx.array,          # (B,S,H,P)
    z: mx.array,          # (B,S,H,P)
    D: mx.array,          # (H,) or (H,P)
    h0: mx.array,         # (B,H,P,N)
    *,
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """SSD 4-step chunked forward, matching _chunked_mamba3_diagonal_scan.

    Returns (out, h_last) with out shaped (B,S,H,P).
    """
    batch, seq, nheads, headdim, d_state = inp.shape
    if log_decay.shape != (batch, seq, nheads, 1, 1):
        raise ValueError(f"log_decay must be {(batch, seq, nheads, 1, 1)}, got {log_decay.shape}")
    if seq % chunk_size != 0:
        raise ValueError(
            f"prototype requires seq ({seq}) divisible by chunk_size ({chunk_size})"
        )
    nchunks = seq // chunk_size
    L = chunk_size

    if D.shape == (nheads,):
        D_skip = D[:, None]
    elif D.shape == (nheads, headdim):
        D_skip = D
    else:
        raise ValueError(f"D must be (H,) or (H,P), got {D.shape}")

    # a[t] = A*dt = log_decay  -> (B,S,H)
    a = log_decay.reshape(batch, seq, nheads)
    # reshape to chunks: (B, c, l, H)
    a_c = a.reshape(batch, nchunks, L, nheads)
    a_c = mx.transpose(a_c, (0, 3, 1, 2))  # (B,H,c,l)
    A_cumsum = mx.cumsum(a_c, axis=-1)  # (B,H,c,l)

    # inp chunks: (B,c,l,H,P,N); x/z/C chunks
    inp_c = inp.reshape(batch, nchunks, L, nheads, headdim, d_state)
    x_c = x.reshape(batch, nchunks, L, nheads, headdim)
    z_c = z.reshape(batch, nchunks, L, nheads, headdim)
    C_c = C.reshape(batch, nchunks, L, nheads, d_state)

    # --- Step 1: intra-chunk diagonal (Y_diag) ---
    # decay matrix Lmat[b,h,c,l,s] = exp(segsum(a)[...,l,s]) = exp(A_cs[l]-A_cs[s]) for l>=s
    Lmat = mx.exp(_segsum(a_c))  # (B,H,c,l,s)
    # OUR inp already includes B as outer product: inp[...,p,n] = x[...,p]*B[...,n].
    # Y_diag[b,c,l,h,p] = sum_{s<=l} C[l,n] * inp[s,p,n] * Lmat[l,s]
    # = sum_s Lmat[l,s] * sum_n C[l,n]*inp[s,p,n]
    # Compute CB[b,c,h,l,s] = sum_n C[l,n] * (sum over inp? no) -> we need inp at s.
    # inp[s,p,n] = x[s,p]*B[s,n]. So sum_n C[l,n]*inp[s,p,n] = x[s,p]*sum_n C[l,n]*B[s,n].
    # Recover B from inp is unstable; instead carry B explicitly via a CB built from C and inp.
    # CB[b,c,h,l,s,p] would be 6D & huge. Use the SSD identity with explicit B:
    # We pass inp = x (x) B, so define Bmat[b,c,s,h,n] by extracting the n-vector at p=0 scaled.
    # To stay faithful & cheap, recompute B from the relation is avoided: instead require caller
    # to also pass B. For the standalone proto we reconstruct B and x from inp deterministically
    # is not needed -- chunked_mamba3_forward_bx() below takes B,x directly. Keep this signature
    # operating on inp by the exact contraction:
    #   Y_diag[b,c,l,h,p] = sum_s Lmat[l,s] * sum_n C[l,n] * inp[s,h,p,n]
    # einsum over (s,n):
    Y_diag = mx.einsum("bclhn,bhcls,bcshpn->bclhp", C_c, Lmat, inp_c)

    # --- Step 2: per-chunk final state (from h0=0 within chunk) ---
    # decay_states[b,h,c,l] = exp(A_cs[c,-1] - A_cs[c,l])
    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (B,H,c,l)
    # states[b,c,h,p,n] = sum_l decay_states[l] * inp[l,h,p,n]
    states = mx.einsum("bhcl,bclhpn->bchpn", decay_states, inp_c)  # (B,c,H,P,N)

    # --- Step 3: inter-chunk recurrence (the ONLY sequential part, O(nchunks)) ---
    # initial_states from h0: (B,1,H,P,N)
    init = h0[:, None]  # (B,1,H,P,N)
    states_cat = mx.concatenate([init, states], axis=1)  # (B,c+1,H,P,N)
    # chunk-boundary decay: segsum over padded A_cumsum[...,-1]
    chunk_tail = A_cumsum[:, :, :, -1]  # (B,H,c)
    chunk_tail_pad = mx.pad(chunk_tail, [(0, 0), (0, 0), (1, 0)])  # (B,H,c+1)
    decay_chunk = mx.exp(_segsum(chunk_tail_pad))  # (B,H,c+1,c+1)
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)  # (B,c+1,H,P,N)
    chunk_states = new_states[:, :-1]  # entry state per chunk (B,c,H,P,N)
    final_state = new_states[:, -1]    # (B,H,P,N)

    # --- Step 4: state -> output (Y_off) + diag + skip + gate ---
    state_decay_out = mx.exp(A_cumsum)  # (B,H,c,l)
    # Y_off[b,c,l,h,p] = sum_n C[l,n] * chunk_states[c,h,p,n] * state_decay_out[l]
    Y_off = mx.einsum(
        "bclhn,bchpn,bhcl->bclhp", C_c, chunk_states, state_decay_out
    )

    Y = Y_diag + Y_off  # (B,c,l,H,P)
    y = Y.reshape(batch, seq, nheads, headdim)
    # skip + gate
    y = y + D_skip.astype(y.dtype) * x
    out = nn.silu(z) * y
    return out, final_state
