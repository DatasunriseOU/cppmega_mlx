"""Chunked parallel-scan BACKWARD prototype for OUR mamba3 diagonal selective SSM.

Step 4 of docs/MAMBA3-PARALLEL-FEASIBILITY.md. The backward is the TRANSPOSE of
the proven chunked forward (scratch/mamba3_chunked_forward_proto.py): the same SSD
4-step chunk decomposition, time-reversed.

OUR exact recurrence (cppmega_mlx/nn/mamba3.py:_chunked_mamba3_diagonal_scan):

    log_decay[t] = A[t] * dt[t]                      # (B,S,H), scalar-per-head decay
    inp[t]       = x[t,:,None] * B[t,None,:]         # (B,S,H,P,N) outer product
    h[t]         = exp(log_decay[t]) * h[t-1] + inp[t]
    y[t]         = sum(h[t] * C[t,None,:], -1) + D*x[t]
    out[t]       = silu(z[t]) * y[t]

The forward proto computes (with a = log_decay = A*dt, in log-space):
    A_cumsum     = cumsum(a)                                  per chunk
    Lmat         = exp(segsum(a))                             intra-chunk decay matrix
    Y_diag       = einsum(C, Lmat, inp)                       intra-chunk
    decay_states = exp(A_cs[-1] - A_cs)                       per-chunk final-state decay
    states       = einsum(decay_states, inp)                  per-chunk summary
    decay_chunk  = exp(segsum(pad(A_cs[-1])))                 inter-chunk decay (lower-tri)
    new_states   = einsum(decay_chunk, [h0; states])          INTER-CHUNK RECURRENCE
    chunk_states = new_states[:-1]                            per-chunk entry state
    final_state  = new_states[-1]
    state_decay  = exp(A_cumsum)                              entry-state -> position decay
    Y_off        = einsum(C, chunk_states, state_decay)       state -> output
    Y            = Y_diag + Y_off
    y            = Y + D*x
    out          = silu(z) * y

The BACKWARD is the exact analytic transpose of every one of those ops, computed
in the same chunk layout. The inter-chunk einsum with the lower-triangular
`decay_chunk` transposes to an UPPER-triangular einsum -> the REVERSE inter-chunk
scan (ssd_state_passing _bwd, O(S/C) steps). Crucially this REUSES the per-chunk
boundary states (chunk_states / decay tensors) computed in the forward instead of
the serial checkpoint-replay.

Cumulative decay is always in LOG-SPACE (segsum/cumsum of a), exponentiated once
-- never a product of per-step exp -- per the doc's underflow caveat. The decay
derivative wrt `a` is handled by the chain rule through Lmat/decay_states/
decay_chunk/state_decay (each is exp(<linear in a>), so d/da just multiplies the
upstream grad by the same exp factor and contracts the matching index).

Validation oracle: the VJP of OUR serial _chunked_mamba3_diagonal_scan
(cppmega_mlx/nn/mamba3.py). See test_mamba3_chunked_backward_parity.py.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mamba3_chunked_forward_proto import _segsum


def _silu_grad(z: mx.array) -> mx.array:
    """d/dz silu(z) = sigmoid(z) * (1 + z*(1 - sigmoid(z)))."""
    s = mx.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def chunked_mamba3_forward_full(
    log_decay: mx.array,  # (B,S,H,1,1)
    inp: mx.array,        # (B,S,H,P,N)
    C: mx.array,          # (B,S,H,N)
    x: mx.array,          # (B,S,H,P)
    z: mx.array,          # (B,S,H,P)
    D: mx.array,          # (H,) or (H,P)
    h0: mx.array,         # (B,H,P,N)
    *,
    chunk_size: int,
):
    """Forward that also returns the intermediate tensors the backward reuses.

    Identical math to chunked_mamba3_forward; returns a cache dict so the
    explicit transpose backward can reuse the per-chunk boundary states (no
    replay). out shaped (B,S,H,P).
    """
    batch, seq, nheads, headdim, d_state = inp.shape
    nchunks = seq // chunk_size
    L = chunk_size

    if D.shape == (nheads,):
        D_skip = D[:, None]
    elif D.shape == (nheads, headdim):
        D_skip = D
    else:
        raise ValueError(f"D must be (H,) or (H,P), got {D.shape}")

    a = log_decay.reshape(batch, seq, nheads)
    a_c = a.reshape(batch, nchunks, L, nheads)
    a_c = mx.transpose(a_c, (0, 3, 1, 2))  # (B,H,c,l)
    A_cumsum = mx.cumsum(a_c, axis=-1)     # (B,H,c,l)

    inp_c = inp.reshape(batch, nchunks, L, nheads, headdim, d_state)
    x_c = x.reshape(batch, nchunks, L, nheads, headdim)
    C_c = C.reshape(batch, nchunks, L, nheads, d_state)

    Lmat = mx.exp(_segsum(a_c))                                   # (B,H,c,l,s)
    Y_diag = mx.einsum("bclhn,bhcls,bcshpn->bclhp", C_c, Lmat, inp_c)

    decay_states = mx.exp(A_cumsum[:, :, :, -1:] - A_cumsum)      # (B,H,c,l)
    states = mx.einsum("bhcl,bclhpn->bchpn", decay_states, inp_c)  # (B,c,H,P,N)

    init = h0[:, None]
    states_cat = mx.concatenate([init, states], axis=1)          # (B,c+1,H,P,N)
    chunk_tail = A_cumsum[:, :, :, -1]                            # (B,H,c)
    chunk_tail_pad = mx.pad(chunk_tail, [(0, 0), (0, 0), (1, 0)])  # (B,H,c+1)
    decay_chunk = mx.exp(_segsum(chunk_tail_pad))                 # (B,H,c+1,c+1)
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    chunk_states = new_states[:, :-1]                            # (B,c,H,P,N)
    final_state = new_states[:, -1]                             # (B,H,P,N)

    state_decay_out = mx.exp(A_cumsum)                          # (B,H,c,l)
    Y_off = mx.einsum("bclhn,bchpn,bhcl->bclhp", C_c, chunk_states, state_decay_out)

    Y = Y_diag + Y_off
    y = Y.reshape(batch, seq, nheads, headdim)
    y = y + D_skip.astype(y.dtype) * x
    gate = nn.silu(z)
    out = gate * y

    cache = dict(
        a_c=a_c, A_cumsum=A_cumsum, inp_c=inp_c, x_c=x_c, C_c=C_c,
        Lmat=Lmat, decay_states=decay_states, states=states,
        states_cat=states_cat, decay_chunk=decay_chunk,
        new_states=new_states, chunk_states=chunk_states,
        state_decay_out=state_decay_out, y=y, gate=gate, z=z,
        D_skip=D_skip, h0=h0,
        shapes=(batch, seq, nheads, headdim, d_state, nchunks, L),
    )
    return out, final_state, cache


def chunked_mamba3_backward(
    dout: mx.array,   # (B,S,H,P) cotangent for out
    cache: dict,
    *,
    dh_last=None,     # optional cotangent for final_state (B,H,P,N); default zero
):
    """Explicit transpose of the chunked forward. Returns grads matching the
    forward inputs (log_decay, inp, C, x, z, D, h0).

    Every step below is the analytic adjoint of the matching forward line. The
    inter-chunk lower-tri einsum transposes to the reverse (upper-tri) scan.
    """
    batch, seq, nheads, headdim, d_state, nchunks, L = cache["shapes"]
    a_c = cache["a_c"]; A_cumsum = cache["A_cumsum"]
    inp_c = cache["inp_c"]; x_c = cache["x_c"]; C_c = cache["C_c"]
    Lmat = cache["Lmat"]; decay_states = cache["decay_states"]
    states_cat = cache["states_cat"]; decay_chunk = cache["decay_chunk"]
    chunk_states = cache["chunk_states"]; state_decay_out = cache["state_decay_out"]
    gate = cache["gate"]; z = cache["z"]; y = cache["y"]; D_skip = cache["D_skip"]

    # ---- transpose of: out = gate * y ; gate = silu(z) ----
    dgate = dout * y
    dy = dout * gate
    dz = dgate * _silu_grad(z)

    # ---- transpose of: y = Y + D_skip * x ----
    # dD over (B,S,H,P); reduce to D shape later
    dx_from_skip = D_skip.astype(dy.dtype) * dy
    # dD_skip[h(,p)] = sum_{b,s} dy * x
    dD_full = dy * x_c.reshape(batch, seq, nheads, headdim)  # (B,S,H,P)
    dY = dy.reshape(batch, nchunks, L, nheads, headdim)      # (B,c,l,H,P)

    # ---- transpose of: Y = Y_diag + Y_off ----
    dY_diag = dY
    dY_off = dY

    # ===== transpose of Y_off = einsum("bclhn,bchpn,bhcl->bclhp", C, chunk_states, state_decay) =====
    # grads wrt each factor (product of three -> drop one, contract dY_off over l,p / matching idx)
    dC_off = mx.einsum("bclhp,bchpn,bhcl->bclhn", dY_off, chunk_states, state_decay_out)
    dchunk_states = mx.einsum("bclhp,bclhn,bhcl->bchpn", dY_off, C_c, state_decay_out)
    dstate_decay_out = mx.einsum("bclhp,bclhn,bchpn->bhcl", dY_off, C_c, chunk_states)
    # state_decay_out = exp(A_cumsum) -> d/dA_cumsum
    dA_cumsum = dstate_decay_out * state_decay_out  # (B,H,c,l)

    # ===== transpose of inter-chunk: new_states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat) =====
    # dchunk_states is grad wrt new_states[:, :-1]; final_state = new_states[:,-1] gets dh_last.
    if dh_last is None:
        dh_last = mx.zeros((batch, nheads, headdim, d_state), dtype=dchunk_states.dtype)
    dnew_states = mx.concatenate([dchunk_states, dh_last[:, None]], axis=1)  # (B,c+1,H,P,N)
    # transpose over the two factors of the einsum (this is the REVERSE inter-chunk scan):
    #   d states_cat[c] = sum_z decay_chunk[z,c] * dnew_states[z]   (upper-tri contraction)
    dstates_cat = mx.einsum("bhzc,bzhpn->bchpn", decay_chunk, dnew_states)  # (B,c+1,H,P,N)
    #   d decay_chunk[z,c] = sum_{p,n} dnew_states[z] * states_cat[c]
    ddecay_chunk = mx.einsum("bzhpn,bchpn->bhzc", dnew_states, states_cat)  # (B,H,c+1,c+1)
    # decay_chunk = exp(segsum(chunk_tail_pad)); transpose of exp then segsum.
    dseg_chunk = ddecay_chunk * decay_chunk                                 # (B,H,c+1,c+1)
    dchunk_tail_pad = _segsum_vjp(dseg_chunk, n=nchunks + 1)                # (B,H,c+1)
    dchunk_tail = dchunk_tail_pad[:, :, 1:]                                 # drop pad -> (B,H,c)
    # chunk_tail = A_cumsum[..., -1] -> scatter into dA_cumsum's last l index
    dA_cumsum = _add_at_last_l(dA_cumsum, dchunk_tail)                      # (B,H,c,l)

    # split dstates_cat -> dh0 (init) and dstates (per-chunk summaries)
    dh0 = dstates_cat[:, 0]                                                 # (B,H,P,N)
    dstates = dstates_cat[:, 1:]                                            # (B,c,H,P,N)

    # ===== transpose of states = einsum("bhcl,bclhpn->bchpn", decay_states, inp_c) =====
    ddecay_states = mx.einsum("bchpn,bclhpn->bhcl", dstates, inp_c)         # (B,H,c,l)
    dinp_from_states = mx.einsum("bchpn,bhcl->bclhpn", dstates, decay_states)  # (B,c,l,H,P,N)
    # decay_states = exp(A_cs[-1] - A_cs): d/dA_cs[-1] = +ds*decay, d/dA_cs[l] = -ds*decay
    t = ddecay_states * decay_states                                       # (B,H,c,l)
    # contribution to A_cumsum: -t at each l, +sum_l t at the last index
    dA_cumsum = dA_cumsum - t
    dA_cumsum = _add_at_last_l(dA_cumsum, mx.sum(t, axis=-1))               # (B,H,c)

    # ===== transpose of Y_diag = einsum("bclhn,bhcls,bcshpn->bclhp", C, Lmat, inp) =====
    dC_diag = mx.einsum("bclhp,bhcls,bcshpn->bclhn", dY_diag, Lmat, inp_c)
    dLmat = mx.einsum("bclhp,bclhn,bcshpn->bhcls", dY_diag, C_c, inp_c)     # (B,H,c,l,s)
    dinp_from_diag = mx.einsum("bclhp,bclhn,bhcls->bcshpn", dY_diag, C_c, Lmat)  # (B,c,s,H,P,N)
    # Lmat = exp(segsum(a_c)) -> transpose of exp then segsum (L x L per chunk)
    dseg = dLmat * Lmat                                                     # (B,H,c,l,s)
    da_from_Lmat = _segsum_vjp(dseg, n=L)                                   # (B,H,c,l)

    # ===== assemble dinp =====
    dinp_c = dinp_from_states + dinp_from_diag                              # (B,c,l,H,P,N)
    dinp = dinp_c.reshape(batch, seq, nheads, headdim, d_state)

    # ===== assemble dC =====
    dC_c = dC_off + dC_diag                                                 # (B,c,l,H,N)
    dC = dC_c.reshape(batch, seq, nheads, d_state)

    # ===== assemble dx =====
    # dx = dx_from_skip (D path) ; Y has no direct x dependence (x lives in inp)
    dx = dx_from_skip

    # ===== A_cumsum = cumsum(a_c) -> transpose is reverse-cumsum =====
    # plus the direct da_from_Lmat (segsum of a_c).
    da_c = _cumsum_vjp(dA_cumsum) + da_from_Lmat                            # (B,H,c,l)
    # a_c was transpose(a.reshape(B,c,L,H),(0,3,1,2)); invert layout -> log_decay grad
    da = mx.transpose(da_c, (0, 2, 3, 1)).reshape(batch, seq, nheads)
    dlog_decay = da.reshape(batch, seq, nheads, 1, 1)

    # ===== dD: reduce dD_full to D shape (caller decides H vs H,P) =====
    dD = mx.sum(dD_full, axis=(0, 1))  # (H,P); test reduces further if D is (H,)

    grads = dict(
        log_decay=dlog_decay, inp=dinp, C=dC, x=dx, z=dz, D=dD, h0=dh0,
    )
    return grads


def _segsum_vjp(dseg: mx.array, n: int) -> mx.array:
    """Adjoint of _segsum: maps grad wrt seg (..., n, n) back to grad wrt a (..., n).

    Forward: a_masked[...,d,e] = a[...,d] if d>e else 0 ; seg = cumsum(a_masked, axis=-2)
             (the masked-where for d<i is constant -inf -> zero grad there).
    Adjoint: d a_masked = reverse-cumsum over axis -2 of (dseg restricted to lower-tri),
             then sum the d>e entries back over e to a[...,d].
    """
    row = mx.arange(n).reshape(n, 1)
    col = mx.arange(n).reshape(1, n)
    lower = (row >= col)  # where seg carried a finite value
    dseg_m = mx.where(lower, dseg, mx.zeros_like(dseg))
    # adjoint of cumsum(axis=-2) is reverse-cumsum(axis=-2)
    da_masked = _reverse_cumsum(dseg_m, axis=-2)
    # adjoint of the strict-lower mask: keep d>e, then sum over e
    strict_lower = (row > col)
    da_masked = mx.where(strict_lower, da_masked, mx.zeros_like(da_masked))
    return mx.sum(da_masked, axis=-1)


def _cumsum_vjp(g: mx.array) -> mx.array:
    """Adjoint of cumsum(axis=-1) is reverse-cumsum(axis=-1)."""
    return _reverse_cumsum(g, axis=-1)


def _reverse_cumsum(g: mx.array, axis: int) -> mx.array:
    return mx.cumsum(g, axis=axis, reverse=True)


def _add_at_last_l(t: mx.array, addend: mx.array) -> mx.array:
    """Add `addend` (B,H,c) into t[..., -1] of a (B,H,c,l) tensor."""
    B, H, c, l = t.shape
    onehot = (mx.arange(l) == (l - 1)).astype(t.dtype).reshape(1, 1, 1, l)
    return t + addend[..., None] * onehot
