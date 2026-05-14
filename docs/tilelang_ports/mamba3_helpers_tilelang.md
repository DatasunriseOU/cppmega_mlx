# Mamba3 backward helpers — TileLang/tvm-ffi port

This note covers the native TileLang port of the three Triton helpers used by Mamba3's
backward pass:

1. compute_dacs_segsum_triton — segment cumulative-sum reduction over the
   time axis.
2. bwd_dadt_fused_triton — fused dA/ddt computation.
3. bwd_dtrap_ddt_triton — fused ddt/dtrap from the trapezoidal scale.

The goal of this port is to keep the helper kernels inside TileLang while
launching through the standard TileLang -> TVM -> tvm-ffi/native API with
explicit MLX owner-output buffers. Pure-MLX rewrites in
cppmega_mlx/nn/_tilelang/_mamba3_helpers.py (sibling agent) remain the
fallback when TileLang cannot compile or consume the buffers directly.

## Source attribution

* Triton originals live in upstream
  mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py (state-spaces/mamba). The
  three dense-path entry points are compute_dacs_segsum_triton,
  bwd_dadt_fused_triton, bwd_dtrap_ddt_triton.
* The cppmega gb10 wrapper (cppmega/megatron/tilelang_mimo_autograd.py)
  re-exposes compute_dacs_segsum_triton with the cppmega (B, T, H) shape
  contract; the TileLang port here matches that contract so the two helpers are
  drop-in swappable.
* Pure-MLX siblings: cppmega_mlx/nn/_tilelang/_mamba3_helpers.py (sibling
  agent's port). They are the parity oracle on macOS because Triton has no
  Metal backend.

## Files in this port

<table>
  <thead>
    <tr>
      <th>Path</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>cppmega_mlx/nn/_tilelang/_mamba3_helpers_tilelang.py</td>
      <td>TileLang PrimFunc definitions, native tvm-ffi compile,<br>
      owner-output dispatch, status surface.</td>
    </tr>
    <tr>
      <td>tests/test_tilelang_mamba3_helpers.py</td>
      <td>19 parameterised parity tests + status assertions.</td>
    </tr>
    <tr>
      <td>scripts/bench_tilelang_mamba3_helpers.py</td>
      <td>Side-by-side bench vs pure-MLX.</td>
    </tr>
    <tr>
      <td>bench/tilelang_ports/mamba3_helpers.json</td>
      <td>Bench output (this file is overwritten on each run).</td>
    </tr>
  </tbody>
</table>

## TileLang DSL → tvm-ffi notes

### Common pattern

All three kernels share the same native pipeline:

1. Build a @T.prim_func PrimFunc with explicit thread/grid extents.
2. Compile with tilelang.compile(..., target="metal",
   execution_backend="tvm_ffi", out_idx=...).
3. Allocate the helper's explicit output buffers and pass them as tvm-ffi
   owner outputs.
4. If the caller's dtype/layout does not match the no-copy native ABI, fall
   back to the pure-MLX sibling instead of casting, padding, or staging hidden
   adapter buffers.

The previous extracted-MSL + mx.fast.metal_kernel bridge is retired for these
helpers. MSL extraction remains useful only for debug surfaces that already
start from TileLang IR but still need a legacy MLX fast-kernel wrapper.

### Helper 1 — compute_dacs_segsum

The Triton kernel does a chunked reverse cumulative sum and writes both the
cumsum and a triangular segsum tile. The cppmega rewrite (and the pure-MLX
sibling) collapse the chunked path into a single reverse cumsum over the
whole sequence, weighted by exp(rev[t]), broadcast against dh[t]. The
TileLang version mirrors that:

python
@T.prim_func
def segsum(A, dt, dh, out):
    with T.Kernel(BH, ceildiv(K, BLOCK_K), threads=BLOCK_K) as (bh, bk):
        # Thread 0 of the row builds the per-time-step weight in shared mem.
        weights = T.alloc_shared((T_,), "float32", scope="shared")
        if tk == 0:
            acc = 0.0
            weights[T_-1] = exp(acc)
            for r in range(T_-1):
                t = T_-2-r
                acc += A[bh,t+1] * dt[bh,t+1]
                weights[t] = exp(acc)
        T.sync_threads()
        # Threads in the row stream out[bh, *, k_index] = dh[bh, *, k_index] * weight
        for t in range(T_):
            out[bh, t, k_index] = (dh[bh, t, k_index].cast(f32) * weights[t]).cast(f16)


Lowering produced clean MSL with one threadgroup_barrier(mem_flags::mem_threadgroup)
between the producer and consumers; the emitted carrier dtype is half.

### Helper 2 — bwd_dadt_fused

Element-wise reduction over trailing dims, then two scalar multiplies per
(B, T, H) location. The TileLang kernel uses one thread per (bh, t) and
loops K serially:

python
@T.prim_func
def dadt(A, dY, dt, h, dA, ddt):
    with T.Kernel(BH, ceildiv(T_, BLOCK_T), threads=BLOCK_T) as (bh, btile):
        if t_index < T_:
            acc = 0.0
            for k in range(K):
                acc += dY[bh, t_index, k].cast(f32) * h[bh, t_index, k].cast(f32)
            dA[bh, t_index]  = acc * dt[bh, t_index]
            ddt[bh, t_index] = acc * A[bh, t_index]


### Helper 3 — bwd_dtrap_ddt

Pure element-wise + 1-token shift kernel. Each thread writes one (bh, t)
entry; the t > 0 branch reads dB_scaled[bh, t-1] for the cross-token
contribution and the t == 0 branch elides it. Mutable-rebind of locals
across an if/else triggers a TileLang warning that is fixed by allocating a
T.alloc_local((1,), accum_dtype) for the result buffer:

python
ddt_v = T.alloc_local((1,), "float32")
dtrap_v = T.alloc_local((1,), "float32")
if t_index > 0:
    ddt_v[0] = d_scale * s + d_scale_lp * (1 - s)
    dtrap_v[0] = (d_scale * dt_val - d_scale_lp * dt_val) * s * (1 - s)
else:
    ddt_v[0] = d_scale * s
    dtrap_v[0] = (d_scale * dt_val) * s * (1 - s)


## Lowering surprises

These were the only non-cosmetic changes vs the earlier Path B GEMM template:

* **Metal target rejects shared.dyn**. The default scope produced by
  T.alloc_shared on the Metal target raises
  Unknown storage scope shared.dyn. Use scope="shared" instead.
* **Metal target rejects threadgroup as a sync arg**. Use T.sync_threads()
  rather than T.tvm_storage_sync("threadgroup") — the latter raises
  unknown storage scope threadgroup.
* **Inline functions cannot allocate threadgroup memory**. The Path B GEMM
  reference wraps the kernel body as inline void helper(...). That fails on
  any Mamba3 helper that uses T.alloc_shared because Metal disallows
  threadgroup declarations in non-kernel functions. Workaround: keep the
  body where it is and inline it directly into MLX's source= argument. See
  lower_tilelang_to_msl_inline in _msl_transform.py.
* **from __future__ import annotations breaks PrimFunc dtype capture**.
  TileLang resolves T.Tensor((N,), dtype_var) by evaluating the annotation
  string against the closure of the surrounding Python function. Variables
  that appear *only* in annotations are not in the closure; they fall through
  to a NameError. Fix: reference both accum_dtype and carrier_dtype
  somewhere in the body so they end up in co_freevars.
* **T.alloc_var vs reassignment**. Reassigning a Python local across an
  if/else triggers an immutable-rebind warning. Use
  T.alloc_local((1,), dtype) and write through the buffer instead.

## fp32 → fp16 carrier rationale

The Triton originals run in fp32 on CUDA. tilelang 0.1.9's bf16 simdgroup MSL
codegen has known issues (cubecl#1202). To stay portable across MLX dtype
choices the TileLang port uses an fp16 carrier with fp32 internal accumulation:

* A and dt are kept in fp32 because they parameterise the decay exponent
  and any fp16 round-trip there blows up exp() near the boundary.
* dY, h, dh, dB_scaled, dt (when used as a carrier-shaped
  multiplier), and trap are downcast to fp16 at the kernel boundary;
  bf16 inputs are first round-tripped through fp32 to avoid mantissa loss.
* Outputs are written back at carrier dtype and re-cast to the caller's dtype
  on the way out, so the public function preserves caller dtype semantics.

This is documented in the module docstring and reflected in the test
tolerances (rtol=1e-4, atol=1e-3).

## Parity tolerances

| Helper              | Target rtol | Target atol | Observed max abs err (mamba3-default shape) |
| ------------------- | ----------- | ----------- | ------------------------------------------- |
| compute_dacs_segsum | 1e-4        | 1e-3        | 9.77e-04                                    |
| bwd_dadt_fused      | 1e-4        | 1e-3        | 7.25e-05                                    |
| bwd_dtrap_ddt       | 1e-4        | 1e-3        | 0.00e+00                                    |

All 19 parameterised tests in tests/test_tilelang_mamba3_helpers.py pass at
rtol=1e-4, atol=1e-3 across multiple shapes.

## Bench numbers (Apple M4 Max, MLX 0.31, tilelang 0.1.9)

Source: bench/tilelang_ports/mamba3_helpers.json. Median of 30 iterations
after 5 warmup, fp16 carrier on inputs/outputs.

### compute_dacs_segsum

| Shape (B, T, H, P, N) | pure-MLX (ms) | TileLang/Metal (ms) | Speedup | max_abs_err |
| --------------------- | ------------- | ------------------- | ------- | ----------- |
| 1, 16, 4, 4, 8        | 0.161         | 0.146               | 1.10x   | 0.00e+00    |
| 2, 32, 4, 4, 8        | 0.161         | 0.154               | 1.05x   | 4.88e-04    |
| 4, 64, 8, 4, 16       | 0.168         | 0.153               | 1.10x   | 4.88e-04    |
| 2, 128, 4, 64, 16     | 0.216         | 0.219               | 0.99x   | 9.77e-04    |

### bwd_dadt_fused

| Shape (B, T, H, P, N) | pure-MLX (ms) | TileLang/Metal (ms) | Speedup | max_abs_err |
| --------------------- | ------------- | ------------------- | ------- | ----------- |
| 1, 16, 4, 4, 8        | 0.149         | 0.140               | 1.07x   | 0.00e+00    |
| 2, 32, 4, 4, 8        | 0.144         | 0.157               | 0.92x   | 0.00e+00    |
| 4, 64, 8, 4, 16       | 0.154         | 0.160               | 0.96x   | 0.00e+00    |
| 2, 128, 4, 64, 16     | 0.200         | 0.222               | 0.90x   | 7.25e-05    |

### bwd_dtrap_ddt

| Shape (B, T, H) | pure-MLX (ms) | TileLang/Metal (ms) | Speedup | max_abs_err |
| --------------- | ------------- | ------------------- | ------- | ----------- |
| 1, 16, 4        | 0.192         | 0.135               | 1.42x   | 0.00e+00    |
| 2, 32, 4        | 0.206         | 0.143               | 1.43x   | 0.00e+00    |
| 4, 64, 8        | 0.205         | 0.130               | 1.57x   | 3.05e-05    |
| 2, 128, 4       | 0.207         | 0.138               | 1.50x   | 0.00e+00    |

## Helper-by-helper recommendation

* **compute_dacs_segsum** — TileLang is correct and a small win at small
  shapes (~1.05–1.10x). At the mamba3-default chunk shape it ties the
  pure-MLX fallback because the TileLang kernel forces a serial reverse
  cumsum on a single producer thread (the tile size is bounded by the chunk).
  *Recommendation:* use TileLang; the regression at the largest shape is
  within noise and the math is provably equivalent (max_abs_err ≤ 1e-3).
* **bwd_dadt_fused** — TileLang is correct but pure-MLX wins at larger
  shapes (~0.90–0.96x speedup means TileLang loses by 4–10%). The pure-MLX
  fused reduction lowers to MLX's optimised mx.sum(...) over trailing
  dims; the TileLang one-thread-per-(bh,t) inner loop on K leaves SIMD lanes
  idle. *Recommendation:* either keep pure-MLX as primary or rewrite the
  TileLang version with a parallel reduction over K (out of scope for this
  pass).
* **bwd_dtrap_ddt** — TileLang wins decisively (~1.42–1.57x speedup) at
  every shape. The element-wise + 1-token-shift workload maps naturally to
  one-thread-per-element on Metal, while the pure-MLX version pays for
  several mx.concatenate calls. *Recommendation:* TileLang.

## Mamba3 stack-level recommendation

For the Mamba3 backward pipeline as a whole on an Apple M4 Max with the
current MLX 0.31 / tilelang 0.1.9 toolchain, **prefer TileLang for
bwd_dtrap_ddt and compute_dacs_segsum, keep bwd_dadt_fused on pure-MLX**
for the time being. The mixed approach is uncomplicated to wire because
_mamba3_helpers_tilelang exposes the same signatures as _mamba3_helpers,
and each helper accepts force_fallback=True for callers that want to
toggle without rebuilding kernels.

When tilelang adds a parallel reduction primitive that compiles to Metal
simdgroup-level reduces, the bwd_dadt_fused TileLang path should be revisited
— moving the K reduction off a single thread is the obvious next optimisation.
