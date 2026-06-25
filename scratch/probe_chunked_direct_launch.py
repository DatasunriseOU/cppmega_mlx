"""PROBE: per-kernel direct_pipeline_launches for the chunked B2->B1->B0->fused bwd.

Drives mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd (the production chunked
backward) through the MLX custom_function and reads the native MLX-TVM-FFI debug
counters (direct_pipeline_launches / direct_device_launches / launches) so we can
see whether the chunked kernels take the direct-pipeline path or fall back to the
host-wrapper TVMFFIFunctionCall.

Runs under memguard 70 (mandatory). Production cache path: this process loads the
cached kernels via from_database UNLESS TILELANG_DISABLE_CACHE=1 is set in the env.

No fabrication: the counters are read straight from the native debug_state().
"""
import os
import sys
import threading
import time

_MEMGUARD_LIMIT_KB = 70 * 1024 * 1024
_PEAK = 0


def _rss_kb():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def _guard():
    global _PEAK
    while True:
        r = _rss_kb()
        if r > _PEAK:
            _PEAK = r
        if r > _MEMGUARD_LIMIT_KB:
            sys.stderr.write(f"[memguard70] KILL rss_kb={r}\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.25)


threading.Thread(target=_guard, daemon=True).start()
sys.path.insert(0, "/Volumes/external/sources/cppmega.mlx")
sys.path.insert(0, "/Volumes/external/sources/tilelang")

import numpy as np
import mlx.core as mx

from tilelang.jit.adapter import _mlx_tvm_ffi as _mlxffi

CACHE_DISABLED = os.environ.get("TILELANG_DISABLE_CACHE", "0") == "1"
print(f"TILELANG_DISABLE_CACHE={os.environ.get('TILELANG_DISABLE_CACHE','<unset>')} "
      f"(cache_disabled={CACHE_DISABLED})")


def _counters():
    return _mlxffi.debug_state()


def _snap(label):
    st = _counters()
    print(f"[{label}] launches={st.get('launches')} "
          f"direct_device_launches={st.get('direct_device_launches')} "
          f"direct_pipeline_launches={st.get('direct_pipeline_launches')} "
          f"direct_compute_encoder_launches={st.get('direct_compute_encoder_launches')}")
    return st


from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd,
)

# nam56r config-exact dims.
b, seq, H, P, N, chunk = 1, 128, 128, 64, 64, 64
rng = np.random.RandomState(0)


def f32(*shape, s=0.1):
    return mx.array((rng.randn(*shape) * s).astype(np.float32))


x = f32(b, seq, H, P)
B = f32(b, seq, H, N)
C = f32(b, seq, H, N)
z = f32(b, seq, H, P, s=0.5)
A_head = (-rng.rand(H)).astype(np.float32)
A = mx.array(np.broadcast_to(A_head[None, None, :], (b, seq, H)).copy())
dt = mx.array((rng.rand(b, seq, H) * 0.05).astype(np.float32))
D = mx.array((rng.randn(H)).astype(np.float32))
h0 = f32(b, H, P, N)
cot_y = mx.array((rng.randn(b, seq, H, P) * 0.1).astype(np.float32))


def fwd_y(x, B, C, z, A, dt, D, h0):
    out = mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
    return out[0]


# WARMUP run (forces compile/load + a full fwd+bwd dispatch). We reset counters
# AFTER warmup so the measured pass reflects steady-state production dispatch.
print("\n=== WARMUP (compile/load + first dispatch) ===")
_mlxffi.reset_debug_state()
y, h_last, cb, dA_cumsum, prev_states, y_stash = (
    mamba3_mimo_apply_with_state_path_c_fwd_path_c_bwd(x, B, C, z, A, dt, D, h0)
)
mx.eval(y, h_last)
_, grads = mx.vjp(fwd_y, (x, B, C, z, A, dt, D, h0), (cot_y,))
mx.eval(*grads)
_snap("warmup_after")

print("\n=== MEASURED PASS (counters reset, then one full fwd+bwd) ===")
_mlxffi.reset_debug_state()
_snap("measured_before")
y2 = fwd_y(x, B, C, z, A, dt, D, h0)
mx.eval(y2)
_snap("after_fwd")
_, grads2 = mx.vjp(fwd_y, (x, B, C, z, A, dt, D, h0), (cot_y,))
mx.eval(*grads2)
final = _snap("after_bwd")

print(f"\nPEAK_RSS_KB={_PEAK} (~{_PEAK/1048576:.3f}GB) memguard70=ON")
print("PROBE_DIRECT_PIPELINE_LAUNCHES=", final.get("direct_pipeline_launches"))
print("PROBE_DIRECT_DEVICE_LAUNCHES=", final.get("direct_device_launches"))
print("PROBE_TOTAL_LAUNCHES=", final.get("launches"))
