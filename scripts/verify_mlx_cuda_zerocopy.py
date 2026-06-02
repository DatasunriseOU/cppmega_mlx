#!/usr/bin/env python
"""Verify MLX-CUDA -> TileLang(target='cuda') zero-copy DLPack round-trip.

Run on a CUDA host (gb10/sm_121) with the CUDA MLX build:

    CPPMEGA_TILELANG_CUDA_ZEROCOPY=1 \
    /home/dave/cppmega-venv/bin/python scripts/verify_mlx_cuda_zerocopy.py

It builds a small MLX-CUDA array, exports it as a kDLCUDA DLPack capsule via
``cppmega_mlx.nn._tilelang._cuda_zerocopy`` (NO host roundtrip), imports it with
``tvm.runtime.from_dlpack``, runs a trivial elementwise TileLang target='cuda'
kernel (out = inp * 2 + 1), and checks numeric parity against the MLX reference.

RULE #1: any failure RAISES with where+what; there is no copy fallback.
The script also A/B-tests device_type 2 (kDLCUDA) vs 13 (kDLCUDAManaged) to report
which one tvm-ffi from_dlpack accepts.
"""

import os
import sys

import mlx.core as mx
import numpy as np


def _build_double_plus_one_kernel(n: int):
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),  # noqa: N803
        B: T.Tensor((n,), "float32"),  # noqa: N803
    ):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for i in T.serial(128):
                idx = bx * 128 + i
                if idx < n:
                    B[idx] = A[idx] * T.float32(2.0) + T.float32(1.0)

    return tilelang.compile(main, target="cuda", out_idx=None)


def _run_for_device_type(device_type: int, n: int = 256) -> tuple[bool, str]:
    os.environ["CPPMEGA_TILELANG_CUDA_DLPACK_DEVICE_TYPE"] = str(device_type)
    from cppmega_mlx.nn._tilelang._cuda_zerocopy import mlx_cuda_array_to_tvm_tensor

    src = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(src)
    ref = np.array(src) * 2.0 + 1.0

    a_tvm = mlx_cuda_array_to_tvm_tensor(src)
    out = mx.zeros((n,), dtype=mx.float32)
    mx.eval(out)
    b_tvm = mlx_cuda_array_to_tvm_tensor(out)

    kernel = _build_double_plus_one_kernel(n)
    kernel(a_tvm, b_tvm)

    import torch

    torch.cuda.synchronize()
    mx.eval(out)
    got = np.array(out)
    max_abs = float(np.max(np.abs(got - ref)))
    ok = bool(np.allclose(got, ref, atol=1e-4, rtol=1e-4))
    return ok, f"device_type={device_type}: ok={ok} max_abs_err={max_abs:.3e}"


def main() -> int:
    if not (getattr(mx, "cu", None) and mx.cu.is_available()):
        raise RuntimeError(
            "verify_mlx_cuda_zerocopy: MLX CUDA backend is not available on this host."
        )
    results = []
    last_exc = None
    for dt in (2, 13):
        try:
            ok, msg = _run_for_device_type(dt)
            results.append((dt, ok, msg))
            print("PASS" if ok else "FAIL", msg)
        except Exception as exc:  # noqa: BLE001 - report which device_type was rejected
            last_exc = exc
            print(f"REJECTED device_type={dt}: {type(exc).__name__}: {exc}")
    accepted = [dt for dt, ok, _ in results if ok]
    if not accepted:
        raise RuntimeError(
            f"verify_mlx_cuda_zerocopy: no CUDA DLPack device_type round-tripped "
            f"zero-copy with numeric parity (last error: {last_exc})."
        )
    print(f"ZERO-COPY VERIFIED: accepted device_type(s)={accepted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
