"""Path C FP8 vecmat/GEMV via TileLang DSL lowering.

This module is the TileLang-DSL counterpart to the hand-written Path B MSL
``fp8_scaled_vecmat`` kernel in :mod:`cppmega_mlx.nn._tilelang.fp8_msl_kernels`.
It is intentionally scoped to the inference shape that matters for this lane:
``M == 1``, ``B`` already transposed as ``(N, K)``, and e4m3 storage.

The current TileLang Metal lowering can express the correct thread-level GEMV
shape with ``tvm_thread_allreduce`` across K and now lowers the single-warp
sum to literal Metal ``simd_sum``. Path B remains faster because it emits
packed uint32 global loads instead of scalar FP8 byte decodes in the hot loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TILELANG_METAL_VECMAT_TARGET = "metal -thread_warp_size=32"


@dataclass(frozen=True)
class FP8VecmatPathCStatus:
    """Runtime/lowering status for the Path C TileLang FP8 vecmat kernel."""

    available: bool
    reason: str
    target: str = TILELANG_METAL_VECMAT_TARGET
    transpose_B: bool = True
    m_equals_1: bool = True


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - hosts without TileLang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def fp8_vecmat_path_c_status() -> FP8VecmatPathCStatus:
    """Return whether TileLang is importable for Path C vecmat lowering."""

    ok, reason = _tilelang_available()
    if not ok:
        return FP8VecmatPathCStatus(available=False, reason=reason)
    return FP8VecmatPathCStatus(
        available=True,
        reason="FP8 vecmat Path C TileLang DSL lowering is available",
    )


def _validate_shape(*, N: int, K: int, outputs_per_block: int, reduce_threads: int, vec: int) -> None:
    if N <= 0 or K <= 0:
        raise ValueError(f"N and K must be positive; got N={N}, K={K}")
    if outputs_per_block <= 0:
        raise ValueError(f"outputs_per_block must be positive; got {outputs_per_block}")
    if reduce_threads <= 0:
        raise ValueError(f"reduce_threads must be positive; got {reduce_threads}")
    if vec <= 0:
        raise ValueError(f"vec must be positive; got {vec}")


def make_fp8_vecmat_reduce_kernel(
    *,
    N: int,
    K: int,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
    vectorized_loads: bool = False,
) -> Any:
    """Build a shape-specialized FP8 vecmat reducer.

    Inputs match Path B's vecmat contract:

    * ``A`` is ``(1, K)`` e4m3.
    * ``B`` is ``(N, K)`` e4m3, i.e. already transposed.
    * ``C`` is ``(1, N)`` fp32.

    ``vectorized_loads=True`` mirrors upstream TileLang GEMV examples by
    staging a small local vector with ``T.vectorized(vec)``. On current
    apple-head Metal lowering this is a probe, not a guarantee of packed
    uint32 MSL loads.
    """

    _validate_shape(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
    )

    import tilelang.language as T

    block_k = reduce_threads * vec
    g = globals()
    g.update(
        _FP8_VM_N=N,
        _FP8_VM_K=K,
        _FP8_VM_NP=outputs_per_block,
        _FP8_VM_RT=reduce_threads,
        _FP8_VM_VEC=vec,
        _FP8_VM_BLOCK_K=block_k,
    )

    if vectorized_loads:

        @T.prim_func
        def fp8_vecmat_reduce(
            A: T.Tensor((1, _FP8_VM_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((_FP8_VM_N, _FP8_VM_K), "float8_e4m3"),
            B_scale: T.Tensor((1,), "float32"),
            C: T.Tensor((1, _FP8_VM_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_FP8_VM_N, _FP8_VM_NP),
                threads=(_FP8_VM_RT, _FP8_VM_NP),
            ) as bx:
                A_local = T.alloc_local((_FP8_VM_VEC,), "float8_e4m3")
                B_local = T.alloc_local((_FP8_VM_VEC,), "float8_e4m3")
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                kr = T.get_thread_binding(0)
                ni = T.get_thread_binding(1)
                col = bx * _FP8_VM_NP + ni
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_FP8_VM_K, _FP8_VM_BLOCK_K)):
                    for v in T.vectorized(_FP8_VM_VEC):
                        k = ko * _FP8_VM_BLOCK_K + kr * _FP8_VM_VEC + v
                        if col < _FP8_VM_N and k < _FP8_VM_K:
                            A_local[v] = A[0, k]
                            B_local[v] = B[col, k]
                    for v in T.serial(_FP8_VM_VEC):
                        accum[0] += T.cast(A_local[v], "float32") * T.cast(
                            B_local[v], "float32"
                        )
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accum[0],
                            True,
                            reduced[0],
                            kr,
                            dtype="handle",
                        )
                    )
                if kr == 0 and col < _FP8_VM_N:
                    C[0, col] = reduced[0] * A_scale[0] * B_scale[0]

    else:

        @T.prim_func
        def fp8_vecmat_reduce(
            A: T.Tensor((1, _FP8_VM_K), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((_FP8_VM_N, _FP8_VM_K), "float8_e4m3"),
            B_scale: T.Tensor((1,), "float32"),
            C: T.Tensor((1, _FP8_VM_N), "float32"),
        ):
            with T.Kernel(
                T.ceildiv(_FP8_VM_N, _FP8_VM_NP),
                threads=(_FP8_VM_RT, _FP8_VM_NP),
            ) as bx:
                accum = T.alloc_local((1,), "float32")
                reduced = T.alloc_local((1,), "float32")
                kr = T.get_thread_binding(0)
                ni = T.get_thread_binding(1)
                col = bx * _FP8_VM_NP + ni
                T.clear(accum)
                for ko in T.serial(T.ceildiv(_FP8_VM_K, _FP8_VM_BLOCK_K)):
                    for v in T.serial(_FP8_VM_VEC):
                        k = ko * _FP8_VM_BLOCK_K + kr * _FP8_VM_VEC + v
                        if col < _FP8_VM_N and k < _FP8_VM_K:
                            accum[0] += T.cast(A[0, k], "float32") * T.cast(
                                B[col, k], "float32"
                            )
                with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, "float32")]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accum[0],
                            True,
                            reduced[0],
                            kr,
                            dtype="handle",
                        )
                    )
                if kr == 0 and col < _FP8_VM_N:
                    C[0, col] = reduced[0] * A_scale[0] * B_scale[0]

    try:
        from tilelang.transform.simplify import apply_simplify

        return apply_simplify(fp8_vecmat_reduce)
    except Exception:
        return fp8_vecmat_reduce


def lower_fp8_vecmat_msl(
    *,
    N: int = 4096,
    K: int = 4096,
    outputs_per_block: int = 4,
    reduce_threads: int = 32,
    vec: int = 4,
    vectorized_loads: bool = False,
    target: str = TILELANG_METAL_VECMAT_TARGET,
) -> str:
    """Lower the Path C vecmat reducer and return the emitted MSL source."""

    import tilelang
    from tvm.target import Target

    prim = make_fp8_vecmat_reduce_kernel(
        N=N,
        K=K,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        vectorized_loads=vectorized_loads,
    )
    artifact = tilelang.lower(prim, target=Target(target))
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    if hasattr(artifact, "rt_mod") and hasattr(artifact.rt_mod, "get_source"):
        return str(artifact.rt_mod.get_source())
    return str(artifact)


def fp8_vecmat_msl_features(msl: str) -> dict[str, int]:
    """Return feature counters used by tests and bench receipts."""

    lowered = msl.lower()
    scalar_decode_sites = msl.count("__tvm_fp8_e4m3_to_half(")
    return {
        "kernel_void": msl.count("kernel void"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "tvm_thread_allreduce": msl.count("tvm_thread_allreduce"),
        "simd_shuffle_down": msl.count("simd_shuffle_down"),
        "simd_sum": msl.count("simd_sum"),
        "reinterpret_cast": msl.count("reinterpret_cast"),
        "device_const_uint": msl.count("device const uint"),
        "uint_pointer": msl.count("uint*"),
        "uchar4": lowered.count("uchar4"),
        "scalar_fp8_byte_decode": scalar_decode_sites,
        "scalar_fp8_byte_decode_calls": max(0, scalar_decode_sites - 1),
    }


def fp8_vecmat_msl_blockers(msl: str) -> dict[str, Any]:
    """Summarize why the generated Path C MSL still misses Path B's fast path."""

    features = fp8_vecmat_msl_features(msl)
    missing: list[str] = []
    if features["reinterpret_cast"] == 0 or features["device_const_uint"] == 0:
        missing.append("packed_uint32_fp8_loads")
    if features["simd_sum"] == 0:
        missing.append("metal_simd_sum_reduction")
    if features["scalar_fp8_byte_decode_calls"] > 0:
        missing.append("lut_or_packed_decode_instead_of_scalar_fp8_helper_calls")
    return {
        "path_b_fast_path_ready": not missing,
        "missing": missing,
        "generated_features": features,
        "required_fast_path": {
            "packed_uint32_fp8_loads": "reinterpret_cast<device const uint*> loads for 4 FP8 bytes",
            "metal_simd_sum_reduction": "literal Metal simd_sum(sum) reduction",
            "no_scalar_fp8_helper_calls": "avoid per-byte __tvm_fp8_e4m3_to_half calls in the hot loop",
        },
    }


__all__ = [
    "FP8VecmatPathCStatus",
    "TILELANG_METAL_VECMAT_TARGET",
    "fp8_vecmat_msl_blockers",
    "fp8_vecmat_msl_features",
    "fp8_vecmat_path_c_status",
    "lower_fp8_vecmat_msl",
    "make_fp8_vecmat_reduce_kernel",
]
