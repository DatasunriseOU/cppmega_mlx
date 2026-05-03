"""Path C FP8 sparse-MLA QK probe via TileLang DSL.

This module is intentionally scheduler/status glue, not a production Sparse-MLA
forward.  Path B already ships the direct-MSL FP8 Sparse-MLA kernel in
``sparse_mla_fp8.py``.  Path C must route the Sparse-MLA QK tile through
``T.fp8_scaled_matmul`` before it can be performance- or parity-eligible.

Current apple-head TileLang can lower a square 32x32 FP8 scaled matmul to the
Metal simdgroup path with explicit scale loads, but the literal Sparse-MLA QK
shape (M=1 query row against top-k transposed KV rows) falls back to scalar code
and can drop the scale operands from the emitted kernel.  The public status
surface below fails closed on that shape so benches/tests do not report fake
Path C support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TILELANG_METAL_FP8_SPARSE_MLA_TARGET = "metal"


@dataclass(frozen=True)
class SparseMLAFp8PathCStatus:
    """Lowering status for the Path C TileLang FP8 Sparse-MLA QK tile."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET
    m: int = 1
    n: int = 16
    k: int = 64
    transpose_B: bool = True


def _tilelang_available() -> tuple[bool, str]:
    try:
        import tilelang  # noqa: F401
        from tilelang import tvm as _tvm  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:  # pragma: no cover - hosts without TileLang
        return False, f"tilelang import failed: {exc}"
    return True, "tilelang importable"


def _validate_shape(
    *,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    a_scale_size: int,
    b_scale_size: int,
) -> None:
    values = {
        "M": M,
        "N": N,
        "K": K,
        "BM": BM,
        "BN": BN,
        "BK": BK,
        "a_scale_size": a_scale_size,
        "b_scale_size": b_scale_size,
    }
    bad = {name: value for name, value in values.items() if value <= 0}
    if bad:
        raise ValueError(f"FP8 Sparse-MLA Path C shape values must be positive: {bad}")


def make_fp8_sparse_mla_qk_kernel(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
) -> Any:
    """Build the QK tile used by FP8 Sparse-MLA.

    ``M`` is the number of query rows, ``N`` is the sparse top-k tile, and
    ``B`` is transposed as ``(N, K)`` to match Path B/audiohacking vecmat scale
    semantics: A scale is per query row or scalar, B scale is per gathered KV row
    or scalar.
    """

    _validate_shape(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        a_scale_size=a_scale_size,
        b_scale_size=b_scale_size,
    )

    import tilelang.language as T

    b_shape = (N, K) if transpose_B else (K, N)
    shared_b_shape = (BN, BK) if transpose_B else (BK, BN)

    g = globals()
    g.update(
        _SMFP8_M=M,
        _SMFP8_N=N,
        _SMFP8_K=K,
        _SMFP8_BM=BM,
        _SMFP8_BN=BN,
        _SMFP8_BK=BK,
        _SMFP8_SA=a_scale_size,
        _SMFP8_SB=b_scale_size,
        _SMFP8_B_SHAPE=b_shape,
        _SMFP8_B_SHARED_SHAPE=shared_b_shape,
        _SMFP8_TRANSPOSE_B=transpose_B,
        _SMFP8_NUM_STAGES=num_stages,
    )

    @T.prim_func
    def fp8_sparse_mla_qk_kernel(
        A_fp8: T.Tensor((_SMFP8_M, _SMFP8_K), "float8_e4m3"),
        A_scale: T.Tensor((_SMFP8_SA,), "float32"),
        B_fp8: T.Tensor(_SMFP8_B_SHAPE, "float8_e4m3"),
        B_scale: T.Tensor((_SMFP8_SB,), "float32"),
        C: T.Tensor((_SMFP8_M, _SMFP8_N), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_SMFP8_N, _SMFP8_BN),
            T.ceildiv(_SMFP8_M, _SMFP8_BM),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((_SMFP8_BM, _SMFP8_BK), "float8_e4m3", scope="shared")
            B_shared = T.alloc_shared(_SMFP8_B_SHARED_SHAPE, "float8_e4m3", scope="shared")
            C_local = T.alloc_fragment((_SMFP8_BM, _SMFP8_BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(_SMFP8_K, _SMFP8_BK), num_stages=_SMFP8_NUM_STAGES):
                T.copy(A_fp8[by * _SMFP8_BM, ko * _SMFP8_BK], A_shared)
                if _SMFP8_TRANSPOSE_B:
                    T.copy(B_fp8[bx * _SMFP8_BN, ko * _SMFP8_BK], B_shared)
                else:
                    T.copy(B_fp8[ko * _SMFP8_BK, bx * _SMFP8_BN], B_shared)
                T.fp8_scaled_matmul(
                    A_shared,
                    A_scale,
                    B_shared,
                    B_scale,
                    C_local,
                    transpose_B=_SMFP8_TRANSPOSE_B,
                )
            T.copy(C_local, C[by * _SMFP8_BM, bx * _SMFP8_BN])

    return fp8_sparse_mla_qk_kernel


def lower_fp8_sparse_mla_qk_msl(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> str:
    """Lower the Path C FP8 Sparse-MLA QK probe and return MSL source."""

    import tilelang
    from tilelang import tvm

    prim = make_fp8_sparse_mla_qk_kernel(
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        a_scale_size=a_scale_size,
        b_scale_size=b_scale_size,
        transpose_B=transpose_B,
        num_stages=num_stages,
    )
    artifact = tilelang.lower(prim, target=tvm.target.Target(target))
    if hasattr(artifact, "kernel_source"):
        return str(artifact.kernel_source)
    if hasattr(artifact, "rt_mod") and hasattr(artifact.rt_mod, "get_source"):
        return str(artifact.rt_mod.get_source())
    return str(artifact)


def fp8_sparse_mla_qk_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers used to guard Path C scale and fast-path semantics."""

    body = msl.split("kernel void", 1)[-1] if "kernel void" in msl else msl
    signature = body.split("{", 1)[0]
    lowered = body.lower()
    return {
        "kernel_void": msl.count("kernel void"),
        "simdgroup_multiply_accumulate": msl.count("simdgroup_multiply_accumulate"),
        "simdgroup_load": msl.count("simdgroup_load"),
        "simdgroup_store": msl.count("simdgroup_store"),
        "fp8_e4m3_decode_helper": msl.count("__tvm_fp8_e4m3_to_half"),
        "A_scale_refs": body.count("A_scale["),
        "B_scale_refs": body.count("B_scale["),
        "signature_has_A_scale": "A_scale" in signature,
        "signature_has_B_scale": "B_scale" in signature,
        "float_a_val": "float a_val" in lowered,
        "float_b_val": "float b_val" in lowered,
        "threadgroup_half": "threadgroup half" in lowered,
    }


def fp8_sparse_mla_qk_path_c_status(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 1,
    b_scale_size: int = 16,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_FP8_SPARSE_MLA_TARGET,
) -> SparseMLAFp8PathCStatus:
    """Fail-closed availability probe for the FP8 Sparse-MLA Path C QK tile."""

    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLAFp8PathCStatus(
            available=False,
            reason=reason,
            features={},
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    try:
        msl = lower_fp8_sparse_mla_qk_msl(
            M=M,
            N=N,
            K=K,
            BM=BM,
            BN=BN,
            BK=BK,
            a_scale_size=a_scale_size,
            b_scale_size=b_scale_size,
            transpose_B=transpose_B,
            num_stages=num_stages,
            target=target,
        )
    except Exception as exc:
        return SparseMLAFp8PathCStatus(
            available=False,
            reason=f"TileLang Metal lowering failed for FP8 Sparse-MLA QK shape: {type(exc).__name__}: {exc}",
            features={},
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    features = fp8_sparse_mla_qk_msl_features(msl)
    has_fast_path = bool(features["simdgroup_multiply_accumulate"])
    has_scale_refs = bool(features["A_scale_refs"]) and bool(features["B_scale_refs"])
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(features["signature_has_B_scale"])
    has_scalar_fallback = bool(features["float_a_val"]) or bool(features["float_b_val"])
    if has_fast_path and has_scale_refs and has_scale_signature and not has_scalar_fallback:
        return SparseMLAFp8PathCStatus(
            available=True,
            reason=(
                "TileLang Path C FP8 Sparse-MLA QK probe lowers through "
                "T.fp8_scaled_matmul to Metal simdgroup MMA with scale loads"
            ),
            features=features,
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    blockers: list[str] = []
    if not has_fast_path:
        blockers.append("no simdgroup_multiply_accumulate")
    if not has_scale_refs or not has_scale_signature:
        blockers.append("scale operands disappeared from emitted MSL")
    if has_scalar_fallback:
        blockers.append("scalar fallback markers present")
    if M < 8 or BM < 8:
        blockers.append("Sparse-MLA M=1/topk tile violates current Metal FP8 simdgroup tile constraints")
    return SparseMLAFp8PathCStatus(
        available=False,
        reason="TileLang Path C FP8 Sparse-MLA QK is not safe to dispatch: " + "; ".join(blockers),
        features=features,
        target=target,
        m=M,
        n=N,
        k=K,
        transpose_B=transpose_B,
    )


__all__ = [
    "SparseMLAFp8PathCStatus",
    "TILELANG_METAL_FP8_SPARSE_MLA_TARGET",
    "fp8_sparse_mla_qk_msl_features",
    "fp8_sparse_mla_qk_path_c_status",
    "lower_fp8_sparse_mla_qk_msl",
    "make_fp8_sparse_mla_qk_kernel",
]
