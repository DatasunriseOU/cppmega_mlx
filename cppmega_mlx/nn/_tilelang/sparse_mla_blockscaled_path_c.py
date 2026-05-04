"""Path C E8M0 block-scaled Sparse-MLA QK probe via TileLang DSL.

This module is intentionally a lowering/status surface, not a production
Sparse-MLA forward. Path B already ships the direct-MSL MXFP8 Sparse-MLA
kernel in ``sparse_mla_blockscaled.py``. Path C becomes eligible only when the
Sparse-MLA QK tile can route through ``T.fp8_scaled_matmul`` with the same
logical MXFP8 layout:

* FP8 data is raw e4m3 bytes laid out as ``[B, S, H, D]`` / ``[B, SK, G, D]``.
* E8M0 scales are unswizzled K-axis block scales with one uint8 per 32 values.
* The DSL QK tile therefore uses ``A_scale[K / 32]`` and ``B_scale[K / 32]``.

Current apple-head TileLang can lower a square 32x32x64 control tile to Metal
simdgroup MMA with E8M0 decode in the staging path. The literal Sparse-MLA QK
shape remains ``M=1`` query row against top-k KV rows, which violates the
current Metal FP8 simdgroup tile constraints. The public status fails closed on
that shape so tests and benches do not report fake Path C support.

The second current blocker is K-chunking: Sparse-MLA wants global scale vectors
of size ``K / 32`` and per-``ko`` offsets into them, while the current TileLang
``T.fp8_scaled_matmul`` validator interprets the staged shared tile width
``BK`` as the complete contracted dimension and requires scale size ``BK / 32``.
Until the DSL can pass a scale subregion/offset for each K chunk, Path C cannot
represent the production block-scaled Sparse-MLA loop without changing the ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TILELANG_METAL_E8M0_SPARSE_MLA_TARGET = "metal"
E8M0_BLOCK_SIZE = 32
E8M0_SCALE_FORMAT = "e8m0_block_k32"
E8M0_LAYOUT = "logical_unswizzled_k_axis_blocks"

# TileLang's @T.prim_func decorator resolves shape constants by name from the
# function globals while make_blockscaled_sparse_mla_qk_kernel is running.
# These defaults are overwritten just before decoration; they exist so static
# tooling sees the same global contract that TileLang consumes dynamically.
_BSFP8_M = 1
_BSFP8_N = 16
_BSFP8_K = 64
_BSFP8_BM = 1
_BSFP8_BN = 16
_BSFP8_BK = 64
_BSFP8_SA = 2
_BSFP8_SB = 2
_BSFP8_B_SHAPE = (16, 64)
_BSFP8_B_SHARED_SHAPE = (16, 64)
_BSFP8_TRANSPOSE_B = True
_BSFP8_NUM_STAGES = 0


@dataclass(frozen=True)
class SparseMLABlockScaledPathCStatus:
    """Lowering status for the Path C E8M0 Sparse-MLA QK tile."""

    available: bool
    reason: str
    features: dict[str, int | bool | str]
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET
    m: int = 1
    n: int = 16
    k: int = 64
    transpose_B: bool = True
    scale_block_size: int = E8M0_BLOCK_SIZE
    scale_layout: str = E8M0_LAYOUT


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
        raise ValueError(f"E8M0 Sparse-MLA Path C shape values must be positive: {bad}")
    if K % E8M0_BLOCK_SIZE != 0:
        raise ValueError(f"E8M0 Sparse-MLA Path C requires K divisible by {E8M0_BLOCK_SIZE}; got K={K}")
    expected_scale_size = K // E8M0_BLOCK_SIZE
    if a_scale_size != expected_scale_size:
        raise ValueError(
            "E8M0 Sparse-MLA Path C A scale size must be "
            f"K/{E8M0_BLOCK_SIZE}={expected_scale_size}; got {a_scale_size}"
        )
    if b_scale_size != expected_scale_size:
        raise ValueError(
            "E8M0 Sparse-MLA Path C B scale size must be "
            f"K/{E8M0_BLOCK_SIZE}={expected_scale_size}; got {b_scale_size}"
        )


def make_blockscaled_sparse_mla_qk_kernel(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
) -> Any:
    """Build the E8M0 block-scaled QK tile used by Sparse-MLA.

    ``M`` is query rows, ``N`` is gathered top-k rows, and ``B`` is transposed
    as ``(N, K)`` to match the Path B Sparse-MLA QK loop. Scales are uint8 E8M0
    bytes indexed by contracted-K block, not by row or column.
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
        _BSFP8_M=M,
        _BSFP8_N=N,
        _BSFP8_K=K,
        _BSFP8_BM=BM,
        _BSFP8_BN=BN,
        _BSFP8_BK=BK,
        _BSFP8_SA=a_scale_size,
        _BSFP8_SB=b_scale_size,
        _BSFP8_B_SHAPE=b_shape,
        _BSFP8_B_SHARED_SHAPE=shared_b_shape,
        _BSFP8_TRANSPOSE_B=transpose_B,
        _BSFP8_NUM_STAGES=num_stages,
    )

    @T.prim_func
    def blockscaled_sparse_mla_qk_kernel(
        A_fp8: T.Tensor((_BSFP8_M, _BSFP8_K), "float8_e4m3"),
        A_scale: T.Tensor((_BSFP8_SA,), "uint8"),
        B_fp8: T.Tensor(_BSFP8_B_SHAPE, "float8_e4m3"),
        B_scale: T.Tensor((_BSFP8_SB,), "uint8"),
        C: T.Tensor((_BSFP8_M, _BSFP8_N), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(_BSFP8_N, _BSFP8_BN),
            T.ceildiv(_BSFP8_M, _BSFP8_BM),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((_BSFP8_BM, _BSFP8_BK), "float8_e4m3", scope="shared")
            B_shared = T.alloc_shared(_BSFP8_B_SHARED_SHAPE, "float8_e4m3", scope="shared")
            C_local = T.alloc_fragment((_BSFP8_BM, _BSFP8_BN), "float32")
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(_BSFP8_K, _BSFP8_BK), num_stages=_BSFP8_NUM_STAGES):
                T.copy(A_fp8[by * _BSFP8_BM, ko * _BSFP8_BK], A_shared)
                if _BSFP8_TRANSPOSE_B:
                    T.copy(B_fp8[bx * _BSFP8_BN, ko * _BSFP8_BK], B_shared)
                else:
                    T.copy(B_fp8[ko * _BSFP8_BK, bx * _BSFP8_BN], B_shared)
                T.fp8_scaled_matmul(
                    A_shared,
                    A_scale,
                    B_shared,
                    B_scale,
                    C_local,
                    transpose_B=_BSFP8_TRANSPOSE_B,
                    scale_format=E8M0_SCALE_FORMAT,
                    scale_block_size=E8M0_BLOCK_SIZE,
                )
            T.copy(C_local, C[by * _BSFP8_BM, bx * _BSFP8_BN])

    return blockscaled_sparse_mla_qk_kernel


def lower_blockscaled_sparse_mla_qk_msl(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> str:
    """Lower the Path C E8M0 Sparse-MLA QK probe and return MSL source."""

    import tilelang
    from tilelang import tvm

    prim = make_blockscaled_sparse_mla_qk_kernel(
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


def blockscaled_sparse_mla_qk_msl_features(msl: str) -> dict[str, int | bool | str]:
    """Return source markers used to guard E8M0 scale and fast-path semantics."""

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
        "e8m0_exp2": body.count("exp2"),
        "e8m0_bias_subtract_127": body.count("- 127"),
        "e8m0_sentinel_255": body.count("== 255"),
        "e8m0_zero_sentinel": body.count("== 0"),
        "k_block_shift_5": body.count(">> 5"),
        "k_block_div_32": body.count("/ 32"),
        "A_scale_collapsed_zero": body.count("A_scale[0]"),
        "B_scale_collapsed_zero": body.count("B_scale[0]"),
        "float_a_val": "float a_val" in lowered,
        "float_b_val": "float b_val" in lowered,
        "threadgroup_half": "threadgroup half" in lowered,
        "scale_format": E8M0_SCALE_FORMAT,
        "scale_block_size": E8M0_BLOCK_SIZE,
        "scale_axis": "contracted_k",
        "scale_layout": E8M0_LAYOUT,
    }


def blockscaled_sparse_mla_qk_path_c_status(
    *,
    M: int = 1,
    N: int = 16,
    K: int = 64,
    BM: int = 1,
    BN: int = 16,
    BK: int = 64,
    a_scale_size: int = 2,
    b_scale_size: int = 2,
    transpose_B: bool = True,
    num_stages: int = 0,
    target: str = TILELANG_METAL_E8M0_SPARSE_MLA_TARGET,
) -> SparseMLABlockScaledPathCStatus:
    """Fail-closed availability probe for the E8M0 Sparse-MLA Path C QK tile."""

    ok, reason = _tilelang_available()
    if not ok:
        return SparseMLABlockScaledPathCStatus(
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
        msl = lower_blockscaled_sparse_mla_qk_msl(
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
        return SparseMLABlockScaledPathCStatus(
            available=False,
            reason=(
                "TileLang Metal lowering failed for E8M0 Sparse-MLA QK shape: "
                f"{type(exc).__name__}: {exc}"
            ),
            features={},
            target=target,
            m=M,
            n=N,
            k=K,
            transpose_B=transpose_B,
        )

    features = blockscaled_sparse_mla_qk_msl_features(msl)
    has_fast_path = bool(features["simdgroup_multiply_accumulate"])
    has_scale_refs = bool(features["A_scale_refs"]) and bool(features["B_scale_refs"])
    has_scale_signature = bool(features["signature_has_A_scale"]) and bool(features["signature_has_B_scale"])
    has_e8m0_decode = bool(
        features["e8m0_exp2"]
        and features["e8m0_bias_subtract_127"]
        and features["e8m0_sentinel_255"]
    )
    has_k_block_index = bool(features["k_block_shift_5"] or features["k_block_div_32"])
    has_collapsed_scale = bool(features["A_scale_collapsed_zero"] or features["B_scale_collapsed_zero"])
    has_scalar_fallback = bool(features["float_a_val"]) or bool(features["float_b_val"])
    shape_eligible = M >= 8 and BM >= 8 and N >= 8 and BN >= 8

    if (
        shape_eligible
        and has_fast_path
        and has_scale_refs
        and has_scale_signature
        and has_e8m0_decode
        and has_k_block_index
        and not has_collapsed_scale
        and not has_scalar_fallback
    ):
        return SparseMLABlockScaledPathCStatus(
            available=True,
            reason=(
                "TileLang Path C E8M0 Sparse-MLA QK probe lowers through "
                "T.fp8_scaled_matmul to Metal simdgroup MMA with K/32 E8M0 scale loads"
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
    if not has_e8m0_decode:
        blockers.append("E8M0 exp2(byte - 127) decode markers missing")
    if not has_k_block_index:
        blockers.append("scale operands are not indexed by K/32")
    if has_collapsed_scale:
        blockers.append("E8M0 scale operands collapsed to [0]")
    if has_scalar_fallback:
        blockers.append("scalar fallback markers present")
    if not shape_eligible:
        blockers.append("Sparse-MLA M=1/topk tile violates current Metal FP8 simdgroup tile constraints")
    return SparseMLABlockScaledPathCStatus(
        available=False,
        reason="TileLang Path C E8M0 Sparse-MLA QK is not safe to dispatch: " + "; ".join(blockers),
        features=features,
        target=target,
        m=M,
        n=N,
        k=K,
        transpose_B=transpose_B,
    )


__all__ = [
    "E8M0_BLOCK_SIZE",
    "E8M0_LAYOUT",
    "E8M0_SCALE_FORMAT",
    "SparseMLABlockScaledPathCStatus",
    "TILELANG_METAL_E8M0_SPARSE_MLA_TARGET",
    "blockscaled_sparse_mla_qk_msl_features",
    "blockscaled_sparse_mla_qk_path_c_status",
    "lower_blockscaled_sparse_mla_qk_msl",
    "make_blockscaled_sparse_mla_qk_kernel",
]
