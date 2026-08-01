"""Native TileLang scatter for the paged KV offsets (decode) write path.

The pure-MLX compatibility scatter (``serving.scatter_paged_kv_offsets``)
rewrites one pool slice per new token through a Python host loop. This module
compiles two TileLang ``@T.prim_func`` kernels through the native ``tvm_ffi``
backend: one copies the existing pool into caller-owned outputs and one writes
the precomputed token->(block, slot) updates.

Contract:

* Caller-owned outputs (``k_out`` / ``v_out``) following the
  ``NativeTileLangKernel`` / mamba3-helper convention: outputs are passed as
  the trailing positional arguments and the same objects are returned.
* Fail-closed: unsupported dtype, missing tilelang or Metal raise
  ``PagedKvNativeUnavailable`` instead of silently degrading. Callers that
  want a fallback choose it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import mlx.core as mx

from cppmega_mlx.nn._tilelang import _msl_transform


class PagedKvNativeUnavailable(RuntimeError):
    """Raised when the native paged KV scatter cannot dispatch fail-closed."""


@dataclass(frozen=True)
class PagedKvNativeStatus:
    available: bool
    reason: str


_SUPPORTED_DTYPES = {mx.float32, mx.float16, mx.bfloat16}
_DTYPE_TO_TILELANG = {
    mx.float32: "float32",
    mx.float16: "float16",
    mx.bfloat16: "bfloat16",
}


def paged_kv_native_status() -> PagedKvNativeStatus:
    """Return whether the native TileLang paged KV scatter can dispatch."""

    if not _msl_transform.can_run_metal():
        return PagedKvNativeStatus(
            available=False,
            reason="MLX Metal backend is not available on the default GPU device",
        )
    try:
        import tilelang  # noqa: F401
        import tilelang.language as _T  # noqa: F401
    except Exception as exc:
        return PagedKvNativeStatus(
            available=False, reason=f"tilelang import failed: {exc}"
        )
    return PagedKvNativeStatus(available=True, reason="native paged KV scatter ready")


def _compile_tvm_ffi_kernel(prim_func: Any, *, out_idx: list[int]) -> Any:
    import tilelang

    return tilelang.compile(
        prim_func,
        target=_msl_transform._as_metal_target("metal"),
        execution_backend="tvm_ffi",
        out_idx=out_idx,
    )


@lru_cache(maxsize=128)
def _copy_kernel_for(pool_elems: int, dtype: str, threads: int):
    """Build & cache a flat pool copy kernel (k_out=k_pool, v_out=v_pool)."""

    import tilelang.language as T

    @T.prim_func
    def pool_copy(
        k_pool: T.Tensor((pool_elems,), dtype),
        v_pool: T.Tensor((pool_elems,), dtype),
        k_out: T.Tensor((pool_elems,), dtype),
        v_out: T.Tensor((pool_elems,), dtype),
    ):
        with T.Kernel(T.ceildiv(pool_elems, threads), threads=threads) as bx:
            tx = T.get_thread_binding(0)
            x = bx * threads + tx
            if x < pool_elems:
                # Reference dtype inside the body so the closure captures it.
                _scratch = T.alloc_local((1,), dtype)
                _scratch[0] = T.cast(0.0, dtype)
                k_out[x] = k_pool[x]
                v_out[x] = v_pool[x]

    return _compile_tvm_ffi_kernel(pool_copy, out_idx=[2, 3])


@lru_cache(maxsize=128)
def _scatter_offsets_kernel_for(
    pool_elems: int,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    batch: int,
    new_tokens: int,
    layer_idx: int,
    dtype: str,
    threads: int,
):
    """Build & cache the offsets-scatter kernel for a static geometry.

    The kernel only *writes* the output buffers at the scattered destination
    elements (never reads them), so passing the freshly copied pools as
    caller-owned outputs is alias-safe.
    """

    import tilelang.language as T

    token_count = batch * new_tokens
    scatter_elems = token_count * num_kv_heads * head_dim

    @T.prim_func
    def scatter_offsets(
        k_new: T.Tensor((token_count * num_kv_heads * head_dim,), dtype),
        v_new: T.Tensor((token_count * num_kv_heads * head_dim,), dtype),
        tok_block: T.Tensor((token_count,), "int32"),
        tok_slot: T.Tensor((token_count,), "int32"),
        k_out: T.Tensor((pool_elems,), dtype),
        v_out: T.Tensor((pool_elems,), dtype),
    ):
        with T.Kernel(T.ceildiv(scatter_elems, threads), threads=threads) as bx:
            tx = T.get_thread_binding(0)
            sid = bx * threads + tx
            if sid < scatter_elems:
                # Reference dtype inside the body so the closure captures it.
                _scratch = T.alloc_local((1,), dtype)
                _scratch[0] = T.cast(0.0, dtype)
                d = sid % head_dim
                t1 = sid // head_dim
                h = t1 % num_kv_heads
                t = t1 // num_kv_heads
                if t < token_count:
                    b = t // new_tokens
                    s = t % new_tokens
                    src = (
                        ((b * num_kv_heads + h) * new_tokens + s) * head_dim + d
                    )
                    block = tok_block[t]
                    slot = tok_slot[t]
                    dst = (
                        ((block * num_layers + layer_idx) * block_size + slot)
                        * num_kv_heads
                        + h
                    ) * head_dim + d
                    if dst < pool_elems:
                        k_out[dst] = k_new[src]
                        v_out[dst] = v_new[src]

    return _compile_tvm_ffi_kernel(scatter_offsets, out_idx=[4, 5])


def scatter_paged_kv_offsets_native(
    manager: Any,
    block_table: mx.array,
    layer_idx: int,
    k: mx.array,
    v: mx.array,
    write_offsets: Sequence[int],
) -> None:
    """Native TileLang scatter of new-token K/V into the paged pool.

    Mirrors the contract of ``serving.scatter_paged_kv_offsets`` (callers must
    validate shapes, offsets and block-table aliases beforehand) and raises
    ``PagedKvNativeUnavailable`` on any unsupported input instead of falling
    back silently.
    """

    status = paged_kv_native_status()
    if not status.available:
        raise PagedKvNativeUnavailable(status.reason)
    dtype = manager.k_pool.dtype
    if dtype not in _SUPPORTED_DTYPES:
        raise PagedKvNativeUnavailable(
            f"unsupported pool dtype for native paged KV scatter: {dtype}"
        )
    if k.dtype != dtype or v.dtype != dtype:
        raise PagedKvNativeUnavailable(
            f"k/v dtype must match pool dtype {dtype}, got {k.dtype}/{v.dtype}"
        )

    batch = int(block_table.shape[0])
    new_tokens = int(k.shape[2])
    block_size = manager.block_size
    num_layers = manager.num_layers
    num_kv_heads = manager.num_kv_heads
    head_dim = manager.head_dim
    pool_elems = int(manager.k_pool.size)

    # Vectorized token -> (physical block, slot) table, no host loop.
    positions = (
        mx.array(list(write_offsets), dtype=mx.int32)[:, None]
        + mx.arange(new_tokens, dtype=mx.int32)[None, :]
    )
    logical = positions // block_size
    slots = positions % block_size
    tok_block = mx.take_along_axis(
        block_table, logical, axis=1
    ).astype(mx.int32).reshape(-1)
    tok_slot = slots.astype(mx.int32).reshape(-1)

    k_flat = k.reshape(-1)
    v_flat = v.reshape(-1)
    k_pool_flat = manager.k_pool.reshape(-1)
    v_pool_flat = manager.v_pool.reshape(-1)
    k_out = mx.empty(k_pool_flat.shape, dtype=dtype)
    v_out = mx.empty(v_pool_flat.shape, dtype=dtype)

    threads = 256
    tl_dtype = _DTYPE_TO_TILELANG[dtype]
    copy_kernel = _copy_kernel_for(pool_elems, tl_dtype, threads)
    returned = copy_kernel(k_pool_flat, v_pool_flat, k_out, v_out)
    if not (
        isinstance(returned, (list, tuple))
        and len(returned) == 2
        and returned[0] is k_out
        and returned[1] is v_out
    ):
        raise PagedKvNativeUnavailable(
            "native paged KV pool copy did not return the caller-owned outputs"
        )

    scatter_kernel = _scatter_offsets_kernel_for(
        pool_elems,
        num_layers,
        block_size,
        num_kv_heads,
        head_dim,
        batch,
        new_tokens,
        layer_idx,
        tl_dtype,
        threads,
    )
    returned = scatter_kernel(k_flat, v_flat, tok_block, tok_slot, k_out, v_out)
    if not (
        isinstance(returned, (list, tuple))
        and len(returned) == 2
        and returned[0] is k_out
        and returned[1] is v_out
    ):
        raise PagedKvNativeUnavailable(
            "native paged KV scatter did not return the caller-owned outputs"
        )
    mx.synchronize()
    manager.k_pool = k_out.reshape(manager.k_pool.shape)
    manager.v_pool = v_out.reshape(manager.v_pool.shape)


__all__ = [
    "PagedKvNativeStatus",
    "PagedKvNativeUnavailable",
    "paged_kv_native_status",
    "scatter_paged_kv_offsets_native",
]
