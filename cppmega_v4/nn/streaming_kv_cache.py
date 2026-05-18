"""Streaming KV cache for CSA / HCA / NSA-Compress branches.

Avoids recomputing the mean-pool over historical tokens on every forward.
Maintains a running compressed K/V buffer of super-tokens (size ⌈S_seen/m⌉)
plus a partial super-token-in-progress with its accumulated mean and count.

Three operations:
    cache = StreamingPoolCache(m=m, n_heads=H, head_dim=D, dtype=mx.float32)
    cache.append(new_k, new_v)        # append [B, T_new, H, D]
    k_super, v_super = cache.snapshot()   # [B, n_super, H, D] — current view

Pure-MLX implementation (no Metal kernel needed). A future Metal/TileLang
fused kernel could maintain the running mean in shared memory; current
impl is allocation-friendly enough for inference on Apple Silicon.
"""

from typing import Optional, Tuple

import mlx.core as mx


class StreamingPoolCache:
    """Per-(B, H) running mean-pool over m tokens for compressed-attention.

    State:
        - completed: [B, n_super, H, D] fp32, the per-block means already finalized.
        - partial_sum: [B, H, D] fp32, sum-so-far of the in-progress super-token.
        - partial_count: int — number of tokens already added to partial_sum.

    Calling `snapshot()` returns the completed block + the in-progress block
    finalized as `partial_sum / partial_count` so the caller sees a coherent
    [B, n_super_total, H, D] view.
    """

    def __init__(
        self,
        *,
        m: int,
        batch: int,
        n_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")
        if batch <= 0 or n_heads <= 0 or head_dim <= 0:
            raise ValueError("batch / n_heads / head_dim must all be positive")
        self.m = m
        self.batch = batch
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._completed_k: Optional[mx.array] = None   # [B, n_super, H, D]
        self._completed_v: Optional[mx.array] = None
        self._partial_k = mx.zeros((batch, n_heads, head_dim), dtype=dtype)
        self._partial_v = mx.zeros((batch, n_heads, head_dim), dtype=dtype)
        self._partial_count = 0
        self._n_super = 0
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def n_super_completed(self) -> int:
        return self._n_super

    def append(self, k: mx.array, v: mx.array) -> None:
        """Append [B, T_new, H, D] new K/V tokens to the running cache."""
        if k.shape != v.shape:
            raise ValueError(f"k.shape ({k.shape}) != v.shape ({v.shape})")
        if k.ndim != 4:
            raise ValueError(f"expected [B, T, H, D]; got {k.shape}")
        b, t, h, d = k.shape
        if b != self.batch or h != self.n_heads or d != self.head_dim:
            raise ValueError(
                f"k shape {k.shape} incompatible with cache "
                f"(B={self.batch}, H={self.n_heads}, D={self.head_dim})"
            )
        k = k.astype(self.dtype)
        v = v.astype(self.dtype)

        # Walk tokens one at a time: simpler than partial-block arithmetic
        # because we need to flush completed super-tokens mid-batch.
        token_idx = 0
        while token_idx < t:
            tokens_to_fill = self.m - self._partial_count
            tokens_available = t - token_idx
            take = min(tokens_to_fill, tokens_available)
            # Accumulate take tokens into partial.
            chunk_k = k[:, token_idx : token_idx + take]   # [B, take, H, D]
            chunk_v = v[:, token_idx : token_idx + take]
            self._partial_k = self._partial_k + chunk_k.sum(axis=1)
            self._partial_v = self._partial_v + chunk_v.sum(axis=1)
            self._partial_count += take
            token_idx += take
            self._total_tokens += take
            if self._partial_count == self.m:
                # Flush completed super-token.
                mean_k = (self._partial_k / float(self.m))[:, None, :, :]
                mean_v = (self._partial_v / float(self.m))[:, None, :, :]
                if self._completed_k is None:
                    self._completed_k = mean_k
                    self._completed_v = mean_v
                else:
                    self._completed_k = mx.concatenate(
                        [self._completed_k, mean_k], axis=1
                    )
                    self._completed_v = mx.concatenate(
                        [self._completed_v, mean_v], axis=1
                    )
                self._n_super += 1
                self._partial_k = mx.zeros_like(self._partial_k)
                self._partial_v = mx.zeros_like(self._partial_v)
                self._partial_count = 0

    def snapshot(
        self, *, include_partial: bool = True,
    ) -> Tuple[mx.array, mx.array]:
        """Return current compressed K/V super-tokens [B, n_super_total, H, D].

        With ``include_partial=True`` (default), the in-progress block is
        finalized as ``partial_sum / partial_count``. With False, only the
        completed super-tokens are returned.
        """
        completed_k = (
            self._completed_k if self._completed_k is not None
            else mx.zeros((self.batch, 0, self.n_heads, self.head_dim),
                          dtype=self.dtype)
        )
        completed_v = (
            self._completed_v if self._completed_v is not None
            else mx.zeros((self.batch, 0, self.n_heads, self.head_dim),
                          dtype=self.dtype)
        )
        if include_partial and self._partial_count > 0:
            mean_k = (self._partial_k / float(self._partial_count))[:, None, :, :]
            mean_v = (self._partial_v / float(self._partial_count))[:, None, :, :]
            completed_k = mx.concatenate([completed_k, mean_k], axis=1)
            completed_v = mx.concatenate([completed_v, mean_v], axis=1)
        return completed_k, completed_v

    def reset(self) -> None:
        """Drop all state (start a new sequence)."""
        self._completed_k = None
        self._completed_v = None
        self._partial_k = mx.zeros((self.batch, self.n_heads, self.head_dim),
                                    dtype=self.dtype)
        self._partial_v = mx.zeros((self.batch, self.n_heads, self.head_dim),
                                    dtype=self.dtype)
        self._partial_count = 0
        self._n_super = 0
        self._total_tokens = 0


__all__ = ["StreamingPoolCache"]
