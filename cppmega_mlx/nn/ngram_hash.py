"""MLX n-gram hash embedding for cppmega input enrichment."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_PRIMES = (
    499_801,
    499_819,
    499_853,
    499_879,
    499_883,
    499_897,
    499_903,
    499_927,
    499_943,
    499_957,
    499_969,
    499_973,
    499_979,
    500_009,
    500_029,
    500_041,
)


def pick_primes(count: int, target_size: int) -> tuple[int, ...]:
    """Pick cppmega-compatible per-table sizes near ``target_size``."""

    candidates = [p for p in _PRIMES if abs(p - target_size) / target_size < 0.5]
    if len(candidates) >= count:
        return tuple(candidates[:count])
    return tuple(target_size + i for i in range(count))


class NgramHashEmbedding(nn.Module):
    """Hash token n-grams into a unified table and project to hidden size."""

    def __init__(
        self,
        *,
        hidden_size: int,
        orders: tuple[int, ...] = (2, 3),
        num_heads: int = 8,
        table_size: int = 500_000,
        embed_dim: int = 16,
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not orders:
            raise ValueError("orders must contain at least one n-gram order")
        if any(order <= 0 for order in orders):
            raise ValueError("all n-gram orders must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if table_size <= 0:
            raise ValueError("table_size must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_size = hidden_size
        self.orders = tuple(int(order) for order in orders)
        self.num_heads = int(num_heads)
        self.num_tables = len(self.orders) * self.num_heads
        self.embed_dim = int(embed_dim)
        self.max_order = max(self.orders)

        self.table_sizes = pick_primes(self.num_tables, table_size)
        offsets: list[int] = []
        total_entries = 0
        for size in self.table_sizes:
            offsets.append(total_entries)
            total_entries += size

        self.table_offsets = mx.array(offsets, dtype=mx.int64)
        self.table_sizes_t = mx.array(self.table_sizes, dtype=mx.int64)
        self.unified_table = nn.Embedding(total_entries, self.embed_dim)

        if seed is not None:
            mx.random.seed(seed)
        self.hash_mults = mx.random.randint(
            1,
            2**31,
            shape=(self.num_tables, self.max_order),
            dtype=mx.int64,
        )
        self.hash_mults = mx.bitwise_or(self.hash_mults, mx.array(1, dtype=mx.int64))
        self.hash_bias = mx.random.randint(
            0,
            2**31,
            shape=(self.num_tables,),
            dtype=mx.int64,
        )

        order_list: list[int] = []
        for order in self.orders:
            order_list.extend([order] * self.num_heads)
        self.order_for_table = mx.array(order_list, dtype=mx.int64)

        order_mask = mx.zeros((self.max_order, self.num_tables), dtype=mx.int64)
        for table_index, order in enumerate(order_list):
            for position in range(order):
                order_mask[position, table_index] = mx.array(1, dtype=mx.int64)
        self.order_mask = order_mask

        self.out_proj = nn.Linear(self.num_tables * self.embed_dim, hidden_size, bias=False)
        self.out_proj.weight = mx.zeros_like(self.out_proj.weight)
        self.dropout = nn.Dropout(dropout)

        self.freeze(
            recurse=False,
            keys=[
                "table_offsets",
                "table_sizes_t",
                "hash_mults",
                "hash_bias",
                "order_for_table",
                "order_mask",
            ],
        )

    def _shifted_tokens(self, token_ids: mx.array) -> mx.array:
        batch, seq = token_ids.shape
        shifted: list[mx.array] = [token_ids]
        for position in range(1, self.max_order):
            if position >= seq:
                shifted.append(mx.zeros((batch, seq), dtype=mx.int64))
                continue
            prefix = mx.zeros((batch, position), dtype=mx.int64)
            shifted.append(mx.concatenate([prefix, token_ids[:, :-position]], axis=1))
        return mx.stack(shifted, axis=0)

    def _hash_all(self, token_ids: mx.array) -> mx.array:
        """Return unified table indices with shape ``(batch, tables, seq)``."""

        if len(token_ids.shape) != 2:
            raise ValueError(f"token_ids must have shape (batch, seq), got {token_ids.shape}")
        token_ids = token_ids.astype(mx.int64)
        shifted = self._shifted_tokens(token_ids)
        mults = mx.expand_dims(mx.expand_dims(self.hash_mults.T, -1), -1)
        mask = mx.expand_dims(mx.expand_dims(self.order_mask, -1), -1)
        product = (mults * mx.expand_dims(shifted, 1)) * mask

        hashed = product[0]
        for position in range(1, self.max_order):
            hashed = mx.bitwise_xor(hashed, product[position])

        hashed = mx.bitwise_xor(hashed, mx.expand_dims(mx.expand_dims(self.hash_bias, -1), -1))
        hashed = mx.remainder(hashed, mx.expand_dims(mx.expand_dims(self.table_sizes_t, -1), -1))
        hashed = hashed + mx.expand_dims(mx.expand_dims(self.table_offsets, -1), -1)
        return mx.transpose(hashed, (1, 0, 2))

    def __call__(self, token_ids: mx.array) -> mx.array:
        unified_indices = self._hash_all(token_ids)
        batch, num_tables, seq = unified_indices.shape
        embeddings = self.unified_table.weight[unified_indices]
        embeddings = mx.reshape(
            mx.transpose(embeddings, (0, 2, 1, 3)),
            (batch, seq, num_tables * self.embed_dim),
        )
        return self.dropout(self.out_proj(embeddings))


CppMegaNgramHashEmbedding = NgramHashEmbedding

