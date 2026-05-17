"""V4 Engram block — doc_id-aware n-gram memory branch.

The vendored TileKernels engram_hash_ref + engram_gate_ref operate on flat
token sequences. This module wraps them with the doc_id partitioning logic
that mirrors ``cppmega_mlx.nn.engram.EngramBranch`` semantics — but lives
in the V4 plugin and integrates with UnifiedSuperblockV4 + RunTemplate.

Document partitioning:
  - For each unique document_id in the batch, gather the tokens belonging
    to that document and run engram_hash on its windowed n-grams.
  - Results scatter back to the original token positions.
  - Tokens with doc_id == ignore_doc_id (default -1) are pass-through.

This matches the V3 design: per-document n-gram memory prevents the model
from learning n-gram associations across document boundaries (which would
leak across our parquet batches).
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class EngramV4Config:
    """V4 Engram block config — keeps the small-vocab default for tests."""

    hidden_size: int
    num_ngram_layers: int = 2
    max_ngram_size: int = 4
    num_embed_table_per_ngram: int = 4
    embed_dim: int = 64
    embed_table_size: int = 256
    clamp_value: float = 1.0
    ignore_doc_id: int = -1

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_ngram_layers <= 0 or self.max_ngram_size < 2:
            raise ValueError("num_ngram_layers > 0 and max_ngram_size >= 2 required")
        if self.embed_table_size <= 0:
            raise ValueError("embed_table_size must be positive")


class EngramV4Block(nn.Module):
    """V4 Engram block — doc_id-aware n-gram memory.

    Forward signature:
        (x: [B, S, H], token_ids: [B, S], document_ids: [B, S] | None)
        → delta: [B, S, H]
    The caller adds ``delta`` to the residual stream. With document_ids
    None, every token is treated as the same document.
    """

    def __init__(self, config: EngramV4Config):
        super().__init__()
        self.config = config
        # Per-layer multipliers for hash mixing.
        # int64 to avoid overflow during prod * multiplier.
        rng = mx.random.uniform(
            low=1, high=2**31 - 1,
            shape=(config.num_ngram_layers, config.max_ngram_size),
        ).astype(mx.int64)
        self.multipliers = rng
        # Per-layer per-ngram embedding table sizes (uniform for simplicity).
        self.vocab_sizes = mx.full(
            (config.num_ngram_layers, config.max_ngram_size - 1,
             config.num_embed_table_per_ngram),
            config.embed_table_size, dtype=mx.int32,
        )
        # Final embedding table & projection back to hidden_size.
        total_size = (
            config.num_ngram_layers
            * (config.max_ngram_size - 1)
            * config.num_embed_table_per_ngram
            * config.embed_table_size
        )
        self.embed = nn.Embedding(total_size, config.embed_dim)
        self.proj = nn.Linear(
            config.num_ngram_layers
            * (config.max_ngram_size - 1)
            * config.num_embed_table_per_ngram
            * config.embed_dim,
            config.hidden_size,
            bias=False,
        )

    def __call__(
        self,
        x: mx.array,
        token_ids: mx.array,
        document_ids: Optional[mx.array] = None,
    ) -> mx.array:
        cfg = self.config
        B, S, H = x.shape
        if token_ids.shape != (B, S):
            raise ValueError(
                f"token_ids must be (B, S)={B, S}, got {token_ids.shape}"
            )
        if document_ids is not None and document_ids.shape != (B, S):
            raise ValueError(
                f"document_ids must be (B, S)={B, S}, got {document_ids.shape}"
            )

        # Build n-grams within each document via causal sliding window.
        # ngram_token_ids[b, s, i] = token_ids[b, s - (max_ngram_size - 1 - i)]
        # but clamped to the document start.
        ngrams = self._build_ngrams(token_ids, document_ids)  # [B, S, max_n]
        # Flatten batch/seq for hashing.
        flat_ngrams = ngrams.reshape(-1, cfg.max_ngram_size).astype(mx.int32)

        # Compute hashes inline (lightweight subset of engram_hash_ref;
        # the vendored version requires offsets we compute here).
        embed_idx = self._hash_to_embed_idx(flat_ngrams)  # [L, num_tokens, K]
        # Look up embeddings: [L, num_tokens, K, embed_dim]
        embeds = self.embed(embed_idx.astype(mx.int32))
        # Flatten per-token: [num_tokens, L * K * embed_dim]
        num_tokens = flat_ngrams.shape[0]
        embeds = mx.transpose(embeds, (1, 0, 2, 3)).reshape(num_tokens, -1)
        delta = self.proj(embeds).reshape(B, S, H)
        return delta

    def _build_ngrams(
        self, token_ids: mx.array, document_ids: Optional[mx.array]
    ) -> mx.array:
        """Build a [B, S, max_ngram_size] tensor of n-gram token ids.

        For each position (b, s), gather token_ids at positions
        s-(max_n-1)..s. When document_ids is provided, clamp the gather to
        not cross document boundaries — out-of-doc positions repeat the
        first in-doc token.
        """
        cfg = self.config
        B, S = token_ids.shape
        max_n = cfg.max_ngram_size
        # Build offsets [max_n] = [-(max_n-1), ..., 0]
        offsets = mx.arange(max_n) - (max_n - 1)
        # positions [S, max_n]
        positions = mx.arange(S)[:, None] + offsets[None, :]
        # Clamp to >= 0 (no negative indexing).
        positions = mx.maximum(positions, 0)
        # If document_ids is provided, enforce doc-boundary clamping:
        # for each (b, s, i), if document_ids[b, positions[s, i]] !=
        # document_ids[b, s], snap positions[s, i] to the earliest in-doc
        # position. Simple approach: walk back through positions and copy.
        if document_ids is not None:
            # positions: [S, max_n] -> gather doc_ids at each candidate
            # position per batch: gathered_doc[b, s, i] = document_ids[b, positions[s, i]]
            # mx.take(..., axis=1) selects along axis 1 with a fancy index
            # tensor, broadcasting batch dim 0 automatically.
            gathered_doc = mx.take(document_ids, positions, axis=1)  # [B, S, max_n]
            current_doc = document_ids[:, :, None]  # [B, S, 1]
            same_doc = (gathered_doc == current_doc) | (
                gathered_doc == cfg.ignore_doc_id
            )
            # Where the candidate position is out-of-doc, snap to position s
            # itself (the current token — guaranteed in-doc).
            current_pos = mx.broadcast_to(mx.arange(S)[None, :, None], (B, S, max_n))
            clamped_pos = mx.where(same_doc, positions[None, :, :], current_pos)
            # Gather token_ids at clamped positions: [B, S, max_n]
            ngrams_per_b = []
            for b_i in range(B):
                ngrams_per_b.append(mx.take(token_ids[b_i], clamped_pos[b_i], axis=0))
            return mx.stack(ngrams_per_b, axis=0)
        # No doc_ids: simple gather (with the >=0 clamp).
        ngrams = token_ids[:, positions]
        return ngrams

    def _hash_to_embed_idx(self, ngrams: mx.array) -> mx.array:
        """Compute per-layer per-ngram per-table embedding indices.

        Mirrors engram_hash_ref's body, inlined for shape clarity.
        Returns [num_ngram_layers, num_tokens, (max_n-1) * K] int32.
        """
        cfg = self.config
        L = cfg.num_ngram_layers
        max_n = cfg.max_ngram_size
        K = cfg.num_embed_table_per_ngram
        ngrams64 = ngrams.astype(mx.int64)             # [num_tokens, max_n]
        # prod [L, num_tokens, max_n] = ngrams64 * multipliers[:, None, :]
        prod = ngrams64[None] * self.multipliers[:, None, :]
        # Running XOR up to position i; per layer, output [num_tokens, K]
        # using modulo each table's size.
        per_layer_results = []
        for layer_idx in range(L):
            hashes_layer = prod[layer_idx, :, 0]  # [num_tokens]
            cols = []
            for i in range(1, max_n):
                hashes_layer = mx.bitwise_xor(hashes_layer, prod[layer_idx, :, i])
                # vocab_sizes[layer_idx, i-1, k] for k in K
                v_sizes = self.vocab_sizes[layer_idx, i - 1].astype(mx.int64)  # [K]
                idx = (hashes_layer[:, None] % v_sizes[None, :]).astype(mx.int32)
                # Offset within the global embed table:
                #   per-layer-per-ngram-per-table block.
                table_base = (
                    (layer_idx * (max_n - 1) + (i - 1)) * K * cfg.embed_table_size
                )
                idx = idx + (mx.arange(K, dtype=mx.int32) * cfg.embed_table_size + table_base)[None, :]
                cols.append(idx)  # [num_tokens, K]
            per_layer_results.append(mx.concatenate(cols, axis=-1))  # [num_tokens, (max_n-1)*K]
        return mx.stack(per_layer_results, axis=0)  # [L, num_tokens, (max_n-1)*K]


__all__ = ["EngramV4Block", "EngramV4Config"]
