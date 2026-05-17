"""Speculative-decoding adapter: SequentialMTPHead → mlx-lm PR #990 MTPModule API.

PR #990 introduces native MTP speculative decoding into mlx-lm with a
specific module contract:

    MTPModule(hidden_states, next_token_ids, embed_tokens, cache=None) -> fused_h
    # fused_h then fed into the shared lm_head to produce token-t+2 logits.

Our cppmega_v4 SequentialMTPHead exposes a *training-time* loss interface
(returns per-depth logits + loss). This adapter wraps a SequentialMTPHead
(or just its depth-0 block) and presents the (hidden, next_token_ids,
embed_tokens) signature that PR #990's driver expects, so the same head
can serve both training and speculative inference without a separate
inference-only module.

Key behavioural mapping:
  - PR #990's pre_fc_norm_hidden / pre_fc_norm_embedding ↔ our
    SequentialMTPDepthBlock.hidden_norm / embedding_norm
  - PR #990's fc(concat([e, h]))                       ↔ our
    SequentialMTPDepthBlock.proj
  - PR #990's stack of MTPDecoderLayer                 ↔ our
    SequentialMTPDepthBlock.transformer + .output_norm

The adapter uses depth-0's block by default (depth-1+ are for deeper
look-ahead in training; at inference time we predict one token ahead).
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn.mtp_v4 import SequentialMTPDepthBlock, SequentialMTPHead


class SequentialMTPHeadAsMTPModule(nn.Module):
    """Wraps a SequentialMTPDepthBlock to expose the PR #990 MTPModule API.

    Construct directly from a SequentialMTPHead via ``from_head``, or build
    fresh from hidden_size if you only want the inference contract without
    sharing weights with a training head.
    """

    def __init__(self, depth_block: SequentialMTPDepthBlock):
        super().__init__()
        self.depth_block = depth_block

    @classmethod
    def from_head(
        cls,
        head: SequentialMTPHead,
        depth_index: int = 0,
    ) -> "SequentialMTPHeadAsMTPModule":
        if head.config.depth <= 0:
            raise ValueError(
                "Cannot build MTPModule adapter from a depth=0 SequentialMTPHead"
            )
        if not 0 <= depth_index < head.config.depth:
            raise ValueError(
                f"depth_index {depth_index} out of range [0, {head.config.depth})"
            )
        return cls(head.depth_blocks[depth_index])

    @classmethod
    def fresh(cls, hidden_size: int) -> "SequentialMTPHeadAsMTPModule":
        return cls(SequentialMTPDepthBlock(hidden_size))

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Embedding,
        cache: Optional[Any] = None,  # accepted for PR #990 API compat; unused
    ) -> mx.array:
        """Run one MTP fusion + transformer step.

        Args:
            hidden_states: (B, N, H) backbone pre-norm hidden state.
            next_token_ids: (B, N) sampled next-token ids (the speculative draft).
            embed_tokens: the backbone's token embedding (shared with lm_head
                or independent — caller decides).
            cache: KV cache slot list, ignored here (SequentialMTPDepthBlock's
                transformer is a single self-contained block).

        Returns:
            (B, N, H) fused hidden state; caller applies the shared lm_head.
        """
        del cache  # PR #990 API compat; our block has no per-layer KV cache
        embeds = embed_tokens(next_token_ids)
        return self.depth_block(hidden_states, embeds)


__all__ = ["SequentialMTPHeadAsMTPModule"]
