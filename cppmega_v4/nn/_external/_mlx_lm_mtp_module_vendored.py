# Vendored from ml-explore/mlx-lm PR #990 (mlx_lm/models/qwen3_5.py)
# Upstream license: MIT, © 2025-2026 Apple Inc.
#
# Excerpt only — the MTPModule class + its decoder layer.  The full PR also
# touches generate.py / sample_utils.py / server.py / cache.py to wire
# speculative decoding end-to-end through mlx_lm.generate; those drivers are
# out of scope for the v4 plugin (we expose the module via our own dispatch).
#
# Public surface used by cppmega_v4:
#   - MTPDecoderLayer   — full-attention decoder block for the MTP head
#   - MTPModule         — fuses hidden + next-token embedding via (pre_norm,
#                         pre_norm, fc, layers, post_norm) — the "DeepSeek
#                         V3 SequentialMTPHead" pattern that Qwen3.5 inherited.

"""Multi-Token Prediction module (Qwen3.5 native speculative decoding).

Vendored verbatim except for:
  - removed unused imports (Attention/MLP/SparseMoeBlock/SwitchLinear) that
    pull the whole qwen3_5 model into scope. The MTPModule itself only
    constructs sub-layers via *injected* nn.Module classes — callers pass
    in the decoder-layer factory rather than relying on the qwen3_5 module
    globals.

That refactor (factory pattern) is the *minimum* edit to make the vendored
class usable without dragging Qwen3.5-specific Attention/MLP/MoE code into
the v4 plugin.
"""

from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn


def create_attention_mask(x: mx.array, cache: Any) -> Optional[mx.array]:
    """Causal mask helper (lifted from mlx_lm.models.base for self-containment)."""
    L = x.shape[1]
    if L <= 1:
        return None
    offset = 0 if cache is None else cache.offset
    rinds = mx.arange(offset + L)
    linds = mx.arange(offset, offset + L) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class MTPModule(nn.Module):
    """Multi-Token Prediction head (Qwen3.5 native speculative decoding).

    Predicts token t+2 from the backbone hidden state h_t and the sampled
    token t+1, using a shared lm_head with the backbone.

    Architecture:
        e     = pre_fc_norm_embedding(embed_tokens(next_token_ids))   # (B, N, H)
        h     = pre_fc_norm_hidden(hidden_states)                     # (B, N, H)
        fused = fc(concat([e, h], -1))                                # (B, N, H)
        for layer in layers: fused = layer(fused, mask, cache)
        out   = norm(fused)
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        rms_norm_eps: float,
        decoder_layer_factory: Callable[[], nn.Module],
    ):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layers = [decoder_layer_factory() for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Embedding,
        cache: Optional[Any] = None,
    ) -> mx.array:
        embeds = embed_tokens(next_token_ids)  # (B, N, H)
        e = self.pre_fc_norm_embedding(embeds)
        h = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(mx.concatenate([e, h], axis=-1))  # (B, N, H)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(fused, cache[0])
        for layer, c in zip(self.layers, cache):
            fused = layer(fused, mask, c)

        return self.norm(fused)  # (B, N, H)
