"""DeepSeek-V3-style Sequential Multi-Token-Prediction head plugin.

The existing ``MinimalMTPHead`` (``cppmega_mlx/training/mtp.py``) reuses one
shared block recursively for every depth — a contracted reference. DeepSeek-V3
ships D **sequential transformer-style blocks**, one per future-token depth,
each with its own RMSNorms, projection, and transformer kernel, but sharing the
model's token embedding and lm head.

This plugin lands the V3-faithful surface side-by-side without touching the
existing ``MinimalMTPHead``.

Sharing contract (intentional):
    - ``token_embedding`` and ``lm_head`` are aliased to the model's existing
      instances (no copy) — gradients flow back through the shared parameters.
    - Each depth owns its own ``hidden_norm``, ``embedding_norm``, ``proj``,
      transformer kernel, and ``output_norm`` — D distinct module instances.

CUDA FastMTP detach semantics are preserved:
    - ``hidden_states`` enters under ``mx.stop_gradient`` (no backprop through
      the main decoder).
    - ``lm_head.weight`` is detached during the per-depth logits projection.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.training.mtp import (
    MinimalMTPSharedBlock,
    MTPLossConfig,
    _lm_head_with_stopped_weight,
    compute_weighted_mtp_loss,
    mtp_cross_entropy_from_logits,
    roll_and_mask_mtp_ids,
    roll_and_mask_mtp_labels,
)


class SequentialMTPDepthBlock(nn.Module):
    """One MTP depth-block: per-depth norms + projection + transformer kernel."""

    def __init__(self, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        self.hidden_norm = nn.RMSNorm(hidden_size)
        self.embedding_norm = nn.RMSNorm(hidden_size)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # Reuse the existing single-block kernel as the per-depth transformer
        # primitive. This keeps the shape contract identical to MinimalMTPHead.
        self.transformer = MinimalMTPSharedBlock(hidden_size)
        self.output_norm = nn.RMSNorm(hidden_size)

    def __call__(self, hidden_states: mx.array, teacher_emb: mx.array) -> mx.array:
        h_mtp = self.proj(
            mx.concatenate(
                [self.hidden_norm(hidden_states), self.embedding_norm(teacher_emb)],
                axis=-1,
            )
        )
        return self.output_norm(self.transformer(h_mtp))


class SequentialMTPHead(nn.Module):
    """V3-faithful depth-D MTP head with D distinct per-depth blocks.

    Public API mirrors ``MinimalMTPHead`` so the rest of the training loop is
    unchanged: ``__call__(hidden_states, target_tokens, ...)`` returns one
    logits tensor per depth, and ``loss(...)`` returns
    ``(mtp_loss, per_depth_losses, depth_weights)``.
    """

    def __init__(
        self,
        token_embedding: nn.Embedding,
        lm_head: nn.Linear,
        *,
        config: MTPLossConfig | None = None,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.lm_head = lm_head
        self.config = config or MTPLossConfig()
        if self.config.depth < 0:
            raise ValueError("MTP depth must be non-negative")
        hidden_size = int(token_embedding.weight.shape[1])
        # D distinct depth blocks — this is the V3 contract.
        self.depth_blocks = [
            SequentialMTPDepthBlock(hidden_size) for _ in range(self.config.depth)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        target_tokens: mx.array,
        *,
        document_ids: mx.array | None = None,
    ) -> tuple[mx.array, ...]:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must be shaped (B, T, D), got {hidden_states.shape}"
            )
        if hidden_states.shape[:2] != target_tokens.shape:
            raise ValueError(
                f"hidden_states prefix shape {hidden_states.shape[:2]} must match "
                f"target_tokens {target_tokens.shape}"
            )
        if self.config.depth == 0:
            return ()

        teacher_ids = roll_and_mask_mtp_ids(
            target_tokens,
            depth=self.config.depth,
            document_ids=document_ids,
        )
        logits_by_depth: list[mx.array] = []
        # CUDA FastMTP detach: main decoder states do not receive grad through MTP.
        h = mx.stop_gradient(hidden_states)
        for depth_idx, ids in enumerate(teacher_ids):
            teacher_emb = self.token_embedding(ids)
            h = self.depth_blocks[depth_idx](h, teacher_emb)
            logits_by_depth.append(_lm_head_with_stopped_weight(self.lm_head, h))
        return tuple(logits_by_depth)

    def loss(
        self,
        hidden_states: mx.array,
        target_tokens: mx.array,
        *,
        document_ids: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, ...], mx.array]:
        labels = roll_and_mask_mtp_labels(
            target_tokens,
            depth=self.config.depth,
            ignore_index=self.config.ignore_index,
            document_ids=document_ids,
        )
        logits_by_depth = self(
            hidden_states,
            target_tokens,
            document_ids=document_ids,
        )
        per_depth = tuple(
            mtp_cross_entropy_from_logits(
                logits,
                label,
                ignore_index=self.config.ignore_index,
            )
            for logits, label in zip(logits_by_depth, labels, strict=True)
        )
        mtp_loss, depth_weights = compute_weighted_mtp_loss(
            per_depth,
            decay=self.config.decay,
        )
        return mtp_loss, per_depth, depth_weights


def attach_sequential_mtp_head(
    model: nn.Module,
    *,
    config: MTPLossConfig | None = None,
) -> SequentialMTPHead:
    """Attach a persistent SequentialMTPHead to a model via direct aliasing.

    Mirrors ``cppmega_mlx.training.mtp.attach_mtp_head`` but emits the V3
    sequential variant. Does not modify the model's class — only sets the
    ``mtp_head`` attribute.
    """
    token_embedding = getattr(model, "token_embedding", None)
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(token_embedding, nn.Embedding):
        raise TypeError(
            "SequentialMTPHead requires model.token_embedding to be an nn.Embedding"
        )
    if not isinstance(lm_head, nn.Linear):
        raise TypeError("SequentialMTPHead requires model.lm_head to be an nn.Linear")

    head = SequentialMTPHead(token_embedding, lm_head, config=config)
    setattr(model, "mtp_head", head)
    return head


__all__ = [
    "SequentialMTPDepthBlock",
    "SequentialMTPHead",
    "attach_sequential_mtp_head",
]
