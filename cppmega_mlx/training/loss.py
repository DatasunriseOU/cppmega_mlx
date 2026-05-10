"""Loss helpers for local MLX language-model training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping, cast

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch
from cppmega_mlx.nn.structure_embedding import StructureEmbedding
from cppmega_mlx.training.cut_cross_entropy import (
    DEFAULT_CHUNK_ROWS,
    linear_cross_entropy,
)
from cppmega_mlx.training.mtp import (
    MTPLossConfig,
    MTPLossMetrics,
    compute_mtp_step_weights,
    get_or_attach_mtp_head,
    next_token_and_mtp_loss,
)
from cppmega_mlx.training.stp_loss import (
    STPLossConfig,
    STPLossMetrics,
    compute_stp_loss,
    next_token_and_stp_loss,
)

_DOCUMENT_ID_ALIASES = ("document_ids", "doc_ids", "packing_document_ids")


def next_token_cross_entropy(
    model: nn.Module,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
) -> tuple[mx.array, mx.array]:
    """Return masked next-token CE loss and the number of contributing tokens."""

    document_ids = _extract_document_ids(batch)
    lm_batch = ensure_lm_batch(batch)
    model_kwargs = lm_batch.model_kwargs()
    if document_ids is not None:
        model_kwargs["document_ids"] = document_ids[:, :-1]
    logits = model(lm_batch.inputs, **model_kwargs)
    targets = lm_batch.targets

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits prefix shape {logits.shape[:2]} must match targets {targets.shape}"
        )

    token_losses = nn.losses.cross_entropy(
        logits.astype(mx.float32), targets, reduction="none"
    )
    mask = lm_batch.target_mask
    ntokens = mask.sum()
    denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
    loss = (token_losses * mask).astype(mx.float32).sum() / denom
    return loss, ntokens


def next_token_cut_cross_entropy(
    model: nn.Module,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[mx.array, mx.array]:
    """Return masked next-token CE via the MLX-native chunked linear path.

    This mirrors :func:`next_token_cross_entropy` but avoids materializing the
    full ``[B*T, V]`` logits tensor in the forward loss. Under
    ``nn.value_and_grad`` MLX still owns the backward trace, so this is a train
    integration path, not a full manual chunked-backward memory receipt.
    """

    document_ids = _extract_document_ids(batch)
    lm_batch = ensure_lm_batch(batch)
    hidden_states = _decoder_hidden_states_for_mtp(
        model,
        lm_batch,
        document_ids=document_ids[:, :-1] if document_ids is not None else None,
    )
    targets = lm_batch.targets
    if hidden_states.shape[:2] != targets.shape:
        raise ValueError(
            "hidden-states prefix shape "
            f"{hidden_states.shape[:2]} must match targets {targets.shape}"
        )

    lm_head = getattr(model, "lm_head", None)
    head_weight = getattr(lm_head, "weight", None)
    if not isinstance(head_weight, mx.array):
        raise TypeError("CCE loss requires model.lm_head.weight to be an mx.array")

    token_losses = linear_cross_entropy(
        hidden_states,
        head_weight,
        targets,
        reduction="none",
        chunk_rows=chunk_rows,
    )
    mask = lm_batch.target_mask
    ntokens = mask.sum()
    denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
    loss = (token_losses * mask).astype(mx.float32).sum() / denom
    return loss, ntokens


def next_token_cross_entropy_with_mtp(
    model: nn.Module,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    *,
    config: MTPLossConfig | None = None,
) -> tuple[mx.array, mx.array, MTPLossMetrics]:
    """Return ``NTP + lambda*MTP`` loss and per-depth MTP metrics.

    This opt-in path preserves the default next-token loss while wiring the
    M0.5 training-side FastMTP contract onto models exposing ``token_embedding``
    and ``lm_head`` modules. For optimizer training, call ``attach_mtp_head`` on
    the model before constructing ``nn.value_and_grad`` so MLX captures the
    persistent head parameters.
    """

    document_ids = _extract_document_ids(batch)
    lm_batch = ensure_lm_batch(batch)
    next_token_loss, ntokens = next_token_cross_entropy(model, batch)
    cfg = config or MTPLossConfig()

    if cfg.depth == 0:
        mtp_loss = next_token_loss * 0.0
        depth_weights = compute_mtp_step_weights(0, cfg.decay)
        total_loss = next_token_and_mtp_loss(
            next_token_loss,
            mtp_loss,
            loss_weight=cfg.loss_weight,
        )
        return total_loss, ntokens, MTPLossMetrics(
            next_token_loss=next_token_loss,
            mtp_loss=mtp_loss,
            total_loss=total_loss,
            per_depth_losses=(),
            depth_weights=depth_weights,
            loss_weight=cfg.loss_weight,
        )

    head = get_or_attach_mtp_head(model, config=cfg)
    hidden_states = _decoder_hidden_states_for_mtp(
        model,
        lm_batch,
        document_ids=document_ids[:, :-1] if document_ids is not None else None,
    )
    mtp_loss, per_depth_losses, depth_weights = head.loss(
        hidden_states,
        lm_batch.targets,
        document_ids=document_ids[:, 1:] if document_ids is not None else None,
    )
    total_loss = next_token_and_mtp_loss(
        next_token_loss,
        mtp_loss,
        loss_weight=cfg.loss_weight,
    )
    return total_loss, ntokens, MTPLossMetrics(
        next_token_loss=next_token_loss,
        mtp_loss=mtp_loss,
        total_loss=total_loss,
        per_depth_losses=per_depth_losses,
        depth_weights=depth_weights,
        loss_weight=cfg.loss_weight,
    )


def next_token_cross_entropy_with_stp(
    model: nn.Module,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    *,
    config: STPLossConfig | None = None,
) -> tuple[mx.array, mx.array, STPLossMetrics]:
    """Return ``NTP + lambda*STP`` loss and STP metrics.

    This is an opt-in Stream H helper. The default ``next_token_cross_entropy``
    path remains unchanged until a recipe explicitly enables STP.
    """

    document_ids = _extract_document_ids(batch)
    lm_batch = ensure_lm_batch(batch)
    next_token_loss, ntokens = next_token_cross_entropy(model, batch)
    cfg = config or STPLossConfig()
    hidden_states = _decoder_hidden_states_for_mtp(
        model,
        lm_batch,
        document_ids=document_ids[:, :-1] if document_ids is not None else None,
    )
    stp_loss = compute_stp_loss(hidden_states, n_spans=cfg.n_spans)
    total_loss = next_token_and_stp_loss(
        next_token_loss,
        stp_loss,
        loss_weight=cfg.loss_weight,
    )
    return total_loss, ntokens, STPLossMetrics(
        next_token_loss=next_token_loss,
        stp_loss=stp_loss,
        total_loss=total_loss,
        n_spans=cfg.n_spans,
        loss_weight=cfg.loss_weight,
    )


def next_token_cross_entropy_mtp_loss(
    model: nn.Module,
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
    *,
    config: MTPLossConfig | None = None,
) -> tuple[mx.array, mx.array]:
    """Train-loop compatible MTP loss returning only ``(total_loss, ntokens)``."""

    total_loss, ntokens, _ = next_token_cross_entropy_with_mtp(
        model,
        batch,
        config=config,
    )
    return total_loss, ntokens


def _decoder_hidden_states_for_mtp(
    model: nn.Module,
    batch: LMTokenBatch,
    *,
    document_ids: mx.array | None = None,
) -> mx.array:
    """Return normalized pre-lm-head hidden states for local MLX MTP loss.

    FastMTP consumes final decoder hidden states, not raw token embeddings. The
    current local tiny models do not expose a public ``return_hidden_states``
    flag, so this helper reconstructs the existing forward path from stable
    module attributes and fails closed for unsupported model surfaces.
    """

    decoder_hidden_states = getattr(model, "decoder_hidden_states", None)
    if callable(decoder_hidden_states):
        model_kwargs = batch.model_kwargs()
        if document_ids is not None:
            model_kwargs["document_ids"] = document_ids
        return decoder_hidden_states(batch.inputs, **model_kwargs)

    token_embedding = getattr(model, "token_embedding", None)
    position_embedding = getattr(model, "position_embedding", None)
    layers = getattr(model, "layers", None)
    norm = getattr(model, "norm", None)
    if not isinstance(token_embedding, nn.Embedding):
        raise TypeError("MTP loss requires model.token_embedding to be an nn.Embedding")
    if not isinstance(position_embedding, nn.Embedding):
        raise TypeError("MTP loss requires model.position_embedding to be an nn.Embedding")
    if not isinstance(layers, list):
        raise TypeError("MTP loss requires model.layers to be a list of decoder blocks")
    if not callable(norm):
        raise TypeError("MTP loss requires model.norm to be callable")

    input_ids = batch.inputs
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be shaped (B, S), got {input_ids.shape}")
    seq_length = input_ids.shape[1]
    positions = mx.arange(seq_length)[None, :]
    hidden_states = token_embedding(input_ids) + position_embedding(positions)

    ngram_hash_embedding = getattr(model, "ngram_hash_embedding", None)
    if ngram_hash_embedding is not None:
        hidden_states = hidden_states + ngram_hash_embedding(input_ids)

    hidden_states = hidden_states + _structure_hidden_state_delta(
        model,
        batch,
        hidden_dtype=hidden_states.dtype,
        seq_length=seq_length,
    )

    if document_ids is not None:
        if document_ids.shape != input_ids.shape:
            raise ValueError(
                f"document_ids shape {document_ids.shape} must match input_ids shape "
                f"{input_ids.shape}"
            )
        logits_fn = getattr(model, "__call__", None)
        if callable(logits_fn):
            _raise_if_negative_document_ids(document_ids)
        from cppmega_mlx.data.packing import mlx_document_boundary_mask

        mask = mlx_document_boundary_mask(document_ids, causal=True, expand_heads=True)
    else:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(
            seq_length,
            dtype=hidden_states.dtype,
        )
    for layer in layers:
        hidden_states = layer(hidden_states, mask)
    apply_norm = cast(Callable[[mx.array], mx.array], norm)
    return apply_norm(hidden_states)


def _structure_hidden_state_delta(
    model: nn.Module,
    batch: LMTokenBatch,
    *,
    hidden_dtype: mx.Dtype,
    seq_length: int,
) -> mx.array:
    structure_embedding = getattr(model, "structure_embedding", None)
    if structure_embedding is None:
        return mx.array(0.0, dtype=hidden_dtype)

    if isinstance(structure_embedding, StructureEmbedding):
        structure_embeddings = structure_embedding(
            structure_ids=batch.structure_ids[:, :-1] if batch.structure_ids is not None else None,
            dep_levels=batch.dep_levels[:, :-1] if batch.dep_levels is not None else None,
            ast_depth_ids=batch.ast_depth_ids[:, :-1] if batch.ast_depth_ids is not None else None,
            sibling_index_ids=batch.sibling_index_ids[:, :-1]
            if batch.sibling_index_ids is not None
            else None,
            node_type_ids=batch.node_type_ids[:, :-1] if batch.node_type_ids is not None else None,
            target_dtype=hidden_dtype,
        )
        if structure_embeddings.ndim == 3:
            return structure_embeddings
        return mx.array(0.0, dtype=hidden_dtype)

    if not isinstance(structure_embedding, nn.Embedding):
        raise TypeError(
            "MTP loss requires model.structure_embedding to be an nn.Embedding "
            "or StructureEmbedding"
        )

    structure_core_fn = getattr(model, "_structure_core", None)
    if callable(structure_core_fn):
        structure_core = structure_core_fn(
            batch.structure_ids[:, :-1] if batch.structure_ids is not None else None,
            batch.dep_levels[:, :-1] if batch.dep_levels is not None else None,
            seq_length,
        )
        if structure_core is not None:
            delta = structure_embedding(structure_core)
        else:
            delta = mx.array(0.0, dtype=hidden_dtype)
    else:
        delta = mx.array(0.0, dtype=hidden_dtype)

    for optional_ids in (
        batch.ast_depth_ids,
        batch.sibling_index_ids,
        batch.node_type_ids,
    ):
        if optional_ids is not None:
            channel_ids = optional_ids[:, :-1]
            delta = delta + structure_embedding(
                channel_ids % int(structure_embedding.weight.shape[0])
            )
    return delta


def _extract_document_ids(
    batch: LMTokenBatch | Mapping[str, mx.array] | mx.array,
) -> mx.array | None:
    if not isinstance(batch, Mapping):
        return None

    present = [name for name in _DOCUMENT_ID_ALIASES if name in batch]
    if not present:
        return None
    if len(present) > 1:
        raise ValueError(
            "batch mapping must provide only one document-id alias; got "
            f"{tuple(present)!r}"
        )

    tokens = batch.get("tokens")
    if not isinstance(tokens, mx.array):
        raise ValueError("batch mapping must contain a 'tokens' array")
    document_ids = batch[present[0]]
    if not isinstance(document_ids, mx.array):
        raise TypeError(f"{present[0]} must be an mx.array")
    _validate_document_id_batch(document_ids, tokens=tokens, alias=present[0])
    return document_ids.astype(mx.int32)


def _validate_document_id_batch(
    document_ids: mx.array,
    *,
    tokens: mx.array,
    alias: str,
) -> None:
    if document_ids.ndim != 2:
        raise ValueError(f"{alias} must be shaped (B, S), got {document_ids.shape}")
    if document_ids.shape != tokens.shape:
        raise ValueError(f"{alias} must match tokens shape {tokens.shape}, got {document_ids.shape}")
    _raise_if_negative_document_ids(document_ids, alias=alias)


def _raise_if_negative_document_ids(document_ids: mx.array, *, alias: str = "document_ids") -> None:
    has_negative = mx.any(document_ids.astype(mx.int32) < 0)
    mx.eval(has_negative)
    if bool(has_negative.item()):
        raise ValueError(f"{alias} must be non-negative for explicit packed batches")


__all__ = [
    "next_token_cut_cross_entropy",
    "next_token_cross_entropy",
    "next_token_cross_entropy_mtp_loss",
    "next_token_cross_entropy_with_stp",
    "next_token_cross_entropy_with_mtp",
]
