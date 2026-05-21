"""Tiny MLX token batches used by local trainer smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

SideChannelDropoutPolicy = Mapping[str, float]

_SIDE_CHANNEL_FAMILY_FIELDS: Mapping[str, tuple[str, ...]] = {
    "platform": ("platform_ids",),
    "syntax": ("ast_depth_ids", "sibling_index_ids", "node_type_ids"),
    "structure": ("structure_ids", "dep_levels"),
}


@dataclass(frozen=True)
class LMTokenBatch:
    """A dense next-token LM batch plus optional cppmega structure side-channels."""

    tokens: mx.array
    target_tokens: mx.array | None = None
    attention_mask: mx.array | None = None
    loss_mask: mx.array | None = None
    document_ids: mx.array | None = None
    structure_ids: mx.array | None = None
    dep_levels: mx.array | None = None
    ast_depth_ids: mx.array | None = None
    sibling_index_ids: mx.array | None = None
    node_type_ids: mx.array | None = None
    platform_ids: mx.array | None = None
    side_channels: Mapping[str, Mapping[str, mx.array]] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2:
            raise ValueError(f"tokens must be shaped (B, S), got {self.tokens.shape}")
        if self.tokens.shape[1] < 2:
            raise ValueError("tokens sequence length must be at least 2")
        if self.target_tokens is not None:
            if self.target_tokens.ndim != 2:
                raise ValueError(
                    f"target_tokens must be shaped (B, T), got {self.target_tokens.shape}"
                )
            if self.target_tokens.shape[0] != self.tokens.shape[0]:
                raise ValueError(
                    "target_tokens batch dimension must match tokens batch "
                    f"{self.tokens.shape[0]}, got {self.target_tokens.shape[0]}"
                )
            if self.target_tokens.shape[1] not in (
                self.tokens.shape[1],
                self.tokens.shape[1] - 1,
            ):
                raise ValueError(
                    "target_tokens sequence length must equal tokens length or "
                    f"tokens length - 1, got {self.target_tokens.shape[1]} for "
                    f"tokens shape {self.tokens.shape}"
                )

        for name, value in self.structure_fields().items():
            if value is not None and value.shape != self.tokens.shape:
                raise ValueError(
                    f"{name} must match tokens shape {self.tokens.shape}, got {value.shape}"
                )

        if self.attention_mask is not None and self.attention_mask.shape != self.tokens.shape:
            raise ValueError(
                "attention_mask must match tokens shape "
                f"{self.tokens.shape}, got {self.attention_mask.shape}"
            )
        if self.loss_mask is not None:
            if self.loss_mask.ndim != 2:
                raise ValueError(f"loss_mask must be shaped (B, T), got {self.loss_mask.shape}")
            valid_loss_shapes = {self.tokens.shape, self.targets.shape}
            if self.loss_mask.shape not in valid_loss_shapes:
                raise ValueError(
                    f"loss_mask must match tokens shape {self.tokens.shape} or "
                    f"targets shape {self.targets.shape}, got {self.loss_mask.shape}"
                )
        if self.document_ids is not None:
            if self.document_ids.ndim != 2:
                raise ValueError(
                    f"document_ids must be shaped (B, S), got {self.document_ids.shape}"
                )
            if self.document_ids.shape != self.tokens.shape:
                raise ValueError(
                    "document_ids must match tokens shape "
                    f"{self.tokens.shape}, got {self.document_ids.shape}"
                )
            has_negative_doc = mx.any(self.document_ids.astype(mx.int32) < 0)
            mx.eval(has_negative_doc)
            if bool(has_negative_doc.item()):
                raise ValueError("document_ids must be non-negative")
        if self.platform_ids is not None:
            if self.platform_ids.ndim not in (2, 3):
                raise ValueError(
                    "platform_ids must be shaped (B, K) or (B, S, K), "
                    f"got {self.platform_ids.shape}"
                )
            if self.platform_ids.shape[0] != self.tokens.shape[0]:
                raise ValueError(
                    "platform_ids batch dimension must match tokens batch "
                    f"{self.tokens.shape[0]}, got {self.platform_ids.shape[0]}"
                )
            if self.platform_ids.ndim == 3 and self.platform_ids.shape[1] != self.tokens.shape[1]:
                raise ValueError(
                    "token-local platform_ids sequence dimension must match tokens "
                    f"{self.tokens.shape[1]}, got {self.platform_ids.shape[1]}"
                )
            has_negative = mx.any(self.platform_ids.astype(mx.int32) < 0)
            mx.eval(has_negative)
            if bool(has_negative.item()):
                raise ValueError("platform_ids must be non-negative")
        if self.side_channels is not None:
            _validate_side_channel_map(self.side_channels)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided")

    @property
    def inputs(self) -> mx.array:
        if self.target_tokens is not None and self.target_tokens.shape[1] == self.tokens.shape[1]:
            return self.tokens
        return self.tokens[:, :-1]

    @property
    def targets(self) -> mx.array:
        if self.target_tokens is not None:
            return self.target_tokens
        return self.tokens[:, 1:]

    @property
    def target_mask(self) -> mx.array:
        if self.loss_mask is not None:
            return self._target_aligned(self.loss_mask).astype(mx.float32)
        if self.attention_mask is not None:
            return self._target_aligned(self.attention_mask).astype(mx.float32)
        return mx.ones(self.targets.shape, dtype=mx.float32)

    @property
    def input_document_ids(self) -> mx.array | None:
        if self.document_ids is None:
            return None
        return self._input_aligned(self.document_ids).astype(mx.int32)

    @property
    def target_document_ids(self) -> mx.array | None:
        if self.document_ids is None:
            return None
        return self._target_aligned(self.document_ids).astype(mx.int32)

    def structure_fields(self) -> dict[str, mx.array | None]:
        return {
            "structure_ids": self.structure_ids,
            "dep_levels": self.dep_levels,
            "ast_depth_ids": self.ast_depth_ids,
            "sibling_index_ids": self.sibling_index_ids,
            "node_type_ids": self.node_type_ids,
        }

    def model_kwargs(self) -> dict[str, mx.array]:
        kwargs = {
            name: self._input_aligned(value)
            for name, value in self.structure_fields().items()
            if value is not None
        }
        if self.platform_ids is not None:
            kwargs["platform_ids"] = (
                self._input_aligned(self.platform_ids)
                if self.platform_ids.ndim == 3
                else self.platform_ids
            )
        return kwargs

    def side_channel_map(self) -> dict[str, dict[str, mx.array]]:
        """Return side-channel tensors grouped by family and column name."""

        out: dict[str, dict[str, mx.array]] = {
            family: {}
            for family in _SIDE_CHANNEL_FAMILY_FIELDS
        }
        for family, fields in _SIDE_CHANNEL_FAMILY_FIELDS.items():
            for field_name in fields:
                value = getattr(self, field_name)
                if value is not None:
                    out[family][field_name] = value
        if self.side_channels is not None:
            for family, columns in self.side_channels.items():
                out.setdefault(family, {}).update(dict(columns))
        return {family: columns for family, columns in out.items() if columns}

    def with_side_channel_dropout(
        self,
        policy: SideChannelDropoutPolicy | None,
        *,
        seed: int | None = None,
        training: bool = True,
    ) -> "LMTokenBatch":
        """Apply shape-stable family dropout by zeroing selected channels."""

        if not training or not policy:
            return self
        rates = _validate_side_channel_dropout_policy(policy)
        if not rates:
            return self
        rng = np.random.default_rng(seed)
        dropped = {
            family
            for family, rate in rates.items()
            if rate >= 1.0 or (rate > 0.0 and float(rng.random()) < rate)
        }
        if not dropped:
            return self

        kwargs = self.as_dict(include_metadata=True, include_side_channels=True)
        for family in dropped:
            for field_name in _SIDE_CHANNEL_FAMILY_FIELDS.get(family, ()):
                value = kwargs.get(field_name)
                if value is not None:
                    kwargs[field_name] = mx.zeros_like(value)
        if self.side_channels is not None:
            kwargs["side_channels"] = {
                family: {
                    name: mx.zeros_like(value) if family in dropped else value
                    for name, value in columns.items()
                }
                for family, columns in self.side_channels.items()
            }
        return LMTokenBatch(**kwargs)

    def _target_aligned(self, value: mx.array) -> mx.array:
        if tuple(value.shape) == tuple(self.targets.shape):
            return value
        if tuple(value.shape) == tuple(self.tokens.shape):
            if (
                self.target_tokens is not None
                and self.target_tokens.shape[1] == self.tokens.shape[1]
            ):
                return value
            return value[:, 1:]
        raise ValueError(
            f"value shape {value.shape} cannot align to targets {self.targets.shape}"
        )

    def _input_aligned(self, value: mx.array) -> mx.array:
        if value.ndim == self.tokens.ndim + 1:
            if tuple(value.shape[:2]) == tuple(self.inputs.shape):
                return value
            if tuple(value.shape[:2]) == tuple(self.tokens.shape):
                if (
                    self.target_tokens is not None
                    and self.target_tokens.shape[1] == self.tokens.shape[1]
                ):
                    return value
                return value[:, :-1, :]
        if tuple(value.shape) == tuple(self.inputs.shape):
            return value
        if tuple(value.shape) == tuple(self.tokens.shape):
            if (
                self.target_tokens is not None
                and self.target_tokens.shape[1] == self.tokens.shape[1]
            ):
                return value
            return value[:, :-1]
        raise ValueError(
            f"value shape {value.shape} cannot align to inputs {self.inputs.shape}"
        )

    def training_metadata(self) -> dict[str, Any]:
        """Return optional non-logit training metadata as a plain mapping."""

        return {} if self.metadata is None else dict(self.metadata)

    def as_dict(
        self,
        *,
        include_metadata: bool = False,
        include_side_channels: bool = False,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"tokens": self.tokens}
        if self.target_tokens is not None:
            data["target_tokens"] = self.target_tokens
        if self.attention_mask is not None:
            data["attention_mask"] = self.attention_mask
        if self.loss_mask is not None:
            data["loss_mask"] = self.loss_mask
        if self.document_ids is not None:
            data["document_ids"] = self.document_ids
        data.update({k: v for k, v in self.structure_fields().items() if v is not None})
        if self.platform_ids is not None:
            data["platform_ids"] = self.platform_ids
        if include_side_channels and self.side_channels is not None:
            data["side_channels"] = self.side_channels
        if include_metadata and self.metadata is not None:
            data["metadata"] = self.metadata
        return data


_DOCUMENT_ID_ALIASES = ("document_ids", "doc_ids", "packing_document_ids")


def _validate_side_channel_map(
    side_channels: Mapping[str, Mapping[str, mx.array]],
) -> None:
    if not isinstance(side_channels, Mapping):
        raise ValueError("side_channels must be a mapping when provided")
    for family, columns in side_channels.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("side channel family names must be non-empty strings")
        if not isinstance(columns, Mapping):
            raise ValueError(f"side channel family {family!r} must map column names")
        for name, value in columns.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("side channel column names must be non-empty strings")
            if not isinstance(value, mx.array):
                raise ValueError(
                    f"side channel {family}.{name} must be an mlx array"
                )


def _validate_side_channel_dropout_policy(
    policy: SideChannelDropoutPolicy,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for family, rate in policy.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("side channel dropout family names must be non-empty")
        value = float(rate)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"side channel dropout for {family!r} must be in [0, 1], got {rate!r}"
            )
        rates[family] = value
    return rates


def _document_ids_from_mapping(batch: Mapping[str, Any]) -> Any | None:
    present = [
        alias
        for alias in _DOCUMENT_ID_ALIASES
        if alias in batch and batch[alias] is not None
    ]
    if len(present) > 1:
        raise ValueError(
            "only one document-id alias may be provided; got "
            f"{', '.join(present)}"
        )
    if not present:
        return None
    alias = present[0]
    value = batch[alias]
    has_negative_doc = mx.any(value.astype(mx.int32) < 0)
    mx.eval(has_negative_doc)
    if bool(has_negative_doc.item()):
        raise ValueError(f"{alias} must be non-negative")
    return value


def ensure_lm_batch(batch: LMTokenBatch | Mapping[str, Any] | mx.array) -> LMTokenBatch:
    """Normalize supported tiny-trainer batch inputs into LMTokenBatch."""

    if isinstance(batch, LMTokenBatch):
        return batch
    if isinstance(batch, mx.array):
        return LMTokenBatch(tokens=batch)
    if isinstance(batch, Mapping):
        if "tokens" not in batch:
            raise ValueError("batch mapping must contain a 'tokens' array")
        return LMTokenBatch(
            tokens=batch["tokens"],
            target_tokens=batch.get("target_tokens", batch.get("target_ids")),
            attention_mask=batch.get("attention_mask"),
            loss_mask=batch.get("loss_mask"),
            document_ids=_document_ids_from_mapping(batch),
            structure_ids=batch.get("structure_ids"),
            dep_levels=batch.get("dep_levels"),
            ast_depth_ids=batch.get("ast_depth_ids"),
            sibling_index_ids=batch.get("sibling_index_ids"),
            node_type_ids=batch.get("node_type_ids"),
            platform_ids=batch.get("platform_ids"),
            side_channels=batch.get("side_channels"),
            metadata=batch.get("metadata"),
        )
    raise TypeError(f"unsupported batch type: {type(batch)!r}")


def synthetic_token_batch(
    *,
    batch_size: int = 2,
    seq_length: int = 8,
    vocab_size: int = 64,
    seed: int = 0,
    include_structure: bool = False,
) -> LMTokenBatch:
    """Create a deterministic synthetic batch for GPU smoke tests."""

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    if seq_length < 2:
        raise ValueError("seq_length must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    tokens = mx.array(
        rng.integers(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)
    )
    attention_mask = mx.ones((batch_size, seq_length), dtype=mx.float32)

    if not include_structure:
        return LMTokenBatch(tokens=tokens, attention_mask=attention_mask)

    structure_vocab = max(2, min(vocab_size, 32))
    structure_ids = mx.array(
        rng.integers(
            0, structure_vocab, size=(batch_size, seq_length), dtype=np.int32
        )
    )
    dep_levels = mx.array(
        rng.integers(0, 8, size=(batch_size, seq_length), dtype=np.int32)
    )
    ast_depth_ids = mx.array(
        rng.integers(0, 8, size=(batch_size, seq_length), dtype=np.int32)
    )
    sibling_index_ids = mx.array(
        rng.integers(0, 8, size=(batch_size, seq_length), dtype=np.int32)
    )
    node_type_ids = mx.array(
        rng.integers(0, structure_vocab, size=(batch_size, seq_length), dtype=np.int32)
    )
    return LMTokenBatch(
        tokens=tokens,
        attention_mask=attention_mask,
        structure_ids=structure_ids,
        dep_levels=dep_levels,
        ast_depth_ids=ast_depth_ids,
        sibling_index_ids=sibling_index_ids,
        node_type_ids=node_type_ids,
    )


__all__ = [
    "LMTokenBatch",
    "SideChannelDropoutPolicy",
    "ensure_lm_batch",
    "synthetic_token_batch",
]
