"""Tiny MLX token batches used by local trainer smoke tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.graph_packet import GraphBatch
from cppmega_mlx.data.integer_validation import validated_integer_array
from cppmega_mlx.data.nanochat_pipeline.packed_rows_schema import (
    LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1,
)

SideChannelDropoutPolicy = Mapping[str, float]

_BATCH_VALUES_PREVALIDATED: ContextVar[bool] = ContextVar(
    "cppmega_mlx_batch_values_prevalidated",
    default=False,
)


def batch_values_are_prevalidated() -> bool:
    """Return whether eager device-value checks ran before an MLX transform."""

    return _BATCH_VALUES_PREVALIDATED.get()


@contextmanager
def prevalidated_batch_values() -> Iterator[None]:
    """Skip transform-unsafe checks after the caller validates the same batch."""

    token = _BATCH_VALUES_PREVALIDATED.set(True)
    try:
        yield
    finally:
        _BATCH_VALUES_PREVALIDATED.reset(token)

_SIDE_CHANNEL_FAMILY_FIELDS: Mapping[str, tuple[str, ...]] = {
    "domain_routes": ("domain_ids", "role_ids", "confidence_ids"),
    "platform": ("platform_ids",),
    "syntax": ("ast_depth_ids", "sibling_index_ids", "node_type_ids"),
    "structure": ("structure_ids", "dep_levels"),
}
_DOMAIN_ROUTE_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "domain_ids": ("token_domain_ids",),
    "role_ids": ("token_role_ids",),
    "entity_ids": ("token_entity_ids",),
    "scope_ids": ("token_scope_ids",),
    "source_doc_ids": ("token_source_doc_ids",),
    "source_identity_ids": ("token_source_identity_ids",),
    "confidence_ids": ("token_confidence_ids",),
}


@dataclass(frozen=True)
class LMTokenBatch:
    """A dense next-token LM batch plus optional cppmega structure side-channels.

    Persisted ``token_*`` domain aliases are accepted in the domain side-channel
    family, but only canonical ``domain_ids``/``role_ids``/``confidence_ids``
    names are exposed to model kwargs.
    """

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
    domain_ids: mx.array | None = None
    role_ids: mx.array | None = None
    confidence_ids: mx.array | None = None
    graph_batch: GraphBatch | None = None
    graph_attention_bias: mx.array | None = None
    graph_edge_kind_bias: mx.array | None = None
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
            if not isinstance(self.document_ids, mx.array):
                raise TypeError("document_ids must be an mlx array")
            if self.document_ids.ndim != 2:
                raise ValueError(
                    f"document_ids must be shaped (B, S), got {self.document_ids.shape}"
                )
            if self.document_ids.shape != self.tokens.shape:
                raise ValueError(
                    "document_ids must match tokens shape "
                    f"{self.tokens.shape}, got {self.document_ids.shape}"
                )
        if self.document_ids is not None and not batch_values_are_prevalidated():
            _validate_packed_sequence_contract(
                document_ids=self.document_ids,
                attention_mask=self.attention_mask,
                loss_mask=self.loss_mask,
                where="LMTokenBatch",
            )
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
            if not batch_values_are_prevalidated():
                has_negative = mx.any(self.platform_ids.astype(mx.int32) < 0)
                mx.eval(has_negative)
                if bool(has_negative.item()):
                    raise ValueError("platform_ids must be non-negative")
        if self.side_channels is not None:
            _validate_side_channel_map(self.side_channels)
        resolved_domain_routes = self.resolved_domain_route_fields()
        for name, value in resolved_domain_routes.items():
            if value is not None and value.shape != self.tokens.shape:
                raise ValueError(
                    f"{name} must match tokens shape {self.tokens.shape}, got {value.shape}"
                )
        if self.graph_batch is not None:
            if not isinstance(self.graph_batch, GraphBatch):
                raise TypeError(
                    "graph_batch must be a GraphBatch when provided, got "
                    f"{type(self.graph_batch).__name__}"
                )
            if self.graph_batch.batch_size not in (1, int(self.tokens.shape[0])):
                raise ValueError(
                    "graph_batch batch size must be 1 or tokens batch "
                    f"{int(self.tokens.shape[0])}, got {self.graph_batch.batch_size}"
                )
        if self.graph_attention_bias is not None or self.graph_edge_kind_bias is not None:
            if self.graph_batch is not None:
                raise ValueError(
                    "provide graph_batch or fixed graph biases, not both"
                )
            expected = (
                int(self.inputs.shape[0]),
                int(self.inputs.shape[1]),
                int(self.inputs.shape[1]),
            )
            for name, value in (
                ("graph_attention_bias", self.graph_attention_bias),
                ("graph_edge_kind_bias", self.graph_edge_kind_bias),
            ):
                if value is not None and tuple(value.shape) != expected:
                    raise ValueError(
                        f"{name} must be shaped for model inputs as {expected}, "
                        f"got {tuple(value.shape)}"
                    )
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
            # loss_mask[i] gates the label produced from input token i. It is a
            # transition sidecar, not a property of target token i + 1.
            return self._input_aligned(self.loss_mask).astype(mx.float32)
        if self.attention_mask is not None:
            return self._target_aligned(self.attention_mask).astype(mx.float32)
        return mx.ones(self.targets.shape, dtype=mx.float32)

    @property
    def input_document_ids(self) -> mx.array | None:
        if self.document_ids is None:
            return None
        return self._input_aligned(self.document_ids)

    @property
    def target_document_ids(self) -> mx.array | None:
        if self.document_ids is None:
            return None
        return self._target_aligned(self.document_ids)

    def structure_fields(self) -> dict[str, mx.array | None]:
        return {
            "structure_ids": self.structure_ids,
            "dep_levels": self.dep_levels,
            "ast_depth_ids": self.ast_depth_ids,
            "sibling_index_ids": self.sibling_index_ids,
            "node_type_ids": self.node_type_ids,
        }

    def domain_route_fields(self) -> dict[str, mx.array | None]:
        return {
            "domain_ids": self.domain_ids,
            "role_ids": self.role_ids,
            "confidence_ids": self.confidence_ids,
        }

    def resolved_domain_route_fields(self) -> dict[str, mx.array | None]:
        """Return canonical model-facing domain routes without alias ambiguity."""

        nested = _canonical_domain_route_channels(
            None
            if self.side_channels is None
            else self.side_channels.get("domain_routes")
        )
        resolved: dict[str, mx.array | None] = {}
        for field_name, direct in self.domain_route_fields().items():
            nested_value = nested.get(field_name)
            if direct is not None and nested_value is not None:
                raise ValueError(
                    "domain routes cannot be provided both directly and through "
                    f"side_channels.domain_routes: {field_name}"
                )
            resolved[field_name] = direct if direct is not None else nested_value
        return resolved

    def model_kwargs(self) -> dict[str, Any]:
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
        if self.graph_batch is not None:
            kwargs["graph_batch"] = self.graph_batch.input_aligned(
                source_sequence_length=int(self.tokens.shape[1]),
                input_sequence_length=int(self.inputs.shape[1]),
            )
        if self.graph_attention_bias is not None:
            kwargs["block_bias"] = self.graph_attention_bias
        if self.graph_edge_kind_bias is not None:
            kwargs["edge_kind_bias"] = self.graph_edge_kind_bias
        for field_name, value in self.resolved_domain_route_fields().items():
            if value is not None:
                kwargs[field_name] = self._input_aligned(value)
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
                if family == "domain_routes":
                    canonical = _canonical_domain_route_channels(columns)
                    out.setdefault(family, {}).update(canonical)
                    alias_names = {
                        alias
                        for aliases in _DOMAIN_ROUTE_FIELD_ALIASES.values()
                        for alias in aliases
                    }
                    out[family].update(
                        {
                            name: value
                            for name, value in columns.items()
                            if name not in alias_names
                            and name not in _DOMAIN_ROUTE_FIELD_ALIASES
                        }
                    )
                else:
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
        data.update(
            {key: value for key, value in self.domain_route_fields().items() if value is not None}
        )
        if self.graph_batch is not None:
            data["graph_batch"] = self.graph_batch
        if self.graph_attention_bias is not None:
            data["graph_attention_bias"] = self.graph_attention_bias
        if self.graph_edge_kind_bias is not None:
            data["graph_edge_kind_bias"] = self.graph_edge_kind_bias
        if include_side_channels and self.side_channels is not None:
            data["side_channels"] = self.side_channels
        if include_metadata and self.metadata is not None:
            data["metadata"] = self.metadata
        return data


_DOCUMENT_ID_ALIASES = ("document_ids", "doc_ids", "packing_document_ids")


def _canonical_domain_route_channels(
    channels: Mapping[str, mx.array] | None,
) -> dict[str, mx.array]:
    if channels is None:
        return {}
    canonical: dict[str, mx.array] = {}
    for field_name, aliases in _DOMAIN_ROUTE_FIELD_ALIASES.items():
        present = [
            name
            for name in (field_name, *aliases)
            if name in channels and channels[name] is not None
        ]
        if len(present) > 1:
            raise ValueError(
                "domain route declared more than once via canonical/alias keys: "
                f"{present}"
            )
        if present:
            value = channels[present[0]]
            if not isinstance(value, mx.array):
                raise ValueError(
                    f"domain route {present[0]!r} must be an mlx array"
                )
            canonical[field_name] = value
    return canonical


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
    if not batch_values_are_prevalidated():
        _validate_document_id_values(value, where=alias)
    return value


def _validate_document_id_values(value: Any, *, where: str) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{where} must be an mlx array")
    try:
        validated_integer_array(
            value,
            where=where,
            min_value=0,
            allow_integral_float=False,
        )
    except ValueError as error:
        if "values must be >= 0" in str(error):
            raise ValueError(f"{where} must be non-negative") from error
        raise


def _binary_mask(value: Any, *, where: str, shape: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{where} must be shaped {shape}, got {array.shape}")
    invalid = ~np.isfinite(array) | ((array != 0) & (array != 1))
    if np.any(invalid):
        row, column = np.argwhere(invalid)[0]
        raise ValueError(
            f"{where} must contain only finite binary values 0 or 1; "
            f"row {int(row)}, column {int(column)} has {array[row, column]!r}"
        )
    return array != 0


def _validate_packed_sequence_contract(
    *,
    document_ids: Any | None,
    attention_mask: Any | None,
    loss_mask: Any | None,
    where: str,
) -> None:
    """Validate row-local packing boundaries without guessing unseen targets."""

    documents: np.ndarray | None = None
    if document_ids is not None:
        try:
            documents = validated_integer_array(
                document_ids,
                where=f"{where}.document_ids",
                min_value=0,
                allow_integral_float=False,
            )
        except ValueError as error:
            if "values must be >= 0" in str(error):
                raise ValueError(f"{where}.document_ids must be non-negative") from error
            raise
        if documents.ndim != 2:
            raise ValueError(
                f"{where}.document_ids must be shaped (B, S), got {documents.shape}"
            )

    valid_tokens: np.ndarray | None = None
    if attention_mask is not None:
        valid_tokens = _binary_mask(
            attention_mask,
            where=f"{where}.attention_mask",
            shape="(B, S)",
        )
        holes = (~valid_tokens[:, :-1]) & valid_tokens[:, 1:]
        if np.any(holes):
            row, column = np.argwhere(holes)[0]
            raise ValueError(
                f"{where}: padding must be trailing; row {int(row)} has a valid "
                f"token after padding at column {int(column) + 1}"
            )
        if documents is not None and documents.shape != valid_tokens.shape:
            raise ValueError(
                f"{where}: document_ids shape {documents.shape} must match "
                f"attention_mask shape {valid_tokens.shape}"
            )
    elif documents is not None:
        valid_tokens = np.ones(documents.shape, dtype=np.bool_)

    if documents is not None and valid_tokens is not None:
        for row_index, (row_documents, row_valid) in enumerate(
            zip(documents, valid_tokens, strict=True)
        ):
            active = row_documents[row_valid]
            if not active.size:
                continue
            run_ids = active[np.r_[True, active[1:] != active[:-1]]]
            unique_ids, counts = np.unique(run_ids, return_counts=True)
            reused = unique_ids[counts > 1]
            if reused.size:
                raise ValueError(
                    f"{where}: document ID {int(reused[0])} is reused "
                    f"non-contiguously in row {row_index}"
                )

    if loss_mask is None:
        return
    losses = _binary_mask(
        loss_mask,
        where=f"{where}.loss_mask",
        shape="(B, T)",
    )
    if valid_tokens is None:
        return
    batch_size, sequence_length = valid_tokens.shape
    if losses.shape[0] != batch_size or losses.shape[1] not in (
        sequence_length,
        sequence_length - 1,
    ):
        raise ValueError(
            f"{where}.loss_mask shape {losses.shape} cannot align to packed "
            f"sequence shape {valid_tokens.shape}"
        )

    transition_count = min(losses.shape[1], sequence_length - 1)
    valid_pairs = (
        valid_tokens[:, :transition_count]
        & valid_tokens[:, 1 : transition_count + 1]
    )
    allowed = np.zeros(losses.shape, dtype=np.bool_)
    allowed[:, :transition_count] = valid_pairs
    if documents is not None:
        allowed[:, :transition_count] &= (
            documents[:, :transition_count]
            == documents[:, 1 : transition_count + 1]
        )

    # A full-width explicit target or a cut window can supervise the final
    # valid source even though its target is not present in this physical row.
    if losses.shape[1] == sequence_length:
        allowed[:, -1] = valid_tokens[:, -1]
    violations = losses & ~allowed
    if np.any(violations):
        row, column = np.argwhere(violations)[0]
        raise ValueError(
            f"{where}.loss_mask must be zero at cross-document/padding transitions; "
            f"row {int(row)}, source column {int(column)}"
        )


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
            domain_ids=batch.get("domain_ids"),
            role_ids=batch.get("role_ids"),
            confidence_ids=batch.get("confidence_ids"),
            graph_batch=batch.get("graph_batch"),
            graph_attention_bias=batch.get("graph_attention_bias"),
            graph_edge_kind_bias=batch.get("graph_edge_kind_bias"),
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

    # Match the default StructureEmbedding/DenseCppLM category contract. The old
    # arbitrary cap of 32 generated corrupt IDs that only worked because the
    # embedding silently clamped them into its final bucket.
    # The production hybrid contract has 11 structural categories (IDs 0..10).
    # Keep synthetic fixtures inside that contract instead of generating an
    # out-of-range ID that only older clamping behavior happened to tolerate.
    structure_vocab = max(2, min(vocab_size, 11))
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
    "LOSS_MASK_ALIGNMENT_SOURCE_TOKEN_PREDICTS_NEXT_V1",
    "LMTokenBatch",
    "SideChannelDropoutPolicy",
    "batch_values_are_prevalidated",
    "ensure_lm_batch",
    "prevalidated_batch_values",
    "synthetic_token_batch",
]
