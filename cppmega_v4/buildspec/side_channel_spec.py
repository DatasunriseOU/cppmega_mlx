"""Side-channel conditioning configuration.

Pure data layer for token metadata conditioning. This module intentionally
does not know how any language extractor works; it only describes which
families may be consumed and how missing data should be handled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar


class SideChannelMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"
    IF_AVAILABLE = "if_available"


class SideChannelEmbedding(str, Enum):
    CATEGORICAL = "categorical"
    NUMERIC_BUCKET = "numeric_bucket"
    SPAN = "span"
    EDGE_BIAS = "edge_bias"
    NONE = "none"


class SideChannelFallback(str, Enum):
    ZEROS = "zeros"
    UNKNOWN_ID = "unknown_id"
    DROP_FAMILY = "drop_family"
    ERROR = "error"


class InferenceEnrichmentSource(str, Enum):
    NONE = "none"
    PROMPT_ONLY = "prompt_only"
    PARSE_IF_POSSIBLE = "parse_if_possible"
    PROJECT_INDEX = "project_index"
    AUTO = "auto"


class InferenceFailPolicy(str, Enum):
    DROP_FAMILY = "drop_family"
    TEXT_ONLY = "text_only"
    ERROR = "error"


class PackingPolicy(str, Enum):
    SEQUENTIAL = "sequential"
    BEST_FIT = "best_fit"


_EnumT = TypeVar("_EnumT", bound=Enum)


def _coerce_enum(enum_cls: type[_EnumT], value: _EnumT | str, field_name: str) -> _EnumT:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = sorted(v.value for v in enum_cls)
        raise ValueError(f"{field_name}={value!r} not in {allowed}") from exc


def _str_tuple(value: str | tuple[str, ...] | list[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        out = (value,)
    else:
        out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
    return out


@dataclass(frozen=True)
class FamilySpec:
    """Configuration for one side-channel family."""

    mode: SideChannelMode | str = SideChannelMode.IF_AVAILABLE
    columns: tuple[str, ...] | list[str] = ()
    embedding: SideChannelEmbedding | str = SideChannelEmbedding.CATEGORICAL
    dropout: float = 0.0
    residual_scale: float = 1.0
    fallback: SideChannelFallback | str = SideChannelFallback.DROP_FAMILY
    language_scope: tuple[str, ...] | list[str] | str = ("any",)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode",
            _coerce_enum(SideChannelMode, self.mode, "mode"),
        )
        object.__setattr__(
            self, "embedding",
            _coerce_enum(SideChannelEmbedding, self.embedding, "embedding"),
        )
        object.__setattr__(
            self, "fallback",
            _coerce_enum(SideChannelFallback, self.fallback, "fallback"),
        )
        object.__setattr__(
            self, "columns",
            _str_tuple(self.columns, "columns") if self.columns else (),
        )
        object.__setattr__(
            self, "language_scope",
            _str_tuple(self.language_scope, "language_scope"),
        )
        if not 0.0 <= float(self.dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout!r}")
        object.__setattr__(self, "dropout", float(self.dropout))
        if float(self.residual_scale) < 0.0:
            raise ValueError(
                f"residual_scale must be >= 0, got {self.residual_scale!r}"
            )
        object.__setattr__(self, "residual_scale", float(self.residual_scale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "columns": list(self.columns),
            "embedding": self.embedding.value,
            "dropout": self.dropout,
            "residual_scale": self.residual_scale,
            "fallback": self.fallback.value,
            "language_scope": list(self.language_scope),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FamilySpec":
        return cls(**dict(payload))


@dataclass(frozen=True)
class InferenceEnrichmentSpec:
    """How inference attempts to derive side channels from input context."""

    source: InferenceEnrichmentSource | str = InferenceEnrichmentSource.AUTO
    fail_policy: InferenceFailPolicy | str = InferenceFailPolicy.DROP_FAMILY
    timeout_ms: int = 500
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source",
            _coerce_enum(InferenceEnrichmentSource, self.source, "source"),
        )
        object.__setattr__(
            self, "fail_policy",
            _coerce_enum(InferenceFailPolicy, self.fail_policy, "fail_policy"),
        )
        if int(self.timeout_ms) < 0:
            raise ValueError(f"timeout_ms must be >= 0, got {self.timeout_ms!r}")
        object.__setattr__(self, "timeout_ms", int(self.timeout_ms))
        object.__setattr__(self, "cache_enabled", bool(self.cache_enabled))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "fail_policy": self.fail_policy.value,
            "timeout_ms": self.timeout_ms,
            "cache_enabled": self.cache_enabled,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InferenceEnrichmentSpec":
        return cls(**dict(payload))


def default_side_channel_families() -> dict[str, FamilySpec]:
    return {
        "platform": FamilySpec(
            mode=SideChannelMode.AUTO,
            columns=("platform_ids", "source_platform_ids"),
            embedding=SideChannelEmbedding.CATEGORICAL,
            dropout=0.10,
            fallback=SideChannelFallback.DROP_FAMILY,
        ),
        "syntax": FamilySpec(
            mode=SideChannelMode.IF_AVAILABLE,
            columns=(
                "token_ast_depth",
                "token_sibling_index",
                "token_ast_node_type",
            ),
            embedding=SideChannelEmbedding.CATEGORICAL,
            dropout=0.25,
            fallback=SideChannelFallback.DROP_FAMILY,
        ),
        "structure": FamilySpec(
            mode=SideChannelMode.IF_AVAILABLE,
            columns=(
                "token_structure_ids",
                "token_dep_levels",
                "token_chunk_starts",
                "token_chunk_ends",
                "token_chunk_kinds",
                "token_chunk_dep_levels",
            ),
            embedding=SideChannelEmbedding.CATEGORICAL,
            dropout=0.25,
            fallback=SideChannelFallback.DROP_FAMILY,
        ),
        "semantic_graph": FamilySpec(
            mode=SideChannelMode.IF_AVAILABLE,
            columns=(
                "token_symbol_ids",
                "token_call_targets",
                "token_type_refs",
                "token_def_use",
                "token_call_edges",
                "token_type_edges",
            ),
            embedding=SideChannelEmbedding.EDGE_BIAS,
            dropout=0.50,
            fallback=SideChannelFallback.DROP_FAMILY,
        ),
        "temporal_diff": FamilySpec(
            mode=SideChannelMode.OFF,
            columns=(
                "token_change_mask_pre",
                "token_change_mask_post",
                "hunk_id_per_token",
                "edit_op_per_token",
            ),
            embedding=SideChannelEmbedding.CATEGORICAL,
            dropout=0.0,
            fallback=SideChannelFallback.DROP_FAMILY,
        ),
    }


@dataclass(frozen=True)
class SideChannelSpec:
    """Top-level side-channel conditioning policy."""

    mode: SideChannelMode | str = SideChannelMode.AUTO
    families: Mapping[str, FamilySpec | Mapping[str, Any]] = field(
        default_factory=default_side_channel_families
    )
    inference: InferenceEnrichmentSpec | Mapping[str, Any] = field(
        default_factory=InferenceEnrichmentSpec
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mode",
            _coerce_enum(SideChannelMode, self.mode, "mode"),
        )
        families: dict[str, FamilySpec] = {}
        for name, spec in self.families.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("family names must be non-empty strings")
            if isinstance(spec, FamilySpec):
                families[name] = spec
            else:
                families[name] = FamilySpec.from_dict(spec)
        object.__setattr__(self, "families", families)
        if isinstance(self.inference, InferenceEnrichmentSpec):
            inference = self.inference
        else:
            inference = InferenceEnrichmentSpec.from_dict(self.inference)
        object.__setattr__(self, "inference", inference)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "families": {
                name: spec.to_dict()
                for name, spec in self.families.items()
            },
            "inference": self.inference.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SideChannelSpec":
        return cls(**dict(payload))


@dataclass(frozen=True)
class DataMaterializationSpec:
    """Packed-row parquet materialization policy."""

    packing_policy: PackingPolicy | str = PackingPolicy.BEST_FIT
    max_seq_len: int = 4096
    pad_to_max: bool = True
    include_provenance: bool = True
    required_token_fields: tuple[str, ...] | list[str] = (
        "input_ids",
        "target_ids",
        "loss_mask",
        "doc_ids",
        "pack_id",
        "valid_token_count",
        "num_docs",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "packing_policy",
            _coerce_enum(PackingPolicy, self.packing_policy, "packing_policy"),
        )
        if int(self.max_seq_len) <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {self.max_seq_len!r}")
        object.__setattr__(self, "max_seq_len", int(self.max_seq_len))
        object.__setattr__(self, "pad_to_max", bool(self.pad_to_max))
        object.__setattr__(self, "include_provenance", bool(self.include_provenance))
        object.__setattr__(
            self,
            "required_token_fields",
            _str_tuple(self.required_token_fields, "required_token_fields"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "packing_policy": self.packing_policy.value,
            "max_seq_len": self.max_seq_len,
            "pad_to_max": self.pad_to_max,
            "include_provenance": self.include_provenance,
            "required_token_fields": list(self.required_token_fields),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataMaterializationSpec":
        return cls(**dict(payload))


__all__ = [
    "DataMaterializationSpec",
    "FamilySpec",
    "InferenceEnrichmentSpec",
    "InferenceEnrichmentSource",
    "InferenceFailPolicy",
    "PackingPolicy",
    "SideChannelEmbedding",
    "SideChannelFallback",
    "SideChannelMode",
    "SideChannelSpec",
    "default_side_channel_families",
]
