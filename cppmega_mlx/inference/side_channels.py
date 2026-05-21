"""Inference-time side-channel enrichment builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Literal, Protocol, runtime_checkable

import mlx.core as mx

from cppmega_mlx.data.platform_context import (
    PlatformContext,
    encode_platform_context,
    parse_platform_context,
    platform_ids_array,
    render_platform_context,
)

InferenceFailPolicy = Literal["drop_family", "text_only", "error"]


@dataclass(frozen=True)
class AdapterCapabilities:
    """Declarative metadata advertised by a language adapter."""

    language: str
    version: str
    families: tuple[str, ...] = ()


@dataclass(frozen=True)
class TokenMetadata:
    """Token-coordinate metadata produced by a language adapter."""

    structure_ids: Sequence[int] | None = None
    dep_levels: Sequence[int] | None = None
    ast_depth_ids: Sequence[int] | None = None
    sibling_index_ids: Sequence[int] | None = None
    node_type_ids: Sequence[int] | None = None
    side_channels: Mapping[str, Mapping[str, Sequence[int]]] = field(
        default_factory=dict
    )
    provenance: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class CodeMetadataAdapter(Protocol):
    """Language adapter seam for inference-time code metadata."""

    language: str
    version: str

    def probe(self, context: Mapping[str, Any]) -> AdapterCapabilities: ...

    def extract(self, source_or_project: str, options: Mapping[str, Any]) -> Any: ...

    def map_to_tokens(
        self,
        metadata: Any,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata: ...


@dataclass(frozen=True)
class InferenceSideChannelCacheComponents:
    """Stable cache-key parts for enriched inference prompts."""

    content_sha256: str
    tokenizer_id: str
    adapter_language: str | None
    adapter_version: str | None
    platform_context: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "content_sha256": self.content_sha256,
                "tokenizer_id": self.tokenizer_id,
                "adapter_language": self.adapter_language,
                "adapter_version": self.adapter_version,
                "platform_context": self.platform_context,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InferenceSideChannelResult:
    """Prompt IDs plus model kwargs and provenance produced by the builder."""

    prompt_ids: mx.array
    model_kwargs: Mapping[str, mx.array]
    side_channels: Mapping[str, Mapping[str, mx.array]]
    provenance: Mapping[str, str]
    platform_context: PlatformContext
    rendered_platform_context: str
    cache_components: InferenceSideChannelCacheComponents

    @property
    def cache_key(self) -> str:
        return self.cache_components.digest()


class InferenceSideChannelBuilder:
    """Build side-channel tensors for a single inference prompt."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        tokenizer_id: str | None = None,
        adapter: CodeMetadataAdapter | None = None,
        fail_policy: InferenceFailPolicy = "drop_family",
    ) -> None:
        if fail_policy not in {"drop_family", "text_only", "error"}:
            raise ValueError("fail_policy must be drop_family, text_only, or error")
        self.tokenizer = tokenizer
        self.tokenizer_id = tokenizer_id or _infer_tokenizer_id(tokenizer)
        self.adapter = adapter
        self.fail_policy = fail_policy

    def build(
        self,
        text: str,
        *,
        platform_context: PlatformContext | Mapping[str, Any] | str | None = None,
        adapter: CodeMetadataAdapter | None = None,
        language: str | None = None,
        fail_policy: InferenceFailPolicy | None = None,
    ) -> InferenceSideChannelResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        policy = fail_policy or self.fail_policy
        if policy not in {"drop_family", "text_only", "error"}:
            raise ValueError("fail_policy must be drop_family, text_only, or error")

        token_ids = _encode_text(self.tokenizer, text)
        prompt_ids = mx.array([token_ids], dtype=mx.int32)
        ctx = parse_platform_context(platform_context)
        rendered_context = render_platform_context(ctx)
        active_adapter = adapter or self.adapter
        cache_components = _cache_components(
            text=text,
            tokenizer_id=self.tokenizer_id,
            adapter=active_adapter,
            platform_context=rendered_context,
        )

        if policy == "text_only":
            return InferenceSideChannelResult(
                prompt_ids=prompt_ids,
                model_kwargs={},
                side_channels={},
                provenance={"fallback": "text_only"},
                platform_context=ctx,
                rendered_platform_context=rendered_context,
                cache_components=cache_components,
            )

        model_kwargs: dict[str, mx.array] = {}
        side_channels: dict[str, dict[str, mx.array]] = {}
        provenance: dict[str, str] = {}
        platform_ids = _platform_ids_for_context(ctx)
        if platform_ids is not None:
            model_kwargs["platform_ids"] = platform_ids
            side_channels["platform"] = {"platform_ids": platform_ids}
            provenance["platform"] = "platform_context"

        if active_adapter is None:
            if language is not None:
                _handle_adapter_failure(
                    policy,
                    "adapter_missing",
                    provenance=provenance,
                    model_kwargs=model_kwargs,
                    side_channels=side_channels,
                )
            return InferenceSideChannelResult(
                prompt_ids=prompt_ids,
                model_kwargs=model_kwargs,
                side_channels=side_channels,
                provenance=provenance,
                platform_context=ctx,
                rendered_platform_context=rendered_context,
                cache_components=cache_components,
            )

        try:
            metadata = active_adapter.extract(
                text,
                {
                    "language": language or active_adapter.language,
                    "platform_context": ctx,
                    "tokenizer_id": self.tokenizer_id,
                },
            )
            token_metadata = active_adapter.map_to_tokens(
                metadata,
                token_ids,
                self.tokenizer,
            )
            _merge_token_metadata(
                token_metadata,
                token_count=len(token_ids),
                model_kwargs=model_kwargs,
                side_channels=side_channels,
                provenance=provenance,
            )
            if token_metadata.provenance:
                provenance.update(dict(token_metadata.provenance))
            provenance.setdefault(
                "adapter",
                f"{active_adapter.language}:{active_adapter.version}",
            )
        except Exception as exc:
            _handle_adapter_failure(
                policy,
                f"adapter_error:{type(exc).__name__}",
                provenance=provenance,
                model_kwargs=model_kwargs,
                side_channels=side_channels,
            )

        return InferenceSideChannelResult(
            prompt_ids=prompt_ids,
            model_kwargs=model_kwargs,
            side_channels=side_channels,
            provenance=provenance,
            platform_context=ctx,
            rendered_platform_context=rendered_context,
            cache_components=cache_components,
        )


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    if not hasattr(tokenizer, "encode"):
        raise TypeError("tokenizer must expose encode(text)")
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        ids = list(encoded.ids)
    else:
        ids = list(encoded)
    if not ids:
        raise ValueError("encoded prompt must contain at least one token")
    if any(not isinstance(item, int) for item in ids):
        raise ValueError("tokenizer.encode must return integer token ids")
    return ids


def _infer_tokenizer_id(tokenizer: Any) -> str:
    for attr in ("name_or_path", "path"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    return f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"


def _cache_components(
    *,
    text: str,
    tokenizer_id: str,
    adapter: CodeMetadataAdapter | None,
    platform_context: str,
) -> InferenceSideChannelCacheComponents:
    return InferenceSideChannelCacheComponents(
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        tokenizer_id=tokenizer_id,
        adapter_language=getattr(adapter, "language", None),
        adapter_version=getattr(adapter, "version", None),
        platform_context=platform_context,
    )


def _platform_ids_for_context(ctx: PlatformContext) -> mx.array | None:
    if not encode_platform_context(ctx):
        return None
    return mx.array(platform_ids_array([ctx]), dtype=mx.int32)


def _merge_token_metadata(
    metadata: TokenMetadata,
    *,
    token_count: int,
    model_kwargs: dict[str, mx.array],
    side_channels: dict[str, dict[str, mx.array]],
    provenance: dict[str, str],
) -> None:
    field_families = {
        "structure_ids": "structure",
        "dep_levels": "structure",
        "ast_depth_ids": "syntax",
        "sibling_index_ids": "syntax",
        "node_type_ids": "syntax",
    }
    for name, family in field_families.items():
        value = getattr(metadata, name)
        if value is None:
            continue
        array = _token_aligned_array(name, value, token_count=token_count)
        model_kwargs[name] = array
        side_channels.setdefault(family, {})[name] = array
        provenance.setdefault(family, "adapter")

    for family, columns in metadata.side_channels.items():
        for name, value in columns.items():
            array = _token_aligned_array(name, value, token_count=token_count)
            side_channels.setdefault(family, {})[name] = array
            provenance.setdefault(family, "adapter")


def _token_aligned_array(
    name: str,
    value: Sequence[int],
    *,
    token_count: int,
) -> mx.array:
    items = [int(item) for item in value]
    if len(items) != token_count:
        raise ValueError(
            f"{name} must have one value per token: got {len(items)}, expected {token_count}"
        )
    return mx.array([items], dtype=mx.int32)


def _handle_adapter_failure(
    policy: InferenceFailPolicy,
    reason: str,
    *,
    provenance: dict[str, str],
    model_kwargs: dict[str, mx.array],
    side_channels: dict[str, dict[str, mx.array]],
) -> None:
    if policy == "error":
        raise RuntimeError(reason)
    if policy == "text_only":
        model_kwargs.clear()
        side_channels.clear()
        provenance.clear()
        provenance["fallback"] = "text_only"
        return
    provenance["adapter"] = f"dropped:{reason}"


__all__ = [
    "AdapterCapabilities",
    "CodeMetadataAdapter",
    "InferenceFailPolicy",
    "InferenceSideChannelBuilder",
    "InferenceSideChannelCacheComponents",
    "InferenceSideChannelResult",
    "TokenMetadata",
]
