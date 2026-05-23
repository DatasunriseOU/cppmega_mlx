"""Backend inference side-channel tensor preview."""

from __future__ import annotations

import time
from typing import Any, Mapping

import mlx.core as mx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cppmega_mlx.inference.side_channels import InferenceSideChannelBuilder
from cppmega_v4.jsonrpc.cache import LRUCache, canonical_sha256
from cppmega_v4.jsonrpc.schema import SideChannelSpecPayload
from cppmega_v4.jsonrpc.tokenizer_methods import _load


class TensorPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: list[int]
    dtype: str
    sample: list[int | float | bool] = Field(default_factory=list)


class SideChannelPreviewParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_source: str
    text: str
    side_channels: SideChannelSpecPayload = Field(
        default_factory=SideChannelSpecPayload
    )
    platform_context: dict[str, Any] | str | None = None
    language: str | None = None
    adapter: str | None = None
    max_values: int = Field(default=16, ge=0, le=256)


class SideChannelPreviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_count: int
    prompt_ids: TensorPreview
    model_kwargs: dict[str, TensorPreview] = Field(default_factory=dict)
    side_channels: dict[str, dict[str, TensorPreview]] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    rendered_platform_context: str
    cache_key: str
    elapsed_ms: float


_FAMILY_MODEL_KWARGS: Mapping[str, tuple[str, ...]] = {
    "platform": ("platform_ids",),
    "structure": ("structure_ids", "dep_levels"),
    "syntax": ("ast_depth_ids", "sibling_index_ids", "node_type_ids"),
}


def preview_side_channels(
    params: SideChannelPreviewParams,
    *,
    cache: LRUCache | None = None,
) -> SideChannelPreviewResult:
    """Preview real inference side-channel tensors for one prompt."""

    cache_key = "side-channel-preview::" + canonical_sha256(
        params.model_dump(mode="json")
    )
    if params.side_channels.inference.cache_enabled and cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    t0 = time.perf_counter()
    source = params.side_channels.inference.source
    fail_policy = params.side_channels.inference.fail_policy
    effective_fail_policy = "text_only" if (
        params.side_channels.mode == "off" or source == "none"
    ) else fail_policy
    adapter_language = _adapter_language(params)
    if source == "prompt_only":
        adapter_language = None

    loaded = _load(params.tokenizer_source)
    builder = InferenceSideChannelBuilder(
        loaded.tokenizer,
        tokenizer_id=params.tokenizer_source,
        fail_policy=effective_fail_policy,
    )
    result = builder.build(
        params.text,
        platform_context=params.platform_context,
        language=adapter_language,
        fail_policy=effective_fail_policy,
    )

    enabled_families = _enabled_families(params.side_channels)
    model_kwargs = {
        name: _tensor_preview(value, params.max_values)
        for name, value in sorted(result.model_kwargs.items())
        if _model_kwarg_enabled(name, enabled_families)
    }
    side_channels = {
        family: {
            name: _tensor_preview(value, params.max_values)
            for name, value in sorted(columns.items())
        }
        for family, columns in sorted(result.side_channels.items())
        if family in enabled_families
    }
    side_channels = {
        family: columns
        for family, columns in side_channels.items()
        if columns
    }

    out = SideChannelPreviewResult(
        token_count=int(result.prompt_ids.shape[1]),
        prompt_ids=_tensor_preview(result.prompt_ids, params.max_values),
        model_kwargs=model_kwargs,
        side_channels=side_channels,
        provenance=dict(result.provenance),
        rendered_platform_context=result.rendered_platform_context,
        cache_key=result.cache_key,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    if params.side_channels.inference.cache_enabled and cache is not None:
        cache.set(cache_key, out)
    return out


def _adapter_language(params: SideChannelPreviewParams) -> str | None:
    adapter = (params.adapter or "").strip().lower()
    if adapter and adapter != "none":
        return adapter
    language = (params.language or "").strip().lower()
    return language or None


def _enabled_families(spec: SideChannelSpecPayload) -> frozenset[str]:
    if spec.mode == "off":
        return frozenset()
    return frozenset(
        name
        for name, family in spec.families.items()
        if family.mode != "off"
    )


def _model_kwarg_enabled(name: str, enabled_families: frozenset[str]) -> bool:
    return any(
        name in fields and family in enabled_families
        for family, fields in _FAMILY_MODEL_KWARGS.items()
    )


def _tensor_preview(value: mx.array, max_values: int) -> TensorPreview:
    mx.eval(value)
    arr = np.array(value)
    sample_values = arr.reshape(-1)[:max_values].tolist()
    return TensorPreview(
        shape=[int(dim) for dim in value.shape],
        dtype=str(value.dtype).removeprefix("mlx.core."),
        sample=[
            item.item() if hasattr(item, "item") else item
            for item in sample_values
        ],
    )


__all__ = [
    "SideChannelPreviewParams",
    "SideChannelPreviewResult",
    "TensorPreview",
    "preview_side_channels",
]
