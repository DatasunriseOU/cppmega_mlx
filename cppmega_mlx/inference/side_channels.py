"""Inference-time side-channel enrichment builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
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
    available: bool = True
    reason: str | None = None


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


class AdapterUnavailableError(RuntimeError):
    """Raised when an adapter exists but cannot run in this environment."""


@dataclass(frozen=True)
class _SourceMetadata:
    language: str
    source: str
    ast_depth: tuple[int, ...]
    sibling_index: tuple[int, ...]
    node_type: tuple[int, ...]
    provenance: Mapping[str, str]


class UnavailableCodeMetadataAdapter:
    """Explicit placeholder for languages whose extractor is not wired yet."""

    version = "unavailable-v1"

    def __init__(
        self,
        language: str,
        *,
        families: Sequence[str] = (),
        reason: str = "adapter_not_implemented",
    ) -> None:
        self.language = normalize_code_language(language)
        self.families = tuple(families)
        self.reason = reason

    def probe(self, context: Mapping[str, Any]) -> AdapterCapabilities:
        del context
        return AdapterCapabilities(
            language=self.language,
            version=self.version,
            families=self.families,
            available=False,
            reason=self.reason,
        )

    def extract(self, source_or_project: str, options: Mapping[str, Any]) -> Any:
        del source_or_project, options
        raise AdapterUnavailableError(self.reason)

    def map_to_tokens(
        self,
        metadata: Any,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata:
        del metadata, tokens, tokenizer
        raise AdapterUnavailableError(self.reason)


class ClangCodeMetadataAdapter:
    """Clang-backed C/C++ metadata adapter for inference enrichment.

    The adapter parses a source snippet or unsaved file through libclang and
    maps source-coordinate AST signals into the canonical token metadata shape.
    More expensive project-index semantics stay behind the same protocol.
    """

    language = "cpp"
    version = "clang-ast-v1"
    families = ("syntax", "structure")

    def probe(self, context: Mapping[str, Any]) -> AdapterCapabilities:
        try:
            _clang_runtime(context.get("libclang_path"))
        except Exception as exc:
            return AdapterCapabilities(
                language=self.language,
                version=self.version,
                families=self.families,
                available=False,
                reason=str(exc),
            )
        return AdapterCapabilities(
            language=self.language,
            version=self.version,
            families=self.families,
            available=True,
        )

    def extract(
        self,
        source_or_project: str,
        options: Mapping[str, Any],
    ) -> _SourceMetadata:
        if not isinstance(source_or_project, str) or not source_or_project:
            raise AdapterUnavailableError("source must be a non-empty string")
        clang = _clang_runtime(options.get("libclang_path"))
        from tools.clang_indexer.index_project import (
            _sanitize_compile_args_for_clang,
            extract_clang_ast_metadata,
        )

        language = normalize_code_language(str(options.get("language") or self.language))
        repo_root = options.get("repo_root")
        filepath = str(options.get("filepath") or _default_source_path(language))
        compile_args = options.get("compile_args")
        args = (
            _sanitize_compile_args_for_clang(list(compile_args))
            if isinstance(compile_args, Sequence) and not isinstance(compile_args, str)
            else _default_clang_args(language)
        )
        with tempfile.TemporaryDirectory(prefix="cppmega_infer_clang_") as tmpdir:
            if repo_root:
                source_path = filepath if os.path.isabs(filepath) else os.path.join(
                    str(repo_root), filepath
                )
                source_path = os.path.normpath(source_path)
                unsaved_files = [(source_path, source_or_project)]
            else:
                source_path = os.path.join(tmpdir, Path(filepath).name)
                with open(source_path, "w", encoding="utf-8", errors="replace") as handle:
                    handle.write(source_or_project)
                unsaved_files = None

            index = clang.Index.create()
            try:
                tu = index.parse(
                    source_path,
                    args=args,
                    unsaved_files=unsaved_files,
                    options=(
                        clang.TranslationUnit.PARSE_INCOMPLETE
                        | clang.TranslationUnit.PARSE_PRECOMPILED_PREAMBLE
                    ),
                )
            except Exception as exc:
                raise AdapterUnavailableError(f"clang_parse_failed:{exc}") from exc

            ast_depth, sibling_index, node_type = extract_clang_ast_metadata(
                source_or_project,
                tu,
                source_path,
            )
        return _SourceMetadata(
            language=language,
            source=source_or_project,
            ast_depth=tuple(ast_depth),
            sibling_index=tuple(sibling_index),
            node_type=tuple(node_type),
            provenance={
                "adapter": f"{self.language}:{self.version}",
                "syntax": "clang_ast",
                "structure": "clang_ast",
            },
        )

    def map_to_tokens(
        self,
        metadata: Any,
        tokens: Sequence[int],
        tokenizer: Any,
    ) -> TokenMetadata:
        if not isinstance(metadata, _SourceMetadata):
            raise TypeError("metadata must be _SourceMetadata")
        source_indices = _token_source_indices(metadata.source, tokens, tokenizer)
        ast_depth = _source_values_at_indices(metadata.ast_depth, source_indices)
        sibling_index = _source_values_at_indices(metadata.sibling_index, source_indices)
        node_type = _source_values_at_indices(metadata.node_type, source_indices)
        structure_ids = [value if value > 0 else 0 for value in node_type]
        return TokenMetadata(
            structure_ids=structure_ids,
            dep_levels=ast_depth,
            ast_depth_ids=ast_depth,
            sibling_index_ids=sibling_index,
            node_type_ids=node_type,
            provenance=metadata.provenance,
        )


_LANGUAGE_ALIASES = {
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "py": "python",
    "rs": "rust",
    "golang": "go",
}

_UNAVAILABLE_ADAPTER_FAMILIES = {
    "rust": ("syntax", "structure", "semantic_graph"),
    "go": ("syntax", "structure", "semantic_graph"),
    "python": ("syntax", "structure"),
}


def normalize_code_language(language: str) -> str:
    normalized = str(language).strip().lower().replace("_", "-")
    return _LANGUAGE_ALIASES.get(normalized, normalized)


def get_builtin_code_metadata_adapter(language: str) -> CodeMetadataAdapter:
    normalized = normalize_code_language(language)
    if normalized in {"cpp", "c", "cuda", "hip"}:
        return ClangCodeMetadataAdapter()
    if normalized in _UNAVAILABLE_ADAPTER_FAMILIES:
        return UnavailableCodeMetadataAdapter(
            normalized,
            families=_UNAVAILABLE_ADAPTER_FAMILIES[normalized],
        )
    return UnavailableCodeMetadataAdapter(
        normalized,
        reason="adapter_unknown_language",
    )


def builtin_code_metadata_adapters() -> Mapping[str, CodeMetadataAdapter]:
    return {
        "cpp": ClangCodeMetadataAdapter(),
        "rust": UnavailableCodeMetadataAdapter(
            "rust",
            families=_UNAVAILABLE_ADAPTER_FAMILIES["rust"],
        ),
        "go": UnavailableCodeMetadataAdapter(
            "go",
            families=_UNAVAILABLE_ADAPTER_FAMILIES["go"],
        ),
        "python": UnavailableCodeMetadataAdapter(
            "python",
            families=_UNAVAILABLE_ADAPTER_FAMILIES["python"],
        ),
    }


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
        if active_adapter is None and language is not None:
            active_adapter = get_builtin_code_metadata_adapter(language)
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
            return InferenceSideChannelResult(
                prompt_ids=prompt_ids,
                model_kwargs=model_kwargs,
                side_channels=side_channels,
                provenance=provenance,
                platform_context=ctx,
                rendered_platform_context=rendered_context,
                cache_components=cache_components,
            )

        adapter_context = {
            "language": language or active_adapter.language,
            "platform_context": ctx,
            "tokenizer_id": self.tokenizer_id,
        }
        try:
            capabilities = active_adapter.probe(adapter_context)
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
        if not capabilities.available:
            _handle_adapter_failure(
                policy,
                f"adapter_unavailable:{capabilities.reason or 'unknown'}",
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
                adapter_context,
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


def _clang_runtime(libclang_path: Any) -> Any:
    from tools.clang_indexer.index_project import _configure_libclang

    _configure_libclang(str(libclang_path) if libclang_path else None)
    import clang.cindex as clang_cindex  # type: ignore[import-not-found]

    return clang_cindex


def _default_source_path(language: str) -> str:
    if language == "c":
        return "snippet.c"
    if language == "cuda":
        return "snippet.cu"
    if language == "hip":
        return "snippet.hip"
    return "snippet.cpp"


def _default_clang_args(language: str) -> list[str]:
    if language == "c":
        return ["-x", "c", "-std=c11", "-fsyntax-only", "-Wno-everything"]
    return ["-x", "c++", "-std=c++20", "-fsyntax-only", "-Wno-everything"]


def _source_values_at_indices(
    values: Sequence[int],
    indices: Sequence[int],
) -> list[int]:
    if not indices:
        return []
    if not values:
        return [0] * len(indices)
    value_count = len(values)
    return [int(values[min(value_count - 1, max(0, index))]) for index in indices]


def _token_source_indices(
    source: str,
    tokens: Sequence[int],
    tokenizer: Any,
) -> list[int]:
    token_count = len(tokens)
    if token_count <= 0:
        return []
    offsets = _token_offsets(source, tokens, tokenizer)
    if offsets is not None:
        source_len = len(source)
        return [
            min(
                source_len - 1,
                max(0, start if end <= start else (start + end - 1) // 2),
            )
            if source_len > 0 else 0
            for start, end in offsets
        ]
    source_len = len(source)
    if source_len <= 0:
        return [0] * token_count
    return [
        min(source_len - 1, (index * source_len) // token_count)
        for index in range(token_count)
    ]


def _token_offsets(
    source: str,
    tokens: Sequence[int],
    tokenizer: Any,
) -> list[tuple[int, int]] | None:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        encoded = encode(source)
    except Exception:
        return None
    offsets = getattr(encoded, "offsets", None)
    ids = getattr(encoded, "ids", None)
    if offsets is None or ids is None:
        return None
    if list(ids) != list(tokens) or len(offsets) != len(tokens):
        return None
    out: list[tuple[int, int]] = []
    for item in offsets:
        if not isinstance(item, Sequence) or len(item) != 2:
            return None
        start, end = int(item[0]), int(item[1])
        out.append((start, end))
    return out


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
    "AdapterUnavailableError",
    "AdapterCapabilities",
    "ClangCodeMetadataAdapter",
    "CodeMetadataAdapter",
    "InferenceFailPolicy",
    "InferenceSideChannelBuilder",
    "InferenceSideChannelCacheComponents",
    "InferenceSideChannelResult",
    "TokenMetadata",
    "UnavailableCodeMetadataAdapter",
    "builtin_code_metadata_adapters",
    "get_builtin_code_metadata_adapter",
    "normalize_code_language",
]
