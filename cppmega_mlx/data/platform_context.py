"""Canonical platform context parsing, rendering, and nanochat-compatible IDs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

import numpy as np

from cppmega_mlx.data.nanochat_pipeline.platform_vocab import (
    MAX_PLATFORM_IDS,
    PLATFORM_VOCAB,
    PLATFORM_VOCAB_SIZE,
)

_ENCODED_LIST_FIELDS = ("os", "rtos", "gpu", "arch", "compiler")
_TEXT_ONLY_LIST_FIELDS = ("stdlib", "backend")
_LIST_FIELDS = (*_ENCODED_LIST_FIELDS, *_TEXT_ONLY_LIST_FIELDS)
_CONTEXT_ORDER = (
    "language",
    "os",
    "rtos",
    "gpu",
    "arch",
    "compiler",
    "cpp_std",
    "stdlib",
    "backend",
    "target_triple",
)
_ALIASES = {
    "language": {"c++": "cpp", "cc": "cpp", "cxx": "cpp"},
    "os": {"darwin": "macos", "mac": "macos", "osx": "macos", "gnu/linux": "linux"},
    "arch": {"aarch64": "arm64", "amd64": "x64", "x86_64": "x64"},
    "compiler": {
        "appleclang": "clang",
        "clang++": "clang",
        "g++": "gcc",
        "gcc++": "gcc",
        "cl": "msvc",
        "cl.exe": "msvc",
    },
    "stdlib": {"libstdcxx": "libstdc++", "libcpp": "libc++", "msvc": "msvc-stl"},
    "gpu": {"accelerator": "cuda", "nvidia": "cuda", "apple-metal": "metal"},
    "backend": {"mlx-metal": "mlx", "pytorch": "torch"},
}
_KEY_ALIASES = {
    "std": "cpp_std",
    "standard": "cpp_std",
    "cpp": "cpp_std",
    "cpp_standard": "cpp_std",
    "accelerator": "gpu",
    "accelerators": "gpu",
    "target_arch": "arch",
    "target": "target_triple",
    "triple": "target_triple",
}


@dataclass(frozen=True)
class PlatformContext:
    """Canonical platform facts shared by prompt rendering and side-channel IDs.

    Only nanochat vocab fields are encoded into integer IDs. ``stdlib`` and
    ``backend`` are intentionally text-only until the upstream parquet producer
    adds them to ``PLATFORM_VOCAB``.
    """

    language: str | None = None
    os: tuple[str, ...] = ()
    rtos: tuple[str, ...] = ()
    gpu: tuple[str, ...] = ()
    arch: tuple[str, ...] = ()
    compiler: tuple[str, ...] = ()
    cpp_std: str | None = None
    stdlib: tuple[str, ...] = ()
    backend: tuple[str, ...] = ()
    target_triple: str | None = None


def parse_platform_context(value: Any) -> PlatformContext:
    """Normalize a mapping, JSON string, or rendered cppmega context block."""

    if value is None:
        return PlatformContext()
    if isinstance(value, PlatformContext):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return PlatformContext()
        if stripped.startswith("{"):
            decoded = json.loads(stripped)
            if not isinstance(decoded, Mapping):
                raise ValueError("platform context JSON must decode to an object")
            return _context_from_mapping(decoded)
        return _context_from_mapping(_parse_context_text(stripped))
    if isinstance(value, Mapping):
        return _context_from_mapping(value)
    raise TypeError(f"unsupported platform context type: {type(value).__name__}")


def render_platform_context(context: PlatformContext | Mapping[str, Any]) -> str:
    """Render a stable C++ comment block for text-side conditioning."""

    ctx = parse_platform_context(context)
    lines = ["/* cppmega-context:"]
    for key in _CONTEXT_ORDER:
        value = getattr(ctx, key)
        if value is None or value == ():
            continue
        rendered = ",".join(value) if isinstance(value, tuple) else str(value)
        lines.append(f"{key}={rendered}")
    lines.append("*/")
    return "\n".join(lines) + "\n"


def encode_platform_context(
    context: PlatformContext | Mapping[str, Any] | str | None,
) -> tuple[int, ...]:
    """Encode context with the exact nanochat ``platform_info_to_ids`` contract."""

    ctx = parse_platform_context(context)
    platform_info = {
        "os": ctx.os,
        "rtos": ctx.rtos,
        "gpu": ctx.gpu,
        "arch": ctx.arch,
        "compiler": ctx.compiler,
        "cpp_std": ctx.cpp_std,
    }
    ids: list[int] = []
    for category in _ENCODED_LIST_FIELDS:
        vocab = PLATFORM_VOCAB[category]
        labels = platform_info.get(category, ())
        for label in labels if isinstance(labels, tuple) else ():
            encoded = vocab.get(label)
            if encoded is not None:
                ids.append(encoded)
    if ctx.cpp_std:
        encoded = PLATFORM_VOCAB["cpp_std"].get(ctx.cpp_std)
        if encoded is not None:
            ids.append(encoded)
    return tuple(sorted(set(ids)))


def platform_ids_array(
    contexts: Sequence[PlatformContext | Mapping[str, Any] | str | None],
    *,
    width: int = MAX_PLATFORM_IDS,
) -> np.ndarray:
    """Return a padded int32 ``(B, width)`` platform ID array."""

    if width <= 0:
        raise ValueError("width must be positive")
    rows = []
    for context in contexts:
        ids = encode_platform_context(context)
        if len(ids) > width:
            raise ValueError(f"platform context encodes {len(ids)} IDs; width={width}")
        rows.append((*ids, *(0 for _ in range(width - len(ids)))))
    return np.asarray(rows, dtype=np.int32)


def _context_from_mapping(value: Mapping[str, Any]) -> PlatformContext:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_key(str(raw_key))
        if key in _LIST_FIELDS:
            normalized[key] = _normalize_many(key, raw_value)
        elif key == "language":
            normalized[key] = _normalize_scalar(key, raw_value)
        elif key == "cpp_std":
            normalized[key] = _normalize_cpp_standard(raw_value)
        elif key == "target_triple":
            normalized[key] = _empty_to_none(raw_value)
    return PlatformContext(**normalized)


def _parse_context_text(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line in {"/*", "*/"}:
            continue
        line = re.sub(r"^/\*\s*", "", line)
        line = re.sub(r"\s*\*/$", "", line)
        if line.startswith("cppmega-context:"):
            line = line[len("cppmega-context:") :].strip()
            if not line:
                continue
        if line.startswith("//"):
            line = line[2:].strip()
        if line.startswith("platform:"):
            line = line[len("platform:") :].strip()
            for item in line.split():
                _update_key_value(values, item)
            continue
        _update_key_value(values, line)
    return values


def _update_key_value(values: dict[str, Any], item: str) -> None:
    if "=" not in item:
        return
    key, raw_value = item.split("=", 1)
    normalized_key = _normalize_key(key)
    if normalized_key in _LIST_FIELDS:
        values[normalized_key] = [part for part in raw_value.split(",") if part]
    elif normalized_key == "cpp_std":
        values[normalized_key] = raw_value
    elif normalized_key in {"language", "target_triple"}:
        values[normalized_key] = raw_value


def _normalize_key(key: str) -> str:
    stripped = key.strip().lower().replace("-", "_")
    return _KEY_ALIASES.get(stripped, stripped)


def _normalize_many(field: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = re.split(r"[,;\s]+", value.strip())
    elif isinstance(value, Sequence):
        raw_values = [str(item) for item in value]
    else:
        raw_values = [str(value)]
    return tuple(
        dict.fromkeys(
            normalized
            for item in raw_values
            if (normalized := _normalize_scalar(field, item)) is not None
        )
    )


def _normalize_scalar(field: str, value: Any) -> str | None:
    normalized = _empty_to_none(value)
    if normalized is None:
        return None
    lowered = normalized.lower().replace(" ", "").replace("_", "-")
    return _ALIASES.get(field, {}).get(lowered, lowered)


def _normalize_cpp_standard(value: Any) -> str | None:
    normalized = _empty_to_none(value)
    if normalized is None:
        return None
    lowered = normalized.lower().replace(" ", "")
    if lowered.startswith("gnu++"):
        return "c++" + lowered[len("gnu++") :]
    if lowered.startswith("std="):
        lowered = lowered[len("std=") :]
    return lowered


def _empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


__all__ = [
    "MAX_PLATFORM_IDS",
    "PLATFORM_VOCAB",
    "PLATFORM_VOCAB_SIZE",
    "PlatformContext",
    "encode_platform_context",
    "parse_platform_context",
    "platform_ids_array",
    "render_platform_context",
]
