"""Backend tokenizer playground — F-G handlers.

Implements ``tokenizer.encode_visualize`` and ``tokenizer.list_presets``
JSON-RPC methods. Backed by the HuggingFace ``tokenizers`` Python
library so any ``tokenizer.json`` source works uniformly. Tiktoken /
SentencePiece formats are folded in later via the runtime registry on
the frontend (lazy-loaded WASM); the backend exposes one shape.

The encode_visualize result carries per-token byte spans + decoded
literals so the GUI can paint char-aligned color chips and emit
hover-cross-panel highlight events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.probe import TokenizerCapabilities, introspect_tokenizer


# Built-in preset library — paths are placeholders that map to real
# tokenizer.json files when shipped via the widget bundle. Resolving
# real paths is the GUI's job; the backend just declares the names.
PRESET_LIBRARY: tuple[str, ...] = (
    "cppmega_v3", "nanochat_v3",
    "gpt-4-o200k", "gpt-3.5-cl100k", "gpt-2-p50k",
    "llama-3", "mistral", "gemma", "qwen",
    "deepseek-v3", "phi-3", "claude",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TokenSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    text: str
    start: int
    end: int
    is_special: bool = False


class EncodeVisualizeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokenizer_source: str
    text: str
    add_special_tokens: bool = True


class EncodeVisualizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokens: list[TokenSpan]
    token_count: int
    bytes_total: int
    bytes_per_token_avg: float
    bytes_per_token_max: int
    capabilities: dict[str, Any]
    elapsed_ms: float


class ListPresetsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    presets: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizer cache — separate from the main RPC cache because keys differ.
# ---------------------------------------------------------------------------


@dataclass
class _LoadedTokenizer:
    tokenizer: Tokenizer
    capabilities: TokenizerCapabilities


_TOKENIZER_CACHE: dict[str, _LoadedTokenizer] = {}


def _load(source: str) -> _LoadedTokenizer:
    hit = _TOKENIZER_CACHE.get(source)
    if hit is not None:
        return hit
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer source not found: {source}")
    tok = Tokenizer.from_file(str(path))
    caps = introspect_tokenizer(path)
    loaded = _LoadedTokenizer(tokenizer=tok, capabilities=caps)
    _TOKENIZER_CACHE[source] = loaded
    return loaded


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def encode_visualize(
    params: EncodeVisualizeParams,
    *,
    cache: LRUCache | None = None,
) -> EncodeVisualizeResult:
    """Encode ``text`` through ``tokenizer_source`` and emit per-token spans."""
    cache_key = _cache_key(params)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    t0 = time.perf_counter()
    loaded = _load(params.tokenizer_source)
    encoded = loaded.tokenizer.encode(
        params.text, add_special_tokens=params.add_special_tokens,
    )

    text_bytes = params.text.encode("utf-8")
    special_ids = set(loaded.capabilities.special_ids.values())

    spans: list[TokenSpan] = []
    max_bytes = 0
    for tok_id, tok_text, (start, end) in zip(
        encoded.ids, encoded.tokens, encoded.offsets,
    ):
        span_bytes = max(0, end - start)
        max_bytes = max(max_bytes, span_bytes)
        spans.append(TokenSpan(
            id=int(tok_id),
            text=tok_text,
            start=int(start),
            end=int(end),
            is_special=int(tok_id) in special_ids,
        ))

    avg = (sum((s.end - s.start) for s in spans) / len(spans)) if spans else 0.0
    elapsed = (time.perf_counter() - t0) * 1000.0
    out = EncodeVisualizeResult(
        tokens=spans,
        token_count=len(spans),
        bytes_total=len(text_bytes),
        bytes_per_token_avg=round(avg, 3),
        bytes_per_token_max=max_bytes,
        capabilities=_caps_to_dict(loaded.capabilities),
        elapsed_ms=elapsed,
    )
    if cache is not None:
        cache.set(cache_key, out)
    return out


def list_presets() -> ListPresetsResult:
    return ListPresetsResult(presets=list(PRESET_LIBRARY))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(p: EncodeVisualizeParams) -> str:
    # No layout fields here, so canonical JSON via sorted-keys is fine.
    return f"tokenize::{p.tokenizer_source}::{p.add_special_tokens}::{hash(p.text)}"


def _caps_to_dict(c: TokenizerCapabilities) -> dict[str, Any]:
    return {
        "vocab_size": c.vocab_size,
        "special_ids": dict(c.special_ids),
        "has_fim": c.has_fim,
        "has_space_nl": c.has_space_nl,
        "has_code_start": c.has_code_start,
        "has_instruction": c.has_instruction,
        "byte_roundtrip": c.byte_roundtrip,
        "decoder_kind": c.decoder_kind,
        "source": c.source,
    }
