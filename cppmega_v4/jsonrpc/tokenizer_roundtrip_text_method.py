"""V7-H46: tokenizer.roundtrip_text RPC for TokenizerPlayground.

DataInspector's data.roundtrip_check operates over a parquet shard
(needs an `original_text` column). The TokenizerPlayground only has a
user-typed prompt, so we need a parquet-independent check: encode the
prompt, decode the resulting ids, return whether the byte sequence
round-trips intact + the decoded preview + tokenizer capability flag.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.tokenizer_methods import _load


class TokenizerRoundtripTextParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer_source: str
    text: str


class TokenizerRoundtripTextResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: bool
    decoded: str
    original_bytes: int
    decoded_bytes: int
    byte_diff: int
    tokenizer_capability: str
    elapsed_ms: float


def _capability_for(tokenizer_source: str) -> str:
    """Mirror DataInspector's heuristic: cppmega = exact, others approx."""
    if "cppmega" in tokenizer_source.lower():
        return "exact"
    return "approx"


def roundtrip_text(
    params: TokenizerRoundtripTextParams,
    *, cache: LRUCache | None = None,
) -> TokenizerRoundtripTextResult:
    t0 = time.perf_counter()
    tok = _load(params.tokenizer_source)
    enc = tok.encode(params.text)
    ids = list(getattr(enc, "ids", enc))
    try:
        decoded = tok.decode(ids)
    except Exception:
        decoded = ""
    orig_bytes = len(params.text.encode("utf-8"))
    dec_bytes = len(decoded.encode("utf-8"))
    matches = decoded == params.text
    byte_diff = abs(orig_bytes - dec_bytes)
    return TokenizerRoundtripTextResult(
        matches=matches,
        decoded=decoded,
        original_bytes=orig_bytes,
        decoded_bytes=dec_bytes,
        byte_diff=byte_diff,
        tokenizer_capability=_capability_for(params.tokenizer_source),
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
    )


__all__ = [
    "TokenizerRoundtripTextParams", "TokenizerRoundtripTextResult",
    "roundtrip_text",
]
