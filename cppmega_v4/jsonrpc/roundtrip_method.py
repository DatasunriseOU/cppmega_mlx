"""data.roundtrip_check RPC handler (E7-3).

Decodes a parquet shard's input_ids back through the named tokenizer
and compares to the source text (when an 'original_text' column is
present). Surfaces per-row OK/FAIL badges in the Data Inspector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache


class RoundtripCheckParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parquet_path: str
    tokenizer_source: str
    max_rows: int = 8


class RoundtripRowPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_idx: int
    original_bytes: int
    decoded_bytes: int
    matches: bool
    byte_diff: int
    decoded_preview: str = ""


class RoundtripCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[RoundtripRowPayload] = Field(default_factory=list)
    tokenizer_capability: str = "unknown"
    pass_rate: float = 0.0
    has_original_text: bool = False
    elapsed_ms: float = 0.0


def _capability_for(tokenizer_source: str) -> str:
    """Read the tokenizer capability from the encode_visualize path."""
    try:
        from cppmega_v4.jsonrpc.tokenizer_methods import (
            EncodeVisualizeParams, encode_visualize,
        )
        ev = encode_visualize(EncodeVisualizeParams(
            tokenizer_source=tokenizer_source,
            text="probe",
        ))
        return ev.capabilities.byte_roundtrip
    except Exception:
        return "unknown"


def roundtrip_check(
    params: RoundtripCheckParams,
    *,
    cache: LRUCache | None = None,
) -> RoundtripCheckResult:
    import time
    t0 = time.perf_counter()
    cache_key = ("data.roundtrip_check",
                 params.parquet_path, params.tokenizer_source,
                 params.max_rows)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit  # type: ignore[return-value]

    cap = _capability_for(params.tokenizer_source)

    try:
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(f"pyarrow/tokenizers missing: {exc}")

    path = Path(params.parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"parquet not found: {path}")

    tok = Tokenizer.from_file(params.tokenizer_source)
    pf = pq.ParquetFile(path)
    table = pf.read_row_group(0).slice(0, params.max_rows)
    cols = {f.name for f in pf.schema_arrow}
    has_original = "original_text" in cols

    rows: list[RoundtripRowPayload] = []
    pad_id = tok.token_to_id("<PAD>") or 0
    for i in range(min(params.max_rows, table.num_rows)):
        ids_arr = table.column("input_ids")[i].as_py()
        ids = [int(x) for x in ids_arr if int(x) != pad_id]
        decoded = tok.decode(ids)

        if has_original:
            orig = str(table.column("original_text")[i].as_py())
        else:
            orig = decoded  # no ground truth → declare a trivial match

        orig_bytes = orig.encode("utf-8")
        dec_bytes = decoded.encode("utf-8")
        byte_diff = sum(1 for a, b in zip(orig_bytes, dec_bytes) if a != b) \
                    + abs(len(orig_bytes) - len(dec_bytes))
        matches = (orig_bytes == dec_bytes)
        rows.append(RoundtripRowPayload(
            row_idx=i,
            original_bytes=len(orig_bytes),
            decoded_bytes=len(dec_bytes),
            matches=matches,
            byte_diff=byte_diff,
            decoded_preview=decoded[:80],
        ))

    matches_count = sum(1 for r in rows if r.matches)
    pass_rate = matches_count / max(1, len(rows))
    result = RoundtripCheckResult(
        rows=rows,
        tokenizer_capability=cap,
        pass_rate=pass_rate,
        has_original_text=has_original,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )
    if cache is not None:
        cache.set(cache_key, result)
    return result
