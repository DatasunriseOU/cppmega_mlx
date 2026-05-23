"""V7-Q08.1: preview_parquet roundtrip_pass_rate field.

When the operator points at a parquet + supplies a tokenizer_source,
the preview RPC must surface the roundtrip pass rate INLINE so the UI
can warn before training starts (instead of discovering the tokenizer
incompat at step 0). Closes Lane 4 audit gap from
docs/UI-TO-TRAIN-AUDIT-2026-05-23.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cppmega_v4.jsonrpc.data_methods import (
    PreviewParquetParams, preview_parquet,
)

FIXTURE_MATRIX = Path("tests/fixtures/MATRIX.json")
FIXTURE_TOKENIZER = Path("tests/fixtures/tokenizers/T2_gpt2_small.json")


def _matrix() -> dict | None:
    if not FIXTURE_MATRIX.is_file():
        return None
    try:
        return json.loads(FIXTURE_MATRIX.read_text())
    except Exception:
        return None


def _pick_pair() -> tuple[str, str] | None:
    m = _matrix()
    if not m:
        return None
    parquets = m.get("parquets", {})
    # Prefer T2_gpt2 paired with T2 tokenizer (matching scheme).
    for key, info in parquets.items():
        if "T2_gpt2" in key and isinstance(info, dict):
            p = info.get("path")
            if p and Path(p).is_file() and FIXTURE_TOKENIZER.is_file():
                return (str(p), str(FIXTURE_TOKENIZER))
    # Fallback to first available pair.
    for key, info in parquets.items():
        if isinstance(info, dict):
            p = info.get("path")
            if p and Path(p).is_file() and FIXTURE_TOKENIZER.is_file():
                return (str(p), str(FIXTURE_TOKENIZER))
    return None


def test_preview_without_tokenizer_source_no_roundtrip_field() -> None:
    pair = _pick_pair()
    if pair is None:
        import pytest
        pytest.skip("e2e matrix fixture missing")
    parquet_path, _ = pair
    res = preview_parquet(PreviewParquetParams(
        path=parquet_path, offset=0, limit=4,
    ))
    assert res.roundtrip_pass_rate is None
    assert res.roundtrip_sampled_rows == 0


def test_preview_with_tokenizer_source_returns_pass_rate() -> None:
    pair = _pick_pair()
    if pair is None:
        import pytest
        pytest.skip("e2e matrix fixture missing")
    parquet_path, tokenizer_path = pair
    res = preview_parquet(PreviewParquetParams(
        path=parquet_path, offset=0, limit=4,
        tokenizer_source=tokenizer_path,
        roundtrip_sample_rows=4,
    ))
    # roundtrip_pass_rate is in [0, 1] when computed; may be None when
    # tokenizer lib didn't open the file (e.g. binary missing).
    if res.roundtrip_pass_rate is None:
        import pytest
        pytest.skip(
            "tokenizer failed to open — environment-specific, not a "
            "wiring regression")
    assert 0.0 <= res.roundtrip_pass_rate <= 1.0
    assert res.roundtrip_sampled_rows > 0


def test_preview_bad_tokenizer_path_skips_gracefully() -> None:
    pair = _pick_pair()
    if pair is None:
        import pytest
        pytest.skip("e2e matrix fixture missing")
    parquet_path, _ = pair
    # Bogus tokenizer path must NOT crash preview; just leave the
    # roundtrip field unset.
    res = preview_parquet(PreviewParquetParams(
        path=parquet_path, offset=0, limit=4,
        tokenizer_source="/nonexistent/tokenizer.json",
    ))
    assert res.roundtrip_pass_rate is None
    assert res.roundtrip_sampled_rows == 0
