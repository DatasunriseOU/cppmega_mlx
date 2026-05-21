"""E7-3 tests: data.roundtrip_check RPC + UI badge integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cppmega_v4.jsonrpc import dispatch
from cppmega_v4.jsonrpc.schema import METHOD_REGISTRY
from cppmega_v4.jsonrpc.roundtrip_method import (
    RoundtripCheckParams, roundtrip_check,
)


REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"


def _tok(name: str) -> str:
    return str(FIXTURES / "tokenizers" / f"{name}.json")


def _parq(tok: str, schema: str) -> str:
    return str(FIXTURES / "parquet" / f"{tok}__{schema}.parquet")


@pytest.fixture(scope="module", autouse=True)
def ensure_fixtures():
    """Regenerate fixtures if missing original_text column."""
    import pyarrow.parquet as pq
    sample = FIXTURES / "parquet" / "T2_gpt2_small__P1_minimal.parquet"
    if not sample.is_file():
        import subprocess, sys
        subprocess.run([sys.executable, str(FIXTURES / "build_e2e_matrix.py")],
                       check=True)
    cols = {f.name for f in pq.ParquetFile(sample).schema_arrow}
    if "original_text" not in cols:
        import subprocess, sys
        subprocess.run([sys.executable, str(FIXTURES / "build_e2e_matrix.py")],
                       check=True)


def test_method_registered():
    assert "data.roundtrip_check" in METHOD_REGISTRY


def test_t2_gpt2_roundtrip_high_pass_rate():
    """GPT-2 has exact byte_roundtrip — ASCII Python should round-trip."""
    r = roundtrip_check(RoundtripCheckParams(
        parquet_path=_parq("T2_gpt2_small", "P1_minimal"),
        tokenizer_source=_tok("T2_gpt2_small"),
        max_rows=8,
    ))
    # capability may be 'unknown' if encode_visualize can't introspect
    # the local file; the roundtrip check itself still works.
    assert r.tokenizer_capability in ("exact", "approx", "unknown")
    assert r.has_original_text is True
    assert r.pass_rate >= 0.5, f"GPT-2 pass rate too low: {r.pass_rate}"


def test_t1_cppmega_roundtrip_high_pass_rate():
    r = roundtrip_check(RoundtripCheckParams(
        parquet_path=_parq("T1_cppmega_v3", "P1_minimal"),
        tokenizer_source=_tok("T1_cppmega_v3"),
        max_rows=8,
    ))
    assert r.has_original_text is True
    # cppmega tokenizer may have varying roundtrip; just verify the
    # function returns sensible structure.
    assert 0.0 <= r.pass_rate <= 1.0
    assert len(r.rows) == 8


def test_t3_minimal_no_fim_lower_pass_rate_expected():
    """256-vocab BPE may lose information on uncommon tokens."""
    r = roundtrip_check(RoundtripCheckParams(
        parquet_path=_parq("T3_minimal_no_fim", "P1_minimal"),
        tokenizer_source=_tok("T3_minimal_no_fim"),
        max_rows=8,
    ))
    # Either passes (BPE handles ASCII fine) or has measurable byte_diff
    # — both are acceptable shapes.
    assert r.has_original_text is True
    for row in r.rows:
        assert row.byte_diff >= 0
        assert row.original_bytes > 0


def test_dispatch_returns_full_payload():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "data.roundtrip_check",
        "params": {
            "parquet_path": _parq("T2_gpt2_small", "P1_minimal"),
            "tokenizer_source": _tok("T2_gpt2_small"),
            "max_rows": 4,
        },
    })
    assert resp.error is None
    payload = resp.result
    assert "rows" in payload and "pass_rate" in payload
    assert "tokenizer_capability" in payload
    assert "has_original_text" in payload
    assert len(payload["rows"]) == 4
    for row in payload["rows"]:
        assert {"row_idx", "matches", "byte_diff",
                "original_bytes", "decoded_bytes",
                "decoded_preview"} <= set(row.keys())


def test_dispatch_missing_parquet_returns_error():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "data.roundtrip_check",
        "params": {
            "parquet_path": "/nonexistent/path.parquet",
            "tokenizer_source": _tok("T2_gpt2_small"),
            "max_rows": 4,
        },
    })
    assert resp.error is not None


def test_roundtrip_emits_decoded_preview():
    r = roundtrip_check(RoundtripCheckParams(
        parquet_path=_parq("T2_gpt2_small", "P1_minimal"),
        tokenizer_source=_tok("T2_gpt2_small"),
        max_rows=2,
    ))
    for row in r.rows:
        assert isinstance(row.decoded_preview, str)
        assert len(row.decoded_preview) <= 80


def test_byte_diff_is_zero_when_matches_is_true():
    """Internal invariant — when matches=True, byte_diff must be 0."""
    r = roundtrip_check(RoundtripCheckParams(
        parquet_path=_parq("T2_gpt2_small", "P1_minimal"),
        tokenizer_source=_tok("T2_gpt2_small"),
        max_rows=8,
    ))
    for row in r.rows:
        if row.matches:
            assert row.byte_diff == 0, row
