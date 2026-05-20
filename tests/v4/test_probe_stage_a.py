"""Contract Probe Stage A tests — capability introspection.

Covers ``introspect_tokenizer`` against the vendored nanochat
``tokenizer.json`` and ``introspect_parquet`` against an in-test fixture
that pyarrow writes to a temporary path. We avoid checking in a real
parquet shard — it would be a binary blob with weak provenance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.probe import (
    ColumnSpec,
    ParquetCapabilities,
    TokenizerCapabilities,
    introspect_parquet,
    introspect_tokenizer,
)


_VENDORED_TOKENIZER = Path("cppmega_mlx/tokenizer/tokenizer.json")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_introspect_vendored_tokenizer_returns_capabilities():
    caps = introspect_tokenizer(_VENDORED_TOKENIZER)
    assert isinstance(caps, TokenizerCapabilities)
    assert caps.vocab_size > 0
    assert caps.source == str(_VENDORED_TOKENIZER)


def test_vendored_tokenizer_has_full_special_id_contract():
    """Per bd memory tokenizer-architecture: 0=PAD..47=NL contract."""
    caps = introspect_tokenizer(_VENDORED_TOKENIZER)
    for name in (
        "PAD", "UNK", "BOS", "EOS",
        "FIM_PREFIX", "FIM_MIDDLE", "FIM_SUFFIX",
        "CODE_START", "FIM_INSTRUCTION",
        "SPACE", "NL",
    ):
        assert name in caps.special_ids, f"missing special id {name!r}"
    assert caps.special_ids["PAD"] == 0
    assert caps.special_ids["BOS"] == 2
    assert caps.special_ids["FIM_PREFIX"] == 4
    assert caps.special_ids["SPACE"] == 46
    assert caps.special_ids["NL"] == 47


def test_vendored_tokenizer_has_capability_flags():
    caps = introspect_tokenizer(_VENDORED_TOKENIZER)
    assert caps.has_fim is True
    assert caps.has_space_nl is True
    assert caps.has_code_start is True
    assert caps.has_instruction is True


def test_vendored_tokenizer_decoder_is_custom_approx():
    """decoder=null in nanochat tokenizer → custom CppTokenizer.decode."""
    caps = introspect_tokenizer(_VENDORED_TOKENIZER)
    assert caps.decoder_kind == "custom"
    assert caps.byte_roundtrip == "approx"


def test_introspect_missing_tokenizer_raises():
    with pytest.raises(FileNotFoundError):
        introspect_tokenizer("/nonexistent/tokenizer.json")


def test_introspect_minimal_tokenizer_without_fim(tmp_path: Path):
    """Tokenizer missing the FIM trio must report has_fim=False."""
    tjson = {
        "model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}},
        "added_tokens": [
            {"id": 0, "content": "<PAD>"},
            {"id": 1, "content": "<UNK>"},
            {"id": 2, "content": "<BOS>"},
            {"id": 3, "content": "<EOS>"},
        ],
        "decoder": {"type": "ByteLevel"},
    }
    p = tmp_path / "tiny_tokenizer.json"
    p.write_text(json.dumps(tjson))
    caps = introspect_tokenizer(p)
    assert caps.vocab_size == 256
    assert caps.has_fim is False
    assert caps.has_space_nl is False
    assert caps.has_code_start is False
    assert caps.decoder_kind == "hf"
    assert caps.byte_roundtrip == "exact"


def test_capabilities_has_helper():
    caps = introspect_tokenizer(_VENDORED_TOKENIZER)
    assert caps.has("BOS", "EOS")
    assert not caps.has("MADE_UP_TOKEN")


def test_tokenizer_probe_under_200ms():
    """Performance gate: probe must stay GUI-responsive."""
    t0 = time.perf_counter()
    introspect_tokenizer(_VENDORED_TOKENIZER)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200, f"introspect_tokenizer took {elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def _write_fixture_parquet(
    path: Path,
    *,
    n_rows: int = 32,
    extra_cols: dict[str, list] | None = None,
) -> None:
    cols: dict[str, list] = {
        "input_ids": [list(range(8 + i % 4)) for i in range(n_rows)],
        "doc_ids":   [i // 4 for i in range(n_rows)],
    }
    if extra_cols:
        cols.update(extra_cols)
    table = pa.table(cols)
    pq.write_table(table, path)


def test_introspect_parquet_returns_capabilities(tmp_path: Path):
    p = tmp_path / "shard_00.parquet"
    _write_fixture_parquet(p)
    caps = introspect_parquet(p)
    assert isinstance(caps, ParquetCapabilities)
    assert caps.row_count == 32
    assert caps.total_bytes > 0
    assert caps.source == str(p)


def test_introspect_parquet_detects_token_stream(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(p)
    caps = introspect_parquet(p)
    assert caps.has_token_ids is True
    assert caps.has_doc_ids is True
    assert caps.has_chunk_spans is False
    assert caps.has_call_edges is False


def test_introspect_parquet_side_channels_exclude_canonical(tmp_path: Path):
    """input_ids/labels/etc are NOT side-channels; everything else is."""
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(
        p,
        extra_cols={
            "call_edges": [[(0, 1)] for _ in range(32)],
            "labels":     [list(range(8)) for _ in range(32)],
        },
    )
    caps = introspect_parquet(p)
    assert "input_ids" not in caps.side_channels
    assert "labels" not in caps.side_channels
    assert "doc_ids" in caps.side_channels
    assert "call_edges" in caps.side_channels
    assert caps.has_call_edges is True


def test_introspect_parquet_schema_columns_have_ratios(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(p)
    caps = introspect_parquet(p)
    names = {c.name for c in caps.schema_columns}
    assert names == {"input_ids", "doc_ids"}
    for c in caps.schema_columns:
        assert isinstance(c, ColumnSpec)
        assert 0.0 <= c.non_null_ratio <= 1.0


def test_introspect_parquet_sample_seq_lens_match_sample_size(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(p, n_rows=8)
    caps = introspect_parquet(p, sample_rows=4)
    assert len(caps.sample_seq_lens) == 4
    assert all(L > 0 for L in caps.sample_seq_lens)


def test_introspect_parquet_provenance_detection(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(
        p,
        extra_cols={
            "constituent_provenance_offsets": [[0] for _ in range(32)],
        },
    )
    caps = introspect_parquet(p)
    assert caps.has_provenance is True


def test_introspect_missing_parquet_raises():
    with pytest.raises(FileNotFoundError):
        introspect_parquet("/nonexistent/file.parquet")


def test_introspect_parquet_column_helper(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(p)
    caps = introspect_parquet(p)
    col = caps.column("input_ids")
    assert col is not None
    assert col.name == "input_ids"
    assert caps.column("nonexistent") is None


def test_parquet_probe_under_200ms(tmp_path: Path):
    p = tmp_path / "f.parquet"
    _write_fixture_parquet(p, n_rows=256)
    t0 = time.perf_counter()
    introspect_parquet(p, sample_rows=256)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200, f"introspect_parquet took {elapsed_ms:.1f} ms"
