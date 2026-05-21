"""E-1 fixture generator tests.

Locks the on-disk shape: 4 tokenizer JSONs + 16 parquet shards + a
MATRIX.json index that ties everything together. The generator must
stay idempotent (re-running emits identical content) and every shard
must decode back into non-empty text under its matching tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from tests.fixtures.build_e2e_matrix import (
    PARQUET_DIR,
    PARQUET_SCHEMAS,
    TOKENIZERS_DIR,
    TOKENIZER_SPECS,
    INDEX_PATH,
    generate_parquets,
    generate_tokenizers,
    validate_round_trip,
)


REQUIRED_TOKENIZERS = {"T1_cppmega_v3", "T2_gpt2_small",
                       "T3_minimal_no_fim", "T4_fim_only"}
REQUIRED_SCHEMAS = set(PARQUET_SCHEMAS)


@pytest.fixture(scope="module", autouse=True)
def ensure_fixtures_present():
    """Generate once per module so each test runs against fresh artefacts."""
    generate_tokenizers()
    generate_parquets()


# ---------------------------------------------------------------------------
# Coverage gates
# ---------------------------------------------------------------------------


def test_tokenizer_specs_lock_four_entries():
    names = {s.name for s in TOKENIZER_SPECS}
    assert names == REQUIRED_TOKENIZERS


def test_tokenizer_specs_have_distinct_vocab_sizes():
    sizes = {s.vocab_target for s in TOKENIZER_SPECS}
    assert len(sizes) == len(TOKENIZER_SPECS)


def test_parquet_schemas_lock_four_variants():
    assert set(PARQUET_SCHEMAS) == REQUIRED_SCHEMAS
    assert len(PARQUET_SCHEMAS) == 4


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REQUIRED_TOKENIZERS))
def test_tokenizer_file_exists_and_loads(name):
    p = TOKENIZERS_DIR / f"{name}.json"
    assert p.is_file(), f"missing {p}"
    tok = Tokenizer.from_file(str(p))
    assert tok.get_vocab_size() > 0


def test_t1_is_full_cppmega_contract():
    tok = Tokenizer.from_file(str(TOKENIZERS_DIR / "T1_cppmega_v3.json"))
    assert tok.get_vocab_size() == 65536
    assert tok.token_to_id("<BOS>") is not None
    assert tok.token_to_id("<FIM_PREFIX>") is not None


def test_t2_is_gpt2_50257():
    tok = Tokenizer.from_file(str(TOKENIZERS_DIR / "T2_gpt2_small.json"))
    assert tok.get_vocab_size() == 50257


def test_t3_has_no_fim_specials():
    tok = Tokenizer.from_file(str(TOKENIZERS_DIR / "T3_minimal_no_fim.json"))
    assert tok.token_to_id("<FIM_PREFIX>") is None
    assert tok.token_to_id("<BOS>") is not None
    assert tok.get_vocab_size() <= 260  # 256 target + a couple slack


def test_t4_has_fim_trio():
    tok = Tokenizer.from_file(str(TOKENIZERS_DIR / "T4_fim_only.json"))
    assert tok.token_to_id("<FIM_PREFIX>") is not None
    assert tok.token_to_id("<FIM_MIDDLE>") is not None
    assert tok.token_to_id("<FIM_SUFFIX>") is not None


def test_generate_tokenizers_is_idempotent():
    """Second call should report nothing fresh."""
    first = generate_tokenizers()
    second = generate_tokenizers()
    assert all(v["fresh"] is False for v in second.values()), \
        f"non-fresh expected on rerun: {second}"
    assert {k: v["digest"] for k, v in first.items()} == \
           {k: v["digest"] for k, v in second.items()}


# ---------------------------------------------------------------------------
# Parquet shards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokenizer", sorted(REQUIRED_TOKENIZERS))
@pytest.mark.parametrize("schema", sorted(REQUIRED_SCHEMAS))
def test_parquet_shard_exists_and_has_token_column(tokenizer, schema):
    p = PARQUET_DIR / f"{tokenizer}__{schema}.parquet"
    assert p.is_file(), f"missing {p}"
    pf = pq.ParquetFile(p)
    cols = [f.name for f in pf.schema_arrow]
    assert "input_ids" in cols
    assert pf.metadata.num_rows == 32


@pytest.mark.parametrize("schema,extra_col", [
    ("P1_minimal", None),
    ("P2_doc", "doc_ids"),
    ("P3_engram", "call_edges"),
    ("P4_full", "loss_mask"),
])
def test_schema_carries_expected_extra(schema, extra_col):
    p = PARQUET_DIR / f"T1_cppmega_v3__{schema}.parquet"
    cols = {f.name for f in pq.ParquetFile(p).schema_arrow}
    if extra_col:
        assert extra_col in cols, f"{schema} missing {extra_col}: {cols}"


def test_p4_full_carries_all_enriched_columns():
    p = PARQUET_DIR / "T1_cppmega_v3__P4_full.parquet"
    cols = {f.name for f in pq.ParquetFile(p).schema_arrow}
    expected = {"input_ids", "doc_ids", "loss_mask", "chunk_boundaries",
                "call_edges", "type_edges",
                "constituent_provenance_offsets"}
    assert expected <= cols, f"missing: {expected - cols}"


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_validate_round_trip_emits_non_empty_decoded_text():
    rtt = validate_round_trip()
    assert len(rtt) == len(REQUIRED_TOKENIZERS) * len(REQUIRED_SCHEMAS)
    for key, payload in rtt.items():
        assert payload["non_empty"] == "True", \
            f"empty decode for {key}: {payload}"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_matrix_index_is_machine_readable():
    assert INDEX_PATH.is_file(), "MATRIX.json missing"
    payload = json.loads(INDEX_PATH.read_text())
    for key in ("tokenizers", "parquets", "round_trip"):
        assert key in payload
    assert len(payload["tokenizers"]) == 4
    assert len(payload["parquets"]) == 16
    assert len(payload["round_trip"]) == 16


def test_matrix_index_paths_resolve():
    payload = json.loads(INDEX_PATH.read_text())
    for entry in payload["tokenizers"].values():
        assert Path(entry["path"]).is_file()
    for entry in payload["parquets"].values():
        assert Path(entry["path"]).is_file()
