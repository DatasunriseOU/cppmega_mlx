from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.domain_schema import (
    DOMAIN_SCHEMA_SHA256,
    DOMAIN_SCHEMA_SHA256_METADATA_KEY,
)
from cppmega_mlx.data.tokenizer_contract import (
    TOKENIZER_CONTRACT_SHA256,
    TOKENIZER_CONTRACT_SHA256_METADATA_KEY,
)
from scripts import materialize_megatron_objectives as materializer


def _contract_metadata() -> dict[bytes, bytes]:
    return {
        DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8"): DOMAIN_SCHEMA_SHA256.encode(
            "ascii"
        ),
        TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode(
            "utf-8"
        ): TOKENIZER_CONTRACT_SHA256.encode("ascii"),
    }


def test_materialized_schema_binds_full_frozen_contract_hashes() -> None:
    metadata = materializer.materialized_schema().metadata or {}

    assert metadata[DOMAIN_SCHEMA_SHA256_METADATA_KEY.encode("utf-8")] == (
        DOMAIN_SCHEMA_SHA256.encode("ascii")
    )
    assert metadata[TOKENIZER_CONTRACT_SHA256_METADATA_KEY.encode("utf-8")] == (
        TOKENIZER_CONTRACT_SHA256.encode("ascii")
    )


@pytest.mark.parametrize(
    ("metadata_key", "replacement"),
    [
        (DOMAIN_SCHEMA_SHA256_METADATA_KEY, None),
        (TOKENIZER_CONTRACT_SHA256_METADATA_KEY, b"0" * 64),
    ],
)
def test_objective_materializer_rejects_missing_or_stale_input_hashes(
    tmp_path: Path,
    metadata_key: str,
    replacement: bytes | None,
) -> None:
    metadata = _contract_metadata()
    encoded_key = metadata_key.encode("utf-8")
    if replacement is None:
        metadata.pop(encoded_key)
    else:
        metadata[encoded_key] = replacement
    schema = pa.schema(
        [pa.field("input_ids", pa.list_(pa.uint32()))],
        metadata=metadata,
    )
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([{"input_ids": [1]}], schema=schema), path)

    with pytest.raises(ValueError, match="missing or stale frozen CASE5"):
        next(materializer._iter_sources([str(path)], seed=7))


def test_objective_receipt_hash_binding_rejects_stale_values() -> None:
    receipt: dict[str, object] = {}
    materializer._bind_case5_contract_hashes(receipt)
    assert receipt == {
        "domain_schema_sha256": DOMAIN_SCHEMA_SHA256,
        "tokenizer_contract_sha256": TOKENIZER_CONTRACT_SHA256,
    }

    stale = {"domain_schema_sha256": "0" * 64}
    with pytest.raises(ValueError, match="stale domain_schema_sha256"):
        materializer._bind_case5_contract_hashes(stale)
