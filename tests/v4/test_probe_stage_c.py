"""Contract Probe Stage C tests — JSON serialisation + schema parity.

Stable wire format for GUI/CLI consumers. The exported schema lives at
``docs/contract_probe_schema.json`` and MUST stay in sync with what
:func:`cppmega_v4.probe.json_schema` returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.buildspec import (
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
    ifim_shaped_loss,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.probe import (
    SCHEMA_VERSION,
    contract_probe,
    from_dict,
    json_schema,
    to_dict,
)


_VENDORED_TOKENIZER = Path("cppmega_mlx/tokenizer/tokenizer.json")
_EXPORTED_SCHEMA = Path("docs/contract_probe_schema.json")


def _write_full_parquet(p: Path, n_rows: int = 16, with_edges: bool = True):
    cols: dict[str, list] = {
        "input_ids": [list(range(8)) for _ in range(n_rows)],
        "doc_ids":   [i // 4 for i in range(n_rows)],
    }
    if with_edges:
        cols["call_edges"] = [[(0, 1)] for _ in range(n_rows)]
        cols["type_edges"] = [[(0, 2)] for _ in range(n_rows)]
    pq.write_table(pa.table(cols), p)


def _make_spec(preset: str = "llama3_8b", loss=None) -> ModelBuildSpec:
    specs = build_preset_specs(preset, hidden_size=64)
    graph = from_block_specs(specs, hidden_size=64, instantiate=False)
    head = graph.nodes[-1].name
    return ModelBuildSpec(
        graph=graph,
        loss=loss or cross_entropy_loss(head_output_name=head),
        optim=adamw(),
    )


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip.
# ---------------------------------------------------------------------------


def test_to_dict_emits_schema_version():
    spec = _make_spec()
    pqp = Path("/tmp/_probe_test.parquet")
    _write_full_parquet(pqp)
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    d = to_dict(report)
    assert d["schema_version"] == SCHEMA_VERSION


def test_to_dict_then_from_dict_round_trip(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec()
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    restored = from_dict(to_dict(report))
    assert restored == report


def test_to_dict_then_from_dict_round_trip_with_findings(tmp_path: Path):
    """IFIM on minimal tokenizer → unsatisfied findings + alternatives."""
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    tok = tmp_path / "tiny.json"
    tok.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"a": 0}},
        "added_tokens": [{"id": 0, "content": "<PAD>"}],
        "decoder": {"type": "ByteLevel"},
    }))
    spec = _make_spec(loss=ifim_shaped_loss())
    report = contract_probe(spec, tok, pqp, run_dry_forward=False)
    assert not report.is_clean
    restored = from_dict(to_dict(report))
    assert restored == report
    # Alternatives survived too:
    assert any(f.alternatives for f in restored.findings)


def test_from_dict_rejects_schema_version_mismatch():
    fake = {
        "schema_version": "0.0.1",
        "tokenizer": {}, "parquet": {}, "findings": [],
        "elapsed_ms": 0.0, "probe_hidden_size": 64,
        "dry_forward_verdict": "skipped",
    }
    with pytest.raises(ValueError, match="schema_version"):
        from_dict(fake)


# ---------------------------------------------------------------------------
# JSON-Schema validity.
# ---------------------------------------------------------------------------


def test_schema_is_valid_draft_2020_12():
    schema = json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)


def test_report_validates_against_schema(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec()
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    payload = to_dict(report)
    jsonschema.validate(payload, json_schema())


def test_report_with_findings_validates_against_schema(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    tok = tmp_path / "tiny.json"
    tok.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {"a": 0}},
        "added_tokens": [{"id": 0, "content": "<PAD>"}],
        "decoder": {"type": "ByteLevel"},
    }))
    spec = _make_spec(loss=ifim_shaped_loss())
    report = contract_probe(spec, tok, pqp, run_dry_forward=False)
    jsonschema.validate(to_dict(report), json_schema())


# ---------------------------------------------------------------------------
# Exported schema parity.
# ---------------------------------------------------------------------------


def test_exported_schema_matches_live_schema():
    """docs/contract_probe_schema.json MUST equal json_schema() output.

    Regenerate via:
        python -c 'import json; from cppmega_v4.probe import json_schema;
                   json.dump(json_schema(),
                             open("docs/contract_probe_schema.json","w"),
                             indent=2)'
    """
    assert _EXPORTED_SCHEMA.exists(), (
        "docs/contract_probe_schema.json missing — Stage C ships the "
        "exported schema alongside the runtime json_schema()"
    )
    on_disk = json.loads(_EXPORTED_SCHEMA.read_text())
    live = json_schema()
    assert on_disk == live, (
        "exported schema drifted from json_schema(); regenerate "
        "docs/contract_probe_schema.json"
    )


def test_exported_schema_carries_schema_version_constant():
    on_disk = json.loads(_EXPORTED_SCHEMA.read_text())
    assert on_disk["properties"]["schema_version"]["const"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# JSON encoding shape (sanity for GUI consumers).
# ---------------------------------------------------------------------------


def test_to_dict_is_json_dumpable(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec()
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    serialised = json.dumps(to_dict(report))
    assert SCHEMA_VERSION in serialised
    parsed = json.loads(serialised)
    assert parsed["is_clean"] is True


def test_round_trip_preserves_is_clean(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec()
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    assert report.is_clean
    restored = from_dict(to_dict(report))
    assert restored.is_clean == report.is_clean
