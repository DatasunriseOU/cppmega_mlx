"""JSON serialisation surface for ContractProbeReport.

Stable wire format for GUI/CLI consumers (Stage C of the Contract
Probe epic). The schema is also written out to
``docs/contract_probe_schema.json`` and kept in sync via the
``test_schema_matches_export`` test in tests/v4/test_probe_stage_c.py.

Round-trip identity is guaranteed for clean (no-warning) reports —
``from_dict(to_dict(r)) == r``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from cppmega_v4.probe.alternatives import Alternative
from cppmega_v4.probe.capabilities import (
    ColumnSpec,
    ParquetCapabilities,
    SideChannelFamilyCoverage,
    TokenizerCapabilities,
)
from cppmega_v4.probe.probe import ContractProbeReport, ProbeFinding
from cppmega_v4.probe.requirements import DataRequirement


SCHEMA_VERSION: str = "1.0.0"


def to_dict(report: ContractProbeReport) -> dict[str, Any]:
    """Convert a ContractProbeReport to a plain JSON-compatible dict."""
    d: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tokenizer": _tokenizer_to_dict(report.tokenizer),
        "parquet": _parquet_to_dict(report.parquet),
        "findings": [_finding_to_dict(f) for f in report.findings],
        "elapsed_ms": report.elapsed_ms,
        "probe_hidden_size": report.probe_hidden_size,
        "dry_forward_verdict": report.dry_forward_verdict,
        "dry_forward_detail": report.dry_forward_detail,
        "is_clean": report.is_clean,
    }
    return d


def from_dict(data: Mapping[str, Any]) -> ContractProbeReport:
    """Reconstruct a ContractProbeReport from :func:`to_dict` output."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: got {version!r}, "
            f"this build expects {SCHEMA_VERSION!r}"
        )
    return ContractProbeReport(
        tokenizer=_tokenizer_from_dict(data["tokenizer"]),
        parquet=_parquet_from_dict(data["parquet"]),
        findings=tuple(_finding_from_dict(f) for f in data["findings"]),
        elapsed_ms=float(data["elapsed_ms"]),
        probe_hidden_size=int(data["probe_hidden_size"]),
        dry_forward_verdict=data["dry_forward_verdict"],
        dry_forward_detail=data.get("dry_forward_detail", ""),
    )


def json_schema() -> dict[str, Any]:
    """Return the JSON-Schema (draft 2020-12) for ContractProbeReport."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://cppmega.mlx/schemas/contract_probe_report.json",
        "title": "ContractProbeReport",
        "type": "object",
        "required": [
            "schema_version", "tokenizer", "parquet", "findings",
            "elapsed_ms", "probe_hidden_size",
            "dry_forward_verdict", "is_clean",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "tokenizer": {"$ref": "#/$defs/TokenizerCapabilities"},
            "parquet": {"$ref": "#/$defs/ParquetCapabilities"},
            "findings": {
                "type": "array",
                "items": {"$ref": "#/$defs/ProbeFinding"},
            },
            "elapsed_ms": {"type": "number", "minimum": 0},
            "probe_hidden_size": {"type": "integer", "minimum": 1},
            "dry_forward_verdict": {
                "enum": ["ok", "shape_mismatch", "exception", "skipped"],
            },
            "dry_forward_detail": {"type": "string"},
            "is_clean": {"type": "boolean"},
        },
        "$defs": {
            "TokenizerCapabilities": {
                "type": "object",
                "required": [
                    "vocab_size", "special_ids", "has_fim", "has_space_nl",
                    "has_code_start", "has_instruction",
                    "byte_roundtrip", "decoder_kind", "source",
                ],
                "properties": {
                    "vocab_size": {"type": "integer", "minimum": 0},
                    "special_ids": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                    "has_fim": {"type": "boolean"},
                    "has_space_nl": {"type": "boolean"},
                    "has_code_start": {"type": "boolean"},
                    "has_instruction": {"type": "boolean"},
                    "byte_roundtrip": {"enum": ["exact", "approx", "none"]},
                    "decoder_kind": {"enum": ["custom", "hf", "none"]},
                    "source": {"type": "string"},
                },
            },
            "ParquetCapabilities": {
                "type": "object",
                "required": [
                    "schema_columns", "row_count", "total_bytes",
                    "has_token_ids", "has_doc_ids", "has_chunk_spans",
                    "has_call_edges", "has_type_edges", "has_provenance",
                    "side_channels", "sample_seq_lens", "source",
                ],
                "properties": {
                    "schema_columns": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/ColumnSpec"},
                    },
                    "row_count": {"type": "integer", "minimum": 0},
                    "total_bytes": {"type": "integer", "minimum": 0},
                    "has_token_ids": {"type": "boolean"},
                    "has_doc_ids": {"type": "boolean"},
                    "has_chunk_spans": {"type": "boolean"},
                    "has_call_edges": {"type": "boolean"},
                    "has_type_edges": {"type": "boolean"},
                    "has_provenance": {"type": "boolean"},
                    "side_channels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "side_channel_families": {
                        "type": "object",
                        "additionalProperties": {
                            "$ref": "#/$defs/SideChannelFamilyCoverage"
                        },
                    },
                    "sample_seq_lens": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "source": {"type": "string"},
                },
            },
            "ColumnSpec": {
                "type": "object",
                "required": ["name", "arrow_dtype", "nullable", "non_null_ratio"],
                "properties": {
                    "name": {"type": "string"},
                    "arrow_dtype": {"type": "string"},
                    "nullable": {"type": "boolean"},
                    "non_null_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "SideChannelFamilyCoverage": {
                "type": "object",
                "required": [
                    "family", "status", "columns", "missing_columns",
                    "dropped_columns", "token_alignment", "graph_remapping",
                    "provenance", "non_null_ratio",
                ],
                "properties": {
                    "family": {"type": "string"},
                    "status": {
                        "enum": [
                            "present", "partial", "missing", "derived", "dropped",
                        ],
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "dropped_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "token_alignment": {
                        "enum": ["yes", "no", "unknown", "not_applicable"],
                    },
                    "graph_remapping": {
                        "enum": ["yes", "no", "missing", "not_applicable"],
                    },
                    "provenance": {
                        "enum": [
                            "original", "derived", "missing", "dropped",
                            "mixed", "unknown",
                        ],
                    },
                    "non_null_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
            },
            "DataRequirement": {
                "type": "object",
                "required": ["key", "origin", "required", "reason", "satisfied_by"],
                "properties": {
                    "key": {"type": "string"},
                    "origin": {"enum": ["tokenizer", "parquet", "derived"]},
                    "required": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "satisfied_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "Alternative": {
                "type": "object",
                "required": ["action", "target", "diff", "cost", "reason"],
                "properties": {
                    "action": {
                        "enum": ["swap_loss", "swap_tokenizer", "add_column",
                                 "drop_brick", "relax_requirement"],
                    },
                    "target": {"type": "string"},
                    "diff": {"type": "object"},
                    "cost": {"enum": ["low", "medium", "high"]},
                    "reason": {"type": "string"},
                },
            },
            "ProbeFinding": {
                "type": "object",
                "required": ["kind", "component", "requirement",
                             "message", "alternatives"],
                "properties": {
                    "kind": {"enum": ["satisfied", "unsatisfied", "warning"]},
                    "component": {"type": "string"},
                    "requirement": {"$ref": "#/$defs/DataRequirement"},
                    "message": {"type": "string"},
                    "alternatives": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Alternative"},
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Per-type helpers (kept private; consumers go through to_dict/from_dict).
# ---------------------------------------------------------------------------


def _tokenizer_to_dict(t: TokenizerCapabilities) -> dict[str, Any]:
    return {
        "vocab_size": t.vocab_size,
        "special_ids": dict(t.special_ids),
        "has_fim": t.has_fim,
        "has_space_nl": t.has_space_nl,
        "has_code_start": t.has_code_start,
        "has_instruction": t.has_instruction,
        "byte_roundtrip": t.byte_roundtrip,
        "decoder_kind": t.decoder_kind,
        "source": t.source,
    }


def _tokenizer_from_dict(d: Mapping[str, Any]) -> TokenizerCapabilities:
    return TokenizerCapabilities(
        vocab_size=int(d["vocab_size"]),
        special_ids=dict(d["special_ids"]),
        has_fim=bool(d["has_fim"]),
        has_space_nl=bool(d["has_space_nl"]),
        has_code_start=bool(d["has_code_start"]),
        has_instruction=bool(d["has_instruction"]),
        byte_roundtrip=d["byte_roundtrip"],
        decoder_kind=d["decoder_kind"],
        source=d["source"],
    )


def _column_to_dict(c: ColumnSpec) -> dict[str, Any]:
    return asdict(c)


def _column_from_dict(d: Mapping[str, Any]) -> ColumnSpec:
    return ColumnSpec(
        name=d["name"],
        arrow_dtype=d["arrow_dtype"],
        nullable=bool(d["nullable"]),
        non_null_ratio=float(d["non_null_ratio"]),
    )


def _family_coverage_to_dict(c: SideChannelFamilyCoverage) -> dict[str, Any]:
    return {
        "family": c.family,
        "status": c.status,
        "columns": list(c.columns),
        "missing_columns": list(c.missing_columns),
        "dropped_columns": list(c.dropped_columns),
        "token_alignment": c.token_alignment,
        "graph_remapping": c.graph_remapping,
        "provenance": c.provenance,
        "non_null_ratio": c.non_null_ratio,
    }


def _family_coverage_from_dict(d: Mapping[str, Any]) -> SideChannelFamilyCoverage:
    return SideChannelFamilyCoverage(
        family=d["family"],
        status=d["status"],
        columns=tuple(d.get("columns", ())),
        missing_columns=tuple(d.get("missing_columns", ())),
        dropped_columns=tuple(d.get("dropped_columns", ())),
        token_alignment=d["token_alignment"],
        graph_remapping=d["graph_remapping"],
        provenance=d["provenance"],
        non_null_ratio=float(d["non_null_ratio"]),
    )


def _parquet_to_dict(p: ParquetCapabilities) -> dict[str, Any]:
    return {
        "schema_columns": [_column_to_dict(c) for c in p.schema_columns],
        "row_count": p.row_count,
        "total_bytes": p.total_bytes,
        "has_token_ids": p.has_token_ids,
        "has_doc_ids": p.has_doc_ids,
        "has_chunk_spans": p.has_chunk_spans,
        "has_call_edges": p.has_call_edges,
        "has_type_edges": p.has_type_edges,
        "has_provenance": p.has_provenance,
        "side_channels": sorted(p.side_channels),
        "sample_seq_lens": list(p.sample_seq_lens),
        "source": p.source,
        "side_channel_families": {
            name: _family_coverage_to_dict(coverage)
            for name, coverage in sorted(p.side_channel_families.items())
        },
    }


def _parquet_from_dict(d: Mapping[str, Any]) -> ParquetCapabilities:
    return ParquetCapabilities(
        schema_columns=tuple(_column_from_dict(c) for c in d["schema_columns"]),
        row_count=int(d["row_count"]),
        total_bytes=int(d["total_bytes"]),
        has_token_ids=bool(d["has_token_ids"]),
        has_doc_ids=bool(d["has_doc_ids"]),
        has_chunk_spans=bool(d["has_chunk_spans"]),
        has_call_edges=bool(d["has_call_edges"]),
        has_type_edges=bool(d["has_type_edges"]),
        has_provenance=bool(d["has_provenance"]),
        side_channels=frozenset(d["side_channels"]),
        sample_seq_lens=tuple(int(x) for x in d["sample_seq_lens"]),
        source=d["source"],
        side_channel_families={
            name: _family_coverage_from_dict(coverage)
            for name, coverage in d.get("side_channel_families", {}).items()
        },
    )


def _req_to_dict(r: DataRequirement) -> dict[str, Any]:
    return {
        "key": r.key,
        "origin": r.origin,
        "required": r.required,
        "reason": r.reason,
        "satisfied_by": list(r.satisfied_by),
    }


def _req_from_dict(d: Mapping[str, Any]) -> DataRequirement:
    return DataRequirement(
        key=d["key"],
        origin=d["origin"],
        required=bool(d["required"]),
        reason=d["reason"],
        satisfied_by=tuple(d.get("satisfied_by", ())),
    )


def _alt_to_dict(a: Alternative) -> dict[str, Any]:
    return {
        "action": a.action,
        "target": a.target,
        "diff": dict(a.diff),
        "cost": a.cost,
        "reason": a.reason,
    }


def _alt_from_dict(d: Mapping[str, Any]) -> Alternative:
    return Alternative(
        action=d["action"],
        target=d["target"],
        diff=dict(d["diff"]),
        cost=d["cost"],
        reason=d["reason"],
    )


def _finding_to_dict(f: ProbeFinding) -> dict[str, Any]:
    return {
        "kind": f.kind,
        "component": f.component,
        "requirement": _req_to_dict(f.requirement),
        "message": f.message,
        "alternatives": [_alt_to_dict(a) for a in f.alternatives],
    }


def _finding_from_dict(d: Mapping[str, Any]) -> ProbeFinding:
    return ProbeFinding(
        kind=d["kind"],
        component=d["component"],
        requirement=_req_from_dict(d["requirement"]),
        message=d["message"],
        alternatives=tuple(_alt_from_dict(a) for a in d["alternatives"]),
    )
