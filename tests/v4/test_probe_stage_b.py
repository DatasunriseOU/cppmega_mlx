"""Contract Probe Stage B tests — requirements + solver + alternatives.

Each test exercises one slice of the pipeline:
  - BRICK_REQUIREMENTS is exhaustive (covers every BLOCK_BUILDERS key).
  - LOSS_REQUIREMENTS satisfaction logic per LossKind.
  - contract_probe end-to-end on synthetic tokenizer + parquet fixtures.
  - alternatives generation for each unmet case.
  - dry_forward verdicts (ok / shape_mismatch via brick mismatch / etc).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.buildspec import (
    LossKind,
    ModelBuildSpec,
    adamw,
    cross_entropy_loss,
    ifim_shaped_loss,
    mhc_attn_bias_loss,
    mtp_weighted_loss,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS
from cppmega_v4.probe import (
    Alternative,
    BRICK_REQUIREMENTS,
    ContractProbeReport,
    DataRequirement,
    LOSS_REQUIREMENTS,
    ProbeFinding,
    contract_probe,
    dry_forward,
    generate_alternatives,
    introspect_parquet,
)


_VENDORED_TOKENIZER = Path("cppmega_mlx/tokenizer/tokenizer.json")


def _write_full_parquet(p: Path, *, n_rows: int = 16, with_edges: bool = True):
    cols: dict[str, list] = {
        "input_ids": [list(range(8)) for _ in range(n_rows)],
        "doc_ids":   [i // 4 for i in range(n_rows)],
    }
    if with_edges:
        cols["call_edges"] = [[(0, 1)] for _ in range(n_rows)]
        cols["type_edges"] = [[(0, 2)] for _ in range(n_rows)]
    pq.write_table(pa.table(cols), p)


def _write_token_only_parquet(p: Path):
    pq.write_table(pa.table({
        "token_ids": [list(range(4)), list(range(4, 8))],
    }), p)


def _write_partial_side_channel_parquet(p: Path):
    pq.write_table(pa.table({
        "token_ids": [list(range(4)), list(range(4, 8))],
        # Source-level AST ids from current GB10 samples: useful provenance,
        # but not token-coordinate training side channels.
        "structure_ids": [[1, 2], [3, 4, 5]],
        "call_edges": [[(0, 1)], [(1, 2)]],
        "source_path": ["a.cc", "b.cc"],
    }), p)


def _write_enriched_side_channel_parquet(p: Path):
    rows = 2
    tokens = [list(range(4)), list(range(4, 8))]
    pq.write_table(pa.table({
        "input_ids": tokens,
        "target_ids": [row[1:] + [0] for row in tokens],
        "loss_mask": [[1, 1, 1, 0] for _ in range(rows)],
        "doc_ids": [[1, 1, 2, 2], [1, 1, 1, 1]],
        "pack_id": [0, 1],
        "valid_token_count": [4, 4],
        "num_docs": [2, 2],
        "platform_ids": [[10, 99], [20, 99]],
        "source_platform_ids": [[[10, 99], [11, 99]], [[20, 99]]],
        "token_ast_depth": [[0, 1, 1, 2], [0, 1, 2, 2]],
        "token_sibling_index": [[0, 0, 1, 0], [0, 0, 0, 1]],
        "token_ast_node_type": [[1, 2, 2, 3], [1, 2, 3, 3]],
        "token_structure_ids": [[4, 4, 5, 5], [6, 6, 7, 7]],
        "token_dep_levels": [[0, 1, 1, 2], [0, 1, 2, 2]],
        "token_call_edges": [[(0, 1)], [(1, 3)]],
        "token_type_edges": [[(0, 2)], [(2, 3)]],
        "token_change_mask_pre": [[0, 0, 1, 1], [0, 1, 1, 0]],
        "hunk_id_per_token": [[0, 0, 1, 1], [2, 2, 2, 2]],
        "edit_op_per_token": [[0, 0, 1, 1], [1, 1, 0, 0]],
        "source_file_id": [101, 102],
        "language_id": [1, 1],
        "extractor_name": ["clang", "clang"],
        "extractor_version": ["v10", "v10"],
        "tokenizer_id": ["nanochat", "nanochat"],
    }), p)


def _minimal_tokenizer(p: Path, *, with_fim: bool = False):
    added = [
        {"id": 0, "content": "<PAD>"},
        {"id": 1, "content": "<UNK>"},
        {"id": 2, "content": "<BOS>"},
        {"id": 3, "content": "<EOS>"},
    ]
    if with_fim:
        added += [
            {"id": 4, "content": "<FIM_PREFIX>"},
            {"id": 5, "content": "<FIM_MIDDLE>"},
            {"id": 6, "content": "<FIM_SUFFIX>"},
        ]
    p.write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(256)}},
        "added_tokens": added,
        "decoder": {"type": "ByteLevel"},
    }))


def _make_spec(preset: str, hidden: int = 64, loss=None) -> ModelBuildSpec:
    specs = build_preset_specs(preset, hidden_size=hidden)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    head = graph.nodes[-1].name
    return ModelBuildSpec(
        graph=graph,
        loss=loss or cross_entropy_loss(head_output_name=head),
        optim=adamw(),
    )


# ---------------------------------------------------------------------------
# Coverage gate: every BLOCK_BUILDERS key must have a requirements entry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(BLOCK_BUILDERS.keys()))
def test_brick_requirements_table_covers_every_block_builder(kind):
    assert kind in BRICK_REQUIREMENTS, (
        f"brick {kind!r} is in BLOCK_BUILDERS but missing from "
        f"BRICK_REQUIREMENTS — add an entry (empty tuple is fine)"
    )


def test_loss_requirements_table_covers_every_loss_kind():
    for kind in LossKind:
        assert kind in LOSS_REQUIREMENTS


# ---------------------------------------------------------------------------
# DataRequirement satisfied-by logic.
# ---------------------------------------------------------------------------


def test_data_requirement_direct_match():
    r = DataRequirement("FIM_PREFIX", "tokenizer", True, "")
    assert r.is_satisfied_by(frozenset({"FIM_PREFIX"}))
    assert not r.is_satisfied_by(frozenset({"BOS"}))


def test_data_requirement_satisfied_by_alternative_key():
    r = DataRequirement("labels", "derived", True, "", satisfied_by=("input_ids",))
    assert r.is_satisfied_by(frozenset({"input_ids"}))


# ---------------------------------------------------------------------------
# Side-channel family capability discovery.
# ---------------------------------------------------------------------------


def test_parquet_capabilities_reports_token_only_family_status(tmp_path: Path):
    pqp = tmp_path / "token_only.parquet"
    _write_token_only_parquet(pqp)
    caps = introspect_parquet(pqp)

    assert caps.side_channel_families["universal"].status == "derived"
    assert caps.side_channel_families["universal"].provenance == "derived"
    assert caps.side_channel_families["platform"].status == "missing"
    assert caps.side_channel_families["structure"].status == "missing"
    assert caps.side_channels == frozenset()


def test_parquet_capabilities_reports_dropped_source_level_columns(tmp_path: Path):
    pqp = tmp_path / "partial.parquet"
    _write_partial_side_channel_parquet(pqp)
    caps = introspect_parquet(pqp)

    structure = caps.side_channel_families["structure"]
    graph = caps.side_channel_families["semantic_graph"]
    assert structure.status == "dropped"
    assert structure.token_alignment == "no"
    assert structure.dropped_columns == ("structure_ids",)
    assert graph.status == "dropped"
    assert graph.graph_remapping == "no"
    assert graph.dropped_columns == ("call_edges",)


def test_parquet_capabilities_reports_enriched_family_coverage(tmp_path: Path):
    pqp = tmp_path / "enriched.parquet"
    _write_enriched_side_channel_parquet(pqp)
    caps = introspect_parquet(pqp)

    assert caps.has_provenance is True
    assert caps.side_channel_families["universal"].status == "present"
    assert caps.side_channel_families["platform"].status == "present"
    assert caps.side_channel_families["platform"].token_alignment == "yes"
    assert caps.side_channel_families["syntax"].status == "present"
    assert caps.side_channel_families["structure"].status == "partial"
    assert caps.side_channel_families["structure"].token_alignment == "yes"
    assert caps.side_channel_families["semantic_graph"].status == "partial"
    assert caps.side_channel_families["semantic_graph"].graph_remapping == "yes"
    assert caps.side_channel_families["temporal_diff"].status == "partial"
    assert caps.side_channel_families["temporal_diff"].provenance == "original"


# ---------------------------------------------------------------------------
# Full contract_probe end-to-end.
# ---------------------------------------------------------------------------


def test_contract_probe_clean_on_llama_with_ce_and_full_data(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec("llama3_8b")
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    assert isinstance(report, ContractProbeReport)
    assert report.is_clean, [f.message for f in report.blocking]
    assert report.dry_forward_verdict == "ok"


def test_contract_probe_blocks_ifim_on_tokenizer_without_fim(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    tok = tmp_path / "tiny_tok.json"
    _minimal_tokenizer(tok, with_fim=False)
    spec = _make_spec("llama3_8b", loss=ifim_shaped_loss())
    report = contract_probe(spec, tok, pqp, run_dry_forward=False)
    assert not report.is_clean
    blocking_keys = {f.requirement.key for f in report.blocking}
    assert {"FIM_PREFIX", "FIM_MIDDLE", "FIM_SUFFIX"} <= blocking_keys


def test_contract_probe_allows_ifim_on_tokenizer_with_fim(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    tok = tmp_path / "fim_tok.json"
    _minimal_tokenizer(tok, with_fim=True)
    spec = _make_spec("llama3_8b", loss=ifim_shaped_loss())
    report = contract_probe(spec, tok, pqp, run_dry_forward=False)
    assert report.is_clean


def test_contract_probe_blocks_mhc_when_type_edges_missing(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp, with_edges=False)
    spec = _make_spec("llama3_8b", loss=mhc_attn_bias_loss())
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp, run_dry_forward=False)
    assert not report.is_clean
    assert any(f.requirement.key == "type_edges" for f in report.blocking)


def test_contract_probe_allows_mtp_via_derived_labels(tmp_path: Path):
    """MTP labels_k_shifted is satisfied_by=('input_ids',) — derived path."""
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec("llama3_8b", loss=mtp_weighted_loss(k=2))
    report = contract_probe(spec, _VENDORED_TOKENIZER, pqp, run_dry_forward=False)
    assert report.is_clean


# ---------------------------------------------------------------------------
# Alternatives.
# ---------------------------------------------------------------------------


def test_alternatives_loss_swap_for_fim_unmet(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    tok = tmp_path / "tiny_tok.json"
    _minimal_tokenizer(tok, with_fim=False)
    spec = _make_spec("llama3_8b", loss=ifim_shaped_loss())
    report = contract_probe(spec, tok, pqp, run_dry_forward=False)
    finding = next(f for f in report.blocking if f.requirement.key == "FIM_PREFIX")
    actions = {a.action for a in finding.alternatives}
    assert "swap_loss" in actions
    assert "swap_tokenizer" in actions


def test_alternatives_capped_at_three_per_finding():
    """Design ceiling: max 3 alternatives per finding."""
    for finding_alts in (
        generate_alternatives(
            LOSS_REQUIREMENTS[LossKind.IFIM_SHAPED][0],
            "loss:ifim_shaped",
            _make_spec("llama3_8b", loss=ifim_shaped_loss()),
            _vendored_caps(),
            _empty_parquet_caps(),
        ),
    ):
        assert len(finding_alts) <= 3


def test_alternatives_ordered_by_cost():
    """Cheapest action first (low < medium < high)."""
    alts = generate_alternatives(
        LOSS_REQUIREMENTS[LossKind.IFIM_SHAPED][0],
        "loss:ifim_shaped",
        _make_spec("llama3_8b", loss=ifim_shaped_loss()),
        _vendored_caps(),
        _empty_parquet_caps(),
    )
    cost_seq = [a.cost for a in alts]
    assert cost_seq == sorted(cost_seq, key=lambda c: {"low": 0, "medium": 1, "high": 2}[c])


# ---------------------------------------------------------------------------
# dry_forward.
# ---------------------------------------------------------------------------


def test_dry_forward_ok_on_linear_chain():
    specs = build_preset_specs("llama3_8b", hidden_size=64)
    graph = from_block_specs(specs, hidden_size=64, instantiate=False)
    result = dry_forward(graph, hidden_size=64, seq_len=8)
    assert result.verdict == "ok", result.detail


def test_dry_forward_ok_on_parallel_block():
    from cppmega_v4.architectures.presets import tiny_aya_parallel_specs
    specs = tiny_aya_parallel_specs(64)
    graph = from_block_specs(specs, hidden_size=64, instantiate=False)
    result = dry_forward(graph, hidden_size=64, seq_len=8)
    assert result.verdict == "ok", result.detail


# ---------------------------------------------------------------------------
# Performance.
# ---------------------------------------------------------------------------


def test_contract_probe_under_2s_on_largest_preset(tmp_path: Path):
    pqp = tmp_path / "shard.parquet"
    _write_full_parquet(pqp)
    spec = _make_spec("qwen3_235b_a22b")
    t0 = time.perf_counter()
    contract_probe(spec, _VENDORED_TOKENIZER, pqp)
    elapsed = (time.perf_counter() - t0)
    assert elapsed < 2.0, f"probe took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Helpers — keep at file end to avoid forward-ref hassle.
# ---------------------------------------------------------------------------


def _vendored_caps():
    from cppmega_v4.probe import introspect_tokenizer
    return introspect_tokenizer(_VENDORED_TOKENIZER)


def _empty_parquet_caps():
    from cppmega_v4.probe import ParquetCapabilities
    return ParquetCapabilities(
        schema_columns=(),
        row_count=0,
        total_bytes=0,
        has_token_ids=False,
        has_doc_ids=False,
        has_chunk_spans=False,
        has_call_edges=False,
        has_type_edges=False,
        has_provenance=False,
        side_channels=frozenset(),
        sample_seq_lens=(),
        source="<empty>",
    )
