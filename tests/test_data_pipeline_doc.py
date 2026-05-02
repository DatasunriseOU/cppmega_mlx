from __future__ import annotations

from pathlib import Path

from cppmega_mlx import data


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PIPELINE_DOC = REPO_ROOT / "docs" / "data_pipeline.md"


def test_data_pipeline_doc_states_ingress_packing_and_guardrails() -> None:
    text = DATA_PIPELINE_DOC.read_text(encoding="utf-8")
    normalized_text = " ".join(text.lower().split())

    required_phrases = [
        "npz shards",
        "parquet files",
        "megatron indexed `.bin/.idx`",
        "appends one EOS token",
        "fixed-length packed rows",
        "document-boundary mask",
        "not wired into the training loop",
        "not consumed by the current attention implementation",
    ]
    for phrase in required_phrases:
        assert " ".join(phrase.lower().split()) in normalized_text


def test_sequence_packing_helpers_are_public_data_exports() -> None:
    expected_exports = {
        "OversizedSamplePolicy",
        "PackedSequences",
        "cumulative_doc_ids_from_eos",
        "document_boundary_mask",
        "pack_documents_with_eos",
    }

    assert expected_exports <= set(data.__all__)
    assert callable(data.cumulative_doc_ids_from_eos)
    assert callable(data.document_boundary_mask)
    assert callable(data.pack_documents_with_eos)
