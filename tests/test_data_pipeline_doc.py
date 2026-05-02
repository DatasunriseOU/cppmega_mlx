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
        "Mapping-batch training can carry explicit packed document IDs",
        "loaders still need an owned schema pass",
        "PyTorch DataLoader integration is explicit and optional",
        "M0.1 tokenizer parity is already closed",
        "packed document-id schema ownership",
        "100M-token stress gate",
    ]
    for phrase in required_phrases:
        assert " ".join(phrase.lower().split()) in normalized_text


def test_sequence_packing_helpers_are_public_data_exports() -> None:
    expected_exports = {
        "PackingStrategy",
        "OversizedSamplePolicy",
        "PackedSequences",
        "cumulative_doc_ids_from_eos",
        "document_boundary_mask",
        "mlx_cumulative_doc_ids_from_eos",
        "mlx_document_boundary_mask",
        "mlx_sequence_packing_attention_mask",
        "pack_bos_aligned_best_fit",
        "pack_documents_with_eos",
    }

    assert expected_exports <= set(data.__all__)
    assert callable(data.cumulative_doc_ids_from_eos)
    assert callable(data.document_boundary_mask)
    assert callable(data.mlx_sequence_packing_attention_mask)
    assert callable(data.pack_bos_aligned_best_fit)
    assert callable(data.pack_documents_with_eos)
