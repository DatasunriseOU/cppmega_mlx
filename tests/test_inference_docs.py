from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "inference.md"


def _read_doc() -> str:
    return DOC.read_text()


def test_inference_modes_doc_pins_supported_local_surfaces() -> None:
    text = _read_doc()

    for phrase in (
        "# Inference Modes",
        "## Mode Matrix",
        "generate_tokens(",
        "generate_tokens_with_kv_cache(",
        "stream_generate_tokens(",
        "generate_tokens_with_prompt_cache(",
        "generate_tokens_speculative(",
        "generate_tokens_mtp_self_speculative(",
        "create_local_generation_app(",
        "scripts/bench_inference_quality.py",
        "scripts/bench_inference_long_context.py",
        "scripts/quantize_for_inference.py",
        "JsonConstrainedLogitsProcessor",
        "JsonTokenIds",
        "logits_processors",
        "QuantizedKVCache",
        "model_kwargs_builder",
    ):
        assert phrase in text


def test_inference_modes_doc_keeps_non_claims_explicit() -> None:
    text = _read_doc()

    for phrase in (
        "not an OpenAI-compatible API",
        "not model-integrated paged attention",
        "not a real ARC/MMLU/HumanEval leaderboard run",
        "not a real NIAH/RULER leaderboard run",
        "not a GB10 parity claim",
        "not a full checkpoint converter",
        "not mixed bf16-to-q4 quantized_kv_start > 0 transition coverage",
        "not JSON Schema or raw text parsing",
    ):
        assert phrase in text
