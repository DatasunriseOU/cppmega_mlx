from __future__ import annotations

import json
from pathlib import Path

import pytest

from cppmega_mlx.tokenizer import TokenizerContractError, load_cppmega_tokenizer
from cppmega_mlx.tokenizer.cpp_tokenizer import EXPECTED_SPECIAL_TOKENS

NANOCHAT_ROOT = Path("/Volumes/external/sources/nanochat")


def _write_tokenizer_json(path: Path, vocab: dict[str, int]) -> None:
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab=vocab, merges=[], unk_token="<UNK>")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


def _valid_vocab() -> dict[str, int]:
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        **EXPECTED_SPECIAL_TOKENS,
        "hello": 1_000,
        "world": 1_001,
    }
    used_ids = set(vocab.values())
    next_id = max(EXPECTED_SPECIAL_TOKENS.values()) + 1
    while len(vocab) < 65_536:
        if next_id not in used_ids:
            token = f"tok_{next_id}"
            vocab[token] = next_id
            used_ids.add(next_id)
        next_id += 1
    return vocab


def test_load_cppmega_tokenizer_accepts_exact_m01_contract(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer_json(tokenizer_path, _valid_vocab())

    tokenizer = load_cppmega_tokenizer(tokenizer_path)

    assert tokenizer.vocab_size == 65_536
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.fim_prefix_id == 4
    assert tokenizer.fim_middle_id == 5
    assert tokenizer.fim_suffix_id == 6
    assert tokenizer.code_start_id == 7
    assert tokenizer.fim_instruction_id == 45
    assert tokenizer.token_for_id(7) == "<CODE_START>"
    assert tokenizer.id_for_token("<CODE_START>") == 7
    assert tokenizer.token_for_id(45) == "<FIM_INSTRUCTION>"
    assert tokenizer.id_for_token("<FIM_INSTRUCTION>") == 45
    ids = tokenizer.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(token_id, int) for token_id in ids)
    with_specials = tokenizer.encode("hello", prepend="<BOS>", append="<EOS>")
    assert with_specials[0] == 2
    assert with_specials[-1] == 3


def test_load_cppmega_tokenizer_accepts_directory_path(tmp_path: Path) -> None:
    _write_tokenizer_json(tmp_path / "tokenizer.json", _valid_vocab())

    tokenizer = load_cppmega_tokenizer(tmp_path)

    assert tokenizer.path == tmp_path / "tokenizer.json"


def test_load_cppmega_tokenizer_rejects_wrong_vocab_size(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab = _valid_vocab()
    vocab.pop(next(token for token in vocab if token.startswith("tok_")))
    _write_tokenizer_json(tokenizer_path, vocab)

    with pytest.raises(TokenizerContractError, match="expected vocab size 65536"):
        load_cppmega_tokenizer(tokenizer_path)


def test_load_cppmega_tokenizer_rejects_wrong_reserved_id_token(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab = _valid_vocab()
    del vocab["<FIM_INSTRUCTION>"]
    vocab["<RESERVED_45>"] = 45
    _write_tokenizer_json(tokenizer_path, vocab)

    with pytest.raises(
        TokenizerContractError,
        match="<FIM_INSTRUCTION>.*must use id 45.*id 45 maps to '<RESERVED_45>'",
    ):
        load_cppmega_tokenizer(tokenizer_path)


def test_m01_declared_nanochat_tokenizer_json_fails_closed() -> None:
    tokenizer_path = NANOCHAT_ROOT / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.skip(f"{tokenizer_path} is not available")

    with pytest.raises(TokenizerContractError, match="expected vocab size 65536"):
        load_cppmega_tokenizer(tokenizer_path)


def test_nanochat_v3_artifact_satisfies_special_id_contract() -> None:
    """nanochat tokenizer_v3.json now matches the M0.1 special-id contract."""
    tokenizer_path = NANOCHAT_ROOT / "tokenizer_v3.json"
    if not tokenizer_path.is_file():
        pytest.skip(f"{tokenizer_path} is not available")

    tokenizer = load_cppmega_tokenizer(tokenizer_path)

    assert tokenizer.vocab_size == 65_536
    assert tokenizer.code_start_id == 7
    assert tokenizer.fim_instruction_id == 45
    assert tokenizer.token_for_id(7) == "<CODE_START>"
    assert tokenizer.id_for_token("<CODE_START>") == 7
    assert tokenizer.token_for_id(45) == "<FIM_INSTRUCTION>"
    assert tokenizer.id_for_token("<FIM_INSTRUCTION>") == 45


def test_decode_parity_with_gb10_reference_receipt() -> None:
    """MLX-side decode is byte-identical to gb10's CppTokenizer.decode.

    Each entry pairs an ID stream with the exact decoded string captured from
    gb10's nanochat CppTokenizer wrapper (the CUDA reference). This guards the
    M0.1 acceptance gate: MLX inference, FIM transforms, and RL reward parsing
    must produce the same strings as the CUDA reference for any ID stream.
    """

    tokenizer_path = (
        Path(__file__).resolve().parents[1]
        / "cppmega_mlx"
        / "tokenizer"
        / "tokenizer.json"
    )
    if not tokenizer_path.is_file():
        pytest.skip("vendored tokenizer.json not present")

    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "decode_receipt_gb10.json"
    )
    if not receipt_path.is_file():
        pytest.skip("decode receipt fixture not present")

    receipt = json.loads(receipt_path.read_text())
    tokenizer = load_cppmega_tokenizer(tokenizer_path)
    assert tokenizer.vocab_size == receipt["vocab_size"]

    for sample in receipt["samples"]:
        text = sample["text"]
        ref_ids = sample["ids"]
        ref_decoded = sample["decoded"]
        assert tokenizer.encode(text) == ref_ids, text
        assert tokenizer.decode(ref_ids) == ref_decoded, text


def test_nanochat_v3_fixed_tokens_config_matches_special_id_contract() -> None:
    config_path = NANOCHAT_ROOT / "config" / "tokenizer_v3_fixed_tokens.json"
    if not config_path.is_file():
        pytest.skip(f"{config_path} is not available")

    payload = json.loads(config_path.read_text())
    special_tokens = payload["special_tokens"]["tokens"]

    assert payload["_total_vocab"] == 65_536
    assert special_tokens["<BOS>"] == 2
    assert special_tokens["<EOS>"] == 3
    assert special_tokens["<FIM_PREFIX>"] == 4
    assert special_tokens["<FIM_MIDDLE>"] == 5
    assert special_tokens["<FIM_SUFFIX>"] == 6
    assert special_tokens["<CODE_START>"] == 7
    assert special_tokens["<FIM_INSTRUCTION>"] == 45
