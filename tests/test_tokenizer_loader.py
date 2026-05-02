from __future__ import annotations

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
    assert tokenizer.fim_instruction_id == 7
    assert tokenizer.token_for_id(7) == "<FIM_INSTRUCTION>"
    assert tokenizer.id_for_token("<FIM_INSTRUCTION>") == 7
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
    vocab["<CODE_START>"] = 7
    _write_tokenizer_json(tokenizer_path, vocab)

    with pytest.raises(
        TokenizerContractError, match="<FIM_INSTRUCTION>.*must use id 7"
    ):
        load_cppmega_tokenizer(tokenizer_path)


def test_m01_declared_nanochat_tokenizer_json_fails_closed() -> None:
    tokenizer_path = NANOCHAT_ROOT / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.skip(f"{tokenizer_path} is not available")

    with pytest.raises(TokenizerContractError, match="expected vocab size 65536"):
        load_cppmega_tokenizer(tokenizer_path)


def test_nanochat_v3_artifact_fails_closed_on_fim_instruction_contract() -> None:
    tokenizer_path = NANOCHAT_ROOT / "tokenizer_v3.json"
    if not tokenizer_path.is_file():
        pytest.skip(f"{tokenizer_path} is not available")

    with pytest.raises(
        TokenizerContractError, match="<FIM_INSTRUCTION>.*must use id 7"
    ):
        load_cppmega_tokenizer(tokenizer_path)
