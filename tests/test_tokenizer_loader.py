from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import cast

import pytest

from cppmega_mlx.data.tokenizer_contract import (
    RESERVED_ROLE_TOKEN_IDS,
    SPECIAL_TOKEN_IDS,
)
from cppmega_mlx.tokenizer import TokenizerContractError, load_cppmega_tokenizer
from cppmega_mlx.tokenizer.cpp_tokenizer import (
    EXPECTED_SPECIAL_TOKENS,
    normalize_whitespace_with_offsets,
)

NANOCHAT_ROOT = Path("/Volumes/external/sources/nanochat")
VENDORED_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1] / "cppmega_mlx" / "tokenizer" / "tokenizer.json"
)


def _write_tokenizer_json(path: Path, vocab: dict[str, int]) -> None:
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.BPE(vocab=vocab, merges=[], unk_token="<UNK>")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    control_tokens = [
        f"<{role}>"
        for role, _token_id in sorted(
            SPECIAL_TOKEN_IDS.items(), key=lambda item: item[1]
        )
        if f"<{role}>" in vocab
    ]
    control_tokens.extend(
        f"<RESERVED_{token_id}>"
        for token_id in sorted(set(RESERVED_ROLE_TOKEN_IDS.values()))
        if f"<RESERVED_{token_id}>" in vocab
    )
    tokenizer.add_special_tokens(control_tokens)
    tokenizer.save(str(path))


def _valid_vocab() -> dict[str, int]:
    vocab = {
        **{f"<{role}>": token_id for role, token_id in SPECIAL_TOKEN_IDS.items()},
        **{
            f"<RESERVED_{token_id}>": token_id
            for token_id in RESERVED_ROLE_TOKEN_IDS.values()
        },
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


def test_load_cppmega_tokenizer_accepts_exact_m01_contract() -> None:
    tokenizer = load_cppmega_tokenizer(VENDORED_TOKENIZER_PATH)

    assert tokenizer.vocab_size == 65_536
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.fim_prefix_id == 4
    assert tokenizer.fim_middle_id == 5
    assert tokenizer.fim_suffix_id == 6
    assert tokenizer.code_start_id == 7
    assert tokenizer.code_end_id == 8
    assert tokenizer.think_start_id == 9
    assert tokenizer.think_end_id == 10
    assert tokenizer.query_tool_id == 11
    assert tokenizer.tool_result_id == 19
    assert tokenizer.fim_instruction_id == 45
    assert tokenizer.token_for_id(7) == "<CODE_START>"
    assert tokenizer.id_for_token("<CODE_START>") == 7
    assert tokenizer.token_for_id(8) == "<CODE_END>"
    assert tokenizer.id_for_token("<CODE_END>") == 8
    assert tokenizer.token_for_id(9) == "<THINK_START>"
    assert tokenizer.id_for_token("<THINK_START>") == 9
    assert tokenizer.token_for_id(10) == "<THINK_END>"
    assert tokenizer.id_for_token("<THINK_END>") == 10
    assert tokenizer.token_for_id(11) == "<QUERY_TOOL>"
    assert tokenizer.id_for_token("<QUERY_TOOL>") == 11
    assert tokenizer.token_for_id(19) == "<TOOL_RESULT>"
    assert tokenizer.id_for_token("<TOOL_RESULT>") == 19
    assert tokenizer.token_for_id(45) == "<FIM_INSTRUCTION>"
    assert tokenizer.id_for_token("<FIM_INSTRUCTION>") == 45
    assert tokenizer.space_token_id == 46
    assert tokenizer.nl_token_id == 47
    assert tokenizer.token_for_id(46) == "<SPACE>"
    assert tokenizer.token_for_id(47) == "<NL>"
    ids = tokenizer.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(token_id, int) for token_id in ids)
    with_specials = tokenizer.encode("hello", prepend="<BOS>", append="<EOS>")
    assert with_specials[0] == 2
    assert with_specials[-1] == 3


def test_whitespace_normalizer_keeps_docstring_pre_boundary_after_apostrophe() -> None:
    text = (
        "/*\n"
        " * @discussion\n"
        " * This can't be compiled with aarch32; see https://github.com/a/b. *\n"
        " */\n"
        "// === PRE-COMMIT ===\n"
        "bool f();\n"
    )

    normalized, offsets = normalize_whitespace_with_offsets(text)

    assert len(offsets) == len(normalized)
    assert "can't" in normalized
    assert "https://github.com/a/b" in normalized
    assert ".<SPACE>*<NL><SPACE>*/<NL>//<SPACE>===<SPACE>PRE-COMMIT" in normalized


def test_whitespace_normalizer_still_preserves_string_literal_whitespace() -> None:
    text = 'const char* s = "a  b\\n c";\nint x;\n'

    normalized, offsets = normalize_whitespace_with_offsets(text)

    assert len(offsets) == len(normalized)
    assert '"a  b\\n c"' in normalized
    assert '=<SPACE>"a  b\\n c";<NL>int<SPACE>x;' in normalized


def test_load_cppmega_tokenizer_accepts_directory_path(tmp_path: Path) -> None:
    shutil.copyfile(VENDORED_TOKENIZER_PATH, tmp_path / "tokenizer.json")

    tokenizer = load_cppmega_tokenizer(tmp_path)

    assert tokenizer.path == tmp_path / "tokenizer.json"


def test_load_cppmega_tokenizer_rejects_role_compatible_untrained_artifact(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer_json(tokenizer_path, _valid_vocab())

    with pytest.raises(TokenizerContractError, match="expected 7200 added tokens"):
        load_cppmega_tokenizer(tokenizer_path)


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
        match="id 45 must map to '<FIM_INSTRUCTION>'.*'<RESERVED_45>'",
    ):
        load_cppmega_tokenizer(tokenizer_path)


def test_load_cppmega_tokenizer_rejects_swapped_domain_delimiter_slots(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab = _valid_vocab()
    vocab["<RESERVED_209>"], vocab["<RESERVED_210>"] = 210, 209
    _write_tokenizer_json(tokenizer_path, vocab)

    with pytest.raises(
        TokenizerContractError,
        match="id 209 must map to '<RESERVED_209>'.*'<RESERVED_210>'",
    ):
        load_cppmega_tokenizer(tokenizer_path)


def test_load_cppmega_tokenizer_rejects_missing_added_control_token(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer_json(tokenizer_path, _valid_vocab())
    payload = json.loads(tokenizer_path.read_text())
    payload["added_tokens"] = [
        token for token in payload["added_tokens"] if token["id"] != 209
    ]
    tokenizer_path.write_text(json.dumps(payload))

    with pytest.raises(
        TokenizerContractError,
        match="control token '<RESERVED_209>'.*must be present in added_tokens",
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
    assert tokenizer.code_end_id == 8
    assert tokenizer.query_tool_id == 11
    assert tokenizer.tool_result_id == 19
    assert tokenizer.fim_instruction_id == 45
    assert tokenizer.token_for_id(7) == "<CODE_START>"
    assert tokenizer.id_for_token("<CODE_START>") == 7
    assert tokenizer.token_for_id(45) == "<FIM_INSTRUCTION>"
    assert tokenizer.id_for_token("<FIM_INSTRUCTION>") == 45


def test_decode_parity_with_gb10_reference_receipt() -> None:
    """MLX-side decode is byte-identical to gb10's CppTokenizer.decode.

    Each entry pairs an ID stream with the exact decoded string captured from
    gb10's nanochat CppTokenizer wrapper (the deployed tokenizer receipt).
    This guards the M0.1 acceptance gate: MLX inference, FIM transforms, and
    RL reward parsing must produce the same strings as the deployed tokenizer
    receipt for any ID stream.
    """

    if not VENDORED_TOKENIZER_PATH.is_file():
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
    tokenizer = load_cppmega_tokenizer(VENDORED_TOKENIZER_PATH)
    assert tokenizer.vocab_size == receipt["vocab_size"]

    for sample in receipt["samples"]:
        text = sample["text"]
        ref_ids = sample["ids"]
        ref_decoded = sample["decoded"]
        assert tokenizer.encode(text) == ref_ids, text
        assert tokenizer.decode(ref_ids) == ref_decoded, text


def test_bpe_split_identifier_decodes_without_spurious_spaces() -> None:
    """Regression: identifiers split by BPE (e.g. "sum" -> "s","u","m") must
    decode back as a contiguous identifier. Before the <SPACE>/<NL> redesign,
    decode produced "int s u m = 5" because the wrapper inserted spaces between
    every BPE piece. Now spaces only appear when an explicit <SPACE> token
    (id=46) sits between pieces.
    """

    if not VENDORED_TOKENIZER_PATH.is_file():
        pytest.skip("vendored tokenizer.json not present")

    tokenizer = load_cppmega_tokenizer(VENDORED_TOKENIZER_PATH)

    cases = [
        ("int sum = 5;", "int sum = 5;"),
        ("void foo() { return 0; }", "void foo() { return 0; }"),
        ("int factorial(int n) { return n; }", "int factorial(int n) { return n; }"),
        ("a\nb", "a\nb"),
        ("tab\there", "tab here"),  # tab collapses to a single space
        ("  multi   space\n\nblank", " multi space\nblank"),  # WS runs collapse
    ]
    for original, expected in cases:
        ids = cast(list[int], tokenizer.encode(original))
        assert all(isinstance(token_id, int) for token_id in ids)
        decoded = tokenizer.decode(ids)
        assert decoded == expected, (
            f"input={original!r}\n  expected={expected!r}\n  got={decoded!r}"
        )


def test_decode_and_optional_control_ids_fail_closed_outside_vocab() -> None:
    tokenizer = load_cppmega_tokenizer(VENDORED_TOKENIZER_PATH)

    with pytest.raises(TokenizerContractError, match="outside tokenizer vocab: 65536"):
        tokenizer.decode([65_536])
    with pytest.raises(TokenizerContractError, match="must be an integer, got bool"):
        tokenizer.decode([True])
    with pytest.raises(
        TokenizerContractError, match="id 65536 is outside tokenizer vocab"
    ):
        tokenizer.encode("int x;", prepend=65_536)


def test_line_comment_newline_survives_encode_decode_roundtrip() -> None:
    """A // comment must not absorb the following source line.

    C++ single-line comments are syntactically terminated only by the newline.
    If the tokenizer/detokenizer ever drops that <NL>, clang-format cannot
    reconstruct the program: the next code line becomes part of the comment.
    """

    if not VENDORED_TOKENIZER_PATH.is_file():
        pytest.skip("vendored tokenizer.json not present")

    tokenizer = load_cppmega_tokenizer(VENDORED_TOKENIZER_PATH)
    source = (
        "int a = 0; // keep this comment\n"
        "int b = 1; // second comment\n"
        "return a + b;\n"
    )

    ids = cast(list[int], tokenizer.encode(source))
    tokens = [tokenizer.token_for_id(token_id) for token_id in ids]

    assert "//" in tokens
    assert tokens.count("<NL>") == 3
    assert tokens[tokens.index("//") + 1 :].index("<NL>") >= 1
    assert tokenizer.decode(ids) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("int x; // C comment\nint y;\n", "int x; // C comment\nint y;\n"),
        ("/// C++ doc comment\nint f();\n", "/// C++ doc comment\nint f();\n"),
        ("// CRLF comment\r\nint value;\r\n", "// CRLF comment\nint value;\n"),
        (
            "/** block doc\n * line\n */\nint g();\n",
            "/** block doc\n * line\n */\nint g();\n",
        ),
    ],
)
def test_comment_and_docstring_eol_roundtrip(source: str, expected: str) -> None:
    tokenizer = load_cppmega_tokenizer(VENDORED_TOKENIZER_PATH)

    ids = cast(list[int], tokenizer.encode(source))

    assert ids.count(tokenizer.nl_token_id) == expected.count("\n")
    assert tokenizer.decode(ids) == expected


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
    assert special_tokens["<CODE_END>"] == 8
    assert special_tokens["<THINK_START>"] == 9
    assert special_tokens["<THINK_END>"] == 10
    assert special_tokens["<QUERY_TOOL>"] == 11
    assert special_tokens["<TOOL_RESULT>"] == 19
    assert special_tokens["<FIM_INSTRUCTION>"] == 45
    assert special_tokens["<SPACE>"] == 46
    assert special_tokens["<NL>"] == 47
