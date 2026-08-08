from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)
from cppmega_mlx.data.nanochat_pipeline.tokenized_enriched import (
    materialize_tokenized_enriched_batch,
)


_FREEBSD_DIALOG_TESTDATA_8BIT_B64 = (
    "IyEvYmluL3NoCiMgJElkOiB0ZXN0ZGF0YS04Yml0LHYgMS4yIDIwMTEvMTAvMTYgMjM6MjY6MzIg"
    "dG9tIEV4cCAkCgojIFNlbGVjdCBvbmUgb2YgdGhlICJTQU1QTEU9IiBsaW5lcywgdG8gdGVzdCBo"
    "YW5kbGluZyBvZiBjaGFyYWN0ZXJzIHdoaWNoCiMgYXJlIG5vbnByaW50aW5nIGluIGEgUE9TSVgg"
    "bG9jYWxlOgoKY2FzZSAuJDEgaW4KCSMgQzEgY29udHJvbHMKLjgpCglTQU1QTEU9IoCBgoOEhYaH"
    "iImKi4yNjo8iCgk7OwouOSkKCVNBTVBMRT0ikJGSk5SVlpeYmZqbnJ2enyIKCTs7CgojIExhdGlu"
    "LTEKLlthQV0pCglTQU1QTEU9IqChoqOkpaanqKmqq6ytrq8iCgk7OwouW2JCXSkKCVNBTVBMRT0i"
    "sLGys7S1tre4ubq7vL2+vyIKCTs7Ci5bY0NdKQoJU0FNUExFPSLAwcLDxMXGx8jJysvMzc7PIgoJ"
    "OzsKLltkRF0pCglTQU1QTEU9ItDR0tPU1dbX2Nna29zd3t8iCgk7OwouW2VFXSkKCVNBTVBMRT0i"
    "4OHi4+Tl5ufo6err7O3u7yIKCTs7Ci5bZkZdKQoJU0FNUExFPSLw8fLz9PX29/j5+vv8/f7/IgoJ"
    "OzsKKikKCSMgQzAgY29udHJvbHMgKGV4Y2VwdCBhIGZldyB3aGljaCBhcmUgYWx3YXlzIHRyZWF0"
    "ZWQgc3BlY2lhbGx5IGJ5IGN1cnNlcyk6CglTQU1QTEU9IgECAwQFBgcLDA4PEBESExQVFhcYGRoi"
    "Cgk7Owplc2FjCgojIFRoaXMgc2NyaXB0IGlzIHNvdXJjZSdkIGZyb20gb3RoZXIgc2NyaXB0cywg"
    "YW5kIHVzZXMgdGhlIHBhcmFtZXRlciBsaXN0IGZyb20KIyB0aG9zZSBleHBsaWNpdGx5LiAgQnV0"
    "IHRoZXkgbWF5IHVzZSB0aGUgcGFyYW1ldGVyIGxpc3QgbGF0ZXIsIHRvIHNldCBvcHRpb25zCiMg"
    "c3BlY2lhbGx5IGZvciBkaWFsb2cuICBXb3JrIGFyb3VuZCB0aGUgY29uZmxpY3RpbmcgdXNlcyBi"
    "eSByZW1vdmluZyB0aGUKIyBwYXJhbWV0ZXIgd2hpY2ggd2UganVzdCB1c2VkIHRvIHNlbGVjdCBh"
    "IHNldCBvZiBkYXRhLgppZiB0ZXN0ICQjICE9IDAKdGhlbgoJc2hpZnQgMQpmaQo="
)

_GLIBC_TEST_GENCAT_SHIFT_JIS_B64 = (
    "IyEvYmluL3NoCiMgVGVzdCBlc2NhcGUgY2hhcmFjdGVyIGhhbmRsaW5nIGluIGdlbmNhdC4KIyBD"
    "b3B5cmlnaHQgKEMpIDIwMDAtMjAyNiBGcmVlIFNvZnR3YXJlIEZvdW5kYXRpb24sIEluYy4KIyBU"
    "aGlzIGZpbGUgaXMgcGFydCBvZiB0aGUgR05VIEMgTGlicmFyeS4KCiMgVGhlIEdOVSBDIExpYnJh"
    "cnkgaXMgZnJlZSBzb2Z0d2FyZTsgeW91IGNhbiByZWRpc3RyaWJ1dGUgaXQgYW5kL29yCiMgbW9k"
    "aWZ5IGl0IHVuZGVyIHRoZSB0ZXJtcyBvZiB0aGUgR05VIExlc3NlciBHZW5lcmFsIFB1YmxpYwoj"
    "IExpY2Vuc2UgYXMgcHVibGlzaGVkIGJ5IHRoZSBGcmVlIFNvZnR3YXJlIEZvdW5kYXRpb247IGVp"
    "dGhlcgojIHZlcnNpb24gMi4xIG9mIHRoZSBMaWNlbnNlLCBvciAoYXQgeW91ciBvcHRpb24pIGFu"
    "eSBsYXRlciB2ZXJzaW9uLgoKIyBUaGUgR05VIEMgTGlicmFyeSBpcyBkaXN0cmlidXRlZCBpbiB0"
    "aGUgaG9wZSB0aGF0IGl0IHdpbGwgYmUgdXNlZnVsLAojIGJ1dCBXSVRIT1VUIEFOWSBXQVJSQU5U"
    "WTsgd2l0aG91dCBldmVuIHRoZSBpbXBsaWVkIHdhcnJhbnR5IG9mCiMgTUVSQ0hBTlRBQklMSVRZ"
    "IG9yIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFLiAgU2VlIHRoZSBHTlUKIyBMZXNz"
    "ZXIgR2VuZXJhbCBQdWJsaWMgTGljZW5zZSBmb3IgbW9yZSBkZXRhaWxzLgoKIyBZb3Ugc2hvdWxk"
    "IGhhdmUgcmVjZWl2ZWQgYSBjb3B5IG9mIHRoZSBHTlUgTGVzc2VyIEdlbmVyYWwgUHVibGljCiMg"
    "TGljZW5zZSBhbG9uZyB3aXRoIHRoZSBHTlUgQyBMaWJyYXJ5OyBpZiBub3QsIHNlZQojIDxodHRw"
    "czovL3d3dy5nbnUub3JnL2xpY2Vuc2VzLz4uCgpzZXQgLWUKCmNvbW1vbl9vYmpwZng9JDEKdGVz"
    "dF9wcm9ncmFtX2NtZF9iZWZvcmVfZW52PSQyCnJ1bl9wcm9ncmFtX2Vudj0kMwp0ZXN0X3Byb2dy"
    "YW1fY21kX2FmdGVyX2Vudj0kNAoKIyBSdW4gdGhlIHRlc3QgcHJvZ3JhbS4KJHt0ZXN0X3Byb2dy"
    "YW1fY21kX2JlZm9yZV9lbnZ9IFwKICAke3J1bl9wcm9ncmFtX2Vudn0gXAogIE5MU1BBVEg9JHtj"
    "b21tb25fb2JqcGZ4fWNhdGdldHMvJU4uJWMuY2F0IExDX0FMTD1qYV9KUC5TSklTIFwKICAke3Rl"
    "c3RfcHJvZ3JhbV9jbWRfYWZ0ZXJfZW52fSBcCiAgICA+ICR7Y29tbW9uX29ianBmeH1jYXRnZXRz"
    "L3Rlc3QtZ2VuY2F0Lm91dAoKIyBDb21wYXJlIHdpdGggdGhlIGV4cGVjdGVkIHJlc3VsdC4KY21w"
    "IC0gJHtjb21tb25fb2JqcGZ4fWNhdGdldHMvdGVzdC1nZW5jYXQub3V0IDw8IkVPRiIKTENfTUVT"
    "U0FHRVMgPSBqYV9KUC5TSklTCnNhbXBsZTE6QUJDREVGOgpzYW1wbGUyOpP6lnuM6joKc2FtcGxl"
    "MzqXXJLolVw6CnNhbXBsZTQ6VEVTVAlUQUI6CnNhbXBsZTU6i0CUXAmPXI7tl946CmRvdWJsZSBz"
    "bGFzaFwKYW5vdGhlciBsaW5lCkVPRgpyZXM9JD8KCmNhdCA8PEVPRiB8CiNkZWZpbmUgQW5vdGhl"
    "clNldCAweDIJLyogKnN0YW5kYXJkIGlucHV0KjoxMyAqLwojZGVmaW5lIEFub3RoZXJGT08gMHgx"
    "CS8qICpzdGFuZGFyZCBpbnB1dCo6MTQgKi8KRU9GCmNtcCAke2NvbW1vbl9vYmpwZnh9Y2F0Z2V0"
    "cy90ZXN0LWdlbmNhdC5oIC0gfHwgcmVzPTEKCmV4aXQgJHJlcwo="
)


def _role_tokens(parsed, role: DomainRoleKind) -> set[str]:
    return {
        token.text
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
        if int(role_id) == int(role)
    }


def test_ksh_parser_keeps_domain_and_shell_graph_edges() -> None:
    from cppmega_mlx.data.shell_parsers import parse_ksh

    parsed = parse_ksh("typeset name=value\nprint $name | sed 's/x/y/' > out.txt\n")

    assert parsed.domain == DomainKind.KSH
    assert parsed.metadata["parser_adapter"] == "ksh"
    assert "typeset" in _role_tokens(parsed, DomainRoleKind.KEYWORD)
    assert "print" in _role_tokens(parsed, DomainRoleKind.COMMAND)
    assert "sed" in _role_tokens(parsed, DomainRoleKind.COMMAND)
    assert "out.txt" in _role_tokens(parsed, DomainRoleKind.PATH)
    assert any(
        edge[2] == int(DomainEdgeKind.SHELL_PIPE) for edge in parsed.edges
    )
    assert any(
        edge[2] == int(DomainEdgeKind.SHELL_REDIR_OUT) for edge in parsed.edges
    )

    from cppmega_mlx.data.agent_trajectory import parse_shell_action_domain

    assert parse_shell_action_domain("print ok", shell_kind="ksh").domain == DomainKind.KSH


def test_python_parser_uses_stdlib_ast_and_tokenize_for_typed_spans_and_edges() -> None:
    from cppmega_mlx.data.python_parsers import parse_python

    source = (
        '"""module docs"""\n'
        "# module comment\n"
        "def greet(name: str) -> str:\n"
        "    message = f\"Hello {name}\"\n"
        "    print(message)\n"
        "    return message\n"
    )
    parsed = parse_python(source)

    assert parsed.domain == DomainKind.PYTHON
    assert parsed.metadata["parser_adapter"] == "python-ast-tokenize"
    assert "def" in _role_tokens(parsed, DomainRoleKind.KEYWORD)
    assert "greet" in _role_tokens(parsed, DomainRoleKind.IDENTIFIER)
    assert "# module comment" in _role_tokens(parsed, DomainRoleKind.COMMENT)
    assert any(
        parsed.text[token.start : token.end] == '"""module docs"""'
        and role_id == int(DomainRoleKind.DOCSTRING)
        for token, role_id in zip(parsed.tokens, parsed.role_ids, strict=True)
    )
    assert set(parsed.confidence_ids) == {int(ParseConfidence.EXACT)}
    assert any(
        edge[2] == int(DomainEdgeKind.AST_PARENT) for edge in parsed.edges
    )
    assert any(edge[2] == int(DomainEdgeKind.CALL) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.DEF_USE) for edge in parsed.edges)

    enriched = parsed.to_enriched_document()
    assert enriched["domain_kind"] == int(DomainKind.PYTHON)
    assert len(enriched["domain_ids"]) == len(source)
    assert enriched["domain_edges"]


def test_python_parser_marks_syntax_errors_raw_without_losing_domain_identity() -> None:
    from cppmega_mlx.data.python_parsers import parse_python

    parsed = parse_python("def broken(:\n    pass\n")

    assert parsed.domain == DomainKind.PYTHON
    assert set(parsed.confidence_ids) == {int(ParseConfidence.RAW)}
    assert parsed.metadata["unsupported_syntax"] == "malformed_python_syntax"


def test_python_parser_maps_utf8_ast_columns_to_character_spans() -> None:
    from cppmega_mlx.data.python_parsers import parse_python

    source = "def café(value: str):\n    return value\n"
    parsed = parse_python(source)
    cafe_start = source.index("café")

    cafe_tokens = [
        token
        for token in parsed.tokens
        if token.text == "café"
    ]
    assert len(cafe_tokens) == 1
    assert cafe_tokens[0].start == cafe_start
    assert cafe_tokens[0].end == cafe_start + len("café")
    assert parsed.confidence_ids
    assert set(parsed.confidence_ids) == {int(ParseConfidence.EXACT)}


def test_domain_dispatch_and_discovery_cover_ksh_and_python(tmp_path: Path) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        discover_project_domain_files,
        parse_domain_document,
        resolve_domain_parser,
    )

    ksh_text = "#!/bin/ksh\nprint ok\n"
    python_text = "def answer():\n    return 42\n"
    ksh_path = tmp_path / "script.ksh"
    python_path = tmp_path / "module.py"
    shebang_path = tmp_path / "run"
    ksh_path.write_text(ksh_text)
    python_path.write_text(python_text)
    shebang_path.write_text("#!/usr/bin/env python3\nprint(42)\n")

    assert resolve_domain_parser(ksh_path, ksh_text).domain == DomainKind.KSH
    assert resolve_domain_parser(python_path, python_text).domain == DomainKind.PYTHON
    assert resolve_domain_parser(shebang_path, shebang_path.read_text()).domain == DomainKind.PYTHON
    assert parse_domain_document(python_path, python_text).domain == DomainKind.PYTHON

    discovered = {
        item.path.relative_to(tmp_path).as_posix(): item.domain
        for item in discover_project_domain_files(tmp_path)
    }
    assert discovered == {
        "module.py": DomainKind.PYTHON,
        "run": DomainKind.PYTHON,
        "script.ksh": DomainKind.KSH,
    }


def test_postgres_mule_internal_fixture_round_trips_byte_exactly(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        decode_domain_prefix,
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = (
        b"-- MULE \x92 internal encoding fixture\n"
        b"SELECT '\x81' AS byte_value;\n"
    )
    path = tmp_path / "src/test/mb/sql/mule_internal.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    decoded_prefix = decode_domain_prefix(encoded, path=path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=17))

    assert [item.path for item in discovered] == [path]
    assert decoded_prefix.encode("latin-1") == encoded
    assert {chunk.source_encoding for chunk in chunks} == {"mule-internal"}
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 17 for chunk in chunks)


def test_postgres_mule_internal_contract_requires_signature(tmp_path: Path) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        decode_domain_prefix,
        iter_domain_file_chunks,
    )

    encoded = b"-- unrelated \x92 fixture\nSELECT 1;\n"
    path = tmp_path / "src/test/mb/sql/mule_internal.sql"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    decoded_prefix = decode_domain_prefix(encoded, path=path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=17))

    assert decoded_prefix.encode("cp1252") == encoded
    assert {chunk.source_encoding for chunk in chunks} == {"windows-1252"}
    assert b"".join(chunk.text.encode("cp1252") for chunk in chunks) == encoded


@pytest.mark.parametrize(
    ("relative_path", "payload", "error"),
    [
        (
            "src/test/mb/sql/not_mule_internal.sql",
            b"-- near miss \x92\nSELECT '\x81';\n",
            "invalid UTF-8 or Windows-1252",
        ),
        (
            "src/test/mb/sql/mule_internal.sql",
            b"-- MULE \x92\nSELECT '\x81\x00';\n",
            "NUL byte",
        ),
    ],
)
def test_postgres_mule_internal_contract_fails_closed(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
    error: str,
) -> None:
    from cppmega_mlx.data.domain_ingestion import iter_domain_file_chunks

    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=error):
        list(iter_domain_file_chunks(path))


def test_freebsd_dialog_8bit_fixture_preserves_exact_upstream_bytes(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = base64.b64decode(_FREEBSD_DIALOG_TESTDATA_8BIT_B64)
    assert len(encoded) == 959
    assert hashlib.sha256(encoded).hexdigest() == (
        "8da95be352cc07a792179bb103aa6f7a7a073b59ba007a28b94fd8b30afb37dc"
    )
    path = tmp_path / "contrib/dialog/samples/testdata-8bit"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=96))

    assert [item.path for item in discovered] == [path]
    assert {chunk.source_encoding for chunk in chunks} == {"iso-8859-1"}
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 96 for chunk in chunks)


def test_glibc_shift_jis_expected_output_preserves_exact_upstream_bytes(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import (
        discover_project_domain_files,
        iter_domain_file_chunks,
    )

    encoded = base64.b64decode(_GLIBC_TEST_GENCAT_SHIFT_JIS_B64)
    assert len(encoded) == 1577
    assert hashlib.sha256(encoded).hexdigest() == (
        "88a7a81dc5c99fe901b1fe8966bdee605aea949b2dc20cee26156db55d4cdc4d"
    )
    path = tmp_path / "catgets/test-gencat.sh"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    discovered = discover_project_domain_files(tmp_path)
    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=128))

    assert [item.path for item in discovered] == [path]
    assert {chunk.source_encoding for chunk in chunks} == {
        "mixed-utf-8-shift-jis-byte-preserving"
    }
    assert b"".join(chunk.text.encode("latin-1") for chunk in chunks) == encoded
    assert all(chunk.byte_end - chunk.byte_start <= 128 for chunk in chunks)


def test_glibc_shift_jis_marker_does_not_authorize_bytes_outside_heredoc(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import iter_domain_file_chunks

    encoded = base64.b64decode(_GLIBC_TEST_GENCAT_SHIFT_JIS_B64).replace(
        b"set -e\n",
        b"set -e\x81\n",
        1,
    )
    path = tmp_path / "catgets/test-gencat.sh"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="invalid UTF-8 or Windows-1252"):
        list(iter_domain_file_chunks(path))


def test_nt5_japanese_localized_cmd_round_trips_shift_jis(
    tmp_path: Path,
) -> None:
    from cppmega_mlx.data.domain_ingestion import iter_domain_file_chunks

    text = "@echo off\r\nrem 日本語セットアップ\r\n"
    encoded = text.encode("shift_jis")
    path = (
        tmp_path
        / "nt5src/Source/XPSP1/NT/termsrv/admtools/appcmpt/jpn/msie4usr.cmd"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)

    chunks = list(iter_domain_file_chunks(path, max_chunk_bytes=24))

    assert "".join(chunk.text for chunk in chunks) == text
    assert {chunk.source_encoding for chunk in chunks} == {"shift-jis"}
    assert b"".join(chunk.text.encode("shift_jis") for chunk in chunks) == encoded


class _Encoding:
    def __init__(self, text: str) -> None:
        self.ids = [1000 + (index % 1000) for index in range(len(text))]
        self.offsets = [(index, index + 1) for index in range(len(text))]


class _CharOffsetBackend:
    @staticmethod
    def encode_batch(texts, add_special_tokens=False):
        del add_special_tokens
        return [_Encoding(text) for text in texts]


class _CharOffsetTokenizer:
    _tokenizer = _CharOffsetBackend()

    @staticmethod
    def get_bos_token_id() -> int:
        return 1


def _materialize_domain_row(domain: DomainKind) -> dict:
    text = "print(42)\n"
    return {
        "repo": "fixture/domain-contract",
        "filepath": "script.py",
        "text": text,
        "domain_kind": int(domain),
        "domain_ids": [int(domain)] * len(text),
        "domain_role_ids": [0] * len(text),
        "domain_entity_ids": [0] * len(text),
        "domain_scope_ids": [0] * len(text),
        "domain_source_doc_ids": [1] * len(text),
        "domain_confidence_ids": [int(ParseConfidence.EXACT)] * len(text),
    }


def test_materialization_inserts_new_delimiters_and_preserves_old_row_slots() -> None:
    new_row = materialize_tokenized_enriched_batch(
        [_materialize_domain_row(DomainKind.PYTHON)],
        _CharOffsetTokenizer(),
        num_threads=1,
    )[0]
    old_row = materialize_tokenized_enriched_batch(
        [_materialize_domain_row(DomainKind.TCSH)],
        _CharOffsetTokenizer(),
        num_threads=1,
    )[0]

    assert new_row["token_ids"][1] == 247
    assert new_row["token_ids"][-1] == 248
    assert old_row["token_ids"][1] == 207
    assert old_row["token_ids"][-1] == 208
    assert old_row["token_domain_ids"][1] == int(DomainKind.TCSH)


def test_index_project_materializes_ksh_and_python_as_typed_domain_docs(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    ksh_path = tmp_path / "run.ksh"
    ksh_path.write_text("#!/bin/ksh\nprint ok\n")

    assert index_project.find_shell_files(str(tmp_path)) == [
        (str(ksh_path), "ksh")
    ]

    doc = index_project.build_build_doc(
        "module.py",
        "def answer():\n    return 42\n",
        "python",
        project_id="fixture/domain-contract",
    )
    assert doc["domain_kind"] == int(DomainKind.PYTHON)
    assert doc["doc_type"] == "code"
    assert doc["domain_parse_info"]["parser"] == "python-ast-tokenize"
    assert doc["language_info"]["primary_language"] == "python"
