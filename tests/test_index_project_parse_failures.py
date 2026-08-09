from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest


_GCC_LIMITS_CASELABELS_SOURCE = (
    b"#define LIM1(x) x##0: x##1: x##2: x##3: x##4: x##5: x##6: x##7: "
    b"x##8: x##9: \n"
    b"#define LIM2(x) LIM1(x##0) LIM1(x##1) LIM1(x##2) LIM1(x##3) LIM1(x##4) "
    b"\\\n\t\tLIM1(x##5) LIM1(x##6) LIM1(x##7) LIM1(x##8) LIM1(x##9)\n"
    b"#define LIM3(x) LIM2(x##0) LIM2(x##1) LIM2(x##2) LIM2(x##3) LIM2(x##4) "
    b"\\\n\t\tLIM2(x##5) LIM2(x##6) LIM2(x##7) LIM2(x##8) LIM2(x##9)\n"
    b"#define LIM4(x) LIM3(x##0) LIM3(x##1) LIM3(x##2) LIM3(x##3) LIM3(x##4) "
    b"\\\n\t\tLIM3(x##5) LIM3(x##6) LIM3(x##7) LIM3(x##8) LIM3(x##9)\n"
    b"#define LIM5(x) LIM4(x##0) LIM4(x##1) LIM4(x##2) LIM4(x##3) LIM4(x##4) "
    b"\\\n\t\tLIM4(x##5) LIM4(x##6) LIM4(x##7) LIM4(x##8) LIM4(x##9)\n"
    b"#define LIM6(x) LIM5(x##0) LIM5(x##1) LIM5(x##2) LIM5(x##3) LIM5(x##4) "
    b"\\\n\t\tLIM5(x##5) LIM5(x##6) LIM5(x##7) LIM5(x##8) LIM5(x##9)\n"
    b"#define LIM7(x) LIM6(x##0) LIM6(x##1) LIM6(x##2) LIM6(x##3) LIM6(x##4) "
    b"\\\n\t\tLIM6(x##5) LIM6(x##6) LIM6(x##7) LIM6(x##8) LIM6(x##9)\n"
    b"\nvoid q19_func (long i)\n{\n  switch (i) {\n    LIM5 (case 1)\n"
    b"      break;\n  }\n}\n"
)


def test_index_jsonl_output_is_gzip_streamable(tmp_path: Path) -> None:
    from tools.clang_indexer.index_project import _open_jsonl_output

    output = tmp_path / "source.enriched.jsonl.gz"
    with _open_jsonl_output(output, append=False, compressed=True) as stream:
        stream.write('{"doc":1}\n')
    with _open_jsonl_output(output, append=True, compressed=True) as stream:
        stream.write('{"doc":2}\n')

    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert stream.readlines() == ['{"doc":1}\n', '{"doc":2}\n']


def _load_indexer():
    try:
        from tools.clang_indexer import index_project

        index_project._configure_libclang()
    except Exception as exc:  # pragma: no cover - environment without libclang
        pytest.skip(f"libclang unavailable: {exc}")
    return index_project


def test_parse_file_batch_fails_loud_with_file_and_cause(tmp_path: Path) -> None:
    index_project = _load_indexer()
    source = tmp_path / "broken.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as raised:
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-x", "definitely-not-a-language"],
                str(tmp_path),
                "fixture/parse-failure",
            )
        )

    message = str(raised.value)
    assert str(source) in message
    assert "TranslationUnitLoadError" in message
    assert "libclang parse failed" in message


def test_sequential_project_parse_fails_loud_instead_of_publishing(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "broken.cpp"
    source.write_text("int main() { return 0; }\n", encoding="utf-8")
    (tmp_path / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "file": str(source),
                    "arguments": [
                        "clang++",
                        "-x",
                        "definitely-not-a-language",
                        str(source),
                    ],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as raised:
        index_project.process_project(
            str(tmp_path),
            enriched=True,
            project_id="fixture/parse-failure",
        )

    message = str(raised.value)
    assert str(source) in message
    assert "TranslationUnitLoadError" in message
    assert "libclang parse failed" in message


def test_sane_translation_unit_load_error_uses_bound_lossless_lexical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    source = tmp_path / "native_crash.cpp"
    raw_source = (
        "// libclang native-crash regression: π\n"
        "int preserved_answer() { return 42; }\n"
    ).encode()
    source.write_bytes(raw_source)

    class TranslationUnitLoadError(Exception):
        pass

    def crash_translation_unit(*_args, **_kwargs):
        raise TranslationUnitLoadError("native parser crashed")

    monkeypatch.setattr(
        index_project,
        "_load_translation_unit",
        crash_translation_unit,
    )
    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/native-clang-crash",
        )
    )

    assert parsed_count == 1
    assert payload["functions"] == []
    assert payload["typedefs"] == []
    assert payload["lexical_fallback_files"] == ["native_crash.cpp"]
    assert payload["parse_recovery_records"] == [
        {
            "relative_path": "native_crash.cpp",
            "trigger": "translation_unit_load_error",
            "status": "lexical_fallback",
            "fallback_mode": "lossless_cpp_lexical_v1",
            "fallback_reason": "translation_unit_load_error",
            "compile_args_status": "sane",
            "compile_arg_count": 3,
            "compile_args_sha256": payload["parse_recovery_records"][0][
                "compile_args_sha256"
            ],
            "source_size_bytes": len(raw_source),
            "source_char_count": len(raw_source.decode("utf-8")),
            "source_sha256": hashlib.sha256(raw_source).hexdigest(),
            "source_encoding": "utf-8",
        }
    ]
    summary = index_project._parse_recovery_summary(
        payload["parse_recovery_records"]
    )
    assert summary["status"] == "complete"
    assert summary["recovered_file_count"] == 1
    assert summary["semantic_recovered_file_count"] == 0
    assert summary["lexical_fallback_file_count"] == 1
    assert summary["lexical_fallback_source_bytes"] == len(raw_source)
    assert summary["unresolved_file_count"] == 0


def test_gcc_case_label_ast_recursion_uses_bound_lossless_lexical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    assert len(_GCC_LIMITS_CASELABELS_SOURCE) == 935
    assert hashlib.sha256(_GCC_LIMITS_CASELABELS_SOURCE).hexdigest() == (
        "cd4aaab81ac06ab6265f81567297015b364c8c6e026e757fac87cc38faae868e"
    )
    source = tmp_path / "limits-caselabels.c"
    source.write_bytes(_GCC_LIMITS_CASELABELS_SOURCE)

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    visitor_namespace: dict[str, object] = {}
    exec(
        compile(
            "def _visit():\n"
            "    raise RecursionError('maximum recursion depth exceeded')\n",
            index_project.__file__,
            "exec",
        ),
        visitor_namespace,
    )
    visitor = visitor_namespace["_visit"]
    assert callable(visitor)
    monkeypatch.setattr(
        index_project,
        "parse_translation_unit",
        lambda *_args, **_kwargs: visitor(),
    )
    compile_args = ["-std=c11", "-fsyntax-only", "-Wno-everything"]

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            compile_args,
            str(tmp_path),
            "gcc-mirror/gcc",
        )
    )

    assert parsed_count == 1
    assert payload["functions"] == []
    assert payload["typedefs"] == []
    assert payload["lexical_fallback_files"] == ["limits-caselabels.c"]
    assert payload["parse_recovery_records"] == [
        {
            "relative_path": "limits-caselabels.c",
            "trigger": "ast_recursion_error",
            "status": "lexical_fallback",
            "fallback_mode": "lossless_cpp_lexical_v1",
            "fallback_reason": "ast_recursion_error",
            "compile_args_status": "sane",
            "compile_arg_count": 5,
            "compile_args_sha256": payload["parse_recovery_records"][0][
                "compile_args_sha256"
            ],
            "source_size_bytes": len(_GCC_LIMITS_CASELABELS_SOURCE),
            "source_char_count": len(_GCC_LIMITS_CASELABELS_SOURCE),
            "source_sha256": hashlib.sha256(
                _GCC_LIMITS_CASELABELS_SOURCE
            ).hexdigest(),
            "source_encoding": "utf-8",
        }
    ]

    documents = index_project.emit_cpp_lexical_fallback_documents(
        payload["lexical_fallback_files"],
        index=index_project.ProjectIndex(),
        project_dir=str(tmp_path),
        project_id="gcc-mirror/gcc",
        compile_db={},
        default_args=compile_args,
        default_build_info=None,
        parse_recovery_records=payload["parse_recovery_records"],
        enriched=True,
    )

    assert len(documents) == 1
    assert documents[0]["text"].encode("utf-8") == _GCC_LIMITS_CASELABELS_SOURCE
    assert documents[0]["cpp_parse_fallback"]["reason"] == (
        "ast_recursion_error"
    )
    assert documents[0]["domain_parse_info"]["fallback_reason"] == (
        "ast_recursion_error"
    )


def test_unrelated_recursion_error_still_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    source = tmp_path / "unrelated-recursion.cpp"
    source.write_text("int value = 1;\n", encoding="utf-8")

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    monkeypatch.setattr(
        index_project,
        "parse_translation_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RecursionError("unrelated recursion bug")
        ),
    )

    with pytest.raises(RuntimeError, match="RecursionError"):
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
                str(tmp_path),
                "fixture/unrelated-recursion",
            )
        )


def test_non_translation_unit_error_still_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.clang_indexer import index_project

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    source = tmp_path / "unexpected_failure.cpp"
    source.write_text("int preserved = 1;\n")
    monkeypatch.setattr(
        index_project,
        "_load_translation_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("not a libclang TU load error")
        ),
    )

    with pytest.raises(RuntimeError, match="ValueError"):
        index_project._parse_file_batch(
            (
                [str(source)],
                {},
                ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
                str(tmp_path),
                "fixture/non-tu-error",
            )
        )


def test_sane_translation_unit_load_error_emits_heuristic_source_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cppmega_mlx.data.domain_schema import ParseConfidence
    from tools.clang_indexer import index_project

    class FakeIndex:
        @staticmethod
        def create() -> object:
            return object()

    monkeypatch.setattr(index_project, "_configure_libclang", lambda: None)
    monkeypatch.setattr(index_project, "Index", FakeIndex)
    source = tmp_path / "driver.cpp"
    raw_source = (
        b"// Useful driver implementation must not be quarantined.\n"
        b"int driver_value(int input) { return input + 7; }\n"
    )
    source.write_bytes(raw_source)

    class TranslationUnitLoadError(Exception):
        pass

    monkeypatch.setattr(
        index_project,
        "_load_translation_unit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TranslationUnitLoadError("native parser crashed")
        ),
    )
    documents = index_project.process_project(
        str(tmp_path),
        enriched=True,
        project_id="fixture/lossless-lexical-fallback",
    )

    assert len(documents) == 1
    document = documents[0]
    assert document["doc_type"] == "code"
    assert document["filepath"] == "driver.cpp"
    assert document["text"].encode("utf-8") == raw_source
    assert document["cpp_parse_fallback"] == {
        "schema": "cppmega.cpp_parse_fallback_v1",
        "mode": "lossless_cpp_lexical_v1",
        "reason": "translation_unit_load_error",
        "compile_args_status": "sane",
        "source_sha256": hashlib.sha256(raw_source).hexdigest(),
        "source_encoding": "utf-8",
        "source_span": document["source_span"],
    }
    assert document["source_span"] == {
        "chunk_index": 0,
        "byte_start": 0,
        "byte_end": len(raw_source),
        "char_start": 0,
        "char_end": len(raw_source.decode("utf-8")),
        "source_size_bytes": len(raw_source),
        "chunk_limit_bytes": index_project.CPP_LEXICAL_FALLBACK_CHUNK_BYTES,
        "split_reason": "eof",
        "source_encoding": "utf-8",
    }
    text_len = len(document["text"])
    assert document["domain_confidence_ids"] == [
        int(ParseConfidence.HEURISTIC)
    ] * text_len
    for field in (
        "call_edges",
        "type_edges",
        "build_edges",
        "shell_edges",
        "diagnostic_edges",
    ):
        assert document[field] == []
    for field in (
        "structure_ids",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "symbol_ids",
        "call_targets",
        "type_refs",
        "def_use",
        "domain_scope_ids",
    ):
        assert document[field] == [0] * text_len
    for field in (
        "domain_ids",
        "domain_role_ids",
        "domain_entity_ids",
        "domain_source_doc_ids",
        "domain_source_identity_ids",
    ):
        assert len(document[field]) == text_len


def test_cpp_lexical_fallback_chunks_preserve_utf8_bytes_and_spans(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    source = tmp_path / "utf8.cpp"
    raw_source = (
        "// αβγδεζηθ\n"
        "int first = 1;\n"
        "int second = 2;\n"
    ).encode()
    source.write_bytes(raw_source)

    chunks = list(
        index_project._iter_cpp_lexical_fallback_chunks(
            str(source),
            max_chunk_bytes=17,
        )
    )

    assert b"".join(text.encode("utf-8") for text, _span in chunks) == raw_source
    assert [span["chunk_index"] for _text, span in chunks] == list(
        range(len(chunks))
    )
    assert [span["byte_start"] for _text, span in chunks] == [
        0,
        *[span["byte_end"] for _text, span in chunks[:-1]],
    ]
    assert chunks[-1][1]["byte_end"] == len(raw_source)


def test_cpp_lexical_fallback_rejects_project_root_escape(tmp_path: Path) -> None:
    from tools.clang_indexer import index_project

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    outside_source = tmp_path / "outside.cpp"
    outside_source.write_text("int outside = 1;\n")

    class TranslationUnitLoadError(Exception):
        pass

    with pytest.raises(RuntimeError, match="escapes the project root"):
        index_project._record_cpp_lexical_fallback(
            str(outside_source),
            ["-x", "c++", "-std=c++17"],
            str(project_dir),
            TranslationUnitLoadError("native parser crashed"),
            [],
        )


def test_cpp_lexical_fallback_rechecks_compile_args_digest(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    source = tmp_path / "changed_context.cpp"
    source.write_text("int changed_context = 1;\n")

    class TranslationUnitLoadError(Exception):
        pass

    parse_args = index_project._resolve_file_args(
        str(source),
        {},
        ["-std=c++17"],
    )
    records: list[dict[str, object]] = [
        {
            "relative_path": "changed_context.cpp",
            "trigger": "missing_include_diagnostic",
            "status": "unresolved",
        }
    ]
    relative_path = index_project._record_cpp_lexical_fallback(
        str(source),
        parse_args,
        str(tmp_path),
        TranslationUnitLoadError("native parser crashed"),
        records,
    )

    assert relative_path == "changed_context.cpp"
    assert records[0]["trigger"] == "translation_unit_load_error"
    with pytest.raises(RuntimeError, match="compile args changed"):
        index_project.emit_cpp_lexical_fallback_documents(
            [relative_path],
            index=index_project.ProjectIndex(),
            project_dir=str(tmp_path),
            project_id="fixture/compile-args-drift",
            compile_db=None,
            default_args=["-std=c++20"],
            default_build_info=None,
            parse_recovery_records=records,
            enriched=True,
        )


def test_cp1252_source_round_trips_without_clang_token_utf8_decode(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "legacy.cpp"
    raw_type = b"struct /* compiler\x92s type */ LegacyType { int value; };\r\n"
    raw_function = (
        b"int /* compiler\x92s invalid candidate */ "
        b"legacy_answer() { return 42; }\r\n"
    )
    raw_source = raw_type + raw_function
    source.write_bytes(raw_source)

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/cp1252-source",
        )
    )

    assert parsed_count == 1
    function = next(
        item for item in payload["functions"]
        if item["name"] == "legacy_answer"
    )
    type_definition = next(
        item for item in payload["typedefs"]
        if item["name"] == "LegacyType"
    )
    assert function["text"] == raw_function[:-2].decode("cp1252")
    assert "compiler’s invalid candidate" in function["text"]
    assert type_definition["text"] == raw_type[:-3].decode("cp1252")
    assert "compiler’s type" in type_definition["text"]
    for field in (
        "ast_depth",
        "sibling_index",
        "ast_node_type",
        "semantic_symbol_ids",
        "semantic_call_targets",
        "semantic_type_refs",
        "semantic_def_use",
    ):
        assert len(function[field]) == len(function["text"])


def test_parse_batch_recovers_nested_legacy_include_context(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    source_dir = tmp_path / "legacy" / "src"
    include_dir = tmp_path / "legacy" / "vendor" / "include"
    nested_include_dir = tmp_path / "legacy" / "vendor" / "inc.next"
    decoy_include_dir = tmp_path / "other" / "include"
    source_dir.mkdir(parents=True)
    include_dir.mkdir(parents=True)
    nested_include_dir.mkdir(parents=True)
    decoy_include_dir.mkdir(parents=True)
    source = source_dir / "main.cpp"
    source.write_text(
        "int legacy_answer() { return LEGACY_ANSWER; }\n",
        encoding="utf-8",
    )
    (include_dir / "legacy_prelude.h").write_text(
        '#include "legacy_value.h"\n',
        encoding="utf-8",
    )
    (nested_include_dir / "legacy_value.h").write_text(
        "#define LEGACY_ANSWER 42\n",
        encoding="utf-8",
    )
    (decoy_include_dir / "legacy_prelude.h").write_text(
        "#define LEGACY_ANSWER 13\n",
        encoding="utf-8",
    )

    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-include", "legacy_prelude.h"],
            str(tmp_path),
            "fixture/nested-include-recovery",
        )
    )

    assert parsed_count == 1
    assert [item["name"] for item in payload["functions"]] == [
        "legacy_answer"
    ]
    assert payload["parse_recovery_records"] == [
        {
            "relative_path": "legacy/src/main.cpp",
            "trigger": "missing_include_diagnostic",
            "added_include_dir_examples": [
                "legacy/vendor/include",
                "legacy/vendor/inc.next",
            ],
            "added_include_dir_count": 2,
            "added_include_dirs_sha256": (
                "5a4a956e5ff09b51e6b2a395d66cb09a9407c0414f8b6b8c563"
                "f0433660d1704"
            ),
            "added_include_dir_examples_truncated": False,
            "requested_include_name_count": 2,
            "requested_include_names_sha256": (
                "9a69b537257773714a90564239bc0d5722c84333c6e26c2ac851"
                "8af299ac3241"
            ),
            "requested_include_name_examples": [
                "legacy_prelude.h",
                "legacy_value.h",
            ],
            "requested_include_name_examples_truncated": False,
            "unresolved_include_name_count": 0,
            "retry_round_count": 2,
            "initial_missing_include_count": 1,
            "status": "recovered",
            "retry_missing_include_count": 0,
        }
    ]


def test_source_with_nul_byte_is_rejected() -> None:
    index_project = _load_indexer()

    with pytest.raises(ValueError, match="source contains NUL byte"):
        index_project._decode_source_bytes(b"int value = 0;\0\n", "binary.cpp")


@pytest.mark.parametrize("quoted_standard", ['"23"', "23"])
def test_cmake_quoted_cxx_standard_drives_truthful_parser_dialect(
    tmp_path: Path,
    quoted_standard: str,
) -> None:
    from cppmega_mlx.data.nanochat_pipeline.build_context import (
        detect_build_context,
    )

    index_project = _load_indexer()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(quoted_standard LANGUAGES CXX)\n"
        f"set(CMAKE_CXX_STANDARD {quoted_standard} CACHE INTERNAL \"\")\n",
        encoding="utf-8",
    )
    source = tmp_path / "quoted_standard.cpp"
    source.write_text(
        "constexpr int quoted_standard(bool value) {\n"
        "    if consteval { return 23; }\n"
        "    return value ? 1 : 0;\n"
        "}\n",
        encoding="utf-8",
    )

    platform_info, detected_args, compile_index = detect_build_context(
        str(tmp_path)
    )

    assert compile_index is None
    assert platform_info["build_system"] == "cmake"
    assert platform_info["standard"] == "c++23"
    assert "-std=c++23" in detected_args
    file_args = index_project._resolve_file_args(
        str(source),
        {},
        index_project.get_default_compile_args(str(tmp_path)),
    )
    assert "-std=c++23" in file_args
    translation_unit = index_project._load_translation_unit(
        str(source),
        index_project.Index.create(),
        file_args,
    )
    assert not [
        diagnostic
        for diagnostic in translation_unit.diagnostics
        if int(diagnostic.severity) >= 3
    ]


@pytest.mark.parametrize(
    ("bom", "disk_encoding"),
    (
        (b"\xff\xfe", "utf-16-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\x00\x00\xfe\xff", "utf-32-be"),
    ),
)
def test_bom_marked_wide_cpp_is_transcoded_losslessly_for_libclang(
    tmp_path: Path,
    bom: bytes,
    disk_encoding: str,
) -> None:
    index_project = _load_indexer()
    source = tmp_path / "wide.cpp"
    text = (
        "// BOM-marked source\r\n"
        "int wide_answer(int value) { return value + 42; }\r\n"
    )
    source.write_bytes(bom + text.encode(disk_encoding))

    decoded, detected_encoding = index_project._decode_source_bytes(
        source.read_bytes(),
        str(source),
    )
    parser_text, parser_bytes, parser_encoding = (
        index_project._read_source_file(str(source))
    )
    payload, parsed_count = index_project._parse_file_batch(
        (
            [str(source)],
            {},
            ["-std=c++17", "-fsyntax-only", "-Wno-everything"],
            str(tmp_path),
            "fixture/wide-source",
        )
    )

    assert detected_encoding == disk_encoding
    assert decoded.encode(disk_encoding) == source.read_bytes()
    assert parser_encoding == "utf-8"
    assert parser_bytes == parser_text.encode("utf-8")
    assert parsed_count == 1
    assert "wide_answer" in {
        function["name"] for function in payload["functions"]
    }


def test_malformed_bom_marked_wide_source_fails_closed() -> None:
    index_project = _load_indexer()

    with pytest.raises(UnicodeDecodeError):
        index_project._decode_source_bytes(
            b"\xff\xfe\x00",
            "malformed-wide.cpp",
        )


def test_textual_nul_inside_comment_or_literal_round_trips() -> None:
    index_project = _load_indexer()
    source = (
        b'char normal[] = "value\\0";\n'
        b'char embedded[] = "value\x00";\n'
        b'char raw[] = R"tag(value\x00)tag";\n'
        b"/* glyph \x81 '\x00' */\n"
    )

    text, encoding = index_project._decode_source_bytes(source, "fixture.cpp")

    assert encoding == "latin-1"
    assert text.encode(encoding) == source


def test_header_macro_emission_preserves_mixed_legacy_bytes(
    tmp_path: Path,
) -> None:
    from tools.clang_indexer import index_project

    header = tmp_path / "mixed.h"
    raw = b'#define MIXED "\x8d"\n'
    header.write_bytes(raw)
    docs: list[dict] = []

    stats = index_project.emit_header_documents(
        index=index_project.ProjectIndex(),
        header_files=[str(header)],
        project_dir=str(tmp_path),
        project_id="fixture/mixed-header",
        compile_db=None,
        default_args=[],
        default_build_info=None,
        max_tokens=4096,
        enriched=True,
        chunk_claims=None,
        emit_doc=docs.append,
    )

    assert stats["header_macro"] == 1
    assert len(docs) == 1
    assert docs[0]["text"].encode("latin-1") == raw
    assert "\ufffd" not in docs[0]["text"]


def test_gnu_c_standard_header_keeps_consistent_language_family(
    tmp_path: Path,
) -> None:
    index_project = _load_indexer()
    header = tmp_path / "KeychainSyncAccountUpdater.h"
    header.write_text(
        "#import <UAUPlugin/UAUSession.h>\n\n"
        "@interface KeychainSyncAccountUpdater : NSObject "
        "<UserAccountUpdaterProtocol>\n\n"
        "@end\n",
        encoding="utf-8",
    )

    adapted = index_project._adapt_args_for_file(
        ["-std=gnu2x", "-fblocks", "-fsyntax-only", "-Wno-everything"],
        str(header),
    )

    assert adapted[:3] == ["-x", "c-header", "-std=gnu2x"]
    assert index_project._is_sane_compile_args(adapted)
    translation_unit = index_project._load_translation_unit(
        str(header),
        index_project.Index.create(),
        adapted,
    )
    assert translation_unit.spelling == str(header)


@pytest.mark.parametrize(
    ("standard_args", "expected_language", "expected_standard"),
    [
        (["--std=c11", "--std=gnu2x"], "c-header", "--std=gnu2x"),
        (["-cl-std=CL1.2", "-cl-std=CL3.0"], "cl", "-cl-std=CL3.0"),
    ],
)
def test_header_adaptation_keeps_only_last_standard_alias(
    tmp_path: Path,
    standard_args: list[str],
    expected_language: str,
    expected_standard: str,
) -> None:
    index_project = _load_indexer()
    header = tmp_path / "dialect.h"
    header.write_text("int dialect_fixture;\n", encoding="utf-8")

    adapted = index_project._adapt_args_for_file(
        [*standard_args, "-fsyntax-only", "-Wno-everything"],
        str(header),
    )

    standard_flags = [
        arg
        for arg in adapted
        if arg.startswith(("-std=", "--std=", "-cl-std="))
    ]
    assert adapted[:2] == ["-x", expected_language]
    assert standard_flags == [expected_standard]
    assert index_project._is_sane_compile_args(adapted)
