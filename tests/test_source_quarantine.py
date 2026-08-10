from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from tools.clang_indexer import index_project as ip
from tools.clang_indexer.source_quarantine import (
    LEGACY_MANIFEST_SCHEMA,
    MANIFEST_SCHEMA,
    ProjectSourceQuarantine,
    RECEIPT_SCHEMA,
    SourceQuarantineError,
)


PROJECT_ID = "fixture/source-quarantine"
RELATIVE_XML = "sdk/license.cc"
RELATIVE_CRASH_FIXTURE = "tools/clang/test/Parser/crash-report.c"
RELATIVE_INDEX_CRASH_FIXTURE = "tools/clang/test/Index/crash-recovery.c"
RELATIVE_INDEX_REMAP_CRASH_FIXTURE = (
    "tools/clang/test/Index/Inputs/crash-recovery-code-complete-remap.c"
)
RELATIVE_NUL_DIAGNOSTIC_FIXTURE = "clang/test/Misc/diag-null-bytes-in-line.cpp"
RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE = "clang/test/Lexer/newline-nul.c"
RELATIVE_CERTIFICATE_PAIR = "vectors/certpairs/reverseCertificatePair.cp"
CERTIFICATE_PAIR_PREFIX = "vectors/certpairs/"
RELATIVE_GENERATED_BLOB = "ports_module/example_build/module_code.c"
RELATIVE_EXECUTABLE_ARCHIVE = "bin/self-executing-tool"
RELATIVE_NUL_FF_BLOB = "unknown_version_2/Source/drivers/spb/spbcx/sys/driver.h"
RELATIVE_WINDOWS_NUL_FF_BLOB = (
    "windows_10_shared_source_kit/unknown_version_2/Source/drivers/spb/spbcx/sys/driver.h"
)
RELATIVE_TRUNCATED_UTF32BE_BOM = "Tests/RunCMake/Syntax/Broken-BOM-UTF-32-BE.cmake"
RELATIVE_BIG5_SHELL_HEREDOC = (
    "external/gpl2/gettext/dist/gettext-tools/tests/msgconv-1"
)


def _xml_bytes() -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-16"?>\r\n'
        "<license><name>not C++</name></license>\r\n"
    ).encode("utf-16")


def _clang_crash_fixture_bytes() -> bytes:
    return (
        b"// RUN: not --crash %clang_cc1 %s 2>&1 | FileCheck %s\n"
        b"// REQUIRES: crash-recovery\n"
        b"\n"
        b"// FIXME: CHECKs might be incompatible to win32.\n"
        b"// Stack traces also require back traces.\n"
        b"// REQUIRES: shell, backtrace\n"
        b"\n"
        b"#prag\\\n"
        b"ma clang __debug crash\n"
        b"\n"
        b"// CHECK: prag\\\n"
        b"// CHECK-NEXT: ma\n"
        b"\n"
    )


def _clang_index_crash_fixture_bytes() -> bytes:
    return (
        b"// RUN: not c-index-test -test-load-source all %s 2> %t.err\n"
        b"// RUN: FileCheck < %t.err -check-prefix=CHECK-LOAD-SOURCE-CRASH %s\n"
        b"// CHECK-LOAD-SOURCE-CRASH: Unable to load translation unit\n"
        b"// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 not --crash "
        b"c-index-test -test-load-source all %s\n"
        b"//\n"
        b"// REQUIRES: crash-recovery\n"
        b"\n"
        b"#pragma clang __debug crash\n"
    )


def _clang_index_remap_crash_fixture_bytes() -> bytes:
    return (
        b"// RUN: echo env CINDEXTEST_EDITING=1 \\\n"
        b"// RUN:   not c-index-test -test-load-source-reparse 1 local \\\n"
        b'// RUN:   -remap-file="%s,%S/Inputs/crash-recovery-code-complete-remap.c" \\\n'
        b"// RUN:   %s 2> %t.err\n"
        b"// RUN: FileCheck < %t.err -check-prefix=CHECK-CODE-COMPLETE-CRASH %s\n"
        b"// CHECK-CODE-COMPLETE-CRASH: Unable to reparse translation unit\n"
        b"\n"
        b"#warning parsing original file\n"
        b"\n"
        b"#pragma clang __debug crash\n"
    )


def _clang_embedded_nul_diagnostic_bytes() -> bytes:
    return (
        b"// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | "
        b"FileCheck -strict-whitespace %s\n"
        b"\n"
        b"int x[sizeof\0int];\n"
        b"// CHECK: warning: null character ignored\n"
        b"// CHECK-NEXT: int x[sizeof<U+0000>int];\n"
        b"// CHECK-NEXT:             ^\n"
        b"\n"
        b"// CHECK: error: expected parentheses around type name in "
        b"sizeof expression\n"
        b"// CHECK-NEXT: int x[sizeof<U+0000>int];\n"
        b"// CHECK-NEXT:             ^\n"
        b"// CHECK-NEXT:             (          )\n"
    )


def _clang_newline_nul_diagnostic_bytes() -> bytes:
    return (
        b"// RUN: %clang_cc1 -E %s -verify\n"
        b"\n"
        b"// We used to crash if a line continuation was followed by a nul byte "
        b"within a\n"
        b"// preprocessing directive.\n"
        b"# if 1 \\\n"
        b"\0#if something_else // expected-warning {{null character ignored}} "
        b"expected-error {{not a valid binary operator}}\n"
        b"#error error\n"
        b"#endif\n"
        b"#endif // expected-error {{#endif without #if}}\n"
    )


def _der(tag: int, payload: bytes) -> bytes:
    if len(payload) < 0x80:
        length = bytes([len(payload)])
    else:
        encoded = len(payload).to_bytes((len(payload).bit_length() + 7) // 8, "big")
        length = bytes([0x80 | len(encoded)]) + encoded
    return bytes([tag]) + length + payload


def _certificate_pair_bytes(*, wrapper_tag: int = 0xA1) -> bytes:
    certificate = _der(
        0x30,
        _der(0x30, b"\x02\x01\x01")
        + _der(0x30, b"\x06\x03\x2a\x03\x04")
        + _der(0x03, b"\x00\x01"),
    )
    return _der(0x30, _der(wrapper_tag, certificate))


def _utf16le_generated_c_array_bytes(*, byte_count: int = 1024) -> bytes:
    byte_literals = ", ".join(
        f"0x{value % 256:02X}" for value in range(byte_count)
    )
    generated = (
        "/* \r\n"
        "   Input ELF file: sample_threadx_module.axf\r\n"
        "   Output C Array file: module_code.c\r\n"
        "*/\r\n\r\n"
        "__align(4096) unsigned char  module_code[] = {\r\n"
        "/* Address  Contents */\r\n"
        f"/* 0x00000000 */ {byte_literals}}};\r\n"
    ).encode("utf-16le")
    return b"\xff\xfe" + generated


def _self_executing_zip_bytes() -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("payload.txt", "exact fixture payload\n")
    return b'#!/bin/sh\nexec java -jar "$0" "$@"\nexit 1\n' + archive_buffer.getvalue()


def _truncated_utf32be_bom_bytes() -> bytes:
    return b"\x00\x00\xfe"


def _big5_shell_heredoc_bytes() -> bytes:
    big5_translation = bytes.fromhex(
        "a6b9a55cafe0bbddad6eabeaa66eabfca977a8e2add3bfe9a44ac0c9"
    )
    utf8_translation = (
        "此功能需要恰好指定兩個輸入檔"
    ).encode("utf-8")
    po_body = (
        b"# Chinese translation for GNU gettext messages.\n"
        b"#\n"
        b'msgid ""\n'
        b'msgstr ""\n'
        b'"MIME-Version: 1.0\\n"\n'
        b'"Content-Type: text/plain; charset=big5\\n"\n'
        b'"Content-Transfer-Encoding: 8bit\\n"\n\n'
        b"#: src/msgcmp.c:155 src/msgmerge.c:273\n"
        b'msgid "exactly 2 input files required"\n'
        b'msgstr "'
        + big5_translation
        + b'"\n'
    )
    ok_body = (
        b"# Chinese translation for GNU gettext messages.\n"
        b"#\n"
        b'msgid ""\n'
        b'msgstr ""\n'
        b'"MIME-Version: 1.0\\n"\n'
        b'"Content-Type: text/plain; charset=UTF-8\\n"\n'
        b'"Content-Transfer-Encoding: 8bit\\n"\n\n'
        b"#: src/msgcmp.c:155 src/msgmerge.c:273\n"
        b'msgid "exactly 2 input files required"\n'
        b'msgstr "'
        + utf8_translation
        + b'"\n'
    )
    return (
        b"#! /bin/sh\n\n"
        b"# Test conversion from BIG5 to UTF-8.\n\n"
        b'tmpfiles=""\n'
        b"trap 'rm -fr $tmpfiles' 1 2 3 15\n\n"
        b'tmpfiles="$tmpfiles mco-test1.po"\n'
        b"cat <<\\EOF > mco-test1.po\n"
        + po_body
        + b"EOF\n\n"
        + b'tmpfiles="$tmpfiles mco-test1.out"\n'
        + b": ${MSGCONV=msgconv}\n"
        + b"${MSGCONV} --to-code=UTF-8 -o mco-test1.out mco-test1.po\n"
        + b"test $? = 0 || { rm -fr $tmpfiles; exit 1; }\n\n"
        + b'tmpfiles="$tmpfiles mco-test1.ok"\n'
        + b"cat <<\\EOF > mco-test1.ok\n"
        + ok_body
        + b"EOF\n"
        + b"\n: ${DIFF=diff}\n"
        + b"# Redirect stdout, so as not to fill the user's screen with "
        + b"non-ASCII bytes.\n"
        + b"${DIFF} mco-test1.ok mco-test1.out >/dev/null\n"
        + b"result=$?\n\n"
        + b"rm -fr $tmpfiles\n\n"
        + b"exit $result\n"
    )


def _write_manifest(
    path: Path,
    payload: bytes,
    *,
    sha256: str | None = None,
    classification: str = "mislabeled_non_cpp",
    detected_format: str = "xml_utf16le",
    relative_path: str = RELATIVE_XML,
    reason: str = "fixture XML stored under a .cc suffix",
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "entries": [
                    {
                        "project_id": PROJECT_ID,
                        "relative_path": relative_path,
                        "size_bytes": len(payload),
                        "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
                        "classification": classification,
                        "detected_format": detected_format,
                        "reason": reason,
                    }
                ],
                "collections": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_collection_manifest(
    path: Path,
    payloads: dict[str, bytes],
    *,
    expected_file_count: int | None = None,
    content_set_sha256: str | None = None,
) -> None:
    rows = [
        [relative_path, len(payload), hashlib.sha256(payload).hexdigest()]
        for relative_path, payload in sorted(payloads.items())
    ]
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "entries": [],
                "collections": [
                    {
                        "project_id": PROJECT_ID,
                        "relative_path_prefix": CERTIFICATE_PAIR_PREFIX,
                        "relative_path_suffix": ".cp",
                        "expected_file_count": (
                            expected_file_count
                            if expected_file_count is not None
                            else len(payloads)
                        ),
                        "content_set_sha256": content_set_sha256 or digest,
                        "classification": "mislabeled_non_cpp",
                        "detected_format": "asn1_der_x509_certificate_pair",
                        "reason": "DER certificate-pair fixtures stored under .cp",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_cpp_discovery_preserves_large_and_nonproduction_source_trees(
    tmp_path: Path,
) -> None:
    fixtures = {
        "src/main.cpp": b"int main() { return 0; }\n",
        "tests/test_main.cpp": b"void test_main() {}\n",
        "third_party/vendor.hpp": b"#pragma once\n",
        "examples/demo.cc": b"void demo() {}\n",
        "docs/snippet.cxx": b"void documented() {}\n",
        "fuzzing/fuzz.cpp": b"void fuzz() {}\n",
        "src/large.cpp": b"//" + b"x" * 500_001,
    }
    for relative_path, payload in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    discovered = {
        Path(path).relative_to(tmp_path).as_posix()
        for path in ip.find_cpp_files(str(tmp_path))
    }
    assert discovered == set(fixtures)

    explicitly_filtered = {
        Path(path).relative_to(tmp_path).as_posix()
        for path in ip.find_cpp_files(
            str(tmp_path),
            extra_exclude_dirs={"third_party"},
        )
    }
    assert explicitly_filtered == set(fixtures) - {"third_party/vendor.hpp"}


def test_exact_quarantine_filters_verified_non_cpp_and_builds_receipt(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    code = tmp_path / "src/main.cpp"
    code.parent.mkdir(parents=True)
    code.write_text("int main() { return 0; }\n", encoding="utf-8")
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload)

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    candidates = ip.find_cpp_files(str(tmp_path))
    kept, receipt = policy.filter_candidates(tmp_path, candidates)

    assert kept == [str(code)]
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["candidate_count_before_quarantine"] == 2
    assert receipt["candidate_count_after_quarantine"] == 1
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"] == [
        {
            "project_id": PROJECT_ID,
            "relative_path": RELATIVE_XML,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "classification": "mislabeled_non_cpp",
            "detected_format": "xml_utf16le",
            "reason": "fixture XML stored under a .cc suffix",
        }
    ]


def test_quarantine_hash_mismatch_fails_without_filtering(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload, sha256="0" * 64)

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="SHA-256 mismatch"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_legacy_point_manifest_remains_supported(tmp_path: Path) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["schema"] = LEGACY_MANIFEST_SCHEMA
    del raw["collections"]
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1


def test_exact_quarantine_filters_deliberate_clang_crash_fixture(
    tmp_path: Path,
) -> None:
    payload = _clang_crash_fixture_bytes()
    candidate = tmp_path / RELATIVE_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=RELATIVE_CRASH_FIXTURE,
        reason="fixture deliberately crashes Clang",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "deliberate_compiler_crash_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "clang_debug_crash_pragma"
    )


def test_exact_quarantine_filters_deliberate_clang_index_crash_fixture(
    tmp_path: Path,
) -> None:
    payload = _clang_index_crash_fixture_bytes()
    candidate = tmp_path / RELATIVE_INDEX_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=RELATIVE_INDEX_CRASH_FIXTURE,
        reason="fixture deliberately crashes Clang indexer",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["detected_format"] == (
        "clang_debug_crash_pragma"
    )


def test_exact_quarantine_filters_deliberate_clang_remap_crash_fixture(
    tmp_path: Path,
) -> None:
    payload = _clang_index_remap_crash_fixture_bytes()
    candidate = tmp_path / RELATIVE_INDEX_REMAP_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=RELATIVE_INDEX_REMAP_CRASH_FIXTURE,
        reason="fixture deliberately crashes Clang during remap parsing",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1


def test_exact_quarantine_filters_clang_embedded_nul_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _clang_embedded_nul_diagnostic_bytes()
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_embedded_nul_diagnostic",
        relative_path=RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        reason="fixture intentionally embeds a NUL for Clang diagnostics",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "deliberate_compiler_diagnostic_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "clang_embedded_nul_diagnostic"
    )


def test_exact_quarantine_filters_clang_newline_nul_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _clang_newline_nul_diagnostic_bytes()
    candidate = tmp_path / RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_escaped_newline_nul_preprocessor_diagnostic",
        relative_path=RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE,
        reason="fixture intentionally embeds a NUL after an escaped newline",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["classification"] == (
        "deliberate_compiler_diagnostic_fixture"
    )
    assert receipt["entries"][0]["detected_format"] == (
        "clang_escaped_newline_nul_preprocessor_diagnostic"
    )


def test_clang_newline_nul_quarantine_requires_preprocessor_signature(
    tmp_path: Path,
) -> None:
    payload = _clang_newline_nul_diagnostic_bytes().replace(
        b"expected-error {{not a valid binary operator}}",
        b"expected-error {{unrelated diagnostic}}",
    )
    candidate = tmp_path / RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_escaped_newline_nul_preprocessor_diagnostic",
        relative_path=RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE,
        reason="forged newline/NUL diagnostic fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="contract is incomplete"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_der_x509_certificate_pair(
    tmp_path: Path,
) -> None:
    payload = _certificate_pair_bytes()
    candidate = tmp_path / RELATIVE_CERTIFICATE_PAIR
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="asn1_der_x509_certificate_pair",
        relative_path=RELATIVE_CERTIFICATE_PAIR,
        reason="DER certificate-pair fixture stored under a .cp suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == (
        "asn1_der_x509_certificate_pair"
    )


def test_exact_collection_quarantine_filters_complete_der_set(
    tmp_path: Path,
) -> None:
    payloads = {
        f"{CERTIFICATE_PAIR_PREFIX}forward.cp": _certificate_pair_bytes(),
        f"{CERTIFICATE_PAIR_PREFIX}reverse.cp": _certificate_pair_bytes(
            wrapper_tag=0xA0
        ),
    }
    candidates = []
    for relative_path, payload in payloads.items():
        candidate = tmp_path / relative_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(payload)
        candidates.append(str(candidate))
    manifest = tmp_path / "quarantine.json"
    _write_collection_manifest(manifest, payloads)

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, candidates)

    assert kept == []
    assert receipt["project_manifest_entry_count"] == 1
    assert receipt["quarantined_count"] == 2
    assert [entry["relative_path"] for entry in receipt["entries"]] == sorted(payloads)


def test_collection_quarantine_rejects_incomplete_or_drifted_set(
    tmp_path: Path,
) -> None:
    relative_path = f"{CERTIFICATE_PAIR_PREFIX}forward.cp"
    payload = _certificate_pair_bytes()
    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_collection_manifest(
        manifest,
        {relative_path: payload},
        expected_file_count=2,
    )
    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="count mismatch"):
        policy.filter_candidates(tmp_path, [str(candidate)])

    _write_collection_manifest(
        manifest,
        {relative_path: payload},
        content_set_sha256="0" * 64,
    )
    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="content-set SHA-256 mismatch"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_utf16le_generated_binary_blob(
    tmp_path: Path,
) -> None:
    payload = _utf16le_generated_c_array_bytes()
    candidate = tmp_path / RELATIVE_GENERATED_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_binary_blob",
        detected_format="utf16le_generated_c_array",
        relative_path=RELATIVE_GENERATED_BLOB,
        reason="generated binary blob fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["classification"] == "generated_binary_blob"


def test_exact_quarantine_filters_nul_ff_binary_blob(tmp_path: Path) -> None:
    payload = b"\0\xff" * 386 + b"\0"
    candidate = tmp_path / RELATIVE_NUL_FF_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="nul_ff_binary_blob",
        relative_path=RELATIVE_NUL_FF_BLOB,
        reason="binary 0x00/0xff payload stored under a header suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["detected_format"] == "nul_ff_binary_blob"


def test_exact_quarantine_filters_self_executing_zip(
    tmp_path: Path,
) -> None:
    payload = _self_executing_zip_bytes()
    candidate = tmp_path / RELATIVE_EXECUTABLE_ARCHIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_executable_archive",
        detected_format="posix_shell_appended_zip",
        relative_path=RELATIVE_EXECUTABLE_ARCHIVE,
        reason="self-executing archive fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["detected_format"] == (
        "posix_shell_appended_zip"
    )


def test_executable_archive_quarantine_rejects_invalid_zip(
    tmp_path: Path,
) -> None:
    payload = b'#!/bin/sh\nexec java -jar "$0" "$@"\nPK\x03\x04not-a-zip\n'
    candidate = tmp_path / RELATIVE_EXECUTABLE_ARCHIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_executable_archive",
        detected_format="posix_shell_appended_zip",
        relative_path=RELATIVE_EXECUTABLE_ARCHIVE,
        reason="forged self-executing archive",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="appended ZIP is invalid"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_truncated_utf32be_bom(tmp_path: Path) -> None:
    payload = _truncated_utf32be_bom_bytes()
    candidate = tmp_path / RELATIVE_TRUNCATED_UTF32BE_BOM
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="truncated_utf32be_bom",
        relative_path=RELATIVE_TRUNCATED_UTF32BE_BOM,
        reason="truncated UTF-32BE BOM fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == "truncated_utf32be_bom"


def test_truncated_utf32be_bom_quarantine_rejects_other_payload(
    tmp_path: Path,
) -> None:
    payload = b"\x00\x00\xff"
    candidate = tmp_path / RELATIVE_TRUNCATED_UTF32BE_BOM
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="truncated_utf32be_bom",
        relative_path=RELATIVE_TRUNCATED_UTF32BE_BOM,
        reason="forged truncated UTF-32BE BOM fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="exactly the three-byte"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_exact_quarantine_filters_big5_shell_heredoc(tmp_path: Path) -> None:
    payload = _big5_shell_heredoc_bytes()
    candidate = tmp_path / RELATIVE_BIG5_SHELL_HEREDOC
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="big5_shell_heredoc",
        relative_path=RELATIVE_BIG5_SHELL_HEREDOC,
        reason="BIG5 shell heredoc fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["entries"][0]["detected_format"] == "big5_shell_heredoc"


def test_big5_shell_heredoc_quarantine_rejects_changed_message(
    tmp_path: Path,
) -> None:
    payload = _big5_shell_heredoc_bytes().replace(b"--to-code=UTF-8", b"--to-code=BIG5")
    candidate = tmp_path / RELATIVE_BIG5_SHELL_HEREDOC
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="big5_shell_heredoc",
        relative_path=RELATIVE_BIG5_SHELL_HEREDOC,
        reason="forged BIG5 shell heredoc fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="conversion and cleanup"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_nul_ff_binary_blob_verification_streams_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"\0" * (1024 * 1024) + b"\xff" * (1024 * 1024)
    candidate = tmp_path / RELATIVE_NUL_FF_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="nul_ff_binary_blob",
        relative_path=RELATIVE_NUL_FF_BLOB,
        reason="binary 0x00/0xff payload stored under a header suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    original_read_bytes = Path.read_bytes

    def reject_candidate_read_bytes(path: Path) -> bytes:
        if path == candidate:
            raise AssertionError("nul_ff_binary_blob verification must stream input")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_candidate_read_bytes)

    kept, receipt = policy.filter_candidates(tmp_path, [str(candidate)])

    assert kept == []
    assert receipt["quarantined_count"] == 1


@pytest.mark.parametrize(
    "payload",
    [b"", b"\0" * 4, b"\xff" * 4, b"\0\xff\x01"],
)
def test_nul_ff_binary_blob_requires_both_values_and_no_others(
    tmp_path: Path,
    payload: bytes,
) -> None:
    candidate = tmp_path / RELATIVE_NUL_FF_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="nul_ff_binary_blob",
        relative_path=RELATIVE_NUL_FF_BLOB,
        reason="binary 0x00/0xff payload stored under a header suffix",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="only 0x00 and 0xff"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_generated_binary_blob_quarantine_rejects_small_c_array(
    tmp_path: Path,
) -> None:
    payload = _utf16le_generated_c_array_bytes(byte_count=16)
    candidate = tmp_path / RELATIVE_GENERATED_BLOB
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_binary_blob",
        detected_format="utf16le_generated_c_array",
        relative_path=RELATIVE_GENERATED_BLOB,
        reason="forged generated binary blob",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="contract is incomplete"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_certificate_pair_quarantine_rejects_non_certificate_der(
    tmp_path: Path,
) -> None:
    payload = _der(0x30, _der(0xA1, _der(0x30, _der(0x02, b"\x01"))))
    candidate = tmp_path / RELATIVE_CERTIFICATE_PAIR
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="mislabeled_non_cpp",
        detected_format="asn1_der_x509_certificate_pair",
        relative_path=RELATIVE_CERTIFICATE_PAIR,
        reason="forged certificate pair",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="field layout"):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_checked_in_clang_crash_manifest_matches_reference_fixture() -> None:
    payload = _clang_crash_fixture_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = [
        item
        for item in manifest["entries"]
        if item["project_id"]
        in {"google/filament", "microsoft/DirectXShaderCompiler"}
        and item["relative_path"].endswith(RELATIVE_CRASH_FIXTURE)
    ]

    assert len(payload) == 271
    assert {entry["project_id"] for entry in entries} == {
        "google/filament",
        "microsoft/DirectXShaderCompiler",
    }
    for entry in entries:
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["classification"] == "deliberate_compiler_crash_fixture"
        assert entry["detected_format"] == "clang_debug_crash_pragma"


def test_checked_in_filament_index_crash_manifest_matches_reference_fixture() -> None:
    fixtures = {
        RELATIVE_INDEX_CRASH_FIXTURE: _clang_index_crash_fixture_bytes(),
        RELATIVE_INDEX_REMAP_CRASH_FIXTURE: _clang_index_remap_crash_fixture_bytes(),
    }
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entries = {
        relative_path: next(
            item
            for item in manifest["entries"]
            if item["project_id"] == "google/filament"
            and item["relative_path"].endswith(relative_path)
        )
        for relative_path in fixtures
    }

    assert len(fixtures[RELATIVE_INDEX_CRASH_FIXTURE]) == 344
    assert hashlib.sha256(fixtures[RELATIVE_INDEX_CRASH_FIXTURE]).hexdigest() == (
        "1dae510e0b173890f77aa3ef905b892614b3b5c7a98add3df7b58a555ccef727"
    )
    assert len(fixtures[RELATIVE_INDEX_REMAP_CRASH_FIXTURE]) == 398
    assert hashlib.sha256(
        fixtures[RELATIVE_INDEX_REMAP_CRASH_FIXTURE]
    ).hexdigest() == "4170335b0ad9450e204fcf9625e6d7f506f84308b10857b7b57eb37973b66590"
    assert set(entries) == set(fixtures)
    for relative_path, payload in fixtures.items():
        entry = entries[relative_path]
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["classification"] == "deliberate_compiler_crash_fixture"
        assert entry["detected_format"] == "clang_debug_crash_pragma"


def test_checked_in_intel_nul_diagnostic_manifest_matches_reference_fixture() -> None:
    payload = _clang_embedded_nul_diagnostic_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "intel/llvm"
        and item["relative_path"] == RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    )

    assert len(payload) == 398
    assert hashlib.sha256(payload).hexdigest() == (
        "acba383a8c05e95c15d885e06c467d58edf39f5c7f84a0376f86cdb20d40be3a"
    )
    assert entry["relative_path"] == RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_diagnostic_fixture"
    assert entry["detected_format"] == "clang_embedded_nul_diagnostic"


def test_checked_in_intel_newline_nul_manifest_matches_reference_fixture() -> None:
    payload = _clang_newline_nul_diagnostic_bytes()
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "intel/llvm"
        and item["relative_path"] == RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    )

    assert len(payload) == 332
    assert hashlib.sha256(payload).hexdigest() == (
        "11548a466e2e5eb0a686cf422f7872ea94f616fe43f4bdb9311a0f1764391474"
    )
    assert entry["size_bytes"] == len(payload)
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "deliberate_compiler_diagnostic_fixture"
    assert entry["detected_format"] == (
        "clang_escaped_newline_nul_preprocessor_diagnostic"
    )


def test_checked_in_cmake_truncated_bom_manifest_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "Kitware/CMake"
    )

    payload = _truncated_utf32be_bom_bytes()
    assert entry["relative_path"] == RELATIVE_TRUNCATED_UTF32BE_BOM
    assert entry["size_bytes"] == len(payload) == 3
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "mislabeled_non_cpp"
    assert entry["detected_format"] == "truncated_utf32be_bom"


def test_checked_in_xemu_certificate_pair_manifest_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    collection = next(
        item
        for item in manifest["collections"]
        if item["project_id"] == "xemu-project/xemu"
    )

    assert collection["relative_path_prefix"] == (
        "roms/edk2/CryptoPkg/Library/OpensslLib/openssl/pyca-cryptography/"
        "vectors/cryptography_vectors/x509/PKITS_data/certpairs/"
    )
    assert collection["relative_path_suffix"] == ".cp"
    assert collection["expected_file_count"] == 348
    assert collection["content_set_sha256"] == (
        "4d92e2254cef41f0a84525e6e30a1d6fcda5237d0b878522ed911cfa973c6ef7"
    )
    assert collection["classification"] == "mislabeled_non_cpp"
    assert collection["detected_format"] == "asn1_der_x509_certificate_pair"

    security_collection = next(
        item
        for item in manifest["collections"]
        if item["project_id"] == "xemu-project/xemu"
        and item["relative_path_prefix"]
        == (
            "roms/edk2/SecurityPkg/DeviceSecurity/SpdmLib/libspdm/os_stub/"
            "openssllib/openssl/pyca-cryptography/vectors/"
            "cryptography_vectors/x509/PKITS_data/certpairs/"
        )
    )
    assert security_collection["relative_path_suffix"] == ".cp"
    assert security_collection["expected_file_count"] == 348
    assert security_collection["content_set_sha256"] == (
        "f109a9ffd7ad27256039925e3222ccd21fba5c6e734b2a12f0b243122e36701e"
    )
    assert security_collection["classification"] == "mislabeled_non_cpp"
    assert security_collection["detected_format"] == "asn1_der_x509_certificate_pair"


def test_checked_in_windows_nul_ff_manifest_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "corpus.local/windows_10_shared_source_kit"
    )

    assert entry["relative_path"] == RELATIVE_WINDOWS_NUL_FF_BLOB
    assert entry["size_bytes"] == 773
    assert entry["sha256"] == (
        "38f8873ec81398a0a0e025690b3e2baa55c5c1221441d68ca0144b0b557fcb4b"
    )
    assert entry["classification"] == "mislabeled_non_cpp"
    assert entry["detected_format"] == "nul_ff_binary_blob"


def test_checked_in_threadx_generated_blob_manifest_matches_frozen_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "eclipse-threadx/threadx"
    )

    assert entry["size_bytes"] == 60766
    assert entry["sha256"] == (
        "303a817f33b086755103778421c37b3aa716e5d49c95c2a9a2d9a6f31cafcd9c"
    )
    assert entry["classification"] == "generated_binary_blob"
    assert entry["detected_format"] == "utf16le_generated_c_array"


def test_checked_in_netbsd_big5_shell_manifest_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "NetBSD/src"
    )

    payload = _big5_shell_heredoc_bytes()
    assert entry["relative_path"] == RELATIVE_BIG5_SHELL_HEREDOC
    assert entry["size_bytes"] == len(payload) == 1155
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["classification"] == "mislabeled_non_cpp"
    assert entry["detected_format"] == "big5_shell_heredoc"


def test_checked_in_cpython_executable_archive_matches_archive_receipt() -> None:
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/source_quarantine_manifest.json"
        ).read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in manifest["entries"]
        if item["project_id"] == "python/cpython"
        and item["relative_path"] == "Lib/test/archivetestdata/exe_with_zip"
    )

    assert entry["size_bytes"] == 990
    assert entry["sha256"] == (
        "2f27f5c9108936a693fd496565e5c5050b5c62cfbb61d1d5da9d97c89533d637"
    )
    assert entry["classification"] == "generated_executable_archive"
    assert entry["detected_format"] == "posix_shell_appended_zip"


def test_clang_crash_quarantine_requires_independent_fixture_signature(
    tmp_path: Path,
) -> None:
    payload = b"int main(void) { return 0; }\n"
    candidate = tmp_path / RELATIVE_CRASH_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_crash_fixture",
        detected_format="clang_debug_crash_pragma",
        relative_path=RELATIVE_CRASH_FIXTURE,
        reason="forged crash fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(
        SourceQuarantineError,
        match="crash-test contract is incomplete",
    ):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_clang_embedded_nul_quarantine_requires_diagnostic_signature(
    tmp_path: Path,
) -> None:
    payload = b"int x[sizeof\0int];\n"
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_embedded_nul_diagnostic",
        relative_path=RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        reason="forged diagnostic fixture",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(
        SourceQuarantineError,
        match="embedded-NUL diagnostic contract is incomplete",
    ):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_clang_embedded_nul_quarantine_requires_both_caret_markers(
    tmp_path: Path,
) -> None:
    payload = _clang_embedded_nul_diagnostic_bytes().replace(
        b"// CHECK-NEXT:             ^\n",
        b"// CHECK-NEXT:             x\n",
        1,
    )
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    _write_manifest(
        manifest,
        payload,
        classification="deliberate_compiler_diagnostic_fixture",
        detected_format="clang_embedded_nul_diagnostic",
        relative_path=RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        reason="forged diagnostic fixture missing one caret marker",
    )

    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(
        SourceQuarantineError,
        match="embedded-NUL diagnostic contract is incomplete",
    ):
        policy.filter_candidates(tmp_path, [str(candidate)])


def test_quarantine_entry_must_be_discovered_and_cannot_hide_parse_errors(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    manifest = tmp_path / "quarantine.json"
    _write_manifest(manifest, payload)
    policy = ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)
    with pytest.raises(SourceQuarantineError, match="were not discovered"):
        policy.filter_candidates(tmp_path, [])

    _write_manifest(
        manifest,
        payload,
        classification="parse_error",
        detected_format="xml_utf16le",
    )
    with pytest.raises(SourceQuarantineError, match="unsupported quarantine"):
        ProjectSourceQuarantine.load(manifest, project_id=PROJECT_ID)


def test_process_project_writes_atomic_bound_receipt(
    tmp_path: Path,
) -> None:
    payload = _xml_bytes()
    candidate = tmp_path / RELATIVE_XML
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    receipt_path = tmp_path / "receipts/source.json"
    _write_manifest(manifest, payload)

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id=PROJECT_ID,
        source_quarantine_manifest=str(manifest),
        source_quarantine_receipt=str(receipt_path),
    )

    assert documents == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["project_id"] == PROJECT_ID
    assert receipt["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert receipt["quarantined_count"] == 1
    omission_receipt = receipt["external_reference_omissions"]
    assert omission_receipt["schema"] == "cppmega.external_reference_omissions_v1"
    assert omission_receipt["status"] == "complete"
    assert omission_receipt["reason"] == "unknown_external_provider"
    assert omission_receipt["observation_count"] == 0
    assert omission_receipt["unique_reference_count"] == 0
    assert omission_receipt["location_count"] == 0
    assert omission_receipt["locations"] == []


def test_process_project_quarantines_clang_embedded_nul_diagnostic(
    tmp_path: Path,
) -> None:
    payload = _clang_embedded_nul_diagnostic_bytes()
    candidate = tmp_path / RELATIVE_NUL_DIAGNOSTIC_FIXTURE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    newline_payload = _clang_newline_nul_diagnostic_bytes()
    newline_candidate = tmp_path / RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE
    newline_candidate.parent.mkdir(parents=True)
    newline_candidate.write_bytes(newline_payload)
    manifest = Path(__file__).parents[1] / "configs/source_quarantine_manifest.json"
    receipt_path = tmp_path / "receipts/source.json"

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id="intel/llvm",
        source_quarantine_manifest=str(manifest),
        source_quarantine_receipt=str(receipt_path),
    )

    assert documents == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["project_id"] == "intel/llvm"
    assert receipt["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert receipt["quarantined_count"] == 2
    assert {entry["relative_path"] for entry in receipt["entries"]} == {
        RELATIVE_NUL_DIAGNOSTIC_FIXTURE,
        RELATIVE_NEWLINE_NUL_DIAGNOSTIC_FIXTURE,
    }


def test_process_project_quarantines_non_cpp_executable_archive(
    tmp_path: Path,
) -> None:
    payload = _self_executing_zip_bytes()
    candidate = tmp_path / RELATIVE_EXECUTABLE_ARCHIVE
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(payload)
    manifest = tmp_path / "quarantine.json"
    receipt_path = tmp_path / "receipts/source.json"
    _write_manifest(
        manifest,
        payload,
        classification="generated_executable_archive",
        detected_format="posix_shell_appended_zip",
        relative_path=RELATIVE_EXECUTABLE_ARCHIVE,
        reason="self-executing archive fixture",
    )

    documents = ip.process_project(
        str(tmp_path),
        enriched=True,
        project_id=PROJECT_ID,
        source_quarantine_manifest=str(manifest),
        source_quarantine_receipt=str(receipt_path),
    )

    assert documents == []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["quarantined_count"] == 1
    assert receipt["entries"][0]["relative_path"] == RELATIVE_EXECUTABLE_ARCHIVE
