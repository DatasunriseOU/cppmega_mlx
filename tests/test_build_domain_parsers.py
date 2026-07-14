from __future__ import annotations

import numpy as np

from cppmega_mlx.data.build_parsers import (
    parse_autoconf,
    parse_bazel,
    parse_cmake,
    parse_make,
    parse_ninja,
)
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


def _roles(parsed):
    out = {}
    for token, role in zip(parsed.tokens, parsed.role_ids):
        out.setdefault(token.text, []).append(role)
    return out


def _has_role(parsed, text: str, role: DomainRoleKind) -> bool:
    return int(role) in _roles(parsed).get(text, [])


def test_cmake_parser_marks_target_sources_and_libraries() -> None:
    parsed = parse_cmake(
        "add_executable(app main.cpp util.cc)\n"
        "target_link_libraries(app PRIVATE ssl crypto)\n"
    )
    assert parsed.domain == DomainKind.CMAKE
    assert _has_role(parsed, "add_executable", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "app", DomainRoleKind.TARGET)
    assert _has_role(parsed, "main.cpp", DomainRoleKind.SOURCE)
    assert _has_role(parsed, "ssl", DomainRoleKind.LIBRARY)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_SOURCE) for edge in parsed.edges)

    packet = parsed.to_packet()
    assert packet.domain == DomainKind.CMAKE
    assert int(np.asarray(packet.domain_ids)[0]) == int(DomainKind.CMAKE)


def test_cmake_parser_routes_balanced_multiline_command_arguments() -> None:
    parsed = parse_cmake(
        "add_executable(\n"
        "  app\n"
        "  src/main.cpp\n"
        "  src/util.cc\n"
        ")\n"
        "target_link_libraries(\n"
        "  app\n"
        "  PRIVATE\n"
        "  ssl\n"
        "  crypto\n"
        ")\n"
    )

    assert _has_role(parsed, "app", DomainRoleKind.TARGET)
    assert _has_role(parsed, "src/main.cpp", DomainRoleKind.SOURCE)
    assert _has_role(parsed, "src/util.cc", DomainRoleKind.SOURCE)
    assert _has_role(parsed, "ssl", DomainRoleKind.LIBRARY)
    assert _has_role(parsed, "crypto", DomainRoleKind.LIBRARY)
    source_edges = [
        edge
        for edge in parsed.edges
        if edge[2] == int(DomainEdgeKind.BUILD_TARGET_SOURCE)
    ]
    dependency_edges = [
        edge
        for edge in parsed.edges
        if edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP)
    ]
    assert len(source_edges) == 2
    assert len(dependency_edges) == 2
    assert parsed.metadata["targets_seen"] == 2


def test_cmake_parser_marks_missing_and_extra_command_parentheses_raw() -> None:
    missing = parse_cmake("add_executable(app\n  main.cpp\n")
    extra = parse_cmake("add_executable(app main.cpp))\n")
    no_open = parse_cmake("add_executable app main.cpp\n")
    comment = parse_cmake(
        "# add_executable(fake ( in a line comment\n"
        "#[=[\n"
        "target_link_libraries(fake ( missing\n"
        "]=]\n"
        'set(TEXT "unmatched ) in a string")\n'
        "add_executable(app main.cpp)\n"
    )

    for parsed in (missing, extra, no_open):
        assert set(parsed.confidence_ids) == {int(ParseConfidence.RAW)}
        assert parsed.metadata["unsupported_syntax"] == "malformed_cmake_command_parentheses"
    assert set(comment.confidence_ids) == {int(ParseConfidence.HEURISTIC)}


def test_make_parser_marks_targets_prereqs_and_recipe_command() -> None:
    parsed = parse_make("app: main.o util.o\n\t$(CXX) -o app main.o util.o\n")
    assert parsed.domain == DomainKind.MAKE
    assert _has_role(parsed, "app", DomainRoleKind.TARGET)
    assert _has_role(parsed, "main.o", DomainRoleKind.PREREQUISITE)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP) for edge in parsed.edges)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_RULE_COMMAND) for edge in parsed.edges)


def test_make_parser_does_not_emit_separator_comment_or_inline_recipe_edges() -> None:
    parsed = parse_make(
        "app: main.o | generated.h ; echo fake.cc # ignored.o\n"
    )
    edge_targets = {
        parsed.tokens[dst].text
        for _, dst, kind in parsed.edges
        if kind == int(DomainEdgeKind.BUILD_TARGET_DEP)
    }
    assert edge_targets == {"main.o", "generated.h"}
    for false_target in ("|", ";", "echo", "fake.cc", "#", "ignored.o"):
        assert not _has_role(parsed, false_target, DomainRoleKind.PREREQUISITE)


def test_ninja_parser_marks_rule_build_inputs_outputs() -> None:
    parsed = parse_ninja("rule cc\n  command = clang++ -c $in -o $out\nbuild main.o: cc main.cc\n")
    roles = _roles(parsed)
    assert parsed.domain == DomainKind.NINJA
    assert int(DomainRoleKind.RULE) in roles["cc"]
    assert _has_role(parsed, "main.o", DomainRoleKind.OUTPUT)
    assert _has_role(parsed, "main.cc", DomainRoleKind.INPUT)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_ACTION_INPUT) for edge in parsed.edges)


def test_ninja_parser_skips_separator_and_comment_pseudo_inputs() -> None:
    parsed = parse_ninja(
        "build out.o | out.d: cc in.cc | implicit.h || order.h # bogus.h\n"
    )
    edge_targets = {
        parsed.tokens[dst].text
        for _, dst, kind in parsed.edges
        if kind == int(DomainEdgeKind.BUILD_ACTION_INPUT)
    }
    assert edge_targets == {"in.cc", "implicit.h", "order.h"}
    assert "|" not in edge_targets
    assert "||" not in edge_targets
    assert "bogus.h" not in edge_targets


def test_bazel_parser_marks_rule_target_srcs_deps() -> None:
    parsed = parse_bazel('cc_library(name = "core", srcs = ["core.cc"], deps = ["//base:base"])\n')
    assert parsed.domain == DomainKind.BAZEL
    assert _has_role(parsed, "cc_library", DomainRoleKind.RULE)
    assert _has_role(parsed, '"core"', DomainRoleKind.TARGET)
    assert _has_role(parsed, '"core.cc"', DomainRoleKind.SOURCE)
    assert _has_role(parsed, '"//base:base"', DomainRoleKind.LABEL)
    assert any(edge[2] == int(DomainEdgeKind.BUILD_TARGET_DEP) for edge in parsed.edges)


def test_bazel_parser_repairs_edges_when_name_attribute_is_last() -> None:
    parsed = parse_bazel(
        'cc_library(srcs = ["core.cc"], deps = ["//base:base"], name = "core")\n'
    )
    target_idx = next(
        idx
        for idx, token in enumerate(parsed.tokens)
        if token.text == '"core"'
    )
    assert {
        (parsed.tokens[dst].text, kind)
        for src, dst, kind in parsed.edges
        if src == target_idx
    } == {
        ('"core.cc"', int(DomainEdgeKind.BUILD_TARGET_SOURCE)),
        ('"//base:base"', int(DomainEdgeKind.BUILD_TARGET_DEP)),
    }


def test_autoconf_parser_marks_macros() -> None:
    parsed = parse_autoconf("AC_INIT([demo], [1.0])\nAC_PROG_CXX\n")
    assert parsed.domain == DomainKind.AUTOCONF
    assert _has_role(parsed, "AC_INIT", DomainRoleKind.COMMAND)
    assert _has_role(parsed, "AC_PROG_CXX", DomainRoleKind.COMMAND)


def test_autoconf_m4_bracket_quotes_hide_literal_parentheses() -> None:
    parsed = parse_autoconf(
        "AC_MSG_NOTICE([literal ( and nested [still ) quoted]])\nAC_PROG_CXX\n"
    )
    assert set(parsed.confidence_ids) == {int(ParseConfidence.HEURISTIC)}
    assert _has_role(parsed, "AC_PROG_CXX", DomainRoleKind.COMMAND)

    malformed = parse_autoconf("AC_INIT([demo], [1.0)\n")
    assert set(malformed.confidence_ids) == {int(ParseConfidence.RAW)}
