"""CMake domain parser."""

from __future__ import annotations

from cppmega_mlx.data.build_parsers.base import (
    ParsedDomainDocument,
    is_option,
    is_source_path,
    strip_quotes,
)
from cppmega_mlx.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


_TARGET_COMMANDS = {
    "add_executable",
    "add_library",
    "target_sources",
    "target_link_libraries",
    "target_include_directories",
    "target_compile_options",
    "target_compile_definitions",
}
_COMMANDS = _TARGET_COMMANDS | {"project", "set", "option", "find_package", "include"}


def parse_cmake(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.CMAKE,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "cmake"},
    )
    next_entity = 1
    last_target_by_line: dict[int, int] = {}

    for idx, token in enumerate(doc.tokens):
        word = strip_quotes(token.text)
        lower = word.lower()
        if lower in _COMMANDS:
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            command_entity = next_entity
            next_entity += 1
            line_indices = doc.token_indices_on_line(token.line)
            after = [i for i in line_indices if i > idx and doc.tokens[i].text not in {"(", ")"}]
            if lower in _TARGET_COMMANDS and after:
                target_idx = after[0]
                doc.set_role(target_idx, DomainRoleKind.TARGET, entity=next_entity, scope=command_entity)
                target_entity = next_entity
                next_entity += 1
                last_target_by_line[token.line] = target_idx
                doc.add_edge(idx, target_idx, DomainEdgeKind.BUILD_COMMAND_TARGET)
                for arg_idx in after[1:]:
                    value = strip_quotes(doc.tokens[arg_idx].text)
                    if value.upper() in {"PRIVATE", "PUBLIC", "INTERFACE"}:
                        doc.set_role(arg_idx, DomainRoleKind.KEYWORD, scope=target_entity)
                    elif is_source_path(value):
                        doc.set_role(arg_idx, DomainRoleKind.SOURCE, entity=next_entity, scope=target_entity)
                        next_entity += 1
                        doc.add_edge(target_idx, arg_idx, DomainEdgeKind.BUILD_TARGET_SOURCE)
                    elif is_option(value):
                        doc.set_role(arg_idx, DomainRoleKind.OPTION, scope=target_entity)
                    elif lower == "target_link_libraries":
                        doc.set_role(arg_idx, DomainRoleKind.LIBRARY, entity=next_entity, scope=target_entity)
                        next_entity += 1
                        doc.add_edge(target_idx, arg_idx, DomainEdgeKind.BUILD_TARGET_DEP)
            elif lower == "set" and after:
                doc.set_role(after[0], DomainRoleKind.VARIABLE, entity=next_entity, scope=command_entity)
                next_entity += 1
        elif word.startswith("${") or word.startswith("$("):
            doc.set_role(idx, DomainRoleKind.VARIABLE)

    doc.metadata["targets_seen"] = len(last_target_by_line)
    return doc


__all__ = ["parse_cmake"]
