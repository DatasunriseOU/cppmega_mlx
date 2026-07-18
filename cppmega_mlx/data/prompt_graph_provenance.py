"""Shared prompt-graph provenance contracts for the MLX producer."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .symbol_identity import (
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    canonical_external_provider_file,
    canonical_external_usr_identity,
    compute_symbol_id,
    external_provider_project,
    is_repo_file_location_identity,
    parse_repo_file_location_identity,
)


PRODUCTION_IDENTITY_PROVENANCE_CONTRACT = (
    "case4_symbol_reference_v3_repo_binding_v1"
)
INDEX_INTEGRITY_VERSION = "1"
INDEX_PAYLOAD_HASH_KEY = "index_payload_sha256"
INDEXER_DEPENDENCY_HASH_KEY = "indexer_dependency_closure_sha256"
INDEXER_DEPENDENCY_MANIFEST_KEY = "indexer_dependency_manifest"
INDEXER_DEPENDENCY_POLICY = "imported_local_python_closure_v1"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha_json(value: Any) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _package_parts(path: Path, checkout_root: Path) -> tuple[str, ...]:
    relative = path.relative_to(checkout_root)
    return tuple(part for part in relative.parts[:-1] if part != "__pycache__")


def _resolve_local_module(
    module_name: str,
    *,
    checkout_root: Path,
    current_path: Path,
) -> tuple[Path, ...]:
    if not module_name:
        return ()
    parts = tuple(part for part in module_name.split(".") if part)
    candidates = [
        checkout_root.joinpath(*parts),
        current_path.parent.joinpath(*parts),
    ]
    resolved: list[Path] = []

    def add_path(candidate: Path) -> None:
        try:
            resolved_candidate = candidate.resolve(strict=True)
            relative_parts = resolved_candidate.relative_to(checkout_root).parts
        except (OSError, RuntimeError, ValueError):
            return
        package_parts = relative_parts[:-1]
        for index in range(1, len(package_parts) + 1):
            initializer = checkout_root.joinpath(
                *package_parts[:index],
                "__init__.py",
            )
            try:
                initializer = initializer.resolve(strict=True)
                initializer.relative_to(checkout_root)
            except (OSError, RuntimeError, ValueError):
                continue
            if initializer.is_file() and initializer not in resolved:
                resolved.append(initializer)
        if (
            resolved_candidate.is_file()
            and resolved_candidate.suffix == ".py"
            and resolved_candidate not in resolved
        ):
            resolved.append(resolved_candidate)

    for candidate in candidates:
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            add_path(path)
    return tuple(resolved)


def _imports(tree: ast.AST) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    result: list[tuple[str, int, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((alias.name, 0, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(
                (
                    node.module or "",
                    int(node.level),
                    tuple(alias.name for alias in node.names if alias.name != "*"),
                )
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            result.append((node.args[0].value, 0, ()))
    return tuple(result)


def indexer_dependency_manifest(
    indexer_path: str | Path,
    checkout_root: str | Path,
) -> dict[str, str]:
    entrypoint = Path(indexer_path).expanduser().resolve()
    root = Path(checkout_root).expanduser().resolve()
    try:
        entrypoint.relative_to(root)
    except ValueError as exc:
        raise ValueError("indexer entrypoint is outside its checkout") from exc
    pending = [entrypoint]
    visited: set[Path] = set()
    manifest: dict[str, str] = {}
    while pending:
        current = pending.pop().resolve()
        if current in visited:
            continue
        visited.add(current)
        manifest[current.relative_to(root).as_posix()] = sha256(
            current.read_bytes()
        ).hexdigest()
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        package = _package_parts(current, root)
        for module, level, names in _imports(tree):
            if level:
                base = package[: max(0, len(package) - level + 1)]
                qualified = ".".join((*base, module))
            else:
                qualified = module
            candidates = _resolve_local_module(
                qualified,
                checkout_root=root,
                current_path=current,
            )
            for candidate in candidates:
                if candidate not in visited:
                    pending.append(candidate)
            for name in names:
                candidates = _resolve_local_module(
                    f"{qualified}.{name}",
                    checkout_root=root,
                    current_path=current,
                )
                for candidate in candidates:
                    if candidate not in visited:
                        pending.append(candidate)
    return dict(sorted(manifest.items()))


def indexer_dependency_hash(
    indexer_path: str | Path,
    checkout_root: str | Path,
) -> tuple[dict[str, str], str]:
    manifest = indexer_dependency_manifest(indexer_path, checkout_root)
    return manifest, _sha_json(manifest)


def validate_indexer_dependency_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_indexer_root: str | Path,
) -> None:
    root = Path(expected_indexer_root).expanduser().resolve()
    raw_path = provenance.get("indexer_path")
    raw_manifest = provenance.get(INDEXER_DEPENDENCY_MANIFEST_KEY)
    hashes = provenance.get("hashes")
    if not isinstance(raw_path, str) or not isinstance(raw_manifest, Mapping):
        raise ValueError("prompt graph indexer dependency provenance is missing")
    if not isinstance(hashes, Mapping) or not _is_sha256(
        hashes.get(INDEXER_DEPENDENCY_HASH_KEY)
    ):
        raise ValueError("prompt graph indexer dependency hash is missing")
    expected_path = root / "tools" / "clang_indexer" / "index_project.py"
    checkout_root = provenance.get("indexer_checkout_root")
    if (
        not isinstance(checkout_root, str)
        or Path(checkout_root).expanduser().resolve() != root
    ):
        raise ValueError("prompt graph indexer checkout provenance is not same-checkout")
    if Path(raw_path).expanduser().resolve(strict=False) != expected_path:
        raise ValueError("prompt graph indexer dependency path is not same-checkout")
    actual_manifest = indexer_dependency_manifest(expected_path, root)
    if dict(raw_manifest) != actual_manifest:
        raise ValueError("prompt graph indexer dependency provenance is stale")
    if _sha_json(actual_manifest) != hashes[INDEXER_DEPENDENCY_HASH_KEY]:
        raise ValueError("prompt graph indexer dependency hash mismatch")


def validate_external_references(
    provenance: Mapping[str, Any],
    *,
    project_id: str,
    index: Any | None = None,
) -> None:
    if "external_references" not in provenance:
        raise ValueError("prompt graph external_references are missing")
    references = provenance.get("external_references")
    if not isinstance(references, list):
        raise ValueError("prompt graph external_references must be a list")
    documents = {
        int(document.id): document for document in getattr(index, "documents", ())
    }
    seen: set[tuple[Any, ...]] = set()
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ValueError("prompt graph external reference must be an object")
        relation = reference.get("relation")
        if relation not in {"call", "type", "def_use"}:
            raise ValueError("prompt graph external reference relation is invalid")
        provider = reference.get("provider")
        include = reference.get("include_provenance")
        provider_project = reference.get("project")
        file = reference.get("file")
        if not all(
            isinstance(value, str) and value
            for value in (provider, include, provider_project, file)
        ):
            raise ValueError("prompt graph external reference provenance is incomplete")
        try:
            expected_project = external_provider_project(
                provider,
                source="prompt graph external reference",
            )
            expected_file = canonical_external_provider_file(
                provider,
                include,
                source="prompt graph external reference",
            )
        except Exception as exc:
            raise ValueError(
                "prompt graph external reference provider is not trusted"
            ) from exc
        if provider_project != expected_project or file != expected_file:
            raise ValueError(
                "prompt graph external reference provider identity is inconsistent"
            )
        if provider_project == project_id:
            raise ValueError("prompt graph external reference is incorrectly local")
        line = reference.get("line")
        column = reference.get("column")
        if (
            isinstance(line, bool)
            or not isinstance(line, int)
            or line <= 0
            or isinstance(column, bool)
            or not isinstance(column, int)
            or column <= 0
        ):
            raise ValueError("prompt graph external reference location is invalid")
        symbol_key = reference.get("symbol_key")
        symbol_id = reference.get("symbol_id")
        if (
            not isinstance(symbol_key, str)
            or not symbol_key
            or isinstance(symbol_id, bool)
            or not isinstance(symbol_id, int)
            or symbol_id != compute_symbol_id(symbol_key)
        ):
            raise ValueError("prompt graph external reference identity is incomplete")
        usr = reference.get("usr")
        signature = reference.get("canonical_signature")
        qname = reference.get("qname")
        symbol_kind = reference.get("symbol_kind")
        if not isinstance(usr, str) or not isinstance(signature, str):
            raise ValueError("prompt graph external reference semantic fields are invalid")
        if not isinstance(qname, str) or not isinstance(symbol_kind, str) or not symbol_kind:
            raise ValueError("prompt graph external reference semantic fields are invalid")
        if usr:
            expected_key = canonical_external_usr_identity(
                usr=usr,
                canonical_signature=signature,
                provider=provider,
                include_provenance=include,
                project=provider_project,
                source="prompt graph external reference",
            )
            if symbol_key != expected_key:
                raise ValueError(
                    "prompt graph external reference USR identity is inconsistent"
                )
        elif is_repo_file_location_identity(symbol_key):
            identity = parse_repo_file_location_identity(
                symbol_key,
                source="prompt graph external reference",
            )
            if (
                identity.project != provider_project
                or identity.file != file
                or identity.line != line
                or identity.column != column
                or identity.kind != symbol_kind
                or identity.qname != qname
            ):
                raise ValueError(
                    "prompt graph external reference location identity is inconsistent"
                )
        elif signature and symbol_key.startswith("fallback:"):
            fields: dict[str, str] = {}
            for part in symbol_key.removeprefix("fallback:").split("\x1f"):
                key, separator, value = part.partition("=")
                if separator:
                    fields[key] = value
            scope = {
                key: value
                for key, separator, value in (
                    part.partition("=") for part in fields.get("scope", "").split("|")
                )
                if separator
            }
            if (
                fields.get("schema") != f"v{SYMBOL_IDENTITY_SCHEMA_VERSION}"
                or fields.get("qname") != qname
                or fields.get("kind") != symbol_kind
                or fields.get("sig") != " ".join(signature.split())
                or scope.get("project") != provider_project
                or scope.get("file") != file
            ):
                raise ValueError(
                    "prompt graph external reference signature identity is inconsistent"
                )
        else:
            raise ValueError(
                "prompt graph external reference requires a trusted USR/signature identity"
            )

        document_id = reference.get("document_id")
        source_path = reference.get("source_path")
        start = reference.get("start")
        end = reference.get("end")
        if (
            isinstance(document_id, bool)
            or not isinstance(document_id, int)
            or document_id <= 0
            or not isinstance(source_path, str)
            or not source_path
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError("prompt graph external reference source span is invalid")
        source_parts = Path(source_path).as_posix().split("/")
        if (
            Path(source_path).is_absolute()
            or any(part in {"", ".", ".."} for part in source_parts)
        ):
            raise ValueError("prompt graph external reference source path is invalid")
        if index is not None:
            document = documents.get(document_id)
            if document is None or document.source_path != source_path:
                raise ValueError(
                    "prompt graph external reference source document is invalid"
                )
            if end > len(document.source):
                raise ValueError("prompt graph external reference source span is invalid")
        identity_key = (relation, document_id, source_path, start, end, symbol_key)
        if identity_key in seen:
            raise ValueError("prompt graph external references contain a duplicate")
        seen.add(identity_key)
    count = provenance.get("external_reference_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(references)
    ):
        raise ValueError("prompt graph external reference count does not match payload")
    expected_order = sorted(
        references,
        key=lambda row: (
            row["document_id"],
            row["start"],
            row["end"],
            row["relation"],
            row["symbol_key"],
        ),
    )
    if references != expected_order:
        raise ValueError("prompt graph external references are not canonicalized")


def validate_shared_provenance(
    index: Any,
    *,
    expected_indexer_root: str | Path,
) -> None:
    provenance = index.provenance
    if provenance.get("index_integrity_version") != INDEX_INTEGRITY_VERSION:
        raise ValueError("production repository index integrity version mismatch")
    if provenance.get("indexer_dependency_policy") != INDEXER_DEPENDENCY_POLICY:
        raise ValueError("production repository indexer dependency policy mismatch")
    validate_indexer_dependency_provenance(
        provenance,
        expected_indexer_root=expected_indexer_root,
    )
    validate_external_references(
        provenance,
        project_id=index.project_id,
        index=index,
    )


__all__ = [
    "INDEX_INTEGRITY_VERSION",
    "INDEX_PAYLOAD_HASH_KEY",
    "INDEXER_DEPENDENCY_HASH_KEY",
    "INDEXER_DEPENDENCY_MANIFEST_KEY",
    "INDEXER_DEPENDENCY_POLICY",
    "PRODUCTION_IDENTITY_PROVENANCE_CONTRACT",
    "indexer_dependency_hash",
    "indexer_dependency_manifest",
    "validate_shared_provenance",
]
