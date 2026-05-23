"""V7-Q03.1: ckpt.list_history RPC.

Scans a directory recursively for ``*.safetensors`` files, reads
self-describing metadata (header-only, no full tensor load) and
returns a sorted-by-mtime descending list capped at 100. Pairs with
``ckpt.inspect`` for the single-file inspector view and feeds the
``CheckpointHistoryDropdown`` UI component that lets the operator pick
a past checkpoint for resume without copy-pasting paths.

Closes Lane 6 audit gap from docs/UI-TO-TRAIN-AUDIT-2026-05-23.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache


class CkptListHistoryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = "."
    # Cap to prevent runaway scans on large workspaces.
    max_entries: int = 100


class CkptHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    mtime: float
    size_bytes: int
    arch_hash: str | None = None
    opt_kind: str | None = None
    global_step: int | None = None
    has_opt_sidecar: bool = False


class CkptListHistoryResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    directory: str
    scanned: int = 0
    entries: list[CkptHistoryEntry] = Field(default_factory=list)
    error: str | None = None


def ckpt_list_history(
    params: CkptListHistoryParams, *, cache: LRUCache | None = None,
) -> CkptListHistoryResult:
    root = Path(params.directory).expanduser()
    try:
        root = root.resolve()
    except Exception as exc:  # noqa: BLE001 -- surface to caller
        return CkptListHistoryResult(
            directory=str(params.directory), error=f"resolve failed: {exc}",
        )
    if not root.exists():
        return CkptListHistoryResult(
            directory=str(root), error="directory does not exist",
        )
    if not root.is_dir():
        return CkptListHistoryResult(
            directory=str(root), error="path is not a directory",
        )

    from cppmega_v4.runner.stages import read_ckpt_metadata

    raw_entries: list[CkptHistoryEntry] = []
    scanned = 0
    for path in root.rglob("*.safetensors"):
        # Skip opt-state sidecars — they're surfaced via has_opt_sidecar.
        name = path.name
        if name.endswith(".opt.safetensors") or name.endswith(".opt"):
            continue
        scanned += 1
        try:
            stat = path.stat()
            meta: dict[str, Any] | None = None
            try:
                meta = read_ckpt_metadata(str(path))
            except Exception:
                meta = None
            arch = (meta or {}).get("arch")
            train = (meta or {}).get("train")
            opt = (meta or {}).get("opt")
            arch_hash = (
                arch.get("config_hash") if isinstance(arch, dict) else None
            )
            opt_kind = (
                opt.get("kind") if isinstance(opt, dict) else None
            )
            global_step = (
                train.get("global_step") if isinstance(train, dict) else None
            )
            sidecar = path.with_suffix(path.suffix + ".opt")
            raw_entries.append(CkptHistoryEntry(
                path=str(path),
                mtime=stat.st_mtime,
                size_bytes=stat.st_size,
                arch_hash=arch_hash,
                opt_kind=opt_kind,
                global_step=global_step,
                has_opt_sidecar=sidecar.is_file(),
            ))
        except Exception:
            # Unreadable file — skip silently.
            continue

    raw_entries.sort(key=lambda e: e.mtime, reverse=True)
    capped = raw_entries[: max(1, int(params.max_entries))]
    return CkptListHistoryResult(
        directory=str(root), scanned=scanned, entries=capped,
    )


class CkptListSubdirsParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str = "."


class CkptListSubdirsResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    current: str
    parent: str | None = None
    subdirs: list[str] = Field(default_factory=list)
    error: str | None = None


def ckpt_list_subdirs(
    params: CkptListSubdirsParams, *, cache: LRUCache | None = None,
) -> CkptListSubdirsResult:
    root = Path(params.directory).expanduser()
    try:
        root = root.resolve()
    except Exception as exc:
        return CkptListSubdirsResult(
            current=params.directory, error=f"resolve failed: {exc}"
        )
    if not root.exists():
        return CkptListSubdirsResult(
            current=str(root), error="directory does not exist"
        )
    if not root.is_dir():
        return CkptListSubdirsResult(
            current=str(root), error="path is not a directory"
        )

    subdirs: list[str] = []
    try:
        for p in root.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                subdirs.append(p.name)
    except Exception as exc:
        return CkptListSubdirsResult(
            current=str(root), error=f"read failed: {exc}"
        )

    subdirs.sort()
    parent = str(root.parent) if root.parent != root else None
    return CkptListSubdirsResult(
        current=str(root),
        parent=parent,
        subdirs=subdirs,
    )


__all__ = [
    "CkptListHistoryParams",
    "CkptHistoryEntry",
    "CkptListHistoryResult",
    "ckpt_list_history",
    "CkptListSubdirsParams",
    "CkptListSubdirsResult",
    "ckpt_list_subdirs",
]
