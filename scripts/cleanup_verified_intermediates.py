#!/usr/bin/env python3
"""Delete extract-cache JSONL only after lossless parquet verification.

The cleanup contract is deliberately fail-closed:

* the extractor publication must bind the JSONL path, size, SHA-256, and lines;
* a conveyor manifest must prove exact, gap-free commit-range completion;
* every non-empty range parquet must be ZSTD-compressed and match the manifest;
* the per-repository conveyor lock must be available for the whole audit/delete;
* an external receipt is written before any directory is removed.

Dry-run is the default. Deletion additionally requires ``--execute`` and
``--receipt``. Source-cache deletion is intentionally unsupported until an
equivalent source-to-parquet receipt exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, BinaryIO, Iterable

import pyarrow.compute as pc
import pyarrow.parquet as pq

OUTPUT_BASE = Path("outputs")
EXTRACT_CACHE = OUTPUT_BASE / "extract_cache_case5_shared"
DEFAULT_PROOFS = (
    (
        OUTPUT_BASE / "conveyor_case5_v11_commits_20260719_180000" / "_done.json",
        OUTPUT_BASE / "reindexed_case5_v7_20260715_130725_commits",
    ),
    (
        OUTPUT_BASE / "conveyor_case5_v8_retry_blender_20260716_0045" / "_done.json",
        OUTPUT_BASE / "reindexed_case5_v8_retry_blender_20260716_0045_commits",
    ),
    (
        OUTPUT_BASE / "conveyor_case5_v4_20260714_093120" / "_done.json",
        OUTPUT_BASE / "reindexed_case5_v4_20260714_093120_commits",
    ),
)
REQUIRED_PARQUET_COLUMNS = (
    "valid_token_count",
    "trained_token_count",
    "slack_tokens",
)


class VerificationError(RuntimeError):
    """A candidate cannot be deleted without risking data loss."""


@dataclass(frozen=True)
class Proof:
    manifest_path: Path
    parquet_root: Path
    manifest_sha256: str
    done: dict[str, Any]


@dataclass
class HeldLock:
    path: Path
    handle: BinaryIO

    def close(self) -> None:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_and_lines(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
    return digest.hexdigest(), lines


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise VerificationError(f"{label} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise VerificationError(f"{label} must be a regular non-symlink file: {path}")
    return info


def _contained_direct_child(root: Path, child: Path) -> None:
    root_resolved = root.resolve(strict=True)
    child_resolved = child.resolve(strict=True)
    if child_resolved.parent != root_resolved:
        raise VerificationError(
            f"cleanup target is not a direct child of cache root: {child}"
        )


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_proof(value: str) -> tuple[Path, Path]:
    manifest, separator, parquet_root = value.partition("=")
    if not separator or not manifest or not parquet_root:
        raise argparse.ArgumentTypeError(
            "proof must be MANIFEST_PATH=COMMIT_PARQUET_ROOT"
        )
    return Path(manifest), Path(parquet_root)


def load_proofs(specifications: Iterable[tuple[Path, Path]]) -> list[Proof]:
    proofs: list[Proof] = []
    for manifest_path, parquet_root in specifications:
        manifest_path = manifest_path.resolve(strict=True)
        parquet_root = parquet_root.resolve(strict=True)
        _regular_file(manifest_path, label="manifest")
        payload = _read_json_object(manifest_path)
        done = payload.get("done")
        if not isinstance(done, dict):
            raise VerificationError(f"manifest has no done object: {manifest_path}")
        proofs.append(
            Proof(
                manifest_path=manifest_path,
                parquet_root=parquet_root,
                manifest_sha256=_sha256_file(manifest_path),
                done=done,
            )
        )
    if not proofs:
        raise VerificationError("at least one proof is required")
    return proofs


def select_exact_proof(repo: str, proofs: Iterable[Proof]) -> Proof | None:
    for proof in proofs:
        completion = proof.done.get(f"{repo}::commits")
        if (
            isinstance(completion, dict)
            and completion.get("complete") is True
            and completion.get("completion_proof") == "commit_plan_exact_range_coverage"
        ):
            return proof
    return None


def _validated_ranges(repo: str, proof: Proof, expected_records: int) -> list[dict]:
    completion = proof.done[f"{repo}::commits"]
    plan = proof.done.get(f"{repo}::commit_plan")
    if not isinstance(plan, dict):
        raise VerificationError(f"{repo}: missing commit_plan in {proof.manifest_path}")
    if completion.get("n_records") != expected_records:
        raise VerificationError(
            f"{repo}: completion records do not match publication: "
            f"{completion.get('n_records')} != {expected_records}"
        )
    if plan.get("n_records") != expected_records:
        raise VerificationError(
            f"{repo}: plan records do not match publication: "
            f"{plan.get('n_records')} != {expected_records}"
        )

    ranges: list[dict] = []
    prefix = f"{repo}::r"
    for key, value in proof.done.items():
        if (
            isinstance(key, str)
            and key.startswith(prefix)
            and isinstance(value, dict)
            and value.get("source") == "commits"
        ):
            bounds = value.get("range")
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(isinstance(item, int) for item in bounds)
            ):
                raise VerificationError(f"{repo}: invalid range receipt {key}")
            ranges.append({"key": key, "start": bounds[0], "end": bounds[1], **value})

    ranges.sort(key=lambda item: (item["start"], item["end"]))
    cursor = 0
    for item in ranges:
        start = item["start"]
        end = item["end"]
        if start != cursor or end <= start or item.get("n_records") != end - start:
            raise VerificationError(
                f"{repo}: non-exact range coverage at {item['key']}: "
                f"cursor={cursor}, range=[{start}, {end}], "
                f"records={item.get('n_records')}"
            )
        expected_stage = f"commit:{repo}:r{start}:{end}"
        stage = item.get("dedup_stage_promoted")
        empty_after_dedup = item.get("empty_after_dedup") is True
        lengths = item.get("lengths")
        if not isinstance(lengths, dict):
            raise VerificationError(f"{repo}: missing length receipt at {item['key']}")
        if stage != {"stage_id": expected_stage}:
            nonempty_rows = sum(
                int(metrics.get("rows", 0))
                for metrics in lengths.values()
                if isinstance(metrics, dict)
            )
            if not empty_after_dedup or nonempty_rows != 0:
                raise VerificationError(
                    f"{repo}: range has no promoted dedup stage: {item['key']}"
                )
        cursor = end

    if cursor != expected_records:
        raise VerificationError(
            f"{repo}: ranges end at {cursor}, expected {expected_records}"
        )
    if completion.get("range_count") != len(ranges):
        raise VerificationError(
            f"{repo}: range_count mismatch: "
            f"{completion.get('range_count')} != {len(ranges)}"
        )
    return ranges


def _verify_parquet(path: Path, *, length: int, metrics: dict[str, Any]) -> dict:
    file_info = _regular_file(path, label="commit parquet")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise VerificationError(f"cannot open parquet {path}: {exc}") from exc

    missing = set(REQUIRED_PARQUET_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise VerificationError(f"{path}: missing columns {sorted(missing)}")
    codecs = {
        parquet.metadata.row_group(row_group).column(column).compression
        for row_group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }
    if codecs != {"ZSTD"}:
        raise VerificationError(f"{path}: expected only ZSTD, found {sorted(codecs)}")

    rows = int(metrics.get("rows", -1))
    valid_tokens = int(metrics.get("valid_tokens", -1))
    pad_tokens = int(metrics.get("pad_tokens", -1))
    capacity_tokens = int(metrics.get("capacity_tokens", -1))
    if rows <= 0 or capacity_tokens != rows * length:
        raise VerificationError(f"{path}: invalid manifest row/capacity receipt")
    if (
        valid_tokens < 0
        or pad_tokens < 0
        or valid_tokens + pad_tokens != capacity_tokens
    ):
        raise VerificationError(f"{path}: invalid manifest token accounting")
    if parquet.metadata.num_rows != rows:
        raise VerificationError(
            f"{path}: parquet rows {parquet.metadata.num_rows} != receipt {rows}"
        )

    table = parquet.read(columns=list(REQUIRED_PARQUET_COLUMNS))
    actual_valid = int(pc.sum(table["valid_token_count"]).as_py() or 0)
    actual_trained = int(pc.sum(table["trained_token_count"]).as_py() or 0)
    actual_pad = int(pc.sum(table["slack_tokens"]).as_py() or 0)
    if actual_valid != valid_tokens or actual_pad != pad_tokens:
        raise VerificationError(
            f"{path}: parquet token totals do not match receipt: "
            f"valid={actual_valid}/{valid_tokens}, pad={actual_pad}/{pad_tokens}"
        )
    row_capacity_ok = bool(
        pc.all(
            pc.equal(
                pc.add(table["valid_token_count"], table["slack_tokens"]),
                length,
            )
        ).as_py()
    )
    trained_bounds_ok = bool(
        pc.all(
            pc.and_(
                pc.greater_equal(table["trained_token_count"], 0),
                pc.less_equal(
                    table["trained_token_count"],
                    table["valid_token_count"],
                ),
            )
        ).as_py()
    )
    if not row_capacity_ok:
        raise VerificationError(
            f"{path}: one or more rows violate valid_token_count + slack_tokens "
            f"== {length}"
        )
    if not trained_bounds_ok:
        raise VerificationError(
            f"{path}: one or more rows violate 0 <= trained_token_count "
            "<= valid_token_count"
        )
    return {
        "bytes": file_info.st_size,
        "rows": rows,
        "valid_tokens": actual_valid,
        "trained_tokens": actual_trained,
        "pad_tokens": actual_pad,
    }


def acquire_repo_lock(repo_dir: Path) -> HeldLock:
    lock_path = repo_dir / ".conveyor-cache.lock"
    _regular_file(lock_path, label="conveyor cache lock")
    handle = lock_path.open("rb")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise VerificationError(
            f"cache entry is active; lock is busy: {lock_path}"
        ) from exc
    return HeldLock(lock_path, handle)


def verify_repo(repo_dir: Path, proof: Proof) -> dict[str, Any]:
    repo = repo_dir.name
    source = repo_dir / f"{repo}_commits.jsonl"
    publication_path = Path(f"{source}.extract-checkpoint") / "publication.json"
    _regular_file(source, label="published commit JSONL")
    _regular_file(publication_path, label="extractor publication")
    publication = _read_json_object(publication_path)
    output = publication.get("output")
    if publication.get("status") != "done" or not isinstance(output, dict):
        raise VerificationError(f"{repo}: publication is not status=done")
    try:
        publication_path_value = Path(publication["output_path"]).resolve(strict=True)
        expected_size = int(output["size_bytes"])
        expected_lines = int(output["line_count"])
        expected_sha256 = str(output["sha256"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise VerificationError(f"{repo}: malformed publication receipt") from exc
    if publication_path_value != source.resolve(strict=True):
        raise VerificationError(
            f"{repo}: publication points at {publication_path_value}, not {source}"
        )
    if source.stat().st_size != expected_size:
        raise VerificationError(f"{repo}: JSONL size does not match publication")

    actual_sha256, actual_lines = _sha256_and_lines(source)
    if actual_sha256 != expected_sha256 or actual_lines != expected_lines:
        raise VerificationError(
            f"{repo}: JSONL digest/lines do not match publication: "
            f"sha={actual_sha256}/{expected_sha256}, "
            f"lines={actual_lines}/{expected_lines}"
        )

    ranges = _validated_ranges(repo, proof, expected_lines)
    parquet_totals = {
        "files": 0,
        "bytes": 0,
        "rows": 0,
        "valid_tokens": 0,
        "trained_tokens": 0,
        "pad_tokens": 0,
    }
    for item in ranges:
        for raw_length, metrics in item["lengths"].items():
            if not isinstance(metrics, dict):
                raise VerificationError(
                    f"{repo}: invalid length receipt at {item['key']}"
                )
            rows = int(metrics.get("rows", 0))
            if rows == 0:
                continue
            length = int(raw_length)
            parquet_path = (
                proof.parquet_root / str(length) / f"{repo}_r{item['start']}.parquet"
            )
            verified = _verify_parquet(parquet_path, length=length, metrics=metrics)
            parquet_totals["files"] += 1
            for key, value in verified.items():
                parquet_totals[key] += value

    return {
        "repo": repo,
        "cache_path": str(repo_dir.resolve(strict=True)),
        "source": {
            "path": str(source.resolve(strict=True)),
            "bytes": expected_size,
            "lines": expected_lines,
            "sha256": actual_sha256,
        },
        "publication": {
            "path": str(publication_path.resolve(strict=True)),
            "job_fingerprint": publication.get("job_fingerprint"),
            "written_at": publication.get("written_at"),
        },
        "proof": {
            "manifest": str(proof.manifest_path),
            "manifest_sha256": proof.manifest_sha256,
            "parquet_root": str(proof.parquet_root),
            "ranges": len(ranges),
        },
        "parquet": parquet_totals,
        "status": "verified",
    }


def build_plan(
    cache_root: Path, proofs: list[Proof]
) -> tuple[list[dict], list[dict], list[HeldLock]]:
    cache_root = cache_root.resolve(strict=True)
    if not cache_root.is_dir():
        raise VerificationError(f"extract cache is not a directory: {cache_root}")
    verified: list[dict] = []
    blocked: list[dict] = []
    locks: list[HeldLock] = []
    try:
        for repo_dir in sorted(cache_root.iterdir()):
            if not repo_dir.is_dir() or repo_dir.is_symlink():
                blocked.append(
                    {"path": str(repo_dir), "reason": "not_direct_directory"}
                )
                continue
            _contained_direct_child(cache_root, repo_dir)
            repo = repo_dir.name
            source = repo_dir / f"{repo}_commits.jsonl"
            publication = Path(f"{source}.extract-checkpoint") / "publication.json"
            if not publication.is_file():
                blocked.append(
                    {
                        "repo": repo,
                        "path": str(repo_dir),
                        "reason": "no_done_publication",
                    }
                )
                continue
            proof = select_exact_proof(repo, proofs)
            if proof is None:
                blocked.append(
                    {
                        "repo": repo,
                        "path": str(repo_dir),
                        "reason": "no_exact_range_completion",
                    }
                )
                continue
            lock = acquire_repo_lock(repo_dir)
            locks.append(lock)
            verified.append(verify_repo(repo_dir, proof))
            print(
                f"verified {repo}: {verified[-1]['source']['bytes'] / 2**30:.3f} GiB, "
                f"{verified[-1]['parquet']['files']} parquet files",
                flush=True,
            )
    except Exception:
        for lock in reversed(locks):
            lock.close()
        raise
    return verified, blocked, locks


def execute_cleanup(
    cache_root: Path, receipt_path: Path, receipt: dict[str, Any]
) -> None:
    cache_root = cache_root.resolve(strict=True)
    receipt_path = receipt_path.resolve()
    try:
        receipt_path.relative_to(cache_root)
    except ValueError:
        pass
    else:
        raise VerificationError("receipt must be outside the cache being deleted")

    receipt["status"] = "validated"
    receipt["validated_at"] = _utc_now()
    _write_json_atomic(receipt_path, receipt)
    for entry in receipt["verified"]:
        repo_dir = Path(entry["cache_path"])
        _contained_direct_child(cache_root, repo_dir)
        shutil.rmtree(repo_dir)
        if repo_dir.exists():
            raise VerificationError(f"cleanup target still exists: {repo_dir}")
        entry["status"] = "removed"
        entry["removed_at"] = _utc_now()
        _write_json_atomic(receipt_path, receipt)
    receipt["status"] = "complete"
    receipt["completed_at"] = _utc_now()
    _write_json_atomic(receipt_path, receipt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-cache", action="store_true")
    parser.add_argument("--source-cache", action="store_true")
    parser.add_argument("--extract-cache-root", type=Path, default=EXTRACT_CACHE)
    parser.add_argument(
        "--proof",
        action="append",
        type=_parse_proof,
        help="MANIFEST_PATH=COMMIT_PARQUET_ROOT; repeat in newest-first order",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    if args.source_cache:
        parser.error(
            "source-cache deletion is disabled: no source-to-parquet receipt contract exists"
        )
    if not args.extract_cache:
        parser.error("--extract-cache is required")
    if args.execute and args.receipt is None:
        parser.error("--execute requires --receipt outside the extract cache")

    specifications = args.proof if args.proof is not None else list(DEFAULT_PROOFS)
    proofs = load_proofs(specifications)
    verified, blocked, locks = build_plan(args.extract_cache_root, proofs)
    try:
        receipt = {
            "schema_version": 1,
            "mode": "execute" if args.execute else "dry_run",
            "status": "planned",
            "generated_at": _utc_now(),
            "cache_root": str(args.extract_cache_root.resolve(strict=True)),
            "planned_repos": len(verified),
            "planned_bytes": sum(item["source"]["bytes"] for item in verified),
            "blocked": blocked,
            "verified": verified,
        }
        print(
            f"plan: {len(verified)} verified repos, "
            f"{receipt['planned_bytes'] / 2**30:.3f} GiB; "
            f"{len(blocked)} blocked",
            flush=True,
        )
        for entry in blocked:
            print(
                f"blocked {entry.get('repo', entry['path'])}: {entry['reason']}",
                flush=True,
            )
        if args.execute:
            execute_cleanup(args.extract_cache_root, args.receipt, receipt)
            print(f"cleanup complete; receipt={args.receipt.resolve()}", flush=True)
        elif args.receipt is not None:
            _write_json_atomic(args.receipt.resolve(), receipt)
            print(f"dry-run receipt={args.receipt.resolve()}", flush=True)
    finally:
        for lock in reversed(locks):
            lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
