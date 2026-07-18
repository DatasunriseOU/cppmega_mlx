from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


MLX_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MLX_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_cpp_revision(repo: Path, revision: int) -> None:
    source = repo / "src" / "cache.cpp"
    source.parent.mkdir(parents=True, exist_ok=True)
    body = ["int cache_contract(int value) {"]
    body.extend(
        f"    value += {revision * 100 + offset};" for offset in range(1, 20)
    )
    body.extend(["    return value;", "}", ""])
    source.write_text("\n".join(body), encoding="utf-8")


def _make_git_repo(root: Path) -> Path:
    repo = root / "cache-repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Cache Test")
    _git(repo, "config", "user.email", "cache@example.invalid")
    _git(repo, "remote", "add", "origin", "https://github.com/tests/cache-repo.git")
    _write_cpp_revision(repo, 0)
    _git(repo, "add", "src/cache.cpp")
    _git(repo, "commit", "-qm", "initial source")
    _write_cpp_revision(repo, 1)
    _git(repo, "add", "src/cache.cpp")
    _git(repo, "commit", "-qm", "modify source")
    return repo


def _set_external_cache(monkeypatch, module, root: Path) -> None:
    monkeypatch.setattr(module, "EXTRACT_CACHE_ROOT", root.resolve())
    monkeypatch.setattr(
        module,
        "EXTRACT_CACHE_MODE",
        module.EXTRACT_CACHE_MODE_EXTERNAL,
    )


def _ensure(module, repo: Path, run_root: Path):
    return module.ensure_commit_records(
        repo.name,
        repo,
        run_root / "work",
        run_root / "work-parent",
        module.Manifest(run_root / "manifest.json"),
        resume=True,
    )


def test_external_cache_reuses_completed_publication_across_run_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import streaming_conveyor

    repo = _make_git_repo(tmp_path / "source")
    cache_root = tmp_path / "shared-extract-cache"
    _set_external_cache(monkeypatch, streaming_conveyor, cache_root)

    output, first_count, first_status = _ensure(
        streaming_conveyor,
        repo,
        tmp_path / "v4-run",
    )
    first_stat = output.stat()
    first_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    extracted_again = False

    def forbidden_reextract(*_args, **_kwargs):
        nonlocal extracted_again
        extracted_again = True
        raise AssertionError("completed external publication must be reused")

    monkeypatch.setattr(
        streaming_conveyor,
        "stage_extract_commits",
        forbidden_reextract,
    )

    reused, second_count, second_status = _ensure(
        streaming_conveyor,
        repo,
        tmp_path / "v5-run",
    )

    assert first_status == "fresh"
    assert second_status == "hit"
    assert first_count == second_count > 0
    assert extracted_again is False
    assert reused == output
    assert reused.stat().st_mtime_ns == first_stat.st_mtime_ns
    publication = json.loads(
        (
            Path(str(output) + ".extract-checkpoint") / "publication.json"
        ).read_text(encoding="utf-8")
    )
    assert publication["status"] == "done"
    assert publication["output"]["sha256"] == first_digest
    assert streaming_conveyor.extract_cache_access_receipt(second_status) == {
        "root": str(cache_root.resolve()),
        "mode": "external",
        "status": "hit",
        "hit": True,
        "reused": True,
    }


def test_external_cache_reuses_legacy_directory_identity_with_explicit_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import streaming_conveyor

    repo = _make_git_repo(tmp_path / "source")
    _git(repo, "remote", "remove", "origin")
    cache_root = tmp_path / "shared-extract-cache"
    output = cache_root / repo.name / f"{repo.name}_commits.jsonl"
    output.parent.mkdir(parents=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(streaming_conveyor.EXTRACT_GIT),
            "--repo",
            str(repo),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    first_record = json.loads(output.read_text().splitlines()[0])
    assert first_record["repo"] == repo.name

    _set_external_cache(monkeypatch, streaming_conveyor, cache_root)
    monkeypatch.setattr(
        streaming_conveyor,
        "stage_extract_commits",
        lambda *_args, **_kwargs: pytest.fail("legacy publication must be reused"),
    )

    reused, count, status = streaming_conveyor.ensure_commit_records(
        repo.name,
        repo,
        tmp_path / "work",
        tmp_path / "work-parent",
        streaming_conveyor.Manifest(tmp_path / "manifest.json"),
        resume=True,
        project_id="tests/cache-repo",
    )

    assert reused == output
    assert count > 0
    assert status == "hit_legacy_identity_override"
    assert streaming_conveyor.extract_cache_access_receipt(status) == {
        "root": str(cache_root.resolve()),
        "mode": "external",
        "status": "hit_legacy_identity_override",
        "hit": False,
        "reused": True,
        "legacy_identity_override": True,
    }


def test_external_cache_rejects_stale_head_without_reextracting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import streaming_conveyor

    repo = _make_git_repo(tmp_path / "source")
    _set_external_cache(
        monkeypatch,
        streaming_conveyor,
        tmp_path / "shared-extract-cache",
    )
    output, _count, _status = _ensure(
        streaming_conveyor,
        repo,
        tmp_path / "v4-run",
    )
    published_digest = hashlib.sha256(output.read_bytes()).hexdigest()

    _write_cpp_revision(repo, 2)
    _git(repo, "add", "src/cache.cpp")
    _git(repo, "commit", "-qm", "advance head")
    monkeypatch.setattr(
        streaming_conveyor,
        "stage_extract_commits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale completed cache must fail, not re-extract")
        ),
    )

    with pytest.raises(streaming_conveyor.RepoFailure) as caught:
        _ensure(streaming_conveyor, repo, tmp_path / "v5-run")

    assert caught.value.stage == "extract_cache_validate"
    assert "does not match the current repository/config" in caught.value.detail
    assert hashlib.sha256(output.read_bytes()).hexdigest() == published_digest


def test_external_cache_lock_serializes_processes(tmp_path: Path) -> None:
    cache_root = tmp_path / "shared-extract-cache"
    ready = tmp_path / "holder-ready"
    release = tmp_path / "holder-release"
    contender_entered = tmp_path / "contender-entered"
    prefix = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
        "import streaming_conveyor as c; "
        f"c.EXTRACT_CACHE_ROOT = __import__('pathlib').Path({str(cache_root)!r}); "
        "c.EXTRACT_CACHE_MODE = c.EXTRACT_CACHE_MODE_EXTERNAL; "
    )
    holder_code = (
        prefix
        + "from pathlib import Path; import time; "
        f"ready=Path({str(ready)!r}); release=Path({str(release)!r}); "
        "lock=c.extract_cache_repo_lock('repo'); lock.__enter__(); "
        "ready.write_text('ready'); "
        "\nwhile not release.exists(): time.sleep(0.02)\n"
        "lock.__exit__(None, None, None)\n"
    )
    contender_code = (
        prefix
        + "from pathlib import Path; "
        "lock=c.extract_cache_repo_lock('repo'); lock.__enter__(); "
        f"Path({str(contender_entered)!r}).write_text('entered'); "
        "lock.__exit__(None, None, None)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    contender = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            if holder.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.exists(), holder.communicate(timeout=1)

        contender = subprocess.Popen(
            [sys.executable, "-c", contender_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.25)
        assert contender.poll() is None
        assert not contender_entered.exists()

        release.write_text("release", encoding="utf-8")
        holder_stdout, holder_stderr = holder.communicate(timeout=10)
        contender_stdout, contender_stderr = contender.communicate(timeout=10)
        assert (holder.returncode, holder_stdout, holder_stderr) == (0, "", "")
        assert (contender.returncode, contender_stdout, contender_stderr) == (
            0,
            "",
            "",
        )
        assert contender_entered.read_text(encoding="utf-8") == "entered"
    finally:
        release.touch()
        if holder.poll() is None:
            holder.kill()
            holder.wait()
        if contender is not None and contender.poll() is None:
            contender.kill()
            contender.wait()


@pytest.mark.parametrize("fail_repo", [False, True], ids=["success", "failure"])
def test_process_one_repo_never_deletes_external_cache(
    tmp_path: Path,
    monkeypatch,
    fail_repo: bool,
) -> None:
    import streaming_conveyor

    repo = "repo"
    work_root = tmp_path / "run" / "work"
    repo_dir = work_root / repo / "_src"
    repo_dir.mkdir(parents=True)
    cache_root = tmp_path / "shared-extract-cache"
    marker = cache_root / repo / "retained.marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("durable", encoding="utf-8")
    _set_external_cache(monkeypatch, streaming_conveyor, cache_root)

    def commits_half(*_args, **_kwargs):
        if fail_repo:
            raise streaming_conveyor.RepoFailure(repo, "range", "expected failure")
        return 0, 0, True

    monkeypatch.setattr(streaming_conveyor, "run_commits_half", commits_half)
    manifest = streaming_conveyor.Manifest(tmp_path / "manifest.json")
    with ThreadPoolExecutor(max_workers=1) as pool:
        streaming_conveyor.process_one_repo(
            repo=repo,
            repo_dir=repo_dir,
            lengths_code=(),
            lengths_commits=(1024,),
            range_size=500,
            range_target_bytes=0,
            work_root=work_root,
            work_parent=tmp_path / "run" / "work-parent",
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            keep_temp=False,
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=None,
            streams="commits",
        )

    assert not (work_root / repo).exists()
    assert marker.read_text(encoding="utf-8") == "durable"
    assert (f"{repo}::commits" in manifest.failed) is fail_repo


@pytest.mark.parametrize(
    "cache_state",
    ["filename_only", "corrupt_publication"],
)
def test_corrupt_external_cache_fails_without_extraction(
    tmp_path: Path,
    monkeypatch,
    cache_state: str,
) -> None:
    import streaming_conveyor

    repo = "repo"
    cache_root = tmp_path / "shared-extract-cache"
    output = cache_root / repo / f"{repo}_commits.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text('{"untrusted": true}\n', encoding="utf-8")
    if cache_state == "corrupt_publication":
        publication = Path(str(output) + ".extract-checkpoint") / "publication.json"
        publication.parent.mkdir(parents=True)
        publication.write_text("{not-json", encoding="utf-8")
    _set_external_cache(monkeypatch, streaming_conveyor, cache_root)
    extracted = False

    def forbidden_extract(*_args, **_kwargs):
        nonlocal extracted
        extracted = True
        raise AssertionError("corrupt external cache must not be replaced")

    monkeypatch.setattr(
        streaming_conveyor,
        "stage_extract_commits",
        forbidden_extract,
    )
    expected = (
        "refusing filename-only reuse or automatic replacement"
        if cache_state == "filename_only"
        else "invalid external extraction publication"
    )
    with pytest.raises(streaming_conveyor.RepoFailure, match=expected):
        streaming_conveyor.ensure_commit_records(
            repo,
            tmp_path / "repo-src",
            tmp_path / "work",
            tmp_path / "work-parent",
            streaming_conveyor.Manifest(tmp_path / "manifest.json"),
            resume=True,
        )

    assert extracted is False


def test_external_cache_flag_and_per_repo_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import streaming_conveyor

    for name in (
        "CONVEYOR_ROOT",
        "CONVEYOR_MANIFEST",
        "DEFAULT_PROGRESS_JSONL",
        "DEFAULT_RUN_LOCK_DIR",
        "DEFAULT_WORK_PARENT",
        "DEFAULT_RESERVATION_FILE",
        "EXTRACT_CACHE_ROOT",
        "EXTRACT_CACHE_MODE",
    ):
        monkeypatch.setattr(
            streaming_conveyor,
            name,
            getattr(streaming_conveyor, name),
        )
    cache_root = tmp_path / "shared-extract-cache"
    args = streaming_conveyor.parse_args(
        [
            "--conveyor-root",
            str(tmp_path / "v5-conveyor"),
            "--extract-cache-root",
            str(cache_root),
        ]
    )
    streaming_conveyor.configure_runtime_paths_from_args(args)
    assert streaming_conveyor.EXTRACT_CACHE_ROOT == cache_root.resolve()
    assert (
        streaming_conveyor.EXTRACT_CACHE_MODE
        == streaming_conveyor.EXTRACT_CACHE_MODE_EXTERNAL
    )

    repo = "repo"
    records = cache_root / repo / f"{repo}_commits.jsonl"
    records.parent.mkdir(parents=True)
    records.write_text("{}\n", encoding="utf-8")

    def records_provider(*_args):
        return records, 1, "hit"

    def range_runner(
        range_repo,
        _repo_dir,
        _records_jsonl,
        start,
        end,
        lengths,
        *_args,
        **_kwargs,
    ):
        target = str(lengths[0])
        return {
            "source": "commits",
            "repo": range_repo,
            "range": [start, end],
            "lengths": {
                target: {
                    "rows": 1,
                    "valid_tokens": 8,
                    "pad_tokens": 0,
                    "capacity_tokens": 8,
                }
            },
            "stage_timings_s": {},
        }

    manifest = streaming_conveyor.Manifest(tmp_path / "manifest.json")
    progress = streaming_conveyor.ProgressWriter(tmp_path / "progress.jsonl")
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = streaming_conveyor.run_commits_half(
            repo=repo,
            repo_dir=tmp_path / "repo-src",
            repo_work=tmp_path / "repo-work",
            work_root=tmp_path / "work",
            work_parent=tmp_path / "work-parent",
            lengths_commits=(1024,),
            range_size=500,
            pool=pool,
            manifest=manifest,
            manifest_lock=streaming_conveyor.threading.Lock(),
            resume=True,
            cumulative={"valid": 0},
            dedup_db=None,
            dedup_near=False,
            pr_store=None,
            repo_list=None,
            progress=progress,
            range_target_bytes=0,
            range_runner_override=range_runner,
            commit_records_override=records_provider,
        )

    expected = {
        "root": str(cache_root.resolve()),
        "mode": "external",
        "status": "hit",
        "hit": True,
        "reused": True,
    }
    assert result == (1, 0, True)
    assert manifest.done[f"{repo}::commit_plan"]["extract_cache"] == expected
    assert manifest.done[f"{repo}::r0"]["extract_cache"] == expected
    assert manifest.done[f"{repo}::commits"]["extract_cache"] == expected
    cache_event = next(
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "extract_cache"
    )
    assert {key: cache_event[key] for key in expected} == expected
