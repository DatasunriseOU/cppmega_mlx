from __future__ import annotations

from pathlib import Path

import pytest


def _stats(path: Path, target_length: int) -> dict:
    return {
        "rows": 1,
        "capacity_tokens": target_length,
        "valid_tokens": target_length - 1,
        "pad_tokens": 1,
        "pad_frac": 1 / target_length,
        "payload": path.read_text(encoding="utf-8"),
    }


def _assert_no_publication_temps(root: Path) -> None:
    assert not list(root.rglob("*.staged.parquet"))
    assert not list(root.rglob("*.backup.parquet"))
    transaction_root = root / ".transactions"
    assert not transaction_root.exists() or not list(transaction_root.iterdir())


def test_code_bucket_publication_rolls_back_every_bucket_on_late_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import streaming_reindex as sr

    output_root = tmp_path / "outputs"
    packed_by_length: dict[int, Path] = {}
    for target_length in (8, 16):
        packed = tmp_path / f"new-{target_length}.parquet"
        packed.write_text(f"new-{target_length}", encoding="utf-8")
        packed_by_length[target_length] = packed
        destination = output_root / str(target_length) / "repo.parquet"
        destination.parent.mkdir(parents=True)
        destination.write_text(f"old-{target_length}", encoding="utf-8")
    stale_destination = output_root / "32" / "repo.parquet"
    stale_destination.parent.mkdir(parents=True)
    stale_destination.write_text("old-32", encoding="utf-8")

    real_replace = sr._replace_publication_path
    staged_replaces = 0

    def fail_second_staged_replace(source: Path, destination: Path) -> None:
        nonlocal staged_replaces
        if "staged" in source.parts:
            staged_replaces += 1
            if staged_replaces == 2:
                raise OSError("late bucket replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(sr, "_replace_publication_path", fail_second_staged_replace)
    with pytest.raises(OSError, match="late bucket replace failure"):
        sr.publish_bucket_outputs_atomically(
            "repo",
            packed_by_length,
            output_root=output_root,
            filename="repo.parquet",
            stats_reader=_stats,
            remove_lengths=(32,),
        )

    assert (output_root / "8" / "repo.parquet").read_text() == "old-8"
    assert (output_root / "16" / "repo.parquet").read_text() == "old-16"
    assert stale_destination.read_text() == "old-32"
    _assert_no_publication_temps(output_root)


def test_successful_publication_removes_stale_bucket_file(tmp_path: Path) -> None:
    from scripts import streaming_reindex as sr

    output_root = tmp_path / "outputs"
    packed = tmp_path / "new-16.parquet"
    packed.write_text("new-16", encoding="utf-8")
    for target_length in (8, 16):
        destination = output_root / str(target_length) / "repo.parquet"
        destination.parent.mkdir(parents=True)
        destination.write_text(f"old-{target_length}", encoding="utf-8")

    stats = sr.publish_bucket_outputs_atomically(
        "repo",
        {16: packed},
        output_root=output_root,
        filename="repo.parquet",
        stats_reader=_stats,
        remove_lengths=(8,),
    )

    assert not (output_root / "8" / "repo.parquet").exists()
    assert (output_root / "16" / "repo.parquet").read_text() == "new-16"
    assert set(stats) == {"16"}
    _assert_no_publication_temps(output_root)


def test_commit_bucket_preparation_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import streaming_reindex_commits as src

    output_root = tmp_path / "commit-outputs"
    monkeypatch.setattr(src, "COMMIT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(src, "_parquet_stats", _stats)
    packed_by_length: dict[int, Path] = {}
    for target_length in (8, 16):
        packed = tmp_path / f"commit-new-{target_length}.parquet"
        packed.write_text(f"new-{target_length}", encoding="utf-8")
        packed_by_length[target_length] = packed
        destination = output_root / str(target_length) / "repo_r0.parquet"
        destination.parent.mkdir(parents=True)
        destination.write_text(f"old-{target_length}", encoding="utf-8")

    preparations = 0

    def fail_second_prepare(path: Path) -> None:
        nonlocal preparations
        preparations += 1
        if preparations == 2:
            raise RuntimeError("late recompress failure")
        path.write_text(path.read_text() + "-zstd", encoding="utf-8")

    monkeypatch.setattr(src, "recompress_zstd_max", fail_second_prepare)
    with pytest.raises(RuntimeError, match="late recompress failure"):
        src.publish_range_outputs("repo", 0, packed_by_length)

    assert (output_root / "8" / "repo_r0.parquet").read_text() == "old-8"
    assert (output_root / "16" / "repo_r0.parquet").read_text() == "old-16"
    _assert_no_publication_temps(output_root)


def test_manifest_failure_and_restart_remove_stale_done_state(tmp_path: Path) -> None:
    from scripts import streaming_reindex as sr

    path = tmp_path / "_done.json"
    manifest = sr.Manifest(
        path=path,
        done={"repo": {"old": True}, "repo::r0": {"old": True}},
        failed={},
    )
    manifest.mark_failed("repo", "pack", "late failure")
    manifest.mark_started_prefix("repo::r")

    reloaded = sr.Manifest.load(path)
    assert "repo" not in reloaded.done
    assert "repo::r0" not in reloaded.done
    assert reloaded.failed["repo"]["stage"] == "pack"
