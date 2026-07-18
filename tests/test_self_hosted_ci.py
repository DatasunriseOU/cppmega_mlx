from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from scripts import run_self_hosted_ci as ci


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "scripts" / "self_hosted_hosts.json"


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        nested = set(value)
        for child in value.values():
            nested.update(_keys(child))
        return nested
    if isinstance(value, list):
        nested: set[str] = set()
        for child in value:
            nested.update(_keys(child))
        return nested
    return set()


def _local_inventory(*, required: bool = True) -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine.lower() == "arm64":
        lane = "macos-mlx"
    elif system == "Linux" and machine.lower() in {"x86_64", "amd64"}:
        lane = "linux-portable"
    else:
        pytest.skip(f"unsupported test host: {system}/{machine}")
    return {
        "schema_version": 1,
        "repository": "test/local",
        "hosts": [
            {
                "id": "local-test",
                "address": "local",
                "transport": "local",
                "system": system.lower(),
                "machines": [machine.lower()],
                "lane": lane,
                "required": required,
                "python": sys.executable,
                "timeout_seconds": 30,
                "github": None,
            }
        ],
    }


def test_inventory_routes_required_lanes_to_registered_runner_labels() -> None:
    inventory = ci._load_inventory(INVENTORY_PATH)
    hosts = {host["id"]: host for host in inventory["hosts"]}

    assert set(hosts) == {
        "mac-studio",
        "legion-linux",
        "windows-10-0-0-11",
        "untrusted-10-0-0-12",
    }
    assert hosts["mac-studio"]["address"] == "10.0.0.8"
    assert hosts["mac-studio"]["python"] == (
        "/Volumes/external/sources/.venvs/cppmega.mlx/bin/python"
    )
    assert hosts["mac-studio"]["github"] == {
        "runner_name": "mac-studio-cppmega-mlx",
        "labels": ["self-hosted", "macOS", "ARM64", "cppmega-mlx-macos"],
    }
    assert hosts["legion-linux"]["address"] == "10.0.0.16"
    assert hosts["legion-linux"]["github"] == {
        "runner_name": "davidgor-Legion-R9000P-ARX8-cppmega-mlx",
        "labels": ["self-hosted", "Linux", "X64", "cppmega-mlx"],
    }
    assert {
        host["lane"] for host in hosts.values() if host.get("required")
    } == set(ci.LANES)
    assert hosts["windows-10-0-0-11"]["lane"] is None
    assert hosts["untrusted-10-0-0-12"]["lane"] is None


def test_inventory_contains_no_credential_fields_or_values() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    keys = {key.lower() for key in _keys(inventory)}
    serialized = json.dumps(inventory).lower()

    assert not keys.intersection(
        {"password", "passwd", "token", "secret", "private_key", "credential"}
    )
    assert "github_pat_" not in serialized
    assert "gho_" not in serialized
    assert "authorization:" not in serialized


def test_inventory_parser_rejects_credential_fields(tmp_path: Path) -> None:
    inventory = _local_inventory()
    inventory["hosts"][0]["password"] = "not-allowed"
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ci.SelfHostedCIError, match="credentials are forbidden"):
        ci._load_inventory(path)


def test_inventory_parser_rejects_required_host_without_lane(tmp_path: Path) -> None:
    inventory = _local_inventory()
    inventory["hosts"][0]["lane"] = None
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ci.SelfHostedCIError, match="has no supported lane"):
        ci._load_inventory(path)


def test_ssh_transport_is_noninteractive_and_has_no_password_argument() -> None:
    host = ci._load_inventory(INVENTORY_PATH)["hosts"][1]
    command = ci._ssh_base(host, 7)
    rendered = " ".join(command)

    assert "BatchMode=yes" in rendered
    assert "PasswordAuthentication=no" in rendered
    assert "KbdInteractiveAuthentication=no" in rendered
    assert "ConnectTimeout=7" in rendered
    assert "password" not in host


@pytest.mark.parametrize(
    ("message", "classification"),
    [
        ("Host key verification failed.", "host_key_verification_failed"),
        ("Permission denied (publickey).", "ssh_authentication_failed"),
        ("connect to host x port 22: Connection refused", "ssh_connection_refused"),
        ("ssh: connect to host x port 22: Operation timed out", "ssh_timeout"),
    ],
)
def test_ssh_failures_are_classified_explicitly(
    message: str, classification: str
) -> None:
    assert ci.classify_ssh_failure(message) == classification


def test_lane_platform_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ci.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ci.platform, "machine", lambda: "x86_64")

    with pytest.raises(ci.SelfHostedCIError, match="requires Darwin/arm64"):
        ci._validate_lane_platform(ci.LANES["macos-mlx"])


def test_step_timeout_terminates_the_process_group(tmp_path: Path) -> None:
    started = time.monotonic()
    result = ci.run_step(
        name="bounded-sleep",
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
        cwd=tmp_path,
        log_path=tmp_path / "bounded-sleep.log",
        timeout_seconds=0.1,
    )

    assert result["status"] == "timed_out"
    assert result["exit_code"] == 124
    assert time.monotonic() - started < 5


def test_direct_cli_dry_run_writes_a_passed_local_probe(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(_local_inventory()), encoding="utf-8")
    receipt_base = tmp_path / "receipts"

    exit_code = ci.main(
        [
            "orchestrate",
            "--inventory",
            str(inventory_path),
            "--repo-root",
            str(REPO_ROOT),
            "--receipt-dir",
            str(receipt_base),
            "--run-id",
            "local-dry-run",
            "--dry-run",
        ]
    )

    receipt = json.loads(
        (receipt_base / "local-dry-run" / "orchestration.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 0
    assert receipt["status"] == "dry_run_passed"
    assert receipt["required_unavailable_hosts"] == []
    assert receipt["probes"][0]["status"] == "available"


def test_direct_cli_blocks_an_unreachable_required_host(tmp_path: Path) -> None:
    inventory = _local_inventory()
    host = inventory["hosts"][0]
    host.update(
        {
            "id": "missing-linux",
            "address": "127.0.0.1",
            "transport": "ssh",
            "ssh_target": "nobody@127.0.0.1",
            "ssh_port": 1,
            "system": "linux",
            "machines": ["x86_64"],
            "lane": "linux-portable",
            "python": "python3",
        }
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    receipt_base = tmp_path / "receipts"

    exit_code = ci.main(
        [
            "orchestrate",
            "--inventory",
            str(inventory_path),
            "--repo-root",
            str(REPO_ROOT),
            "--receipt-dir",
            str(receipt_base),
            "--run-id",
            "blocked-dry-run",
            "--connect-timeout",
            "1",
            "--dry-run",
        ]
    )

    receipt = json.loads(
        (receipt_base / "blocked-dry-run" / "orchestration.json").read_text(
            encoding="utf-8"
        )
    )
    assert exit_code == 2
    assert receipt["status"] == "blocked_unavailable_hosts"
    assert receipt["required_unavailable_hosts"] == ["missing-linux"]
    assert receipt["probes"][0]["status"] == "unavailable"


def test_both_lane_manifests_run_the_orchestration_regressions() -> None:
    assert "tests/test_self_hosted_ci.py" in ci.MACOS_TESTS
    assert "tests/test_self_hosted_ci.py" in ci.LINUX_TESTS
    assert "tests/test_workflow_runner_policy.py" in ci.MACOS_TESTS
    assert "tests/test_workflow_runner_policy.py" in ci.LINUX_TESTS


def test_lane_manifests_cover_current_case5_and_training_regressions() -> None:
    for test_path in (
        "tests/test_case5_domain_ingestion.py",
        "tests/test_convert_megatron_dense500m_torchdist_to_mlx.py",
        "tests/test_materialize_megatron_dependency_provenance.py",
        "tests/test_production_objective_mixer.py",
        "tests/test_stage1_combined_graph_objective.py",
    ):
        assert test_path in ci.MACOS_TESTS

    for test_path in (
        "tests/test_domain_sidecar_parquet.py",
        "tests/test_packer_edge_remap.py",
        "tests/test_streaming_conveyor_revision.py",
        "tests/test_streaming_reindex_run_checked.py",
        "tests/test_tokenizer_contract.py",
    ):
        assert test_path in ci.MACOS_TESTS
        assert test_path in ci.LINUX_TESTS


def test_macos_lane_covers_case1_to_case5_graph_identity_contracts() -> None:
    for test_path in (
        "tests/test_atomic_identity_publication.py",
        "tests/test_clang_usr_identity.py",
        "tests/test_graph_recipe.py",
        "tests/test_inference_repository_prompt_graph.py",
        "tests/test_objective_schedule.py",
        "tests/test_prompt_graph.py",
        "tests/test_prompt_graph_index.py",
        "tests/test_stage1_graph_domain_production.py",
        "tests/test_train_stage1_smoke.py",
    ):
        assert test_path in ci.MACOS_TESTS
