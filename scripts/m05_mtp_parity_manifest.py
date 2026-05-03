#!/usr/bin/env python3
"""M0.5 FastMTP parity manifest scaffold.

This script records CUDA-reference FastMTP metadata preflight without running
CUDA, MLX, Hopper/Liger fused CE, or numerical parity. By default it writes a
refused manifest when the required CUDA artifact is missing. An explicitly
omitted artifact writes a blocked manifest. A metadata-valid artifact remains
valid_not_evaluated and refused until a separate numerical harness evaluates it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega_mlx.training.parity import (  # noqa: E402
    M05_MTP_BETA,
    M05_MTP_CUDA_ARTIFACT_CONTRACT,
    M05_MTP_CUDA_ARTIFACT_PATH,
    M05_MTP_DEPTH,
    M05_MTP_LAMBDA,
    M05_MTP_PARITY_OUTPUT,
    M05_MTP_PARITY_PROFILE,
    build_m05_mtp_parity_manifest,
    validate_m05_cuda_reference_artifact_dict,
)

DEFAULT_OUTPUT = ROOT / M05_MTP_PARITY_OUTPUT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the fail-closed M0.5 FastMTP parity manifest. Preflight only "
            "validates external CUDA artifact metadata and never closes M0.5."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cuda-reference-artifact",
        type=Path,
        default=ROOT / M05_MTP_CUDA_ARTIFACT_PATH,
        help=(
            "Path to the external CUDA FastMTP artifact. The default is the "
            "required M0.5 contract path; missing/invalid artifacts are refused."
        ),
    )
    parser.add_argument(
        "--no-cuda-reference-artifact",
        action="store_true",
        help="Record an explicit not-supplied CUDA artifact state instead of probing the default path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the written manifest JSON to stdout.",
    )
    return parser


def cuda_reference_artifact_preflight(artifact_path: Path | None) -> dict[str, Any]:
    preflight: dict[str, Any] = {
        "required_artifact": M05_MTP_CUDA_ARTIFACT_PATH,
        "artifact_contract": M05_MTP_CUDA_ARTIFACT_CONTRACT,
        "evaluates_mtp_losses": False,
        "preflight_is_acceptance": False,
    }
    if artifact_path is None:
        return {
            **preflight,
            "artifact_preflight_status": "not_supplied",
            "artifact_error": "No CUDA FastMTP reference artifact path was supplied.",
        }
    if not artifact_path.exists():
        return {
            **preflight,
            "artifact_preflight_status": "missing",
            "artifact_error": f"CUDA FastMTP reference artifact not found: {artifact_path}",
        }
    if not artifact_path.is_file():
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": (
                f"CUDA FastMTP reference artifact is not a regular file: {artifact_path}"
            ),
        }
    try:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": f"CUDA FastMTP reference artifact could not be read: {exc}",
        }
    except json.JSONDecodeError as exc:
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": f"CUDA FastMTP reference artifact is not valid JSON: {exc}",
        }
    if not isinstance(artifact_payload, Mapping):
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": "CUDA FastMTP reference artifact JSON must be an object.",
        }
    try:
        validate_m05_cuda_reference_artifact_dict(artifact_payload)
    except ValueError as exc:
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": str(exc),
        }
    return {
        **preflight,
        "artifact_preflight_status": "valid_not_evaluated",
        "artifact_error": None,
        "artifact_format": artifact_payload["format"],
        "artifact_loss_values_sha256": artifact_payload["loss_values_sha256"],
        "artifact_source_commit": artifact_payload["source_commit"],
        "artifact_hardware": artifact_payload["hardware"],
        "artifact_cuda_runtime": artifact_payload["cuda_runtime"],
    }


def run_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cuda_artifact = None if args.no_cuda_reference_artifact else args.cuda_reference_artifact
    cuda_preflight = cuda_reference_artifact_preflight(cuda_artifact)
    metadata = {
        "script": "scripts/m05_mtp_parity_manifest.py",
        "git_commit": git_commit(),
        "scaffold_only": True,
        "metadata_preflight_only": True,
    }
    manifest = build_m05_mtp_parity_manifest(
        [],
        profile=M05_MTP_PARITY_PROFILE,
        mtp_depth=M05_MTP_DEPTH,
        mtp_beta=M05_MTP_BETA,
        mtp_lambda=M05_MTP_LAMBDA,
        source="m05_mtp_parity_manifest.py",
        cuda_reference_artifact=cuda_artifact,
        cuda_reference_preflight=cuda_preflight,
        mlx_reference={
            "script": "scripts/m05_mtp_parity_manifest.py",
            "local_mlx_mtp_losses_evaluated": False,
            "readiness_is_cuda_parity": False,
            "readiness_note": (
                "Manifest scaffold only; no MLX FastMTP losses were evaluated and "
                "no CUDA FastMTP numerical comparison was run."
            ),
        },
        metadata=metadata,
    )
    return manifest, 0


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, exit_code = run_manifest(args)
    write_json(args.output, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
        print(f"status: {manifest['status']}")
        print(f"full_m0_5_acceptance: {manifest['acceptance_gate']['full_m0_5_acceptance']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
