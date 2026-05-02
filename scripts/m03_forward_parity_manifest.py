#!/usr/bin/env python3
"""M0.3 forward-parity manifest scaffold.

This script records M0.3 CUDA-reference versus MLX forward-parity evidence
without pretending to run CUDA locally. By default it writes a blocked manifest.
Supplying a CUDA reference artifact path only records that artifact as
not-evaluated and writes a refused scaffold; it never closes M0.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cppmega_mlx.training.parity import (  # noqa: E402
    M03_FORWARD_PARITY_BATCH_SIZE,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT,
    M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH,
    M03_FORWARD_PARITY_OUTPUT,
    M03_FORWARD_PARITY_PROFILE,
    M03_FORWARD_PARITY_SEED,
    M03_FORWARD_PARITY_SEQ_LEN,
    M03_FORWARD_PARITY_VOCAB_SIZE,
    build_m03_forward_parity_manifest,
    validate_m03_cuda_reference_artifact_dict,
)
from cppmega_mlx.recipes.model_factory import (  # noqa: E402
    build_local_gb10_quarter_tiny_smoke_model,
    local_gb10_quarter_profile,
)

DEFAULT_OUTPUT = ROOT / M03_FORWARD_PARITY_OUTPUT
DEFAULT_SEED = M03_FORWARD_PARITY_SEED
DEFAULT_BATCH_SIZE = M03_FORWARD_PARITY_BATCH_SIZE
DEFAULT_SEQ_LEN = M03_FORWARD_PARITY_SEQ_LEN
DEFAULT_VOCAB_SIZE = M03_FORWARD_PARITY_VOCAB_SIZE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the M0.3 random-init seed-matched forward parity manifest. "
            "Defaults to a blocked scaffold until the external CUDA artifact "
            "is present and passes metadata preflight. Preflight never evaluates "
            "logits and never closes M0.3."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cuda-reference-artifact",
        type=Path,
        default=ROOT / M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH,
        help=(
            "Path to the external CUDA logits artifact. The default is the "
            "required M0.3 contract path; missing/invalid artifacts are refused."
        ),
    )
    parser.add_argument(
        "--no-cuda-reference-artifact",
        action="store_true",
        help="Record an explicit not-supplied CUDA artifact state instead of probing the default path.",
    )
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument(
        "--skip-mlx-readiness",
        action="store_true",
        help=(
            "Do not run the local MLX tiny-smoke forward probe. The manifest "
            "still remains blocked/refused for full M0.3 acceptance."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the written manifest JSON to stdout.",
    )
    return parser


def deterministic_input_tokens(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> np.ndarray:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.uint32)


def input_tokens_sha256(tokens: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(tokens).tobytes()).hexdigest()


def local_mlx_readiness(tokens: np.ndarray, *, seed: int) -> dict[str, Any]:
    """Run MLX-side structural and tiny forward readiness without CUDA claims."""

    import mlx.core as mx

    profile = local_gb10_quarter_profile()
    expanded = profile.expanded_pattern
    mx.random.seed(seed)
    model = build_local_gb10_quarter_tiny_smoke_model()
    smoke_tokens = np.remainder(tokens, model.config.vocab_size).astype(np.uint32, copy=False)
    input_ids = mx.array(smoke_tokens, dtype=mx.uint32)
    logits = model(input_ids)
    mx.eval(logits)

    logits_np = np.asarray(logits)
    finite = bool(np.isfinite(logits_np).all())
    return {
        "readiness_status": "pass" if finite else "fail",
        "local_mlx_forward_executed": True,
        "local_mlx_forward_scope": "tiny_smoke_only",
        "readiness_is_cuda_parity": False,
        "full_profile_allocation_executed": False,
        "full_profile_forward_executed": False,
        "tiny_smoke_forward_executed": True,
        "closure_required_mlx_forward_scope": "full_local_gb10_quarter_logits",
        "execution_note": (
            "This script records full local_gb10_quarter profile metadata but "
            "does not allocate or forward the full profile; it only executes "
            "the tiny smoke model below."
        ),
        "full_profile": {
            "name": profile.name,
            "pattern": profile.pattern,
            "depth": profile.depth,
            "hidden_size": profile.hidden_size,
            "ffn_hidden_size": profile.ffn_hidden_size,
            "num_attention_heads": profile.num_attention_heads,
            "head_dim": profile.head_dim,
            "vocab_size": profile.vocab_size,
            "route_symbols": list(expanded.symbols),
            "route_roles": [layer.role for layer in expanded.layers],
            "dsa_layer_numbers": list(expanded.dsa_layer_numbers),
            "mla_layer_numbers": list(expanded.mla_layer_numbers),
        },
        "tiny_smoke_forward": {
            "seed": seed,
            "input_tokens_modulo_vocab": True,
            "input_tokens_sha256": input_tokens_sha256(smoke_tokens),
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "max_seq_length": model.config.max_seq_length,
            "route_symbols": list(model.route_symbols),
            "route_roles": list(model.route_roles),
            "logits_shape": list(logits.shape),
            "logits_dtype": str(logits.dtype),
            "logits_all_finite": finite,
            "logits_sha256": hashlib.sha256(np.ascontiguousarray(logits_np).tobytes()).hexdigest(),
        },
        "mlx_version": package_version("mlx"),
        "mlx_lm_version": package_version("mlx-lm"),
        "hardware": platform.platform(),
    }


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def cuda_reference_artifact_preflight(
    artifact_path: Path | None,
    *,
    input_hash: str,
    dtype: str,
) -> dict[str, Any]:
    preflight: dict[str, Any] = {
        "required_artifact": M03_FORWARD_PARITY_CUDA_ARTIFACT_PATH,
        "artifact_contract": M03_FORWARD_PARITY_CUDA_ARTIFACT_CONTRACT,
        "evaluates_logits": False,
        "preflight_is_acceptance": False,
    }
    if artifact_path is None:
        return {
            **preflight,
            "artifact_preflight_status": "not_supplied",
            "artifact_error": "No CUDA reference artifact path was supplied.",
        }
    if not artifact_path.exists():
        return {
            **preflight,
            "artifact_preflight_status": "missing",
            "artifact_error": f"CUDA reference artifact not found: {artifact_path}",
        }
    if not artifact_path.is_file():
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": f"CUDA reference artifact is not a regular file: {artifact_path}",
        }
    try:
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": f"CUDA reference artifact could not be read: {exc}",
        }
    except json.JSONDecodeError as exc:
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": f"CUDA reference artifact is not valid JSON: {exc}",
        }
    if not isinstance(artifact_payload, Mapping):
        return {
            **preflight,
            "artifact_preflight_status": "invalid",
            "artifact_error": "CUDA reference artifact JSON must be an object.",
        }
    try:
        validate_m03_cuda_reference_artifact_dict(
            artifact_payload,
            input_tokens_sha256=input_hash,
            dtype=dtype,
        )
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
        "artifact_tensor_name": artifact_payload["tensor_name"],
        "artifact_logits_sha256": artifact_payload["logits_sha256"],
        "artifact_source_commit": artifact_payload["source_commit"],
        "artifact_hardware": artifact_payload["hardware"],
        "artifact_cuda_runtime": artifact_payload["cuda_runtime"],
    }


def run_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    tokens = deterministic_input_tokens(
        seed=DEFAULT_SEED,
        batch_size=DEFAULT_BATCH_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
        vocab_size=DEFAULT_VOCAB_SIZE,
    )
    token_hash = input_tokens_sha256(tokens)
    dtype = args.dtype or "bf16"
    cuda_artifact = None if args.no_cuda_reference_artifact else args.cuda_reference_artifact
    cuda_preflight = cuda_reference_artifact_preflight(
        cuda_artifact,
        input_hash=token_hash,
        dtype=dtype,
    )
    readiness = (
        {
            "readiness_status": "skipped",
            "local_mlx_forward_executed": False,
            "local_mlx_forward_scope": "skipped",
            "readiness_is_cuda_parity": False,
            "full_profile_allocation_executed": False,
            "full_profile_forward_executed": False,
            "tiny_smoke_forward_executed": False,
            "closure_required_mlx_forward_scope": "full_local_gb10_quarter_logits",
        }
        if args.skip_mlx_readiness
        else local_mlx_readiness(tokens, seed=DEFAULT_SEED)
    )
    metadata = {
        "script": "scripts/m03_forward_parity_manifest.py",
        "git_commit": git_commit(),
        "scaffold_only": True,
        "local_mlx_readiness_status": readiness["readiness_status"],
    }
    manifest = build_m03_forward_parity_manifest(
        [],
        seed=DEFAULT_SEED,
        batch_size=DEFAULT_BATCH_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
        profile=M03_FORWARD_PARITY_PROFILE,
        dtype=dtype,
        source="m03_forward_parity_manifest.py",
        input_tokens_sha256=token_hash,
        cuda_reference_artifact=cuda_artifact,
        cuda_reference_preflight=cuda_preflight,
        mlx_reference={
            "script": "scripts/m03_forward_parity_manifest.py",
            **readiness,
            "readiness_note": (
                "Local MLX structural/tiny forward readiness only; no CUDA logits "
                "were evaluated and this does not satisfy full M0.3 acceptance."
            ),
        },
        metadata=metadata,
    )
    exit_code = 0 if readiness["readiness_status"] in {"pass", "skipped"} else 1
    return manifest, exit_code


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
    if args.json or exit_code != 0:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"wrote {args.output}")
        print(f"status: {manifest['status']}")
        print(f"full_m0_3_acceptance: {manifest['acceptance_gate']['full_m0_3_acceptance']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
