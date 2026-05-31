from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.m04_train_step as m04_train_step
from cppmega_mlx.models.hybrid_lm import PathCActivationBufferCapture
from cppmega_mlx.recipes.model_factory import build_local_gb10_quarter_tiny_smoke_model
from cppmega_mlx.runtime.path_c_physical_abi import (
    PathCLogicalBufferOwner,
    make_physical_abi_bank_owner,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (
    PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR,
)
from cppmega_mlx.training.optimizers import (
    ADAMW_BASE_CLASS,
    ADAMW_FP32_MOMENTS_CLASS,
    AdamWFP32Moments,
    collect_adamw_moment_dtypes,
    dtype_name,
    make_adamw,
)
from scripts.m04_train_step import (
    OBSERVED_OPTIMIZER_IDENTITY,
    GRAD_CHECKPOINT_EXPECTATION,
    REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
    REQUIRED_DTYPE,
    REQUIRED_MODEL_GEOMETRY,
    REQUIRED_MODEL_SOURCE,
    acceptance_gate_payload,
    applied_memory_limit_api_path_from_payload,
    local_gb10_quarter_preflight_payload,
    target_dataset_path,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m04_train_step.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_FUSION_COMPILE_RECEIPT = ROOT / "reports" / "path_c_fusion_compile_receipt.json"
PRODUCTION_FUSION_COMPILE_RECEIPT = (
    ROOT / "reports" / "path_c_fusion_production_smoke_receipt.json"
)
LOCAL_MLX_PYTHON = Path("/Volumes/external/sources/mlx/python")
LOCAL_MLX_LIB = LOCAL_MLX_PYTHON / "mlx" / "lib"
BASELINE_RECEIPT = ROOT / "bench" / "baselines" / "m04_train_step.json"
GB10_SAMPLE = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
TARGET_PARQUET = "data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet"
REAL_PARQUET_COLUMNS = (
    "token_ids",
    "structure_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
)


def _write_m04_token_parquet(
    path: Path,
    rows: list[list[int]],
    *,
    tokenizer_fingerprint: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"token_ids": rows}
    if tokenizer_fingerprint is not None:
        data["tokenizer_fingerprint"] = [tokenizer_fingerprint] * len(rows)
    pq.write_table(pa.table(data), path)
    return path


@contextmanager
def temporary_env(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def path_c_fusion_auto_env() -> Iterator[None]:
    with temporary_env({"CPPMEGA_PATH_C_FUSION": "auto"}):
        yield


@pytest.fixture
def path_c_fusion_force_env() -> Iterator[None]:
    with temporary_env({"CPPMEGA_PATH_C_FUSION": "force"}):
        yield


def strip_known_tvm_stderr_noise(stderr: str) -> str:
    lines = [
        line
        for line in stderr.splitlines()
        if not (
            "arm_aprofile.cc:125: Warning: Cannot parse Arm(R)-based target features"
            in line
            and "without LLVM support" in line
        )
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def canonical_allocation_probe(**overrides: Any) -> dict[str, Any]:
    probe = {
        "status": "ok",
        "allocation_ready": True,
        "source": REQUIRED_MODEL_SOURCE,
        "allocation_mode": "full_profile_allocation_probe",
        "required_geometry": REQUIRED_MODEL_GEOMETRY,
        "profile_geometry": REQUIRED_MODEL_GEOMETRY,
        "geometry_matches_required": True,
        "profile_name": "local_gb10_quarter",
        "model_class": "HybridTinyLM",
        "eval_scope": "parameters_only_no_forward_no_training",
        "forward_executed": False,
        "training_executed": False,
        "memory_before": {"active_memory_bytes": 0},
        "memory_after": {"active_memory_bytes": 1024},
    }
    probe.update(overrides)
    return probe


def test_m04_parquet_shards_stream_in_deterministic_order(tmp_path: Path) -> None:
    shard0 = _write_m04_token_parquet(
        tmp_path / "corpus" / "val_00000.parquet",
        [[1, 2, 3, 4]],
    )
    shard1 = _write_m04_token_parquet(
        tmp_path / "corpus" / "val_00001.parquet",
        [[5, 6, 7, 8]],
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--data-path",
            str(shard0),
            "--data-format",
            "parquet",
            "--token-key",
            "token_ids",
            "--seq-len",
            "4",
            "--batch-size",
            "1",
            "--data-shard",
            str(shard1),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=shard0)

    dataset = m04_train_step.training_dataset_from_args(
        args,
        config=config,
        data_path=shard0,
        loop=False,
    )
    batches = dataset.iter_batches(loop=False)
    first = next(batches)
    second = next(batches)

    assert np.array(first.tokens).tolist() == [[1, 2, 3, 4]]
    assert np.array(second.tokens).tolist() == [[5, 6, 7, 8]]
    assert first.metadata["parquet_stream"]["shard_index"] == 0
    assert second.metadata["parquet_stream"]["shard_index"] == 1
    payload = m04_train_step.dataset_payload(dataset, config)
    stream = payload["dataset_receipt"]["parquet_receipt"]["stream"]
    assert stream["shard_count"] == 2
    assert stream["shards"][1]["row_count"] == 1


def test_m04_rejects_mismatched_parquet_shard_tokenizers(tmp_path: Path) -> None:
    shard0 = _write_m04_token_parquet(
        tmp_path / "corpus" / "val_00000.parquet",
        [[1, 2, 3, 4]],
        tokenizer_fingerprint="a" * 64,
    )
    shard1 = _write_m04_token_parquet(
        tmp_path / "corpus" / "val_00001.parquet",
        [[5, 6, 7, 8]],
        tokenizer_fingerprint="b" * 64,
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--data-path",
            str(shard0),
            "--data-format",
            "parquet",
            "--token-key",
            "token_ids",
            "--seq-len",
            "4",
            "--batch-size",
            "1",
            "--data-shard",
            str(shard1),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=shard0)

    with pytest.raises(ValueError, match="tokenizer_fingerprint mismatch"):
        m04_train_step.training_dataset_from_args(
            args,
            config=config,
            data_path=shard0,
            loop=False,
        )


class _ValueAndGradPathCDirectFusionChainTrainingRuntime(
    m04_train_step.PathCDirectFusionChainTrainingRuntime
):
    def value_and_grad(
        self,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        loss_and_grad: Any,
    ) -> tuple[tuple[mx.array, mx.array], Any]:
        return loss_and_grad(model, batch)


class _ContractedValueAndGradPathCDirectFusionChainTrainingRuntime(
    _ValueAndGradPathCDirectFusionChainTrainingRuntime
):
    def value_and_grad_contract(self) -> dict[str, Any]:
        return {
            "contract": m04_train_step.PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_direct_chain_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": True,
            "returns_full_model_grads": True,
            "gradient_scope": "full_model",
            "loss_cotangent_bridge_ready": True,
            "model_gradient_tree_ready": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }


class _ReadyFusedTrainBlockTrainingRuntime:
    contract = m04_train_step.PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_CONTRACT
    training_critical_path = True
    hidden_packing_performed = False
    no_hidden_allocation_policy = True
    owner_name = "local_gb10_quarter.path_c_fused_train_block_runtime"

    def __init__(self) -> None:
        self._binding: dict[str, Any] | None = {
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
        }

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("unit tests only inspect the fused train-block contract")

    def backward(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("unit tests only inspect the fused train-block contract")

    def value_and_grad(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("unit tests only inspect the fused train-block contract")

    def bind_training_graph(self, **binding: Any) -> None:
        self._binding = dict(binding)

    def unbind_training_graph(self, *, owner: str) -> None:
        if self._binding is not None and self._binding.get("owner") == owner:
            self._binding = None

    def training_graph_binding(self) -> dict[str, Any]:
        return dict(self._binding or {})

    def value_and_grad_contract(self) -> dict[str, Any]:
        return {
            "contract": m04_train_step.PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": True,
            "returns_full_model_grads": True,
            "gradient_scope": "full_model",
            "loss_cotangent_bridge_ready": True,
            "model_gradient_tree_ready": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }


class _UnboundReadyFusedTrainBlockTrainingRuntime(
    _ReadyFusedTrainBlockTrainingRuntime
):
    def __init__(self) -> None:
        super().__init__()
        self.unbind_training_graph(owner="CompiledPretrainingStep")


class _ContractedFusedTrainBlockArtifact:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("call", tuple(kwargs)))

    def forward(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("forward", tuple(kwargs)))

    def backward(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("backward", tuple(kwargs)))

    def value_and_grad(self, model: Any, batch: Mapping[str, mx.array], *, bank_owner: Any) -> tuple[Any, Any]:
        self.calls.append(("value_and_grad", tuple(sorted(batch))))
        del model, bank_owner
        raise AssertionError("unit test only validates automatic runtime binding")

    def value_and_grad_contract(self) -> dict[str, Any]:
        return {
            "contract": m04_train_step.PATH_C_FUSED_TRAIN_BLOCK_VALUE_AND_GRAD_CONTRACT,
            "owner": "CompiledPretrainingStep",
            "uses_fused_train_block_runtime": True,
            "uses_forward_hook": True,
            "uses_backward_or_vjp_hook": True,
            "returns_model_grads": True,
            "returns_full_model_grads": True,
            "gradient_scope": "full_model",
            "loss_cotangent_bridge_ready": True,
            "model_gradient_tree_ready": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }


class _PhysicalAbiBankOwnerFactoryModel:
    def __init__(self, wrapped: Any, owner: Any) -> None:
        self._wrapped = wrapped
        self._owner = owner
        self.owner_factory_sequence_lengths: list[int | None] = []
        self.path_c_fused_train_block_artifact = _ContractedFusedTrainBlockArtifact()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def make_path_c_physical_abi_bank_owner(
        self,
        *,
        sequence_length: int | None = None,
    ) -> Any:
        self.owner_factory_sequence_lengths.append(sequence_length)
        return self._owner


class _ContractedLossCotangentBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def loss_cotangent_bridge_contract(self) -> dict[str, Any]:
        return {
            "contract": m04_train_step.PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT,
            "returns_required_loss_cotangents": True,
            "delegates_to_eager_loss_and_grad": False,
            "hidden_packing_performed": False,
        }

    def __call__(
        self,
        *,
        model: nn.Module,
        batch: Mapping[str, mx.array],
        logical_buffers: Mapping[str, mx.array],
        required_loss_cotangent_buffers: Sequence[str],
        chain: Any,
    ) -> dict[str, Any]:
        del model, batch, chain
        self.calls.append(
            (
                "loss_cotangent_bridge",
                tuple(required_loss_cotangent_buffers),
            )
        )
        return {
            "loss": mx.array(1.25, dtype=mx.float32),
            "ntokens": mx.array(7, dtype=mx.uint32),
            "cotangents": {
                name: logical_buffers[name]
                for name in required_loss_cotangent_buffers
            },
        }


def test_fp8_path_policies_set_explicit_runtime_routes(tmp_path: Path) -> None:
    path_c_args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--dtype", "fp8_path_c", "--output", str(tmp_path / "c.json")]
    )
    path_b_args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--dtype", "fp8_path_b", "--output", str(tmp_path / "b.json")]
    )
    with temporary_env(
        {
            "CPPMEGA_KERNEL_PATH__SPARSE_MLA": None,
            "CPPMEGA_SPARSE_MLA_FP8_ROUTE": None,
            "CPPMEGA_SPARSE_MLA_FP8_BWD": None,
            "CPPMEGA_MAMBA3_PATH_C_BWD": None,
        }
    ):
        with m04_train_step.fp8_path_c_kernel_policy(
            path_c_args,
            ensure_dev_env=lambda: None,
        ):
            assert os.environ["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] == "path_c"
            assert os.environ["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] == "path_c"
            assert os.environ["CPPMEGA_SPARSE_MLA_FP8_BWD"] == "path_c"
            assert os.environ["CPPMEGA_MAMBA3_PATH_C_BWD"] == "path_b"
        assert "CPPMEGA_SPARSE_MLA_FP8_ROUTE" not in os.environ
        assert "CPPMEGA_SPARSE_MLA_FP8_BWD" not in os.environ
        assert "CPPMEGA_MAMBA3_PATH_C_BWD" not in os.environ

        with m04_train_step.fp8_path_b_kernel_policy(path_b_args):
            assert os.environ["CPPMEGA_KERNEL_PATH__SPARSE_MLA"] == "path_b"
            assert os.environ["CPPMEGA_SPARSE_MLA_FP8_ROUTE"] == "path_b"
            assert "CPPMEGA_MAMBA3_PATH_C_BWD" not in os.environ


def test_fp8_path_c_direct_chain_capture_is_opt_in(tmp_path: Path) -> None:
    base_args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--dtype", "fp8_path_c", "--output", str(tmp_path / "base.json")]
    )
    runtime_args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--use-path-c-direct-chain-runtime",
            "--output",
            str(tmp_path / "runtime.json"),
        ]
    )
    profile_args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--profile-path-c-direct-chain-runtime",
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )

    assert not m04_train_step.path_c_direct_chain_capture_requested(
        base_args,
        compile_enabled=False,
    )
    assert m04_train_step.path_c_direct_chain_capture_requested(
        runtime_args,
        compile_enabled=False,
    )
    assert m04_train_step.path_c_direct_chain_capture_requested(
        profile_args,
        compile_enabled=False,
    )
    assert not m04_train_step.path_c_direct_chain_capture_requested(
        runtime_args,
        compile_enabled=True,
    )


def test_tilelang_dev_env_points_to_build_root_and_runtime_libs(tmp_path: Path) -> None:
    source_root = tmp_path / "tl_apache_tvm_swap"
    build_root = source_root / "build"
    (source_root / "tilelang").mkdir(parents=True)
    (source_root / "3rdparty" / "tvm" / "python").mkdir(parents=True)
    (build_root / "lib").mkdir(parents=True)
    (build_root / "tvm").mkdir(parents=True)

    with temporary_env(
        {
            "TILELANG_DEV_BUILD_ROOT": str(source_root),
            "TVM_LIBRARY_PATH": None,
            "DYLD_LIBRARY_PATH": None,
        }
    ):
        m04_train_step.ensure_tilelang_dev_env_for_path_c()

        assert os.environ["TILELANG_DEV_BUILD_ROOT"] == str(build_root)
        assert os.environ["TVM_LIBRARY_PATH"] == str(build_root / "lib")
        assert str(build_root / "lib") in os.environ["DYLD_LIBRARY_PATH"].split(
            os.pathsep
        )


def test_m04_import_preserves_recipes_package_exports() -> None:
    import cppmega_mlx.recipes as recipes

    assert recipes is sys.modules["cppmega_mlx.recipes"]
    assert isinstance(recipes.__all__, list)
    assert "local_gb10_quarter" in recipes.__all__
    assert hasattr(recipes, "local_gb10_quarter")


def test_profile_hold_seconds_is_opt_in(tmp_path: Path) -> None:
    args = m04_train_step.build_parser().parse_args(
        ["--synthetic", "--output", str(tmp_path / "receipt.json")]
    )

    assert args.profile_hold_seconds == 0.0


class _FakeCacheLimitMLX:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def set_cache_limit(self, limit: int) -> int:
        self.calls.append(limit)
        return 987654321


def test_path_c_local_gb10_defaults_cache_limit_to_zero(tmp_path: Path) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    fake = _FakeCacheLimitMLX()

    with temporary_env({"CPPMEGA_KERNEL_PATH": "path_c"}):
        payload = m04_train_step.apply_cache_limit_payload(args, mx_module=fake)

    assert fake.calls == [0]
    assert payload == {
        "configured": True,
        "applied": True,
        "limit_bytes": 0,
        "source": "path_c_default",
        "api_path": "mx.set_cache_limit",
        "previous_limit_bytes": 987654321,
    }


def test_non_path_c_keeps_mlx_cache_default(tmp_path: Path) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    fake = _FakeCacheLimitMLX()

    with temporary_env(
        {
            "CPPMEGA_KERNEL_PATH": None,
            "CPPMEGA_MLX_CACHE_LIMIT_BYTES": None,
        }
    ):
        payload = m04_train_step.apply_cache_limit_payload(args, mx_module=fake)

    assert fake.calls == []
    assert payload["configured"] is False
    assert payload["source"] == "mlx_default"


def run_script(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath = [str(ROOT)]
    if LOCAL_MLX_PYTHON.exists():
        pythonpath.insert(0, str(LOCAL_MLX_PYTHON))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if LOCAL_MLX_LIB.exists():
        dyld_path = [str(LOCAL_MLX_LIB)]
        if env.get("DYLD_LIBRARY_PATH"):
            dyld_path.append(env["DYLD_LIBRARY_PATH"])
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(dyld_path)
    result = subprocess.run(
        [str(PYTHON if PYTHON.exists() else sys.executable), str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=strip_known_tvm_stderr_noise(result.stderr),
    )


def load_json_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def copy_real_parquet_head(
    source_path: Path, sample_path: Path, *, row_count: int = 4
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_file = pq.ParquetFile(source_path)
    batch = next(
        parquet_file.iter_batches(
            batch_size=row_count,
            columns=list(REAL_PARQUET_COLUMNS),
        )
    )
    pq.write_table(pa.Table.from_batches([batch]), sample_path)


def tiny_args(output: Path, *, steps: int = 1) -> list[str]:
    return [
        "--synthetic",
        "--steps",
        str(steps),
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--hidden-size",
        "8",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--output",
        str(output),
        "--json",
    ]


def assert_m04_receipt_contract(payload: dict[str, Any]) -> None:
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_mlx_m04_train_step"
    assert payload["issue"]["id"] == "cppmega-mlx-t8f.4"
    assert payload["local_only"] is True
    assert payload["m4_vs_gb10_throughput_parity_claim"] is False
    gate = payload["acceptance_gate"]
    full_gate_completed = bool(gate["full_local_gb10_quarter_gate_completed"])
    assert payload["gb10_training_correctness_claim"] is full_gate_completed
    assert payload["full_m0_4_acceptance_claim"] is full_gate_completed
    assert gate["full_target_dataset"] == TARGET_PARQUET
    assert gate["full_target_dataset_100_step_required"] is True
    assert gate["local_gb10_quarter_required"] is True
    assert gate["required_model_profile"] == "local_gb10_quarter"
    assert gate["required_dtype"] == REQUIRED_DTYPE
    assert gate["observed_dtype"] == "bfloat16"
    assert gate["dtype_ok"] is True
    assert gate["required_optimizer_name"] == "AdamW"
    assert gate["grad_checkpoint_required"] is True
    assert gate["full_local_gb10_quarter_gate_required"] is True
    for key in (
        "real_parquet_source_identity",
        "target_parquet_path_ok",
        "dataset_name_ok",
        "dataset_format_ok",
        "dtype_ok",
        "local_gb10_quarter_preflight",
        "local_gb10_quarter_preflight_ok",
        "model_identity_ok",
        "model_identity",
        "optimizer_identity_ok",
        "optimizer_identity",
        "required_adamw_master_moment_dtype",
        "observed_adamw_master_moment_dtypes",
        "fp32_adamw_master_moments_ok",
        "adamw_ok",
        "grad_checkpoint_expectation_ok",
        "grad_checkpoint_identity",
        "step_count_ok",
        "loss_decrease_ok",
        "loss_fields_ok",
        "all_finite_ok",
        "optimizer_update_ok",
        "m4_runtime_metadata",
        "m4_runtime_metadata_ok",
        "full_local_gb10_quarter_gate_completed",
        "full_local_gb10_quarter_gate_blockers",
    ):
        assert key in gate
    assert gate["real_parquet_source_identity"]["required_path"] == TARGET_PARQUET
    assert (
        payload["local_gb10_quarter_preflight"] == gate["local_gb10_quarter_preflight"]
    )
    preflight = payload["local_gb10_quarter_preflight"]
    assert preflight["profile_name"] == "local_gb10_quarter"
    assert preflight["source"] == REQUIRED_MODEL_SOURCE
    assert preflight["required_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert preflight["profile_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert preflight["geometry_matches_required"] is True
    assert preflight["tokenizer_contract"]["resolved"] is True
    assert preflight["tokenizer_contract"]["expected_vocab_size"] == 65_536
    assert preflight["tokenizer_contract"]["blocker_id"] == "cppmega-mlx-t8f.1"
    assert preflight["tokenizer_contract"]["milestone"] == "M0.1"
    assert (
        "<FIM_INSTRUCTION>"
        in preflight["tokenizer_contract"]["required_special_tokens"]
    )
    assert "CODE_START" in preflight["tokenizer_contract"]["reason"]
    assert "M0.1 is closed" in preflight["tokenizer_contract"]["reason"]
    if preflight["allocation_attempted"]:
        assert preflight["allocation_attempted"] is True
        assert preflight["allocation_ready"] is True
        assert preflight["allocation_mode"] == "full_profile_allocation_probe"
        assert preflight["allocation_probe"]["status"] == "ok"
        assert preflight["allocation_probe"]["source"] == REQUIRED_MODEL_SOURCE
        assert preflight["allocation_probe"]["allocation_mode"] == (
            "full_profile_allocation_probe"
        )
        assert preflight["allocation_probe"]["required_geometry"] == (
            REQUIRED_MODEL_GEOMETRY
        )
        assert preflight["allocation_probe"]["profile_geometry"] == (
            REQUIRED_MODEL_GEOMETRY
        )
        assert preflight["allocation_probe"]["geometry_matches_required"] is True
        assert preflight["allocation_probe"]["model_class"] == "HybridTinyLM"
        assert preflight["allocation_probe"]["forward_executed"] is False
        assert preflight["allocation_probe"]["training_executed"] is False
        assert preflight["ok"] is True
        assert preflight["blockers"] == []
        assert gate["local_gb10_quarter_preflight_ok"] is True
    else:
        assert preflight["allocation_attempted"] is False
        assert preflight["allocation_ready"] is False
        assert preflight["allocation_mode"] == "allocation_free_preflight"
        assert preflight["ok"] is False
        assert {"allocation_attempted", "allocation_ready"}.issubset(
            set(preflight["blockers"])
        )
        assert "tokenizer_contract_resolved" not in preflight["blockers"]
        assert gate["local_gb10_quarter_preflight_ok"] is False
    if gate["full_target_dataset_100_step_completed"]:
        assert gate["uses_full_target_dataset"] is True
        assert gate["real_parquet_source_identity"]["ok"] is True
        assert gate["full_target_dataset_blocker"] is None
    else:
        assert gate["full_target_dataset_blocker"]
    assert gate["full_local_gb10_quarter_gate_completed"] is full_gate_completed
    if full_gate_completed:
        assert gate["full_local_gb10_quarter_gate_blockers"] == []
    else:
        assert gate["full_local_gb10_quarter_gate_blockers"]
    model_identity = gate["model_identity"]
    assert model_identity["required_name"] == "local_gb10_quarter"
    assert model_identity["required_source"] == REQUIRED_MODEL_SOURCE
    assert model_identity["required_profile"] == "local_gb10_quarter"
    assert model_identity["required_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert model_identity["ok"] is gate["model_identity_ok"]
    optimizer_identity = gate["optimizer_identity"]
    assert gate["required_adamw_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["required_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["master_moment_evidence"]["required_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert (
        optimizer_identity["master_moment_dtype_ok"]
        is (gate["fp32_adamw_master_moments_ok"])
    )
    assert payload["workload"]["dtype"] == "bfloat16"
    model_payload = payload["model"]
    if model_payload.get("metadata_only"):
        assert payload["workload"]["mode"] == "metadata_only_no_forward_no_training"
        assert model_payload["source"] is None
        assert model_payload["name"] is None
        assert model_payload["required_source"] == REQUIRED_MODEL_SOURCE
        assert model_payload["required_profile"] == "local_gb10_quarter"
        assert model_payload["profile_matches_required"] is False
        assert model_payload["local_gb10_quarter_preflight"] == preflight
        assert model_payload["forward_executed"] is False
        assert model_payload["training_executed"] is False
    else:
        if gate["model_identity_ok"]:
            assert model_payload["source"] == REQUIRED_MODEL_SOURCE
            assert model_payload["name"] == "local_gb10_quarter"
            assert model_payload["profile_matches_required"] is True
        else:
            assert model_payload["source"] == "cppmega_mlx.models.hybrid_lm"
            assert model_payload["name"] == "HybridTinyLM"
            assert model_payload["profile_matches_required"] is False
        assert model_payload["required_profile"] == "local_gb10_quarter"
        assert model_payload["local_gb10_quarter_preflight"] == preflight
    assert payload["training"]["optimizer"]["name"] == "AdamW"
    assert payload["training"]["optimizer"]["class"] == (ADAMW_FP32_MOMENTS_CLASS)
    assert payload["training"]["optimizer"]["base_class"] == ADAMW_BASE_CLASS
    assert payload["training"]["optimizer"]["adamw"] is True
    assert payload["training"]["optimizer"]["required_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert (
        payload["training"]["optimizer"]["master_moment_evidence"]
        == (optimizer_identity["master_moment_evidence"])
    )
    grad_checkpoint_expected = bool(payload["workload"].get("grad_checkpoint", False))
    assert payload["training"]["grad_checkpoint"]["required"] is True
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is (
        grad_checkpoint_expected
    )
    assert payload["training"]["grad_checkpoint"]["expectation_satisfied"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_observed_enabled"] is grad_checkpoint_expected
    assert gate["grad_checkpoint_expectation_ok"] is grad_checkpoint_expected
    assert gate["grad_checkpoint_identity"]["observed_enabled"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_identity"]["expectation_satisfied"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_identity"]["ok"] is grad_checkpoint_expected
    expected_model = (
        "metadata_only_no_observed_model"
        if model_payload.get("metadata_only")
        else model_payload["name"]
    )
    expected_route = (
        "metadata_only_no_forward_no_training"
        if model_payload.get("metadata_only")
        else model_payload["route_symbols"]
    )
    assert payload["baseline_row"] == {
        "batch_size": payload["workload"]["batch_size"],
        "commit": payload["software"]["git_commit"] or "unknown",
        "dtype": "bfloat16",
        "gb10_parity_claim": False,
        "hardware": payload["baseline_row"]["hardware"],
        "local_only": True,
        "mode": payload["workload"]["mode"],
        "model": expected_model,
        "route": expected_route,
        "seq_len": payload["workload"]["seq_len"],
        "tokens_per_second": payload["timing"]["tokens_per_second"] or 0.0,
    }


def assert_regression_report_matches_payload(payload: dict[str, Any]) -> None:
    report = payload["regression_report"]
    route_dispatch = report["route_dispatch"]
    producer_gate = report["fp8_path_c_producer_gate"]

    assert route_dispatch["raw"] == payload["training"].get("kernel_dispatch", [])
    assert "fallback_reason" in route_dispatch
    assert report["fallback_reason"] == route_dispatch["fallback_reason"]
    assert report["dtype"]["requested"] == payload["workload"]["dtype"]
    assert report["optimizer"]["key"] == payload["workload"]["optimizer"]["key"]
    assert report["memory"]["peak_memory_bytes"] == payload["memory"][
        "peak_memory_bytes"
    ]
    assert report["training"]["all_finite"] == payload["training"]["all_finite"]
    assert report["training"]["initial_loss"] == payload["training"]["initial_loss"]
    assert report["training"]["final_loss"] == payload["training"]["final_loss"]
    assert report["training"]["mean_loss"] == payload["training"].get("mean_loss")
    assert report["training"]["loss_decreased"] == payload["training"][
        "loss_decreased"
    ]
    assert report["gate_summary"]["dtype"] == payload["workload"]["dtype"]
    assert report["gate_summary"]["optimizer"] == payload["workload"]["optimizer"][
        "key"
    ]
    assert report["gate_summary"]["fallback_reason"] == route_dispatch[
        "fallback_reason"
    ]
    assert report["gate_summary"]["fp8_path_c_producer_status"] == producer_gate[
        "status"
    ]
    assert report["gate_summary"]["fp8_path_c_producer_ok"] == producer_gate["ok"]
    assert producer_gate["name"] == "fp8_path_c_sparse_mla_producer"
    assert producer_gate["large_tensor_staging_allowed"] is False
    assert producer_gate["hidden_wrapper_quantization_allowed"] is False
    assert producer_gate["kernel_boundary_quantization_allowed"] is False
    assert "regression_report.fp8_path_c_producer_gate" in producer_gate[
        "receipt_field_paths"
    ]
    if payload["workload"]["dtype"] == "fp8_path_c":
        assert producer_gate["required"] is True
        assert producer_gate["fallback_to_path_b_allowed"] is False
    else:
        assert producer_gate["required"] is False
        assert producer_gate["status"] == "not_requested"
    assert "path_b_observed" in route_dispatch
    assert "path_c_observed" in route_dispatch
    assert "path_summary" in route_dispatch
    assert report["throughput"]["tokens_per_second"] == payload["timing"][
        "tokens_per_second"
    ]
    assert report["throughput"]["claim_gate"]["ok"] is True
    assert report["gate_summary"]["tokens_per_second_claim_ok"] is True
    assert report["gate_summary"]["bogus_tok_sec_claim_detected"] is False
    assert report["visibility_gate"] == {
        "route_dispatch_visible": True,
        "dtype_visible": True,
        "optimizer_visible": True,
        "memory_peak_visible": True,
        "tokens_per_second_visible": payload["timing"]["tokens_per_second"]
        is not None,
        "finite_visible": True,
        "loss_visible": True,
        "fallback_reason_visible": True,
    }


def assert_m04_20step_matrix_plan(payload: dict[str, Any]) -> None:
    matrix = payload["m04_20step_matrix"]

    assert matrix["name"] == "m04_local_gb10_20step_dtype_optimizer_matrix"
    assert matrix["status"] == "commands_prepared_not_executed_by_this_receipt"
    assert matrix["profile"] == "local_gb10_quarter"
    assert matrix["dataset"] == TARGET_PARQUET
    assert matrix["steps"] == 20
    assert matrix["acceptance_steps"] == 100
    assert matrix["batch_size"] == 1
    assert matrix["seq_len"] == 4096
    assert matrix["dtype_routes"] == ["bf16", "fp8_path_b", "fp8_path_c", "int8"]
    assert matrix["optimizers"] == [
        "adamw",
        "muon",
        "muon_adamw",
        "lion",
        "lion8bit",
        "adam8bit",
    ]
    baseline = matrix["baseline_comparison"]
    assert baseline["baseline_tokens_per_second"] == 900.0
    assert baseline["baseline_kind"] == (
        "existing_real_parquet_bs1_seq4096_20step_receipts"
    )
    assert baseline["baseline_scope"] == "local_m4_only_not_gb10_parity"
    assert {
        row["case_id"] for row in baseline["reference_receipts"]
    } == {
        "lion8bit_sym_lr1e-4",
        "adam8bit_sym_lr1e-4",
        "adam8bit_dyn_lr1e-4",
    }
    assert any(
        row["meets_900_tok_s_baseline"] is True
        for row in baseline["reference_receipts"]
    )
    assert matrix["command_sets"] == [
        "dry_run",
        "smoke_1step",
        "real_20step",
        "real_100step",
    ]
    cases = {case["case_id"]: case for case in matrix["cases"]}
    assert len(cases) == 24
    assert len(matrix["real_20step_commands"]) == 24
    assert len(matrix["real_100step_commands"]) == 24
    assert len(matrix["dry_run_commands"]) == 24
    assert len(matrix["smoke_commands"]) == 24
    assert cases["bf16_adamw_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["bf16_adamw_20step"]["command"]
    assert "--optimizer adamw" in cases["bf16_adamw_20step"]["command"]
    assert "--dry-run-json" in cases["bf16_adamw_20step"]["dry_run_command"]
    assert "--steps 1" in cases["bf16_adamw_20step"]["smoke_command"]
    assert "--steps 20" in cases["bf16_adamw_20step"]["real_20step_command"]
    assert "--steps 100" in matrix["real_100step_commands"][0]
    assert "--require-loss-decrease" in matrix["real_100step_commands"][0]
    assert cases["bf16_muon_20step"]["supported"] is True
    assert "--optimizer muon" in cases["bf16_muon_20step"]["command"]
    assert cases["fp8_path_b_muon_adamw_20step"]["supported"] is True
    assert "--dtype fp8_path_b" in cases["fp8_path_b_muon_adamw_20step"]["command"]
    assert "--optimizer muon_adamw" in cases[
        "fp8_path_b_muon_adamw_20step"
    ]["command"]
    assert cases["fp8_path_c_muon_adamw_20step"]["supported"] is True
    assert "--dtype fp8_path_c" in cases["fp8_path_c_muon_adamw_20step"]["command"]
    assert "--optimizer muon_adamw" in cases[
        "fp8_path_c_muon_adamw_20step"
    ]["command"]
    assert cases["fp8_path_c_muon_20step"]["supported"] is True
    assert "--dtype fp8_path_c" in cases["fp8_path_c_muon_20step"]["command"]
    assert "--optimizer muon" in cases["fp8_path_c_muon_20step"]["command"]
    assert cases["int8_muon_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_muon_20step"]["command"]
    assert "--optimizer int8" in cases["int8_muon_20step"]["command"]
    assert cases["int8_muon_adamw_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_muon_adamw_20step"]["command"]
    assert "--optimizer int8" in cases["int8_muon_adamw_20step"]["command"]
    assert cases["int8_adam8bit_20step"]["supported"] is True
    assert cases["int8_lion8bit_20step"]["supported"] is True
    assert cases["int8_adamw_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_adamw_20step"]["command"]
    assert "--optimizer adam8bit" in cases["int8_adamw_20step"]["command"]
    assert cases["int8_lion_20step"]["supported"] is True
    assert "--dtype bfloat16" in cases["int8_lion_20step"]["command"]
    assert "--optimizer lion8bit" in cases["int8_lion_20step"]["command"]


def _receipt_args_for_regression_report(
    tmp_path: Path,
    *,
    dtype: str = "bfloat16",
    optimizer: str = "adamw",
    pattern: str = "M",
    depth: str = "1",
    dsa_a_layer_ranks: str = "",
) -> argparse.Namespace:
    return m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            dtype,
            "--optimizer",
            optimizer,
            "--pattern",
            pattern,
            "--depth",
            depth,
            "--dsa-a-layer-ranks",
            dsa_a_layer_ranks,
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )


def test_regression_report_rejects_bogus_tokens_per_second_claim(
    tmp_path: Path,
) -> None:
    args = _receipt_args_for_regression_report(tmp_path)
    train_payload = {
        "step_metrics": [
            {
                "loss": 2.0,
                "seconds": 1.0,
                "ntokens": 128,
                "tokens_per_second": 999_999.0,
                "updated": True,
            }
        ],
        "kernel_dispatch": [
            {
                "op_name": "mamba3_mimo",
                "path": "path_b",
                "kernel_used": "metal_kernel_fwd_v1",
            }
        ],
    }

    report = m04_train_step.regression_report_payload(
        args,
        config=args,
        train_payload=train_payload,
        optimizer=adamw_identity(),
        memory_after={"peak_memory_bytes": 4096},
        tokens_per_second=999_999.0,
        status="ok",
    )

    claim_gate = report["throughput"]["claim_gate"]
    step_check = claim_gate["step_checks"][0]
    assert claim_gate["ok"] is False
    assert claim_gate["bogus_tok_sec_claim_detected"] is True
    assert claim_gate["reported_tokens_per_second_finite"] is True
    assert claim_gate["step_rates_consistent"] is False
    assert step_check["expected_tokens_per_second"] == 128.0
    assert step_check["reported_tokens_per_second"] == 999_999.0
    assert step_check["rate_consistent_with_ntokens_and_seconds"] is False
    assert report["gate_summary"]["tokens_per_second_claim_ok"] is False
    assert report["gate_summary"]["bogus_tok_sec_claim_detected"] is True


@pytest.mark.parametrize(
    (
        "dtype",
        "optimizer",
        "pattern",
        "depth",
        "dsa_a_layer_ranks",
        "kernel_dispatch",
        "expected_path_b",
        "expected_path_c",
        "expected_fallback",
    ),
    [
        (
            "bfloat16",
            "adamw",
            "M",
            "1",
            "",
            [
                {
                    "op_name": "mamba3_mimo",
                    "path": "path_b",
                    "kernel_used": "metal_kernel_fwd_v1",
                }
            ],
            True,
            False,
            None,
        ),
        (
            "fp8_path_c",
            "lion",
            "A",
            "1",
            "0",
            [
                {
                    "op_name": "mamba3_mimo",
                    "path": "path_c",
                    "kernel_used": "mamba3_mimo_path_c",
                },
                {
                    "op_name": "m2rnn",
                    "path": "path_c",
                    "kernel_used": "path_c_tilelang_dsl_packed",
                },
                {
                    "op_name": "sparse_mla",
                    "path": "path_c",
                    "kernel_used": "sparse_mla_fp8_path_c_apply",
                },
            ],
            False,
            True,
            None,
        ),
    ],
)
def test_regression_report_records_path_b_vs_path_c_receipt_gate_fields(
    tmp_path: Path,
    dtype: str,
    optimizer: str,
    pattern: str,
    depth: str,
    dsa_a_layer_ranks: str,
    kernel_dispatch: list[dict[str, Any]],
    expected_path_b: bool,
    expected_path_c: bool,
    expected_fallback: str | None,
) -> None:
    args = _receipt_args_for_regression_report(
        tmp_path,
        dtype=dtype,
        optimizer=optimizer,
        pattern=pattern,
        depth=depth,
        dsa_a_layer_ranks=dsa_a_layer_ranks,
    )
    train_payload = {
        "mean_loss": 1.5,
        "step_metrics": [
            {
                "loss": 2.0,
                "seconds": 0.5,
                "ntokens": 128,
                "tokens_per_second": 256.0,
                "updated": True,
            },
            {
                "loss": 1.0,
                "seconds": 0.25,
                "ntokens": 128,
                "tokens_per_second": 512.0,
                "updated": True,
            },
        ],
        "kernel_dispatch": kernel_dispatch,
    }

    report = m04_train_step.regression_report_payload(
        args,
        config=args,
        train_payload=train_payload,
        optimizer=adamw_identity(name="Lion" if optimizer == "lion" else "AdamW"),
        memory_after={"peak_memory_bytes": 123_456},
        tokens_per_second=384.0,
        status="ok",
    )

    route = report["route_dispatch"]
    summary = report["gate_summary"]
    assert report["dtype"]["requested"] == dtype
    assert report["optimizer"]["key"] == optimizer
    assert report["memory"]["peak_memory_bytes"] == 123_456
    assert route["path_b_observed"] is expected_path_b
    assert route["path_c_observed"] is expected_path_c
    assert route["fallback_reason"] == expected_fallback
    assert summary["dtype"] == dtype
    assert summary["optimizer"] == optimizer
    assert summary["path_b_observed"] is expected_path_b
    assert summary["path_c_observed"] is expected_path_c
    assert summary["fallback_reason"] == expected_fallback
    assert summary["tokens_per_second_claim_ok"] is True
    assert summary["bogus_tok_sec_claim_detected"] is False
    if dtype == "fp8_path_c":
        assert route["requested_path_c_ops"] == ["m2rnn", "mamba3_mimo", "sparse_mla"]
        assert route["unobserved_requested_path_c_ops"] == []
        assert route["producer_missing"] is False
        assert route["producer_unobserved"] is False
        assert report["fp8_path_c_producer_gate"]["ok"] is True
        assert summary["fp8_path_c_producer_status"] == (
            m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
        )
    else:
        assert route["requested_path_c_ops"] == []
        assert report["fp8_path_c_producer_gate"]["required"] is False
        assert summary["fp8_path_c_producer_status"] == "not_requested"


def test_checked_in_receipt_records_full_m0_4_acceptance() -> None:
    payload = json.loads(BASELINE_RECEIPT.read_text())

    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["full_m0_4_acceptance_claim"] is True
    assert payload["workload"]["synthetic"] is False
    assert payload["workload"]["data_format"] == "parquet"
    assert payload["workload"]["mode"] == "eager"
    assert payload["workload"]["steps_requested"] == 100
    assert payload["workload"]["batch_size"] == 1
    assert payload["workload"]["seq_len"] == 128
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is True
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is True
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is True
    assert payload["acceptance_gate"]["optimizer_identity_ok"] is True
    assert payload["acceptance_gate"]["adamw_ok"] is True
    assert payload["acceptance_gate"]["fp32_adamw_master_moments_ok"] is True
    assert payload["acceptance_gate"]["observed_adamw_master_moment_dtypes"]
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["m4_runtime_metadata_ok"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_blockers"] == []
    assert payload["acceptance_gate"]["full_target_dataset_blocker"] is None
    assert payload["training"]["steps_completed"] == 100
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["loss_decreased"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["training"]["final_loss"] < payload["training"]["initial_loss"]
    assert payload["training"]["optimizer"]["name"] == "AdamW"
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["model"]["name"] == "local_gb10_quarter"
    assert payload["model"]["profile_matches_required"] is True
    assert payload["baseline_row"]["model"] == "local_gb10_quarter"
    assert payload["baseline_row"]["route"] == "AEMEAEMEAEMRA"
    assert payload["baseline_row"]["seq_len"] == 128
    assert payload["acceptance_blockers"] == []


def test_synthetic_one_step_writes_finite_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output))
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert payload["status"] == "ok"
    assert payload["workload"]["synthetic"] is True
    assert payload["workload"]["data_format"] == "npz"
    assert payload["workload"]["model_profile"] == "hybrid_tiny"
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["full_target_dataset_blocker"]
    assert payload["training"]["final_loss"] > 0
    assert payload["training"]["step_metrics"][0]["updated"] is True
    interpretation = payload["timing"]["throughput_interpretation"]
    assert interpretation["reported_tokens_per_second_kind"] == (
        "loss_target_tokens_per_second"
    )
    assert interpretation["denominator"] == "sum(step_metrics[].ntokens)"
    assert interpretation["input_tokens_per_step"] == 4
    assert interpretation["nominal_target_tokens_per_step"] == 3
    assert interpretation["measured_target_tokens_per_step"] == [3]
    assert interpretation["workload_scope"] == "tiny_or_hybrid_smoke"
    assert interpretation["production_shape"] is False
    assert interpretation["excluded_from_step_timer"] == [
        "dataset construction",
        "next(batches) parquet/npz batch fetch",
        "model allocation",
        "optimizer initialization",
        "receipt JSON serialization",
        "post-step cache clear cadence",
    ]
    assert payload["memory"]["peak_memory_bytes"] is None or (
        payload["memory"]["peak_memory_bytes"] >= 0
    )


def test_throughput_interpretation_marks_short_local_gb10_sequence() -> None:
    config = m04_train_step.TrainHybridTinyConfig(
        model_profile="local_gb10_quarter",
        data_format="parquet",
        batch_size=1,
        seq_len=1024,
        steps=1,
        dtype="bfloat16",
        grad_checkpoint=True,
    )
    interpretation = m04_train_step.throughput_interpretation_payload(
        config,
        train_payload={"tokens_per_second": 480.0},
        step_metrics=[
            {
                "ntokens": 1023,
                "seconds": 2.0,
                "tokens_per_second": 511.5,
            }
        ],
        tokens_per_second_values=[511.5],
    )

    assert interpretation["workload_scope"] == "short_sequence_full_profile_smoke"
    assert interpretation["production_seq_len"] == 4096
    assert interpretation["production_shape"] is False
    assert interpretation["input_tokens_per_step"] == 1024
    assert interpretation["nominal_target_tokens_per_step"] == 1023
    assert interpretation["total_input_tokens"] == 1024
    assert interpretation["total_target_tokens"] == 1023
    assert interpretation["input_tokens_per_second"] == 512.0
    assert interpretation["target_tokens_per_second"] == 511.5
    assert "underfills" in interpretation["warning"]


def test_synthetic_grad_checkpoint_receipt_marks_gate_without_m0_4_claim(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output), "--grad-checkpoint")
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["workload"]["model_profile"] == "hybrid_tiny"
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["training"]["grad_checkpoint"]["expectation_satisfied"] is True
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "local_gb10_quarter_preflight_ok",
        "model_identity_ok",
    }.issubset(set(payload["acceptance_gate"]["full_local_gb10_quarter_gate_blockers"]))


def test_require_loss_decrease_fails_single_step_but_writes_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output), "--require-loss-decrease")

    assert result.returncode == 2
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "failed"
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["loss_decrease_required"] is True
    assert payload["training"]["loss_decrease_satisfied"] is False


def test_missing_dataset_dry_run_reports_blocked_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    missing = tmp_path / "missing.parquet"

    result = run_script(
        "--data-path",
        str(missing),
        "--dry-run-json",
        "--output",
        str(output),
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "blocked"
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert payload["regression_report"]["fallback_reason"].startswith(
        "missing_dataset:"
    )
    assert payload["blockers"][0]["type"] == "missing_dataset"
    assert payload["training"]["steps_completed"] == 0
    assert payload["workload"]["data_path"] == str(missing)
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_fp8_path_c_training_dtype_route_blocks_missing_sparse_mla_producer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"

    result = run_script(
        *tiny_args(output),
        "--dtype",
        "fp8_path_c",
    )

    assert result.returncode == 2, result.stderr
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "blocked"
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert payload["blockers"][0]["type"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert payload["blockers"][0]["reason"].startswith("fp8_path_c requested")
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["kernel_dispatch"] == []
    assert payload["workload"]["dtype"] == "fp8_path_c"
    assert payload["workload"]["optimizer"]["key"] == "adamw"
    precision_route = payload["workload"]["precision_route"]
    assert precision_route["requested"] == "fp8_path_c"
    assert precision_route["kind"] == "fp8_path_c"
    assert precision_route["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert precision_route["blocker_type"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert precision_route["carrier_dtype"] == "bfloat16"
    assert precision_route["native_fp8_producer_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert precision_route["kernel_surface_status"] == (
        m04_train_step.FP8_PATH_C_KERNEL_SURFACE_STATUS
    )
    assert precision_route["kernel_surface_available"] is True
    assert precision_route["full_end_to_end_training_available"] is False
    assert precision_route["bridge_target"] == m04_train_step.FP8_PATH_C_BRIDGE_TARGET
    assert precision_route["bridge_status"] == m04_train_step.FP8_PATH_C_BRIDGE_STATUS
    assert precision_route["zero_copy_required"] is True
    assert precision_route["large_tensor_staging_allowed"] is False
    assert precision_route["hidden_wrapper_quantization_allowed"] is False
    assert precision_route["kernel_boundary_quantization_allowed"] is False
    assert precision_route["prepared_buffers_configured"] is False
    producer = precision_route["sparse_mla_fp8_producer"]
    assert producer["configured"] is False
    assert producer["prepared_buffers_configured"] is False
    assert producer["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert producer["required_prepared_buffers"] == [
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    ]
    assert producer["hidden_wrapper_quantization_allowed"] is False
    assert producer["kernel_boundary_quantization_allowed"] is False
    assert producer["producer_stage"] == m04_train_step.FP8_PATH_C_PRODUCER_STAGE
    assert producer["producer_quantization"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_QUANTIZATION
    )
    route = payload["training"]["fp8_path_c_training_route"]
    assert route["requested"] is True
    assert route["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert route["blocker_type"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert route["reason"].startswith("producer_missing:")
    assert route["carrier_dtype"] == "bfloat16"
    assert route["native_fp8_producer_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert route["sparse_mla_fp8_producer"] == producer
    assert route["kernel_surface_status"] == (
        m04_train_step.FP8_PATH_C_KERNEL_SURFACE_STATUS
    )
    assert route["kernel_surface_available"] is True
    assert route["full_end_to_end_training_available"] is False
    assert route["end_to_end_training_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert route["direct_mx_array_artifact_call_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert route["bridge_target"] == m04_train_step.FP8_PATH_C_BRIDGE_TARGET
    assert route["bridge_status"] == m04_train_step.FP8_PATH_C_BRIDGE_STATUS
    assert route["bridge_evidence"] == {
        "mlx_array_exports_dlpack": True,
        "mlx_public_from_dlpack_available": False,
        "tvm_ffi_from_dlpack_available": True,
        "mlx_metal_dlpack_device": "kDLMetal:0",
        "tvm_from_dlpack_device": "metal:0",
        "native_mlx_array_wrapper_linked": True,
        "native_tvm_ffi_graph_outputs": True,
        "dlpack_used_for_path_c_graph_bridge": False,
        "standalone_mlx_to_tvm_metal_kernel_verified": True,
        "m04_bridge_wired": True,
    }
    assert route["zero_copy_required"] is True
    assert route["large_tensor_staging_allowed"] is False
    assert route["hidden_wrapper_quantization_allowed"] is False
    assert route["kernel_boundary_quantization_allowed"] is False
    assert route["prepared_buffers_configured"] is False
    assert route["hidden_dtype_cast_allowed"] is False
    assert route["hidden_shape_staging_allowed"] is False
    assert route["fallback_to_path_b_allowed"] is False
    assert route["selected_action"] == "fail_closed_producer_missing"
    assert route["kernel_policy_env"] == {
        "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
        "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
        "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
    }
    assert {
        "fp8_scaled_vecmat_path_c",
        "mamba3_mimo_path_c",
        "m2rnn_path_c",
        "sparse_mla_fp8_path_c_apply",
        "matmul_tl_fp8_scaled_matmul",
    } == {surface["name"] for surface in route["available_path_c_surfaces"]}
    surfaces = {
        surface["name"]: surface for surface in route["available_path_c_surfaces"]
    }
    assert surfaces["matmul_tl_fp8_scaled_matmul"]["kernel_surface_available"] is True
    assert surfaces["mamba3_mimo_path_c"]["training_surface"] is False
    assert surfaces["mamba3_mimo_path_c"]["full_path_c_backward_available"] is True
    assert surfaces["mamba3_mimo_path_c"]["default_backward_route"] == "path_b"
    assert surfaces["mamba3_mimo_path_c"]["fp8_route_auto_selected"] is True
    assert surfaces["m2rnn_path_c"]["training_surface"] is True
    assert surfaces["m2rnn_path_c"]["fallback_to_path_b_allowed"] is False
    assert surfaces["m2rnn_path_c"]["fp8_route_auto_selected"] is True
    assert surfaces["sparse_mla_fp8_path_c_apply"]["training_surface"] is False
    assert surfaces["sparse_mla_fp8_path_c_apply"]["producer_required"] is True
    assert surfaces["sparse_mla_fp8_path_c_apply"]["producer_status"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    )
    assert (
        surfaces["sparse_mla_fp8_path_c_apply"]["backward_surface"]
        == "native_tvm_ffi_graph_output_scatter"
    )
    assert surfaces["sparse_mla_fp8_path_c_apply"]["fallback_backward_surface"] == (
        "prepared_fp8_path_b_reference_vjp"
    )
    assert surfaces["sparse_mla_fp8_path_c_apply"]["default_backward_route"] == (
        "path_c"
    )
    assert (
        "FP8 parameter/weight producers that create the required dtype/layout "
        "before matmul kernel boundaries" in route["missing_training_surfaces"]
    )
    assert (
        "absorbed MLA producer split for NoPE/RoPE KV layout and calibrated "
        "separate K/V scale lifecycle" in route["missing_training_surfaces"]
    )
    assert (
        route["higher_level_owner"]["sparse_mla_fp8_next_owner"]
        == m04_train_step.FP8_PATH_C_PRODUCER_OWNER
    )
    assert "without DSA Sparse-MLA producer" in (
        route["higher_level_owner"]["current_m04_route_owner"]
    )
    dispatch_report = payload["regression_report"]["route_dispatch"]
    assert dispatch_report["requested_path_c_ops"] == ["m2rnn", "mamba3_mimo", "sparse_mla"]
    assert dispatch_report["observed_path_c_ops"] == []
    assert dispatch_report["unobserved_requested_path_c_ops"] == [
        "m2rnn",
        "mamba3_mimo",
        "sparse_mla",
    ]
    assert dispatch_report["producer_missing"] is True
    assert dispatch_report["fp8_sparse_mla_producer"] == producer
    assert dispatch_report["fallback_detected"] is True
    assert dispatch_report["fallback_reason"].startswith("producer_missing:")
    producer_gate = payload["regression_report"]["fp8_path_c_producer_gate"]
    assert producer_gate["required"] is True
    assert producer_gate["ok"] is False
    assert producer_gate["status"] == m04_train_step.FP8_PATH_C_PRODUCER_MISSING_STATUS
    assert producer_gate["fail_closed"] is True
    assert producer_gate["producer"] == producer
    assert producer_gate["reason"].startswith("producer_missing:")


def test_fp8_path_c_dsa_attention_route_metadata_is_configured(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")

    producer = m04_train_step.sparse_mla_fp8_producer_payload(config)
    producer_gate = m04_train_step.fp8_path_c_producer_gate_payload(config)
    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )

    assert config.dsa_a_layer_ranks == (0,)
    assert producer["configured"] is True
    assert producer["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer["dsa_layer_numbers"] == [1]
    assert producer["owner"] == m04_train_step.FP8_PATH_C_PRODUCER_OWNER
    assert producer["prepared_buffers_configured"] is True
    assert producer["producer_stage"] == m04_train_step.FP8_PATH_C_PRODUCER_STAGE
    assert producer["producer_quantization"] == (
        m04_train_step.FP8_PATH_C_PRODUCER_QUANTIZATION
    )
    assert producer["hidden_wrapper_quantization_allowed"] is False
    assert producer["kernel_boundary_quantization_allowed"] is False
    assert producer["required_prepared_buffers"] == [
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    ]
    assert route["status"] == m04_train_step.FP8_PATH_C_SPLIT_TRAINING_STATUS
    assert route["blocker_type"] is None
    assert route["prepared_buffers_configured"] is True
    assert route["hidden_wrapper_quantization_allowed"] is False
    assert route["kernel_boundary_quantization_allowed"] is False
    assert route["split_end_to_end_training_available"] is True
    assert route["full_end_to_end_training_available"] is False
    assert route["fused_train_block_runtime_available"] is False
    assert route["fused_train_block_blocker_type"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS
    )
    assert route["direct_mx_array_artifact_call_status"] == (
        "m04_uses_split_model_graph_route_not_fused_train_block"
    )
    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["path_c_fusion"]["mode"] == "auto"
    assert route["path_c_fusion"]["status"] == "plan_ready_not_default"
    assert route["path_c_fusion"]["runtime_training_binding"]["status"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS
    )
    assert route["path_c_fusion"]["runtime_training_binding"][
        "runtime_uses_fused_train_block"
    ] is False
    expected_physical_banks = [
        "path_c_float32_activation_abi_bank",
        "path_c_float32_parameter_abi_bank",
        "path_c_float32_state_abi_bank",
        "path_c_uint8_abi_bank",
        "path_c_float32_attention_abi_bank",
        "path_c_int32_abi_bank",
        "path_c_float32_activation_gradient_abi_bank",
        "path_c_float32_parameter_gradient_abi_bank",
    ]
    assert route["path_c_fusion"]["runtime_training_binding"][
        "required_bank_buffers"
    ] == expected_physical_banks
    assert route["path_c_fusion"]["runtime_training_binding"][
        "missing_bank_buffers"
    ] == expected_physical_banks
    assert route["path_c_fusion"]["runtime_training_binding"][
        "no_hidden_allocation_policy"
    ] is True
    assert route["path_c_fusion"]["lowering_boundary"] == "tilelang_tvm_ir"
    assert route["path_c_fusion"]["compiler"] == "tilelang.engine.fusion"
    assert route["path_c_fusion"]["requires_msl_post_fusion"] is False
    assert route["path_c_fusion"]["large_tensor_staging_allowed"] is False
    selected_schedule_id = route["path_c_fusion"]["graph_construction"][
        "selected_model_region_schedule_id"
    ]
    assert selected_schedule_id.startswith("path_c_descriptor_chain_")
    assert route["path_c_fusion"]["graph_construction"] == {
        "builder": "PathCFusionScheduleOptimizer",
        "input_model": "local_gb10_quarter_profile_path_c_bricks",
        "route_symbols": list("AEMEAEMEAEMRA"),
        "region_source": "build_path_c_model_regions_from_model",
        "edge_policy": "infer_from_outputs_to_inputs",
        "dependency_ordering": "topological",
        "schedule_construction": "dynamic_brick_descriptors",
        "optimization_scope": "all_discovered_supported_path_c_brick_segments",
        "static_acceptance_fixture_used_for_selection": False,
        "selected_model_region": "local_gb10_quarter_path_c_10_12",
        "selected_model_region_op_signature": [
            "entry_rmsnorm",
            "mamba3_mimo",
            "residual_rmsnorm",
            "m2rnn",
            "residual_rmsnorm",
            "attention_qkv_projection",
            "sparse_mla_fp8_apply",
        ],
        "selected_model_region_schedule_id": selected_schedule_id,
        "preset_only": False,
    }
    model_route_candidates = route["path_c_fusion"]["model_route_candidates"]
    assert model_route_candidates["profile"] == "local_gb10_quarter"
    assert "".join(model_route_candidates["route_symbols"]) == "AEMEAEMEAEMRA"
    assert model_route_candidates["region_source"] == (
        "build_path_c_model_regions_from_model"
    )
    assert model_route_candidates["selection_policy"] == (
        "largest_supported_contiguous_route_segment"
    )
    assert model_route_candidates["selected_candidate"]["name"] == (
        "local_gb10_quarter_path_c_10_12"
    )
    assert len(model_route_candidates["candidate_regions"]) == 1
    candidate = model_route_candidates["candidate_regions"][0]
    assert candidate["name"] == "local_gb10_quarter_path_c_10_12"
    assert candidate["brick_names"] == [
        "local_gb10_quarter_brick_10_M",
        "local_gb10_quarter_brick_11_R",
        "local_gb10_quarter_brick_12_A",
    ]
    assert candidate["brick_kinds"] == ["mamba3", "m2rnn", "attention"]
    assert candidate["brick_route_symbols"] == ["M", "R", "A"]
    assert candidate["node_names"] == [
        "local_gb10_quarter_brick_10_M_entry_rmsnorm",
        "local_gb10_quarter_brick_10_M",
        "local_gb10_quarter_brick_10_M_residual_norm",
        "local_gb10_quarter_brick_11_R",
        "local_gb10_quarter_brick_11_R_residual_norm",
        "local_gb10_quarter_brick_12_A_qkv_projection",
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply",
    ]
    assert candidate["op_signature"] == [
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    ]
    assert candidate["edge_count"] == 11
    assert candidate["z3_sync"] == {
        "enabled": True,
        "objective": "minimize_sync_async",
        "proof_required": True,
    }
    assert candidate["schedule_target"]["schedule_id"].startswith(
        "path_c_descriptor_chain_"
    )
    assert candidate["schedule_target"]["implementation_kind"] == "production"
    assert candidate["schedule_target"]["schedule_generator"] == (
        PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    assert "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight" in (
        candidate["schedule_target"]["required_real_abi_inputs"]
    )
    assert "local_gb10_quarter_brick_12_A_qkv_projection_attention_q_proj_weight" in (
        candidate["schedule_target"]["required_real_abi_inputs"]
    )
    assert candidate["plan"]["schedule_contract_status"] == (
        "registered_not_lowered"
    )
    assert candidate["plan"]["autograd_status"] == "ready"
    assert route["path_c_fusion"]["fullgraph_required"] is True
    assert route["path_c_fusion"]["graph_break_policy"] == "fail_closed"
    assert route["path_c_fusion"]["schedule_name"] == (
        "local_gb10_quarter_path_c_10_12:descriptor_generated_fwd_bwd"
    )
    assert route["path_c_fusion"]["schedule_status"] == "ready"
    assert route["path_c_fusion"]["schedule_registry"] == {
        "selector": "PathCFusionScheduleRegistry",
        "match_policy": "op_signature_or_descriptor_chain",
        "selected_schedule_id": selected_schedule_id,
        "selected_schedule_name": (
            "local_gb10_quarter_path_c_10_12:descriptor_generated_fwd_bwd"
        ),
        "selected_from": "selected_model_region",
    }
    assert route["path_c_fusion"]["schedule_contract"]["status"] == (
        "registered_not_lowered"
    )
    assert route["path_c_fusion"]["schedule_contract"][
        "declared_implementation_kind"
    ] == "production"
    assert route["path_c_fusion"]["schedule_contract"][
        "declared_schedule_id"
    ] == selected_schedule_id
    assert "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight" in (
        route["path_c_fusion"]["schedule_contract"][
            "declared_required_real_abi_inputs"
        ]
    )
    assert route["path_c_fusion"]["production_schedule"]["schedule_id"] == (
        selected_schedule_id
    )
    assert route["path_c_fusion"]["production_schedule"]["schedule_name"] == (
        "local_gb10_quarter_path_c_10_12:descriptor_generated_fwd_bwd"
    )
    assert route["path_c_fusion"]["production_schedule"]["source"] == (
        "selected_model_region"
    )
    assert route["path_c_fusion"]["production_schedule"]["shape_env_key"]
    assert route["path_c_fusion"]["schedule_contract"]["shape_env_key"] == (
        route["path_c_fusion"]["production_schedule"]["shape_env_key"]
    )
    assert route["path_c_fusion"]["production_schedule"]["implementation_kind"] == (
        "production"
    )
    assert route["path_c_fusion"]["production_schedule"]["implementation_status"] == (
        "ready"
    )
    assert route["path_c_fusion"]["production_schedule"]["trusted_by_default"] is False
    required_codegen_steps = route["path_c_fusion"]["production_schedule"][
        "required_codegen_steps"
    ]
    assert required_codegen_steps[:3] == [
        "dynamic_region_graph_walk",
        "brick_descriptor_chain_resolution",
        "single_entry_tilelang_region",
    ]
    assert "real_model_parameter_abi_contract" in required_codegen_steps
    assert "mamba3_scan_descriptor" in required_codegen_steps
    assert "m2rnn_descriptor" in required_codegen_steps
    assert "m2rnn_bwd_descriptor" in required_codegen_steps
    assert "m2rnn_bwd_final_gradient_owner_outputs" in required_codegen_steps
    assert "attention_qkv_projection_bwd_descriptor" in required_codegen_steps
    assert "sparse_mla_fp8_apply_row_phased_prepared_apply" in required_codegen_steps
    assert "z3_sync_async_schedule_points" in required_codegen_steps
    assert "cache_key_shape_specialization_audit" in required_codegen_steps
    assert (
        route["path_c_fusion"]["production_schedule"]["schedule_generator"]
        == PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    assert (
        route["path_c_fusion"]["production_schedule"][
            "schedule_generator_status"
        ]
        == "production_region_fragments"
    )
    assert route["path_c_fusion"]["production_schedule"]["internal_buffer_policy"] == (
        "row_local_hidden"
    )
    assert route["path_c_fusion"]["production_schedule"]["loop_policy"] == (
        "row_phased_hidden"
    )
    assert route["path_c_fusion"]["production_schedule"]["buffer_extent"] == (
        m04_train_step.MATRIX_SEQ_LEN
    )
    assert route["path_c_fusion"]["production_schedule"]["loop_extent"] == (
        m04_train_step.MATRIX_SEQ_LEN
        * m04_train_step.REQUIRED_MODEL_GEOMETRY["hidden_size"]
    )
    assert route["path_c_fusion"]["production_schedule"]["brick_ops"] == (
        route["path_c_fusion"]["schedule_contract"]["op_signature"]
    )
    assert set(
        route["path_c_fusion"]["production_schedule"]["brick_schedule_families"]
    ) == {"loop_descriptor_dataflow"}
    assert "mamba3_mimo:descriptor_codegen_ready" in (
        route["path_c_fusion"]["production_schedule"]["brick_descriptor_statuses"]
    )
    fragment_statuses = route["path_c_fusion"]["production_schedule"][
        "brick_production_fragment_statuses"
    ]
    fragment_reasons = route["path_c_fusion"]["production_schedule"][
        "brick_production_fragment_reasons"
    ]
    fragment_blockers = route["path_c_fusion"]["production_schedule"][
        "brick_production_fragment_blockers"
    ]
    assert route["path_c_fusion"]["production_schedule"][
        "production_fragments_complete"
    ] is True
    assert fragment_blockers == []
    assert any(
        status.startswith("mamba3_mimo:production_region_inlined:")
        for status in fragment_statuses
    )
    assert any(
        reason.startswith(
            "mamba3_mimo:production_region_inlined:"
            "row-phased descriptor codegen fuses Mamba3 dense input projection"
        )
        for reason in fragment_reasons
    )
    assert not any(
        blocker.startswith("mamba3_mimo:")
        for blocker in fragment_blockers
    )
    assert any(
        status.startswith("residual_rmsnorm:production_region_inlined:")
        for status in fragment_statuses
    )
    assert any(
        status.startswith("residual_rmsnorm_bwd:production_region_inlined:")
        for status in fragment_statuses
    )
    assert any(
        reason.startswith(
            "residual_rmsnorm_bwd:production_region_inlined:"
            "row-phased descriptor codegen recomputes residual/RMSNorm"
        )
        for reason in fragment_reasons
    )
    assert not any(
        blocker.startswith("residual_rmsnorm_bwd:")
        for blocker in fragment_blockers
    )
    assert any(
        reason.startswith(
            "residual_rmsnorm:production_region_inlined:"
            "row-phased descriptor codegen emits the residual bridge"
        )
        for reason in fragment_reasons
    )
    assert not any(
        blocker.startswith("residual_rmsnorm:production_region_inlined:")
        for blocker in fragment_blockers
    )
    assert any(
        status.startswith("m2rnn:production_region_inlined:")
        for status in fragment_statuses
    )
    assert not any(blocker.startswith("m2rnn:") for blocker in fragment_blockers)
    assert (
        route["path_c_fusion"]["production_schedule"][
            "real_abi_contract_complete"
        ]
        is True
    )
    assert (
        route["path_c_fusion"]["production_schedule"]["missing_real_abi_inputs"]
        == []
    )
    assert "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight" in (
        route["path_c_fusion"]["production_schedule"]["required_external_buffers"]
    )
    assert "local_gb10_quarter_brick_10_M_residual_norm_weight" in (
        route["path_c_fusion"]["production_schedule"][
            "required_real_abi_inputs"
        ]
    )
    assert "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight:(67321856,)" in (
        route["path_c_fusion"]["production_schedule"][
            "required_real_abi_input_shapes"
        ]
    )
    assert "local_gb10_quarter_brick_10_M_residual_norm_weight:(3584,)" in (
        route["path_c_fusion"]["production_schedule"][
            "required_real_abi_input_shapes"
        ]
    )
    assert (
        route["path_c_fusion"]["production_schedule"]["contract_key"]
        == route["path_c_fusion"]["schedule_contract"]["key"]
    )
    required_internal_buffers = route["path_c_fusion"]["schedule_contract"][
        "required_internal_buffers"
    ]
    for expected_internal in [
        "local_gb10_quarter_brick_10_M_entry_rmsnorm_hidden",
        "local_gb10_quarter_brick_10_M_delta",
        "local_gb10_quarter_brick_10_M_residual_norm_hidden",
        "local_gb10_quarter_brick_11_R_delta",
        "local_gb10_quarter_brick_11_R_residual_norm_hidden",
        "local_gb10_quarter_brick_12_A_qkv_projection_indices",
        "local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad",
        "local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad",
        "local_gb10_quarter_brick_10_M_residual_norm_hidden_grad",
        "local_gb10_quarter_brick_10_M_delta_grad",
        "local_gb10_quarter_brick_10_M_entry_rmsnorm_hidden_grad",
    ]:
        assert expected_internal in required_internal_buffers
    assert route["path_c_fusion"]["production_compile_receipt"]["status"] == "mismatch"
    assert route["path_c_fusion"]["production_compile_receipt"]["verified"] is False
    assert route["path_c_fusion"]["production_compile_receipt"]["native_compile_ok"] is True
    assert (
        route["path_c_fusion"]["production_compile_receipt"][
            "runtime_execution_status"
        ]
        == "compile_only_not_runtime_ready"
    )
    assert (
        route["path_c_fusion"]["production_compile_receipt"]["runtime_smoke_mode"]
        == "production_1b"
    )
    assert (
        route["path_c_fusion"]["production_compile_receipt"][
            "runtime_smoke_actually_executed"
        ]
        is False
    )
    assert (
        route["path_c_fusion"]["production_compile_receipt"][
            "production_runtime_smoke_uses_fused_train_block"
        ]
        is False
    )
    assert route["path_c_fusion"]["production_compile_receipt"]["failed_checks"] == [
        "runtime_execution_ready",
        "production_runtime_smoke_ok",
        "production_smoke_uses_fused_train_block",
    ]
    assert (
        route["path_c_fusion"]["production_compile_receipt"]["checks"][
            "schedule_id_matches"
        ]
        is True
    )
    assert [blocker["kind"] for blocker in route["path_c_fusion"]["schedule_blockers"]] == [
        "selected_model_schedule_not_default",
        "production_schedule_not_compile_verified",
        "fused_train_block_runtime_not_bound",
        "production_1b_matrix_profile_missing",
    ]
    assert route["path_c_fusion"]["schedule_blockers"][0]["schedule_id"] == (
        selected_schedule_id
    )
    assert route["path_c_fusion"]["schedule_blockers"][0]["schedule_generator"] == (
        PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    assert route["path_c_fusion"]["semantic_blockers"] == []
    assert "diagnostic_raw_abi_region" not in route["path_c_fusion"]
    assert route["path_c_fusion"]["autograd_plan"]["status"] == "ready"
    assert route["path_c_fusion"]["autograd_plan"]["missing_backward_nodes"] == []
    assert route["path_c_fusion"]["autograd_plan"]["backward_nodes"] == [
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd",
        "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
        "local_gb10_quarter_brick_11_R_residual_norm_bwd",
        "local_gb10_quarter_brick_11_R_bwd",
        "local_gb10_quarter_brick_10_M_residual_norm_bwd",
        "local_gb10_quarter_brick_10_M_bwd",
        "local_gb10_quarter_brick_10_M_entry_rmsnorm_bwd",
    ]
    assert route["path_c_fusion"]["autograd_plan"]["backward_edges"] == [
        [
            "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_kv_scale_grad",
        ],
        [
            "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad",
        ],
        [
            "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_q_scale_grad",
        ],
        [
            "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
            "local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad",
        ],
        [
            "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
            "local_gb10_quarter_brick_11_R_residual_norm_bwd",
            "local_gb10_quarter_brick_11_R_residual_norm_hidden_grad",
        ],
        [
            "local_gb10_quarter_brick_11_R_residual_norm_bwd",
            "local_gb10_quarter_brick_11_R_bwd",
            "local_gb10_quarter_brick_11_R_delta_grad",
        ],
        [
            "local_gb10_quarter_brick_11_R_residual_norm_bwd",
            "local_gb10_quarter_brick_10_M_residual_norm_bwd",
            "local_gb10_quarter_brick_10_M_hidden_after_grad",
        ],
        [
            "local_gb10_quarter_brick_11_R_bwd",
            "local_gb10_quarter_brick_10_M_residual_norm_bwd",
            "local_gb10_quarter_brick_10_M_residual_norm_hidden_grad",
        ],
        [
            "local_gb10_quarter_brick_10_M_residual_norm_bwd",
            "local_gb10_quarter_brick_10_M_bwd",
            "local_gb10_quarter_brick_10_M_delta_grad",
        ],
        [
            "local_gb10_quarter_brick_10_M_bwd",
            "local_gb10_quarter_brick_10_M_entry_rmsnorm_bwd",
            "local_gb10_quarter_brick_10_M_entry_rmsnorm_hidden_grad",
        ],
    ]
    assert route["path_c_fusion"]["node_names"] == [
        "local_gb10_quarter_brick_10_M_entry_rmsnorm",
        "local_gb10_quarter_brick_10_M",
        "local_gb10_quarter_brick_10_M_residual_norm",
        "local_gb10_quarter_brick_11_R",
        "local_gb10_quarter_brick_11_R_residual_norm",
        "local_gb10_quarter_brick_12_A_qkv_projection",
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply",
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_bwd",
        "local_gb10_quarter_brick_12_A_qkv_projection_bwd",
        "local_gb10_quarter_brick_11_R_residual_norm_bwd",
        "local_gb10_quarter_brick_11_R_bwd",
        "local_gb10_quarter_brick_10_M_residual_norm_bwd",
        "local_gb10_quarter_brick_10_M_bwd",
        "local_gb10_quarter_brick_10_M_entry_rmsnorm_bwd",
    ]
    assert route["path_c_fusion"]["z3_sync"] == {
        "enabled": True,
        "objective": "minimize_sync_async",
        "candidates": ["sync", "async"],
        "proof_required": True,
    }
    assert route["path_c_fusion"]["acceptance_gate"]["ignores_bad_path_b"] is True
    assert route["path_c_fusion"]["acceptance_gate"]["requires_ready_fusion_plan"] is True
    assert (
        route["path_c_fusion"]["acceptance_gate"]["requires_compile_verified_single_kernel"]
        is True
    )
    assert (
        route["path_c_fusion"]["acceptance_gate"]["requires_verified_schedule_contract"]
        is True
    )
    assert (
        route["path_c_fusion"]["acceptance_gate"][
            "requires_complete_real_abi_contract"
        ]
        is True
    )
    assert route["path_c_fusion"]["acceptance_gate"]["current_plan_default_eligible"] is False
    assert route["path_c_fusion"]["cache_audit_required"] is True
    assert producer_gate["required"] is True
    assert producer_gate["ok"] is True
    assert producer_gate["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer_gate["fail_closed"] is False
    assert producer_gate["producer"] == producer


class _PathCBankLike:
    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


def _model_route_regions(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> tuple[Any, ...]:
    if model is None:
        _, _, regions = m04_train_step._local_gb10_path_c_model_regions()
        return regions
    return tuple(
        model.path_c_fusion_regions(
            include_backward=False,
            min_route_bricks=2,
            sequence_length=sequence_length,
        )
    )


def _model_route_physical_bank_buffers(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> dict[str, mx.array]:
    regions = _model_route_regions(model, sequence_length=sequence_length)
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    assert selected_region is not None
    scheduled = m04_train_step.plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    assert scheduled.schedule_target is not None
    prim_func = scheduled.schedule_target.schedule_template(scheduled.region)
    bridge = m04_train_step.plan_physical_abi_runtime_bridge(
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map"),
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes"),
    )
    bank_shapes = getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes")
    return {
        bank: mx.zeros(
            tuple(int(dim) for dim in tuple(bank_shapes[bank])),
            dtype=getattr(mx, str(bridge["bank_dtypes"][bank])),
        )
        for bank in bridge["required_bank_buffers"]
    }


def _model_route_physical_bank_owner(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
):
    regions = _model_route_regions(model, sequence_length=sequence_length)
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    assert selected_region is not None
    scheduled = m04_train_step.plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    assert scheduled.schedule_target is not None
    prim_func = scheduled.schedule_target.schedule_template(scheduled.region)
    owner_prefix = (
        str(getattr(model, "path_c_profile_name"))
        if model is not None and getattr(model, "path_c_profile_name", None)
        else "local_gb10_quarter"
    )
    return make_physical_abi_bank_owner(
        f"{owner_prefix}.path_c_physical_abi_banks",
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map"),
        getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_shapes"),
        _model_route_physical_bank_buffers(
            model,
            sequence_length=sequence_length,
        ),
    )


def _model_route_generated_stage_physical_abi(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    regions = _model_route_regions(model, sequence_length=sequence_length)
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    assert selected_region is not None
    scheduled = m04_train_step.plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    assert scheduled.schedule_target is not None
    abi_prim_func = scheduled.schedule_target.schedule_template(scheduled.region)
    stage_groups = m04_train_step.plan_path_c_descriptor_stage_groups(
        scheduled.region
    )
    stage_prim_funcs = tuple(
        m04_train_step._path_c_generated_stage_schedule_template(
            schedule_target=scheduled.schedule_target,
            region=scheduled.region,
            abi_prim_func=abi_prim_func,
            execution_stage=group.execution_stage,
            active_node_names=group.active_node_names,
            stage_suffix=group.stage_suffix,
            row_dispatch_mode=group.row_dispatch_mode,
        )(scheduled.region)
        for group in stage_groups
    )
    return m04_train_step._path_c_merged_physical_abi_for_prim_funcs(
        (abi_prim_func, *stage_prim_funcs)
    )


def _model_route_generated_stage_physical_bank_owner(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
):
    physical_abi_map, physical_abi_shapes = _model_route_generated_stage_physical_abi(
        model,
        sequence_length=sequence_length,
    )
    bridge = m04_train_step.plan_physical_abi_runtime_bridge(
        physical_abi_map,
        physical_abi_shapes,
    )
    owner_prefix = (
        str(getattr(model, "path_c_profile_name"))
        if model is not None and getattr(model, "path_c_profile_name", None)
        else "local_gb10_quarter"
    )
    return make_physical_abi_bank_owner(
        f"{owner_prefix}.path_c_physical_abi_banks",
        physical_abi_map,
        physical_abi_shapes,
        {
            bank: mx.zeros(
                tuple(int(dim) for dim in tuple(physical_abi_shapes[bank])),
                dtype=getattr(mx, str(bridge["bank_dtypes"][bank])),
            )
            for bank in bridge["required_bank_buffers"]
        },
    )


def test_fp8_path_c_route_metadata_fails_closed_when_planner_import_fails(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    def failing_planner(**_kwargs: Any) -> Mapping[str, Any]:
        raise ImportError("unit tilelang circular import")

    route = m04_train_step.fp8_path_c_training_route_payload(
        args,
        path_c_fusion_fn=failing_planner,
    )

    assert route["requested"] is False
    assert route["status"] == "not_requested"
    assert route["full_end_to_end_training_available"] is False
    assert route["fused_train_block_runtime_available"] is False
    path_c_fusion = route["path_c_fusion"]
    assert path_c_fusion["status"] == "path_c_fusion_planner_unavailable"
    assert path_c_fusion["planner_exception"]["type"] == "ImportError"
    assert path_c_fusion["runtime_training_binding"]["status"] == (
        "path_c_fusion_planner_unavailable"
    )
    assert path_c_fusion["model_route_candidates"]["profile"] == (
        "local_gb10_quarter"
    )


def _model_route_direct_chain(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
):
    regions = _model_route_regions(model, sequence_length=sequence_length)
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    assert selected_region is not None
    scheduled = m04_train_step.plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    return m04_train_step.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region,
        include_backward=True,
    )


def _model_route_direct_chain_logical_buffers(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> dict[str, _PathCBankLike]:
    buffers: dict[str, _PathCBankLike] = {}
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)
    assert chain.status == "ready"
    for name, spec in m04_train_step._path_c_direct_chain_required_logical_buffer_specs(
        chain
    ).items():
        candidate = _PathCBankLike(
            tuple(spec["shape"]),
            str(spec["dtype"]),
        )
        existing = buffers.setdefault(name, candidate)
        assert existing.shape == candidate.shape
        assert existing.dtype == candidate.dtype
    return buffers


def _model_route_direct_chain_mx_buffers(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> dict[str, mx.array]:
    buffers: dict[str, mx.array] = {}
    for name, spec in _model_route_direct_chain_logical_buffers(
        model,
        sequence_length=sequence_length,
    ).items():
        dtype = getattr(mx, str(spec.dtype))
        buffers[name] = mx.zeros(tuple(spec.shape), dtype=dtype)
    return buffers


def _model_route_direct_chain_artifacts(
    model: Any | None = None,
    *,
    sequence_length: int | None = None,
) -> tuple[Any, ...]:
    return tuple(
        lambda *args: None
        for _ in _model_route_direct_chain(
            model,
            sequence_length=sequence_length,
        ).segments
    )


def test_path_c_fusion_runtime_binding_accepts_model_owned_physical_banks(
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    _, _, regions = m04_train_step._local_gb10_path_c_model_regions()
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    assert selected_region is not None
    scheduled = m04_train_step.plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    assert scheduled.schedule_target is not None
    bank_buffers = _model_route_physical_bank_buffers()

    payload = m04_train_step.path_c_fusion_runtime_training_binding_payload(
        region=scheduled.region,
        schedule_target=scheduled.schedule_target,
        bank_buffers=bank_buffers,
        bank_buffer_owner="local_gb10_quarter.path_c_physical_abi_banks",
        fused_artifact=lambda *args: None,
    )

    assert payload["status"] == "ok"
    assert payload["binding_status"] == "ok"
    assert payload["physical_abi_binding_ready"] is True
    assert payload["fused_artifact_bound"] is True
    assert payload["runtime_uses_fused_train_block"] is True
    assert payload["missing_bank_buffers"] == []
    assert payload["provided_bank_buffers"] == list(bank_buffers)
    assert payload["bank_buffer_owner"] == "local_gb10_quarter.path_c_physical_abi_banks"
    assert payload["hidden_packing_performed"] is False


def test_path_c_fusion_runtime_binding_accepts_bank_owner_object(
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    _, _, regions = m04_train_step._local_gb10_path_c_model_regions()
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    assert selected_region is not None
    scheduled = m04_train_step.plan_path_c_fusion_schedule_for_region(
        selected_region,
        include_backward=True,
    )
    assert scheduled.schedule_target is not None

    payload = m04_train_step.path_c_fusion_runtime_training_binding_payload(
        region=scheduled.region,
        schedule_target=scheduled.schedule_target,
        bank_owner=_model_route_physical_bank_owner(),
        fused_artifact=lambda *args: None,
    )

    assert payload["status"] == "ok"
    assert payload["physical_abi_binding_ready"] is True
    assert payload["fused_artifact_bound"] is True
    assert payload["runtime_uses_fused_train_block"] is True
    assert payload["bank_buffer_owner"] == "local_gb10_quarter.path_c_physical_abi_banks"
    assert payload["provided_bank_buffers"] == list(
        _model_route_physical_bank_buffers()
    )


def test_fp8_path_c_training_route_keeps_split_when_direct_chain_is_standalone(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--seq-len",
            "513",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        direct_chain_artifacts=_model_route_direct_chain_artifacts(
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
    )

    direct_chain = route["path_c_fusion"]["direct_chained_fusion"]
    assert direct_chain["runtime_binding"]["status"] == "ok"
    assert direct_chain["runtime_binding"]["runtime_uses_direct_fusion_chain"] is True
    assert direct_chain["standalone_dispatch_available"] is True
    assert direct_chain["training_critical_path"] is False
    assert direct_chain["training_runtime_available"] is False
    assert direct_chain["runtime_binding"]["segment_count"] == 7
    assert [
        segment["execution_phase"]
        for segment in direct_chain["runtime_binding"]["segments"]
    ] == [
        "forward",
        "forward",
        "backward",
        "backward",
        "backward",
        "backward",
        "backward",
    ]
    assert route["status"] == m04_train_step.FP8_PATH_C_SPLIT_TRAINING_STATUS
    assert route["end_to_end_training_status"] == (
        m04_train_step.FP8_PATH_C_SPLIT_TRAINING_STATUS
    )
    assert route["full_end_to_end_training_available"] is False
    assert route["direct_fusion_chain_runtime_available"] is True
    assert route["direct_fusion_chain_standalone_dispatch_available"] is True
    assert route["direct_fusion_chain_training_critical_path"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    assert route["direct_fusion_chain_training_runtime_contract"]["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert route["fused_train_block_runtime_available"] is False
    assert route["fused_train_block_blocker_type"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS
    )
    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["direct_mx_array_artifact_call_status"] == (
        "m04_direct_fusion_chain_standalone_only_not_training_route"
    )


def test_bf16_path_c_policy_route_is_requested_without_fp8_producer(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "bfloat16",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()

    with temporary_env(
        {
            "CPPMEGA_KERNEL_PATH": "path_c",
            "CPPMEGA_KERNEL_PATH__MAMBA3_MIMO": "path_c",
            "CPPMEGA_KERNEL_PATH__M2RNN": "path_c",
            "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
        }
    ):
        route = m04_train_step.fp8_path_c_training_route_payload_for_model(
            config,
            model,
        )

    assert m04_train_step.fp8_path_c_route_requested(config) is False
    assert route["requested"] is True
    assert route["dtype"] == "bfloat16"
    assert route["sparse_mla_fp8_producer"]["requested"] is False
    assert route["status"] == m04_train_step.FP8_PATH_C_SPLIT_TRAINING_STATUS
    assert route["split_end_to_end_training_available"] is True
    assert route["full_end_to_end_training_available"] is False
    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["blocker_type"] is None


def test_path_c_direct_chain_value_and_grad_bridge_plan_reports_loss_and_tree_gaps(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
        auto_install_fused_train_block=True,
    )

    bridge = route["path_c_fusion"]["direct_chained_fusion"][
        "value_and_grad_bridge_plan"
    ]
    assert bridge["status"] == "blocked"
    assert bridge["contract"] == m04_train_step.PATH_C_DIRECT_FUSION_VALUE_AND_GRAD_CONTRACT
    assert bridge["loss_cotangent_bridge_ready"] is False
    assert bridge["model_gradient_tree_ready"] is False
    assert bridge["delegates_to_eager_loss_and_grad"] is False
    assert bridge["required_gradient_buffer_count"] == 39
    assert bridge["covered_parameter_gradient_buffer_count"] == 28
    assert bridge["parameter_gradient_tree_name_count"] == 28
    assert bridge["bridge_only_gradient_buffer_count"] == 11
    assert bridge["required_loss_cotangent_buffers"] == [
        "local_gb10_quarter_brick_11_R_hidden_after_grad",
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_grad",
    ]
    assert "hidden_grad" not in bridge["required_loss_cotangent_buffers"]
    assert "local_gb10_quarter_brick_10_M_hidden_grad" in (
        bridge["bridge_only_gradient_buffers"]
    )
    assert "local_gb10_quarter_brick_10_M_hidden_grad" in (
        bridge["required_runtime_bridge_gradients"]
    )
    assert "local_gb10_quarter_brick_10_M_mamba3_h0_grad" in bridge[
        "required_runtime_bridge_gradients"
    ]
    assert "local_gb10_quarter_brick_10_M_entry_rmsnorm_hidden_grad" in (
        bridge["bridge_only_gradient_buffers"]
    )
    assert "local_gb10_quarter_brick_10_M_entry_rmsnorm_weight_grad" in (
        bridge["covered_parameter_gradient_buffers"]
    )
    assert "local_gb10_quarter_brick_12_A_qkv_projection_attention_q_proj_weight_grad" in (
        bridge["covered_parameter_gradient_buffers"]
    )
    assert "layers.12.block.q_proj.weight_grad" in bridge[
        "parameter_gradient_tree_names"
    ]
    assert "norm.weight_grad" in bridge["parameter_gradient_tree_names"]
    assert "lm_head.weight_grad" in bridge["parameter_gradient_tree_names"]
    assert {blocker["kind"] for blocker in bridge["blockers"]} == {
        "loss_cotangent_bridge_missing",
        "model_gradient_tree_extraction_missing",
        "runtime_bridge_gradient_outputs_required",
    }
    # Direct-chain bridge blockers stand; route action now switches to the
    # fused train-block route because the mixed-mode runtime auto-installs.
    assert route["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_split_training_route",
    }


def test_path_c_direct_chain_diagnostics_do_not_report_unavailable_for_entry_rmsnorm(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    direct_chain = route["path_c_fusion"]["direct_chained_fusion"]
    for key in (
        "model_binding_audit",
        "pre_step_owner_plan",
        "value_and_grad_bridge_plan",
    ):
        status = direct_chain[key]["status"]
        assert not status.endswith("_unavailable"), (key, direct_chain[key])
    assert direct_chain["pre_step_owner_plan"]["status"] == (
        "pre_step_runtime_owner_missing"
    )
    assert direct_chain["value_and_grad_bridge_plan"]["status"] == "blocked"


def test_path_c_model_gradient_tree_from_direct_buffers_maps_parameter_aliases(
    tmp_path: Path,
) -> None:
    del tmp_path
    model = build_local_gb10_quarter_tiny_smoke_model()
    buffers = _model_route_direct_chain_mx_buffers(model)
    chain = _model_route_direct_chain(model)
    bridge = m04_train_step.path_c_direct_fusion_chain_value_and_grad_bridge_plan(
        chain=chain,
        model=model,
    )

    payload = m04_train_step.path_c_model_gradient_tree_extraction_payload(
        model=model,
        logical_buffers=buffers,
        parameter_gradient_names=bridge["parameter_gradient_tree_names"],
    )
    gradient_tree = m04_train_step.path_c_model_gradient_tree_from_direct_buffers(
        model=model,
        logical_buffers=buffers,
        parameter_gradient_names=bridge["parameter_gradient_tree_names"],
    )
    flat = dict(tree_flatten(gradient_tree))

    assert payload["status"] == "ok"
    assert payload["gradient_tree_ready"] is True
    assert payload["mapped_parameter_gradient_count"] == 28
    assert payload["parameter_gradient_alias_count"] == 28
    assert payload["missing_parameter_gradient_names"] == []
    assert payload["missing_logical_gradient_buffers"] == []
    assert "layers.12.block.q_proj.weight_grad" in flat
    logical_name = (
        "local_gb10_quarter_brick_12_A_qkv_projection_attention_q_proj_weight_grad"
    )
    assert flat["layers.12.block.q_proj.weight_grad"] is buffers[logical_name]
    assert flat["norm.weight_grad"] is buffers["final_norm_weight_grad"]
    assert flat["lm_head.weight_grad"] is buffers["lm_head_weight_grad"]


def test_path_c_direct_chain_bridge_plan_accepts_gradient_tree_buffers(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    buffers = _model_route_direct_chain_mx_buffers(
        model,
        sequence_length=sequence_length,
    )

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_logical_buffers=buffers,
        direct_chain_logical_buffer_owner="unit.path_c_direct_buffers",
    )
    bridge = route["path_c_fusion"]["direct_chained_fusion"][
        "value_and_grad_bridge_plan"
    ]

    assert bridge["status"] == "blocked"
    assert bridge["loss_cotangent_bridge_ready"] is False
    assert bridge["model_gradient_tree_ready"] is True
    assert bridge["runtime_bridge_gradient_outputs_ready"] is True
    assert bridge["model_gradient_tree_extraction"]["status"] == "ok"
    assert bridge["model_gradient_tree_extraction"][
        "mapped_parameter_gradient_count"
    ] == 28
    assert {blocker["kind"] for blocker in bridge["blockers"]} == {
        "loss_cotangent_bridge_missing"
    }
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_path_c_direct_chain_runtime_value_and_grad_uses_loss_cotangent_bridge(
    tmp_path: Path,
) -> None:
    del tmp_path
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    buffers = _model_route_direct_chain_mx_buffers(model)
    bridge = _ContractedLossCotangentBridge()
    runtime = m04_train_step.PathCDirectFusionChainTrainingRuntime(
        chain=chain,
        artifacts=_model_route_direct_chain_artifacts(model),
        logical_buffers=buffers,
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=True,
        loss_cotangent_bridge=bridge,
    )
    runtime.bind_training_graph(
        owner="CompiledPretrainingStep",
        uses_direct_chain_runtime=True,
        uses_forward_hook=True,
        uses_backward_or_vjp_hook=True,
    )

    contract = m04_train_step._direct_chain_value_and_grad_contract_payload(runtime)
    assert contract["status"] == "ok"
    assert contract["loss_cotangent_bridge_ready"] is True
    assert contract["model_gradient_tree_ready"] is True

    def forbidden_loss_and_grad(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Path C direct runtime must not delegate to eager")

    (loss, ntokens), grads = runtime.value_and_grad(
        model,
        {},
        forbidden_loss_and_grad,
    )
    flat = dict(tree_flatten(grads))

    assert float(loss.item()) == pytest.approx(1.25)
    assert int(ntokens.item()) == 7
    assert bridge.calls == [
        (
            "loss_cotangent_bridge",
            (
                "local_gb10_quarter_brick_11_R_hidden_after_grad",
                "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_grad",
            ),
        )
    ]
    logical_name = (
        "local_gb10_quarter_brick_12_A_qkv_projection_attention_q_proj_weight_grad"
    )
    assert flat["layers.12.block.q_proj.weight_grad"] is buffers[logical_name]


def test_path_c_direct_chain_value_and_grad_probe_payload_is_post_step_only(
    tmp_path: Path,
) -> None:
    del tmp_path
    model = build_local_gb10_quarter_tiny_smoke_model()
    logical_buffers = _model_route_direct_chain_mx_buffers(model)
    runtime = m04_train_step.PathCDirectFusionChainTrainingRuntime(
        chain=_model_route_direct_chain(model),
        artifacts=_model_route_direct_chain_artifacts(model),
        logical_buffers=logical_buffers,
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=False,
        loss_cotangent_bridge=m04_train_step.PathCResidualSumSuffixLossCotangentBridge(
            chunk_rows=128,
        ),
        model=model,
    )
    seq_len = logical_buffers[
        "local_gb10_quarter_brick_11_R_hidden_after"
    ].shape[1]
    tokens = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)

    payload = m04_train_step.path_c_direct_chain_value_and_grad_probe_payload(
        runtime=runtime,
        model=model,
        batch={"tokens": tokens},
    )

    assert payload["status"] == "ok"
    assert payload["execution_phase"] == "post_step_profile_probe"
    assert payload["training_critical_path"] is False
    assert payload["delegated_to_eager_loss_and_grad"] is False
    assert payload["gradient_count"] > 0
    assert payload["value_and_grad_contract"]["status"] == "incomplete"
    assert payload["value_and_grad_contract"]["returns_full_model_grads"] is True
    assert payload["training_runtime_contract"]["training_runtime_available"] is False


def test_path_c_direct_chain_runtime_value_and_grad_bridges_after_forward_segments(
    tmp_path: Path,
) -> None:
    del tmp_path
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    buffers = _model_route_direct_chain_mx_buffers(model)
    events: list[str] = []

    def _artifact_for_segment(segment: Any) -> Any:
        def _artifact(*_args: Any) -> None:
            events.append(
                f"segment:{segment.index}:{segment.execution_phase}"
            )

        return _artifact

    artifacts = tuple(_artifact_for_segment(segment) for segment in chain.segments)

    class _ForwardBoundaryBridge(_ContractedLossCotangentBridge):
        def __call__(
            self,
            *,
            model: nn.Module,
            batch: Mapping[str, mx.array],
            logical_buffers: Mapping[str, mx.array],
            required_loss_cotangent_buffers: Sequence[str],
            chain: Any,
        ) -> dict[str, Any]:
            assert events == ["segment:0:forward", "segment:1:forward"]
            events.append("loss_cotangent_bridge")
            return super().__call__(
                model=model,
                batch=batch,
                logical_buffers=logical_buffers,
                required_loss_cotangent_buffers=required_loss_cotangent_buffers,
                chain=chain,
            )

    runtime = m04_train_step.PathCDirectFusionChainTrainingRuntime(
        chain=chain,
        artifacts=artifacts,
        logical_buffers=buffers,
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=True,
        loss_cotangent_bridge=_ForwardBoundaryBridge(),
    )
    runtime.bind_training_graph(
        owner="CompiledPretrainingStep",
        uses_direct_chain_runtime=True,
        uses_forward_hook=True,
        uses_backward_or_vjp_hook=True,
    )

    def forbidden_loss_and_grad(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Path C direct runtime must not delegate to eager")

    runtime.value_and_grad(model, {}, forbidden_loss_and_grad)

    assert events == [
        "segment:0:forward",
        "segment:1:forward",
        "loss_cotangent_bridge",
        "segment:2:backward",
        "segment:3:backward",
        "segment:4:backward",
        "segment:5:backward",
        "segment:6:backward",
    ]


def test_path_c_local_gb10_suffix_bridge_returns_real_boundary_cotangents() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    buffers = _model_route_direct_chain_mx_buffers(model)
    residual_name = "local_gb10_quarter_brick_11_R_hidden_after"
    attention_name = "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out"
    residual_grad_name = f"{residual_name}_grad"
    attention_grad_name = f"{attention_name}_grad"
    residual = mx.ones_like(buffers[residual_name]) * mx.array(0.125)
    attention = mx.ones_like(buffers[attention_name]) * mx.array(-0.0625)
    buffers[residual_name] = residual
    buffers[attention_name] = attention
    tokens = mx.arange(residual.shape[1] + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)
    bridge = m04_train_step.PathCResidualSumSuffixLossCotangentBridge(chunk_rows=128)

    contract = bridge.loss_cotangent_bridge_contract()
    assert contract["contract"] == m04_train_step.PATH_C_LOSS_COTANGENT_BRIDGE_CONTRACT
    assert contract["returns_required_loss_cotangents"] is True
    assert contract["delegates_to_eager_loss_and_grad"] is False
    assert contract["hidden_packing_performed"] is False

    payload = bridge(
        model=model,
        batch={"tokens": tokens},
        logical_buffers=buffers,
        required_loss_cotangent_buffers=(residual_grad_name, attention_grad_name),
        chain=chain,
    )

    def expected_suffix_loss(
        residual_hidden: mx.array,
        attention_out: mx.array,
        norm_weight: mx.array,
        head_weight: mx.array,
    ) -> tuple[mx.array, mx.array]:
        targets = tokens[:, 1:]
        hidden = residual_hidden + attention_out
        inv_rms = mx.rsqrt(
            mx.mean(hidden * hidden, axis=-1, keepdims=True)
            + mx.array(model.norm.eps, dtype=hidden.dtype)
        )
        normed = hidden * inv_rms * norm_weight
        logits = normed @ head_weight.T
        token_losses = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            targets,
            reduction="none",
        )
        mask = mx.ones(targets.shape, dtype=mx.float32)
        ntokens = mask.sum()
        denom = mx.maximum(ntokens, mx.array(1.0, dtype=mx.float32))
        return (token_losses * mask).astype(mx.float32).sum() / denom, ntokens

    (expected_loss, expected_ntokens), expected_grads = mx.value_and_grad(
        expected_suffix_loss,
        argnums=(0, 1, 2, 3),
    )(residual, attention, model.norm.weight, model.lm_head.weight)
    mx.eval(
        payload["loss"],
        payload["ntokens"],
        payload["cotangents"][residual_grad_name],
        payload["cotangents"][attention_grad_name],
        payload["parameter_grads"]["norm.weight_grad"],
        payload["parameter_grads"]["lm_head.weight_grad"],
        expected_loss,
        expected_ntokens,
        expected_grads[0],
        expected_grads[1],
        expected_grads[2],
        expected_grads[3],
    )

    assert float(payload["loss"].item()) == pytest.approx(
        float(expected_loss.item()),
        rel=1e-5,
        abs=1e-6,
    )
    assert int(payload["ntokens"].item()) == int(expected_ntokens.item())
    assert set(payload["cotangents"]) == {residual_grad_name, attention_grad_name}
    residual_delta = mx.max(
        mx.abs(payload["cotangents"][residual_grad_name] - expected_grads[0])
    )
    attention_delta = mx.max(
        mx.abs(payload["cotangents"][attention_grad_name] - expected_grads[1])
    )
    norm_delta = mx.max(
        mx.abs(payload["parameter_grads"]["norm.weight_grad"] - expected_grads[2])
    )
    head_delta = mx.max(
        mx.abs(payload["parameter_grads"]["lm_head.weight_grad"] - expected_grads[3])
    )
    mx.eval(residual_delta, attention_delta, norm_delta, head_delta)
    assert float(residual_delta.item()) < 1e-5
    assert float(attention_delta.item()) < 1e-5
    assert float(norm_delta.item()) < 1e-5
    assert float(head_delta.item()) < 1e-5


def test_path_c_prefix_hidden_and_vjp_cover_pre_region_parameters() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    start_layer = m04_train_step._path_c_direct_chain_start_layer_index(
        model,
        chain,
    )
    assert start_layer == 10

    seq_len = 512
    tokens = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)
    capture = PathCActivationBufferCapture(
        aliases={
            "local_gb10_quarter_brick_10_M_hidden": "hidden",
            "local_gb10_quarter_brick_10_M_residual_norm_hidden": "normed_hidden",
        },
        owner_name="unit.path_c_prefix_boundary_capture",
    )
    model.attach_path_c_activation_probe(capture)
    captured_suffix_hidden = model.decoder_hidden_states(tokens[:, :-1])
    prefix_hidden = m04_train_step.path_c_model_prefix_hidden_states(
        model,
        {"tokens": tokens},
        end_layer_index=start_layer,
    )
    boundary_delta = mx.max(mx.abs(prefix_hidden - capture.buffers["hidden"]))
    mx.eval(captured_suffix_hidden, prefix_hidden, boundary_delta)

    assert float(boundary_delta.item()) < 1e-6

    hidden_grad = mx.ones_like(prefix_hidden) * mx.array(0.001, dtype=mx.float32)
    normed_hidden_grad = (
        mx.ones_like(capture.buffers["normed_hidden"]) * mx.array(0.002, dtype=mx.float32)
    )
    prefix_grads = m04_train_step.path_c_prefix_gradient_tree_from_hidden_cotangent(
        model=model,
        batch={"tokens": tokens},
        hidden_cotangent=hidden_grad,
        normed_hidden_cotangent=normed_hidden_grad,
        chain=chain,
    )
    flat = dict(tree_flatten(prefix_grads))
    mx.eval(prefix_grads)

    assert "token_embedding.weight" in flat
    assert "position_embedding.weight" in flat
    assert "layers.8.block.out_proj.weight" in flat
    assert "layers.10.norm.weight" in flat
    assert "layers.10.block.in_proj.weight" not in flat
    assert "norm.weight" not in flat
    assert "lm_head.weight" not in flat


def test_fp8_path_c_training_route_rejects_legacy_direct_chain_bool(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    model.path_c_direct_fusion_chain_training_critical_path = True

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
    )

    contract = route["direct_fusion_chain_training_runtime_contract"]
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert contract["runtime_installed"] is False
    assert route["direct_fusion_chain_training_critical_path"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_fp8_path_c_training_route_rejects_partial_direct_chain_runtime(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    runtime = SimpleNamespace(
        contract=m04_train_step.PATH_C_DIRECT_FUSION_TRAINING_RUNTIME_CONTRACT,
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=True,
        forward=lambda *args: None,
        no_hidden_allocation_policy=True,
        hidden_packing_performed=False,
    )

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
        direct_chain_training_runtime=runtime,
    )

    contract = route["direct_fusion_chain_training_runtime_contract"]
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS
    )
    assert contract["forward_callable"] is True
    assert contract["backward_callable"] is False
    assert contract["vjp_callable"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    assert route["fused_train_block_runtime_available"] is False
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_fp8_path_c_training_route_rejects_direct_chain_runtime_without_graph_binding(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    runtime = _ContractedValueAndGradPathCDirectFusionChainTrainingRuntime(
        chain=_model_route_direct_chain(model, sequence_length=sequence_length),
        artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        logical_buffers=_model_route_direct_chain_mx_buffers(
            model,
            sequence_length=sequence_length,
        ),
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=True,
    )
    runtime_payload = runtime.forward()
    assert runtime_payload["status"] == "ok"
    assert runtime_payload["runtime_uses_direct_fusion_chain"] is True

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
        direct_chain_training_runtime=runtime,
    )

    direct_chain = route["path_c_fusion"]["direct_chained_fusion"]
    contract = route["direct_fusion_chain_training_runtime_contract"]
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_NOT_CRITICAL_STATUS
    )
    assert contract["contract_matches"] is True
    assert (
        contract["runtime_class"]
        == "_ContractedValueAndGradPathCDirectFusionChainTrainingRuntime"
    )
    assert contract["value_and_grad_contract"]["status"] == "ok"
    assert contract["runtime_owner"] == "unit.path_c_direct_training_runtime"
    assert contract["training_critical_path_declared"] is True
    assert contract["value_and_grad_callable"] is True
    assert contract["value_and_grad_contract_ok"] is True
    assert contract["training_graph_bound"] is False
    assert contract["training_graph_binding"]["status"] == "missing"
    assert contract["training_critical_path_verified"] is False
    assert direct_chain["training_runtime_contract"]["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_NOT_CRITICAL_STATUS
    )
    assert direct_chain["training_critical_path"] is False
    assert direct_chain["training_runtime_available"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    assert route["fused_train_block_runtime_available"] is False
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_fp8_path_c_training_route_rejects_graph_bound_runtime_without_loss_bridge(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    runtime = m04_train_step.PathCDirectFusionChainTrainingRuntime(
        chain=_model_route_direct_chain(model, sequence_length=sequence_length),
        artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        logical_buffers=_model_route_direct_chain_mx_buffers(
            model,
            sequence_length=sequence_length,
        ),
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=True,
    )
    runtime.bind_training_graph(
        owner="CompiledPretrainingStep",
        uses_direct_chain_runtime=True,
        uses_forward_hook=True,
        uses_backward_or_vjp_hook=True,
    )

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
        direct_chain_training_runtime=runtime,
    )

    contract = route["direct_fusion_chain_training_runtime_contract"]
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS
    )
    assert contract["forward_callable"] is True
    assert contract["backward_callable"] is True
    assert contract["value_and_grad_callable"] is True
    assert contract["value_and_grad_contract"]["status"] == "incomplete"
    assert (
        contract["value_and_grad_contract"]["loss_cotangent_bridge_contract"][
            "status"
        ]
        == "missing"
    )
    assert contract["training_graph_bound"] is True
    assert contract["training_critical_path_verified"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    assert route["fused_train_block_runtime_available"] is False
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_fp8_path_c_training_route_rejects_graph_bound_value_and_grad_without_direct_contract(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    runtime = _ValueAndGradPathCDirectFusionChainTrainingRuntime(
        chain=_model_route_direct_chain(model, sequence_length=sequence_length),
        artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        logical_buffers=_model_route_direct_chain_mx_buffers(
            model,
            sequence_length=sequence_length,
        ),
        owner_name="unit.path_c_direct_training_runtime",
        training_critical_path=True,
    )
    runtime.bind_training_graph(
        owner="CompiledPretrainingStep",
        uses_direct_chain_runtime=True,
        uses_forward_hook=True,
        uses_backward_or_vjp_hook=True,
    )

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=_model_route_direct_chain_artifacts(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
        direct_chain_training_runtime=runtime,
    )

    contract = route["direct_fusion_chain_training_runtime_contract"]
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS
    )
    assert contract["value_and_grad_callable"] is True
    assert contract["value_and_grad_contract"]["status"] == "incomplete"
    assert (
        contract["value_and_grad_contract"]["loss_cotangent_bridge_ready"]
        is False
    )
    assert contract["value_and_grad_contract"]["model_gradient_tree_ready"] is False
    assert "contracted loss-to-region cotangent bridge" in str(
        contract["value_and_grad_contract"]["reason"]
    )
    assert (
        contract["value_and_grad_contract"]["loss_cotangent_bridge_contract"][
            "status"
        ]
        == "missing"
    )
    assert contract["value_and_grad_contract_ok"] is False
    assert contract["training_graph_bound"] is True
    assert contract["training_critical_path_verified"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    assert route["fused_train_block_runtime_available"] is False
    assert route["full_end_to_end_training_available"] is False
    assert route["status"] == m04_train_step.FP8_PATH_C_SPLIT_TRAINING_STATUS
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_path_c_direct_chain_native_artifacts_bind_with_direct_buffers(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)

    artifacts = m04_train_step.compile_path_c_direct_fusion_chain_artifacts(chain)

    assert len(artifacts) == len(chain.segments)
    assert all(callable(artifact) for artifact in artifacts)
    assert {type(artifact).__name__ for artifact in artifacts} == {"JITKernel"}

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=artifacts,
        direct_chain_logical_buffers=_model_route_direct_chain_logical_buffers(
            model,
            sequence_length=sequence_length,
        ),
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
    )
    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]

    assert runtime_binding["status"] == "ok"
    assert runtime_binding["direct_chain_artifacts_bound"] is True
    assert runtime_binding["runtime_uses_direct_fusion_chain"] is True
    assert runtime_binding["missing_artifact_segments"] == []
    assert route["path_c_fusion"]["direct_chained_fusion"][
        "standalone_dispatch_available"
    ] is True
    assert route["path_c_fusion"]["direct_chained_fusion"][
        "training_runtime_available"
    ] is False
    assert route["path_c_fusion"]["direct_chained_fusion"][
        "training_runtime_contract"
    ]["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_path_c_direct_chain_runtime_executor_runs_native_segments(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)
    buffers = _model_route_direct_chain_mx_buffers(
        model,
        sequence_length=sequence_length,
    )
    artifacts = m04_train_step.compile_path_c_direct_fusion_chain_artifacts(chain)
    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        model=model,
        direct_chain_artifacts=artifacts,
        direct_chain_logical_buffers=buffers,
        direct_chain_logical_buffer_owner="local_gb10_quarter.path_c_direct_buffers",
    )

    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["direct_mx_array_artifact_call_status"] == (
        "m04_direct_fusion_chain_standalone_only_not_training_route"
    )

    payload = m04_train_step.run_path_c_direct_fusion_chain_route(
        chain=chain,
        logical_buffers=buffers,
        artifacts=artifacts,
    )

    assert payload["status"] == "ok"
    assert payload["runtime_uses_direct_fusion_chain"] is True
    assert payload["segment_count"] == 7
    assert [segment["execution_phase"] for segment in payload["segments"]] == [
        "forward",
        "forward",
        "backward",
        "backward",
        "backward",
        "backward",
        "backward",
    ]
    assert [segment["status"] for segment in payload["segments"]] == [
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
    ]
    # The mamba3 mimo BACKWARD segment is TIME-CHUNKED (launcher_chunks): its
    # kernel additionally declares the path_c_row_chunk_index /
    # path_c_row_subchunk_index / path_c_backward_stage_index scalar params so the
    # runtime can split the reverse-time scan into watchdog-safe per-launch
    # command buffers. That segment therefore carries 3 extra scalar args beyond
    # the backward gate; every other segment keeps grid_chunks dispatch.
    time_chunked_segment_indices = {
        segment["index"]
        for segment in payload["segments"]
        if segment.get("time_chunk_launch_count")
    }
    assert time_chunked_segment_indices == {5}, time_chunked_segment_indices
    mamba3_bwd_launch_count = next(
        segment["time_chunk_launch_count"]
        for segment in payload["segments"]
        if segment["index"] == 5
    )
    assert mamba3_bwd_launch_count > 1
    assert [segment["kernel_arg_count"] for segment in payload["segments"]] == [
        len(rb_segment["required_logical_buffers"])
        # backward segments additionally pass the path_c_run_backward gate scalar
        + (1 if payload_segment["execution_phase"] == "backward" else 0)
        # the time-chunked mamba3 backward also binds the launcher-chunk index
        # scalars (path_c_row_chunk_index + path_c_row_subchunk_index +
        # path_c_backward_stage_index) so the runtime can select each launch.
        + (3 if payload_segment["index"] in time_chunked_segment_indices else 0)
        for payload_segment, rb_segment in zip(
            payload["segments"],
            route["path_c_fusion"]["direct_chained_fusion"]["runtime_binding"][
                "segments"
            ],
        )
    ]


def test_path_c_direct_chain_runtime_installer_keeps_probe_off_critical_path(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)
    logical_buffers = _model_route_direct_chain_mx_buffers(
        model,
        sequence_length=sequence_length,
    )
    logical_owner = PathCLogicalBufferOwner(
        owner_name="unit.path_c_direct_fusion_chain_buffers",
        buffers=logical_buffers,
    )

    install_payload = (
        m04_train_step.install_path_c_direct_chain_training_runtime_for_model(
            model=model,
            chain=chain,
            artifacts=_model_route_direct_chain_artifacts(
                model,
                sequence_length=sequence_length,
            ),
            logical_owner=logical_owner,
            owner_name="unit.path_c_direct_training_runtime",
            training_critical_path=False,
            run_probe=True,
        )
    )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )
    contract = route["direct_fusion_chain_training_runtime_contract"]

    assert install_payload["status"] == "ok"
    assert install_payload["runtime_uses_direct_fusion_chain"] is True
    assert install_payload["execution"]["status"] == "ok"
    assert install_payload["training_critical_path"] is False
    assert contract["runtime_class"] == "PathCDirectFusionChainTrainingRuntime"
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS
    )
    assert contract["value_and_grad_callable"] is True
    assert contract["value_and_grad_contract"]["status"] == "incomplete"
    assert "contracted loss-to-region cotangent bridge" in str(
        contract["value_and_grad_contract"]["reason"]
    )
    assert (
        contract["value_and_grad_contract"]["loss_cotangent_bridge_contract"][
            "status"
        ]
        == "missing"
    )
    assert route["direct_fusion_chain_runtime_available"] is True
    assert route["direct_fusion_chain_training_runtime_available"] is False
    # Either path is a valid Path C training action; fused-train-block
    # wins precedence when both are bound (HybridTinyLM's mixed-mode
    # runtime is now installed automatically).
    assert route["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_direct_fusion_chain_route",
        "run_path_c_split_training_route",
    }


def test_path_c_direct_chain_runtime_installer_blocks_incomplete_critical_path(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)
    logical_buffers = _model_route_direct_chain_mx_buffers(
        model,
        sequence_length=sequence_length,
    )
    logical_owner = PathCLogicalBufferOwner(
        owner_name="unit.path_c_direct_fusion_chain_buffers",
        buffers=logical_buffers,
    )

    install_payload = (
        m04_train_step.install_path_c_direct_chain_training_runtime_for_model(
            model=model,
            chain=chain,
            artifacts=_model_route_direct_chain_artifacts(
                model,
                sequence_length=sequence_length,
            ),
            logical_owner=logical_owner,
            owner_name="unit.path_c_direct_training_runtime",
            training_critical_path=True,
            run_probe=False,
        )
    )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert install_payload["status"] == "blocked"
    assert install_payload["training_critical_path"] is False
    assert install_payload["reason"] == "direct-chain value_and_grad runtime incomplete"
    assert install_payload["value_and_grad_contract"]["status"] == "incomplete"
    assert (
        install_payload["value_and_grad_contract"]["loss_cotangent_bridge_ready"]
        is False
    )
    assert (
        install_payload["value_and_grad_contract"]["model_gradient_tree_ready"]
        is False
    )
    assert not hasattr(model, "path_c_direct_fusion_chain_training_runtime")
    # Either path is a valid Path C training action; fused-train-block
    # wins precedence when both are bound (HybridTinyLM's mixed-mode
    # runtime is now installed automatically).
    assert route["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_direct_fusion_chain_route",
        "run_path_c_split_training_route",
    }
    assert route["direct_fusion_chain_training_runtime_available"] is False


def test_path_c_direct_chain_runtime_installer_accepts_suffix_bridge_off_critical_path(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)
    logical_buffers = _model_route_direct_chain_mx_buffers(
        model,
        sequence_length=sequence_length,
    )
    logical_owner = PathCLogicalBufferOwner(
        owner_name="unit.path_c_direct_fusion_chain_buffers",
        buffers=logical_buffers,
    )

    install_payload = (
        m04_train_step.install_path_c_direct_chain_training_runtime_for_model(
            model=model,
            chain=chain,
            artifacts=_model_route_direct_chain_artifacts(
                model,
                sequence_length=sequence_length,
            ),
            logical_owner=logical_owner,
            owner_name="unit.path_c_direct_training_runtime",
            training_critical_path=False,
            run_probe=False,
            loss_cotangent_bridge=m04_train_step.PathCResidualSumSuffixLossCotangentBridge(
                chunk_rows=128,
            ),
        )
    )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )
    contract = route["direct_fusion_chain_training_runtime_contract"]

    assert install_payload["status"] == "ok"
    assert install_payload["training_critical_path"] is False
    assert install_payload["value_and_grad_contract"]["loss_cotangent_bridge_ready"]
    assert install_payload["value_and_grad_contract"]["returns_model_grads"] is True
    assert (
        install_payload["value_and_grad_contract"]["returns_full_model_grads"]
        is True
    )
    assert (
        install_payload["value_and_grad_contract"][
            "full_model_gradient_coverage"
        ]["missing_parameter_names"]
        == []
    )
    assert install_payload["value_and_grad_contract"][
        "full_model_gradient_coverage"
    ]["inactive_zero_gradient_parameter_names"] == [
        "layers.12.block.k_proj.weight",
        "layers.12.block.v_proj.weight",
    ]
    assert contract["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_TRAINING_RUNTIME_INCOMPLETE_STATUS
    )
    assert contract["training_runtime_available"] is False
    runtime = model.path_c_direct_fusion_chain_training_runtime
    seq_len = logical_buffers[
        "local_gb10_quarter_brick_11_R_hidden_after"
    ].shape[1]
    tokens = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)

    def forbidden_loss_and_grad(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Path C production bridge must not delegate to eager")

    (loss, ntokens), grads = runtime.value_and_grad(
        model,
        {"tokens": tokens},
        forbidden_loss_and_grad,
    )
    flat = dict(tree_flatten(grads))

    mx.eval(loss, ntokens)
    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == seq_len
    assert "layers.10.norm.weight" in flat
    assert "layers.12.block.q_proj.weight" in flat
    assert "layers.12.block.k_proj.weight" in flat
    assert "layers.12.block.v_proj.weight" in flat
    assert "norm.weight" in flat
    assert "lm_head.weight" in flat
    assert contract["training_graph_bound"] is False
    assert contract["value_and_grad_contract_ok"] is False
    assert route["direct_fusion_chain_training_runtime_available"] is False
    # The fused-train-block runtime is explicitly opt-in via
    # --use-path-c-fused-train-block-runtime (verified end-to-end separately). The
    # route does NOT auto-install it by default here, so without the flag full
    # end-to-end fused training is not advertised and the route uses the split
    # path (default auto-install would change fp8_path_c routing and is a separate
    # opt-in that must not regress paths B/D/E).
    assert route["full_end_to_end_training_available"] is False
    assert route["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_split_training_route",
    }


def test_path_c_direct_chain_runtime_blocks_incomplete_production_bridge(
    tmp_path: Path,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model, sequence_length=sequence_length)
    logical_owner = PathCLogicalBufferOwner(
        owner_name="unit.path_c_direct_fusion_chain_buffers",
        buffers=_model_route_direct_chain_mx_buffers(
            model,
            sequence_length=sequence_length,
        ),
    )

    install_payload = (
        m04_train_step.install_path_c_direct_chain_training_runtime_for_model(
            model=model,
            chain=chain,
            artifacts=_model_route_direct_chain_artifacts(
                model,
                sequence_length=sequence_length,
            ),
            logical_owner=logical_owner,
            owner_name="unit.path_c_direct_training_runtime",
            training_critical_path=True,
            run_probe=False,
            loss_cotangent_bridge=m04_train_step.PathCResidualSumSuffixLossCotangentBridge(
                chunk_rows=128,
            ),
        )
    )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert install_payload["status"] == "ok"
    assert install_payload["training_critical_path"] is True
    assert install_payload["value_and_grad_contract"]["status"] == "ok"
    assert install_payload["value_and_grad_contract"]["returns_model_grads"] is True
    assert (
        install_payload["value_and_grad_contract"]["returns_full_model_grads"]
        is True
    )
    coverage = install_payload["value_and_grad_contract"][
        "full_model_gradient_coverage"
    ]
    assert coverage["missing_parameter_names"] == []
    assert coverage["inactive_zero_gradient_parameter_names"] == [
        "layers.12.block.k_proj.weight",
        "layers.12.block.v_proj.weight",
    ]
    assert hasattr(model, "path_c_direct_fusion_chain_training_runtime")
    assert route["direct_fusion_chain_training_runtime_available"] is True
    # Either path is a valid Path C training action; fused-train-block
    # wins precedence when both are bound (HybridTinyLM's mixed-mode
    # runtime is now installed automatically).
    assert route["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_direct_fusion_chain_route",
        "run_path_c_split_training_route",
    }


def test_fp8_path_c_training_route_keeps_split_until_fused_artifact_is_bound(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        bank_buffers=_model_route_physical_bank_buffers(),
        bank_buffer_owner="local_gb10_quarter.path_c_physical_abi_banks",
    )

    binding = route["path_c_fusion"]["runtime_training_binding"]
    assert binding["physical_abi_binding_ready"] is True
    assert binding["fused_artifact_bound"] is False
    assert binding["runtime_uses_fused_train_block"] is False
    direct_chain = route["path_c_fusion"]["direct_chained_fusion"]
    assert direct_chain["status"] == "ready"
    assert direct_chain["covers_full_region"] is True
    audit = direct_chain["model_binding_audit"]
    assert audit["status"] == "runtime_activation_owner_missing"
    assert audit["requires_runtime_activation_owner"] is True
    assert audit["runtime_activation_or_grad_count"] > 0
    assert any(
        name == "hidden" or name.endswith("_hidden")
        for name in audit["runtime_activation_or_grad_examples"]
    )
    assert direct_chain["runtime_binding"]["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_LOGICAL_BUFFERS_MISSING_STATUS
    )
    assert (
        direct_chain["runtime_binding"]["runtime_uses_direct_fusion_chain"]
        is False
    )
    assert route["fused_train_block_runtime_available"] is False
    assert route["fused_train_block_blocker_type"] == (
        "fused_train_block_artifact_missing"
    )
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_fp8_path_c_training_route_keeps_split_until_fused_training_runtime_is_bound(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        bank_owner=_model_route_physical_bank_owner(),
        fused_artifact=lambda *args: None,
    )

    binding = route["path_c_fusion"]["runtime_training_binding"]
    assert binding["runtime_uses_fused_train_block"] is True
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["single_fused_train_block_runtime_available"] is False
    assert route["fused_train_block_training_runtime_available"] is False
    assert route["fused_train_block_training_runtime_contract"]["status"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert route["fused_train_block_runtime_available"] is False
    assert route["fused_train_block_blocker_type"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert route["selected_action"] == "run_path_c_split_training_route"


def test_fp8_path_c_training_route_reports_model_owned_physical_bank_plan(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    binding = route["path_c_fusion"]["runtime_training_binding"]
    assert binding["status"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_BANKS_MISSING_STATUS
    )
    assert binding["runtime_uses_fused_train_block"] is False
    assert binding["physical_abi_binding_ready"] is False
    assert binding["fused_artifact_bound"] is False
    assert binding["missing_bank_buffers"] == binding["required_bank_buffers"]
    assert binding["provided_bank_buffers"] == []
    assert not hasattr(model, "path_c_physical_abi_bank_owner")
    bank_plan = binding["model_owned_physical_abi_bank_plan"]
    assert bank_plan["status"] == "model_owned_physical_abi_banks_required"
    assert bank_plan["owner_attribute"] == "path_c_physical_abi_bank_owner"
    assert bank_plan["owner_name"] == "local_gb10_quarter.path_c_physical_abi_banks"
    assert bank_plan["allocation_required_before_binding"] is True
    assert bank_plan["hidden_packing_performed"] is False
    assert bank_plan["no_hidden_allocation_policy"] is True
    assert bank_plan["required_bank_buffers"] == binding["required_bank_buffers"]
    assert [spec["name"] for spec in bank_plan["bank_specs"]] == (
        binding["required_bank_buffers"]
    )
    assert {spec["dtype"] for spec in bank_plan["bank_specs"]} == {
        "float32",
        "uint8",
        "int32",
    }
    assert bank_plan["total_nbytes"] == sum(
        spec["nbytes"] for spec in bank_plan["bank_specs"]
    )
    assert bank_plan["total_nbytes"] > 0
    assert all(
        spec["logical_buffer_count"] == len(spec["logical_buffers"])
        for spec in bank_plan["bank_specs"]
    )


def test_path_c_internal_scratch_abi_specs_coalesces_physical_bank_shape() -> None:
    prim_func = SimpleNamespace(
        _cppmega_path_c_spilled_shared_scratch_shapes={
            "q_fp8_grad": {
                "dtype": "float32",
                "param_name": "path_c_float32_scratch_bank",
                "shape": (1024,),
                "coalesced_scratch_bank": True,
                "bank": "path_c_float32_scratch_bank",
                "offset": 0,
            },
            "kv_fp8_grad": {
                "dtype": "float32",
                "param_name": "path_c_float32_scratch_bank",
                "shape": (1024,),
                "coalesced_scratch_bank": True,
                "bank": "path_c_float32_scratch_bank",
                "offset": 1024,
            },
            "q_scale_grad": {
                "dtype": "float32",
                "param_name": "path_c_float32_scratch_bank",
                "shape": (256,),
                "coalesced_scratch_bank": True,
                "bank": "path_c_float32_scratch_bank",
                "offset": 2048,
            },
            "local_tile": {
                "dtype": "int32",
                "param_name": "standalone_int32_scratch",
                "shape": (4, 8),
            },
        }
    )

    specs = m04_train_step._path_c_internal_scratch_abi_specs(prim_func)

    assert specs["path_c_float32_scratch_bank"] == {
        "shape": (2304,),
        "dtype": "float32",
    }
    assert specs["standalone_int32_scratch"] == {
        "shape": (4, 8),
        "dtype": "int32",
    }
    assert "q_fp8_grad" not in specs
    assert "kv_fp8_grad" not in specs


def test_path_c_direct_chain_scratch_bank_merge_uses_largest_segment_shape() -> None:
    specs: dict[str, dict[str, Any]] = {}

    m04_train_step._path_c_merge_direct_chain_buffer_spec(
        specs,
        name="path_c_float32_scratch_bank",
        shape=(2032,),
        dtype="float32",
        category="runtime_activation_or_grad",
        segment_index=0,
        source="scratch",
    )
    m04_train_step._path_c_merge_direct_chain_buffer_spec(
        specs,
        name="path_c_float32_scratch_bank",
        shape=(7112,),
        dtype="float32",
        category="runtime_activation_or_grad",
        segment_index=2,
        source="scratch",
    )

    assert specs["path_c_float32_scratch_bank"]["shape"] == (7112,)
    assert specs["path_c_float32_scratch_bank"]["segments"] == [0, 2]


def test_path_c_internal_scratch_validator_accepts_larger_coalesced_bank() -> None:
    binding = m04_train_step._path_c_validate_internal_scratch_abi_buffers(
        {
            "path_c_float32_scratch_bank": {
                "shape": (8192,),
                "dtype": "float32",
            },
        },
        {
            "path_c_float32_scratch_bank": mx.zeros((28672,), dtype=mx.float32),
        },
    )

    assert binding["status"] == "ok"
    assert binding["shape_mismatch_internal_scratch_buffers"] == []


def test_path_c_fused_train_block_installer_resizes_auto_owner_to_selected_artifact(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    physical_abi_map = {
        "loss": {
            "bank": "path_c_bfloat16_abi_bank",
            "dtype": "bfloat16",
            "offset": 0,
            "shape": (2,),
            "logical_shape": (2,),
            "size": 2,
        }
    }
    physical_abi_shapes = {"path_c_bfloat16_abi_bank": (2,)}

    class _Artifact:
        def __call__(self, *args: Any) -> None:
            del args

    _Artifact.physical_abi_map = physical_abi_map
    _Artifact.physical_abi_shapes = physical_abi_shapes
    _Artifact.kernel_buffer_shapes = {
        "path_c_bfloat16_abi_bank": (2,),
        "path_c_float32_scratch_bank": (8,),
    }

    stale_owner = make_physical_abi_bank_owner(
        "unit.path_c_physical_abi_banks",
        physical_abi_map,
        physical_abi_shapes,
        {
            "path_c_bfloat16_abi_bank": mx.zeros((2,), dtype=mx.bfloat16),
            "path_c_float32_scratch_bank": mx.zeros((4,), dtype=mx.float32),
        },
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    model.path_c_physical_abi_bank_owner = stale_owner

    install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
        model=model,
        fused_artifact=_Artifact(),
        training_runtime=_ReadyFusedTrainBlockTrainingRuntime(),
        sequence_length=m04_train_step.path_c_training_sequence_length(config),
    )

    assert install["status"] == "ok"
    assert install["artifact_kernel_buffer_binding"]["status"] == "ok"
    assert model.path_c_physical_abi_bank_owner is not stale_owner
    owner_buffers = model.path_c_physical_abi_bank_owner.buffers
    assert tuple(owner_buffers["path_c_float32_scratch_bank"].shape) == (8,)
    assert owner_buffers["path_c_bfloat16_abi_bank"].dtype == mx.bfloat16


def test_path_c_fused_train_block_installer_rejects_explicit_undersized_owner(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    physical_abi_map = {
        "loss": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "shape": (2,),
            "logical_shape": (2,),
            "size": 2,
        }
    }
    physical_abi_shapes = {"path_c_float32_abi_bank": (2,)}

    class _Artifact:
        def __call__(self, *args: Any) -> None:
            del args

    _Artifact.physical_abi_map = physical_abi_map
    _Artifact.physical_abi_shapes = physical_abi_shapes
    _Artifact.kernel_buffer_shapes = {
        "path_c_float32_abi_bank": (2,),
        "path_c_float32_scratch_bank": (8,),
    }

    stale_owner = make_physical_abi_bank_owner(
        "unit.path_c_physical_abi_banks",
        physical_abi_map,
        physical_abi_shapes,
        {
            "path_c_float32_abi_bank": mx.zeros((2,), dtype=mx.float32),
            "path_c_float32_scratch_bank": mx.zeros((4,), dtype=mx.float32),
        },
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()

    install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
        model=model,
        bank_owner=stale_owner,
        fused_artifact=_Artifact(),
        training_runtime=_ReadyFusedTrainBlockTrainingRuntime(),
        sequence_length=m04_train_step.path_c_training_sequence_length(config),
    )

    assert install["status"] == "blocked"
    assert install["artifact_kernel_buffer_binding"]["status"] == "failed"
    assert install["artifact_kernel_buffer_binding"]["undersized_kernel_buffers"] == [
        {
            "name": "path_c_float32_scratch_bank",
            "expected_shape": [8],
            "expected_size": 8,
            "actual_shape": [4],
            "actual_size": 4,
        }
    ]


def test_path_c_fusion_payload_keeps_compile_blocker_without_matching_receipt(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env

    payload = m04_train_step.path_c_fusion_payload(
        model=build_local_gb10_quarter_tiny_smoke_model(),
        compile_receipt_path=tmp_path / "missing_receipt.json",
    )

    assert payload["production_compile_receipt"]["status"] == "missing"
    compile_blockers = [
        blocker
        for blocker in payload["schedule_blockers"]
        if blocker["kind"] == "production_schedule_not_compile_verified"
    ]
    assert len(compile_blockers) == 1
    assert compile_blockers[0]["compile_receipt_status"] == "missing"


def test_path_c_fusion_payload_rejects_tiny_smoke_compile_receipt(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    receipt_payload = json.loads(
        PRODUCTION_FUSION_COMPILE_RECEIPT.read_text(encoding="utf-8")
    )
    receipt_payload["runtime_execution_contract"] = {
        **receipt_payload["runtime_execution_contract"],
        "status": "compile_only_not_runtime_ready",
        "runtime_route_uses_fused_region": False,
        "physical_abi_runtime_binding_status": "not_bound",
        "physical_abi_missing_bank_buffers": [
            "path_c_float32_abi_bank",
            "path_c_uint8_abi_bank",
            "path_c_int32_abi_bank",
        ],
    }
    receipt_payload["runtime_smoke"] = {
        "status": "ok",
        "mode": "tiny_mra",
        "actually_executed": True,
    }
    receipt_payload["reporting_contract"] = {
        **receipt_payload.get("reporting_contract", {}),
        "production_runtime_smoke_uses_fused_train_block": False,
    }
    tiny_receipt_path = tmp_path / "tiny_smoke_compile_receipt.json"
    tiny_receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")

    payload = m04_train_step.path_c_fusion_payload(
        model=build_local_gb10_quarter_tiny_smoke_model(),
        compile_receipt_path=tiny_receipt_path,
    )

    receipt = payload["production_compile_receipt"]
    assert receipt["status"] == "mismatch"
    assert receipt["runtime_execution_status"] == "compile_only_not_runtime_ready"
    assert receipt["runtime_route_uses_fused_region"] is False
    assert receipt["runtime_smoke_status"] == "ok"
    assert receipt["runtime_smoke_mode"] == "tiny_mra"
    assert receipt["runtime_smoke_actually_executed"] is True
    assert receipt["production_runtime_smoke_uses_fused_train_block"] is False
    assert {
        "runtime_execution_ready",
        "production_runtime_smoke_ok",
        "production_smoke_uses_fused_train_block",
    }.issubset(set(receipt["failed_checks"]))

    compile_blockers = [
        blocker
        for blocker in payload["schedule_blockers"]
        if blocker["kind"] == "production_schedule_not_compile_verified"
    ]
    assert len(compile_blockers) == 1
    assert compile_blockers[0]["compile_receipt_status"] == "mismatch"
    assert {
        "runtime_execution_ready",
        "production_runtime_smoke_ok",
        "production_smoke_uses_fused_train_block",
    }.issubset(set(compile_blockers[0]["failed_checks"]))


def test_fp8_path_c_training_route_selects_fused_action_when_banks_are_bound(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")

    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        bank_buffers=_model_route_physical_bank_buffers(),
        bank_buffer_owner="local_gb10_quarter.path_c_physical_abi_banks",
        fused_artifact=lambda *args: None,
        fused_train_block_training_runtime=_ReadyFusedTrainBlockTrainingRuntime(),
    )

    assert route["full_end_to_end_training_available"] is True
    assert route["status"] == m04_train_step.FP8_PATH_C_E2E_TRAINING_STATUS
    assert route["fused_train_block_runtime_available"] is True
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["single_fused_train_block_runtime_available"] is True
    assert route["fused_train_block_training_runtime_available"] is True
    assert route["fused_train_block_training_runtime_contract"]["status"] == "ok"
    assert route["fused_train_block_blocker_type"] is None
    assert route["direct_mx_array_artifact_call_status"] == (
        "m04_uses_fused_train_block_route"
    )
    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    assert route["path_c_fusion"]["runtime_training_binding"]["status"] == "ok"
    assert route["path_c_fusion"]["runtime_training_binding"][
        "runtime_uses_fused_train_block"
    ] is True


def test_fp8_path_c_training_route_for_model_reads_bank_owner(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model.path_c_physical_abi_bank_owner = model.make_path_c_physical_abi_bank_owner(
        sequence_length=sequence_length,
    )
    model.path_c_fused_train_block_artifact = lambda *args: None

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["fused_train_block_training_runtime_available"] is False
    assert route["fused_train_block_training_runtime_contract"]["status"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert route["path_c_fusion"]["runtime_training_binding"]["bank_buffer_owner"] == (
        "local_gb10_quarter.path_c_physical_abi_banks"
    )


def test_fp8_path_c_training_route_for_model_reads_bank_owner_factory(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    wrapped = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    owner = _model_route_physical_bank_owner(
        wrapped,
        sequence_length=sequence_length,
    )
    model = _PhysicalAbiBankOwnerFactoryModel(wrapped, owner)

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert model.owner_factory_sequence_lengths == [sequence_length]
    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["fused_train_block_training_runtime_available"] is False
    assert route["single_fused_train_block_runtime_available"] is False
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["fused_train_block_training_runtime_contract"]["status"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS
    )
    # This flag reports that a standalone fused artifact is bound; the
    # selected training action still stays split until the value-and-grad
    # training runtime contract is available.
    assert route["path_c_fusion"]["runtime_training_binding"][
        "runtime_uses_fused_train_block"
    ] is True
    assert route["path_c_fusion"]["runtime_training_binding"]["bank_buffer_owner"] == (
        "local_gb10_quarter.path_c_physical_abi_banks"
    )


def test_fp8_path_c_training_route_for_model_auto_compiles_fused_artifact_when_banks_exist(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model.path_c_physical_abi_bank_owner = model.make_path_c_physical_abi_bank_owner(
        sequence_length=sequence_length,
    )
    lowerer_calls: list[dict[str, Any]] = []

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod
        lowerer_calls.append({"target": target, "kwargs": dict(kwargs)})
        return lambda *args: None

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
        auto_install_fused_train_block=True,
        fused_train_block_artifact_lowerer=fake_lowerer,
    )

    auto_install = route["path_c_fusion"]["fused_train_block_auto_install"]
    assert auto_install["status"] == "ok"
    assert auto_install["artifact_compile"]["status"] == "ok"
    assert auto_install["artifact_compile"]["artifact_bound"] is True
    assert auto_install["training_runtime_available"] is True
    assert auto_install["training_runtime_contract"]["status"] == "ok"
    assert auto_install["training_runtime_contract"]["runtime_installed"] is True
    assert (
        auto_install["training_runtime_contract"]["returns_full_model_grads"]
        is True
    )
    assert auto_install["training_runtime_contract"]["runtime_class"] == (
        "PathCFusedPlusEagerTrainingRuntime"
    )
    assert auto_install["hidden_packing_performed"] is False
    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["fused_train_block_training_runtime_available"] is True
    assert route["path_c_fusion"]["status"] == "runtime_bound_not_default"
    assert route["path_c_fusion"]["runtime_training_binding"][
        "runtime_uses_fused_train_block"
    ] is True
    assert route["path_c_fusion"]["graph_construction"][
        "schedule_construction"
    ] == "dynamic_brick_descriptors"
    assert auto_install["artifact_compile"]["native_compile_ok"] is True
    assert auto_install["artifact_compile"]["plan"]["single_generated_artifact"] is True
    assert (
        auto_install["artifact_compile"]["plan"]["runtime_schedule_contract_status"]
        == "single_launcher_verified"
    )
    assert route["path_c_fusion"]["runtime_training_binding"][
        "hidden_packing_performed"
    ] is False
    assert callable(model.path_c_fused_train_block_artifact)
    assert lowerer_calls[0]["target"] == "metal"


def test_fp8_path_c_training_route_for_model_auto_wraps_contracted_fused_artifact(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model.path_c_physical_abi_bank_owner = model.make_path_c_physical_abi_bank_owner(
        sequence_length=sequence_length,
    )
    artifact = _ContractedFusedTrainBlockArtifact()
    model.path_c_fused_train_block_artifact = artifact

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
        auto_install_fused_train_block=True,
    )

    runtime = model.path_c_fused_train_block_training_runtime
    assert runtime.artifact is artifact
    assert runtime.bank_owner is model.path_c_physical_abi_bank_owner
    assert runtime.training_graph_binding() == {
        "owner": "CompiledPretrainingStep",
        "uses_fused_train_block_runtime": True,
        "uses_forward_hook": True,
        "uses_backward_or_vjp_hook": True,
    }
    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    assert route["fused_train_block_training_runtime_available"] is True
    assert route["single_fused_train_block_runtime_available"] is True
    assert route["fused_train_block_training_runtime_contract"]["status"] == "ok"
    assert route["path_c_fusion"]["runtime_training_binding"][
        "runtime_uses_fused_train_block"
    ] is True


def test_fp8_path_c_training_route_for_model_auto_binds_model_training_runtime(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model.path_c_physical_abi_bank_owner = model.make_path_c_physical_abi_bank_owner(
        sequence_length=sequence_length,
    )
    runtime = _UnboundReadyFusedTrainBlockTrainingRuntime()
    model.path_c_fused_train_block_training_runtime = runtime
    lowerer_calls: list[dict[str, Any]] = []

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod
        lowerer_calls.append({"target": target, "kwargs": dict(kwargs)})
        return lambda *args: None

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
        auto_install_fused_train_block=True,
        fused_train_block_artifact_lowerer=fake_lowerer,
    )

    auto_install = route["path_c_fusion"]["fused_train_block_auto_install"]
    assert auto_install["status"] == "ok"
    assert auto_install["training_runtime_available"] is True
    assert auto_install["training_runtime_contract"]["status"] == "ok"
    assert runtime.training_graph_binding() == {
        "owner": "CompiledPretrainingStep",
        "uses_fused_train_block_runtime": True,
        "uses_forward_hook": True,
        "uses_backward_or_vjp_hook": True,
    }
    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    assert route["fused_train_block_training_runtime_available"] is True
    assert route["single_fused_train_block_runtime_available"] is True
    assert callable(model.path_c_fused_train_block_artifact)
    assert lowerer_calls[0]["target"] == "metal"


def test_path_c_fusion_payload_reads_model_bound_fused_train_block_runtime(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model.path_c_physical_abi_bank_owner = _model_route_physical_bank_owner(
        model,
        sequence_length=sequence_length,
    )
    model.path_c_fused_train_block_artifact = lambda *args: None
    model.path_c_fused_train_block_training_runtime = (
        _ReadyFusedTrainBlockTrainingRuntime()
    )

    payload = m04_train_step.path_c_fusion_payload(
        model=model,
        sequence_length=sequence_length,
    )

    assert payload["runtime_training_binding"]["status"] == "ok"
    assert payload["runtime_training_binding"]["runtime_uses_fused_train_block"] is True
    assert payload["fused_train_block_training_runtime_contract"]["status"] == "ok"
    assert "fused_train_block_runtime_not_bound" not in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }


def test_path_c_fused_train_block_runtime_installer_binds_model_owned_banks(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)

    install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
        model=model,
        bank_owner=_model_route_physical_bank_owner(
            model,
            sequence_length=sequence_length,
        ),
        fused_artifact=lambda *args: None,
        sequence_length=sequence_length,
    )

    assert install["status"] == "blocked"
    assert install["runtime_uses_fused_train_block"] is True
    assert install["runtime_binding"]["status"] == "ok"
    assert install["training_runtime_available"] is False
    assert install["training_runtime_contract"]["status"] == (
        m04_train_step.FP8_PATH_C_FUSED_TRAIN_BLOCK_TRAINING_RUNTIME_MISSING_STATUS
    )
    assert install["hidden_packing_performed"] is False
    assert model.path_c_physical_abi_bank_owner.owner_name == (
        "local_gb10_quarter.path_c_physical_abi_banks"
    )
    assert callable(model.path_c_fused_train_block_artifact)

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert route["selected_action"] == "run_path_c_split_training_route"
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["fused_train_block_training_runtime_available"] is False
    assert route["path_c_fusion"]["runtime_training_binding"]["bank_buffer_owner"] == (
        "local_gb10_quarter.path_c_physical_abi_banks"
    )


def test_path_c_fused_train_block_runtime_installer_attaches_training_runtime_contract(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    runtime = _ReadyFusedTrainBlockTrainingRuntime()

    install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
        model=model,
        bank_owner=_model_route_physical_bank_owner(
            model,
            sequence_length=sequence_length,
        ),
        fused_artifact=lambda *args: None,
        training_runtime=runtime,
        sequence_length=sequence_length,
    )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert install["status"] == "ok"
    assert install["training_runtime_available"] is True
    assert install["training_runtime_contract"]["status"] == "ok"
    assert model.path_c_fused_train_block_training_runtime is runtime
    assert route["fused_train_block_training_runtime_available"] is True
    assert route["selected_action"] == "run_path_c_fused_train_block_route"


def test_compile_path_c_fused_train_block_artifact_for_model_lowers_selected_aot_region(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    expected_route = m04_train_step.path_c_fusion_payload(
        model=model,
        sequence_length=sequence_length,
    )
    lowerer_calls: list[dict[str, Any]] = []

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod
        lowerer_calls.append({"target": target, "kwargs": dict(kwargs)})
        return lambda *args: None

    compiled = m04_train_step.compile_path_c_fused_train_block_artifact_for_model(
        model=model,
        sequence_length=sequence_length,
        lowerer=fake_lowerer,
    )

    assert compiled["status"] == "ok"
    assert compiled["native_compile_ok"] is True
    assert compiled["hidden_packing_performed"] is False
    assert callable(compiled["artifact"])
    assert isinstance(
        compiled["artifact"],
        m04_train_step.PathCFusedTrainBlockCallableArtifact,
    )
    assert compiled["route_region"] == expected_route["graph_construction"][
        "selected_model_region"
    ]
    assert compiled["implementation_kind"] == "production"
    assert compiled["plan"]["single_kernel_fused"] is True
    assert compiled["plan"]["single_generated_artifact"] is True
    assert compiled["plan"]["generated_stage_artifact"] is False
    assert compiled["plan"]["monolithic_native_compile_skipped"] is True
    assert compiled["plan"]["monolithic_runtime_blocked"] is False
    assert compiled["plan"]["monolithic_grid_runtime_blocked"] is True
    assert compiled["plan"]["monolithic_grid_runtime_blocker"]["kind"] == (
        "monolithic_grid_chunks_recurrent_backward_scalar_replay"
    )
    assert compiled["plan"]["selected_runtime_artifact"] == (
        "single_generated_launcher_chunks"
    )
    assert compiled["plan"]["all_stages_single_kernel_fused"] is True
    assert compiled["plan"]["single_launcher_compile_verified"] is True
    assert compiled["plan"]["single_launcher_runtime_blocked"] is False
    assert compiled["plan"]["single_launcher_runtime_blocker"] is None
    assert compiled["plan"]["runtime_schedule_contract_status"] == (
        "single_launcher_verified"
    )
    assert compiled["plan"]["single_launcher_row_launch"][
        "row_chunk_index_param"
    ] == "path_c_row_chunk_index"
    assert compiled["plan"]["single_launcher_row_launch"][
        "row_subchunk_index_param"
    ] == "path_c_row_subchunk_index"
    assert compiled["plan"]["single_launcher_row_launch"][
        "rows_per_kernel_launch"
    ] == 1
    assert compiled["plan"]["single_launcher_row_launch"][
        "row_subchunk_count"
    ] == 64
    assert compiled["plan"]["single_launcher_backward_stage"][
        "backward_stage_count"
    ] == 7
    assert compiled["plan"]["single_launcher_backward_stage"][
        "backward_stage_index_param"
    ] == "path_c_backward_stage_index"
    assert compiled["artifact"].row_chunk_count == 2
    assert compiled["artifact"].row_chunk_index_param == "path_c_row_chunk_index"
    assert compiled["artifact"].row_subchunk_count == 64
    assert compiled["artifact"].row_subchunk_index_param == (
        "path_c_row_subchunk_index"
    )
    assert compiled["artifact"].rows_per_kernel_launch == 1
    assert compiled["artifact"].backward_stage_count == 7
    assert compiled["artifact"].backward_stage_index_param == (
        "path_c_backward_stage_index"
    )
    assert len(lowerer_calls) == 1
    assert compiled["plan"]["backward_graph"] == "aot_autograd"
    training_abi_contract = compiled["training_abi_contract"]
    assert training_abi_contract["status"] == "ok"
    assert training_abi_contract["can_back_value_and_grad"] is True
    assert training_abi_contract["loss_output_available"] is False
    assert training_abi_contract["ntokens_output_available"] is False
    assert training_abi_contract["loss_outputs_source"] == (
        "eager_suffix_replay_cotangent_bridge"
    )
    assert training_abi_contract["train_step_output_abi_declared"] is False
    assert training_abi_contract["train_step_suffix_loss_input_abi_declared"] is False
    assert training_abi_contract["suffix_loss_inputs_available"] is False
    assert training_abi_contract["missing_suffix_loss_inputs"] == []
    assert training_abi_contract["train_step_outputs_computed"] is False
    assert training_abi_contract["train_step_computed_outputs"] == []
    assert training_abi_contract["train_step_pending_outputs"] == []
    loss_source_buffers = training_abi_contract["train_step_loss_source_buffers"]
    assert len(loss_source_buffers) == 2
    assert loss_source_buffers[0].endswith("_R_hidden_after")
    assert loss_source_buffers[1].endswith("_A_sparse_mla_fp8_apply_out")
    assert training_abi_contract["train_step_loss_cotangents_computed"] is False
    assert training_abi_contract["replay_cotangent_boundary_available"] is True
    assert (
        training_abi_contract[
            "train_step_suffix_loss_parameter_grads_computed"
        ]
        is False
    )
    assert training_abi_contract[
        "train_step_suffix_loss_parameter_gradient_buffers"
    ] == []
    assert training_abi_contract[
        "train_step_suffix_loss_parameter_grad_abi"
    ]["gradients_computed"] is False
    assert training_abi_contract["gradient_output_count"] > 0
    assert training_abi_contract["logical_buffer_count"] > (
        training_abi_contract["kernel_parameter_count"]
    )
    assert training_abi_contract["missing_value_and_grad_outputs"] == []
    assert lowerer_calls[0]["target"] == "metal"


def test_compile_path_c_fused_train_block_short_sequence_still_uses_launcher_stage_selector(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--seq-len",
            "16",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    lowerer_calls: list[dict[str, Any]] = []

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod
        lowerer_calls.append({"target": target, "kwargs": dict(kwargs)})
        return lambda *args: None

    compiled = m04_train_step.compile_path_c_fused_train_block_artifact_for_model(
        model=model,
        sequence_length=sequence_length,
        lowerer=fake_lowerer,
    )

    assert compiled["status"] == "ok"
    assert compiled["plan"]["selected_runtime_artifact"] == (
        "single_generated_launcher_chunks"
    )
    assert compiled["plan"]["monolithic_grid_runtime_blocked"] is True
    assert compiled["plan"]["monolithic_grid_runtime_blocker"]["row_chunk_count"] == 1
    assert compiled["plan"]["single_launcher_runtime_blocked"] is False
    assert compiled["artifact"].row_chunk_count == 1
    assert compiled["artifact"].row_chunk_index_param == "path_c_row_chunk_index"
    assert compiled["artifact"].row_subchunk_count == 64
    assert compiled["artifact"].row_subchunk_index_param == (
        "path_c_row_subchunk_index"
    )
    assert compiled["artifact"].backward_stage_count == 7
    assert compiled["artifact"].backward_stage_index_param == (
        "path_c_backward_stage_index"
    )
    assert len(lowerer_calls) == 1


def test_path_c_fused_train_block_runtime_installer_compiles_artifact_when_banks_exist(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    lowerer_calls: list[dict[str, Any]] = []

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod
        lowerer_calls.append({"target": target, "kwargs": dict(kwargs)})
        return lambda *args: None

    model.path_c_physical_abi_bank_owner = (
        _model_route_generated_stage_physical_bank_owner(
            model,
            sequence_length=sequence_length,
        )
    )
    install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
        model=model,
        compile_artifact=True,
        artifact_lowerer=fake_lowerer,
        sequence_length=sequence_length,
    )

    # Mixed-mode PathCFusedPlusEagerTrainingRuntime flips this to ok.
    assert install["status"] == "ok"
    assert install["artifact_compile"]["status"] == "ok"
    assert install["artifact_compile"]["native_compile_ok"] is True
    assert install["artifact_compile"]["artifact_bound"] is True
    assert install["artifact_compile"]["artifact_type"] == (
        "PathCFusedTrainBlockCallableArtifact"
    )
    assert install["artifact_compile"]["plan"]["single_generated_artifact"] is True
    assert install["artifact_compile"]["plan"]["generated_stage_artifact"] is False
    assert (
        install["artifact_compile"]["plan"]["selected_runtime_artifact"]
        == "single_generated_launcher_chunks"
    )
    assert (
        install["artifact_compile"]["plan"]["runtime_schedule_contract_status"]
        == "single_launcher_verified"
    )
    assert install["artifact_compile"]["training_abi_contract"]["status"] == "ok"
    assert install["artifact_kernel_buffer_binding"]["status"] == "ok"
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "can_back_value_and_grad"
        ]
        is True
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "loss_output_available"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "ntokens_output_available"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "train_step_suffix_loss_input_abi_declared"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "suffix_loss_inputs_available"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "missing_suffix_loss_inputs"
        ]
        == []
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "train_step_outputs_computed"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "train_step_computed_outputs"
        ]
        == []
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "train_step_pending_outputs"
        ]
        == []
    )
    install_loss_source_buffers = install["artifact_compile"][
        "training_abi_contract"
    ]["train_step_loss_source_buffers"]
    assert len(install_loss_source_buffers) == 2
    assert install_loss_source_buffers[0].endswith("_R_hidden_after")
    assert install_loss_source_buffers[1].endswith("_A_sparse_mla_fp8_apply_out")
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "train_step_loss_cotangents_computed"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "replay_cotangent_boundary_available"
        ]
        is True
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "train_step_suffix_loss_parameter_grads_computed"
        ]
        is False
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "missing_train_step_suffix_loss_parameter_gradient_buffers"
        ]
        == []
    )
    assert (
        install["artifact_compile"]["training_abi_contract"][
            "missing_value_and_grad_outputs"
        ]
        == []
    )
    assert "artifact" not in install["artifact_compile"]
    assert install["runtime_uses_fused_train_block"] is True
    assert install["training_runtime_available"] is True
    assert install["training_runtime_contract"]["status"] == "ok"
    assert install["training_runtime_contract"]["runtime_installed"] is True
    assert install["training_runtime_contract"]["returns_full_model_grads"] is True
    assert install["training_runtime_contract"]["runtime_class"] == (
        "PathCFusedPlusEagerTrainingRuntime"
    )
    assert install["hidden_packing_performed"] is False
    assert callable(model.path_c_fused_train_block_artifact)
    assert lowerer_calls[0]["target"] == "metal"

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert route["selected_action"] == "run_path_c_fused_train_block_route"
    assert route["single_fused_train_block_standalone_dispatch_available"] is True
    assert route["fused_train_block_training_runtime_available"] is True
    assert route["path_c_fusion"]["runtime_training_binding"][
        "runtime_uses_fused_train_block"
    ] is True


def test_path_c_fused_train_block_runtime_installer_wraps_compiled_artifact_honestly(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    sequence_length = m04_train_step.path_c_training_sequence_length(config)

    def fake_lowerer(func_or_mod: Any, *, target: str, **kwargs: Any) -> Any:
        del func_or_mod, target, kwargs
        return lambda *kernel_args: None

    model.path_c_physical_abi_bank_owner = (
        _model_route_generated_stage_physical_bank_owner(
            model,
            sequence_length=sequence_length,
        )
    )
    install = m04_train_step.install_path_c_fused_train_block_runtime_for_model(
        model=model,
        compile_artifact=True,
        artifact_lowerer=fake_lowerer,
        sequence_length=sequence_length,
    )

    # PathCFusedPlusEagerTrainingRuntime wraps the fused artifact + bank
    # owner and installs the replay/cotangent custom function. The install
    # path reports status='ok' only when the generated artifact owns the
    # in-region block gradients and MLX eager suffix owns loss/lm_head.
    # The wrapped artifact still honestly reports partial coverage
    # (gradient_scope='selected_train_block', covered_count < trainable).
    assert install["status"] == "ok"
    assert install["artifact_compile"]["status"] == "ok"
    assert install["runtime_uses_fused_train_block"] is True
    contract = install["training_runtime_contract"]
    assert contract["runtime_installed"] is True
    assert contract["status"] == "ok"
    assert contract["runtime_class"] == "PathCFusedPlusEagerTrainingRuntime"
    assert contract["value_and_grad_callable"] is True
    value_and_grad_contract = contract["value_and_grad_contract"]
    assert value_and_grad_contract["status"] == "ok"
    assert value_and_grad_contract["returns_model_grads"] is True
    assert value_and_grad_contract["returns_full_model_grads"] is True
    coverage = value_and_grad_contract["full_model_gradient_coverage"]
    assert contract["full_model_gradient_coverage"] == coverage
    assert coverage["full_model_gradient_tree_ready"] is True
    assert (
        coverage["gradient_scope"]
        == "full_model_via_fused_replay_cotangent_bridge"
    )
    assert coverage["missing_parameter_count"] == 0
    assert value_and_grad_contract["suffix_bypass_available"] is False
    assert value_and_grad_contract["replay_cotangent_bridge_available"] is True
    assert value_and_grad_contract["bank_grad_overlay_active"] is True
    assert value_and_grad_contract["merged_bank_resident_parameter_count"] > 0
    assert coverage["bank_grad_overlay_active"] is True
    assert value_and_grad_contract["delegates_to_eager_loss_and_grad"] is False
    assert value_and_grad_contract["hidden_packing_performed"] is False
    inner = model.path_c_fused_train_block_artifact.value_and_grad_contract()
    assert inner["gradient_scope"] == "selected_train_block"
    assert inner["returns_full_model_grads"] is False
    assert "layers.0.block.q_proj.weight" in (
        inner["full_model_gradient_coverage"]["missing_parameter_names"]
    )


def test_fp8_path_c_training_route_for_model_uses_model_derived_regions(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    graph = route["path_c_fusion"]["graph_construction"]
    assert graph["input_model"] == "local_gb10_quarter.path_c_bricks"
    assert graph["selected_model_region"] == "local_gb10_quarter_path_c_10_12"
    assert route["path_c_fusion"]["model_route_candidates"]["profile"] == (
        "local_gb10_quarter"
    )
    candidate = route["path_c_fusion"]["model_route_candidates"][
        "selected_candidate"
    ]
    assert candidate["brick_names"] == [
        "local_gb10_quarter_brick_10_M",
        "local_gb10_quarter_brick_11_R",
        "local_gb10_quarter_brick_12_A",
    ]
    audit = route["path_c_fusion"]["direct_chained_fusion"]["model_binding_audit"]
    assert audit["status"] == "runtime_backward_or_state_owner_missing"
    assert audit["required_logical_buffer_count"] == 94
    assert audit["model_parameter_or_constant_count"] == 26
    assert audit["runtime_activation_or_grad_count"] == 62
    assert audit["backward_gradient_count"] == 39
    assert audit["forward_activation_probe_surface_available"] is True
    assert audit["parameter_gradient_probe_surface_available"] is True
    assert audit["model_parameter_logical_owner_available"] is True
    assert audit["model_parameter_logical_owner"] == (
        "local_gb10_quarter.path_c_model_parameter_buffers"
    )
    assert audit["model_parameter_logical_owner_buffer_count"] == 26
    assert audit["model_parameter_logical_owner_total_buffer_count"] > 25
    assert audit["backward_gradient_parameter_alias_coverage_count"] == 28
    assert audit["backward_gradient_uncovered_count"] == 11
    assert audit["profile_brick_names_attached"] is True
    assert audit["forward_activation_or_prepared_count"] == 23
    assert audit["runtime_state_count"] == 6
    pre_step_plan = route["path_c_fusion"]["direct_chained_fusion"][
        "pre_step_owner_plan"
    ]
    assert pre_step_plan["status"] == "pre_step_runtime_owner_missing"
    assert pre_step_plan["training_critical_path_ready"] is False
    assert pre_step_plan["model_parameter_or_constant_available_count"] == 26
    assert pre_step_plan["model_parameter_or_constant_missing_count"] == 0
    assert pre_step_plan["batch_dependent_forward_or_prepared_count"] == 23
    assert pre_step_plan["batch_dependent_forward_or_prepared_missing_count"] == 23
    assert pre_step_plan["runtime_state_count"] == 6
    assert pre_step_plan["runtime_state_missing_count"] == 6
    assert pre_step_plan["pre_step_runtime_missing_count"] == 29
    assert pre_step_plan["backward_workspace_gradient_count"] == 39
    assert (
        "local_gb10_quarter_brick_10_M_hidden"
        in pre_step_plan["batch_dependent_forward_or_prepared_missing_examples"]
    )
    assert pre_step_plan["runtime_state_missing_examples"] == [
        "local_gb10_quarter_brick_10_M_mamba3_h0",
        "local_gb10_quarter_brick_10_M_state",
        "local_gb10_quarter_brick_10_M_state_in",
        "local_gb10_quarter_brick_11_R_m2rnn_conv_state",
        "local_gb10_quarter_brick_11_R_m2rnn_h0",
        "mamba3_angle_grad_state",
    ]
    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]
    assert runtime_binding["logical_buffer_owner"] == (
        "local_gb10_quarter.path_c_model_parameter_buffers"
    )
    assert runtime_binding["provided_logical_buffer_count"] == 26
    assert runtime_binding["shape_mismatch_buffers"] == []
    assert runtime_binding["missing_logical_buffer_count"] == 66
    assert (
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_mla_sm_scale"
        in runtime_binding["missing_logical_buffers"]
    )
    assert runtime_binding["hidden_packing_performed"] is False


def test_path_c_direct_chain_pre_step_owner_is_dynamic_batch_abi(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    assert int(config.seq_len) != 512
    seq_len = 512
    tokens = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)

    owner = m04_train_step.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain,
        model=model,
        batch={"tokens": tokens},
    )
    binding = m04_train_step.path_c_direct_fusion_chain_runtime_binding_payload(
        chain=chain,
        logical_owner=owner,
        artifacts=_model_route_direct_chain_artifacts(model),
    )
    pre_step_plan = m04_train_step.path_c_direct_chain_pre_step_owner_plan(
        chain=chain,
        logical_owner=owner,
    )

    assert owner.owner_name == "local_gb10_quarter.path_c_pre_step_runtime_buffers"
    assert len(owner.buffers) == 94
    assert owner.hidden_packing_performed is False
    assert owner.no_hidden_allocation_policy is True
    assert owner.buffers[
        "local_gb10_quarter_brick_10_M_hidden"
    ].shape == (1, seq_len, 16)
    parameters = dict(tree_flatten(model.trainable_parameters()))
    assert owner.buffers[
        "local_gb10_quarter_brick_10_M_mamba3_conv_weight"
    ] is parameters["layers.10.block.conv_weight"]
    assert binding["status"] == "ok"
    assert binding["runtime_uses_direct_fusion_chain"] is True
    assert binding["provided_logical_buffer_count"] == 92
    assert binding["missing_logical_buffer_count"] == 0
    assert binding["unexpected_logical_buffer_count"] == 0
    assert pre_step_plan["status"] == "pre_step_runtime_owner_ready"
    assert pre_step_plan["training_critical_path_ready"] is True
    assert pre_step_plan["model_parameter_or_constant_available_count"] == 26
    assert pre_step_plan["batch_dependent_forward_or_prepared_available_count"] == 23
    assert pre_step_plan["runtime_state_available_count"] == 6
    assert pre_step_plan["backward_workspace_gradient_available_count"] == 39


def test_path_c_direct_chain_pre_step_owner_preserves_bf16_model_dtype(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "bfloat16",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    del config
    model = build_local_gb10_quarter_tiny_smoke_model(dtype=mx.bfloat16)
    chain = _model_route_direct_chain(model)
    seq_len = 512
    tokens = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)

    owner = m04_train_step.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain,
        model=model,
        batch={"tokens": tokens},
    )
    binding = m04_train_step.path_c_direct_fusion_chain_runtime_binding_payload(
        chain=chain,
        logical_owner=owner,
        artifacts=_model_route_direct_chain_artifacts(model),
    )

    assert owner.buffers[
        "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight"
    ].dtype == mx.bfloat16
    assert binding["status"] == "ok"
    assert binding["runtime_uses_direct_fusion_chain"] is True
    assert binding["dtype_mismatch_count"] == 0
    assert binding["dtype_mismatch_buffers"] == []


def test_path_c_direct_chain_plans_for_runtime_input_sequence_length(
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    model = build_local_gb10_quarter_tiny_smoke_model()
    input_sequence_length = 128
    profile_name = str(getattr(model, "path_c_profile_name", "HybridTinyLM"))
    region_prefix = m04_train_step._path_c_direct_chain_region_prefix(
        model,
        profile_name,
    )
    direct_chains = m04_train_step.plan_path_c_direct_fusion_chains_for_model(
        model,
        region_prefix=region_prefix,
        include_backward=True,
        sequence_length=input_sequence_length,
    )
    regions = m04_train_step.build_path_c_model_regions_from_model(
        model,
        region_prefix=region_prefix,
        include_backward=False,
        sequence_length=input_sequence_length,
    )
    selected_region = m04_train_step._select_path_c_model_route_region(regions)
    chain = m04_train_step._select_path_c_direct_chain_for_region(
        direct_chains,
        selected_region,
    )
    assert chain is not None
    tokens = mx.arange(input_sequence_length + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)

    owner = m04_train_step.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain,
        model=model,
        batch={"tokens": tokens},
    )
    artifacts = tuple(lambda *args: None for _ in chain.segments)
    binding = m04_train_step.path_c_direct_fusion_chain_runtime_binding_payload(
        chain=chain,
        logical_owner=owner,
        artifacts=artifacts,
    )

    assert owner.buffers[
        "local_gb10_quarter_brick_10_M_hidden"
    ].shape == (1, input_sequence_length, model.config.hidden_size)
    assert binding["status"] == "ok"
    assert binding["runtime_uses_direct_fusion_chain"] is True
    assert binding["shape_mismatch_buffers"] == []


def test_local_gb10_direct_chain_m2rnn_state_weight_uses_value_square_shape(
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    profile = m04_train_step.local_gb10_quarter_profile()
    model_descriptor = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    chains = m04_train_step.plan_path_c_direct_fusion_chains_for_model(
        model_descriptor,
        region_prefix=f"{profile.name}_path_c",
        include_backward=True,
    )
    selected_chain = max(
        chains,
        key=lambda chain: len(getattr(getattr(chain, "source_region", None), "nodes", ())),
    )
    state_weight_shape = None
    for segment in selected_chain.segments:
        target = segment.schedule_target
        assert target is not None
        prim_func = target.schedule_template(segment.region)
        abi_map = getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map")
        key = "local_gb10_quarter_brick_11_R_m2rnn_state_weight"
        if key in abi_map:
            state_weight_shape = tuple(int(dim) for dim in abi_map[key]["shape"])
            break

    m2rnn_config = profile.hybrid_config().m2rnn_config()
    assert state_weight_shape == (
        m2rnn_config.num_weight_heads,
        m2rnn_config.v_head_dim,
        m2rnn_config.v_head_dim,
    )


def test_path_c_direct_chain_runtime_rebuilds_pre_step_owner_on_step(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--seq-len",
            "513",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    artifacts = _model_route_direct_chain_artifacts(model)
    seq_len = 512
    tokens_a = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens_a = tokens_a % mx.array(model.config.vocab_size, dtype=mx.int32)
    tokens_b = (tokens_a + 3) % mx.array(model.config.vocab_size, dtype=mx.int32)
    initial_owner = m04_train_step.make_path_c_direct_chain_pre_step_runtime_owner(
        chain=chain,
        model=model,
        batch={"tokens": tokens_a},
    )
    factory_calls: list[tuple[int, ...]] = []

    def owner_factory(
        model_arg: nn.Module,
        batch: Mapping[str, mx.array],
    ) -> PathCLogicalBufferOwner:
        tokens = batch["tokens"]
        factory_calls.append(tuple(int(dim) for dim in tokens.shape))
        return m04_train_step.make_path_c_direct_chain_pre_step_runtime_owner(
            chain=chain,
            model=model_arg,
            batch=batch,
        )

    install_payload = (
        m04_train_step.install_path_c_direct_chain_training_runtime_for_model(
            model=model,
            chain=chain,
            artifacts=artifacts,
            logical_owner=initial_owner,
            owner_name="unit.path_c_direct_training_runtime",
            training_critical_path=True,
            run_probe=False,
            loss_cotangent_bridge=m04_train_step.PathCResidualSumSuffixLossCotangentBridge(
                chunk_rows=128,
            ),
            pre_step_owner_factory=owner_factory,
        )
    )
    runtime = model.path_c_direct_fusion_chain_training_runtime
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())

    def forbidden_loss_fn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Path C critical path must not delegate to eager loss")

    stepper = m04_train_step.CompiledPretrainingStep(
        model,
        optimizer,
        compile=False,
        loss_fn=forbidden_loss_fn,
        path_c_training_runtime=runtime,
    )

    metrics = stepper({"tokens": tokens_b})
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )
    grad_name = next(
        name
        for name, value in runtime.last_pre_step_owner.buffers.items()
        if name.endswith("_grad") and isinstance(value, mx.array)
    )
    post_step_capture = m04_train_step.PathCGradientBufferCapture(
        owner_name="local_gb10_quarter.path_c_parameter_gradient_capture",
    )
    post_step_capture(
        {
            "logical_names": (grad_name,),
            "tensor": mx.ones_like(runtime.last_pre_step_owner.buffers[grad_name]),
        }
    )
    model.path_c_direct_fusion_chain_logical_buffer_owners = (post_step_capture,)
    route_after_capture = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    assert install_payload["status"] == "ok"
    assert install_payload["training_critical_path"] is True
    assert factory_calls == [(1, seq_len + 1)]
    assert runtime.last_pre_step_owner is not initial_owner
    assert runtime.last_pre_step_binding["status"] == "ok"
    assert runtime.binding["runtime_uses_direct_fusion_chain"] is True
    assert metrics.compiled is False
    assert metrics.updated is True
    assert metrics.ntokens == seq_len
    assert route["direct_fusion_chain_training_runtime_available"] is True
    # Mixed-mode fused-train-block runtime now auto-installs and wins
    # precedence over the explicit direct-chain runtime; either action
    # is a valid Path C training entry. The direct-chain runtime is
    # still installed and its pre-step owner book-keeping is unchanged.
    assert route["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_direct_fusion_chain_route",
    }
    assert route_after_capture["selected_action"] in {
        "run_path_c_fused_train_block_route",
        "run_path_c_direct_fusion_chain_route",
    }
    assert route_after_capture["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]["logical_buffer_owner"] == runtime.last_pre_step_owner.owner_name


def test_path_c_fusion_payload_reads_model_bound_direct_chain_runtime(
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    model = build_local_gb10_quarter_tiny_smoke_model()
    chain = _model_route_direct_chain(model)
    seq_len = 512
    tokens = mx.arange(seq_len + 1, dtype=mx.int32)[None, :]
    tokens = tokens % mx.array(model.config.vocab_size, dtype=mx.int32)

    def owner_factory(
        model_arg: nn.Module,
        batch: Mapping[str, mx.array],
    ) -> PathCLogicalBufferOwner:
        return m04_train_step.make_path_c_direct_chain_pre_step_runtime_owner(
            chain=chain,
            model=model_arg,
            batch=batch,
        )

    initial_owner = owner_factory(model, {"tokens": tokens})
    install_payload = (
        m04_train_step.install_path_c_direct_chain_training_runtime_for_model(
            model=model,
            chain=chain,
            artifacts=_model_route_direct_chain_artifacts(model),
            logical_owner=initial_owner,
            training_critical_path=True,
            loss_cotangent_bridge=m04_train_step.PathCResidualSumSuffixLossCotangentBridge(
                chunk_rows=128,
            ),
            pre_step_owner_factory=owner_factory,
        )
    )

    payload = m04_train_step.path_c_fusion_payload(
        model=model,
        sequence_length=seq_len,
    )

    assert install_payload["status"] == "ok"
    direct_chain = payload["direct_chained_fusion"]
    assert direct_chain["runtime_binding"]["runtime_uses_direct_fusion_chain"] is True
    assert direct_chain["training_runtime_contract"]["status"] == "ok"
    assert payload["fused_train_block_training_critical_path"] is False
    assert payload["fused_train_block_training_runtime_available"] is False
    assert direct_chain["training_critical_path"] is True
    assert "fused_train_block_runtime_not_bound" in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }


def test_path_c_fusion_payload_reports_model_level_direct_chain_planner() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model()

    payload = m04_train_step.path_c_fusion_payload(model=model)

    direct_chain = payload["direct_chained_fusion"]
    assert direct_chain["source_region_name"] == "local_gb10_quarter_path_c_10_12"
    assert direct_chain["construction"]["planner"] == (
        "plan_path_c_direct_fusion_chains_for_model"
    )
    assert direct_chain["construction"]["region_prefix"] == (
        "local_gb10_quarter_path_c"
    )
    assert direct_chain["construction"]["candidate_chain_count"] == 1
    assert payload["graph_construction"]["static_acceptance_fixture_used_for_selection"] is False


def test_path_c_direct_chain_runtime_capture_owners_reduce_binding_gap() -> None:
    model = build_local_gb10_quarter_tiny_smoke_model()
    activation_capture, gradient_capture = (
        m04_train_step._path_c_direct_chain_runtime_capture_owners_for_model(model)
    )
    sequence_length = 127
    fake_buffers = _model_route_direct_chain_logical_buffers(
        model,
        sequence_length=sequence_length,
    )

    def tensor_for(name: str) -> mx.array:
        spec = fake_buffers[name]
        dtype = getattr(mx, str(spec.dtype))
        return mx.zeros(tuple(spec.shape), dtype=dtype)

    activation_capture(
        {
            "logical_names": ("local_gb10_quarter_brick_10_M_hidden",),
            "tensor": tensor_for("local_gb10_quarter_brick_10_M_hidden"),
        }
    )
    for name, tensor in fake_buffers.items():
        if (
            name.endswith(
                (
                    "_mamba3_h0",
                    "_state",
                    "_state_in",
                    "_m2rnn_h0",
                    "_m2rnn_conv_state",
                )
            )
            or any(
                token in name
                for token in (
                    "_hidden",
                    "_residual_norm_hidden",
                    "_delta",
                    "_hidden_after",
                    "_qkv_projection_q_fp8",
                    "_qkv_projection_q_scale",
                    "_qkv_projection_kv_fp8",
                    "_qkv_projection_kv_scale",
                    "_qkv_projection_indices",
                    "_sparse_mla_fp8_apply_lse",
                    "_sparse_mla_fp8_apply_out",
                    "_sparse_mla_fp8_apply_sparse_mla_has_sinks",
                    "_sparse_mla_fp8_apply_sparse_mla_sinks",
                    "_sparse_mla_fp8_apply_sparse_mla_sm_scale",
                )
            )
        ) and not name.endswith("_grad"):
            activation_capture({"logical_names": (name,), "tensor": tensor_for(name)})
    for source, targets in gradient_capture.aliases.items():
        for target in targets:
            if target in fake_buffers:
                gradient_capture(
                    {"logical_names": (source,), "tensor": tensor_for(target)}
                )
                break
    model.path_c_direct_fusion_chain_logical_buffer_owners = (
        activation_capture,
        gradient_capture,
    )

    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            "/tmp/path-c-runtime-capture-owner-receipt.json",
        ]
    )
    config = m04_train_step.config_from_args(
        args,
        data_path=Path("/tmp/path-c-runtime-capture-owner-tokens.npz"),
    )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )
    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]
    suffix_loss_gradient_buffers = {"final_norm_weight_grad", "lm_head_weight_grad"}
    segment_buffer_count = len(set(fake_buffers).difference(suffix_loss_gradient_buffers))

    assert runtime_binding["logical_buffer_owner"] == (
        "local_gb10_quarter.path_c_direct_fusion_chain_buffers"
    )
    assert runtime_binding["provided_logical_buffer_count"] == (
        segment_buffer_count - runtime_binding["missing_logical_buffer_count"]
    )
    assert runtime_binding["missing_logical_buffer_count"] > 0
    assert runtime_binding["shape_mismatch_count"] == 0
    assert runtime_binding["dtype_mismatch_count"] == 0
    assert "hidden" not in runtime_binding["missing_logical_buffers"]
    assert (
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_sparse_mla_sm_scale"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert (
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert (
        "local_gb10_quarter_brick_12_A_qkv_projection_q_fp8_grad"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert (
        "local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8_grad"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert (
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_grad"
        in runtime_binding["missing_logical_buffers"]
    )
    assert runtime_binding["hidden_packing_performed"] is False

    vjp_activation_grad_sources = {
        "local_gb10_quarter_brick_10_M_hidden_grad": (
            "local_gb10_quarter_brick_10_M_hidden_grad"
        ),
        "local_gb10_quarter_brick_10_M_delta_grad": (
            "local_gb10_quarter_brick_10_M_delta_grad"
        ),
        "local_gb10_quarter_brick_10_M_hidden_after_grad": (
            "local_gb10_quarter_brick_10_M_hidden_after_grad"
        ),
        "local_gb10_quarter_brick_11_R_delta_grad": (
            "local_gb10_quarter_brick_11_R_delta_grad"
        ),
        "local_gb10_quarter_brick_11_R_hidden_after_grad": (
            "local_gb10_quarter_brick_11_R_hidden_after_grad"
        ),
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_grad": (
            "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_grad"
        ),
    }
    for logical_name, buffer_name in vjp_activation_grad_sources.items():
        activation_capture(
            {
                "logical_names": (logical_name,),
                "tensor": tensor_for(buffer_name),
                "phase": "value_and_grad",
            }
        )
    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )
    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]

    assert runtime_binding["provided_logical_buffer_count"] == (
        segment_buffer_count - runtime_binding["missing_logical_buffer_count"]
    )
    assert runtime_binding["missing_logical_buffer_count"] > 0
    assert runtime_binding["shape_mismatch_count"] == 0
    assert (
        "local_gb10_quarter_brick_10_M_hidden_grad"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert (
        "local_gb10_quarter_brick_10_M_delta_grad"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert (
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out_grad"
        not in runtime_binding["missing_logical_buffers"]
    )
    assert runtime_binding["missing_logical_buffers"] == [
        "local_gb10_quarter_brick_10_M_entry_rmsnorm_hidden_grad",
        "local_gb10_quarter_brick_10_M_mamba3_h0_grad",
        "local_gb10_quarter_brick_10_M_residual_norm_hidden_grad",
        "local_gb10_quarter_brick_10_M_state_in_grad",
        "local_gb10_quarter_brick_11_R_m2rnn_h0_grad",
        "m2rnn_h_checkpoint",
        "mamba3_angle_checkpoint",
        "mamba3_h_checkpoint",
        "path_c_float32_scratch_bank",
        "path_c_int32_scratch_bank",
    ]

    workspace_owner = (
        m04_train_step.make_path_c_direct_fusion_chain_workspace_owner(
            chain=_model_route_direct_chain(
                model,
                sequence_length=sequence_length,
            ),
            logical_buffer_names=runtime_binding["missing_logical_buffers"],
            owner_name="local_gb10_quarter.path_c_direct_fusion_chain_workspace",
        )
    )
    assert set(workspace_owner.buffers) == set(
        runtime_binding["missing_logical_buffers"]
    )
    assert workspace_owner.buffers[
        "local_gb10_quarter_brick_10_M_state_in_grad"
    ].shape == workspace_owner.buffers[
        "local_gb10_quarter_brick_10_M_mamba3_h0_grad"
    ].shape
    model.path_c_direct_fusion_chain_logical_buffer_owners = (
        activation_capture,
        gradient_capture,
        workspace_owner,
    )

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )
    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]

    assert runtime_binding["provided_logical_buffer_count"] == segment_buffer_count
    assert runtime_binding["missing_logical_buffer_count"] == 0
    assert runtime_binding["logical_tensor_binding_ready"] is True
    assert runtime_binding["direct_chain_artifacts_bound"] is False
    assert runtime_binding["status"] == (
        m04_train_step.FP8_PATH_C_DIRECT_CHAIN_ARTIFACTS_MISSING_STATUS
    )
    assert runtime_binding["missing_artifact_segments"] == [0, 1, 2, 3, 4, 5, 6]
    assert runtime_binding["hidden_packing_performed"] is False


def test_fp8_path_c_training_route_composes_model_and_runtime_logical_owners(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    extra_owner = PathCLogicalBufferOwner(
        "local_gb10_quarter.path_c_runtime_activation_buffers",
        {
            "hidden": _PathCBankLike((1, sequence_length, 16), "float32"),
            "hidden_grad": _PathCBankLike((1, sequence_length, 16), "float32"),
        },
    )
    model.path_c_direct_fusion_chain_logical_buffer_owners = (extra_owner,)

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]
    assert runtime_binding["logical_buffer_owner"] == (
        "local_gb10_quarter.path_c_direct_fusion_chain_buffers"
    )
    assert runtime_binding["hidden_packing_performed"] is False
    assert runtime_binding["provided_logical_buffer_count"] == 26
    assert runtime_binding["missing_logical_buffer_count"] == 66
    assert [
        segment["execution_phase"] for segment in runtime_binding["segments"]
    ] == [
        "forward",
        "forward",
        "backward",
        "backward",
        "backward",
        "backward",
        "backward",
    ]
    assert "hidden" not in runtime_binding["missing_logical_buffers"]
    assert "hidden_grad" not in runtime_binding["missing_logical_buffers"]
    assert "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_out" in (
        runtime_binding["missing_logical_buffers"]
    )
    assert "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_lse" in (
        runtime_binding["missing_logical_buffers"]
    )


def test_fp8_path_c_training_route_accepts_real_activation_capture_owner(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    sequence_length = m04_train_step.path_c_training_sequence_length(config)
    model = build_local_gb10_quarter_tiny_smoke_model()
    capture = PathCActivationBufferCapture(
        aliases={"local_gb10_quarter_brick_10_M_hidden": "hidden"},
        owner_name="local_gb10_quarter.path_c_forward_activation_capture",
    )
    model.attach_path_c_activation_probe(capture)
    hidden = model.decoder_hidden_states(
        mx.zeros((1, sequence_length), dtype=mx.int32)
    )
    mx.eval(hidden)
    model.path_c_direct_fusion_chain_logical_buffer_owners = (capture,)

    route = m04_train_step.fp8_path_c_training_route_payload_for_model(
        config,
        model,
    )

    runtime_binding = route["path_c_fusion"]["direct_chained_fusion"][
        "runtime_binding"
    ]
    assert runtime_binding["logical_buffer_owner"] == (
        "local_gb10_quarter.path_c_direct_fusion_chain_buffers"
    )
    assert runtime_binding["shape_mismatch_buffers"] == [
        "local_gb10_quarter_brick_10_M_mamba3_h0",
        "local_gb10_quarter_brick_10_M_state",
        "local_gb10_quarter_brick_10_M_state_in",
        "local_gb10_quarter_brick_11_R_m2rnn_conv_state",
        "local_gb10_quarter_brick_11_R_m2rnn_h0",
    ]
    assert "hidden" not in runtime_binding["missing_logical_buffers"]
    assert "local_gb10_quarter_brick_10_M_delta" not in (
        runtime_binding["missing_logical_buffers"]
    )
    assert capture.buffers["hidden"] is capture.buffers[
        "local_gb10_quarter_brick_10_M_hidden"
    ]


def test_receipt_preserves_bound_fp8_path_c_route_from_train_payload(
    tmp_path: Path,
    path_c_fusion_auto_env: None,
) -> None:
    del path_c_fusion_auto_env
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--dtype",
            "fp8_path_c",
            "--pattern",
            "A",
            "--depth",
            "1",
            "--dsa-a-layer-ranks",
            "0",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=tmp_path / "tokens.npz")
    route = m04_train_step.fp8_path_c_training_route_payload(
        config,
        bank_owner=_model_route_physical_bank_owner(),
        fused_artifact=lambda *args: None,
        fused_train_block_training_runtime=_ReadyFusedTrainBlockTrainingRuntime(),
    )

    receipt = m04_train_step.receipt_from_train_payload(
        args,
        config=config,
        train_payload={
            "status": "ok",
            "tokens_per_step": 8,
            "trained_tokens": 16,
            "final_loss": 1.0,
            "mean_loss": 1.5,
            "step_metrics": [
                {
                    "loss": 2.0,
                    "seconds": 0.5,
                    "ntokens": 8,
                    "tokens_per_second": 16.0,
                    "updated": True,
                    "trained_tokens": 8,
                },
                {
                    "loss": 1.0,
                    "seconds": 0.25,
                    "ntokens": 8,
                    "tokens_per_second": 32.0,
                    "updated": True,
                    "trained_tokens": 16,
                },
            ],
            "kernel_dispatch": [],
            "fp8_path_c_training_route": route,
        },
        memory_before={"active_memory_bytes": 0},
        memory_after={"active_memory_bytes": 0, "peak_memory_bytes": 0},
    )

    assert receipt["workload"]["fp8_path_c_training_route"]["selected_action"] == (
        "run_path_c_fused_train_block_route"
    )
    assert receipt["training"]["fp8_path_c_training_route"]["selected_action"] == (
        "run_path_c_fused_train_block_route"
    )
    assert receipt["training"]["fp8_path_c_training_route"][
        "fused_train_block_runtime_available"
    ] is True


def test_path_c_fusion_force_mode_fails_closed_until_runtime_is_bound(
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env

    payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )

    assert payload["mode"] == "force"
    assert payload["status"] == "force_blocked_schedule_unverified"
    assert payload["single_kernel_fused"] is False
    assert payload["fullgraph_required"] is True
    assert payload["graph_break_policy"] == "fail_closed"
    assert payload["autograd_plan"]["status"] == "ready"
    assert payload["schedule_name"] == (
        "local_gb10_quarter_path_c_10_12:descriptor_generated_fwd_bwd"
    )
    assert payload["schedule_status"] == "ready"
    assert payload["schedule_contract"]["status"] == "registered_not_lowered"
    assert payload["schedule_contract"]["declared_implementation_kind"] == "production"
    assert payload["schedule_contract"]["declared_schedule_id"] == (
        payload["production_schedule"]["schedule_id"]
    )
    assert payload["production_schedule"]["schedule_id"].startswith(
        "path_c_descriptor_chain_"
    )
    assert payload["production_schedule"]["implementation_status"] == (
        "ready"
    )
    assert (
        payload["production_schedule"]["schedule_generator_status"]
        == "production_region_fragments"
    )
    assert payload["production_schedule"]["internal_buffer_policy"] == (
        "row_local_hidden"
    )
    assert payload["production_schedule"]["loop_policy"] == "row_phased_hidden"
    assert payload["production_schedule"]["real_abi_contract_complete"] is True
    assert payload["production_schedule"]["missing_real_abi_inputs"] == []
    assert payload["production_schedule"]["production_fragments_complete"] is True
    assert any(
        status.startswith("m2rnn:production_region_inlined:")
        for status in payload["production_schedule"][
            "brick_production_fragment_statuses"
        ]
    )
    assert "selected_model_schedule_not_default" in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }
    assert "production_1b_matrix_profile_missing" in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }
    assert "production_schedule_not_compile_verified" in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }
    assert payload["production_compile_receipt"]["status"] == "mismatch"
    assert payload["production_compile_receipt"]["verified"] is False
    assert payload["production_compile_receipt"]["native_compile_ok"] is True
    assert payload["production_compile_receipt"]["cache_key_recompile_status"] == (
        "key_stable"
    )
    assert payload["production_compile_receipt"]["runtime_execution_status"] == (
        "compile_only_not_runtime_ready"
    )
    assert payload["production_compile_receipt"]["runtime_route_uses_fused_region"] is False
    assert payload["production_compile_receipt"]["runtime_smoke_status"] == "ok"
    assert payload["production_compile_receipt"]["runtime_smoke_mode"] == "production_1b"
    assert (
        payload["production_compile_receipt"]["runtime_smoke_actually_executed"] is False
    )
    assert (
        payload["production_compile_receipt"][
            "production_runtime_smoke_uses_fused_train_block"
        ]
        is False
    )
    assert payload["production_compile_receipt"]["failed_checks"] == [
        "runtime_execution_ready",
        "production_runtime_smoke_ok",
        "production_smoke_uses_fused_train_block",
    ]
    assert "production_schedule_uses_descriptor_loop_fragments" not in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }
    assert "missing_real_abi_inputs" not in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }
    assert payload["semantic_blockers"] == []
    assert "diagnostic_raw_abi_region" not in payload
    assert payload["default_allowed"] is False
    assert "not yet trusted" in payload["reason"]
    assert "compile-verified" in payload["reason"]


def test_path_c_fusion_payload_accepts_matching_matrix_profile_receipt(
    tmp_path: Path,
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env
    baseline_payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )
    schedule = baseline_payload["production_schedule"]
    matrix_receipt_path = tmp_path / "path_c_fusion_matrix_profile.json"
    matrix_receipt_path.write_text(
        json.dumps(
            {
                "kind": "cppmega_path_c_fusion_matrix_profile_receipt",
                "status": "ok",
                "model_profile": m04_train_step.REQUIRED_MODEL_PROFILE,
                "schedule_id": schedule["schedule_id"],
                "schedule_name": schedule["schedule_name"],
                "single_cppmega_commit": True,
                "cppmega_sha": "abc123",
                "full_1b_matrix_captured": True,
                "profiling_traces_captured": True,
                "memory_non_regression_ok": True,
                "cache_receipts_captured": True,
                "path_b_baselines_clean": True,
                "path_c_default_gate_rows_passed": True,
                "path_c_peak_memory_non_regression": True,
                "path_c_warm_cache_hit_observed": True,
                "matrix_rows": [
                    {
                        "dtype_route": dtype_route,
                        "optimizer": optimizer,
                        "status": "ok",
                    }
                    for dtype_route in m04_train_step.MATRIX_DTYPE_ROUTES
                    for optimizer in m04_train_step.MATRIX_OPTIMIZERS
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
        matrix_profile_receipt_path=matrix_receipt_path,
    )

    assert payload["production_matrix_profile_receipt"]["status"] == "verified"
    assert payload["production_matrix_profile_receipt"]["verified"] is True
    assert payload["production_matrix_profile_receipt"]["schedule_id"] == (
        schedule["schedule_id"]
    )
    assert "production_1b_matrix_profile_missing" not in {
        blocker["kind"] for blocker in payload["schedule_blockers"]
    }
    assert payload["acceptance_gate"]["current_matrix_profile_verified"] is True
    assert payload["default_allowed"] is False


def test_path_c_fusion_payload_rejects_incomplete_matrix_profile_receipt(
    tmp_path: Path,
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env
    baseline_payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )
    schedule = baseline_payload["production_schedule"]
    matrix_receipt_path = tmp_path / "incomplete_path_c_fusion_matrix_profile.json"
    matrix_receipt_path.write_text(
        json.dumps(
            {
                "kind": "cppmega_path_c_fusion_matrix_profile_receipt",
                "status": "ok",
                "model_profile": m04_train_step.REQUIRED_MODEL_PROFILE,
                "schedule_id": schedule["schedule_id"],
                "schedule_name": schedule["schedule_name"],
                "single_cppmega_commit": True,
                "cppmega_sha": "abc123",
                "full_1b_matrix_captured": True,
                "profiling_traces_captured": True,
                "memory_non_regression_ok": True,
                "cache_receipts_captured": True,
                "path_b_baselines_clean": True,
                "path_c_default_gate_rows_passed": True,
                "path_c_peak_memory_non_regression": True,
                "path_c_warm_cache_hit_observed": True,
                "row_check_summary": {
                    "total_rows": 1,
                    "row_status_ok": 1,
                    "path_b_baseline_clean": 1,
                    "path_c_default_gate_passed": 1,
                    "path_c_peak_memory_non_regression": 1,
                    "path_c_warm_cache_hit_observed": 1,
                    "path_c_cold_cache_miss_observed": 1,
                    "profiling_trace_captured": 1,
                },
                "failed_rows_by_check": {},
                "matrix_rows": [
                    {
                        "dtype_route": "fp8_path_c",
                        "optimizer": "adamw",
                        "status": "ok",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
        matrix_profile_receipt_path=matrix_receipt_path,
    )

    receipt = payload["production_matrix_profile_receipt"]
    assert receipt["status"] == "mismatch"
    assert receipt["verified"] is False
    assert "matrix_rows_cover_required_grid" in receipt["failed_checks"]
    matrix_blockers = [
        blocker
        for blocker in payload["schedule_blockers"]
        if blocker["kind"] == "production_1b_matrix_profile_missing"
    ]
    assert len(matrix_blockers) == 1
    assert "matrix_rows_cover_required_grid" in matrix_blockers[0]["failed_checks"]
    assert matrix_blockers[0]["row_check_summary"] == receipt["row_check_summary"]
    assert matrix_blockers[0]["failed_rows_by_check"] == (
        receipt["failed_rows_by_check"]
    )
    assert payload["acceptance_gate"]["current_matrix_profile_verified"] is False


def _complete_matrix_profile_rows(
    *,
    cppmega_sha: str = "abc123",
) -> list[dict[str, Any]]:
    return [
        {
            "dtype_route": dtype_route,
            "optimizer": optimizer,
            "status": "ok",
            "cppmega_sha": cppmega_sha,
            "path_b_status": "ok",
            "path_b_tok_sec": 100.0,
            "path_b_peak_memory_gb": 10.0,
            "path_c_warm_status": "ok",
            "path_c_warm_tok_sec": 125.0,
            "path_c_peak_memory_gb": 9.5,
            "path_c_warm_cache_hit": True,
            "path_c_cold_cache_hit": False,
            "profiling_trace_path": (
                f"reports/profiling/{dtype_route}_{optimizer}_path_c.json"
            ),
        }
        for dtype_route in m04_train_step.MATRIX_DTYPE_ROUTES
        for optimizer in m04_train_step.MATRIX_OPTIMIZERS
    ]


def _path_matrix_profile_rows(
    *,
    cppmega_sha: str = "abc123",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in ("bf16", "fp8"):
        for optimizer in (
            *m04_train_step.MATRIX_OPTIMIZERS,
            "muon_int8",
        ):
            for path, cache_hit, tok_sec, peak_memory_gb in (
                ("path_b", None, 100.0, 10.0),
                ("path_c_cold", False, 110.0, 9.75),
                ("path_c_warm", True, 125.0, 9.5),
            ):
                rows.append(
                    {
                        "dtype": dtype,
                        "optimizer": optimizer,
                        "path": path,
                        "status": "ok",
                        "pass_fail_reason": "ok",
                        "cppmega_sha": cppmega_sha,
                        "tok_sec": tok_sec,
                        "peak_memory_gb": peak_memory_gb,
                        "cache_hit": cache_hit,
                    }
                )
    return rows


def test_path_c_fusion_matrix_profile_receipt_derives_verified_grid_from_report(
    tmp_path: Path,
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env
    baseline_payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )
    schedule = baseline_payload["production_schedule"]

    receipt = m04_train_step.path_c_fusion_matrix_profile_receipt_from_report(
        {
            "software": {"cppmega_sha": "abc123"},
            "results": _complete_matrix_profile_rows(cppmega_sha="abc123"),
        },
        schedule_id=schedule["schedule_id"],
        schedule_name=schedule["schedule_name"],
    )

    assert receipt["status"] == "ok"
    assert receipt["full_1b_matrix_captured"] is True
    assert receipt["single_cppmega_commit"] is True
    assert receipt["cppmega_sha"] == "abc123"
    assert receipt["profiling_traces_captured"] is True
    assert receipt["memory_non_regression_ok"] is True
    assert receipt["cache_receipts_captured"] is True
    assert receipt["path_b_baselines_clean"] is True
    assert receipt["path_c_default_gate_rows_passed"] is True
    assert receipt["path_c_peak_memory_non_regression"] is True
    assert receipt["path_c_warm_cache_hit_observed"] is True
    assert len(receipt["matrix_rows"]) == (
        len(m04_train_step.MATRIX_DTYPE_ROUTES)
        * len(m04_train_step.MATRIX_OPTIMIZERS)
    )
    row_count = len(m04_train_step.MATRIX_DTYPE_ROUTES) * len(
        m04_train_step.MATRIX_OPTIMIZERS
    )
    assert receipt["row_check_summary"] == {
        "total_rows": row_count,
        "row_status_ok": row_count,
        "path_b_baseline_clean": row_count,
        "path_c_default_gate_passed": row_count,
        "path_c_peak_memory_non_regression": row_count,
        "path_c_warm_cache_hit_observed": row_count,
        "path_c_cold_cache_miss_observed": row_count,
        "profiling_trace_captured": row_count,
    }
    assert receipt["failed_rows_by_check"] == {}

    receipt_path = tmp_path / "derived_path_c_fusion_matrix_profile.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
        matrix_profile_receipt_path=receipt_path,
    )

    assert payload["production_matrix_profile_receipt"]["status"] == "verified"
    assert payload["production_matrix_profile_receipt"]["row_check_summary"] == (
        receipt["row_check_summary"]
    )
    assert payload["production_matrix_profile_receipt"]["failed_rows_by_check"] == {}
    assert payload["acceptance_gate"]["current_matrix_profile_verified"] is True


def test_path_c_fusion_matrix_profile_receipt_derives_grid_from_path_matrix_report(
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env
    baseline_payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )
    schedule = baseline_payload["production_schedule"]

    receipt = m04_train_step.path_c_fusion_matrix_profile_receipt_from_report(
        {
            "software": {"cppmega_sha": "abc123"},
            "results": _path_matrix_profile_rows(cppmega_sha="abc123"),
        },
        schedule_id=schedule["schedule_id"],
        schedule_name=schedule["schedule_name"],
    )

    assert receipt["status"] == "mismatch"
    assert receipt["single_cppmega_commit"] is True
    assert receipt["full_1b_matrix_captured"] is True
    assert receipt["profiling_traces_captured"] is False
    assert "profiling_traces_captured" in receipt["failed_checks"]
    assert len(receipt["matrix_rows"]) == (
        len(m04_train_step.MATRIX_DTYPE_ROUTES)
        * len(m04_train_step.MATRIX_OPTIMIZERS)
    )
    assert receipt["missing_matrix_rows"] == []
    row_count = len(m04_train_step.MATRIX_DTYPE_ROUTES) * len(
        m04_train_step.MATRIX_OPTIMIZERS
    )
    assert receipt["row_check_summary"] == {
        "total_rows": row_count,
        "row_status_ok": row_count,
        "path_b_baseline_clean": row_count,
        "path_c_default_gate_passed": row_count,
        "path_c_peak_memory_non_regression": row_count,
        "path_c_warm_cache_hit_observed": row_count,
        "path_c_cold_cache_miss_observed": row_count,
        "profiling_trace_captured": 0,
    }
    assert receipt["failed_rows_by_check"] == {
        "profiling_trace_captured": [
            f"{dtype_route}:{optimizer}"
            for dtype_route in m04_train_step.MATRIX_DTYPE_ROUTES
            for optimizer in m04_train_step.MATRIX_OPTIMIZERS
        ],
    }


def test_path_c_fusion_matrix_profile_receipt_counts_per_row_check_failures(
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env
    baseline_payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )
    schedule = baseline_payload["production_schedule"]
    rows = _complete_matrix_profile_rows(cppmega_sha="abc123")
    rows[0]["path_c_peak_memory_gb"] = 10.25
    rows[1]["path_c_warm_tok_sec"] = 90.0
    rows[2].pop("profiling_trace_path")

    receipt = m04_train_step.path_c_fusion_matrix_profile_receipt_from_report(
        {
            "software": {"cppmega_sha": "abc123"},
            "results": rows,
        },
        schedule_id=schedule["schedule_id"],
        schedule_name=schedule["schedule_name"],
    )

    row_count = len(m04_train_step.MATRIX_DTYPE_ROUTES) * len(
        m04_train_step.MATRIX_OPTIMIZERS
    )
    assert receipt["status"] == "mismatch"
    assert receipt["row_check_summary"] == {
        "total_rows": row_count,
        "row_status_ok": row_count,
        "path_b_baseline_clean": row_count,
        "path_c_default_gate_passed": row_count - 1,
        "path_c_peak_memory_non_regression": row_count - 1,
        "path_c_warm_cache_hit_observed": row_count,
        "path_c_cold_cache_miss_observed": row_count,
        "profiling_trace_captured": row_count - 1,
    }
    assert receipt["failed_rows_by_check"] == {
        "path_c_default_gate_passed": ["bf16:muon"],
        "path_c_peak_memory_non_regression": ["bf16:adamw"],
        "profiling_trace_captured": ["bf16:muon_adamw"],
    }


def test_path_c_fusion_matrix_profile_receipt_rejects_mixed_commits_and_peak_regression(
    path_c_fusion_force_env: None,
) -> None:
    del path_c_fusion_force_env
    baseline_payload = m04_train_step.path_c_fusion_payload(
        compile_receipt_path=PRODUCTION_FUSION_COMPILE_RECEIPT,
    )
    schedule = baseline_payload["production_schedule"]
    rows = _complete_matrix_profile_rows(cppmega_sha="abc123")
    rows[0]["cppmega_sha"] = "def456"
    rows[1]["path_c_peak_memory_gb"] = 10.25

    receipt = m04_train_step.path_c_fusion_matrix_profile_receipt_from_report(
        {
            "software": {"cppmega_sha": "abc123"},
            "results": rows,
        },
        schedule_id=schedule["schedule_id"],
        schedule_name=schedule["schedule_name"],
    )

    assert receipt["status"] == "mismatch"
    assert receipt["single_cppmega_commit"] is False
    assert receipt["memory_non_regression_ok"] is False
    assert receipt["path_c_peak_memory_non_regression"] is False
    assert "single_cppmega_commit" in receipt["failed_checks"]
    assert "path_c_peak_memory_non_regression" in receipt["failed_checks"]


def test_fp8_path_c_local_gb10_profile_uses_model_factory_dsa_producer() -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--dtype",
            "fp8_path_c",
        ]
    )

    producer = m04_train_step.sparse_mla_fp8_producer_payload(args)
    producer_gate = m04_train_step.fp8_path_c_producer_gate_payload(args)

    assert producer["configured"] is True
    assert producer["route_source"] == (
        "cppmega_mlx.recipes.model_factory.local_gb10_quarter"
    )
    assert producer["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer["dsa_a_layer_ranks"] == [1, 2, 3]
    assert producer["dsa_layer_numbers"] == [5, 9, 13]
    assert producer["prepared_buffers_configured"] is True
    assert producer["hidden_wrapper_quantization_allowed"] is False
    assert producer["kernel_boundary_quantization_allowed"] is False
    assert producer["reason"] is None
    assert producer_gate["required"] is True
    assert producer_gate["ok"] is True
    assert producer_gate["status"] == m04_train_step.FP8_PATH_C_NATIVE_PRODUCER_STATUS
    assert producer_gate["producer"] == producer


def assert_local_gb10_metadata_dry_run_contract(payload: dict[str, Any]) -> None:
    status = payload["status"]
    assert status in {"dry_run", "failed"}
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_mlx_m04_train_step"
    assert payload["local_only"] is True
    assert payload["gb10_training_correctness_claim"] is False
    assert payload["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["full_m0_4_acceptance_claim"] is False
    assert "blockers" not in payload
    assert_regression_report_matches_payload(payload)
    assert_m04_20step_matrix_plan(payload)
    assert {item["id"] for item in payload["acceptance_blockers"]} == {
        "cppmega-mlx-t8f.4.local_gb10_quarter_gate",
    }
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["mode"] == "metadata_only_no_forward_no_training"
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["optimizer_updated"] is False
    assert payload["training"]["losses"] == []
    assert payload["training"]["optimizer"]["update_observed"] is False
    assert payload["training"]["optimizer"]["master_moment_evidence"]["skipped"] is True
    assert payload["model"]["source"] is None
    assert payload["model"]["name"] is None
    assert payload["model"]["observed_source"] is None
    assert payload["model"]["observed_name"] is None
    assert payload["model"]["required_source"] == REQUIRED_MODEL_SOURCE
    assert payload["model"]["required_name"] == "local_gb10_quarter"
    assert payload["model"]["requested_profile"] == "local_gb10_quarter"
    assert payload["model"]["profile"] is None
    assert payload["model"]["requested_profile_matches_required"] is True
    assert payload["model"]["profile_matches_required"] is False
    assert payload["model"]["metadata_only"] is True
    assert payload["model"]["forward_executed"] is False
    assert payload["model"]["training_executed"] is False
    assert payload["baseline_row"]["model"] == "metadata_only_no_observed_model"
    assert payload["baseline_row"]["tokens_per_second"] == 0.0
    assert payload["baseline_row"]["local_only"] is True
    assert payload["baseline_row"]["gb10_parity_claim"] is False
    gate = payload["acceptance_gate"]
    assert gate["required_model_profile"] == "local_gb10_quarter"
    assert gate["observed_model_name"] is None
    assert gate["observed_model_source"] is None
    assert gate["model_identity_ok"] is False
    assert gate["optimizer_update_ok"] is False
    assert gate["adamw_ok"] is False
    assert gate["full_target_dataset_100_step_completed"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "real_parquet_source_identity_ok",
        "target_parquet_path_ok",
        "dataset_name_ok",
        "dataset_format_ok",
        "model_identity_ok",
        "optimizer_update_ok",
        "loss_decrease_ok",
        "loss_fields_ok",
        "all_finite_ok",
    }.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))


def test_local_gb10_quarter_dry_run_is_metadata_only_preflight(
    tmp_path: Path,
) -> None:
    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    def fail_allocation_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("full local_gb10_quarter allocation must be opt-in")

    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(
        args,
        dry_run_payload_fn=fail_route,
        train_hybrid_tiny_fn=fail_route,
        allocation_probe_fn=fail_allocation_probe,
    )

    assert exit_code == 0
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["local_gb10_quarter_preflight"]["allocation_attempted"] is False
    assert payload["local_gb10_quarter_preflight"]["allocation_mode"] == (
        "allocation_free_preflight"
    )
    assert payload["acceptance_gate"]["local_gb10_quarter_preflight_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_require_loss_decrease_fails_closed(
    tmp_path: Path,
) -> None:
    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--require-loss-decrease",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(
        args,
        dry_run_payload_fn=fail_route,
        train_hybrid_tiny_fn=fail_route,
    )

    assert exit_code == 2
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["status"] == "failed"
    assert payload["training"]["loss_decreased"] is False
    assert payload["training"]["loss_decrease_required"] is True
    assert payload["training"]["loss_decrease_satisfied"] is False
    assert payload["acceptance_gate"]["loss_decrease_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_cli_writes_requested_output_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_local_gb10_metadata.json"
    baseline_before = (
        BASELINE_RECEIPT.read_text(encoding="utf-8")
        if BASELINE_RECEIPT.exists()
        else None
    )

    result = run_script(
        "--synthetic",
        "--model-profile",
        "local_gb10_quarter",
        "--dry-run-json",
        "--output",
        str(output),
        "--json",
    )

    payload = load_json_result(result)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert_local_gb10_metadata_dry_run_contract(payload)
    if baseline_before is None:
        assert not BASELINE_RECEIPT.exists()
    else:
        assert BASELINE_RECEIPT.read_text(encoding="utf-8") == baseline_before


def test_local_gb10_quarter_dry_run_records_non_default_optimizer_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_local_gb10_lion.json"
    result = run_script(
        "--synthetic",
        "--model-profile",
        "local_gb10_quarter",
        "--dry-run-json",
        "--optimizer",
        "lion",
        "--output",
        str(output),
        "--json",
    )

    payload = load_json_result(result)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["optimizer"] == {
        "requested": "lion",
        "key": "lion",
        "quant_scheme": "dynamic_int8_v1",
        "source": "cli",
    }
    optimizer = payload["training"]["optimizer"]
    assert optimizer["name"] == "Lion"
    assert optimizer["key"] == "lion"
    assert optimizer["class"] == "cppmega_mlx.training.optimizers.LionFP32Moments"
    assert optimizer["adamw"] is False
    assert optimizer["master_moment_evidence"]["skipped"] is True
    assert payload["acceptance_gate"]["observed_optimizer_name"] == "Lion"
    assert payload["acceptance_gate"]["optimizer_identity_ok"] is False
    assert payload["acceptance_gate"]["adamw_ok"] is False


def test_non_default_optimizer_is_blocked_outside_local_gb10_route(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_hybrid_lion.json"
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--optimizer",
            "lion",
            "--output",
            str(output),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "unsupported_optimizer_route"
    assert payload["workload"]["optimizer"]["key"] == "lion"
    assert payload["training"]["optimizer"]["name"] == "Lion"


def test_local_gb10_quarter_training_routes_to_injected_seam(
    tmp_path: Path,
) -> None:
    route_calls: list[tuple[m04_train_step.TrainHybridTinyConfig, Path]] = []

    def fail_train(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HybridTinyLM training route must not be called")

    def fake_local_gb10_route(
        _args: Any,
        *,
        config: m04_train_step.TrainHybridTinyConfig,
        data_path: Path,
    ) -> tuple[dict[str, Any], int]:
        route_calls.append((config, data_path))
        return (
            m04_train_step.blocked_receipt(
                _args,
                "unit-test local_gb10_quarter route called",
                "unit_test_route_called",
            ),
            2,
        )

    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(
        args,
        train_hybrid_tiny_fn=fail_train,
        local_gb10_route_fn=fake_local_gb10_route,
    )

    assert len(route_calls) == 1
    config, data_path = route_calls[0]
    assert config.model_profile == "local_gb10_quarter"
    assert config.grad_checkpoint is False
    assert config.data_format == "npz"
    assert data_path.suffix == ".npz"
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["full_m0_4_acceptance_claim"] is False
    assert payload["blockers"][0]["type"] == "unit_test_route_called"
    assert (
        payload["blockers"][0]["reason"] == "unit-test local_gb10_quarter route called"
    )
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["training"]["steps_completed"] == 0
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["model_identity_ok"] is False


def test_local_gb10_quarter_grad_checkpoint_routes_to_injected_seam(
    tmp_path: Path,
) -> None:
    route_calls: list[tuple[m04_train_step.TrainHybridTinyConfig, Path]] = []

    def fail_train(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HybridTinyLM training route must not be called")

    def fake_local_gb10_route(
        _args: Any,
        *,
        config: m04_train_step.TrainHybridTinyConfig,
        data_path: Path,
    ) -> tuple[dict[str, Any], int]:
        route_calls.append((config, data_path))
        return (
            m04_train_step.blocked_receipt(
                _args,
                "unit-test local_gb10_quarter grad-checkpoint route called",
                "unit_test_route_called",
            ),
            2,
        )

    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--grad-checkpoint",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(
        args,
        train_hybrid_tiny_fn=fail_train,
        local_gb10_route_fn=fake_local_gb10_route,
    )

    assert len(route_calls) == 1
    config, data_path = route_calls[0]
    assert config.model_profile == "local_gb10_quarter"
    assert config.grad_checkpoint is True
    assert config.data_format == "npz"
    assert data_path.suffix == ".npz"
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "unit_test_route_called"
    assert payload["blockers"][0]["reason"] == (
        "unit-test local_gb10_quarter grad-checkpoint route called"
    )
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_with_allocation_probe_is_preflight_only(
    tmp_path: Path,
) -> None:
    probe_called = False

    def fake_probe() -> dict[str, Any]:
        nonlocal probe_called
        probe_called = True
        return canonical_allocation_probe()

    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--probe-local-gb10-quarter-allocation",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(
        args,
        dry_run_payload_fn=fail_route,
        train_hybrid_tiny_fn=fail_route,
        allocation_probe_fn=fake_probe,
    )

    assert probe_called is True
    assert exit_code == 0
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is True
    preflight = payload["local_gb10_quarter_preflight"]
    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is True
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["forward_executed"] is False
    assert preflight["allocation_probe"]["training_executed"] is False
    assert preflight["ok"] is True
    assert payload["acceptance_gate"]["local_gb10_quarter_preflight_ok"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_allocation_probe_success_is_preflight_only(
    tmp_path: Path,
) -> None:
    def fake_probe() -> dict[str, Any]:
        return canonical_allocation_probe()

    args = m04_train_step.build_parser().parse_args(
        [
            "--probe-local-gb10-quarter-allocation",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    preflight = m04_train_step.local_gb10_quarter_preflight_from_args(
        args,
        allocation_probe_fn=fake_probe,
    )
    gate = local_gb10_gate(
        model_name="HybridTinyLM",
        grad_checkpoint=grad_checkpoint_identity(enabled=False),
        local_gb10_quarter_preflight=preflight,
    )

    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is True
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["forward_executed"] is False
    assert preflight["allocation_probe"]["training_executed"] is False
    assert preflight["ok"] is True
    assert preflight["blockers"] == []
    assert gate["local_gb10_quarter_preflight_ok"] is True
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "model_identity_ok",
        "grad_checkpoint_expectation_ok",
    }.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))


def test_local_gb10_allocation_probe_failure_fails_closed(
    tmp_path: Path,
) -> None:
    def fake_probe() -> dict[str, Any]:
        return canonical_allocation_probe(
            status="blocked",
            allocation_ready=False,
            memory_after={"active_memory_bytes": 0},
            error_type="RuntimeError",
            error="synthetic allocation failure",
        )

    args = m04_train_step.build_parser().parse_args(
        [
            "--probe-local-gb10-quarter-allocation",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    preflight = m04_train_step.local_gb10_quarter_preflight_from_args(
        args,
        allocation_probe_fn=fake_probe,
    )
    gate = local_gb10_gate(local_gb10_quarter_preflight=preflight)

    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is False
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["error_type"] == "RuntimeError"
    assert preflight["allocation_probe"]["error"] == "synthetic allocation failure"
    assert preflight["ok"] is False
    assert preflight["blockers"] == ["allocation_ready"]
    assert gate["local_gb10_quarter_preflight_ok"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert (
        "local_gb10_quarter_preflight_ok"
        in (gate["full_local_gb10_quarter_gate_blockers"])
    )


def test_applied_memory_limit_api_path_preserves_actual_fallback_path() -> None:
    payload = {
        "applied": True,
        "metal_limit_api_path": "mx.set_memory_limit",
    }

    assert applied_memory_limit_api_path_from_payload(payload) == "mx.set_memory_limit"


class _Bf16Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.ones((2, 2), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sum(x @ self.weight)


def _run_bf16_probe_update(
    optimizer: optim.Optimizer,
) -> tuple[_Bf16Probe, dict[str, str]]:
    model = _Bf16Probe()

    def loss_fn(probe: _Bf16Probe, x: mx.array) -> mx.array:
        return probe(x)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, mx.ones((2, 2), dtype=mx.bfloat16))
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return model, collect_adamw_moment_dtypes(optimizer.state)


@pytest.mark.parametrize(
    ("optimizer_key", "expected_key", "expected_name", "quantized"),
    [
        ("adamw", "adamw", "AdamW", False),
        ("muon_adamw", "muon_adamw", "MuonAdamW", False),
        ("nam56r", "muon_adamw", "MuonAdamW", False),
        ("lion", "lion", "Lion", False),
        ("adam8bit", "adam8bit", "Adam8bit", True),
        ("lion8bit", "lion8bit", "Lion8bit", True),
        ("int8", "int8", "MuonAdamWInt8", True),
    ],
)
def test_local_gb10_optimizer_selector_initializes_supported_variants(
    optimizer_key: str,
    expected_key: str,
    expected_name: str,
    quantized: bool,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--optimizer",
            optimizer_key,
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=GB10_SAMPLE)
    model = _Bf16Probe()

    optimizer = m04_train_step.make_local_gb10_optimizer(
        args,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    identity = m04_train_step.optimizer_identity_for_selected_optimizer(
        args,
        config,
        optimizer,
        model,
        optimizer_updated=True,
    )

    assert identity["key"] == expected_key
    assert identity["name"] == expected_name
    assert identity["quantized_state"] is quantized
    assert identity["learning_rate"] == config.learning_rate
    assert identity["weight_decay"] == config.weight_decay
    assert identity["variant"]["requested"] == optimizer_key
    expected_quant_scheme = None if optimizer_key == "adamw" else "dynamic_int8_v1"
    assert identity["variant"]["quant_scheme"] == expected_quant_scheme
    if optimizer_key == "adamw":
        assert identity["adamw"] is True
        assert identity["name_matches_required"] is True
        assert identity["master_moment_dtype_ok"] is True
    else:
        assert identity["adamw"] is False
        assert identity["name_matches_required"] is False
        assert identity["master_moment_dtype_ok"] is False
        assert identity["state_evidence"]["state_dtype_breakdown_bytes"]


def test_stock_mlx_adamw_uses_bf16_moments_for_bf16_params() -> None:
    model, moment_dtypes = _run_bf16_probe_update(
        optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    )

    assert dtype_name(model.weight) == "bfloat16"
    assert moment_dtypes == {
        "weight/m": "bfloat16",
        "weight/v": "bfloat16",
    }


def test_repo_local_adamw_keeps_bf16_params_with_fp32_moments() -> None:
    optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.0)
    model, moment_dtypes = _run_bf16_probe_update(optimizer)

    assert isinstance(optimizer, AdamWFP32Moments)
    assert dtype_name(model.weight) == "bfloat16"
    assert moment_dtypes == {
        "weight/m": "float32",
        "weight/v": "float32",
    }


def test_repo_local_adamw_weight_decay_preserves_fp32_moments() -> None:
    optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.1)
    model, moment_dtypes = _run_bf16_probe_update(optimizer)

    assert isinstance(optimizer, AdamWFP32Moments)
    assert dtype_name(model.weight) == "bfloat16"
    assert bool(mx.all(mx.isfinite(model.weight)).item())
    assert moment_dtypes == {
        "weight/m": "float32",
        "weight/v": "float32",
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"applied": False, "metal_limit_api_path": "mx.set_memory_limit"},
        {"applied": True},
        {"applied": True, "metal_limit_api_path": ""},
    ],
)
def test_applied_memory_limit_api_path_requires_applied_recorded_path(
    payload: Any,
) -> None:
    assert applied_memory_limit_api_path_from_payload(payload) is None


def test_real_parquet_dry_run_uses_gb10_sample_receipt(tmp_path: Path) -> None:
    if not GB10_SAMPLE.exists():
        pytest.skip(f"GB10 parquet sample is not present: {GB10_SAMPLE}")

    sample_path = tmp_path / "clang_semantic_4k_v10_head.parquet"
    copy_real_parquet_head(GB10_SAMPLE, sample_path)
    output = tmp_path / "m04_train_step.json"
    result = run_script(
        "--data-path",
        str(sample_path),
        "--dry-run-json",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        "64",
        "--hidden-size",
        "8",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--output",
        str(output),
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "dry_run"
    assert payload["workload"]["synthetic"] is False
    assert payload["workload"]["data_format"] == "parquet"
    assert payload["workload"]["data_path"] == str(sample_path)
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["full_target_dataset_blocker"]
    assert payload["dataset"]["metadata"]["source_format"] == "parquet"
    assert payload["dataset"]["dataset_receipt"]["source_dataset_name"] == (
        "clang_semantic_4k_v10"
    )
    assert payload["training"]["steps_completed"] == 0


def target_dataset_receipt(
    *,
    source_path: str = TARGET_PARQUET,
    source_format: str = "parquet",
    source_dataset_name: str = "clang_semantic_4k_v10",
) -> dict[str, Any]:
    return {
        "path": source_path,
        "dataset_receipt": {
            "source_path": source_path,
            "source_format": source_format,
            "source_dataset_name": source_dataset_name,
        },
        "metadata": {"source_format": source_format},
    }


def adamw_moment_evidence(*, moment_dtype: str = "float32") -> dict[str, Any]:
    return {
        "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "observed_parameter_dtype": REQUIRED_DTYPE,
        "observed_moment_dtypes": {
            "weight/m": moment_dtype,
            "weight/v": moment_dtype,
        },
        "optimizer_class": ADAMW_FP32_MOMENTS_CLASS,
        "optimizer_base_class": ADAMW_BASE_CLASS,
        "state_keys": ["learning_rate", "step", "weight"],
        "ok": moment_dtype == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
    }


def adamw_identity(
    *,
    name: str = "AdamW",
    updated: bool = True,
    moment_dtype: str = "float32",
) -> dict[str, Any]:
    master_moment_evidence = adamw_moment_evidence(moment_dtype=moment_dtype)
    return {
        **OBSERVED_OPTIMIZER_IDENTITY,
        "name": name,
        "required_name": "AdamW",
        "name_matches_required": name == "AdamW",
        "adamw": name == "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "update_observed": updated,
        "required_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "master_moment_evidence": master_moment_evidence,
        "master_moment_dtype_ok": master_moment_evidence["ok"],
    }


def grad_checkpoint_identity(*, enabled: bool = True) -> dict[str, Any]:
    return {
        "required": True,
        "observed_enabled": enabled,
        "source": GRAD_CHECKPOINT_EXPECTATION["source"],
        "expectation_satisfied": enabled,
    }


def m4_device_metadata(*, device_name: str = "Apple M4 Max") -> dict[str, Any]:
    return {
        "machine": "arm64",
        "metal_available": True,
        "platform": "macOS-26.4.1-arm64-arm-64bit-Mach-O",
        "mlx_device_info": {
            "device_name": device_name,
            "memory_size": 137438953472,
        },
    }


def local_gb10_model_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "profile": "local_gb10_quarter",
        **REQUIRED_MODEL_GEOMETRY,
    }
    config["mtp"] = dict(REQUIRED_MODEL_GEOMETRY["mtp"])
    config.update(overrides)
    return config


def resolved_local_gb10_preflight(**overrides: Any) -> dict[str, Any]:
    allocation_probe = canonical_allocation_probe()
    preflight = local_gb10_quarter_preflight_payload(
        allocation_attempted=True,
        allocation_ready=True,
        allocation_mode="full_profile_allocation_probe",
        allocation_probe=allocation_probe,
    )
    preflight["tokenizer_contract"] = {
        **preflight["tokenizer_contract"],
        "resolved": True,
    }
    preflight["ok"] = True
    preflight["blockers"] = []
    preflight.update(overrides)
    return preflight


def local_gb10_gate(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_path": target_dataset_path(),
        "data_format": "parquet",
        "dtype": REQUIRED_DTYPE,
        "dataset": target_dataset_receipt(),
        "steps_requested": 100,
        "steps_completed": 100,
        "loss_decreased": True,
        "all_finite": True,
        "optimizer_updated": True,
        "model_name": "local_gb10_quarter",
        "model_source": REQUIRED_MODEL_SOURCE,
        "model_config": local_gb10_model_config(),
        "optimizer": adamw_identity(),
        "grad_checkpoint": grad_checkpoint_identity(),
        "device": m4_device_metadata(),
        "local_gb10_quarter_preflight": resolved_local_gb10_preflight(),
    }
    payload.update(overrides)
    return acceptance_gate_payload(**payload)


def test_acceptance_gate_accepts_complete_local_gb10_quarter_evidence() -> None:
    gate = local_gb10_gate()

    assert gate["real_parquet_source_identity"]["ok"] is True
    assert gate["full_target_dataset_100_step_completed"] is True
    assert gate["dtype_ok"] is True
    assert gate["local_gb10_quarter_preflight_ok"] is True
    assert gate["local_gb10_quarter_preflight"]["source"] == REQUIRED_MODEL_SOURCE
    assert gate["local_gb10_quarter_preflight"]["required_geometry"] == (
        REQUIRED_MODEL_GEOMETRY
    )
    assert gate["local_gb10_quarter_preflight"]["profile_geometry"] == (
        REQUIRED_MODEL_GEOMETRY
    )
    assert gate["model_identity_ok"] is True
    assert gate["optimizer_identity_ok"] is True
    assert gate["required_adamw_master_moment_dtype"] == "float32"
    assert gate["observed_adamw_master_moment_dtypes"] == {
        "weight/m": "float32",
        "weight/v": "float32",
    }
    assert gate["fp32_adamw_master_moments_ok"] is True
    assert gate["adamw_ok"] is True
    assert gate["grad_checkpoint_expectation_ok"] is True
    assert gate["m4_runtime_metadata_ok"] is True
    assert gate["full_local_gb10_quarter_gate_completed"] is True
    assert gate["full_local_gb10_quarter_gate_blockers"] == []


@pytest.mark.parametrize(
    ("overrides", "failed_checks"),
    [
        (
            {
                "dataset": target_dataset_receipt(
                    source_path="/tmp/fake/clang_semantic_4k_v10/val_00000.parquet"
                )
            },
            {"real_parquet_source_identity_ok", "target_parquet_path_ok"},
        ),
        (
            {"dataset": target_dataset_receipt(source_dataset_name="not_clang")},
            {"real_parquet_source_identity_ok", "dataset_name_ok"},
        ),
        (
            {
                "data_format": "npz",
                "dataset": target_dataset_receipt(source_format="npz"),
            },
            {"real_parquet_source_identity_ok", "dataset_format_ok"},
        ),
        (
            {"dtype": "float32"},
            {"dtype_ok"},
        ),
        (
            {
                "model_name": "HybridTinyLM",
                "model_config": local_gb10_model_config(),
            },
            {"model_identity_ok"},
        ),
        (
            {
                "model_name": "local_gb10_quarter",
                "model_config": local_gb10_model_config(profile="HybridTinyLM"),
            },
            {"model_identity_ok"},
        ),
        (
            {"model_source": "fake.local_gb10_quarter"},
            {"model_identity_ok"},
        ),
        (
            {"model_config": local_gb10_model_config(hidden_size=16)},
            {"model_identity_ok"},
        ),
        (
            {
                "model_config": local_gb10_model_config(
                    mtp={"depth": 1, "beta": 0.6, "loss_weight": 0.3}
                )
            },
            {"model_identity_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    source="fake.local_gb10_quarter"
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    profile_geometry={
                        **REQUIRED_MODEL_GEOMETRY,
                        "hidden_size": 16,
                    }
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    tokenizer_contract={
                        **resolved_local_gb10_preflight()["tokenizer_contract"],
                        "resolved": False,
                    }
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {"optimizer": adamw_identity(name="SGD")},
            {"optimizer_identity_ok", "adamw_ok"},
        ),
        (
            {"optimizer": {**adamw_identity(), "class": "fake.AdamW"}},
            {"optimizer_identity_ok", "adamw_ok"},
        ),
        (
            {"optimizer": adamw_identity(moment_dtype="bfloat16")},
            {"fp32_adamw_master_moments_ok"},
        ),
        (
            {"grad_checkpoint": grad_checkpoint_identity(enabled=False)},
            {"grad_checkpoint_expectation_ok"},
        ),
        (
            {
                "grad_checkpoint": {
                    **grad_checkpoint_identity(),
                    "source": "unit-test-local-gb10-quarter",
                }
            },
            {"grad_checkpoint_expectation_ok"},
        ),
        (
            {"device": m4_device_metadata(device_name="Apple M3 Max")},
            {"m4_runtime_metadata_ok"},
        ),
        (
            {"steps_completed": 99},
            {"step_count_ok"},
        ),
        (
            {"steps_requested": 101, "steps_completed": 100},
            {"step_count_ok"},
        ),
        (
            {"loss_decreased": False},
            {"loss_decrease_ok", "loss_fields_ok"},
        ),
        (
            {"all_finite": False},
            {"all_finite_ok", "loss_fields_ok"},
        ),
    ],
)
def test_acceptance_gate_fail_closes_on_fake_or_incomplete_evidence(
    overrides: dict[str, Any],
    failed_checks: set[str],
) -> None:
    gate = local_gb10_gate(**overrides)

    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert failed_checks.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))
    if any(
        check in failed_checks
        for check in (
            "real_parquet_source_identity_ok",
            "target_parquet_path_ok",
            "dataset_name_ok",
            "dataset_format_ok",
            "dtype_ok",
            "step_count_ok",
            "loss_fields_ok",
            "optimizer_update_ok",
        )
    ):
        assert gate["full_target_dataset_100_step_completed"] is False


@pytest.mark.parametrize(
    "preflight_overrides",
    [
        {"allocation_probe": None},
        {"allocation_probe": canonical_allocation_probe(status="blocked")},
        {"allocation_probe": canonical_allocation_probe(allocation_ready=False)},
        {
            "allocation_probe": canonical_allocation_probe(
                source="fake.local_gb10_quarter"
            )
        },
        {"allocation_probe": canonical_allocation_probe(source=None)},
        {
            "allocation_probe": canonical_allocation_probe(
                allocation_mode="caller_supplied_allocation_evidence"
            )
        },
        {"allocation_probe": canonical_allocation_probe(allocation_mode=None)},
        {"allocation_probe": canonical_allocation_probe(profile_name="HybridTinyLM")},
        {"allocation_probe": canonical_allocation_probe(model_class="FakeTinyLM")},
        {"allocation_probe": canonical_allocation_probe(model_class=None)},
        {"allocation_probe": canonical_allocation_probe(eval_scope="forward_smoke")},
        {"allocation_probe": canonical_allocation_probe(forward_executed=True)},
        {"allocation_probe": canonical_allocation_probe(training_executed=True)},
        {
            "allocation_probe": canonical_allocation_probe(
                geometry_matches_required=False
            )
        },
        {
            "allocation_probe": canonical_allocation_probe(
                required_geometry={**REQUIRED_MODEL_GEOMETRY, "hidden_size": 16}
            )
        },
        {
            "allocation_probe": canonical_allocation_probe(
                profile_geometry={**REQUIRED_MODEL_GEOMETRY, "hidden_size": 16}
            )
        },
        {
            "allocation_mode": "caller_supplied_allocation_evidence",
            "allocation_probe": canonical_allocation_probe(),
        },
    ],
)
def test_acceptance_gate_requires_canonical_allocation_probe(
    preflight_overrides: dict[str, Any],
) -> None:
    gate = local_gb10_gate(
        local_gb10_quarter_preflight=resolved_local_gb10_preflight(
            **preflight_overrides
        )
    )

    assert gate["local_gb10_quarter_preflight_ok"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert (
        "local_gb10_quarter_preflight_ok"
        in (gate["full_local_gb10_quarter_gate_blockers"])
    )


def _fp8_path_c_gate_args(
    *,
    dtype: str = "fp8_path_c",
    seq_len: int = 512,
    use_direct_chain: bool = False,
    use_fused_train_block: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        dtype=dtype,
        seq_len=seq_len,
        use_path_c_direct_chain_runtime=use_direct_chain,
        use_path_c_fused_train_block_runtime=use_fused_train_block,
    )


def test_fp8_path_c_gate_default_threshold_is_2048() -> None:
    with temporary_env({m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None}):
        assert (
            m04_train_step.fp8_path_c_max_seq_len()
            == m04_train_step.FP8_PATH_C_LONG_SEQ_GATE_DEFAULT
            == 2048
        )


def test_fp8_path_c_auto_route_selectable_below_threshold() -> None:
    args = _fp8_path_c_gate_args(seq_len=512)
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            m04_train_step.FP8_PATH_C_EXPLICIT_OVERRIDE_ENV: None,
        }
    ):
        gate = m04_train_step.fp8_path_c_long_seq_gate(args)
        assert gate["gated"] is False
        assert m04_train_step.fp8_path_c_route_requested(args) is True
        assert m04_train_step.fp8_path_b_route_requested(args) is False


def test_fp8_path_c_auto_route_gated_at_and_above_threshold() -> None:
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            m04_train_step.FP8_PATH_C_EXPLICIT_OVERRIDE_ENV: None,
        }
    ):
        for seq_len in (2048, 4096):
            args = _fp8_path_c_gate_args(seq_len=seq_len)
            gate = m04_train_step.fp8_path_c_long_seq_gate(args)
            assert gate["gated"] is True
            assert (
                gate["status"]
                == m04_train_step.FP8_PATH_C_LONG_SEQ_GATE_STATUS
            )
            assert gate["fallback_dtype"] == m04_train_step.FP8_PATH_B_DTYPE
            assert gate["reason"] and "Path C" in gate["reason"]
            # Effective route fails closed onto the Path B baseline.
            assert m04_train_step.fp8_path_c_route_requested(args) is False
            assert m04_train_step.fp8_path_b_route_requested(args) is True


def test_fp8_path_c_explicit_runtime_override_bypasses_gate() -> None:
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            m04_train_step.FP8_PATH_C_EXPLICIT_OVERRIDE_ENV: None,
        }
    ):
        direct = _fp8_path_c_gate_args(seq_len=4096, use_direct_chain=True)
        assert m04_train_step.fp8_path_c_long_seq_gate(direct)["gated"] is False
        assert m04_train_step.fp8_path_c_route_requested(direct) is True

        fused = _fp8_path_c_gate_args(seq_len=4096, use_fused_train_block=True)
        assert m04_train_step.fp8_path_c_long_seq_gate(fused)["gated"] is False
        assert m04_train_step.fp8_path_c_route_requested(fused) is True


def test_fp8_path_c_override_env_bypasses_gate_for_flagless_config() -> None:
    # A flag-less config (TrainHybridTinyConfig-style) still honors the override
    # that run_receipt mirrors into the environment.
    args = _fp8_path_c_gate_args(seq_len=4096)
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            m04_train_step.FP8_PATH_C_EXPLICIT_OVERRIDE_ENV: "1",
        }
    ):
        assert m04_train_step.fp8_path_c_long_seq_gate(args)["gated"] is False
        assert m04_train_step.fp8_path_c_route_requested(args) is True


def test_fp8_path_c_gate_does_not_touch_bf16_path_c() -> None:
    # bf16 Path C (CPPMEGA_KERNEL_PATH=path_c, dtype != fp8_path_c) is untouched.
    args = _fp8_path_c_gate_args(dtype="bfloat16", seq_len=4096)
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            "CPPMEGA_KERNEL_PATH": "path_c",
            "CPPMEGA_KERNEL_PATH__SPARSE_MLA": "path_c",
        }
    ):
        assert m04_train_step.fp8_path_c_long_seq_gate(args)["gated"] is False
        # bf16 Path C is still requested via the kernel policy env.
        assert m04_train_step.path_c_kernel_policy_requested() is True
        assert m04_train_step.path_c_training_route_requested(args) is True


def test_fp8_path_c_gate_does_not_touch_fp8_path_b() -> None:
    args = _fp8_path_c_gate_args(dtype="fp8_path_b", seq_len=4096)
    with temporary_env({m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None}):
        assert m04_train_step.fp8_path_c_long_seq_gate(args)["gated"] is False
        assert m04_train_step.fp8_path_b_route_requested(args) is True


def test_fp8_path_c_gate_threshold_is_configurable() -> None:
    # Custom threshold via env.
    with temporary_env({m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: "1024"}):
        assert m04_train_step.fp8_path_c_max_seq_len() == 1024
        gated = _fp8_path_c_gate_args(seq_len=1024)
        assert m04_train_step.fp8_path_c_route_requested(gated) is False
        ok = _fp8_path_c_gate_args(seq_len=1023)
        assert m04_train_step.fp8_path_c_route_requested(ok) is True
    # Disabled via threshold 0.
    with temporary_env({m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: "0"}):
        assert m04_train_step.fp8_path_c_max_seq_len() == 0
        args = _fp8_path_c_gate_args(seq_len=4096)
        assert m04_train_step.fp8_path_c_long_seq_gate(args)["gated"] is False
        assert m04_train_step.fp8_path_c_route_requested(args) is True


def test_fp8_path_c_gate_records_receipt_note_on_precision_route() -> None:
    args = _fp8_path_c_gate_args(seq_len=2048)
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            m04_train_step.FP8_PATH_C_EXPLICIT_OVERRIDE_ENV: None,
        }
    ):
        payload = m04_train_step.precision_route_payload(args)
    assert payload["kind"] == "fp8_path_c_auto_long_seq_gated_to_path_b"
    assert payload["status"] == m04_train_step.FP8_PATH_C_LONG_SEQ_GATE_STATUS
    assert payload["path_c_used"] is False
    assert payload["fp8_path_c_long_seq_gate"]["gated"] is True


def test_fp8_path_c_gate_kernel_policy_env_falls_back_to_path_b() -> None:
    args = _fp8_path_c_gate_args(seq_len=2048)
    route_env = m04_train_step.SPARSE_MLA_FP8_ROUTE_ENV
    with temporary_env(
        {
            m04_train_step.FP8_PATH_C_MAX_SEQ_LEN_ENV: None,
            m04_train_step.FP8_PATH_C_EXPLICIT_OVERRIDE_ENV: None,
            route_env: "path_c",
        }
    ):
        with m04_train_step.fp8_path_b_kernel_policy(
            args
        ), m04_train_step.fp8_path_c_kernel_policy(args):
            # Gated AUTO Path C demotes the live Sparse-MLA route to Path B.
            assert os.environ[route_env] == "path_b"
        # Restored on exit.
        assert os.environ[route_env] == "path_c"
