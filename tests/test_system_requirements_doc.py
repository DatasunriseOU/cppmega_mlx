from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "system_requirements.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _normalized_doc() -> str:
    return " ".join(_doc().split())


def test_system_requirements_doc_scopes_local_mac_readiness_fail_closed() -> None:
    doc = _doc()

    assert "# MLX Mac Local Training System Requirements" in doc
    assert "fail-closed readiness slice" in doc
    assert "report `NOT READY`" in doc
    assert "M4-vs-GB10 parity is not proven" in doc
    assert "Distributed Megatron parity is not claimed" in doc
    assert "Trainable Metal-kernel adoption is not claimed" in doc
    assert "Full NAM56R readiness is not claimed" in doc


def test_doc_requires_mlx_device_and_apple_silicon_receipt_fields() -> None:
    doc = _doc()

    required = (
        'importlib.metadata.version("mlx")',
        "platform.platform()",
        "platform.machine()",
        "mx.default_device()",
        "mx.metal.is_available()",
        "mx.device_info()",
        "device_name",
        "memory_size",
        "max_recommended_working_set_size",
        "architecture",
        "macOS on Apple Silicon (`arm64`)",
        "Unified RAM must fit",
        "CPU-only fallback",
        "`NOT READY` for a Mac training receipt",
    )
    for phrase in required:
        assert phrase in doc


def test_doc_requires_file_descriptor_memory_and_thermal_guardrails() -> None:
    doc = _doc()
    normalized = _normalized_doc()

    required = (
        "ulimit -n >= 65536",
        "resource.RLIMIT_NOFILE",
        "Do not silently continue with a lower limit",
        "cppmega_mlx/runtime/memory.py",
        "DEFAULT_WIRED_RATIO = 0.70",
        "DEFAULT_METAL_RATIO = 0.85",
        "mx.set_wired_limit(limit_bytes)",
        "mx.set_memory_limit(limit_bytes)",
        "mx.metal.set_memory_limit(limit_bytes)",
        "fail closed if `mx.set_wired_limit` is unavailable",
        "Do not raise `iogpu.wired_limit_mb` inside a training script",
        "powermetrics",
        "Thermal throttling invalidates performance receipts",
    )
    for phrase in required:
        assert phrase in doc

    assert "The helper is dry-run by default" in normalized
    assert "must not mutate process-global MLX limits unless the caller explicitly applies the plan" in normalized


def test_doc_keeps_distributed_and_jaccl_future_measured_only() -> None:
    doc = _doc()
    normalized = _normalized_doc()

    required = (
        "mlx.core.distributed",
        'mx.distributed.is_available("jaccl")',
        "future measured-only inputs",
        "does not prove that cppmega.mlx implements Megatron TP/PP/VPP/EP/SP",
        "mx.distributed.init(backend=...)",
        "mlx.launch",
        "Thunderbolt 5 RDMA",
        "ibv_devices",
        "fully connected mesh",
        "MLX_METAL_FAST_SYNCH",
        "not distributed Megatron parity",
        "all distributed training readiness is `NOT READY`",
    )
    for phrase in required:
        assert phrase in doc

    assert "JACCL remains a future backend candidate" in normalized


def test_doc_does_not_claim_forbidden_wave14_surfaces() -> None:
    doc = _doc()

    forbidden_overclaims = (
        "GB10 parity is proven",
        "M4 Max parity with GB10 is proven",
        "M4 Max matches GB10",
        "M4-only rows prove GB10 parity",
        "distributed Megatron parity is proven",
        "full distributed Megatron parity",
        "trainable Metal kernels are adopted",
        "Metal kernels are on the training path",
        "full NAM56R readiness is proven",
        "cppmega.mlx is NAM56R-ready",
        "JACCL proves Megatron parity",
    )
    for phrase in forbidden_overclaims:
        assert phrase not in doc
