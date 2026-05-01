#!/usr/bin/env python3
"""Compare matched M4 Max and GB10 benchmark rows without local-only parity claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, NamedTuple


MISSING = "<missing>"
COMPARE_PACKAGE_SCHEMA_VERSION = 1
COMPARE_PACKAGE_KIND = "cppmega.mlx.gb10_matched_compare_package"
COMPARE_REPORT_FILENAME = "compare_report.json"
MATCHED_COMPARISONS_FILENAME = "matched_comparisons.jsonl"
REFUSED_PAIRS_FILENAME = "refused_pairs.jsonl"
COMPARE_PACKAGE_MANIFEST_FILENAME = "manifest.json"

MATCH_FIELDS: tuple[str, ...] = (
    "profile",
    "route",
    "model_route",
    "route_plan",
    "backend_plan",
    "model_source",
    "dtype",
    "compile",
    "warmup_steps",
    "measured_steps",
    "batch_size",
    "seq_len",
    "vocab_size",
    "d_model",
    "n_heads",
    "n_layers",
    "mlp_dim",
    "learning_rate",
    "seed",
    "include_structure",
    "data_contract",
    "framework",
    "backend",
    "python_version",
    "mlx_version",
    "mlx_lm_version",
    "mlx_metal_version",
    "torch_version",
    "cuda_version",
    "driver_version",
)

REQUIRED_MATCH_FIELDS: tuple[str, ...] = (
    "model_route",
    "route_plan",
    "backend_plan",
    "model_source",
    "dtype",
    "compile",
    "warmup_steps",
    "measured_steps",
    "batch_size",
    "seq_len",
    "vocab_size",
    "d_model",
    "n_heads",
    "n_layers",
    "mlp_dim",
    "learning_rate",
    "seed",
    "include_structure",
    "data_contract",
    "framework",
    "backend",
)

MLX_REQUIRED_MATCH_FIELDS: tuple[str, ...] = (
    "mlx_version",
    "mlx_metal_version",
)

TORCH_REQUIRED_MATCH_FIELDS: tuple[str, ...] = (
    "torch_version",
    "device_name",
)

CUDA_REQUIRED_MATCH_FIELDS: tuple[str, ...] = (
    "cuda_version",
    "driver_version",
    "device_capability",
)

MATCHED_COMPARISON_KEY_REQUIREMENT = (
    "M4/GB10 ratios require explicit matched-key provenance on both rows: "
    "comparison_key.workload + comparison_key.software, "
    "bench_receipt.comparison_key, workload_key + software_key, or legacy "
    "matched_run.key with run_metadata.framework. The selected workload and "
    "software keys must be identical across rows, including extra fields. "
    "When modern comparison-key sources are present in the same row, they must "
    "also agree with each other; legacy matched_run provenance is used only "
    "when no modern comparison key exists."
)


class ComparisonKeySource(NamedTuple):
    name: str
    workload: dict[str, Any]
    software: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load bench_matrix JSON/JSONL rows from M4 Max and GB10, require "
            "matched workload and framework fields, then emit comparison ratios."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSON summary/object/array or NDJSON file. Use '-' for stdin.",
    )
    parser.add_argument(
        "--m4-label",
        default="M4 Max",
        help="Case-insensitive substring identifying M4 rows by hardware_label.",
    )
    parser.add_argument(
        "--gb10-label",
        default="GB10",
        help="Case-insensitive substring identifying GB10 rows by hardware_label.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Emit one comparison object per line instead of one summary object.",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help=(
            "Write a GB10 matched-run package with manifest.json, full report, "
            "matched comparison JSONL, and refusal JSONL artifacts."
        ),
    )
    return parser


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _iter_json_values(text: str, *, source: str) -> Iterable[Any]:
    stripped = text.strip()
    if not stripped:
        return
    try:
        yield json.loads(stripped)
        return
    except json.JSONDecodeError:
        pass

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_no}: invalid JSON row: {exc.msg}") from exc


def _iter_rows_from_value(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_rows_from_value(item)
        return
    if not isinstance(value, dict):
        return
    cases = value.get("cases")
    if isinstance(cases, list):
        for item in cases:
            yield from _iter_rows_from_value(item)
        return
    yield value


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for value in _iter_json_values(_read_text(path), source=path):
            rows.extend(_iter_rows_from_value(value))
    return rows


def _nested(row: dict[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first_present(row: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        value = _nested(row, *path)
        if value is not None:
            return value
    return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _scalar_value(value: Any) -> Any:
    if isinstance(value, dict):
        return None
    return value


def _sequence_label(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ".".join(str(part) for part in value)
    return value


def _gib_to_bytes(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value) * 1024**3)
    except (TypeError, ValueError):
        return None


def _freeze_key_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_key_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_key_value(item) for item in value)
    return value


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == MISSING


def _has_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _explicit_comparison_key_sources(row: dict[str, Any]) -> list[ComparisonKeySource]:
    sources: list[ComparisonKeySource] = []
    comparison_key = _dict_value(row.get("comparison_key"))
    comparison_workload = _dict_value(comparison_key.get("workload"))
    comparison_software = _dict_value(comparison_key.get("software"))
    if comparison_workload and comparison_software:
        sources.append(
            ComparisonKeySource("comparison_key", comparison_workload, comparison_software)
        )

    bench_receipt = _dict_value(row.get("bench_receipt"))
    receipt_comparison_key = _dict_value(bench_receipt.get("comparison_key"))
    receipt_workload = _dict_value(receipt_comparison_key.get("workload"))
    receipt_software = _dict_value(receipt_comparison_key.get("software"))
    if receipt_workload and receipt_software:
        sources.append(
            ComparisonKeySource(
                "bench_receipt.comparison_key",
                receipt_workload,
                receipt_software,
            )
        )

    workload_key = _dict_value(row.get("workload_key"))
    software_key = _dict_value(row.get("software_key"))
    if workload_key and software_key:
        sources.append(
            ComparisonKeySource("workload_key+software_key", workload_key, software_key)
        )
    if sources:
        return sources

    matched_key = _dict_value(
        _first_present(
            row,
            (
                ("matched_run_key",),
                ("matched_run", "key"),
                ("run_metadata", "matched_run", "key"),
            ),
        )
    )
    framework_key = _dict_value(_nested(row, "run_metadata", "framework"))
    if matched_key and _has_mapping(framework_key):
        sources.append(
            ComparisonKeySource(
                "matched_run+run_metadata.framework",
                matched_key,
                framework_key,
            )
        )

    return sources


def _comparison_key_conflicts(
    sources: list[ComparisonKeySource],
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    if len(sources) < 2:
        return conflicts
    first = sources[0]
    for source in sources[1:]:
        for section, first_value, source_value in (
            ("workload", first.workload, source.workload),
            ("software", first.software, source.software),
        ):
            if _freeze_key_value(first_value) == _freeze_key_value(source_value):
                continue
            conflicts.append(
                {
                    "section": section,
                    "left_source": first.name,
                    "right_source": source.name,
                    "left": first_value,
                    "right": source_value,
                }
            )
    return conflicts


def missing_matched_comparison_key(row: dict[str, Any]) -> bool:
    return (
        not row.get("matched_comparison_key_sources")
        or bool(row.get("matched_comparison_key_conflicts"))
    )


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    config = _dict_value(row.get("config"))
    args = _dict_value(row.get("args"))
    framework_metadata = _dict_value(row.get("framework"))
    raw_device = row.get("device")
    device = _dict_value(raw_device)
    memory = _dict_value(row.get("memory"))
    after_measured = _dict_value(memory.get("after_measured_steps"))
    bench_receipt = _dict_value(row.get("bench_receipt"))
    receipt_timing = _dict_value(bench_receipt.get("timing"))
    receipt_device = _dict_value(bench_receipt.get("device"))
    workload_key = _dict_value(row.get("workload_key"))
    software_key = _dict_value(row.get("software_key"))
    comparison_key = _dict_value(row.get("comparison_key"))
    comparison_workload = _dict_value(comparison_key.get("workload"))
    comparison_software = _dict_value(comparison_key.get("software"))
    receipt_comparison_key = _dict_value(bench_receipt.get("comparison_key"))
    receipt_comparison_workload = _dict_value(receipt_comparison_key.get("workload"))
    receipt_comparison_software = _dict_value(receipt_comparison_key.get("software"))
    receipt_workload = _dict_value(bench_receipt.get("workload"))
    receipt_software = _dict_value(bench_receipt.get("software"))
    explicit_comparison_key_sources = _explicit_comparison_key_sources(row)
    comparison_key_conflicts = _comparison_key_conflicts(explicit_comparison_key_sources)
    explicit_comparison_keys = (
        None if comparison_key_conflicts else explicit_comparison_key_sources[0]
        if explicit_comparison_key_sources
        else None
    )

    matched_key = _dict_value(
        _first_present(
            row,
            (
                ("matched_run_key",),
                ("matched_run", "key"),
                ("run_metadata", "matched_run", "key"),
            ),
        )
    )
    metadata_workload = _dict_value(_nested(row, "run_metadata", "workload"))
    metadata_framework = {
        **_dict_value(_nested(row, "run_metadata", "framework")),
        **framework_metadata,
    }
    workload_sources = (
        matched_key,
        metadata_workload,
        workload_key,
        comparison_workload,
        receipt_workload,
        receipt_comparison_workload,
    )
    software_sources = (
        software_key,
        comparison_software,
        receipt_software,
        receipt_comparison_software,
    )

    include_structure = _coalesce(
        row.get("include_structure") if "include_structure" in row else None,
        config.get("include_structure"),
        matched_key.get("include_structure"),
        metadata_workload.get("include_structure"),
        workload_key.get("include_structure"),
        comparison_workload.get("include_structure"),
        receipt_workload.get("include_structure"),
        receipt_comparison_workload.get("include_structure"),
    )
    data_contract = _first_present(
        row,
        (
            ("data_contract",),
            ("dataset",),
            ("data", "contract"),
            ("config", "data_contract"),
            ("config", "dataset"),
            ("matched_run_key", "data_contract"),
            ("matched_run", "key", "data_contract"),
            ("run_metadata", "matched_run", "key", "data_contract"),
            ("run_metadata", "workload", "data_contract"),
        ),
    )
    data_contract = _coalesce(
        data_contract,
        *(source.get("data_contract") for source in workload_sources),
    )
    if data_contract is None:
        data_contract = "synthetic_structure" if include_structure else "synthetic_plain"

    framework = _coalesce(
        _scalar_value(row.get("framework")),
        row.get("framework_name"),
        metadata_framework.get("framework"),
        metadata_framework.get("name"),
        *(source.get("framework") for source in software_sources),
        *(source.get("name") for source in software_sources),
        "mlx" if _coalesce(device.get("mlx"), metadata_framework.get("mlx")) is not None else None,
        "torch"
        if _coalesce(
            row.get("torch_version"),
            row.get("torch"),
            device.get("torch"),
            metadata_framework.get("torch"),
        )
        is not None
        else None,
    )
    backend = _coalesce(
        row.get("backend"),
        row.get("framework_backend"),
        row.get("execution_backend"),
        device.get("backend"),
        metadata_framework.get("backend"),
        args.get("backend"),
        *(source.get("backend") for source in software_sources),
        *(source.get("execution_backend") for source in software_sources),
        *(source.get("framework_backend") for source in software_sources),
        "cuda"
        if _coalesce(
            row.get("cuda_version"),
            row.get("cuda"),
            row.get("torch_cuda"),
            row.get("cuda_device"),
            row.get("cuda_cap"),
            row.get("cuda_capability"),
            device.get("cuda"),
            device.get("torch_cuda"),
            device.get("cuda_device"),
            device.get("cuda_cap"),
            device.get("cuda_capability"),
            metadata_framework.get("cuda"),
            metadata_framework.get("torch_cuda"),
            metadata_framework.get("cuda_device"),
            metadata_framework.get("cuda_cap"),
            metadata_framework.get("cuda_capability"),
        )
        is not None
        else None,
        "metal"
        if framework == "mlx"
        and _coalesce(
            device.get("mlx_metal"),
            device.get("metal"),
            metadata_framework.get("mlx_metal"),
            metadata_framework.get("metal"),
            *(source.get("mlx_metal") for source in software_sources),
            *(source.get("metal") for source in software_sources),
        )
        is not None
        else None,
        "mlx" if framework == "mlx" else None,
    )
    device_name = _coalesce(
        row.get("device_name"),
        row.get("cuda_device"),
        device.get("device_name"),
        device.get("name"),
        device.get("cuda_device"),
        _nested(row, "device", "mlx_device_info", "device_name"),
        metadata_framework.get("device_name"),
        metadata_framework.get("cuda_device"),
        _nested(row, "run_metadata", "framework", "mlx_device_info", "device_name"),
        *(source.get("device_name") for source in software_sources),
        *(source.get("cuda_device") for source in software_sources),
        receipt_device.get("device_name"),
        raw_device if isinstance(raw_device, str) else None,
    )
    device_capability = _sequence_label(
        _coalesce(
            row.get("device_capability"),
            row.get("cuda_capability"),
            row.get("cuda_cap"),
            row.get("capability"),
            device.get("device_capability"),
            device.get("cuda_capability"),
            device.get("cuda_cap"),
            device.get("capability"),
            metadata_framework.get("device_capability"),
            metadata_framework.get("cuda_capability"),
            metadata_framework.get("cuda_cap"),
            metadata_framework.get("capability"),
            *(source.get("device_capability") for source in software_sources),
            *(source.get("cuda_capability") for source in software_sources),
            *(source.get("cuda_cap") for source in software_sources),
            *(source.get("capability") for source in software_sources),
        )
    )

    normalized = dict(row)
    normalized.update(
        {
            "hardware_label": _coalesce(
                row.get("hardware_label"),
                config.get("hardware_label"),
                bench_receipt.get("hardware_label"),
            ),
            "profile": _coalesce(
                row.get("profile"),
                config.get("profile"),
                matched_key.get("profile"),
                metadata_workload.get("profile"),
                workload_key.get("profile"),
                comparison_workload.get("profile"),
                receipt_workload.get("profile"),
                receipt_comparison_workload.get("profile"),
                bench_receipt.get("profile"),
                args.get("profile"),
            ),
            "route": _coalesce(
                row.get("route"),
                config.get("route"),
                matched_key.get("route"),
                metadata_workload.get("route"),
                workload_key.get("route"),
                comparison_workload.get("route"),
                receipt_workload.get("route"),
                receipt_comparison_workload.get("route"),
                bench_receipt.get("route"),
                args.get("route"),
            ),
            "model_route": _coalesce(
                row.get("model_route"),
                config.get("model_route"),
                matched_key.get("model_route"),
                metadata_workload.get("model_route"),
                workload_key.get("model_route"),
                comparison_workload.get("model_route"),
                receipt_workload.get("model_route"),
                receipt_comparison_workload.get("model_route"),
                bench_receipt.get("model_route"),
                args.get("model_route"),
            ),
            "route_plan": _coalesce(
                row.get("route_plan"),
                config.get("route_plan"),
                matched_key.get("route_plan"),
                metadata_workload.get("route_plan"),
                workload_key.get("route_plan"),
                comparison_workload.get("route_plan"),
                receipt_workload.get("route_plan"),
                receipt_comparison_workload.get("route_plan"),
            ),
            "backend_plan": _coalesce(
                row.get("backend_plan"),
                config.get("backend_plan"),
                matched_key.get("backend_plan"),
                metadata_workload.get("backend_plan"),
                workload_key.get("backend_plan"),
                comparison_workload.get("backend_plan"),
                receipt_workload.get("backend_plan"),
                receipt_comparison_workload.get("backend_plan"),
            ),
            "model_source": _coalesce(
                row.get("model_source"),
                config.get("model_source"),
                matched_key.get("model_source"),
                metadata_workload.get("model_source"),
                workload_key.get("model_source"),
                comparison_workload.get("model_source"),
                receipt_workload.get("model_source"),
                receipt_comparison_workload.get("model_source"),
            ),
            "dtype": _coalesce(
                row.get("dtype"),
                config.get("dtype"),
                matched_key.get("dtype"),
                metadata_workload.get("dtype"),
                workload_key.get("dtype"),
                comparison_workload.get("dtype"),
                receipt_workload.get("dtype"),
                receipt_comparison_workload.get("dtype"),
                bench_receipt.get("dtype"),
                args.get("dtype"),
            ),
            "compile": _coalesce(
                row.get("compile") if "compile" in row else None,
                config.get("compile"),
                matched_key.get("compile"),
                metadata_workload.get("compile"),
                workload_key.get("compile"),
                comparison_workload.get("compile"),
                receipt_workload.get("compile"),
                receipt_comparison_workload.get("compile"),
                bench_receipt.get("compile"),
                receipt_timing.get("compile"),
                args.get("compile"),
            ),
            "warmup_steps": _coalesce(
                row.get("warmup_steps"),
                config.get("warmup_steps"),
                matched_key.get("warmup_steps"),
                metadata_workload.get("warmup_steps"),
                workload_key.get("warmup_steps"),
                comparison_workload.get("warmup_steps"),
                receipt_workload.get("warmup_steps"),
                receipt_comparison_workload.get("warmup_steps"),
                bench_receipt.get("warmup_steps"),
                receipt_timing.get("warmup_steps"),
                args.get("warmup"),
            ),
            "measured_steps": _coalesce(
                row.get("measured_steps"),
                config.get("steps"),
                matched_key.get("measured_steps"),
                metadata_workload.get("measured_steps"),
                workload_key.get("measured_steps"),
                comparison_workload.get("measured_steps"),
                receipt_workload.get("measured_steps"),
                receipt_comparison_workload.get("measured_steps"),
                bench_receipt.get("measured_steps"),
                receipt_timing.get("measured_steps"),
                args.get("repeat"),
            ),
            "batch_size": _coalesce(
                row.get("batch_size"),
                config.get("batch_size"),
                matched_key.get("batch_size"),
                metadata_workload.get("batch_size"),
                workload_key.get("batch_size"),
                comparison_workload.get("batch_size"),
                receipt_workload.get("batch_size"),
                receipt_comparison_workload.get("batch_size"),
                bench_receipt.get("batch_size"),
                args.get("batch_size"),
            ),
            "seq_len": _coalesce(
                row.get("seq_len"),
                config.get("seq_len"),
                matched_key.get("seq_len"),
                metadata_workload.get("seq_len"),
                workload_key.get("seq_len"),
                comparison_workload.get("seq_len"),
                receipt_workload.get("seq_len"),
                receipt_comparison_workload.get("seq_len"),
                bench_receipt.get("seq_len"),
                args.get("seq_len"),
                args.get("seqlen"),
            ),
            "vocab_size": _coalesce(
                row.get("vocab_size"),
                config.get("vocab_size"),
                matched_key.get("vocab_size"),
                metadata_workload.get("vocab_size"),
                workload_key.get("vocab_size"),
                comparison_workload.get("vocab_size"),
                receipt_workload.get("vocab_size"),
                receipt_comparison_workload.get("vocab_size"),
            ),
            "d_model": _coalesce(
                row.get("d_model"),
                config.get("d_model"),
                matched_key.get("d_model"),
                metadata_workload.get("d_model"),
                workload_key.get("d_model"),
                comparison_workload.get("d_model"),
                receipt_workload.get("d_model"),
                receipt_comparison_workload.get("d_model"),
            ),
            "n_heads": _coalesce(
                row.get("n_heads"),
                config.get("n_heads"),
                matched_key.get("n_heads"),
                metadata_workload.get("n_heads"),
                workload_key.get("n_heads"),
                comparison_workload.get("n_heads"),
                receipt_workload.get("n_heads"),
                receipt_comparison_workload.get("n_heads"),
            ),
            "n_layers": _coalesce(
                row.get("n_layers"),
                config.get("n_layers"),
                matched_key.get("n_layers"),
                metadata_workload.get("n_layers"),
                workload_key.get("n_layers"),
                comparison_workload.get("n_layers"),
                receipt_workload.get("n_layers"),
                receipt_comparison_workload.get("n_layers"),
            ),
            "mlp_dim": _coalesce(
                row.get("mlp_dim"),
                config.get("mlp_dim"),
                matched_key.get("mlp_dim"),
                metadata_workload.get("mlp_dim"),
                workload_key.get("mlp_dim"),
                comparison_workload.get("mlp_dim"),
                receipt_workload.get("mlp_dim"),
                receipt_comparison_workload.get("mlp_dim"),
            ),
            "learning_rate": _coalesce(
                row.get("learning_rate"),
                config.get("learning_rate"),
                matched_key.get("learning_rate"),
                metadata_workload.get("learning_rate"),
                workload_key.get("learning_rate"),
                comparison_workload.get("learning_rate"),
                receipt_workload.get("learning_rate"),
                receipt_comparison_workload.get("learning_rate"),
            ),
            "seed": _coalesce(
                row.get("seed"),
                config.get("seed"),
                matched_key.get("seed"),
                metadata_workload.get("seed"),
                workload_key.get("seed"),
                comparison_workload.get("seed"),
                receipt_workload.get("seed"),
                receipt_comparison_workload.get("seed"),
            ),
            "include_structure": include_structure,
            "data_contract": data_contract,
            "framework": framework,
            "backend": backend,
            "python_version": _coalesce(
                row.get("python_version"),
                device.get("python"),
                metadata_framework.get("python"),
                software_key.get("python_version"),
                software_key.get("python"),
                comparison_software.get("python_version"),
                comparison_software.get("python"),
                receipt_software.get("python_version"),
                receipt_software.get("python"),
                receipt_comparison_software.get("python_version"),
                receipt_comparison_software.get("python"),
            ),
            "mlx_version": _coalesce(
                row.get("mlx_version"),
                device.get("mlx"),
                metadata_framework.get("mlx"),
                *(source.get("mlx_version") for source in software_sources),
                *(source.get("mlx") for source in software_sources),
            ),
            "mlx_lm_version": _coalesce(
                row.get("mlx_lm_version"),
                device.get("mlx_lm"),
                metadata_framework.get("mlx_lm"),
                *(source.get("mlx_lm_version") for source in software_sources),
                *(source.get("mlx_lm") for source in software_sources),
            ),
            "mlx_metal_version": _coalesce(
                row.get("mlx_metal_version"),
                device.get("mlx_metal"),
                metadata_framework.get("mlx_metal"),
                *(source.get("mlx_metal_version") for source in software_sources),
                *(source.get("mlx_metal") for source in software_sources),
            ),
            "torch_version": _coalesce(
                row.get("torch_version"),
                row.get("torch"),
                device.get("torch"),
                metadata_framework.get("torch"),
                *(source.get("torch_version") for source in software_sources),
                *(source.get("torch") for source in software_sources),
            ),
            "cuda_version": _coalesce(
                row.get("cuda_version"),
                row.get("cuda"),
                row.get("torch_cuda"),
                device.get("cuda"),
                device.get("torch_cuda"),
                metadata_framework.get("cuda"),
                metadata_framework.get("torch_cuda"),
                *(source.get("cuda_version") for source in software_sources),
                *(source.get("cuda") for source in software_sources),
                *(source.get("torch_cuda") for source in software_sources),
            ),
            "driver_version": _coalesce(
                row.get("driver_version"),
                row.get("driver"),
                row.get("cuda_driver"),
                device.get("driver"),
                device.get("driver_version"),
                metadata_framework.get("driver"),
                metadata_framework.get("driver_version"),
                *(source.get("driver_version") for source in software_sources),
                *(source.get("driver") for source in software_sources),
                *(source.get("cuda_driver") for source in software_sources),
            ),
            "device_name": device_name,
            "device_capability": device_capability,
            "compile_time_s": _coalesce(
                row.get("compile_time_s"),
                bench_receipt.get("compile_time_s"),
                receipt_timing.get("compile_time_s"),
            ),
            "mean_step_time_s": _coalesce(
                row.get("mean_step_time_s"),
                bench_receipt.get("mean_step_time_s"),
                bench_receipt.get("wall_time_s"),
                receipt_timing.get("mean_step_time_s"),
                receipt_timing.get("wall_time_s"),
            ),
            "tokens_per_second": _coalesce(
                row.get("tokens_per_second"),
                row.get("tokens_per_sec"),
                row.get("tokens_per_s"),
                bench_receipt.get("tokens_per_second"),
                receipt_timing.get("tokens_per_second"),
            ),
            "peak_memory_bytes": row.get("peak_memory_bytes")
            if row.get("peak_memory_bytes") is not None
            else _coalesce(
                bench_receipt.get("peak_memory_bytes"),
                _nested(bench_receipt, "memory", "peak_bytes"),
                after_measured.get("peak_bytes"),
                _gib_to_bytes(row.get("max_alloc_gib")),
                _gib_to_bytes(row.get("peak_memory_gib")),
            ),
            "matched_comparison_key_sources": [
                source.name for source in explicit_comparison_key_sources
            ],
            "matched_comparison_key_conflicts": comparison_key_conflicts,
            "matched_comparison_key_source": (
                explicit_comparison_keys.name if explicit_comparison_keys else None
            ),
            "matched_comparison_workload_key": (
                explicit_comparison_keys.workload if explicit_comparison_keys else None
            ),
            "matched_comparison_software_key": (
                explicit_comparison_keys.software if explicit_comparison_keys else None
            ),
        }
    )
    return normalized


def host_kind(row: dict[str, Any], *, m4_label: str, gb10_label: str) -> str | None:
    hardware_label = str(row.get("hardware_label") or "").casefold()
    device_name = str(row.get("device_name") or "").casefold()
    m4 = m4_label.casefold()
    gb10 = gb10_label.casefold()
    if gb10 in hardware_label:
        return "gb10"
    if m4 in hardware_label:
        return "m4"
    if gb10 in device_name:
        return "gb10"
    if m4 in device_name:
        return "m4"
    return None


def match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_freeze_key_value(row.get(field, MISSING)) for field in MATCH_FIELDS)


def strict_match_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        match_key(row),
        _freeze_key_value(row.get("matched_comparison_workload_key", MISSING)),
        _freeze_key_value(row.get("matched_comparison_software_key", MISSING)),
    )


def _uses_cuda_metadata(row: dict[str, Any]) -> bool:
    hardware_label = str(row.get("hardware_label") or "").casefold()
    backend = str(row.get("backend") or "").casefold()
    device_name = str(row.get("device_name") or "").casefold()
    return (
        "gb10" in hardware_label
        or "dgx spark" in hardware_label
        or "cuda" in backend
        or not _is_missing(row.get("cuda_version"))
        or not _is_missing(row.get("driver_version"))
        or not _is_missing(row.get("device_capability"))
        or "nvidia" in device_name
        or "gb10" in device_name
    )


def required_match_fields_for_row(row: dict[str, Any]) -> tuple[str, ...]:
    fields = list(REQUIRED_MATCH_FIELDS)
    framework = str(row.get("framework") or "").casefold()
    if framework == "mlx":
        fields.extend(MLX_REQUIRED_MATCH_FIELDS)
    elif framework == "torch":
        fields.extend(TORCH_REQUIRED_MATCH_FIELDS)
        if _uses_cuda_metadata(row):
            fields.extend(CUDA_REQUIRED_MATCH_FIELDS)
    return tuple(dict.fromkeys(fields))


def missing_required_match_fields(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in required_match_fields_for_row(row):
        value = row.get(field, MISSING)
        if _is_missing(value):
            missing.append(field)
    return missing


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "hardware_label": row.get("hardware_label"),
        "tokens_per_second": row.get("tokens_per_second"),
        "peak_memory_bytes": row.get("peak_memory_bytes"),
        "compile_time_s": row.get("compile_time_s"),
        "mean_step_time_s": row.get("mean_step_time_s"),
        "framework": row.get("framework"),
        "backend": row.get("backend"),
        "device_name": row.get("device_name"),
        "device_capability": row.get("device_capability"),
        "python_version": row.get("python_version"),
        "mlx_version": row.get("mlx_version"),
        "mlx_lm_version": row.get("mlx_lm_version"),
        "mlx_metal_version": row.get("mlx_metal_version"),
        "torch_version": row.get("torch_version"),
        "cuda_version": row.get("cuda_version"),
        "driver_version": row.get("driver_version"),
        "matched_comparison_key_sources": row.get("matched_comparison_key_sources", []),
        "matched_comparison_key_conflicts": row.get("matched_comparison_key_conflicts", []),
        "matched_comparison_key_source": row.get("matched_comparison_key_source"),
    }


def _mismatched_comparison_keys(
    m4: dict[str, Any],
    gb10: dict[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for section, field in (
        ("workload", "matched_comparison_workload_key"),
        ("software", "matched_comparison_software_key"),
    ):
        m4_value = m4.get(field, MISSING)
        gb10_value = gb10.get(field, MISSING)
        if _freeze_key_value(m4_value) == _freeze_key_value(gb10_value):
            continue
        mismatches.append(
            {
                "field": f"comparison_key.{section}",
                "m4": MISSING if _is_missing(m4_value) else m4_value,
                "gb10": MISSING if _is_missing(gb10_value) else gb10_value,
            }
        )
    return mismatches


def _mismatched_fields(m4: dict[str, Any], gb10: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for field in MATCH_FIELDS:
        m4_value = m4.get(field, MISSING)
        gb10_value = gb10.get(field, MISSING)
        if _freeze_key_value(m4_value) == _freeze_key_value(gb10_value):
            continue
        mismatches.append(
            {
                "field": field,
                "m4": MISSING if _is_missing(m4_value) else m4_value,
                "gb10": MISSING if _is_missing(gb10_value) else gb10_value,
            }
        )
    return mismatches


def unmatched_pair_row(m4: dict[str, Any], gb10: dict[str, Any]) -> dict[str, Any]:
    m4_missing = missing_required_match_fields(m4)
    gb10_missing = missing_required_match_fields(gb10)
    m4_missing_key = missing_matched_comparison_key(m4)
    gb10_missing_key = missing_matched_comparison_key(gb10)
    mismatched_comparison_keys = _mismatched_comparison_keys(m4, gb10)
    mismatched_fields = _mismatched_fields(m4, gb10)
    if m4_missing or gb10_missing:
        reason = "missing_required_match_fields"
    elif m4_missing_key or gb10_missing_key:
        reason = "missing_matched_comparison_key"
    elif mismatched_fields:
        reason = "match_field_mismatch"
    elif mismatched_comparison_keys:
        reason = "matched_comparison_key_mismatch"
    else:
        reason = "match_field_mismatch"
    return {
        "status": "unmatched",
        "parity_claim_refused": True,
        "reason": reason,
        "matched_comparison_key_requirement": MATCHED_COMPARISON_KEY_REQUIREMENT,
        "missing_required_fields": {
            "m4": m4_missing,
            "gb10": gb10_missing,
        },
        "missing_matched_comparison_key": {
            "m4": m4_missing_key,
            "gb10": gb10_missing_key,
        },
        "matched_comparison_key_conflicts": {
            "m4": m4.get("matched_comparison_key_conflicts", []),
            "gb10": gb10.get("matched_comparison_key_conflicts", []),
        },
        "mismatched_comparison_keys": mismatched_comparison_keys,
        "mismatched_fields": mismatched_fields,
        "m4": _row_identity(m4),
        "gb10": _row_identity(gb10),
    }


def unmatched_pairs(
    m4_rows: list[dict[str, Any]],
    gb10_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for m4 in m4_rows:
        m4_missing = missing_required_match_fields(m4)
        m4_missing_key = missing_matched_comparison_key(m4)
        for gb10 in gb10_rows:
            gb10_missing = missing_required_match_fields(gb10)
            gb10_missing_key = missing_matched_comparison_key(gb10)
            if (
                not m4_missing
                and not gb10_missing
                and not m4_missing_key
                and not gb10_missing_key
                and strict_match_key(m4) == strict_match_key(gb10)
            ):
                continue
            pairs.append(unmatched_pair_row(m4, gb10))
    return pairs


def comparison_row(key: tuple[Any, ...], m4: dict[str, Any], gb10: dict[str, Any]) -> dict[str, Any]:
    fields = {
        field: MISSING if _is_missing(m4.get(field, MISSING)) else m4.get(field)
        for field in MATCH_FIELDS
    }
    return {
        "status": "matched",
        "match": fields,
        "m4": _row_identity(m4),
        "gb10": _row_identity(gb10),
        "ratios": {
            "m4_tokens_per_second_over_gb10": _ratio(
                m4.get("tokens_per_second"),
                gb10.get("tokens_per_second"),
            ),
            "gb10_tokens_per_second_over_m4": _ratio(
                gb10.get("tokens_per_second"),
                m4.get("tokens_per_second"),
            ),
            "m4_peak_memory_bytes_over_gb10": _ratio(
                m4.get("peak_memory_bytes"),
                gb10.get("peak_memory_bytes"),
            ),
            "gb10_peak_memory_bytes_over_m4": _ratio(
                gb10.get("peak_memory_bytes"),
                m4.get("peak_memory_bytes"),
            ),
            "m4_compile_time_s_over_gb10": _ratio(
                m4.get("compile_time_s"),
                gb10.get("compile_time_s"),
            ),
            "gb10_compile_time_s_over_m4": _ratio(
                gb10.get("compile_time_s"),
                m4.get("compile_time_s"),
            ),
        },
    }


def build_report(
    rows: list[dict[str, Any]],
    *,
    m4_label: str = "M4 Max",
    gb10_label: str = "GB10",
) -> dict[str, Any]:
    normalized = [normalized_row(row) for row in rows]
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = {}
    host_rows: dict[str, list[dict[str, Any]]] = {"m4": [], "gb10": []}
    host_counts = {"m4": 0, "gb10": 0, "ignored": 0}
    incomplete_match_metadata_counts = {"m4": 0, "gb10": 0}
    missing_matched_comparison_key_counts = {"m4": 0, "gb10": 0}
    matched_comparison_key_conflict_counts = {"m4": 0, "gb10": 0}
    for row in normalized:
        kind = host_kind(row, m4_label=m4_label, gb10_label=gb10_label)
        if kind is None:
            host_counts["ignored"] += 1
            continue
        host_counts[kind] += 1
        host_rows[kind].append(row)
        missing_required = missing_required_match_fields(row)
        missing_key = missing_matched_comparison_key(row)
        key_conflict = bool(row.get("matched_comparison_key_conflicts"))
        if missing_required:
            incomplete_match_metadata_counts[kind] += 1
        if missing_key:
            missing_matched_comparison_key_counts[kind] += 1
        if key_conflict:
            matched_comparison_key_conflict_counts[kind] += 1
        if missing_required or missing_key:
            continue
        grouped.setdefault(strict_match_key(row), {"m4": [], "gb10": []})[kind].append(row)

    comparisons: list[dict[str, Any]] = []
    for key, hosts in grouped.items():
        if not hosts["m4"] or not hosts["gb10"]:
            continue
        for m4 in hosts["m4"]:
            for gb10 in hosts["gb10"]:
                comparisons.append(comparison_row(key, m4, gb10))

    refused_pairs = unmatched_pairs(host_rows["m4"], host_rows["gb10"])
    if comparisons:
        status = "ok"
    elif host_counts["m4"] == 0 or host_counts["gb10"] == 0:
        status = "insufficient_matched_rows"
    else:
        status = "no_matching_rows"

    return {
        "status": status,
        "row_count": len(rows),
        "host_counts": host_counts,
        "match_fields": list(MATCH_FIELDS),
        "required_match_fields": list(REQUIRED_MATCH_FIELDS),
        "matched_comparison_key_requirement": MATCHED_COMPARISON_KEY_REQUIREMENT,
        "conditional_required_match_fields": {
            "mlx": list(MLX_REQUIRED_MATCH_FIELDS),
            "torch": list(TORCH_REQUIRED_MATCH_FIELDS),
            "cuda": list(CUDA_REQUIRED_MATCH_FIELDS),
        },
        "incomplete_match_metadata_counts": incomplete_match_metadata_counts,
        "missing_matched_comparison_key_counts": missing_matched_comparison_key_counts,
        "matched_comparison_key_conflict_counts": matched_comparison_key_conflict_counts,
        "parity_claim_refused": not comparisons,
        "matched_comparison_count": len(comparisons),
        "unmatched_pair_count": len(refused_pairs),
        "unmatched_pairs": refused_pairs,
        "comparisons": comparisons,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _packaged_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        **report,
        "comparisons": [
            {
                key: value
                for key, value in comparison.items()
                if key != "ratios"
            }
            for comparison in report["comparisons"]
        ],
        "ratio_artifact": MATCHED_COMPARISONS_FILENAME,
        "artifact_policy": (
            "Package compare_report.json intentionally omits ratios. "
            "Use matched_comparisons.jsonl for ratio-bearing matched rows only."
        ),
    }


def write_compare_package(
    report: dict[str, Any],
    package_dir: Path,
    *,
    inputs: list[str],
    m4_label: str,
    gb10_label: str,
) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=True)
    report_path = package_dir / COMPARE_REPORT_FILENAME
    matched_path = package_dir / MATCHED_COMPARISONS_FILENAME
    refused_path = package_dir / REFUSED_PAIRS_FILENAME
    manifest_path = package_dir / COMPARE_PACKAGE_MANIFEST_FILENAME

    _write_json(report_path, _packaged_report(report))
    _write_jsonl(matched_path, report["comparisons"])
    _write_jsonl(refused_path, report["unmatched_pairs"])

    manifest = {
        "schema_version": COMPARE_PACKAGE_SCHEMA_VERSION,
        "kind": COMPARE_PACKAGE_KIND,
        "status": report["status"],
        "package_dir": str(package_dir),
        "inputs": list(inputs),
        "hardware_labels": {
            "m4": m4_label,
            "gb10": gb10_label,
        },
        "artifacts": {
            "compare_report": COMPARE_REPORT_FILENAME,
            "matched_comparisons": MATCHED_COMPARISONS_FILENAME,
            "refused_pairs": REFUSED_PAIRS_FILENAME,
            "manifest": COMPARE_PACKAGE_MANIFEST_FILENAME,
        },
        "host_counts": report["host_counts"],
        "incomplete_match_metadata_counts": report[
            "incomplete_match_metadata_counts"
        ],
        "missing_matched_comparison_key_counts": report[
            "missing_matched_comparison_key_counts"
        ],
        "matched_comparison_key_conflict_counts": report[
            "matched_comparison_key_conflict_counts"
        ],
        "matched_comparison_count": report["matched_comparison_count"],
        "unmatched_pair_count": report["unmatched_pair_count"],
        "parity_claim_refused": report["parity_claim_refused"],
        "matched_comparison_key_requirement": report[
            "matched_comparison_key_requirement"
        ],
        "artifact_policy": (
            "Only matched_comparisons.jsonl may contain ratios. "
            "refused_pairs.jsonl is refusal evidence and must not be used for "
            "M4-vs-GB10 throughput or memory claims."
        ),
        "workload_software_key_guard": (
            "GB10 comparisons require identical selected "
            "comparison_key.workload and comparison_key.software objects on "
            "both rows; unmatched or row-local conflicting keys stay refused."
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(
            load_rows(args.input),
            m4_label=args.m4_label,
            gb10_label=args.gb10_label,
        )
        if args.package_dir is not None:
            report = {
                **report,
                "compare_package": write_compare_package(
                    report,
                    args.package_dir,
                    inputs=args.input,
                    m4_label=args.m4_label,
                    gb10_label=args.gb10_label,
                ),
            }
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    if args.jsonl:
        if report["comparisons"]:
            for comparison in report["comparisons"]:
                print(json.dumps(comparison, sort_keys=True))
        else:
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "host_counts": report["host_counts"],
                        "parity_claim_refused": report["parity_claim_refused"],
                        "matched_comparison_count": 0,
                        "unmatched_pair_count": report["unmatched_pair_count"],
                        "unmatched_pairs": report["unmatched_pairs"],
                    },
                    sort_keys=True,
                )
            )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
