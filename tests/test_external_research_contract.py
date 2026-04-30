from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DOCS = (
    ROOT / "docs" / "research" / "mlx_core_and_metal.md",
    ROOT / "docs" / "research" / "mlx_lm_training_patterns.md",
    ROOT / "docs" / "research" / "apple_kernel_survey.md",
)
OWNED_CONTRACT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "porting_plan.md",
    ROOT / "docs" / "parity_anchors.md",
    ROOT / "docs" / "profile_kernel_gate.md",
    ROOT / "docs" / "checkpointing.md",
)


def _read_docs() -> tuple[str, str]:
    return (
        (ROOT / "docs" / "porting_plan.md").read_text(),
        (ROOT / "README.md").read_text(),
    )


def _read_research_docs() -> dict[str, str]:
    return {path.name: path.read_text() for path in RESEARCH_DOCS}


def _read_owned_contract_docs() -> str:
    return "\n".join(path.read_text() for path in OWNED_CONTRACT_DOCS)


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_external_framework_decisions_keep_mlx_native_contract() -> None:
    porting_plan, _ = _read_docs()

    assert "## External Framework Decisions" in porting_plan
    assert "Checked references: MLX core, MLX-LM" in porting_plan
    assert "not imported dependencies" in porting_plan
    assert "Use MLX core as the P0 runtime" in porting_plan
    assert "Use MLX-LM as a pattern source" in porting_plan
    assert "Do not make MLX-LM the cppmega trainer base" in porting_plan
    assert "mlx-tune" in porting_plan
    assert "mlx-forge" in porting_plan
    assert "ForgeLLM" in porting_plan
    assert "MLX-GRPO" in porting_plan


def test_research_docs_include_primary_external_receipts() -> None:
    docs = _read_research_docs()
    combined = "\n".join(docs.values())

    for text in docs.values():
        assert "Primary Receipts Refresh" in text
        assert "https://raw.githubusercontent.com/ml-explore/mlx/main/README.md" in text
        assert "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md" in text
        assert "https://huggingface.co/kernels?hardware=apple-m4&sort=trending" in text

    assert "MLX README direct fetch returned HTTP 200" in combined
    assert "MLX-LM README direct fetch returned HTTP 200" in combined
    assert "Hugging Face Apple M4 kernel listing direct fetch returned HTTP 200" in combined
    assert "Apple Silicon array framework" in combined or "Apple Silicon arrays" in combined
    assert "`mlx.nn`" in combined
    assert "`mlx.optimizers`" in combined
    assert "`mx.distributed`" in combined


def test_hf_apple_m4_kernel_snapshot_stays_reference_only() -> None:
    docs = _read_research_docs()
    combined = "\n".join(docs.values())
    normalized = " ".join(combined.split())

    assert "10 Apple M4 kernel entries" in combined or "showed 10 entries" in combined
    for kernel_name in (
        "mlx-rmsnorm",
        "mlx-quantization-metal-kernels",
        "metal-flash-sdpa",
        "paged-attention",
        "gpt-oss-metal-kernels",
        "bitsandbytes-mps",
        "activation",
    ):
        assert kernel_name in combined

    assert "reference-only" in combined
    assert "not training dependencies" in normalized or "not pretraining dependencies" in normalized
    assert "No HF kernel is on the training path" in combined
    assert "pure MLX fallback" in combined
    assert "parity tests" in combined


def test_external_kernel_and_nanochat_boundaries_are_documented() -> None:
    porting_plan, readme = _read_docs()
    combined = f"{porting_plan}\n{readme}"
    normalized_plan = " ".join(porting_plan.split())

    assert "Hugging Face Apple M4 kernels" in combined
    assert "source references" in porting_plan
    assert "Metal flash SDPA" in porting_plan
    assert "not adoption decisions" in normalized_plan
    assert "Do not borrow a kernel into the training path" in normalized_plan
    assert "../nanochat" in combined
    assert "Torch reference only" in combined
    assert "not Metal-native" in combined


def test_m4_gb10_performance_contract_remains_matched_row_only() -> None:
    porting_plan, readme = _read_docs()
    research = "\n".join(_read_research_docs().values())
    combined = f"{porting_plan}\n{readme}\n{research}"

    forbidden_standalone_overclaims = (
        "M4 Max parity with GB10 is proven",
        "M4 Max is not worse than GB10.",
        "M4 Max is not worse than GB10\n",
        "M4 Max matches GB10",
        "GB10 is slower",
        "M4-only rows prove GB10 parity",
    )
    for phrase in forbidden_standalone_overclaims:
        assert phrase not in combined

    assert 'Do not claim M4 Max is "not worse than GB10"' in porting_plan
    assert "Strict rule: an M4 Max run is never a GB10 parity claim by itself" in porting_plan
    assert "matched-row protocol" in porting_plan
    assert "No M4 Max vs GB10 parity claim without matched GB10 data" in research
    assert "M4-only rows cannot support" in research


def test_package_dependency_contract_matches_documented_runtime() -> None:
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    dependency_names = {str(item).split(">=", maxsplit=1)[0] for item in dependencies}

    assert {"mlx", "mlx-lm", "numpy", "safetensors"} <= dependency_names

    optional_dependencies = project["optional-dependencies"]
    assert isinstance(optional_dependencies, dict)
    parquet_extra = optional_dependencies["parquet"]
    assert isinstance(parquet_extra, list)
    assert any(str(item).startswith("pyarrow>=") for item in parquet_extra)

    docs = _read_owned_contract_docs()
    assert "Base package dependencies are `mlx`, `mlx-lm`, `numpy`, and `safetensors`." in docs
    assert "Parquet loading stays optional" in docs


def test_package_init_exports_stable_public_helpers() -> None:
    import cppmega_mlx.config as config
    import cppmega_mlx.models as models
    import cppmega_mlx.recipes as recipes
    from cppmega_mlx.config import Nam56RModelConfig
    from cppmega_mlx.models import HybridTinyConfig, HybridTinyLM, TinyLM, TinyLMConfig
    from cppmega_mlx.recipes import REFERENCE_PATTERN, build_nam56r_pattern

    assert "Nam56RModelConfig" in config.__all__
    assert "HybridTinyLM" in models.__all__
    assert "build_nam56r_pattern" in recipes.__all__
    assert Nam56RModelConfig().depth == 52
    assert TinyLMConfig(vocab_size=8, hidden_size=8, num_heads=2).vocab_size == 8
    assert TinyLM.__name__ == "TinyLM"
    assert HybridTinyConfig(vocab_size=8, hidden_size=8, num_attention_heads=2).depth == 4
    assert HybridTinyLM.__name__ == "HybridTinyLM"
    assert REFERENCE_PATTERN == "AEMEAEMEAEMR"
    assert build_nam56r_pattern().depth == 52


def test_no_tracked_parquet_samples_or_runtime_overclaims() -> None:
    docs = _read_owned_contract_docs()
    normalized = " ".join(docs.split())

    tracked_parquet = subprocess.run(
        ["git", "ls-files", "*.parquet", "data/parquet_samples/*"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.splitlines()
    assert tracked_parquet == []

    forbidden_overclaims = (
        "M4 Max parity with GB10 is proven",
        "M4 Max matches GB10",
        "M4-only rows prove GB10 parity",
        "distributed Megatron parity is proven",
        "full distributed Megatron parity",
        "production-scale Megatron `.bin/.idx` input is proven",
    )
    for phrase in forbidden_overclaims:
        assert phrase not in docs

    assert "not checked-in fixtures" in normalized
    assert "M4 Max vs GB10 parity is not proven" in docs
    assert "Distributed MLX training is not implemented in this repo" in docs
    assert "full Megatron launcher/training parity remains outside the tiny-local scaffold" in normalized


def test_external_research_contract_is_importable_from_clean_process() -> None:
    script = "\n".join(
        [
            "from cppmega_mlx.config import Nam56RModelConfig",
            "from cppmega_mlx.models import HybridTinyConfig, TinyLMConfig",
            "from cppmega_mlx.recipes import REFERENCE_PATTERN, build_nam56r_pattern",
            "assert Nam56RModelConfig().depth == 52",
            "assert TinyLMConfig().vocab_size == 64",
            "assert HybridTinyConfig().depth == 4",
            "assert REFERENCE_PATTERN == 'AEMEAEMEAEMR'",
            "assert build_nam56r_pattern().dsa_layer_numbers == (5, 9, 13, 21, 25, 29, 37, 41, 45)",
        ]
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)
