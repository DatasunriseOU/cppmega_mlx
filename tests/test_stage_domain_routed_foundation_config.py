from __future__ import annotations

from pathlib import Path

from cppmega_mlx.data.domain_schema import DomainKind
from cppmega_mlx.data.production_bundle import _DOMAIN_ROUTE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "stage_domain_routed_foundation.yaml"


def _yaml_list(text: str, key: str) -> set[str]:
    lines = text.splitlines()
    start = lines.index(f"  {key}:") + 1
    values: set[str] = set()
    for line in lines[start:]:
        if not line.startswith("    - "):
            break
        values.add(line.removeprefix("    - ").strip())
    return values


def test_domain_routed_recipe_tracks_frozen_build_and_shell_domains() -> None:
    text = RECIPE.read_text(encoding="utf-8")
    configured_build_kinds = _yaml_list(text, "build_kinds")
    configured_shell_kinds = _yaml_list(text, "shell_kinds")

    enum_build_kinds = {
        kind.name.lower() for kind in DomainKind if 2 <= int(kind) < 20
    }
    enum_shell_kinds = {
        kind.name.lower() for kind in DomainKind if 20 <= int(kind) < 30
    }

    assert configured_build_kinds == enum_build_kinds
    assert configured_shell_kinds == enum_shell_kinds


def test_recipe_requires_production_source_provenance_as_data_only() -> None:
    text = RECIPE.read_text(encoding="utf-8")
    required_token_sidecars = _yaml_list(text, "require_token_sidecars")
    production_source_provenance = {
        name for name in _DOMAIN_ROUTE_COLUMNS if name.startswith("token_source_")
    }
    model_inputs = text.split("model_inputs:\n", 1)[1].split("\nstages:\n", 1)[0]

    assert production_source_provenance == {
        "token_source_doc_ids",
        "token_source_identity_ids",
    }
    assert production_source_provenance <= required_token_sidecars
    assert "source_build_kinds" not in model_inputs
