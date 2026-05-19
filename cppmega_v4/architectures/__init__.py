"""Architecture preset registry — JSON-shaped block specs for the
gallery of public hybrid LLM architectures we want to be able to compose
from V4 bricks.

Each preset is a list of ``{"kind": ..., "name": ..., "params": ...}``
dicts ready to feed to
:func:`cppmega_v4.fusion.brick_graph.from_block_specs`. The preset
covers ONE repeat-unit of the architecture; callers replicate the
returned list ``num_layers / repeat_unit_size`` times to build a full
model.

Public API:
  - :data:`PRESETS`: ``dict[str, callable(int)]`` mapping a preset name
    to a function ``hidden_size -> list[dict]``.
  - :func:`build_preset_specs(name, hidden_size, *, num_layers=None)`:
    return the concatenated spec list ready to instantiate.
  - :func:`available_presets()`: sorted list of preset names.
"""

from __future__ import annotations

from cppmega_v4.architectures.presets import (
    PRESETS,
    available_presets,
    build_preset_specs,
)

__all__ = [
    "PRESETS",
    "available_presets",
    "build_preset_specs",
]
