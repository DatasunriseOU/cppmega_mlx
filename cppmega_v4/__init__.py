"""cppmega_v4 — DeepSeek V4 / GatedDeltaNet plugin port for cppmega.mlx.

This package is a **side-by-side plugin**: it imports from ``cppmega_mlx`` but
never modifies it. The existing ``cppmega_mlx`` modules continue to receive
parallel performance work on the TileLang / TVM / TVM-FFI stack while this
package ports new architectural primitives in isolation.

Public surface (filled in as ROIs land):
    - ``cppmega_v4.nn.moe_v4`` — aux-loss-free balancing, sqrt(softplus) scoring
    - ``cppmega_v4.nn.linear_attention`` — GDN block ("L" symbol)
    - ``cppmega_v4.nn.kimi_delta_attention`` — KDA block ("K" symbol)
    - ``cppmega_v4.nn.mhc_v4`` — Sinkhorn-Knopp Manifold HyperConnection
    - ``cppmega_v4.nn.engram_v4`` — TileKernels-grade Engram
    - ``cppmega_v4.nn.attention_v4`` — FlashMLA absorb, Lightning Indexer
    - ``cppmega_v4.nn.mtp_v4`` — Sequential depth-D MTP heads
    - ``cppmega_v4.models.hybrid_v4`` — V4HybridTinyConfig / V4HybridTinyLM

Build invariants:
    - No file under ``cppmega_mlx/`` is modified by this port.
    - No entry is added to the ``cppmega_mlx*`` glob in ``pyproject.toml``.
    - Tests live under ``tests/v4/`` and use ``pytest.importorskip`` for
      anything that depends on optional kernels.
    - Final cut-over (swapping ``HybridTinyLM`` users to V4 wrappers) is a
      separate ROI handled only after every block here is green.
"""

__version__ = "0.0.0-dev"

__all__ = ["__version__"]
