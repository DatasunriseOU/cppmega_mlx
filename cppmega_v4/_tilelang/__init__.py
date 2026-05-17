"""Side-by-side TileLang/Metal dispatch scaffolding for V4 blocks.

Each Path B/C/D/E module exposes the same API surface as Path A and falls
back to Path A's reference kernel when the underlying backend (Metal MSL /
TileLang DSL / Triton frontend / vendored mlx-lm op) is not available on
the host or is still pending implementation.

This keeps the v4 plugin runnable end-to-end on day one while leaving room
for each backend to be filled in incrementally.
"""
