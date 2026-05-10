# AGENTS.md — cppmega.mlx agent contract

## Tensor memory rule

- Wrappers/adapters must not silently allocate or copy large tensors.
- If a design appears to require staging a large tensor, casting a large tensor,
  or copying a large tensor only to satisfy a wrapper boundary, treat that
  design as wrong and keep looking.
- Prefer passing references to existing GPU buffers, fusion, views/broadcasts,
  or IR-level lowering through the TileLang/TVM/tvm-ffi pipeline.
- If zero-copy/fused lowering is not currently possible, fail explicitly or
  fall back to the existing production path without pretending Path C handled it.
- Dtype/shape mismatches must be solved at the right level:
  - first look higher in the graph and make the producer create the tensor in
    the required dtype/shape from the start;
  - if the current dtype/layout is the right system-level format (for example
    FP8 for the active config), move lower and teach the kernel to consume that
    format directly;
  - only use casts/reshapes as explicit graph decisions, not hidden adapter
    staging.
