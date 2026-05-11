# MLX Runtime Bridge Gate

`cppmega_mlx.nn._tilelang._mlx_runtime.wrap_tilelang_metal_kernel` is a
legacy migration bridge from TileLang-emitted MSL text to
`mx.fast.metal_kernel`. It is not the production Path C boundary.

Production Path C work must use the TileLang/TVM/tvm-ffi owner-output route so
callers pass existing GPU buffers and explicit output ownership through the
graph. The raw MLX fast-kernel bridge now fails closed unless the caller passes
`allow_mx_fast_metal_kernel=True`.

That opt-in is reserved for tests, POC harnesses, and explicit migration tools
that need to inspect or run a lowered MSL body. The gate is control-flow only:
it does not allocate tensors, copy buffers, cast dtypes, or reshape data.
