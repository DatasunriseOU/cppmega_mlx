# Metal Kernel Policy

This repo keeps custom Metal kernels behind explicit, optional seams. The
current prototype is cppmega_mlx.kernels.metal_ops.squared_relu, a
non-critical forward-only relu(x) ** 2 op with a pure MLX fallback.

## Policy

- Start every custom kernel with a pure MLX reference implementation.
- Keep backend="auto" safe: use Metal only when the local MLX process and
  input dtype are eligible, otherwise return the pure MLX result.
- Keep backend="metal" fail-closed: if Metal is unavailable or the input is
  unsupported, raise an explicit unsupported-path error instead of silently
  falling back.
- Keep can_run_metal() narrow: it only reports whether the current process has
  a default MLX GPU device with mx.metal.is_available(). Per-kernel readiness
  belongs in metal_kernel_status(...).
- Require metal_kernel_status(...) to check the local mx.fast.metal_kernel
  API, supported dtype, non-empty input, and constructed kernel object before
  reporting available=True.
- Do not put prototype Metal kernels in the required training graph. Training
  callers must pass through the pure MLX fallback unless the Metal path is
  explicitly wrapped by mx.custom_function and supplies the required VJP/JVP.
- Do not load remote Hugging Face kernels, depend on HF kernel packages, or
  make HF kernel repositories part of the training path. External Apple/Metal
  kernels are source-review and parity-fixture inputs only until pinned,
  vendored or reimplemented in-tree, licensed, profiled, and covered by the same
  fallback and VJP/JVP gates.
- Verify fallback parity unconditionally and Metal parity only when Metal is
  available. Covered dtypes are mx.float32, mx.float16, and mx.bfloat16;
  every new dtype needs an explicit parity row before it is advertised as
  supported.
- Treat benchmarks as secondary. Correctness, fallback behavior, and explicit
  failure modes are required before timing claims.
- Do not add CUDA-only runtime branches to the local MLX/Metal path. CUDA,
  Triton, TE, H200, or GB10 details may appear in comparison docs only, not in
  cppmega_mlx.kernels.metal_ops.

## Current Local Receipt

Verified on 2026-04-30 from the local Apple Silicon helper environment:

text
mlx==0.31.1
mlx-lm==0.31.2
mlx-metal==0.31.1
default_device Device(gpu, 0)
mx.metal.is_available() True
mx.fast.metal_kernel present True
mx.custom_function present True
device_name Apple M4 Max
architecture applegpu_g16s


For this host, metal_kernel_status() reports available=True for supported
non-empty floating tensors, and the squared_relu(..., backend="metal") smoke
test runs for mx.float32, mx.float16, and mx.bfloat16. bfloat16
assertions cast through mx.float32 before NumPy conversion because direct
NumPy conversion of MLX bfloat16 arrays is not a reliable test oracle in the
installed stack. This is local availability evidence only. It is not M4 Max vs
GB10 performance parity evidence.

## squared_relu Fallback Matrix

| Condition                                       | backend="auto"    | backend="metal"        |
| ----------------------------------------------- | ----------------- | ---------------------- |
| Metal device unavailable                        | Pure MLX fallback | MetalKernelUnsupported |
| mx.fast.metal_kernel unavailable                | Pure MLX fallback | MetalKernelUnsupported |
| Unsupported dtype                               | Pure MLX fallback | MetalKernelUnsupported |
| Empty tensor                                    | Pure MLX fallback | MetalKernelUnsupported |
| Constructed kernel missing                      | Pure MLX fallback | MetalKernelUnsupported |
| Supported non-empty float tensor on this M4 Max | Metal path        | Metal path             |

backend="mlx" always uses the pure MLX reference. Unknown backend labels,
including "cuda", raise ValueError; they are not aliases for a fallback.

## Differentiability Gate

Plain mx.fast.metal_kernel is forward-only for cppmega.mlx policy purposes.
Any kernel adopted into differentiated training must first be wrapped with
mx.custom_function and define a VJP for every trainable input. JVP support is
required when the training caller or diagnostic path depends on forward-mode
transforms. Scatter, gather, and routing-style backward kernels must also handle
initialization and atomic accumulation explicitly before training adoption.

Until that VJP gate is satisfied, a Metal kernel may be used only for
forward-only diagnostics, preprocessing, or optional inference-style paths that
retain a pure MLX fallback.

Remote or community kernels do not bypass this gate. A kernel read from
Hugging Face, MLX examples, or MLX-LM can inform an in-tree implementation or
test fixture, but it is not a supported cppmega.mlx training dependency unless
the differentiability, fallback, source pin, license, local parity, and hotspot
receipts are all checked into this repo.

squared_relu(..., training=True) is the concrete enforcement surface. With
backend="auto" or backend="mlx" it always uses the pure MLX implementation,
so mx.grad and mx.jvp transform the normal MLX graph. With
backend="metal" it raises MetalKernelUnsupported because the prototype
Metal kernel is forward-only and has no custom VJP/JVP.

The public TrainingKernelStatus object is also fail-closed. It rejects
training_safe=True unless the kernel is owned in-tree, the source is pinned,
license coverage is recorded, the pure-MLX fallback is covered, local parity is
covered, hotspot evidence exists, the operation is marked differentiable,
VJP/backward coverage is present, and the fallback backend remains "mlx".
jvp_covered=False can be explicit for kernels that are VJP-backed but not used
by forward-mode diagnostics; a caller that needs mx.jvp must still require the
JVP field before training adoption.

## Sources Checked

- docs/research/mlx_core_and_metal.md
- MLX custom Metal kernel docs:
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- Official MLX docs state that each mx.fast.metal_kernel construction creates
  a Metal library and may trigger JIT compilation, so cppmega builds prototype
  kernel objects once and reuses them.
- The same docs state ensure_row_contiguous=True can copy non-contiguous
  inputs, so hot kernels need explicit profiling and stride-aware tests before
  adoption.
- The same docs show differentiable kernels through mx.custom_function and
  VJP/JVP definitions; scatter-style backward examples use init_value=0 and
  atomic_outputs=True.
- W3.5 direct MLX/MLX-LM API refresh on 2026-05-01 returned HTTP 200 for the
  GitHub repo and latest-release endpoints. It recorded MLX v0.31.2,
  published 2026-04-22T01:40:04Z, and MLX-LM v0.31.3, published
  2026-04-22T07:43:57Z. Mutable star, fork, and updated_at fields are not a
  policy contract.
- Direct MLX-LM loss-source refresh:
  https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/tuner/losses.py
  returned HTTP 200 and still includes can_run_metal(),
  mx.fast.metal_kernel, @mx.custom_function, and .vjp loss kernels with
  non-Metal fallback paths.
  This is reference evidence only: cppmega training paths still require local
  fallback, explicit Metal fail-closed behavior, and VJP/JVP gates.
- Installed MLX stubs:
  .venv/lib/python3.13/site-packages/mlx/core/fast.pyi
  and .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi
- Installed MLX-LM differentiable Metal examples:
  .venv/lib/python3.13/site-packages/mlx_lm/tuner/losses.py
- Hugging Face Apple/Metal kernel references:
  kernels-community/metal-flash-sdpa,
  kernels-community/mlx-quantization-metal-kernels,
  kernels-community/activation, and
  kernels-community/gpt-oss-metal-kernels
- MLX examples reference repo:
  https://github.com/ml-explore/mlx-examples

mlx-examples and Hugging Face Apple M4 kernels are source-reading candidates
only. They do not weaken the in-repo fallback, explicit Metal mode,
profile-before-kernel, or VJP/JVP training-path gates above.
The 2026-04-30 Hugging Face Apple M4 listing showed 10 entries, and the W3.5
2026-05-01 refresh still found 10 entries after HTML-unescaping the embedded
KernelList payload. This is catalog evidence only, not adoption evidence.
The same listing refresh returned HTTP 200 from
https://huggingface.co/kernels?hardware=apple-m4&sort=trending; the guessed
direct API https://huggingface.co/api/kernels?hardware=apple-m4&sort=trending
returned HTTP 404, so the recorded rows come from the HTML-embedded
KernelList metadata.
Direct git ls-remote checks on the same HF kernel repos showed that live
repository HEADs can differ from the HTML listing sha values. Treat listing
metadata as a mutable catalog snapshot; pin and verify a direct revision before
any source import or dependency decision.

The current M4 Max Mamba3/M2RNN receipts in docs/perf_mamba_m2rnn.md are
smoke-scale route and checkpoint evidence. They do not justify adopting a custom
Metal kernel or external HF kernel; use them to decide what to profile next.
