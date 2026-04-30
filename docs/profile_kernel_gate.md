# Profile-Before-Kernel Gate

Custom Metal kernels are blocked from the cppmega.mlx training path until the
repo has local profile evidence for the exact route and shape that motivates
the replacement. The gate is implemented in
`cppmega_mlx.training.profile.require_kernel_hotspot_evidence(...)`.

## Contract

Before adopting a custom kernel, collect hotspot records with:

- `candidate_kernel`: the proposed kernel name or package entry.
- `name`: measured operation or scope, for example `mamba3_scan`.
- `seconds` and `total_seconds`: elapsed hotspot time and comparable profile
  window time.
- `calls`: number of measured calls in the profile window.
- `route`, `backend`, and `operation`: route metadata such as `A`, `M`, `E`, or
  `R`, plus `mlx`/`metal`/reference backend labels.
- `source`: profiler source, for example `profile_step`, MLX trace, or external
  profiler report.

`assess_kernel_adoption(...)` returns a JSON-safe verdict. The stricter
`require_kernel_hotspot_evidence(...)` raises `KernelAdoptionBlocked` when the
evidence is absent or below threshold.

The default threshold is intentionally conservative for tiny smoke tests:

- at least one profile sample,
- top hotspot elapsed time at least `0.001s`,
- top hotspot fraction at least `0.10` of the measured window.

Production adoption should raise these thresholds and use repeated warm runs.

## Fail-Closed Rules

A kernel candidate must remain blocked when:

- there are no profile samples,
- the strongest hotspot is below the minimum elapsed-time threshold,
- the strongest hotspot is below the minimum profile-fraction threshold,
- the candidate only has an external HF/MLX example but no cppmega route
  profile,
- the differentiated training path lacks custom VJP/JVP coverage.

This means HF Apple M4 kernels and MLX-LM Metal examples are implementation
references only until local hotspot, parity, dtype, fallback, and backward
evidence exist in this repo.

## Minimal Usage

```python
from cppmega_mlx.training.profile import (
    hotspot_from_profile_metrics,
    profile_step,
    require_kernel_hotspot_evidence,
)

with profile_step("mamba3_scan", tokens=4096, extra={"route": "M"}) as prof:
    loss = run_train_step()
    prof.add_eval_args(loss)

hotspot = hotspot_from_profile_metrics(
    prof.metrics,
    total_seconds=full_step_seconds,
    source="profile_step",
)
assessment = require_kernel_hotspot_evidence(
    "mamba3-metal-scan",
    [hotspot],
    min_hotspot_fraction=0.20,
    min_hotspot_seconds=0.05,
)
```

Archive `assessment.to_dict()` next to the benchmark row or profile report. A
human-readable summary can be generated with `summarize_hotspots(...)`.

## Current Decision

No custom training-path kernel is adopted by this lane. The gate only creates
the measurable decision surface. Existing local docs already record the current
Apple M4/HF kernel survey and MLX custom-kernel constraints: `mx.fast`
`metal_kernel` is available for forward kernels, but differentiated training
replacement requires a custom-function VJP/JVP and parity tests before
adoption.
