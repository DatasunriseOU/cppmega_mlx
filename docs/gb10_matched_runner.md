# GB10 Matched Comparison Guard

This workspace must not claim M4 Max versus GB10 parity unless both sides
produce rows for the same workload and the same software stack identity.

`scripts/compare_bench_rows.py` is the gate. It may emit ratios only when both
rows contain explicit matched-key provenance through one of these schemas:

- `comparison_key.workload` plus `comparison_key.software`
- `bench_receipt.comparison_key.workload` plus `bench_receipt.comparison_key.software`
- `workload_key` plus `software_key`
- legacy `matched_run.key` plus `run_metadata.framework`

Rows that only happen to share top-level shape fields are not enough. The tool
must report `parity_claim_refused: true` and no `ratios` when the matched key is
absent or when any required workload/software field differs. The selected
explicit workload key and selected explicit software key must also be identical
as complete machine-readable objects. If a GB10 row carries an extra software
flag, a different data-contract label, or any other key-only distinction that
normalizes to the same top-level fields, the comparison remains unmatched.
When modern key sources coexist inside one row, they must agree with each other
before the row is usable. For example, top-level `comparison_key`,
`bench_receipt.comparison_key`, and `workload_key` plus `software_key` are
checked as redundant provenance. If they disagree, the report increments
`matched_comparison_key_conflict_counts`, marks the row as missing a usable
matched comparison key, and refuses ratios. The legacy `matched_run.key` plus
`run_metadata.framework` path is fallback provenance only when no modern key
source exists.

Local Apple Silicon matrix receipts also carry `receipt_scope: local_only`,
`local_only: true`, and `gb10_parity_claim: false`. These markers are not a
replacement for the matched-row guard; they are a visible reminder that a local
M4 row by itself is single-host evidence.

## Exact Workload Fields

Matched rows must carry identical values for:

- `profile`
- `route`
- `model_route`
- `route_plan`
- `backend_plan`
- `model_source`
- `dtype`
- `compile`
- `warmup_steps`
- `measured_steps`
- `batch_size`
- `seq_len`
- `vocab_size`
- `d_model`
- `n_heads`
- `n_layers`
- `mlp_dim`
- `learning_rate`
- `seed`
- `include_structure`
- `data_contract`

For MLX/Metal rows, the software identity must also include matching
`framework`, `backend`, `python_version`, `mlx_version`, `mlx_lm_version`, and
`mlx_metal_version`. For Torch/CUDA rows, include matching `framework`,
`backend`, `torch_version`, `cuda_version`, `driver_version`, `device_name`, and
`device_capability`.

## GB10 Quarter Reference Is Not An MLX Match

The sibling CUDA/Megatron GB10 debug lane in `../cppmega` currently documents
the local quarter profile as:

```text
B = 4
S = 4096
H = 3584
Heads = 28
V = 65536
Main layers = 13
Pattern = *EME*EME*EMM*
MTP predictors = 2
```

That profile is useful as a shape target for future MLX work, but it is not a
matched M4 comparison row by itself. It runs through CUDA, Megatron,
Transformer Engine, and GB10-specific kernels, while this repo's local path runs
through MLX/Metal. A report may store both rows, but a parity statement requires
matching `comparison_key.workload` and `comparison_key.software`; otherwise the
right output is an unmatched/refused comparison.

## Producing Real Rows

Use the same matrix command on each host and keep the raw JSON. Do not edit the
keys by hand to force a match.

```bash
./.venv/bin/python scripts/bench_matrix.py \
  --hardware-label "M4 Max" \
  --batch-sizes 1,2 \
  --seq-lens 32,64 \
  --profiles smoke,tiny,hybrid-smoke \
  --routes plain,structure,mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager,compiled \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --json > runs/m4max_matrix.json

./.venv/bin/python scripts/bench_matrix.py \
  --hardware-label "GB10" \
  --batch-sizes 1,2 \
  --seq-lens 32,64 \
  --profiles smoke,tiny,hybrid-smoke \
  --routes plain,structure,mamba3,m2rnn,hybrid-aemr \
  --compile-modes eager,compiled \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --json > runs/gb10_matrix.json

./.venv/bin/python scripts/compare_bench_rows.py \
  --input runs/m4max_matrix.json \
  --input runs/gb10_matrix.json \
  --package-dir runs/gb10_matched_package > runs/matched_compare.json
```

If the final status is `no_matching_rows` or `insufficient_matched_rows`, keep
the output as refusal evidence and describe the mismatched keys instead of
reporting a ratio. A valid ratio appears only under `comparisons[*].ratios`.
The package directory is the durable handoff artifact: `manifest.json` records
the guard status and input paths, `compare_report.json` keeps the guard/refusal
summary without ratios, `matched_comparisons.jsonl` is the only ratio-bearing
JSONL file, and `refused_pairs.jsonl` is mismatch evidence only.

## Parquet Samples

Local Parquet samples under `data/parquet_samples/gb10/` are ignored by git and
are only data-contract smoke inputs. They can help produce the same
`data_contract` on both machines, but they do not replace a matched benchmark
receipt. A valid real-data comparison still needs both M4 and GB10 rows to name
the same Parquet-derived contract and to pass the comparison-key guard above.
