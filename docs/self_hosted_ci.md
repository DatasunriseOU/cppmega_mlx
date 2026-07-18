# Self-hosted CI orchestration

`cppmega_mlx` uses GitHub Actions only as a coordinator. Every job with a
`runs-on` key targets a repository-owned runner; hosted macOS, Linux, and
Windows labels are rejected by `tests/test_workflow_runner_policy.py`.

The registration pattern comes from the live `cloud_love` setup:

- workflows target stable custom labels in addition to `self-hosted` and the
  operating-system label;
- runner registration tokens are temporary GitHub secrets consumed by
  `cloud_love/.github/workflows/bootstrap-cppmega-runners.yml`;
- no runner token, SSH password, private key, or other credential is stored in
  this repository;
- all external workflow actions are pinned to full commit SHAs.

## Verified inventory

Snapshot taken 2026-07-14. Online/busy state is live data and must be refreshed
before an operational claim:

| Inventory ID | Address | Role | GitHub runner and labels | Direct status at verification |
|---|---:|---|---|---|
| `mac-studio` | `10.0.0.8` | required macOS MLX | `mac-studio-cppmega-mlx`; `self-hosted,macOS,ARM64,cppmega-mlx-macos` | local and available |
| `legion-linux` | `10.0.0.16` | required portable Linux | `davidgor-Legion-R9000P-ARX8-cppmega-mlx`; `self-hosted,Linux,X64,cppmega-mlx` | TCP/22 reachable; batch SSH authentication unavailable from the Mac Studio |
| `windows-10-0-0-11` | `10.0.0.11` | inventory only | no `cppmega_mlx` runner | Windows is not eligible for either lane; batch SSH authentication unavailable |
| `untrusted-10-0-0-12` | `10.0.0.12` | quarantined inventory only | none | TCP/22 reachable, but the current host key does not match the saved key |

Do not bypass the `10.0.0.12` host-key error with
`StrictHostKeyChecking=no`. Verify the replacement machine and fingerprint out
of band, update `known_hosts`, and only then assign it a test lane.

Refresh GitHub's runner view without printing credentials:

```bash
gh api repos/DatasunriseOU/cppmega_mlx/actions/runners \
  --jq '.runners[] | {name,os,status,busy,labels:[.labels[].name]}'
```

## Shared suites

Both GitHub jobs and the direct orchestrator call
`scripts/run_self_hosted_ci.py lane`:

- `macos-mlx` runs the focused data, model, inference, evaluation, and
  orchestration policy tests with the established local MLX environment.
- `linux-portable` creates an isolated temporary venv, installs only the
  already-established portable test dependencies, runs tests with
  `--noconftest`, compiles portable scripts, and never imports Apple's MLX
  runtime.

Each process is started in its own process group. A timeout terminates the
whole group, returns exit code 124, and records `timed_out` instead of leaving
children behind. Each step has its own log and the lane always writes
`receipt.json` with host, source commit, timing, command, status, and exit code.
The workflows upload these directories with 14-day retention even when a lane
fails.

## Direct CLI

The direct path does not use GitHub Actions. It probes all selected hosts with
non-interactive SSH, aborts before dispatch if any required host is
unavailable, stages the exact requested Git commit, runs the same lane entry
point, and collects each host's receipt and logs.

Read-only probe and plan:

```bash
python3 scripts/run_self_hosted_ci.py orchestrate \
  --dry-run \
  --receipt-dir /tmp/cppmega-mlx-self-hosted
```

Run only the local macOS lane:

```bash
python3 scripts/run_self_hosted_ci.py orchestrate \
  --host mac-studio \
  --ref HEAD \
  --receipt-dir /tmp/cppmega-mlx-self-hosted
```

Run the full matrix after key-based SSH to `10.0.0.16` is available:

```bash
python3 scripts/run_self_hosted_ci.py orchestrate \
  --ref HEAD \
  --receipt-dir /tmp/cppmega-mlx-self-hosted
```

The CLI never accepts a password or token argument. SSH uses `BatchMode=yes`,
disables password and keyboard-interactive authentication, and preserves normal
host-key verification. Configure an agent-backed SSH key outside the
repository. The source worktree may be dirty; dispatch still uses the exact
resolved commit and records `source_dirty` in the orchestration receipt.

Receipts are written under:

```text
<receipt-dir>/<run-id>/orchestration.json
<receipt-dir>/<run-id>/<host-id>/probe.json
<receipt-dir>/<run-id>/<host-id>/receipt.json
<receipt-dir>/<run-id>/<host-id>/*.log
```

Exit code 2 means orchestration/preflight was blocked, exit code 1 means a
dispatched lane failed, and exit code 0 means every selected required lane
passed (or a dry-run found every selected required host available).

## Preset shard incident

PR run `29301626406`, job `86986403882`, proved that increasing shard1 to 40
minutes was not a fix: setup finished at `03:00:37Z`, the Playwright step ran
until `03:36:05Z`, and the job hit its 40-minute cap. The matrix is one spec
file and the base config has `fullyParallel: false`, so Playwright's file-level
sharding assigned all 912 tests to shard 1 while the other shards had no tests.

The self-hosted workflow now passes `--fully-parallel`, retaining four shards
that contain test cells rather than files. Playwright has a 12-minute global
cap and each job has a 20-minute cap, leaving setup and artifact-upload
headroom while still producing logs on a test timeout. Corrected run
`29306678792` also exposed a 1.9 GB `setup-node` cache-save attempt after shard
2 had passed; it spent 101 seconds packing the cache and then failed to reserve
the key. GitHub's npm cache hook is disabled for these persistent runners;
npm's host-local cache remains available to `npm ci` without a remote post-job
upload.
