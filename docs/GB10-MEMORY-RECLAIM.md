# gb10 — reclaiming "stuck" memory WITHOUT a reboot

On the GB10 (Grace‑Blackwell, unified LPDDR5X), GPU allocations are carved out of
system RAM. When a CUDA process is `kill -9`'d **mid‑allocation**, the runtime
never releases the device memory, so `free` keeps showing it as **used** even
though no process appears to own it. This looks like a 100 GB leak and a near‑frozen
box (`avail` collapses to single‑digit GB).

## Why `ps`/`pgrep`/`nvidia-smi` don't find it

- The leaked memory is **device‑mapped**, not anonymous: `grep AnonPages /proc/meminfo`
  stays tiny (e.g. 495 MB) while `free` shows ~100 GB used. `ps --sort=-rss` shows
  nothing large (top process ~0.2 GB).
- `nvidia-smi --query-compute-apps` / FB memory report **N/A** on GB10 (unified mem).
- A pattern `pkill -f m04_train_step` can **miss** the holder: a `timeout 1800 python …`
  wrapper or a re‑exec'd child has a different cmdline, so the orphan survives.

## The reclaim recipe (no reboot, no `--gpu-reset`)

1. Find the real holder of CUDA managed memory — it lives on **`/dev/nvidia-uvm`**,
   not `/dev/nvidia0`:
   ```sh
   ssh gb10 'fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools'
   ```
   This lists PIDs that `ps`/`pgrep`‑by‑name do not. The anomalous `python` entry
   (next to legit Xorg/gnome/firefox display holders) is the orphan.
2. Kill it **cleanly** so the CUDA runtime releases the allocation:
   ```sh
   ssh gb10 'kill -TERM <pid>'      # SIGTERM first — lets CUDA free device mem
   # only escalate to kill -9 if TERM fails; a -9 mid-alloc is what leaks in the first place
   ```
   Memory returns to `free` within a few seconds once the true holder dies.
3. Drop the page cache — reclaims **buff/cache** (file-backed pages), NOT memory
   held by a live process:
   ```sh
   ssh gb10 "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
   ```

## Two distinct cases — pick the right tool

The `free` "used" can stay high for **two different reasons**; the fix differs:

| symptom | cause | fix |
|---|---|---|
| `fuser -v /dev/nvidia*` shows an **anomalous python/m04** PID (next to Xorg/gnome/firefox); `nvidia_uvm` refcount > 0 | a live/orphan CUDA process still **holds** GPU memory (device-mapped; `ps`/`nvidia-smi` may not show it) | `kill -TERM <pid>` the holder (step 2). Memory returns when it dies. |
| `fuser` shows **no** compute holder (even under `sudo`); `nvidia_uvm` refcount **0**; `AnonPages` tiny; but `used` is tens of GB | **page-cache accumulation** — e.g. a fresh-process-per-cell sweep where each process re-reads weights/parquet/tilelang artifacts from disk | `sync; echo 3 > /proc/sys/vm/drop_caches` (step 3). Frees it instantly. |

Observed both in one session: a `kill -9`'d batch=4 m04 left an orphan **holding** ~100 GB
(fixed by `fuser`+kill); a 54-cell fresh-process regen left ~60 GB of **page cache** with
no holder (fixed by `drop_caches`, which dropped 65 G → 5 G). Check `fuser` first to tell
which case you're in.

## Verify

```sh
ssh gb10 'free -g | awk "/^Mem:/{print \"used=\"\$3\"G avail=\"\$7\"G\"}"'
# healthy gb10 idle baseline: used≈5-6G, avail≈115-119G of 121G
```

## Prevention

- Never `kill -9` a CUDA process unless `kill -TERM` failed. SIGTERM lets MLX/torch
  tear down the context and release device memory; SIGKILL mid‑alloc leaks it.
- After any kill, confirm with `fuser -v /dev/nvidia-uvm` that no orphan remains,
  then re‑check `free`.
- Memory discipline for benchmarks on this box: ramp slowly (start bs=1 seq=512),
  sample `free -g` live, and abort if used grows past **70 GB** — `bs=4 × seq=4096`
  (= 16,384 tok/step, 32× the bs=1/seq=512 footprint) does not fit this config in
  MLX‑eager and must not be forced.
