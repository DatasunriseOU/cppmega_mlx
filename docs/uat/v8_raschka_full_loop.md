# UAT — V8 Raschka Full-Loop (10-step user journey)

> Manual acceptance script. Walks a real user through the full
> preset → train loop. Take a screenshot at every checkpoint. The
> automated counterpart is `vbgui/e2e/scenarios/v8_raschka_full_loop.spec.ts`.

## Pre-flight

- Start the backend: `uvicorn cppmega_v4.jsonrpc.server:app --port 8767`.
- Start the frontend: `pnpm -C vbgui dev`.
- Open `http://127.0.0.1:5176/`.

## Step 1 — Pick a preset (R01)

1. Click the **Preset launcher** dropdown in the top bar.
2. Select `llama3_8b`.

**Expected**:

- Canvas populates with the preset's brick chain (≥ 2 nodes).
- Switch to the **Optim** sidebar tab — `kind=adamw`,
  `optim-clip` field shows `1.0`, `optim-mp` checked, the first
  group's lr is `0.0003`.
- Click the group's clock icon → ScheduleEditor opens with `kind=wsd`.

## Step 2 — Scale down (R02)

1. Click the **Gallery** tab.
2. The `GalleryScaleDownSlider` panel appears above the per-preset
   table. The picker says `llama3_8b`.
3. Drag the slider down until the target reads `1.00 GB`.

**Expected**:

- `gallery-scaledown-est-bytes` shows a value < 1 GB after a 250 ms
  pause.
- `gallery-scaledown-shape` reads `H=512 L=32 (from H=4096 L=32)`.
- `gallery-scaledown-fits` chip is green and says "fits budget".

4. Click **Apply scaled preset**.

**Expected**: canvas swaps to the scaled chain (64 nodes for 32
layers of `attn+mlp`).

## Step 3 — Verify memory (R03)

1. Click the **Memory** sidebar tab.

**Expected**:

- A 4×5 grid renders within ~200 ms.
- Most cells under `bf16/fp8/mxfp4` are green (`fits`).
- Hover a cell — a tooltip lists every component byte count.

## Step 4 — Auto-fit (R04)

1. Back to the **Gallery** tab.
2. Click **Auto-fit to my devbox**.

**Expected**:

- A blue chip appears below the slider:
  `gb10_quarter · hidden=4096, layers=32, axis=dp×1, peak=… GB / … GB`
  (or your local topology).

## Step 5 — Inject MTP (R08)

1. Click the **Canvas** tab.
2. In the yellow `FeatureInjectionBar`, select `mtp_weighted`.
3. Click **Apply**.

**Expected**: the applied-list at the right of the bar shows
`mtp_weighted`. Open the **Rewriters** sidebar tab — `MTPRewriter`
appears with `K=2, weight=0.5`.

## Step 6 — Data quickstart (R09)

1. Click the **Data** tab.
2. Click **HF quickstart** in the header.

**Expected**: modal opens; dataset field shows
`HuggingFaceFW/fineweb-edu`; n_tokens shows `8192`.

3. Click **Run**.

**Expected**: within ~30 s the modal shows a green result block with
the parquet path and `n tokens, n docs, … ms` counters.

4. Click **Close**. The DataInspector path field is now the new
   parquet path.

## Step 7 — Use for training

1. Click **Use for training** next to the path. The button turns
   green (`✓ Training`).

## Step 8 — Run 4 steps

1. Click **Run pipeline** in the top bar → choose **Train**.
2. Set `num_steps` to `4`.
3. Click **Train**.

**Expected**:

- `LiveTrainPanel` appears with a sparkline that fills in 4 dots.
- The toast fires when training completes.
- The `extras.train` overlay on the run-result modal shows
  `preset_origin: llama3_8b`, `scale_down_factor < 1.0`,
  `feature_injections: [mtp_weighted]`, `data_source.kind:
  hf_quickstart`.

## Step 9 — Inspect outputs

Open the run-history menu (top-right of the canvas) and confirm the
run lands with green status, with the v8 extras visible in the
overlay.

## Step 10 — Reset

Refresh the page. Cycle through Steps 1-2 with a different preset
(e.g. `qwen3_dense_4b`) to confirm defaults switch and the loop
remains stable.

## Pass / fail

This UAT passes if every "Expected" block above is satisfied without
any browser-console error in `chrome://devtools/console` or the
backend uvicorn log.
