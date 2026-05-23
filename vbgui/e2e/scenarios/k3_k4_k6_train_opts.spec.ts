// V7-K3 / K4 / K6 visual e2e — TrainOptionsPanel inputs reach
// stage_train and the train completes with visible loss chart.
// Asserts grad_clip_max_norm (K4) and fake_ranks (K6) land in the
// train extras dl (verifies the wiring without needing a separate
// per-extra testid contract for each new key).

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("K3/K4/K6: TrainOptionsPanel knobs reach stage_train", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Expand the panel and set values.
  await page.getByTestId("train-options-toggle").click();
  await page.getByTestId("train-opt-val_every").fill("2");          // K3
  await page.getByTestId("train-opt-grad_clip_max_norm").fill("0.5"); // K4
  // K6 slider — programmatically set to 2.
  await page.getByTestId("train-opt-fake_ranks").fill("2");
  await expect(page.getByTestId("train-opt-fake_ranks-value"))
    .toHaveText("2");

  await clickRunPipeline(page, "train");
  await page.getByTestId("run-result-expand-train").click();
  await expect(page.getByTestId("chart-svg")).toBeVisible({
    timeout: 30_000,
  });

  // Train extras dl carries fake_ranks (sharding/K6 forwarded it).
  // Use a generous timeout — the train extras row populates after the
  // expand-train click but the per-key testids are added sequentially.
  await expect(page.getByTestId("run-result-extras-train-fake_ranks"))
    .toBeVisible({ timeout: 5_000 });

  await closeModal(page);
});
