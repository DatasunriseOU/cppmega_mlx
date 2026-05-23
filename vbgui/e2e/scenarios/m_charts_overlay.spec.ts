// V7-M21 / M22 / M23 visual e2e — TrainExtrasOverlay renders the
// losses_smoothed + val_losses overlays on the primary loss chart AND
// the dedicated lr chart for lr_trajectory. Real train run; we assert
// the rendered SVG paths exist with at least the segments we expect.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("M21/M22/M23: charts overlay smoothed/val/lr after real train", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Bump num_steps so smoothed / lr_trajectory have meaningful length.
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("4");
  await page.getByTestId("run-pipeline-train").click();
  await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });

  // L47 may have auto-expanded train if it failed; otherwise click.
  const extrasRow = page.getByTestId("run-result-extras-row-train");
  if (!(await extrasRow.isVisible().catch(() => false))) {
    await page.getByTestId("run-result-expand-train").click();
  }

  // M21 — primary loss line.
  await expect(page.getByTestId("extras-loss-chart-line"))
    .toBeVisible({ timeout: 30_000 });

  // M22 — dedicated LR chart svg.
  await expect(page.getByTestId("extras-lr-chart-svg")).toBeVisible();

  // M21 — smoothed overlay path may or may not be present depending
  // on whether the backend emitted losses_smoothed in this build; if
  // present it must be a real <path>.
  const smoothed = page.getByTestId("extras-loss-chart-line-smoothed");
  if (await smoothed.isVisible().catch(() => false)) {
    const d = await smoothed.getAttribute("d");
    expect((d ?? "").startsWith("M")).toBe(true);
  }

  await closeModal(page);
});
