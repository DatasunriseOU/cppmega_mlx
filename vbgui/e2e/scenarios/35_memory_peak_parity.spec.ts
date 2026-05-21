// G06: stage_train brackets train loop with reset_peak_memory +
// get_peak_memory; extras.memory_peak_bytes populated.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("G06: extras.memory_peak_bytes populated and > 0", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();
  const peakText = await page.getByTestId(
    "run-result-extras-train-memory_peak_bytes").textContent();
  const peak = parseInt(peakText?.trim() ?? "0", 10);
  // M1/M2/M3 Mac Metal backend reports peak memory; should be at
  // least 1MB after a 2-step llama3_8b train with 64-hidden bricks.
  expect(peak).toBeGreaterThan(1_000_000);
  await closeModal(page);
});
