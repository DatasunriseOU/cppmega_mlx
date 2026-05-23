// G06: stage_train brackets train loop with reset_peak_memory +
// get_peak_memory; extras.memory_peak_bytes populated.
// V7-I02: Tightened parity verification between verify-time worst_rank estimate
// and actual measured peak Metal HBM (ratio < 4.0x) on single-device topology.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("G06/V7-I02: memory peak within 4.0x of estimate on matched topology", async ({ page }) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Select a matched single-device topology (world_size=1) so estimate and actual run are aligned
  await page.getByTestId("topology-selector").selectOption("m3_ultra_solo");
  await page.waitForTimeout(500);

  // Pre-Train: wait for estimate to settle
  await expect.poll(async () => {
    const raw = await page.getByTestId("memory-bar-estimate").getAttribute("data-bytes");
    return raw == null ? 0 : parseInt(raw, 10);
  }, { timeout: 8_000 }).toBeGreaterThan(0);

  // Trigger N=2 steps training
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("2");
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await closeModal(page);

  // Read both values from the MemoryBar dual display
  const estText = await page.getByTestId("memory-bar-estimate").getAttribute("data-bytes");
  const actText = await page.getByTestId("memory-bar-actual").getAttribute("data-bytes");

  const estimate = parseInt(estText ?? "0", 10);
  const actual = parseInt(actText ?? "0", 10);

  expect(estimate).toBeGreaterThan(0);
  expect(actual).toBeGreaterThan(0);

  // Tight parity bound: ratio is strictly less than 4.0x
  const ratio = Math.max(actual, estimate) / Math.min(actual, estimate);
  expect(ratio).toBeLessThan(4.0);

  await closeModal(page);
});
