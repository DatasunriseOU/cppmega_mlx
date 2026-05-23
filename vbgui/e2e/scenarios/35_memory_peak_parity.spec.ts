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

  // Wait for initial estimate to settle first to avoid stale-response race conditions
  await expect.poll(async () => {
    const raw = await page.getByTestId("memory-bar-estimate").getAttribute("data-bytes");
    return raw == null ? 0 : parseInt(raw, 10);
  }, { timeout: 30_000 }).toBeGreaterThan(0);

  // Select a matched single-device topology (world_size=1) so estimate and actual run are aligned
  await page.getByTestId("topology-selector").selectOption("m3_ultra_solo");
  await page.waitForTimeout(500);

  // Change custom axis DP degree to 1 to match the m3_ultra_solo topology
  await page.getByTestId("sidebar-tab-sharding").click();
  await page.getByTestId("sharding-axis-0-degree").fill("1");
  await page.waitForTimeout(1000);

  // Wait for the new topology's estimate to settle
  await expect.poll(async () => {
    const raw = await page.getByTestId("memory-bar-estimate").getAttribute("data-bytes");
    return raw == null ? 0 : parseInt(raw, 10);
  }, { timeout: 30_000 }).toBeGreaterThan(0);

  // Trigger N=2 steps training
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("2");
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });

  // Read both values from the run result modal to get apples-to-apples high-fidelity values
  await page.getByTestId("run-result-expand-estimate_memory").click();
  const estText = await page.getByTestId("run-result-extras-estimate_memory-estimated_peak_bytes").textContent();

  await page.getByTestId("run-result-expand-train").click();
  const actText = await page.getByTestId("run-result-extras-train-memory_peak_bytes").textContent();

  const estimate = parseInt(estText?.trim() ?? "0", 10);
  const actual = parseInt(actText?.trim() ?? "0", 10);

  expect(estimate).toBeGreaterThan(0);
  expect(actual).toBeGreaterThan(0);

  // Tight parity bound: ratio is strictly less than 4.0x
  const ratio = Math.max(actual, estimate) / Math.min(actual, estimate);
  expect(ratio).toBeLessThan(4.0);

  await closeModal(page);
});
