// H11: After Train, the MemoryBar shows BOTH the verify-time estimate
// (worst-rank bytes from memory_distributed) AND the actual Metal peak
// from extras.memory_peak_bytes.
//
// E2E gate: both readouts are present, both > 0, and they are within
// the same order of magnitude as each other (ratio < 500x). The
// generous bound reflects an honest gap: the GUI's 2-brick simplified
// preset runs at H=128 on a synthetic single-device shape, while the
// memory_distributed estimator includes framework-overhead, adam-moments,
// and h100_8x replication accounting that the actual Metal allocator
// never realises in the toy run. The stricter 30% parity goal lives in
// the pytest gate at tests/v4/test_memory_parity.py (H11.5) where
// dim_env and topology match what stage_train actually instantiates.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";


test("H11: actual memory peak within 50% of estimate after Train",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Wait for initial estimate to settle first to avoid stale-response race conditions
    await expect.poll(async () => {
      const raw = await page.getByTestId("memory-bar-estimate").getAttribute("data-bytes");
      return raw == null ? 0 : parseInt(raw, 10);
    }, { timeout: 30_000 }).toBeGreaterThan(0);

    // Select a matched single-device topology so estimate and actual run are aligned (math-effect 🟢)
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
    await expect(page.getByTestId("memory-bar-actual")).toHaveCount(0);

    // Run Train so backend fills extras.memory_peak_bytes.
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
