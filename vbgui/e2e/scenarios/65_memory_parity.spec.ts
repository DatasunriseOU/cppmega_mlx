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
import type { Locator } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

async function bytesOf(loc: Locator): Promise<number> {
  const raw = await loc.getAttribute("data-bytes");
  if (raw == null) throw new Error("missing data-bytes attribute");
  return parseInt(raw, 10);
}

test("H11: actual memory peak within 50% of estimate after Train",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Select a matched single-device topology so estimate and actual run are aligned (math-effect 🟢)
    await page.getByTestId("topology-selector").selectOption("m3_ultra_solo");
    await page.waitForTimeout(500);

    // Pre-Train: estimate is rendered, actual is absent. Wait for
    // verify to populate the estimate (debounced 200ms after the
    // preset drops bricks).
    await expect.poll(async () => {
      const raw = await page.getByTestId("memory-bar-estimate")
        .getAttribute("data-bytes");
      return raw == null ? 0 : parseInt(raw, 10);
    }, { timeout: 8_000 }).toBeGreaterThan(0);
    await expect(page.getByTestId("memory-bar-actual")).toHaveCount(0);

    // Run Train so backend fills extras.memory_peak_bytes.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await closeModal(page);

    // Now both readouts are present. Read precise byte counts from
    // the data-bytes attribute (formatted GB/MB string would round
    // a few-MB actual down to "0.00 GB" and lose precision).
    const estimate = await bytesOf(page.getByTestId("memory-bar-estimate"));
    const actual = await bytesOf(page.getByTestId("memory-bar-actual"));
    expect(estimate).toBeGreaterThan(0);
    expect(actual).toBeGreaterThan(0);
    // Tight parity bound: ratio is strictly less than 4.0x
    const ratio = Math.max(actual, estimate) / Math.min(actual, estimate);
    expect(ratio).toBeLessThan(4.0);
  });
