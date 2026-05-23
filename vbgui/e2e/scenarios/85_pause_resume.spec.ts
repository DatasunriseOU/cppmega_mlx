// V7-H06: pause / resume mid-train. UI button toggles between Pause
// and Resume, calls pipeline.pause / pipeline.resume RPCs, and the
// backend job_control module blocks the train loop until resumed.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V7-H06: pause button toggles label and disables when no run is active",
  async ({ page }) => {
    test.setTimeout(30_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // No train running yet → pause button visible but disabled.
    const pauseBtn = page.getByTestId("run-pipeline-pause");
    await expect(pauseBtn).toBeVisible();
    await expect(pauseBtn).toBeDisabled();
    await expect(pauseBtn).toHaveText(/Pause/i);
  });

test("V7-H06: long-running train can be paused then resumed",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    // Use enough steps so we have time to hit pause mid-run.
    await page.getByTestId("train-num-steps").fill("8");
    await page.getByTestId("run-pipeline-train").click();

    const pauseBtn = page.getByTestId("run-pipeline-pause");
    // Wait until trainInFlight engages — pause button becomes enabled.
    await expect(pauseBtn).toBeEnabled({ timeout: 10_000 });

    await pauseBtn.click();
    // Button label flips to Resume.
    await expect(pauseBtn).toHaveText(/Resume/i, { timeout: 5_000 });

    // After short hold, click Resume.
    await page.waitForTimeout(500);
    await pauseBtn.click();
    await expect(pauseBtn).toHaveText(/Pause/i, { timeout: 5_000 });

    // Eventually the modal appears with status ok.
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 120_000 });
    await closeModal(page);
  });
