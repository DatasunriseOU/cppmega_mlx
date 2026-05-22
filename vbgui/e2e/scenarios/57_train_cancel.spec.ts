// H03: Train cancellation reaches backend pipeline.abort and returns a
// cancelled pipeline report with partial train extras.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H03: Cancel aborts in-flight N=64 Train and shows partial losses",
  async ({ page }) => {
    test.setTimeout(90_000);
    await gotoApp(page);
    await selectPreset(page, "tiny_aya");

    await expect(page.getByTestId("run-pipeline-cancel")).toBeDisabled();
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("64");
    await page.getByTestId("run-pipeline-train").click();

    const cancel = page.getByTestId("run-pipeline-cancel");
    await expect(cancel).toBeEnabled();
    await page.waitForTimeout(100);
    await cancel.click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await expect(page.getByTestId("run-result-overall"))
      .toContainText("cancelled");
    await expect(page.getByTestId("run-result-stage-train"))
      .toContainText("cancelled");

    await page.getByTestId("run-result-expand-train").click();
    await expect(page.getByTestId("run-result-extras-train-aborted"))
      .toHaveText("true");
    const steps = Number(await page.getByTestId(
      "run-result-extras-train-num_steps").textContent());
    expect(steps).toBeGreaterThan(0);
    expect(steps).toBeLessThan(64);

    const losses = page.locator(
      "[data-testid^='run-result-extras-train-losses-']");
    await expect(losses).toHaveCount(steps);
    await closeModal(page);
  });
