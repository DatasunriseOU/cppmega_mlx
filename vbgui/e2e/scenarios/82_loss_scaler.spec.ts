// V7-D03: LossScaler integration into stage_train with fp16 master_dtype.
// Honest-closure: the LossScaler class existed but was not engaged. After
// wiring, master_dtype=fp16 surfaces extras.train.loss_scaler with
// {mode, scale, overflow_count, overflow_steps, clean_steps_since_overflow}
// — visible in the StageExtras DOM render so the UI shows the math.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V7-D03: fp16 master_dtype engages LossScaler and reports snapshot",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("top-bar-precision-mode").selectOption("fp16");
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    // extras.train.loss_scaler is a nested object; the StageExtras
    // recursive renderer produces -loss_scaler-<field> testids.
    const scalerBlock = page.getByTestId("run-result-extras-train-loss_scaler");
    await expect(scalerBlock).toBeVisible();

    const mode = await page.getByTestId(
      "run-result-extras-train-loss_scaler-mode").textContent();
    expect(["dynamic", "static"]).toContain(mode);

    const overflowCount = await page.getByTestId(
      "run-result-extras-train-loss_scaler-overflow_count").textContent();
    const overflowN = parseInt(overflowCount ?? "-1", 10);
    expect(overflowN).toBeGreaterThanOrEqual(0);

    const scaleText = await page.getByTestId(
      "run-result-extras-train-loss_scaler-scale").textContent();
    expect(parseFloat(scaleText ?? "0")).toBeGreaterThan(0);

    await closeModal(page);
  });

test("V7-D03: bf16 master_dtype leaves loss_scaler=null",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("top-bar-precision-mode").selectOption("bf16");
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    // The non-object null renders as text "null" via ExtrasEntry.
    const scalerText = await page.getByTestId(
      "run-result-extras-train-loss_scaler").textContent();
    expect(scalerText).toBe("null");

    await closeModal(page);
  });
