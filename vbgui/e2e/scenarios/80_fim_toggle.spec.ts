// V7-G05: FIM (Fill-In-Middle) toggle in TopBar threads through
// stage_options.train.fim_enabled and surfaces extras.train.fim_active
// + fim_ratio. The visible-row assertion uses the StageExtras render
// in RunResultModal so this is a true "UI shows backend math" check,
// not just RPC payload echo.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V7-G05: FIM toggle surfaces fim_active=true + fim_ratio in train extras",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("train-fim-enabled").check();
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    const fimActive = await page
      .getByTestId("run-result-extras-train-fim_active").textContent();
    expect(fimActive).toBe("true");

    const fimRatioText = await page
      .getByTestId("run-result-extras-train-fim_ratio").textContent();
    expect(fimRatioText).not.toBe("null");
    const fimRatio = parseFloat(fimRatioText ?? "0");
    expect(fimRatio).toBeGreaterThan(0);
    expect(fimRatio).toBeLessThanOrEqual(1);

    await closeModal(page);
  });

test("V7-G05: FIM off by default — fim_active=false",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    // Do not check train-fim-enabled.
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    const fimActive = await page
      .getByTestId("run-result-extras-train-fim_active").textContent();
    expect(fimActive).toBe("false");

    await closeModal(page);
  });
