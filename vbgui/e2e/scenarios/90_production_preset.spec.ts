// V7-P5: E2E Playwright test verifying that production-scale presets (H=4096)
// can be loaded and validated analytically through the visual builder.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickRunPipeline, closeModal } from "../fixtures";

test("V7-P5: Snap to llama3_8b production scale in DimEnvEditor and validate through UI", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);

  // 1. Select the llama3_8b production scale in DimEnvEditor
  await page.getByTestId("dim-env-preset").selectOption("llama3_8b");
  await page.waitForTimeout(500);

  // 2. Select the llama3_8b preset dropdown to drop it on the canvas
  await selectPreset(page, "llama3_8b");
  await page.waitForTimeout(1000);

  // 3. Trigger the full pipeline run
  const modal = await clickRunPipeline(page, "full");
  await expect(modal).toBeVisible();

  // 4. Assert that all verification stages complete successfully with "ok"
  await expect(modal.getByTestId("run-result-stage-parse")).toContainText("ok");
  await expect(modal.getByTestId("run-result-stage-verify_build_spec")).toContainText("ok");
  await expect(modal.getByTestId("run-result-stage-resolve_shapes")).toContainText("ok");
  await expect(modal.getByTestId("run-result-stage-estimate_memory")).toContainText("ok");

  // 5. Expand verification details and check the H=4096 estimate values
  await page.getByTestId("run-result-expand-estimate_memory").click();
  const estimateText = await page.getByTestId("run-result-extras-estimate_memory-total_bytes").textContent();
  const estimateBytes = parseInt(estimateText?.trim() ?? "0", 10);

  // A 1-unit llama3_8b at H=4096 has > 200 million parameters (approx 400MB in FP16/BF16)
  expect(estimateBytes).toBeGreaterThan(200_000_000);

  await closeModal(page);
});
