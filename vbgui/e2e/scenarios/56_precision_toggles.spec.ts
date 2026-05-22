// H02: TopBar precision toggles flip extras.{master_dtype,fp8_active}.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H02: toggle mixed_precision OFF → extras.master_dtype=='bf16'",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("top-bar-mixed-precision").uncheck();
    await page.waitForTimeout(300);
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    const master = await page.getByTestId(
      "run-result-extras-train-master_dtype").textContent();
    expect(master?.trim()).toBe("bf16");
    await closeModal(page);
  });

test("H02: toggle fp8_enabled ON → extras.fp8_active=='true'",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("top-bar-fp8-enabled").check();
    await page.waitForTimeout(300);
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    const fp8 = await page.getByTestId(
      "run-result-extras-train-fp8_active").textContent();
    expect(fp8?.trim().toLowerCase()).toBe("true");
    await closeModal(page);
  });
