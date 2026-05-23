// V7-D06: master_dtype dropdown renders real ms/token cost from
// dtype.cost_estimate RPC. Honest-closure: previously the bench script
// produced a static CSV/HTML report but the live UI showed only dtype
// names — now each option carries " · X.XXXX ms/tok" next to its label.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("V7-D06: master_dtype options show ms/tok cost after dropdown opens",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    // Probe fires once when the menu first opens — wait for either
    // measured indicator OR the loading indicator to disappear.
    await expect(page.getByTestId("dtype-cost-summary"))
      .toBeVisible({ timeout: 30_000 });

    const select = page.getByTestId("top-bar-precision-mode");
    const fp32Text = await select.locator("option[value='fp32']").textContent();
    const bf16Text = await select.locator("option[value='bf16']").textContent();
    const fp16Text = await select.locator("option[value='fp16']").textContent();
    // Each supported dtype carries the ms/tok suffix.
    expect(fp32Text).toMatch(/ms\/tok/);
    expect(bf16Text).toMatch(/ms\/tok/);
    expect(fp16Text).toMatch(/ms\/tok/);
  });
