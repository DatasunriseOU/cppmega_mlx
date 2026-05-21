// G07: precision toggles reach extras (propagation-only for v5;
// real dtype switching deferred to v6+).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("G07: extras.train_dtype + master_dtype + fp8_active populated",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    const trainDtype = await page.getByTestId(
      "run-result-extras-train-train_dtype").textContent();
    const masterDtype = await page.getByTestId(
      "run-result-extras-train-master_dtype").textContent();
    const fp8 = await page.getByTestId(
      "run-result-extras-train-fp8_active").textContent();

    expect(["bf16", "fp16", "fp32"]).toContain(trainDtype?.trim());
    expect(["bf16", "fp16", "fp32"]).toContain(masterDtype?.trim());
    expect(["true", "false"]).toContain(fp8?.trim().toLowerCase());

    await closeModal(page);
  });
