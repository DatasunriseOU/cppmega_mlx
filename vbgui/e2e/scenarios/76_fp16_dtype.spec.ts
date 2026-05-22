// H23: TopBar precision dropdown → fp16 → Train extras report
// master_dtype="fp16" and train_dtype="fp16".

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H23: top-bar-precision-mode=fp16 → extras.train_dtype=='fp16'",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("top-bar-precision-mode").selectOption("fp16");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const td = ((await page.getByTestId(
      "run-result-extras-train-train_dtype").textContent()) ?? "").trim();
    expect(td).toBe("fp16");
    const md = ((await page.getByTestId(
      "run-result-extras-train-master_dtype").textContent()) ?? "").trim();
    expect(md).toBe("fp16");
    await closeModal(page);
  });
