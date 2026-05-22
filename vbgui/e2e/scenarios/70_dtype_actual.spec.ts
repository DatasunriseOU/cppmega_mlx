// H16: toggle mixed_precision / fp8 in TopBar → Train → assert
// extras.dtype_actual reflects the post-cast dtype (not just the
// requested string). Closes the V5-G07 honesty gap where
// master_dtype/train_dtype/fp8_active were string echos only.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H16: mixed_precision=False → master_dtype_actual is bf16",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("top-bar-mixed-precision").uncheck();
    await page.waitForTimeout(300);
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const actual = ((await page.getByTestId(
      "run-result-extras-train-dtype_actual-master_dtype_actual")
      .textContent()) ?? "").trim();
    // mlx stringifies as "mlx.core.bfloat16".
    expect(actual.toLowerCase()).toContain("bfloat16");
    await closeModal(page);
  });

test("H16: fp8_enabled=True → fp8_attempted true + fallback reason",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("top-bar-fp8-enabled").check();
    await page.waitForTimeout(300);
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const fp8Attempted = ((await page.getByTestId(
      "run-result-extras-train-dtype_actual-fp8_attempted")
      .textContent()) ?? "").trim().toLowerCase();
    expect(fp8Attempted).toBe("true");
    // On Apple Silicon mlx build mx.float8 is absent → fallback reason
    // is a non-empty string. (If a future build adds fp8 support this
    // assertion will need to relax to "either succeeded or fell back".)
    const reason = ((await page.getByTestId(
      "run-result-extras-train-dtype_actual-fp8_fallback_reason")
      .textContent()) ?? "").trim();
    expect(reason.length).toBeGreaterThan(0);
    expect(reason).not.toBe("null");
    await closeModal(page);
  });
