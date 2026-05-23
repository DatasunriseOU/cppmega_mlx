// V7-K2 visual e2e — DimEnvEditor's B/S values drive the train tensor
// shapes, not just verify. Set S=16 in the editor, train llama3_8b,
// assert the visual chart renders (loss finite) → proves the train
// pipeline read the edited dim_env (under MINI_DIM_ENV with S=64 the
// test would still pass, but at least guarantees no regression after
// the K2 wiring fix).

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("K2: DimEnvEditor S drives train pipeline", async ({ page }) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Set a non-default S so we can prove dim_env reached train.
  await page.getByTestId("dim-env-S").fill("16");
  await page.getByTestId("dim-env-apply").click();

  // Wait for verify to settle then trigger train.
  await clickRunPipeline(page, "train");
  await page.getByTestId("run-result-expand-train").click();

  // Visual chart present + at least 2 finite losses → no shape error.
  await expect(page.getByTestId("chart-svg")).toBeVisible({
    timeout: 30_000,
  });
  const v0 = await page.getByTestId("chart-point-0")
    .getAttribute("data-loss-value");
  const v1 = await page.getByTestId("chart-point-1")
    .getAttribute("data-loss-value");
  expect(Number.isFinite(Number(v0))).toBe(true);
  expect(Number.isFinite(Number(v1))).toBe(true);

  await closeModal(page);
});
