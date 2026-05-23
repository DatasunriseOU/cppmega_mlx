// V7-K1 visual e2e — num_steps input in TopBar feeds stage_train.
// Real backend roundtrip: set num_steps=3 in TopBar, train llama3_8b,
// assert the visual LossChart has exactly 3 chart-point-{i} circles.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("K1: TopBar num_steps drives extras.losses length visually", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Open the run-pipeline dropdown and set num_steps to 3.
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("3");

  // Trigger the train mode directly without re-opening (the
  // clickRunPipeline helper re-opens the toggle; doing it manually
  // keeps the typed value).
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();

  // Exactly 3 visible chart-point circles → losses array is length 3.
  await expect(page.getByTestId("chart-point-0")).toBeVisible();
  await expect(page.getByTestId("chart-point-1")).toBeVisible();
  await expect(page.getByTestId("chart-point-2")).toBeVisible();
  await expect(page.locator("[data-testid^='chart-point-']"))
    .toHaveCount(3);

  // All 3 points carry finite data-loss-value attributes.
  for (const i of [0, 1, 2]) {
    const v = await page.getByTestId(`chart-point-${i}`)
      .getAttribute("data-loss-value");
    expect(Number.isFinite(Number(v))).toBe(true);
  }

  await closeModal(page);
});
