// V7-P8: multi-epoch corpus via UI. The default smoke corpus has a
// tiny tokens-per-batch count, so a modest num_steps already covers
// > 1 epoch. We pick the dev_128 dim_env preset (S=512, B=1) and
// num_steps=32 — given the smoke iterator's bounded byte budget,
// that wraps the dataset at least once. The visible chart proves
// the wrap didn't blow up.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, closeModal,
} from "../fixtures";

test("P8: multi-epoch wrap via dev_128 preset + 32 train steps", async ({
  page,
}) => {
  test.setTimeout(240_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await page.getByTestId("dim-env-preset").selectOption("dev_128");

  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("32");
  await page.getByTestId("run-pipeline-train").click();
  await page.getByTestId("run-result-modal").waitFor({ timeout: 120_000 });

  const extras = page.getByTestId("run-result-extras-row-train");
  if (!(await extras.isVisible().catch(() => false))) {
    await page.getByTestId("run-result-expand-train").click();
  }

  // At least 8 visible chart points (subset of 32; smoke renders a
  // subset for huge runs but always emits >=2 + every-N anchor).
  let pointCount = 0;
  for (let i = 0; i < 32; i++) {
    const p = page.getByTestId(`extras-loss-chart-point-${i}`);
    if (await p.isVisible().catch(() => false)) pointCount++;
  }
  expect(pointCount).toBeGreaterThanOrEqual(8);

  // No NaN/Inf across visible points.
  for (let i = 0; i < 32; i++) {
    const p = page.getByTestId(`extras-loss-chart-point-${i}`);
    if (!(await p.isVisible().catch(() => false))) continue;
    const v = Number(await p.getAttribute("data-loss-value"));
    expect(Number.isFinite(v)).toBe(true);
  }
  await closeModal(page);
});
