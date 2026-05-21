// G23: gradient_clip_norm activation observable in extras.gradient_clip.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("G23: tight clip (0.001) triggers num_clips > 0 in extras", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("sidebar-tab-optim").click();
  await page.getByTestId("optim-clip").fill("0.001");
  await page.getByTestId("optim-apply").click();

  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("4");
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();

  const numClips = parseInt(
    (await page.getByTestId("run-result-extras-train-gradient_clip-num_clips")
      .textContent()) ?? "0", 10);
  const threshold = parseFloat(
    (await page.getByTestId("run-result-extras-train-gradient_clip-threshold")
      .textContent()) ?? "0");
  expect(threshold).toBeCloseTo(0.001, 6);
  expect(numClips).toBeGreaterThan(0);

  await closeModal(page);
});
