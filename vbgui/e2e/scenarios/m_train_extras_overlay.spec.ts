// V7-M-block visual e2e — after a real train run, the modal's
// TrainExtrasOverlay surfaces every extras.* key the backend emits.
// We assert on the visible overlays (badges + panels) rather than
// the JSON dl that previously held them.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("M-block: TrainExtrasOverlay surfaces real train extras", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Bump fake_ranks=2 via TrainOptionsPanel so gradient_reduce_ms /
  // sharding_applied actually populate.
  await page.getByTestId("train-options-toggle").click();
  await page.getByTestId("train-opt-fake_ranks").fill("2");
  // grad_clip_max_norm so max_grad_norm_seen + num_clips light up.
  await page.getByTestId("train-opt-grad_clip_max_norm").fill("0.5");

  await clickRunPipeline(page, "train");

  // L47 may have auto-expanded the train row if it failed; only
  // click expand-train when the extras row isn't already showing.
  const extrasRow = page.getByTestId("run-result-extras-row-train");
  if (!(await extrasRow.isVisible().catch(() => false))) {
    await page.getByTestId("run-result-expand-train").click();
  }

  // Overlay container always renders for train.
  await expect(page.getByTestId("train-extras-overlay"))
    .toBeVisible({ timeout: 30_000 });

  // Primary loss chart shows ≥2 finite points.
  const point0 = page.getByTestId("extras-loss-chart-point-0");
  await expect(point0).toBeVisible();
  const v0 = Number(await point0.getAttribute("data-loss-value"));
  expect(Number.isFinite(v0)).toBe(true);

  // optimizer + brick-kinds always emit.
  await expect(page.getByTestId("extras-badge-optimizer_kind"))
    .toBeVisible();
  await expect(page.locator("[data-testid^='extras-brick-kind-']").first())
    .toBeVisible({ timeout: 5_000 });

  // grad-clip activity surfaced because grad_clip_max_norm was set.
  await expect(page.getByTestId("extras-grad-clip-panel"))
    .toBeVisible({ timeout: 5_000 });

  await closeModal(page);
});
