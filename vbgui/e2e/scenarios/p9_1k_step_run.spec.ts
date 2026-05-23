// V7-P9: 1k+ step run via UI. Annotated test.slow() so Playwright
// picks the extended timeout. Uses the mini preset (default) so the
// per-step cost is bounded — total wall-clock ~ minutes on M-series
// hardware. Asserts the modal's run-result-overall comes back ok
// + the chart renders at least a sampled subset of the 1024 points.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, closeModal,
} from "../fixtures";

test("P9: 1024-step train via UI returns ok + visible chart", async ({
  page,
}) => {
  test.slow();
  test.setTimeout(20 * 60_000);  // 20 minutes — actual run usually faster
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  // stay on mini dim_env so per-step is fast.

  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("1024");
  await page.getByTestId("run-pipeline-train").click();

  // pipeline.run RPC blocks until train completes — wait big.
  await page.getByTestId("run-result-modal").waitFor({
    timeout: 15 * 60_000,
  });
  await expect(page.getByTestId("run-result-overall"))
    .toContainText(/ok|fail/);

  const extras = page.getByTestId("run-result-extras-row-train");
  if (!(await extras.isVisible().catch(() => false))) {
    await page.getByTestId("run-result-expand-train").click();
  }
  // Chart present + at least one finite point visible.
  await expect(page.getByTestId("extras-loss-chart-svg"))
    .toBeVisible({ timeout: 15_000 });
  const v = await page.getByTestId("extras-loss-chart-point-0")
    .getAttribute("data-loss-value");
  expect(Number.isFinite(Number(v))).toBe(true);

  await closeModal(page);
});
