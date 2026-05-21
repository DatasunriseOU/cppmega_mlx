// G04: RewritersTab Apply chain actually mutates the spec graph in
// stage_train. V4-8 only proved rewriter names propagated to extras.
// G04 asserts extras.graph_diff = {added, removed} reflects what the
// rewriter actually did to the build_spec.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

async function applyRewriterAndTrain(
  page: import("@playwright/test").Page, rewriter: string,
): Promise<void> {
  await page.getByTestId("sidebar-tab-rewriters").click();
  await page.getByTestId("rewriters-tab").waitFor();
  await page.getByTestId(`rewriter-add-${rewriter}`).click();
  await page.getByTestId("rewriter-apply").click();
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();
}

test("G04: MTPRewriter adds K-1 head nodes to graph_diff.added",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await applyRewriterAndTrain(page, "MTPRewriter");

    // graph_diff is a nested object → recursive ExtrasEntry renders it.
    // 'added' is an array → ol with per-index testids.
    const addedCount = await page.locator(
      "[data-testid^='run-result-extras-train-graph_diff-added-']").count();
    expect(addedCount).toBeGreaterThanOrEqual(1);
    // MTPRewriter defaults k=2 → one extra head added (head_0 + head_1
    // where head_0 replaces the original)
    const removedCount = await page.locator(
      "[data-testid^='run-result-extras-train-graph_diff-removed-']").count();
    expect(removedCount).toBeGreaterThanOrEqual(1);

    // MTP rewriter also upgrades loss to MTP_WEIGHTED → extras.mtp populated
    const mtpK = parseInt(
      (await page.getByTestId("run-result-extras-train-mtp-k")
        .textContent()) ?? "0", 10);
    expect(mtpK).toBeGreaterThanOrEqual(2);

    await closeModal(page);
  });

test("G04: no rewriters → empty graph_diff added/removed", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();
  const addedCount = await page.locator(
    "[data-testid^='run-result-extras-train-graph_diff-added-']").count();
  expect(addedCount).toBe(0);
  await closeModal(page);
});
