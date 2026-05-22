// H14: AblationsTab row expand reveals full extras subtree.
//
// V5-G18 added structural parity at the backend; the UI only showed
// final loss + mini chart. v6 adds an `ablation-row-{variant}-expand`
// button per row that toggles an extras subtree row containing the
// per-variant losses array plus every key of train extras
// (model_summary, optimizer_kind, schedule_kind, data_source, …).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset } from "../fixtures";

test("H14: ablation row expand renders extras subtree", async ({ page }) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Open Ablations tab.
  await page.getByTestId("sidebar-tab-ablations").click();
  await page.getByTestId("ablations-tab").waitFor();

  // Default activation axis with glu + swiglu pre-selected.
  await page.getByTestId("ablation-num-steps").fill("2");
  await page.getByTestId("ablation-run").click();
  await page.getByTestId("ablation-results").waitFor({ timeout: 60_000 });

  // Pick whichever variant ended up first in the ranking — both glu
  // and swiglu are present in VARIANTS_PER_AXIS defaults.
  const row = page.locator(
    "[data-testid^='ablation-row-glu']").first();
  await row.waitFor();
  // Expand row.
  await page.getByTestId("ablation-row-glu-expand").click();
  await page.getByTestId("ablation-row-glu-extras").waitFor();
  // Per-row losses array rendered.
  const lossesText = await page.getByTestId(
    "ablation-row-glu-losses").textContent();
  expect(lossesText).toContain("[");
  expect(lossesText).toContain("]");
  // Backend-emitted extras keys are rendered.
  await expect(page.getByTestId(
    "ablation-row-glu-extras-optimizer_kind")).toBeVisible();
  await expect(page.getByTestId(
    "ablation-row-glu-extras-data_source")).toBeVisible();
  // model_summary is a nested object; rendered as a JSON-stringified dd.
  const ms = await page.getByTestId(
    "ablation-row-glu-extras-model_summary").textContent();
  expect(ms).toContain("optimizer_kind");
});
