// V4-10/H17: side_channels toggle in train dropdown reaches stage_train
// and doc_ids has a real forward effect through the attention mask.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("V4-10: doc_ids toggle reaches stage_train side_channels_observed",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-side-channel-doc_ids").check();
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    // Array rendered with one entry
    const sc0 = await page.getByTestId(
      "run-result-extras-train-side_channels_observed-0").textContent();
    expect(sc0?.trim()).toBe("doc_ids");
    const docMaskApplied = await page.getByTestId(
      "run-result-extras-train-side_channels_forward_effect-doc_mask_applied")
      .textContent();
    expect(docMaskApplied?.trim()).toBe("true");
    const densityText = await page.getByTestId(
      "run-result-extras-train-side_channels_forward_effect-doc_ids_mask_density")
      .textContent();
    expect(Number(densityText)).toBeGreaterThan(0.1);

    await closeModal(page);
  });

test("V4-10: both toggles enabled → both observed", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-side-channel-doc_ids").check();
  await page.getByTestId("train-side-channel-token_ids").check();
  await page.getByTestId("run-pipeline-train").click();

  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();

  const items = page.locator(
    "[data-testid^='run-result-extras-train-side_channels_observed-']");
  const count = await items.count();
  expect(count).toBe(2);
  const tokenNorm = await page.getByTestId(
    "run-result-extras-train-side_channels_forward_effect-token_ids_added_norm")
    .textContent();
  expect(Number(tokenNorm)).toBeGreaterThan(0);

  await closeModal(page);
});

test("V4-10: no toggle → empty observed list", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();

  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();

  const items = page.locator(
    "[data-testid^='run-result-extras-train-side_channels_observed-']");
  const count = await items.count();
  expect(count).toBe(0);

  await closeModal(page);
});
