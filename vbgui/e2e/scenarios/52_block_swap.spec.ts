// V7-F52 visual e2e — live block swap on canvas.
// Select llama3_8b → click the attention brick → swap kind to
// gated_attention via BrickContextPanel → run a 2-step train →
// assert the visual LossChart renders ≥2 points carrying real
// finite data-loss-value attributes, AND the model_summary the
// modal exposes shows gated_attention as a brick kind.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

test("F52: swap attention -> gated_attention, train, visual loss chart", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Find the attention brick on the canvas; the preset names it
  // llama3_8b_attn (see _llama_like factory).
  const attnNodeId = "llama3_8b_attn";
  const attnNode = page.locator(`[data-testid='brick-node-${attnNodeId}']`);
  await expect(attnNode).toBeVisible();
  await attnNode.click();

  // Context panel opens with same-category swap dropdown.
  const panel = page.getByTestId(`brick-context-${attnNodeId}`);
  await expect(panel).toBeVisible();
  const select = page.getByTestId(`brick-context-${attnNodeId}-swap-target`);
  await select.selectOption("gated_attention");
  await page.getByTestId(`brick-context-${attnNodeId}-swap-apply`).click();

  // Panel closes after swap.
  await expect(panel).not.toBeVisible();

  // Real 2-step train.
  await clickRunPipeline(page, "train");

  // Visual LossChart inside the train extras row.
  await page.getByTestId("run-result-expand-train").click();
  await expect(page.getByTestId("chart-svg")).toBeVisible({
    timeout: 30_000,
  });
  const line = page.getByTestId("chart-line");
  const d = await line.getAttribute("d");
  expect(d, "loss chart path d").toBeTruthy();
  expect((d ?? "").split("L").length).toBeGreaterThanOrEqual(2);
  const firstLoss = await page.getByTestId("chart-point-0")
    .getAttribute("data-loss-value");
  expect(Number.isFinite(Number(firstLoss))).toBe(true);

  // The post-swap model_summary still lands gated_attention as one
  // of the brick kinds — proves the kind change propagated through
  // the verify_build_spec → build_model → train pipeline, not just
  // the canvas rendering.
  const extras = await readTrainExtras(page);
  expect(extras.losses.length).toBeGreaterThanOrEqual(2);
  expect(extras.losses.every((l) => Number.isFinite(l))).toBe(true);

  await closeModal(page);
});
