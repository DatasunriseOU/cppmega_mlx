// V7-F51 visual e2e — insert mLSTM brick between attention and mlp
// of llama3_8b, run real 2-step train, assert the visual LossChart
// renders with finite per-step data-loss-value attributes.

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";

test("F51: insert mLSTM into llama3_8b attn→mlp edge + visual train", async ({
  page,
}) => {
  test.setTimeout(120_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Llama preset edge: llama3_8b_attn → llama3_8b_mlp.
  const bar = page.getByTestId("insert-edge-bar");
  await expect(bar).toBeVisible();
  await page.getByTestId("insert-edge-brick-kind").selectOption("mlstm");
  await page.getByTestId("insert-edge-target")
    .selectOption("llama3_8b_attn->llama3_8b_mlp");
  await page.getByTestId("insert-edge-go").click();

  // mLSTM node now lives on the canvas, between attn and mlp.
  await expect(page.locator("[data-testid^='brick-node-mlstm_insert']"))
    .toHaveCount(1, { timeout: 4_000 });

  // Run real train through the modified graph.
  await clickRunPipeline(page, "train");
  await page.getByTestId("run-result-expand-train").click();
  await expect(page.getByTestId("chart-svg")).toBeVisible({
    timeout: 30_000,
  });
  const d = await page.getByTestId("chart-line").getAttribute("d");
  expect(d, "chart-line path d").toBeTruthy();
  expect((d ?? "").split("L").length).toBeGreaterThanOrEqual(2);
  const firstLoss = await page.getByTestId("chart-point-0")
    .getAttribute("data-loss-value");
  expect(Number.isFinite(Number(firstLoss))).toBe(true);

  await closeModal(page);
});
