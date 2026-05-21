// V4-1: DataInspector → stage_options.train.parquet_path threading.
//
// Walks: load parquet in Data tab → click "Use for training" → switch
// to canvas → Train → assert extras.data_source === "parquet" AND
// extras.token_count > 0 (proves UI selection actually reached backend).
//
// Closes G1 from V4 audit (UI parquet decorative for training).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab, closeModal } from "../fixtures";
import { loadMatrix } from "../utils/matrix";
import { readTrainExtras } from "../utils/train_extras";

test("V4-1: DataInspector parquet reaches stage_train via stage_options",
  async ({ page }) => {
    test.setTimeout(120_000);
    const matrix = loadMatrix();
    const parquet = matrix.parquets.T2_gpt2_small__P1_minimal.path;

    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Default indicator: synthetic
    await expect(page.getByTestId("train-data-source")).toContainText(
      "synthetic");

    // Load parquet in Data tab
    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path").fill(parquet);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });

    // Use this parquet for training
    await page.getByTestId("data-use-for-train").click();

    // Indicator now shows parquet basename
    const indicator = page.getByTestId("train-data-source");
    await expect(indicator).toContainText("parquet:");
    await expect(indicator).toContainText("T2_gpt2_small__P1_minimal");

    // Train
    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });

    const extras = await readTrainExtras(page);
    // Direct DOM check on new data_source / token_count primitives.
    const dataSource = await page.getByTestId(
      "run-result-extras-train-data_source").textContent();
    expect(dataSource?.trim()).toBe("parquet");
    const tokenCount = parseInt(
      (await page.getByTestId(
        "run-result-extras-train-token_count").textContent()) ?? "0", 10);
    expect(tokenCount).toBeGreaterThan(0);
    // Weights moved
    expect(extras.weight_delta_norm).toBeGreaterThan(0);

    await closeModal(page);
  });

test("V4-1 negative: no Use-for-train click → stage_train falls back to synthetic",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Don't click Use-for-train; just run Train.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });

    await page.getByTestId("run-result-expand-train").click();
    const dataSource = await page.getByTestId(
      "run-result-extras-train-data_source").textContent();
    expect(dataSource?.trim()).toBe("synthetic");
    await closeModal(page);
  });
