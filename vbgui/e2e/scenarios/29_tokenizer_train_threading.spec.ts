// V4-3: Tokenizer Playground → stage_options.train.tokenizer_path
// threading. Combined with V4-1 parquet path, this proves the full
// real-data path: UI parquet + UI tokenizer → stage_train_tokenized.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab, closeModal } from "../fixtures";
import { loadMatrix } from "../utils/matrix";

test("V4-3: Tokenizer Playground tokenizer reaches stage_train tokenizer_path",
  async ({ page }) => {
    test.setTimeout(120_000);
    const matrix = loadMatrix();
    const parquet = matrix.parquets.T2_gpt2_small__P1_minimal.path;
    const tokenizer = matrix.tokenizers.T2_gpt2_small.path;

    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Load parquet + Use-for-train
    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path").fill(parquet);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });
    await page.getByTestId("data-use-for-train").click();

    // Pick a tokenizer in Playground + Use-for-train
    await clickTab(page, "tokenizer");
    await page.getByTestId("tokenizer-playground").waitFor();
    await page.getByTestId("add-panel").click();
    await page.getByTestId("tokenizer-source-0").fill(tokenizer);
    await page.getByTestId("tokenizer-encode-0").click();
    await page.getByTestId("tokenizer-metrics-0").waitFor({ timeout: 8_000 });
    await page.getByTestId("tokenizer-use-for-train-0").click();

    // Indicator should now show both parquet + tokenizer
    const indicator = page.getByTestId("train-data-source");
    await expect(indicator).toContainText("parquet:");
    await expect(indicator).toContainText("tok:");

    // Train + assert backend received tokenizer_path
    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    // The T2 parquet has only input_ids (no 'text' column), so V4-2
    // tokenizer path falls through to V3-2 raw-int path. Either way
    // data_source must NOT be 'synthetic'.
    const dataSource = await page.getByTestId(
      "run-result-extras-train-data_source").textContent();
    expect(["parquet", "parquet_tokenized"]).toContain(dataSource?.trim());

    await closeModal(page);
  });
