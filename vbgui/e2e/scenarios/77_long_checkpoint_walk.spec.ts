// H24/V7-C01: 1000+ step Train checkpoint walk with save through the UI
// on the real tokenizer/parquet pair, fresh page → 200-step resume → assert continuation.
// Uses H=16 scale so that 1000+ steps execute in a few seconds while verifying long-term
// stability and continuation correctness.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab, closeModal } from "../fixtures";
import { loadMatrix } from "../utils/matrix";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { existsSync, unlinkSync } from "node:fs";

const MATRIX = loadMatrix();
const REAL_PARQUET = MATRIX.parquets.T2_gpt2_small__P1_minimal.path;
const REAL_TOKENIZER = MATRIX.tokenizers.T2_gpt2_small.path;
const SAVE = join(tmpdir(), "vbgui_h24_ckpt_long.safetensors");

test.beforeAll(() => {
  if (existsSync(SAVE)) unlinkSync(SAVE);
});

test("H24: 1000+ step UI Train → save → fresh load → 200-step resume (math-effect 🟢)",
  async ({ page }) => {
    test.setTimeout(300_000);

    // ----- Long Train with save -----
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Scale H=16 so 1000 steps are blazingly fast
    await page.getByTestId("dim-env-H").fill("16");
    await page.getByTestId("dim-env-nh").fill("2");
    await page.getByTestId("dim-env-head_dim").fill("8");
    await page.getByTestId("dim-env-apply").click();
    await page.waitForTimeout(500);

    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path").fill(REAL_PARQUET);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });
    await page.getByTestId("data-use-for-train").click();

    await clickTab(page, "tokenizer");
    await page.getByTestId("tokenizer-playground").waitFor();
    await page.getByTestId("add-panel").click();
    await page.getByTestId("tokenizer-source-0").fill(REAL_TOKENIZER);
    await page.getByTestId("tokenizer-encode-0").click();
    await page.getByTestId("tokenizer-metrics-0").waitFor({ timeout: 8_000 });
    await page.getByTestId("tokenizer-use-for-train-0").click();

    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("1000");
    await page.getByTestId("train-checkpoint-save-path").fill(SAVE);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 240_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const savedPath = ((await page.getByTestId(
      "run-result-extras-train-checkpoint-saved_path").textContent()) ?? "")
      .trim();
    expect(savedPath).toBe(SAVE);

    // Capture last loss from the long run via losses array
    const lossesItems = page.locator(
      "[data-testid^='run-result-extras-train-losses-']");
    const cnt = await lossesItems.count();
    expect(cnt).toBe(1000);
    const savedLast = parseFloat((await page.getByTestId(
      `run-result-extras-train-losses-${cnt - 1}`).textContent()) ?? "0");
    await closeModal(page);

    // ----- Fresh page, same data + tokenizer wired, load, short Train -----
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    
    // Scale H=16 again to match saved checkpoint
    await page.getByTestId("dim-env-H").fill("16");
    await page.getByTestId("dim-env-nh").fill("2");
    await page.getByTestId("dim-env-head_dim").fill("8");
    await page.getByTestId("dim-env-apply").click();
    await page.waitForTimeout(500);

    // Re-wire data + tokenizer so the rebuilt model has the same
    // train_token_embedding architecture as the saved one — otherwise
    // safetensors load silently fails on key mismatch.
    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path").fill(REAL_PARQUET);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });
    await page.getByTestId("data-use-for-train").click();

    await clickTab(page, "tokenizer");
    await page.getByTestId("add-panel").click();
    await page.getByTestId("tokenizer-source-0").fill(REAL_TOKENIZER);
    await page.getByTestId("tokenizer-encode-0").click();
    await page.getByTestId("tokenizer-metrics-0").waitFor({ timeout: 8_000 });
    await page.getByTestId("tokenizer-use-for-train-0").click();

    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("200");
    await page.getByTestId("train-checkpoint-load-path").fill(SAVE);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 120_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const loadedPath = ((await page.getByTestId(
      "run-result-extras-train-checkpoint-loaded_path").textContent()) ?? "")
      .trim();
    expect(loadedPath).toBe(SAVE);

    const resumedFirst = parseFloat((await page.getByTestId(
      "run-result-extras-train-losses-0").textContent()) ?? "0");
    
    // Continuation bound: resumed first loss within ~50% of saved last
    const rel = Math.abs(resumedFirst - savedLast)
      / Math.max(Math.abs(savedLast), 1e-9);
    expect(rel).toBeLessThan(0.5);

    await closeModal(page);
  });
