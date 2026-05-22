// H08: TopBar train-probe-text textarea threads opts.inference_probe_text
// into stage_train (V5-G20). Backend pairs the text with the active
// tokenizer to encode real token ids for the pre-vs-post forward
// divergence probe, replacing the synthetic random Gaussian input.
//
// Assertions:
//   - extras.inference_probe.real_tokens === true
//   - extras.inference_probe.text_len > 0
//   - extras.inference_probe.top1_token_drift >= 0 (number, not "null")

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab, closeModal } from "../fixtures";
import { loadMatrix } from "../utils/matrix";

const MATRIX = loadMatrix();
const REAL_TOKENIZER = MATRIX.tokenizers.T2_gpt2_small.path;

test("H08: train-probe-text → extras.inference_probe.real_tokens=true",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Wire tokenizer so the backend can encode the probe text.
    await clickTab(page, "tokenizer");
    await page.getByTestId("tokenizer-playground").waitFor();
    await page.getByTestId("add-panel").click();
    await page.getByTestId("tokenizer-source-0").fill(REAL_TOKENIZER);
    await page.getByTestId("tokenizer-encode-0").click();
    await page.getByTestId("tokenizer-metrics-0").waitFor({ timeout: 8_000 });
    await page.getByTestId("tokenizer-use-for-train-0").click();

    // Drive Train with probe text.
    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("train-probe-text")
      .fill("hello world this is a probe sentence");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const real = await page.getByTestId(
      "run-result-extras-train-inference_probe-real_tokens").textContent();
    expect(real?.trim().toLowerCase()).toBe("true");

    const textLen = parseInt(await page.getByTestId(
      "run-result-extras-train-inference_probe-text_len")
      .textContent() ?? "0", 10);
    expect(textLen).toBeGreaterThan(0);

    const drift = await page.getByTestId(
      "run-result-extras-train-inference_probe-top1_token_drift").textContent();
    expect(drift?.trim()).not.toBe("null");
    expect(parseFloat(drift!.trim())).toBeGreaterThanOrEqual(0);

    await closeModal(page);
  });
