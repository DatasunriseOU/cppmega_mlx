// V4-4: Real-data convergence — pick fixture parquet+tokenizer via
// UI, set 8 steps, assert tokenized path activated AND losses fell.
// Closes G10 from the V4 audit: V3-6 convergence used synthetic
// random targets; this is the real-corpus version.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab, closeModal } from "../fixtures";
import { loadMatrix } from "../utils/matrix";
import { readTrainExtras } from "../utils/train_extras";

const MATRIX = loadMatrix();

// Tokenizer + parquet pair from the existing E-1 matrix fixture
// (parquet has an 'original_text' column the V4-2 helper recognises).
const REAL_PARQUET = MATRIX.parquets.T2_gpt2_small__P1_minimal.path;
const REAL_TOKENIZER = MATRIX.tokenizers.T2_gpt2_small.path;

// Deep presets that produced loss-fall in V3-6 synthetic test.
const CONVERGENCE_PRESETS = ["llama3_8b", "mistral_small_3_1"];

for (const preset of CONVERGENCE_PRESETS) {
  test(`V4-4: real-data convergence on ${preset}`, async ({ page }) => {
    test.setTimeout(150_000);
    await gotoApp(page);
    await selectPreset(page, preset);

    // Wire parquet
    await clickTab(page, "data");
    await page.getByTestId("data-inspector").waitFor();
    await page.getByTestId("data-path").fill(REAL_PARQUET);
    await page.getByTestId("data-load").click();
    await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });
    await page.getByTestId("data-use-for-train").click();

    // Wire tokenizer
    await clickTab(page, "tokenizer");
    await page.getByTestId("tokenizer-playground").waitFor();
    await page.getByTestId("add-panel").click();
    await page.getByTestId("tokenizer-source-0").fill(REAL_TOKENIZER);
    await page.getByTestId("tokenizer-encode-0").click();
    await page.getByTestId("tokenizer-metrics-0").waitFor({ timeout: 8_000 });
    await page.getByTestId("tokenizer-use-for-train-0").click();

    // Train N=8
    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("8");
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 90_000 });

    const extras = await readTrainExtras(page);

    // Real-data path activated
    const dataSource = await page.getByTestId(
      "run-result-extras-train-data_source").textContent();
    expect(dataSource?.trim()).toBe("parquet_tokenized");

    // tokenizer_used reports a path basename (not 'null')
    const tokUsed = await page.getByTestId(
      "run-result-extras-train-tokenizer_used").textContent();
    expect(tokUsed?.trim()).toContain(".json");

    // 8 steps actually executed
    expect(extras.num_steps).toBe(8);
    expect(extras.losses.length).toBe(8);
    expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

    // Convergence floor: at least one of {strict last<first,
    // tailAvg<headAvg} must hold. Real tokens give signal; if neither
    // direction holds the loss kernel is broken.
    const first = extras.losses[0];
    const last = extras.losses[extras.losses.length - 1];
    const head = extras.losses.slice(0, 3);
    const tail = extras.losses.slice(-3);
    const headAvg = head.reduce((a, b) => a + b, 0) / head.length;
    const tailAvg = tail.reduce((a, b) => a + b, 0) / tail.length;
    expect(last < first || tailAvg < headAvg).toBe(true);

    // Weights actually moved
    expect(extras.weight_delta_norm).toBeGreaterThan(1e-4);

    await closeModal(page);
  });
}
