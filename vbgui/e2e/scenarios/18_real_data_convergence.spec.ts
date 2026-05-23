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

    // G14: Train N=16 with strict 5% floor — V4-4 was weak (last<first
    // OR tailAvg<headAvg, satisfied by single-step random drift).
    await clickTab(page, "canvas");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("16");
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 120_000 });

    const extras = await readTrainExtras(page);

    // Real-data path activated
    const dataSource = await page.getByTestId(
      "run-result-extras-train-data_source").textContent();
    expect(["parquet_tokenized", "parquet_tokenized_stream"]).toContain(dataSource?.trim());

    // tokenizer_used reports a path basename (not 'null')
    const tokUsed = await page.getByTestId(
      "run-result-extras-train-tokenizer_used").textContent();
    expect(tokUsed?.trim()).toContain(".json");

    // 16 steps actually executed
    expect(extras.num_steps).toBe(16);
    expect(extras.losses.length).toBe(16);
    expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

    // G14/H03 follow-up: real-token training now feeds stable token
    // embeddings instead of fresh random embeddings every step. The
    // tiny batch may legitimately fit hard, so the lower 0.3×initial
    // floor would reject successful convergence. What's REQUIRED:
    // bounded behaviour plus a real final loss drop.
    const first = extras.losses[0];
    const secondHalf = extras.losses.slice(8);
    expect(Math.max(...secondHalf)).toBeLessThan(first * 1.5);
    expect(Math.min(...secondHalf)).toBeGreaterThanOrEqual(0);
    expect(extras.losses[extras.losses.length - 1]).toBeLessThan(first * 0.3);

    // Weights moved meaningfully (V4-4 had 1e-4 — bump to 1e-3 for N=16)
    expect(extras.weight_delta_norm).toBeGreaterThan(1e-3);

    await closeModal(page);
  });
}
