// H06: UI N=100 long-run real-corpus walk through Visual Builder GUI.
//
// Closes V5-G15 "long-N convergence on real corpus" from the UI side:
// pick the real parquet+tokenizer pair from MATRIX.json, set Train
// steps to 100, dispatch Train through the TopBar, then assert that
// the resulting extras prove the model actually trained for 100 steps
// on tokens (not synthetic random), made non-trivial parameter
// movement, and the inference probe distinguishes pre-vs-post weights.
//
// This is a long-running test (~30-90s) and is tagged @long-running so
// CI can opt-in. Local runs include it by default.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, clickTab, closeModal } from "../fixtures";
import { loadMatrix } from "../utils/matrix";
import { readTrainExtras } from "../utils/train_extras";

const MATRIX = loadMatrix();
const REAL_PARQUET = MATRIX.parquets.T2_gpt2_small__P1_minimal.path;
const REAL_TOKENIZER = MATRIX.tokenizers.T2_gpt2_small.path;

test("H06 @long-running: N=100 real-corpus walk through UI", async ({ page }) => {
  // 180s budget; plan target is 120s but Apple-Silicon CI variance is real.
  test.setTimeout(180_000);

  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  // Wire real parquet via Data tab.
  await clickTab(page, "data");
  await page.getByTestId("data-inspector").waitFor();
  await page.getByTestId("data-path").fill(REAL_PARQUET);
  await page.getByTestId("data-load").click();
  await page.getByTestId("data-metrics").waitFor({ timeout: 8_000 });
  await page.getByTestId("data-use-for-train").click();

  // Wire real tokenizer via Tokenizer tab.
  await clickTab(page, "tokenizer");
  await page.getByTestId("tokenizer-playground").waitFor();
  await page.getByTestId("add-panel").click();
  await page.getByTestId("tokenizer-source-0").fill(REAL_TOKENIZER);
  await page.getByTestId("tokenizer-encode-0").click();
  await page.getByTestId("tokenizer-metrics-0").waitFor({ timeout: 8_000 });
  await page.getByTestId("tokenizer-use-for-train-0").click();

  // Drive N=100 Train from TopBar Train menu on Canvas tab.
  await clickTab(page, "canvas");
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("train-num-steps").fill("100");
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 150_000 });

  const extras = await readTrainExtras(page);

  // (H06.2) exactly 100 steps executed.
  expect(extras.num_steps).toBe(100);
  expect(extras.losses.length).toBe(100);
  expect(extras.losses.every((l) => Number.isFinite(l))).toBe(true);

  // (H06.3) losses_smoothed final window < initial window
  //   — the smoothed series strips per-step noise so a monotone-window
  //     check is meaningful. We read the smoothed array directly.
  const smoothed = await arrayOf(
    page, "run-result-extras-train-losses_smoothed");
  expect(smoothed.length).toBe(100);
  const headWin = smoothed.slice(0, 20);
  const tailWin = smoothed.slice(-20);
  const headAvg = headWin.reduce((a, b) => a + b, 0) / headWin.length;
  const tailAvg = tailWin.reduce((a, b) => a + b, 0) / tailWin.length;
  expect(tailAvg).toBeLessThan(headAvg);

  // (H06.4) weights moved meaningfully on a 100-step real-token run.
  expect(extras.weight_delta_norm).toBeGreaterThan(0.01);

  // (H06.5) inference probe shows real divergence (100 steps is significant).
  const l2 = parseFloat(await textOf(
    page, "run-result-extras-train-inference_probe-l2_diff"));
  expect(l2).toBeGreaterThan(0.1);

  // Real-data path activated (not synthetic).
  const dataSource = await page.getByTestId(
    "run-result-extras-train-data_source").textContent();
  // V7-G01 added a multi-shard `parquet_tokenized_stream` label
  // alongside the single-shard `parquet_tokenized`; H06 accepts both.
  expect(["parquet_tokenized", "parquet_tokenized_stream"])
    .toContain(dataSource?.trim() ?? "");

  await closeModal(page);
});

import type { Page } from "@playwright/test";

async function textOf(page: Page, testid: string): Promise<string> {
  const t = await page.getByTestId(testid).textContent();
  if (t == null) throw new Error(`testid ${testid} produced no text`);
  return t.trim();
}

async function arrayOf(page: Page, base: string): Promise<number[]> {
  const items = page.locator(`[data-testid^='${base}-']`);
  const count = await items.count();
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = await page.getByTestId(`${base}-${i}`).textContent();
    if (t == null) throw new Error(`testid ${base}-${i} empty`);
    out.push(parseFloat(t.trim()));
  }
  return out;
}
