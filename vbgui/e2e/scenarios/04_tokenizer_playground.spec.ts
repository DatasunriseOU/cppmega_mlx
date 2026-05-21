// Tokenizer Playground — 3-panel side-by-side comparison across all 4
// fixture tokenizers. Same input text → distinct token counts +
// capabilities surfaced per panel.

import { test, expect } from "@playwright/test";
import { gotoApp, clickTab } from "../fixtures";
import { snapshot } from "../utils/screenshot";
import { loadMatrix, TOKENIZER_NAMES } from "../utils/matrix";

test("Tokenizer Playground — 3-panel side-by-side compare", async ({ page }) => {
  const matrix = loadMatrix();
  await gotoApp(page);
  await clickTab(page, "tokenizer");
  await page.getByTestId("tokenizer-playground").waitFor();

  // Set deterministic input to make the panels comparable.
  const textarea = page.getByTestId("tokenizer-input");
  await textarea.fill("def foo(x): return x + 1\nclass Bar: pass\n");

  // Add 2 more panels (default starts with 0) → 3 total.
  await page.getByTestId("add-panel").click();
  await page.getByTestId("add-panel").click();
  await page.getByTestId("add-panel").click();

  const triple = [
    TOKENIZER_NAMES[0], TOKENIZER_NAMES[1], TOKENIZER_NAMES[2],
  ] as const;
  for (const [i, name] of triple.entries()) {
    const src = matrix.tokenizers[name].path;
    await page.getByTestId(`tokenizer-source-${i}`).fill(src);
    await page.getByTestId(`tokenizer-encode-${i}`).click();
    await page.getByTestId(`tokenizer-metrics-${i}`).waitFor();
  }

  // All three panels must show distinct token-count metrics (different
  // vocab sizes encode the same text into different numbers of tokens).
  const counts = await Promise.all(triple.map(async (_, i) => {
    const text = await page.getByTestId(`tokenizer-metrics-${i}`).textContent();
    const match = text?.match(/(\d+) tokens/);
    return match ? Number(match[1]) : -1;
  }));
  expect(counts.every((c) => c > 0)).toBe(true);
  // T1 (65536 BPE) vs T3 (256 mini-BPE) MUST give different counts.
  expect(counts[0]).not.toBe(counts[2]);

  await snapshot(page, "04_tokenizer_playground", "three_panels_compare");
});

test("Tokenizer Playground — encoding round-trip yields shape spans", async ({ page }) => {
  const matrix = loadMatrix();
  await gotoApp(page);
  await clickTab(page, "tokenizer");
  await page.getByTestId("add-panel").click();
  await page.getByTestId("tokenizer-source-0")
            .fill(matrix.tokenizers.T1_cppmega_v3.path);
  await page.getByTestId("tokenizer-input").fill("hello world");
  await page.getByTestId("tokenizer-encode-0").click();
  await page.getByTestId("tokenizer-metrics-0").waitFor();
  // Each chip carries id + offset tooltip.
  await expect(page.getByTestId("tokenizer-chip-0-0")).toBeVisible();
});
