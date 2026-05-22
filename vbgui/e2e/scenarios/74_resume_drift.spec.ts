// H21: UI walk — save ckpt → fresh page → load → resume Train →
// inference probe l2_diff is bounded.
//
// Also covers the corrupt-checkpoint path: feeding a non-safetensors
// file as load-path triggers a silent fallback and Train still
// completes (no spurious error modal).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { existsSync, unlinkSync, writeFileSync } from "node:fs";

const SAVE = join(tmpdir(), "vbgui_h21_ck.safetensors");
const BAD = join(tmpdir(), "vbgui_h21_bad.safetensors");

test.beforeAll(() => {
  for (const p of [SAVE, BAD]) {
    if (existsSync(p)) unlinkSync(p);
  }
  // Pre-create a corrupt file for the negative test.
  writeFileSync(BAD, "not a safetensors file");
});

test("H21: save → fresh → load → resume Train → l2_diff finite",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("4");
    await page.getByTestId("train-checkpoint-save-path").fill(SAVE);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await closeModal(page);

    // Fresh page → load only → resume 1 step.
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("1");
    await page.getByTestId("train-checkpoint-load-path").fill(SAVE);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();

    const l2 = parseFloat(((await page.getByTestId(
      "run-result-extras-train-inference_probe-l2_diff").textContent()) ?? "0")
      .trim());
    expect(Number.isFinite(l2)).toBe(true);
    expect(l2).toBeGreaterThanOrEqual(0);
    await closeModal(page);
  });

test("H21: corrupt checkpoint → load swallowed silently, Train still ok",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("train-checkpoint-load-path").fill(BAD);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    const trainRow = page.getByTestId("run-result-stage-train");
    await expect(trainRow).toContainText("ok");
    await page.getByTestId("run-result-expand-train").click();
    const loaded = ((await page.getByTestId(
      "run-result-extras-train-checkpoint-loaded_path").textContent()) ?? "")
      .trim();
    // Load swallowed → loaded_path is the rendered "null" string.
    expect(loaded).toBe("null");
    await closeModal(page);
  });
