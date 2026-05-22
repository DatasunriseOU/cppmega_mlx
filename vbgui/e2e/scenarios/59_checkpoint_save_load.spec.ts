// H05: TopBar checkpoint save/load path inputs forward into stage_train
// G12. The two tests run sequentially in this file (--workers=1):
//   1. Save-path → extras.checkpoint.saved_path matches and the
//      safetensors file lands on disk for the next test to load.
//   2. Load-path with that file → extras.checkpoint.loaded_path matches.
//
// The strict loss-continuation round-trip (H05.6) is deferred to H19
// (strict identical-loss-continuation) where the spec/data also gets
// pinned to a deterministic seed. H05 closes the path-forwarding gap.

import { test, expect } from "@playwright/test";
import { unlinkSync, existsSync } from "node:fs";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

const SAVE = "/tmp/vbgui_h05_ckpt.safetensors";

test.beforeAll(() => {
  if (existsSync(SAVE)) unlinkSync(SAVE);
});

test("H05.A: save-path forwards → checkpoint.saved_path matches",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("train-checkpoint-save-path").fill(SAVE);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();
    const saved = await page.getByTestId(
      "run-result-extras-train-checkpoint-saved_path").textContent();
    expect(saved?.trim()).toBe(SAVE);
    await closeModal(page);
    expect(existsSync(SAVE)).toBe(true);
  });

test("H05.B: load-path forwards → checkpoint.loaded_path matches",
  async ({ page }) => {
    test.setTimeout(120_000);
    // Guard: depends on file written by H05.A.
    expect(existsSync(SAVE)).toBe(true);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("train-checkpoint-load-path").fill(SAVE);
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();
    await page.getByTestId("run-result-extras-row-train").waitFor();
    const loaded = await page.getByTestId(
      "run-result-extras-train-checkpoint-loaded_path").textContent();
    expect(loaded?.trim()).toBe(SAVE);
    await closeModal(page);
  });
