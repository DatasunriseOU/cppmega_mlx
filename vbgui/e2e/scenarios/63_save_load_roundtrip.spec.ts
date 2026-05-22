// H09: Save → Load round-trip extras parity through the GUI.
//
// V5-G11 added Save/Load buttons to TopBar. v6 closes the honesty gap:
// previously nobody had asserted that a Loaded spec actually produces
// the same training extras as building the same spec from scratch.
// This test does:
//   1. Build spec via preset (llama3_8b) → click Save → capture the
//      download as a Blob → Train and snapshot extras_baseline.
//   2. Fresh page (gotoApp again) → Load the captured file via the
//      file input → Train and snapshot extras_after_load.
//   3. Assert model_summary fields match exactly + losses match
//      within 1e-4 (synthetic data is seed-deterministic in stage_train).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeFileSync, readFileSync, existsSync, unlinkSync } from "node:fs";

const SAVE_PATH = join(tmpdir(), "vbgui_h09_spec.json");

test.beforeAll(() => {
  if (existsSync(SAVE_PATH)) unlinkSync(SAVE_PATH);
});

test("H09: Save → fresh page → Load → Train extras match baseline",
  async ({ page }) => {
    test.setTimeout(180_000);

    // ----- 1) Build, Save, Train baseline ---------------------------------
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Trigger Save and intercept the download.
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByTestId("spec-save").click(),
    ]);
    const tmp = await download.path();
    expect(tmp).toBeTruthy();
    const bytes = readFileSync(tmp!);
    writeFileSync(SAVE_PATH, bytes);
    expect(existsSync(SAVE_PATH)).toBe(true);

    // Baseline Train.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    const baseline = await readTrainExtras(page);
    await closeModal(page);

    // ----- 2) Fresh page + Load + Train -----------------------------------
    await gotoApp(page);
    // No preset — feed the saved spec via the file input so the test
    // proves Load alone reproduces the canvas state.
    const loadInput = page.getByTestId("spec-load-input");
    await loadInput.setInputFiles(SAVE_PATH);
    // Wait for nodes to materialise from the loaded spec.
    await expect.poll(async () =>
      await page.locator("[data-testid^='brick-node-']").count(),
      { timeout: 8_000 },
    ).toBeGreaterThan(0);

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    const afterLoad = await readTrainExtras(page);
    await closeModal(page);

    // ----- 3) Assert parity ------------------------------------------------
    expect(afterLoad.model_summary).toEqual(baseline.model_summary);
    expect(afterLoad.num_steps).toBe(baseline.num_steps);
    expect(afterLoad.losses.length).toBe(baseline.losses.length);
    for (let i = 0; i < baseline.losses.length; i++) {
      expect(Math.abs(afterLoad.losses[i] - baseline.losses[i]))
        .toBeLessThan(1e-4);
    }
  });
