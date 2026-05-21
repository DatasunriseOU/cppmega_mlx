// V3-6: multi-step convergence — train for N=8 steps via the new
// TopBar train-num-steps input, assert that losses actually decrease.
// Closes the "did the model learn anything" gap that 2-step matrix
// runs left open (2 steps proves non-NaN, not learning).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

// Strict-convergence presets (deeper architectures): training math
// must actually move loss downward on average.
const CONVERGENCE_PRESETS = ["llama3_8b", "mistral_small_3_1"];

// Multi-step sanity presets (shallower architectures): assert that 8
// steps execute, weights move, losses stay finite. Synthetic random
// targets + small models make loss-curve direction noisy.
const SANITY_PRESETS = ["gemma3_270m", "tiny_aya"];

for (const preset of CONVERGENCE_PRESETS) {
  test(`multi-step convergence (N=8): ${preset} loss decreases`, async ({
    page,
  }) => {
    test.setTimeout(120_000);  // 8-step train + RPC + LR head warmup
    await gotoApp(page);
    await selectPreset(page, preset);

    // Open the run dropdown, set num_steps=8, click Train.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("8");
    await page.getByTestId("run-pipeline-train").click();

    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    const extras = await readTrainExtras(page);

    // V3-6 acceptance: losses must show a decrease floor.
    expect(extras.num_steps).toBe(8);
    expect(extras.losses.length).toBe(8);
    expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

    // Strict monotone-ish: losses[7] strictly smaller than losses[0]
    // and average of last 3 strictly smaller than average of first 3.
    // Synthetic Gaussian embeds + random targets do not produce a
    // perfect convergence curve, but any genuine training signal
    // pushes the trailing window below the leading window.
    const first = extras.losses[0];
    const last = extras.losses[extras.losses.length - 1];
    expect(last).toBeLessThan(first);

    const head = extras.losses.slice(0, 3);
    const tail = extras.losses.slice(-3);
    const headAvg = head.reduce((a, b) => a + b, 0) / head.length;
    const tailAvg = tail.reduce((a, b) => a + b, 0) / tail.length;
    expect(tailAvg).toBeLessThan(headAvg);

    // Weights actually moved (not numerical noise).
    expect(extras.weight_delta_norm).toBeGreaterThan(1e-4);

    await closeModal(page);
  });
}

for (const preset of SANITY_PRESETS) {
  test(`multi-step sanity (N=8): ${preset} runs 8 steps with movement`,
    async ({ page }) => {
      test.setTimeout(120_000);
      await gotoApp(page);
      await selectPreset(page, preset);

      await page.getByTestId("run-pipeline-toggle").click();
      await page.getByTestId("train-num-steps").fill("8");
      await page.getByTestId("run-pipeline-train").click();

      const modal = page.getByTestId("run-result-modal");
      await modal.waitFor({ timeout: 60_000 });
      const extras = await readTrainExtras(page);

      expect(extras.num_steps).toBe(8);
      expect(extras.losses.length).toBe(8);
      expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);
      expect(extras.weight_delta_norm).toBeGreaterThan(1e-4);

      await closeModal(page);
    });
}
