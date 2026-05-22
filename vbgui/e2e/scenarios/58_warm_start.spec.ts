// H04: train-warm-start checkbox forwards continue_from_run_id so the
// backend G10 LRU cache restores opt.state for the next Train run.
//
// First Train (warm-start OFF):  cold opt.state → extras.opt_state_carried=false
// Second Train (warm-start ON):  same spec re-runs → opt_state_carried=true.
//
// The second run does not require a strictly lower losses[0] (the cached
// opt.state pairs with a freshly built model whose weights re-randomise,
// so the loss surface differs). What MUST hold is that the warm-start
// flag actually flips the backend's opt_state_carried extras.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

test("H04: warm-start checkbox flips extras.opt_state_carried",
  async ({ page }) => {
    test.setTimeout(120_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // First Train — warm-start OFF (default). Expect opt_state_carried=false.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("2");
    // Confirm the checkbox starts unchecked.
    await expect(page.getByTestId("train-warm-start")).not.toBeChecked();
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    const firstExtras = await readTrainExtras(page);
    expect(firstExtras.losses.length).toBe(2);
    const firstCarried = await page.getByTestId(
      "run-result-extras-train-opt_state_carried").textContent();
    expect(firstCarried?.trim().toLowerCase()).toBe("false");
    await closeModal(page);

    // Second Train — warm-start ON. Expect opt_state_carried=true.
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-warm-start").check();
    await expect(page.getByTestId("train-warm-start")).toBeChecked();
    await page.getByTestId("run-pipeline-train").click();
    await page.getByTestId("run-result-modal").waitFor({ timeout: 60_000 });
    const secondCarried = await page.getByTestId(
      "run-result-extras-train-opt_state_carried").textContent();
    expect(secondCarried?.trim().toLowerCase()).toBe("true");
    await closeModal(page);
  });
