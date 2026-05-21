// V4-9: muon_adamw_hybrid optimizer routes 2D matmul weights to Muon
// and 1D/3D+ tensors to AdamW. extras.muon_group_size + adamw_group_size
// must both be > 0 to prove the split predicate actually fired.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

test("V4-9: hybrid optimizer split reports non-empty muon + adamw buckets",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("sidebar-tab-optim").click();
    await page.getByTestId("optim-kind").selectOption("muon_adamw_hybrid");
    await page.getByTestId("optim-apply").click();

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    const extras = await readTrainExtras(page);

    expect(extras.optimizer_kind).toBe("muon_adamw_hybrid");

    const muonSize = parseInt(
      (await page.getByTestId("run-result-extras-train-muon_group_size")
        .textContent()) ?? "0", 10);
    const adamwSize = parseInt(
      (await page.getByTestId("run-result-extras-train-adamw_group_size")
        .textContent()) ?? "0", 10);

    expect(muonSize).toBeGreaterThan(0);
    expect(adamwSize).toBeGreaterThan(0);
    // Sanity: Muon handles big matmul weights → its bucket should be
    // at least as large as the AdamW bucket (norm scalars + biases).
    expect(muonSize).toBeGreaterThanOrEqual(adamwSize);

    await closeModal(page);
  });
