// G22: muon_adamw_hybrid produces distinct per-bucket update deltas.
// V4-9 only counted bucket sizes; G22 asserts the update math actually
// diverges between Muon and AdamW groups.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("G22: hybrid optimizer produces distinct muon/adamw bucket deltas",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("sidebar-tab-optim").click();
    await page.getByTestId("optim-kind").selectOption("muon_adamw_hybrid");
    await page.getByTestId("optim-apply").click();

    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("train-num-steps").fill("4");
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    const muonNorm = parseFloat(
      (await page.getByTestId(
        "run-result-extras-train-hybrid_deltas-muon_norm").textContent())
        ?? "0");
    const adamwNorm = parseFloat(
      (await page.getByTestId(
        "run-result-extras-train-hybrid_deltas-adamw_norm").textContent())
        ?? "0");
    const ratio = parseFloat(
      (await page.getByTestId(
        "run-result-extras-train-hybrid_deltas-ratio").textContent())
        ?? "0");

    // Both buckets must receive updates
    expect(muonNorm).toBeGreaterThan(0);
    expect(adamwNorm).toBeGreaterThan(0);
    // Ratio MUST differ from 1.0 by more than 5% — otherwise the
    // hybrid split would have produced identical updates (i.e., the
    // two optimizers' math is decorative).
    expect(Math.abs(ratio - 1.0)).toBeGreaterThan(0.05);
    await closeModal(page);
  });
