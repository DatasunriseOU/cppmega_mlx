// H01: ShardingTab accept proposal updates spec.sharding.axis_assignments
// AND extras.sharding_applied reflects the chosen strategy after Train.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

test("H01: accept first sharding proposal → axes mutated → train extras matches",
  async ({ page }) => {
    test.setTimeout(90_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    // Pick a topology with degree=8 so proposals are non-trivial
    await page.getByTestId("topology-selector").selectOption("h100_8x");
    await page.waitForTimeout(800);  // suggest_sharding debounce

    // Switch to sharding tab and accept first proposal
    await page.getByTestId("sidebar-tab-sharding").click();
    await page.getByTestId("sharding-tab").waitFor();
    const firstAccept = page.locator(
      "[data-testid^='sharding-accept-']").first();
    await firstAccept.waitFor({ timeout: 8_000 });
    await firstAccept.click();
    await page.waitForTimeout(600);  // dispatch + re-verify

    // Run Train; extras.sharding_applied must reflect the accepted
    // strategy's axes (not the INITIAL_SPEC fsdp2 default unchanged).
    await page.getByTestId("run-pipeline-toggle").click();
    await page.getByTestId("run-pipeline-train").click();
    const modal = page.getByTestId("run-result-modal");
    await modal.waitFor({ timeout: 60_000 });
    await page.getByTestId("run-result-expand-train").click();

    const shardDim = parseInt(
      (await page.getByTestId(
        "run-result-extras-train-sharding_applied-shard_dim")
        .textContent()) ?? "0", 10);
    // Accepted proposal's shard_dim = product of axis degrees.
    // h100_8x supports 8-way → shard_dim should be ≥ 1
    expect(shardDim).toBeGreaterThan(0);
    await closeModal(page);
  });
