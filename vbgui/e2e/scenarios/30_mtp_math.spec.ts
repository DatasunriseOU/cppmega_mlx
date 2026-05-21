// G01: LossKind MTP_WEIGHTED actually runs K-head weighted loss.
// V4-7 only proved extras.loss_kind echo; G01 asserts the math:
// extras.mtp.{k, betas, per_head_losses} populated AND per_head_losses
// values differ across the K shifted-label heads (proves distinct
// supervision, not all heads sharing the same labels).

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

async function configureMTP(page: import("@playwright/test").Page,
                            k: string, beta: string): Promise<void> {
  await page.getByTestId("sidebar-tab-loss").click();
  await page.getByTestId("loss-tab").waitFor();
  await page.getByTestId("loss-kind").selectOption("mtp_weighted");
  await page.getByTestId("loss-mtp-k").fill(k);
  await page.getByTestId("loss-mtp-beta").fill(beta);
  await page.getByTestId("loss-apply").click();
}

async function trainAndReadMTP(page: import("@playwright/test").Page) {
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();
  const k = parseInt(
    (await page.getByTestId("run-result-extras-train-mtp-k")
      .textContent()) ?? "0", 10);
  const betas: number[] = [];
  const heads: number[] = [];
  for (let i = 0; i < k; i++) {
    const b = parseFloat(
      (await page.getByTestId(`run-result-extras-train-mtp-betas-${i}`)
        .textContent()) ?? "NaN");
    const h = parseFloat(
      (await page.getByTestId(`run-result-extras-train-mtp-per_head_losses-${i}`)
        .textContent()) ?? "NaN");
    betas.push(b);
    heads.push(h);
  }
  return { k, betas, per_head_losses: heads };
}

test("G01: MTP K=2 beta=0.6 produces extras.mtp with 2 distinct head losses",
  async ({ page }) => {
    test.setTimeout(60_000);
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");
    await configureMTP(page, "2", "0.6");
    const mtp = await trainAndReadMTP(page);
    expect(mtp.k).toBe(2);
    expect(mtp.betas).toHaveLength(2);
    expect(mtp.betas[0]).toBeCloseTo(0.6, 4);
    expect(mtp.betas[1]).toBeCloseTo(0.6, 4);
    expect(mtp.per_head_losses).toHaveLength(2);
    // Distinct supervision → losses MUST differ
    expect(Math.abs(mtp.per_head_losses[0] - mtp.per_head_losses[1]))
      .toBeGreaterThan(1e-4);
    await closeModal(page);
  });

test("G01: MTP K=3 → 3 heads + 3 betas + 3 per_head_losses", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await configureMTP(page, "3", "0.33");
  const mtp = await trainAndReadMTP(page);
  expect(mtp.k).toBe(3);
  expect(mtp.betas).toHaveLength(3);
  expect(mtp.per_head_losses).toHaveLength(3);
  await closeModal(page);
});

test("G01: cross_entropy does NOT populate extras.mtp", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  // Default loss kind is cross_entropy; just Train without changing.
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();
  // Expanded extras render dt for primitives + dl/ol for objects/arrays.
  // 'mtp' key would appear as a nested object section; assert no
  // mtp-k testid present.
  const count = await page.locator(
    "[data-testid='run-result-extras-train-mtp-k']").count();
  expect(count).toBe(0);
  await closeModal(page);
});
