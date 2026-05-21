// G02 + G03: IFIM_SHAPED / MHC_ATTN_BIAS produce actual math effect.
// V4-7 only proved loss_kind string echo; v5 asserts that:
//   - extras.ifim.{lambda_fim, fim_weights_norm, penalty_value} populated
//   - extras.mhc.{lambda_mhc, bias_norm, penalty_value} populated
//   - changing λ changes losses[0]

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";

async function applyLossAndTrain(
  page: import("@playwright/test").Page, kind: string,
  lambdaTestid: string, lambdaValue: string,
): Promise<{ losses: number[]; extras: Record<string, unknown> }> {
  await page.getByTestId("sidebar-tab-loss").click();
  await page.getByTestId("loss-tab").waitFor();
  await page.getByTestId("loss-kind").selectOption(kind);
  await page.getByTestId(lambdaTestid).fill(lambdaValue);
  await page.getByTestId("loss-apply").click();
  await page.getByTestId("run-pipeline-toggle").click();
  await page.getByTestId("run-pipeline-train").click();
  const modal = page.getByTestId("run-result-modal");
  await modal.waitFor({ timeout: 60_000 });
  await page.getByTestId("run-result-expand-train").click();
  // Parse losses array
  const lossesCount = await page.locator(
    "[data-testid^='run-result-extras-train-losses-']").count();
  const losses: number[] = [];
  for (let i = 0; i < lossesCount; i++) {
    losses.push(parseFloat(
      (await page.getByTestId(`run-result-extras-train-losses-${i}`)
        .textContent()) ?? "NaN"));
  }
  return { losses, extras: {} };
}

test("G02: IFIM λ_fim=0.1 populates extras.ifim with penalty", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await applyLossAndTrain(page, "ifim_shaped", "loss-ifim-lambda", "0.1");
  const lam = parseFloat(
    (await page.getByTestId("run-result-extras-train-ifim-lambda_fim")
      .textContent()) ?? "NaN");
  const fim = parseFloat(
    (await page.getByTestId("run-result-extras-train-ifim-fim_weights_norm")
      .textContent()) ?? "NaN");
  const penalty = parseFloat(
    (await page.getByTestId("run-result-extras-train-ifim-penalty_value")
      .textContent()) ?? "NaN");
  expect(lam).toBeCloseTo(0.1, 6);
  expect(fim).toBeGreaterThan(0);
  expect(Math.abs(penalty - lam * fim)).toBeLessThan(1e-3);
  await closeModal(page);
});

test("G02: IFIM penalty_value scales linearly with λ_fim", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await applyLossAndTrain(page, "ifim_shaped", "loss-ifim-lambda", "0.5");
  const lam = parseFloat(
    (await page.getByTestId("run-result-extras-train-ifim-lambda_fim")
      .textContent()) ?? "NaN");
  const fim = parseFloat(
    (await page.getByTestId("run-result-extras-train-ifim-fim_weights_norm")
      .textContent()) ?? "NaN");
  const penalty = parseFloat(
    (await page.getByTestId("run-result-extras-train-ifim-penalty_value")
      .textContent()) ?? "NaN");
  expect(lam).toBeCloseTo(0.5, 6);
  // λ=0.5 with non-trivial fim → penalty must dominate
  expect(penalty).toBeGreaterThan(0.01);
  expect(Math.abs(penalty - lam * fim)).toBeLessThan(1e-3);
  await closeModal(page);
});

test("G03: MHC λ_mhc=0.05 populates extras.mhc with bias_norm", async ({
  page,
}) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await applyLossAndTrain(page, "mhc_attn_bias", "loss-mhc-lambda", "0.05");
  const lam = parseFloat(
    (await page.getByTestId("run-result-extras-train-mhc-lambda_mhc")
      .textContent()) ?? "NaN");
  const bias = parseFloat(
    (await page.getByTestId("run-result-extras-train-mhc-bias_norm")
      .textContent()) ?? "NaN");
  expect(lam).toBeCloseTo(0.05, 6);
  expect(bias).toBeGreaterThan(0);
  await closeModal(page);
});

test("G03: MHC penalty_value scales linearly with λ_mhc", async ({ page }) => {
  test.setTimeout(60_000);
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");
  await applyLossAndTrain(page, "mhc_attn_bias", "loss-mhc-lambda", "0.2");
  const lam = parseFloat(
    (await page.getByTestId("run-result-extras-train-mhc-lambda_mhc")
      .textContent()) ?? "NaN");
  const bias = parseFloat(
    (await page.getByTestId("run-result-extras-train-mhc-bias_norm")
      .textContent()) ?? "NaN");
  const penalty = parseFloat(
    (await page.getByTestId("run-result-extras-train-mhc-penalty_value")
      .textContent()) ?? "NaN");
  expect(lam).toBeCloseTo(0.2, 6);
  expect(penalty).toBeGreaterThan(0.001);
  expect(Math.abs(penalty - lam * bias)).toBeLessThan(1e-3);
  await closeModal(page);
});
