// V3-5: deep UI → backend → real mini-training, with assertions on
// training math (not just status row presence). Each scenario must
// have ≥3 content assertions reading from UI-surfaced extras (V3-4),
// proving the UI mutation propagated through to stage_train.
//
// Replaces the v2 vacuous `expect(["ok","fail"]).toContain(status)`
// pattern that hid bugs B1 (optimizer ignored), B2 (data ignored),
// and B3 (extras hidden).

import { test, expect } from "@playwright/test";
import {
  gotoApp, selectPreset, clickRunPipeline, closeModal,
} from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

// ---------------------------------------------------------------------------
// 1) Activation switch propagates into model_summary.mlp_activation
// ---------------------------------------------------------------------------

test("UI activation change (swiglu) lands in extras.model_summary", async ({
  page,
}) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.locator("[data-testid='brick-node-llama3_8b_mlp']").click();
  const panel = page.getByTestId("brick-context-llama3_8b_mlp");
  await expect(panel).toBeVisible({ timeout: 4_000 });
  await page.locator("[data-testid='brick-context-llama3_8b_mlp-activation']")
    .selectOption("swiglu");
  await page.locator("[data-testid='brick-context-llama3_8b_mlp-apply']")
    .click();

  await clickRunPipeline(page, "train");
  const extras = await readTrainExtras(page);

  // Content assertions — not status theatre:
  expect(extras.model_summary.mlp_activation).toBe("swiglu");
  expect(extras.losses.length).toBeGreaterThanOrEqual(2);
  expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);
  expect(extras.weight_delta_norm).toBeGreaterThan(0);

  await closeModal(page);
});

// ---------------------------------------------------------------------------
// 2) Schedule change shapes lr_trajectory + reports schedule_kind
// ---------------------------------------------------------------------------

test("UI linear_warmup w=4 reaches train as scheduled lr_trajectory", async ({
  page,
}) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("sidebar-tab-optim").click();
  await page.getByTestId("optim-group-0-schedule-toggle").click();
  await page.getByTestId("schedule-kind-0").selectOption("linear_warmup");
  await page.getByTestId("schedule-warmup-0").fill("4");
  await page.getByTestId("optim-apply").click();

  await clickRunPipeline(page, "train");
  const extras = await readTrainExtras(page);

  // schedule_kind matches selection
  expect(extras.schedule_kind).toBe("linear_warmup");
  expect(extras.model_summary.schedule_kind).toBe("linear_warmup");
  // lr_trajectory[0] is ramp (warmup), not the peak lr
  expect(extras.lr_trajectory.length).toBeGreaterThan(0);
  expect(extras.lr_trajectory[0]).toBeLessThan(
    Math.max(...extras.lr_trajectory) + 1e-9);
  // Strictly: at step 0 of linear_warmup(w=4), lr is 0
  expect(extras.lr_trajectory[0]).toBeCloseTo(0, 6);

  await closeModal(page);
});

// ---------------------------------------------------------------------------
// 3) pre_norm switch reaches model_summary.attention_pre_norm
// ---------------------------------------------------------------------------

test("UI pre_norm switch attention rmsnorm→layernorm propagates", async ({
  page,
}) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.locator("[data-testid='brick-node-llama3_8b_attn']").click();
  const panel = page.getByTestId("brick-context-llama3_8b_attn");
  await expect(panel).toBeVisible();
  await page.locator(
    "[data-testid='brick-context-llama3_8b_attn-pre-norm']")
    .selectOption("layernorm");
  await page.locator(
    "[data-testid='brick-context-llama3_8b_attn-apply']").click();

  await clickRunPipeline(page, "train");
  const extras = await readTrainExtras(page);

  expect(extras.model_summary.attention_pre_norm).toBe("layernorm");
  expect(extras.weight_delta_norm).toBeGreaterThan(0);
  expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

  await closeModal(page);
});

// ---------------------------------------------------------------------------
// 4) Optimizer change (Lion via OptimTab) → extras.optimizer_kind=='lion'
// ---------------------------------------------------------------------------

test("UI optimizer change to Lion propagates to extras.optimizer_kind", async ({
  page,
}) => {
  await gotoApp(page);
  await selectPreset(page, "llama3_8b");

  await page.getByTestId("sidebar-tab-optim").click();
  // OptimTab kind selector for group 0
  await page.getByTestId("optim-kind").selectOption("lion");
  await page.getByTestId("optim-apply").click();

  await clickRunPipeline(page, "train");
  const extras = await readTrainExtras(page);

  // The big V3-1 / B1 assertion:
  expect(extras.optimizer_kind).toBe("lion");
  expect(extras.model_summary.optimizer_kind).toBe("lion");
  expect(extras.weight_delta_norm).toBeGreaterThan(0);

  await closeModal(page);
});

// ---------------------------------------------------------------------------
// 5) Optimizer change to Muon → different math from AdamW (B1 regression)
// ---------------------------------------------------------------------------

test("UI optimizer change to Muon propagates and produces weight delta",
  async ({ page }) => {
    await gotoApp(page);
    await selectPreset(page, "llama3_8b");

    await page.getByTestId("sidebar-tab-optim").click();
    await page.getByTestId("optim-kind").selectOption("muon");
    await page.getByTestId("optim-apply").click();

    await clickRunPipeline(page, "train");
    const extras = await readTrainExtras(page);

    expect(extras.optimizer_kind).toBe("muon");
    expect(extras.model_summary.optimizer_kind).toBe("muon");
    expect(extras.weight_delta_norm).toBeGreaterThan(0);
    expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

    await closeModal(page);
  });
