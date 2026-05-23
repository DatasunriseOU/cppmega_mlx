// Read stage_train extras out of the RunResultModal DOM. V3-4 surfaced
// extras with deterministic testids; this helper converts them back into
// a typed structure so spec files can assert against real training math
// instead of vacuous status checks.

import { expect, type Page } from "@playwright/test";

export type TrainExtras = {
  losses: number[];
  lr_trajectory: number[];
  weight_delta_norm: number;
  num_steps: number;
  schedule_kind: string;
  optimizer_kind: string;
  model_summary: {
    mlp_activation: string | null;
    attention_pre_norm: string;
    attention_post_norm: string;
    mlp_pre_norm: string;
    mlp_post_norm: string;
    optimizer_kind: string;
    schedule_kind: string;
    num_brick_kinds: number;
  };
  // V7-Q07.1: extended badge / panel fields read via TrainExtrasOverlay
  // testids. All optional — scenarios that don't activate these
  // features still see undefined and skip the assertion.
  losses_smoothed?: number[];
  val_losses?: number[];
  perplexity?: number;
  bits_per_byte?: number;
  master_dtype?: string;
  dtype_actual?: string;
  fp8_active?: boolean;
  fim_active?: boolean;
  fim_ratio?: number;
  sharding_applied?: boolean;
  side_channels_observed?: string[];
  per_brick_grad_norms?: Record<string, number>;
  routing_entropy?: number;
  load_balance_loss?: number;
  per_expert_load?: number[];
  capacity_factor?: number;
  num_experts?: number;
  gradient_reduce_ms?: number;
  loss_scaler_overflows?: number[];
};

async function textOf(page: Page, testid: string): Promise<string> {
  const t = await page.getByTestId(testid).textContent();
  if (t == null) throw new Error(`testid ${testid} produced no text`);
  return t.trim();
}

async function arrayOf(page: Page, baseTestid: string): Promise<number[]> {
  const items = page.locator(`[data-testid^='${baseTestid}-']`);
  const count = await items.count();
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const text = await page.getByTestId(`${baseTestid}-${i}`).textContent();
    if (text == null) throw new Error(`testid ${baseTestid}-${i} empty`);
    out.push(parseFloat(text.trim()));
  }
  return out;
}

export async function readTrainExtras(page: Page): Promise<TrainExtras> {
  // Wait for the train stage status to be "ok" (generous 45s timeout for compilation/run under load).
  await page.getByTestId("run-result-status-train").waitFor({ state: "visible", timeout: 45_000 });
  
  await expect(page.getByTestId("run-result-status-train")).toHaveText("ok", { timeout: 45_000 });

  // Open the train expand row.
  await page.getByTestId("run-result-expand-train").click();
  await page.getByTestId("run-result-extras-row-train").waitFor({ timeout: 10_000 });

  const losses = await arrayOf(page, "run-result-extras-train-losses");
  const lr_trajectory = await arrayOf(
    page, "run-result-extras-train-lr_trajectory");
  const weight_delta_norm = parseFloat(
    await textOf(page, "run-result-extras-train-weight_delta_norm"));
  const num_steps = parseInt(
    await textOf(page, "run-result-extras-train-num_steps"), 10);
  const schedule_kind = await textOf(
    page, "run-result-extras-train-schedule_kind");
  const optimizer_kind = await textOf(
    page, "run-result-extras-train-optimizer_kind");

  const ms_base = "run-result-extras-train-model_summary";
  const model_summary = {
    mlp_activation: (await textOf(page, `${ms_base}-mlp_activation`))
      || null,
    attention_pre_norm: await textOf(page,
      `${ms_base}-attention_pre_norm`),
    attention_post_norm: await textOf(page,
      `${ms_base}-attention_post_norm`),
    mlp_pre_norm: await textOf(page, `${ms_base}-mlp_pre_norm`),
    mlp_post_norm: await textOf(page, `${ms_base}-mlp_post_norm`),
    optimizer_kind: await textOf(page, `${ms_base}-optimizer_kind`),
    schedule_kind: await textOf(page, `${ms_base}-schedule_kind`),
    num_brick_kinds: parseInt(
      await textOf(page, `${ms_base}-num_brick_kinds`), 10),
  };

  // V7-Q07.1: read optional badge values + chart series + arrays via
  // the TrainExtrasOverlay testid contract. Each lookup is wrapped in
  // a try so scenarios that don't activate the feature stay green
  // (extras key absent on backend -> testid not in DOM -> undefined).
  async function optText(testid: string): Promise<string | undefined> {
    try {
      const txt = await page.getByTestId(testid).textContent({
        timeout: 500,
      });
      return txt?.trim();
    } catch {
      return undefined;
    }
  }
  async function optNum(testid: string): Promise<number | undefined> {
    const t = await optText(testid);
    if (t == null || t === "") return undefined;
    const n = parseFloat(t);
    return Number.isFinite(n) ? n : undefined;
  }
  async function optBool(testid: string): Promise<boolean | undefined> {
    const t = await optText(testid);
    if (t == null) return undefined;
    return /on|true|1/i.test(t);
  }
  async function optArray(base: string): Promise<number[] | undefined> {
    try {
      const count = await page.locator(`[data-testid^='${base}-']`).count();
      if (count === 0) return undefined;
      const out: number[] = [];
      for (let i = 0; i < count; i++) {
        const t = await optText(`${base}-${i}`);
        if (t == null) continue;
        const n = parseFloat(t);
        if (Number.isFinite(n)) out.push(n);
      }
      return out;
    } catch {
      return undefined;
    }
  }

  const losses_smoothed = await optArray(
    "run-result-extras-train-losses_smoothed");
  const val_losses = await optArray("run-result-extras-train-val_losses");
  const perplexity = await optNum("extras-badge-perplexity");
  const bits_per_byte = await optNum("extras-badge-bpb");
  const master_dtype = await optText("extras-badge-master_dtype");
  const dtype_actual = await optText("extras-badge-dtype_actual");
  const fp8_active = await optBool("extras-badge-fp8_active");
  const fim_active = await optBool("extras-badge-fim_active");
  const fim_ratio = await optNum("extras-badge-fim_ratio");
  const sharding_applied = await optBool("extras-sharding-panel");
  const gradient_reduce_ms = await optNum("extras-badge-gradient_reduce_ms");
  const loss_scaler_overflows = await optArray(
    "run-result-extras-train-loss_scaler_overflows");

  return {
    losses, lr_trajectory, weight_delta_norm, num_steps,
    schedule_kind, optimizer_kind, model_summary,
    losses_smoothed, val_losses, perplexity, bits_per_byte,
    master_dtype, dtype_actual, fp8_active, fim_active, fim_ratio,
    sharding_applied, gradient_reduce_ms, loss_scaler_overflows,
  };
}
