// Read stage_train extras out of the RunResultModal DOM. V3-4 surfaced
// extras with deterministic testids; this helper converts them back into
// a typed structure so spec files can assert against real training math
// instead of vacuous status checks.

import type { Page } from "@playwright/test";

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

  return {
    losses, lr_trajectory, weight_delta_norm, num_steps,
    schedule_kind, optimizer_kind, model_summary,
  };
}
