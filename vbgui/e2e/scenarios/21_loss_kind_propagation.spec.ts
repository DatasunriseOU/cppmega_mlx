// V4-7: All 4 (non-custom) LossKind values propagate UI → stage_train
// → extras.loss_kind + extras.model_summary.loss_kind. V3-5 only proved
// cross_entropy.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

interface Scenario {
  kind: string;
  fields?: { testid: string; value: string }[];
}

const SCENARIOS: Scenario[] = [
  { kind: "cross_entropy" },
  { kind: "mtp_weighted",
    fields: [
      { testid: "loss-mtp-k", value: "2" },
      { testid: "loss-mtp-beta", value: "0.5" },
    ] },
  { kind: "ifim_shaped",
    fields: [{ testid: "loss-ifim-lambda", value: "0.1" }] },
  { kind: "mhc_attn_bias",
    fields: [{ testid: "loss-mhc-lambda", value: "0.05" }] },
];

for (const { kind, fields } of SCENARIOS) {
  test(`V4-7: loss kind '${kind}' propagates to extras.loss_kind`,
    async ({ page }) => {
      test.setTimeout(60_000);
      await gotoApp(page);
      await selectPreset(page, "llama3_8b");

      await page.getByTestId("sidebar-tab-loss").click();
      await page.getByTestId("loss-tab").waitFor();
      await page.getByTestId("loss-kind").selectOption(kind);
      for (const f of fields ?? []) {
        await page.getByTestId(f.testid).fill(f.value);
      }
      await page.getByTestId("loss-apply").click();

      await page.getByTestId("run-pipeline-toggle").click();
      await page.getByTestId("run-pipeline-train").click();
      const modal = page.getByTestId("run-result-modal");
      await modal.waitFor({ timeout: 60_000 });
      const extras = await readTrainExtras(page);

      // Propagation assertion: loss_kind reaches extras AND model_summary
      const lossText = await page.getByTestId(
        "run-result-extras-train-loss_kind").textContent();
      expect(lossText?.trim()).toBe(kind);
      expect(extras.model_summary).toBeDefined();
      // model_summary.loss_kind in dict-style render
      const msLossText = await page.getByTestId(
        "run-result-extras-train-model_summary-loss_kind").textContent();
      expect(msLossText?.trim()).toBe(kind);
      // Train ran end-to-end
      expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

      await closeModal(page);
    });
}
