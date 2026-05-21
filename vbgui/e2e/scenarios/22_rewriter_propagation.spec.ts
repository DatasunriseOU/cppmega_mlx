// V4-8: RewritersTab chain reaches stage_train via spec.rewriters →
// extras.model_summary.rewriters_applied. Closes G7 from V4 audit.

import { test, expect } from "@playwright/test";
import { gotoApp, selectPreset, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

const REWRITERS = ["MTPRewriter", "IFIMRewriter", "MHCRewriter"];

for (const rw of REWRITERS) {
  test(`V4-8: rewriter '${rw}' propagates to extras.model_summary`,
    async ({ page }) => {
      test.setTimeout(60_000);
      await gotoApp(page);
      await selectPreset(page, "llama3_8b");

      await page.getByTestId("sidebar-tab-rewriters").click();
      await page.getByTestId("rewriters-tab").waitFor();
      await page.getByTestId(`rewriter-add-${rw}`).click();
      // Chip should appear
      await expect(page.getByTestId("rewriter-chip-0")).toContainText(rw);
      await page.getByTestId("rewriter-apply").click();

      await page.getByTestId("run-pipeline-toggle").click();
      await page.getByTestId("run-pipeline-train").click();
      const modal = page.getByTestId("run-result-modal");
      await modal.waitFor({ timeout: 60_000 });
      const extras = await readTrainExtras(page);

      // Top-level extras has the list (array DOM testid pattern)
      const rwListRoot = page.locator(
        "[data-testid^='run-result-extras-train-model_summary-rewriters_applied']");
      const count = await rwListRoot.count();
      // model_summary.rewriters_applied renders as nested object → dl entry
      // OR as array → ol. Either way the chosen rewriter name must appear.
      // Easier: assert top-level extras renders the array via the recursive
      // ExtrasEntry pattern (model_summary is an object → values rendered
      // as JSON.stringify when nested). Check via JSON parse of cell text.
      const msEntry = await page.getByTestId(
        "run-result-extras-train-model_summary-rewriters_applied").textContent();
      expect(msEntry).toContain(rw);
      expect(extras.losses.every(l => Number.isFinite(l))).toBe(true);

      await closeModal(page);
    });
}
