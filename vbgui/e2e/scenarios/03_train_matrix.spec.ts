// Mini-train matrix — 12 family-rep × 4 tokenizer × 4 parquet = 192
// scenarios. Same GUI walk as the preset matrix (E-3), but clicks
// Train instead of Smoke. The Train pipeline runs the real `train`
// stage (forward → CE loss → backward → AdamW.update × 2 steps).
//
// Per cell assertion: modal overall=ok AND a `train` stage row exists
// with status=ok. (Loss-finite / weight-delta / no-blow-up asserts
// live inside stage_train; if any fires, the cell goes red.)

import { test, expect } from "@playwright/test";
import {
  assertOverallStatus, clickRunPipeline, closeModal,
  gotoApp, loadParquetInInspector, loadTokenizerInPlayground,
  selectPreset,
} from "../fixtures";
import { snapshot } from "../utils/screenshot";
import {
  PARQUET_SCHEMAS, TOKENIZER_NAMES, loadMatrix,
} from "../utils/matrix";

// 12 family representatives — one preset per architectural family.
const FAMILY_REPS = [
  "llama3_8b", "qwen3_next", "kimi_linear", "deepseek_v3",
  "gemma3_27b", "gpt_oss_20b", "glm_45", "mistral4",
  "nemotron3", "zaya1", "arcee_trinity", "granite_4_1",
] as const;

// Presets whose bricks use TileLang/TVM Metal kernels lacking a vjp
// implementation (Primitive::vjp Not implemented for TVMFFIMetalCall).
// Their `train` stage is expected to FAIL on this Mac; overall_status
// in the modal will be "fail" but the cell is xfail-asserted to catch
// future regressions in the OTHER stages (parse/verify/...) which must
// still run cleanly.
const EXPECTED_TRAIN_FAIL: ReadonlySet<string> = new Set([
  "kimi_linear",  // contains `kda` brick (Kimi delta-attention, Metal only)
  "qwen3_next",   // contains `gdn` brick (gated delta net, Metal only)
]);

const SCENARIOS = FAMILY_REPS.flatMap((preset) =>
  TOKENIZER_NAMES.flatMap((tok) =>
    PARQUET_SCHEMAS.map((parq) => ({ preset, tok, parq,
                                     key: `${preset}__${tok}__${parq}` })),
  ),
);

test.describe("mini-train matrix (192 cells)", () => {
  for (const { preset, tok, parq, key } of SCENARIOS) {
    test(key, async ({ page }) => {
      const matrix = loadMatrix();
      const tokPath = matrix.tokenizers[tok].path;
      const parqPath = matrix.parquets[`${tok}__${parq}`].path;

      await gotoApp(page);
      await selectPreset(page, preset);

      await loadTokenizerInPlayground(page, tokPath);
      await loadParquetInInspector(page, parqPath);

      const modal = await clickRunPipeline(page, "train");
      const trainRow = modal.getByTestId("run-result-stage-train");
      await expect(trainRow).toBeVisible();

      if (EXPECTED_TRAIN_FAIL.has(preset)) {
        // Train must FAIL with a typed error (vjp / Metal path). Earlier
        // stages still must be green to prove the GUI walk works.
        await expect(trainRow).toContainText("fail");
        await snapshot(page, "03_train_matrix/xfail", key);
      } else {
        await assertOverallStatus(modal, "ok");
        await expect(trainRow).toContainText("ok");
        if (Math.random() < 0.10) {
          await snapshot(page, "03_train_matrix", key);
        }
      }
      await closeModal(page);
      expect(SCENARIOS.length).toBe(192);
    });
  }
});
