// Preset matrix — 57 PRESETS × 4 tokenizer × 4 parquet variants = 912
// scenarios. Every cell walks the GUI end-to-end:
//   open / → preset launcher → tokenizer tab + load → data tab + load →
//   canvas → Smoke → assert modal overall=ok → screenshot.
//
// Known cells expected to fail are marked via EXPECTED_FAILURES so the
// matrix report stays signal-rich (a cell flipping from xfail to pass
// surfaces in the report's "unexpected_pass" bucket).

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

// 57 PRESETS (mirrors cppmega_v4.architectures.PRESETS).
const PRESETS = [
  "arcee_trinity", "deepseek_v3", "deepseek_v4_flash",
  "gemma3_270m", "gemma3_27b", "gemma4", "gemma4_31b",
  "glm_45", "glm_45_air", "glm_47", "glm_5", "glm_51",
  "gpt_oss_120b", "gpt_oss_20b", "granite_4_1", "grok25",
  "intellect_3", "kimi_k2", "kimi_linear", "laguna_xs2",
  "ling25", "ling26", "llama3_2_1b", "llama3_2_3b", "llama3_8b",
  "llama4_maverick", "longcat", "mimo_v2_5", "mimo_v2_5_pro",
  "mimo_v2_flash", "minimax_m2", "minimax_m2_5", "minimax_m2_7",
  "mistral4", "mistral_small_3_1", "nanbeige_4_1", "nemotron3",
  "olmo2_7b", "olmo3_32b", "olmo3_7b", "phi4",
  "qwen3_235b_a22b", "qwen3_30b_a3b", "qwen3_6_27b",
  "qwen3_coder_flash", "qwen3_dense_0_6b", "qwen3_dense_32b",
  "qwen3_dense_4b", "qwen3_dense_8b", "qwen3_next",
  "sarvam_105b", "sarvam_30b", "smollm3", "step3_5_flash",
  "tencent_hy3", "tiny_aya", "zaya1",
] as const;

// Cells where Smoke is expected to fail on the mini-spec (so flagged
// xfail not red). Populate empirically — start empty, add as we observe.
const EXPECTED_FAILURES: Set<string> = new Set();

const SCENARIOS = PRESETS.flatMap((preset) =>
  TOKENIZER_NAMES.flatMap((tok) =>
    PARQUET_SCHEMAS.map((parq) => ({ preset, tok, parq,
                                     key: `${preset}__${tok}__${parq}` })),
  ),
);

test.describe("preset matrix (912 cells)", () => {
  for (const { preset, tok, parq, key } of SCENARIOS) {
    test(key, async ({ page }) => {
      const matrix = loadMatrix();
      const tokPath = matrix.tokenizers[tok].path;
      const parqPath = matrix.parquets[`${tok}__${parq}`].path;

      await gotoApp(page);
      await selectPreset(page, preset);

      await loadTokenizerInPlayground(page, tokPath);
      await loadParquetInInspector(page, parqPath);

      const modal = await clickRunPipeline(page, "smoke");
      if (EXPECTED_FAILURES.has(key)) {
        await assertOverallStatus(modal, "fail");
        await snapshot(page, "02_preset_matrix/xfail", key);
      } else {
        await assertOverallStatus(modal, "ok");
        // Capture screenshot only on the first 30 cells to keep the
        // artefact bundle bounded; failures already screenshot via
        // playwright.config.ts screenshot:"only-on-failure".
        if (Math.random() < 0.04) {
          await snapshot(page, "02_preset_matrix/sample", key);
        }
      }
      await closeModal(page);
      // sanity assertion for the matrix-report generator
      expect(SCENARIOS.length).toBe(912);
    });
  }
});
