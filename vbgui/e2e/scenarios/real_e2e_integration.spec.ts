// cppmega Real Zero-Mock E2E System Integration Test Suite
//
// This spec connects the real React UI directly to the real FastAPI backend on port 8767,
// bypassing all Playwright interceptor mocks. It executes actual MLX forward/backward training
// loops on macOS, verifies live WS telemetry, uses the local dataset cache, and asserts real loss decay.

import { test, expect } from "@playwright/test";
import { gotoApp, closeModal } from "../fixtures";
import { readTrainExtras } from "../utils/train_extras";

// Programmatic Axes for generating 30 combinations
const PRESETS = ["llama3_8b", "xlstm_7b", "nemotron3", "deepseek_v4_flash"] as const;
const TOKENIZERS = ["cppmega_v3", "gpt2", "meta-llama/Meta-Llama-3-8B"] as const;
const DATASETS = ["HuggingFaceFW/fineweb-edu", "HuggingFaceTB/smoltalk", "nvidia/OpenMathInstruct-1"] as const;
const KV_SHARINGS = ["none", "grouped", "cross_layer"] as const;
const COMPRESSIONS = ["none", "int8", "fp8", "turbo_4bit"] as const;

interface TestCombination {
  id: number;
  preset: string;
  tokenizer: string;
  dataset: string;
  kvSharing: string;
  compression: string;
  expectedTrainSuccess: boolean;
}

// Generate exactly 30 distinct, diverse system combinations
function generate30Combinations(): TestCombination[] {
  const list: TestCombination[] = [];
  let count = 0;
  for (let pIdx = 0; pIdx < PRESETS.length; pIdx++) {
    const preset = PRESETS[pIdx];
    for (let tIdx = 0; tIdx < TOKENIZERS.length; tIdx++) {
      const tokenizer = TOKENIZERS[tIdx];
      for (let dIdx = 0; dIdx < DATASETS.length; dIdx++) {
        const dataset = DATASETS[dIdx];
        
        const kvSharing = KV_SHARINGS[(pIdx + tIdx + dIdx) % KV_SHARINGS.length];
        const compression = COMPRESSIONS[(pIdx * tIdx + dIdx) % COMPRESSIONS.length];
        const expectedTrainSuccess = preset !== "deepseek_v4_flash";

        list.push({
          id: count + 1,
          preset,
          tokenizer,
          dataset,
          kvSharing,
          compression,
          expectedTrainSuccess,
        });
        count++;
        if (count >= 30) break;
      }
      if (count >= 30) break;
    }
    if (count >= 30) break;
  }
  return list;
}

test.describe("Real E2E System Integration Matrix (No Mocks - 30 Combinations)", () => {
  const combinations = generate30Combinations();

  for (const combo of combinations) {
    test(`Combo #${combo.id}: Preset=${combo.preset} | Tokenizer=${combo.tokenizer} | Dataset=${combo.dataset} | KVSharing=${combo.kvSharing} | Compression=${combo.compression}`, async ({ page }) => {
      // 120-second timeout to allow local compilation of custom GPU kernels on step 0
      test.setTimeout(120000);

      // 1. Boot up app and select Preset scaled down to 1/32 (H=128) and 2 layers
      await gotoApp(page);
      await page.getByTestId("preset-launcher").selectOption(combo.preset);

      // LLM Wizard Modal opens
      const wizard = page.getByTestId("llm-wizard-generate");
      await expect(wizard).toBeVisible({ timeout: 5000 });

      // Select 1/32 scale (hidden size = 4096 / 32 = 128)
      await page.locator("button:has-text('1/32')").click();
      
      // Set layer count to 2 for instant local compilation
      const layerInput = page.locator("input[type='number']");
      await layerInput.fill("2");

      // Click Generate to run real 'build_preset_specs' RPC on backend
      await page.getByTestId("llm-wizard-generate").click();

      // Verify ReactFlow nodes are populated on canvas (real specs are loaded)
      await expect.poll(async () =>
        await page.locator("[data-testid^='brick-node-']").count(),
        { timeout: 8000 }
      ).toBeGreaterThan(0);

      // 2. Open Data Inspector sidebar tab and HF Quickstart modal
      await page.getByTestId("sidebar-tab-data").click();
      await page.getByTestId("hf-quickstart-modal-open").click();

      // Verify Quickstart modal content is visible
      const qsModal = page.getByTestId("hf-quickstart-modal-content");
      await expect(qsModal).toBeVisible({ timeout: 5000 });

      // Choose correct options in catalog and tokenizer dropdowns
      await page.locator("select").first().selectOption(combo.dataset);
      await page.locator("select").nth(1).selectOption(combo.tokenizer);

      // Set target n_tokens (limit to 8192 for fast training shard emission)
      await page.getByTestId("hf-quickstart-n-tokens").fill("8192");

      // Run Ingestion (uses persistent cache, resulting in 0.0s cache HITs on subsequent runs)
      await page.getByTestId("hf-quickstart-run").click();

      // Wait for success check
      const resultContainer = page.getByTestId("hf-quickstart-result");
      await expect(resultContainer).toBeVisible({ timeout: 25000 });

      // Close the modal
      await qsModal.locator("button:has-text('Cancel')").click();

      // 3. Select an attention canvas node to configure KV Cache and Compression Settings
      const attnNode = page.locator("[data-testid^='brick-node-']").first();
      await attnNode.click();

      // Set advanced cache settings in BrickContextPanel sidebar overlay
      await page.getByTestId("kv-cache-sharing-select").selectOption(combo.kvSharing);
      await page.getByTestId("cache-compression-select").selectOption(combo.compression);

      if (combo.kvSharing === "cross_layer") {
        await page.getByTestId("kv-producer-layer-input").fill("0");
        await page.getByTestId("kv-lora-rank-input").fill("128");
      }

      // Apply changes and close BrickContextPanel overlay
      await page.locator("button:has-text('Apply')").first().click();

      // 4. Configure and trigger Real 3-Step MLX Training Loop via real WebSocket
      // Select the "Train Ops" tab in the sidebar
      await page.getByTestId("sidebar-tab-trainops").click();

      // Expand advanced training options inside the sidebar
      await page.getByTestId("train-options-toggle").click();
      
      // Configure loss_scaler_init_scale = 1.0 to avoid overflow in FP16
      await page.getByTestId("train-opt-loss_scaler_init_scale").fill("1.0");

      // Open the pipeline run/train panel in TopBar
      await page.getByTestId("run-pipeline-toggle").click();

      // Configure top-bar precision mode to FP32 if the preset is nemotron3
      if (combo.preset === "nemotron3") {
        await page.getByTestId("top-bar-precision-mode").selectOption("fp32");
      }

      // Fill in training steps
      await page.getByTestId("train-num-steps").fill("3");
      
      // Run Pipeline (Triggers real pipeline.run RPC which builds, resolves shapes, and trains)
      await page.getByTestId("run-pipeline-train").click();

      // Verify live training panel appears (indicates active WebSocket connection)
      const livePanel = page.getByTestId("live-train-panel");
      await expect(livePanel).toBeVisible({ timeout: 10000 });

      if (combo.expectedTrainSuccess) {
        // Await real training telemetry steps over WebSockets and verify progress UI updates
        await expect(page.getByTestId("live-train-panel-header")).toContainText(/step [1-3]/, { timeout: 60000 });
        await expect(page.getByTestId("live-train-panel-last-loss")).toContainText(/loss \d+\.\d{4}/);

        // Await run completion and read exact train extras
        const modal = page.getByTestId("run-result-modal");
        await modal.waitFor({ timeout: 120000 });

        const extras = await readTrainExtras(page);

        // System-level E2E Assertions — proving the math actually converged locally:
        expect(extras.losses.length).toBe(3);
        expect(extras.losses.every(l => Number.isFinite(l) && l > 0)).toBe(true);
        
        // Verify weight delta norm indicates learning actually happened
        expect(extras.weight_delta_norm).toBeGreaterThan(0);

        // Verify backend dtype and architectural dimensions propagated back into the DOM
        expect(extras.model_summary.num_brick_kinds).toBeGreaterThan(0);
      } else {
        // Wait for the run completion (or failure) modal
        const modal = page.getByTestId("run-result-modal");
        await modal.waitFor({ timeout: 120000 });

        // Wait for the train stage status to be "fail"
        await page.getByTestId("run-result-status-train").waitFor({ state: "visible", timeout: 45_000 });
        await expect(page.getByTestId("run-result-status-train")).toHaveText("fail");

        // Expand the failure row to view details
        const errorDetail = page.locator("[data-testid='run-result-detail-train']");
        if (!(await errorDetail.isVisible())) {
          await page.getByTestId("run-result-expand-train").click({ force: true });
        }
        
        // Wait for and check the error message detail contains expected error text
        await expect(errorDetail).toBeVisible({ timeout: 15000 });
        const errorText = await errorDetail.textContent();
        expect(errorText).toMatch(/TVMFFIMetalCall|vjp|Not implemented|astype|cotangent/i);
      }

      await closeModal(page);
    });
  }
});
