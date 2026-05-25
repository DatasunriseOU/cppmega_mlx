import { test, expect } from "@playwright/test";

const ALL_PRESETS = [
  "arcee_trinity", "deepseek_v3", "deepseek_v4_flash", "gemma3_270m", "gemma3_27b",
  "gemma4", "gemma4_31b", "glm_45", "glm_45_air", "glm_47", "glm_5", "glm_51",
  "gpt_oss_120b", "gpt_oss_20b", "granite_4_1", "grok25", "intellect_3", "kimi_k2",
  "kimi_linear", "laguna_xs2", "ling25", "ling26", "llama3_2_1b", "llama3_2_3b",
  "llama3_8b", "llama4_maverick", "longcat", "mimo_v2_5", "mimo_v2_5_pro", "mimo_v2_flash",
  "minimax_m2", "minimax_m2_5", "minimax_m2_7", "mistral4", "mistral_small_3_1",
  "nanbeige_4_1", "nemotron3", "olmo2_7b", "olmo3_32b", "olmo3_7b", "phi4",
  "qwen3_235b_a22b", "qwen3_30b_a3b", "qwen3_6_27b", "qwen3_coder_flash", "qwen3_dense_0_6b",
  "qwen3_dense_32b", "qwen3_dense_4b", "qwen3_dense_8b", "qwen3_next", "sarvam_105b",
  "sarvam_30b", "smollm3", "step3_5_flash", "tencent_hy3", "tiny_aya", "zaya1",
  "gpt2_xl", "tiny_aya_parallel", "xlstm_7b", "gemma_4_e2b", "gemma_4_e4b"
] as const;

const CORE_PRESETS = ["llama3_8b", "xlstm_7b", "nemotron3", "deepseek_v4_flash"] as const;
const ART_DIR = "/Users/dave/.gemini/antigravity-cli/brain/adc0dd7f-7358-462b-b86f-95349940d112";

test.describe("All Presets Canvas Verification", () => {
  for (const preset of ALL_PRESETS) {
    test(`Capture Wizard & Canvas for Preset: ${preset}`, async ({ page }) => {
      // Fast 30-second timeout for canvas resolution
      test.setTimeout(30000);

      // Connect directly to the dev server
      await page.goto("/");
      await page.waitForTimeout(1000);

      // 1. Select Preset in dropdown launcher
      await page.getByTestId("preset-launcher").selectOption(preset);

      // Await wizard modal to become visible
      const wizard = page.getByTestId("llm-wizard-generate");
      await expect(wizard).toBeVisible({ timeout: 5000 });

      // Downscale to 1/32 and set layers to 2 for instant compilation
      await page.locator("button:has-text('1/32')").click();
      await page.locator("input[type='number']").fill("2");

      // Take step 1 screenshot: Wizard Active
      await page.screenshot({ path: `${ART_DIR}/step1_wizard_${preset}.png` });

      // Click Generate to build specs dynamically on the backend
      await page.getByTestId("llm-wizard-generate").click();

      // Await canvas render showing visual brick nodes
      await expect.poll(async () =>
        await page.locator("[data-testid^='brick-node-']").count(),
        { timeout: 10000 }
      ).toBeGreaterThan(0);

      // Take step 2 screenshot: Canvas Loaded
      await page.screenshot({ path: `${ART_DIR}/step2_canvas_${preset}.png` });
    });
  }
});

test.describe("Core Representative Presets Ingest & Train", () => {
  for (const preset of CORE_PRESETS) {
    test(`Capture Ingest & Train for Preset: ${preset}`, async ({ page }) => {
      // 120-second timeout to allow local compilation of custom GPU kernels
      test.setTimeout(120000);

      // Connect directly to the dev server
      await page.goto("/");
      await page.waitForTimeout(1500);

      // 1. Select Preset
      await page.getByTestId("preset-launcher").selectOption(preset);

      const wizard = page.getByTestId("llm-wizard-generate");
      await expect(wizard).toBeVisible({ timeout: 5000 });

      // Downscale & Set 2 layers
      await page.locator("button:has-text('1/32')").click();
      await page.locator("input[type='number']").fill("2");
      await page.getByTestId("llm-wizard-generate").click();

      // Await canvas render
      await expect.poll(async () =>
        await page.locator("[data-testid^='brick-node-']").count(),
        { timeout: 10000 }
      ).toBeGreaterThan(0);

      // 2. Select data tab and trigger quickstart ingestion
      await page.getByTestId("app-tab-data").click();
      await page.getByTestId("hf-quickstart-modal-open").click();

      // Configure default quickstart options inside the modal
      const modalContent = page.getByTestId("hf-quickstart-modal-content");
      await modalContent.locator("select").first().selectOption("HuggingFaceFW/fineweb-edu");
      await modalContent.locator("select").nth(1).selectOption("cppmega_v3");
      await page.getByTestId("hf-quickstart-n-tokens").fill("8192");

      // Ingest
      await page.getByTestId("hf-quickstart-run").click();
      await expect(page.getByTestId("hf-quickstart-result")).toBeVisible({ timeout: 25000 });

      // Take step 3 screenshot: Data Cached
      await page.screenshot({ path: `${ART_DIR}/step3_data_${preset}.png` });

      // Close modal
      await modalContent.locator("button:has-text('Cancel')").click();

      // Go back to Canvas tab
      await page.getByTestId("app-tab-canvas").click();

      // 3. Select train ops, set loss scale, and start training
      await page.getByTestId("sidebar-tab-trainops").click();
      await page.getByTestId("train-options-toggle").click();
      await page.getByTestId("train-opt-loss_scaler_init_scale").fill("1.0");

      await page.getByTestId("run-pipeline-toggle").click();
      if (preset === "nemotron3") {
        await page.getByTestId("top-bar-precision-mode").selectOption("fp32");
      }
      await page.getByTestId("train-num-steps").fill("3");

      // Run pipeline
      await page.getByTestId("run-pipeline-train").click();

      // Wait for training telemetry to appear in live panel
      const livePanel = page.getByTestId("live-train-panel");
      await expect(livePanel).toBeVisible({ timeout: 10000 });

      if (preset !== "deepseek_v4_flash") {
        // Successful run
        await expect(page.getByTestId("live-train-panel-header")).toContainText(/step [1-3]/, { timeout: 45000 });
        await page.waitForTimeout(1000); // let UI update loss text
        
        // Take step 4 screenshot: Training Progress
        await page.screenshot({ path: `${ART_DIR}/step4_train_${preset}.png` });

        // Wait for completion modal
        const modal = page.getByTestId("run-result-modal");
        await modal.waitFor({ timeout: 45000 });
        
        // Take step 5 screenshot: Completed
        await page.screenshot({ path: `${ART_DIR}/step5_complete_${preset}.png` });
      } else {
        // Expected fail case (DeepSeek custom kernel exception)
        const modal = page.getByTestId("run-result-modal");
        await modal.waitFor({ timeout: 45000 });
        await page.getByTestId("run-result-status-train").waitFor({ state: "visible", timeout: 15000 });
        
        // Expand errors
        const errorDetail = page.locator("[data-testid='run-result-detail-train']");
        if (!(await errorDetail.isVisible())) {
          await page.getByTestId("run-result-expand-train").click({ force: true });
        }
        await expect(errorDetail).toBeVisible({ timeout: 10000 });

        // Take step 4 (failure) screenshot
        await page.screenshot({ path: `${ART_DIR}/step4_fail_${preset}.png` });
      }

      // Close the final pipeline run modal
      await page.getByTestId("run-result-close").click();
    });
  }
});
