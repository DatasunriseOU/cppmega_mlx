import { expect, test } from "@playwright/test";

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

const ART_DIR = "/Users/dave/.gemini/antigravity-cli/brain/adc0dd7f-7358-462b-b86f-95349940d112";

test.describe("Interactive Neural Debugger & Sweep Validator", () => {
  // Scenario 1: Matrix Preset Debugger Check (All 62 Presets)
  test("Matrix Presets Debugger Verification (All 62 Presets)", async ({ page }) => {
    test.setTimeout(240_000); // 4 minutes max
    await page.goto("/");
    await page.waitForTimeout(1500);
    
    for (const preset of ALL_PRESETS) {
      console.log(`[Preset Debugger Boot Check] ${preset}`);
      
      // Select Preset in launcher
      await page.getByTestId("preset-launcher").selectOption(preset);
      
      // Wait for wizard modal
      const wizard = page.getByTestId("llm-wizard-generate");
      await expect(wizard).toBeVisible({ timeout: 5000 });
      
      // Downscale and set 2 layers
      await page.locator("button:has-text('1/32')").click();
      await page.locator("input[type='number']").fill("2");
      await wizard.click();
      
      // Wait for canvas render
      await expect.poll(async () =>
        await page.locator("[data-testid^='brick-node-']").count(),
        { timeout: 8000 }
      ).toBeGreaterThan(0);
      
      // Toggle Debugger Mode
      const toggleBtn = page.getByTestId("toggle-debugger-mode");
      await toggleBtn.click();
      
      // Wait for virtual nodes & dashboard to boot
      const dashboard = page.getByTestId("debugger-dashboard");
      await expect(dashboard).toBeVisible({ timeout: 5000 });
      
      const tokenizerNode = page.getByTestId("tokenizer-virtual-node");
      await expect(tokenizerNode).toBeVisible({ timeout: 2000 });
      
      // Toggle Debugger Mode back off
      await toggleBtn.click();
      await expect(dashboard).not.toBeVisible();
    }
  });

  // Scenario 2: Activation Sweep Verification (11 activations)
  test("Activation Sweep Verification (11 activations)", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.waitForTimeout(1500);
    
    // Select llama3_8b and generate downscaled
    await page.getByTestId("preset-launcher").selectOption("llama3_8b");
    const wizard = page.getByTestId("llm-wizard-generate");
    await expect(wizard).toBeVisible();
    await page.locator("button:has-text('1/32')").click();
    await page.locator("input[type='number']").fill("2");
    await wizard.click();
    
    // Wait for nodes
    await expect.poll(async () =>
      await page.locator("[data-testid^='brick-node-']").count(),
      { timeout: 8000 }
    ).toBeGreaterThan(0);
    
    const ACTIVATIONS = [
      "glu", "gelu", "relu", "relu2", "sqrelu", "silu", "mish",
      "swiglu", "geglu", "reglu", "xielu"
    ];
    
    for (const activation of ACTIVATIONS) {
      console.log(`Testing activation: ${activation}`);
      
      // Open MLP/gated_mlp brick context panel (excluding sidebar palette using class filter)
      const mlpNode = page.locator(".react-flow__node[data-testid*='gated_mlp'], .react-flow__node[data-testid*='mlp']").first();
      const mlpId = await mlpNode.getAttribute("data-testid").then(id => id?.replace("brick-node-", ""));
      await mlpNode.click();
      
      // Select activation
      const select = page.getByTestId(`brick-context-${mlpId}-activation`);
      await select.selectOption(activation);
      
      // Save
      await page.getByTestId(`brick-context-${mlpId}-apply`).click();
      
      // Open Debugger
      const toggleBtn = page.getByTestId("toggle-debugger-mode");
      await toggleBtn.click();
      
      const dashboard = page.getByTestId("debugger-dashboard");
      await expect(dashboard).toBeVisible();
      
      // Step forward once
      await page.getByTestId("debugger-btn-step-fwd").click();
      await expect(dashboard).toContainText("Step Index: 0");
      
      // Close Debugger
      await toggleBtn.click();
      await expect(dashboard).not.toBeVisible();
    }
  });

  // Scenario 3: Optimizer Sweep Verification (10 optimizers)
  test("Optimizer Sweep Verification (10 optimizers)", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.waitForTimeout(1500);
    
    // Select llama3_8b and generate downscaled
    await page.getByTestId("preset-launcher").selectOption("llama3_8b");
    const wizard = page.getByTestId("llm-wizard-generate");
    await expect(wizard).toBeVisible();
    await page.locator("button:has-text('1/32')").click();
    await page.locator("input[type='number']").fill("2");
    await wizard.click();
    
    await expect.poll(async () =>
      await page.locator("[data-testid^='brick-node-']").count(),
      { timeout: 8000 }
    ).toBeGreaterThan(0);
    
    const OPTIMIZERS = [
      "adamw", "muon", "muon_adamw_hybrid", "lion", "lion8bit",
      "adam8bit", "sgd", "adam", "adafactor", "rmsprop"
    ];
    
    // Select the OptimTab
    await page.getByTestId("sidebar-tab-optim").click();
    
    for (const optimizer of OPTIMIZERS) {
      console.log(`Testing optimizer: ${optimizer}`);
      
      // Select optimizer kind
      await page.getByTestId("optim-kind").selectOption(optimizer);
      
      // Apply changes
      await page.getByTestId("optim-apply").click();
      
      // Open Debugger
      const toggleBtn = page.getByTestId("toggle-debugger-mode");
      await toggleBtn.click();
      
      const dashboard = page.getByTestId("debugger-dashboard");
      await expect(dashboard).toBeVisible();
      
      // Run automated Full Train Step animation
      const fullTrainBtn = page.getByTestId("debugger-btn-full-train");
      await expect(fullTrainBtn).toBeVisible();
      await fullTrainBtn.click();
      
      // Wait for the gold optimizer weight update pulse to appear
      const pulseElement = page.getByTestId("debugger-weight-update-pulse");
      await expect(pulseElement).toBeVisible({ timeout: 10000 });
      
      // Close Debugger
      await toggleBtn.click();
      await expect(dashboard).not.toBeVisible();
    }
  });

  // Scenario 4: Deep 40-Step Simulation & Screenshot Capture
  test("40-Step Deep Simulation Walkthrough", async ({ page }) => {
    test.setTimeout(90_000);
    await page.goto("/");
    await page.waitForTimeout(1500);
    
    // Select llama3_8b preset
    await page.getByTestId("preset-launcher").selectOption("llama3_8b");
    const wizard = page.getByTestId("llm-wizard-generate");
    await expect(wizard).toBeVisible();
    await page.locator("button:has-text('1/32')").click();
    await page.locator("input[type='number']").fill("2");
    await wizard.click();
    
    await expect.poll(async () =>
      await page.locator("[data-testid^='brick-node-']").count(),
      { timeout: 8000 }
    ).toBeGreaterThan(0);
    
    // Open Debugger
    const toggleBtn = page.getByTestId("toggle-debugger-mode");
    await toggleBtn.click();
    
    const dashboard = page.getByTestId("debugger-dashboard");
    await expect(dashboard).toBeVisible();
    await expect(dashboard).toContainText("Step Index: -1");
    
    // Take Step 1 screenshot: Tokenized prompt
    await page.screenshot({ path: `${ART_DIR}/step1_dbg_init.png` });
    
    const stepFwdBtn = page.getByTestId("debugger-btn-step-fwd");
    const stepBwdBtn = page.getByTestId("debugger-btn-step-bwd");
    
    // Step forward 20 times (Forward Pass execution)
    for (let i = 0; i < 20; i++) {
      await stepFwdBtn.click();
      await page.waitForTimeout(50); // fast simulation
      
      if (i === 1) {
        // Embedder node step
        await page.screenshot({ path: `${ART_DIR}/step2_dbg_fwd_embed.png` });
      } else if (i === 4) {
        // Attention brick step
        await page.screenshot({ path: `${ART_DIR}/step3_dbg_fwd_attn.png` });
      } else if (i === 19) {
        // Loss computations step
        await page.screenshot({ path: `${ART_DIR}/step4_dbg_loss_logits.png` });
      }
    }
    
    // Step backward 20 times (Backprop simulation)
    for (let i = 0; i < 20; i++) {
      await stepBwdBtn.click();
      await page.waitForTimeout(50);
    }
    
    // Final check to verify it safely returned to prompt tokenize state
    await expect(dashboard).toContainText("Step Index: -1");
    await page.screenshot({ path: `${ART_DIR}/step5_dbg_backprop_complete.png` });
  });
});
