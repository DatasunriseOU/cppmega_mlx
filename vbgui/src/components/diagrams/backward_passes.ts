/**
 * Backward-pass walkthroughs per brick. Each entry describes:
 *   - differentiates: what scalar function and w.r.t. which params/inputs
 *   - chain_rule:     how the upstream cotangent ḡ = dL/dy propagates
 *                     through the brick's local Jacobian (named tensors,
 *                     explicit reverse steps, MLX `mx.grad` invocation)
 *   - key_identity:   one-line equation capturing the central VJP
 *
 * Sourced from a research agent (Exa-verified canonical forms) +
 * cross-checked against textbook derivations. Reverse-mode AD in MLX
 * = `mx.grad(fn, argnums=[...])` returns a function that maps the
 * primal arguments to dL/d{arg_i}. The cotangent ḡ shown below is
 * the gradient signal arriving from layers downstream of this brick.
 */


export interface BackwardEntry {
  differentiates: string;
  chain_rule: string;
  key_identity: string;
}


export const BACKWARD_TOPICS: Record<string, BackwardEntry> = {
  brick_attention: {
    differentiates:
      "Scalar loss L w.r.t. block inputs and projection weights: " +
      "dL/dX, dL/dW_q, dL/dW_k, dL/dW_v, dL/dW_o.",
    chain_rule:
      "Forward: S = QKᵀ/√d_k; P = softmax(S); C = PV; Y = C·W_o. " +
      "Given ḡ = dL/dY: push back as dL/dW_o = Cᵀ·ḡ and dL/dC = ḡ·W_oᵀ. " +
      "Then dL/dP = dL/dC · Vᵀ and dL/dV = Pᵀ · dL/dC. The softmax step " +
      "uses its Jacobian per row: dL/dS_ij = P_ij · (dL/dP_ij − Σ_k " +
      "P_ik · dL/dP_ik). Finally dL/dQ = (dL/dS · K)/√d_k and dL/dK = " +
      "(dL/dSᵀ · Q)/√d_k, then propagate through W_{q,k,v}. In MLX: " +
      "`mx.grad(fn, argnums=[0,1,2,3])(X, W_q, W_k, W_v, ...)`.",
    key_identity:
      "dL/dS = P ⊙ (dL/dP − rowsum(P ⊙ dL/dP)·1ᵀ)  " +
      "(softmax row-Jacobian)",
  },

  brick_gated_attention: {
    differentiates:
      "L w.r.t. dL/dX, dL/dW_{q,k,v,o}, and the gate projection dL/dW_g.",
    chain_rule:
      "Forward adds a gate: Ctx = PV (as in SDPA), G = X·W_g, " +
      "Y = (σ(G) ⊙ Ctx)·W_o. With ḡ = dL/dY: first dL/dW_o = " +
      "(σ(G)⊙Ctx)ᵀ·ḡ and dL/d(σ(G)⊙Ctx) = ḡ·W_oᵀ. Product rule splits " +
      "this into dL/dCtx = dL/d(σ(G)⊙Ctx) ⊙ σ(G) and dL/dσ(G) = " +
      "dL/d(σ(G)⊙Ctx) ⊙ Ctx. Then dL/dG = dL/dσ(G) ⊙ σ(G)(1−σ(G)) " +
      "and dL/dW_g = Xᵀ·dL/dG; the dL/dCtx branch enters the standard " +
      "SDPA backward (softmax-Jacobian, dL/dQ, dL/dK, dL/dV).",
    key_identity: "∂σ(G)/∂G = σ(G) ⊙ (1 − σ(G))",
  },

  brick_mla: {
    differentiates:
      "L w.r.t. dL/dX and the LoRA factors dL/dW_{dq,uq,dkv,uk,uv}. " +
      "RoPE phase has no learnable params.",
    chain_rule:
      "Forward: C_q = X·W_dq, Q = C_q·W_uq; C_kv = X·W_dkv, " +
      "K = C_kv·W_uk, V = C_kv·W_uv; Q,K split per head and RoPE-" +
      "rotated. SDPA backward returns dL/dQ_rot, dL/dK_rot, dL/dV. " +
      "RoPE is an orthogonal rotation R(θ), so dL/dQ = R(θ)ᵀ · " +
      "dL/dQ_rot — just rotate by the negative angle. Up-projection " +
      "VJPs: dL/dW_uq = C_qᵀ·dL/dQ and dL/dC_q = dL/dQ·W_uqᵀ, sym " +
      "for K,V (sum both into dL/dC_kv). Then dL/dW_dq = Xᵀ·dL/dC_q, " +
      "dL/dW_dkv = Xᵀ·dL/dC_kv; dL/dX accumulates from both. MLX: " +
      "`mx.grad(mla_fn, argnums=[1,2,3,4,5])(X, W_dq, W_uq, W_dkv, " +
      "W_uk, W_uv)`.",
    key_identity:
      "RoPE backward: dL/dQ = R(−θ) · dL/dQ_rot  " +
      "(rotation transpose = inverse rotation)",
  },

  brick_mlp: {
    differentiates:
      "L w.r.t. dL/dX, dL/dW_gate, dL/dW_up, dL/dW_down.",
    chain_rule:
      "Forward: a = X·W_gate, b = X·W_up, h = silu(a) ⊙ b, " +
      "Y = h·W_down. Given ḡ = dL/dY, dL/dW_down = hᵀ·ḡ and dL/dh = " +
      "ḡ·W_downᵀ. Product rule: dL/db = dL/dh ⊙ silu(a), " +
      "dL/d(silu(a)) = dL/dh ⊙ b. Then dL/da = dL/d(silu(a)) ⊙ " +
      "silu'(a) with silu'(a) = σ(a) + a·σ(a)(1−σ(a)). Finally " +
      "dL/dW_gate = Xᵀ·dL/da, dL/dW_up = Xᵀ·dL/db, and dL/dX = " +
      "dL/da·W_gateᵀ + dL/db·W_upᵀ. MLX: `mx.grad(swiglu_fn, " +
      "argnums=[1,2,3])(X, W_gate, W_up, W_down)`.",
    key_identity: "silu'(a) = σ(a) · (1 + a·(1 − σ(a)))",
  },

  brick_moe: {
    differentiates:
      "L_total = L_task + α·L_aux w.r.t. dL/dX, expert weights " +
      "dL/dW_e, and router weights dL/dW_router.",
    chain_rule:
      "Forward: ℓ = X·W_router; g = softmax(ℓ); (idx, g_topk) = " +
      "top_k(g); Y = Σ_{e∈topk} g_e · Expert_e(X). Top-k is " +
      "non-differentiable on the indices — straight-through " +
      "estimator: treat the top-k mask as a constant in backward " +
      "but pass full gradients through the selected gate weights. " +
      "dL/dExpert_e_out = g_e · ḡ feeds each chosen expert's MLP " +
      "backward (yielding dL/dW_e and a partial dL/dX). dL/dg_e = " +
      "⟨ḡ, Expert_e(X)⟩ on selected positions, then routed through " +
      "the softmax Jacobian to dL/dℓ, and dL/dW_router = Xᵀ·dL/dℓ. " +
      "Aux load-balance L_aux = N·Σ_e f_e·P_e adds α·N·f_e/T into " +
      "dL/dℓ via the same softmax VJP.",
    key_identity:
      "STE for top-k: ∂Y/∂g_e ≈ Expert_e(X) on selected experts, " +
      "0 elsewhere (mask = stop-grad)",
  },

  brick_mamba3: {
    differentiates:
      "L w.r.t. dL/dx and SSM params dL/d{A, B, C, Δ} — B, C, Δ are " +
      "input-dependent projections so each gets gradient via x.",
    chain_rule:
      "Forward recurrence: h_t = Ā_t · h_{t-1} + B̄_t · x_t, " +
      "y_t = C_t · h_t, with Ā_t = exp(Δ_t·A), B̄_t = Δ_t·B_t. The " +
      "Mamba-2 dual form rewrites this as Y = M·X where M is a " +
      "1-semiseparable matrix from cumulative products of Ā — so " +
      "backward is also a parallel scan in reverse time. Given " +
      "ḡ_t = dL/dy_t: run a reverse scan for adjoint states " +
      "s_t = C_tᵀ·ḡ_t + Ā_{t+1}ᵀ·s_{t+1}; then dL/dB_t = s_t · " +
      "x_tᵀ · Δ_t, dL/dC_t = ḡ_t · h_tᵀ, dL/dx_t = B̄_tᵀ·s_t, and " +
      "dL/dΔ_t, dL/dA accumulate through exp(Δ·A). MLX: " +
      "`mx.grad(ssm_fn, argnums=[1,2,3,4])(x, A, B, C, Δ)` — dual " +
      "form makes this O(T log T) instead of sequential.",
    key_identity:
      "Adjoint scan: s_t = Ā_{t+1}ᵀ · s_{t+1} + C_tᵀ · ḡ_t  " +
      "(reverse-time linear recurrence)",
  },

  brick_mlstm: {
    differentiates:
      "L w.r.t. dL/dx and gate/projection params dL/d{W_i, W_f, " +
      "W_o, W_q, W_k, W_v}. Running max m_t has stop-grad on the " +
      "max-selection branch.",
    chain_rule:
      "Forward: C_t = f_t·C_{t-1} + i_t·v_t·k_tᵀ ; n_t = f_t·n_{t-1} " +
      "+ i_t·k_t ; h_t = o_t ⊙ (C_t·q_t) / max(|n_tᵀ·q_t|, " +
      "exp(−m_t)). Given ḡ_t = dL/dh_t: split through output gate: " +
      "dL/do_t = ḡ_t ⊙ readout/o_t and dL/dreadout = ḡ_t ⊙ o_t. " +
      "Apply quotient rule for the max-normalised readout (max() " +
      "picks one branch — straight-through on the inactive one). " +
      "Reverse-time matrix adjoint Ĉ_t = q_t · dL/d(C_t·q_t)ᵀ + " +
      "f_{t+1}·Ĉ_{t+1} gives dL/dv_t = Ĉ_t·k_t·i_t, dL/dk_t = " +
      "Ĉ_tᵀ·v_t·i_t, dL/di_t and dL/df_t from their multiplicative " +
      "roles, each routed through their sigmoid/exp gate Jacobians.",
    key_identity:
      "Matrix-state adjoint: Ĉ_t = f_{t+1}·Ĉ_{t+1} + q_t · " +
      "∂L/∂(C_t·q_t)ᵀ",
  },

  brick_gdn: {
    differentiates:
      "L w.r.t. dL/dx and delta-rule params dL/d{W_q, W_k, W_v, W_β, " +
      "W_g} (β is the per-step learning rate, g the output gate).",
    chain_rule:
      "Forward delta rule: S_t = (I − β_t·k_t·k_tᵀ)·S_{t-1} + " +
      "β_t·v_t·k_tᵀ ; y_t = g_t ⊙ (S_t·q_t). Given ḡ_t = dL/dy_t: " +
      "dL/dg_t = ḡ_t ⊙ (S_t·q_t) and dL/d(S_t·q_t) = ḡ_t ⊙ g_t. " +
      "Matrix adjoint Ŝ_t = q_t · dL/d(S_t·q_t)ᵀ + (I − " +
      "β_{t+1}·k_{t+1}·k_{t+1}ᵀ)ᵀ · Ŝ_{t+1} (reverse-time, " +
      "mirroring the gated identity update). Then dL/dv_t = " +
      "Ŝ_t·k_t·β_t, dL/dk_t combines both rank-1 update terms, " +
      "dL/dβ_t = ⟨Ŝ_t, (v_t − S_{t-1}·k_t)·k_tᵀ⟩. Route q,k,v,β,g " +
      "through their projections to dL/dW_*. MLX: " +
      "`mx.grad(gdn_fn, argnums=[1,2,3,4,5])(x, W_q, W_k, W_v, " +
      "W_β, W_g)`.",
    key_identity:
      "Ŝ_t = (I − β_{t+1}·k_{t+1}·k_{t+1}ᵀ)ᵀ · Ŝ_{t+1} + " +
      "q_t · ∂L/∂(S_t·q_t)ᵀ",
  },

  adapter_rmsnorm: {
    differentiates:
      "L w.r.t. dL/dx and the learnable scale dL/dγ. ε is constant; " +
      "RMSNorm has no bias.",
    chain_rule:
      "Forward: rms = √(mean(x²) + ε); y_i = (x_i/rms)·γ_i. Scale " +
      "grad is trivial: dL/dγ_i = ḡ_i · (x_i/rms), summed over " +
      "batch/seq. For dL/dx_i we need the Jacobian of x_i/rms — " +
      "both the direct term and the dependence of rms on every x_j. " +
      "Working it out: dL/dx_i = (γ_i·ḡ_i)/rms − x_i · (Σ_j " +
      "γ_j·ḡ_j·x_j) / (d · rms³). Vectorized with u = γ ⊙ ḡ: " +
      "dL/dx = u/rms − x · ⟨u, x⟩ / (d · rms³). The famous " +
      "'norm cancels out' part is that after substituting y = " +
      "γ·x/rms back, the second term becomes a clean per-feature " +
      "subtraction — what makes ‖y‖_RMS = 1 stable under SGD.",
    key_identity:
      "dL/dx = (γ ⊙ ḡ)/rms − x · ⟨γ ⊙ ḡ, x⟩ / (d · rms³)",
  },

  brick_abs_pos_embed: {
    differentiates:
      "L w.r.t. dL/dx (identity passthrough) and dL/dW_pos (the " +
      "learned position table, shape [max_len, d]).",
    chain_rule:
      "Forward: y_{t,d} = x_{t,d} + W_pos[t, d] (lookup by position " +
      "index t). Given ḡ = dL/dy: dL/dx = ḡ (identity). " +
      "dL/dW_pos[t] = Σ_{b} ḡ[b, t] (scatter-add by position — " +
      "every batch element reads the same row t). If positions " +
      "repeat, scatter-add accumulates duplicates. No Jacobian " +
      "computation, just routing.",
    key_identity:
      "dL/dx = ḡ;  dL/dW_pos[t] = Σ_b ḡ[b, t, :]  " +
      "(scatter-add by position)",
  },

  adapter_residual: {
    differentiates:
      "L w.r.t. dL/dx; residual adds no parameters.",
    chain_rule:
      "Forward: y = x + F(x). Jacobian = I + ∂F/∂x. Given ḡ = " +
      "dL/dy, the VJP splits and sums: dL/dx_skip = ḡ (identity " +
      "branch) and dL/dx_F = VJP_F(ḡ) (recurse into the sublayer). " +
      "Total: dL/dx = ḡ + VJP_F(ḡ). This additive split is exactly " +
      "why residuals fight vanishing gradients — the identity path " +
      "guarantees ḡ reaches earlier layers undiminished even when " +
      "∂F/∂x is small or ill-conditioned.",
    key_identity:
      "dL/dx = ḡ + ∂F(x)/∂xᵀ · ḡ  " +
      "(gradient flows through both branches additively)",
  },

  metric_perplexity: {
    differentiates: "L = mean cross-entropy over the batch w.r.t. logits z.",
    chain_rule:
      "softmax + log + NLL collapses into the famous identity " +
      "dL/dz = (p − y)/B where p = softmax(z) and y is the one-hot " +
      "target. This is the single most-used backward formula in " +
      "language modelling. The softmax Jacobian " +
      "J_ij = p_i·(δ_ij − p_j) combined with d/dp(−log p_y) = " +
      "−1/p_y telescopes into a clean (p − y).",
    key_identity: "softmax+CE-VJP:  dL/dz = (p − y) / B",
  },
};
