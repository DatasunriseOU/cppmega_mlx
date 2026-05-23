/**
 * Concrete numerical worked examples per brick / math foundation.
 *
 * Sourced from a research agent that produced Raschka-style small
 * tensors (seq=4, d_k=4, h=2). Each example carries (a) a caption,
 * (b) the named tensors with role + shape + values, and (c) the
 * sequence of steps that connect them. The renderer
 * (WorkedExampleDiagram) lays out the tensors in step order using
 * the new MatrixGrid primitive.
 */

import type { CellRole } from "./TensorDiagram";


export interface WorkedTensor {
  name: string;
  role: CellRole;
  /** 1-D or 2-D or 3-D array. 3-D is treated as a list of 2-D slices
   *  rendered side by side (one per head). */
  values: number[] | number[][] | number[][][];
  shape: number[];
}

export interface WorkedStep {
  label: string;
  from: string[];
  to: string;
}

export interface WorkedExample {
  caption: string;
  tensors: WorkedTensor[];
  steps: WorkedStep[];
  output: string;
}


export const WORKED_EXAMPLES: Record<string, WorkedExample> = {
  attention: {
    caption: "Vanilla scaled dot-product attention (seq=4, d_k=d_v=4)",
    tensors: [
      { name: "Q", role: "q", shape: [4, 4], values: [
        [0.42, -0.31, 0.18, 0.55],
        [-0.22, 0.61, -0.14, 0.09],
        [0.33, 0.07, -0.48, 0.21],
        [-0.51, 0.24, 0.36, -0.17],
      ] },
      { name: "K", role: "k", shape: [4, 4], values: [
        [0.29, 0.44, -0.12, 0.38],
        [-0.41, 0.15, 0.52, -0.23],
        [0.17, -0.36, 0.28, 0.49],
        [0.45, 0.11, -0.27, -0.34],
      ] },
      { name: "V", role: "v", shape: [4, 4], values: [
        [0.21, -0.47, 0.33, 0.12],
        [0.58, 0.19, -0.24, 0.41],
        [-0.15, 0.36, 0.47, -0.29],
        [0.32, -0.18, 0.05, 0.51],
      ] },
      { name: "scores", role: "attn", shape: [4, 4], values: [
        [0.07, -0.10, 0.21, 0.01],
        [0.10, 0.03, -0.17, -0.05],
        [0.13, -0.30, 0.08, 0.09],
        [-0.08, 0.30, -0.05, -0.13],
      ] },
      { name: "probs", role: "attn", shape: [4, 4], values: [
        [0.25, 0.21, 0.29, 0.24],
        [0.27, 0.25, 0.21, 0.23],
        [0.29, 0.19, 0.28, 0.28],
        [0.23, 0.34, 0.24, 0.22],
      ] },
      { name: "y", role: "out", shape: [4, 4], values: [
        [0.22, 0.00, 0.18, 0.18],
        [0.24, -0.02, 0.16, 0.18],
        [0.21, -0.04, 0.19, 0.18],
        [0.31, 0.00, 0.05, 0.27],
      ] },
    ],
    steps: [
      { label: "Q·Kᵀ / √d_k", from: ["Q", "K"], to: "scores" },
      { label: "softmax(row)", from: ["scores"], to: "probs" },
      { label: "probs · V", from: ["probs", "V"], to: "y" },
    ],
    output: "y",
  },
  rmsnorm: {
    caption: "RMSNorm of an 8-dim vector (γ=1, ε=1e-6) → ‖y‖_RMS = 1",
    tensors: [
      { name: "x", role: "q", shape: [8],
        values: [0.42, -0.31, 0.18, 0.55, -0.27, 0.09, 0.36, -0.48] },
      { name: "x²", role: "attn", shape: [8],
        values: [0.18, 0.10, 0.03, 0.30, 0.07, 0.01, 0.13, 0.23] },
      { name: "mean", role: "attn", shape: [1], values: [0.13] },
      { name: "rms", role: "attn", shape: [1], values: [0.36] },
      { name: "γ", role: "k", shape: [8],
        values: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] },
      { name: "y", role: "out", shape: [8],
        values: [1.17, -0.86, 0.50, 1.53, -0.75, 0.25, 1.00, -1.33] },
      { name: "‖y‖_RMS", role: "out", shape: [1], values: [1.00] },
    ],
    steps: [
      { label: "xᵢ²", from: ["x"], to: "x²" },
      { label: "mean(x²)", from: ["x²"], to: "mean" },
      { label: "√(mean+ε)", from: ["mean"], to: "rms" },
      { label: "γ · x / rms", from: ["x", "rms", "γ"], to: "y" },
      { label: "RMS(y) = 1", from: ["y"], to: "‖y‖_RMS" },
    ],
    output: "y",
  },
  softmax: {
    caption: "Softmax: logits → probabilities (Σ = 1)",
    tensors: [
      { name: "logits", role: "q", shape: [4],
        values: [2.00, 1.00, 0.10, -0.50] },
      { name: "exp", role: "attn", shape: [4],
        values: [7.39, 2.72, 1.11, 0.61] },
      { name: "sum", role: "attn", shape: [1], values: [11.83] },
      { name: "probs", role: "out", shape: [4],
        values: [0.62, 0.23, 0.09, 0.05] },
      { name: "Σ probs", role: "out", shape: [1], values: [1.00] },
    ],
    steps: [
      { label: "exp(aᵢ)", from: ["logits"], to: "exp" },
      { label: "Σ exp", from: ["exp"], to: "sum" },
      { label: "exp / Σ", from: ["exp", "sum"], to: "probs" },
      { label: "Σ probs = 1", from: ["probs"], to: "Σ probs" },
    ],
    output: "probs",
  },
  dot_product: {
    caption: "Dot product: [1, 2, 3] · [-1, 0, 4] = 11",
    tensors: [
      { name: "a", role: "q", shape: [3], values: [1.00, 2.00, 3.00] },
      { name: "b", role: "k", shape: [3], values: [-1.00, 0.00, 4.00] },
      { name: "aᵢ·bᵢ", role: "attn", shape: [3],
        values: [-1.00, 0.00, 12.00] },
      { name: "y", role: "out", shape: [1], values: [11.00] },
    ],
    steps: [
      { label: "elementwise", from: ["a", "b"], to: "aᵢ·bᵢ" },
      { label: "Σ", from: ["aᵢ·bᵢ"], to: "y" },
    ],
    output: "y",
  },
  mlp: {
    caption: "SwiGLU MLP (d=8, d_ff=16) — gate ⊙ up → down → y",
    tensors: [
      { name: "x", role: "q", shape: [4, 8], values: [
        [0.31, -0.22, 0.48, 0.11, -0.37, 0.19, 0.05, -0.41],
        [-0.14, 0.39, -0.27, 0.52, 0.18, -0.33, 0.46, 0.08],
        [0.45, 0.18, -0.33, -0.09, 0.27, 0.51, -0.22, 0.14],
        [-0.28, 0.07, 0.41, -0.36, -0.15, 0.24, 0.38, -0.19],
      ] },
      { name: "y", role: "out", shape: [4, 8], values: [
        [0.11, -0.07, 0.18, 0.04, -0.14, 0.09, 0.02, -0.16],
        [-0.05, 0.14, -0.11, 0.19, 0.07, -0.13, 0.16, 0.03],
        [0.17, 0.06, -0.13, -0.03, 0.11, 0.18, -0.08, 0.05],
        [-0.12, 0.03, 0.16, -0.14, -0.06, 0.10, 0.15, -0.07],
      ] },
    ],
    steps: [
      { label: "x · W_gate then SiLU", from: ["x"], to: "y" },
      { label: "x · W_up", from: ["x"], to: "y" },
      { label: "silu(gate) ⊙ up → W_down", from: ["x"], to: "y" },
    ],
    output: "y",
  },
  gated_attention: {
    caption: "Gated attention: σ(G) ⊙ ctx then W_o (seq=4, d_v=4)",
    tensors: [
      { name: "Q", role: "q", shape: [4, 4], values: [
        [0.31, -0.22, 0.48, 0.11],
        [-0.14, 0.39, -0.27, 0.52],
        [0.45, 0.18, -0.33, -0.09],
        [-0.28, 0.07, 0.41, -0.36],
      ] },
      { name: "K", role: "k", shape: [4, 4], values: [
        [0.22, 0.41, -0.18, 0.33],
        [-0.37, 0.12, 0.46, -0.19],
        [0.15, -0.29, 0.31, 0.44],
        [0.39, 0.08, -0.24, -0.31],
      ] },
      { name: "V", role: "v", shape: [4, 4], values: [
        [0.18, -0.42, 0.29, 0.10],
        [0.51, 0.16, -0.21, 0.36],
        [-0.13, 0.31, 0.41, -0.25],
        [0.28, -0.16, 0.04, 0.45],
      ] },
      { name: "ctx", role: "attn", shape: [4, 4], values: [
        [0.20, -0.02, 0.14, 0.16],
        [0.22, 0.00, 0.10, 0.18],
        [0.18, -0.07, 0.17, 0.15],
        [0.28, 0.02, 0.02, 0.25],
      ] },
      { name: "σ(G)", role: "gate", shape: [4, 4], values: [
        [0.70, 0.23, 0.60, 0.82],
        [0.35, 0.79, 0.46, 0.68],
        [0.81, 0.40, 0.71, 0.25],
        [0.43, 0.66, 0.20, 0.77],
      ] },
      { name: "y", role: "out", shape: [4, 4], values: [
        [0.14, 0.00, 0.08, 0.13],
        [0.08, 0.00, 0.05, 0.12],
        [0.15, -0.03, 0.12, 0.04],
        [0.12, 0.01, 0.00, 0.19],
      ] },
    ],
    steps: [
      { label: "softmax(QKᵀ/√d_k)·V", from: ["Q", "K", "V"], to: "ctx" },
      { label: "σ(G) ⊙ ctx", from: ["ctx", "σ(G)"], to: "y" },
    ],
    output: "y",
  },
  rope: {
    caption: "RoPE: rotate (q_x, q_y) pair by θ — preserves length, " +
              "encodes position relatively",
    tensors: [
      { name: "q_pre", role: "q", shape: [4, 2], values: [
        [0.50, 0.87],   // unit vector at 60°
        [-0.71, 0.71],
        [0.95, 0.31],
        [-0.50, -0.87],
      ] },
      { name: "θ", role: "raw", shape: [4],
        values: [0.00, 0.79, 1.57, 2.36] },
      { name: "q_post", role: "q", shape: [4, 2], values: [
        [0.50, 0.87],
        [-1.00, 0.00],
        [-0.31, 0.95],
        [0.16, 0.99],
      ] },
    ],
    steps: [
      { label: "R(θ) · q", from: ["q_pre", "θ"], to: "q_post" },
    ],
    output: "q_post",
  },
  residual: {
    caption: "Residual add: y = x + F(x). Gradient splits both ways " +
              "in backward",
    tensors: [
      { name: "x", role: "q", shape: [8],
        values: [0.42, -0.31, 0.18, 0.55, -0.27, 0.09, 0.36, -0.48] },
      { name: "F(x)", role: "hidden", shape: [8],
        values: [0.11, 0.05, -0.13, -0.08, 0.22, -0.06, 0.04, 0.17] },
      { name: "y", role: "out", shape: [8],
        values: [0.53, -0.26, 0.05, 0.47, -0.05, 0.03, 0.40, -0.31] },
    ],
    steps: [
      { label: "y = x + F(x)", from: ["x", "F(x)"], to: "y" },
    ],
    output: "y",
  },
  matmul: {
    caption: "Matrix multiplication: C = A · B (2×3) · (3×2) = (2×2)",
    tensors: [
      { name: "A", role: "q", shape: [2, 3], values: [
        [1.00, 2.00, 3.00],
        [-1.00, 0.50, 4.00],
      ] },
      { name: "B", role: "k", shape: [3, 2], values: [
        [2.00, -1.00],
        [0.50, 3.00],
        [-1.00, 1.00],
      ] },
      { name: "C", role: "out", shape: [2, 2], values: [
        [0.00, 8.00],
        [-5.75, 6.50],
      ] },
    ],
    steps: [
      { label: "row · col", from: ["A", "B"], to: "C" },
    ],
    output: "C",
  },
  moe: {
    caption: "MoE router: 4 experts, top-k=2",
    tensors: [
      { name: "router_logits", role: "k", shape: [4, 4], values: [
        [1.20, -0.35, 0.78, 0.12],
        [-0.41, 1.55, 0.22, -0.18],
        [0.66, 0.31, -0.87, 1.08],
        [0.14, -1.12, 0.95, 0.48],
      ] },
      { name: "router_probs", role: "attn", shape: [4, 4], values: [
        [0.51, 0.11, 0.34, 0.18],
        [0.10, 0.71, 0.19, 0.13],
        [0.30, 0.21, 0.06, 0.45],
        [0.24, 0.07, 0.54, 0.34],
      ] },
      { name: "top_k_idx", role: "attn", shape: [4, 2], values: [
        [0, 2],
        [1, 2],
        [3, 0],
        [2, 3],
      ] },
      { name: "top_k_weights", role: "attn", shape: [4, 2], values: [
        [0.60, 0.40],
        [0.79, 0.21],
        [0.60, 0.40],
        [0.61, 0.39],
      ] },
      { name: "y", role: "out", shape: [4, 8], values: [
        [0.18, -0.11, 0.27, 0.07, -0.21, 0.13, 0.04, -0.24],
        [-0.09, 0.22, -0.16, 0.30, 0.11, -0.19, 0.26, 0.05],
        [0.26, 0.10, -0.19, -0.05, 0.16, 0.29, -0.13, 0.08],
        [-0.16, 0.04, 0.24, -0.21, -0.09, 0.14, 0.22, -0.11],
      ] },
    ],
    steps: [
      { label: "softmax(router_logits)", from: ["router_logits"],
        to: "router_probs" },
      { label: "top-k=2", from: ["router_probs"], to: "top_k_idx" },
      { label: "Σ wᵢ · Eᵢ(x)", from: ["top_k_idx", "top_k_weights"],
        to: "y" },
    ],
    output: "y",
  },
};
