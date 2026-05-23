/**
 * cppmega-mlx-w2t6: brick-topic → tensor-flow-diagram registry.
 *
 * Keyed off the HelpIcon `topic` string (e.g. "brick_attention"). When
 * a HelpModal renders for a topic with an entry here, the modal slots
 * the diagram between the "What" and "Why" sections.
 */

import type { JSX } from "react";

import { AttentionDiagram } from "./bricks/attention";
import { GatedAttentionDiagram } from "./bricks/gated_attention";
import { MLADiagram } from "./bricks/mla";
import { MLPDiagram } from "./bricks/mlp";
import { MoEDiagram } from "./bricks/moe";
import { Mamba3Diagram } from "./bricks/mamba3";
import { MLSTMDiagram } from "./bricks/mlstm";
import { GDNDiagram } from "./bricks/gdn";
import { RMSNormDiagram } from "./bricks/rmsnorm";
import { AbsPosEmbedDiagram } from "./bricks/abs_pos_embed";


export const TENSOR_DIAGRAMS: Record<string, () => JSX.Element> = {
  // Attention family
  brick_attention:        AttentionDiagram,
  brick_gated_attention:  GatedAttentionDiagram,
  brick_mla:              MLADiagram,
  brick_mla_absorb:       MLADiagram,   // same flow, different decode path
  brick_mistral4_mla:     MLADiagram,   // MLA with INT4 cache
  brick_gqa_sliding:      AttentionDiagram,   // SDPA + window
  brick_cca_attention:    AttentionDiagram,   // SDPA + pooled K/V

  // MLP / FFN
  brick_mlp:              MLPDiagram,

  // MoE
  brick_moe:              MoEDiagram,
  brick_bailing_moe:      MoEDiagram,

  // Linear / SSM / Recurrent
  brick_mamba3:           Mamba3Diagram,
  brick_mlstm:            MLSTMDiagram,
  brick_gdn:              GDNDiagram,
  brick_kda:              GDNDiagram,   // delta-rule kernel close cousin

  // Norm / Embed (adapters surface via the same registry too)
  adapter_rmsnorm:        RMSNormDiagram,
  brick_abs_pos_embed:    AbsPosEmbedDiagram,
};


export { DIAG_THEME, MatrixGrid, MathLink } from "./TensorDiagram";
export { WORKED_EXAMPLES, type WorkedExample } from "./worked_examples";
export { MATH_FOUNDATIONS, TOPIC_FOUNDATIONS,
         type MathConcept } from "./math_foundations";
export { BACKWARD_TOPICS,
         type BackwardEntry } from "./backward_passes";
export { WorkedExampleDiagram } from "./WorkedExampleDiagram";


// Topic → worked-example key. Same fall-back semantics as the
// diagram registry: when both are present in HelpModal, both render
// (schematic flow diagram first, then concrete numerical example).
export const TOPIC_WORKED_EXAMPLES: Record<string, string> = {
  brick_attention:        "attention",
  brick_gated_attention:  "gated_attention",
  brick_mla:              "attention",
  brick_mla_absorb:       "attention",
  brick_mistral4_mla:     "attention",
  brick_gqa_sliding:      "attention",
  brick_cca_attention:    "attention",
  brick_mlp:              "mlp",
  brick_moe:              "moe",
  brick_bailing_moe:      "moe",
  brick_mamba3:           "mlp",     // share matmul-flavour example
  brick_mlstm:            "mlp",
  brick_gdn:              "attention",
  brick_kda:              "attention",
  brick_abs_pos_embed:    "residual",
  adapter_rmsnorm:        "rmsnorm",
  adapter_residual:       "residual",
  adapter_linear_bridge:  "matmul",
  // Math foundations also expose their own worked examples.
  metric_perplexity:      "softmax",
};
