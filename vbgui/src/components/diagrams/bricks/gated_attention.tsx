/**
 * Qwen3-Next gated attention:
 *   x → {Q,K,V,G}; RMSNorm(Q), RMSNorm(K); partial RoPE on first hd_rot dims
 *   scores=QKᵀ/√hd → softmax → ctx → ctx ⊙ sigmoid(G) → Wo → y
 *
 * Ref: https://sebastianraschka.com/llm-architecture-gallery/gated-attention/
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function GatedAttentionDiagram(): JSX.Element {
  return (
    <Diagram width={560} height={260}
             caption="gated_attention: Q/K RMSNorm + partial RoPE + sigmoid(G) gate · ctx">
      <Tensor x={10}  y={110} label="x" shape="(B,S,H)" />
      <Tensor x={120} y={20}  label="Q+RMS" w={70} />
      <Tensor x={120} y={70}  label="K+RMS" w={70} />
      <Tensor x={120} y={120} label="V" w={70} />
      <Tensor x={120} y={170} label="G" w={70} />
      <Op     x={220} y={20}  label="partial RoPE" w={90} />
      <Op     x={220} y={70}  label="partial RoPE" w={90} />
      <Op     x={340} y={45}  label="QKᵀ / √hd" w={90} />
      <Op     x={340} y={120} label="softmax · V" w={90} />
      <Op     x={340} y={170} label="σ(G) ⊙" w={70} />
      <Tensor x={460} y={120} label="ctx·gate" w={80} />
      <Tensor x={460} y={185} label="y" shape="(B,S,H)" w={60} />
      <Arrow x1={100} y1={132} x2={120} y2={42}  label="Wq" />
      <Arrow x1={100} y1={132} x2={120} y2={92}  label="Wk" />
      <Arrow x1={100} y1={132} x2={120} y2={142} label="Wv" />
      <Arrow x1={100} y1={132} x2={120} y2={192} label="Wg" />
      <Arrow x1={190} y1={42}  x2={220} y2={42}  />
      <Arrow x1={190} y1={92}  x2={220} y2={92}  />
      <Arrow x1={310} y1={42}  x2={340} y2={65}  />
      <Arrow x1={310} y1={92}  x2={340} y2={65}  />
      <Arrow x1={190} y1={142} x2={340} y2={140} />
      <Arrow x1={430} y1={70}  x2={460} y2={140} />
      <Arrow x1={430} y1={142} x2={460} y2={142} />
      <Arrow x1={190} y1={192} x2={340} y2={192} />
      <Arrow x1={410} y1={192} x2={460} y2={207} label="Wo" />
      <Residual x1={10} y1={170} x2={460} y2={207} bendY={0.03} label="+x" />
    </Diagram>
  );
}
