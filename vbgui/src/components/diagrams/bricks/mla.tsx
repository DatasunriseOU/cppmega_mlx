/**
 * DeepSeek-V3 MLA: LoRA-down on Q and KV → split nope/pe → RoPE on pe-split
 *   → matmul scores → softmax → attend V → Wo → y
 * Inference 'absorb' fast-path folds W_UK into W_Q.
 * Ref: https://arxiv.org/abs/2412.19437 fig.2
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function MLADiagram(): JSX.Element {
  return (
    <Diagram width={560} height={280}
             caption="MLA: LoRA-Q + LoRA-KV → split nope/pe → RoPE on pe → softmax(QKᵀ)·V">
      <Tensor x={10}  y={120} label="x" shape="(B,S,H)" />
      <Op     x={110} y={45}  label="LoRA-Q ↓↑" w={90} />
      <Op     x={110} y={195} label="LoRA-KV ↓↑" w={100} />
      <Tensor x={230} y={20}  label="q_nope" w={70} shape="hd_nope" />
      <Tensor x={230} y={70}  label="q_pe"   w={70} shape="hd_rope" />
      <Tensor x={230} y={170} label="k_nope" w={70} shape="hd_nope" />
      <Tensor x={230} y={210} label="k_pe"   w={70} shape="hd_rope" />
      <Tensor x={230} y={250} label="v"      w={70} shape="hd_v" />
      <Op     x={330} y={70}  label="RoPE" w={50} />
      <Op     x={330} y={210} label="RoPE" w={50} />
      <Op     x={400} y={120} label="softmax(QKᵀ)·V" w={120} />
      <Tensor x={460} y={250} label="y" shape="(B,S,H)" w={60} />
      <Arrow x1={100} y1={142} x2={110} y2={67}  />
      <Arrow x1={100} y1={142} x2={110} y2={217} />
      <Arrow x1={200} y1={67}  x2={230} y2={42}  />
      <Arrow x1={200} y1={67}  x2={230} y2={92}  />
      <Arrow x1={210} y1={217} x2={230} y2={192} />
      <Arrow x1={210} y1={217} x2={230} y2={232} />
      <Arrow x1={210} y1={217} x2={230} y2={272} />
      <Arrow x1={300} y1={92}  x2={330} y2={92}  />
      <Arrow x1={300} y1={232} x2={330} y2={232} />
      <Arrow x1={380} y1={92}  x2={400} y2={142} />
      <Arrow x1={380} y1={232} x2={400} y2={142} />
      <Arrow x1={300} y1={42}  x2={400} y2={130} />
      <Arrow x1={300} y1={192} x2={400} y2={140} />
      <Arrow x1={300} y1={272} x2={400} y2={150} />
      <Arrow x1={520} y1={142} x2={460} y2={272} label="Wo" />
      <Residual x1={10} y1={170} x2={460} y2={270} bendY={0.05} label="+x" />
    </Diagram>
  );
}
