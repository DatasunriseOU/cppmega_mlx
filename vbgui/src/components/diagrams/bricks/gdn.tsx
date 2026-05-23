/**
 * Gated DeltaNet:
 *   x → q,k,v; β=σ(W_β x); α=σ(W_α x)
 *   delta rule: S_t = α·S_{t-1} + β·(v - S_{t-1}·k)·kᵀ
 *   o = S_t · q  → linear_out → y
 * Ref: https://arxiv.org/abs/2412.06464
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function GDNDiagram(): JSX.Element {
  return (
    <Diagram width={560} height={240}
             caption="Gated DeltaNet: S_t = α·S_{t−1} + β·(v − S·k)·kᵀ → readout S·q">
      <Tensor x={10}  y={100} label="x" shape="(B,S,H)" />
      <Tensor x={120} y={20}  label="q" w={40} />
      <Tensor x={120} y={70}  label="k" w={40} />
      <Tensor x={120} y={120} label="v" w={40} />
      <Tensor x={120} y={170} label="β,α" w={50} shape="σ" />
      <Op     x={230} y={70}  label="v - S·k" w={80} />
      <Op     x={230} y={140} label="S_t = α·S + β·δkᵀ" w={170} />
      <Op     x={430} y={70}  label="readout S·q" w={110} />
      <Tensor x={460} y={140} label="y" shape="(B,S,H)" w={60} />
      <Arrow x1={100} y1={122} x2={120} y2={42}  />
      <Arrow x1={100} y1={122} x2={120} y2={92}  />
      <Arrow x1={100} y1={122} x2={120} y2={142} />
      <Arrow x1={100} y1={122} x2={120} y2={192} />
      <Arrow x1={160} y1={92}  x2={230} y2={92}  />
      <Arrow x1={160} y1={142} x2={230} y2={92}  />
      <Arrow x1={170} y1={192} x2={290} y2={140} label="α,β" />
      <Arrow x1={310} y1={92}  x2={315} y2={140} label="δ" />
      <Arrow x1={400} y1={140} x2={430} y2={92}  label="S_t" />
      <Arrow x1={160} y1={42}  x2={430} y2={92}  label="q" />
      <Arrow x1={540} y1={92}  x2={460} y2={162} label="W_out" />
      <Residual x1={10} y1={150} x2={460} y2={170} bendY={0.05} label="+x" />
    </Diagram>
  );
}
