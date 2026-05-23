/**
 * Selective SSM (Mamba-3):
 *   x → linear_in → split(u,z) → conv1d(u) → silu → project(Δ,B,C)
 *   → selective_scan(A,B,C,Δ) → silu(z) → mul → linear_out → y
 * Ref: https://arxiv.org/abs/2312.00752 fig.3
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function Mamba3Diagram(): JSX.Element {
  return (
    <Diagram width={560} height={240}
             caption="Mamba-3 selective SSM: conv1d → Δ,B,C projections → scan → gate ⊙ → out">
      <Tensor x={10}  y={100} label="x" shape="(B,S,H)" />
      <Op     x={100} y={100} label="linear_in" w={70} />
      <Tensor x={190} y={50}  label="u" w={50} shape="(B,S,D)" />
      <Tensor x={190} y={150} label="z" w={50} shape="(B,S,D)" />
      <Op     x={260} y={50}  label="conv1d→silu" w={100} />
      <Op     x={260} y={130} label="silu" w={50} />
      <Tensor x={380} y={20}  label="Δ" w={40} shape="(B,S,D)" />
      <Tensor x={380} y={70}  label="B" w={40} shape="(B,S,N)" />
      <Tensor x={380} y={120} label="C" w={40} shape="(B,S,N)" />
      <Op     x={440} y={50}  label="scan(A,B,C,Δ)" w={110} />
      <Op     x={440} y={130} label="⊙ gate" w={70} />
      <Tensor x={460} y={185} label="y" shape="(B,S,H)" w={60} />
      <Arrow x1={80}  y1={122} x2={100} y2={122} />
      <Arrow x1={170} y1={122} x2={190} y2={72}  />
      <Arrow x1={170} y1={122} x2={190} y2={172} />
      <Arrow x1={240} y1={72}  x2={260} y2={72}  />
      <Arrow x1={240} y1={172} x2={260} y2={152} />
      <Arrow x1={360} y1={72}  x2={380} y2={42}  />
      <Arrow x1={360} y1={72}  x2={380} y2={92}  />
      <Arrow x1={360} y1={72}  x2={380} y2={142} />
      <Arrow x1={420} y1={42}  x2={440} y2={72}  />
      <Arrow x1={420} y1={92}  x2={440} y2={72}  />
      <Arrow x1={420} y1={142} x2={440} y2={72}  />
      <Arrow x1={550} y1={72}  x2={510} y2={152} label="⊗" />
      <Arrow x1={310} y1={152} x2={440} y2={152} />
      <Arrow x1={510} y1={152} x2={460} y2={207} label="W_out" />
      <Residual x1={10} y1={170} x2={460} y2={207} bendY={0.04} label="+x" />
    </Diagram>
  );
}
