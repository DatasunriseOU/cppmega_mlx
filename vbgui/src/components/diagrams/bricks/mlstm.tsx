/**
 * xLSTM matrix-memory mLSTM:
 *   x → q,k,v + gates i,f,o
 *   C_t = f·C_{t-1} + i·(v kᵀ);  n_t = f·n_{t-1} + i·k
 *   h_t = o · (C_t q) / max(|n_tᵀq|, 1)
 * Ref: https://arxiv.org/abs/2405.04517
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function MLSTMDiagram(): JSX.Element {
  return (
    <Diagram width={560} height={260}
             caption="mLSTM: matrix state C_t = f·C + i·(v kᵀ); h = o · (C·q) / max(|nᵀq|, 1)">
      <Tensor x={10}  y={120} label="x" shape="(B,S,H)" />
      <Tensor x={130} y={20}  label="q" w={40} />
      <Tensor x={130} y={70}  label="k" w={40} />
      <Tensor x={130} y={120} label="v" w={40} />
      <Tensor x={130} y={170} label="i,f,o" w={60} shape="gates" />
      <Op     x={230} y={70}  label="outer v·kᵀ" w={90} />
      <Op     x={230} y={140} label="C_t = f·C + i·(vkᵀ)" w={170} />
      <Op     x={420} y={140} label="readout C_t·q" w={110} />
      <Op     x={420} y={200} label="o ⊙ (C·q)/n̂" w={110} />
      <Tensor x={460} y={20}  label="h" shape="(B,S,d)" w={70} />
      <Arrow x1={100} y1={142} x2={130} y2={42}  />
      <Arrow x1={100} y1={142} x2={130} y2={92}  />
      <Arrow x1={100} y1={142} x2={130} y2={142} />
      <Arrow x1={100} y1={142} x2={130} y2={192} />
      <Arrow x1={170} y1={92}  x2={230} y2={92}  />
      <Arrow x1={170} y1={142} x2={230} y2={92}  />
      <Arrow x1={320} y1={92}  x2={315} y2={140} />
      <Arrow x1={190} y1={192} x2={315} y2={140} label="f,i" />
      <Arrow x1={400} y1={162} x2={420} y2={162} />
      <Arrow x1={170} y1={42}  x2={475} y2={162} label="q" />
      <Arrow x1={530} y1={162} x2={475} y2={222} />
      <Arrow x1={190} y1={192} x2={420} y2={222} label="o" />
      <Arrow x1={530} y1={222} x2={460} y2={42}  label="W_out" bendY={0.95} />
      <Residual x1={10} y1={170} x2={460} y2={42}  bendY={0.05} label="+x" />
    </Diagram>
  );
}
