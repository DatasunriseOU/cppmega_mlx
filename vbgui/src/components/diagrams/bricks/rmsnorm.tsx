/**
 * RMSNorm: y = x / sqrt(mean(x²) + ε) * γ
 */

import { Diagram, Tensor, Op, Arrow } from "../TensorDiagram";

export function RMSNormDiagram(): JSX.Element {
  return (
    <Diagram width={500} height={170}
             caption="RMSNorm: y = γ · x / sqrt(mean(x²) + ε)">
      <Tensor x={10}  y={70}  label="x" shape="(B,S,H)" />
      <Op     x={130} y={20}  label="x²" w={50} />
      <Op     x={210} y={20}  label="mean" w={60} />
      <Op     x={300} y={20}  label="+ε then √" w={90} />
      <Tensor x={410} y={20}  label="rms" shape="(B,S,1)" w={70} />
      <Op     x={210} y={90}  label="x / rms" w={90} />
      <Op     x={330} y={90}  label="· γ" w={50} />
      <Tensor x={420} y={70}  label="y" shape="(B,S,H)" w={60} />
      <Arrow x1={100} y1={92}  x2={130} y2={35}  />
      <Arrow x1={180} y1={35}  x2={210} y2={35}  />
      <Arrow x1={270} y1={35}  x2={300} y2={35}  />
      <Arrow x1={390} y1={35}  x2={410} y2={35}  />
      <Arrow x1={100} y1={92}  x2={210} y2={112} />
      <Arrow x1={445} y1={50}  x2={250} y2={92}  label="÷" />
      <Arrow x1={300} y1={112} x2={330} y2={112} />
      <Arrow x1={380} y1={112} x2={420} y2={92}  />
    </Diagram>
  );
}
