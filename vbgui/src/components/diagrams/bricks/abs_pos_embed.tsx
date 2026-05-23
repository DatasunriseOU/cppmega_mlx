/**
 * abs_pos_embed: y = x + W_pos[:S]
 * Ref: BERT / GPT — learned absolute positions.
 */

import { Diagram, Tensor, Op, Arrow } from "../TensorDiagram";

export function AbsPosEmbedDiagram(): JSX.Element {
  return (
    <Diagram width={460} height={150}
             caption="abs_pos_embed: y = x + W_pos[:S]">
      <Tensor x={10}  y={50}  label="x" shape="(B,S,H)" />
      <Tensor x={10}  y={100} label="W_pos" shape="(L_max,H)" />
      <Op     x={180} y={75}  label="slice + add" w={120} />
      <Tensor x={350} y={75}  label="y" shape="(B,S,H)" />
      <Arrow x1={100} y1={72}  x2={180} y2={92}  />
      <Arrow x1={100} y1={122} x2={180} y2={102} label="[:S]" />
      <Arrow x1={300} y1={97}  x2={350} y2={97}  />
    </Diagram>
  );
}
