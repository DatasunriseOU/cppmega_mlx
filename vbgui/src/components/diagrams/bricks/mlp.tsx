/**
 * SwiGLU MLP:
 *   x → gate = W_gate(x), up = W_up(x)
 *   → silu(gate) * up
 *   → W_down → y    (+ x residual)
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function MLPDiagram(): JSX.Element {
  return (
    <Diagram width={520} height={220}
             caption="SwiGLU MLP: silu(W_gate · x) ⊙ (W_up · x) → W_down → y">
      <Tensor x={10}  y={90}  label="x" shape="(B,S,H)" />
      <Tensor x={140} y={30}  label="gate" shape="(B,S,4H)" />
      <Tensor x={140} y={130} label="up"   shape="(B,S,4H)" />
      <Op     x={290} y={30}  label="silu" w={60} />
      <Op     x={290} y={90}  label="⊙ (mul)" w={80} />
      <Tensor x={400} y={90}  label="hid" shape="(B,S,4H)" w={70} />
      <Tensor x={490} y={90}  label="y"   shape="(B,S,H)" w={50} />
      <Arrow x1={100} y1={112} x2={140} y2={52}  label="W_gate" />
      <Arrow x1={100} y1={112} x2={140} y2={152} label="W_up" />
      <Arrow x1={230} y1={52}  x2={290} y2={45}  />
      <Arrow x1={350} y1={45}  x2={370} y2={100} />
      <Arrow x1={230} y1={152} x2={370} y2={112} />
      <Arrow x1={370} y1={112} x2={400} y2={112} />
      <Arrow x1={470} y1={112} x2={490} y2={112} label="W_down" />
      <Residual x1={10} y1={170} x2={490} y2={130} bendY={0.05}
                 label="+ x" />
    </Diagram>
  );
}
