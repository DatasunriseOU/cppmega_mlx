/**
 * Mixture-of-Experts MLP:
 *   x → router (top-k of N experts)
 *   → each token activates k experts → weighted sum → y
 *   + auxiliary load-balance loss
 */

import { Diagram, Tensor, Op, Arrow } from "../TensorDiagram";

export function MoEDiagram(): JSX.Element {
  return (
    <Diagram width={520} height={240}
             caption="MoE: router → top-k experts (k=2 of N=8) → weighted sum → y">
      <Tensor x={10}  y={100} label="x" shape="(B,S,H)" />
      <Op     x={130} y={100} label="router" w={70} />
      <Tensor x={240} y={30}  label="E_1" shape="MLP" w={50} />
      <Tensor x={240} y={80}  label="E_2" shape="MLP" w={50} />
      <Tensor x={240} y={130} label="…"   shape=""    w={50} />
      <Tensor x={240} y={180} label="E_N" shape="MLP" w={50} />
      <Op     x={340} y={100} label="Σ top-k" w={80} />
      <Tensor x={450} y={100} label="y" shape="(B,S,H)" w={60} />
      <Arrow x1={100} y1={122} x2={130} y2={115} />
      <Arrow x1={200} y1={108} x2={240} y2={52}  label="α_1" />
      <Arrow x1={200} y1={115} x2={240} y2={102} label="α_2" />
      <Arrow x1={200} y1={122} x2={240} y2={152} />
      <Arrow x1={200} y1={130} x2={240} y2={202} />
      <Arrow x1={290} y1={52}  x2={340} y2={108} />
      <Arrow x1={290} y1={102} x2={340} y2={115} />
      <Arrow x1={290} y1={202} x2={340} y2={122} />
      <Arrow x1={420} y1={115} x2={450} y2={122} />
    </Diagram>
  );
}
