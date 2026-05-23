/**
 * Vanilla SDPA attention (Llama-style):
 *   x:(B,S,H) → {Q, K, V} = {Wq, Wk, Wv}(x)
 *   → scores = Q · Kᵀ / √hd  → softmax  → · V
 *   → output proj Wo → y:(B,S,H)
 *   plus residual x + y
 */

import { Diagram, Tensor, Op, Arrow, Residual } from "../TensorDiagram";

export function AttentionDiagram(): JSX.Element {
  return (
    <Diagram width={520} height={240}
             caption="vanilla SDPA: x → {Q,K,V} → softmax(QKᵀ/√hd)·V → Wo → y + x">
      <Tensor x={10}  y={100} label="x" shape="(B,S,H)" testid="t-x" />
      <Tensor x={140} y={20}  label="Q" shape="(B,nh,S,hd)" />
      <Tensor x={140} y={100} label="K" shape="(B,nh,S,hd)" />
      <Tensor x={140} y={180} label="V" shape="(B,nh,S,hd)" />
      <Op     x={290} y={50}  label="softmax(QKᵀ/√hd)" w={140} />
      <Tensor x={290} y={130} label="attn·V" shape="(B,nh,S,hd)" w={140} />
      <Tensor x={460} y={100} label="y" shape="(B,S,H)" w={50}
              testid="t-y" />
      <Arrow x1={100} y1={122} x2={140} y2={42}  label="Wq" />
      <Arrow x1={100} y1={122} x2={140} y2={122} label="Wk" />
      <Arrow x1={100} y1={122} x2={140} y2={202} label="Wv" />
      <Arrow x1={230} y1={42}  x2={290} y2={65}  />
      <Arrow x1={230} y1={122} x2={290} y2={65}  />
      <Arrow x1={230} y1={202} x2={290} y2={152} />
      <Arrow x1={360} y1={80}  x2={360} y2={130} />
      <Arrow x1={430} y1={152} x2={460} y2={122} label="Wo" />
      <Residual x1={10} y1={140} x2={460} y2={140} bendY={0.05}
                 label="+ x (residual)" />
    </Diagram>
  );
}
