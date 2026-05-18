"""KDA Path B forward + backward — same fwd-fast / bwd-correct pattern as GDN."""

import mlx.core as mx

from cppmega_v4._tilelang.kda_path_b import kda_forward_path_b
from cppmega_v4.nn._external.fla_naive_kda import naive_recurrent_kda


@mx.custom_function
def kda_apply_path_b(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array,
) -> mx.array:
    """Forward via fast Path B Metal kernel; backward via Path A reference grad."""
    y, _ = kda_forward_path_b(q, k, v, g, beta, output_final_state=False)
    return y


@kda_apply_path_b.vjp
def _kda_apply_path_b_vjp(primals, cotangent, output):
    del output
    q, k, v, g, beta = primals

    def _loss_proxy(q_, k_, v_, g_, beta_):
        y, _ = naive_recurrent_kda(q_, k_, v_, g_, beta_)
        return (y * cotangent).sum()

    return mx.grad(_loss_proxy, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)


__all__ = ["kda_apply_path_b"]
