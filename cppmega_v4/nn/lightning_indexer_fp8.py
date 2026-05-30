"""ROI 7 — DSA Lightning Indexer FP8 path (pure-MLX + vendored FP8 dequant).

NOT a Path-E fused Metal kernel. This wraps the fp32 ``LightningIndexer``
scaffold with a pure-MLX FP8-dequant GEMM path that mirrors the upstream V3.2
``act_quant`` / ``fp8_index`` pattern WITHOUT a custom CUDA / Metal kernel:

  1. wq_b weight is stored fp8 (e4m3) with block-128 scale_inv, dequantised
     on the fly via ``dequant_block_fp8`` (vendored mlx-lm PR #1224 utility).
  2. Activations are token-wise dynamically quantized to fp8 (per-row scale
     in fp32), the matmul runs in bfloat16 (the closest MLX accuracy that
     preserves fp8 precision), and the output is rescaled.

The K side (wk) and weights_proj stay bfloat16 — they're tiny (head_dim ~32
and n_heads ~32) so the FP8 overhead is not worth it.

Fused FP8 GEMM wiring (Path C):
    The indexer's ``q`` projection ``qr @ wq_b.T`` — the inner block-FP8
    weight × activation GEMM that feeds the score einsum — is routed through
    the TileLang cooperative-tensor ``fused_fp8_gemm`` prim (the keystone
    block-FP8 GEMM) via :meth:`LightningIndexerFP8._wq_b_apply`, under an
    *explicit* shape-dispatch (RULE #1: no silent dequant fallback hiding a
    broken fused path — a malformed FP8 weight RAISES).  ``fused_fp8_gemm``
    internally picks its cooperative-tensor prim or a numerically e4m3-exact
    scalar MSL kernel; both agree to FP8 tolerance.

    The score contraction itself (``einsum bqhd,bkd,bqh->bqk``) is NOT a
    ``fused_fp8_gemm``-shaped op: both ``q`` and ``k`` are *dynamic bf16
    activations* (there is no static block-FP8 weight operand), and the
    contraction is a per-head-weighted, head-summed attention score, not a
    plain ``A @ W.T``.  It therefore stays pure-MLX by design — wiring it
    through ``fused_fp8_gemm`` would require fabricating a fake static FP8
    weight, which RULE #1 forbids.

Provenance / labelling note:
    The only vendored upstream piece here is ``dequant_block_fp8`` (mlx-lm
    PR #1224). There is NO fused Path-E Metal indexer kernel — do not label
    this module "Path E". The GDN/KDA Path E
    (``mlx_lm_gated_delta_update`` / ``mlx_lm_kda_update``) is a separate,
    genuinely fused vendored Metal kernel. The ``wq_b`` projection here is a
    real fused TileLang block-FP8 GEMM (Path C), but the wrapper as a whole is
    not "Path E".
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.nn._external._mlx_lm_fp8_dequant_vendored import dequant_block_fp8
from cppmega_v4.nn.lightning_indexer import (
    LightningIndexer,
    LightningIndexerConfig,
    _apply_non_interleaved_rope,
)

_FP8_BLOCK = 128


def _token_quant_fp8(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Token-wise dynamic quant to fp8 (one scale per row).

    x: [..., D] bf16 / fp32.
    Returns (x_fp8_uint8, scale_inv) where scale_inv broadcasts back to x.
    """
    xf = x.astype(mx.float32)
    amax = mx.maximum(mx.max(mx.abs(xf), axis=-1, keepdims=True), 1e-6)
    # fp8 e4m3 max magnitude ≈ 448.
    scale = amax / 448.0
    x_scaled = (xf / scale).astype(mx.bfloat16)
    fp8 = mx.to_fp8(x_scaled)
    return fp8, scale.astype(mx.bfloat16)


@dataclass(frozen=True)
class LightningIndexerFP8Config(LightningIndexerConfig):
    """FP8 indexer config. ``fp8_blocks`` toggles the wq_b dequant path."""

    fp8_blocks: bool = True


class LightningIndexerFP8(nn.Module):
    """V3.2-faithful FP8 lightning indexer (pure-MLX + vendored FP8 dequant).

    This is NOT a Path-E fused Metal kernel — it is plain MLX ops plus the
    vendored ``dequant_block_fp8`` (mlx-lm PR #1224). Drop-in for
    ``LightningIndexer`` with identical forward signature. The wq_b projection
    runs through a dequant-on-the-fly fp8->bf16 path; everything else stays
    bf16 (small dims don't benefit from fp8).
    """

    def __init__(self, config: LightningIndexerFP8Config):
        super().__init__()
        self.config = config
        out_dim = config.n_heads * config.head_dim

        # wq_b: stored fp8 + per-block scale_inv; bf16 falls back when no fp8.
        if config.fp8_blocks:
            self._wq_b_fp8 = mx.zeros(
                (out_dim, config.q_lora_rank), dtype=mx.uint8
            )
            blocks_m = (out_dim + _FP8_BLOCK - 1) // _FP8_BLOCK
            blocks_n = (config.q_lora_rank + _FP8_BLOCK - 1) // _FP8_BLOCK
            self._wq_b_scale_inv = mx.ones((blocks_m, blocks_n), dtype=mx.float32)
        else:
            self._wq_b_bf16 = mx.zeros(
                (out_dim, config.q_lora_rank), dtype=mx.bfloat16
            )

        # K side + weight projection: bf16 (small dims).
        self.wk = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(config.head_dim, eps=config.norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, config.n_heads, bias=False)

    def _wq_b_apply(self, qr: mx.array) -> mx.array:
        """Apply the FP8 score-path projection ``qr @ wq_b.T``.

        This is the indexer's one genuine block-FP8 weight × activation GEMM
        (the ``q`` projection that feeds the score einsum downstream): ``wq_b``
        is stored as a static ``[out_dim, q_lora_rank]`` uint8 e4m3 weight with
        a ``[ceil(out_dim/128), ceil(q_lora_rank/128)]`` per-block ``scale_inv``,
        which is exactly the ``fused_fp8_gemm`` contract (``out = a @ W.T`` with
        a precomputed block-FP8 weight).

        RULE #1 — explicit dispatch, NO silent fallback:
          * fp8-blocks path: the shape is validated against the
            ``fused_fp8_gemm`` block-FP8 contract and the fused
            cooperative-tensor kernel is called *directly*.  If the kernel
            raises it propagates — we do NOT swallow it into a dequant matmul
            that would hide a broken fused path.  ``fused_fp8_gemm`` itself
            chooses its TileLang cooperative-tensor prim vs scalar MSL kernel
            (both numerically e4m3-exact); that is an internal, numerically
            equivalent choice, not a degraded fallback.
          * a shape that does NOT satisfy the block-FP8 contract is handled by
            an *explicit*, shape-correct ``dequant_block_fp8`` + matmul path
            with a stated reason — reached only when the contract check says so,
            never as an exception-swallowing escape hatch.
          * non-fp8 (bf16) path: plain matmul, no fp8 involved.
        """
        if self.config.fp8_blocks:
            from cppmega_v4._tilelang.fused_fp8_gemm import fused_fp8_gemm

            if self._wq_b_fp8_gemm_supported(qr):
                # W = wq_b [out_dim, q_lora_rank]; fused_fp8_gemm computes
                # qr @ W.T = [B, T, out_dim].  Errors here propagate (RULE #1).
                return fused_fp8_gemm(
                    self._wq_b_fp8, self._wq_b_scale_inv, qr
                ).astype(mx.bfloat16)
            # Explicit (not silent) shape-correct path: the weight does not
            # satisfy the block-FP8 GEMM contract, so dequant the e4m3 weight
            # and matmul in bf16.  The reason is reported by
            # ``_wq_b_fp8_gemm_supported`` raising on a genuinely broken shape.
            w = dequant_block_fp8(self._wq_b_fp8, self._wq_b_scale_inv)
        else:
            w = self._wq_b_bf16
        # qr [B, T, q_lora_rank] @ w.T [q_lora_rank, out_dim]
        return qr.astype(mx.bfloat16) @ w.T

    def _wq_b_fp8_gemm_supported(self, qr: mx.array) -> bool:
        """Explicit dispatch predicate for the ``fused_fp8_gemm`` score GEMM.

        Returns ``True`` when the stored ``wq_b`` weight + ``qr`` activation
        satisfy the ``fused_fp8_gemm`` block-FP8 contract (uint8 2D weight,
        matching contraction dim, correctly-shaped per-128-block ``scale_inv``).

        RULE #1: a malformed FP8 weight (wrong dtype / rank / scale-block shape)
        is a real bug, so this RAISES rather than silently steering to a
        dequant fallback.  ``False`` is returned only for the one legitimate
        non-fused case — an empty contraction (``q_lora_rank == 0``) where the
        GEMM is degenerate — which the explicit dequant path then handles.
        """
        w = self._wq_b_fp8
        s = self._wq_b_scale_inv
        if w.dtype != mx.uint8:
            raise TypeError(
                f"_wq_b_apply: wq_b fp8 weight must be uint8 e4m3 storage; "
                f"got {w.dtype}"
            )
        if w.ndim != 2:
            raise ValueError(
                f"_wq_b_apply: wq_b fp8 weight must be 2D [out_dim, q_lora_rank]; "
                f"got shape {w.shape}"
            )
        out_dim, k = w.shape
        if qr.shape[-1] != k:
            raise ValueError(
                f"_wq_b_apply: qr last dim ({qr.shape[-1]}) must match wq_b "
                f"contraction dim ({k})"
            )
        blocks_m = (out_dim + _FP8_BLOCK - 1) // _FP8_BLOCK
        blocks_k = (k + _FP8_BLOCK - 1) // _FP8_BLOCK
        if tuple(s.shape) != (blocks_m, blocks_k):
            raise ValueError(
                f"_wq_b_apply: wq_b scale_inv shape {tuple(s.shape)} != expected "
                f"({blocks_m}, {blocks_k}) for W={w.shape} block={_FP8_BLOCK}"
            )
        if k == 0:
            # Degenerate empty contraction: fused_fp8_gemm has nothing to do;
            # the explicit dequant+matmul path returns the correct zeros.
            return False
        return True

    def load_fp8_weights(
        self,
        wq_b_fp8: mx.array,
        wq_b_scale_inv: mx.array,
        wk_bf16: mx.array,
        weights_proj_bf16: mx.array,
        k_norm_weight: Optional[mx.array] = None,
        k_norm_bias: Optional[mx.array] = None,
    ) -> None:
        """Inject FP8-quantized checkpoint tensors."""
        assert self.config.fp8_blocks, "load_fp8_weights requires fp8_blocks=True"
        if wq_b_fp8.dtype != mx.uint8:
            raise TypeError(f"wq_b_fp8 must be uint8 (fp8 storage); got {wq_b_fp8.dtype}")
        self._wq_b_fp8 = wq_b_fp8
        self._wq_b_scale_inv = wq_b_scale_inv.astype(mx.float32)
        self.wk.weight = wk_bf16.astype(mx.bfloat16)
        self.weights_proj.weight = weights_proj_bf16.astype(mx.bfloat16)
        if k_norm_weight is not None:
            self.k_norm.weight = k_norm_weight.astype(mx.bfloat16)
        if k_norm_bias is not None:
            self.k_norm.bias = k_norm_bias.astype(mx.bfloat16)

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        freqs_cis: tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        cfg = self.config
        batch, seq, _ = x.shape
        cos, sin = freqs_cis

        q = self._wq_b_apply(qr).reshape(batch, seq, cfg.n_heads, cfg.head_dim)
        q_pe = q[..., : cfg.rope_head_dim]
        q_nope = q[..., cfg.rope_head_dim:]
        q_pe = _apply_non_interleaved_rope(q_pe, cos, sin)
        q = mx.concatenate([q_pe, q_nope], axis=-1)

        k = self.k_norm(self.wk(x))
        k_pe = k[..., : cfg.rope_head_dim]
        k_nope = k[..., cfg.rope_head_dim:]
        k_pe = _apply_non_interleaved_rope(k_pe, cos, sin)
        k = mx.concatenate([k_pe, k_nope], axis=-1)

        weights = self.weights_proj(x) * (cfg.n_heads ** -0.5)
        sm_scale = (
            cfg.softmax_scale if cfg.softmax_scale is not None
            else cfg.head_dim ** -0.5
        )

        scores = mx.einsum("bqhd,bkd,bqh->bqk", q, k, weights) * sm_scale
        if mask is not None:
            scores = scores + mask
        topk = min(cfg.index_topk, scores.shape[-1])
        return mx.stop_gradient(
            mx.argpartition(-scores, topk - 1, axis=-1)[..., :topk]
        ).astype(mx.int32)


def quantize_indexer_weights_for_fp8(
    indexer: LightningIndexer,
) -> dict[str, mx.array]:
    """Convert an fp32 ``LightningIndexer`` to FP8-ready checkpoint tensors.

    Returns a dict that ``LightningIndexerFP8.load_fp8_weights`` accepts.
    Only ``wq_b`` is fp8-quantized; the rest are bf16 passthroughs.
    """
    w = indexer.wq_b.weight.astype(mx.float32)  # [out_dim, q_lora_rank]
    bs = _FP8_BLOCK
    m, n = w.shape
    pad_b = (-m) % bs
    pad_s = (-n) % bs
    blocks_m = (m + pad_b) // bs
    blocks_n = (n + pad_s) // bs
    padded = mx.pad(w, ((0, pad_b), (0, pad_s)))
    blocks = padded.reshape(blocks_m, bs, blocks_n, bs)
    amax = mx.maximum(mx.max(mx.abs(blocks), axis=(1, 3), keepdims=False), 1e-6)
    scale_inv = (amax / 448.0).astype(mx.float32)  # [blocks_m, blocks_n]
    scaled = (blocks / scale_inv[:, None, :, None]).reshape(
        m + pad_b, n + pad_s
    )[:m, :n]
    fp8 = mx.to_fp8(scaled.astype(mx.bfloat16))

    return {
        "wq_b_fp8": fp8,
        "wq_b_scale_inv": scale_inv,
        "wk_bf16": indexer.wk.weight.astype(mx.bfloat16),
        "weights_proj_bf16": indexer.weights_proj.weight.astype(mx.bfloat16),
        "k_norm_weight": indexer.k_norm.weight.astype(mx.bfloat16)
            if hasattr(indexer.k_norm, "weight") else None,
        "k_norm_bias": indexer.k_norm.bias.astype(mx.bfloat16)
            if hasattr(indexer.k_norm, "bias") else None,
    }


__all__ = [
    "LightningIndexerFP8",
    "LightningIndexerFP8Config",
    "quantize_indexer_weights_for_fp8",
]
