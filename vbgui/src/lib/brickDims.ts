/**
 * Per-brick dimension + parameter-count estimator. Used by the
 * BrickContextPanel "Dimensions & params" block so the architect
 * sees what flows in/out of each brick + how many weights it owns,
 * before running verify_build_spec.
 *
 * All formulas are canonical and match the BLOCK_BUILDERS forward
 * passes; numbers are rough (Llama-style: nh*head_dim = H), so the
 * panel labels this section "approx".
 */


export interface BrickDims {
  /** Human-readable input shape, e.g. "(B, S, H)". */
  input: string;
  /** Human-readable output shape. */
  output: string;
  /** Approx parameter count (weights only — bias usually negligible). */
  n_params: number;
  /** Short formula explaining where n_params comes from. */
  formula: string;
}


function fmtNum(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1_000_000) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1_000) return `${(n / 1e3).toFixed(1)} K`;
  return String(Math.round(n));
}


export function fmtParamCount(n: number): string {
  return fmtNum(n);
}


export function computeBrickDims(
  kind: string,
  hidden_size: number,
  params: Record<string, unknown>,
  vocab_size: number = 65536,
): BrickDims {
  const H = hidden_size;
  const num = (k: string, def: number): number => {
    const v = params?.[k];
    return typeof v === "number" ? v : def;
  };

  switch (kind) {
    case "attention":
    case "gqa_sliding":
    case "cca_attention": {
      const nh = num("num_attention_heads", num("num_heads", Math.max(8, H / 64)));
      const nkv = num("num_key_value_heads", nh);
      const hd = num("head_dim", H / nh);
      const Wq = H * nh * hd;
      const Wkv = 2 * H * nkv * hd;
      const Wo = nh * hd * H;
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: Wq + Wkv + Wo,
        formula: `Wq(H·nh·hd) + Wkv(2·H·nkv·hd) + Wo(nh·hd·H) where ` +
                 `H=${H}, nh=${nh}, nkv=${nkv}, hd=${hd}`,
      };
    }
    case "gated_attention": {
      const nh = num("num_attention_heads", Math.max(8, H / 64));
      const nkv = num("num_key_value_heads", nh);
      const hd = num("head_dim", H / nh);
      const Wq = H * nh * hd;
      const Wkv = 2 * H * nkv * hd;
      const Wo = nh * hd * H;
      const Wg = H * nh * hd;
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: Wq + Wkv + Wo + Wg,
        formula: `Wq + Wk + Wv + Wo + Wg, gate adds H·nh·hd extra`,
      };
    }
    case "mla":
    case "mla_absorb":
    case "mistral4_mla":
    case "bailing_mla": {
      const qr = num("q_lora_rank", Math.floor(H / 8));
      const kvr = num("kv_lora_rank", Math.floor(H / 16));
      const hd = num("head_dim", 64);
      const nh = num("num_attention_heads", Math.max(8, H / 64));
      const Wq_lora = H * qr + qr * nh * hd;
      const Wkv_lora = H * kvr + kvr * 2 * nh * hd;
      const Wo = nh * hd * H;
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: Wq_lora + Wkv_lora + Wo,
        formula: `LoRA-Q + LoRA-KV + Wo, q_rank=${qr}, kv_rank=${kvr}`,
      };
    }
    case "mlp": {
      const dff = num("intermediate_size", 4 * H);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: 3 * H * dff,
        formula: `W_gate + W_up + W_down: 3 · H · d_ff = 3 · ${H} · ${dff}`,
      };
    }
    case "moe":
    case "bailing_moe": {
      const ne = num("num_experts", 8);
      const tk = num("top_k", 2);
      const dff = num("intermediate_size", 4 * H);
      const per_expert = 3 * H * dff;
      const router = H * ne;
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: ne * per_expert + router,
        formula: `${ne} experts × 3·H·d_ff (${fmtNum(per_expert)} each) ` +
                 `+ router(H·${ne}). Active per token = ${tk}·${fmtNum(per_expert)} = ` +
                 `${fmtNum(tk * per_expert)}`,
      };
    }
    case "mamba3": {
      const dst = num("d_state", 16);
      const conv = num("conv_size", 4);
      const hd = num("head_dim", 64);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: 5 * H * H + 2 * H * dst + H * conv,
        formula: `linear_in + conv1d + Δ/B/C projections + linear_out, ` +
                 `d_state=${dst}, conv=${conv}, head_dim=${hd}`,
      };
    }
    case "mlstm": {
      const hd = num("head_dim", 64);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: 6 * H * H,
        formula: `W_q+W_k+W_v + W_i+W_f+W_o: 6 · H² (head_dim=${hd})`,
      };
    }
    case "gdn":
    case "kda": {
      const nh = num("num_heads", Math.max(8, H / 64));
      const hd = num("head_dim", 64);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: 5 * H * H,
        formula: `W_q+W_k+W_v+W_β+W_g, nh=${nh}, head_dim=${hd}`,
      };
    }
    case "rmsnorm": {
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: H,
        formula: `γ: (H,) = ${H} learnable scales`,
      };
    }
    case "layernorm": {
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: 2 * H,
        formula: `γ + β: 2·H = 2·${H} learnable affine`,
      };
    }
    case "abs_pos_embed": {
      const max_pos = num("max_position_embeddings", 2048);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: max_pos * H,
        formula: `W_pos: (max_pos=${max_pos}, H=${H})`,
      };
    }
    case "per_layer_embed": {
      const nl = num("num_layers", 32);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: nl * H,
        formula: `per-layer scale vector: num_layers × H = ${nl}·${H}`,
      };
    }
    case "engram": {
      const mem = num("memory_size", 256);
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: mem * H + 3 * H * H,
        formula: `memory_bank(${mem}·H) + Q/K/V projections`,
      };
    }
    case "embedding_table": {
      return {
        input: "(B, S) token ids",
        output: "(B, S, H)",
        n_params: vocab_size * H,
        formula: `vocab × H = ${fmtNum(vocab_size)} × ${H}`,
      };
    }
    default: {
      return {
        input: "(B, S, H)",
        output: "(B, S, H)",
        n_params: 0,
        formula: `no formula for kind=${kind}; check BLOCK_BUILDERS`,
      };
    }
  }
}
