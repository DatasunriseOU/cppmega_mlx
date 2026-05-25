// V7-E-AUDIT-02: client-side cache of compatible (src_kind, dst_kind)
// pairs fetched once from catalog.list_options('compatible_edges').
// FlowCanvas's isValidConnection callback consults this set to reject
// incompatible drags client-side (no server round-trip).

import { useEffect, useState } from "react";
import type { RpcClient } from "@/lib/rpc";

export function useCompatibleEdges(rpc: RpcClient): Set<string> {
  const [pairs, setPairs] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await rpc.call<{
          options: Array<{ name: string; summary: string;
                            paper_ref: string }>
        }>("catalog.list_options",
            { category: "compatible_edges" });
        if (cancelled) return;
        const s = new Set<string>();
        for (const opt of r.options) {
          // name is "src->dst"; paper_ref=src, summary=dst.
          s.add(`${opt.paper_ref}→${opt.summary}`);
        }
        setPairs(s);
      } catch {
        // Empty set => isValidConnection conservatively allows
        // everything, falling back to server-side verify.
      }
    })();
    return () => { cancelled = true; };
  }, [rpc]);

  return pairs;
}

/** isValidConnection predicate built from a compatible-edges set.
 *  Returns true when the set is empty (load-failure fallback) or when
 *  the (src, dst) pair is in the set. */
export function makeIsValidConnection(
  pairs: Set<string>,
  brickKindOf: (nodeId: string) => string | null,
): (conn: { source: string | null; target: string | null }) => boolean {
  return (conn) => {
    const src = conn.source ? brickKindOf(conn.source) : null;
    const dst = conn.target ? brickKindOf(conn.target) : null;
    if (!src || !dst) return false;

    // Tokenizer strictly connects only to embedding_table
    if (src === "tokenizer") {
      return dst === "embedding_table";
    }
    // De-Tokenizer strictly accepts connections only from embedding_table, linear_bridge, mlp or other norm_or_proj bricks
    if (dst === "detokenizer") {
      return src === "embedding_table" || src === "linear_bridge" || src === "mlp" || src === "rmsnorm" || src === "layernorm" || src === "per_layer_embed" || src === "abs_pos_embed";
    }

    if (pairs.size === 0) return true;        // server fallback
    return pairs.has(`${src}→${dst}`);
  };
}
