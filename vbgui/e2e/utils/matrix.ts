// Read tests/fixtures/MATRIX.json so scenarios can iterate the same
// combinations the Python side knows about.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MATRIX_PATH = resolve(__dirname, "..", "..", "..",
                            "tests", "fixtures", "MATRIX.json");

export interface MatrixIndex {
  tokenizers: Record<string, {
    path: string;
    vocab_size: number;
    specials: string[];
    digest: string;
    fresh: boolean;
  }>;
  parquets: Record<string, {
    path: string;
    tokenizer: string;
    schema: string;
    rows: number;
    seq_len: number;
    columns: string[];
  }>;
  round_trip: Record<string, { first_decoded: string; non_empty: string }>;
}

let cache: MatrixIndex | null = null;

export function loadMatrix(): MatrixIndex {
  if (cache) return cache;
  const raw = readFileSync(MATRIX_PATH, "utf-8");
  cache = JSON.parse(raw) as MatrixIndex;
  return cache;
}

export const TOKENIZER_NAMES = [
  "T1_cppmega_v3", "T2_gpt2_small",
  "T3_minimal_no_fim", "T4_fim_only",
] as const;

export const PARQUET_SCHEMAS = [
  "P1_minimal", "P2_doc", "P3_engram", "P4_full",
] as const;

export const PRESET_NAMES = [
  "qwen3_next", "kimi_linear", "kimi_k2", "deepseek_v3",
  "deepseek_v4_flash", "gemma4", "mistral4", "ling26",
  "longcat", "nemotron3", "zaya1", "arcee_trinity",
] as const;
