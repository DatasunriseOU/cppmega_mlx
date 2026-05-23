// cppmega Visual Builder — design tokens (JS references)
//
// Every value here points at a CSS custom property defined in theme.css, so
// theme.css remains the single source of truth: edit a colour there and both
// the stylesheet rules and these inline-style consumers update together.
//
// Usage:  style={{ background: T.surface, color: T.text }}

import type { CSSProperties } from "react";
import type { BrickCategory } from "@/lib/bricks";

export const T = {
  // Surfaces
  bg:          "var(--vb-bg)",
  bgCanvas:    "var(--vb-bg-canvas)",
  surface:     "var(--vb-surface)",
  surface2:    "var(--vb-surface-2)",
  surface3:    "var(--vb-surface-3)",
  glass:       "var(--vb-surface-glass)",

  // Borders
  border:       "var(--vb-border)",
  borderSoft:   "var(--vb-border-soft)",
  borderStrong: "var(--vb-border-strong)",

  // Text
  text:          "var(--vb-text)",
  textSecondary: "var(--vb-text-secondary)",
  textMuted:     "var(--vb-text-muted)",

  // Accent
  accent:         "var(--vb-accent)",
  accentStrong:   "var(--vb-accent-strong)",
  accentSoft:     "var(--vb-accent-soft)",
  accentContrast: "var(--vb-accent-contrast)",

  // Semantic
  success: "var(--vb-success)",
  warning: "var(--vb-warning)",
  danger:  "var(--vb-danger)",
  info:    "var(--vb-info)",

  // Type
  font:     "var(--vb-font)",
  fontMono: "var(--vb-font-mono)",

  // Radii
  radiusSm:   "var(--vb-radius-sm)",
  radiusMd:   "var(--vb-radius-md)",
  radiusLg:   "var(--vb-radius-lg)",
  radiusXl:   "var(--vb-radius-xl)",
  radiusPill: "var(--vb-radius-pill)",

  // Elevation
  shadowPanel: "var(--vb-shadow-panel)",
  shadowPop:   "var(--vb-shadow-pop)",
} as const;

// Per-category neon accent (mirrors lib/bricks.ts categories).
export const CATEGORY_ACCENT: Record<BrickCategory, string> = {
  sdpa_attention: "var(--vb-cat-attn)",
  linear_attn:    "var(--vb-cat-linear)",
  ssm:            "var(--vb-cat-ssm)",
  moe:            "var(--vb-cat-moe)",
  sparse_attn:    "var(--vb-cat-sparse)",
  cross_attn:     "var(--vb-cat-cross)",
  norm_or_proj:   "var(--vb-cat-norm)",
  nonlinear_rnn:  "var(--vb-cat-rnn)",
  io:             "var(--vb-accent)",
};

export function accentForCategory(cat: BrickCategory | undefined): string {
  return cat ? CATEGORY_ACCENT[cat] : T.textSecondary;
}

// Sets the --vb-node-accent custom property consumed by .vb-node / .vb-chip /
// .vb-palette-item hover + glow rules. Cast keeps TS happy about the custom
// property key.
export function accentVar(color: string): CSSProperties {
  return { ["--vb-node-accent"]: color } as CSSProperties;
}

// A category-coloured Tabler-style glyph for palette + node chips. Kept as a
// tiny inline-SVG map so we don't pull in an icon dependency.
export const CATEGORY_ICON: Record<BrickCategory, string> = {
  sdpa_attention: "✦",
  linear_attn:    "≋",
  ssm:            "∿",
  moe:            "⧉",
  sparse_attn:    "⋰",
  cross_attn:     "✕",
  norm_or_proj:   "∥",
  nonlinear_rnn:  "↺",
  io:             "🔠",
};
