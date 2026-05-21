// Library-mode build that produces a single ESM bundle consumable by
// anywidget. Output goes to ../cppmega_v4/widget/static/ so the Python
// package can ship it as package_data.

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  build: {
    outDir: path.resolve(__dirname, "../cppmega_v4/widget/static"),
    emptyOutDir: true,
    lib: {
      entry: path.resolve(__dirname, "src/anywidget.tsx"),
      formats: ["es"],
      fileName: () => "widget.mjs",
    },
    rollupOptions: {
      output: { assetFileNames: "widget.[ext]" },
    },
    cssCodeSplit: false,
  },
});
