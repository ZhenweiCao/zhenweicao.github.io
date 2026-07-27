import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["spec/**/*.cases.ts"],
    reporters: ["default"],
    alias: {
      "./styles/graph.scss": path.resolve(__dirname, "spec/__mocks__/styleMock.ts"),
      "./scripts/graph.inline.ts": path.resolve(__dirname, "spec/__mocks__/scriptMock.ts"),
    },
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "preact",
  },
});
