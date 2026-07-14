import { defineConfig } from "@playwright/test";

// ponytail: local-only smoke test, not wired into CI — needs the full live
// stack (Qdrant, embedding, generation) that CI doesn't run. See e2e/README.md.
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3002",
  },
});
