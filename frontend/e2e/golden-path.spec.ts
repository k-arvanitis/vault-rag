import { test, expect } from "@playwright/test";

// ponytail: exercises the flagship flow (query -> citation -> evidence panel)
// against a live dev stack. Not a substitute for unit coverage on branch logic
// -- it only proves the wiring holds end to end. Run: `npm run test:e2e`
// with `make api` + `make ui` (and Qdrant/embedding/generation) already up.
//
// Known flaky: retrieval grounding for this question is inconsistent (see
// TODO.md's ungrounded-retrieval note) -- occasional failures reflect a real
// answer-quality issue, not a broken test. Retry once before assuming the
// wiring itself regressed.
test("ask a question and see a cited, evidence-backed answer", async ({ page }) => {
  await page.goto("/");

  const input = page.getByPlaceholder("Ask about your documents…");
  await input.fill(
    "According to the procurement policy document (doc_001), when was it approved by the Board of Retirement?"
  );
  await page.getByLabel("Send").click();

  const citation = page.getByRole("button", { name: /^\[1\]/ }).first();
  await expect(citation).toBeVisible({ timeout: 90_000 });

  // EvidencePanel auto-populates from the answer's sources once it lands.
  await expect(page.getByRole("img", { name: /Page \d+/ }).first()).toBeVisible({
    timeout: 15_000,
  });
});
