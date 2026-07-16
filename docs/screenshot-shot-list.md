# Screenshot shot list

Where to save captures and what each one should show, for the README /
portfolio page. File paths assume `docs/screenshots/` (create it if it
doesn't exist yet — no screenshots are committed by this doc, only the plan
for taking them).

| # | File | What it shows | Caption |
|---|---|---|---|
| 1 | `docs/screenshots/01-chat-citations.png` | Main chat view: a streamed answer with inline `[N]` citation chips visible | "Every claim links back to a numbered source — click any chip to see the exact passage." |
| 2 | `docs/screenshots/02-evidence-pdf-highlight.png` | Evidence panel open, PDF page image with the cited passage highlighted | "Citations aren't just text — they point at the actual page and the exact highlighted region." |
| 3 | `docs/screenshots/03-spreadsheet-sql.png` | Technical details tab showing generated SQL for a spreadsheet question | "Structured questions get answered by a real SQL engine over the cleaned table, not guesswork." |
| 4 | `docs/screenshots/04-inspector-raw-vs-cleaned.png` | Document Inspector, raw-vs-cleaned table view with a matched row highlighted in both | "The same row, highlighted in both the original file and the LLM-cleaned table — nothing hidden." |
| 5 | `docs/screenshots/05-comparison-answer.png` | A cross-document comparison answer with citations from two distinct documents visible in the sources list | "Comparisons are deterministic — evidence from every named document, guaranteed, not left to the model's judgment." |
| 6 | `docs/screenshots/06-refusal.png` | A clearly out-of-corpus question answered with a plain refusal | "No hedging, no invented answer when the information isn't there." |
| 7 | `docs/screenshots/07-sidebar-sources.png` | Sidebar with several ingested documents, showing type/status/page-or-sheet counts | "Every source's ingestion status and size, at a glance." |
| 8 | `docs/screenshots/08-eval-dashboard.png` | Quality → Evaluation screen with correctness/faithfulness/relevancy/Hit@5 and the by-question-type breakdown | "Benchmark numbers generated from a real run, not hand-typed — see docs/EVAL_SUMMARY.md." |

## Capture notes

- Use a real ingested corpus (the 18-document eval set or `make seed`'s
  starter subset) — not placeholder/lorem-ipsum content.
- Prefer light mode for README embeds unless dark mode is the deliberate
  brand choice — check contrast either way.
- Crop to the relevant panel, not the full browser chrome, except for #1
  which benefits from showing the full layout once.
- If Playwright is available and the app can run with representative data,
  a screenshot script can automate steps 1–7 (not 8, which needs a real
  `make eval` run first) — see `docs/release-checklist.md` for how to bring
  the stack up. Do not fabricate a screenshot by editing one manually.
