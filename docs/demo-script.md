# Demo script (3–4 minutes)

A live walkthrough script, not a recording — run it against a real `make api`/`make ui`
stack with real documents. Don't fabricate any of these steps; if a service is down,
say so rather than skipping to a canned result.

## Setup (before recording/presenting)

- `make api` + `make ui` running, Qdrant/LiteLLM/Ollama up.
- Have a scan (e.g. `doc_004_foia_invoices_packet.pdf`) and a spreadsheet
  (e.g. `doc_006_purchase_card_transactions_q1_2025_26.xlsx`) ready to drop in,
  or already ingested via `make seed`.

## Script

1. **Upload a scan and a spreadsheet.** Drag both into the sidebar's upload
   dropzone. Narrate: "Scanned pages route through local OCR, born-digital
   pages skip it entirely — per page, not per document." Wait for both to
   hit Ready.

2. **Ask a PDF question.** e.g. "According to the procurement policy, when
   was it approved by the Board of Retirement?" Point out the answer
   streaming token-by-token and the inline `[N]` citation chip appearing.

3. **Click the citation and inspect the original page.** Evidence panel
   opens showing the actual PDF page image with the cited passage
   highlighted (or the honest "Exact region unavailable" fallback — both are
   correct, explain why if it comes up).

4. **Ask a spreadsheet aggregation question.** e.g. "What's the total spend
   on advertising costs?" Point out this routes to a SQL agent, not the
   PDF retrieval path.

5. **Show the generated SQL and the highlighted source row.** Open
   Technical details for the generated SQL; click the spreadsheet citation
   to show the matched row highlighted in Evidence, then open the full
   Document Inspector to show the same row highlighted in both the raw and
   cleaned table views.

6. **Compare two documents.** e.g. "Compare doc_009 and doc_010 — what does
   each say about vacation leave?" Point out the answer draws evidence from
   both documents (not just one), and that this is deterministic — the
   system always retrieves both sides independently, not left to chance.

7. **Ask an unsupported question.** Something plainly not in the corpus.
   Show the plain refusal — no hedging, no invented answer.

8. **Briefly show the evaluation results.** Open the Quality → Evaluation
   screen — correctness/faithfulness/relevancy/Hit@5, broken down by
   question type, generated from a real benchmark run
   (`docs/EVAL_SUMMARY.md`), not hand-typed numbers.

## What NOT to do

- Don't skip a step because a service is slow — narrate the wait instead of
  cutting away, or say plainly "this would normally take Xs."
- Don't reuse a screenshot/recording from a different code version than
  what's actually running.
- Don't claim a metric or behavior this session didn't actually verify.
