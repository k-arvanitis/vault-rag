# Vault RAG — Evaluation Summary

*Measured on a 82-question hold-out benchmark spanning PDFs, scanned/OCR documents,
and spreadsheets. Judge: claim-level RAGAS-style grading (gpt-oss-120b). Larger
215-question validation in progress.*

## Headline

| What it measures | Result |
|---|---|
| **Finds the right source** (retrieval hit@5) | **94%** |
| **Answers are grounded in the sources** (faithfulness) | **86%** |
| **Answers address the question** (relevancy) | **92%** |
| **Refuses to invent answers** (unanswerable / PII questions) | **100%** |
| **Single-document factual & table lookups** | **~94%** |

The system retrieves the correct evidence, grounds its answers in it, and — critically
for a business setting — **declines to answer when the information isn't present**
rather than fabricating it.

## By capability

| Capability | Questions | Score | Notes |
|---|---|---|---|
| Document retrieval (hit@5 / hit@10) | 53 | 94% / 96% | Right source surfaced in the top results |
| Evidence recall@10 | 53 | 91% | Of all needed evidence, fraction retrieved |
| Faithfulness (no hallucination) | 74 | 86% | Claim-level grading; ±5 run-to-run on the LLM judge |
| Answer relevancy | 82 | 92% | |
| Refusal on unanswerable questions | 8 | 100% | Home address, sort code, DOB, etc. — correctly declined |
| Structured data (Excel/CSV → SQL) | 21 | 81% | Exact single-table lookups ~94%; multi-report joins lower |
| Overall answer correctness | 82 | 79% | See "honest limitations" below |

## How it's evaluated (rigor matters to a technical buyer)

- **Hold-out benchmark**, not training data; questions span every document type.
- **Faithfulness graded at the claim level** (RAGAS/DeepEval-style): each factual
  claim must be inferable from the retrieved context, not just plausible.
- **Refusal is explicitly tested** with questions whose answers are *not* in the
  corpus (PII, absent fields) — the system must say "Unsupported."
- Retrieval and answer quality are measured **separately**, so we know whether a
  miss is a retrieval problem or a generation problem.

## Honest limitations (what the 79% overall reflects)

Overall correctness is held down by a deliberately hard subset that does **not**
represent typical end-user questions:

- **Cross-document arithmetic / aggregation** — e.g. summing 25 values across a
  scanned, multi-chunk table. (Mitigated: tables are now loaded into a SQL engine
  so exact `SUM`/`COUNT` works.)
- **Matching a record across two unrelated reports** that share no common key.
- **Questions with a debatable "correct" answer** — e.g. a document's formal title
  vs. its running header.

On **realistic single-document questions** — the bulk of real usage — the system
runs **~85-94%** (94% on the core table-lookup set; 85% on a broader 130-question
verified set generated from the source data). For a specific deployment, the right
benchmark is built from the client's own documents and the questions their users
actually ask.

## Engineering behind the numbers

Hybrid dense + sparse retrieval → cross-encoder reranking → grounded generation
with abstention guards, plus a text-to-SQL path for spreadsheet/tabular questions.
Faithfulness guarding and refusal behavior are first-class, not afterthoughts.
