# Vault RAG — Presentation Brief

Purpose: input for building a slide deck. Pair this with `docs/diagrams.md` (the two
Mermaid pipeline diagrams) and `src/prompts.py` (the actual prompts). This brief names
which Python file does what and explains every mechanism in each pipeline at a level
suitable for an interview presentation — detailed enough to be credible, not a
line-by-line code tour.

Audience: technical interviewers. Tone: an engineer presenting a portfolio project.

The two narrative pillars to keep returning to: **(a) it is agentic** — the LLM drives
control flow through tools and self-correction, not a fixed chain; **(b) it is
multi-modal** — vision + text + structured data, ingested by purpose-built paths and
unified behind one agent. Slide 5 makes that case explicitly.

---

## Slide 1 — The Problem & What Vault RAG Is

**The problem.** Organizations sit on piles of mixed private documents — contracts and
reports as PDFs (some born-digital, some scanned), spreadsheets of transactions, Office
files. Answering one factual question means a human opening files by hand. A naive
"chatbot over documents" does not solve this, for five concrete reasons:

1. Scanned pages have **no text layer** — nothing to embed.
2. Charts and figures are **invisible to text search**.
3. Spreadsheet questions need **exact math** (`SUM`, `WHERE`) — embeddings cannot do that.
4. Plain vector search is **too coarse** to reliably land the right passage.
5. LLMs **fabricate** when they cannot find an answer.

**What Vault RAG is.** A retrieval-augmented question-answering system over that mixed
pile — where each of those five failures is fixed by a specific engineered mechanism,
not a prompt.

- **Local-first.** Documents, the vector database, and the SQL store all stay on the
  machine. Only LLM API calls leave it.
- **Two modalities, one agent.** Unstructured text (PDFs) and structured data
  (spreadsheets) are handled by two different sub-systems, unified behind a single agent.
- **Engineered, not wrapped.** The value is in the retrieval quality and reliability
  work, not in calling an API.

Two pipelines: an **ingestion pipeline** (files → searchable stores) and a **query
pipeline** (question → grounded answer).

---

## Slide 2 — Frameworks & Stack

Being explicit about the stack is itself a talking point — every choice is deliberate.

| Layer | Framework / Tool | Role |
|---|---|---|
| Agent orchestration | **LangGraph** | the ReAct agent loop + the Excel text-to-SQL sub-graphs |
| LLM plumbing | **LangChain** (core) | message types, tool abstractions, `ChatOpenAI` client |
| LLM gateway | **LiteLLM** proxy | one OpenAI-compatible endpoint, multi-provider failover (Slide 4) |
| Vector database | **Qdrant** | dense + sparse vectors, hybrid search |
| SQL store | **DuckDB** | embedded SQL engine for spreadsheet rows |
| Embeddings server | **Ollama** | local `nomic-embed-text` |
| OCR serving | **vLLM** | serves the LightOn OCR vision model (GPU) |
| API backend | **FastAPI** | async `/query`, `/ingest`, `/documents` endpoints |
| Frontend | **Next.js / React** | chat UI + retrieval-trace inspector |
| Chat interface | **Slack Bolt** | Slack query bot (thin client over the API) |
| PDF parsing | **pymupdf4llm** | born-digital page → markdown (CPU, no model) |
| Observability | **Langfuse / LangSmith** | tracing every agent run and tool call |
| Evaluation | RAGAS-style **LLM-judge harness** | correctness / faithfulness / relevancy / refusal |
| Tooling | **uv**, **ruff**, **pytest**, **Docker** | deps, lint, tests, local service stack |

Headline: **LangGraph** for agency, **Qdrant + DuckDB** for the two modalities,
**LiteLLM** for provider resilience, **FastAPI + Next.js** for delivery.

---

## Slide 3 — Infrastructure & Providers

**Runs in Docker (one Compose stack):**

- **Qdrant** — vector database (document embeddings + summaries).
- **Ollama** — local embedding-model server (`nomic-embed-text`).
- **LiteLLM proxy** — single LLM gateway (its own slide, next).
- **LightOn OCR** — OCR vision model on a vLLM server; GPU-only, behind an optional
  Compose profile. Born-digital PDFs, Excel, and Markdown ingest fine without it.

**Not in Docker:** the Python application itself, the reranker model (loaded in-process
on CPU), and DuckDB (an embedded file, not a server).

**External providers in play:** Groq (main reasoning LLM + figure-vision model),
NVIDIA NIM (reasoning fallback via LiteLLM), OpenRouter (chunk-enrichment LLM at
ingest), OpenAI (the Excel text-to-SQL sub-agent). Slide 15 maps every model to its
provider.

---

## Slide 4 — The LiteLLM Gateway

**The problem it solves.** The system makes many LLM calls (agent, HyDE, judges,
repairs, table detection). Calling providers directly would mean provider-specific
SDKs, scattered API-key handling, and a hard failure the moment one provider
rate-limits or goes down — and Groq's free tier rate-limits often.

**The solution.** A **LiteLLM proxy** — one OpenAI-compatible endpoint
(`http://localhost:4000/v1`) that the whole app targets via `GENERATION_API_BASE`.
**We use LiteLLM with Groq as the primary provider and the other providers as automatic
fallbacks** — the app never talks to a provider SDK directly.

**Configured in `litellm_config.yaml`** — three model entries, all registered under the
**same `model_name`** (`qwen/qwen3-32b` — the value of `GENERATION_MODEL`), which makes
LiteLLM treat them as one automatic failover chain:

| Priority | Provider | Model |
|---|---|---|
| 1 — primary | **Groq** | Qwen3 32B — `qwen/qwen3-32b` |
| 2 — fallback | **Groq** | Llama 3.3 70B Versatile |
| 3 — reasoning fallback | **NVIDIA NIM** | Qwen3 Next 80B |

**Router behavior:** `num_retries: 3`, `allowed_fails: 1` before switching provider,
`cooldown_time: 60s` on a failed provider, `simple-shuffle` ordering. `drop_params: true`
strips parameters a given provider rejects, so one request shape works across all three.
Optional `master_key` auth (`LITELLM_MASTER_KEY`).

**What it achieves:**

- **One integration point** — app code targets one URL and one SDK; adding or swapping a
  provider is a YAML edit with **zero code change**.
- **Automatic failover** — a Groq rate-limit transparently rolls to the next provider;
  the user's query does not fail.
- **Resilience** — retries plus per-provider cooldown smooth over transient errors.
- **Cross-provider compatibility** — `drop_params` hides each provider's quirks.

Scope note: LiteLLM gateways the **agent / reasoning** model. Other calls — OCR (vLLM),
figure vision (Groq), chunk enrichment (OpenRouter), Excel text-to-SQL (OpenAI) — go
direct; see Slide 15.

---

## Slide 5 — What Makes It Agentic & Multi-modal

*(The interview-critical slide. These two properties are the whole pitch.)*

**Agentic — the LLM drives control flow, it is not a fixed chain.**

- Not a static RAG chain (`retrieve → stuff → answer`). It is a **LangGraph ReAct
  agent**: a think → act → observe loop where the LLM decides, per question, *which*
  tool to call, with *what* arguments, and *whether* to call again.
- **Tools, not steps.** `search_knowledge_base` and `query_excel` are tools the agent
  chooses between — it can call one, the other, or both, in any order.
- **An agent inside an agent.** `query_excel` is *itself* an agent — a self-correcting
  text-to-SQL LangGraph that loops (write SQL → run → evaluate → retry or try the next
  table) until it succeeds or exhausts candidates.
- **Self-correction / reflection.** The answer-refinement layer runs an LLM-as-judge
  that grades the answer's completeness and triggers targeted re-retrieval — the system
  critiques and repairs its own output.
- **Deterministic scaffolding.** `route_question` pre-resolves the tool choice — agentic
  where judgement is genuinely needed, deterministic where a rule is reliable (Slide 14).

**Multi-modal — vision + text + structured data, unified.**

- **Input modalities:** born-digital PDFs, scanned PDFs (image), figures/charts (image),
  Office docs, and spreadsheets (structured tabular data).
- **Model modalities:** text LLMs (reasoning, SQL), a **vision-language model**
  (figures → text), an **OCR vision model** (scanned pages → markdown), and **embedding
  models** (dense + sparse).
- **Storage modalities:** unstructured text → **Qdrant** (vector DB, semantic search);
  structured rows → **DuckDB** (SQL, exact aggregation). Two stores because the two
  question types are fundamentally different.
- **Unified:** the user asks one natural-language question; the agent routes across the
  text and tabular modalities transparently.

One-line takeaway: *agentic* = the model owns the control flow; *multi-modal* = it
reasons over vision, text, and structured data through one interface.

---

## Slide 6 — Architecture Diagrams

> 🖼️ **DIAGRAM PLACEHOLDER 1 — Ingestion architecture.**
> Source: `docs/diagrams.md` (Mermaid). Should show: uploaded file → file-type router →
> **PDF path** (per-page routing → OCR / VLM → table repair → contextual chunking →
> dual embedding → Qdrant) and **Excel path** (raw loader → LLM structure detection →
> DuckDB rows + per-sheet summaries → Qdrant).

> 🖼️ **DIAGRAM PLACEHOLDER 2 — Query architecture.**
> Source: `docs/diagrams.md` (Mermaid). Should show: question → `route_question` →
> **ReAct agent** → { `search_knowledge_base` retrieval pipeline | `query_excel`
> text-to-SQL agent } → answer-refinement layer → grounded answer; with the
> Qdrant / DuckDB stores and the LiteLLM gateway drawn in.

Present these two diagrams *before* the detail slides so the audience has the map first.

---

## Slide 7 — Ingestion Pipeline: PDF Path

`ingest.py` is the orchestrator — it routes each file by type and runs the stages.

| File | Role |
|---|---|
| `ingest.py` | Orchestrator; file-type router; runs the 5 stages |
| `parser/pdf_parser.py` | **Per-page two-path router** |
| `ingestion/ocr.py` | Sends a scanned page to the OCR model |
| `ingestion/vlm.py` | Sends a figure to the vision-language model |
| `table_processor.py` | Converts ASCII grid tables into row sentences |
| `chunker.py` | Splits markdown into chunks; LLM context enrichment |
| `embedder.py` | Turns chunks into embedding vectors |
| `vector_store.py` | Upserts vectors into Qdrant |

**Key mechanisms:**

- **Per-page routing.** Each PDF page is classified independently. A page with a real
  text layer (≥ 50 characters) goes through a fast CPU markdown extractor — no model.
  A scanned page (no text layer) is rendered to an image and sent to the OCR model.
  Running OCR on born-digital pages would corrupt numbers and equations that are already
  perfect — so the two paths are kept strictly separate.
- **Figure handling.** Charts and diagrams on text pages are invisible to text search.
  They are detected, sent to a vision-language model, and replaced inline with a text
  description — so visual content becomes searchable.
- **Table repair.** OCR mangles tables in three specific ways (split headers, LaTeX
  blobs, missing last column). These are repaired *structurally* — no LLM — to avoid
  hallucinated table data.
- **Contextual chunking.** Each chunk is enriched with a one-line LLM-written context
  and embedded as *context + content*, so a chunk stays findable even when it lacks
  self-contained context ("the payment term in the Services Contract"). The background
  the enrichment LLM sees adapts to document size — the whole document when it is small
  enough (Anthropic Contextual Retrieval), or the document summary plus a window of
  neighbouring chunks when it is too large to send whole — so the per-chunk cost stays
  bounded even on a 150-page report.
- **Dual embedding.** Every chunk is stored with a dense vector (meaning) and a sparse
  vector (exact keywords) — this is what makes hybrid search possible later.
- **Idempotent ingestion.** Each vector gets a deterministic ID derived from
  (file, chunk index), so re-ingesting a file overwrites instead of duplicating.

---

## Slide 8 — Ingestion Pipeline: Excel Path

A spreadsheet is handled completely differently from a PDF.

| File | Role |
|---|---|
| `ingest_table_rows.py` | Orchestrator for spreadsheets |
| `ingest_tables.py` | Raw loader; forward-fills merged cells |
| `preprocessing/excel_cleaner.py` | LLM detects the table structure |
| `duckdb_store.py` | Loads cleaned data into DuckDB |

**Key idea — split storage by what each store is good at:**

- **The rows go to DuckDB** as real SQL tables. Spreadsheet questions need exact lookups
  and aggregation (`SUM`, `WHERE`) — embeddings cannot do that.
- **Only per-sheet summaries go to Qdrant.** Just enough (file, sheet, columns, sample
  values) for the agent to *discover* which table is relevant.

**Mechanisms:**

- **LLM structure detection.** Real spreadsheets have title banners, multi-row headers,
  units rows, and footnotes. An LLM reads the first rows and reports where the real
  table starts and what the columns are — a narrow, low-risk task. A heuristic fallback
  runs if the LLM fails.
- **Sample values in summaries.** Each sheet summary embeds real sample values per
  column, sampled across the whole sheet — so an entity buried deep in the data is still
  discoverable.

---

## Slide 9 — Query Pipeline: The Agent

`rag_agent.py` builds and runs the agent.

- `build_rag_agent` — a startup factory: loads and warms up the reranker, creates the
  LLM client (pointed at the LiteLLM gateway), builds the two tools, and wires them into
  a **LangGraph ReAct agent** (a tool-calling loop: think → call a tool → observe →
  repeat → answer).
- `ask_agent` / `stream_agent` — run the agent per question; `stream_agent` streams
  tokens for the live UI.
- `route_question` — *before* the agent runs, deterministically resolves which tool the
  question needs (Slide 14).

The agent has **two tools**, one per modality:

- `search_knowledge_base` — document retrieval (Slide 10)
- `query_excel` — spreadsheet text-to-SQL (Slide 11)

---

## Slide 10 — Tool 1: `search_knowledge_base` (Retrieval Pipeline)

File: `tools/retrieval_tool.py`. This is not "look in Qdrant" — it is a retrieval
pipeline, each stage fixing a specific failure of naive vector search. Internally it is
three named stages: **resolve scope → retrieve & rerank → format**.

- **HyDE (hypothetical-answer expansion).** A question and its answer share little
  vocabulary. An LLM first writes a short *hypothetical answer*, which embeds much closer
  to real document text.
- **Dual search, then merge.** Retrieval issues **two** Qdrant searches — one with the
  raw question, one with the HyDE hypothetical — and **merges both result lists**. The
  raw query is strong on exact terms (codes, IDs, names); the HyDE query is strong on
  paraphrased, answer-style content. Each covers the other's blind spot. HyDE is
  fault-tolerant (if it fails, retrieval continues on the raw results alone) and can be
  switched off with a flag.
- **Two-stage retrieval.** Stage 1 identifies which *document* is relevant from the
  summaries; Stage 2 runs a search scoped/boosted toward that document — so a search for
  "revenue" does not pull chunks from the wrong file.
- **Hybrid search.** Each search combines the dense vector (meaning) and the sparse
  vector (exact keywords), fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking.** Vector similarity is coarse — the best chunk is often
  ranked seventh. A cross-encoder model re-scores the top candidates precisely. It runs
  in-process on CPU and is warmed up at startup so the first query has no cold-start lag.
- **Neighbour-chunk injection.** For each top hit, the chunks immediately before and
  after it are pulled in too — so a chunk never loses the context next to it.
- **Snippet selection & table formatting.** Long chunks are trimmed to the relevant
  slice; table chunks are re-rendered as key-value text the LLM reads easily.

---

## Slide 11 — Tool 2: `query_excel` (Text-to-SQL Agent)

Files: `tools/excel.py` (the agent) + `duckdb_store.py` (the DuckDB access layer).

This is a **self-correcting text-to-SQL agent** — two nested LangGraph graphs.

- **Decompose.** A multi-part question is split into focused sub-questions.
- **Table selection — by column-name matching.** For each sub-question, candidate
  DuckDB tables are ranked by how well the question's words overlap the table's
  **column names** (and scoped to a document if one is named explicitly). Note: this
  uses the live DuckDB schema, *not* the Qdrant sheet summary.
- **Inner SQL loop, per sub-question:**
  1. Select a candidate table.
  2. Inspect — pull its schema and a few sample rows from DuckDB.
  3. Write SQL — an LLM writes a query from the schema + samples.
  4. Run the SQL.
  5. Evaluate the result, and route accordingly.
- **Self-correction — the key design point.** The evaluator distinguishes two failures:
  a **SQL error** means "fix the query" → retry the *same* table; **zero rows** means
  "wrong table" → move to the *next* candidate table. A naive agent would retry one
  table forever.
- **Fan-out.** Sub-questions run as parallel branches (LangGraph `Send`), then
  **synthesize** merges the sub-answers into one final answer.

The SQL context is the table's exact schema and real sample rows — a lossy summary is
not enough to write correct SQL (exact column names and types matter).

---

## Slide 12 — Prompts

> 📝 **PROMPT PLACEHOLDER — paste 2–3 key prompts verbatim from `src/prompts.py`.**
> All prompts are centralized in one file (nothing hardcoded inline). The deck should
> show the *actual* instructions, not paraphrases, so the audience sees the engineering.

Worth showing (all in `src/prompts.py`):

- **`compose_system_prompt`** — the agent's system prompt, assembled from blocks:
  tools, rules, a **clarification rule**, an **abstention rule** (when to say
  "Unsupported"), and a citation block. The abstention block is what suppresses
  fabrication.
- **`DECOMPOSE_PROMPT`** — splits a multi-part question into sub-questions (Excel agent).
- **`SQL_PROMPT_HEADER` / `SQL_RETRY_HINT`** — instruct the LLM to write DuckDB SQL from
  schema + samples, and how to fix a failed query.
- **`CHUNK_CONTEXT_PROMPT` / `DOCUMENT_SUMMARY_PROMPT`** — the ingest-time enrichment
  prompts (one-line chunk context, per-document summary).

Talking point: prompts are versioned in code, in one file — not scattered f-strings.

---

## Slide 13 — Answer Refinement: The Answer Is a Draft

The agent's first answer is never returned directly. `rag_agent.py` runs a four-step
refinement pipeline (documented in-code as an ordered pipeline):

1. **Repair.** An **LLM-as-judge** grades the answer purely on *completeness* — did it
   cover every part of the question? If not, the judge writes focused search queries for
   the gaps; those are re-retrieved and the answer is regenerated with the bigger
   context. (A guard prevents a correct "I don't know" from being repaired into a
   fabricated answer.)
2. **Fallback.** If the answer still looks broken, answer directly from the retrieved
   chunks; if that fails too, re-retrieve from scratch and answer.
3. **Normalize.** Every "I couldn't find it" phrasing is collapsed into one canonical
   token, so the rest of the system can reliably detect a refusal.
4. **Anti-evasion.** An answer that just names a filename without extracting a value is
   treated as a refusal.

Each step targets a real, observed failure mode of LLM agents.

---

## Slide 14 — Design Principle: Deterministic Scaffolding Around the LLM

A recurring decision: **use deterministic code wherever a rule is reliable, and reserve
the LLM for genuinely ambiguous judgement.** This made the agent measurably more
reliable.

- **Tool routing is deterministic.** Instead of letting the LLM guess "search documents
  or query the spreadsheet?", `route_question` checks which document's summary best
  matches the question and routes by that document's type. Deterministic routing
  removed a class of wrong-tool errors.
- **Table-column reconstruction is deterministic.** Column names for repaired tables are
  derived by counting tokens, not asked of an LLM — no hallucinated headers.
- **Table repair is structural** — no LLM, so no hallucination risk on numeric data.

Takeaway line for the deck: *"The LLM does the reasoning; deterministic code does the
routing and the structure. Constraining the LLM to what it is actually good at is what
made the system reliable."*

---

## Slide 15 — Where the LLMs Are (and Which Models)

This system makes many model calls across several providers. Being explicit about this
is itself a talking point.

| Stage | File | Purpose | Model | Provider |
|---|---|---|---|---|
| Scanned-page OCR | `ingestion/ocr.py` | Image → markdown | `lightonocr-2-1b-ocr-soup` | Local vLLM (GPU) |
| Figure description | `ingestion/vlm.py` | Chart/diagram → text | `llama-4-scout-17b` | Groq |
| Chunk enrichment + doc summary | `chunker.py` | Context line per chunk | `gemma-4-31b` (free tier) | OpenRouter |
| Excel structure detection | `excel_cleaner.py` | Find table layout | `llama-3.3-70b-versatile` | Groq |
| LaTeX table repair (fallback) | `ingest.py` | Rebuild a table | `qwen/qwen3-32b` | Groq via LiteLLM |
| Main ReAct agent | `rag_agent.py` | Reasoning + tool calls | `qwen/qwen3-32b` | Groq via LiteLLM |
| HyDE expansion | `tools/retrieval_tool.py` | Hypothetical answer | `qwen/qwen3-32b` | Groq via LiteLLM |
| Coverage judge + repair + fallbacks | `rag_agent.py` | Verify / fix the answer | `qwen/qwen3-32b` | Groq via LiteLLM |
| Excel text-to-SQL agent | `tools/excel.py` | Write & format SQL | `gpt-4o-mini` | OpenAI |

Every row tagged "via LiteLLM" gets the failover chain from Slide 4 for free.

---

## Slide 16 — Known Limitations (be honest)

- **Many LLM calls per query.** A single question can trigger the ReAct agent loop,
  a HyDE call, a coverage-judge call, repair calls, and — for spreadsheets — a separate
  text-to-SQL sub-agent. This costs **latency and money**, and is the main scaling
  concern.
- **Multiple providers and models.** Groq, NVIDIA NIM, OpenRouter, and OpenAI are all in
  play. This means more API keys and a larger failure surface; the LiteLLM proxy
  mitigates provider outages but not the underlying complexity.
- **Reranker on CPU.** The cross-encoder runs on CPU — fine for a demo, a bottleneck
  under load.
- **OCR needs a GPU.** Scanned-document support requires the GPU OCR container; without
  it, only born-digital PDFs / Excel / Markdown work.
- **Quality machinery adds cost.** The verify-and-repair layer improves correctness but
  spends extra LLM calls on every answer judged incomplete.
- **Retrieval heuristics are eval-tuned.** The retrieval pipeline carries many tuned
  heuristics; they earn their scores but are not all independently proven minimal.
- **Evaluation uses LLM judges.** Correctness and faithfulness are scored by LLM judges,
  which are themselves imperfect — the eval numbers are directional, not exact.

---

## Slide 17 — Evaluation & Results

**Benchmark:** 82 questions across 9 question types, over 14 real mixed-format public
documents (born-digital PDFs, scanned PDFs, Excel, CSV). No eval-set-specific shortcuts —
every answer comes from the model and tool outputs. Correctness and relevancy carry
±2–3 points of run-to-run variance from inference nondeterminism.

**Answer quality** — judged by RAGAS-style LLM judges:

| Metric | Score | What it measures |
|---|---|---|
| Correctness | **80.4%** | Does the answer state the facts the question asks for, vs. the gold answer? Paraphrases, formatting, currency symbols and source labels are accepted; exact matches skip the judge. |
| Faithfulness | **89.3%** | Is every claim in the answer supported by — or inferable from — the retrieved context? Claim-level (RAGAS-style); cross-document conclusions count when their component facts are present; invented facts and wrong-source mixing are penalised. Excludes unanswerable questions. |
| Answer relevancy | **91.5%** | Does the answer actually address the question asked — not off-topic, not padded with irrelevant context? |

**Document retrieval** — 53 PDF/OCR questions, measured against Qdrant:

| Metric | Score | What it measures |
|---|---|---|
| Hit@5 | **100%** | Fraction of questions where a gold evidence chunk lands in the top 5 retrieved |
| Hit@10 | **100%** | Same, within the top 10 retrieved |
| MRR | **89.0%** | Mean reciprocal rank of the first gold evidence chunk (1.0 = always ranked first) |
| Evidence recall@10 | **96.5%** | Fraction of *all* annotated gold evidence chunks recovered within the top 10 |

**Structured retrieval** — 21 Excel/CSV questions, measured against DuckDB:

| Metric | Score | What it measures |
|---|---|---|
| Answer accuracy | **71.4%** | Fraction of spreadsheet questions where the text-to-SQL path returns the correct cell value |

**Refusal** — the unanswerable-question subset:

| Metric | Score | What it measures |
|---|---|---|
| Correct refusal rate | **100%** | Fraction of questions with no answer in the corpus where the agent returns `Unsupported` instead of hallucinating |

Retrieval is **split by modality** on purpose: PDF questions are scored by Qdrant vector
hit rate, Excel/CSV questions by DuckDB answer accuracy — mixing them would penalise the
SQL path for never appearing in Qdrant results. Unanswerable questions are excluded from
the retrieval and faithfulness metrics (a correct refusal makes no claim and retrieves
no context).

**What moved the scores** — the story is the engineering, not the framework: per-page
PDF routing, contextual chunking, HyDE expansion, two-stage retrieval, hybrid
dense+sparse search, the cross-encoder reranker, deterministic tool routing, and the
Excel summaries enriched with sample values. Failed experiments are logged so they are
not retried — disciplined iteration, not guesswork.

**The honest ceiling.** The remaining ~20% of correctness failures are *specific*, not
random: special-character entity names that defeat SQL `ILIKE` matching, LLM
column-disambiguation errors (right document, wrong column), wrong-cell-in-right-document
picks, and a few OCR variances on scanned PDFs.

*(For the deck: this is naturally two slides — answer-quality on one, retrieval +
refusal + the ceiling on the next.)*

---

## Notes for the deck builder

- **Placeholders to fill before presenting:** the two architecture diagrams (Slide 6,
  from `docs/diagrams.md`) and the verbatim prompts (Slide 12, from `src/prompts.py`).
- Use the two Mermaid diagrams in `docs/diagrams.md` — one for ingestion, one for query.
- Keep code references at the **file** level only; no line numbers.
- Suggested flow: Slides 1–6 set context (problem, stack, infra, LiteLLM, the
  agentic/multi-modal pitch, the architecture map); 7–8 ingestion; 9–13 the query side,
  the tools, the prompts, and refinement; 14–17 the honest engineering view (design
  principle, models, limitations, results).
- **Spend the most time on Slides 5, 7–8, 10–11** — the agentic/multi-modal framing and
  the ingestion + query pipelines and their tools are where the engineering depth is.
- The strongest narrative beats: (a) two modalities handled by purpose-built paths,
  (b) retrieval as an engineered pipeline not an API call, (c) the answer treated as a
  draft to be verified, (d) deterministic scaffolding making the LLM reliable,
  (e) LiteLLM giving provider resilience for free.
