[![CI](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FF4136?style=for-the-badge&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)

# Vault RAG

Production-minded document intelligence for heterogeneous business document collections. Vault RAG handles the hard case most portfolio demos avoid: mixed formats (PDFs, scanned pages, spreadsheets, figures), mixed quality, and mixed retrieval needs — all queried through one retrieval stack via an operator console and a Slack delivery surface.

**Latest benchmark:** 82 questions over 14 real mixed-format public documents — **84.6% agent correctness**, **86.7% faithfulness** (answers grounded in retrieved text), **90.5% structured (DuckDB) accuracy**, **87.5% unanswerable refusal rate**.

Key engineering bets: per-page PDF routing (born-digital → pymupdf4llm, scanned → LightOn OCR), contextual retrieval (one-sentence summaries prepended before embedding), hybrid dense+sparse search with RRF fusion, cross-encoder reranking, and a LangGraph ReAct agent that can issue multiple searches before answering.

---

## Interfaces

Three surfaces, one retrieval backend:

| Interface | Who uses it | What it does |
|---|---|---|
| Streamlit | Admin / operator | Ingest documents, inspect parsed output, run eval, debug chunks |
| Slack | Team | Query the indexed corpus — @mention or DM, cited answers |
| CLI | Scripts / CI | Batch ingestion and headless query testing |

```bash
# Ingest
uv run python -m src.ingest --pdf data/input/report.pdf
uv run python -m src.ingest_table_rows data/input/tables.xlsx

# Query
uv run python -m src.rag_agent --query "What are the payment terms?"

# UI
make app        # Streamlit → http://localhost:8501
make slack      # Slack bot via Socket Mode
```

---

## Quickstart

```bash
git clone https://github.com/k-arvanitis/vault-rag.git && cd vault-rag
uv sync
cp .env.example .env   # set GROQ_API_KEY at minimum
make up                # Qdrant + OCR stack
ollama pull nomic-embed-text
make seed              # download and ingest starter documents (~3 files)
make app               # → http://localhost:8501
```

`make seed` downloads three public documents (two PDFs + one CSV) and ingests them so the UI is immediately queryable. GPU required for LightOn OCR (scanned PDFs). Born-digital PDFs, Excel, and Markdown work on CPU only.

---

## Docker deployment

```bash
cp .env.example .env
docker compose up -d --build   # or: make docker-up
```

Starts four services: Qdrant, LiteLLM proxy (Groq primary → OpenRouter fallback), Ollama (pulls nomic-embed-text on first start), and the Streamlit app at `http://localhost:8501`. First start takes a few minutes while Ollama downloads the model (~274 MB).

```bash
make docker-up-gpu   # adds LightOn OCR vLLM container — requires CUDA 12+ and NVIDIA runtime
```

> **Image size:** the `app` image is ~5 GB due to PyTorch CUDA libraries. Normal for ML workloads — the reranker runs on CPU regardless.

---

## Demo walkthrough

```bash
make up && ollama pull nomic-embed-text && make app
make eval         # full 82-question benchmark (14 docs, 9 question types)
```

Suggested flow in the Streamlit console:

1. **Chat** — ask a cross-document question
2. **Retrieved Chunks** — inspect the exact text/table snippets used in the answer
3. **Document Inspector** — compare the original page with parsed Markdown and chunk boundaries
4. **Eval Results** — gold vs generated answers, metrics, and retrieved evidence row by row

Sample questions:

```
A procurement policy and a services contract both include rules about extension periods.
Which allows longer, and what is each period?

In the two Doncaster Council spending documents, what are the amounts for the
Google Ads2372193163 row and the SS SYSTEMS LTD row?

What is the salary of the CEO of Doncaster School Solutions?
```

| Chat | Retrieved chunks |
|---|---|
| ![Chat UI](docs/screenshots/chat.png) | ![Retrieved chunks](docs/screenshots/retrieved_chunks.png) |

| Document inspector | Eval results |
|---|---|
| ![Document inspector](docs/screenshots/document_inspector.png) | ![Eval results](docs/screenshots/eval_results.png) |

---

## Architecture

```
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  INGESTION PIPELINE                                                          ║
 ╚══════════════════════════════════════════════════════════════════════════════╝

  File (PDF / Excel / CSV)
       │
       ▼
  ┌────────────────────┐
  │  src/ingest.py     │  File-type router. Calls parse_pdf() for PDFs,
  │  format detection  │  ingest_table_rows() for Excel/CSV.
  └────────┬───────────┘
           │
     ┌─────┴──────────────────────────────────┐
     │ PDF                                    │ Excel / CSV
     ▼                                        ▼
  ┌──────────────────────────────┐   ┌────────────────────────────────────┐
  │  src/parser/pdf_parser.py    │   │  src/ingest_table_rows.py          │
  │  Per-page router             │   │  openpyxl / pandas → rows          │
  │                              │   │                                    │
  │  page.get_text() < 50 chars? │   │  document_summary → Qdrant (always)│
  │                              │   │  If NOT in EXCEL_FILES:            │
  │  YES (scanned)  NO (digital) │   │    sheet_row, sheet_table → Qdrant │
  └────┬──────────────┬──────────┘   │  If in EXCEL_FILES (DuckDB path):  │
       │              │              │    rows → DuckDB only; Qdrant gets  │
       ▼              ▼              │    summary only (--only-summary)    │
  ┌─────────┐  ┌──────────────────┐  └────────────────────────────────────┘
  │ LightOn │  │  pymupdf4llm     │
  │ OCR     │  │  (CPU, no API)   │
  │ vLLM    │  │                  │
  │ GPU req │  │  Raster figures? │
  │         │  │  → src/ingestion/│
  │ OCR_    │  │    vlm.py        │
  │ API_BASE│  │  → Groq VLM API  │
  │ :8002   │  │  VLM_MODEL=      │
  └────┬────┘  │  llama-4-scout   │
       │       └────────┬─────────┘
       │                │
       └──────┬──────────┘
              │ Markdown text (per page)
              ▼
  ┌───────────────────────────────────────────────────────┐
  │  src/chunker.py  — 5-stage pipeline                   │
  │                                                       │
  │  1. Split on <!-- PAGE N --> markers (page boundaries)│
  │  2. Split on Markdown headers (MarkdownHeaderSplitter)│
  │  3. Re-split oversized chunks > CHUNK_MAX_TOKENS=1024 │
  │  4. Merge tiny chunks < CHUNK_MIN_TOKENS=256          │
  │  5. Contextual enrichment (LLM per chunk):            │
  │     → CHUNK_LLM_API_BASE (OpenRouter default)         │
  │     → CHUNK_LLM_MODEL (google/gemma-4-31b-it:free)    │
  │     → writes: "CONTEXT: <1 sentence>\n\nCONTENT: ..." │
  │  6. Document summary chunk (LLM, same model):         │
  │     → "Document ID: doc_XXX\nFile: ...\n<summary>"    │
  │     → chunk_type = "document_summary"                 │
  └───────────────────┬───────────────────────────────────┘
                      │ chunks (content + vector_text)
                      ▼
  ┌───────────────────────────────────────────────────────┐
  │  src/embedder.py + src/sparse_embedder.py             │
  │                                                       │
  │  Dense:  Ollama /api/embed                            │
  │          OLLAMA_EMBED_MODEL = nomic-embed-text        │
  │          768-dim cosine vectors                       │
  │                                                       │
  │  Sparse: fastembed BM25 (BAAI/bge-m3)                 │
  │          token frequency → sparse indices+values      │
  └───────────────────┬───────────────────────────────────┘
                      │ {dense_vector, sparse_vector, payload}
                      ▼
  ┌───────────────────────────────────────────────────────┐
  │  Qdrant  (collection: documents_chunks)               │
  │  Point ID = SHA-1(file_name + chunk_index)  ← idem-   │
  │  potent: re-ingest overwrites, no duplicates          │
  │  Payload: content, metadata.chunk_type, doc_id, etc.  │
  └───────────────────────────────────────────────────────┘


 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  QUERY PIPELINE                                                              ║
 ╚══════════════════════════════════════════════════════════════════════════════╝

  User question
       │
       ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  src/rag_agent.py — build_rag_agent() + ask_agent() / stream_agent()       │
  │  LangGraph ReAct loop (create_react_agent)                                 │
  │  LLM: GENERATION_MODEL via GENERATION_API_BASE (LiteLLM proxy → Groq)     │
  │  _build_doc_registry(): scrolls Qdrant document_summary chunks →           │
  │    builds {filename_stem: doc_id} map for fuzzy title → doc_id resolution  │
  └────────────────────────────────────────────────────────────────────────────┘
       │                                │
       │ doc NOT in DuckDB table list   │ doc IS in DuckDB table list
       │ (PDFs, Qdrant-only docs)       │ (EXCEL_FILES: doc_006,007,014)
       ▼                                ▼
  ┌──────────────────────┐    ┌─────────────────────────────┐
  │  search_knowledge_   │    │  query_excel tool           │
  │  base tool           │    │  src/excel_tool.py          │
  │  src/rag_agent.py    │    │  Agent writes SQL SELECT    │
  │  _make_unified_tool()│    │  DuckDB executes against    │
  └──────────┬───────────┘    │  cleaned sheet tables       │
             │                └─────────────────────────────┘
             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  HyDE  (src/rag_agent.py  _hyde())                       │
  │  LLM generates 2-3 sentence hypothetical answer          │
  │  → embed hypothetical instead of raw query               │
  │  Both raw + HyDE hit sets are merged (dedup by chunk key)│
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  src/retriever.py — retrieve()                           │
  │                                                          │
  │  1. infer_query_chunk_types() — keyword routing:         │
  │     "row/sheet/xlsx/table/amount/supplier…" terms        │
  │     → include pdf_table_rows + sheet_row chunks          │
  │     everything else → page_content + pdf_table_rows      │
  │                                                          │
  │  2. Qdrant hybrid search:                                │
  │     dense vector (nomic-embed) + sparse (BM25)           │
  │     fused with RRF (Reciprocal Rank Fusion)              │
  │     top_k = RETRIEVAL_TOP_K = 100 candidates             │
  │     optional: scope_doc_id filter, filter_token match    │
  └──────────────────────┬───────────────────────────────────┘
                         │ 100 candidates
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  src/reranker.py — BGEReranker.rerank()                  │
  │  Model: RERANKER_MODEL = BAAI/bge-reranker-v2-m3         │
  │  Architecture: AutoModelForSequenceClassification        │
  │  (cross-encoder — query+chunk scored together)           │
  │  Input: pairs of [query, chunk_text]                     │
  │  Output: logit scores → sorted top RERANK_TOP_N = 10     │
  └──────────────────────┬───────────────────────────────────┘
                         │ top-10 reranked chunks
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Context formatting  (src/rag_agent.py _best_snippet())  │
  │  Chunks truncated to MAX_CHUNK_CHARS=1500                │
  │  Tables truncated to MAX_TABLE_CHARS=3000                │
  │  Table rows reformatted as "Field: Value" key-value      │
  │  Each chunk prefixed: [N] file=<name> chunk=<idx>        │
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Generation LLM  (GENERATION_API_BASE → LiteLLM → Groq)  │
  │  Model: GENERATION_MODEL (default: qwen3-32b)            │
  │  Synthesises cited answer from retrieved chunks          │
  │  Post-processing:                                        │
  │  → _normalize_unsupported(): hedging → "Unsupported"     │
  │  → _repair_deterministic_numeric_answer(): regex fixes   │
  │  → _repair_incomplete_answer(): coverage check + retry   │
  └──────────────────────────────────────────────────────────┘
                         │
                         ▼
                   Cited answer
```

---

## Key engineering decisions

### Per-page PDF routing

Every PDF page is routed independently — not the whole document. Born-digital pages (≥50 chars of extractable text) go through pymupdf4llm on CPU with zero API calls. Scanned pages have no usable text layer and are sent to a local LightOn OCR vLLM server. Mixed documents — a contract where pages 1–10 are digital and page 11 is a faxed addendum — are handled correctly per page with no configuration.

Running OCR on born-digital pages is counterproductive: the vision model transcribes what it *sees* in a rendered image, introducing errors on numbers, equations, and code that are already perfect as text. The 50-char threshold keeps both paths completely separate.

### Contextual Retrieval

For each chunk, a fast LLM writes one sentence describing the topic, entities, and purpose of that specific chunk. This sentence is prepended before embedding:

```
CONTEXT: This chunk describes payment terms under Section 4.2 of the supplier agreement.

CONTENT:
<original chunk text>
```

Each vector in the index captures both what the chunk is *about* and what it *says* — improving recall for short or indirect queries. Implementation of [Anthropic Contextual Retrieval (2024)](https://www.anthropic.com/news/contextual-retrieval).

### Three-pass deterministic table repair

Research papers and GHG inventory reports contain tables that OCR cannot handle correctly. Three specific failure modes are fixed without any LLM involvement:

1. **Two-row HTML headers** — OCR places sub-column names (`R@1 R@5 R@10`) in the first tbody row instead of the thead. Detected structurally; flattened to `KaLMv2 R@1` programmatically.
2. **LaTeX `$\text{...}$` tables** — arXiv papers embed raw LaTeX in OCR output. Header tokens are parsed and a correctly-aligned Markdown table is rebuilt.
3. **Missing rightmost column** — wide tables frequently lose the last column in vision-language OCR. `pypdf` falls back to the PDF text layer and recovers values positionally, including splitting concatenated decimals (`2.9524.141` → `2.952`, `4.141`).

All three repairs are structural — no hallucination risk and alignment is guaranteed by counting data-row tokens.

### Deterministic point IDs (idempotent re-ingestion)

Every Qdrant point is assigned a SHA-1 hash of `(file_name, chunk_index)` rather than a random UUID. Re-ingesting the same file overwrites vectors in place instead of appending duplicates. With random IDs, re-running ingestion silently doubles every chunk in the index — a subtle bug that degrades retrieval precision without any visible error.

### HyDE query expansion

The agent first generates a *hypothetical answer* using a fast LLM call and embeds that instead of the raw question. The embedded hypothetical is semantically much closer to real document text, improving recall for vague queries like "what were the sales figures" where the question shares few tokens with the answer.

### Dual-modality retrieval: vector search + DuckDB SQL

Vector search works well for prose, policies, and narrative reports — the kind of content where semantic similarity reliably surfaces the right passage. It breaks down for spreadsheet data.

When a user asks *"how much did supplier X pay in March?"* or *"what is the total spend for category Y?"*, the correct answer requires exact string matching, numeric filtering, and aggregation (`SUM`, `GROUP BY`). A dense vector for "how much did supplier X pay" is semantically similar to every row in a payment sheet — there is no useful distance signal. The right chunk is determined by column equality and arithmetic, not by proximity in embedding space.

To handle both, the system runs two independent retrieval paths:

- **Qdrant** for PDF and OCR documents — hybrid dense+sparse search, reranked by a cross-encoder.
- **DuckDB** for flat-structure Excel and CSV files — the agent writes a `SELECT` query; the in-process database executes it and returns an exact result.

The agent selects the path based on whether the target document appears in the DuckDB table list injected into its system prompt. PDF-format documents always go to Qdrant even when they contain numeric tables, because vector search on contextualised chunk summaries still outperforms SQL over unstructured OCR output.

DuckDB specifically — rather than Postgres or SQLite — because it is in-process (no server to run, no connection pool), columnar (aggregations over 10k-row CSV files return in milliseconds), and loads directly from a pandas DataFrame. The entire structured store is a single file at `DUCKDB_PATH`.

### Context-overflow retry

When the LLM returns a `400 Bad Request` due to token overflow, the agent retries with `rerank_top_n` halved. The reduction persists only for that single request. A shorter but valid answer is returned instead of a crash, without permanently setting a conservative chunk count.

---

## Tech stack

| Component | Technology | Why |
|---|---|---|
| PDF — born-digital | pymupdf4llm | Reading the existing text layer is faster and more faithful than OCR, especially for numbers, tables, and equations |
| PDF — scanned | LightOn OCR (local vLLM) | Scanned pages have no usable text layer; running OCR locally preserves privacy |
| Figure descriptions | llama-4-scout-17b (Groq) | Turns charts and diagrams into searchable text so evidence inside figures is retrievable |
| Contextual summaries | llama-3.1-8b-instant (Groq) | Fast, low-cost model adds chunk-level context at ingest without adding query latency |
| Embeddings | nomic-embed-text (Ollama) | Strong local embedding model — indexing stays on-prem with no external API per chunk |
| Structured data store | DuckDB | In-process analytical database — zero ops (no server, just a file), columnar storage makes aggregations (SUM, GROUP BY, AVG) over large spreadsheets fast. Postgres would add a running server, connection pooling, and migrations for a use case that is read-only analytics, not transactions. |
| Vector database | Qdrant | Dense + sparse retrieval in one system with simple local Docker deployment |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder: scores query and chunk *together* in one forward pass, so it models their interaction directly. A bi-encoder scores them independently then compares embeddings — sharply less accurate when the relevance signal lives in the relationship between question and passage, not either alone. bge-reranker-v2-m3 is multilingual and trained on diverse document types vs the English-only MS MARCO corpus of smaller alternatives |
| Generation | qwen3-32b (Groq) | Strong answer synthesis and multi-step reasoning without a local high-end GPU |
| Agent | LangGraph (ReAct) | Iterative retrieval and query reformulation when a single retrieve-then-generate pass is insufficient |
| UI | Streamlit | Python-native — faster to build and inspect than a custom frontend for an operator-heavy workflow |
| Observability | Langfuse | End-to-end traces make it possible to inspect prompts, tool calls, retrieved chunks, and token usage |

---

## Privacy & data

- **Parsing** runs locally — raw document bytes never leave the machine.
- **Embeddings** are generated by Ollama — document text never leaves the machine at indexing time.
- **Contextual summaries** send chunk text to Groq at ingest time. To avoid this, point `CHUNK_LLM_API_BASE` at a local vLLM server.
- **Query answering** sends only retrieved chunks and the user query to Groq. To go fully air-gapped, point `GENERATION_API_BASE` at a local vLLM server.

---

## Evaluation

82-question benchmark over 14 real public documents: procurement policies, legal contracts, government annual reports, scanned invoice packets, FOIA disclosures, Excel/CSV spend reports, HR handbooks, and open-data maturity datasets. Nine question types: OCR extraction, table lookup, numeric lookup, figure grounding, table grounding, negation check, cross-document comparison, single-doc factoid, and unanswerable.

### Benchmark corpus

All 14 documents are publicly available. `make seed` automatically downloads and ingests a representative starter subset (doc_001, doc_002, doc_007) so the system is immediately queryable after setup — no manual downloads required.

| Doc | Title | Type | Format | Source |
|---|---|---|---|---|
| doc_001 ★ | Policy for the Procurement of Goods and Services (PGS) | Policy | PDF (born-digital) | [lacera.gov](https://www.lacera.gov/sites/default/files/assets/documents/board/Governing%20Documents/General%20Policies/Purchasing_Policy_Goods_Services.pdf) |
| doc_002 ★ | Appendix C – Terms and Conditions of Contract for Services | Contract | PDF (born-digital) | [publishing.service.gov.uk](https://assets.publishing.service.gov.uk/media/5abcfd7fed915d44eb7e6969/Terms_and_Conditions_for_Services.pdf) |
| doc_003 | 111th Annual Report of the Board of Governors of the Federal Reserve System, 2024 | Annual report | PDF (born-digital) | [federalreserve.gov](https://www.federalreserve.gov/publications/files/2024-annual-report.pdf) |
| doc_004 | Marie Campbell FOIA Complete – Portable & Dumpster Rentals | FOIA / invoices | PDF (scanned) | [bensenville.gov](https://www.bensenville.gov/DocumentCenter/View/20216/17021_Marie_Campbell_FOIA_Complete) |
| doc_005 | Other Pertinent Forms and Reports – Fueling Records | Invoice | PDF (scanned) | [ntsb.gov](https://data.ntsb.gov/Docket/Document/docBLOB?FileExtension=.PDF&FileName=Other+Pertinent+Forms+and+Reports+%28fueling+records%29-Master.PDF&ID=40393413) |
| doc_006 | Purchase Card Transactions Qtr1 2025-26 | Spend table | Excel | [doncaster.gov.uk](https://www.doncaster.gov.uk/services/the-council-democracy/payments-to-suppliers-reports-2025-26) |
| doc_007 ★ | Published Spend Report April 25 | Spend table | CSV | [doncaster.gov.uk](https://www.doncaster.gov.uk/services/the-council-democracy/payments-to-suppliers-reports-2025-26) |
| doc_008 | 2024 Annual Report: Additional Opportunities to Reduce Fragmentation, Overlap, and Duplication | Government report | PDF (born-digital) | [gao.gov](https://www.gao.gov/products/gao-24-106915) |
| doc_009 | Human Resources Policy Manual 2024 | HR policy | PDF (born-digital) | [united-church.ca](https://united-church.ca/sites/default/files/2021-04/hr-policy-manual.pdf) |
| doc_010 | Employee Handbook | Handbook | PDF (born-digital) | [rosemont.com](https://bd-rosemont-images.s3.amazonaws.com/wp-content/uploads/2024/08/15153613/EmployeeHandbook-8.15.24-1.pdf) |
| doc_011 | 2025 Open Data Maturity Questionnaire – Spain | Dataset | Excel | [data.europa.eu](https://data.europa.eu/sites/default/files/2025-12/2025_odm_questionnaire_spain_0.xlsx) |
| doc_012 | OSSE AFE Quarterly and Year-End Reporting Workbook FY2025 | Finance report | Excel | [osse.dc.gov](https://osse.dc.gov/sites/default/files/dc/sites/osse/service_content/attachments/8.%20SAMPLE%20FY25%20OSSE%20AFE%20QUARTERLY%20%26%20YEAR-END%20REPORTING%20WORKBOOK.xlsx) |
| doc_013 | FY26 OSSE AFE Grant Budget & Finance Tracker Workbook | Budget tracker | Excel | [osse.dc.gov](https://osse.dc.gov/sites/default/files/dc/sites/osse/service_content/attachments/4.%20REVISED_FY26%20OSSE%20AFE%20Grant%20Budget%20%26%20Finance%20Tracker%20Workbook%20%24510K_Rev%20Match%20Tab%2015A.5.14.25.xlsx) |
| doc_014 | Supplier Spend Over £500 – April 2024 | Spend table | CSV | [bristol.gov.uk](https://www.bristol.gov.uk/files/documents/8042-supplier-spend-apr-2024) |

★ Downloaded automatically by `make seed`.

### Results

**Agent answer metrics** (all 82 questions)

| Metric | Score | Notes |
|---|---:|---|
| Correctness | **84.6%** | LLM judge + exact-match short-circuit |
| Faithfulness | **86.7%** | Answer claims grounded in retrieved text |
| Answer relevancy | **87.8%** | Answer directly addresses the question |

**Vector retrieval metrics** (53 PDF/OCR questions, Qdrant)

| Metric | Score |
|---|---:|
| Hit@5 | **84.9%** |
| Hit@10 | **92.5%** |
| MRR | **62.3%** |
| Evidence recall@10 | **87.1%** |

Hit@10 at 92.5% means the correct evidence chunk lands in the top 10 candidates for nearly every answerable PDF question — with no domain-specific fine-tuning, purely from hybrid dense+sparse retrieval and contextual chunk summaries. The gap between MRR (62.3%) and Hit@10 (92.5%) reflects cross-document and multi-hop questions where relevant evidence is split across chunks; the cross-encoder reranker recovers most of it by position 10.

**Structured retrieval** (21 Excel/CSV questions, DuckDB)

| Metric | Score |
|---|---:|
| Answer accuracy | **90.5%** |

Excel and CSV questions bypass Qdrant entirely. The agent detects that the target document is loaded in DuckDB and writes a SQL SELECT; the result is returned directly. Measuring these against vector hit rate would be a category error — a structured lookup that returns the exact row has 100% retrieval success by definition.

**Unanswerable questions** (8 questions)

| Metric | Score |
|---|---:|
| Correct refusal rate | **87.5%** |

Questions that cannot be answered from the indexed corpus. The agent is instructed to return the single word `Unsupported` — no hedging, no hallucination. One question received a valid natural-language refusal instead of the bare token, which the exact-match evaluator scores as a miss; semantically the behaviour was correct.

### Methodology notes

- **Faithfulness exceeds correctness.** 86.7% faithfulness vs 84.6% correctness means that when the system gives a wrong answer, the mistake is almost always a retrieval miss (the right chunk was not found) rather than a hallucination over retrieved text. This is the failure mode you want in a production RAG system — missed recall is recoverable by improving retrieval; hallucination is not.
- **Correctness ceiling.** The remaining ~15% failures split into three categories: (1) multi-hop cross-document questions where evidence is split across 100+ page documents and the reranker doesn't surface both chunks together; (2) questions that require arithmetic the agent is explicitly instructed to refuse (by design — a RAG system should not invent calculations); (3) LLM non-determinism on a small number of numeric lookups where the correct row is retrieved but the wrong value is extracted.
- **Retrieval metrics are split by modality.** PDF questions are measured by Qdrant vector hit rate. Excel/CSV questions are measured by DuckDB answer accuracy. Mixing them would penalise the SQL path for never appearing in Qdrant results.
- **Unanswerable questions are excluded from retrieval and faithfulness metrics.** A correct refusal makes no factual claim and retrieves no context — scoring faithfulness against empty evidence would be meaningless.

Full methodology and reproduction steps: [eval/README.md](eval/README.md).

---

## Setup

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| uv | 0.4+ | Dependency management |
| Docker + Compose | 24+ | Qdrant + optional OCR server |
| Ollama | 0.4+ | Local embedding model |
| GPU (optional) | CUDA 12+ | Required for LightOn OCR |

### Configuration

All variables live in `.env`. See `.env.example` for a full annotated list. The only required variable is `GROQ_API_KEY` (free tier at [console.groq.com](https://console.groq.com)).

Key overrides:

| Variable | Default | Notes |
|---|---|---|
| `GENERATION_MODEL` | `qwen3-32b` | Main answering LLM |
| `CHUNK_LLM_MODEL` | `google/gemma-4-31b-it:free` | Contextual summary model (ingest only) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Local embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `VLM_ENABLED` | `true` | Set `false` to skip figure description calls |

### Slack

Create a Slack app with Socket Mode, add the `app_mention` and `message.im` event subscriptions, and add `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` to `.env`. Full walkthrough: [docs/SLACK_SETUP.md](docs/SLACK_SETUP.md).

---

## Tests

126 tests, all mocked — no live services required to run CI.

| File | Tests | What's covered |
|---|---|---|
| `test_chunker.py` | 3 | Token limits, minimum chunk size, output fields |
| `test_config.py` | 5 | Config invariants: types, model family constraints |
| `test_retriever.py` | 17 | Cosine similarity, text filter, dense and hybrid Qdrant paths |
| `test_reranker.py` | 10 | BGEReranker and QwenReranker: output format, sort order, score ranges |
| `test_rag_agent.py` | 52 | History injection, token streaming, `<think>` suppression, cross-document repair |
| `test_table_repair.py` | 13 | Two-row HTML headers, LaTeX column derivation, missing-column recovery |
| `test_pdf_parser.py` | 6 | Per-page routing, VLM enable/disable, exception handling |
| `test_excel_cleaner.py` | 4 | Sheet filtering, row-value skipping, no-data token handling |
| `test_slack.py` | 16 | Message routing, DM handling, mention parsing, answer formatting |

```bash
uv run pytest tests/ -v
```

---

## Project structure

```text
vault-rag/
├── app.py                     # Streamlit operator console
├── slack_app.py               # Slack bot (Socket Mode)
├── litellm_config.yaml        # LiteLLM proxy config
├── src/
│   ├── config.py              # All env vars in one place
│   ├── ingest.py              # File-type router → parser → chunker → Qdrant
│   ├── ingest_table_rows.py   # Row-batched ingestion for Excel / CSV
│   ├── chunker.py             # 5-stage chunking pipeline
│   ├── embedder.py            # Dense embedding (Ollama)
│   ├── sparse_embedder.py     # Sparse embedding (BM25 / fastembed)
│   ├── retriever.py           # Hybrid search, HyDE, RRF fusion
│   ├── reranker.py            # Cross-encoder reranker
│   ├── table_processor.py     # Three-pass deterministic table repair
│   ├── rag_agent.py           # LangGraph ReAct agent + streaming
│   ├── vector_store.py        # Qdrant upsert / scroll helpers
│   ├── excel_tool.py          # Agent tool for structured Excel queries
│   ├── parser/
│   │   ├── pdf_parser.py      # Per-page router (born-digital vs scanned)
│   │   └── lightonocr_parser.py
│   ├── ingestion/
│   │   ├── ocr.py             # LightOn OCR (local vLLM)
│   │   └── vlm.py             # VLM figure descriptions (Groq)
│   └── preprocessing/
│       ├── excel_cleaner.py   # Sheet normalisation and header repair
│       └── chunk_builder.py   # SheetResult → chunk dicts
├── eval/
│   ├── README.md              # Benchmark design and reproduction steps
│   ├── run_eval.py            # Evaluation runner
│   └── data/ results/
├── docs/
│   ├── CASE_STUDY.md
│   ├── SLACK_SETUP.md
│   └── screenshots/
└── docker/
    └── ingestion-stack/
```

---

## Failure modes

| Component | What fails | Symptom | Fix |
|---|---|---|---|
| LightOn OCR | vLLM server not running | Scanned pages fail; born-digital pages unaffected | `make up` |
| VLM (figures) | Groq unavailable | Figures replaced with `[Figure: description unavailable]`; ingestion continues | Set `VLM_ENABLED=false` to opt out |
| Qdrant | Container not running | All queries return empty | `docker compose up -d qdrant` |
| Ollama | Model not pulled | Embedding step fails | `ollama pull nomic-embed-text` |
| Groq API | Missing key | Generation returns 401 | Set `GROQ_API_KEY` in `.env` |
| Reranker | First run | First query slow (~30s, downloads model weights) | Pre-download at startup (Dockerfile does this) |
