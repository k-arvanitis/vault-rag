[![CI](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/k-arvanitis/vault-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-FF4136?style=for-the-badge&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)

# Vault RAG

Production-minded document intelligence for heterogeneous business document collections. Vault RAG handles the hard case most portfolio demos avoid: mixed formats (PDFs, scanned pages, spreadsheets, figures), mixed quality, and mixed retrieval needs — all queried through one retrieval stack via an operator console and a Slack delivery surface.

**Latest benchmark:** 56 questions over 8 real mixed-format public documents — **96.4% correctness**, **97.3% faithfulness**, **98.2% answer relevancy**, **98% Hit@10**.

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
make app               # → http://localhost:8501
```

GPU required for LightOn OCR (scanned PDFs). Born-digital PDFs, Excel, and Markdown work on CPU only.

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
make eval-cross   # cross-document benchmark
make eval         # full 56-question benchmark
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
 INGESTION
 ─────────────────────────────────────────────────────────────────

 Upload
   │
   ▼
 ┌──────────────────┐     ┌───────────────────────────────────┐
 │  File type       │────▶│  Parser                           │
 │  detection       │     │  Excel/CSV → openpyxl / pandas    │
 └──────────────────┘     │  Markdown  → plain text           │
                          │  PDF/DOCX  → per-page router ─────┼──┐
                          └───────────────────────────────────┘  │
                                                                  │
                          ┌──────────── PDF page ────────────┐   │
                          │                                  │◀──┘
                          │  text layer ≥ 50 chars?          │
                          │     YES → pymupdf4llm            │
                          │           (CPU, no API call)     │
                          │           figures → VLM (Groq)   │
                          │     NO  → LightOn OCR            │
                          │           (local vLLM, GPU)      │
                          └────────────────┬─────────────────┘
                                           │ Markdown (per page)
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Chunker                          │
                          │  1. Split on Markdown headers     │
                          │  2. Re-split chunks > 1024 tokens │
                          │  3. Merge tiny chunks < 256 tok   │
                          │  4. Contextual summary per chunk  │
                          └────────────────┬──────────────────┘
                                           │ chunks + context
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Embedder (local Ollama)          │
                          │  nomic-embed-text → dense vector  │
                          │  BM25 (fastembed) → sparse vector │
                          └────────────────┬──────────────────┘
                                           ▼
                                        Qdrant

 QUERY
 ─────────────────────────────────────────────────────────────────

 User question
   │
   ▼
 ┌──────────────────┐     ┌───────────────────────────────────┐
 │  HyDE expansion  │────▶│  Hybrid search (Qdrant)           │
 │  hypothetical    │     │  dense + sparse → RRF fusion      │
 │  answer, embed   │     │  top-100 candidates               │
 └──────────────────┘     └────────────────┬──────────────────┘
                                           ▼
                          ┌───────────────────────────────────┐
                          │  Cross-encoder reranker           │
                          │  ms-marco-MiniLM → top-10         │
                          └────────────────┬──────────────────┘
                                           ▼
                          ┌───────────────────────────────────┐
                          │  ReAct agent (LangGraph)          │
                          │  can call search tool N times     │
                          └────────────────┬──────────────────┘
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
| Vector database | Qdrant | Dense + sparse retrieval in one system with simple local Docker deployment |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | First-pass retrieval maximises recall; cross-encoder recovers precision before generation |
| Generation | llama-3.3-70b-versatile (Groq) | Strong answer synthesis and multi-step reasoning without a local high-end GPU |
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

56-question benchmark over 8 real public documents: procurement policies, legal contracts, government annual reports, scanned invoice packets, FOIA disclosures, and Excel/CSV spend reports. Four categories: single-document factoid, table lookup, cross-document comparison, and unanswerable.

### Results

| Category | Questions | Correctness |
|---|---|---|
| Single-doc factoid | 32 | Included in overall |
| Table lookup | 8 | Included in overall |
| Cross-document comparison | 10 | 90.0% |
| Unanswerable | 6 | 100% |
| **Overall** | **56** | **96.4%** |

| Answer metric | Score |
|---|---:|
| Correctness | 96.4% |
| Faithfulness | 97.3% |
| Answer relevancy | 98.2% |

| Retrieval metric | Score |
|---|---:|
| Hit@5 | 98% |
| Hit@10 | 98% |
| Evidence recall@10 | 96% |
| MRR | 0.92 |

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
