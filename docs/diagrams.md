# Vault RAG — Architecture Diagrams

Mermaid source for the two pipelines. Renders on GitHub, in most markdown
viewers, and at [mermaid.live](https://mermaid.live).
Companion: [ARCHITECTURE.md](ARCHITECTURE.md) (narrative).

---

## 1. Ingestion pipeline

```mermaid
flowchart TD
    F["File: PDF / Excel / CSV"] --> R{File-type router}

    %% ---- PDF branch ----
    R -->|PDF| PP{Per-page router<br/>text layer >= 50 chars?}
    PP -->|Yes — born-digital| BD["Markdown extractor<br/>local, CPU, no model"]
    PP -->|No — scanned| OCR["OCR model<br/>local vLLM, GPU"]
    BD --> FIG{"Figures on page?<br/>(raster / vector)"}
    FIG -->|Yes| VLM["Vision-language model<br/>figure description"]
    FIG -->|No| CH
    VLM --> CH
    OCR --> CH
    CH["Chunker<br/>page split → section split →<br/>re-split large → merge small →<br/>contextual enrichment →<br/>+ document summary"]

    %% ---- Excel / CSV branch ----
    R -->|Excel / CSV| SCH["LLM schema extraction<br/>column names, data-start row,<br/>footnote rows"]
    SCH --> ROWS["Cleaned rows"]
    SCH --> SUM["document_summary +<br/>sheet_summary chunks"]
    ROWS --> DD[("DuckDB<br/>one table per sheet")]

    %% ---- Shared embedding sink ----
    CH --> EMB["Dual embedding<br/>dense (semantic) + sparse (exact tokens)"]
    SUM --> EMB
    EMB --> QD[("Qdrant<br/>hybrid collection<br/>text chunks + summaries")]
```

---

## 2. Query pipeline

```mermaid
flowchart TD
    Q["User question"] --> MP{Multi-part?}
    MP -->|Yes| SPLIT["Split into independent sub-questions"]
    MP -->|No| ROUTE
    SPLIT --> ROUTE

    ROUTE["Deterministic tool routing<br/>match question vs document summaries →<br/>resolve spreadsheet vs text document"]
    ROUTE --> AGENT

    AGENT{"ReAct agent<br/>tool-calling loop"}

    AGENT -->|text document| SKB["search_knowledge_base"]
    SKB --> S1["Stage 1 — doc routing<br/>search summaries → resolve which docs"]
    S1 --> S2["Stage 2 — scoped hybrid search<br/>dense + sparse, RRF-fused +<br/>HyDE expansion + cross-encoder rerank +<br/>neighbour-chunk expansion"]
    S2 --> QD[("Qdrant")]
    QD --> AGENT

    AGENT -->|spreadsheet| QE["query_excel — Excel sub-agent"]
    QE --> SELT["Select table"]
    SELT --> INSP["Inspect schema + sample rows"]
    INSP --> WSQL["Write SQL"]
    WSQL --> RSQL["Run SQL"]
    RSQL --> EVAL{"Evaluate<br/>error / 0 rows?"}
    EVAL -->|retry| WSQL
    EVAL -->|ok| DD[("DuckDB")]
    DD --> AGENT

    AGENT --> POST["Post-processing<br/>multi-part coverage check →<br/>forced retry on bare Unsupported →<br/>strip leaked chunk headers"]
    POST --> ANS["Cited answer<br/>chunk + page (PDF) ·<br/>sheet + SQL trace (Excel/CSV)"]
```
