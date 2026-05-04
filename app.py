"""Vault RAG — Streamlit chat UI.

Features:
- Upload PDFs, Excel, CSV, Word, Markdown, or image files → ingested into Qdrant
- Chat with your documents (ReAct agent, multi-step retrieval)
- Conversation history per session
- Document library: see what's been ingested, delete individual files
- Document inspector: original PDF side-by-side with parsed markdown
"""
from __future__ import annotations

import re
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "data" / "input"
TRANSLATED_DIR = REPO_ROOT / "data" / "output" / "processed"
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"
EVAL_SUMMARY_PATH = EVAL_RESULTS_DIR / "summary.json"
EVAL_ANSWERS_PATH = EVAL_RESULTS_DIR / "answer_results.jsonl"
EVAL_RETRIEVAL_PATH = EVAL_RESULTS_DIR / "retrieval_results.jsonl"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DIRS = (
    TRANSLATED_DIR,
    REPO_ROOT / "data" / "output" / "lightonocr",
    REPO_ROOT / "data" / "output" / "pymupdf",
    REPO_ROOT / "data" / "output" / "translated",
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vault RAG",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── lazy imports ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG agent…")
def _get_agent():
    """Build and cache the RAG agent using current config values."""
    from src.config import (
        GENERATION_API_BASE,
        GENERATION_MODEL,
        QDRANT_COLLECTION,
        QDRANT_URL,
        RERANK_TOP_N,
        RERANKER_MODEL,
        RETRIEVAL_TOP_K,
    )
    from src.rag_agent import build_rag_agent

    return build_rag_agent(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        reranker_model_name=RERANKER_MODEL or None,
        model_name=GENERATION_MODEL,
        generation_api_base=GENERATION_API_BASE,
    )


def _ingest_file(uploaded_file) -> str:
    """Save upload to data/input/ and run the appropriate ingestion pipeline."""
    suffix = Path(uploaded_file.name).suffix.lower()
    dest = INPUT_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())

    from src.config import QDRANT_COLLECTION

    if suffix in {".xlsx", ".xls", ".csv"}:
        from src.ingest_table_rows import ingest_table_rows
        ingest_table_rows(str(dest), collection=QDRANT_COLLECTION)
    elif suffix in {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".png", ".jpg", ".jpeg"}:
        # docx support: extend here if run_ingest does not yet handle .docx natively
        from src.ingest import run_ingest
        run_ingest(pdf_path=dest, collection=QDRANT_COLLECTION)
    else:
        from src.ingest import run_ingest
        run_ingest(pdf_path=dest, collection=QDRANT_COLLECTION)

    return uploaded_file.name


def _list_ingested_files() -> list[str]:
    """Return sorted list of unique source filenames currently in the vector store."""
    from src.config import QDRANT_COLLECTION, QDRANT_URL
    from src.vector_store import scroll_all_payloads
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return []
    seen: set[str] = set()
    for p in payloads:
        meta = p.get("metadata") or {}
        fn = meta.get("source_file") or meta.get("file_name", "")
        if fn:
            seen.add(fn)
    return sorted(seen)


def _delete_file(file_name: str) -> None:
    """Delete all vector store points associated with the given source filename."""
    from src.config import QDRANT_COLLECTION, QDRANT_URL
    from src.vector_store import delete_by_file
    delete_by_file(QDRANT_URL, QDRANT_COLLECTION, file_name)


def _resolve_original_path(source: str) -> Path:
    """Resolve an ingested source path to a local file if it is still present."""
    source_path = Path(source)
    candidates = []
    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append(REPO_ROOT / source_path)
    candidates.append(INPUT_DIR / source_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _resolve_markdown_path(source: str) -> Path:
    """Resolve parsed markdown for a PDF across the parser output directories."""
    stem = Path(source).stem
    candidates = [directory / f"{stem}.md" for directory in MARKDOWN_DIRS]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _split_markdown_by_page(md_text: str) -> tuple[dict[int, str], dict[int, str]]:
    """Split markdown into page content and pipeline labels using <!-- PAGE N --> markers.

    Supports both legacy <!-- PAGE N --> and new <!-- PAGE N | label --> formats.

    Returns:
        Tuple of (pages, pipelines) where each is a dict keyed by 1-based page number.
    """
    import re
    # Capture optional "| label" suffix
    parts = re.split(r"<!--\s*PAGE\s+(\d+)(?:\s*\|([^-]*))?\s*-->", md_text)
    pages: dict[int, str] = {}
    pipelines: dict[int, str] = {}
    # parts = [pre_text, "1", label_or_None, content1, "2", label_or_None, content2, ...]
    i = 1
    while i < len(parts) - 2:
        page_num = int(parts[i])
        label = (parts[i + 1] or "").strip()
        content = parts[i + 2].strip()
        pages[page_num] = content
        if label:
            pipelines[page_num] = label
        i += 3
    return pages, pipelines


@st.cache_data(ttl=10)
def _load_eval_summary() -> dict:
    """Load the latest evaluation summary if present."""
    if not EVAL_SUMMARY_PATH.exists():
        return {}
    try:
        return json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(ttl=10)
def _load_eval_answers() -> list[dict]:
    """Load latest per-question evaluation rows if present."""
    if not EVAL_ANSWERS_PATH.exists():
        return []
    rows: list[dict] = []
    for line in EVAL_ANSWERS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@st.cache_data(ttl=10)
def _load_eval_retrieval() -> dict[str, dict]:
    """Load latest per-question retrieval rows keyed by qa_id."""
    if not EVAL_RETRIEVAL_PATH.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in EVAL_RETRIEVAL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qa_id = row.get("qa_id")
        if qa_id:
            rows[qa_id] = row
    return rows


def _fmt_metric(value: object) -> str:
    """Format a 0-1 metric as a percentage."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _hit_label(hit: dict, index: int) -> str:
    """Build a compact label for a retrieved eval hit."""
    doc_id = hit.get("doc_id") or "unknown"
    chunk_type = hit.get("chunk_type") or "chunk"
    sheet = hit.get("sheet_name")
    row_ref = hit.get("row_ref")
    location = f"sheet={sheet}" if sheet else ""
    if row_ref is not None:
        location = f"{location} row={row_ref}".strip()
    score = hit.get("score")
    score_text = f" · score={float(score):.3f}" if isinstance(score, int | float) else ""
    return f"{index}. {doc_id} · {chunk_type}{(' · ' + location) if location else ''}{score_text}"


def _hit_is_table(hit: dict) -> bool:
    """Return True for retrieved spreadsheet/table evidence."""
    chunk_type = str(hit.get("chunk_type") or "")
    content = str(hit.get("content") or "")
    return bool(hit.get("sheet_name")) or "sheet_" in chunk_type or "[File:" in content


# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []  # list of chunk strings from last query


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗄️ Vault RAG")
    st.caption("Business document Q&A — hybrid search · HyDE · reranking")
    st.divider()

    st.subheader("Add documents")
    uploaded = st.file_uploader(
        "Upload PDF, Excel, CSV, Markdown, Word, or images",
        type=["pdf", "xlsx", "xls", "csv", "md", "docx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for uf in uploaded:
            if uf.name not in st.session_state.ingested:
                with st.spinner(f"Ingesting {uf.name}…"):
                    try:
                        _ingest_file(uf)
                        st.session_state.ingested.append(uf.name)
                        st.success(f"✓ {uf.name}")
                    except Exception as exc:
                        st.error(f"✗ {uf.name}: {exc}")

    st.divider()

    st.subheader("Document library")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()

    @st.cache_data(ttl=30)
    def _cached_files():
        """Cached wrapper around _list_ingested_files with 30-second TTL."""
        return _list_ingested_files()

    files = _cached_files()
    if not files:
        st.caption("No documents ingested yet.")
    else:
        for fn in files:
            col1, col2 = st.columns([4, 1])
            col1.caption(fn)
            if col2.button("🗑", key=f"del_{fn}", help=f"Remove {fn}"):
                with st.spinner(f"Removing {fn}…"):
                    _delete_file(fn)
                st.cache_data.clear()
                st.rerun()

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_chunks, tab_inspect, tab_eval = st.tabs([
    "💬 Chat",
    "🔎 Retrieved Chunks",
    "🔍 Document Inspector",
    "📊 Eval Results",
])

# ── chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # History = all turns before the current user message
        history = st.session_state.messages[:-1]

        from src.rag_agent import stream_agent
        chunks: list[str] = []
        try:
            agent = _get_agent()
            with st.chat_message("assistant"):
                answer = st.write_stream(
                    stream_agent(agent, prompt, history=history, collected_chunks=chunks)
                )
        except Exception as exc:
            answer = f"Error: {exc}"
            with st.chat_message("assistant"):
                st.error(answer)

        st.session_state.last_chunks = chunks
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ── chunks tab ───────────────────────────────────────────────────────────────
with tab_chunks:
    if not st.session_state.last_chunks:
        st.info("Ask a question in the Chat tab to see the retrieved chunks here.")
    else:
        st.caption(f"{len(st.session_state.last_chunks)} chunks retrieved")
        for i, chunk in enumerate(st.session_state.last_chunks, start=1):
            with st.expander(f"Chunk {i} — {chunk[:80]}…", expanded=i == 1):
                st.markdown(chunk)


# ── inspector tab ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _load_file_chunks(source_file: str) -> list[dict]:
    """Load all Qdrant payloads for a given source file (cached 60 s)."""
    from src.config import QDRANT_COLLECTION, QDRANT_URL
    from src.vector_store import get_chunks_by_file
    try:
        return get_chunks_by_file(QDRANT_URL, QDRANT_COLLECTION, source_file)
    except Exception:
        return []


with tab_inspect:
    ingested_files = _cached_files()

    if not ingested_files:
        st.info("No documents ingested yet. Upload a file via the sidebar.")
    else:
        selected = st.selectbox(
            "Select a document",
            options=ingested_files,
            label_visibility="collapsed",
        )
        suffix = Path(selected).suffix.lower()

        # ── PDF inspector ──────────────────────────────────────────────────────
        if suffix == ".pdf":
            pdf_path = _resolve_original_path(selected)
            pdf_name = pdf_path.name
            md_path = _resolve_markdown_path(selected)

            from src.config import QDRANT_COLLECTION, QDRANT_URL
            from src.vector_store import get_document_summary
            md_stem = Path(pdf_name).stem + ".md"
            summary = get_document_summary(QDRANT_URL, QDRANT_COLLECTION, md_stem)
            if summary:
                with st.expander("Document Summary", expanded=True):
                    st.markdown(summary.replace("## Document Summary\n\n", ""))

            pdf_available = pdf_path.exists()
            md_available = md_path.exists()

            if not pdf_available:
                st.warning("Original PDF not available locally — showing parsed markdown only.")
            if not md_available:
                st.info("Markdown not found. Re-ingest this PDF to generate it.")

            if md_available:
                import pypdfium2 as pdfium
                md_text = re.sub(r"\[TABLE_START\]\n?|\[TABLE_END\]\n?", "", md_path.read_text(encoding="utf-8"))
                page_sections, page_pipelines = _split_markdown_by_page(md_text)
                has_markers = bool(page_sections)

                if pdf_available:
                    pdf_doc = pdfium.PdfDocument(str(pdf_path))
                    n_pages = len(pdf_doc)
                else:
                    pdf_doc = None
                    n_pages = max(page_sections.keys()) if page_sections else 0

                st.caption(f"{n_pages} pages · {'page-synced' if has_markers else 'no page markers — showing full markdown'}")
                st.divider()

                if has_markers:
                    for i in range(n_pages):
                        page_num = i + 1
                        pipeline = page_pipelines.get(page_num, "")
                        badge = f" · `{pipeline}`" if pipeline else ""

                        if pdf_available:
                            col_pdf, col_md = st.columns(2)
                            with col_pdf:
                                st.caption(f"Page {page_num}")
                                bitmap = pdf_doc[i].render(scale=2.0)
                                st.image(bitmap.to_pil(), use_container_width=True)
                            with col_md:
                                st.caption(f"Parsed — page {page_num}{badge}")
                                section = page_sections.get(page_num, "_No content for this page._")
                                st.markdown(section, unsafe_allow_html=True)
                        else:
                            st.caption(f"Page {page_num}{badge}")
                            section = page_sections.get(page_num, "_No content for this page._")
                            st.markdown(section, unsafe_allow_html=True)
                        st.divider()
                else:
                    if pdf_available:
                        col_pdf, col_md = st.columns(2)
                        with col_pdf:
                            st.subheader("Original PDF")
                            for i in range(n_pages):
                                bitmap = pdf_doc[i].render(scale=2.0)
                                st.image(bitmap.to_pil(), caption=f"Page {i+1}", use_container_width=True)
                        with col_md:
                            st.subheader("Parsed & enhanced markdown")
                            st.markdown(md_text, unsafe_allow_html=True)
                    else:
                        st.subheader("Parsed & enhanced markdown")
                        st.markdown(md_text, unsafe_allow_html=True)

        # ── Excel / CSV inspector ──────────────────────────────────────────────
        elif suffix in {".xlsx", ".xls", ".csv"}:
            all_payloads = _load_file_chunks(selected)

            summary_payload = next(
                (p for p in all_payloads if (p.get("metadata") or {}).get("chunk_type") == "document_summary"),
                None,
            )
            data_payloads = [
                p for p in all_payloads
                if (p.get("metadata") or {}).get("chunk_type") != "document_summary"
            ]

            if not all_payloads:
                st.info("No chunks found. Re-ingest this file to inspect its contents.")
            else:
                st.caption(f"{len(data_payloads)} data chunks indexed")

                if summary_payload:
                    with st.expander("Document Summary", expanded=True):
                        summary_text = (summary_payload.get("content") or "").replace("## Document Summary\n\n", "")
                        # Strip pandas datetime noise (" 00:00:00") from display
                        summary_text = re.sub(r" 00:00:00", "", summary_text)
                        st.markdown(summary_text)

                # Group by sheet
                sheets: dict[str, list[dict]] = {}
                for p in data_payloads:
                    meta = p.get("metadata") or {}
                    sheet = meta.get("sheet_name") or "—"
                    sheets.setdefault(sheet, []).append(p)

                for sheet_name, chunks in sheets.items():
                    chunks_sorted = sorted(chunks, key=lambda p: (p.get("metadata") or {}).get("part", 0))
                    st.subheader(f"Sheet: {sheet_name}")
                    st.caption(f"{len(chunks_sorted)} chunks")
                    for idx, payload in enumerate(chunks_sorted, start=1):
                        meta = payload.get("metadata") or {}
                        chunk_type = meta.get("chunk_type", "chunk")
                        row_ref = meta.get("row_ref")
                        num_rows = meta.get("num_rows")
                        label_parts = [f"Chunk {idx}", chunk_type]
                        if row_ref is not None:
                            end_row = row_ref + (num_rows or 1) - 1
                            if end_row == row_ref:
                                label_parts.append(f"row {row_ref}")
                            else:
                                label_parts.append(f"rows {row_ref}–{end_row}")
                        label = " · ".join(label_parts)
                        content = payload.get("content") or "_No content._"
                        with st.expander(label, expanded=idx == 1):
                            st.code(content, language="text")
                    st.divider()

        # ── unsupported type ───────────────────────────────────────────────────
        else:
            st.info(f"Inspector view is available for PDF and Excel/CSV files. {suffix} files show chunks in the Retrieved Chunks tab after a query.")


# ── evaluation tab ────────────────────────────────────────────────────────────
with tab_eval:
    summary = _load_eval_summary()
    answer_rows = _load_eval_answers()
    retrieval_rows = _load_eval_retrieval()

    if not summary or not answer_rows:
        st.info("No evaluation results found yet. Run `uv run python eval/run_eval.py` to generate them.")
    else:
        st.subheader("Latest Evaluation")

        agent_metrics = summary.get("agent_answer_metrics", {})
        retrieval_metrics = summary.get("retrieval_metrics", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Correctness", _fmt_metric(agent_metrics.get("correctness")))
        col2.metric("Faithfulness", _fmt_metric(agent_metrics.get("faithfulness")))
        col3.metric("Answer relevancy", _fmt_metric(agent_metrics.get("answer_relevancy")))
        col4.metric("Hit@10", _fmt_metric(retrieval_metrics.get("hit_at_10")))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Questions", str(summary.get("question_count", len(answer_rows))))
        col6.metric("Hit@5", _fmt_metric(retrieval_metrics.get("hit_at_5")))
        col7.metric("Evidence recall@20", _fmt_metric(retrieval_metrics.get("evidence_recall_at_20")))
        col8.metric("MRR", f"{retrieval_metrics.get('mrr', 0):.3f}" if retrieval_metrics.get("mrr") is not None else "n/a")

        with st.expander("Run breakdown", expanded=False):
            st.json({
                "question_breakdown": summary.get("question_breakdown", {}),
                "judge_breakdown": summary.get("judge_breakdown", {}),
            })

        st.divider()

        # The answer result file does not currently store question_type, so infer
        # useful demo filters from qa_id ranges used by the benchmark.
        qa_filters = {
            "All": lambda row: True,
            "Single-doc factoid (qa_001-qa_032)": lambda row: "qa_001" <= row.get("qa_id", "") <= "qa_032",
            "Table lookup (qa_033-qa_040)": lambda row: "qa_033" <= row.get("qa_id", "") <= "qa_040",
            "Cross-document (qa_041-qa_050)": lambda row: "qa_041" <= row.get("qa_id", "") <= "qa_050",
            "Unanswerable (qa_051-qa_056)": lambda row: "qa_051" <= row.get("qa_id", "") <= "qa_056",
            "Needs review": lambda row: min(
                float(row.get("correctness") or 0),
                float(row.get("faithfulness") or 0),
                float(row.get("answer_relevancy") or 0),
            ) < 0.9,
        }
        selected_filter = st.selectbox("Filter QA pairs", options=list(qa_filters), label_visibility="collapsed")
        filtered_rows = [row for row in answer_rows if qa_filters[selected_filter](row)]

        table_rows = [
            {
                "qa_id": row.get("qa_id"),
                "correctness": row.get("correctness"),
                "faithfulness": row.get("faithfulness"),
                "answer_relevancy": row.get("answer_relevancy"),
                "judge": row.get("judge_used"),
                "question": row.get("question"),
            }
            for row in filtered_rows
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        if filtered_rows:
            selected_qa = st.selectbox(
                "Inspect QA pair",
                options=[row.get("qa_id", "") for row in filtered_rows],
                format_func=lambda qa_id: f"{qa_id} — {next((r.get('question', '')[:80] for r in filtered_rows if r.get('qa_id') == qa_id), '')}",
            )
            selected_row = next(row for row in filtered_rows if row.get("qa_id") == selected_qa)

            st.markdown(f"**Question**  \n{selected_row.get('question', '')}")
            gold_col, predicted_col = st.columns(2)
            with gold_col:
                st.caption("Gold answer")
                st.markdown(selected_row.get("gold_answer") or "_No gold answer recorded._")
            with predicted_col:
                st.caption("Pipeline answer")
                st.markdown(selected_row.get("predicted_answer") or "_No pipeline answer recorded._")

            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            score_col1.metric("Correctness", _fmt_metric(selected_row.get("correctness")))
            score_col2.metric("Faithfulness", _fmt_metric(selected_row.get("faithfulness")))
            score_col3.metric("Relevancy", _fmt_metric(selected_row.get("answer_relevancy")))
            score_col4.metric("Judge", selected_row.get("judge_used", "n/a"))

            retrieval_row = retrieval_rows.get(selected_qa, {})
            hits = retrieval_row.get("hits") or []
            matched = retrieval_row.get("matched_evidence") or []
            if hits:
                st.divider()
                st.markdown("**Retrieved evidence**")

                table_hits = [hit for hit in hits if _hit_is_table(hit)]
                text_hits = [hit for hit in hits if not _hit_is_table(hit)]
                evidence_scope = st.radio(
                    "Evidence type",
                    options=["All", "Tables/rows", "Text chunks"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
                if evidence_scope == "Tables/rows":
                    visible_hits = table_hits
                elif evidence_scope == "Text chunks":
                    visible_hits = text_hits
                else:
                    visible_hits = hits

                st.caption(
                    f"{len(hits)} retrieved hits · {len(table_hits)} table/row hits · "
                    f"{len(matched)} gold evidence matches"
                )

                for hit_index, hit in enumerate(visible_hits[:10], start=1):
                    label = _hit_label(hit, hit_index)
                    expanded = hit_index <= 2 or (evidence_scope == "Tables/rows" and hit_index <= 4)
                    with st.expander(label, expanded=expanded):
                        meta_cols = st.columns(4)
                        meta_cols[0].caption(f"doc_id: {hit.get('doc_id') or 'n/a'}")
                        meta_cols[1].caption(f"type: {hit.get('chunk_type') or 'n/a'}")
                        meta_cols[2].caption(f"sheet: {hit.get('sheet_name') or 'n/a'}")
                        meta_cols[3].caption(f"row: {hit.get('row_ref') if hit.get('row_ref') is not None else 'n/a'}")
                        content = hit.get("content") or "_No content recorded._"
                        if _hit_is_table(hit):
                            st.code(content, language="text")
                        else:
                            st.markdown(content)
            elif retrieval_row:
                st.info("Retrieval metrics are present for this QA pair, but no hit content was recorded.")
