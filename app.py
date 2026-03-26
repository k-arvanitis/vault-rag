"""RagForge — Streamlit chat UI.

Features:
- Upload PDFs, Excel, CSV, or Markdown files → ingested into Qdrant
- Chat with your documents (ReAct agent, multi-step retrieval)
- Conversation history per session
- Document library: see what's been ingested, delete individual files
- Document inspector: original PDF side-by-side with parsed markdown
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "data" / "input"
TRANSLATED_DIR = REPO_ROOT / "data" / "output" / "translated"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RagForge",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── lazy imports ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG agent…")
def _get_agent():
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
    else:
        from src.ingest import run_ingest
        run_ingest(pdf_path=dest, collection=QDRANT_COLLECTION)

    return uploaded_file.name


def _list_ingested_files() -> list[str]:
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
    from src.config import QDRANT_COLLECTION, QDRANT_URL
    from src.vector_store import delete_by_file
    delete_by_file(QDRANT_URL, QDRANT_COLLECTION, file_name)


def _split_markdown_by_page(md_text: str) -> dict[int, str]:
    """Split markdown into {page_number: content} using <!-- PAGE N --> markers."""
    import re
    parts = re.split(r"<!--\s*PAGE\s+(\d+)\s*-->", md_text)
    pages: dict[int, str] = {}
    # parts = [pre_text, "1", content1, "2", content2, ...]
    i = 1
    while i < len(parts) - 1:
        page_num = int(parts[i])
        content = parts[i + 1].strip()
        pages[page_num] = content
        i += 2
    return pages


# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = []


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔥 RagForge")
    st.caption("Self-hosted RAG — hybrid search · HyDE · reranking")
    st.divider()

    st.subheader("Add documents")
    uploaded = st.file_uploader(
        "Upload PDF, Excel, CSV, or Markdown",
        type=["pdf", "xlsx", "xls", "csv", "md"],
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
tab_chat, tab_inspect = st.tabs(["💬 Chat", "🔍 Document Inspector"])

# ── chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    agent = _get_agent()
                    from src.rag_agent import ask_agent
                    answer = ask_agent(agent, prompt)
                except Exception as exc:
                    answer = f"Error: {exc}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ── inspector tab ─────────────────────────────────────────────────────────────
with tab_inspect:
    ingested_files = _cached_files()
    pdf_ingested = [f for f in ingested_files if f.lower().endswith(".pdf")]

    if not pdf_ingested:
        st.info("No PDFs ingested yet. Upload a PDF via the sidebar.")
    else:
        selected = st.selectbox(
            "Select a document",
            options=pdf_ingested,
            label_visibility="collapsed",
        )
        pdf_name = Path(selected).name  # strip any leading path like "data/input/"
        pdf_path = INPUT_DIR / pdf_name
        md_path = TRANSLATED_DIR / (Path(pdf_name).stem + ".md")

        if not pdf_path.exists():
            st.warning("PDF not available locally — it was ingested before file saving was enabled.")
        elif not md_path.exists():
            st.info("Markdown not found. Re-ingest this PDF to generate it.")
        else:
            import pypdfium2 as pdfium
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            md_text = md_path.read_text(encoding="utf-8")
            page_sections = _split_markdown_by_page(md_text)
            n_pages = len(pdf_doc)
            has_markers = bool(page_sections)

            st.caption(f"{n_pages} pages · {'page-synced' if has_markers else 'no page markers — showing full markdown'}")
            st.divider()

            if has_markers:
                for i in range(n_pages):
                    page_num = i + 1
                    col_pdf, col_md = st.columns(2)
                    with col_pdf:
                        st.caption(f"Page {page_num}")
                        bitmap = pdf_doc[i].render(scale=2.0)
                        st.image(bitmap.to_pil(), use_container_width=True)
                    with col_md:
                        st.caption(f"Parsed — page {page_num}")
                        section = page_sections.get(page_num, "_No content for this page._")
                        st.markdown(section)
                    st.divider()
            else:
                # Fallback: all pages left, full markdown right
                col_pdf, col_md = st.columns(2)
                with col_pdf:
                    st.subheader("Original PDF")
                    for i in range(n_pages):
                        bitmap = pdf_doc[i].render(scale=2.0)
                        st.image(bitmap.to_pil(), caption=f"Page {i+1}", use_container_width=True)
                with col_md:
                    st.subheader("Parsed & enhanced markdown")
                    st.markdown(md_text)
