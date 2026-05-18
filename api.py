"""Vault RAG — FastAPI backend for the Next.js UI."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.config import (  # noqa: E402
    API_CORS_ORIGINS,
    API_KEY,
    GENERATION_API_BASE,
    GENERATION_MODEL,
    MAX_TOOL_RESULTS,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
from src.vector_store import scroll_all_payloads, get_chunks_by_file, delete_by_file, _request as _qdrant  # noqa: E402
from src.file_resolver import resolve_original_name as _resolve_original_name  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "data" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKDOWN_DIRS = (
    REPO_ROOT / "data" / "output" / "processed",
    REPO_ROOT / "data" / "output" / "lightonocr",
    REPO_ROOT / "data" / "output" / "pymupdf",
    REPO_ROOT / "data" / "output" / "translated",
)

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ── lifespan: warm the agent on startup ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once at startup so the first /query doesn't pay the cold-start penalty."""
    try:
        _get_agent()
        logger.info("Agent warmed; ready to serve")
    except Exception:
        logger.exception("Agent warmup failed; /query will retry per-request")
    yield


app = FastAPI(title="Vault RAG API", lifespan=lifespan)


# ── CORS — explicit origin list (use API_CORS_ORIGINS=* only intentionally) ───

_cors_origins = [o.strip() for o in API_CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── auth dep — required on mutating endpoints when API_KEY is set ─────────────

async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Raise 401 unless the X-API-Key header matches API_KEY (when configured)."""
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ── global exception handler — never leak stack traces ────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the exception and return a generic 500 instead of leaking str(exc)."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── agent singleton ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_agent() -> Any:
    """Build (and cache) the RAG agent — one instance reused across requests."""
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


# ── helpers ────────────────────────────────────────────────────────────────────

def _run_ingest_sync(job_id: str, dest: Path, force_pipeline: str | None = None) -> None:
    """Ingest one uploaded file into Qdrant/DuckDB, updating the job record."""
    suffix = dest.suffix.lower()
    _jobs[job_id]["status"] = "processing"
    try:
        if suffix in {".xlsx", ".xls", ".csv"}:
            _jobs[job_id]["stage"] = "chunking"
            from src.ingest_table_rows import ingest_table_rows
            ingest_table_rows(str(dest), collection=QDRANT_COLLECTION)
            _jobs[job_id]["chunks_created"] = -1
        else:
            from src.ingest import run_ingest
            _jobs[job_id]["stage"] = "parsing"
            result = run_ingest(pdf_path=dest, collection=QDRANT_COLLECTION, force_pipeline=force_pipeline)
            chunks_path = result.get("chunks_path") if isinstance(result, dict) else None
            if chunks_path and Path(chunks_path).exists():
                import json as _json
                _jobs[job_id]["chunks_created"] = len(_json.loads(Path(chunks_path).read_text()))
        _jobs[job_id].update({"status": "done", "stage": "indexed"})
    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "stage": "failed", "error": str(exc)})


def _payloads_to_docs(payloads: list[dict]) -> list[dict]:
    """Group Qdrant payloads into one document card per source file."""
    from collections import defaultdict
    counts: dict[str, int] = defaultdict(int)
    for p in payloads:
        meta = p.get("metadata", {}) or {}
        name = meta.get("source_file") or meta.get("file_name") or ""
        if name:
            counts[name] += 1
    type_map = {
        "pdf": "PDF", "xlsx": "Excel", "xls": "Excel", "csv": "CSV",
        "docx": "Word", "doc": "Word", "md": "MD",
        "png": "Image", "jpg": "Image", "jpeg": "Image",
    }
    return [
        {
            "filename": name,
            "file_type": type_map.get(Path(name).suffix.lstrip(".").lower(), "File"),
            "chunk_count": count,
            "status": "indexed",
        }
        for name, count in sorted(counts.items())
    ]


# Chunk header format: "[1] file=name.pdf chunk=5 score=0.8312"
# or                   "[1] file=name.xlsx sheet=Sheet1 score=0.91"
_HEADER_RE = re.compile(
    r"^\[?\d+\]?\s+file=(?P<file>[^\s]+)"
    r"(?:\s+(?P<loc_key>chunk|sheet|part|repair_query|subquery)=(?P<loc_val>[^\s]+))?"
    r"(?:\s+score=(?P<score>[^\s]+))?",
)
_MD_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_TABLE_MARKER_RE = re.compile(r"\[TABLE_START\]|\[TABLE_END\]")
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE\s+(\d+)")


_CALL_BOUNDARY = "---CALL_BOUNDARY---"

# The model sometimes copies a raw retrieved-chunk header line
# ("[1] file=doc.pdf chunk=2 ...") or a dangling "Sources:" label into its final
# answer. Strip those whole lines so they never reach the user. Inline citation
# markers like "[1]" are NOT matched — the pattern requires "file=" after them.
_LEAKED_HEADER_RE = re.compile(
    r"^[ \t]*(?:\[\d+\][ \t]+file=\S+.*|sources?[ \t]*:[ \t]*)$",
    re.MULTILINE | re.IGNORECASE,
)
# Inline [N] citation markers — dropped because the answer's numbering does not
# correspond to the trace panel's, so they would be misleading. The "Retrieved
# chunks" panel is the source list. Only [N] within the retrieved-chunk range
# (1..MAX_TOOL_RESULTS) is treated as a citation — a bracketed year like [2024]
# or any larger number is left alone. List markers like "1." (no brackets) too.
_INLINE_CITATION_RE = re.compile(r"[ \t]*\[(\d+)\]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _strip_inline_citation(match: re.Match) -> str:
    """Drop a [N] marker only when N is a plausible retrieved-chunk index."""
    return "" if 1 <= int(match.group(1)) <= MAX_TOOL_RESULTS else match.group(0)


def _strip_leaked_headers(text: str) -> str:
    """Remove raw chunk-header lines and inline [N] citation markers the LLM
    echoes into its answer — the trace panel is the source list."""
    cleaned = _LEAKED_HEADER_RE.sub("", text)
    cleaned = _INLINE_CITATION_RE.sub(_strip_inline_citation, cleaned)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _parse_sources(collected: list[str]) -> list[dict]:
    """Parse collected tool chunks into source cards for the UI.

    Groups chunks by tool call (boundaries marked by `---CALL_BOUNDARY---`) and
    iterates calls in REVERSE order — later calls are typically scoped/refined
    and contain the answer-bearing chunks the LLM actually used. This prevents
    the first broad call's noise chunks from monopolizing the displayed list.
    """
    calls: list[list[str]] = [[]]
    for raw in collected:
        if raw == _CALL_BOUNDARY:
            calls.append([])
        else:
            calls[-1].append(raw)
    calls = [c for c in calls if c]

    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for call_chunks in reversed(calls):
        for raw in call_chunks:
            lines = raw.strip().splitlines()
            if not lines:
                continue
            m = _HEADER_RE.match(lines[0])
            filename = m.group("file") if m else "unknown"
            loc_key = m.group("loc_key") or "" if m else ""
            loc_val = m.group("loc_val") or "" if m else ""
            score_str = m.group("score") if m else None

            body_lines = lines[1:] if len(lines) > 1 else []
            body = "\n".join(body_lines).strip()

            page_m = _PAGE_MARKER_RE.search(body)
            page = int(page_m.group(1)) if page_m else None

            if filename.startswith("eval/data/raw/"):
                filename = filename[len("eval/data/raw/"):]
            filename = _resolve_original_name(filename)

            heading_m = _MD_HEADING_RE.search(body[:600])
            if heading_m:
                section = heading_m.group(1).strip()
            elif loc_key == "sheet":
                section = loc_val
            else:
                section = ""

            is_doc_summary = loc_key == "chunk" and loc_val == "-1"
            is_sheet_summary = loc_key == "sheet" and "Sheet summary:" in body[:200]

            if is_doc_summary:
                location = "document summary"
            elif is_sheet_summary:
                location = f"sheet summary: {loc_val}"
            elif loc_key == "chunk":
                location = f"chunk {loc_val}"
            elif loc_key == "sheet":
                location = f"sheet: {loc_val}"
            elif loc_key == "part":
                location = f"part {loc_val}"
            else:
                location = ""

            plain = _TABLE_MARKER_RE.sub("", body)
            plain = re.sub(r"^#{1,3}\s+.+$", "", plain, flags=re.MULTILINE).strip()
            excerpt = " ".join(plain.split())[:350]

            score = float(score_str) if score_str else None
            # 1.0 is the retriever's placeholder for filter/scroll fetches
            # (doc-routed chunks carry no similarity score) — drop it so the UI
            # doesn't show a misleading "perfect match" chip.
            if score == 1.0:
                score = None

            key = (filename, excerpt[:80])
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "filename": filename,
                "section": section,
                "location": location,
                "page": page,
                "excerpt": excerpt,
                "score": round(score, 4) if score else None,
            })
    return sources[:8]


def _resolve_pdf_path(filename: str) -> Path | None:
    """Find the on-disk PDF for a filename, or None if it cannot be located."""
    p = Path(filename)
    candidates = [
        INPUT_DIR / p.name,
        p if p.is_absolute() else None,
        REPO_ROOT / p,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def _resolve_markdown_path(filename: str) -> Path | None:
    """Find the parsed-markdown file for a document, or None if absent."""
    stem = Path(filename).stem
    for d in MARKDOWN_DIRS:
        candidate = d / f"{stem}.md"
        if candidate.exists():
            return candidate
    return None


def _split_markdown_pages(md_text: str) -> tuple[dict[int, str], dict[int, str]]:
    """Split markdown by <!-- PAGE N | label --> markers."""
    parts = re.split(r"<!--\s*PAGE\s+(\d+)(?:\s*\|([^-]*))?\s*-->", md_text)
    pages: dict[int, str] = {}
    pipelines: dict[int, str] = {}
    i = 1
    while i < len(parts) - 2:
        page_num = int(parts[i])
        label = (parts[i + 1] or "").strip()
        content = _TABLE_MARKER_RE.sub("", parts[i + 2]).strip()
        pages[page_num] = content
        if label:
            pipelines[page_num] = label
        i += 3
    return pages, pipelines


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness probe — returns immediately, does not touch Qdrant or the agent."""
    return {"status": "ok"}


@app.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest(file: UploadFile = File(...), pipeline: str = Form("auto")):
    """POST /ingest — save the uploaded file and start a background ingest job."""
    dest = INPUT_DIR / file.filename
    dest.write_bytes(await file.read())
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "stage": "parsing", "chunks_created": 0}
    force_pipeline = pipeline if pipeline in {"ocr", "text"} else None
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_ingest_sync, job_id, dest, force_pipeline)
    return {"job_id": job_id, "status": "processing"}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    """GET /ingest/status — report a background ingest job's progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job["status"], "stage": job["stage"], "chunks_created": job.get("chunks_created", 0)}


@app.get("/documents")
async def list_documents():
    """GET /documents — list ingested documents as UI cards."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return []
    return _payloads_to_docs(payloads)


@app.get("/stats")
async def stats():
    """GET /stats — return total document and chunk counts."""
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return {"total_docs": 0, "total_chunks": 0}
    docs = _payloads_to_docs(payloads)
    return {"total_docs": len(docs), "total_chunks": len(payloads)}


class QueryRequest(BaseModel):
    question: str


_RETRY_INSTRUCTION = (
    "\n\nIMPORTANT: This is a retry. The previous attempt returned Unsupported. "
    "You MUST follow the doc-routing protocol strictly: "
    "(1) call search_knowledge_base with the topic words alone to identify the relevant "
    "document_summary chunk and read its Document ID (doc_XXX). "
    "(2) call search_knowledge_base again with the original question, passing that doc_id "
    "as the doc_id argument to scope the search to that specific document."
)


@app.post("/query")
async def query(req: QueryRequest):
    """POST /query — answer a question with the RAG agent, splitting multi-part questions and merging the sub-answers."""
    from src.rag_agent import stream_agent, route_question, routing_directive
    from src.answer_quality import _is_multi_part_query, _split_multi_part_query
    agent = _get_agent()
    loop = asyncio.get_running_loop()

    async def _run_once(question: str) -> tuple[str, list[str], dict]:
        """Run the agent once for a question; return (answer, collected chunks, trace)."""
        collected: list[str] = []
        tokens: list[str] = []
        trace_holder: dict = {}

        def _run():
            """Drive stream_agent in a worker thread, collecting tokens, chunks and trace."""
            sql_trace: list[str] = []
            tool_calls: list[str] = []
            for token in stream_agent(
                agent,
                question,
                collected_chunks=collected,
                sql_trace=sql_trace,
                tool_calls=tool_calls,
            ):
                tokens.append(token)
            trace_holder["sql"] = sql_trace
            trace_holder["tools"] = tool_calls

        await loop.run_in_executor(_executor, _run)
        return "".join(tokens).strip(), collected, trace_holder

    async def _answer(question: str) -> tuple[str, list[str], dict]:
        """Answer one question, with a forced retry on a bare Unsupported.

        First resolves the tool deterministically: route_question matches the
        question against document summaries in Qdrant and, if it lands on a
        spreadsheet vs a text document, a routing directive is prepended so the
        agent uses query_excel vs search_knowledge_base accordingly — instead of
        guessing the tool from question wording.

        Groq inference at temp=0 still has small nondeterminism; the agent
        occasionally skips doc-routing on the first attempt and returns
        Unsupported despite the answer existing. The retry forces the protocol.
        """
        route = await loop.run_in_executor(_executor, route_question, question)
        q = routing_directive(route) + question
        ans, coll, trace = await _run_once(q)
        if ans.lower() == "unsupported":
            r_ans, r_coll, r_trace = await _run_once(q + _RETRY_INSTRUCTION)
            if r_ans.lower() != "unsupported" and r_ans:
                return r_ans, r_coll, r_trace
        return ans, coll, trace

    # Multi-part questions are split and answered one sub-question at a time,
    # then merged here deterministically. The agent's single-pass synthesis
    # intermittently drops a part, so we never rely on it for that — each
    # sub-question runs on its own and every part is guaranteed in the output.
    parts = (
        _split_multi_part_query(req.question)
        if _is_multi_part_query(req.question)
        else [req.question]
    )
    if len(parts) == 1:
        answer, collected, excel_trace = await _answer(req.question)
    else:
        sub_answers: list[str] = []
        collected = []
        sql_all: list[str] = []
        tools_all: list[str] = []
        for part in parts:
            p_ans, p_coll, p_trace = await _answer(part)
            sub_answers.append(p_ans.strip())
            collected += p_coll
            sql_all += p_trace.get("sql") or []
            tools_all += p_trace.get("tools") or []
        # Blank line between parts (a single \n is only a soft break in
        # markdown); number them so a terse part still reads as its own answer.
        kept = [a for a in sub_answers if a]
        answer = "\n\n".join(f"{i}. {a}" for i, a in enumerate(kept, 1)) or "Unsupported"
        excel_trace = {"sql": sql_all, "tools": tools_all}

    answer = _strip_leaked_headers(answer)
    sources = _parse_sources(collected)
    sql_list = [s for s in (excel_trace.get("sql") or []) if s]
    return {
        "answer": answer,
        "sources": sources,
        "sql": sql_list,
        "tools_used": _tools_used(excel_trace.get("tools") or []),
    }


# Real tool names → frontend pill keys (see TOOL_META in TraceSidebar.tsx).
_TOOL_DISPLAY = {"search_knowledge_base": "search_documents", "query_excel": "query_excel"}


def _tools_used(tool_calls: list[str]) -> list[str]:
    """Map the agent's actual tool invocations to display keys, deduped in call order."""
    seen: list[str] = []
    for name in tool_calls:
        key = _TOOL_DISPLAY.get(name, name)
        if key not in seen:
            seen.append(key)
    return seen


@app.delete("/collection", dependencies=[Depends(require_api_key)])
async def clear_collection():
    """DELETE /collection — drop the Qdrant collection and reset the agent."""
    base = QDRANT_URL.rstrip("/")
    _qdrant("DELETE", f"{base}/collections/{QDRANT_COLLECTION}")
    _get_agent.cache_clear()
    return {"status": "cleared"}


@app.delete("/documents/{filename:path}", dependencies=[Depends(require_api_key)])
async def delete_document(filename: str):
    """Remove all Qdrant points for a single file."""
    try:
        deleted = delete_by_file(QDRANT_URL, QDRANT_COLLECTION, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _get_agent.cache_clear()
    return {"status": "deleted", "filename": filename, "points_deleted": deleted}


# ── inspector endpoints ────────────────────────────────────────────────────────

@app.get("/documents/{filename:path}/chunks")
async def document_chunks(filename: str):
    """All Qdrant chunks for a file, grouped and sorted for the inspector."""
    try:
        payloads = get_chunks_by_file(QDRANT_URL, QDRANT_COLLECTION, filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not payloads:
        return {"summary": None, "chunks": []}

    summary_payload = next(
        (p for p in payloads if (p.get("metadata") or {}).get("chunk_type") == "document_summary"),
        None,
    )
    data_chunks = [
        {
            "content": p.get("content", ""),
            "metadata": p.get("metadata", {}),
        }
        for p in payloads
        if (p.get("metadata") or {}).get("chunk_type") != "document_summary"
    ]
    data_chunks.sort(key=lambda c: c["metadata"].get("chunk_index", c["metadata"].get("part", 0)))

    return {
        "summary": summary_payload.get("content") if summary_payload else None,
        "chunks": data_chunks,
    }


@app.get("/documents/{filename:path}/markdown")
async def document_markdown(filename: str):
    """Return parsed markdown split into pages (for PDF inspector)."""
    md_path = _resolve_markdown_path(filename)
    if not md_path:
        raise HTTPException(status_code=404, detail="Markdown not found")
    raw = md_path.read_text(encoding="utf-8")
    pages, pipelines = _split_markdown_pages(raw)
    has_markers = bool(pages)
    return {
        "has_page_markers": has_markers,
        "full_text": raw if not has_markers else None,
        "pages": [
            {"page": n, "content": pages[n], "pipeline": pipelines.get(n, "")}
            for n in sorted(pages)
        ],
    }


@app.get("/documents/{filename:path}/pdf/{page}")
async def document_pdf_page(filename: str, page: int):
    """Render a single PDF page (1-indexed) and return it as a base64 PNG."""
    pdf_path = _resolve_pdf_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        n = len(doc)
        if page < 1 or page > n:
            raise HTTPException(status_code=404, detail=f"Page {page} out of range (1–{n})")
        bitmap = doc[page - 1].render(scale=1.5)
        img = bitmap.to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image_b64": b64, "page": page, "total_pages": n}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents/{filename:path}/table-sheet/{sheet}")
async def document_table_sheet(filename: str, sheet: str):
    """Return raw rows and cleaned markdown for one sheet of an Excel/CSV file."""
    TABLE_MD_DIR = REPO_ROOT / "data" / "output" / "table_markdowns"
    file_stem = Path(filename).stem
    safe_sheet = sheet.replace("/", "_").replace("\\", "_")
    md_path = TABLE_MD_DIR / f"{file_stem}__{safe_sheet}.md"
    raw_path = INPUT_DIR / filename
    suffix = Path(filename).suffix.lower()

    cleaned_md: str | None = md_path.read_text(encoding="utf-8") if md_path.exists() else None

    raw_rows: list[list[str]] | None = None
    if raw_path.exists():
        try:
            import pandas as pd
            if suffix == ".csv":
                df = pd.read_csv(raw_path, header=None, dtype=str, nrows=60).fillna("")
            else:
                df = pd.read_excel(raw_path, sheet_name=sheet, header=None, dtype=str, nrows=60).fillna("")
            raw_rows = df.values.tolist()
        except Exception:
            raw_rows = None

    if cleaned_md is None and raw_rows is None:
        raise HTTPException(status_code=404, detail="No data found for this sheet")

    return {"sheet": sheet, "raw_rows": raw_rows, "cleaned_md": cleaned_md}


@app.get("/documents/{filename:path}/pdf/info")
async def document_pdf_info(filename: str):
    """Return total page count for a PDF."""
    pdf_path = _resolve_pdf_path(filename)
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF not found")
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf_path))
        return {"total_pages": len(doc)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
