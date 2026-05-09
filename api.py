"""Vault RAG — FastAPI backend for the Next.js UI."""
from __future__ import annotations

import base64
import io
import re
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (
    GENERATION_API_BASE,
    GENERATION_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RETRIEVAL_TOP_K,
)
from src.vector_store import scroll_all_payloads, get_chunks_by_file, _request as _qdrant

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "data" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKDOWN_DIRS = (
    REPO_ROOT / "data" / "output" / "processed",
    REPO_ROOT / "data" / "output" / "lightonocr",
    REPO_ROOT / "data" / "output" / "pymupdf",
    REPO_ROOT / "data" / "output" / "translated",
)

app = FastAPI(title="Vault RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


# ── agent singleton ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_agent() -> Any:
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


def _parse_sources(collected: list[str]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for raw in collected:
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

        # Extract section heading from chunk body (chunker prepends headings)
        heading_m = _MD_HEADING_RE.search(body[:600])
        if heading_m:
            section = heading_m.group(1).strip()
        elif loc_key == "sheet":
            section = loc_val
        else:
            section = ""

        # Build location label shown in the UI badge
        if loc_key == "chunk":
            location = f"chunk {loc_val}"
        elif loc_key == "sheet":
            location = f"sheet: {loc_val}"
        elif loc_key == "part":
            location = f"part {loc_val}"
        else:
            location = ""

        # Excerpt: strip markdown headings and table markers, take first 350 chars
        plain = _TABLE_MARKER_RE.sub("", body)
        plain = re.sub(r"^#{1,3}\s+.+$", "", plain, flags=re.MULTILINE).strip()
        excerpt = " ".join(plain.split())[:350]

        score = float(score_str) if score_str else None

        key = (filename, excerpt[:80])
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "filename": filename,
            "section": section,
            "location": location,
            "excerpt": excerpt,
            "score": round(score, 4) if score else None,
        })
    return sources[:8]


def _resolve_pdf_path(filename: str) -> Path | None:
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

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), pipeline: str = Form("auto")):
    dest = INPUT_DIR / file.filename
    dest.write_bytes(await file.read())
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "stage": "parsing", "chunks_created": 0}
    force_pipeline = pipeline if pipeline in {"ocr", "text"} else None
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_ingest_sync, job_id, dest, force_pipeline)
    return {"job_id": job_id, "status": "processing"}


@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job["status"], "stage": job["stage"], "chunks_created": job.get("chunks_created", 0)}


@app.get("/documents")
async def list_documents():
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return []
    return _payloads_to_docs(payloads)


@app.get("/stats")
async def stats():
    try:
        payloads = scroll_all_payloads(QDRANT_URL, QDRANT_COLLECTION)
    except Exception:
        return {"total_docs": 0, "total_chunks": 0}
    docs = _payloads_to_docs(payloads)
    return {"total_docs": len(docs), "total_chunks": len(payloads)}


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query(req: QueryRequest):
    from src.rag_agent import stream_agent
    agent = _get_agent()
    collected: list[str] = []
    tokens: list[str] = []
    loop = asyncio.get_event_loop()

    def _run():
        for token in stream_agent(agent, req.question, collected_chunks=collected):
            tokens.append(token)

    await loop.run_in_executor(_executor, _run)
    return {"answer": "".join(tokens).strip(), "sources": _parse_sources(collected)}


@app.delete("/collection")
async def clear_collection():
    base = QDRANT_URL.rstrip("/")
    try:
        _qdrant("DELETE", f"{base}/collections/{QDRANT_COLLECTION}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    _get_agent.cache_clear()
    return {"status": "cleared"}


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
