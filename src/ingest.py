import argparse
import importlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader  # noqa: E402

from src import chunker as chunker_module  # noqa: E402
from src.config import (  # noqa: E402
    QDRANT_URL, QDRANT_COLLECTION, OLLAMA_API_BASE, OLLAMA_EMBED_MODEL,
    OCR_API_BASE, OCR_MODEL,
)
from src.embedder import embed_chunks_file  # noqa: E402
from src.ingest_table_rows import ingest_table_rows  # noqa: E402
from src.parser.input_utils import resolve_to_pdf  # noqa: E402
from src.parser.lightonocr_parser import pdf_to_markdown  # noqa: E402
from src.vector_store import delete_by_file, ingest_embeddings  # noqa: E402

_IMAGE_ANALYSIS_API_BASE = os.getenv("IMAGE_ANALYSIS_API_BASE", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))
_IMAGE_ANALYSIS_MODEL = os.getenv("IMAGE_ANALYSIS_MODEL", "Qwen/Qwen3.5-2B")

REPO_ROOT = chunker_module.REPO_ROOT


def _log(message: str, step: int | None = None, debug: bool = False, enabled: bool = True) -> None:
    if not enabled:
        return
    prefix = "[INGEST]"
    if step is not None:
        prefix += f"[{step}/5]"
    if debug:
        prefix += "[DEBUG]"
    print(f"{prefix} {message}")


def _path_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _load_json_list(path: Path, label: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {label} at {path}, got {type(payload).__name__}")
    return payload


def _pdf_page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def _run_ingest_pdf(
    pdf_path: Path,
    collection: str,
    qdrant_url: str = QDRANT_URL,
    ocr_endpoint: str = f"{OCR_API_BASE}/v1/chat/completions",
    ocr_model_name: str = OCR_MODEL,
    ocr_table_mode: str = "grid",
    ollama_api_base: str = OLLAMA_API_BASE,
    ollama_embed_model: str = OLLAMA_EMBED_MODEL,
    enrich_with_llm: bool = True,
    verbose: bool = True,
) -> dict[str, Path]:
    file_started_at = time.perf_counter()
    pdf_path = Path(pdf_path)
    _log(f"Start | file={pdf_path} | collection={collection}", enabled=verbose)
    _log(
        "Endpoints | "
        f"qdrant={qdrant_url} ocr={ocr_endpoint} model={ocr_model_name} "
        f"ollama={ollama_api_base} embed_model={ollama_embed_model}",
        debug=True,
        enabled=verbose,
    )
    _log(f"Input file exists={pdf_path.exists()} | path={pdf_path}", debug=True, enabled=verbose)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
    if pdf_path.suffix.lower() not in {".pdf", ".docx", ".doc", ".ppt", ".pptx"}:
        raise ValueError(
            "run_ingest expects PDF/Office file path ('.pdf', '.doc', '.docx', '.ppt', '.pptx'). "
            f"Received: {pdf_path}. "
            "If you already have chunks JSON, run embedding + vector-store ingest directly "
            "with src.embedder.embed_chunks_file and src.vector_store.ingest_embeddings."
        )
    resolved_pdf_path = resolve_to_pdf(pdf_path)
    page_count = _pdf_page_count(resolved_pdf_path)
    _log(
        f"Input size={_path_size_mb(pdf_path):.2f} MB | pages={page_count} | resolved_pdf={resolved_pdf_path}",
        debug=True,
        enabled=verbose,
    )

    # 1) PDF -> markdown (LightOn OCR)
    _log("Parse PDF -> markdown", step=1, enabled=verbose)
    _log(
        "Parsing config: "
        f"endpoint={ocr_endpoint}, model={ocr_model_name}, table_mode={ocr_table_mode}",
        step=1,
        debug=True,
        enabled=verbose,
    )
    md_path = pdf_to_markdown(
        pdf_path=resolved_pdf_path,
        output_dir=REPO_ROOT / "data/output/lightonocr",
        endpoint=ocr_endpoint,
        model_name=ocr_model_name,
        markdown_tables=True,
        table_mode=ocr_table_mode,
        image_analysis_endpoint=f"{_IMAGE_ANALYSIS_API_BASE.rstrip('/')}/v1/chat/completions",
        image_analysis_model=_IMAGE_ANALYSIS_MODEL,
        image_analysis_timeout_sec=300,
    )
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown output not found after parsing step: {md_path}")
    markdown = md_path.read_text(encoding="utf-8")
    page_markers = markdown.count("<!-- PAGE ")
    _log(
        f"Done | markdown={md_path} | pages={page_markers} | chars={len(markdown)} | size={_path_size_mb(md_path):.2f} MB",
        step=1,
        enabled=verbose,
    )

    # 1b) Save processed markdown to data/output/translated/ (translation removed)
    translated_dir = REPO_ROOT / "data" / "output" / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    translated_md_path = translated_dir / md_path.name
    translated_md_path.write_text(markdown, encoding="utf-8")

    # 1c) Convert ASCII grid tables to row sentences, save to data/output/processed/
    from src.table_processor import process_and_save as _process_md
    processed_dir = REPO_ROOT / "data" / "output" / "processed"
    markdown, _ = _process_md(markdown, md_path.stem, processed_dir)

    # 2) markdown -> chunks JSON
    _log("Chunk markdown -> chunks JSON", step=2, enabled=verbose)
    _log(f"Chunking config: enrich_with_llm={enrich_with_llm}", step=2, debug=True, enabled=verbose)
    # Reload chunker to pick up local prompt/chunking edits in long-running notebook kernels.
    chunker = importlib.reload(chunker_module)
    created_chunks = chunker.chunk_markdown(
        markdown=markdown,
        enrich_with_llm=enrich_with_llm,
        verbose=verbose,
        output_dir=REPO_ROOT / "data/output/chunks",
        file_name=md_path.name,
        source_file=str(pdf_path),
    )
    chunks_path = REPO_ROOT / "data/output/chunks" / f"{md_path.stem}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk output not found after chunking step: {chunks_path}")
    chunk_rows = _load_json_list(chunks_path, "chunks output")
    _log(
        f"Done | chunks={len(chunk_rows)} (in_memory={len(created_chunks)}) | file={chunks_path} | size={_path_size_mb(chunks_path):.2f} MB",
        step=2,
        enabled=verbose,
    )

    # 3) chunks JSON -> embeddings JSON (Ollama)
    _log("Embed chunks -> embeddings JSON", step=3, enabled=verbose)
    _log(
        f"Embedding config: api_base={ollama_api_base}, model={ollama_embed_model}, input_chunks={len(chunk_rows)}",
        step=3,
        debug=True,
        enabled=verbose,
    )
    embeddings_path = embed_chunks_file(
        input_path=chunks_path,
        output_path=REPO_ROOT / "data/output/embeddings" / f"{md_path.stem}_chunks_embeddings.json",
        api_base=ollama_api_base,
        model_name=ollama_embed_model,
        verbose=verbose,
    )
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings output not found after embedding step: {embeddings_path}")
    embedding_rows = _load_json_list(embeddings_path, "embeddings output")
    vector_dim = 0
    if embedding_rows:
        first_vec = embedding_rows[0].get("embedding", [])
        if isinstance(first_vec, list):
            vector_dim = len(first_vec)
    _log(
        f"Done | vectors={len(embedding_rows)} | dim={vector_dim} | file={embeddings_path} | size={_path_size_mb(embeddings_path):.2f} MB",
        step=3,
        enabled=verbose,
    )

    # 4) embeddings JSON -> vector store (Qdrant)
    _log("Upsert embeddings -> Qdrant", step=4, enabled=verbose)
    deleted = delete_by_file(url=qdrant_url, collection=collection, file_name=md_path.name)
    if deleted > 0:
        _log(f"Removed {deleted} existing points for '{md_path.name}'", step=4, enabled=verbose)
    _log(
        f"Qdrant target: url={qdrant_url}, collection={collection}, points={len(embedding_rows)}",
        step=4,
        debug=True,
        enabled=verbose,
    )
    ingest_embeddings(
        input_path=embeddings_path,
        url=qdrant_url,
        collection=collection,
        verbose=verbose,
    )
    _log(f"Done | upserted_points={len(embedding_rows)} | collection={collection}", step=4, enabled=verbose)
    elapsed_seconds = time.perf_counter() - file_started_at
    _log(
        f"Pipeline finished successfully | pages={page_count} | elapsed={elapsed_seconds:.2f}s",
        enabled=verbose,
    )

    return {
        "markdown_path": md_path,
        "chunks_path": chunks_path,
        "embeddings_path": embeddings_path,
    }


def run_ingest(
    collection: str,
    pdf_path: Path | None = None,
    folder_path: Path | None = None,
    qdrant_url: str = "http://127.0.0.1:7333",
    ocr_endpoint: str = "http://127.0.0.1:8002/v1/chat/completions",
    ocr_model_name: str = "lightonocr-2-1b-ocr-soup",
    ocr_table_mode: str = "grid",
    data_format: str = "all",
    ollama_api_base: str = "http://127.0.0.1:11434",
    ollama_embed_model: str = "bge-m3",
    enrich_with_llm: bool = True,
    recursive: bool = False,
    verbose: bool = True,
) -> dict[str, Path] | list[dict[str, Path]]:
    if (pdf_path is None) == (folder_path is None):
        raise ValueError("Provide exactly one of `pdf_path` or `folder_path`.")

    if pdf_path is not None:
        return _run_ingest_pdf(
            pdf_path=pdf_path,
            collection=collection,
            qdrant_url=qdrant_url,
            ocr_endpoint=ocr_endpoint,
            ocr_model_name=ocr_model_name,
            ocr_table_mode=ocr_table_mode,
            ollama_api_base=ollama_api_base,
            ollama_embed_model=ollama_embed_model,
            enrich_with_llm=enrich_with_llm,
            verbose=verbose,
        )

    target_dir = Path(folder_path)
    if not target_dir.exists():
        raise FileNotFoundError(f"Folder not found: {target_dir}")
    if not target_dir.is_dir():
        raise ValueError(f"Expected a directory for `folder_path`, got: {target_dir}")

    patterns_by_format = {
        "pdf": ["*.pdf"],
        "docx": ["*.docx", "*.doc"],
        "pptx": ["*.pptx", "*.ppt"],
        "all": ["*.pdf", "*.docx", "*.doc", "*.pptx", "*.ppt"],
    }
    if data_format not in patterns_by_format:
        raise ValueError("data_format must be one of: pdf, docx, pptx, all")

    _TABLE_EXTS = {".xlsx", ".xls", ".csv"}
    _DOC_PATTERNS = patterns_by_format[data_format]
    search_patterns = [f"**/{p}" for p in _DOC_PATTERNS] if recursive else _DOC_PATTERNS
    doc_files: list[Path] = []
    for pattern in search_patterns:
        doc_files.extend(target_dir.glob(pattern))

    table_pattern = "**/*.xlsx **/*.xls **/*.csv".split() if recursive else ["*.xlsx", "*.xls", "*.csv"]
    table_files: list[Path] = []
    for pattern in table_pattern:
        table_files.extend(target_dir.glob(pattern))

    all_files = sorted(set(doc_files + table_files))
    if not all_files:
        raise ValueError(
            f"No input files found in {target_dir} (recursive={recursive})"
        )

    _log(
        f"Folder mode | dir={target_dir} | docs={len(doc_files)} | tables={len(table_files)}",
        enabled=verbose,
    )
    results: list[dict[str, Path]] = []
    for index, file_path in enumerate(all_files, start=1):
        file_started_at = time.perf_counter()
        if file_path.suffix.lower() in _TABLE_EXTS:
            _log(f"File {index}/{len(all_files)} | {file_path.name} [TABLE]", enabled=verbose)
            out_path = ingest_table_rows(str(file_path), collection=collection, verbose=verbose)
            results.append({"chunks_path": out_path})
        else:
            file_size_mb = _path_size_mb(file_path)
            _log(
                f"File {index}/{len(all_files)} | {file_path.name} | size={file_size_mb:.2f} MB",
                enabled=verbose,
            )
            out = _run_ingest_pdf(
                pdf_path=file_path,
                collection=collection,
                qdrant_url=qdrant_url,
                ocr_endpoint=ocr_endpoint,
                ocr_model_name=ocr_model_name,
                ocr_table_mode=ocr_table_mode,
                ollama_api_base=ollama_api_base,
                ollama_embed_model=ollama_embed_model,
                enrich_with_llm=enrich_with_llm,
                verbose=False,
            )
            results.append(out)
        elapsed_seconds = time.perf_counter() - file_started_at
        _log(f"File {index}/{len(all_files)} done | elapsed={elapsed_seconds:.2f}s", enabled=verbose)

    _log(f"Folder ingest finished | processed={len(results)} files", enabled=verbose)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--pdf", type=Path, help="Single PDF or DOCX file")
    source_group.add_argument("--folder", type=Path, help="Directory containing PDF/DOCX files")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recursively scan --folder for PDF/DOCX files",
    )
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--ocr-endpoint", default=f"{OCR_API_BASE}/v1/chat/completions")
    parser.add_argument("--ocr-model-name", default=OCR_MODEL)
    parser.add_argument("--ocr-table-mode", choices=["pipe", "grid"], default="grid")
    parser.add_argument("--data-format", choices=["pdf", "docx", "pptx", "all"], default="all")
    parser.add_argument("--ollama-api-base", default=OLLAMA_API_BASE)
    parser.add_argument("--ollama-embed-model", default=OLLAMA_EMBED_MODEL)
    parser.add_argument(
        "--enrich-with-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable context generation during chunking",
    )
    args = parser.parse_args()

    out = run_ingest(
        collection=args.collection,
        pdf_path=args.pdf,
        folder_path=args.folder,
        qdrant_url=args.qdrant_url,
        ocr_endpoint=args.ocr_endpoint,
        ocr_model_name=args.ocr_model_name,
        ocr_table_mode=args.ocr_table_mode,
        data_format=args.data_format,
        ollama_api_base=args.ollama_api_base,
        ollama_embed_model=args.ollama_embed_model,
        enrich_with_llm=args.enrich_with_llm,
        recursive=args.recursive,
        verbose=True,
    )
    if isinstance(out, list):
        print(f"Done. Processed {len(out)} files.")
    else:
        print(
            "Done.\n"
            f"markdown: {out['markdown_path']}\n"
            f"chunks: {out['chunks_path']}\n"
            f"embeddings: {out['embeddings_path']}"
        )


if __name__ == "__main__":
    main()
