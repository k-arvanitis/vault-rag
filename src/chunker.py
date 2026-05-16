import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI
import os
import tiktoken

from src.config import CHUNK_LLM_API_BASE, CHUNK_LLM_MODEL, CHUNK_LLM_API_KEY
from src.prompts import CHUNK_CONTEXT_PROMPT, DOCUMENT_SUMMARY_PROMPT

REPO_ROOT = Path(__file__).resolve().parents[1]


def _debug(message: str) -> None:
    print(f"[CHUNKER][DEBUG] {message}")


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_text: str | None = None

    @property
    def token_count(self) -> int:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(self.content))


def generate_document_summary(client: OpenAI, model_name: str, markdown: str) -> str:
    """Generate a 3-5 sentence document-level summary for the whole document."""
    max_input_chars = int(os.getenv("SUMMARY_MAX_INPUT_CHARS", "6000"))
    truncated = markdown[:max_input_chars]
    prompt = DOCUMENT_SUMMARY_PROMPT.format(document=truncated)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Summary unavailable: {e}"


def contextualize_chunk(client: OpenAI, model_name: str, doc_context: str, chunk_content: str) -> str:
    max_output_tokens = int(os.getenv("CONTEXT_ENRICH_MAX_OUTPUT_TOKENS", "100"))
    max_input_chars = int(os.getenv("CONTEXT_ENRICH_MAX_INPUT_CHARS", "3000"))
    chunk_content = chunk_content[:max_input_chars]

    heading_match = re.search(r"(?m)^#{1,3}\s+(.+?)\s*$", chunk_content)
    table_match = re.search(r"(?i)\btable\s+\d+[^\n]*", chunk_content)
    heading_hint = (heading_match.group(1).strip() if heading_match else "none")
    table_hint = (table_match.group(0).strip() if table_match else "none")

    prompt = CHUNK_CONTEXT_PROMPT.format(
        heading_hint=heading_hint, table_hint=table_hint, chunk_content=chunk_content
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_output_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {e}"


def chunk_markdown(
    markdown: str,
    max_tokens: int = 1024,
    min_tokens: int = 256,
    min_chars: int = 300,
    chunk_overlap: int = 100,
    enrich_with_llm: bool = True,
    verbose: bool = True,
    output_dir: Path | None = None,
    file_name: str = "unknown",
    source_file: str = "",
) -> list[Chunk]:
    load_dotenv(override=True)

    api_base = os.getenv("CHUNK_LLM_API_BASE", CHUNK_LLM_API_BASE)
    model_name = os.getenv("CHUNK_LLM_MODEL", CHUNK_LLM_MODEL)
    api_key = os.getenv("CHUNK_LLM_API_KEY", CHUNK_LLM_API_KEY) or "no-key"
    client = OpenAI(base_url=api_base, api_key=api_key)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",
    )

    _figure_block_re = re.compile(
        r'\[FIGURE_START\].*?\[FIGURE_END\]', re.DOTALL
    )

    # Split at page-boundary markers emitted by pymupdf4llm before header splitting.
    # Without this, a ## heading from page N and a table/figure starting on page N+1
    # land in the same chunk — the reranker then sees the wrong lead content (the
    # heading) and demotes the chunk for table/figure queries.
    _page_marker_re = re.compile(r'(?=<!--\s*PAGE\s+\d+)', re.IGNORECASE)

    def _split_protecting_figures(content: str, metadata: dict) -> list[Chunk]:
        """Split content on token limit while keeping figure blocks atomic."""
        result: list[Chunk] = []
        parts = _figure_block_re.split(content)
        figures = _figure_block_re.findall(content)
        for idx, part in enumerate(parts):
            part = part.strip()
            if part:
                token_count = len(tokenizer.encode(part))
                if token_count <= max_tokens:
                    result.append(Chunk(content=part, metadata=dict(metadata)))
                else:
                    for sub_doc in text_splitter.create_documents([part], metadatas=[metadata]):
                        sub_content = sub_doc.page_content.strip()
                        if sub_content:
                            result.append(Chunk(content=sub_content, metadata=dict(sub_doc.metadata or {})))
            if idx < len(figures):
                result.append(Chunk(content=figures[idx].strip(), metadata=dict(metadata)))
        return result

    # Split by page boundaries first so each page is processed independently,
    # then split each page by Markdown headers.
    page_sections = _page_marker_re.split(text)

    chunks: list[Chunk] = []
    for page_text in page_sections:
        page_text = page_text.strip()
        if not page_text:
            continue
        for section in header_splitter.split_text(page_text):
            content = section.page_content.strip()
            if not content:
                continue

            metadata = dict(section.metadata or {})
            token_count = len(tokenizer.encode(content))
            if token_count <= max_tokens:
                chunks.append(Chunk(content=content, metadata=dict(metadata)))
            else:
                chunks.extend(_split_protecting_figures(content, metadata))

    def _has_section_header(content: str) -> bool:
        """Return True if the chunk starts with a Markdown section header (## or deeper).

        A named section is an intentional document boundary and must never be
        dissolved into a neighbour regardless of its token count.
        """
        stripped = content.lstrip()
        return stripped.startswith("##")

    compact_chunks: list[Chunk] = []
    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk.content))
        if (
            compact_chunks
            and chunk_tokens < min_tokens
            and not _has_section_header(chunk.content)
        ):
            compact_chunks[-1].content += "\n\n" + chunk.content
        else:
            compact_chunks.append(chunk)

    # Merge tiny chunks into previous or next chunk to reduce retrieval noise.
    # Chunks that open with a Markdown section header are kept as-is — the
    # document author drew an intentional boundary there.
    merged_chunks: list[Chunk] = []
    i = 0
    while i < len(compact_chunks):
        chunk = compact_chunks[i]
        chunk_chars = len(chunk.content)
        has_table = "[TABLE_START]" in chunk.content
        has_figure = "[FIGURE_START]" in chunk.content

        if has_table or has_figure or chunk_chars >= min_chars or _has_section_header(chunk.content):
            merged_chunks.append(chunk)
            i += 1
            continue

        if merged_chunks and "[TABLE_START]" not in merged_chunks[-1].content and "[FIGURE_START]" not in merged_chunks[-1].content:
            merged_chunks[-1].content += "\n\n" + chunk.content
            i += 1
            continue

        if i + 1 < len(compact_chunks) and "[TABLE_START]" not in compact_chunks[i + 1].content and "[FIGURE_START]" not in compact_chunks[i + 1].content:
            compact_chunks[i + 1].content = chunk.content + "\n\n" + compact_chunks[i + 1].content
            i += 1
            continue

        merged_chunks.append(chunk)
        i += 1

    compact_chunks = merged_chunks

    if enrich_with_llm:
        total_chunks = len(compact_chunks)
        if verbose:
            _debug(f"Enriching {total_chunks} chunks using Groq model: {model_name}")
        for i, chunk in enumerate(compact_chunks, start=1):
            context = contextualize_chunk(client, model_name, markdown, chunk.content)
            context = context.strip()
            chunk.metadata["context"] = context
            chunk.vector_text = f"CONTEXT: {context}\n\nCONTENT:\n{chunk.content}"
            if verbose and (i == 1 or i == total_chunks or i % 5 == 0):
                _debug(f"Enriched {i}/{total_chunks} chunks")

        # Document-level summary — prepended as a dedicated chunk so general
        # questions ("what is this paper about?") hit it directly in retrieval.
        if verbose:
            _debug("Generating document summary chunk")
        doc_summary = generate_document_summary(client, model_name, markdown)
        doc_id_match = re.search(r"doc_\d+", file_name)
        doc_id = doc_id_match.group(0) if doc_id_match else ""
        id_header = f"Document ID: {doc_id}\nFile: {file_name}\n\n" if doc_id else f"File: {file_name}\n\n"
        summary_content = f"## Document Summary\n\n{id_header}{doc_summary}"
        summary_chunk = Chunk(
            content=summary_content,
            metadata={
                "chunk_type": "document_summary",
                "doc_id": doc_id,
                "file_name": file_name,
                "source_file": source_file,
                "chunk_index": -1,
                "chunk_size_chars": len(summary_content),
                "token_count": len(tokenizer.encode(summary_content)),
            },
        )
        summary_chunk.vector_text = f"DOCUMENT SUMMARY:\n{id_header}{doc_summary}"
        compact_chunks.insert(0, summary_chunk)

    output_chunks = []
    for i, chunk in enumerate(compact_chunks):
        context = str(chunk.metadata.get("context", ""))
        metadata = dict(chunk.metadata or {})
        metadata["title"] = str(metadata.get("title", "") or "").strip()
        metadata["section"] = str(metadata.get("section", "") or "").strip()
        metadata["subsection"] = str(metadata.get("subsection", "") or "").strip()
        metadata["file_name"] = file_name or metadata.get("file_name", "unknown")
        metadata["source_file"] = source_file or metadata.get("source_file", "")
        metadata["chunk_index"] = i
        metadata["chunk_size_chars"] = len(chunk.content)
        metadata["token_count"] = chunk.token_count

        title = metadata["title"]
        section = metadata["section"]
        subsection = metadata["subsection"]
        header_lines = []
        if title:
            header_lines.append(f"# {title}")
        if section:
            header_lines.append(f"## {section}")
        if subsection:
            header_lines.append(f"### {subsection}")
        if header_lines and not chunk.content.lstrip().startswith("#"):
            chunk.content = "\n".join(header_lines) + "\n\n" + chunk.content

        if context and not chunk.vector_text:
            chunk.vector_text = f"CONTEXT: {context}\n\nCONTENT:\n{chunk.content}"
        chunk.metadata = metadata

        output_chunks.append(
            {
                "context": context,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "vector_text": chunk.vector_text,
            }
        )

    if output_dir is None:
        final_output_dir = REPO_ROOT / "data/output/chunks"
    else:
        final_output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    final_output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{Path(file_name).stem}_chunks.json" if file_name and file_name != "unknown" else "chunks.json"
    output_path = final_output_dir / output_name
    output_path.write_text(json.dumps(output_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"Saved: {output_path}")

    return compact_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/output/chandra/WHO East. Med. dsa1184.md",
        help="Input markdown file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/output/chunks",
        help="Directory for chunk JSON output",
    )
    parser.add_argument(
        "--enrich-with-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable context generation via LLM",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Merge chunks smaller than this character length into previous/next chunk.",
    )
    args = parser.parse_args()
    full_document_text = args.input.read_text(encoding="utf-8")
    chunk_markdown(
        full_document_text,
        enrich_with_llm=args.enrich_with_llm,
        min_chars=args.min_chars,
        output_dir=args.output_dir,
        file_name=args.input.name,
        source_file=str(args.input),
    )


if __name__ == "__main__":
    main()
