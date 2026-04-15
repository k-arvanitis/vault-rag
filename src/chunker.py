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

from src.config import OLLAMA_API_BASE, GENERATION_MODEL, CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _debug(message: str) -> None:
    print(f"[CHUNKER][DEBUG] {message}")


def _ollama_chat_base() -> str:
    return f"{OLLAMA_API_BASE.rstrip('/')}/v1"


def _resolve_chat_model(client: OpenAI, requested_model: str, verbose: bool) -> str:
    try:
        models = client.models.list()
        available = [m.id for m in getattr(models, "data", []) if getattr(m, "id", None)]
    except Exception as exc:
        if verbose:
            _debug(f"Could not list models from vLLM endpoint: {exc}")
        return requested_model

    if requested_model in available:
        return requested_model
    if available:
        if verbose:
            _debug(
                f"Requested model '{requested_model}' not found. "
                f"Falling back to '{available[0]}'."
            )
        return available[0]
    return requested_model


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_text: str | None = None

    @property
    def token_count(self) -> int:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(self.content))


_TABLE_ROWS_PER_CHUNK = int(os.getenv("TABLE_ROWS_PER_CHUNK", "20"))


def _parse_ascii_grid(table_text: str) -> tuple[list[str], list[list[str]]]:
    """Parse an ASCII +---+ grid table into (headers, data_rows).

    Returns headers as a list of column names and data_rows as a list of
    row value lists. Separator lines (+---+) and empty lines are skipped.
    """
    headers: list[str] = []
    rows: list[list[str]] = []
    past_header = False

    for line in table_text.splitlines():
        line = line.strip()
        if not line or line.startswith("+"):
            if "=" in line:
                past_header = True
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not past_header and not headers:
                headers = cells
            elif past_header:
                rows.append(cells)
    return headers, rows


def _ascii_table_to_chunks(
    table_block: str,
    metadata: dict,
    rows_per_chunk: int = _TABLE_ROWS_PER_CHUNK,
) -> list["Chunk"]:
    """Convert a [TABLE_START]...[TABLE_END] block into row-sentence chunks.

    Each chunk contains up to `rows_per_chunk` rows formatted as sentences,
    prefixed with the table title and part number so chunks are self-contained.
    """
    inner = re.sub(r"\[TABLE_START\]|\[TABLE_END\]", "", table_block).strip()
    # Strip ```text fences if present
    inner = re.sub(r"^```\w*\n?", "", inner, flags=re.MULTILINE)
    inner = re.sub(r"```\s*$", "", inner, flags=re.MULTILINE).strip()

    headers, rows = _parse_ascii_grid(inner)
    if not headers or not rows:
        # Unparseable — fall back to storing the whole block as-is
        return [Chunk(content=table_block.strip(), metadata=dict(metadata))]

    col_names = " | ".join(headers)
    total_parts = max(1, (len(rows) + rows_per_chunk - 1) // rows_per_chunk)
    title = metadata.get("subsection") or metadata.get("section") or metadata.get("title") or "Table"
    chunks: list[Chunk] = []

    for part_idx, start in enumerate(range(0, len(rows), rows_per_chunk), start=1):
        batch = rows[start: start + rows_per_chunk]
        sentences = []
        for row in batch:
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v]
            if pairs:
                sentences.append(" | ".join(pairs))

        header_line = f"Table: {title} | Part {part_idx} of {total_parts} | Columns: {col_names}"
        content = header_line + "\n\n" + "\n".join(sentences)
        chunk_meta = dict(metadata)
        chunk_meta["chunk_type"] = "pdf_table_rows"
        chunk_meta["table_part"] = part_idx
        chunk_meta["table_total_parts"] = total_parts
        chunks.append(Chunk(content=content, metadata=chunk_meta))

    return chunks


def contextualize_chunk(client: OpenAI, model_name: str, doc_context: str, chunk_content: str) -> str:
    max_output_tokens = int(os.getenv("CONTEXT_ENRICH_MAX_OUTPUT_TOKENS", "100"))
    heading_match = re.search(r"(?m)^#{1,3}\s+(.+?)\s*$", chunk_content)
    table_match = re.search(r"(?i)\btable\s+\d+[^\n]*", chunk_content)
    heading_hint = (heading_match.group(1).strip() if heading_match else "none")
    table_hint = (table_match.group(0).strip() if table_match else "none")

    prompt = f"""You are generating retrieval context for one chunk.
Write exactly one concise sentence (max 30 words) that describes the main topic, entities, and purpose of this chunk.
Be specific to this chunk only. Do not use generic filler.
Use these chunk-specific hints when relevant:
- heading_hint: {heading_hint}
- table_hint: {table_hint}
Return only the sentence.

Here is the chunk:
<chunk>
{chunk_content}
</chunk>
"""

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
    load_dotenv()

    api_base = os.getenv("CHUNK_LLM_API_BASE", _ollama_chat_base())
    requested_model_name = os.getenv("CHUNK_LLM_MODEL", GENERATION_MODEL)
    client = OpenAI(base_url=api_base, api_key="ollama")
    model_name = _resolve_chat_model(client, requested_model_name, verbose)

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

    chunks: list[Chunk] = []
    for section in header_splitter.split_text(text):
        content = section.page_content.strip()
        if not content:
            continue

        metadata = dict(section.metadata or {})
        token_count = len(tokenizer.encode(content))
        if token_count <= max_tokens:
            chunks.append(Chunk(content=content, metadata=dict(metadata)))
        else:
            sub_docs = text_splitter.create_documents([content], metadatas=[metadata])
            for sub_doc in sub_docs:
                sub_content = sub_doc.page_content.strip()
                if sub_content:
                    chunks.append(Chunk(content=sub_content, metadata=dict(sub_doc.metadata or {})))

    compact_chunks: list[Chunk] = []
    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk.content))
        if (
            compact_chunks
            and chunk_tokens < min_tokens
        ):
            compact_chunks[-1].content += "\n\n" + chunk.content
        else:
            compact_chunks.append(chunk)

    # Merge tiny chunks into previous or next chunk to reduce retrieval noise.
    merged_chunks: list[Chunk] = []
    i = 0
    while i < len(compact_chunks):
        chunk = compact_chunks[i]
        chunk_chars = len(chunk.content)
        has_table = "[TABLE_START]" in chunk.content

        if has_table or chunk_chars >= min_chars:
            merged_chunks.append(chunk)
            i += 1
            continue

        if merged_chunks and "[TABLE_START]" not in merged_chunks[-1].content:
            merged_chunks[-1].content += "\n\n" + chunk.content
            i += 1
            continue

        if i + 1 < len(compact_chunks) and "[TABLE_START]" not in compact_chunks[i + 1].content:
            compact_chunks[i + 1].content = chunk.content + "\n\n" + compact_chunks[i + 1].content
            i += 1
            continue

        merged_chunks.append(chunk)
        i += 1

    compact_chunks = merged_chunks

    if enrich_with_llm:
        total_chunks = len(compact_chunks)
        if verbose:
            _debug(f"Enriching {total_chunks} chunks using vLLM model: {model_name}")
        for i, chunk in enumerate(compact_chunks, start=1):
            context = contextualize_chunk(client, model_name, markdown, chunk.content)
            context = context.strip()
            chunk.metadata["context"] = context
            chunk.vector_text = f"CONTEXT: {context}\n\nCONTENT:\n{chunk.content}"
            if verbose and (i == 1 or i == total_chunks or i % 5 == 0):
                _debug(f"Enriched {i}/{total_chunks} chunks")

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
