"""Tests for src/chunker.py.

Calls real chunker code with enrich_with_llm=False to avoid LLM network calls.
Uses a temporary directory for output to avoid polluting the data/ tree.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import tiktoken

from src.chunker import chunk_markdown, extract_literal_title
from src.config import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS


def _make_long_markdown(n_sections: int = 10, words_per_section: int = 400) -> str:
    """Build a markdown document large enough to produce multiple chunks."""
    lines = []
    for i in range(n_sections):
        lines.append(f"## Section {i + 1}\n")
        # ~400 words per section to ensure splitting
        lines.append(
            (" ".join([f"word{j}" for j in range(words_per_section)]) + "\n\n")
        )
    return "".join(lines)


def test_chunk_respects_max_tokens():
    md = _make_long_markdown()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    with tempfile.TemporaryDirectory() as tmp:
        chunks = chunk_markdown(
            md,
            max_tokens=CHUNK_MAX_TOKENS,
            min_tokens=CHUNK_MIN_TOKENS,
            enrich_with_llm=False,
            verbose=False,
            output_dir=Path(tmp),
            file_name="test.md",
        )
    assert chunks, "chunk_markdown should return at least one chunk"
    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk.content))
        assert token_count <= CHUNK_MAX_TOKENS + 50, (
            f"Chunk exceeds CHUNK_MAX_TOKENS ({CHUNK_MAX_TOKENS}): {token_count} tokens"
        )


def test_chunk_minimum_size():
    md = _make_long_markdown()
    tokenizer = tiktoken.get_encoding("cl100k_base")
    with tempfile.TemporaryDirectory() as tmp:
        chunks = chunk_markdown(
            md,
            max_tokens=CHUNK_MAX_TOKENS,
            min_tokens=CHUNK_MIN_TOKENS,
            enrich_with_llm=False,
            verbose=False,
            output_dir=Path(tmp),
            file_name="test.md",
        )
    # All chunks except the last should meet the minimum token size.
    for chunk in chunks[:-1]:
        token_count = len(tokenizer.encode(chunk.content))
        assert token_count >= CHUNK_MIN_TOKENS, (
            f"Non-final chunk is below CHUNK_MIN_TOKENS ({CHUNK_MIN_TOKENS}): {token_count} tokens"
        )


def test_chunk_output_has_required_fields():
    md = "## Introduction\n\nThis is a short test document with enough text to form a chunk.\n"
    with tempfile.TemporaryDirectory() as tmp:
        chunks = chunk_markdown(
            md,
            enrich_with_llm=False,
            verbose=False,
            output_dir=Path(tmp),
            file_name="test.md",
        )
    assert chunks, "chunk_markdown should return at least one chunk"
    for chunk in chunks:
        assert hasattr(chunk, "content"), "Chunk is missing 'content' attribute"
        assert hasattr(chunk, "metadata"), "Chunk is missing 'metadata' attribute"
        assert isinstance(chunk.content, str), "'content' should be a str"
        assert isinstance(chunk.metadata, dict), "'metadata' should be a dict"


def test_extract_literal_title_skips_page_markers_and_figures():
    md = (
        "<!-- PAGE 1 | pymupdf4llm -->\n\n"
        "## POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)\n\n"
        "**Board of Retirement Approved on September 4, 2024**\n\n"
        "[FIGURE_START]\nA logo.\n[FIGURE_END]\n"
    )
    assert (
        extract_literal_title(md)
        == "POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)"
    )


def test_extract_literal_title_empty_when_no_heading_near_top():
    md = "Just plain prose with no heading anywhere in the cover area.\n" * 5
    assert extract_literal_title(md) == ""


def test_every_chunk_gets_doc_id_from_file_name():
    """Only the document_summary chunk used to carry doc_id -- every regular
    narrative chunk had none, so doc-scoped retrieval filters silently matched
    nothing for most of the corpus. doc_id must be derived for every chunk."""
    md = "## Introduction\n\nThis is a short test document with enough text to form a chunk.\n"
    with tempfile.TemporaryDirectory() as tmp:
        chunks = chunk_markdown(
            md,
            enrich_with_llm=False,
            verbose=False,
            output_dir=Path(tmp),
            file_name="doc_042_example.md",
        )
    assert chunks
    for chunk in chunks:
        assert chunk.metadata.get("doc_id") == "doc_042"


def test_document_summary_chunk_carries_literal_title():
    """generate_document_summary() only ever produces an LLM paraphrase -- a
    title question has nothing verbatim to match against without this line,
    and generation tends to pick a more prominent section heading instead."""
    md = (
        "## POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)\n\n"
        "**Board of Retirement Approved on September 4, 2024**\n\n"
        "## V. Purchasing and Contracting Policy\n\n"
        "This is the actual policy body text, long enough to form its own chunk.\n"
    )
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch(
            "src.chunker.generate_document_summary",
            return_value="This document is a procurement policy.",
        ),
        patch("src.chunker.contextualize_chunk", return_value="context"),
    ):
        chunks = chunk_markdown(
            md,
            enrich_with_llm=True,
            verbose=False,
            output_dir=Path(tmp),
            file_name="doc_001_procurement_policy.md",
        )
    summary_chunk = next(
        c for c in chunks if c.metadata.get("chunk_type") == "document_summary"
    )
    assert (
        "Title: POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)"
        in summary_chunk.content
    )
