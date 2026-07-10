"""Tests for src/chunker.py.

Calls real chunker code with enrich_with_llm=False to avoid LLM network calls.
Uses a temporary directory for output to avoid polluting the data/ tree.
"""
import tempfile
from pathlib import Path

import tiktoken

from src.chunker import chunk_markdown
from src.config import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS


def _make_long_markdown(n_sections: int = 10, words_per_section: int = 400) -> str:
    """Build a markdown document large enough to produce multiple chunks."""
    lines = []
    for i in range(n_sections):
        lines.append(f"## Section {i + 1}\n")
        # ~400 words per section to ensure splitting
        lines.append((" ".join([f"word{j}" for j in range(words_per_section)]) + "\n\n"))
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
