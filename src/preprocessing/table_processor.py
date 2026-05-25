"""Post-process parsed markdown: convert ASCII grid tables to row sentences.

Runs after OCR + translation, before chunking. Saves the result to
data/output/processed/ so the inspector can show exactly what gets ingested.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROWS_PER_BLOCK = int(os.getenv("TABLE_ROWS_PER_CHUNK", "20"))


def _parse_ascii_grid(table_text: str) -> tuple[list[str], list[list[str]]]:
    """Parse +---+ ASCII grid into (headers, data_rows)."""
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


def _table_to_sentences(table_text: str, title: str) -> str:
    """Convert one ASCII grid table to a block of row sentences."""
    inner = re.sub(r"```\w*\n?|```", "", table_text).strip()
    headers, rows = _parse_ascii_grid(inner)

    if not headers or not rows:
        return table_text  # unparseable — keep as-is

    col_names = " | ".join(headers)
    total_parts = max(1, (len(rows) + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK)
    blocks: list[str] = []

    for part_idx, start in enumerate(range(0, len(rows), ROWS_PER_BLOCK), start=1):
        batch = rows[start: start + ROWS_PER_BLOCK]
        sentences = []
        for row in batch:
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v]
            if pairs:
                sentences.append(" | ".join(pairs))

        header_line = f"**Table: {title} | Part {part_idx} of {total_parts}**  \nColumns: {col_names}\n"
        blocks.append(header_line + "\n".join(sentences))

    return "\n\n".join(blocks)


def process_markdown(markdown: str) -> str:
    """Replace all ASCII grid tables in markdown with row-sentence blocks."""
    # Match contiguous blocks of lines starting with + or |
    def replace_table(match: re.Match) -> str:
        table_text = match.group(0)
        # Try to find a title from the nearest preceding heading
        return _table_to_sentences(table_text, title="Table")

    return re.sub(r"((?:^[+|][^\n]*\n)+)", replace_table, markdown, flags=re.MULTILINE)


def process_and_save(markdown: str, stem: str, output_dir: Path) -> tuple[str, Path]:
    """Process markdown and save to output_dir/{stem}.md. Returns (processed_text, path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = process_markdown(markdown)
    out_path = output_dir / f"{stem}.md"
    out_path.write_text(processed, encoding="utf-8")
    return processed, out_path
