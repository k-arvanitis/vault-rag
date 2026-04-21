"""Table row → text chunk ingestion. No LLM, no SQL, no schema extraction.

Each row becomes a text sentence and is embedded + stored in the same
documents_chunks Qdrant collection as PDFs, so the existing retriever
finds table data automatically.

Usage:
    python -m src.ingest_table_rows data/myfile.xlsx
    python -m src.ingest_table_rows data/myfile.csv --collection documents_chunks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from openai import OpenAI

from src.ingest_tables import load_sheets
from src.config import (
    OLLAMA_API_BASE as _DEFAULT_OLLAMA_BASE,
    OLLAMA_EMBED_MODEL as _DEFAULT_EMBED_MODEL,
    QDRANT_URL as _DEFAULT_QDRANT_URL,
    MAX_ROWS_PER_CHUNK,
    MAX_CHARS_PER_TABLE_CHUNK,
    SKIP_SHEET_KEYWORDS,
    SKIP_ROW_VALUES,
    NO_DATA_TOKENS,
)


# ---------------------------------------------------------------------------
# Config helpers (reuse same env vars as the rest of the project)
# ---------------------------------------------------------------------------

def _ollama_base() -> str:
    return os.getenv("OLLAMA_API_BASE", _DEFAULT_OLLAMA_BASE).rstrip("/")

def _embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", _DEFAULT_EMBED_MODEL)

def _qdrant_url() -> str:
    return os.getenv("QDRANT_URL", _DEFAULT_QDRANT_URL)


# ---------------------------------------------------------------------------
# Header detection (heuristic, no LLM)
# ---------------------------------------------------------------------------

def _find_header_row(rows: list[list[Any]]) -> int:
    """Return index of the best header row.

    Skips:
    - empty rows
    - rows where all non-empty values are identical (merged title cells)
    - rows where most values are numeric

    Picks the first row with multiple *distinct* text values.
    """
    for i, row in enumerate(rows[:20]):
        non_empty = [str(v).strip() for v in row if v is not None and str(v).strip()]
        if len(non_empty) < 2:
            continue
        # Skip title rows: cells are mostly identical (forward-filled merged cells)
        # Real header rows have at least 3 distinct values
        if len(set(non_empty)) < 3:
            continue
        text_count = sum(1 for v in non_empty if not _is_numeric(v))
        if text_count / len(non_empty) >= 0.5:
            return i
    return 0


def _is_unit_like(v: str) -> bool:
    """Return True if the value looks like a unit annotation, e.g. '(kt)', 'Gg', '%', 'TJ'."""
    v = v.strip()
    if not v:
        return True  # empty cells are fine in a units row
    if len(v) > 15:
        return False
    if v.startswith("(") and v.endswith(")"):
        return True
    unit_tokens = {"kt", "gg", "tg", "pg", "tj", "gj", "mj", "pj", "%", "t", "mg",
                   "kg", "g", "mw", "gw", "tw", "kwh", "mwh", "gwh", "twh", "na", "no",
                   "yes", "n/a", "n.a.", "-", "–"}
    return v.lower() in unit_tokens


def _merge_units_into_headers(headers: list[str], units_row: list[Any]) -> list[str]:
    """Append unit annotations from the units row into the header names."""
    merged = []
    for header, unit_val in zip(headers, units_row):
        unit_str = str(unit_val).strip() if unit_val is not None else ""
        if unit_str and unit_str.lower() not in {"na", "no", "-", "–", "n/a"}:
            # normalise: ensure parens
            unit_str = unit_str if unit_str.startswith("(") else f"({unit_str})"
            merged.append(f"{header} {unit_str}".strip())
        else:
            merged.append(header)
    return merged


def _detect_units_row(rows: list[list[Any]], header_idx: int) -> int | None:
    """If the row immediately after the header is a units row, return its index."""
    next_idx = header_idx + 1
    if next_idx >= len(rows):
        return None
    row = rows[next_idx]
    non_empty = [str(v).strip() for v in row if v is not None and str(v).strip()]
    if not non_empty:
        return None
    unit_like = sum(1 for v in non_empty if _is_unit_like(v))
    if unit_like / len(non_empty) >= 0.6:
        return next_idx
    return None

def _is_numeric(v: Any) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Row → text
# ---------------------------------------------------------------------------

def _extract_sheet_title(rows: list[list[Any]], header_idx: int) -> str:
    """Collect all distinct text values from pre-header rows (rows 0..header_idx-1).

    These rows are typically merged title cells that contain things like:
    'GREECE', '2019', 'TABLE 1 SECTORAL REPORT FOR ENERGY', etc.
    All distinct values are joined with ' | ' so nothing is lost.
    Returns empty string if nothing useful is found.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for row in rows[:header_idx]:
        for v in row:
            s = str(v).strip() if v is not None else ""
            if s and not _is_numeric(s) and len(s) > 2 and s not in seen:
                seen.add(s)
                parts.append(s)
    return " | ".join(parts)


def sheet_summary_text(
    file_name: str,
    sheet_name: str,
    headers: list[str],
    data_rows: list[list[Any]],
    sheet_title: str = "",
) -> str:
    """Build one summary chunk per sheet from headers + first-column category values.

    No LLM — pure string concatenation. Catches queries like
    'what sheets have CO2 data?' or 'which table covers energy industries?'
    """
    prefix_parts = [f"File: {file_name}", f"Sheet: {sheet_name}"]
    if sheet_title:
        prefix_parts.append(sheet_title)

    # Unique non-numeric values from first column = category index of the table
    seen: set[str] = set()
    categories: list[str] = []
    for row in data_rows:
        v = row[0] if row else None
        s = str(v).strip() if v is not None else ""
        if s and not _is_numeric(s) and s.lower() not in SKIP_ROW_VALUES and s not in seen:
            seen.add(s)
            categories.append(s)

    lines = [
        f"[{' | '.join(prefix_parts)}]",
        f"Sheet summary: {len(data_rows)} rows.",
        f"Columns: {', '.join(headers)}",
    ]
    if categories:
        lines.append(f"Categories: {', '.join(categories[:20])}")
    return "\n".join(lines)


def _qualify_headers(headers: list[str], subheader_rows: list[list[Any]]) -> list[str]:
    """Combine duplicate column names with their subheader values.

    E.g. ['EMISSIONS', 'EMISSIONS', 'EMISSIONS'] + subheaders with ['CO2(1)', 'CH4', 'N2O']
    → ['EMISSIONS', 'EMISSIONS (CO2(1))', 'EMISSIONS (CH4)', 'EMISSIONS (N2O)']
    """
    if not subheader_rows:
        return headers
    from collections import Counter
    counts = Counter(h for h in headers if h and not h.startswith("col_"))
    duplicate_headers = {h for h, c in counts.items() if c > 1}
    if not duplicate_headers:
        return headers
    qualified = []
    for i, h in enumerate(headers):
        if h not in duplicate_headers:
            qualified.append(h)
            continue
        sub_parts = []
        for sh_row in subheader_rows:
            val = str(sh_row[i]).strip() if i < len(sh_row) and sh_row[i] is not None else ""
            if val and val not in {"-", "–"}:
                sub_parts.append(val)
        qualified.append(f"{h} ({' '.join(sub_parts)})" if sub_parts else h)
    return qualified


def _collect_subheaders(rows: list[list[Any]], after_idx: int) -> tuple[list[list[Any]], int]:
    """Collect sub-header rows that appear after the main header/units row.

    Sub-header rows have an empty first cell and contain only text annotations
    (e.g. 'CO2(1)', 'Amount captured', '(TJ)') — no numeric data values.
    Stops as soon as a row contains numeric data (real data row).
    Returns (subheader_rows, first_data_row_idx).
    """
    subheaders = []
    i = after_idx
    while i < len(rows):
        row = rows[i]
        first = str(row[0]).strip() if row and row[0] is not None else ""
        # Stop if first cell has a real label
        if first and not _is_unit_like(first):
            break
        non_empty = [str(v).strip() for v in row if v is not None and str(v).strip()]
        if not non_empty:
            break
        # Stop if the row contains numeric values — it's a data row, not annotation
        numeric_count = sum(1 for v in non_empty if _is_numeric(v))
        if numeric_count > 0:
            break
        subheaders.append(row)
        i += 1
    return subheaders, i


def _format_cell(v: Any) -> str:
    """Format one cell value for markdown/text chunk output."""
    s = str(v).strip() if v is not None else ""
    if not s:
        return ""
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
        if len(s) <= 12:
            return s
        return f"{f:.6g}"
    except ValueError:
        return s


def _should_skip_data_row(row: list[Any]) -> bool:
    """Return True when a table row is documentation or contains only no-data values."""
    first_val = str(row[0]).strip().lower() if row and row[0] is not None else ""
    if first_val in SKIP_ROW_VALUES:
        return True
    data_vals = [str(v).strip().lower() for v in row[1:] if v is not None and str(v).strip()]
    return bool(data_vals) and all(v in NO_DATA_TOKENS for v in data_vals)


def sheet_to_markdown(
    headers: list[str],
    data_rows: list[list[Any]],
    subheader_rows: list[list[Any]] | None = None,
) -> str:
    """Render data rows as a markdown table.

    Skips rows that are all NO/NA/NE (no-data) and documentation rows.
    Truncates numbers to 4 significant figures to keep chunks compact.
    Sub-header rows (column annotations, units) are prepended after the separator
    so every chunk is self-contained.
    """
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    if subheader_rows:
        for sh_row in subheader_rows:
            cells = [_format_cell(sh_row[i]) if i < len(sh_row) else "" for i in range(len(headers))]
            lines.append("| " + " | ".join(cells) + " |")

    for row in data_rows:
        if _should_skip_data_row(row):
            continue
        cells = [_format_cell(row[i]) if i < len(row) else "" for i in range(len(headers))]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def row_to_chunk(
    file_name: str,
    sheet_name: str,
    headers: list[str],
    row: list[Any],
    sheet_title: str = "",
) -> str:
    """Build a compact row-level chunk using the resolved table context."""
    prefix_parts = [f"File: {file_name}", f"Sheet: {sheet_name}"]
    if sheet_title:
        prefix_parts.append(sheet_title)

    row_label = _format_cell(row[0]) if row else ""
    pairs = []
    for idx, header in enumerate(headers):
        if idx >= len(row):
            continue
        value = _format_cell(row[idx])
        if not value:
            continue
        pairs.append(f"{header}: {value}")

    lines = [f"[{' | '.join(prefix_parts)}]"]
    if row_label:
        lines.append(f"Row summary: {row_label}")
    lines.append(" | ".join(pairs))
    return "\n".join(lines)


def sheet_to_chunk(
    file_name: str,
    sheet_name: str,
    headers: list[str],
    data_rows: list[list[Any]],
    sheet_title: str = "",
    subheader_rows: list[list[Any]] | None = None,
) -> str:
    """Build a single chunk for a sheet: description header + full markdown table.

    The description enables semantic retrieval; the markdown table lets the LLM
    read exact values without needing SQL or row-by-row search.
    """
    description = sheet_summary_text(file_name, sheet_name, headers, data_rows, sheet_title)
    table_md = sheet_to_markdown(headers, data_rows, subheader_rows=subheader_rows)
    return description + "\n\n" + table_md


# ---------------------------------------------------------------------------
# Embed + store
# ---------------------------------------------------------------------------

_MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "24000"))  # BGE-M3 supports 8192 tokens (~24000 chars)


def _embed(text: str) -> list[float]:
    import httpx
    client = OpenAI(
        base_url=f"{_ollama_base()}/v1",
        api_key="ollama",
        http_client=httpx.Client(timeout=300),
    )
    return client.embeddings.create(model=_embed_model(), input=text[:_MAX_EMBED_CHARS]).data[0].embedding

def _qdrant_request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except HTTPError as e:
        raise RuntimeError(f"Qdrant {e.code}: {e.read().decode()}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot connect to Qdrant: {e}") from e

def _ensure_collection(collection: str, dim: int) -> None:
    base = _qdrant_url().rstrip("/")
    try:
        _qdrant_request("GET", f"{base}/collections/{collection}")
    except RuntimeError:
        _qdrant_request("PUT", f"{base}/collections/{collection}", {
            "vectors": {"size": dim, "distance": "Cosine"},
            "sparse_vectors": {"sparse": {}},
        })
        print(f"  Created collection '{collection}'")

def _upsert(collection: str, points: list[dict]) -> None:
    base = _qdrant_url().rstrip("/")
    _qdrant_request("PUT", f"{base}/collections/{collection}/points?wait=true", {"points": points})


def _build_vector_field(text: str, sparse_embedder: Any) -> Any:
    """Build dense or dense+sparse vectors for one chunk of table text."""
    vec = _embed(text)
    if sparse_embedder is None:
        return vec
    indices, values = sparse_embedder.embed(text)
    return {"": vec, "sparse": {"indices": indices, "values": values}}


def _build_table_point(
    file_name: str,
    sheet_name: str,
    chunk_text: str,
    chunk_type: str,
    part_idx: int,
    num_rows: int,
    vector_field: Any,
) -> dict[str, Any]:
    """Build one Qdrant point for a table-derived chunk."""
    prefix = "sheet" if chunk_type == "sheet_table" else "row"
    return {
        "id": _point_id(file_name, sheet_name, f"{prefix}:{part_idx}"),
        "vector": vector_field,
        "payload": {
            "content": chunk_text,
            "source_type": "table",
            "metadata": {
                "source_file": file_name,
                "sheet_name": sheet_name,
                "chunk_type": chunk_type,
                "part": part_idx,
                "num_rows": num_rows,
            },
        },
    }

def _point_id(file_name: str, sheet_name: str, key_suffix: Any) -> int:
    key = f"{file_name}::{sheet_name}::{key_suffix}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _save_chunks_json(file_path: str, all_chunks: list[dict]) -> Path:
    """Save all row chunks for a file to data/output/chunks/<stem>_table_chunks.json."""
    stem = Path(file_path).stem
    out_dir = REPO_ROOT / "data" / "output" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_table_chunks.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def ingest_table_rows(
    file_path: str,
    collection: str = "documents_chunks",
    batch_size: int = 50,
    verbose: bool = True,
) -> Path:
    """Ingest all sheets from an Excel/CSV file as row-text chunks into Qdrant.

    Metadata (country, year, report name, etc.) is auto-extracted from the
    pre-header title rows in each sheet — no manual input needed.

    Also saves all chunks to data/output/chunks/<stem>_table_chunks.json.
    Returns the path to that file.
    """
    file_name = Path(file_path).name
    sheets = load_sheets(file_path)
    all_chunks: list[dict] = []  # accumulate across all sheets for JSON output

    # Delete any existing points for this file before re-ingesting (deduplication)
    base = _qdrant_url().rstrip("/")
    try:
        _qdrant_request("POST", f"{base}/collections/{collection}/points/delete?wait=true", {
            "filter": {"must": [{"key": "metadata.source_file", "match": {"value": file_name}}]}
        })
        if verbose:
            print(f"  Deleted existing points for '{file_name}'")
    except Exception:
        pass  # collection may not exist yet — that's fine

    for sheet_name, rows in sheets.items():
        if any(kw in sheet_name.lower() for kw in SKIP_SHEET_KEYWORDS):
            if verbose:
                print(f"  Skipping '{sheet_name}'")
            continue
        if len(rows) < 2:
            continue

        header_idx = _find_header_row(rows)
        headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[header_idx])]

        # Extract descriptive title from skipped title rows (before the header)
        sheet_title = _extract_sheet_title(rows, header_idx)

        # Merge units row (e.g. "(kt)", "Gg") into header names if present
        units_idx = _detect_units_row(rows, header_idx)
        if units_idx is not None:
            headers = _merge_units_into_headers(headers, rows[units_idx])
            subheader_rows, data_start = _collect_subheaders(rows, units_idx + 1)
        else:
            subheader_rows, data_start = _collect_subheaders(rows, header_idx + 1)
        data_rows = rows[data_start:]

        if verbose:
            title_info = f" | title='{sheet_title[:40]}'" if sheet_title else ""
            print(f"  {sheet_name}: {len(data_rows)} rows, headers={headers[:4]}...{title_info}")

        # Qualify duplicate column names with subheader content for better embeddings
        qualified_headers = _qualify_headers(headers, subheader_rows)

        # Split into chunks of MAX_ROWS_PER_CHUNK rows; then further split any window
        # whose rendered text exceeds MAX_CHARS_PER_TABLE_CHUNK (wide sheets have few rows
        # but many columns, so row count alone doesn't bound chunk size).
        def _split_window(rows: list) -> list[list]:
            if not rows:
                return [rows]
            text = sheet_to_chunk(file_name, sheet_name, qualified_headers, rows, sheet_title, subheader_rows=subheader_rows)
            if len(text) <= MAX_CHARS_PER_TABLE_CHUNK or len(rows) == 1:
                return [rows]
            mid = len(rows) // 2
            return _split_window(rows[:mid]) + _split_window(rows[mid:])

        coarse_windows = [
            data_rows[i:i + MAX_ROWS_PER_CHUNK]
            for i in range(0, max(1, len(data_rows)), MAX_ROWS_PER_CHUNK)
        ]
        row_windows = []
        for w in coarse_windows:
            row_windows.extend(_split_window(w))

        try:
            from src.sparse_embedder import get_sparse_embedder
            sparse_embedder = get_sparse_embedder()
        except Exception as e:
            print(f"  [WARNING] Sparse embedder failed to load: {e}. Storing dense vectors only.")
            sparse_embedder = None

        _ensure_collection_done = False
        points: list[dict[str, Any]] = []
        # Save full sheet as markdown before chunking
        full_sheet_md = sheet_to_chunk(file_name, sheet_name, qualified_headers, data_rows, sheet_title, subheader_rows=subheader_rows)
        md_out_dir = REPO_ROOT / "data" / "output" / "table_markdowns"
        md_out_dir.mkdir(parents=True, exist_ok=True)
        safe_sheet = sheet_name.replace("/", "_").replace("\\", "_")
        md_out_path = md_out_dir / f"{Path(file_path).stem}__{safe_sheet}.md"
        md_out_path.write_text(full_sheet_md, encoding="utf-8")

        for part_idx, window in enumerate(row_windows):
            chunk_text = sheet_to_chunk(file_name, sheet_name, qualified_headers, window, sheet_title, subheader_rows=subheader_rows)

            all_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source_file": file_name,
                    "sheet_name": sheet_name,
                    "sheet_title": sheet_title,
                    "chunk_type": "sheet_table",
                    "part": part_idx,
                    "num_rows": len(window),
                },
            })

            try:
                vector_field = _build_vector_field(chunk_text, sparse_embedder)
            except Exception as e:
                print(f"  [WARNING] Sparse embedding failed for part {part_idx}: {e}. Using dense only.")
                vector_field = _embed(chunk_text)
            if not _ensure_collection_done:
                vec_size = len(vector_field[""]) if isinstance(vector_field, dict) else len(vector_field)
                _ensure_collection(collection, vec_size)
                _ensure_collection_done = True
            points.append(_build_table_point(
                file_name=file_name,
                sheet_name=sheet_name,
                chunk_text=chunk_text,
                chunk_type="sheet_table",
                part_idx=part_idx,
                num_rows=len(window),
                vector_field=vector_field,
            ))

        row_chunk_count = 0
        for row_idx, row in enumerate(data_rows):
            if _should_skip_data_row(row):
                continue
            chunk_text = row_to_chunk(file_name, sheet_name, qualified_headers, row, sheet_title)
            all_chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source_file": file_name,
                    "sheet_name": sheet_name,
                    "sheet_title": sheet_title,
                    "chunk_type": "sheet_row",
                    "part": row_idx,
                    "num_rows": 1,
                },
            })

            try:
                vector_field = _build_vector_field(chunk_text, sparse_embedder)
            except Exception as e:
                print(f"  [WARNING] Sparse embedding failed for row {row_idx}: {e}. Using dense only.")
                vector_field = _embed(chunk_text)
            if not _ensure_collection_done:
                vec_size = len(vector_field[""]) if isinstance(vector_field, dict) else len(vector_field)
                _ensure_collection(collection, vec_size)
                _ensure_collection_done = True
            points.append(_build_table_point(
                file_name=file_name,
                sheet_name=sheet_name,
                chunk_text=chunk_text,
                chunk_type="sheet_row",
                part_idx=row_idx,
                num_rows=1,
                vector_field=vector_field,
            ))
            row_chunk_count += 1

        if points:
            _upsert(collection, points)

        n_parts = len(row_windows)
        if verbose:
            print(f"    → {n_parts} table chunk(s) + {row_chunk_count} row chunk(s) upserted ({len(data_rows)} rows)")

    out_path = _save_chunks_json(file_path, all_chunks)
    if verbose:
        print(f"\nSaved {len(all_chunks)} chunks → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest table rows as text chunks into Qdrant.")
    parser.add_argument("file_path", help="Path to .xlsx or .csv file")
    parser.add_argument("--collection", default="documents_chunks")
    args = parser.parse_args()
    ingest_table_rows(args.file_path, collection=args.collection)


if __name__ == "__main__":
    main()
