"""Table ingestion — DuckDB for data, Qdrant for metadata.

Pipeline per file:
  1. LLM-assisted cleaning (excel_cleaner) → cleaned DataFrames
  2. Load cleaned DataFrames into DuckDB (persistent, cached)
  3. Upsert document_summary + per-sheet sheet_summary to Qdrant (discovery only)

The agent finds which sheet is relevant via Qdrant, then queries DuckDB directly.
No sheet_table or sheet_row chunks are stored in Qdrant.

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
    DUCKDB_PATH,
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
    """Build one summary chunk per sheet from headers + per-column sample values.

    No LLM — pure string concatenation. For each named column, collects up to
    _SAMPLES_PER_COL unique non-numeric text values sampled evenly across the
    entire file (not just the first rows) so that entity names deep in large
    sheets are still discoverable via vector or keyword search.
    """
    _SAMPLES_PER_COL = 20

    prefix_parts = [f"File: {file_name}", f"Sheet: {sheet_name}"]
    if sheet_title:
        prefix_parts.append(sheet_title)

    sample_parts: list[str] = []
    for col_idx, header in enumerate(headers):
        h = str(header).strip() if header is not None else ""
        if not h or h.lower().startswith("unnamed"):
            continue
        # Collect ALL unique text values for the column first,
        # then pick a distributed sample so rare values deep in the file
        # are still represented (e.g. an entity that appears once at row 2121).
        unique_vals: list[str] = []
        seen_col: set[str] = set()
        for row in data_rows:
            val = row[col_idx] if col_idx < len(row) else None
            if val is None:
                continue
            s = str(val).strip()
            if not s or _is_numeric(s) or s.lower() in SKIP_ROW_VALUES:
                continue
            if s not in seen_col:
                seen_col.add(s)
                unique_vals.append(s)
        if not unique_vals:
            continue
        # Pick _SAMPLES_PER_COL evenly distributed values from the unique list.
        n_u = len(unique_vals)
        if n_u <= _SAMPLES_PER_COL:
            samples = unique_vals
        else:
            step = n_u / _SAMPLES_PER_COL
            samples = [unique_vals[int(i * step)] for i in range(_SAMPLES_PER_COL)]
        sample_parts.append(f"{h}: {', '.join(samples)}")

    lines = [
        f"[{' | '.join(prefix_parts)}]",
        f"Sheet summary: {len(data_rows)} rows.",
        f"Columns: {', '.join(headers)}",
    ]
    if sample_parts:
        lines.append(f"Sample values — {' | '.join(sample_parts[:8])}")
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
    return _embed_batch([text])[0]


def _embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    import httpx
    client = OpenAI(
        base_url=f"{_ollama_base()}/v1",
        api_key="ollama",
        http_client=httpx.Client(timeout=300),
    )
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:_MAX_EMBED_CHARS] for t in texts[i : i + batch_size]]
        response = client.embeddings.create(model=_embed_model(), input=batch)
        results.extend([d.embedding for d in sorted(response.data, key=lambda d: d.index)])
    return results

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

def _upsert(collection: str, points: list[dict], batch_size: int = 200) -> None:
    base = _qdrant_url().rstrip("/")
    for i in range(0, len(points), batch_size):
        _qdrant_request("PUT", f"{base}/collections/{collection}/points?wait=true", {"points": points[i : i + batch_size]})


def _point_id(file_name: str, sheet_name: str, key_suffix: Any) -> int:
    key = f"{file_name}::{sheet_name}::{key_suffix}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_file_document_summary(
    file_name: str,
    sheet_summaries: list[str],
    collection: str,
) -> None:
    """Upsert one document_summary point per table file so summary-based routing finds it.

    The text aggregates all per-sheet summary headers (file, sheet, columns, categories)
    — no LLM call needed.
    """
    if not sheet_summaries:
        return
    parts = file_name.split("_")
    doc_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
    combined = f"## Document Summary\n\nDocument ID: {doc_id}\nFile: {file_name}\n\n" + "\n\n".join(sheet_summaries)
    vec = _embed(combined)
    point_id = _point_id(file_name, "__summary__", "document_summary")
    point = {
        "id": point_id,
        "vector": vec,
        "payload": {
            "content": combined,
            "source_type": "table",
            "metadata": {
                "source_file": file_name,
                "doc_id": doc_id,
                "chunk_type": "document_summary",
                "chunk_index": -1,
            },
        },
    }
    _upsert(collection, [point])


def _save_chunks_json(file_path: str, all_chunks: list[dict]) -> Path:
    """Save all row chunks for a file to data/output/chunks/<stem>_table_chunks.json."""
    stem = Path(file_path).stem
    out_dir = REPO_ROOT / "data" / "output" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_table_chunks.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _load_into_duckdb(file_path: str, file_name: str, verbose: bool) -> dict[str, list[str]]:
    """Clean file with LLM and load all sheets into DuckDB. Returns {sheet_name: [columns]}."""
    import duckdb
    from src.preprocessing.excel_cleaner import process_file
    from src.duckdb_store import _table_name, _normalize_dates, _make_groq_llm_fn

    db_path = os.getenv("DUCKDB_PATH", DUCKDB_PATH)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    llm_fn = _make_groq_llm_fn()

    try:
        sheet_results = process_file(file_path, llm_fn)
    except Exception as e:
        print(f"  [WARNING] excel_cleaner failed for '{file_name}': {e}. Skipping DuckDB load.")
        con.close()
        return {}

    sheet_columns: dict[str, list[str]] = {}
    for sheet_name, sr in sheet_results.items():
        tname = _table_name(file_name, sheet_name)
        existing = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [tname]
        ).fetchone()[0]
        if existing:
            if verbose:
                print(f"  DuckDB cache hit: {tname}")
        else:
            if verbose:
                print(f"  DuckDB loading: {tname} ({len(sr.df)} rows)")
            normalized = _normalize_dates(sr.df)
            con.register("_tmp_df", normalized)
            con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM _tmp_df')
            con.unregister("_tmp_df")
        cols = [row[0] for row in con.execute(f'DESCRIBE "{tname}"').fetchall()]
        sheet_columns[sheet_name] = cols
    con.close()
    return sheet_columns


def _build_sheet_summary_point(
    file_name: str,
    sheet_name: str,
    columns: list[str],
    description: str,
    collection: str,
) -> None:
    """Upsert one sheet_summary point to Qdrant for discovery.

    Includes the DuckDB table name so the agent can map directly to the right table.
    """
    from src.duckdb_store import _table_name
    tname = _table_name(file_name, sheet_name)
    parts = file_name.split("_")
    doc_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]

    content = (
        f"[File: {file_name} | Sheet: {sheet_name}]\n"
        f"DuckDB table: {tname}\n"
        f"Document ID: {doc_id}\n"
        f"Columns: {', '.join(columns)}\n"
    )
    if description:
        content += f"Description: {description}\n"

    vec = _embed(content)
    point_id = _point_id(file_name, sheet_name, "sheet_summary")
    _ensure_collection(collection, len(vec))
    _upsert(collection, [{
        "id": point_id,
        "vector": vec,
        "payload": {
            "content": content,
            "source_type": "table",
            "metadata": {
                "source_file": file_name,
                "doc_id": doc_id,
                "sheet_name": sheet_name,
                "duckdb_table": tname,
                "chunk_type": "sheet_summary",
                "chunk_index": -1,
            },
        },
    }])


def ingest_table_rows(
    file_path: str,
    collection: str = "documents_chunks",
    verbose: bool = True,
) -> Path:
    """Ingest a table file: clean with LLM → DuckDB, metadata → Qdrant.

    DuckDB receives the LLM-cleaned DataFrames for all sheets.
    Qdrant receives one document_summary + one sheet_summary per sheet (no row chunks).
    Returns the path to the saved chunks JSON (summaries only).
    """
    file_name = Path(file_path).name
    all_chunks: list[dict] = []

    # Delete any existing Qdrant points for this file
    base = _qdrant_url().rstrip("/")
    try:
        _qdrant_request("POST", f"{base}/collections/{collection}/points/delete?wait=true", {
            "filter": {"must": [{"key": "metadata.source_file", "match": {"value": file_name}}]}
        })
        if verbose:
            print(f"  Deleted existing Qdrant points for '{file_name}'")
    except Exception:
        pass

    # Step 1: LLM cleaning + DuckDB load
    if verbose:
        print("  Cleaning and loading into DuckDB...")
    sheet_columns = _load_into_duckdb(file_path, file_name, verbose)

    # Step 2: generate sheet_summary points for Qdrant
    # Fall back to heuristic headers if DuckDB load failed for a sheet
    raw_sheets = load_sheets(file_path)
    sheet_summary_texts: list[str] = []

    for sheet_name, rows in raw_sheets.items():
        if any(kw in sheet_name.lower() for kw in SKIP_SHEET_KEYWORDS):
            continue
        if len(rows) < 2:
            continue

        # Use cleaned column names from DuckDB if available, else heuristic
        if sheet_name in sheet_columns:
            columns = sheet_columns[sheet_name]
            description = ""
        else:
            header_idx = _find_header_row(rows)
            headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[header_idx])]
            units_idx = _detect_units_row(rows, header_idx)
            if units_idx is not None:
                headers = _merge_units_into_headers(headers, rows[units_idx])
            subheader_rows, data_start = _collect_subheaders(rows, header_idx + 1)
            columns = _qualify_headers(headers, subheader_rows)
            description = ""

        sheet_title = _extract_sheet_title(rows, _find_header_row(rows))
        data_rows = rows[_find_header_row(rows) + 1:]

        if verbose:
            print(f"  {sheet_name}: {len(data_rows)} rows, {len(columns)} columns")

        summary_text = sheet_summary_text(file_name, sheet_name, columns, data_rows, sheet_title)
        sheet_summary_texts.append(summary_text)
        all_chunks.append({
            "content": summary_text,
            "metadata": {
                "source_file": file_name,
                "sheet_name": sheet_name,
                "sheet_title": sheet_title,
                "chunk_type": "sheet_summary",
            },
        })

        _build_sheet_summary_point(file_name, sheet_name, columns, description, collection)
        if verbose:
            print("    → sheet_summary upserted to Qdrant")

        # Save markdown for inspector UI
        header_idx = _find_header_row(rows)
        units_idx = _detect_units_row(rows, header_idx)
        raw_headers = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(rows[header_idx])]
        if units_idx is not None:
            raw_headers = _merge_units_into_headers(raw_headers, rows[units_idx])
            subheader_rows, data_start = _collect_subheaders(rows, units_idx + 1)
        else:
            subheader_rows, data_start = _collect_subheaders(rows, header_idx + 1)
        full_sheet_md = sheet_to_chunk(file_name, sheet_name, raw_headers, rows[data_start:], sheet_title, subheader_rows=subheader_rows)
        md_out_dir = REPO_ROOT / "data" / "output" / "table_markdowns"
        md_out_dir.mkdir(parents=True, exist_ok=True)
        safe_sheet = sheet_name.replace("/", "_").replace("\\", "_")
        (md_out_dir / f"{Path(file_path).stem}__{safe_sheet}.md").write_text(full_sheet_md, encoding="utf-8")

    # Step 3: document_summary
    _build_file_document_summary(file_name, sheet_summary_texts, collection)
    if verbose:
        print("  document_summary upserted to Qdrant")

    out_path = _save_chunks_json(file_path, all_chunks)
    if verbose:
        print(f"\nSaved {len(all_chunks)} summary chunks → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest table file: LLM cleaning → DuckDB, metadata → Qdrant.")
    parser.add_argument("file_path", help="Path to .xlsx or .csv file")
    parser.add_argument("--collection", default="documents_chunks")
    args = parser.parse_args()
    ingest_table_rows(args.file_path, collection=args.collection)


if __name__ == "__main__":
    main()
