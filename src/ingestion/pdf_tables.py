"""Load tables embedded in parsed PDF/Office markdown into DuckDB.

PDF-embedded tables (HTML ``<table>`` blocks emitted by the OCR/parse step, or
GitHub-flavoured markdown pipe tables) are loaded into DuckDB as ``doc_NNN_*``
tables — the same store the Excel/CSV path uses — and a per-table summary is
upserted to Qdrant. This lets the ``query_excel`` SQL agent answer aggregation
and exact-lookup questions (``SUM``, ``COUNT``, exact row filters) that an LLM
cannot do reliably over retrieved free text.

General by design: runs for every ingested document, keyed only on table shape,
never on a specific file or column name.

Calls: pandas (table parsing), duckdb (load), src.duckdb_store (_table_name,
_normalize_dates), src.ingest_table_rows (sheet/document summary upserts).
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DUCKDB_PATH
from src.duckdb_store import _normalize_dates, _table_name

# Tables smaller than this are headers/decorative layout, not data worth querying.
_MIN_ROWS = 3
_MIN_COLS = 2


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-empty columns and give blank/duplicate headers stable names."""
    df = df.dropna(axis=1, how="all")
    seen: dict[str, int] = {}
    names: list[str] = []
    for i, col in enumerate(df.columns):
        name = re.sub(r"\s+", " ", str(col)).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df = df.copy()
    df.columns = names
    return df


def _parse_markdown_pipe_tables(markdown: str) -> list[pd.DataFrame]:
    """Parse GitHub-flavoured ``| a | b |`` pipe tables (no HTML) into DataFrames."""
    tables: list[pd.DataFrame] = []
    block: list[str] = []
    for line in markdown.splitlines() + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            block.append(line.strip())
            continue
        if len(block) >= 3:  # header + separator + >=1 row
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in block]
            sep = rows[1]
            if all(set(c) <= set("-: ") and c for c in sep):
                header, data = rows[0], rows[2:]
                width = len(header)
                data = [r for r in data if len(r) == width]
                if data:
                    tables.append(pd.DataFrame(data, columns=header))
        block = []
    return tables


def _extract_tables(markdown: str) -> list[pd.DataFrame]:
    """Return every substantial table in the markdown (HTML first, then pipe)."""
    tables: list[pd.DataFrame] = []
    if "<table" in markdown.lower():
        try:
            tables.extend(pd.read_html(StringIO(markdown)))
        except ValueError:
            pass
    if not tables:
        tables.extend(_parse_markdown_pipe_tables(markdown))
    return [t for t in tables if t.shape[0] >= _MIN_ROWS and t.shape[1] >= _MIN_COLS]


def load_tables_to_duckdb(
    doc_name: str, markdown: str, db_path: str | None = None, verbose: bool = True
) -> list[tuple[str, pd.DataFrame]]:
    """Load every substantial table in ``markdown`` into DuckDB.

    Returns the list of (sheet_name, cleaned DataFrame) loaded, so the caller can
    build Qdrant discovery summaries. Each table becomes ``doc_NNN_..._table_K``.
    """
    tables = _extract_tables(markdown)
    if not tables:
        return []
    db = db_path or DUCKDB_PATH
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db)
    loaded: list[tuple[str, pd.DataFrame]] = []
    try:
        for idx, raw in enumerate(tables, start=1):
            df = _clean_columns(raw)
            sheet = f"table_{idx}"
            tname = _table_name(doc_name, sheet)
            try:
                con.register("_tmp_df", _normalize_dates(df))
                con.execute(
                    f'CREATE OR REPLACE TABLE "{tname}" AS SELECT * FROM _tmp_df'
                )
                con.unregister("_tmp_df")
                loaded.append((sheet, df))
                if verbose:
                    print(f"  DuckDB: {tname} ({df.shape[0]} rows, {df.shape[1]} cols)")
            except Exception as exc:  # noqa: BLE001 — one bad table must not abort the doc
                if verbose:
                    print(f"  [WARN] table {idx} load failed for {doc_name}: {exc}")
    finally:
        con.close()
    return loaded


def ingest_pdf_tables(
    doc_name: str,
    markdown: str,
    collection: str = "documents_chunks",
    verbose: bool = True,
) -> int:
    """Load a document's embedded tables into DuckDB and upsert Qdrant summaries.

    ``doc_name`` is the document stem (e.g. ``doc_005_fueling_records_invoice``).
    Returns the number of tables loaded. Safe to re-run (CREATE OR REPLACE).
    """
    from src.ingest_table_rows import (  # noqa: PLC0415 — avoid import cycle at module load
        _build_file_document_summary,
        _build_sheet_summary_point,
        sheet_summary_text,
    )

    loaded = load_tables_to_duckdb(doc_name, markdown, verbose=verbose)
    if not loaded:
        return 0

    file_name = f"{doc_name}.pdf"
    summaries: list[str] = []
    for sheet, df in loaded:
        columns = [str(c) for c in df.columns]
        data_rows = df.astype(object).where(pd.notna(df), None).values.tolist()
        summary = sheet_summary_text(file_name, sheet, columns, data_rows)
        summaries.append(summary)
        _build_sheet_summary_point(file_name, sheet, columns, "", collection)
    _build_file_document_summary(file_name, summaries, collection)
    if verbose:
        print(f"  Qdrant: {len(loaded)} table summaries upserted for {doc_name}")
    return len(loaded)
