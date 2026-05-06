"""DuckDB-backed Excel query tool.

Pipeline: Raw Excel → excel_cleaner.process_file() → cleaned DataFrame → DuckDB table.
DuckDB persists to disk so the LLM-assisted cleaning only runs once per file.
Agent queries via SQL through a single query_excel tool.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

import duckdb
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.config import DUCKDB_PATH, GROQ_API_KEY
from src.preprocessing.excel_cleaner import process_file


def _make_groq_llm_fn() -> Callable[[str], str]:
    """Return a Groq LLM callable for excel_cleaner.process_file."""
    import openai

    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    return _call


def _table_name(file: str, sheet: str) -> str:
    """Derive a valid DuckDB table name from file basename + sheet name."""
    stem = Path(file).stem
    combined = f"{stem}__{sheet}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", combined).strip("_")


def _normalize_dates(df: "Any") -> "Any":
    """Convert all date/datetime columns to ISO 'YYYY-MM-DD' strings.

    Handles both datetime64 dtype columns and object columns whose values
    match common date patterns (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.).
    dayfirst=True is used for ambiguous formats (matches EU convention in source data).
    """
    import pandas as pd

    df = df.copy()
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series.dt.strftime("%Y-%m-%d").where(series.notna(), other=None)
            continue
        if series.dtype != object:
            continue
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            continue
        sample = non_null.head(20)
        matching = sample.str.match(r"^\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$").sum()
        if matching / len(sample) < 0.8:
            continue
        try:
            parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
            valid_ratio = parsed.notna().sum() / max(series.notna().sum(), 1)
            if valid_ratio < 0.8:
                continue
            df[col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), other=None)
        except Exception:
            pass
    return df


_WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "REPLACE")


class DuckDBStore:
    """Load Excel files via excel_cleaner and persist cleaned DataFrames as DuckDB tables."""

    def __init__(self, excel_files: list[str], db_path: str = DUCKDB_PATH) -> None:
        """Process Excel files and register cleaned sheets as persistent DuckDB tables.

        Uses the DuckDB file as a cache — tables already present are not reprocessed.

        Args:
            excel_files: Paths to .xlsx or .csv files.
            db_path: Path to the persistent DuckDB database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._lock = threading.Lock()
        self._tables: dict[str, list[str]] = {}
        self._file_sheet_map: dict[str, dict[str, str]] = {}

        llm_fn = _make_groq_llm_fn()
        for path in excel_files:
            basename = os.path.basename(path)
            try:
                sheet_results = process_file(path, llm_fn)
            except Exception as e:
                print(f"[DuckDBStore] ERROR processing {basename}: {e}")
                continue

            file_map: dict[str, str] = {}
            for sheet_name, sr in sheet_results.items():
                tname = _table_name(basename, sheet_name)
                existing = self._con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [tname]
                ).fetchone()[0]
                if existing:
                    print(f"[DuckDBStore] cache hit: {tname}")
                else:
                    print(f"[DuckDBStore] inserting: {tname}")
                    normalized_df = _normalize_dates(sr.df)
                    self._con.register("_tmp_df", normalized_df)
                    self._con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM _tmp_df')
                    self._con.unregister("_tmp_df")
                cols = [row[0] for row in self._con.execute(f'DESCRIBE "{tname}"').fetchall()]
                self._tables[tname] = cols
                file_map[sheet_name] = tname
            self._file_sheet_map[basename] = file_map

    def tables(self) -> dict[str, list[str]]:
        """Return {table_name: [column_names]} for all loaded sheets."""
        return self._tables

    def file_sheet_map(self) -> dict[str, dict[str, str]]:
        """Return {file_basename: {sheet_name: table_name}}."""
        return self._file_sheet_map

    def execute(self, sql: str) -> Any:
        """Execute a SQL query, fetch all results as a DataFrame (thread-safe)."""
        with self._lock:
            return self._con.execute(sql).df()

    def fetchone(self, sql: str) -> tuple | None:
        """Execute a SQL query and return the first row (thread-safe)."""
        with self._lock:
            return self._con.execute(sql).fetchone()


def build_excel_tools(store: DuckDBStore) -> list[StructuredTool]:
    """Return a single LangChain StructuredTool for SQL-based Excel queries.

    Args:
        store: Populated DuckDBStore instance.
    """

    class _SQLInput(BaseModel):
        sql: str

    def query_excel(sql: str) -> str:
        """Execute a SQL SELECT query against the structured Excel/CSV tables in DuckDB."""
        normalized = sql.strip().upper().lstrip("(")
        for kw in _WRITE_PREFIXES:
            if normalized.startswith(kw):
                return "Write operations are not allowed. Use SELECT queries only."
        try:
            df = store.execute(sql)
        except Exception as e:
            return f"SQL error: {e}"
        if df.empty:
            return "Query returned no rows."
        _ROW_LIMIT = 100
        truncated = len(df) > _ROW_LIMIT
        out = df.head(_ROW_LIMIT).to_string(index=False, max_colwidth=80)
        if truncated:
            out += f"\n… (truncated at {_ROW_LIMIT} rows)"
        return out

    return [
        StructuredTool.from_function(
            func=query_excel,
            name="query_excel",
            description=(
                "Execute a SQL SELECT query against cleaned Excel/CSV data stored in DuckDB. "
                "Use for value lookups, filters, aggregations (SUM, COUNT, AVG, GROUP BY, ORDER BY), "
                "and cross-column comparisons. Available tables and their columns are listed in the "
                "system prompt. Always double-quote column names that contain spaces or special characters."
            ),
            args_schema=_SQLInput,
        )
    ]
