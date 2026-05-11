"""DuckDB-backed Excel query tool.

Pipeline: Raw Excel → excel_cleaner.process_file() → cleaned DataFrame → DuckDB table.
DuckDB persists to disk so the LLM-assisted cleaning only runs once per file.
Agent queries via SQL through a single query_excel tool.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Callable

import duckdb

from src.config import DUCKDB_PATH, GROQ_API_KEY


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


# DD/MM/YYYY or D/M/YYYY inside SQL string literals → YYYY-MM-DD
_DATE_DMY = re.compile(r"'(\d{1,2})/(\d{1,2})/(\d{4})'")
# ILIKE string values for truncation retry: captures the inner text of '%...%' patterns
_ILIKE_VAL = re.compile(r"(ILIKE\s+'%)([^%']+)(%')", re.IGNORECASE)


def _normalize_sql(sql: str) -> str:
    """Rewrite SQL literals before DuckDB execution.

    - DD/MM/YYYY → YYYY-MM-DD (dates stored as ISO strings in all ingested tables)
    """
    return _DATE_DMY.sub(lambda m: f"'{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}'", sql)


def _truncate_ilike(sql: str, chars: int = 1) -> str:
    """Shorten each ILIKE '%...%' value by `chars` trailing characters.

    Used as a fallback when a query returns no rows and the agent may have
    auto-completed a truncated supplier/beneficiary name (e.g. "Yorkshir" → "Yorkshire").
    Only shortens values that end with a letter (not space or digit).
    """
    def _shorten(m: re.Match) -> str:
        prefix, text, suffix = m.group(1), m.group(2), m.group(3)
        if len(text) > chars + 3 and text[-1].isalpha():
            return f"{prefix}{text[:-chars]}{suffix}"
        return m.group(0)
    return _ILIKE_VAL.sub(_shorten, sql)


class DuckDBStore:
    """Connect to the persistent DuckDB populated by ingest_table_rows."""

    def __init__(self, db_path: str = DUCKDB_PATH) -> None:
        """Connect to the existing DuckDB file. Tables are loaded at ingest time."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._lock = threading.Lock()
        # Discover all user tables currently in the DB
        rows = self._con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        self._tables: dict[str, list[str]] = {}
        for (tname,) in rows:
            cols = [r[0] for r in self._con.execute(f'DESCRIBE "{tname}"').fetchall()]
            self._tables[tname] = cols
        if self._tables:
            print(f"[DuckDBStore] connected — {len(self._tables)} table(s) available.")
        else:
            print("[DuckDBStore] connected — no tables yet. Ingest table files first.")

    def tables(self) -> dict[str, list[str]]:
        """Return {table_name: [column_names]} for all tables in the DB."""
        return self._tables

    def execute(self, sql: str) -> Any:
        """Execute a SQL query, fetch all results as a DataFrame (thread-safe)."""
        with self._lock:
            return self._con.execute(sql).df()

    def fetchone(self, sql: str) -> tuple | None:
        """Execute a SQL query and return the first row (thread-safe)."""
        with self._lock:
            return self._con.execute(sql).fetchone()

    def describe(self, table_name: str) -> list[tuple[str, str]]:
        """Return [(column_name, column_type), ...] for a table."""
        with self._lock:
            rows = self._con.execute(f'DESCRIBE "{table_name}"').fetchall()
        return [(r[0], r[1]) for r in rows]

    def sample(self, table_name: str, n: int = 3) -> list[dict]:
        """Return up to n sample rows as a list of dicts."""
        with self._lock:
            df = self._con.execute(f'SELECT * FROM "{table_name}" LIMIT {n}').df()
        return df.to_dict(orient="records")
