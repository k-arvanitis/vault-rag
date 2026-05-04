"""Direct Excel query tools — bypasses Qdrant, queries cleaned DataFrames in memory.

Temporary retrieval path until tables move to Postgres. Uses excel_cleaner.process_file
to produce cleaned SheetResult objects, then exposes tool functions for LangChain agents.
"""

from __future__ import annotations

import os
import pickle
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.config import EXCEL_CACHE_DIR, GROQ_API_KEY
from src.preprocessing.excel_cleaner import SheetResult, process_file


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


def _fuzzy_find(query: str, candidates: list[str], top_n: int = 3) -> list[str]:
    """Return top_n candidates ranked by SequenceMatcher ratio."""
    q = query.lower()
    scored = [(SequenceMatcher(None, q, c.lower()).ratio(), c) for c in candidates]
    scored.sort(reverse=True)
    return [c for ratio, c in scored[:top_n] if ratio > 0.2]


def _label_col(df: pd.DataFrame) -> str | None:
    """Heuristically pick the row-label column (first mostly-string column)."""
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(10)
        if not sample.empty:
            non_num = sum(
                1 for v in sample
                if not v.replace(",", "").replace(".", "").replace("-", "").strip().isnumeric()
            )
            if non_num > len(sample) // 2:
                return col
    return df.columns[0] if len(df.columns) > 0 else None


def _find_rows(df: pd.DataFrame, label_col: str, query: str) -> pd.DataFrame:
    """Return rows where the label column contains query (exact then fuzzy)."""
    q = query.lower()
    mask = df[label_col].astype(str).str.lower().str.contains(q, regex=False, na=False)
    matches = df[mask]
    if not matches.empty:
        return matches
    labels = df[label_col].dropna().astype(str).tolist()
    close = _fuzzy_find(query, labels, top_n=1)
    if close:
        mask2 = df[label_col].astype(str).str.lower() == close[0].lower()
        return df[mask2]
    return pd.DataFrame()


class ExcelStore:
    """Loads and caches cleaned SheetResults for a list of Excel files."""

    def __init__(self, excel_files: list[str], cache_dir: str = EXCEL_CACHE_DIR) -> None:
        """Load Excel files, using pickle cache to skip repeated LLM calls.

        Args:
            excel_files: Absolute or relative paths to .xlsx files.
            cache_dir: Directory for pickle cache files.
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, SheetResult]] = {}
        llm_fn = _make_groq_llm_fn()
        for path in excel_files:
            basename = os.path.basename(path)
            cache_path = self._cache_dir / f"{basename}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    self._store[basename] = pickle.load(f)
                print(f"[ExcelStore] cache hit: {basename}")
            else:
                print(f"[ExcelStore] processing (LLM): {basename}")
                results = process_file(path, llm_fn)
                self._store[basename] = results
                with open(cache_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"[ExcelStore] cached → {cache_path}")

    def files(self) -> list[str]:
        """Return loaded file basenames."""
        return list(self._store.keys())

    def sheets(self, file: str) -> dict[str, SheetResult] | None:
        """Return sheet name → SheetResult mapping for a file."""
        return self._store.get(file)

    def get(self, file: str, sheet: str) -> SheetResult | None:
        """Return a single SheetResult or None."""
        return (self._store.get(file) or {}).get(sheet)

    def find_file(self, query: str) -> str | None:
        """Fuzzy-find closest file basename (substring then difflib)."""
        q = query.lower()
        for f in self._store:
            if q in f.lower():
                return f
        matches = get_close_matches(query, self._store.keys(), n=1, cutoff=0.3)
        return matches[0] if matches else None

    def find_sheet(self, file: str, query: str) -> str | None:
        """Fuzzy-find closest sheet name within a file."""
        sheets = self._store.get(file) or {}
        q = query.lower()
        for s in sheets:
            if q in s.lower():
                return s
        matches = get_close_matches(query, sheets.keys(), n=1, cutoff=0.3)
        return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------


class _FileInput(BaseModel):
    file: str


class _FileSheetInput(BaseModel):
    file: str
    sheet: str


class _SearchInput(BaseModel):
    file: str
    sheet: str
    query: str


class _RowInput(BaseModel):
    file: str
    sheet: str
    row_match: str


class _CellInput(BaseModel):
    file: str
    sheet: str
    row_match: str
    column_match: str


class _ColumnInput(BaseModel):
    file: str
    sheet: str
    column_match: str


class _QueryRowsInput(BaseModel):
    file: str
    sheet: str
    filter_column: str
    filter_value: str
    filter_column2: str = ""
    filter_value2: str = ""
    sum_column: str = ""
    max_rows: int = 0


# ---------------------------------------------------------------------------
# Build LangChain tools from an ExcelStore
# ---------------------------------------------------------------------------


def build_excel_tools(store: ExcelStore) -> list[StructuredTool]:
    """Return LangChain StructuredTools that query the given ExcelStore.

    Args:
        store: Populated ExcelStore instance.
    """

    def list_excel_sources() -> str:
        """List available Excel workbooks."""
        files = store.files()
        return "\n".join(f"- {f}" for f in files) if files else "No Excel files loaded."

    def describe_workbook(file: str) -> str:
        """Return sheet names and dimensions for a workbook.

        Args:
            file: Filename or fuzzy match (e.g. 'GRC_2023').
        """
        matched = store.find_file(file)
        if not matched:
            return f"No workbook matched '{file}'. Available: {store.files()}"
        lines = [f"Workbook: {matched}"]
        for name, sr in (store.sheets(matched) or {}).items():
            lines.append(f"  Sheet '{name}': {sr.metadata.row_count} rows × {sr.metadata.column_count} cols")
        return "\n".join(lines)

    def describe_sheet(file: str, sheet: str) -> str:
        """Return columns, row count, description, notes, and sample row labels.

        Args:
            file: Filename or fuzzy match.
            sheet: Sheet name or fuzzy match.
        """
        mf = store.find_file(file)
        if not mf:
            return f"No workbook matched '{file}'."
        ms = store.find_sheet(mf, sheet)
        if not ms:
            return f"No sheet matched '{sheet}' in '{mf}'."
        sr = store.get(mf, ms)
        meta = sr.metadata
        lc = _label_col(sr.df)
        samples = sr.df[lc].dropna().astype(str).head(8).tolist() if lc else []
        lines = [
            f"Workbook: {mf}  Sheet: {ms}",
            f"Description: {meta.table_description}",
            f"Rows: {meta.row_count}  Columns: {meta.column_count}",
            f"Columns: {list(sr.df.columns)}",
            f"Sample row labels ({lc}): {samples}",
        ]
        if meta.notes:
            lines.append(f"Notes: {meta.notes[:3]}")
        return "\n".join(lines)

    def search_sheet(file: str, sheet: str, query: str) -> str:
        """Search a sheet's row labels and column names for a keyword.

        Args:
            file: Filename or fuzzy match.
            sheet: Sheet name or fuzzy match.
            query: Text to search (e.g. a category name, year, or entity).
        """
        mf = store.find_file(file)
        if not mf:
            return f"No workbook matched '{file}'."
        ms = store.find_sheet(mf, sheet)
        if not ms:
            return f"No sheet matched '{sheet}' in '{mf}'."
        sr = store.get(mf, ms)
        df = sr.df
        q = query.lower()
        hits: list[str] = []
        for col in df.columns:
            if q in col.lower():
                hits.append(f"Column match: '{col}'")
        lc = _label_col(df)
        if lc:
            for i, val in df[lc].items():
                if val and q in str(val).lower():
                    hits.append(f"Row {i} label: '{val}'")
        if not hits:
            candidates = list(df.columns) + (df[lc].dropna().astype(str).tolist() if lc else [])
            for c in _fuzzy_find(query, candidates):
                hits.append(f"Fuzzy match: '{c}'")
        if not hits:
            return f"No matches for '{query}' in {mf}/{ms}."
        return f"Matches in {mf}/{ms}:\n" + "\n".join(hits[:10])

    def get_row(file: str, sheet: str, row_match: str) -> str:
        """Return all column values for the best matching row.

        Args:
            file: Filename or fuzzy match.
            sheet: Sheet name or fuzzy match.
            row_match: Text to match against row labels.
        """
        mf = store.find_file(file)
        if not mf:
            return f"No workbook matched '{file}'."
        ms = store.find_sheet(mf, sheet)
        if not ms:
            return f"No sheet matched '{sheet}' in '{mf}'."
        sr = store.get(mf, ms)
        lc = _label_col(sr.df)
        if not lc:
            return "Could not identify a row-label column."
        matches = _find_rows(sr.df, lc, row_match)
        if matches.empty:
            return f"No row matched '{row_match}' in {mf}/{ms}."
        row = matches.iloc[0]
        pairs = " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v) and str(v).strip())
        return (
            f"Source: {mf} / {ms}\n"
            f"Row index: {matches.index[0]}  Label column: {lc}\n"
            f"{pairs}"
        )

    def get_cell(file: str, sheet: str, row_match: str, column_match: str) -> str:
        """Return the value at the intersection of a row and column.

        Args:
            file: Filename or fuzzy match.
            sheet: Sheet name or fuzzy match.
            row_match: Row label to match.
            column_match: Column name to match.
        """
        mf = store.find_file(file)
        if not mf:
            return f"No workbook matched '{file}'."
        ms = store.find_sheet(mf, sheet)
        if not ms:
            return f"No sheet matched '{sheet}' in '{mf}'."
        sr = store.get(mf, ms)
        df = sr.df
        # Match column
        exact_col: str | None = next((c for c in df.columns if column_match.lower() in c.lower()), None)
        if not exact_col:
            close = _fuzzy_find(column_match, list(df.columns), top_n=1)
            exact_col = close[0] if close else None
        if not exact_col:
            return f"No column matched '{column_match}'. Available: {list(df.columns)}"
        # Match row
        lc = _label_col(df)
        if not lc:
            return "Could not identify a row-label column."
        matches = _find_rows(df, lc, row_match)
        if matches.empty:
            return f"No row matched '{row_match}' in {mf}/{ms}."
        row = matches.iloc[0]
        value = row[exact_col]
        return (
            f"Value: {value}\n"
            f"Source: {mf} / {ms}\n"
            f"Row: '{row[lc]}' (index {matches.index[0]})\n"
            f"Column: '{exact_col}'"
        )

    def get_column(file: str, sheet: str, column_match: str) -> str:
        """Return all non-empty values in a column with their row labels.

        Args:
            file: Filename or fuzzy match.
            sheet: Sheet name or fuzzy match.
            column_match: Column name to match.
        """
        mf = store.find_file(file)
        if not mf:
            return f"No workbook matched '{file}'."
        ms = store.find_sheet(mf, sheet)
        if not ms:
            return f"No sheet matched '{sheet}' in '{mf}'."
        sr = store.get(mf, ms)
        df = sr.df
        exact_col: str | None = next((c for c in df.columns if column_match.lower() in c.lower()), None)
        if not exact_col:
            close = _fuzzy_find(column_match, list(df.columns), top_n=1)
            exact_col = close[0] if close else None
        if not exact_col:
            return f"No column matched '{column_match}'. Available: {list(df.columns)}"
        lc = _label_col(df)
        lines = [f"Column '{exact_col}' in {mf}/{ms}:"]
        _ROW_LIMIT = 100
        count = 0
        for i, val in df[exact_col].items():
            if pd.notna(val) and str(val).strip():
                label = df.at[i, lc] if lc else i
                lines.append(f"  row {i} ({label}): {val}")
                count += 1
                if count >= _ROW_LIMIT:
                    lines.append(f"  … (truncated at {_ROW_LIMIT} rows, use query_rows for filtered results)")
                    break
        return "\n".join(lines) if len(lines) > 1 else f"Column '{exact_col}' is empty."

    def query_rows(
        file: str,
        sheet: str,
        filter_column: str,
        filter_value: str,
        filter_column2: str = "",
        filter_value2: str = "",
        sum_column: str = "",
        max_rows: int = 0,
    ) -> str:
        """Filter rows where filter_column contains filter_value and return matches + optional sum.

        Use this for large spreadsheets to avoid dumping all rows. Prefer this over get_column
        when you need rows matching a category, department, or entity.

        Args:
            file: Filename or fuzzy match.
            sheet: Sheet name or fuzzy match.
            filter_column: Column to filter on (e.g. 'Department', 'Directorate').
            filter_value: Value to match (case-insensitive substring).
            filter_column2: Optional second column for AND filtering (e.g. 'Date').
            filter_value2: Value to match in filter_column2 (case-insensitive substring).
            sum_column: ONLY provide when the question asks for a total/aggregate. Leave empty when the question asks for a specific row's amount — read it from the returned rows instead.
            max_rows: Maximum number of rows to return (0 = no limit). Use max_rows=1 when the question asks for a specific named entity's row (e.g. "the Google Ads row") to return only the first/earliest match.
        """
        mf = store.find_file(file)
        if not mf:
            return f"No workbook matched '{file}'."
        ms = store.find_sheet(mf, sheet)
        if not ms:
            return f"No sheet matched '{sheet}' in '{mf}'."
        sr = store.get(mf, ms)
        df = sr.df

        exact_fc: str | None = next((c for c in df.columns if filter_column.lower() in c.lower()), None)
        if not exact_fc:
            close = _fuzzy_find(filter_column, list(df.columns), top_n=1)
            exact_fc = close[0] if close else None
        if not exact_fc:
            return f"No column matched '{filter_column}'. Available: {list(df.columns)}"

        mask = df[exact_fc].astype(str).str.contains(filter_value, case=False, na=False, regex=False)
        matches = df[mask]
        if matches.empty:
            # Fuzzy column resolution may have picked the wrong column — try all string-like columns.
            str_cols = [c for c in df.columns if df[c].dtype == object]
            for alt_col in str_cols:
                if alt_col == exact_fc:
                    continue
                alt_mask = df[alt_col].astype(str).str.contains(filter_value, case=False, na=False, regex=False)
                alt_matches = df[alt_mask]
                if not alt_matches.empty:
                    exact_fc = alt_col
                    matches = alt_matches
                    break
        if matches.empty:
            return f"No rows where any column contains '{filter_value}' in {mf}/{ms}."

        # Apply optional second AND filter.
        if filter_column2 and filter_value2:
            exact_fc2: str | None = next((c for c in df.columns if filter_column2.lower() in c.lower()), None)
            if exact_fc2:
                mask2 = matches[exact_fc2].astype(str).str.contains(filter_value2, case=False, na=False, regex=False)
                narrowed = matches[mask2]
                if not narrowed.empty:
                    matches = narrowed

        row_limit = max_rows if max_rows > 0 else 50
        lines = [f"Rows where '{exact_fc}' contains '{filter_value}' in {mf}/{ms} ({len(matches)} rows):"]
        for i, (_, row) in enumerate(matches.iterrows()):
            if i >= row_limit:
                lines.append(f"  … ({len(matches) - row_limit} more rows not shown, use max_rows to increase)")
                break
            pairs = " | ".join(f"{c}: {v}" for c, v in row.items() if pd.notna(v) and str(v).strip())
            lines.append(f"  {pairs}")

        if sum_column:
            exact_sc: str | None = next((c for c in df.columns if sum_column.lower() in c.lower()), None)
            if exact_sc:
                numeric = pd.to_numeric(
                    matches[exact_sc].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
                total = numeric.sum()
                lines.append(f"\nSum of '{exact_sc}': {total:,.2f} ({numeric.notna().sum()} rows summed)")
        return "\n".join(lines)

    specs: list[tuple[Any, str, str, type[BaseModel]]] = [
        (list_excel_sources, "list_excel_sources",
         "List available Excel workbooks. Call with no arguments before inspecting sheets.",
         BaseModel),
        (describe_workbook, "describe_workbook",
         "Return sheet names and dimensions for an Excel workbook. Arg: file (filename or partial match).",
         _FileInput),
        (describe_sheet, "describe_sheet",
         "Return columns, row count, description, notes, and sample row labels for a sheet. Args: file, sheet.",
         _FileSheetInput),
        (search_sheet, "search_sheet",
         "Search a sheet's row labels and column names for a keyword. Args: file, sheet, query.",
         _SearchInput),
        (get_row, "get_row",
         "Return all column values for the best matching row. Args: file, sheet, row_match.",
         _RowInput),
        (get_cell, "get_cell",
         "Return the single value at a (row, column) intersection. Args: file, sheet, row_match, column_match.",
         _CellInput),
        (get_column, "get_column",
         "Return all values in a column with their row labels. Args: file, sheet, column_match.",
         _ColumnInput),
        (query_rows, "query_rows",
         "Filter rows where filter_column contains filter_value, return matches and optional sum. "
         "Use for large spreadsheets instead of get_column. "
         "Supports AND filtering via filter_column2+filter_value2. "
         "Use max_rows=1 when the question asks for a specific named entity's row (e.g. 'the Google Ads row') to return only the first/earliest match. "
         "Args: file, sheet, filter_column, filter_value, filter_column2 (optional), filter_value2 (optional), sum_column (optional), max_rows (optional, default 0=no limit).",
         _QueryRowsInput),
    ]

    tools: list[StructuredTool] = []
    for fn, name, desc, schema in specs:
        if schema is BaseModel:
            # No-arg tool — StructuredTool.from_function handles this directly
            tools.append(StructuredTool.from_function(func=fn, name=name, description=desc))
        else:
            tools.append(StructuredTool.from_function(func=fn, name=name, description=desc, args_schema=schema))
    return tools
