"""Excel/CSV loading and schema extraction utilities.

Provides:
  load_sheets()    — parse .xlsx / .csv into raw row lists
  extract_schema() — LLM-based column name / metadata extraction

Used by ingest_table_rows.py (row-as-text ingestion) and table_cleaner.py.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import openpyxl
from openai import OpenAI

SKIP_SHEET_KEYWORDS = ("index", "contents", "cover", "notes")
SPECIAL_VALUES = {"NO", "NA", "IE", "NE", "NO,IE", "NO,NA", "NE,NO"}
PREVIEW_ROWS = 15


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

def _vllm_client() -> OpenAI:
    from src.config import OLLAMA_API_BASE
    base = os.getenv("TABLE_LLM_API_BASE", f"{OLLAMA_API_BASE.rstrip('/')}/v1")
    return OpenAI(base_url=base, api_key="ollama")


def _vllm_model() -> str:
    from src.config import GENERATION_MODEL
    return os.getenv("TABLE_LLM_MODEL", GENERATION_MODEL)


# ---------------------------------------------------------------------------
# Sheet loading
# ---------------------------------------------------------------------------

def _sanitize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _forward_fill(row: list[Any]) -> list[Any]:
    last = None
    out: list[Any] = []
    for value in row:
        v = _sanitize_cell(value)
        if v is not None:
            last = v
        out.append(last)
    return out


def load_sheets(file_path: str) -> dict[str, list[list[Any]]]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return _load_xlsx(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Expected .xlsx or .csv")


def _load_xlsx(path: Path) -> dict[str, list[list[Any]]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    sheets: dict[str, list[list[Any]]] = {}
    for name in wb.sheetnames:
        if any(keyword in name.lower() for keyword in SKIP_SHEET_KEYWORDS):
            print(f"  Skipping '{name}'")
            continue
        ws = wb[name]
        rows: list[list[Any]] = []
        for row in ws.iter_rows():
            values = [cell.value for cell in row]
            if not any(v is not None and str(v).strip() != "" for v in values):
                continue
            rows.append(_forward_fill(values))
        sheets[name] = rows
    return sheets


def _load_csv(path: Path) -> dict[str, list[list[Any]]]:
    rows: list[list[Any]] = []
    with path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not any(str(v).strip() != "" for v in row):
                continue
            normalized = [v if str(v).strip() != "" else None for v in row]
            rows.append(_forward_fill(normalized))
    return {path.stem: rows}


# ---------------------------------------------------------------------------
# Schema extraction (LLM-based)
# ---------------------------------------------------------------------------

def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output, tolerating fences/preamble."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model output")
    if raw.startswith("```"):
        raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
        return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    parsed = json.loads(raw[start: end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def _deduplicate_columns(columns: list[str]) -> list[str]:
    result: list[str] = []
    seen: dict[str, int] = {}
    for col in columns:
        count = seen.get(col, 0)
        seen[col] = count + 1
        result.append(f"{col}_{count + 1}" if count else col)
    return result


def _looks_numeric(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def _to_snake(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return cleaned or "col"


def _heuristic_schema(sheet_name: str, rows: list[list[Any]]) -> dict[str, Any]:
    """Deterministic fallback schema when LLM output is empty/invalid."""
    if not rows:
        return {
            "table_title": sheet_name,
            "data_starts_row": 0,
            "footnote_start_row": None,
            "columns": ["category"],
            "description": f"Table '{sheet_name}'.",
        }
    header_row = 0
    best_score = float("-inf")
    window = rows[: min(len(rows), 10)]
    for i, row in enumerate(window):
        non_empty = [v for v in row if v is not None and str(v).strip() != ""]
        if not non_empty:
            continue
        numeric_count = sum(1 for v in non_empty if _looks_numeric(v))
        text_count = len(non_empty) - numeric_count
        next_numeric = 0
        if i + 1 < len(window):
            next_numeric = sum(1 for v in window[i + 1] if _looks_numeric(v))
        score = (text_count * 2) + next_numeric - numeric_count
        if score > best_score:
            best_score = score
            header_row = i
    header = rows[header_row]
    columns: list[str] = []
    used: dict[str, int] = {}
    for idx, cell in enumerate(header):
        base = _to_snake(str(cell) if cell is not None else f"col_{idx+1}")
        count = used.get(base, 0)
        used[base] = count + 1
        columns.append(f"{base}_{count+1}" if count else base)
    if not columns:
        columns = ["category"]
    data_starts_row = min(header_row + 1, max(0, len(rows) - 1))
    footnote_start_row: int | None = None
    for i in range(data_starts_row, len(rows)):
        row = rows[i]
        numeric_count = sum(1 for v in row if _looks_numeric(v))
        first = str(row[0]).strip() if row and row[0] is not None else ""
        if len(first) > 90 and numeric_count == 0:
            footnote_start_row = i
            break
    return {
        "table_title": sheet_name,
        "data_starts_row": data_starts_row,
        "footnote_start_row": footnote_start_row,
        "columns": columns,
        "description": f"Structured table extracted from sheet '{sheet_name}'.",
    }


def _validate_schema(schema: dict[str, Any], rows: list[list[Any]]) -> str | None:
    data_start = schema.get("data_starts_row")
    columns = schema.get("columns", [])
    if not isinstance(data_start, int) or data_start < 0:
        return f"data_starts_row must be a non-negative int, got {data_start!r}"
    if data_start >= len(rows):
        return f"data_starts_row={data_start} exceeds row count={len(rows)}"
    if not columns:
        return "columns list is empty"
    data_rows = rows[data_start: data_start + 5]
    if not data_rows:
        return "no data rows after data_starts_row"
    avg_cells = sum(len(r) for r in data_rows) / len(data_rows)
    if len(columns) > avg_cells * 2:
        return (
            f"columns count ({len(columns)}) is much larger than avg cells per row ({avg_cells:.0f})"
        )
    footnote_start = schema.get("footnote_start_row")
    if footnote_start is not None:
        try:
            footnote_start = int(footnote_start)
        except (TypeError, ValueError):
            return f"footnote_start_row must be int or null, got {footnote_start!r}"
        if footnote_start <= data_start:
            return f"footnote_start_row={footnote_start} must be greater than data_starts_row={data_start}"
    return None


def extract_schema(
    sheet_name: str,
    rows: list[list[Any]],
    llm: OpenAI,
    model_name: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    preview = rows[:PREVIEW_ROWS]
    base_prompt = f"""You are analyzing a raw spreadsheet table. Merged cells may have been forward-filled.

Sheet: {sheet_name}
First rows:
{json.dumps(preview, indent=2, ensure_ascii=False)}

Return ONLY a JSON object with:
- "table_title": string
- "data_starts_row": int (0-based row index where numeric/data rows begin, AFTER all header rows)
- "footnote_start_row": int or null (0-based row index where footnotes/notes/sources begin; null if none)
- "columns": list of clean snake_case column names, one per data column, ALL MUST BE UNIQUE.
  CRITICAL for multi-level / merged headers: combine the parent header with the child header
  to form a fully-qualified unique name.
  Example: parent "Emissions" + child "CO2 kt" → "co2_emissions_kt".
  Include units in each name where visible e.g. "co2_emissions_kt", "consumption_tj".
  NEVER repeat the same column name — if you're unsure, append the column index (_1, _2, …).
- "description": 2-3 sentences describing this table for semantic search.

Return ONLY valid JSON, no extra text."""

    last_error: str | None = None
    schema: dict[str, Any] = {}
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_error:
            prompt = (
                base_prompt
                + f"\n\nPrevious attempt returned an invalid schema. Error: {last_error}. "
                "Please fix the issue and return a corrected JSON."
            )
        response = llm.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_content = response.choices[0].message.content or ""
        try:
            schema = _parse_json_object(raw_content)
            if schema.get("columns"):
                schema["columns"] = _deduplicate_columns(schema["columns"])
        except Exception:
            fallback = llm.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt + "\n\nRespond with ONLY a single JSON object. Do not use markdown code fences."}],
                temperature=0,
                max_tokens=800,
            )
            fallback_content = fallback.choices[0].message.content or ""
            try:
                schema = _parse_json_object(fallback_content)
                if schema.get("columns"):
                    schema["columns"] = _deduplicate_columns(schema["columns"])
            except Exception as exc:
                last_error = f"invalid/empty fallback JSON ({exc})"
                print(f"    Schema parsing failed (attempt {attempt + 1}/{max_retries + 1}): {last_error}")
                continue
        error = _validate_schema(schema, rows)
        if error is None:
            return schema
        last_error = error
        print(f"    Schema validation failed (attempt {attempt + 1}/{max_retries + 1}): {error}")

    print("    Falling back to heuristic schema inference.")
    return _heuristic_schema(sheet_name, rows)
