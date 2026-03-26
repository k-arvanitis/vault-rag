"""LLM-based Excel/CSV sheet cleaner.

Standalone tool — not wired into the ingestion pipeline yet.

Two-stage process:
  1. Schema alignment validation — LLM verifies that extracted column names actually
     match the column values in the first few data rows, and fixes any misalignment.
  2. Row cleaning (batched) — LLM normalises each batch of raw rows into clean JSON:
     resolves special values (NO/IE/NE), separates subtotals, strips footnotes,
     infers null vs zero.

Usage (standalone):
    from src.table_cleaner import clean_sheet
    from src.ingest_tables import load_sheets, extract_schema, _vllm_client, _vllm_model

    sheets = load_sheets("GRC.xlsx")
    rows   = sheets["Table1s1"]
    llm    = _vllm_client()
    model  = _vllm_model()
    schema = extract_schema("Table1s1", rows, llm, model)
    result = clean_sheet("Table1s1", rows, schema, llm, model, verbose=True)
    print(result.columns)
    print(result.data[:3])
    print(result.validation_report)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

SPECIAL_VALUES = {"NO", "NA", "IE", "NE", "NO,IE", "NO,NA", "NE,NO", "C"}
BATCH_SIZE = int(os.getenv("TABLE_CLEANER_BATCH_SIZE", "20"))


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class CleanedSheet:
    sheet_name: str
    table_title: str
    columns: list[str]               # validated & fixed column names
    data: list[dict[str, Any]]       # cleaned rows, one dict per row
    metadata: dict[str, Any]         # units, footnotes, source lines
    validation_report: dict[str, Any]  # what was changed vs original schema


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_json(
    llm: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 2000,
) -> Any:
    """Call the LLM, requesting JSON output. Returns parsed Python object."""
    response = llm.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    return json.loads(raw)


def _data_rows(raw_rows: list[list[Any]], schema: dict[str, Any]) -> list[list[Any]]:
    """Slice raw_rows to the data range defined by the schema."""
    data_start = int(schema.get("data_starts_row") or 0)
    footnote_start = schema.get("footnote_start_row")
    if footnote_start is not None:
        try:
            footnote_start = int(footnote_start)
        except (TypeError, ValueError):
            footnote_start = None
    return raw_rows[data_start:footnote_start]


# ---------------------------------------------------------------------------
# Stage 1 — Schema alignment validation
# ---------------------------------------------------------------------------

def validate_schema_alignment(
    sheet_name: str,
    raw_rows: list[list[Any]],
    schema: dict[str, Any],
    llm: OpenAI,
    model: str,
    sample_rows: int = 5,
) -> dict[str, Any]:
    """Verify that schema column names actually align with data values.

    Returns a (possibly corrected) schema dict plus a validation_report.
    """
    columns = schema.get("columns", [])
    data_start = int(schema.get("data_starts_row") or 0)
    sample = raw_rows[data_start : data_start + sample_rows]

    # Build a side-by-side table for the LLM
    aligned: list[dict] = []
    for row in sample:
        entry: dict[str, Any] = {}
        for i, col in enumerate(columns):
            entry[col] = row[i] if i < len(row) else None
        aligned.append(entry)

    prompt = f"""You are validating whether the column names assigned to a spreadsheet table
actually match the values in the data rows.

Sheet: {sheet_name}
Table title: {schema.get('table_title', sheet_name)}

Assigned column names (left) mapped to first {sample_rows} data rows (right):
{json.dumps(aligned, indent=2, ensure_ascii=False)}

Full raw header rows (rows 0 to {data_start}):
{json.dumps(raw_rows[:data_start + 1], indent=2, ensure_ascii=False)}

Tasks:
1. Check if the column names correctly describe the values. Pay special attention to:
   - Numeric columns named after wrong gas/metric (e.g. column named "ch4_kt" but values match CO2 magnitudes)
   - Columns that appear shifted (value in column N actually belongs to column N-1)
   - Duplicate or generic names like "emissions" that should be more specific
2. Return a corrected list of column names in the SAME ORDER and COUNT as the input.
3. Identify any rows in the sample that are subtotals, section headers, or footnotes
   (not actual data rows).
4. Extract any units mentioned in the header rows (e.g. "kt CO2 equivalent", "TJ").
5. Extract any footnote/source text visible in the header area.

Return ONLY a JSON object with:
- "columns": list[str] — corrected column names (same length as input: {len(columns)})
- "changes": list[str] — description of each change made (empty list if no changes)
- "subtotal_row_indices": list[int] — 0-based indices within the sample that are subtotals/headers
- "units": dict[str, str] — column_name -> unit string (e.g. {{"co2_kt": "kt"}})
- "footnotes": list[str] — any footnote or source lines found in header rows
- "confidence": "high" | "medium" | "low" — how confident you are in the alignment"""

    try:
        result = _call_json(llm, model, prompt, max_tokens=1500)
    except Exception as exc:
        return {
            **schema,
            "validation_report": {"error": str(exc), "changes": [], "confidence": "error"},
        }

    corrected_cols = result.get("columns", columns)
    # Safety: must be same length
    if len(corrected_cols) != len(columns):
        corrected_cols = columns

    updated_schema = {
        **schema,
        "columns": corrected_cols,
    }
    report = {
        "original_columns": columns,
        "corrected_columns": corrected_cols,
        "changes": result.get("changes", []),
        "subtotal_row_indices": result.get("subtotal_row_indices", []),
        "units": result.get("units", {}),
        "footnotes": result.get("footnotes", []),
        "confidence": result.get("confidence", "unknown"),
    }
    updated_schema["validation_report"] = report
    return updated_schema


# ---------------------------------------------------------------------------
# Stage 2 — Batch row cleaning
# ---------------------------------------------------------------------------

def clean_rows_batch(
    batch: list[list[Any]],
    columns: list[str],
    units: dict[str, str],
    llm: OpenAI,
    model: str,
) -> list[dict[str, Any] | None]:
    """Clean a batch of raw rows.

    Returns one dict per input row (None for rows identified as non-data).
    """
    prompt = f"""You are cleaning raw spreadsheet rows into structured JSON.

Column names: {json.dumps(columns)}
Units per column: {json.dumps(units) if units else "not specified"}

Special value meanings:
- "NO"  = Not Occurring (treat as null)
- "NA"  = Not Applicable (treat as null)
- "IE"  = Included Elsewhere (treat as null, flag it)
- "NE"  = Not Estimated (treat as null)
- "C"   = Confidential (treat as null)
- ""    = empty cell (treat as null)

Raw rows to clean (list of lists, one per row):
{json.dumps(batch, indent=2, ensure_ascii=False)}

For EACH row return one of:
- A JSON object mapping column names to clean values:
  - Numeric strings → float (e.g. "27,242.75" → 27242.75)
  - Special values (NO/NA/IE/NE/C) → null, plus add a "<col>_flag" key with the original code
  - Empty → null
  - Text → string (stripped)
  - If the entire row looks like a subtotal, section header, or footnote → return null
- null for non-data rows

Return ONLY a JSON object with:
- "rows": list — one entry per input row (clean dict or null)"""

    try:
        result = _call_json(llm, model, prompt, max_tokens=3000)
        rows = result.get("rows", [])
        # Pad/trim to match batch length
        while len(rows) < len(batch):
            rows.append(None)
        return rows[:len(batch)]
    except Exception:
        # Fallback: return raw dicts without cleaning
        fallback = []
        for raw in batch:
            entry: dict[str, Any] = {}
            for i, col in enumerate(columns):
                entry[col] = raw[i] if i < len(raw) else None
            fallback.append(entry)
        return fallback


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_sheet(
    sheet_name: str,
    raw_rows: list[list[Any]],
    schema: dict[str, Any],
    llm: OpenAI,
    model: str,
    batch_size: int = BATCH_SIZE,
    verbose: bool = False,
) -> CleanedSheet:
    """Full two-stage cleaning pipeline for one sheet.

    Stage 1: validate & fix schema alignment.
    Stage 2: clean all data rows in batches.
    """
    if verbose:
        print(f"[cleaner] {sheet_name} — stage 1: schema alignment validation")

    validated = validate_schema_alignment(sheet_name, raw_rows, schema, llm, model)
    report = validated.pop("validation_report", {})
    columns = validated.get("columns", schema.get("columns", []))
    units = report.get("units", {})

    if verbose:
        changes = report.get("changes", [])
        print(f"  confidence={report.get('confidence')} changes={len(changes)}")
        for c in changes:
            print(f"    • {c}")

    # Stage 2: clean data rows in batches
    data_rows = _data_rows(raw_rows, validated)
    if verbose:
        print(f"[cleaner] {sheet_name} — stage 2: cleaning {len(data_rows)} rows in batches of {batch_size}")

    cleaned_data: list[dict[str, Any]] = []
    for batch_start in range(0, len(data_rows), batch_size):
        batch = data_rows[batch_start : batch_start + batch_size]
        results = clean_rows_batch(batch, columns, units, llm, model)
        for row_result in results:
            if row_result is not None:
                cleaned_data.append(row_result)
        if verbose:
            kept = sum(1 for r in results if r is not None)
            print(f"  batch {batch_start // batch_size + 1}: {kept}/{len(batch)} rows kept")

    # Collect metadata
    data_start = int(validated.get("data_starts_row") or 0)
    header_rows = raw_rows[:data_start]
    metadata: dict[str, Any] = {
        "units": units,
        "footnotes": report.get("footnotes", []),
        "header_rows": header_rows,
        "table_title": validated.get("table_title", sheet_name),
    }

    return CleanedSheet(
        sheet_name=sheet_name,
        table_title=validated.get("table_title", sheet_name),
        columns=columns,
        data=cleaned_data,
        metadata=metadata,
        validation_report=report,
    )
