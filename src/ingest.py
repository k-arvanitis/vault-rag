import argparse
import importlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from pypdf import PdfReader  # noqa: E402

from src import chunker as chunker_module  # noqa: E402
from src.config import (  # noqa: E402
    QDRANT_URL, QDRANT_COLLECTION, OLLAMA_API_BASE, OLLAMA_EMBED_MODEL,
    OCR_API_BASE, OCR_MODEL,
)
from src.embedder import embed_chunks_file  # noqa: E402
from src.ingest_table_rows import ingest_table_rows  # noqa: E402
from src.parser.input_utils import resolve_to_pdf  # noqa: E402
from src.parser.pdf_parser import parse_pdf  # noqa: E402
from src.vector_store import delete_by_file, ingest_embeddings  # noqa: E402

# ---------------------------------------------------------------------------
# HTML table multi-row header → flat markdown table
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402

_HTML_TABLE_RE = _re.compile(r"(?is)<table\b[^>]*>.*?</table>")
_SUB_COL_TOKEN_RE = _re.compile(r"^(R@\d+|NDCG@\d+|[PF]@\d+|MRR@?\d*|@\w+|-)$")


def _parse_html_cells(row_html: str, tag: str = "td") -> list[str]:
    """Extract cell text from a single <tr> row."""
    return [
        _re.sub(r"<[^>]+>", "", cell).strip()
        for cell in _re.findall(rf"(?is)<{tag}\b[^>]*>(.*?)</{tag}>", row_html)
    ]


def _fix_html_multirow_header(table_html: str) -> str | None:
    """Convert an HTML table with a two-row header into a flat markdown pipe table.

    Handles the pattern where the OCR drops colspan attributes, placing group
    names in <thead> (with empty filler cells) and sub-column names (R@1/R@5…)
    in the first <tbody> row.

    Returns a markdown pipe-table string, or None if the pattern doesn't apply.
    """
    # Extract <thead> row
    thead_m = _re.search(r"(?is)<thead\b[^>]*>(.*?)</thead>", table_html)
    tbody_m = _re.search(r"(?is)<tbody\b[^>]*>(.*?)</tbody>", table_html)
    if not thead_m or not tbody_m:
        return None

    thead_cells = _parse_html_cells(thead_m.group(1), "th")
    # Also try <td> in thead (some OCR outputs use td inside thead)
    if not thead_cells:
        thead_cells = _parse_html_cells(thead_m.group(1), "td")

    # Get all <tr> rows from tbody
    body_rows_html = _re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", tbody_m.group(1))
    if not body_rows_html:
        return None

    first_body_cells = _parse_html_cells(body_rows_html[0], "td")

    # Check: is the first body row a sub-column header row?
    non_empty = [c for c in first_body_cells if c]
    if not non_empty:
        return None
    sub_col_count = sum(1 for c in non_empty if _SUB_COL_TOKEN_RE.match(c))
    if sub_col_count / len(non_empty) < 0.5:
        return None  # Not a sub-column row

    # Reconstruct the flat header using _derive_column_names
    group_names = [c for c in thead_cells if c]  # non-empty group names + standalones
    sub_col_names = first_body_cells  # sub-column tokens

    # Build the header string in the same format _derive_column_names expects:
    # "<group1> <group2> ... <groupN> <sub1> <sub2> ..."
    header_str = " ".join(group_names + sub_col_names)

    # Build clean_data from remaining body rows
    data_rows_text = []
    for row_html in body_rows_html[1:]:
        cells = _parse_html_cells(row_html, "td")
        data_rows_text.append(" ".join(cells))
    clean_data = "\n".join(data_rows_text)

    derived_cols = _derive_column_names(header_str, clean_data)
    if derived_cols:
        result = _build_markdown_table(derived_cols, clean_data)
        if result:
            return result

    # Fallback: produce a flat pipe table without derivation
    # Just combine: for empty thead cells, pull sub-col name; for non-empty, append sub-col
    flat_headers: list[str] = []
    for i, h in enumerate(first_body_cells):
        # Try to assign a group name based on position
        group = thead_cells[i] if i < len(thead_cells) else ""
        sub = h
        if group and not _SUB_COL_TOKEN_RE.match(group):
            flat_headers.append(f"{group} {sub}" if _SUB_COL_TOKEN_RE.match(sub) else group)
        else:
            flat_headers.append(sub)

    if not flat_headers:
        return None

    sep = "| " + " | ".join(["---"] * len(flat_headers)) + " |"
    header_row = "| " + " | ".join(flat_headers) + " |"
    data_md_rows = []
    for row_html in body_rows_html[1:]:
        cells = _parse_html_cells(row_html, "td")
        cells = cells[:len(flat_headers)]
        while len(cells) < len(flat_headers):
            cells.append("")
        data_md_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, sep] + data_md_rows)


def _fix_html_table_headers_in_markdown(markdown: str, verbose: bool = True) -> str:
    """Find HTML tables with two-row headers and convert them to flat markdown tables."""
    matches = list(_HTML_TABLE_RE.finditer(markdown))
    if not matches:
        return markdown

    result = markdown
    fixed = 0
    for m in reversed(matches):
        table_html = m.group(0)
        flat = _fix_html_multirow_header(table_html)
        if flat:
            result = result[:m.start()] + "\n\n" + flat + "\n\n" + result[m.end():]
            fixed += 1

    if verbose and fixed:
        _log(f"Fixed {fixed} HTML table(s) with multi-row headers")
    return result


def _fill_empty_last_column_from_text(markdown: str, pdf_path: Path, verbose: bool = True) -> str:
    """Fill empty last-column cells in pipe tables using the PDF text layer.

    Born-digital PDFs (arXiv papers) have a text layer that contains all values.
    The OCR sometimes misses the rightmost column of wide tables; this function
    recovers those values by scanning the PDF text layer.
    """
    _decimal_re = _re.compile(r"\b\d+\.\d{2,4}\b")
    _pipe_table_re = _re.compile(
        r"(?m)^(\|[^\n]+\|)\n(\|[-| :]+\|)\n((?:\|[^\n]+\|\n?)+)"
    )

    def _has_empty_last_col(table_str: str) -> bool:
        """Return True if >50% of data rows have an empty last cell."""
        rows = [line for line in table_str.strip().splitlines() if line.startswith("|") and "---" not in line]
        if len(rows) < 2:
            return False
        empty = sum(1 for r in rows[1:] if r.rstrip().endswith("|  |") or r.rstrip().endswith("| |"))
        return empty / max(len(rows) - 1, 1) > 0.5

    def _extract_page_text(page_num: int) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            if 0 <= page_num < len(reader.pages):
                return reader.pages[page_num].extract_text() or ""
        except Exception:
            pass
        return ""

    def _find_row_value_in_text(label: str, page_text: str, n_expected: int) -> str | None:
        r"""Find the n_expected-th decimal number in the row starting with label.

        PDF text layers concatenate adjacent numbers with no space (e.g. "2.9524.141").
        Using \d+\.\d{3} correctly splits these since research table values have exactly
        3 decimal places. Falls back to 2-decimal matching if needed.
        """
        label_clean = _re.escape(label.strip())
        m = _re.search(label_clean, page_text)
        if not m:
            return None
        tail = page_text[m.end(): m.end() + 300]
        # Try 3-decimal match first (most precise, avoids greedy over-capture)
        nums = _re.findall(r"\d+\.\d{3}", tail)
        if len(nums) < n_expected:
            # Fall back: 2-decimal
            nums = _re.findall(r"\d+\.\d{2,3}", tail)
        if len(nums) >= n_expected:
            return nums[n_expected - 1]
        return None

    # Find page markers in markdown to know which PDF page each table is on
    page_marker_re = _re.compile(r"<!--\s*PAGE\s+(\d+)\s*-->")
    markers = [(m.start(), int(m.group(1))) for m in page_marker_re.finditer(markdown)]

    def _page_for_pos(pos: int) -> int:
        """Return the 0-indexed PDF page number for a position in the markdown."""
        page = 1
        for m_pos, m_page in markers:
            if m_pos <= pos:
                page = m_page
        return page - 1  # convert to 0-indexed

    result = markdown
    total_filled = 0

    for m in reversed(list(_pipe_table_re.finditer(markdown))):
        table_str = m.group(0)
        if not _has_empty_last_col(table_str):
            continue

        page_idx = _page_for_pos(m.start())
        page_text = _extract_page_text(page_idx)
        if not page_text:
            continue

        header_row = m.group(1)
        sep_row = m.group(2)
        data_rows_str = m.group(3)

        # Count how many numeric columns exist in a complete row (to know which column is last)
        data_rows = [r for r in data_rows_str.strip().splitlines() if r.strip().startswith("|")]
        new_rows: list[str] = []
        filled = 0

        for row in data_rows:
            if row.rstrip().endswith("|  |") or row.rstrip().endswith("| |"):
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                label = cells[0] if cells else ""
                n_numeric = sum(1 for c in cells[1:] if _decimal_re.match(c))
                # We want the (n_numeric + 1)-th decimal in this row in the text layer
                value = _find_row_value_in_text(label, page_text, n_numeric + 1)
                if value:
                    cells[-1] = value
                    row = "| " + " | ".join(cells) + " |"
                    filled += 1
            new_rows.append(row)

        if filled:
            new_table = header_row + "\n" + sep_row + "\n" + "\n".join(new_rows) + "\n"
            result = result[:m.start()] + new_table + result[m.end():]
            total_filled += filled

    if verbose and total_filled:
        _log(f"Recovered {total_filled} missing cell(s) from PDF text layer")
    return result


# ---------------------------------------------------------------------------
# LaTeX table → markdown conversion
# ---------------------------------------------------------------------------

# Matches $\text{...}$ blocks that are long enough to be a table
_LATEX_TEXT_TABLE_RE = _re.compile(r'\$\\text\{(.{80,}?)\}\$', _re.DOTALL)


def _is_latex_table(inner: str) -> bool:
    """Return True if the inner text of a $\\text{...}$ block looks like a table.

    Distinguishes data tables from mathematical formulas by requiring:
    - A \\textbf{} header block (tables always have bold column headers)
    - Many decimal numbers (multiple rows × multiple metric columns)
    - Or explicit retrieval metric patterns (R@K, NDCG@K, etc.)
    """
    # Must have a bold header (\textbf{}) — formulas rarely start with one
    if not _re.search(r"\\textbf\{", inner):
        return False
    # Need enough decimal values to suggest multiple data rows (≥ 8)
    decimal_count = len(_re.findall(r"\b\d+\.\d+\b", inner))
    if decimal_count >= 8:
        return True
    # Or: fewer decimals but has retrieval metric annotations
    if decimal_count >= 4 and _re.search(r"R@\d+|NDCG@\d+|@adaptive", inner):
        return True
    return False


def _preprocess_latex_table(inner: str) -> tuple[str, str, list[str]]:
    """Separate headers from data and derive column names where possible.

    Returns (header_str, clean_data_str, derived_columns).
    derived_columns is a non-empty list when we can programmatically determine
    the correct column names (avoids LLM hallucination on multi-level headers).
    """
    first_bold = _re.search(r"\\textbf\{([^}]+)\}", inner)
    header_str = first_bold.group(1).strip() if first_bold else ""
    after_headers = inner[first_bold.end():].strip() if first_bold else inner
    clean_data = _re.sub(r"\\textbf\{([^}]+)\}", r"\1", after_headers).strip()

    derived = _derive_column_names(header_str, clean_data)
    return header_str, clean_data, derived


def _derive_column_names(header_str: str, clean_data: str) -> list[str]:
    """Programmatically derive column names for tables with repeating sub-columns.

    Handles the research table two-row-header pattern flattened into a single
    string, e.g.:
      "Tokens Qwen3-4B Qwen3-4B (RR) Hipporag2 MSA (Ours) Dataset R@1 R@5 R@10
       R@1 R@5 R@10 R@1 R@5 R@10 @adaptive"

    In this format ALL group names appear first, then ALL sub-column tokens follow.
    """
    sub_re = _re.compile(r"^(R@\d+|NDCG@\d+|[PF]@\d+|MRR@\d*|@\w+)$")
    standalone_re = _re.compile(r"^(Dataset|Tokens?|Size|#\w+)$", _re.I)

    tokens = header_str.split()
    if not tokens:
        return []

    # ── Step 1: find where sub-col tokens start ───────────────────────────────
    first_sub_idx = next((i for i, t in enumerate(tokens) if sub_re.match(t)), None)
    if first_sub_idx is None:
        return []  # no sub-col pattern at all

    prefix_tokens = tokens[:first_sub_idx]
    sub_col_tokens = [t for t in tokens[first_sub_idx:] if sub_re.match(t)]

    # ── Step 2: derive repeating unit from sub-col sequence ──────────────────
    unit: list[str] = []
    for s in sub_col_tokens:
        if unit and s == unit[0]:
            break
        unit.append(s)
    if not unit:
        return []

    # ── Step 3: parse prefix into standalones + groups ────────────────────────
    # Each non-connector, non-parenthetical, non-standalone token starts a NEW group.
    # Connector tokens (+, &, vs, /) and parentheticals (...) extend the current group.
    _connector_re = _re.compile(r"^[+&/]$|^vs\.?$", _re.I)
    standalones: list[str] = []
    groups: list[str] = []
    for t in prefix_tokens:
        if t.startswith("(") or _connector_re.match(t):
            # Glue to the last group (or standalone) as a suffix
            if groups:
                groups[-1] = groups[-1] + " " + t
            elif standalones:
                standalones[-1] = standalones[-1] + " " + t
        elif standalone_re.match(t):
            standalones.append(t)
        else:
            # Start a new group — but if previous group ends with a connector,
            # this token continues that group
            if groups and _connector_re.match(groups[-1].split()[-1]):
                groups[-1] = groups[-1] + " " + t
            else:
                groups.append(t)

    if not groups:
        return []

    # ── Step 4: assign sub-col units to groups ────────────────────────────────
    # Each group gets one "slice" of sub-col tokens.
    # The total sub-col tokens should equal groups × len(unit), ± last group variant.
    n_unit = len(unit)
    # Allow last group to have fewer/different sub-cols (e.g. just @adaptive)
    sub_slices: list[list[str]] = []
    idx = 0
    for i in range(len(groups)):
        if i < len(groups) - 1:
            # Normal group: take exactly n_unit sub-cols
            slice_ = sub_col_tokens[idx: idx + n_unit]
            if len(slice_) < n_unit:
                return []  # not enough sub-cols — fall back to LLM
            sub_slices.append(slice_)
            idx += n_unit
        else:
            # Last group: take whatever remains
            slice_ = sub_col_tokens[idx:]
            sub_slices.append(slice_ if slice_ else unit)
            idx = len(sub_col_tokens)

    # ── Step 5: build final column list ──────────────────────────────────────
    cols: list[str] = list(standalones)
    for g, subs in zip(groups, sub_slices):
        for s in subs:
            cols.append(f"{g} {s}")

    # ── Step 6: sanity-check against first data row ──────────────────────────
    num_re = _re.compile(r"^\d[\d.,MKBTkm%]*$")
    cite_re = _re.compile(r"^\[\d+\]$")
    data_tokens = clean_data.split()
    numeric_in_first_row = 0
    counting = False
    for tok in data_tokens:
        if cite_re.match(tok):
            continue
        if not counting and num_re.match(tok):
            counting = True
        if counting:
            if num_re.match(tok):
                numeric_in_first_row += 1
            elif numeric_in_first_row > 2:
                break
    # Expected columns = text labels (standalones) + numeric values
    # Number of standalones ≈ 1–3, so total = numeric + n_standalones
    expected_total = numeric_in_first_row + len(standalones)
    if cols and abs(len(cols) - expected_total) <= 2:
        return cols
    return []  # derived count doesn't match — fall back to LLM


def _build_markdown_table(derived_cols: list[str], clean_data: str) -> str | None:
    """Build a markdown table directly from derived column names and raw data text.

    Avoids LLM entirely — no risk of two-row headers or column misalignment.
    Each data row: one or more non-numeric text tokens (label) + N-1 numeric tokens.
    """
    n_cols = len(derived_cols)
    _num_re = _re.compile(r"^\d[\d.,MKBTkm%]*$|^-$")
    _cite_re = _re.compile(r"^\[[\d,]+\]$")

    tokens = [t for t in clean_data.split() if not _cite_re.match(t)]

    rows: list[list[str]] = []
    i = 0
    while i < len(tokens):
        # Collect text label (all non-numeric leading tokens)
        label_parts: list[str] = []
        while i < len(tokens) and not _num_re.match(tokens[i]):
            label_parts.append(tokens[i])
            i += 1
        if not label_parts:
            i += 1
            continue
        # Collect numeric values until the next text label or end
        values: list[str] = []
        while i < len(tokens) and _num_re.match(tokens[i]):
            values.append(tokens[i])
            i += 1
        if not values:
            continue
        row = [" ".join(label_parts)] + values
        row = row[:n_cols]
        while len(row) < n_cols:
            row.append("")
        rows.append(row)

    if not rows:
        return None

    header = "| " + " | ".join(derived_cols) + " |"
    sep = "| " + " | ".join(["---"] * n_cols) + " |"
    return "\n".join([header, sep] + ["| " + " | ".join(r) + " |" for r in rows])


def _convert_latex_table(inner: str) -> str:
    """Convert a flat OCR'd LaTeX text block into a markdown table.

    If column names can be programmatically derived, build the table directly
    without the LLM (avoids two-row header hallucinations).  Falls back to LLM
    only when derivation fails.
    """
    header_str, clean_data, derived_cols = _preprocess_latex_table(inner)

    # Fast path: derived columns available → build table without LLM
    if derived_cols:
        result = _build_markdown_table(derived_cols, clean_data)
        if result:
            return result

    # Fallback: ask LLM to reconstruct (no derived columns available)
    from openai import OpenAI
    from src.config import GENERATION_API_BASE, GENERATION_MODEL, GROQ_API_KEY

    api_base = os.getenv("GENERATION_API_BASE", GENERATION_API_BASE)
    model = os.getenv("GENERATION_MODEL", GENERATION_MODEL)
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY) or "no-key"
    client = OpenAI(base_url=api_base, api_key=api_key)

    prompt = (
        "Reconstruct the following research paper table as a correct Markdown table.\n\n"
        f"HEADERS (from the first bold block — may be two physical header rows merged):\n{header_str}\n\n"
        f"DATA (bold markers stripped, content kept):\n{clean_data}\n\n"
        "Rules:\n"
        "- Combine parent group names with sub-column labels into ONE header row "
        "(e.g. 'MethodA R@1', 'MethodA R@5'). NEVER generate two header rows.\n"
        "- NEVER merge distinct groups.\n"
        "- Map data values to columns strictly left to right.\n"
        "- Return ONLY the markdown table — no explanation, no code fences."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )
        result = (resp.choices[0].message.content or "").strip()
        result = _re.sub(r"^```(?:markdown)?\n?", "", result)
        result = _re.sub(r"\n?```$", "", result).strip()
        return result if "|" in result else f"$\\text{{{inner}}}$"
    except Exception:
        return f"$\\text{{{inner}}}$"  # leave original on failure


def _convert_latex_tables_in_markdown(markdown: str, verbose: bool = True) -> str:
    """Find and convert all $\\text{...}$ blocks that look like tables."""
    matches = list(_LATEX_TEXT_TABLE_RE.finditer(markdown))
    table_matches = [(m, m.group(1)) for m in matches if _is_latex_table(m.group(1))]
    if not table_matches:
        return markdown
    if verbose:
        _log(f"Converting {len(table_matches)} LaTeX table block(s) to markdown")
    result = markdown
    for m, inner in reversed(table_matches):  # reversed so offsets stay valid
        converted = _convert_latex_table(inner)
        result = result[:m.start()] + "\n\n" + converted + "\n\n" + result[m.end():]
    return result


REPO_ROOT = chunker_module.REPO_ROOT


def _log(message: str, step: int | None = None, debug: bool = False, enabled: bool = True) -> None:
    """Print a prefixed ingestion log line; no-ops when enabled=False."""
    if not enabled:
        return
    prefix = "[INGEST]"
    if step is not None:
        prefix += f"[{step}/5]"
    if debug:
        prefix += "[DEBUG]"
    print(f"{prefix} {message}")


def _path_size_mb(path: Path) -> float:
    """Return the file size of path in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def _load_json_list(path: Path, label: str) -> list[dict]:
    """Read a JSON file and assert it is a list; raises ValueError otherwise."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {label} at {path}, got {type(payload).__name__}")
    return payload


def _pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF without loading the full content."""
    return len(PdfReader(str(pdf_path)).pages)


def _run_ingest_pdf(
    pdf_path: Path,
    collection: str,
    qdrant_url: str = QDRANT_URL,
    ocr_endpoint: str = f"{OCR_API_BASE}/v1/chat/completions",
    ocr_model_name: str = OCR_MODEL,
    ocr_table_mode: str = "grid",
    ollama_api_base: str = OLLAMA_API_BASE,
    ollama_embed_model: str = OLLAMA_EMBED_MODEL,
    enrich_with_llm: bool = True,
    verbose: bool = True,
    force_pipeline: str | None = None,
) -> dict[str, Path]:
    file_started_at = time.perf_counter()
    pdf_path = Path(pdf_path)
    _log(f"Start | file={pdf_path} | collection={collection}", enabled=verbose)
    _log(
        "Endpoints | "
        f"qdrant={qdrant_url} ocr={ocr_endpoint} model={ocr_model_name} "
        f"ollama={ollama_api_base} embed_model={ollama_embed_model}",
        debug=True,
        enabled=verbose,
    )
    _log(f"Input file exists={pdf_path.exists()} | path={pdf_path}", debug=True, enabled=verbose)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input file not found: {pdf_path}")
    if pdf_path.suffix.lower() not in {".pdf", ".docx", ".doc", ".ppt", ".pptx"}:
        raise ValueError(
            "run_ingest expects PDF/Office file path ('.pdf', '.doc', '.docx', '.ppt', '.pptx'). "
            f"Received: {pdf_path}. "
            "If you already have chunks JSON, run embedding + vector-store ingest directly "
            "with src.embedder.embed_chunks_file and src.vector_store.ingest_embeddings."
        )
    resolved_pdf_path = resolve_to_pdf(pdf_path)
    page_count = _pdf_page_count(resolved_pdf_path)
    _log(
        f"Input size={_path_size_mb(pdf_path):.2f} MB | pages={page_count} | resolved_pdf={resolved_pdf_path}",
        debug=True,
        enabled=verbose,
    )

    # 1) PDF -> markdown (two-path router: pymupdf4llm for text-layer, LightOn OCR for scanned)
    _log("Parse PDF -> markdown (two-path router)", step=1, enabled=verbose)
    pages = parse_pdf(str(resolved_pdf_path), force_pipeline=force_pipeline)
    markdown = "".join(
        f"\n\n<!-- PAGE {i + 1} | {label} -->\n\n{page_text}"
        for i, (page_text, label) in enumerate(pages)
    )
    output_dir = REPO_ROOT / "data/output/lightonocr"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{resolved_pdf_path.stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    _log(
        f"Done | markdown={md_path} | pages={len(pages)} | chars={len(markdown)} | size={_path_size_mb(md_path):.2f} MB",
        step=1,
        enabled=verbose,
    )

    # 1a) Fix HTML tables with two-row headers (OCR drops colspan → group names in
    #     thead + sub-cols in first tbody row). Must run BEFORE table_processor.
    markdown = _fix_html_table_headers_in_markdown(markdown, verbose=verbose)

    # 1a-ii) Fill empty last-column cells from the PDF text layer (born-digital PDFs).
    #        OCR sometimes misses the rightmost column of wide tables.
    markdown = _fill_empty_last_column_from_text(markdown, resolved_pdf_path, verbose=verbose)

    # 1b-i) Convert any LaTeX-encoded tables ($\text{...}$) to markdown tables
    markdown = _convert_latex_tables_in_markdown(markdown, verbose=verbose)
    md_path.write_text(markdown, encoding="utf-8")

    # 1c) Convert ASCII grid tables to row sentences, save to data/output/processed/
    from src.preprocessing.table_processor import process_and_save as _process_md
    processed_dir = REPO_ROOT / "data" / "output" / "processed"
    markdown, _ = _process_md(markdown, md_path.stem, processed_dir)

    # 2) markdown -> chunks JSON
    _log("Chunk markdown -> chunks JSON", step=2, enabled=verbose)
    _log(f"Chunking config: enrich_with_llm={enrich_with_llm}", step=2, debug=True, enabled=verbose)
    # Reload chunker to pick up local prompt/chunking edits in long-running notebook kernels.
    chunker = importlib.reload(chunker_module)
    created_chunks = chunker.chunk_markdown(
        markdown=markdown,
        enrich_with_llm=enrich_with_llm,
        verbose=verbose,
        output_dir=REPO_ROOT / "data/output/chunks",
        file_name=md_path.name,
        source_file=str(pdf_path),
    )
    chunks_path = REPO_ROOT / "data/output/chunks" / f"{md_path.stem}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk output not found after chunking step: {chunks_path}")
    chunk_rows = _load_json_list(chunks_path, "chunks output")
    _log(
        f"Done | chunks={len(chunk_rows)} (in_memory={len(created_chunks)}) | file={chunks_path} | size={_path_size_mb(chunks_path):.2f} MB",
        step=2,
        enabled=verbose,
    )

    # 3) chunks JSON -> embeddings JSON (Ollama)
    _log("Embed chunks -> embeddings JSON", step=3, enabled=verbose)
    _log(
        f"Embedding config: api_base={ollama_api_base}, model={ollama_embed_model}, input_chunks={len(chunk_rows)}",
        step=3,
        debug=True,
        enabled=verbose,
    )
    embeddings_path = embed_chunks_file(
        input_path=chunks_path,
        output_path=REPO_ROOT / "data/output/embeddings" / f"{md_path.stem}_chunks_embeddings.json",
        api_base=ollama_api_base,
        model_name=ollama_embed_model,
        batch_size=2,
        verbose=verbose,
    )
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings output not found after embedding step: {embeddings_path}")
    embedding_rows = _load_json_list(embeddings_path, "embeddings output")
    vector_dim = 0
    if embedding_rows:
        first_vec = embedding_rows[0].get("embedding", [])
        if isinstance(first_vec, list):
            vector_dim = len(first_vec)
    _log(
        f"Done | vectors={len(embedding_rows)} | dim={vector_dim} | file={embeddings_path} | size={_path_size_mb(embeddings_path):.2f} MB",
        step=3,
        enabled=verbose,
    )

    # 4) embeddings JSON -> vector store (Qdrant)
    _log("Upsert embeddings -> Qdrant", step=4, enabled=verbose)
    deleted = delete_by_file(url=qdrant_url, collection=collection, file_name=md_path.name)
    if deleted > 0:
        _log(f"Removed {deleted} existing points for '{md_path.name}'", step=4, enabled=verbose)
    _log(
        f"Qdrant target: url={qdrant_url}, collection={collection}, points={len(embedding_rows)}",
        step=4,
        debug=True,
        enabled=verbose,
    )
    ingest_embeddings(
        input_path=embeddings_path,
        url=qdrant_url,
        collection=collection,
        verbose=verbose,
    )
    _log(f"Done | upserted_points={len(embedding_rows)} | collection={collection}", step=4, enabled=verbose)

    # 5) Load any tables embedded in the document into DuckDB (+ Qdrant summaries)
    #    so the SQL agent can answer aggregation / exact-lookup questions over them.
    _log("Load embedded tables -> DuckDB", step=5, enabled=verbose)
    try:
        from src.ingestion.pdf_tables import ingest_pdf_tables
        n_tables = ingest_pdf_tables(md_path.stem, markdown, collection=collection, verbose=verbose)
        _log(f"Done | tables_loaded={n_tables}", step=5, enabled=verbose)
    except Exception as exc:  # noqa: BLE001 — table load must not crash the doc's ingest
        _log(f"[WARN] embedded-table ingestion failed: {exc}", step=5, enabled=verbose)

    elapsed_seconds = time.perf_counter() - file_started_at
    _log(
        f"Pipeline finished successfully | pages={page_count} | elapsed={elapsed_seconds:.2f}s",
        enabled=verbose,
    )

    return {
        "markdown_path": md_path,
        "chunks_path": chunks_path,
        "embeddings_path": embeddings_path,
    }


def run_ingest(
    collection: str,
    pdf_path: Path | None = None,
    folder_path: Path | None = None,
    qdrant_url: str = "http://127.0.0.1:7333",
    ocr_endpoint: str = "http://127.0.0.1:8002/v1/chat/completions",
    ocr_model_name: str = "lightonocr-2-1b-ocr-soup",
    ocr_table_mode: str = "grid",
    data_format: str = "all",
    ollama_api_base: str = "http://127.0.0.1:11434",
    ollama_embed_model: str = OLLAMA_EMBED_MODEL,
    enrich_with_llm: bool = True,
    recursive: bool = False,
    verbose: bool = True,
    force_pipeline: str | None = None,
) -> dict[str, Path] | list[dict[str, Path]]:
    if (pdf_path is None) == (folder_path is None):
        raise ValueError("Provide exactly one of `pdf_path` or `folder_path`.")

    if pdf_path is not None:
        return _run_ingest_pdf(
            pdf_path=pdf_path,
            collection=collection,
            qdrant_url=qdrant_url,
            ocr_endpoint=ocr_endpoint,
            ocr_model_name=ocr_model_name,
            ocr_table_mode=ocr_table_mode,
            ollama_api_base=ollama_api_base,
            ollama_embed_model=ollama_embed_model,
            enrich_with_llm=enrich_with_llm,
            verbose=verbose,
            force_pipeline=force_pipeline,
        )

    target_dir = Path(folder_path)
    if not target_dir.exists():
        raise FileNotFoundError(f"Folder not found: {target_dir}")
    if not target_dir.is_dir():
        raise ValueError(f"Expected a directory for `folder_path`, got: {target_dir}")

    patterns_by_format = {
        "pdf": ["*.pdf"],
        "docx": ["*.docx", "*.doc"],
        "pptx": ["*.pptx", "*.ppt"],
        "all": ["*.pdf", "*.docx", "*.doc", "*.pptx", "*.ppt"],
    }
    if data_format not in patterns_by_format:
        raise ValueError("data_format must be one of: pdf, docx, pptx, all")

    _TABLE_EXTS = {".xlsx", ".xls", ".csv"}
    _DOC_PATTERNS = patterns_by_format[data_format]
    search_patterns = [f"**/{p}" for p in _DOC_PATTERNS] if recursive else _DOC_PATTERNS
    doc_files: list[Path] = []
    for pattern in search_patterns:
        doc_files.extend(target_dir.glob(pattern))

    table_pattern = "**/*.xlsx **/*.xls **/*.csv".split() if recursive else ["*.xlsx", "*.xls", "*.csv"]
    table_files: list[Path] = []
    for pattern in table_pattern:
        table_files.extend(target_dir.glob(pattern))

    all_files = sorted(set(doc_files + table_files))
    if not all_files:
        raise ValueError(
            f"No input files found in {target_dir} (recursive={recursive})"
        )

    _log(
        f"Folder mode | dir={target_dir} | docs={len(doc_files)} | tables={len(table_files)}",
        enabled=verbose,
    )
    results: list[dict[str, Path]] = []
    for index, file_path in enumerate(all_files, start=1):
        file_started_at = time.perf_counter()
        if file_path.suffix.lower() in _TABLE_EXTS:
            _log(f"File {index}/{len(all_files)} | {file_path.name} [TABLE]", enabled=verbose)
            out_path = ingest_table_rows(str(file_path), collection=collection, verbose=verbose)
            results.append({"chunks_path": out_path})
        else:
            file_size_mb = _path_size_mb(file_path)
            _log(
                f"File {index}/{len(all_files)} | {file_path.name} | size={file_size_mb:.2f} MB",
                enabled=verbose,
            )
            out = _run_ingest_pdf(
                pdf_path=file_path,
                collection=collection,
                qdrant_url=qdrant_url,
                ocr_endpoint=ocr_endpoint,
                ocr_model_name=ocr_model_name,
                ocr_table_mode=ocr_table_mode,
                ollama_api_base=ollama_api_base,
                ollama_embed_model=ollama_embed_model,
                enrich_with_llm=enrich_with_llm,
                verbose=False,
            )
            results.append(out)
        elapsed_seconds = time.perf_counter() - file_started_at
        _log(f"File {index}/{len(all_files)} done | elapsed={elapsed_seconds:.2f}s", enabled=verbose)

    _log(f"Folder ingest finished | processed={len(results)} files", enabled=verbose)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--pdf", type=Path, help="Single PDF or DOCX file")
    source_group.add_argument("--folder", type=Path, help="Directory containing PDF/DOCX files")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recursively scan --folder for PDF/DOCX files",
    )
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--ocr-endpoint", default=f"{OCR_API_BASE}/v1/chat/completions")
    parser.add_argument("--ocr-model-name", default=OCR_MODEL)
    parser.add_argument("--ocr-table-mode", choices=["pipe", "grid"], default="grid")
    parser.add_argument("--data-format", choices=["pdf", "docx", "pptx", "all"], default="all")
    parser.add_argument("--ollama-api-base", default=OLLAMA_API_BASE)
    parser.add_argument("--ollama-embed-model", default=OLLAMA_EMBED_MODEL)
    parser.add_argument(
        "--enrich-with-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable context generation during chunking",
    )
    parser.add_argument(
        "--force-pipeline",
        choices=["ocr", "text"],
        default=None,
        help="Override per-page routing: 'ocr' forces LightOn OCR on all pages, 'text' forces pymupdf4llm",
    )
    args = parser.parse_args()

    out = run_ingest(
        collection=args.collection,
        pdf_path=args.pdf,
        folder_path=args.folder,
        qdrant_url=args.qdrant_url,
        ocr_endpoint=args.ocr_endpoint,
        ocr_model_name=args.ocr_model_name,
        ocr_table_mode=args.ocr_table_mode,
        data_format=args.data_format,
        ollama_api_base=args.ollama_api_base,
        ollama_embed_model=args.ollama_embed_model,
        enrich_with_llm=args.enrich_with_llm,
        force_pipeline=args.force_pipeline,
        recursive=args.recursive,
        verbose=True,
    )
    if isinstance(out, list):
        print(f"Done. Processed {len(out)} files.")
    else:
        print(
            "Done.\n"
            f"markdown: {out['markdown_path']}\n"
            f"chunks: {out['chunks_path']}\n"
            f"embeddings: {out['embeddings_path']}"
        )


if __name__ == "__main__":
    main()
