from __future__ import annotations

import argparse
import os
import re
from io import StringIO
from pathlib import Path
from typing import Literal, Union

from chandra.input import load_pdf_images
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from pypdf import PdfReader

from PIL import Image
from src.parser.input_utils import resolve_to_pdf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/output/chandra"

TABLE_RE = re.compile(r"(?is)<table\b[^>]*>.*?</table>")
TABLE_START = "\n\n[TABLE_START]\n"
TABLE_END = "\n[TABLE_END]\n\n"
IMAGE_MD_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]*\)")
SUP_TAG_RE = re.compile(r"(?is)<sup>\s*(?P<sup>.*?)\s*</sup>")
SUB_TAG_RE = re.compile(r"(?is)<sub>\s*(?P<sub>.*?)\s*</sub>")
FOOTNOTE_PREFIX_RE = re.compile(r"(?m)^(?P<indent>\s*)\^(?P<label>[A-Za-z0-9]+):\s*")
TITLE_LINE_RE = re.compile(
    r"""(?im)^\s*
    (?P<line>.*?
    table\s*\d+[a-z]?\s*
    (?:[:.\-–—])\s*
    .+?)\s*$
    """,
    re.VERBOSE,
)
TITLE_LINE_NO_PUNCT_RE = re.compile(
    r"""(?im)^\s*
    (?P<line>.*?
    table\s*\d+[a-z]?\s+
    .+?)\s*$
    """,
    re.VERBOSE,
)
ENABLE_NO_PUNCT_TITLES = False


def _md_escape(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    s = s.replace("|", r"\|")
    return s.strip()


def _flatten_columns(df):
    if hasattr(df.columns, "values") and type(df.columns).__name__ == "MultiIndex":
        new_cols = []
        for col in df.columns.values:
            parts = [str(x).strip() for x in col if str(x).strip() and str(x) != "nan"]
            dedup = []
            for p in parts:
                if not dedup or dedup[-1] != p:
                    dedup.append(p)
            new_cols.append(" > ".join(dedup) if dedup else "")
        df.columns = new_cols
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _sanitize_df_for_markdown(df):
    df = df.copy()
    df = _flatten_columns(df)
    df.columns = [_md_escape(c) for c in df.columns]
    for c in df.columns:
        df[c] = df[c].map(_md_escape)
    return df


def _is_malformed_df(df) -> bool:
    cols = [str(c) for c in df.columns]
    if not cols:
        return True

    # Pandas can mis-parse complex HTML headers and produce giant synthetic names.
    if any(len(c) > 160 for c in cols):
        return True
    if sum(len(c) for c in cols) > 600:
        return True
    if any(c.count(" > ") > 3 and len(c) > 120 for c in cols):
        return True

    return False


def _table_is_complex(table_html: str, max_cols_for_pipe: int = 10) -> bool:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(table_html, "lxml")
    if soup.select("[rowspan], [colspan]"):
        return True
    first_tr = soup.find("tr")
    if first_tr:
        cells = first_tr.find_all(["th", "td"])
        approx_cols = sum(int(x.get("colspan", 1)) for x in cells)
        if approx_cols > max_cols_for_pipe:
            return True
    return False


def normalize_table_title(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", line)
    line = re.sub(r"\*{2,}", " ", line)
    line = re.sub(r"(?i)([A-Za-z0-9])(?=Table\s*\d)", r"\1 ", line)
    line = " ".join(line.split())
    return line


def normalize_image_annotations(md_text: str) -> str:
    def _replace(match) -> str:
        alt_text = " ".join(match.group("alt").split())
        if not alt_text:
            return "[IMAGE]"
        return f"[IMAGE] {alt_text}"

    return IMAGE_MD_RE.sub(_replace, md_text)


def normalize_superscripts(md_text: str) -> str:
    def _replace(match) -> str:
        sup_text = " ".join(match.group("sup").split())
        return f"^{sup_text}" if sup_text else ""

    return SUP_TAG_RE.sub(_replace, md_text)


def normalize_subscripts(md_text: str) -> str:
    def _replace(match) -> str:
        sub_text = " ".join(match.group("sub").split())
        return f"_{sub_text}" if sub_text else ""

    return SUB_TAG_RE.sub(_replace, md_text)


def normalize_footnote_prefixes(md_text: str) -> str:
    return FOOTNOTE_PREFIX_RE.sub(r"\g<indent>", md_text)


def html_table_to_markdown(
    table_html: str,
    mode: Literal["pipe", "grid"] = "grid",
    index: bool = False,
    max_cols_for_pipe: int = 10,
) -> str:
    import pandas as pd

    if mode == "pipe" and _table_is_complex(table_html, max_cols_for_pipe=max_cols_for_pipe):
        mode = "grid"

    dfs = pd.read_html(StringIO(table_html), flavor="lxml")
    out = []

    for df in dfs:
        df = df.fillna("")
        df = _sanitize_df_for_markdown(df)
        if _is_malformed_df(df):
            continue

        if mode == "pipe":
            rendered = df.to_markdown(index=index, tablefmt="github")
        else:
            from tabulate import tabulate

            grid = tabulate(df, headers="keys", tablefmt="grid", showindex=index)
            rendered = f"```text\n{grid}\n```"

        out.append(rendered)

    if not out:
        return table_html

    return "\n\n".join(out)


def _find_last_title_in_window(window: str):
    last = None
    for m in TITLE_LINE_RE.finditer(window):
        last = m
    if last is not None:
        return last
    if ENABLE_NO_PUNCT_TITLES:
        for m in TITLE_LINE_NO_PUNCT_RE.finditer(window):
            last = m
        return last
    return None


def replace_html_tables_with_titles(
    md_text: str,
    mode: Literal["pipe", "grid"] = "grid",
    lookback_chars: int = 1200,
) -> str:
    pieces = []
    last_end = 0

    for m in TABLE_RE.finditer(md_text):
        t_start, t_end = m.span()
        before = md_text[last_end:t_start]
        table_html = m.group(0)

        window = before[-lookback_chars:]
        title_match = _find_last_title_in_window(window)

        pulled_title = ""
        if title_match:
            raw_line = title_match.group("line").strip()
            raw_line = normalize_table_title(raw_line)
            pulled_title = raw_line

            win_start = len(before) - len(window)
            abs0 = win_start + title_match.start(0)
            abs1 = win_start + title_match.end(0)

            line_start = before.rfind("\n", 0, abs0) + 1
            line_end = before.find("\n", abs1)
            if line_end == -1:
                before = before[:line_start]
            else:
                before = before[:line_start] + before[line_end + 1 :]

        try:
            rendered = html_table_to_markdown(table_html, mode=mode)
        except Exception:
            rendered = table_html

        block = TABLE_START
        if pulled_title:
            block += pulled_title + "\n\n"
        block += rendered + TABLE_END

        pieces.append(before)
        pieces.append(block)
        last_end = t_end

    pieces.append(md_text[last_end:])
    return "".join(pieces)


def pdf_to_markdown(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    api_base: str = "http://127.0.0.1:8001/v1",
    model_name: str = "chandra",
    markdown_tables: bool = False,
    table_mode: Literal["pipe", "grid"] = "grid",
) -> Path:
    pdf_path = resolve_to_pdf(pdf_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    os.environ["VLLM_API_BASE"] = api_base
    os.environ["VLLM_MODEL_NAME"] = model_name

    n_pages = len(PdfReader(str(pdf_path)).pages)
    page_range = list(range(n_pages))

    manager = InferenceManager(method="vllm")
    pil_images = load_pdf_images(str(pdf_path), page_range=page_range)
    batch = [BatchInputItem(image=im, prompt_type="ocr_layout") for im in pil_images]
    results = manager.generate(batch)


    combined: list[str] = []
    for page_idx, result in enumerate(results, start=1):
        combined.append(f"\n\n<!-- PAGE {page_idx} -->\n\n")
        combined.append(result.markdown)

    markdown_content = "".join(combined)
    if markdown_tables:
        markdown_content = replace_html_tables_with_titles(markdown_content, mode=table_mode)
    markdown_content = normalize_superscripts(markdown_content)
    markdown_content = normalize_subscripts(markdown_content)
    markdown_content = normalize_footnote_prefixes(markdown_content)
    markdown_content = normalize_image_annotations(markdown_content)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}.md"
    out_path.write_text(markdown_content, encoding="utf-8")
    return out_path


def png_to_markdown(
    image_path: Union[str, Path],
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    api_base: str = "http://127.0.0.1:8001/v1",
    model_name: str = "chandra",
    markdown_tables: bool = False,
    table_mode: Literal["pipe", "grid"] = "grid",
) -> Path:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    # Setup environment for the VLLM backend
    os.environ["VLLM_API_BASE"] = api_base
    os.environ["VLLM_MODEL_NAME"] = model_name

    # 1. Load the PNG directly using PIL
    with Image.open(image_path) as img:
        # Convert to RGB to ensure compatibility (handles RGBA/transparency)
        img_rgb = img.convert("RGB")
        
        # 2. Initialize the Manager and wrap the image in a BatchItem
        manager = InferenceManager(method="vllm")
        batch = [BatchInputItem(image=img_rgb, prompt_type="ocr_layout")]
        
        # 3. Generate the OCR results
        results = manager.generate(batch)

    # 4. Extract Markdown (since it's one image, we just take the first result)
    markdown_content = results[0].markdown
    if markdown_tables:
        markdown_content = replace_html_tables_with_titles(markdown_content, mode=table_mode)
    markdown_content = normalize_superscripts(markdown_content)
    markdown_content = normalize_subscripts(markdown_content)
    markdown_content = normalize_footnote_prefixes(markdown_content)
    markdown_content = normalize_image_annotations(markdown_content)

    # 5. Save and return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}.md"
    out_path.write_text(markdown_content, encoding="utf-8")
    
    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chandra OCR (vLLM) on a PDF/Office file or PNG and save markdown."
    )
    parser.add_argument("input_path", help="Path to a PDF/Office file (.doc/.docx/.ppt/.pptx) or PNG.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where output markdown will be saved.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8001/v1",
        help="vLLM OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model-name",
        default="chandra",
        help="Model name exposed by vLLM.",
    )
    parser.add_argument(
        "--markdown-tables",
        action="store_true",
        help="Convert HTML tables to markdown blocks with [TABLE_START]/[TABLE_END].",
    )
    parser.add_argument(
        "--table-mode",
        choices=["pipe", "grid"],
        default="grid",
        help="Markdown table rendering mode when --markdown-tables is enabled.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".pdf", ".docx", ".doc", ".ppt", ".pptx"}:
        out_path = pdf_to_markdown(
            pdf_path=input_path,
            output_dir=Path(args.output_dir),
            api_base=args.api_base,
            model_name=args.model_name,
            markdown_tables=args.markdown_tables,
            table_mode=args.table_mode,
        )
    elif suffix == ".png":
        out_path = png_to_markdown(
            image_path=input_path,
            output_dir=Path(args.output_dir),
            api_base=args.api_base,
            model_name=args.model_name,
            markdown_tables=args.markdown_tables,
            table_mode=args.table_mode,
        )
    else:
        raise ValueError(f"Unsupported input type: {suffix}. Use .pdf/.doc/.docx/.ppt/.pptx or .png")

    print(f"Saved markdown to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
