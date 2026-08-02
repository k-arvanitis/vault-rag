"""Minimal shim: text-normalisation helpers extracted from the original chandra_parser.

The full chandra_parser (which required the proprietary `chandra` package) has been
removed. This shim retains the five functions imported by lightonocr_parser so the
active OCR pipeline continues to work without modification.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Literal

IMAGE_MD_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]*\)")
SUP_TAG_RE = re.compile(r"(?is)<sup>\s*(?P<sup>.*?)\s*</sup>")
SUB_TAG_RE = re.compile(r"(?is)<sub>\s*(?P<sub>.*?)\s*</sub>")
FOOTNOTE_PREFIX_RE = re.compile(r"(?m)^(?P<indent>\s*)\^(?P<label>[A-Za-z0-9]+):\s*")
TABLE_RE = re.compile(r"(?is)<table\b[^>]*>.*?</table>")
TABLE_START = "\n\n[TABLE_START]\n"
TABLE_END = "\n[TABLE_END]\n\n"
TITLE_LINE_RE = re.compile(
    r"""(?im)^\s*
    (?P<line>.*?
    table\s*\d+[a-z]?\s*
    (?:[:.\-\u2013\u2014])\s*
    .+?)\s*$
    """,
    re.VERBOSE,
)
ENABLE_NO_PUNCT_TITLES = False


def normalize_image_annotations(md_text: str) -> str:
    def _replace(match: re.Match) -> str:
        alt_text = " ".join(match.group("alt").split())
        if not alt_text:
            return "[IMAGE]"
        return f"[IMAGE] {alt_text}"

    return IMAGE_MD_RE.sub(_replace, md_text)


def normalize_superscripts(md_text: str) -> str:
    def _replace(match: re.Match) -> str:
        sup_text = " ".join(match.group("sup").split())
        return f"^{sup_text}" if sup_text else ""

    return SUP_TAG_RE.sub(_replace, md_text)


def normalize_subscripts(md_text: str) -> str:
    def _replace(match: re.Match) -> str:
        sub_text = " ".join(match.group("sub").split())
        return f"_{sub_text}" if sub_text else ""

    return SUB_TAG_RE.sub(_replace, md_text)


def normalize_footnote_prefixes(md_text: str) -> str:
    return FOOTNOTE_PREFIX_RE.sub(r"\g<indent>", md_text)


def _md_escape(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = " ".join(s.split())
    s = s.replace("|", r"\|")
    return s.strip()


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
    if any(len(c) > 160 for c in cols):
        return True
    if sum(len(c) for c in cols) > 600:
        return True
    if any(c.count(" > ") > 3 and len(c) > 120 for c in cols):
        return True
    return False


def normalize_table_title(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\*{1,3}\s*(.*?)\s*\*{1,3}$", r"\1", line)
    line = re.sub(r"\*{2,}", " ", line)
    line = re.sub(r"(?i)([A-Za-z0-9])(?=Table\s*\d)", r"\1 ", line)
    line = " ".join(line.split())
    return line


def html_table_to_markdown(
    table_html: str,
    mode: Literal["pipe", "grid"] = "grid",
    index: bool = False,
    max_cols_for_pipe: int = 10,
) -> str:
    import pandas as pd

    if mode == "pipe" and _table_is_complex(
        table_html, max_cols_for_pipe=max_cols_for_pipe
    ):
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
    return last


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
