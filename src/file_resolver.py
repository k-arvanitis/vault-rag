"""Resolve parsed .md filenames back to their original source (PDF/XLSX/etc)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_DIRS = (
    REPO_ROOT / "data" / "input",
    REPO_ROOT / "eval" / "data" / "raw",
)


@lru_cache(maxsize=1)
def _stem_map() -> dict[str, str]:
    """Map of {stem: original_filename_with_extension} across all source dirs."""
    out: dict[str, str] = {}
    for d in ORIGINAL_DIRS:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() != ".md":
                out.setdefault(p.stem, p.name)
    return out


def resolve_original_name(filename: str) -> str:
    """Return the original source filename for a parsed .md, or filename unchanged."""
    p = Path(filename)
    if p.suffix.lower() != ".md":
        return p.name
    return _stem_map().get(p.stem, p.name)
