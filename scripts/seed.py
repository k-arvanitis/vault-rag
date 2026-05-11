"""Download and ingest a starter document set.

Covers both retrieval paths so the demo is immediately queryable:
  - Vector (Qdrant): two born-digital PDFs for prose and policy queries
  - SQL (DuckDB):    one CSV for spend filtering and aggregation

Run via:  make seed
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

SEED_DOCS = [
    {
        "doc_id": "doc_001",
        "file_name": "doc_001_procurement_policy.pdf",
        "url": (
            "https://www.lacera.gov/sites/default/files/assets/documents/board/"
            "Governing%20Documents/General%20Policies/Purchasing_Policy_Goods_Services.pdf"
        ),
        "ingest": "pdf",
        "label": "LACERA procurement policy (PDF)",
    },
    {
        "doc_id": "doc_002",
        "file_name": "doc_002_services_contract_terms.pdf",
        "url": (
            "https://assets.publishing.service.gov.uk/media/"
            "5abcfd7fed915d44eb7e6969/Terms_and_Conditions_for_Services.pdf"
        ),
        "ingest": "pdf",
        "label": "UK Government contract terms (PDF)",
    },
    {
        "doc_id": "doc_007",
        "file_name": "doc_007_published_spend_report_april_25.csv",
        "url": (
            "https://www.doncaster.gov.uk/Documents/DocumentView/Stream/Media/Default/"
            "Council%20and%20Democracy/Published%20Spend%20Report%20April%2025.csv"
        ),
        "ingest": "table",
        "label": "Doncaster Council spend report (CSV → DuckDB)",
    },
]

RAW_DIR = Path("eval/data/raw")

SAMPLE_QUESTIONS = [
    'What is the LACERA policy on sole-source procurement?',
    'Both documents include rules about termination. What are they and how do they differ?',
    'In the Doncaster spend report, list the payments to Arco Limited.',
]


def _download(url: str, dest: Path) -> None:
    """Download url to dest, skipping if already present."""
    if dest.exists():
        print(f"  already present: {dest.name}")
        return
    print(f"  downloading {dest.name} ...", end="", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vault-rag/seed"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
    except Exception as exc:
        dest.unlink(missing_ok=True)
        print(f" FAILED ({exc})")
        sys.exit(1)
    print(f" done ({dest.stat().st_size // 1024} KB)")


def _ingest_pdf(path: Path) -> None:
    """Ingest a born-digital PDF via the standard ingestion pipeline."""
    subprocess.run(
        ["uv", "run", "python", "-m", "src.ingest", "--pdf", str(path)],
        check=True,
    )


def _ingest_table(path: Path) -> None:
    """Ingest an Excel/CSV file: cleaned rows → DuckDB, metadata → Qdrant.

    Qdrant gets only a document_summary + per-sheet sheet_summary (for discovery);
    structured queries for these documents are answered from DuckDB.
    """
    subprocess.run(
        ["uv", "run", "python", "-m", "src.ingest_table_rows", str(path)],
        check=True,
    )


def main() -> None:
    """Download and ingest the seed document set."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for doc in SEED_DOCS:
        dest = RAW_DIR / doc["file_name"]
        print(f"\n[{doc['doc_id']}] {doc['label']}")
        _download(doc["url"], dest)
        print("  ingesting ...", flush=True)
        if doc["ingest"] == "pdf":
            _ingest_pdf(dest)
        else:
            _ingest_table(dest)

    print("\n" + "─" * 60)
    print("Seed complete. Try these sample questions in the Streamlit UI:")
    for q in SAMPLE_QUESTIONS:
        print(f'  • "{q}"')
    print()
    print("Note: the CSV spend report is also available via DuckDB SQL if")
    print("eval/data/raw/doc_007_published_spend_report_april_25.csv is listed")
    print("in the EXCEL_FILES env var (set by default in .env.example).")


if __name__ == "__main__":
    main()
