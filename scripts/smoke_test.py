"""End-to-end backend smoke test: reingest real documents, ask real questions,
verify the answer pipeline actually works — not mocked, hits live Qdrant/DuckDB/LLM.

Not part of the pytest suite (which mocks all external services per convention) —
run manually with `make smoke` before a release, or whenever ingestion, retrieval,
or generation code changes in a way that unit tests can't catch.

Checks:
  1. PDF reingest -> ask a prose question -> answer has a valid citation.
  2. Excel reingest -> ask a table-lookup question -> answer contains the right value.
  3. Cross-document comparison -> both named documents appear in sources.

Exits 1 if any check fails, prints a pass/fail line per check either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answer_pipeline import answer_query
from src.config import QDRANT_COLLECTION
from src.rag_agent import build_rag_agent

RAW_DIR = Path(__file__).resolve().parent.parent / "eval" / "data" / "raw"

_failures: list[str] = []


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        _failures.append(label)


def check_pdf_reingest_and_query(agent) -> None:
    """Reingest a real PDF, then ask a question and verify a real citation comes back."""
    from src.ingest import run_ingest

    pdf_path = RAW_DIR / "doc_001_procurement_policy.pdf"
    print(f"\n-- Reingesting {pdf_path.name} --")
    result = run_ingest(pdf_path=pdf_path, collection=QDRANT_COLLECTION, verbose=False)
    _check("PDF reingest completes", isinstance(result, dict) and bool(result))

    print("-- Asking a prose question against it --")
    r = answer_query(agent, "What is the title of this procurement policy document?")
    _check(
        "PDF question returns a non-empty answer",
        bool(r["answer"]) and r["answer"].strip().lower() != "unsupported",
        detail=repr(r["answer"])[:200],
    )
    _check(
        "PDF question returns at least one source",
        len(r["sources"]) > 0,
        detail=f"sources={r['sources']}",
    )
    if r["sources"]:
        src = r["sources"][0]
        _check(
            "Citation has a real filename and a quote",
            bool(src.get("filename")) and bool(src.get("quote")),
            detail=repr(src)[:200],
        )


def check_excel_reingest_and_query(agent) -> None:
    """Reingest a real spreadsheet, then ask a table-lookup question with a known answer."""
    from src.ingest_table_rows import ingest_table_rows

    xlsx_path = RAW_DIR / "doc_006_purchase_card_transactions_q1_2025_26.xlsx"
    print(f"\n-- Reingesting {xlsx_path.name} --")
    chunks_path = ingest_table_rows(str(xlsx_path), collection=QDRANT_COLLECTION, verbose=False)
    _check("Excel reingest completes", chunks_path is not None and Path(chunks_path).exists())

    print("-- Asking a table-lookup question against it --")
    r = answer_query(
        agent,
        "For Supplier Name 'Screwfix Direct' on 2025-04-03 in PLACE / STREET SCENE, "
        "what is the NET Amount?",
    )
    _check(
        "Excel question answer contains the known gold value (39.54)",
        "39.54" in r["answer"],
        detail=repr(r["answer"])[:200],
    )


def check_cross_document_comparison(agent) -> None:
    """A comparison question naming two real documents must cite both."""
    print("\n-- Asking a cross-document comparison question --")
    r = answer_query(
        agent,
        "Which document allows a longer extension period: the procurement policy "
        "or the services contract terms?",
    )
    filenames = {s["filename"] for s in r["sources"]}
    both_present = any("procurement" in f.lower() for f in filenames) and any(
        "services_contract" in f.lower() for f in filenames
    )
    _check(
        "Comparison answer cites both named documents",
        both_present,
        detail=f"filenames={filenames}",
    )
    _check(
        "Comparison answer is not a flat refusal",
        r["answer"].strip().lower() != "unsupported",
        detail=repr(r["answer"])[:200],
    )


def main() -> int:
    print("Building agent (loads reranker, connects Qdrant/DuckDB)...")
    agent = build_rag_agent()

    check_pdf_reingest_and_query(agent)
    check_excel_reingest_and_query(agent)
    check_cross_document_comparison(agent)

    print(f"\n{'='*60}")
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
