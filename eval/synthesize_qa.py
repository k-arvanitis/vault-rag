"""
Generate QA pairs for single-doc narrative questions using DeepEval Synthesizer.
Covers doc_001–doc_005 and doc_008 (PDFs) + doc_006/doc_007 (tables).
Run: uv run python eval/synthesize_qa.py
"""

import json
import os
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data/output/processed"
TABLE_MD_DIR = REPO_ROOT / "data/output/table_markdowns"
OUT_PATH = Path(__file__).parent / "questions.jsonl"

# Single-doc narrative docs — synthesizer handles these
NARRATIVE_DOCS = [
    ("doc_001", PROCESSED_DIR / "doc_001_procurement_policy.md"),
    ("doc_002", PROCESSED_DIR / "doc_002_services_contract_terms.md"),
    ("doc_003", PROCESSED_DIR / "doc_003_fed_annual_report_2024.md"),
    ("doc_008", PROCESSED_DIR / "doc_008_gao_24_106915.md"),
]

# Table docs
TABLE_DOCS = [
    (
        "doc_006",
        TABLE_MD_DIR / "doc_006_purchase_card_transactions_q1_2025_26__DataAnalysis.md",
    ),
    (
        "doc_007",
        TABLE_MD_DIR
        / "doc_007_published_spend_report_april_25__doc_007_published_spend_report_april_25.md",
    ),
]

# OCR docs — hand-written questions only, skip synthesis
# doc_004, doc_005


def synthesize_for_doc(doc_id: str, md_path: Path, n: int = 8) -> list[dict]:
    """Generate n QA pairs from a single markdown document."""
    if not md_path.exists():
        print(f"  [SKIP] {doc_id} — {md_path.name} not found")
        return []

    text = md_path.read_text(encoding="utf-8")
    # Trim to avoid huge token counts on large docs
    text = text[:40_000]

    styling = StylingConfig(
        scenario=f"A researcher is auditing document {doc_id} for factual accuracy.",
        task="Answer questions about the document content based only on what is stated in the document.",
        expected_output_format="A concise factual answer, one to three sentences.",
    )

    synthesizer = Synthesizer(
        model="gpt-4o-mini", async_mode=False, styling_config=styling
    )
    contexts = [[text]]
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=n,
        source_files=[str(md_path)],
    )

    results = []
    for i, golden in enumerate(goldens):
        results.append(
            {
                "doc_id": doc_id,
                "question": golden.input,
                "gold_answer": golden.expected_output,
                "context": golden.context,
            }
        )
        print(f"  [{doc_id}] Q{i + 1}: {golden.input[:80]}")
    return results


def main() -> None:
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    all_goldens = []

    print("=== Narrative PDFs ===")
    for doc_id, path in NARRATIVE_DOCS:
        print(f"\n{doc_id} ({path.name})")
        goldens = synthesize_for_doc(doc_id, path, n=8)
        all_goldens.extend(goldens)

    print("\n=== Table docs ===")
    for doc_id, path in TABLE_DOCS:
        print(f"\n{doc_id} ({path.name})")
        goldens = synthesize_for_doc(doc_id, path, n=5)
        all_goldens.extend(goldens)

    # Write output
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in all_goldens:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_goldens)} QA pairs → {OUT_PATH}")


if __name__ == "__main__":
    main()
