"""Generate verified QA pairs from the structured data in DuckDB.

Every gold answer is read directly from a table cell or computed by SQL, so it is
provably correct — no LLM-invented answers. Lookups are keyed only on values that
are UNIQUE in their table, so each question has exactly one unambiguous answer.

Output: eval/data/qa_pairs/generated_verified_qa.json (table_lookup / numeric_*),
plus generated_unanswerable_qa.json (realistic absent-info questions).

Run: python -m eval.generate_verified_qa
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import duckdb

from src.config import DUCKDB_PATH

random.seed(7)
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "eval" / "data" / "qa_pairs"

# Human-readable document names so questions read naturally (not "doc_007").
DOC_NAMES = {
    "doc_005_fueling_records_invoice__table_1": ("doc_005", "NTSB fueling records invoice"),
    "doc_006_purchase_card_transactions_q1_2025_26__DataAnalysis": ("doc_006", "Doncaster Council purchase card transactions for Q1 2025/26"),
    "doc_007_published_spend_report_april_25__doc_007_published_spend_report_april_25": ("doc_007", "Doncaster Council published spend report for April 2025"),
    "doc_011_spain_open_data_maturity_questionnaire_2025__Portal": ("doc_011", "Spain open data maturity questionnaire (Portal section)"),
    "doc_014_bristol_supplier_spend_april_2024__doc_014_bristol_supplier_spend_april_2024": ("doc_014", "Bristol supplier spend for April 2024"),
}

# Per table: (unique-key column, [answerable target columns]). Keys are columns
# whose values are unique enough to identify one row.
LOOKUP_SPEC = {
    "doc_005_fueling_records_invoice__table_1": ("Tran #", ["Quantity", "Total", "Time", "Grade", "Unit Cost"]),
    "doc_007_published_spend_report_april_25__doc_007_published_spend_report_april_25": (
        "Transaction Number", ["Beneficiary", "Total", "Merchant Category", "Directorate", "Local Authority Dept"]),
    "doc_014_bristol_supplier_spend_april_2024__doc_014_bristol_supplier_spend_april_2024": (
        "Transaction Number", ["Name", "amount", "Pay Date"]),
    "doc_011_spain_open_data_maturity_questionnaire_2025__Portal": ("ID", ["Answer from 2024"]),
}

_SHEET = {  # DuckDB table -> sheet_name for evidence
    k: k.split("__", 1)[1] for k in DOC_NAMES
}


def _fmt(v) -> str:
    """Format a cell value to a clean gold string (drop trailing .0 on integers)."""
    if v is None:
        return ""
    s = str(v).strip()
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else f"{f:.2f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def _unique_key_rows(con, table, key, targets, n):
    """Return rows whose key value is unique in the table and targets are non-empty."""
    cols = ", ".join(f'"{c}"' for c in [key, *targets])
    df = con.execute(f'SELECT {cols} FROM "{table}"').df()
    counts = df[key].value_counts()
    uniq = df[df[key].map(lambda k: counts.get(k, 0) == 1)]
    uniq = uniq.dropna(subset=[key])
    rows = uniq.to_dict("records")
    random.shuffle(rows)
    out = []
    for r in rows:
        kv = _fmt(r[key])
        good_targets = [t for t in targets if _fmt(r.get(t)) and _fmt(r.get(t)).lower() not in ("nan", "0", "none")]
        if kv and good_targets:
            out.append((kv, r))
        if len(out) >= n:
            break
    return out


def gen_lookups(con) -> list[dict]:
    qa: list[dict] = []
    per_table = {"doc_007_published_spend_report_april_25__doc_007_published_spend_report_april_25": 52,
                 "doc_014_bristol_supplier_spend_april_2024__doc_014_bristol_supplier_spend_april_2024": 38,
                 "doc_011_spain_open_data_maturity_questionnaire_2025__Portal": 14,
                 "doc_005_fueling_records_invoice__table_1": 10}
    for table, (key, targets) in LOOKUP_SPEC.items():
        doc_id, name = DOC_NAMES[table]
        rows = _unique_key_rows(con, table, key, targets, per_table[table])
        for kv, r in rows:
            target = random.choice([t for t in targets if _fmt(r.get(t))])
            gold = _fmt(r[target])
            q = f"In the {name}, what is the {target} for {key} {kv}?"
            qa.append({
                "question": q, "gold_answer": gold, "answer_type": "short_text",
                "question_type": "table_lookup", "requires_cross_document": False,
                "document_count": 1, "source_document_ids": [doc_id],
                "gold_evidence": [{"doc_id": doc_id, "evidence_type": "sheet_row",
                                   "sheet_name": _SHEET[table],
                                   "quote": f"{key}: {kv}; {target}: {gold}"}],
            })
    return qa


def gen_aggregations(con) -> list[dict]:
    qa: list[dict] = []
    f = "doc_005_fueling_records_invoice__table_1"
    doc_id, name = DOC_NAMES[f]
    specs = [
        (f'SELECT ROUND(SUM("Quantity"),2) FROM "{f}"', f"In the {name}, what is the total Quantity across all transactions?"),
        (f'SELECT ROUND(SUM("Total"),2) FROM "{f}"', f"In the {name}, what is the total Total amount across all transactions?"),
        (f'SELECT COUNT(*) FROM "{f}"', f"How many transactions are in the {name}?"),
        (f'SELECT ROUND(MAX("Total"),2) FROM "{f}"', f"In the {name}, what is the largest single Total amount?"),
        (f'SELECT ROUND(AVG("Quantity"),2) FROM "{f}"', f"In the {name}, what is the average Quantity per transaction?"),
    ]
    d6 = "doc_006_purchase_card_transactions_q1_2025_26__DataAnalysis"
    did6, n6 = DOC_NAMES[d6]
    specs += [
        (f'SELECT COUNT(*) FROM "{d6}"', f"How many transactions are recorded in the {n6}?"),
        (f'SELECT COUNT(DISTINCT "Directorate") FROM "{d6}"', f"How many distinct Directorates appear in the {n6}?"),
    ]
    for sql, q in specs:
        gold = _fmt(con.execute(sql).fetchone()[0])
        did = did6 if d6 in sql else doc_id
        qa.append({
            "question": q, "gold_answer": gold, "answer_type": "number",
            "question_type": "numeric_reasoning", "requires_cross_document": False,
            "document_count": 1, "source_document_ids": [did],
            "gold_evidence": [{"doc_id": did, "evidence_type": "sheet_table",
                               "sheet_name": _SHEET.get(d6 if d6 in sql else f, ""),
                               "quote": f"computed: {sql}"}],
        })
    return qa


# Realistic questions whose answer is genuinely absent from these business docs.
# All clearly personal/identifying info that business transaction & policy
# documents do not contain — so "Unsupported" is unambiguously the right answer.
UNANSWERABLE = [
    ("What is the home address of the authorizing manager named in the procurement policy?", "doc_001"),
    ("What is the personal mobile phone number of the authorizing manager in the procurement policy?", "doc_001"),
    ("What is the email address of the supplier Cash Converters in the purchase card transactions?", "doc_006"),
    ("What is the phone number of the supplier Screwfix Direct in the purchase card transactions?", "doc_006"),
    ("What is the bank sort code for transaction 6110140 in the published spend report?", "doc_007"),
    ("What is the date of birth of the beneficiary for transaction 6110140 in the published spend report?", "doc_007"),
    ("What is the National Insurance number of any beneficiary in the published spend report?", "doc_007"),
    ("What is the email address of any supplier in the Bristol supplier spend?", "doc_014"),
    ("What is the driver's name recorded in the NTSB fueling records invoice?", "doc_005"),
    ("What is the home address of the Federal Reserve chair in the annual report?", "doc_003"),
    ("What is the personal email of any employee in the Rosemont employee handbook?", "doc_010"),
    ("What is the credit card number used for the Google Ads transaction in the purchase card report?", "doc_006"),
]


def gen_unanswerable() -> list[dict]:
    return [{
        "question": q, "gold_answer": "Unsupported", "answer_type": "unanswerable",
        "question_type": "unanswerable", "requires_cross_document": False,
        "document_count": 1, "source_document_ids": [doc_id], "gold_evidence": [],
    } for q, doc_id in UNANSWERABLE]


def main() -> None:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    lookups = gen_lookups(con)
    aggs = gen_aggregations(con)
    unans = gen_unanswerable()
    con.close()

    for i, item in enumerate(lookups + aggs, start=1):
        item["qa_id"] = f"gen_{i}"
    for i, item in enumerate(unans, start=1):
        item["qa_id"] = f"genu_{i}"

    (OUT / "generated_verified_qa.json").write_text(
        json.dumps(lookups + aggs, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "generated_unanswerable_qa.json").write_text(
        json.dumps(unans, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"lookups={len(lookups)} aggregations={len(aggs)} unanswerable={len(unans)} "
          f"| total new={len(lookups)+len(aggs)+len(unans)}")


if __name__ == "__main__":
    main()
