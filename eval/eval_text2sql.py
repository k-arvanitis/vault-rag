"""Measure the text-to-SQL agent (src/tools/excel.py) in isolation.

`table_lookup` correctness in run_eval.py conflates three different things: did the
agent pick the right table, did it write correct SQL, and did the final natural-
language formatting pass phrase the value the way the gold answer happens to be
phrased. A wrong NL phrasing (or a judge/date-format quirk) then reads as a SQL bug.

This script isolates SQL execution accuracy: it re-executes the exact SQL the agent
generated (captured via query_excel's artifact) and compares the raw DuckDB result
directly against gold_answer, bypassing the NL-formatting model call entirely.

Usage:
    uv run python eval/eval_text2sql.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from eval.run_eval import _exact_match_score, _load_questions_for_run  # noqa: E402
from vault_rag.duckdb_store import DuckDBStore  # noqa: E402
from vault_rag.tools.excel import _execute_sql, build_excel_agent_tools  # noqa: E402

RESULTS_PATH = REPO_ROOT / "eval" / "results" / "text2sql_eval.json"


def _raw_sql_value(store: DuckDBStore, sql: str) -> tuple[bool, str]:
    """Re-execute one generated SQL statement and return (ran_ok, raw_result_text)."""
    try:
        ok, result, single_col, _ambiguous = _execute_sql(store, sql)
    except Exception as exc:
        return False, f"execution error: {exc}"
    if not ok:
        return False, result
    return True, single_col if single_col is not None else result


def evaluate_text2sql(category_filter: str = "table_lookup") -> dict[str, Any]:
    """Run every table_lookup question through query_excel and score SQL execution
    accuracy directly, independent of the agent's final NL-formatted answer."""
    store = DuckDBStore()
    tool = build_excel_agent_tools(store)[0]
    questions = _load_questions_for_run(category_filter, None)

    rows: list[dict[str, Any]] = []
    for q in questions:
        question_text = q["question"]
        gold = str(q.get("gold_answer", ""))
        try:
            nl_answer, artifact = tool.func(question=question_text)
        except Exception as exc:
            rows.append(
                {
                    "qa_id": q["qa_id"],
                    "question": question_text,
                    "gold_answer": gold,
                    "sql": None,
                    "sql_ran_ok": False,
                    "raw_value": f"tool error: {exc}",
                    "nl_answer": None,
                    "sql_value_score": 0.0,
                    "nl_answer_score": 0.0,
                }
            )
            continue

        sql_list = (artifact or {}).get("sql") or []
        sql = sql_list[0] if sql_list else None
        if sql:
            ran_ok, raw_value = _raw_sql_value(store, sql)
        else:
            ran_ok, raw_value = (
                False,
                "(no SQL captured — agent abstained before writing one)",
            )

        rows.append(
            {
                "qa_id": q["qa_id"],
                "question": question_text,
                "gold_answer": gold,
                "sql": sql,
                "sql_ran_ok": ran_ok,
                "raw_value": raw_value,
                "nl_answer": nl_answer,
                "sql_value_score": _exact_match_score(raw_value, gold)
                if ran_ok
                else 0.0,
                "nl_answer_score": _exact_match_score(nl_answer, gold),
            }
        )

    n = len(rows)
    sql_execution_success_rate = sum(r["sql_ran_ok"] for r in rows) / n if n else 0.0
    sql_value_accuracy = (
        sum(r["sql_value_score"] == 1.0 for r in rows) / n if n else 0.0
    )
    nl_answer_accuracy = (
        sum(r["nl_answer_score"] == 1.0 for r in rows) / n if n else 0.0
    )
    # Cases where the raw SQL value was right but the NL-formatting step corrupted it —
    # the gap this script exists to isolate.
    formatting_regressions = [
        r for r in rows if r["sql_value_score"] == 1.0 and r["nl_answer_score"] < 1.0
    ]

    summary = {
        "question_count": n,
        "sql_execution_success_rate": sql_execution_success_rate,
        "sql_value_accuracy": sql_value_accuracy,
        "nl_answer_accuracy": nl_answer_accuracy,
        "formatting_regression_count": len(formatting_regressions),
        "formatting_regression_qa_ids": [r["qa_id"] for r in formatting_regressions],
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    result = evaluate_text2sql()
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nFull per-question detail written to {RESULTS_PATH}")
