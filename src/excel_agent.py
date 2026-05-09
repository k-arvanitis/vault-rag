"""LangGraph-orchestrated text-to-SQL agent for Excel/CSV questions.

Architecture
------------
Outer graph (per call):
    decompose ─→ Send(per-subquestion) ─→ inner_graph ─→ synthesize ─→ END

Inner graph (per subquestion):
    select_table → inspect → write_sql → run_sql → evaluate ─→ END
                                  ▲                    │
                                  └──── retry ─────────┘   (retries on 0 rows / SQL errors)

The split fixes two failure modes:
1. Cross-document Excel questions used to be answered with a single SQL pass — the
   `decompose` node now produces one subquestion per source document.
2. Single-doc lookups failed silently when the first SQL returned no rows. The inner
   loop now feeds the empty result back to the model and re-writes SQL up to N times.
"""
from __future__ import annotations

import json
import operator
import re
from typing import Annotated, Any, TypedDict

import openai
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from src.config import (
    EXCEL_AGENT_API_BASE,
    EXCEL_AGENT_API_KEY,
    EXCEL_AGENT_MODEL,
)
from src.excel_tool import DuckDBStore, _normalize_sql, _truncate_ilike

_MAX_SQL_ATTEMPTS = 2
_ROW_LIMIT = 50

_DOC_TABLE_PREFIX_RE = re.compile(r"^doc_\d+_", re.IGNORECASE)

_TABLE_STOPWORDS = {
    # Generic English function words only — never strip column-name candidates
    # (amount, total, name, number, date, etc. are real columns in our schemas).
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "of", "in", "on",
    "for", "to", "with", "by", "at", "from", "what", "which", "who", "whom",
    "where", "when", "why", "how", "this", "that", "these", "those", "it", "its",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "row", "rows", "table", "tables",
    "doc", "document", "file", "sheet",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Drop <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _llm_chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 700) -> str:
    """One-shot chat completion against the configured Excel-agent endpoint."""
    if not EXCEL_AGENT_API_KEY:
        raise RuntimeError("EXCEL_AGENT_API_KEY is not configured")
    client = openai.OpenAI(base_url=EXCEL_AGENT_API_BASE, api_key=EXCEL_AGENT_API_KEY)
    resp = client.chat.completions.create(
        model=EXCEL_AGENT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _strip_think(resp.choices[0].message.content or "")


def _doc_tables(store: DuckDBStore) -> dict[str, list[str]]:
    """Return only the doc_NNN_* tables, dropping legacy / test artefacts."""
    return {t: c for t, c in store.tables().items() if _DOC_TABLE_PREFIX_RE.match(t)}


def _question_keywords(question: str) -> set[str]:
    """Keyword set used for ranking tables and detecting doc-name mentions."""
    return {
        t for t in re.findall(r"[a-z][a-z0-9]{2,}", question.lower())
        if t not in _TABLE_STOPWORDS
    }


def _rank_tables(tables: dict[str, list[str]], question: str) -> list[str]:
    """Order tables by descending overlap of question tokens with COLUMN NAMES.

    Ranking by column names is more discriminating than table names — questions
    mention the field they want ("Beneficiary", "Merchant Category") which maps
    cleanly to the column that holds it. Falls back to table-name overlap on ties.
    """
    q_tokens = _question_keywords(question)

    def score(tname: str) -> tuple[int, int, str]:
        col_tokens: set[str] = set()
        for col in tables.get(tname, []):
            col_tokens.update(re.findall(r"[a-z][a-z0-9]{2,}", col.lower()))
        col_score = len(q_tokens & col_tokens)
        name_tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", tname.lower()))
        name_score = len(q_tokens & name_tokens)
        return (-col_score, -name_score, tname)

    return sorted(tables.keys(), key=score)


def _table_summary(tname: str, schema: list[tuple[str, str]]) -> str:
    """One-line schema digest for the table listing."""
    cols = ", ".join(f'"{c}" ({t})' for c, t in schema)
    return f'"{tname}": {cols}'


def _format_samples(samples: list[dict]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in samples)


def _execute_sql(store: DuckDBStore, sql: str) -> tuple[bool, str, str | None]:
    """Run SQL and return (ok, formatted_text, single_col_first_value).

    - ok=True with rendered table on success (including 0-row results)
    - ok=False with `SQL error: ...` on failure
    - single_col_first_value: the first row's only-cell value when the SQL
      projects exactly one column; None otherwise. Used as a deterministic
      fallback when the format LLM punts on multi-row single-column results.

    Auto-retries 0-row ILIKE queries with progressively-truncated string filters
    to handle truncated supplier/beneficiary names common in the source data.
    """
    sql = _normalize_sql(sql)
    try:
        df = store.execute(sql)
    except Exception as exc:
        return False, f"SQL error: {exc}", None

    if df.empty:
        for trim in (1, 2, 3):
            relaxed = _truncate_ilike(sql, trim)
            if relaxed == sql:
                break
            try:
                df = store.execute(relaxed)
            except Exception:
                break
            if not df.empty:
                sql = relaxed
                break

    if df.empty:
        return True, "Query returned 0 rows.", None
    # Aggregate queries (SUM/COUNT/etc.) on a missing key collapse to all-NaN —
    # treat that as an empty match so the agent retries on a different table/filter
    # rather than reporting "NaN" to the user.
    if df.shape == (1, 1) and df.iloc[0, 0] != df.iloc[0, 0]:  # NaN check
        return True, "Query returned 0 rows.", None

    single_col_first = None
    if df.shape[1] == 1:
        first = df.iloc[0, 0]
        if first is not None and first == first:  # not NaN
            single_col_first = str(first)

    truncated = len(df) > _ROW_LIMIT
    out = df.head(_ROW_LIMIT).to_string(index=False, max_colwidth=80)
    if truncated:
        out += f"\n… (truncated at {_ROW_LIMIT} rows)"
    return True, out, single_col_first


# ---------------------------------------------------------------------------
# Decomposition: split cross-document questions into per-doc subqueries
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a question-decomposition planner for a SQL agent over per-document tables.
Split the user's question into one subquestion per source document or per distinct \
filter target. If the question targets a single source, return a one-element list.

Output a JSON array of strings. No prose, no fences.

Examples:
Q: "For supplier Foo on 2025-04-01 in doc_006 what is NET Amount? And for transaction 123 in doc_007 what is Beneficiary?"
A: ["For supplier Foo on 2025-04-01 in doc_006 what is the NET Amount?", "For transaction 123 in doc_007 what is the Beneficiary?"]

Q: "What is the NET Amount for supplier Bar in the purchase card transactions?"
A: ["What is the NET Amount for supplier Bar in the purchase card transactions?"]
"""


def _decompose(question: str) -> list[str]:
    """Return [question] if single-target, else 2+ focused subquestions."""
    try:
        raw = _llm_chat(
            messages=[
                {"role": "system", "content": _DECOMPOSE_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=300,
        )
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
        parts = json.loads(raw)
        if isinstance(parts, list) and parts and all(isinstance(p, str) for p in parts):
            return [p.strip() for p in parts if p.strip()]
    except Exception:
        pass
    return [question]


# ---------------------------------------------------------------------------
# Inner SQL agent: select_table → inspect → write_sql → run_sql → evaluate
# ---------------------------------------------------------------------------

_SQL_PROMPT_HEADER = """\
You are a DuckDB SQL expert. Write ONE SQL SELECT query to answer the question.

## Selected table
{table_name}

## Columns
{schema}

## Sample rows
{samples}

## SQL rules
- Always double-quote column names: `"Column Name"`.
- Use ILIKE for ALL string filters — VARCHAR columns often have trailing whitespace.
- Special chars in ILIKE patterns (*, ?, &, /, +) are treated as literals — no escaping.
- Dates stored as `'YYYY-MM-DD'`. Rewrite any date in the question to that format.
- BIGINT/DOUBLE numeric columns: use `=`. VARCHAR amount columns: use ILIKE.
- Keep filters minimal — fewer filters = better recall when text is messy.
- Wrap your SQL in ```sql ... ``` fences. Output the SQL only, no commentary.
"""

_SQL_RETRY_HINT = """\

## Previous attempt(s)
{history}

The previous attempt did not return rows or hit a SQL error. Try ONE adjustment:
- SQL error → fix the column name (read "candidate bindings" if listed in the error).
- 0 rows → shorten ONE ILIKE pattern by trimming a few trailing characters
  (data values are sometimes truncated). Keep ALL other filters intact.
Do NOT drop the date/department/transaction-number filters — that risks matching the wrong row.
Wrap your new SQL in ```sql ... ``` fences.
"""

_FORMAT_PROMPT = """\
SQL result rows (table "{table_name}"):

{result}

Question: {question}

Read the rows above. Extract the answer.

RULES (follow EXACTLY):
1. The rows have at least one row of data → ALWAYS output a value, never abstain.
2. Multiple data rows → use the FIRST data row. Never refuse because of multiple rows.
3. Multiple fields asked → "Field1=value; Field2=value".
4. Single field asked → output the value alone (no field name, no units).
5. Copy values verbatim (e.g. 99.99 not "$99.99", text values exactly).
6. Output only: "Query returned 0 rows." → output exactly: Unsupported
7. No preamble, no markdown, no explanation. Just the value(s).
"""


class _SQLState(TypedDict):
    """Inner-loop state for one subquestion."""
    question: str
    candidate_tables: list[str]
    table_index: int
    selected_table: str
    schema: list[tuple[str, str]]
    samples: list[dict]
    sql_history: list[tuple[str, str]]   # (sql, result)
    last_single_col_value: str | None    # deterministic fallback when LLM punts
    attempts: int
    answer: str


def _extract_sql(text: str) -> str | None:
    """Pull the first ```sql ... ``` block, falling back to a SELECT-prefixed line."""
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    if text.strip().upper().startswith("SELECT"):
        return text.strip()
    return None


def _build_inner_graph(store: DuckDBStore) -> Any:
    """Compile the per-subquestion SQL ReAct graph."""

    def select_table(state: _SQLState) -> dict:
        idx = state.get("table_index", 0)
        candidates = state.get("candidate_tables") or []
        if idx >= len(candidates):
            return {"answer": "Unsupported"}
        return {"selected_table": candidates[idx]}

    def inspect(state: _SQLState) -> dict:
        tname = state["selected_table"]
        try:
            schema = store.describe(tname)
            samples = store.sample(tname, n=3)
        except Exception:
            return {"schema": [], "samples": []}
        return {"schema": schema, "samples": samples}

    def write_sql(state: _SQLState) -> dict:
        history_text = ""
        if state.get("sql_history"):
            blocks = []
            for prev_sql, prev_result in state["sql_history"][-3:]:
                snippet = prev_result[:600]
                blocks.append(f"SQL:\n```sql\n{prev_sql}\n```\nResult:\n{snippet}")
            history_text = "\n---\n".join(blocks)

        schema_text = "\n".join(f'  "{c}" ({t})' for c, t in state["schema"])
        samples_text = _format_samples(state["samples"]) or "(no rows in sample)"
        prompt = _SQL_PROMPT_HEADER.format(
            table_name=state["selected_table"],
            schema=schema_text,
            samples=samples_text,
        )
        if history_text:
            prompt += _SQL_RETRY_HINT.format(history=history_text)

        try:
            raw = _llm_chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": state["question"]},
                ],
                max_tokens=400,
            )
        except Exception as exc:
            sql = ""
            error = f"LLM error: {exc}"
            return {
                "sql_history": state.get("sql_history", []) + [(sql, error)],
                "attempts": state.get("attempts", 0) + 1,
            }

        sql = _extract_sql(raw) or ""
        return {
            "sql_history": state.get("sql_history", []) + [(sql, "")],
            "attempts": state.get("attempts", 0) + 1,
        }

    def run_sql(state: _SQLState) -> dict:
        history = state.get("sql_history") or []
        if not history:
            return {}
        sql, _ = history[-1]
        if not sql:
            return {
                "sql_history": history[:-1] + [(sql, "No SQL extracted from model output.")],
                "last_single_col_value": None,
            }
        ok, result, single_col = _execute_sql(store, sql)
        return {
            "sql_history": history[:-1] + [(sql, result)],
            "last_single_col_value": single_col,
        }

    def evaluate(state: _SQLState) -> dict:
        history = state.get("sql_history") or []
        last_result = history[-1][1] if history else ""
        last_sql = history[-1][0] if history else ""
        is_empty = last_result.strip() == "Query returned 0 rows."
        is_error = (
            last_result.lower().startswith("sql error")
            or last_result.startswith("LLM error")
            or not last_sql.strip()
            or last_result.startswith("No SQL extracted")
        )

        if not is_empty and not is_error and last_result.strip():
            try:
                answer = _llm_chat(
                    messages=[
                        {"role": "user", "content": _FORMAT_PROMPT.format(
                            table_name=state["selected_table"],
                            result=last_result[:2000],
                            question=state["question"],
                        )},
                    ],
                    max_tokens=200,
                ).strip()
            except Exception:
                answer = ""
            # When the SQL projects exactly one column and rows came back, the answer is
            # unambiguous — a punted "Unsupported" from the format LLM is a false negative.
            if (not answer or answer.lower() == "unsupported") and state.get("last_single_col_value"):
                return {"answer": state["last_single_col_value"]}
            return {"answer": answer or "Unsupported"}

        # Routing rules:
        #  - SQL/column error on this table: retry SAME table once (let the model fix the column).
        #  - 0 rows: this table doesn't hold the answer — move to the next candidate table.
        #  - Out of attempts and out of tables: Unsupported.
        if is_error and state.get("attempts", 0) < _MAX_SQL_ATTEMPTS:
            return {}

        next_idx = state.get("table_index", 0) + 1
        candidates = state.get("candidate_tables") or []
        if next_idx < min(len(candidates), 3):
            return {
                "table_index": next_idx,
                "attempts": 0,
                "sql_history": [],
                "last_single_col_value": None,
            }

        return {"answer": "Unsupported"}

    def route_after_evaluate(state: _SQLState) -> str:
        if state.get("answer"):
            return END
        # State after evaluate:
        #   - retry same table on column-error → sql_history kept, attempts < MAX
        #   - move to next table → sql_history cleared (length 0), attempts reset
        #   - give up → answer set to Unsupported (handled above)
        if state.get("sql_history") and state.get("attempts", 0) < _MAX_SQL_ATTEMPTS:
            return "write_sql"
        if state.get("candidate_tables") and state.get("table_index", 0) < len(state["candidate_tables"]):
            return "select_table"
        return END

    builder: StateGraph = StateGraph(_SQLState)
    builder.add_node("select_table", select_table)
    builder.add_node("inspect", inspect)
    builder.add_node("write_sql", write_sql)
    builder.add_node("run_sql", run_sql)
    builder.add_node("evaluate", evaluate)
    builder.set_entry_point("select_table")
    builder.add_conditional_edges(
        "select_table",
        lambda s: END if s.get("answer") else "inspect",
        {END: END, "inspect": "inspect"},
    )
    builder.add_edge("inspect", "write_sql")
    builder.add_edge("write_sql", "run_sql")
    builder.add_edge("run_sql", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {END: END, "write_sql": "write_sql", "select_table": "select_table"},
    )
    return builder.compile()


# ---------------------------------------------------------------------------
# Outer graph: decompose → fan-out → synthesize
# ---------------------------------------------------------------------------

class _OuterState(TypedDict):
    question: str
    subquestions: list[str]
    candidate_tables_per_sub: list[list[str]]
    answers: Annotated[list[str], operator.add]
    final_answer: str


def _candidate_tables_for(store: DuckDBStore, question: str) -> list[str]:
    """Return ranked candidate tables for a subquestion (doc-prefixed only)."""
    tables = _doc_tables(store)
    explicit = re.findall(r"\bdoc_\d+\b", question.lower())
    if explicit:
        scoped = {t: c for t, c in tables.items() if any(d in t.lower() for d in explicit)}
        if scoped:
            tables = scoped
    return _rank_tables(tables, question)


def _build_outer_graph(store: DuckDBStore) -> Any:
    """Compile the top-level decompose → per-subquestion-fanout → synthesize graph."""
    inner = _build_inner_graph(store)

    def decompose_node(state: _OuterState) -> dict:
        subqs = _decompose(state["question"])
        cands = [_candidate_tables_for(store, sq) for sq in subqs]
        return {"subquestions": subqs, "candidate_tables_per_sub": cands}

    def fan_out(state: _OuterState) -> Any:
        return [
            Send(
                "sql_agent",
                {
                    "question": sq,
                    "candidate_tables": cands,
                    "table_index": 0,
                    "selected_table": "",
                    "schema": [],
                    "samples": [],
                    "sql_history": [],
                    "last_single_col_value": None,
                    "attempts": 0,
                    "answer": "",
                },
            )
            for sq, cands in zip(state["subquestions"], state["candidate_tables_per_sub"])
        ]

    def sql_agent_node(state: _SQLState) -> dict:
        try:
            result = inner.invoke(state, config={"recursion_limit": 60})
        except Exception:
            return {"answers": ["Unsupported"]}
        return {"answers": [result.get("answer") or "Unsupported"]}

    def synthesize_node(state: _OuterState) -> dict:
        answers = state.get("answers") or []
        subqs = state.get("subquestions") or []
        if not answers:
            return {"final_answer": "Unsupported"}
        if len(answers) == 1:
            return {"final_answer": answers[0]}
        if all(a.strip().lower() == "unsupported" for a in answers):
            return {"final_answer": "Unsupported"}
        # Pair each subquestion with its answer for clarity in multi-part outputs.
        joined = "; ".join(
            f"{sq.split(' in ')[-1].rstrip('?').strip() or f'part {i+1}'}: {a}"
            if a.strip().lower() != "unsupported"
            else f"part {i+1}: Unsupported"
            for i, (sq, a) in enumerate(zip(subqs, answers))
        )
        return {"final_answer": joined}

    builder: StateGraph = StateGraph(_OuterState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("sql_agent", sql_agent_node)
    builder.add_node("synthesize", synthesize_node)
    builder.set_entry_point("decompose")
    builder.add_conditional_edges("decompose", fan_out, ["sql_agent"])
    builder.add_edge("sql_agent", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Public tool factory
# ---------------------------------------------------------------------------

def build_excel_agent_tools(store: DuckDBStore) -> list[StructuredTool]:
    """Return the LangChain StructuredTool for the agentic Excel SQL pipeline.

    Args:
        store: Connected DuckDBStore instance.
    """
    if not EXCEL_AGENT_API_KEY:
        def _disabled(question: str) -> str:
            """Excel agent is disabled because EXCEL_AGENT_API_KEY is not configured."""
            return "Excel agent not configured: EXCEL_AGENT_API_KEY is missing."

        class _Input(BaseModel):
            question: str

        return [StructuredTool.from_function(
            func=_disabled,
            name="query_excel",
            description="Disabled (missing API key).",
            args_schema=_Input,
        )]

    graph = _build_outer_graph(store)

    class _Input(BaseModel):
        question: str

    def query_excel(question: str) -> str:
        """Answer a structured-data question by decomposing per-source, running a SQL ReAct agent per part, and synthesising."""
        try:
            result = graph.invoke({
                "question": question,
                "subquestions": [],
                "candidate_tables_per_sub": [],
                "answers": [],
                "final_answer": "",
            })
        except Exception as exc:
            return f"Excel agent error: {exc}"
        return result.get("final_answer") or "Unsupported"

    return [
        StructuredTool.from_function(
            func=query_excel,
            name="query_excel",
            description=(
                "Answer any question about structured data (Excel/CSV) stored in DuckDB. "
                "Pass the full question as 'question' — include all filter details "
                "(dates, amounts, supplier names, transaction numbers, departments) "
                "exactly as stated. The agent decomposes cross-document questions, "
                "selects relevant tables, writes SQL with retries on errors and empty "
                "results, and returns the answer. Call once per logical question."
            ),
            args_schema=_Input,
        )
    ]
