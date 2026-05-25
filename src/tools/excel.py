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

Entry point: build_excel_agent_tools(store) returns the `query_excel`
StructuredTool.
Called by: the main RAG agent tool registry (src/rag_agent.py / src/tools), which
exposes `query_excel` to the orchestrating LLM.
Calls: src.duckdb_store (DuckDBStore for data, _normalize_sql/_truncate_ilike for
SQL rewriting), src.config (Excel-agent LLM endpoint), src.prompts (DECOMPOSE,
SQL, FORMAT, RETRY prompts), the OpenAI-compatible chat API, and LangGraph.
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
from src.duckdb_store import DuckDBStore, _normalize_sql, _truncate_ilike
from src.prompts import (
    DECOMPOSE_PROMPT,
    FORMAT_PROMPT,
    SQL_PROMPT_HEADER,
    SQL_RETRY_HINT,
)

_MAX_SQL_ATTEMPTS = 3
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
# Helpers: LLM calls, table ranking, and SQL execution
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Drop <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _llm_chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 700) -> str:
    """One-shot chat completion against the configured Excel-agent endpoint."""
    if not EXCEL_AGENT_API_KEY:
        raise RuntimeError("EXCEL_AGENT_API_KEY is not configured")
    # Build the client and run a single completion, stripping reasoning tags.
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
        """Score a table by how many query tokens overlap its column names."""
        # Count question-token overlap with the table's column names.
        col_tokens: set[str] = set()
        for col in tables.get(tname, []):
            col_tokens.update(re.findall(r"[a-z][a-z0-9]{2,}", col.lower()))
        col_score = len(q_tokens & col_tokens)
        # Tie-break on overlap with the table name itself.
        name_tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", tname.lower()))
        name_score = len(q_tokens & name_tokens)
        return (-col_score, -name_score, tname)

    return sorted(tables.keys(), key=score)


def _format_samples(samples: list[dict]) -> str:
    """Render sample rows as one JSON object per line."""
    return "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in samples)


def _split_where_predicates(sql: str) -> tuple[str, list[str], str] | None:
    """Split a flat WHERE clause into its top-level AND predicates (quote-aware).

    Returns (text_up_to_and_including_WHERE, [predicate, ...], trailing_clause) or
    None when there is no WHERE clause or fewer than two predicates. The ` AND `
    split respects single-quoted literals, so a value containing " AND " is intact.
    """
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if not m:
        return None
    head, rest = sql[: m.end()], sql[m.end():]
    # The WHERE body ends at the first GROUP BY / ORDER BY / LIMIT / HAVING, if any.
    tail_m = re.search(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b", rest, re.IGNORECASE)
    where_body, tail = (rest[: tail_m.start()], rest[tail_m.start():]) if tail_m else (rest, "")

    # Walk the body splitting on top-level ` AND `, ignoring quoted string contents.
    preds, buf, i, in_str = [], "", 0, False
    while i < len(where_body):
        ch = where_body[i]
        if ch == "'":
            in_str = not in_str
        if not in_str and where_body[i : i + 5].upper() == " AND ":
            preds.append(buf.strip())
            buf, i = "", i + 5
            continue
        buf += ch
        i += 1
    if buf.strip():
        preds.append(buf.strip())
    return (head + " ", preds, tail) if len(preds) >= 2 else None


def _relax_by_dropping_predicate(store: DuckDBStore, sql: str) -> Any:
    """Retry an over-constrained 0-row query by dropping one AND-predicate at a time.

    A query returns nothing when one filter value is assigned to the wrong column
    or is otherwise too strict. Dropping each predicate in turn and keeping the
    most-specific variant that still returns rows recovers the row without knowing
    anything about the schema. Returns the chosen DataFrame, or None if none helps.
    """
    parts = _split_where_predicates(sql)
    if not parts:
        return None
    head, preds, tail = parts
    best = None
    for drop in range(len(preds)):
        variant = head + " AND ".join(p for j, p in enumerate(preds) if j != drop) + tail
        try:
            vdf = store.execute(variant)
        except Exception:
            continue
        # Prefer the variant returning the fewest (non-zero) rows — the most specific.
        if not vdf.empty and (best is None or len(vdf) < len(best)):
            best = vdf
    return best


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
    # Rewrite date literals, then run the SQL.
    sql = _normalize_sql(sql)
    try:
        df = store.execute(sql)
    except Exception as exc:
        return False, f"SQL error: {exc}", None

    # On 0 rows, retry with progressively-truncated ILIKE values.
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

    # Still empty: the query is likely over-constrained (a value filtered on the
    # wrong column). Drop one predicate at a time and keep the most-specific match.
    if df.empty:
        relaxed_df = _relax_by_dropping_predicate(store, sql)
        if relaxed_df is not None:
            df = relaxed_df

    if df.empty:
        return True, "Query returned 0 rows.", None
    # Aggregate queries (SUM/COUNT/etc.) on a missing key collapse to all-NaN —
    # treat that as an empty match so the agent retries on a different table/filter
    # rather than reporting "NaN" to the user.
    if df.shape == (1, 1) and df.iloc[0, 0] != df.iloc[0, 0]:  # NaN check
        return True, "Query returned 0 rows.", None

    # Capture the lone cell value for single-column results (deterministic fallback).
    single_col_first = None
    if df.shape[1] == 1:
        first = df.iloc[0, 0]
        if first is not None and first == first:  # not NaN
            single_col_first = str(first)

    # Render the result table, capping the number of rows shown.
    truncated = len(df) > _ROW_LIMIT
    out = df.head(_ROW_LIMIT).to_string(index=False, max_colwidth=80)
    if truncated:
        out += f"\n… (truncated at {_ROW_LIMIT} rows)"
    return True, out, single_col_first


# ---------------------------------------------------------------------------
# Decomposition: split cross-document questions into per-doc subqueries
# ---------------------------------------------------------------------------

def _decompose(question: str) -> list[str]:
    """Return [question] if single-target, else 2+ focused subquestions."""
    try:
        # Ask the LLM for a JSON list of subquestions, stripping any code fences.
        raw = _llm_chat(
            messages=[
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=300,
        )
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
        # Parse and accept only a non-empty list of strings.
        parts = json.loads(raw)
        if isinstance(parts, list) and parts and all(isinstance(p, str) for p in parts):
            return [p.strip() for p in parts if p.strip()]
    except Exception:
        pass
    # Any failure falls back to treating the question as a single target.
    return [question]


# ---------------------------------------------------------------------------
# Inner SQL agent: select_table → inspect → write_sql → run_sql → evaluate
# ---------------------------------------------------------------------------

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
    final_sql: str


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
        """Pick the current candidate table for the SQL attempt, or abstain."""
        idx = state.get("table_index", 0)
        candidates = state.get("candidate_tables") or []
        if idx >= len(candidates):
            return {"answer": "Unsupported"}
        return {"selected_table": candidates[idx]}

    def inspect(state: _SQLState) -> dict:
        """Load the selected table's schema and sample rows into state."""
        tname = state["selected_table"]
        try:
            schema = store.describe(tname)
            samples = store.sample(tname, n=8)
        except Exception:
            return {"schema": [], "samples": []}
        return {"schema": schema, "samples": samples}

    def write_sql(state: _SQLState) -> dict:
        """Prompt the LLM to write the next SQL query from schema, samples and history."""
        # Summarise the last few SQL attempts and their results for the retry hint.
        history_text = ""
        if state.get("sql_history"):
            blocks = []
            for prev_sql, prev_result in state["sql_history"][-3:]:
                snippet = prev_result[:600]
                blocks.append(f"SQL:\n```sql\n{prev_sql}\n```\nResult:\n{snippet}")
            history_text = "\n---\n".join(blocks)

        # Build the SQL-writing prompt from the table schema and sample rows.
        schema_text = "\n".join(f'  "{c}" ({t})' for c, t in state["schema"])
        samples_text = _format_samples(state["samples"]) or "(no rows in sample)"
        prompt = SQL_PROMPT_HEADER.format(
            table_name=state["selected_table"],
            schema=schema_text,
            samples=samples_text,
        )
        if history_text:
            prompt += SQL_RETRY_HINT.format(history=history_text)

        # Call the LLM; on failure record the error as a history entry.
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

        # Extract the SQL from the model output and append it pending execution.
        sql = _extract_sql(raw) or ""
        return {
            "sql_history": state.get("sql_history", []) + [(sql, "")],
            "attempts": state.get("attempts", 0) + 1,
        }

    def run_sql(state: _SQLState) -> dict:
        """Execute the most recent SQL against DuckDB and record its result."""
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
        """Judge the last SQL result and decide retry, next table, or final answer."""
        # Classify the most recent result as empty, errored, or usable.
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

        # Usable result: ask the LLM to format it into a natural-language answer.
        if not is_empty and not is_error and last_result.strip():
            try:
                answer = _llm_chat(
                    messages=[
                        {"role": "user", "content": FORMAT_PROMPT.format(
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
                return {"answer": state["last_single_col_value"], "final_sql": last_sql}
            return {"answer": answer or "Unsupported", "final_sql": last_sql}

        # Routing rules:
        #  - SQL/column error OR 0 rows on this table: retry the SAME table (let the
        #    model fix the column or relax a too-strict filter) until attempts run out.
        #  - Out of attempts: move to the next candidate table.
        #  - Out of attempts and out of tables: Unsupported.
        if (is_error or is_empty) and state.get("attempts", 0) < _MAX_SQL_ATTEMPTS:
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
        """Route the SQL graph after evaluation — retry, next table, or END."""
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

    # Wire the inner graph: select_table → inspect → write_sql → run_sql → evaluate.
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
    sql_trace: Annotated[list[str], operator.add]
    final_answer: str


def _candidate_tables_for(store: DuckDBStore, question: str) -> list[str]:
    """Return ranked candidate tables for a subquestion (doc-prefixed only)."""
    tables = _doc_tables(store)
    # If the question names explicit doc ids, restrict candidates to those docs.
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
        """Split the question into sub-questions and find candidate tables for each."""
        subqs = _decompose(state["question"])
        cands = [_candidate_tables_for(store, sq) for sq in subqs]
        return {"subquestions": subqs, "candidate_tables_per_sub": cands}

    def fan_out(state: _OuterState) -> Any:
        """Fan out one parallel sql_agent branch per sub-question (LangGraph Send)."""
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
                    "final_sql": "",
                },
            )
            for sq, cands in zip(state["subquestions"], state["candidate_tables_per_sub"])
        ]

    def sql_agent_node(state: _SQLState) -> dict:
        """Run the inner SQL graph for one sub-question; abstain on error."""
        # Invoke the compiled inner graph; any failure abstains with "Unsupported".
        try:
            result = inner.invoke(state, config={"recursion_limit": 60})
        except Exception:
            return {"answers": ["Unsupported"], "sql_trace": []}
        # Collect the answer and the generated SQL for the UI trace.
        sql = (result.get("final_sql") or "").strip()
        return {
            "answers": [result.get("answer") or "Unsupported"],
            "sql_trace": [sql] if sql else [],
        }

    def synthesize_node(state: _OuterState) -> dict:
        """Merge the per-sub-question answers into one final answer."""
        # Trivial cases: no answers, a single answer, or all-unsupported.
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

    # Wire the outer graph: decompose → fan-out to sql_agent branches → synthesize.
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
# Public tool factory: wrap the graph as a LangChain query_excel tool
# ---------------------------------------------------------------------------

def build_excel_agent_tools(store: DuckDBStore) -> list[StructuredTool]:
    """Return the LangChain StructuredTool for the agentic Excel SQL pipeline.

    Args:
        store: Connected DuckDBStore instance.
    """
    # No API key: return a stub tool that always reports the agent is disabled.
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

    # Compile the decompose → fan-out → synthesize graph once at tool-build time.
    graph = _build_outer_graph(store)

    class _Input(BaseModel):
        question: str

    def query_excel(question: str) -> tuple[str, dict[str, Any]]:
        """Answer a structured-data question by decomposing per-source, running a SQL ReAct agent per part, and synthesising.

        Returns (answer, artifact). The artifact carries the generated SQL and
        subquestions for the UI trace panel; it rides on the ToolMessage and is
        not shown to the LLM, so the tool's content stays clean.
        """
        # Run the full graph from a fresh empty state.
        try:
            result = graph.invoke({
                "question": question,
                "subquestions": [],
                "candidate_tables_per_sub": [],
                "answers": [],
                "sql_trace": [],
                "final_answer": "",
            })
        except Exception as exc:
            return f"Excel agent error: {exc}", {}
        # Package the generated SQL and subquestions as the UI-only artifact.
        artifact = {
            "sql": list(result.get("sql_trace") or []),
            "subquestions": list(result.get("subquestions") or []),
        }
        return result.get("final_answer") or "Unsupported", artifact

    return [
        StructuredTool.from_function(
            func=query_excel,
            name="query_excel",
            response_format="content_and_artifact",
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
