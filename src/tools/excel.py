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
from contextvars import ContextVar
from typing import Annotated, Any, TypedDict

from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from src.config import (
    EXCEL_AGENT_API_BASE,
    EXCEL_AGENT_API_KEY,
    EXCEL_AGENT_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from src.duckdb_store import (
    DuckDBStore,
    _normalize_special_chars_ilike,
    _normalize_sql,
    _truncate_ilike,
)
from src.llm_utils import (
    _get_openai_client,
    _is_thinking_model,
    _openrouter_provider_extra_body,
)
from src.prompts import (
    DECOMPOSE_PROMPT,
    FORMAT_PROMPT,
    SQL_PROMPT_HEADER,
    SQL_RETRY_HINT,
)
from src.vector_store import get_source_by_duckdb_table

_MAX_SQL_ATTEMPTS = 3
_ROW_LIMIT = 50

# Set once per top-level agent turn (src/answer_pipeline.py's run_once, around the
# stream_agent/agent.invoke() call) to the question the agent was actually asked --
# read by _column_matches_question's gate so a column-match check can't be fooled
# by a self-rewritten retry. Reproduced live: on an internal SQL retry, the ReAct
# agent rewrote its own query_excel(question=...) argument to directly ask for the
# wrong column ("...what is the Merchant Category for the transaction...") instead
# of the user's real question ("what payment method...") -- no text pattern on that
# rewritten string can distinguish it from a genuine user question, since by that
# point it genuinely does name the wrong field. The fix is not smarter text
# matching, it's not trusting agent-mutable text for this check at all: this
# ContextVar mirrors FORCED_DOC_ID's pattern (src/tools/retrieval_tool.py) to give
# the gate access to the one piece of text the agent's own tool calls can't rewrite.
ORIGINAL_QUESTION: ContextVar[str | None] = ContextVar(
    "ORIGINAL_QUESTION", default=None
)

_DOC_TABLE_PREFIX_RE = re.compile(r"^doc_\d+_", re.IGNORECASE)

# Detects an aggregate SELECT (SUM/COUNT/AVG/MIN/MAX) -- see query_excel's
# citation-building for why this changes what gets used as the row-match quote.
_AGGREGATE_SQL_RE = re.compile(r"\b(?:SUM|COUNT|AVG|MIN|MAX)\s*\(", re.IGNORECASE)
# First quoted string literal after an ILIKE/= comparison in a WHERE clause
# (e.g. WHERE "Purchase of Expenditure" ILIKE 'MATERIALS' -> "MATERIALS").
_WHERE_LITERAL_RE = re.compile(r"(?:ILIKE|=)\s*'([^']+)'", re.IGNORECASE)

_TABLE_STOPWORDS = {
    # Generic English function words only — never strip column-name candidates
    # (amount, total, name, number, date, etc. are real columns in our schemas).
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "by",
    "at",
    "from",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "row",
    "rows",
    "table",
    "tables",
    "doc",
    "document",
    "file",
    "sheet",
}


# ---------------------------------------------------------------------------
# Helpers: LLM calls, table ranking, and SQL execution
# ---------------------------------------------------------------------------


def _strip_think(text: str) -> str:
    """Drop <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _llm_chat(
    messages: list[dict], temperature: float = 0.0, max_tokens: int = 700
) -> str:
    """One-shot chat completion against the configured Excel-agent endpoint."""
    if not EXCEL_AGENT_API_KEY:
        raise RuntimeError("EXCEL_AGENT_API_KEY is not configured")
    # Thinking models (Qwen3, QwQ, DeepSeek-R1, ...) emit chain-of-thought prose
    # ahead of the answer unless told not to — suppress it on the first message,
    # same convention as src/rag_agent.py and src/tools/retrieval_tool.py.
    if _is_thinking_model(EXCEL_AGENT_MODEL) and messages:
        first = messages[0]
        if not first["content"].startswith("/no_think"):
            messages = [
                {**first, "content": "/no_think " + first["content"]}
            ] + messages[1:]
    # Reuse a cached client and run a single completion, stripping reasoning tags.
    client = _get_openai_client(EXCEL_AGENT_API_BASE, EXCEL_AGENT_API_KEY)
    resp = client.chat.completions.create(
        model=EXCEL_AGENT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_openrouter_provider_extra_body(EXCEL_AGENT_API_BASE),
    )
    return _strip_think(resp.choices[0].message.content or "")


def _doc_tables(store: DuckDBStore) -> dict[str, list[str]]:
    """Return only the doc_NNN_* tables, dropping legacy / test artefacts."""
    return {t: c for t, c in store.tables().items() if _DOC_TABLE_PREFIX_RE.match(t)}


def _question_keywords(question: str) -> set[str]:
    """Keyword set used for ranking tables and detecting doc-name mentions."""
    return {
        t
        for t in re.findall(r"[a-z][a-z0-9]{2,}", question.lower())
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
    return "\n".join(
        json.dumps(row, ensure_ascii=False, default=str) for row in samples
    )


def _split_where_predicates(sql: str) -> tuple[str, list[str], str] | None:
    """Split a flat WHERE clause into its top-level AND predicates (quote-aware).

    Returns (text_up_to_and_including_WHERE, [predicate, ...], trailing_clause) or
    None when there is no WHERE clause or fewer than two predicates. The ` AND `
    split respects single-quoted literals, so a value containing " AND " is intact.
    """
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if not m:
        return None
    head, rest = sql[: m.end()], sql[m.end() :]
    # The WHERE body ends at the first GROUP BY / ORDER BY / LIMIT / HAVING, if any.
    tail_m = re.search(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b", rest, re.IGNORECASE)
    where_body, tail = (
        (rest[: tail_m.start()], rest[tail_m.start() :]) if tail_m else (rest, "")
    )

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
        variant = (
            head + " AND ".join(p for j, p in enumerate(preds) if j != drop) + tail
        )
        try:
            vdf = store.execute(variant)
        except Exception:
            continue
        # Prefer the variant returning the fewest (non-zero) rows — the most specific.
        if not vdf.empty and (best is None or len(vdf) < len(best)):
            best = vdf
    return best


def _execute_sql(
    store: DuckDBStore, sql: str
) -> tuple[bool, str, str | None, list[str] | None]:
    """Run SQL and return (ok, formatted_text, single_col_first_value, ambiguous_values).

    - ok=True with rendered table on success (including 0-row results)
    - ok=False with `SQL error: ...` on failure
    - single_col_first_value: the first row's only-cell value when the SQL
      projects exactly one column; None otherwise. Used as a deterministic
      fallback when the format LLM punts on multi-row single-column results.
    - ambiguous_values: when the SQL projects one column and matched rows carry
      more than one distinct value for it, the question under-constrains which
      row is meant (e.g. "the Trainline transaction" matching 44 rows across 5
      different expense categories) — this lists the distinct values found (up
      to 8) so the caller can ask for a qualifier instead of picking one at
      random. None when there's no such ambiguity.

    Auto-retries 0-row ILIKE queries with progressively-truncated string filters
    to handle truncated supplier/beneficiary names common in the source data.
    """
    # Rewrite date literals, then run the SQL.
    sql = _normalize_sql(sql)
    try:
        df = store.execute(sql)
    except Exception as exc:
        return False, f"SQL error: {exc}", None, None

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

    # Still empty: the filter value's punctuation/conjunction style may not match
    # the stored data (e.g. "Smith & Jones" vs "Smith and Jones"). Retry with
    # both sides normalized to ignore '&' vs "and" and other punctuation.
    if df.empty:
        normalized = _normalize_special_chars_ilike(sql)
        if normalized != sql:
            try:
                relaxed = store.execute(normalized)
            except Exception:
                relaxed = None
            if relaxed is not None and not relaxed.empty:
                df = relaxed
                sql = normalized

    # Still empty: the query is likely over-constrained (a value filtered on the
    # wrong column). Drop one predicate at a time and keep the most-specific match.
    if df.empty:
        relaxed_df = _relax_by_dropping_predicate(store, sql)
        if relaxed_df is not None:
            df = relaxed_df

    if df.empty:
        return True, "Query returned 0 rows.", None, None
    # Aggregate queries (SUM/COUNT/etc.) on a missing key collapse to all-NaN —
    # treat that as an empty match so the agent retries on a different table/filter
    # rather than reporting "NaN" to the user.
    if df.shape == (1, 1) and df.iloc[0, 0] != df.iloc[0, 0]:  # NaN check
        return True, "Query returned 0 rows.", None, None

    # Capture the lone cell value for single-column results (deterministic fallback),
    # and detect genuine ambiguity: multiple rows matched but they disagree on the
    # one projected value, meaning the question under-constrains which row is meant.
    single_col_first = None
    ambiguous_values: list[str] | None = None
    if df.shape[1] == 1:
        col = df.iloc[:, 0]
        first = col.iloc[0]
        if first is not None and first == first:  # not NaN
            single_col_first = str(first)
        distinct = [str(v) for v in col.dropna().unique()]
        if len(distinct) > 1:
            ambiguous_values = distinct[:8]

    # Render the result table, capping the number of rows shown.
    truncated = len(df) > _ROW_LIMIT
    out = df.head(_ROW_LIMIT).to_string(index=False, max_colwidth=80)
    if truncated:
        out += f"\n… (truncated at {_ROW_LIMIT} rows)"
    return True, out, single_col_first, ambiguous_values


def _answer_in_result(answer: str, result_text: str) -> bool:
    """True when every value in a formatted answer appears in the SQL result text.

    The FORMAT step is copy-only by contract (FORMAT_PROMPT rule 5), so any
    output value absent from the result rows is a fabrication — reproduced
    live: a single-row result containing only "Purchase of Expenditure:
    MATERIALS" was formatted as "150.00", a number that exists elsewhere in
    the table but nowhere in this call's own input. Numeric values compare
    float-equal so "150.0" in the rows matches "150.00" in the answer; text
    values compare case- and whitespace-insensitively.
    """
    norm_result = re.sub(r"\s+", " ", result_text).lower()
    result_nums: set[float] = set()
    for tok in re.findall(r"-?\d[\d,]*\.?\d*", result_text):
        try:
            result_nums.add(float(tok.replace(",", "")))
        except ValueError:
            pass
    for part in answer.split(";"):
        value = part.split("=", 1)[-1].strip()
        if not value:
            continue
        try:
            if float(value.replace(",", "")) in result_nums:
                continue
        except ValueError:
            pass
        if re.sub(r"\s+", " ", value).lower() in norm_result:
            continue
        return False
    return True


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
    sql_history: list[tuple[str, str]]  # (sql, result)
    last_single_col_value: str | None  # deterministic fallback when LLM punts
    last_ambiguous_values: list[str] | None  # distinct values when rows disagree
    attempts: int
    answer: str
    final_sql: str
    final_result: str


def _is_readonly_select(sql: str) -> bool:
    """True if sql is exactly one SELECT/WITH statement, no statement stacking.

    query_excel is a lookup tool over user-uploaded data; nothing should ever
    write to or drop from the shared DuckDB file through it. Confirmed live
    that DuckDB's execute() runs semicolon-stacked statements by default
    ("SELECT 1; DROP TABLE t;" actually drops t) -- a SQL-generation call
    that's been prompt-injected (via a malicious question, or adversarial
    retrieved content in its history/samples) could otherwise destroy data
    for the whole app, not just answer wrong. Reject anything but a single
    read-only statement rather than trying to sanitize it.
    """
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) != 1:
        return False
    return bool(re.match(r"^(SELECT|WITH)\b", statements[0], re.IGNORECASE))


def _extract_sql(text: str) -> str | None:
    """Pull the first ```sql ... ``` block, falling back to a SELECT-prefixed line.

    Returns None (treated the same as "no SQL extracted") for anything that
    isn't a single read-only SELECT/WITH statement -- see _is_readonly_select.
    """
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = m.group(1).strip() if m and m.group(1).strip() else None
    if candidate is None and text.strip().upper().startswith("SELECT"):
        candidate = text.strip()
    if candidate is None or not _is_readonly_select(candidate):
        return None
    return candidate


# Column-name words too common to signal a real match on their own — two genuinely
# different real-world fields ("invoice number" vs "transaction number") both contain
# "number" without meaning the same thing, so it's excluded before checking overlap.
_GENERIC_COLUMN_WORDS = {
    "number",
    "num",
    "no",
    "id",
    "code",
    "amount",
    "value",
    "date",
    "name",
    "total",
    "type",
    "category",
    "description",
    "ref",
    "reference",
}


def _extract_selected_columns(sql: str) -> list[str]:
    """Return the column names projected by a SELECT, ignoring AS-aliases and quoting."""
    m = re.search(r"select\s+(.*?)\s+from\s", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    columns = []
    for part in m.group(1).split(","):
        part = re.split(r"\bAS\b", part.strip(), maxsplit=1, flags=re.IGNORECASE)[
            0
        ].strip()
        part = part.strip('"').strip("'")
        if (
            part
            and part != "*"
            and not part.upper().startswith(("SUM(", "COUNT(", "AVG(", "MIN(", "MAX("))
        ):
            columns.append(part)
    return columns


# Row-qualifier clauses ("for the transaction with the highest...", "on 2025-04-03",
# "in PLACE / STREET SCENE") describe which ROW is meant, not which FIELD is being
# asked for -- but their words (e.g. "transaction") can accidentally overlap a wrong
# column's name (e.g. "Transaction Number" answering an "invoice number" question),
# passing the vocabulary gate on a coincidence instead of a real match. Cut the
# question off at the first such clause before comparing.
_ROW_QUALIFIER_RE = re.compile(r"\b(?:for|with|on|in)\s+the\b", re.IGNORECASE)
_TARGET_FIELD_PREFIX_RE = re.compile(
    r"^\s*what\s+(?:is|was|are|were)\s+(?:the\s+)?", re.IGNORECASE
)


def _target_field_phrase(question: str) -> str:
    """Return the leading phrase naming the field asked for, dropping any
    trailing row-qualifier clause. Falls back to the full question if no
    clause boundary is found.

    Parenthetical asides are stripped first -- reproduced live: on a retry,
    the outer ReAct agent rewrote its own query_excel call to "...what
    payment method (Merchant Category) was used..." -- self-injecting its
    wrong-column guess as parenthetical text in the SAME question string this
    gate compares against, so "Merchant Category" trivially vocabulary-matched
    a question that now, by the agent's own doing, contained the words
    "Merchant Category". The gate exists specifically to not trust that guess
    (see _column_matches_question's docstring); a caller-injected aside must
    not be able to satisfy the check it's supposed to fail.
    """
    q = re.sub(r"\([^)]*\)", " ", question).strip().rstrip("?")
    m = _ROW_QUALIFIER_RE.search(q)
    head = q[: m.start()] if m else q
    head = _TARGET_FIELD_PREFIX_RE.sub("", head).strip()
    return head or q


def _column_matches_question(column_name: str, question: str) -> bool:
    """Hard gate backing the SQL-writing prompt's column-matching rule: does this
    column's own wording appear anywhere in the field the question actually asks
    for, ignoring generic words and any trailing row-qualifier clause?

    Relying on the LLM to self-police "is this really the same concept" is unreliable
    (verified: it still substitutes "Merchant Category" for "payment method" about half
    the time even when explicitly told not to) — this catches the cases where the
    selected column shares no real vocabulary with what was asked at all. Matching
    against the whole question let a row-qualifier's incidental words (e.g.
    "transaction" in "...for the transaction with the highest NET Amount") pass a
    wrong column ("Transaction Number") that has nothing to do with the field asked
    for ("invoice number") -- so only the target-field phrase is compared now.
    """
    q_words = set(re.findall(r"[a-z]+", _target_field_phrase(question).lower()))
    raw_col_words = set(re.findall(r"[a-z]+", column_name.lower()))
    col_words = raw_col_words - _GENERIC_COLUMN_WORDS
    if not col_words:
        # Column name is entirely generic words (e.g. bare "Amount", "Date") --
        # too little signal to compare on stripped vocabulary, but an unconditional
        # pass here let a column like "Total" answer a "payment method" question
        # (reproduced: doc_007 qa_9). Fall back to raw (unstripped) overlap instead:
        # "Amount" still matches "what is the amount", "Total" no longer matches
        # "payment method".
        return bool(raw_col_words & q_words)
    return bool(col_words & q_words)


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
        # Hard gate: don't trust the model's own judgment that a column matches the
        # question — verified unreliable (src/prompts.py comment). If none of the
        # selected columns share real vocabulary with the question, discard the SQL
        # exactly as if the model had refused, so it flows through the same
        # retry/next-table/Unsupported path as a genuine no-match.
        #
        # Gate against ORIGINAL_QUESTION (the real top-level ask), not
        # state["question"] -- state["question"] is whatever text the outer
        # ReAct agent's own retry loop chose to pass to this tool call, and
        # it's allowed to rewrite that freely between calls. Reproduced live:
        # after a rejected "Merchant Category" attempt, the agent's own next
        # retry rewrote its query_excel call to ask "...what is the Merchant
        # Category for the transaction..." outright -- at that point the text
        # genuinely does name the wrong field, so no pattern match on it can
        # catch the substitution; only a source of truth the agent can't
        # rewrite can. Falls back to state["question"] when unset (e.g. tests
        # that build the graph without going through run_once).
        gate_question = ORIGINAL_QUESTION.get() or state["question"]
        if sql:
            selected_columns = _extract_selected_columns(sql)
            if selected_columns and not any(
                _column_matches_question(col, gate_question)
                for col in selected_columns
            ):
                sql = ""
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
                "sql_history": history[:-1]
                + [(sql, "No SQL extracted from model output.")],
                "last_single_col_value": None,
                "last_ambiguous_values": None,
            }
        ok, result, single_col, ambiguous_values = _execute_sql(store, sql)
        return {
            "sql_history": history[:-1] + [(sql, result)],
            "last_single_col_value": single_col,
            "last_ambiguous_values": ambiguous_values,
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

        # Usable result, but rows disagree on the one projected value: the question
        # under-constrains which row is meant (e.g. "the Trainline transaction"
        # matches 44 rows across 5 different expense categories). Asking the format
        # LLM to pick one from a truncated dump of disagreeing rows is exactly how a
        # confident wrong answer gets produced — short-circuit to a clarification
        # instead, same "Clarify:" convention the main agent uses for broad questions.
        ambiguous_values = state.get("last_ambiguous_values")
        if not is_empty and not is_error and ambiguous_values:
            values_preview = ", ".join(f'"{v}"' for v in ambiguous_values[:5])
            return {
                "answer": (
                    f"Clarify: multiple different values match this query "
                    f"({values_preview}"
                    f"{', …' if len(ambiguous_values) > 5 else ''}) — please add a "
                    "date, transaction number, or other identifying detail to narrow "
                    "it to one row."
                ),
                "final_sql": last_sql,
            }

        # Usable result: ask the LLM to format it into a natural-language answer.
        if not is_empty and not is_error and last_result.strip():
            try:
                answer = _llm_chat(
                    messages=[
                        {
                            "role": "user",
                            "content": FORMAT_PROMPT.format(
                                table_name=state["selected_table"],
                                result=last_result[:2000],
                                question=state["question"],
                            ),
                        },
                    ],
                    max_tokens=200,
                ).strip()
            except Exception:
                answer = ""
            # The format LLM occasionally echoes back the internal "no rows" sentinel
            # phrase as if it were a real answer, even when real rows were passed in —
            # it's never a legitimate output value, only ever a literal-input pattern
            # this same prompt asks it to recognize. Bypass the single-column fallback
            # below: that fallback exists for a genuine false-negative "Unsupported" on
            # a good column, not for this confusion, and reusing it here would just
            # resurface the same (likely wrong) column value.
            if answer.strip().lower() == "query returned 0 rows.":
                return {"answer": "Unsupported", "final_sql": last_sql}
            # Fabrication guard: the format step may only copy values that exist in
            # the result rows (reproduced live: correct single-row "MATERIALS" input
            # formatted as a fabricated "150.00"). A non-copied answer is discarded
            # so the deterministic single-column fallback / Unsupported path below
            # applies instead of trusting it.
            if (
                answer
                and answer.lower() != "unsupported"
                and not _answer_in_result(answer, last_result)
            ):
                answer = ""
            # When the SQL projects exactly one column and rows came back, the answer is
            # unambiguous — a punted "Unsupported" from the format LLM is a false negative.
            if (not answer or answer.lower() == "unsupported") and state.get(
                "last_single_col_value"
            ):
                return {
                    "answer": state["last_single_col_value"],
                    "final_sql": last_sql,
                    "final_result": last_result,
                }
            return {
                "answer": answer or "Unsupported",
                "final_sql": last_sql,
                "final_result": last_result,
            }

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
                "last_ambiguous_values": None,
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
        if state.get("candidate_tables") and state.get("table_index", 0) < len(
            state["candidate_tables"]
        ):
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
    table_trace: Annotated[list[str], operator.add]
    result_trace: Annotated[list[str], operator.add]
    final_answer: str


def _candidate_tables_for(store: DuckDBStore, question: str) -> list[str]:
    """Return ranked candidate tables for a subquestion (doc-prefixed only)."""
    tables = _doc_tables(store)
    # If the question names explicit doc ids, restrict candidates to those docs.
    explicit = re.findall(r"\bdoc_\d+\b", question.lower())
    if explicit:
        scoped = {
            t: c for t, c in tables.items() if any(d in t.lower() for d in explicit)
        }
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
                    "last_ambiguous_values": None,
                    "attempts": 0,
                    "answer": "",
                    "final_sql": "",
                    "final_result": "",
                },
            )
            for sq, cands in zip(
                state["subquestions"], state["candidate_tables_per_sub"]
            )
        ]

    def sql_agent_node(state: _SQLState) -> dict:
        """Run the inner SQL graph for one sub-question; abstain on error."""
        # Invoke the compiled inner graph; any failure abstains with "Unsupported".
        try:
            result = inner.invoke(state, config={"recursion_limit": 60})
        except Exception:
            return {"answers": ["Unsupported"], "sql_trace": []}
        # Collect the answer, the generated SQL, and the table it ran against
        # for the UI trace / citation lookup.
        sql = (result.get("final_sql") or "").strip()
        table = (result.get("selected_table") or "").strip()
        result_text = (result.get("final_result") or "").strip()
        return {
            "answers": [result.get("answer") or "Unsupported"],
            "sql_trace": [sql] if sql else [],
            "table_trace": [table] if sql and table else [],
            "result_trace": [result_text] if sql and table else [],
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
        # _target_field_phrase strips row-qualifier clauses (dates, department
        # names) so the label reads as the field asked for, not a garbled
        # fragment of the row filters (e.g. "PLACE / STREET SCENE what is...").
        joined = "; ".join(
            f"{_target_field_phrase(sq) or f'part {i + 1}'}: {a}"
            if a.strip().lower() != "unsupported"
            else f"part {i + 1}: Unsupported"
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

        return [
            StructuredTool.from_function(
                func=_disabled,
                name="query_excel",
                description="Disabled (missing API key).",
                args_schema=_Input,
            )
        ]

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
            result = graph.invoke(
                {
                    "question": question,
                    "subquestions": [],
                    "candidate_tables_per_sub": [],
                    "answers": [],
                    "sql_trace": [],
                    "table_trace": [],
                    "result_trace": [],
                    "final_answer": "",
                }
            )
        except Exception as exc:
            return f"Excel agent error: {exc}", {}
        # Package the generated SQL and subquestions as the UI-only artifact.
        artifact: dict[str, Any] = {
            "sql": list(result.get("sql_trace") or []),
            "subquestions": list(result.get("subquestions") or []),
        }
        # Attribute the answer to its source spreadsheet(s) -- one citation per
        # DISTINCT table actually queried, not gated on a single subquestion.
        # table_trace/result_trace only ever get an entry together (see
        # sql_agent_node), and only on a genuinely non-empty successful SQL
        # result (see evaluate()'s final_result), so every (table, result) pair
        # here is real, citable evidence -- including when a field-only question
        # ("X and Y?") was internally decomposed into 2+ subquestions against the
        # SAME table, which previously fell through this gate and got zero
        # citation for a genuinely single-source answer (reproduced live
        # 2026-07-22: the Screwfix Purchase-of-Expenditure-and-NET-Amount
        # question decomposed into 2 subquestions, both against doc_006, and
        # got no citation at all despite a real evidence row existing).
        tables = result.get("table_trace") or []
        results = result.get("result_trace") or []
        # Only trust index-alignment with sql_trace when all three lists are
        # the same length -- sql_trace can get an entry with no matching
        # table_trace/result_trace entry (a sub-question that produced SQL
        # but no table, see sql_agent_node), which would desync a naive zip.
        # Skipping the aggregate-quote improvement below rather than
        # guessing wrong is the safe default.
        sql_list = result.get("sql_trace") or []
        aligned_sql = sql_list if len(sql_list) == len(tables) == len(results) else None
        final_answer = result.get("final_answer") or "Unsupported"
        if final_answer.strip().lower() != "unsupported" and not final_answer.startswith(
            "Clarify:"
        ):
            citations: list[dict[str, Any]] = []
            seen_tables: set[str] = set()
            for idx, (table, table_result) in enumerate(zip(tables, results)):
                if table in seen_tables:
                    continue
                seen_tables.add(table)
                source = get_source_by_duckdb_table(QDRANT_URL, QDRANT_COLLECTION, table)
                if not source:
                    continue
                # The matched row(s), rendered as text, give SpreadsheetEvidence's
                # row-highlight heuristic (frontend, cell-value-in-quote match)
                # something real to match against instead of nothing.
                if table_result:
                    source["quote"] = table_result[:500]
                # An aggregate result (SUM/COUNT/AVG/...) is never itself one
                # row's value, so the quote above never matches any real row
                # -- SpreadsheetEvidence then fell back to showing the
                # sheet's first N rows, unrelated to the question. Reproduced
                # live 2026-07-25: "total NET Amount spent on MATERIALS"
                # showed unrelated transactions with no "MATERIALS" in
                # sight. Swap the quote for the WHERE clause's own filter
                # literal (e.g. "MATERIALS") instead -- that DOES appear
                # verbatim in the real matching rows' cells, so the existing
                # row-highlight heuristic finds a genuine example row. The
                # frontend shows a note that this is a computed total, not
                # the literal row value, when is_aggregate is set.
                sql = aligned_sql[idx] if aligned_sql else None
                if sql and _AGGREGATE_SQL_RE.search(sql):
                    literal_m = _WHERE_LITERAL_RE.search(sql)
                    if literal_m:
                        source["quote"] = literal_m.group(1)
                        source["is_aggregate"] = True
                citations.append(source)
            if citations:
                artifact["citations"] = citations
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
