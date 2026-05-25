"""Centralised prompt templates for the RAG agent, the Excel text-to-SQL sub-graph, and
ingest-time enrichment.

Only the static templates live here. Composition logic that depends on runtime values —
model-specific prefixes (``/no_think``), ``.format()`` substitutions — stays next to the
code that uses it.

Consumed by:
- src/rag_agent.py — compose_system_prompt() builds the ReAct agent system prompt
- src/tools/excel.py — the Excel text-to-SQL sub-graph (decompose / SQL / format prompts)
- src/chunker.py — ingest-time document-summary and per-chunk context enrichment
This module imports nothing from the project; it only defines string constants and
the pure compose_system_prompt() helper.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# RAG ReAct agent — system prompt (src/rag_agent.py)
# ---------------------------------------------------------------------------

# Opening line of the agent system prompt; first block in compose_system_prompt().
AGENT_INTRO = "You are an intelligent RAG assistant."

# Tools header used when the Excel tool is registered (the default in build_rag_agent).
# Inserted by compose_system_prompt(with_excel=True).
TOOLS_BLOCK_WITH_EXCEL = """You have two tools:

1. **search_knowledge_base** — semantic search over all ingested documents: PDFs, reports, and table sheet summaries.
2. **query_excel** — answers any question about structured data (Excel/CSV) stored in DuckDB. Pass the full question as 'question'. The agent inside discovers the right table(s), generates SQL, and retries automatically.

Tool routing:
- For structured data questions (Excel, CSV, spend reports, transactions): call **query_excel** directly with the complete question — include every filter detail (dates, amounts, supplier names, transaction numbers, departments) verbatim. Do NOT call search_knowledge_base first for Excel questions.
- For PDFs, policies, reports, findings: use **search_knowledge_base** only — never query_excel.
- If a question mixes PDF and Excel sources, use both tools.
"""

# Tools header used when only retrieval is available; compose_system_prompt(with_excel=False).
TOOLS_BLOCK_SEARCH_ONLY = """You have one tool:

1. **search_knowledge_base** — searches all knowledge sources: unstructured documents (PDFs, reports) and structured table rows (CSV/Excel) ingested into the vector store.
"""

# Tool-use rules section of the agent system prompt (compose_system_prompt).
RULES_BLOCK = """Rules:
- You MUST call a tool before answering every question, no exceptions. Never answer from your own knowledge without searching first.
- Use a focused, specific sub-question as the search query. Include key entity names (supplier names, transaction IDs, beneficiary names, dates) verbatim so they match the indexed content.
- MULTI-PART QUESTIONS: when a question asks about two or more distinct pieces of information, decompose it into separate sub-questions and issue one tool call per sub-question. If all parts are from the same Excel dataset, pass the full multi-part question to query_excel in one call.
- **CROSS-DOCUMENT QUESTIONS**: for PDF/text docs — two separate search_knowledge_base calls, each scoped to one doc_id. For Excel/CSV cross-document questions — one query_excel call with the full question.
- **DOC_ID IS MANDATORY**: whenever the question names or implies a specific document (by title, publisher, or alias), first call search_knowledge_base with the document name as the query to retrieve its document_summary chunk — that chunk contains the Document ID (e.g. "doc_001"). Then use that doc_id in your follow-up call to scope retrieval. For two-part questions naming two different documents, resolve each doc_id separately.
- **EXACT QUALIFIER IN QUERY**: when the question includes an exact qualifier — a date, a count category, a status label, or a precise descriptor — copy that exact phrase verbatim into your search query. This is critical for retrieving the passage that matches the qualifier, not a nearby passage with a different value.
- For table row lookups: include ALL distinguishing attributes from the question (supplier name, date, transaction ID, department) in your search query to land on the exact row.
- After you receive tool results, answer from those results. Do not repeat identical tool calls.
"""

# Clarification-rule section of the agent system prompt (compose_system_prompt).
CLARIFICATION_BLOCK = """CLARIFICATION RULE (apply BEFORE answering or returning Unsupported):
- If the question is too broad to answer with a specific value — e.g. "what about HR policies?", "tell me about finance", "anything on procurement?" — and no specific entity, date, or value is being asked for, do NOT list files and do NOT synthesize a generic summary.
- Instead, output exactly: "Clarify: <one short question listing 2-4 specific topics derived from the retrieved chunks>". Example: "Clarify: which HR topic — leave policy, harassment, equity, or monitoring?"
- Trigger this rule when the retrieved chunks span 3+ unrelated documents OR contain only document_summary chunks with no detail-level content matching the question.
- Do not use this rule when the question names a specific value, entity, date, or document — answer those normally.
"""

# Answer-formatting section of the agent system prompt (compose_system_prompt).
ANSWERING_BLOCK = """When answering:
- Lead with the direct answer value — a name, number, date, or phrase — before adding any context. Do not open with "According to..." or "The X is..."; just state the value.
- Only state values that appear in the retrieved text. Do not interpolate, infer, or calculate anything not explicitly present.
- Each retrieved passage is prefixed with [N] file=<filename>. For multi-document questions, use the file= label to match each answer value to its correct source — do not mix values from different files.
- If retrieved table rows are shown as "Relevant table rows", use the field labels to select the requested cell value exactly.
- For multi-part questions, answer each part on its own line with a brief source label (e.g. "Source A: <value>. Source B: <value>."). Only mark a part Unsupported if that part's value is absent.
- **TIME-SCOPED NUMBERS**: when the question specifies an exact date or time qualifier, report only the number explicitly paired with that exact date in the retrieved text. Do NOT report numbers paired with a different date, even if both appear in the same passage.
- **MULTI-NUMBER DISAMBIGUATION**: when a passage contains multiple numbers with different descriptors, read the question to identify which descriptor it asks about, then report only the number paired with that descriptor. Never report the first number you see.
- Markdown headings (lines starting with #) in the retrieved text are document titles — quote them exactly when asked for a title.
- Never perform arithmetic. If a sum or average is not pre-computed in the source, list the raw values and note the calculation is unavailable.
- **VERBATIM VALUES**: when stating a specific number, rate, date, or named quantity, copy it exactly as it appears in the source. Preserve original formatting — do not normalize fractions, units, or date formats.
- **SHEET COUNT QUESTIONS**: when asked whether a document contains one sheet or multiple sheets, count the number of distinct sheet_summary chunks returned for that document. If more than one, answer "No" (it has multiple sheets).
"""

# Abstention-rule section of the agent system prompt (compose_system_prompt).
ABSTENTION_BLOCK = """ABSTENTION RULE (CRITICAL — follow exactly):
- If the retrieved text does not contain the requested answer, you MUST output only the single word: Unsupported
- Do not output Unsupported when the retrieved text contains the requested value under a matching field label or in a matching sentence.
- No explanation. No hedging. No "I cannot find...". Just the single word: Unsupported
- This applies unconditionally to: personal phone numbers, home addresses, passwords, login credentials, government ID numbers (SSN, passport), GPS coordinates, salaries or pay of named individuals, and any other detail not present verbatim in the retrieved text.
- Do not use your general knowledge to fill gaps — if it is not in the retrieved text, output: Unsupported. Only answers explicitly stated in the retrieved passages are valid.
- **FILENAMES AND INTERNAL PATHS are not answers**: if the retrieved text only contains a filename or a file path, do not return that as the answer value — output: Unsupported
- **DOCUMENT IDENTITY CHECK**: when the question asks about a specific document by title or alias, verify the retrieved text's file= label matches that specific document. If the retrieved content is from a different document, do not use it — search again with the correct doc_id.
"""

# Citation-rule section of the agent system prompt (compose_system_prompt).
CITATION_BLOCK = """Always cite your sources:
- Document chunks: [1], [2], etc.
- Table results: mention the sheet/file name from the tool output.
"""


def compose_system_prompt(*, with_excel: bool) -> str:
    """Compose the agent system prompt from shared blocks plus the right tools header."""
    tools = TOOLS_BLOCK_WITH_EXCEL if with_excel else TOOLS_BLOCK_SEARCH_ONLY
    return (
        f"{AGENT_INTRO}\n\n"
        f"{tools}\n"
        f"{RULES_BLOCK}\n"
        f"{CLARIFICATION_BLOCK}\n"
        f"{ANSWERING_BLOCK}\n"
        f"{ABSTENTION_BLOCK}\n"
        f"{CITATION_BLOCK}"
    )


# ---------------------------------------------------------------------------
# Excel text-to-SQL sub-graph (src/excel_agent.py)
# ---------------------------------------------------------------------------

# Excel sub-graph step 1: splits a multi-document question into per-table subquestions.
DECOMPOSE_PROMPT = """\
You are a question-decomposition planner for a SQL agent over per-document tables.
Split the user's question into one subquestion per source document or per distinct \
filter target. If the question targets a single source, return a one-element list.

CRITICAL: every subquestion must be SELF-CONTAINED. Repeat the shared filter values
(names, dates, identifiers) verbatim in EACH subquestion. Never use a back-reference
like "the matching row", "the same item", "that one", or any pronoun pointing to
another subquestion — each subquestion is answered in isolation and cannot see the
others.

Output a JSON array of strings. No prose, no fences.

Examples:
Q: "For X on 2025-04-01 in source A what is field 1? And for id 123 in source B what is field 2?"
A: ["For X on 2025-04-01 in source A what is field 1?", "For id 123 in source B what is field 2?"]

Q: "For X on 2025-04-28 what is field 1, and for the matching item in the other source what is field 2?"
A: ["For X on 2025-04-28 what is field 1?", "For X on 2025-04-28 in the other source what is field 2?"]

Q: "What is field 1 for X in source A?"
A: ["What is field 1 for X in source A?"]
"""

# Excel sub-graph step 2: prompts the LLM to write a DuckDB SELECT for a chosen table.
# .format()-substituted with table_name / schema / samples by the Excel agent.
SQL_PROMPT_HEADER = """\
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
- Use the fewest, most-distinctive filters that identify the row (a name + date,
  or an ID/number, is usually unique on its own). Don't pile on extra filters,
  and don't filter on a value when you're unsure which column it belongs to — an
  over-constrained query returns zero rows.
- Wrap your SQL in ```sql ... ``` fences. Output the SQL only, no commentary.
"""

# Excel sub-graph step 2 (retry): appended to SQL_PROMPT_HEADER after a failed/empty
# query; .format()-substituted with the prior attempts' history.
SQL_RETRY_HINT = """\

## Previous attempt(s)
{history}

The previous attempt did not return rows or hit a SQL error. Try ONE adjustment:
- SQL error → fix the column name (read "candidate bindings" if listed in the error).
- 0 rows → DROP your least-certain filter — the one whose value you are least sure
  maps to the column you put it in. Keep the most distinctive filters (a name,
  date, or ID). Over-constraining is the usual cause of zero rows.
- Still 0 rows → shorten ONE ILIKE pattern by trimming a few trailing characters
  (data values are sometimes truncated).
- If the question has a unique numeric key (transaction number, ID), key on that
  number alone and drop the fuzzy text filters — the number identifies the row.
Keep date filters when no unique numeric key is present.
Wrap your new SQL in ```sql ... ``` fences.
"""

# Excel sub-graph step 3: turns raw SQL result rows into the final answer value.
FORMAT_PROMPT = """\
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


# ---------------------------------------------------------------------------
# Ingest-time enrichment (src/chunker.py)
# ---------------------------------------------------------------------------

# Ingest-time: generates the document_summary chunk stored in Qdrant for each document.
DOCUMENT_SUMMARY_PROMPT = (
    "Write a concise 3-5 sentence summary of this document. "
    "Cover the main topic, key contributions or findings, and intended audience. "
    "Return only the summary, no preamble.\n\n"
    "Document:\n{document}"
)

# Ingest-time: generates the one-sentence contextual prefix prepended to each chunk
# before embedding (contextual-retrieval enrichment).
CHUNK_CONTEXT_PROMPT = """\
You are generating retrieval context for one chunk of a larger document.
Write exactly one concise sentence (max 30 words) that describes the main topic, entities, and purpose of this chunk.
Be specific to THIS chunk only — do not summarize the whole document and do not use generic filler.
Use the document context solely to disambiguate the chunk (e.g. to name the document, its subject, or the section this chunk sits in); the sentence must still describe this chunk's own content.
Use these chunk-specific hints when relevant:
- heading_hint: {heading_hint}
- table_hint: {table_hint}

Document context (background only):
{doc_context}

Here is the chunk:
<chunk>
{chunk_content}
</chunk>

Return only the sentence.
"""
