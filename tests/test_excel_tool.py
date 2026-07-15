"""Tests for the SQL column-match hard gate in src/tools/excel.py."""

from __future__ import annotations

from src.tools.excel import _column_matches_question, _extract_sql, _target_field_phrase


class TestSqlInjectionGuard:
    """query_excel is a read-only lookup tool over the shared DuckDB file.
    Confirmed live that DuckDB's execute() runs semicolon-stacked statements
    by default ("SELECT 1; DROP TABLE t;" actually drops t) -- a
    prompt-injected SQL-generation call must not be able to reach anything
    but a single SELECT/WITH statement."""

    def test_rejects_statement_stacked_drop(self):
        assert _extract_sql("```sql\nSELECT 1; DROP TABLE t;\n```") is None

    def test_rejects_statement_stacked_delete(self):
        assert _extract_sql("```sql\nSELECT * FROM x; DELETE FROM x;\n```") is None

    def test_rejects_bare_drop_table(self):
        assert _extract_sql("```sql\nDROP TABLE doc_006_transactions;\n```") is None

    def test_rejects_update(self):
        assert _extract_sql("```sql\nUPDATE t SET x = 1;\n```") is None

    def test_allows_plain_select(self):
        sql = '```sql\nSELECT "NET Amount" FROM t WHERE "Supplier Name" ILIKE \'x\';\n```'
        assert _extract_sql(sql) is not None

    def test_allows_select_without_fence(self):
        assert _extract_sql("SELECT * FROM t") == "SELECT * FROM t"

    def test_allows_cte_with_clause(self):
        sql = "```sql\nWITH cte AS (SELECT 1 AS x) SELECT * FROM cte;\n```"
        assert _extract_sql(sql) is not None


def test_target_field_phrase_drops_row_qualifier_clause():
    q = (
        "What is the invoice number for the transaction with the highest "
        "NET Amount in the purchase card transactions dataset?"
    )
    assert _target_field_phrase(q) == "invoice number"


def test_target_field_phrase_falls_back_to_whole_question_without_clause():
    q = "What is the total amount for transaction number 6091984?"
    # No "for/with/on/in the ..." row-qualifier clause here ("for transaction
    # number", not "for the transaction") -- only the leading "what is the"
    # prefix is stripped, the rest is kept intact.
    assert _target_field_phrase(q) == "total amount for transaction number 6091984"


def test_column_match_blocks_wrong_column_hidden_behind_row_qualifier_overlap():
    """ "Transaction Number" shares "transaction" with the question's row-qualifier
    clause ("...for the transaction with..."), not with the field actually asked
    for ("invoice number") -- must not pass on that coincidence."""
    q = (
        "What is the invoice number for the transaction with the highest "
        "NET Amount in the purchase card transactions dataset?"
    )
    assert _column_matches_question("Transaction Number", q) is False


def test_column_match_allows_real_match_on_target_field():
    q = (
        "What is the invoice number for the transaction with the highest "
        "NET Amount in the purchase card transactions dataset?"
    )
    assert _column_matches_question("Invoice Number", q) is True


def test_column_match_still_blocks_unrelated_column_without_row_qualifier():
    q = "What payment method was used for the transaction with the largest Total?"
    assert _column_matches_question("Merchant Category", q) is False


def test_column_match_blocks_generic_column_with_no_real_overlap():
    """ "Total" is a generic word, so it used to auto-pass the gate no matter what
    was asked -- reproduced: doc_007_qa_9 let "Total" answer a "payment method"
    question, hallucinating an answer instead of refusing. A generic column must
    still share a literal word with the question to pass."""
    q = "What payment method was used for the transaction with the largest Total in the published spend report?"
    assert _column_matches_question("Total", q) is False


def test_column_match_allows_generic_column_with_real_overlap():
    """Generic columns should still pass when the question actually asks for
    that generic field (e.g. "Amount" answering "what is the amount")."""
    q = "What is the amount for transaction number 6091984?"
    assert _column_matches_question("Amount", q) is True
