"""Tests for the SQL column-match hard gate in src/tools/excel.py."""

from __future__ import annotations

from src.tools.excel import _column_matches_question, _target_field_phrase


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
