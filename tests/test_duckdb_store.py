"""Unit tests for SQL-literal rewriting helpers in src/duckdb_store.py."""
from __future__ import annotations

from vault_rag.duckdb_store import _normalize_special_chars_ilike


def test_normalizes_ampersand_to_match_and_variant():
    sql = "SELECT * FROM t WHERE \"Supplier\" ILIKE '%Smith & Jones%'"
    out = _normalize_special_chars_ilike(sql)
    assert "'%smithandjones%'" in out
    assert "regexp_replace" in out


def test_normalizes_and_variant_to_same_form():
    sql = "SELECT * FROM t WHERE \"Supplier\" ILIKE '%Smith and Jones%'"
    out = _normalize_special_chars_ilike(sql)
    assert "'%smithandjones%'" in out


def test_leaves_sql_without_ilike_unchanged():
    sql = "SELECT * FROM t WHERE \"Amount\" = 5"
    assert _normalize_special_chars_ilike(sql) == sql
