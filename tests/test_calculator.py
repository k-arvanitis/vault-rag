"""Tests for the controlled arithmetic tool in src/tools/calculator.py."""

from src.tools.calculator import calculate


def test_calculate_basic_arithmetic():
    assert calculate("2 + 2") == "4"


def test_calculate_percentage_change():
    result = calculate("(1510000 - 1240000) / 1240000 * 100")
    assert result.startswith("21.7")


def test_calculate_strips_currency_and_commas():
    assert calculate("$1,240,000 + $10,000") == "1250000"


def test_calculate_rejects_name_lookup():
    result = calculate("__import__('os').system('echo hi')")
    assert result.startswith("Calculator error")


def test_calculate_rejects_function_call():
    result = calculate("len([1, 2, 3])")
    assert result.startswith("Calculator error")


def test_calculate_rejects_attribute_access():
    result = calculate("(1).__class__")
    assert result.startswith("Calculator error")
