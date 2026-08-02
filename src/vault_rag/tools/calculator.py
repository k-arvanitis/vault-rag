"""A controlled arithmetic tool: evaluates a numeric expression the agent builds
from values it already retrieved and cited -- it never invents numbers itself.

Only literals and +-*/()** are accepted; parsed via `ast` and evaluated by walking
the parse tree, so no name lookups, attribute access, or function calls are
possible (unlike a raw `eval()`).
"""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a restricted arithmetic AST node, or raise ValueError."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def _safe_eval(expression: str) -> float:
    """Parse and evaluate a numeric expression, rejecting anything but +-*/()**."""
    cleaned = expression.replace(",", "").replace("$", "").strip()
    tree = ast.parse(cleaned, mode="eval")
    return _eval_node(tree.body)


class _CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "A numeric arithmetic expression built only from values you already "
            "retrieved and cited, e.g. '(1510000 - 1240000) / 1240000 * 100'. "
            "Never include a value you have not already retrieved."
        )
    )


def calculate(expression: str) -> str:
    """Evaluate a numeric expression and return the result as a string, or an error."""
    try:
        result = _safe_eval(expression)
    except Exception as exc:
        return f"Calculator error: could not evaluate '{expression}' ({exc})"
    return str(result)


def build_calculator_tool() -> StructuredTool:
    """Return the LangChain StructuredTool for controlled arithmetic over cited values."""
    return StructuredTool.from_function(
        func=calculate,
        name="calculate",
        description=(
            "Evaluate a numeric arithmetic expression (+, -, *, /, **, parentheses) "
            "built ONLY from values you already retrieved and cited via another tool. "
            "Use this for sums, differences, percentages, and ratios over numbers "
            "pulled from PDF/OCR text -- never guess or round the inputs yourself, "
            "and never use this tool to invent a number that wasn't retrieved."
        ),
        args_schema=_CalculatorInput,
    )
