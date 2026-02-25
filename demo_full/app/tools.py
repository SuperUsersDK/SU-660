from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '23*47'."""
    allowed = set("0123456789+-*/(). %")
    if any(c not in allowed for c in expression):
        return "Unsupported characters in expression."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def doc_stats(sources: list[str]) -> dict[str, Any]:
    """Return basic stats from a list of source filenames (from retrieved metadata)."""
    out: dict[str, int] = {}
    for src in sources:
        out[src] = out.get(src, 0) + 1
    return {"chunks_by_source": out, "unique_sources": len(out)}


@tool
def search_in_context(needle: str, context: str) -> str:
    """Find occurrences of a phrase in the provided context text."""
    if not needle.strip():
        return "Empty needle."
    idx = context.lower().find(needle.lower())
    if idx == -1:
        return "Not found."
    start = max(0, idx - 200)
    end = min(len(context), idx + 200)
    return context[start:end]

