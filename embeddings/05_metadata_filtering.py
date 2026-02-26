"""
Demo 05: Metadata filtering + embeddings.

Viser:
- hvorfor metadata-filter ofte boer koeres foer vector similarity
- hvordan det forbedrer precision og governance
"""

from __future__ import annotations

try:
    from embeddings.common import DOCS, print_hits, retrieve_dense
except ModuleNotFoundError:
    from common import DOCS, print_hits, retrieve_dense


def filter_docs(team: str | None = None, lang: str | None = None):
    out = DOCS
    if team is not None:
        out = [d for d in out if d.metadata.get("team") == team]
    if lang is not None:
        out = [d for d in out if d.metadata.get("lang") == lang]
    return out


def main() -> None:
    query = "How should support handle password reset?"
    print("=== Eksempel 5: Metadata filtering ===")
    print("Query:", query)
    print()
    print_hits("No filter", retrieve_dense(query, docs=DOCS, top_k=5))
    print_hits("Filter team=support", retrieve_dense(query, docs=filter_docs(team='support'), top_k=5))
    print_hits("Filter team=legal", retrieve_dense(query, docs=filter_docs(team='legal'), top_k=5))
    print("Pointen: metadata reducerer soege-rum og mindsker irrelevante hits.")


if __name__ == "__main__":
    main()

