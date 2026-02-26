"""
Demo 01: Lexical vs dense retrieval (didaktisk mock).

Viser:
- hvorfor embeddings kan finde semantisk relaterede tekster uden exact keyword match
- hvorfor lexical stadig er nyttig som baseline
"""

from __future__ import annotations

try:
    from embeddings.common import QUERIES, print_hits, retrieve_dense, retrieve_sparse
except ModuleNotFoundError:
    from common import QUERIES, print_hits, retrieve_dense, retrieve_sparse


def main() -> None:
    query = "How can a customer cancel a purchase and get money back?"
    print("=== Eksempel 1: Lexical vs dense retrieval ===")
    print("Query:", query)
    print()
    print_hits("Sparse / lexical", retrieve_sparse(query, top_k=4))
    print_hits("Dense / embedding (mock)", retrieve_dense(query, top_k=4))

    print("Ekstra queries:")
    for q in QUERIES:
        print(f"- {q}")


if __name__ == "__main__":
    main()

