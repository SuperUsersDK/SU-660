"""
Demo 03: Query embedding strategier.

Viser:
- baseline query
- query rewrite (normalisering af brugerord)
- multi-query retrieval og fusion
"""

from __future__ import annotations

try:
    from embeddings.common import print_hits, reciprocal_rank_fusion, retrieve_dense
except ModuleNotFoundError:
    from common import print_hits, reciprocal_rank_fusion, retrieve_dense


def main() -> None:
    user_query = "How do we stop a contract and what must the notice include?"
    rewritten_query = "How do we terminate a contract and what must the termination notice include?"
    subqueries = [
        "How do we terminate a contract?",
        "What must a termination notice include?",
    ]

    print("=== Eksempel 3: Query embedding strategier ===")
    print("User query:", user_query)
    print()
    print_hits("A) Baseline dense retrieval", retrieve_dense(user_query, top_k=4))
    print_hits("B) Query rewrite before embedding", retrieve_dense(rewritten_query, top_k=4))

    fused = reciprocal_rank_fusion([retrieve_dense(q, top_k=4) for q in subqueries])
    print_hits("C) Multi-query + RRF fusion", fused[:4])
    print("Note: Multi-query giver ofte bedre recall, men oeger latency/cost.")


if __name__ == "__main__":
    main()

