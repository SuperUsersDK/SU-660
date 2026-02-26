"""
Demo 06: Hybrid retrieval (sparse + dense) med RRF.

Viser:
- hvordan lexical og dense retrieval supplerer hinanden
- enkel fusion-strategi (Reciprocal Rank Fusion)
"""

from __future__ import annotations

try:
    from embeddings.common import print_hits, reciprocal_rank_fusion, retrieve_dense, retrieve_sparse
except ModuleNotFoundError:
    from common import print_hits, reciprocal_rank_fusion, retrieve_dense, retrieve_sparse


def main() -> None:
    query = "billing dispute after invoice date"
    sparse_hits = retrieve_sparse(query, top_k=4)
    dense_hits = retrieve_dense(query, top_k=4)
    fused = reciprocal_rank_fusion([sparse_hits, dense_hits])

    print("=== Eksempel 6: Hybrid retrieval + RRF ===")
    print("Query:", query)
    print()
    print_hits("Sparse", sparse_hits)
    print_hits("Dense", dense_hits)
    print_hits("Hybrid (RRF)", fused[:4])
    print("Pointen: Hybrid kan forbedre recall/robusthed ved ordlyd + semantik.")


if __name__ == "__main__":
    main()

