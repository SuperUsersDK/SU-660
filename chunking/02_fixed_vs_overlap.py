"""
Demo 02: Fixed-size chunking uden overlap vs med overlap.

Viser:
- forskellen mellem samme chunk-stoerrelse med og uden overlap
- hvordan overlap paavirker antal chunks og retrieval-resultater

Saadan tolkes output:
- flere chunks ved overlap er forventet (mere redundans)
- sammenlign top-hits for at se om overlap hjælper boundary-cases
- hvis hits bliver meget ens/dublerede, kan overlap vaere for hoejt

Maalet er at forstaa tradeoff mellem recall og redundans/cost.
"""

from __future__ import annotations

try:
    from chunking.common import SAMPLE_DOCS, chunk_fixed_words, retrieve, format_hits
except ModuleNotFoundError:
    from common import SAMPLE_DOCS, chunk_fixed_words, retrieve, format_hits


def main() -> None:
    docs = [d for d in SAMPLE_DOCS if d["id"] == "rag_notes"]
    query = "How does overlap affect recall around chunk boundaries?"

    no_overlap = chunk_fixed_words(docs, chunk_words=10, overlap_words=0, strategy_name="fixed_no_overlap")
    with_overlap = chunk_fixed_words(docs, chunk_words=10, overlap_words=3, strategy_name="fixed_overlap")

    print("=== Eksempel 2: Fixed-size uden overlap vs med overlap ===")
    print("Query:", query)
    print(f"Chunks uden overlap: {len(no_overlap)}")
    print(f"Chunks med overlap:  {len(with_overlap)}")

    print("\n--- Uden overlap ---")
    print(format_hits(retrieve(query, no_overlap, top_k=4)))

    print("\n--- Med overlap (3 ord) ---")
    print(format_hits(retrieve(query, with_overlap, top_k=4)))


if __name__ == "__main__":
    main()
