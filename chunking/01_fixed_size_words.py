"""
Demo 01: Fixed-size chunking (ord) uden overlap.

Viser:
- hvordan et dokument bliver delt i faste ord-vinduer
- hvordan retrieval-hits ser ud med denne simple baseline-strategi

Saadan tolkes output:
- `Antal chunks` viser hvor mange stykker teksten blev delt i
- `tok~` er et groft token-estimat (didaktisk, ikke rigtig tokenizer)
- `score` er en simpel lexical similarity-score (ikke embeddings)
- Hoje scores betyder kun "mere ord-overlap med query", ikke noedvendigvis bedst semantik

Brug demoen som baseline for sammenligning med de senere strategier.
"""

from __future__ import annotations

try:
    from chunking.common import SAMPLE_DOCS, chunk_fixed_words, print_chunk_preview, retrieve, format_hits
except ModuleNotFoundError:
    from common import SAMPLE_DOCS, chunk_fixed_words, print_chunk_preview, retrieve, format_hits


def main() -> None:
    docs = [d for d in SAMPLE_DOCS if d["id"] == "rag_notes"]
    query = "Why should top-k be tuned in RAG?"

    chunks = chunk_fixed_words(docs, chunk_words=12, overlap_words=0)

    print("=== Eksempel 1: Fixed-size chunking (ord) uden overlap ===")
    print("Query:", query)
    print()
    print_chunk_preview(chunks)
    print("\nTop hits:")
    print(format_hits(retrieve(query, chunks, top_k=4)))


if __name__ == "__main__":
    main()
