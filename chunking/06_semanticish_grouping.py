"""
Demo 06: "Semantic-ish" grouping vs fast sentence grouping.

Viser:
- en didaktisk approximation af semantisk chunking (lexical similarity, ikke embeddings)
- hvordan adaptive grupper kan afvige fra faste sentence-grupper

Saadan tolkes output:
- `Semantic-ish` er kun en undervisningsmodel, ikke produktionsklar semantic chunking
- sammenlign antal chunks og top-hits for at se effekten af adaptive graenser
- brug output til at forklare hvorfor "semantisk chunking" er dyrere og mere kompleks i praksis

Pointen er at illustrere ideen bag semantiske chunkgraenser uden ekstra dependencies.
"""

from __future__ import annotations

try:
    from chunking.common import SAMPLE_DOCS, chunk_semanticish_sentences, chunk_sentence_groups, format_hits, retrieve
except ModuleNotFoundError:
    from common import SAMPLE_DOCS, chunk_semanticish_sentences, chunk_sentence_groups, format_hits, retrieve


def main() -> None:
    docs = SAMPLE_DOCS
    query = "What should we log, and how should we handle tool failures?"

    baseline = chunk_sentence_groups(docs, group_size=2, sentence_overlap=0, strategy_name="sentence_2")
    semanticish = chunk_semanticish_sentences(docs, max_sentences_per_chunk=3, similarity_threshold=0.10)

    print("=== Eksempel 6: 'Semantic-ish' grouping vs fast sentence grouping ===")
    print("NB: Dette er en didaktisk lexical-approximation, ikke rigtig embedding-baseret semantic chunking.")
    print(f"Sentence groups: {len(baseline)} chunks")
    print(f"Semantic-ish:    {len(semanticish)} chunks")

    print("\n--- Baseline (sentence groups) ---")
    print(format_hits(retrieve(query, baseline, top_k=5)))

    print("\n--- Semantic-ish grouping ---")
    print(format_hits(retrieve(query, semanticish, top_k=5)))


if __name__ == "__main__":
    main()
