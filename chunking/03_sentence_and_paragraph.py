"""
Demo 03: Sentence-based vs paragraph-based chunking.

Viser:
- hvordan naturlige tekstgraenser (saetninger/afsnit) giver andre chunks end fixed-size
- hvordan retrieval reagerer paa de to strategier for samme query

Saadan tolkes output:
- `Sentence groups` giver typisk mindre, mere fokuserede chunks
- `Paragraph chunks` bevarer mere struktur, men kan blive ujævne i størrelse
- sammenlign top-hits for relevans og læsbarhed/citation-egnethed

Brug demoen til at diskutere semantik vs. stoerrelseskontrol.
"""

from __future__ import annotations

try:
    from chunking.common import (
        MARKDOWN_DOC,
        SAMPLE_DOCS,
        chunk_paragraphs,
        chunk_sentence_groups,
        format_hits,
        print_chunk_preview,
        retrieve,
    )
except ModuleNotFoundError:
    from common import (
        MARKDOWN_DOC,
        SAMPLE_DOCS,
        chunk_paragraphs,
        chunk_sentence_groups,
        format_hits,
        print_chunk_preview,
        retrieve,
    )


def main() -> None:
    query = "What does the termination clause say about termination notice?"
    sentence_chunks = chunk_sentence_groups([MARKDOWN_DOC], group_size=2, sentence_overlap=0)
    paragraph_chunks = chunk_paragraphs([MARKDOWN_DOC], max_paragraphs_per_chunk=1)

    print("=== Eksempel 3: Sentence-based vs paragraph-based chunking ===")
    print("Query:", query)

    print("\n--- Sentence groups (2 saetninger) ---")
    print_chunk_preview(sentence_chunks, max_chunks=5)
    print("\nTop hits:")
    print(format_hits(retrieve(query, sentence_chunks, top_k=3)))

    print("\n--- Paragraph chunks ---")
    print_chunk_preview(paragraph_chunks, max_chunks=5)
    print("\nTop hits:")
    print(format_hits(retrieve(query, paragraph_chunks, top_k=3)))


if __name__ == "__main__":
    main()
