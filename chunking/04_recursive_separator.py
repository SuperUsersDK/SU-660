"""
Demo 04: Recursive chunking med separator-hierarki.

Viser:
- hvordan teksten splittes fra grove til fine separatorer (afsnit -> linje -> saetning -> ord)
- hvordan man kan kombinere basisstørrelse og efterfølgende overlap

Saadan tolkes output:
- chunk-længder vil variere (det er meningen)
- overlap kan ses som lidt gentaget tekst i naeste chunk
- max_chars begrænser basis-segmentet før overlap; færdige chunks kan være længere
- top-hits skal vurderes paa om de bevarer nok lokal kontekst uden for meget stoej

Dette er ofte den mest nyttige "standardstrategi" at starte med i RAG.
"""

from __future__ import annotations

try:
    from chunking.common import MARKDOWN_DOC, chunk_recursive, format_hits, print_chunk_preview, retrieve
except ModuleNotFoundError:
    from common import MARKDOWN_DOC, chunk_recursive, format_hits, print_chunk_preview, retrieve


def main() -> None:
    query = "What do the billing terms say about invoices and due dates?"
    chunks = chunk_recursive([MARKDOWN_DOC], max_chars=180, overlap_chars=30)

    print("=== Eksempel 4: Recursive chunking (separator-hierarki) ===")
    print("Separators: [\\\\n\\\\n, \\\\n, '. ', ' ', '']")
    print("Basis-segment: højst 180 tegn; derefter tilføjes op til 30 tegn overlap.")
    print("Færdige chunks kan derfor være op til 211 tegn inklusive mellemrum.")
    print()
    print_chunk_preview(chunks, max_chunks=10)

    print("\nTop hits:")
    print(format_hits(retrieve(query, chunks, top_k=4)))


if __name__ == "__main__":
    main()
