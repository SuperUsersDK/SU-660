"""
Demo 05: Heading-aware chunking for Markdown.

Viser:
- hvordan headings/sektioner kan bruges som primære chunkgraenser
- hvordan heading-metadata forbedrer explainability og retrieval-inspektion

Saadan tolkes output:
- `heading=...` viser hvilken sektion chunket kommer fra
- hoej score + korrekt heading er et stærkt signal for god retrieval-kvalitet
- brug output til at diskutere hvorfor strukturbevarelse er vigtig i policies/manuals

Demoen illustrerer værdien af strukturbaseret chunking frem for kun størrelse-baseret chunking.
"""

from __future__ import annotations

try:
    from chunking.common import MARKDOWN_DOC, chunk_markdown_headings, retrieve
except ModuleNotFoundError:
    from common import MARKDOWN_DOC, chunk_markdown_headings, retrieve


def main() -> None:
    query = "What does the termination clause say about written notice?"
    chunks = chunk_markdown_headings(MARKDOWN_DOC, max_chars_per_section_chunk=160)

    print("=== Eksempel 5: Heading-aware chunking (Markdown) ===")
    print(f"Antal heading-aware chunks: {len(chunks)}")
    print()
    for c in chunks:
        print(
            f"- {c.chunk_id} | heading={c.metadata.get('heading')} | "
            f"len={len(c.text)} | {c.text[:90].replace(chr(10), ' ')}..."
        )

    print("\nTop hits:")
    for i, (chunk, score) in enumerate(retrieve(query, chunks, top_k=4), start=1):
        print(
            f"{i}. score={score:.3f} | heading={chunk.metadata.get('heading')} | "
            f"{chunk.text[:110].replace(chr(10), ' ')}"
        )


if __name__ == "__main__":
    main()
