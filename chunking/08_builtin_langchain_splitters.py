"""
Demo 08: Comparison of built-in LangChain chunking strategies.

Shows:
- how built-in splitters in `langchain_text_splitters` split the same text differently
- how chunk boundaries affect retrieval hits and scores in a controlled comparison

Strategies in this demo:
- CharacterTextSplitter
- RecursiveCharacterTextSplitter
- TokenTextSplitter
- MarkdownHeaderTextSplitter (on markdown text)

How to interpret results:
- compare chunk count, chunk previews, and top-k hits side-by-side
- do not over-index on top-1 score only (look at top-k and chunk usefulness)
- scores are simple lexical similarity scores (didactic), not embedding similarity

Use this demo in class to discuss built-in chunking tradeoffs before custom chunking logic.
"""

from __future__ import annotations

from typing import Any

try:
    from chunking.common import Chunk, MARKDOWN_DOC, SAMPLE_DOCS, format_hits, print_chunk_preview, retrieve
except ModuleNotFoundError:
    from common import Chunk, MARKDOWN_DOC, SAMPLE_DOCS, format_hits, print_chunk_preview, retrieve


def _wrap_text_chunks(
    texts: list[str],
    *,
    doc_id: str,
    strategy: str,
    extra_meta: list[dict[str, Any]] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, text in enumerate(texts):
        meta = {"strategy": strategy}
        if extra_meta and i < len(extra_meta):
            meta.update(extra_meta[i])
        text = text.strip()
        if not text:
            continue
        chunks.append(Chunk(chunk_id=f"{doc_id}_{strategy}_{i}", doc_id=doc_id, text=text, metadata=meta))
    return chunks


def _header_aware_chunks() -> list[Chunk]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
    docs = splitter.split_text(MARKDOWN_DOC["text"])

    out: list[Chunk] = []
    for i, doc in enumerate(docs):
        out.append(
            Chunk(
                chunk_id=f"{MARKDOWN_DOC['id']}_markdown_header_{i}",
                doc_id=MARKDOWN_DOC["id"],
                text=doc.page_content.strip(),
                metadata={"strategy": "markdown_header", **doc.metadata},
            )
        )
    return out


def _print_strategy_report(name: str, chunks: list[Chunk], query: str, top_k: int = 4) -> None:
    print(f"\n=== {name} ===")
    print_chunk_preview(chunks, max_chunks=6)
    print("\nTop hits:")
    print(format_hits(retrieve(query, chunks, top_k=top_k)))


def main() -> None:
    try:
        from langchain_text_splitters import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
            TokenTextSplitter,
        )
    except ImportError:
        print("This demo requires `langchain_text_splitters` (and `tiktoken` for TokenTextSplitter).")
        print("Install example: pip install langchain-text-splitters tiktoken")
        return

    plain_doc = next(d for d in SAMPLE_DOCS if d["id"] == "rag_notes")
    plain_text = plain_doc["text"]
    query = "Why should top-k be tuned in RAG, and how does overlap affect recall?"

    char_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=90,
        chunk_overlap=20,
        keep_separator=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=25,
        separators=[". ", " ", ""],
    )
    token_splitter = TokenTextSplitter(
        model_name="gpt-4o-mini",
        chunk_size=28,
        chunk_overlap=6,
    )

    char_chunks = _wrap_text_chunks(char_splitter.split_text(plain_text), doc_id=plain_doc["id"], strategy="character")
    recursive_chunks = _wrap_text_chunks(
        recursive_splitter.split_text(plain_text),
        doc_id=plain_doc["id"],
        strategy="recursive_character",
    )
    token_chunks = _wrap_text_chunks(token_splitter.split_text(plain_text), doc_id=plain_doc["id"], strategy="token")

    print("=== Demo 08: Built-in LangChain splitters ===")
    print("Same text + same query. Only the chunking strategy changes.")
    print("Query:", query)

    _print_strategy_report("CharacterTextSplitter", char_chunks, query)
    _print_strategy_report("RecursiveCharacterTextSplitter", recursive_chunks, query)
    _print_strategy_report("TokenTextSplitter", token_chunks, query)

    header_query = "What does the termination clause say about written notice?"
    header_chunks = _header_aware_chunks()
    print("\n=== MarkdownHeaderTextSplitter (heading-aware on markdown text) ===")
    print("Query:", header_query)
    print(f"Antal chunks: {len(header_chunks)}")
    for chunk in header_chunks:
        print(
            f"- {chunk.chunk_id} | h1={chunk.metadata.get('h1')} | h2={chunk.metadata.get('h2')} | "
            f"{chunk.text[:100].replace(chr(10), ' ')}"
        )
    print("\nTop hits:")
    print(format_hits(retrieve(header_query, header_chunks, top_k=4)))

    print("\nInterpretation prompts:")
    print("- Which splitter preserves the most useful local context?")
    print("- Which splitter is easiest to explain/debug?")
    print("- Which strategy would you use for markdown policies vs plain text PDFs?")


if __name__ == "__main__":
    main()

