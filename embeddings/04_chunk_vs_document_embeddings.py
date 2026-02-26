"""
Demo 04: Chunk-level vs document-level embeddings.

Viser:
- hvorfor hele dokumenter kan blive for brede/noisy
- hvorfor chunk embeddings typisk er bedre til RAG retrieval
"""

from __future__ import annotations

from dataclasses import replace

try:
    from embeddings.common import DOCS, Doc, print_hits, retrieve_dense
except ModuleNotFoundError:
    from common import DOCS, Doc, print_hits, retrieve_dense


def split_into_chunks(doc: Doc) -> list[Doc]:
    sentences = [s.strip() for s in doc.text.split(". ") if s.strip()]
    chunks: list[Doc] = []
    for i, sentence in enumerate(sentences, start=1):
        text = sentence if sentence.endswith(".") else sentence + "."
        chunks.append(
            replace(
                doc,
                doc_id=f"{doc.doc_id}_c{i}",
                text=text,
                metadata={**doc.metadata, "parent_doc": doc.doc_id, "chunk_index": str(i)},
            )
        )
    return chunks


def main() -> None:
    query = "What information must be in a termination notice?"
    doc_level = DOCS
    chunk_level = [chunk for d in DOCS for chunk in split_into_chunks(d)]

    print("=== Eksempel 4: Document vs chunk embeddings ===")
    print("Query:", query)
    print()
    print_hits("Document-level retrieval", retrieve_dense(query, docs=doc_level, top_k=4))
    print_hits("Chunk-level retrieval", retrieve_dense(query, docs=chunk_level, top_k=6))
    print("Pointen: chunk-level retrieval giver ofte mere specifik grounding til LLM-svar.")


if __name__ == "__main__":
    main()

