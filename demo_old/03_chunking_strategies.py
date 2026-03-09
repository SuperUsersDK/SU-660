from common import SAMPLE_DOCS, build_chunks_fixed, build_chunks_sentence, format_hits, retrieve


def main() -> None:
    docs = [doc for doc in SAMPLE_DOCS if doc["id"] == "doc_rag"]
    query = "Hvordan paavirker chunking og top-k relevansen i RAG?"

    fixed_chunks = build_chunks_fixed(docs, chunk_words=12, overlap_words=3)
    sentence_chunks = build_chunks_sentence(docs, group_size=1)

    print("=== Chunking-strategier pa samme dokument ===")
    print(f"Query: {query}\n")
    print(f"Fast chunking: {len(fixed_chunks)} chunks")
    print(f"Sentence chunking: {len(sentence_chunks)} chunks")

    print("\n--- Retrieval med fast chunking ---")
    print(format_hits(retrieve(query, fixed_chunks, top_k=3)))

    print("\n--- Retrieval med sentence chunking ---")
    print(format_hits(retrieve(query, sentence_chunks, top_k=3)))


if __name__ == "__main__":
    main()

