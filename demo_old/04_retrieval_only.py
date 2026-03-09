from common import SAMPLE_DOCS, build_chunks_sentence, format_hits, retrieve


def main() -> None:
    chunks = build_chunks_sentence(SAMPLE_DOCS, group_size=2)
    query = "Hvilke pauser er der pa kurset, og hvornar er de?"
    hits = retrieve(query, chunks, top_k=4)

    print("=== Demo: Forste retrieval uden generering (kun sogning/top-k) ===")
    print(f"Query: {query}\n")
    print(format_hits(hits))

    print("\nBemark: Ingen generering her, kun retrieval-resultater som grundlag for senere RAG.")


if __name__ == "__main__":
    main()

