from common import SAMPLE_DOCS, build_chunks_sentence, format_hits, grounded_answer, retrieve


def answer_question(question: str) -> None:
    chunks = build_chunks_sentence(SAMPLE_DOCS, group_size=2)
    hits = retrieve(question, chunks, top_k=3)
    answer = grounded_answer(question, hits, prompt_variant="strict")
    print(f"\nSporgsmal: {question}")
    print("Top retrieval hits:")
    print(format_hits(hits))
    print("Svar:")
    print(answer)


def main() -> None:
    print("=== Demo: Simpel RAG-chat i Python ===")
    answer_question("Hvorfor er metadata-filters nyttige i RAG?")
    answer_question("Hvilke ting skal vi have logging pa i en LLM-losning?")


if __name__ == "__main__":
    main()
