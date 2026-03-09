from common import SAMPLE_DOCS, build_chunks_sentence, fake_tool_call, format_hits, grounded_answer, retrieve


def main() -> None:
    question = "Hvad skal vi logge i en LLM-losning, og hvad er 100 USD i DKK?"
    chunks = build_chunks_sentence(SAMPLE_DOCS, group_size=2)

    retrieval_query = "logging prompts model settings latency in LLM solution"
    hits = retrieve(retrieval_query, chunks, top_k=2, metadata_filter={"category": "operations"})
    rag_answer = grounded_answer(retrieval_query, hits, prompt_variant="strict")
    fx = fake_tool_call("fx_rate", base="USD", quote="DKK")

    print("=== Demo: Chain der kombinerer retrieval og et eksternt tool-kald ===")
    print(f"Brugersporgsmal: {question}\n")
    print("1) Retrieval-trin")
    print(format_hits(hits))
    print("\n2) RAG-svar (grounded)")
    print(rag_answer)
    print("\n3) Tool-kald (fx rate)")
    print(fx)
    print("\n4) Sammensat svar")
    print(
        "Log prompts, model settings og latency. Definer fallback ved tool failures/timeouts. "
        f"100 USD er ca. {100 * fx['rate']:.2f} DKK (mock rate)."
    )


if __name__ == "__main__":
    main()
