from common import SAMPLE_DOCS, BENCHMARK_QA, build_chunks_sentence, grounded_answer, retrieve, score_keywords, print_table


def run_variant(name: str, prompt_variant: str, top_k: int) -> dict:
    chunks = build_chunks_sentence(SAMPLE_DOCS, group_size=2)
    scores = []
    for qa in BENCHMARK_QA:
        hits = retrieve(qa["question"], chunks, top_k=top_k)
        answer = grounded_answer(qa["question"], hits, prompt_variant=prompt_variant)
        scores.append(score_keywords(answer, qa["expected_keywords"]))
    return {
        "variant": name,
        "prompt_variant": prompt_variant,
        "top_k": top_k,
        "avg_keyword_score": round(sum(scores) / len(scores), 2),
    }


def main() -> None:
    print("=== Demo: Sammenlign to promptversioner eller to retrieval-opsaetninger ===")
    rows = [
        run_variant("A", "baseline", 2),
        run_variant("B", "strict", 2),
        run_variant("C", "strict", 3),
    ]
    print_table(rows, ["variant", "prompt_variant", "top_k", "avg_keyword_score"])
    print("\nBrug demoen til at vise A/B-testlogik: hold alt konstant og varier kun 1 parameter ad gangen.")


if __name__ == "__main__":
    main()

