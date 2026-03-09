from common import BENCHMARK_QA, SAMPLE_DOCS, append_jsonl, benchmark_retrieval, build_chunks_sentence, print_table


def main() -> None:
    chunks = build_chunks_sentence(SAMPLE_DOCS, group_size=2)
    rows = benchmark_retrieval(chunks, top_k=2, metadata_filter_by_category=False)

    print("=== Demo: Simpel eval-loop i Python (manuelle ratings + basale malinger) ===")
    table_rows = []
    for idx, row in enumerate(rows, start=1):
        manual_rating = 4 if row["keyword_score"] >= 0.66 else 2
        result = {
            "case": idx,
            "target_category": row["target_category"],
            "top_hit_category": row["top_hit_category"],
            "keyword_score": row["keyword_score"],
            "manual_rating_1_5": manual_rating,
        }
        append_jsonl("demo/output/eval_results.jsonl", result)
        table_rows.append(result)

    print_table(
        table_rows,
        ["case", "target_category", "top_hit_category", "keyword_score", "manual_rating_1_5"],
    )

    print("\nEval-cases:")
    for i, qa in enumerate(BENCHMARK_QA, start=1):
        print(f"{i}. {qa['question']}")
    print("\nSkrevet til: demo/output/eval_results.jsonl")


if __name__ == "__main__":
    main()

