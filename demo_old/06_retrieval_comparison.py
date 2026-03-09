from common import SAMPLE_DOCS, benchmark_retrieval, build_chunks_fixed, build_chunks_sentence, print_table


def summarize(label: str, rows: list[dict]) -> dict:
    avg_keyword = round(sum(r["keyword_score"] for r in rows) / len(rows), 2)
    avg_top_score = round(sum(r["top_score"] for r in rows) / len(rows), 3)
    correct_top_category = sum(1 for r in rows if r["top_hit_category"] == r["target_category"])
    return {
        "config": label,
        "avg_keyword_score": avg_keyword,
        "avg_top_score": avg_top_score,
        "top_category_hits": f"{correct_top_category}/{len(rows)}",
    }


def main() -> None:
    fixed = build_chunks_fixed(SAMPLE_DOCS, chunk_words=16, overlap_words=4)
    sentence = build_chunks_sentence(SAMPLE_DOCS, group_size=2)

    fixed_rows = benchmark_retrieval(fixed, top_k=2, metadata_filter_by_category=False)
    sentence_rows = benchmark_retrieval(sentence, top_k=2, metadata_filter_by_category=False)
    sentence_filtered_rows = benchmark_retrieval(sentence, top_k=2, metadata_filter_by_category=True)

    summary_rows = [
        summarize("fixed_16_4", fixed_rows),
        summarize("sentence_g2", sentence_rows),
        summarize("sentence_g2+category_filter", sentence_filtered_rows),
    ]

    print("=== Demo: Sammenligning af retrieval-resultater ved forskellige indstillinger ===")
    print_table(summary_rows, ["config", "avg_keyword_score", "avg_top_score", "top_category_hits"])


if __name__ == "__main__":
    main()

