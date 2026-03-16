"""
Demo 07: Parameter-grid evaluering (chunk size + overlap).

Viser:
- hvordan man sammenligner chunking-parametre systematisk i et lille grid
- hvorfor tuning boer ske med maalinger frem for mavefornemmelse

Saadan tolkes output:
- `n_chunks` viser storage/ingest-kompleksitet (flere chunks = mere cost/overhead)
- `avg_keyword_score` er en simpel retrieval-coverage indikator i benchmark-sporgsmaal
- `avg_top_score` er en grov signalmaaling for hvor godt top-hit matcher query lexicalt

Vigtig pointe:
- tallene er didaktiske mock-metrikker; brug samme metode til at sammenligne configs,
  ikke som absolut kvalitetsdom.
"""

from __future__ import annotations

try:
    from chunking.common import QUERY_SET, SAMPLE_DOCS, chunk_fixed_words, keyword_score, retrieve
except ModuleNotFoundError:
    from common import QUERY_SET, SAMPLE_DOCS, chunk_fixed_words, keyword_score, retrieve


def evaluate(chunk_words: int, overlap_words: int) -> tuple[float, float, int]:
    chunks = chunk_fixed_words(SAMPLE_DOCS, chunk_words=chunk_words, overlap_words=overlap_words)
    scores: list[float] = []
    top_scores: list[float] = []
    for case in QUERY_SET:
        hits = retrieve(case["question"], chunks, top_k=3)
        scores.append(keyword_score(case["question"], hits, case["expected_keywords"]))
        top_scores.append(hits[0][1] if hits else 0.0)
    avg_keyword = round(sum(scores) / len(scores), 2)
    avg_top_score = round(sum(top_scores) / len(top_scores), 3)
    return avg_keyword, avg_top_score, len(chunks)


def main() -> None:
    print("=== Eksempel 7: Parameter-grid (chunk size + overlap) ===")
    print("Maal: vis at chunking boer tunes med maalinger og ikke kun intuition.\n")
    print("chunk_words | overlap_words | n_chunks | avg_keyword_score | avg_top_score")
    print("-----------+---------------+----------+-------------------+-------------")
    for chunk_words, overlap_words in [
        (8, 0),
        (8, 2),
        (12, 0),
        (12, 3),
        (18, 0),
        (18, 4),
    ]:
        avg_kw, avg_top, n_chunks = evaluate(chunk_words, overlap_words)
        print(
            f"{chunk_words:<10} | {overlap_words:<13} | {n_chunks:<8} | "
            f"{avg_kw:<17} | {avg_top:<11}"
        )


if __name__ == "__main__":
    main()
