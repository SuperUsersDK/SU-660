"""
Demo 07: Modelvalg og tradeoffs (didaktisk tabel).

Viser:
- dimension, latency, cost og retrieval quality som designtradeoff
- hvorfor der ikke findes "bedste embedding model" uden kontekst
"""

from __future__ import annotations

try:
    from embeddings.common import fake_cost_usd, fake_latency_ms
except ModuleNotFoundError:
    from common import fake_cost_usd, fake_latency_ms


MODELS = [
    {"name": "small-general", "dim": 768, "quality": 0.72, "price_1k": 0.00002},
    {"name": "medium-general", "dim": 1536, "quality": 0.79, "price_1k": 0.00006},
    {"name": "large-general", "dim": 3072, "quality": 0.84, "price_1k": 0.00013},
]


def main() -> None:
    print("=== Eksempel 7: Embedding model tradeoffs ===")
    print("Antagelse: 12,000 chunks, ~220 tokens/chunk")
    print()
    print("model           | dim  | quality* | ingest_cost_usd | topk_latency_ms | topk+rereank_ms")
    print("----------------+------+----------+-----------------+----------------+---------------")
    for m in MODELS:
        ingest_cost = fake_cost_usd(num_chunks=12000, avg_tokens=220, price_per_1k_tokens=m["price_1k"])
        lat = fake_latency_ms(dim=m["dim"], top_k=8, rerank=False)
        lat_rr = fake_latency_ms(dim=m["dim"], top_k=8, rerank=True)
        print(
            f"{m['name']:<15} | {m['dim']:<4} | {m['quality']:<8.2f} | "
            f"{ingest_cost:<15.4f} | {lat:<14.1f} | {lat_rr:<13.1f}"
        )
    print()
    print("* quality er kun en didaktisk score. Evaluer paa egne queries/dokumenter.")


if __name__ == "__main__":
    main()

