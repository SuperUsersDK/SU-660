"""
Demo 02: Similarity metrics og normalisering.

Viser:
- forskellen mellem dot product og cosine
- hvorfor L2-normalisering ofte bruges i retrieval pipelines
"""

from __future__ import annotations

try:
    from embeddings.common import cosine_dense, dense_embedding, dot_dense, l2_normalize
except ModuleNotFoundError:
    from common import cosine_dense, dense_embedding, dot_dense, l2_normalize


def scaled(vec: list[float], factor: float) -> list[float]:
    return [v * factor for v in vec]


def main() -> None:
    q = dense_embedding("refund request timeline", normalize=False)
    d_same_direction = scaled(q, 5.0)
    d_related = dense_embedding("customer reimbursement period", normalize=False)

    print("=== Eksempel 2: Similarity metrics ===")
    print("Uden normalisering:")
    print("dot(q, same_direction*5):", round(dot_dense(q, d_same_direction), 4))
    print("dot(q, related):         ", round(dot_dense(q, d_related), 4))
    print("cos(q, same_direction*5):", round(cosine_dense(q, d_same_direction), 4))
    print("cos(q, related):         ", round(cosine_dense(q, d_related), 4))
    print()

    qn = l2_normalize(q)
    sn = l2_normalize(d_same_direction)
    rn = l2_normalize(d_related)
    print("Med L2-normalisering:")
    print("dot(qn, sn)  ~= cosine:", round(dot_dense(qn, sn), 4))
    print("dot(qn, rn)  ~= cosine:", round(dot_dense(qn, rn), 4))
    print("cos(qn, rn):           ", round(cosine_dense(qn, rn), 4))


if __name__ == "__main__":
    main()

