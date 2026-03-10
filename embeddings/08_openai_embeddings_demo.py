"""
Demo 08: Rigtigt embedding-API kald (OpenAI).

Kraever:
- OPENAI_API_KEY i repoets .env i roden
- `openai` package installeret (findes i root requirements.txt)

Viser:
- hvordan man kalder embedding endpoint
- dimension/latency output
- lille retrieval eksempel med rigtige embeddings
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - teaching fallback
    OpenAI = None  # type: ignore[assignment]

try:
    from embeddings.common import DOCS, env_openai_key, print_hits, time_call
except ModuleNotFoundError:
    from common import DOCS, env_openai_key, print_hits, time_call


_REPO_ROOT = Path(__file__).resolve().parents[1]


load_dotenv(_REPO_ROOT / ".env", override=True)


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main() -> None:
    print("=== Eksempel 8: OpenAI embeddings demo ===")
    if OpenAI is None:
        print("Manglende dependency: install root requirements (openai).")
        return

    api_key = env_openai_key()
    if not api_key:
        print("OPENAI_API_KEY mangler. Saet den i .env i projektets rod.")
        return

    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    query = "What must a termination notice contain?"
    texts = [query] + [d.text for d in DOCS]
    client = OpenAI(api_key=api_key)

    response, ms = time_call(lambda: client.embeddings.create(model=model, input=texts))
    vectors = [item.embedding for item in response.data]
    qv = vectors[0]

    hits = []
    for doc, dv in zip(DOCS, vectors[1:]):
        score = cosine(qv, dv)
        hits.append((doc, score))
    hits.sort(key=lambda x: x[1], reverse=True)

    print(f"Model: {model}")
    print(f"Latency: {ms:.1f} ms")
    print(f"Embedding dimension: {len(qv)}")
    print()
    for i, (doc, score) in enumerate(hits[:4], start=1):
        print(f"{i}. score={score:.4f} {doc.doc_id} type={doc.metadata.get('type')}")
        print(f"   {' '.join(doc.text.split())[:100]}{'...' if len(doc.text) > 100 else ''}")
    print()
    print("Undervisningspoint: Embed query + docs med samme model og sammenlign med cosine.")


if __name__ == "__main__":
    main()
