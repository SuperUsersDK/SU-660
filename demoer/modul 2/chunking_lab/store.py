"""Qdrant-lagring for chunk-laboratoriet.

Alle kørsler deler én collection pr. embedding-model. Hver kombination af tekst,
strategi og parametre har sit eget `run_id` i payload, og søgningen filtrerer på det.
Så kan samme spørgsmål søges mod hver strategi for sig, og en kørsel genbruges,
hvis den allerede er embeddet.
"""

from __future__ import annotations

import os
import re
import uuid

from qdrant_client import QdrantClient, models

from chunkers import Chunk


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
    )


def collection_name(model: str) -> str:
    return "chunk_lab__" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def ensure_collection(client: QdrantClient, name: str, dims: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(name, vectors_config=models.VectorParams(size=dims, distance=models.Distance.COSINE))
    client.create_payload_index(name, field_name="run_id", field_schema=models.PayloadSchemaType.KEYWORD)


def _run_filter(run_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="run_id", match=models.MatchValue(value=run_id))])


def stored_vectors(client: QdrantClient, name: str, run_id: str, expected: int) -> list[list[float]] | None:
    """Vektorerne for en tidligere kørsel i chunk-rækkefølge, eller None hvis de mangler."""
    if not client.collection_exists(name):
        return None
    points, _ = client.scroll(name, scroll_filter=_run_filter(run_id), limit=max(expected, 1) + 1, with_vectors=True)
    if len(points) != expected:
        return None
    points.sort(key=lambda point: point.payload["index"])
    return [list(point.vector) for point in points]


def save_run(
    client: QdrantClient,
    name: str,
    run_id: str,
    *,
    chunks: list[Chunk],
    vectors: list[list[float]],
    strategy: str,
    source: str,
) -> None:
    client.delete(name, points_selector=models.FilterSelector(filter=_run_filter(run_id)))
    client.upsert(
        name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}/{index}")),
                vector=vector,
                payload={
                    "run_id": run_id,
                    "index": index,
                    "strategy": strategy,
                    "source": source,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ],
    )


def search(client: QdrantClient, name: str, run_id: str, vector: list[float], limit: int) -> list[tuple[int, float]]:
    """De nærmeste chunks i én kørsel som (chunk-indeks, cosine-score)."""
    response = client.query_points(name, query=vector, query_filter=_run_filter(run_id), limit=limit, with_payload=["index"])
    return [(point.payload["index"], float(point.score)) for point in response.points]
