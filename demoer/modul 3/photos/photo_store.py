"""Fælles Qdrant-opslag for foto-demoet, så app, tagging og agent bruger samme logik."""

from __future__ import annotations

import re
from dataclasses import dataclass

from qdrant_client import QdrantClient, models as qdrant_models

from image_embeddings import get_image_embed_model

COLLECTION_NAME = "modul3_photos_demo"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def collection_for_model(model_name: str | None = None) -> str:
    """Hver embedding-model har sin egen collection, fordi vektorerne ikke kan blandes."""
    model_name = model_name or get_image_embed_model()
    if model_name == DEFAULT_MODEL:
        return COLLECTION_NAME
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"{COLLECTION_NAME}__{slug}"


@dataclass
class StoredPhoto:
    id: str
    filename: str
    vector: list[float]
    tag: str | None
    tag_score: float | None


def search(
    client: QdrantClient,
    vector: list[float],
    *,
    collection: str = COLLECTION_NAME,
    limit: int = 5,
    tag: str | None = None,
    exclude_filename: str | None = None,
) -> list[dict[str, str | float | None]]:
    """Find de nærmeste billeder. Et tag filtrerer på payload før vektorsøgningen."""
    must = []
    must_not = []
    if tag:
        must.append(qdrant_models.FieldCondition(key="tag", match=qdrant_models.MatchValue(value=tag)))
    if exclude_filename:
        must_not.append(
            qdrant_models.FieldCondition(key="filename", match=qdrant_models.MatchValue(value=exclude_filename))
        )
    query_filter = qdrant_models.Filter(must=must, must_not=must_not) if (must or must_not) else None

    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )

    results: list[dict[str, str | float | None]] = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append(
            {
                "filename": str(payload.get("filename", "ukendt.jpg")),
                "score": float(hit.score),
                "tag": payload.get("tag"),
            }
        )
    return results


def fetch_all(client: QdrantClient, *, collection: str = COLLECTION_NAME, with_vectors: bool = True) -> list[StoredPhoto]:
    """Hent alle punkter i collectionen side for side."""
    photos: list[StoredPhoto] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        for point in points:
            payload = point.payload or {}
            photos.append(
                StoredPhoto(
                    id=str(point.id),
                    filename=str(payload.get("filename", "")),
                    vector=list(point.vector) if with_vectors and point.vector is not None else [],
                    tag=payload.get("tag"),
                    tag_score=payload.get("tag_score"),
                )
            )
        if offset is None:
            break
    photos.sort(key=lambda photo: photo.filename)
    return photos


def get_vector(client: QdrantClient, filename: str, *, collection: str = COLLECTION_NAME) -> list[float] | None:
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="filename", match=qdrant_models.MatchValue(value=filename))]
        ),
        limit=1,
        with_vectors=True,
    )
    if not points or points[0].vector is None:
        return None
    return list(points[0].vector)


def collection_exists(client: QdrantClient, collection: str) -> bool:
    return client.collection_exists(collection)
