"""Zero-shot tagging: CLIP sammenligner hvert billede med en liste tekst-labels.

Der trænes ingenting. Hver label skrives som en lille engelsk sætning
("a photo of ..."), fordi CLIP er trænet på billedtekster og rammer bedre med
hele sætninger end med enkeltord. Ret LABELS og kør `python tag_photos.py` for
at tagge igen uden at embedde billederne forfra.
"""

from __future__ import annotations

import math

from qdrant_client import QdrantClient, models as qdrant_models

from image_embeddings import texts_to_vectors
from photo_store import COLLECTION_NAME, fetch_all

# Dansk tag -> engelsk prompt, som CLIP matcher billedet mod.
LABELS: dict[str, str] = {
    "vrag": "a photo of a sunken shipwreck underwater",
    "fisk": "a photo of a fish",
    "koralrev": "a photo of a coral reef underwater",
    "dykker": "a photo of a scuba diver wearing a mask and an air tank",
    "smådyr": "a close-up photo of a small sea creature such as a starfish, sea slug or worm",
    "båd": "a photo of a boat on the water",
    "mennesker": "a photo of people above water, on a boat or on land",
    "solnedgang": "a photo of the sunset or the sea surface and the sky",
}

# CLIP ganger cosine-ligheden med en lært temperatur (logit scale ≈ 100) før softmax.
# Derfor bliver små forskelle i cosine til tydelige sandsynligheder.
LOGIT_SCALE = 100.0


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def classify(
    image_vectors: list[list[float]],
    label_vectors: list[list[float]],
    tags: list[str],
) -> list[tuple[str, float]]:
    """Returnér (tag, sandsynlighed) for hvert billede ud fra cosine mod hver label."""
    results: list[tuple[str, float]] = []
    for image_vector in image_vectors:
        cosines = [sum(a * b for a, b in zip(image_vector, label_vector)) for label_vector in label_vectors]
        probabilities = softmax([LOGIT_SCALE * cosine for cosine in cosines])
        best = max(range(len(tags)), key=probabilities.__getitem__)
        results.append((tags[best], probabilities[best]))
    return results


def tag_collection(
    client: QdrantClient,
    *,
    collection: str = COLLECTION_NAME,
    model_name: str | None = None,
    labels: dict[str, str] | None = None,
) -> dict[str, int]:
    """Tag alle billeder i collectionen og gem tag + sikkerhed i payload."""
    labels = labels or LABELS
    tags = list(labels)
    label_vectors = texts_to_vectors([labels[tag] for tag in tags], model_name)

    photos = fetch_all(client, collection=collection)
    predictions = classify([photo.vector for photo in photos], label_vectors, tags)

    operations = [
        qdrant_models.SetPayloadOperation(
            set_payload=qdrant_models.SetPayload(
                payload={"tag": tag, "tag_score": round(score, 4)},
                points=[photo.id],
            )
        )
        for photo, (tag, score) in zip(photos, predictions)
    ]
    for start in range(0, len(operations), 200):
        client.batch_update_points(collection_name=collection, update_operations=operations[start:start + 200])

    # Et keyword-index gør filtrering på tag hurtig, også i store collections.
    client.create_payload_index(
        collection_name=collection,
        field_name="tag",
        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
    )

    counts = {tag: 0 for tag in tags}
    for tag, _ in predictions:
        counts[tag] += 1
    return counts
