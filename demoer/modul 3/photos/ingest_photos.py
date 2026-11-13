from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from qdrant_client import QdrantClient, models as qdrant_models

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BOLD, CYAN, DIM, GREEN, RED, YELLOW, colorize, print_label, print_section
from image_embeddings import get_image_embed_model, image_to_vector
from photo_store import collection_for_model
from tagging import tag_collection

DATA_DIR = Path(__file__).resolve().parent / "data"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}
BATCH_SIZE = 50


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def get_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def get_qdrant_api_key() -> str | None:
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    return api_key or None


def file_id(path: Path) -> str:
    digest = hashlib.sha1(path.read_bytes()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def load_image_paths() -> list[Path]:
    image_paths: list[Path] = []
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        image_paths.append(path)
    return image_paths


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def upsert_batch(
    client: QdrantClient,
    collection: str,
    batch: list[qdrant_models.PointStruct],
    *,
    batch_no: int,
    total_ingested: int,
) -> None:
    if not batch:
        return
    client.upsert(collection_name=collection, points=batch)
    print(
        colorize(
            f"BATCH {batch_no}: upserted {len(batch)} billeder "
            f"(samlet {total_ingested})",
            GREEN,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed billederne i data/ og gem dem i Qdrant.")
    parser.add_argument(
        "--model",
        help="CLIP-model fra Hugging Face. Standard er IMAGE_EMBED_MODEL. "
        "Hver model får sin egen collection, fx til /compare.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Behold collectionen og tilføj kun billeder, der ikke allerede er gemt "
        "(fx mens fetch_sample_photos.py stadig henter).",
    )
    return parser.parse_args()


def existing_points(client: QdrantClient, collection: str) -> dict[str, str]:
    """Punkt-id -> filnavn for alt, der allerede ligger i collectionen."""
    points_by_id: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=512, offset=offset, with_payload=["filename"], with_vectors=False
        )
        points_by_id.update({str(point.id): str((point.payload or {}).get("filename", "")) for point in points})
        if offset is None:
            return points_by_id


def main() -> None:
    args = parse_args()
    load_env()
    model_name = args.model or get_image_embed_model()
    collection = collection_for_model(model_name)

    image_paths = load_image_paths()
    if not image_paths:
        raise RuntimeError(f"Ingen .jpg/.jpeg billeder fundet i {DATA_DIR}")

    qdrant_api_key = get_qdrant_api_key()
    client = QdrantClient(
        url=get_qdrant_url(),
        api_key=qdrant_api_key,
    )

    points_batch: list[qdrant_models.PointStruct] = []
    vector_size: int | None = None
    total_ingested = 0
    batch_no = 0
    known: set[str] = set()
    removed = 0
    if args.incremental and client.collection_exists(collection):
        stored = existing_points(client, collection)
        # Fjern punkter for billeder, der er slettet fra mappen siden sidst.
        current = {path.name for path in image_paths}
        stale = [point_id for point_id, name in stored.items() if name not in current]
        if stale:
            client.delete(collection_name=collection, points_selector=qdrant_models.PointIdsList(points=stale))
            removed = len(stale)
        known = set(stored) - set(stale)
        # Collectionen findes allerede; den skal ikke slettes og genskabes.
        vector_size = client.get_collection(collection).config.params.vectors.size

    print_section("Photo Ingest")
    print_label("KONFIGURATION", CYAN)
    print(colorize(f"Data mappe: {DATA_DIR}", DIM))
    print(colorize(f"Qdrant URL: {get_qdrant_url()}", DIM))
    print(colorize(f"Qdrant API key: {'ja' if qdrant_api_key else 'nej'}", DIM))
    print(colorize(f"Collection: {collection}", DIM))
    print(colorize(f"Image model: {model_name}", DIM))
    print(colorize(f"Batch size:  {BATCH_SIZE}", DIM))
    if args.incremental:
        print(colorize(f"Inkrementel: {len(known)} billeder ligger allerede i Qdrant, {removed} fjernet", DIM))

    for path in image_paths:
        point_id = file_id(path)
        if point_id in known:
            continue
        try:
            with Image.open(path) as image:
                vector = image_to_vector(image, model_name)
                width, height = image.size
        except (UnidentifiedImageError, OSError):
            # Fx et billede, der stadig er ved at blive skrevet; det tages med næste gang.
            print(colorize(f"SKIP {path.name}: ikke et gyldigt billede", BOLD, RED))
            continue

        if vector_size is None:
            vector_size = len(vector)
            ensure_collection(client, collection, vector_size)

        points_batch.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "filename": path.name,
                    "relative_path": path.name,
                    "width": width,
                    "height": height,
                },
            )
        )
        print(colorize(f"INGEST {path.name}", GREEN))
        total_ingested += 1

        if len(points_batch) >= BATCH_SIZE:
            batch_no += 1
            upsert_batch(
                client,
                collection,
                points_batch,
                batch_no=batch_no,
                total_ingested=total_ingested,
            )
            points_batch = []

    if args.incremental and total_ingested == 0 and known:
        print(colorize("Ingen nye billeder.", YELLOW))
        if removed:
            tag_collection(client, collection=collection, model_name=model_name)
        return
    if total_ingested == 0 or vector_size is None:
        raise RuntimeError("Ingen billeder kunne embeddes.")

    if points_batch:
        batch_no += 1
        upsert_batch(
            client,
            collection,
            points_batch,
            batch_no=batch_no,
            total_ingested=total_ingested,
        )

    print_label("RESULTAT", YELLOW)
    print(colorize(f"Ingested billeder: {total_ingested}", GREEN))
    print(colorize(f"Antal batches:    {batch_no}", DIM))
    print(colorize(f"Vector størrelse:  {vector_size}", DIM))

    print_label("ZERO-SHOT TAGS", CYAN)
    counts = tag_collection(client, collection=collection, model_name=model_name)
    for tag, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(colorize(f"{tag:<12} {count}", DIM))


if __name__ == "__main__":
    main()
