from __future__ import annotations

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

DATA_DIR = Path(__file__).resolve().parent / "data"
COLLECTION_NAME = "modul3_photos_demo"
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


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def upsert_batch(
    client: QdrantClient,
    batch: list[qdrant_models.PointStruct],
    *,
    batch_no: int,
    total_ingested: int,
) -> None:
    if not batch:
        return
    client.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(
        colorize(
            f"BATCH {batch_no}: upserted {len(batch)} billeder "
            f"(samlet {total_ingested})",
            GREEN,
        )
    )


def main() -> None:
    load_env()

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

    print_section("Photo Ingest")
    print_label("KONFIGURATION", CYAN)
    print(colorize(f"Data mappe: {DATA_DIR}", DIM))
    print(colorize(f"Qdrant URL: {get_qdrant_url()}", DIM))
    print(colorize(f"Qdrant API key: {'ja' if qdrant_api_key else 'nej'}", DIM))
    print(colorize(f"Collection: {COLLECTION_NAME}", DIM))
    print(colorize(f"Image model: {get_image_embed_model()}", DIM))
    print(colorize(f"Batch size:  {BATCH_SIZE}", DIM))

    for path in image_paths:
        try:
            with Image.open(path) as image:
                vector = image_to_vector(image)
                width, height = image.size
        except UnidentifiedImageError:
            print(colorize(f"SKIP {path.name}: ikke et gyldigt billede", BOLD, RED))
            continue

        if vector_size is None:
            vector_size = len(vector)
            ensure_collection(client, vector_size)

        points_batch.append(
            qdrant_models.PointStruct(
                id=file_id(path),
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
                points_batch,
                batch_no=batch_no,
                total_ingested=total_ingested,
            )
            points_batch = []

    if total_ingested == 0 or vector_size is None:
        raise RuntimeError("Ingen billeder kunne embeddes.")

    if points_batch:
        batch_no += 1
        upsert_batch(
            client,
            points_batch,
            batch_no=batch_no,
            total_ingested=total_ingested,
        )

    print_label("RESULTAT", YELLOW)
    print(colorize(f"Ingested billeder: {total_ingested}", GREEN))
    print(colorize(f"Antal batches:    {batch_no}", DIM))
    print(colorize(f"Vector størrelse:  {vector_size}", DIM))


if __name__ == "__main__":
    main()
