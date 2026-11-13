"""Tag billederne igen med det label-sæt, PHOTO_LABEL_SET vælger (se tagging.py).

Embeddings genbruges fra Qdrant, så det tager sekunder i stedet for minutter.
Ret labels eller prompts i tagging.py, og kør scriptet igen for at se effekten.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import CYAN, DIM, GREEN, colorize, print_label, print_section
from image_embeddings import get_image_embed_model
from photo_store import collection_for_model
from tagging import get_label_set_name, get_labels, tag_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model hvis collection skal tagges. Standard er IMAGE_EMBED_MODEL.")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env", override=True)
    model_name = args.model or get_image_embed_model()
    collection = collection_for_model(model_name)
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
    )
    if not client.collection_exists(collection):
        raise RuntimeError(f"Collection {collection} findes ikke. Kør ingest_photos.py først.")

    print_section("Zero-shot tagging")
    print_label(f"LABELS ({get_label_set_name()})", CYAN)
    for tag, prompt in get_labels().items():
        print(colorize(f"{tag:<12} \"{prompt}\"", DIM))

    counts = tag_collection(client, collection=collection, model_name=model_name)

    print_label("RESULTAT", GREEN)
    for tag, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(colorize(f"{tag:<12} {count}", GREEN))


if __name__ == "__main__":
    main()
