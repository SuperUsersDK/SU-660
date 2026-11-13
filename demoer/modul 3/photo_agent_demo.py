"""Agent med billed-tools: CLIP-søgning i Qdrant + en vision-model, der beskriver billederne.

Kræver, at foto-demoet er ingestet: python "demoer/modul 3/photos/ingest_photos.py"
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from PIL import Image, ImageOps
from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parents[2]
PHOTOS_DIR = Path(__file__).resolve().parent / "photos"
for path in (ROOT_DIR, PHOTOS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, MAGENTA, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue
from image_embeddings import get_image_embed_model, text_to_vector
import photo_store
from tagging import get_label_set_name, get_labels

DATA_DIR = PHOTOS_DIR / "data"


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def get_chat_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-6-luna")


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
    )


def collection() -> str:
    return photo_store.collection_for_model(get_image_embed_model())


def print_status(label: str, message: str, color: str = CYAN) -> None:
    print(colorize(f"{label:<12}", BOLD, color) + " " + colorize(message, DIM))


def image_data_url(path: Path, max_side: int = 512) -> str:
    """Skaler billedet ned før det sendes til modellen; det sparer tokens og tid."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@tool
def list_photo_tags() -> str:
    """List the zero-shot tags in the photo collection and how many photos have each tag."""
    print_status("TOOL tags", "Tæller tags i Qdrant", YELLOW)
    counts: dict[str, int] = {}
    for photo in photo_store.fetch_all(get_client(), collection=collection(), with_vectors=False):
        counts[photo.tag or "uden tag"] = counts.get(photo.tag or "uden tag", 0) + 1
    return "[tags] " + ", ".join(f"{tag}={count}" for tag, count in sorted(counts.items(), key=lambda item: -item[1]))


@tool
def search_photos(description: str, tag: str = "", limit: int = 5) -> str:
    """Search the photo library with CLIP. Write the description in English for best results.

    Optionally filter on one tag; call list_photo_tags to see which tags exist.
    Returns filenames with similarity scores (text-to-image scores are typically 0.2-0.35).
    """
    tag = tag.strip()
    labels = get_labels()
    if tag and tag not in labels:
        return f"[photos] Ukendt tag '{tag}'. Gyldige tags: {', '.join(labels)}"
    print_status("TOOL search", f"CLIP-søgning: '{description}'" + (f" (tag={tag})" if tag else ""), MAGENTA)
    hits = photo_store.search(
        get_client(),
        text_to_vector(description),
        collection=collection(),
        limit=max(1, min(int(limit), 10)),
        tag=tag or None,
    )
    if not hits:
        return "[photos] Ingen billeder fundet."
    return "\n".join(
        f"[photos] filename={hit['filename']} score={hit['score']:.3f} tag={hit['tag']}"
        for hit in hits
    )


@tool
def describe_photo(filename: str) -> str:
    """Look at one photo from the library with a vision model and describe what it shows."""
    path = DATA_DIR / Path(filename).name
    if not path.is_file():
        return f"[vision] Billedet {filename} findes ikke."
    print_status("TOOL vision", f"Vision-model kigger på {path.name}", BLUE)
    llm = ChatOpenAI(model=get_chat_model(), temperature=0, reasoning_effort="none")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Beskriv kort på dansk, hvad billedet viser (2-3 sætninger). "
                    "Nævn motiv, lys og komposition. Gæt ikke på steder eller navne."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_data_url(path)}},
        ]
    )
    return f"[vision {path.name}] {llm.invoke([message]).content}"


# Spørgsmålet passer til det label-sæt, billederne er tagget med (PHOTO_LABEL_SET).
QUESTIONS = {
    "eksempler": (
        "Find de tre bedste billeder af fugle og de tre bedste af insekter, "
        "beskriv kort hvad der ses på hvert af dem, og anbefal ét af dem som forsidebillede "
        "til en plakat om dyreliv. Svar på dansk."
    ),
    "dykning": (
        "Find de tre bedste billeder af skibsvrag, hvor der også er en dykker med, "
        "beskriv hvad der ses på hvert af dem, og anbefal ét af dem som forsidebillede "
        "til et dykkerkursus. Svar på dansk."
    ),
}


def build_agent() -> Any:
    llm = ChatOpenAI(model=get_chat_model(), temperature=0, reasoning_effort="none")
    return create_agent(
        llm,
        tools=[list_photo_tags, search_photos, describe_photo],
        system_prompt=(
            "Du er en fotoassistent for et billedarkiv. "
            "Brug search_photos til at finde kandidater (beskrivelser på engelsk virker bedst), "
            "og brug describe_photo til at se på billederne, før du udtaler dig om dem. "
            "Påstå aldrig noget om et billede, du ikke har fået beskrevet. "
            "Nævn altid filnavnene."
        ),
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        ).strip()
    return str(message)


def main() -> None:
    clear_screen()
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY mangler i repo-roden .env")
    if not get_client().collection_exists(collection()):
        raise RuntimeError("Foto-collectionen findes ikke. Kør først: python \"demoer/modul 3/photos/ingest_photos.py\"")

    agent = build_agent()
    question = QUESTIONS[get_label_set_name()]

    print_section("Foto-agent")
    print_label("WORKFLOW", CYAN)
    for step in ("Spørgsmål", "Agent", "CLIP-søgning i Qdrant", "Vision-model ser billederne", "Svar"):
        print(colorize(step, DIM))
        if step != "Svar":
            print(colorize("↓", DIM))

    print_label("TOOLS", BLUE)
    print(colorize("- list_photo_tags -> Qdrant payload (zero-shot tags)", BLUE))
    print(colorize("- search_photos   -> CLIP tekstvektor + Qdrant (kan filtrere på tag)", BLUE))
    print(colorize("- describe_photo  -> vision-model ser selve billedet", BLUE))
    prompt_to_continue("Tryk Enter for at se spørgsmålet...")

    print_section("Spørgsmål")
    print_label("SPØRGSMÅL", YELLOW)
    print(colorize(question, YELLOW))
    prompt_to_continue("Tryk Enter for at lade agenten bruge sine tools...")

    print_section("Agent Run")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    final_text = _message_text(messages[-1]) if messages else str(result)

    print_label("SVAR", GREEN)
    print(colorize(final_text, GREEN))


if __name__ == "__main__":
    main()
