from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, CYAN, DIM, GREEN, RED, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "RAG kombinerer retrieval og generation for at give mere grounded svar.",
        "metadata": {"category": "rag", "language": "da"},
    },
    {
        "id": "doc_2",
        "text": "Embeddings are used to match a user query with relevant chunks.",
        "metadata": {"category": "rag", "language": "en"},
    },
    {
        "id": "doc_3",
        "text": "Chunking opdeler dokumenter i mindre bidder, som kan bruges i retrieval.",
        "metadata": {"category": "rag", "language": "da"},
    },
    {
        "id": "doc_4",
        "text": "FAQ: How to reset your password.",
        "metadata": {"category": "support", "language": "en"},
    },
    {
        "id": "doc_5",
        "text": "Supportprocessen beskriver eskalering ved driftsfejl.",
        "metadata": {"category": "support", "language": "da"},
    },
    {
        "id": "doc_6",
        "text": "RAG kombinerar sokning och generering for att ge mer precisa svar.",
        "metadata": {"category": "rag", "language": "sv"},
    },
]


def filter_documents(
    documents: list[dict],
    *,
    category: str,
    language: str,
) -> list[dict]:
    filtered = []

    for document in documents:
        metadata = document.get("metadata", {})
        if metadata.get("category") != category:
            continue
        if metadata.get("language") != language:
            continue
        filtered.append(document)

    return filtered


def print_documents(documents: list[dict], matched: bool) -> None:
    for document in documents:
        metadata = document["metadata"]
        status_color = GREEN if matched else RED
        print(colorize(f"- {document['id']}", status_color))
        print(
            colorize(
                f"  metadata: category={metadata['category']}, language={metadata['language']}",
                DIM,
            )
        )
        print(f"  text: {document['text']}")


def main() -> None:
    clear_screen()
    print_section("Metadata Filter Demo")
    print_label("POINTE", CYAN)
    print(
        colorize(
            "Metadata kan bruges til at begraense retrieval, saa man kun soeger i de dokumenter der er relevante.",
            DIM,
        )
    )
    print(
        colorize(
            "Her filtrerer vi til kun category='rag' og language='da'.",
            DIM,
        )
    )

    print_section("Alle Dokumenter")
    print_label("INPUT", BLUE)
    print_documents(DOCUMENTS, matched=False)

    prompt_to_continue("Tryk Enter for at se det filtrerede udsnit...")

    filtered_documents = filter_documents(DOCUMENTS, category="rag", language="da")

    print_section("Filtreret Resultat")
    print_label("KUN RAG + LANGUAGE=DA", GREEN)
    print_documents(filtered_documents, matched=True)

    print_label("BEMAERK", YELLOW)
    print(
        colorize(
            "Nu vil retrieval kun kigge i danske RAG-dokumenter og ignorere engelsk, svensk og support-data.",
            YELLOW,
        )
    )


if __name__ == "__main__":
    main()
