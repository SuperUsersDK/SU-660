import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BOLD, GREEN, RED, YELLOW, colorize, print_label, print_section

documents = [
    {
        "text": "RAG bruges til at kombinere retrieval og generation.",
        "metadata": {"language": "da", "status": "approved", "department": "ai"},
    },
    {
        "text": "Draft: foreloebige noter om embeddings.",
        "metadata": {"language": "da", "status": "draft", "department": "ai"},
    },
    {
        "text": "Support process documentation in English.",
        "metadata": {"language": "en", "status": "approved", "department": "support"},
    },
    {
        "metadata": {"language": "da", "status": "approved", "department": "ai"},
    },
]


def filter_documents(docs: list[dict] | None) -> list[dict]:
    if not docs:
        return []

    filtered: list[dict] = []

    for doc in docs:
        if not isinstance(doc, dict):
            continue

        metadata = doc.get("metadata")
        if not isinstance(metadata, dict):
            continue

        if metadata.get("language") != "da":
            continue

        if metadata.get("status") != "approved":
            continue

        if metadata.get("department") != "ai":
            continue

        filtered.append(doc)

    return filtered


def main() -> None:
    filtered_docs = filter_documents(documents)
    print_section("Filter 3 - Metadata")

    if not filtered_docs:
        print(colorize("Ingen dokumenter matchede metadata-filteret.", BOLD, YELLOW))
        return

    print_label("MATCHENDE DOKUMENTER", GREEN)
    for doc in filtered_docs:
        text = doc.get("text")
        if text:
            print(colorize(f"- {text}", GREEN))
        else:
            print(colorize("- <mangler text>", RED))


if __name__ == "__main__":
    main()
