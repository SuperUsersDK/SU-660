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

    if not filtered_docs:
        print("Ingen dokumenter matchede metadata-filteret.")
        return

    for doc in filtered_docs:
        print(doc.get("text", "<mangler text>"))


if __name__ == "__main__":
    main()
