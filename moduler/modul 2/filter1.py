from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}
ALLOWED_FOLDER = "knowledge_base"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def find_documents(base_path: str | Path, required_folder: str | None = ALLOWED_FOLDER) -> list[Path]:
    base = Path(base_path)
    if not base.exists():
        return []

    documents: list[Path] = []

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        if required_folder and required_folder not in file_path.parts:
            continue

        documents.append(file_path)

    return sorted(documents)


def main() -> None:
    docs = find_documents(DEFAULT_DATA_DIR)

    if not docs:
        fallback_docs = find_documents(DEFAULT_DATA_DIR, required_folder=None)

        if fallback_docs:
            print(
                f"Ingen dokumenter fundet i mappen {ALLOWED_FOLDER!r} under {DEFAULT_DATA_DIR}."
            )
            print("Viser i stedet alle filer med gyldige filendelser:")
            for doc in fallback_docs:
                print(doc)
            return

        print(f"Ingen dokumenter fundet i {DEFAULT_DATA_DIR}")
        print(
            f"Forventede filtyper: {sorted(ALLOWED_EXTENSIONS)}"
            f" og mappe med navn: {ALLOWED_FOLDER!r}"
        )
        return

    for doc in docs:
        print(doc)


if __name__ == "__main__":
    main()
