from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BOLD, CYAN, GREEN, YELLOW, colorize, print_label, print_section

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
    print_section("Filter 1 - Find dokumenter")

    if not docs:
        fallback_docs = find_documents(DEFAULT_DATA_DIR, required_folder=None)

        if fallback_docs:
            print(colorize(
                f"Ingen dokumenter fundet i mappen {ALLOWED_FOLDER!r} under {DEFAULT_DATA_DIR}.",
                BOLD,
                YELLOW,
            ))
            print(colorize("Viser i stedet alle filer med gyldige filendelser:", YELLOW))
            print_label("DOKUMENTER", GREEN)
            for doc in fallback_docs:
                print(colorize(f"- {doc}", GREEN))
            return

        print(colorize(f"Ingen dokumenter fundet i {DEFAULT_DATA_DIR}", BOLD, YELLOW))
        print(colorize(
            f"Forventede filtyper: {sorted(ALLOWED_EXTENSIONS)} og mappe med navn: {ALLOWED_FOLDER!r}",
            YELLOW,
        ))
        return

    print_label("DOKUMENTER", GREEN)
    for doc in docs:
        print(colorize(f"- {doc}", GREEN))


if __name__ == "__main__":
    main()
