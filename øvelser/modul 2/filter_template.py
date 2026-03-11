from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import GREEN, YELLOW, colorize, print_label, print_section

DATA_DIR = Path(__file__).resolve().parent / "test_dokumenter"


def load_documents(data_dir: Path) -> list[dict]:
    documents = []

    for path in sorted(data_dir.iterdir()):
        # Her skal der filtreres på tilladte filtypert, og kun filer der matcher skal indlæses.


        documents.append(
            {
                "path": path,
                "text": path.read_text(encoding="utf-8").strip(),
            }
        )

    return documents


def filter_documents(documents: list[dict]) -> tuple[list[dict], list[tuple[Path, str]]]:
    kept_documents: list[dict] = []
    removed_documents: list[tuple[Path, str]] = []

    for document in documents:
        path = document["path"]
        text = document["text"]

        # HER skal deltagerne skrive deres filter.
        # Eksempler paa regler:
        # - frasorter korte tekster
        # - frasorter navigation eller cookie-stoej
        # - frasorter kladder og loese noter
        # - frasorter dubletter
        # Naar et dokument ikke bestaar filteret, saa tilfoej det til
        # removed_documents med en kort begrundelse.

        kept_documents.append({"path": path, "text": text})

    return kept_documents, removed_documents


def main() -> None:
    documents = load_documents(DATA_DIR)
    kept_documents, removed_documents = filter_documents(documents)

    print_section("Template - Dokumentfilter")
    print_label("BEHOLDTE DOKUMENTER", GREEN)
    for document in kept_documents:
        print(colorize(f"- {document['path'].name}", GREEN))

    print_label("FRASORTEREDE DOKUMENTER", YELLOW)
    for path, reason in removed_documents:
        print(colorize(f"- {path.name}: {reason}", YELLOW))


if __name__ == "__main__":
    main()
