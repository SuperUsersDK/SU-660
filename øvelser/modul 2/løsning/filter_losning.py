from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import GREEN, RED, YELLOW, colorize, print_label, print_section

DATA_DIR = Path(__file__).resolve().parent.parent / "test_dokumenter"
ALLOWED_EXTENSIONS = {".md", ".txt"}
MIN_LENGTH = 120
MIN_ALPHA_RATIO = 0.6
NEAR_DUPLICATE_THRESHOLD = 0.96

NOISE_PATTERNS = [
    r"home\s*\|",
    r"contact",
    r"pricing",
    r"login",
    r"privacy policy",
    r"cookie",
    r"accept all",
    r"manage settings",
    r"reset password",
    r"open ticket",
    r"track shipment",
]

DRAFT_PATTERNS = [
    r"\bdraft\b",
    r"\btodo\b",
    r"foreloebige noter",
    r"ikke klar til produktion",
    r"loese noter",
]


def load_documents(data_dir: Path) -> list[dict]:
    documents = []

    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        documents.append(
            {
                "path": path,
                "text": path.read_text(encoding="utf-8").strip(),
            }
        )

    return documents


def normalize_text(text: str) -> str:
    lowered = text.lower()
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed.strip()


def alpha_ratio(text: str) -> float:
    letters = sum(char.isalpha() for char in text)
    return letters / max(len(text), 1)


def looks_like_noise(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in NOISE_PATTERNS)


def looks_like_draft(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DRAFT_PATTERNS)


def is_near_duplicate(text: str, kept_texts: list[str]) -> bool:
    for kept_text in kept_texts:
        similarity = SequenceMatcher(None, text, kept_text).ratio()
        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            return True
    return False


def filter_documents(documents: list[dict]) -> tuple[list[dict], list[tuple[Path, str]]]:
    kept_documents: list[dict] = []
    removed_documents: list[tuple[Path, str]] = []
    kept_normalized_texts: list[str] = []

    for document in documents:
        path = document["path"]
        text = document["text"]
        normalized = normalize_text(text)

        if not text:
            removed_documents.append((path, "tom fil"))
            continue

        if len(text) < MIN_LENGTH:
            removed_documents.append((path, "for kort"))
            continue

        if alpha_ratio(text) < MIN_ALPHA_RATIO:
            removed_documents.append((path, "for meget stoej"))
            continue

        if looks_like_noise(text):
            removed_documents.append((path, "navigation eller ui-stoej"))
            continue

        if looks_like_draft(text):
            removed_documents.append((path, "kladde eller noter"))
            continue

        if normalized in kept_normalized_texts:
            removed_documents.append((path, "eksakt dublet"))
            continue

        if is_near_duplicate(normalized, kept_normalized_texts):
            removed_documents.append((path, "naesten dublet"))
            continue

        kept_documents.append(document)
        kept_normalized_texts.append(normalized)

    return kept_documents, removed_documents


def main() -> None:
    documents = load_documents(DATA_DIR)
    kept_documents, removed_documents = filter_documents(documents)

    print_section("Loesning - Dokumentfilter")
    print_label("BEHOLDTE DOKUMENTER", GREEN)
    for document in kept_documents:
        print(colorize(f"- {document['path'].name}", GREEN))

    print_label("FRASORTEREDE DOKUMENTER", YELLOW)
    for path, reason in removed_documents:
        print(colorize(f"- {path.name}: {reason}", RED))


if __name__ == "__main__":
    main()
