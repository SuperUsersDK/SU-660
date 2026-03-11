import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BOLD, CYAN, GREEN, RED, colorize, print_label, print_section


def is_useful_text(text: str | None, min_length: int = 80) -> bool:
    if text is None:
        return False

    cleaned = text.strip()
    if not cleaned:
        return False

    if min_length < 0:
        raise ValueError("min_length skal vaere 0 eller stoerre")

    if len(cleaned) < min_length:
        return False

    # For lidt alfabetisk indhold = sandsynligvis støj
    letters = sum(c.isalpha() for c in cleaned)
    ratio = letters / max(len(cleaned), 1)
    if ratio < 0.5:
        return False

    # Frasortér meget typisk navigations-/støjtekst
    noise_patterns = [
        r"cookie",
        r"accept all",
        r"privacy policy",
        r"home\s*\|\s*about\s*\|\s*contact",
    ]

    lowered = cleaned.lower()
    return not any(re.search(pattern, lowered) for pattern in noise_patterns)


def main() -> None:
    examples = [
        "OK",
        "Home | About | Contact",
        None,
        "Embeddings repraesenterer tekst som numeriske vektorer, som bruges til similarity search i et RAG-system.",
    ]

    print_section("Filter 2 - Brugbar tekst")
    print_label("RESULTATER", CYAN)
    for text in examples:
        result = is_useful_text(text)
        status = colorize(str(result), BOLD, GREEN if result else RED)
        print(f"{status} -> {repr(text)}")


if __name__ == "__main__":
    main()
