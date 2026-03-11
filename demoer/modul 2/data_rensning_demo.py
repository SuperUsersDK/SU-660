from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, RED, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

RAW_DOCUMENT = """<html>
<head>
<title>Embeddings Guide</title>
<script>
window.analytics.track("page_view");
</script>
<style>
body { font-family: Arial; }
</style>
</head>
<body>
<header>
Home | Docs | Pricing | Login
</header>

<main>
<h1>Embeddings i RAG</h1>
<p>Embeddings repraesenterer tekst som numeriske vektorer.</p>
<p>De bruges til at matche brugerens spoergsmaal med relevante dokumentbidder.</p>
</main>

<aside>
Download brochure
</aside>

<footer>
Privacy Policy | Cookie Settings | Contact
</footer>
</body>
</html>"""


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]

    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def highlight_noise(text: str) -> str:
    removable_block_patterns = [
        r"<script[\s\S]*?</script>",
        r"<style[\s\S]*?</style>",
        r"<header[\s\S]*?</header>",
        r"<footer[\s\S]*?</footer>",
        r"<aside[\s\S]*?</aside>",
        r"<head[\s\S]*?</head>",
    ]

    spans: list[tuple[int, int]] = []

    for pattern in removable_block_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            spans.append(match.span())

    for match in re.finditer(r"</?[a-zA-Z0-9]+[^>]*>", text):
        spans.append(match.span())

    merged_spans = merge_spans(spans)
    if not merged_spans:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end in merged_spans:
        if cursor < start:
            parts.append(text[cursor:start])
        parts.append(colorize(text[start:end], BOLD, RED))
        cursor = end

    if cursor < len(text):
        parts.append(text[cursor:])

    return "".join(parts)


def clean_document(text: str) -> str:
    cleaned = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<style[\s\S]*?</style>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<head[\s\S]*?</head>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<header[\s\S]*?</header>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<footer[\s\S]*?</footer>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<aside[\s\S]*?</aside>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?(html|body|main|h1|p)[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def main() -> None:
    clear_screen()
    print_section("Data Rensning Demo")
    print_label("POINTE", CYAN)
    print(
        colorize(
            "Foer embedding boer dokumenter renses for stoej som scripts, tags, navigation, header og footer.",
            DIM,
        )
    )

    print_section("Foer Rensning")
    print_label("RAA DATA", BLUE)
    print(highlight_noise(RAW_DOCUMENT))

    print_label("BEMAERK", YELLOW)
    print(colorize("Det roede indhold er typisk det, man vil fjerne foer embedding.", YELLOW))

    prompt_to_continue("Tryk Enter for at se den rensede version...")

    cleaned = clean_document(RAW_DOCUMENT)

    print_section("Efter Rensning")
    print_label("RENSERESULTAT", GREEN)
    print(colorize(cleaned, GREEN))

    print_label("HVORFOR", CYAN)
    print(
        colorize(
            "Den rensede tekst er mere egnet til embeddings, fordi den fokuserer paa det faglige indhold i stedet for UI-stoej.",
            DIM,
        )
    )


if __name__ == "__main__":
    main()
