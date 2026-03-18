from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, CYAN, DIM, GREEN, MAGENTA, YELLOW, colorize, print_label, print_section

SOURCE_PATH = ROOT_DIR / "demoer" / "modul 2" / "data" / "knowledge_base" / "chunking_workshop.md"
MODEL_NAME = "gpt-4o-mini"


def chunk_fixed_size(text: str, *, chunk_tokens: int, overlap_tokens: int = 0) -> list[str]:
    # TODO: Her skal kursisterne skrive koden til fixed-size chunking.
    # Idé:
    # 1. Tokenizér teksten.
    # 2. Gå gennem tokens i vinduer på chunk_tokens.
    # 3. Brug overlap_tokens til at lade nabo-chunks dele noget tekst.
    # 4. Dekod hver token-blok tilbage til tekst og tilføj den til chunks.

    # En anden mulighed er at arbejde direkte med tekst og splitte per x characterer (nemmere at debugger)
    return []


def chunk_structure_aware(markdown_text: str, *, max_tokens_per_chunk: int) -> list[str]:
    # TODO: Her skal kursisterne skrive koden til structure-aware chunking.
    # Idé:
    # 1. Split dokumentet i sektioner ud fra markdown-overskrifter.
    # 2. Del hver sektion videre op i mindre chunks.
    # 3. Sørg for at et chunk ikke bliver for langt.
    # 4. Bevar gerne overskriften sammen med chunkens tekst.
    return []


def load_document() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8").strip()


def tokenize(text: str) -> list[int] | list[str]:
    try:
        import tiktoken
    except ImportError:
        return text.split()

    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return encoding.encode(text)


def decode(tokens: list[int] | list[str]) -> str:
    try:
        import tiktoken
    except ImportError:
        return " ".join(tokens)

    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return encoding.decode(tokens)


def approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text, flags=re.UNICODE)))


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "Introduktion"
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        if line.startswith("#"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = line.lstrip("#").strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(heading, body) for heading, body in sections if body]


def count_boundary_issues(chunks: list[str]) -> int:
    issues = 0
    for chunk in chunks[:-1]:
        trimmed = chunk.rstrip()
        if trimmed and trimmed[-1] not in ".!?":
            issues += 1
    return issues


def print_chunk_list(title: str, chunks: list[str], *, color: str) -> None:
    print_section(title)
    print(colorize(f"Antal chunks: {len(chunks)}", GREEN))
    print(colorize(f"Chunks der sandsynligvis slutter midt i en tanke: {count_boundary_issues(chunks)}", DIM))

    if not chunks:
        print(colorize("Ingen chunks endnu. Implementér chunking-koden i templaten.", YELLOW))
        return

    for index, chunk in enumerate(chunks, start=1):
        print_label(f"Chunk {index}", color)
        print(colorize(f"Ca. {approx_tokens(chunk)} tokens", DIM))
        print(chunk)
        print()


def print_summary(
    token_no_overlap: list[str],
    token_with_overlap: list[str],
    structure_aware: list[str],
) -> None:
    print_section("Sammenligning")
    rows = [
        ("Fixed-size uden overlap", token_no_overlap),
        ("Fixed-size med overlap", token_with_overlap),
        ("Structure-aware", structure_aware),
    ]

    for name, chunks in rows:
        print(
            colorize(
                f"- {name:<24} chunks={len(chunks):<2} | brud ved chunk-slutning={count_boundary_issues(chunks)}",
                YELLOW,
            )
        )


def main() -> None:
    document = load_document()

    print_section("Chunking Template")
    print_label("KILDE", CYAN)
    print(colorize(str(SOURCE_PATH), DIM))

    token_no_overlap = chunk_fixed_size(document, chunk_tokens=120, overlap_tokens=0)
    token_with_overlap = chunk_fixed_size(document, chunk_tokens=120, overlap_tokens=30)
    structure_aware = chunk_structure_aware(document, max_tokens_per_chunk=120)

    print_chunk_list("1. Fixed-size uden overlap", token_no_overlap, color=BLUE)
    print_chunk_list("2. Fixed-size med overlap", token_with_overlap, color=MAGENTA)
    print_chunk_list("3. Structure-aware chunking", structure_aware, color=CYAN)
    print_summary(token_no_overlap, token_with_overlap, structure_aware)


if __name__ == "__main__":
    main()
