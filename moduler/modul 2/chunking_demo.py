from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, MAGENTA, RED, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

from chunking.common import (  # noqa: E402
    Chunk,
    approx_tokens,
    chunk_markdown_headings,
    chunk_semanticish_sentences,
    chunk_sentence_groups,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "knowledge_base" / "chunking_workshop.md"


def load_document() -> dict[str, str]:
    text = DATA_PATH.read_text(encoding="utf-8")
    return {
        "id": "chunking_workshop",
        "title": "Chunking Workshop Notes",
        "text": text,
    }


def tokenize_via_tiktoken(text: str, model_name: str = "gpt-4o-mini") -> list[int]:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - teaching fallback
        raise RuntimeError("Dette demo kræver tiktoken for token-baseret chunking.") from exc

    encoding = tiktoken.encoding_for_model(model_name)
    return encoding.encode(text)


def decode_via_tiktoken(tokens: list[int], model_name: str = "gpt-4o-mini") -> str:
    import tiktoken

    encoding = tiktoken.encoding_for_model(model_name)
    return encoding.decode(tokens)


def chunk_by_tokens(
    document: dict[str, str],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
    strategy_name: str,
    model_name: str = "gpt-4o-mini",
) -> list[Chunk]:
    tokens = tokenize_via_tiktoken(document["text"], model_name=model_name)
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(len(tokens), start + chunk_tokens)
        chunk_text = decode_via_tiktoken(tokens[start:end], model_name=model_name).strip()
        overlap_prefix = ""
        if overlap_tokens > 0 and start > 0:
            overlap_prefix = decode_via_tiktoken(
                tokens[start : min(end, start + overlap_tokens)],
                model_name=model_name,
            ).lstrip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{document['id']}_{strategy_name}_{idx}",
                    doc_id=document["id"],
                    text=chunk_text,
                    metadata={
                        "strategy": strategy_name,
                        "start_token": start,
                        "end_token": end,
                        "chunk_tokens": chunk_tokens,
                        "overlap_tokens": overlap_tokens,
                        "overlap_prefix": overlap_prefix,
                    },
                )
            )
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)
        idx += 1

    return chunks


def chunk_preview(chunk: Chunk) -> str:
    plain_text = chunk.text.replace("\n", " ")
    preview = plain_text
    overlap_prefix = str(chunk.metadata.get("overlap_prefix", ""))
    if not overlap_prefix:
        return preview

    overlap_preview = overlap_prefix.replace("\n", " ")
    overlap_len = min(len(overlap_preview), len(preview))
    return colorize(preview[:overlap_len], BOLD, RED) + preview[overlap_len:]


def print_chunk_report(
    title: str,
    chunks: list[Chunk],
    *,
    description: list[str] | None = None,
    max_chunks: int = 4,
) -> None:
    print_section(title)
    if description:
        print_strategy_description(description)
    print(colorize(f"Antal chunks: {len(chunks)}", GREEN))

    for chunk in chunks[:max_chunks]:
        print_label(chunk.chunk_id, BLUE)
        print(
            colorize(
                f"strategy={chunk.metadata.get('strategy')} | tok~{approx_tokens(chunk.text)} | len={len(chunk.text)} chars",
                DIM,
            )
        )
        print(chunk_preview(chunk))

    if len(chunks) > max_chunks:
        print(colorize(f"\n... {len(chunks) - max_chunks} flere chunks", YELLOW))


def color_chunk_with_overlap(
    chunk: Chunk,
    *,
    previous_overlap: str = "",
    next_overlap: str = "",
) -> str:
    text = chunk.text.replace("\n", " ")

    if previous_overlap:
        prefix_len = min(len(previous_overlap), len(text))
        text = colorize(text[:prefix_len], BOLD, RED) + text[prefix_len:]

    if next_overlap:
        plain_prefix_len = len(previous_overlap)
        plain_text = chunk.text.replace("\n", " ")
        suffix_len = min(len(next_overlap), len(plain_text))
        split_at = len(plain_text) - suffix_len
        middle = plain_text[plain_prefix_len:split_at]
        suffix = plain_text[split_at:]
        prefix = plain_text[:plain_prefix_len]

        colored_prefix = colorize(prefix, BOLD, RED) if prefix else ""
        colored_suffix = colorize(suffix, BOLD, GREEN)
        text = colored_prefix + middle + colored_suffix

    return text


def print_overlap_chunk_report(
    title: str,
    chunks: list[Chunk],
    *,
    description: list[str] | None = None,
    max_chunks: int = 4,
) -> None:
    print_section(title)
    if description:
        print_strategy_description(description)
    print(colorize(f"Antal chunks: {len(chunks)}", GREEN))
    print(colorize("Grønt = overlap i slutningen af forrige chunk. Rødt = samme overlap i starten af næste chunk.", DIM))

    limit = min(len(chunks), max_chunks)
    for index in range(limit):
        chunk = chunks[index]
        previous_overlap = str(chunk.metadata.get("overlap_prefix", "")).replace("\n", " ")
        next_overlap = ""
        if index + 1 < len(chunks):
            next_overlap = str(chunks[index + 1].metadata.get("overlap_prefix", "")).replace("\n", " ")

        print_label(chunk.chunk_id, BLUE)
        print(
            colorize(
                f"strategy={chunk.metadata.get('strategy')} | tok~{approx_tokens(chunk.text)} | len={len(chunk.text)} chars",
                DIM,
            )
        )
        print(color_chunk_with_overlap(chunk, previous_overlap=previous_overlap, next_overlap=next_overlap))

    if len(chunks) > max_chunks:
        print(colorize(f"\n... {len(chunks) - max_chunks} flere chunks", YELLOW))


def langchain_markdown_chunks(document: dict[str, str]) -> list[Chunk]:
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except ImportError as exc:  # pragma: no cover - teaching fallback
        raise RuntimeError("Dette demo kræver langchain_text_splitters.") from exc

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    docs = splitter.split_text(document["text"])

    chunks: list[Chunk] = []
    for idx, doc in enumerate(docs):
        chunks.append(
            Chunk(
                chunk_id=f"{document['id']}_langchain_md_{idx}",
                doc_id=document["id"],
                text=doc.page_content.strip(),
                metadata={"strategy": "langchain_markdown", **doc.metadata},
            )
        )
    return chunks


def print_langchain_report(
    chunks: list[Chunk],
    *,
    description: list[str] | None = None,
    max_chunks: int = 4,
) -> None:
    print_section("6. LangChain chunking på markdown-filen")
    if description:
        print_strategy_description(description)
    print(colorize(f"Antal chunks: {len(chunks)}", GREEN))

    for chunk in chunks[:max_chunks]:
        print_label(chunk.chunk_id, MAGENTA)
        print(
            colorize(
                f"h1={chunk.metadata.get('h1')} | h2={chunk.metadata.get('h2')} | h3={chunk.metadata.get('h3')}",
                DIM,
            )
        )
        print(chunk.text.replace("\n", " "))

    if len(chunks) > max_chunks:
        print(colorize(f"\n... {len(chunks) - max_chunks} flere chunks", YELLOW))


def print_strategy_description(lines: list[str]) -> None:
    for line in lines:
        print(colorize(line, DIM))
    print()


def main() -> None:
    clear_screen()
    document = load_document()

    print_section("Chunking Demo")
    print_label("KILDE", CYAN)
    print(colorize(str(DATA_PATH), DIM))
    print(colorize("Vi chunker samme markdown-fil på flere forskellige måder.", DIM))

    token_no_overlap = chunk_by_tokens(
        document,
        chunk_tokens=180,
        overlap_tokens=0,
        strategy_name="token_no_overlap",
    )
    print_chunk_report(
        "1. Token-baseret chunking uden overlap",
        token_no_overlap,
        description=[
            "Her splitter vi udelukkende efter token-antal og uden overlap mellem chunks.",
            "Det er en enkel baseline, som er let at implementere og sammenligne med andre strategier.",
            "Ulempen er, at chunk-grænser kan falde midt i en idé eller lige der, hvor vigtig kontekst burde have fortsat.",
        ],
    )
    prompt_to_continue("Tryk Enter for at gå videre til token-chunking med overlap...")

    token_with_overlap = chunk_by_tokens(
        document,
        chunk_tokens=180,
        overlap_tokens=40,
        strategy_name="token_with_overlap",
    )
    print_overlap_chunk_report(
        "2. Token-baseret chunking med overlap",
        token_with_overlap,
        description=[
            "Her bruger vi stadig faste token-vinduer, men nu gentages noget af slutningen af en chunk i starten af den næste.",
            "Det kan beskytte mod tab af kontekst ved chunk-grænser og forbedre recall.",
            "Prisen er mere redundans, flere chunks og højere tokenforbrug.",
        ],
    )
    prompt_to_continue("Tryk Enter for at gå videre til sentence-based chunking...")

    sentence_chunks = chunk_sentence_groups(
        [document],
        group_size=3,
        sentence_overlap=0,
        strategy_name="sentence_based",
    )
    print_chunk_report(
        "3. Sentence-based chunking",
        sentence_chunks,
        description=[
            "Sentence-based chunking grupperer hele sætninger i stedet for at splitte efter et fast tokenbudget.",
            "Det giver ofte mere læsbare chunks, fordi hver chunk bevarer komplette tanker.",
            "Til gengæld bliver chunk-størrelserne mere ujævne, fordi sætninger kan variere meget i længde.",
        ],
    )
    prompt_to_continue("Tryk Enter for at gå videre til structure-aware chunking...")

    structure_chunks = chunk_markdown_headings(
        document,
        max_chars_per_section_chunk=500,
    )
    print_chunk_report(
        "4. Structure-aware chunking",
        structure_chunks,
        description=[
            "Structure-aware chunking bruger dokumentets overskrifter som naturlige grænser, før teksten deles yderligere.",
            "Det er særligt nyttigt i markdown, manualer og politikker, hvor sektionstitler bærer vigtig betydning.",
            "Fordelen er bedre explainability, fordi man lettere kan se, hvilken sektion et chunk kommer fra.",
        ],
    )
    prompt_to_continue("Tryk Enter for at gå videre til semantic chunking...")

    semantic_chunks = chunk_semanticish_sentences(
        [document],
        max_sentences_per_chunk=3,
        similarity_threshold=0.08,
    )
    print_chunk_report(
        "5. Semantic chunking (didaktisk approx)",
        semantic_chunks,
        description=[
            "Semantic chunking prøver at gruppere sætninger, der hører sammen indholdsmæssigt, i stedet for kun at bruge faste grænser.",
            "I dette demo er det en didaktisk approximation baseret på tekstlig lighed, ikke rigtige embeddings.",
            "Pointen er at vise idéen bag adaptive chunk-grænser, som bedre følger meningsenheder i teksten.",
        ],
    )
    prompt_to_continue("Tryk Enter for at gå videre til LangChain chunking...")

    langchain_chunks = langchain_markdown_chunks(document)
    print_langchain_report(
        langchain_chunks,
        description=[
            "Her ser vi, hvordan en færdig chunking-komponent fra LangChain splitter den samme markdown-fil.",
            "Den bruger dokumentets markdown-struktur og gemmer samtidig heading-metadata på hvert chunk.",
            "Det er nyttigt, fordi du kan få en hurtig standardløsning uden selv at skrive hele splitter-logikken.",
        ],
    )


if __name__ == "__main__":
    main()
