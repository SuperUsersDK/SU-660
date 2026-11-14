"""Chunking-strategier, der husker hvor i teksten hver chunk kommer fra.

Alle strategier returnerer `Chunk(start, end, text, meta)`, hvor `text == kilde[start:end]`.
Positionerne bruges af webappen til at vise og animere, hvor hver strategi klipper.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Chunk:
    start: int
    end: int
    text: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    default: int
    minimum: int
    maximum: int
    step: int


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    description: str
    params: tuple[Param, ...]
    func: Callable[..., list[Chunk]]


def make_chunk(text: str, start: int, end: int, **meta) -> Chunk | None:
    """Skær whitespace af i begge ender, så chunken starter og slutter med indhold."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return Chunk(start=start, end=end, text=text[start:end], meta=meta)


def _collect(chunks: list[Chunk | None]) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk is not None]


# --------------------------------------------------------------------------- #
# Hjælpere
# --------------------------------------------------------------------------- #

def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Sætninger slutter ved . ! ? efterfulgt af mellemrum, eller ved et linjeskift."""
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    current = start
    for match in re.finditer(r"[.!?]+(?=\s)|\n", text[start:end]):
        boundary = start + match.end()
        if text[current:boundary].strip():
            spans.append((current, boundary))
        current = boundary
    if text[current:end].strip():
        spans.append((current, end))
    return spans


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Afsnit er adskilt af mindst én tom linje."""
    return [(m.start(), m.end()) for m in re.finditer(r"(?:[^\n]|\n(?![ \t]*\n))+", text) if m.group().strip()]


def group_spans(spans: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Saml nabo-spans, så længe den samlede længde holder sig under max_chars."""
    groups: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        if groups and span_end - groups[-1][0] <= max_chars:
            groups[-1] = (groups[-1][0], span_end)
        else:
            groups.append((span_start, span_end))
    return groups


# --------------------------------------------------------------------------- #
# Strategier
# --------------------------------------------------------------------------- #

def fixed_chars(text: str, size: int = 300) -> list[Chunk]:
    """Klip hver `size` tegn, uanset ord, sætninger og overskrifter."""
    return _collect([make_chunk(text, s, min(s + size, len(text))) for s in range(0, len(text), size)])


def fixed_chars_overlap(text: str, size: int = 300, overlap: int = 60) -> list[Chunk]:
    """Som fast antal tegn, men hver chunk starter `overlap` tegn før den forrige sluttede."""
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap
    chunks = []
    for s in range(0, len(text), step):
        e = min(s + size, len(text))
        chunks.append(make_chunk(text, s, e))
        if e >= len(text):
            break
    return _collect(chunks)


def sentences(text: str, max_chars: int = 400) -> list[Chunk]:
    """Saml hele sætninger, indtil chunken når max_chars."""
    return _collect([make_chunk(text, s, e) for s, e in group_spans(sentence_spans(text), max_chars)])


def paragraphs(text: str, max_chars: int = 800) -> list[Chunk]:
    """Ét afsnit pr. chunk. Afsnit over max_chars deles ved sætningsgrænser."""
    chunks: list[Chunk | None] = []
    for p_start, p_end in paragraph_spans(text):
        if p_end - p_start <= max_chars:
            chunks.append(make_chunk(text, p_start, p_end))
        else:
            for s, e in group_spans(sentence_spans(text, p_start, p_end), max_chars):
                chunks.append(make_chunk(text, s, e, split_paragraph=True))
    return _collect(chunks)


def recursive(text: str, size: int = 400, overlap: int = 40) -> list[Chunk]:
    """LangChains RecursiveCharacterTextSplitter: prøver afsnit, så linjer, så ord, så tegn."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=max(0, min(overlap, size - 1)),
        add_start_index=True,
    )
    chunks = []
    for document in splitter.create_documents([text]):
        start = document.metadata["start_index"]
        chunks.append(make_chunk(text, start, start + len(document.page_content)))
    return _collect(chunks)


HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)


def markdown_sections(text: str) -> list[Chunk]:
    """Én chunk pr. Markdown-sektion (fra en overskrift til den næste)."""
    headings = list(HEADING.finditer(text))
    if not headings:
        raise ValueError("Teksten har ingen Markdown-overskrifter (#, ##, ...).")

    chunks: list[Chunk | None] = []
    if text[: headings[0].start()].strip():
        chunks.append(make_chunk(text, 0, headings[0].start(), heading="(før første overskrift)"))

    path: list[tuple[int, str]] = []
    for index, match in enumerate(headings):
        level, title = len(match.group(1)), match.group(2)
        path = [(lvl, name) for lvl, name in path if lvl < level] + [(level, title)]
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        chunks.append(make_chunk(text, match.start(), end, heading=" › ".join(name for _, name in path)))
    return _collect(chunks)


def python_source(text: str) -> list[Chunk]:
    """Én chunk pr. funktion eller klasse; imports og konstanter samles i én chunk."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError("Teksten er ikke gyldig Python-kode.") from exc

    line_starts = [0]
    for line in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    lines = text.splitlines()

    def offset(line_no: int) -> int:
        return line_starts[min(line_no - 1, len(line_starts) - 1)]

    chunks: list[Chunk | None] = []
    pending: tuple[int, int] | None = None  # sammenhængende kode, der ikke er def/class

    def flush() -> None:
        nonlocal pending
        if pending:
            chunks.append(make_chunk(text, offset(pending[0]), offset(pending[1] + 1), kind="modul", name="imports og konstanter"))
            pending = None

    for node in tree.body:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        last = node.end_lineno or node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush()
            # Kommentarlinjer lige over en funktion hører til funktionen.
            while first > 1 and lines[first - 2].lstrip().startswith("#"):
                first -= 1
            kind = "klasse" if isinstance(node, ast.ClassDef) else "funktion"
            chunks.append(make_chunk(text, offset(first), offset(last + 1), kind=kind, name=node.name))
        else:
            pending = (pending[0] if pending else first, last)
    flush()
    return _collect(chunks)


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        "fixed", "Fast antal tegn",
        "Klipper blindt hver N tegn. Hurtigt og simpelt, men skærer midt i ord og sætninger.",
        (Param("size", "Tegn pr. chunk", 300, 50, 2000, 50),),
        fixed_chars,
    ),
    Strategy(
        "overlap", "Tegn + overlap",
        "Som fast antal tegn, men hver chunk gentager slutningen af den forrige, så en sætning ved et snit findes i mindst én chunk.",
        (Param("size", "Tegn pr. chunk", 300, 50, 2000, 50), Param("overlap", "Overlap (tegn)", 80, 0, 500, 10)),
        fixed_chars_overlap,
    ),
    Strategy(
        "sentences", "Sætninger",
        "Samler hele sætninger op til en maksimal længde. Klipper aldrig midt i en sætning.",
        (Param("max_chars", "Maks. tegn", 400, 100, 2000, 50),),
        sentences,
    ),
    Strategy(
        "paragraphs", "Afsnit",
        "Ét afsnit pr. chunk (tom linje = nyt afsnit). Meget lange afsnit deles ved sætninger.",
        (Param("max_chars", "Maks. tegn", 800, 100, 3000, 50),),
        paragraphs,
    ),
    Strategy(
        "recursive", "Rekursiv (LangChain)",
        "Prøver først at dele ved afsnit, så linjer, så ord og til sidst tegn, for at holde chunks under maks. størrelsen.",
        (Param("size", "Maks. tegn", 400, 100, 2000, 50), Param("overlap", "Overlap (tegn)", 40, 0, 400, 10)),
        recursive,
    ),
    Strategy(
        "markdown", "Markdown-overskrifter",
        "Én chunk pr. sektion under en overskrift. Holder emner samlet, men virker kun på Markdown.",
        (),
        markdown_sections,
    ),
    Strategy(
        "code", "Kildekode (Python)",
        "Én chunk pr. funktion eller klasse via Pythons syntakstræ. Virker kun på Python-kode.",
        (),
        python_source,
    ),
)

STRATEGY_BY_ID = {strategy.id: strategy for strategy in STRATEGIES}


def run_strategy(strategy_id: str, text: str, params: dict | None = None) -> list[Chunk]:
    """Kør en strategi med parametre, der holdes inden for de tilladte grænser."""
    strategy = STRATEGY_BY_ID[strategy_id]
    values = {}
    for param in strategy.params:
        raw = (params or {}).get(param.name, param.default)
        values[param.name] = max(param.minimum, min(param.maximum, int(raw)))
    return strategy.func(text, **values)
