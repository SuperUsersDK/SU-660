from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


SAMPLE_DOCS: list[dict[str, str]] = [
    {
        "id": "rag_notes",
        "title": "RAG Design Notes",
        "text": (
            "RAG combines retrieval and generation. Chunking quality strongly impacts relevance. "
            "Metadata filters can limit retrieval to a product line or date range. "
            "Top-k should be tuned because too much context can reduce answer quality. "
            "Grounded answers should reference retrieved passages. "
            "Overlap can improve recall around chunk boundaries but may increase redundancy."
        ),
    },
    {
        "id": "ops_notes",
        "title": "LLM Operations Checklist",
        "text": (
            "Add logging for prompts, model settings, and response latency. "
            "Define fallback behavior for timeouts and tool failures. "
            "Collect user feedback and review failure cases weekly. "
            "Document privacy constraints and retention periods. "
            "Track prompt and retrieval versions for reproducibility."
        ),
    },
]


MARKDOWN_DOC = {
    "id": "policy_md",
    "title": "Service Policy",
    "text": (
        "# Service Policy\n\n"
        "## Scope\n"
        "This policy applies to customer-facing support workflows and internal escalation handling.\n\n"
        "## Termination Clause\n"
        "Either party may terminate with 30 days written notice. Immediate termination is allowed for material breach.\n"
        "A termination notice must include an effective date and authorized signature.\n\n"
        "## Billing Terms\n"
        "Invoices are due within 14 days. Late fees may apply after the due date.\n"
        "Usage is reported monthly and disputed items must be reported within 10 business days.\n\n"
        "## Security\n"
        "Access logs are retained for 90 days. Administrative actions must be auditable."
    ),
}


QUERY_SET: list[dict[str, Any]] = [
    {
        "question": "Why should top-k be tuned in RAG?",
        "expected_keywords": ["top-k", "context", "quality"],
    },
    {
        "question": "What should we log in an LLM solution?",
        "expected_keywords": ["logging", "latency", "fallback"],
    },
    {
        "question": "What does the termination clause say about termination notice?",
        "expected_keywords": ["terminate", "notice", "termination"],
    },
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def approx_tokens(text: str) -> int:
    # Didactic estimate only, not a tokenizer.
    return max(1, len(tokenize(text)))


def shorten(text: str, max_len: int = 110) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def text_vector(text: str) -> Counter[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "is",
        "are",
        "det",
        "den",
        "de",
        "og",
        "i",
        "pa",
        "til",
        "er",
        "hvad",
        "hvorfor",
        "skal",
        "med",
        "for",
        "om",
        "ved",
        "that",
        "this",
        "with",
    }
    toks = [t for t in tokenize(text) if t not in stopwords and len(t) > 1]
    return Counter(toks)


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0) for t in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[Chunk, float]]:
    qv = text_vector(query)
    scored = [(chunk, cosine_similarity(qv, text_vector(chunk.text))) for chunk in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def format_hits(hits: list[tuple[Chunk, float]], max_len: int = 110) -> str:
    lines = []
    for i, (chunk, score) in enumerate(hits, start=1):
        strategy = chunk.metadata.get("strategy", "?")
        lines.append(
            f"{i}. score={score:.3f} | {chunk.doc_id} | {chunk.chunk_id} | {strategy} | "
            f"{shorten(chunk.text, max_len)}"
        )
    return "\n".join(lines) if lines else "(ingen hits)"


def print_chunk_preview(chunks: list[Chunk], max_chunks: int = 8) -> None:
    print(f"Antal chunks: {len(chunks)}")
    for chunk in chunks[:max_chunks]:
        print(
            f"- {chunk.chunk_id} | {chunk.metadata.get('strategy')} | "
            f"len={len(chunk.text)} chars | tok~{approx_tokens(chunk.text)}"
        )
        print(f"  {shorten(chunk.text, 120)}")
    if len(chunks) > max_chunks:
        print(f"... ({len(chunks) - max_chunks} flere)")


def _make_chunk(doc: dict[str, str], text: str, idx: int, strategy: str, **meta: Any) -> Chunk:
    return Chunk(
        chunk_id=f"{doc['id']}_{strategy}_{idx}",
        doc_id=doc["id"],
        text=text.strip(),
        metadata={"strategy": strategy, **meta},
    )


def chunk_fixed_words(
    docs: list[dict[str, str]],
    chunk_words: int,
    overlap_words: int = 0,
    strategy_name: str = "fixed_words",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        words = doc["text"].split()
        start = 0
        idx = 0
        while start < len(words):
            end = min(len(words), start + chunk_words)
            text = " ".join(words[start:end])
            chunks.append(
                _make_chunk(doc, text, idx, strategy_name, start_word=start, end_word=end)
            )
            if end == len(words):
                break
            start = max(0, end - overlap_words)
            idx += 1
    return chunks


def chunk_fixed_chars(
    docs: list[dict[str, str]],
    chunk_chars: int,
    overlap_chars: int = 0,
    strategy_name: str = "fixed_chars",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        text = " ".join(doc["text"].split())
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            chunk_text = text[start:end]
            chunks.append(_make_chunk(doc, chunk_text, idx, strategy_name, start_char=start, end_char=end))
            if end == len(text):
                break
            start = max(0, end - overlap_chars)
            idx += 1
    return chunks


def chunk_sentence_groups(
    docs: list[dict[str, str]],
    group_size: int = 2,
    sentence_overlap: int = 0,
    strategy_name: str = "sentence_group",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        sentences = split_sentences(doc["text"])
        if group_size <= 0:
            raise ValueError("group_size skal vaere > 0")
        step = max(1, group_size - sentence_overlap)
        idx = 0
        for start in range(0, len(sentences), step):
            group = sentences[start : start + group_size]
            if not group:
                continue
            chunks.append(
                _make_chunk(
                    doc,
                    " ".join(group),
                    idx,
                    strategy_name,
                    sentence_start=start,
                    sentence_end=start + len(group),
                )
            )
            idx += 1
            if start + group_size >= len(sentences):
                break
    return chunks


def chunk_paragraphs(
    docs: list[dict[str, str]],
    max_paragraphs_per_chunk: int = 1,
    strategy_name: str = "paragraph",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        paragraphs = split_paragraphs(doc["text"])
        idx = 0
        for start in range(0, len(paragraphs), max_paragraphs_per_chunk):
            group = paragraphs[start : start + max_paragraphs_per_chunk]
            chunks.append(
                _make_chunk(
                    doc,
                    "\n\n".join(group),
                    idx,
                    strategy_name,
                    paragraph_start=start,
                    paragraph_end=start + len(group),
                )
            )
            idx += 1
    return chunks


def _recursive_split_text(text: str, max_chars: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    sep = separators[0]
    if sep:
        pieces = [p.strip() for p in text.split(sep)]
        joiner = sep
    else:
        pieces = list(text)
        joiner = ""

    # If separator does not split meaningfully, try smaller separators.
    if len(pieces) <= 1:
        return _recursive_split_text(text, max_chars, separators[1:])

    out: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            out.extend(_recursive_split_text(current, max_chars, separators[1:]))
        if len(piece) <= max_chars:
            current = piece
        else:
            out.extend(_recursive_split_text(piece, max_chars, separators[1:]))
            current = ""
    if current:
        out.extend(_recursive_split_text(current, max_chars, separators[1:]))
    return [p.strip() for p in out if p.strip()]


def _apply_char_overlap(segments: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return segments
    out: list[str] = []
    prev = ""
    for seg in segments:
        if prev:
            prefix = prev[-overlap_chars:]
            out.append((prefix + " " + seg).strip())
        else:
            out.append(seg)
        prev = seg
    return out


def chunk_recursive(
    docs: list[dict[str, str]],
    max_chars: int = 280,
    overlap_chars: int = 40,
    separators: list[str] | None = None,
    strategy_name: str = "recursive",
) -> list[Chunk]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[Chunk] = []
    for doc in docs:
        base_segments = _recursive_split_text(doc["text"], max_chars=max_chars, separators=separators)
        segments = _apply_char_overlap(base_segments, overlap_chars)
        for idx, seg in enumerate(segments):
            chunks.append(
                _make_chunk(
                    doc,
                    seg,
                    idx,
                    strategy_name,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
    return chunks


def chunk_markdown_headings(
    markdown_doc: dict[str, str],
    max_chars_per_section_chunk: int = 260,
) -> list[Chunk]:
    text = markdown_doc["text"]
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "ROOT"
    current_lines: list[str] = []

    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    for line in lines:
        m = heading_re.match(line)
        if m:
            if current_lines:
                sections.append((current_heading, current_lines))
            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading = f"h{level}:{title}"
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    out: list[Chunk] = []
    idx = 0
    for heading, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        temp_doc = {
            "id": markdown_doc["id"],
            "title": markdown_doc.get("title", markdown_doc["id"]),
            "text": section_text,
        }
        sub_chunks = chunk_recursive(
            [temp_doc],
            max_chars=max_chars_per_section_chunk,
            overlap_chars=30,
            strategy_name="heading_recursive",
        )
        for sub in sub_chunks:
            sub.chunk_id = f"{markdown_doc['id']}_heading_{idx}"
            sub.metadata["heading"] = heading
            out.append(sub)
            idx += 1
    return out


def chunk_semanticish_sentences(
    docs: list[dict[str, str]],
    max_sentences_per_chunk: int = 4,
    similarity_threshold: float = 0.12,
) -> list[Chunk]:
    """
    Simple lexical-similarity grouping (teaching demo only).
    It approximates "semantic chunking" without embeddings.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        sentences = split_sentences(doc["text"])
        if not sentences:
            continue
        groups: list[list[str]] = []
        current_group = [sentences[0]]
        current_vec = text_vector(sentences[0])

        for sentence in sentences[1:]:
            sv = text_vector(sentence)
            sim = cosine_similarity(current_vec, sv)
            if sim >= similarity_threshold and len(current_group) < max_sentences_per_chunk:
                current_group.append(sentence)
                current_vec = current_vec + sv
            else:
                groups.append(current_group)
                current_group = [sentence]
                current_vec = sv
        groups.append(current_group)

        for idx, group in enumerate(groups):
            chunks.append(
                _make_chunk(
                    doc,
                    " ".join(group),
                    idx,
                    "semanticish",
                    max_sentences=max_sentences_per_chunk,
                    similarity_threshold=similarity_threshold,
                )
            )
    return chunks


def keyword_score(question: str, chunk_hits: list[tuple[Chunk, float]], expected_keywords: Iterable[str]) -> float:
    joined = " ".join(chunk.text.lower() for chunk, _ in chunk_hits)
    expected = [kw.lower() for kw in expected_keywords]
    if not expected:
        return 0.0
    return round(sum(1 for kw in expected if kw in joined) / len(expected), 2)
