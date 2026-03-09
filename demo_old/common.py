from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    category: str
    text: str
    metadata: dict[str, Any]


SAMPLE_DOCS: list[dict[str, str]] = [
    {
        "id": "doc_prompting",
        "title": "Prompting Patterns for Support Bots",
        "source": "internal/prompting-guide.md",
        "category": "prompting",
        "text": (
            "Use a clear system role and define success criteria. "
            "Provide constraints such as tone, output format, and forbidden actions. "
            "Few-shot examples improve consistency for structured output. "
            "Ask the model to cite evidence when working with retrieved context."
        ),
    },
    {
        "id": "doc_rag",
        "title": "RAG Design Notes",
        "source": "internal/rag-notes.md",
        "category": "rag",
        "text": (
            "RAG combines retrieval and generation. Chunking quality strongly impacts relevance. "
            "Metadata filters can limit retrieval to a product line or date range. "
            "Top-k should be tuned because too much context can reduce answer quality. "
            "Grounded answers should reference retrieved passages."
        ),
    },
    {
        "id": "doc_ops",
        "title": "LLM Operations Checklist",
        "source": "internal/ops-checklist.md",
        "category": "operations",
        "text": (
            "Add logging for prompts, model settings, and response latency. "
            "Define fallback behavior for timeouts and tool failures. "
            "Collect user feedback and review failure cases weekly. "
            "Document privacy constraints and retention periods."
        ),
    },
    {
        "id": "doc_course",
        "title": "Course Logistics",
        "source": "internal/course-logistics.md",
        "category": "course",
        "text": (
            "Undervisning er fra 09:00 til 16:00. Pauser er 10:30-10:45, 12:30-13:15 og 14:30-14:45. "
            "Dag 2 fokuserer pa RAG-implementering og workflows. "
            "Dag 3 fokuserer pa evaluering, kvalitet, drift og mini-projekt."
        ),
    },
]


BENCHMARK_QA: list[dict[str, Any]] = [
    {
        "question": "Hvad er vigtigst for robust prompting til struktureret output?",
        "expected_keywords": ["few-shot", "format", "constraints"],
        "category": "prompting",
    },
    {
        "question": "Hvorfor skal top-k i RAG tunes?",
        "expected_keywords": ["top-k", "context", "quality"],
        "category": "rag",
    },
    {
        "question": "Hvilke driftstiltag anbefales ved tool failures?",
        "expected_keywords": ["fallback", "logging", "timeouts"],
        "category": "operations",
    },
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def approx_tokens(text: str) -> int:
    return max(1, len(tokenize(text)))


def render_prompt(
    system_role: str,
    task: str,
    user_input: str,
    examples: list[tuple[str, str]] | None = None,
    context_blocks: list[str] | None = None,
) -> str:
    parts = [f"SYSTEM:\n{system_role}", f"TASK:\n{task}"]
    if context_blocks:
        context_text = "\n\n".join(f"[Kontekst {i+1}] {c}" for i, c in enumerate(context_blocks))
        parts.append(f"CONTEXT:\n{context_text}")
    if examples:
        ex_lines = []
        for i, (inp, out) in enumerate(examples, start=1):
            ex_lines.append(f"Eksempel {i} INPUT: {inp}\nEksempel {i} OUTPUT: {out}")
        parts.append("FEW_SHOT:\n" + "\n\n".join(ex_lines))
    parts.append(f"USER:\n{user_input}")
    return "\n\n".join(parts)


def mock_llm(prompt: str, context_blocks: list[str] | None = None) -> str:
    prompt_lower = prompt.lower()
    wants_json = "json" in prompt_lower
    asks_citation = "cite" in prompt_lower or "citer" in prompt_lower or "evidence" in prompt_lower
    sources = []
    if context_blocks:
        for idx, block in enumerate(context_blocks, start=1):
            if any(tok in block.lower() for tok in tokenize(prompt)[:10]):
                sources.append((idx, block))
        if not sources:
            sources = list(enumerate(context_blocks, start=1))

    if sources:
        bullets = []
        for idx, block in sources[:2]:
            short = shorten(block, 100)
            if asks_citation:
                bullets.append(f"- Svarpunkt fra [Kontekst {idx}]: {short}")
            else:
                bullets.append(f"- {short}")
        answer = "Her er et grounded svar:\n" + "\n".join(bullets)
    else:
        answer = "Forslag: brug tydelig rolle, constraints og eksempler for mere stabilt output."

    if wants_json:
        payload = {
            "summary": "Robust prompt med rolle, constraints og eksempler",
            "actions": ["Definer format", "Tilfoj few-shot", "Valider output"],
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)
    return answer


def shorten(text: str, max_len: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_chunks_fixed(
    docs: list[dict[str, str]], chunk_words: int = 20, overlap_words: int = 5
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        words = doc["text"].split()
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(len(words), start + chunk_words)
            text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['id']}_fixed_{chunk_idx}",
                    doc_id=doc["id"],
                    title=doc["title"],
                    source=doc["source"],
                    category=doc["category"],
                    text=text,
                    metadata={"strategy": "fixed", "start_word": start, "end_word": end},
                )
            )
            if end == len(words):
                break
            start = max(0, end - overlap_words)
            chunk_idx += 1
    return chunks


def build_chunks_sentence(docs: list[dict[str, str]], group_size: int = 2) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc["text"]) if s.strip()]
        for i in range(0, len(sentences), group_size):
            group = sentences[i : i + group_size]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['id']}_sent_{i//group_size}",
                    doc_id=doc["id"],
                    title=doc["title"],
                    source=doc["source"],
                    category=doc["category"],
                    text=" ".join(group),
                    metadata={"strategy": "sentence", "sentence_start": i},
                )
            )
    return chunks


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
        "ved",
        "der",
        "et",
        "en",
        "som",
    }
    toks = [t for t in tokenize(text) if t not in stopwords and len(t) > 1]
    return Counter(toks)


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[token] * b.get(token, 0) for token in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query_vec = text_vector(query)
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        if metadata_filter:
            if any(getattr(chunk, key, None) != value for key, value in metadata_filter.items()):
                continue
        score = cosine_similarity(query_vec, text_vector(chunk.text))
        results.append({"chunk": chunk, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def format_hits(hits: Iterable[dict[str, Any]]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit["chunk"]
        lines.append(
            f"{i}. score={hit['score']:.3f} | {chunk.title} | {chunk.chunk_id} | {shorten(chunk.text, 120)}"
        )
    return "\n".join(lines)


def grounded_answer(
    question: str,
    hits: list[dict[str, Any]],
    prompt_variant: str = "baseline",
) -> str:
    if not hits or hits[0]["score"] <= 0:
        return "Jeg kan ikke finde relevant kontekst i de hentede dokumenter."
    citation_mode = prompt_variant in {"baseline", "strict"}
    lines = []
    for hit in hits[:2]:
        chunk = hit["chunk"]
        prefix = f"[{chunk.source}] " if citation_mode else ""
        lines.append(prefix + shorten(chunk.text, 110))
    lead = (
        "Svar (strict grounded):" if prompt_variant == "strict" else "Svar (grounded):"
    )
    return f"{lead}\n- " + "\n- ".join(lines)


def simple_chat_reply(messages: list[dict[str, str]], max_context_tokens: int = 80) -> dict[str, Any]:
    total = 0
    kept: list[dict[str, str]] = []
    for msg in reversed(messages):
        size = approx_tokens(msg["content"])
        if kept and total + size > max_context_tokens:
            break
        kept.append(msg)
        total += size
    kept.reverse()
    prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in kept)
    reply = mock_llm(prompt)
    return {"kept_messages": kept, "token_count": total, "reply": reply}


def fake_fx_rate(base: str, quote: str) -> float:
    rates = {
        ("USD", "DKK"): 6.92,
        ("EUR", "DKK"): 7.46,
        ("DKK", "EUR"): 0.134,
    }
    return rates.get((base.upper(), quote.upper()), 1.0)


def fake_tool_call(name: str, **kwargs: Any) -> dict[str, Any]:
    if name == "fx_rate":
        base = kwargs.get("base", "USD")
        quote = kwargs.get("quote", "DKK")
        return {"tool": name, "base": base, "quote": quote, "rate": fake_fx_rate(base, quote)}
    if name == "calc_total":
        items = kwargs.get("items", [])
        total = round(sum(float(item.get("price", 0)) for item in items), 2)
        return {"tool": name, "items": len(items), "total": total}
    return {"tool": name, "error": "unknown tool"}


def append_jsonl(path: str | Path, event: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")


def score_keywords(text: str, expected_keywords: list[str]) -> float:
    text_l = text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text_l)
    return found / max(1, len(expected_keywords))


def benchmark_retrieval(
    chunks: list[Chunk],
    top_k: int = 2,
    metadata_filter_by_category: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in BENCHMARK_QA:
        mfilter = {"category": item["category"]} if metadata_filter_by_category else None
        hits = retrieve(item["question"], chunks, top_k=top_k, metadata_filter=mfilter)
        answer = grounded_answer(item["question"], hits)
        keyword_score = score_keywords(answer, item["expected_keywords"])
        top_hit_category = hits[0]["chunk"].category if hits else None
        rows.append(
            {
                "question": item["question"],
                "target_category": item["category"],
                "top_hit_category": top_hit_category,
                "keyword_score": round(keyword_score, 2),
                "top_score": round(hits[0]["score"], 3) if hits else 0.0,
                "answer": answer,
                "hits": hits,
            }
        )
    return rows


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
