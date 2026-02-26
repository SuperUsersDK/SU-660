from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable


@dataclass
class Doc:
    doc_id: str
    text: str
    metadata: dict[str, str]


@dataclass
class Hit:
    doc: Doc
    score: float


DOCS: list[Doc] = [
    Doc(
        doc_id="policy_refund",
        text=(
            "Refund requests must be submitted within 14 days after purchase. "
            "The customer must include the order number and a short reason."
        ),
        metadata={"type": "policy", "lang": "en", "team": "support"},
    ),
    Doc(
        doc_id="policy_termination",
        text=(
            "Termination requires 30 days written notice unless there is a material breach. "
            "Notice must contain the effective date and authorized signature."
        ),
        metadata={"type": "policy", "lang": "en", "team": "legal"},
    ),
    Doc(
        doc_id="faq_password",
        text=(
            "If a user forgot a password, verify identity and send the password reset flow. "
            "Do not share credentials over email."
        ),
        metadata={"type": "faq", "lang": "en", "team": "support"},
    ),
    Doc(
        doc_id="dk_ops_logs",
        text=(
            "Log prompts, modelnavn, latency og fejl. Gem versionsnummer for prompt og retrieval "
            "sa det er muligt at reproducere svar."
        ),
        metadata={"type": "ops", "lang": "da", "team": "platform"},
    ),
    Doc(
        doc_id="billing_invoice",
        text=(
            "Invoices are due within 14 days. Disputed billing items must be reported within "
            "10 business days after invoice date."
        ),
        metadata={"type": "billing", "lang": "en", "team": "finance"},
    ),
]


QUERIES: list[str] = [
    "How long do customers have to request a refund?",
    "What should a termination notice contain?",
    "Hvad skal vi logge i en LLM loesning?",
]


SYNONYM_GROUPS: list[set[str]] = [
    {"refund", "reimbursement", "moneyback"},
    {"terminate", "termination", "cancel", "cancellation"},
    {"invoice", "billing", "bill"},
    {"password", "credential", "login"},
    {"log", "logging", "logs"},
    {"latency", "response_time"},
    {"modelnavn", "model", "model_name"},
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def shorten(text: str, max_len: int = 90) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= max_len else one_line[: max_len - 3] + "..."


def sparse_vector(text: str) -> Counter[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "is",
        "are",
        "if",
        "a",
        "og",
        "det",
        "den",
        "de",
        "i",
        "pa",
        "sa",
        "vi",
        "en",
        "et",
        "skal",
    }
    return Counter(t for t in tokenize(text) if t not in stop and len(t) > 1)


def _canonical_token(token: str) -> str:
    for group in SYNONYM_GROUPS:
        if token in group:
            return sorted(group)[0]
    return token


def _hash_unit(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0


def dense_embedding(
    text: str,
    dim: int = 24,
    use_synonyms: bool = True,
    use_char_ngrams: bool = False,
    normalize: bool = True,
) -> list[float]:
    vec = [0.0] * dim
    toks = tokenize(text)
    for i, tok in enumerate(toks):
        key = _canonical_token(tok) if use_synonyms else tok
        base = 1.0 + min(i, 8) * 0.03
        for d in range(dim):
            vec[d] += base * _hash_unit(f"{key}|{d}")
    if use_char_ngrams:
        compact = "".join(toks)
        for n in (3, 4):
            for i in range(max(0, len(compact) - n + 1)):
                gram = compact[i : i + n]
                for d in range(dim):
                    vec[d] += 0.25 * _hash_unit(f"ng:{gram}|{d}")
    if normalize:
        vec = l2_normalize(vec)
    return vec


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec[:]
    return [v / norm for v in vec]


def cosine_dense(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def dot_dense(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_sparse(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve_sparse(query: str, docs: list[Doc] | None = None, top_k: int = 3) -> list[Hit]:
    docs = docs or DOCS
    qv = sparse_vector(query)
    hits = [Hit(d, cosine_sparse(qv, sparse_vector(d.text))) for d in docs]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def retrieve_dense(
    query: str,
    docs: list[Doc] | None = None,
    top_k: int = 3,
    score_fn: Callable[[list[float], list[float]], float] = cosine_dense,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[Hit]:
    docs = docs or DOCS
    embed = embed_fn or (lambda t: dense_embedding(t))
    qv = embed(query)
    hits = [Hit(d, score_fn(qv, embed(d.text))) for d in docs]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def reciprocal_rank_fusion(result_sets: list[list[Hit]], k: int = 60) -> list[Hit]:
    by_id = {d.doc_id: d for d in DOCS}
    scores: dict[str, float] = {}
    for hits in result_sets:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.doc.doc_id] = scores.get(hit.doc.doc_id, 0.0) + 1.0 / (k + rank)
    out = [Hit(by_id[doc_id], score) for doc_id, score in scores.items()]
    out.sort(key=lambda h: h.score, reverse=True)
    return out


def print_hits(title: str, hits: list[Hit]) -> None:
    print(title)
    for i, hit in enumerate(hits, start=1):
        meta = f"type={hit.doc.metadata.get('type')} lang={hit.doc.metadata.get('lang')}"
        print(f"{i}. score={hit.score:.4f} {hit.doc.doc_id} ({meta})")
        print(f"   {shorten(hit.doc.text)}")
    if not hits:
        print("(no hits)")
    print()


def fake_latency_ms(dim: int, top_k: int, rerank: bool = False) -> float:
    base = 8 + dim * 0.03 + top_k * 1.5
    if rerank:
        base += top_k * 6
    return round(base, 1)


def fake_cost_usd(num_chunks: int, avg_tokens: int, price_per_1k_tokens: float) -> float:
    return round((num_chunks * avg_tokens / 1000.0) * price_per_1k_tokens, 4)


def env_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def time_call(fn: Callable[[], object]) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0

