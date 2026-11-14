"""Embeddings til chunk-laboratoriet.

Standard er en lokal, flersproget model (intfloat/multilingual-e5-small, 384 dimensioner,
op til 512 tokens). Den kører offline efter første download og kan dansk. Sæt
CHUNK_LAB_EMBEDDINGS=openai i .env for at bruge OPENAI_EMBED_MODEL i stedet.

e5-modellerne er trænet med præfikser: tekster gemmes som "passage: ..." og spørgsmål
som "query: ...". Uden præfikserne bliver søgningen mærkbart dårligere.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

LOCAL_MODEL = "intfloat/multilingual-e5-small"
_LOCK = threading.Lock()


@dataclass(frozen=True)
class EmbedderInfo:
    provider: str
    model: str
    dims: int
    max_tokens: int | None


def provider() -> str:
    return os.getenv("CHUNK_LAB_EMBEDDINGS", "local").strip().lower() or "local"


@lru_cache(maxsize=1)
def _local():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModel.from_pretrained(LOCAL_MODEL).eval()
    return tokenizer, model, torch


def _load_local():
    # Webappen kører forespørgsler i tråde; to samtidige indlæsninger fejler i PyTorch.
    with _LOCK:
        return _local()


def info() -> EmbedderInfo:
    if provider() == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return EmbedderInfo("openai", model, 3072 if "large" in model else 1536, 8191)
    return EmbedderInfo("local", LOCAL_MODEL, 384, 512)


def _embed_local(texts: list[str]) -> list[list[float]]:
    tokenizer, model, torch = _load_local()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 32):
        batch = tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # gennemsnit over tokens
        vectors += torch.nn.functional.normalize(pooled, dim=-1).tolist()
    return vectors


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    response = OpenAI().embeddings.create(model=info().model, input=texts)
    return [item.embedding for item in response.data]


def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if provider() == "openai":
        return _embed_openai(texts)
    return _embed_local([f"passage: {text}" for text in texts])


def embed_query(text: str) -> list[float]:
    if provider() == "openai":
        return _embed_openai([text])[0]
    return _embed_local([f"query: {text}"])[0]


def token_counts(texts: list[str]) -> list[int] | None:
    """Antal tokens pr. tekst, så appen kan advare om chunks, modellen skærer af."""
    if provider() == "openai":
        return None
    tokenizer, _, _ = _load_local()
    return [len(tokenizer(f"passage: {text}")["input_ids"]) for text in texts]
