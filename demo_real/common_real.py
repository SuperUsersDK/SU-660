from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv(override=True)


def require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def get_models() -> tuple[str, str]:
    chat_model = require_env("OPENAI_MODEL", "gpt-4.1-mini")
    embed_model = require_env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return chat_model, embed_model


def sample_documents() -> list[dict[str, Any]]:
    return [
        {
            "id": "prompting-1",
            "source": "internal/prompting-guide.md",
            "category": "prompting",
            "text": (
                "Use a clear system role, success criteria, and explicit output schema. "
                "Few-shot examples increase consistency for structured tasks."
            ),
        },
        {
            "id": "rag-1",
            "source": "internal/rag-notes.md",
            "category": "rag",
            "text": (
                "RAG combines retrieval and generation. Chunking and metadata filters influence relevance. "
                "Tune top-k carefully to avoid noisy context."
            ),
        },
        {
            "id": "ops-1",
            "source": "internal/ops-checklist.md",
            "category": "operations",
            "text": (
                "Log prompts, model settings, latency, and failures. "
                "Define fallbacks for tool errors and timeouts. Review feedback regularly."
            ),
        },
    ]


def chroma_dir() -> str:
    path = Path("demo_real/.chroma")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
