from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_DEMO_FULL_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT_DIR = Path(__file__).resolve().parents[2]

# Load local demo_full env first, then repo-root .env second.
# Both use override=True so .env values replace stale shell vars; root wins because it is loaded last.
load_dotenv(_DEMO_FULL_DIR / ".env", override=True)
load_dotenv(_REPO_ROOT_DIR / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "pdf_demo")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")


SETTINGS = Settings()
