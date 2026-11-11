"""Use the same persistent tokenizer cache for classroom preparation and demos."""

from __future__ import annotations

import os
from pathlib import Path


def configure_tokenizer_cache() -> str:
    cache_dir = os.environ.get(
        "TIKTOKEN_CACHE_DIR",
        os.environ.get("DATA_GYM_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache" / "tiktoken")),
    )
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return cache_dir
