"""Download the classroom tokenizer data once, before running offline examples."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunking.tokenizer_cache import configure_tokenizer_cache


def main() -> int:
    cache_dir = configure_tokenizer_cache()
    if not cache_dir:
        print("Caching er slået fra. Sæt TIKTOKEN_CACHE_DIR til en mappe før warmup.", file=sys.stderr)
        return 1
    try:
        import tiktoken

        print(f"Tokenizer-cache: {cache_dir}", flush=True)
        for name in ("o200k_base", "cl100k_base"):
            encoding = tiktoken.get_encoding(name)
            sample = "Chunking med overlap bevarer kontekst: æ, ø og å."
            assert encoding.decode(encoding.encode(sample)) == sample
            print(f"Klar: {name}", flush=True)
    except Exception as exc:
        print(f"Warmup fejlede: {exc}", file=sys.stderr)
        print("Aktivér projektets miljø og kør med internetadgang før undervisningen.", file=sys.stderr)
        return 1
    print("Tokenizer-data er klar til lokal brug. Der er ikke foretaget model-API-kald.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
