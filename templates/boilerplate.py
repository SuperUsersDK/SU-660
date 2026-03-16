from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for path in [current.parent, *current.parents]:
        if (path / ".env.example").exists() or (path / ".gitignore").exists():
            return path
    raise RuntimeError("Kunne ikke finde repo-roden.")


def load_root_env() -> Path:
    root_dir = find_repo_root()
    load_dotenv(root_dir / ".env", override=True)
    return root_dir


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in repo-root .env")
    return OpenAI(api_key=api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OpenAI boilerplate")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to OpenAI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = load_root_env()
    client = build_client()
    prompt = " ".join(args.prompt).strip() or "hvad er RAG?"

    print(f"Repo root: {root_dir}")
    print("OpenAI client er klar.")
    print(f"Prompt: {prompt}")

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content or ""
    print()
    print("Eksempelsvar:")
    print(answer)

    # ----------
    # Skriv din løsning her
    #
    # Eksempel:
    # 1. Lav en prompt
    # 2. Kald modellen via `client`
    # 3. Udskriv svaret
    # ----------
    _ = client


if __name__ == "__main__":
    main()
