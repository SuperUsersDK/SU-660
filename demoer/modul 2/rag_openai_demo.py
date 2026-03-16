from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, DIM, GREEN, MAGENTA, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

STOPWORDS = {
    "hvad",
    "hvorfor",
    "hvordan",
    "man",
    "og",
    "i",
    "er",
    "det",
    "de",
    "den",
    "der",
    "for",
    "til",
    "en",
    "et",
    "at",
    "bruges",
    "bruger",
}

KNOWLEDGE_BASE = [
    {
        "id": "chunking_basics",
        "title": "Chunking til RAG",
        "text": (
            "Chunking betyder, at lange dokumenter deles op i mindre bidder. "
            "Hvis chunks er for store, bliver retrieval ofte mindre praecis. "
            "Hvis chunks er for smaa, mister man vigtig kontekst."
        ),
    },
    {
        "id": "chunk_overlap",
        "title": "Overlap i chunking",
        "text": (
            "Overlap betyder, at to nabo-chunks deler noget af den samme tekst. "
            "Det beskytter mod, at vigtig information forsvinder ved chunk-graenser. "
            "For meget overlap giver dog flere tokens, mere redundans og dyrere retrieval."
        ),
    },
    {
        "id": "embeddings_intro",
        "title": "Embeddings i RAG",
        "text": (
            "Embeddings repraesenterer tekst som numeriske vektorer. "
            "I RAG bruges de til at matche brugerens spoergsmaal med relevante chunks."
        ),
    },
    {
        "id": "metadata_filters",
        "title": "Metadata og retrieval",
        "text": (
            "Metadata kan bruges til at afgraense retrieval til bestemte dokumenttyper, "
            "versioner eller afdelinger. Det forbedrer ofte precision."
        ),
    },
]


def load_root_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def render_prompt(question: str, context_blocks: list[str] | None = None) -> str:
    if not context_blocks:
        return (
            "Du er en teknisk assistent.\n\n"
            "Svar paa spoergsmaalet saa godt du kan.\n\n"
            f"Spoergsmaal:\n{question}\n\n"
            "Svar kort og konkret."
        )

    joined_context = "\n\n".join(
        f"[Kontekst {index + 1}] {block}" for index, block in enumerate(context_blocks)
    )
    return (
        "Du er en teknisk assistent.\n\n"
        "Svar kun ud fra konteksten.\n\n"
        f"Kontekst:\n{joined_context}\n\n"
        f"Spoergsmaal:\n{question}\n\n"
        "Svar kort og konkret."
    )


def retrieve_context(question: str, top_k: int = 2) -> list[dict[str, str]]:
    question_tokens = content_tokens(question)
    scored_documents: list[tuple[int, dict[str, str]]] = []

    for document in KNOWLEDGE_BASE:
        text_overlap = len(question_tokens.intersection(content_tokens(document["text"])))
        title_overlap = len(question_tokens.intersection(content_tokens(document["title"])))
        overlap = text_overlap + (2 * title_overlap)
        scored_documents.append((overlap, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return [document for score, document in scored_documents if score > 0][:top_k]


def call_openai(client: OpenAI, prompt: str) -> str:
    model = os.getenv("OPENAI_MODEL", MODEL)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Du er en teknisk assistent. Svar kort, fagligt og paa dansk.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content or ""


def wait_for_next_step() -> None:
    prompt_to_continue("Tryk Enter for at gaa videre til delen med RAG...")


def main() -> None:
    clear_screen()
    load_root_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in repo root .env")

    client = OpenAI(api_key=api_key)
    question = "Hvorfor bruger man overlap i chunking, og hvad er ulempen?"

    print_section("Prompt uden RAG")
    no_rag_prompt = render_prompt(question)
    print_label("MODEL", YELLOW)
    print(colorize(os.getenv("OPENAI_MODEL", MODEL), DIM))
    print_label("PROMPT", BLUE)
    print(colorize(no_rag_prompt, DIM))
    print_label("SVAR", GREEN)
    print(colorize(call_openai(client, no_rag_prompt), GREEN))
    wait_for_next_step()

    print_section("Prompt med simuleret RAG")
    retrieved_documents = retrieve_context(question, top_k=2)
    rag_prompt = render_prompt(
        question,
        context_blocks=[document["text"] for document in retrieved_documents],
    )
    print_label("MODEL", YELLOW)
    print(colorize(os.getenv("OPENAI_MODEL", MODEL), DIM))
    print_label("PROMPT", BLUE)
    print(colorize(rag_prompt, DIM))
    print_label("RETRIEVED KONTEKST", YELLOW)
    for document in retrieved_documents:
        title = colorize(document["title"], BOLD, YELLOW)
        print(f"- {title}: {document['text']}")
    print_label("SVAR", GREEN)
    print(colorize(call_openai(client, rag_prompt), GREEN))
    print(colorize(f"\nKilder: {', '.join(document['title'] for document in retrieved_documents)}", BOLD, MAGENTA))


if __name__ == "__main__":
    main()
