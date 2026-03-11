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

from tools.terminal_ui import BLUE, DIM, GREEN, MAGENTA, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

MODEL = "gpt-4o-mini"


def load_root_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"[a-zA-Z0-9_]+", text)))


def build_prompt_parts(memory: str) -> dict[str, str]:
    return {
        "System prompt": (
            "Du er en teknisk assistent, der svarer grundigt, korrekt og struktureret. "
            "Du skal forklare som til en erfaren udvikler og gerne inkludere ekstra nuancer."
        ),
        "Instruktioner": (
            "Svar i flere trin. Brug fagbegreber korrekt. Giv eksempler. "
            "Naevn tradeoffs, risici og mulige alternativer."
        ),
        "Conversation memory": memory,
        "Retrieved dokumenter": (
            "Dokument 1: Chunking betyder, at dokumenter deles op i mindre bidder. "
            "Hvis chunks er for store, bliver retrieval mindre praecis. "
            "Dokument 2: Overlap betyder, at nabo-chunks deler noget tekst. "
            "Det beskytter mod tab af kontekst, men oeger tokenforbrug. "
            "Dokument 3: Embeddings repraesenterer tekst som numeriske vektorer. "
            "De bruges til at matche brugerens spoergsmaal med relevante tekststykker. "
            "Dokument 4: Metadata-filtre kan forbedre precision ved at begraense hvilke dokumenter der soeges i."
        )
        * 6,
        "User question": (
            "Hvordan haenger embeddings, chunking, overlap og metadata sammen i et RAG-system?"
        ),
        "Output": "Svar i op til 8 saetninger med eksempler og opsummering.",
    }


def render_prompt(parts: dict[str, str]) -> str:
    ordered_names = [
        "System prompt",
        "Instruktioner",
        "Conversation memory",
        "Retrieved dokumenter",
        "User question",
        "Output",
    ]
    sections = [f"{name}:\n{parts[name]}" for name in ordered_names]
    return "\n\n".join(sections)


def total_tokens(parts: dict[str, str]) -> int:
    return sum(approx_tokens(text) for text in parts.values())


def compress_memory(client: OpenAI, memory: str, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du forkorter conversation memory til brug i en prompt. "
                    "Behold kun de detaljer, der er relevante for at besvare det afsluttende spoergsmaal. "
                    "Svar paa dansk i hoejst 45 ord."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Slutspoergsmaal:\n{question}\n\n"
                    f"Conversation memory:\n{memory}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def answer_with_prompt(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Du er en teknisk assistent. Svar kort, klart og fagligt paa dansk.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    clear_screen()
    load_root_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in repo root .env")

    client = OpenAI(api_key=api_key)

    original_memory = (
        "Bruger: Hvad er embeddings? "
        "Assistent: Embeddings er numeriske vektorer brugt til semantisk lighed. "
        "Bruger: Hvordan bruges de i retrieval? "
        "Assistent: De bruges til at finde relevante chunks. "
        "Bruger: Hvad er chunking? "
        "Assistent: Chunking opdeler dokumenter i mindre bidder. "
        "Bruger: Hvorfor er overlap nyttigt? "
        "Assistent: Overlap beskytter information ved chunk-graenser. "
        "Bruger: Hvad er problemet med for meget overlap? "
        "Assistent: Det giver flere tokens og mere redundans. "
    ) * 6

    original_parts = build_prompt_parts(original_memory)
    original_prompt = render_prompt(original_parts)

    print_section("Token Budget 2")
    print_label("MODEL", YELLOW)
    print(colorize(MODEL, DIM))
    print_label("OPRINDELIG PROMPT", BLUE)
    print(colorize(original_prompt, DIM))

    prompt_to_continue("Tryk Enter for at forkorte Conversation memory...")

    shortened_memory = compress_memory(
        client,
        original_parts["Conversation memory"],
        original_parts["User question"],
    )
    shortened_parts = build_prompt_parts(shortened_memory)

    print_section("Forkortet Memory, via OpenAI")
    print_label("NY CONVERSATION MEMORY", MAGENTA)
    print(colorize(shortened_memory, GREEN))

    print_label("TOKEN SAMMENLIGNING", YELLOW)
    print(
        f"Conversation memory foer: {approx_tokens(original_parts['Conversation memory'])} tokens"
    )
    print(
        f"Conversation memory nu:   {approx_tokens(shortened_parts['Conversation memory'])} tokens"
    )
    print(f"Samlet prompt foer:       {total_tokens(original_parts)} tokens")
    print(f"Samlet prompt nu:         {total_tokens(shortened_parts)} tokens")

    prompt_to_continue("Tryk Enter for at koere den nye prompt...")

    new_prompt = render_prompt(shortened_parts)
    answer = answer_with_prompt(client, new_prompt)

    print_section("Svar Fra Ny Prompt")
    print_label("NY PROMPT", BLUE)
    print(colorize(new_prompt, DIM))
    print_label("SVAR", GREEN)
    print(colorize(answer, GREEN))


if __name__ == "__main__":
    main()
