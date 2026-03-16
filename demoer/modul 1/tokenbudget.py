from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, DIM, GREEN, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

MAX_CONTEXT_WINDOW = 800


def approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"[a-zA-Z0-9_]+", text)))


def print_budget(title: str, parts: dict[str, str]) -> None:
    print_section(title)
    print_label("PROMPT-DELE", BLUE)

    total = 0
    for name, text in parts.items():
        count = approx_tokens(text)
        total += count
        print(f"- {name:<22} {count:>4} tokens")

    print_label("OPSUMMERING", GREEN)
    print(f"Samlet promptbudget: {total} tokens")
    print(f"Maks context window: {MAX_CONTEXT_WINDOW} tokens")

    if total > MAX_CONTEXT_WINDOW:
        print(colorize("Status: Over budget", YELLOW))
    else:
        print(colorize("Status: Inden for budget", GREEN))


def main() -> None:
    clear_screen()
    oversized_prompt = {
        "System prompt": (
            "Du er en teknisk assistent, der svarer grundigt, korrekt og struktureret. "
            "Du skal forklare som til en erfaren udvikler og gerne inkludere ekstra nuancer."
        ),
        "Instruktioner": (
            "Svar i flere trin. Brug fagbegreber korrekt. Giv eksempler. "
            "Naevn tradeoffs, risici og mulige alternativer."
        ),
        "Conversation memory": (
            "Bruger: Hvad er embeddings? "
            "Assistent: Embeddings er numeriske vektorer brugt til semantisk lighed. "
            "Bruger: Hvordan bruges de i retrieval? "
            "Assistent: De bruges til at finde relevante chunks. "
            "Bruger: Hvad er chunking? "
            "Assistent: Chunking opdeler dokumenter i mindre bidder. "
            "Bruger: Hvorfor er overlap nyttigt? "
            "Assistent: Overlap beskytter information ved chunk-graenser. "
            "Bruger: Hvad er problemet med for meget overlap? "
            "Assistent: Det giver flere tokens og mere redundans."
        )
        * 6,
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

    optimized_prompt = {
        "System prompt": (
            "Du er en teknisk assistent, der svarer kort og grounded."
        ),
        "Instruktioner": (
            "Svar kun med de vigtigste pointer. Hold dig til konteksten."
        ),
        "Conversation memory": (
            "Opsummering: Brugeren arbejder med en RAG-app og har allerede spurgt om embeddings, chunking og overlap."
        ),
        "Retrieved dokumenter": (
            "Dokument 1: Chunking opdeler dokumenter i mindre bidder til retrieval. "
            "Dokument 2: Overlap beskytter mod tab af kontekst, men giver flere tokens. "
            "Dokument 3: Embeddings bruges til at matche spoergsmaal med relevante chunks."
        ),
        "User question": (
            "Hvordan haenger embeddings, chunking og overlap sammen i et RAG-system?"
        ),
        "Output": "Svar i maks 3 saetninger.",
    }

    print_section("Token Budget Demo")
    print_label("POINTE", YELLOW)
    print(
        colorize(
            "Token budgetering betyder at planlaegge, hvordan de tilgaengelige tokens fordeles i prompten.",
            DIM,
        )
    )
    print(
        colorize(
            "Prompten bestaar typisk af system prompt, instruktioner, memory, retrieval, user question og output.",
            DIM,
        )
    )

    print_budget("Eksempel 1 - For stort prompt", oversized_prompt)
    prompt_to_continue("Tryk Enter for at gaa videre til det trimmede eksempel...")
    print_budget("Eksempel 2 - Trimmet prompt", optimized_prompt)


if __name__ == "__main__":
    main()
