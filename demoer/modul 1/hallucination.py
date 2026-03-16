from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, DIM, GREEN, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

MODEL = "gpt-4o-mini"


def load_root_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def print_prompt(system_prompt: str, user_prompt: str) -> None:
    print_label("PROMPT", BLUE)
    print(colorize("System:", YELLOW))
    print(system_prompt)
    print()
    print(colorize("User:", YELLOW))
    print(user_prompt)


def main() -> None:
    clear_screen()
    load_root_env()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in repo root .env")

    client = OpenAI(api_key=api_key)
    question = "Hvad er hovedresultaterne i Anders Moellers artikel fra 2019 'Quantum Governance in Nordic Municipalities'?"
    naive_system_prompt = (
        "Du er en hjaelpsom assistent. Svar kort, klart og konkret paa dansk. "
        "Giv gerne specifikke detaljer, hvis det er relevant."
    )

    print_section("Hallucination Demo 1")
    print_label("MODEL", YELLOW)
    print(colorize(MODEL, DIM))
    print_label("SPOERGSMAAL", BLUE)
    print(colorize(question, DIM))
    print_prompt(naive_system_prompt, question)

    naive_response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": naive_system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    naive_answer = naive_response.choices[0].message.content or ""
    print_label("SVAR", GREEN)
    print(colorize(naive_answer, GREEN))
    print(colorize("\nUndervisningspoint: Svarer modellen selvsikkert paa noget, der maaske ikke findes?", YELLOW))

    prompt_to_continue("Tryk Enter for at gaa videre til den forsigtige version...")

    print_section("Hallucination Demo 2")
    print_label("MODEL", YELLOW)
    print(colorize(MODEL, DIM))
    print_label("SPOERGSMAAL", BLUE)
    print(colorize(question, DIM))
    careful_system_prompt = (
        "Du er en forsigtig assistent. Hvis du ikke er sikker paa, at en standard findes "
        "eller hvad den indeholder, saa sig tydeligt at du er usikker og undgaa at opfinde detaljer."
    )
    print_prompt(careful_system_prompt, question)

    careful_response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": careful_system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    careful_answer = careful_response.choices[0].message.content or ""
    print_label("SVAR", GREEN)
    print(colorize(careful_answer, GREEN))
    print(colorize("\nUndervisningspoint: Prompten kan reducere hallucination, men ikke garantere sandhed.", YELLOW))


if __name__ == "__main__":
    main()
