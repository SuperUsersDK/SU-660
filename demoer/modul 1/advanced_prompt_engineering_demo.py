from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_root_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def approx_tokens(text: str) -> int:
    return max(1, len(tokenize(text)))


def load_knowledge_base() -> list[dict[str, str]]:
    documents = []

    for path in sorted(DATA_DIR.glob("*.md")):
        documents.append(
            {
                "id": path.stem,
                "source": path.name,
                "text": path.read_text(encoding="utf-8").strip(),
            }
        )

    return documents


def retrieve_context(question: str, documents: list[dict[str, str]], top_k: int = 2) -> list[dict[str, str]]:
    question_tokens = set(tokenize(question))
    scored = []

    for document in documents:
        doc_tokens = tokenize(document["text"])
        overlap = len(question_tokens.intersection(doc_tokens))
        scored.append((overlap, approx_tokens(document["text"]), document))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [document for score, _, document in scored if score > 0][:top_k]


def render_prompt(
    system_role: str,
    question: str,
    context_blocks: list[str] | None = None,
    history: list[str] | None = None,
    extra_rules: list[str] | None = None,
    max_sentences: int | None = None,
    output_schema: str | None = None,
) -> str:
    parts = [f"SYSTEM:\n{system_role}"]

    if extra_rules:
        parts.append("REGLER:\n" + "\n".join(f"- {rule}" for rule in extra_rules))

    if history:
        parts.append("MEMORY:\n" + "\n".join(history))

    if context_blocks:
        context = "\n\n".join(f"[Kontekst {index + 1}] {block}" for index, block in enumerate(context_blocks))
        parts.append("KONTEKST:\n" + context)

    parts.append(f"SPOERGSMAAL:\n{question}")

    if max_sentences is not None:
        parts.append(f"SVARFORMAT:\nSvar i maks {max_sentences} saetninger.")

    if output_schema:
        parts.append(f"OUTPUT:\n{output_schema}")

    return "\n\n".join(parts)


def call_openai(client: OpenAI, prompt: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Du er en teknisk assistent. Foelg brugerens prompt noejagtigt og svar paa dansk.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content or ""


def call_openai_json(client: OpenAI, prompt: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rag_explanation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "begreb": {"type": "string"},
                        "forklaring": {"type": "string"},
                        "anvendelse_i_rag": {"type": "string"},
                        "noegleord": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": [
                        "begreb",
                        "forklaring",
                        "anvendelse_i_rag",
                        "noegleord",
                        "sources",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Du er en teknisk assistent. Returner kun gyldig JSON, der matcher schemaet noejagtigt."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content or ""


def print_prompt_and_response(prompt: str, response: str) -> None:
    print_label("PROMPT", BLUE)
    print(colorize(prompt, DIM))
    print_label("SVAR", GREEN)
    print(colorize(response, GREEN))


def pretty_json_if_possible(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def wait_for_next_step() -> None:
    prompt_to_continue("Tryk Enter for at gaa videre til naeste del...")


def build_long_history() -> list[str]:
    return [
        "Bruger: Hvad er embeddings?",
        "Assistent: Embeddings er numeriske vektorer, som bruges til semantisk sammenligning.",
        "Bruger: Hvordan bruges de i retrieval?",
        "Assistent: De bruges til at finde tekst, der ligner spoergsmaalets betydning.",
        "Bruger: Hvad er chunking?",
        "Assistent: Chunking deler dokumenter i mindre bidder.",
        "Bruger: Hvorfor er overlap nyttigt?",
        "Assistent: Overlap beskytter vigtig information ved chunk-graenser.",
        "Bruger: Hvad er problemet med for meget overlap?",
        "Assistent: Det giver flere tokens og mere redundans.",
    ]


def summarize_history(history: list[str]) -> str:
    return (
        "Brugeren bygger en RAG-applikation og har allerede spurgt om embeddings, "
        "chunking, retrieval og overlap. Assistenten skal bevare teknisk kontekst "
        "uden at inkludere hele samtalehistorikken."
    )


def token_budget_report(parts: dict[str, str]) -> dict[str, int]:
    return {name: approx_tokens(text) for name, text in parts.items()}


def print_budget_report(before: dict[str, int], after: dict[str, int]) -> None:
    print_label("TOKEN BUDGET", CYAN)
    print(colorize("Foer trimming:", BOLD, YELLOW))
    for name, count in before.items():
        print(f"- {name}: {count}")
    print(f"- total: {sum(before.values())}")

    print(colorize("\nEfter trimming:", BOLD, YELLOW))
    for name, count in after.items():
        print(f"- {name}: {count}")
    print(f"- total: {sum(after.values())}")


def demo() -> None:
    clear_screen()
    load_root_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in repo root .env")

    client = OpenAI(api_key=api_key)
    documents = load_knowledge_base()
    question = "Hvad er chunking, og hvad betyder overlap?"
    follow_up = "Hvordan bruges de i RAG?"

    print_section("Del 1 - Statisk prompt")
    static_prompt = f"Svar paa spoergsmaalet:\n\n{question}"
    static_response = call_openai(client, static_prompt)
    print_prompt_and_response(static_prompt, static_response)
    wait_for_next_step()

    print_section("Del 2 - Statisk prompt med regler")
    improved_static_prompt = render_prompt(
        system_role="Du er en teknisk assistent for Python-udviklere.",
        question=question,
        extra_rules=[
            "Svar kort og praecist",
            "Forklar som til en Python-udvikler",
            "Hvis du ikke ved det, sig det",
        ],
        max_sentences=3,
    )
    improved_static_response = call_openai(client, improved_static_prompt)
    print_prompt_and_response(improved_static_prompt, improved_static_response)
    wait_for_next_step()

    print_section("Del 3 - Dynamisk prompt")
    retrieved = retrieve_context(question, documents, top_k=2)
    dynamic_prompt = render_prompt(
        system_role="Du er ekspert i RAG og LLM-udvikling.",
        question=question,
        context_blocks=[doc["text"] for doc in retrieved],
        extra_rules=[
            "Svar baseret paa konteksten",
            "Brug faglige termer korrekt",
        ],
        max_sentences=4,
    )
    dynamic_response = call_openai(client, dynamic_prompt)
    print_prompt_and_response(dynamic_prompt, dynamic_response)
    wait_for_next_step()

    print_section("Del 4 - Memory styring")
    no_memory_prompt = render_prompt(
        system_role="Du er en teknisk assistent.",
        question=follow_up,
        max_sentences=2,
    )
    no_memory_response = call_openai(client, no_memory_prompt)
    print(colorize("\nUden memory", BOLD, YELLOW))
    print_prompt_and_response(no_memory_prompt, no_memory_response)

    full_history = build_long_history()
    with_memory_prompt = render_prompt(
        system_role="Du er en teknisk assistent.",
        question=follow_up,
        history=full_history[-4:],
        context_blocks=[doc["text"] for doc in retrieved],
        max_sentences=3,
    )
    with_memory_response = call_openai(client, with_memory_prompt)
    print(colorize("\nMed memory", BOLD, YELLOW))
    print_prompt_and_response(with_memory_prompt, with_memory_response)
    wait_for_next_step()

    print_section("Del 5 - Token budgetering")
    oversized_context = [doc["text"] for doc in retrieve_context("Hvad er embeddings og chunking i RAG med overlap?", documents, top_k=4)]
    budget_before_parts = {
        "system": "Du er en teknisk assistent, der skal svare grundigt med eksempler, kilder og forklaringer.",
        "memory": "\n".join(full_history),
        "context": "\n\n".join(oversized_context),
        "question": "Hvordan haenger embeddings, chunking og overlap sammen i RAG?",
        "output_budget": "Svar i op til 8 saetninger og inkluder gerne ekstra forklaring.",
    }
    summary_memory = summarize_history(full_history)
    trimmed_context = oversized_context[:2]
    budget_after_parts = {
        "system": "Du er en teknisk assistent, der svarer kort og grounded.",
        "memory": summary_memory,
        "context": "\n\n".join(trimmed_context),
        "question": "Hvordan haenger embeddings, chunking og overlap sammen i RAG?",
        "output_budget": "Svar i maks 3 saetninger.",
    }
    print_budget_report(token_budget_report(budget_before_parts), token_budget_report(budget_after_parts))

    budget_prompt = render_prompt(
        system_role=budget_after_parts["system"],
        question=budget_after_parts["question"],
        history=[summary_memory],
        context_blocks=trimmed_context,
        extra_rules=["Hold dig inden for token-budgettet", "Prioriter kun den mest relevante kontekst"],
        max_sentences=3,
    )
    budget_response = call_openai(client, budget_prompt)
    print_prompt_and_response(budget_prompt, budget_response)
    wait_for_next_step()

    print_section("Del 6 - Kontrol over output")
    json_question = "Forklar embeddings i RAG."
    json_context = [doc["text"] for doc in retrieve_context(json_question, documents, top_k=2)]
    controlled_prompt = render_prompt(
        system_role="Du er en teknisk assistent.",
        question=json_question,
        context_blocks=json_context,
        extra_rules=[
            "Svar kun ud fra konteksten",
            "Hvis svaret ikke findes, returner en fallback",
        ],
        output_schema=(
            'Returner JSON med felterne: "begreb", "forklaring", '
            '"anvendelse_i_rag", "noegleord", "sources".'
        ),
    )
    controlled_response = pretty_json_if_possible(call_openai_json(client, controlled_prompt))
    print_prompt_and_response(controlled_prompt, controlled_response)


if __name__ == "__main__":
    demo()
