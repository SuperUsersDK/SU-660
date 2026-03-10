from __future__ import annotations

import re
from terminal_ui import BLUE, BOLD, DIM, GREEN, MAGENTA, YELLOW, colorize, print_label, print_section, prompt_to_continue

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
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def render_prompt(question: str, context_blocks: list[str] | None = None) -> str:
    if not context_blocks:
        return f"Svar paa spoergsmaalet:\n\n{question}"

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


def simulate_llm_without_rag(question: str) -> str:
    if "overlap" in question.lower():
        return (
            "Overlap bruges ofte i chunking for at bevare sammenhaeng mellem tekststykker. "
            "Det kan vaere nyttigt, men svaret er lidt generelt uden adgang til konkret kontekst."
        )

    return "Det er et generelt svar uden grounding i konkrete dokumenter."


def extract_grounded_answer(question: str, retrieved_documents: list[dict[str, str]]) -> tuple[str, list[str]]:
    question_tokens = content_tokens(question)
    supporting_sentences: list[tuple[int, str]] = []
    sources: list[str] = []

    for document in retrieved_documents:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", document["text"]) if sentence.strip()]
        matched = False

        for sentence in sentences:
            sentence_overlap = len(question_tokens.intersection(content_tokens(sentence)))
            title_overlap = len(question_tokens.intersection(content_tokens(document["title"])))
            score = sentence_overlap + (2 * title_overlap)
            if score > 0:
                supporting_sentences.append((score, sentence))
                matched = True

        if matched:
            sources.append(document["title"])

    if not supporting_sentences:
        return "Ikke fundet i konteksten.", []

    supporting_sentences.sort(key=lambda item: item[0], reverse=True)
    answer = " ".join(sentence for _, sentence in supporting_sentences[:2])
    return answer, sources


def simulate_llm_with_rag(question: str, retrieved_documents: list[dict[str, str]]) -> str:
    grounded_answer, sources = extract_grounded_answer(question, retrieved_documents)

    if not sources:
        return "Ikke fundet i konteksten."

    return (
        f"{grounded_answer}\n"
        f"{colorize('Kilder:', BOLD, MAGENTA)} {', '.join(sources)}"
    )
def wait_for_next_step() -> None:
    prompt_to_continue("Tryk Enter for at gaa videre til delen med RAG...")


def main() -> None:
    question = "Hvorfor bruger man overlap i chunking, og hvad er ulempen?"

    print_section("Prompt uden RAG")
    no_rag_prompt = render_prompt(question)
    print_label("PROMPT", BLUE)
    print(colorize(no_rag_prompt, DIM))
    print_label("SVAR", GREEN)
    print(colorize(simulate_llm_without_rag(question), GREEN))
    wait_for_next_step()

    print_section("Prompt med simuleret RAG")
    retrieved_documents = retrieve_context(question, top_k=2)
    rag_prompt = render_prompt(
        question,
        context_blocks=[document["text"] for document in retrieved_documents],
    )
    print_label("PROMPT", BLUE)
    print(colorize(rag_prompt, DIM))
    print_label("RETRIEVED KONTEKST", YELLOW)
    for document in retrieved_documents:
        title = colorize(document["title"], BOLD, YELLOW)
        print(f"- {title}: {document['text']}")
    print_label("SVAR", GREEN)
    print(colorize(simulate_llm_with_rag(question, retrieved_documents), GREEN))


if __name__ == "__main__":
    main()
