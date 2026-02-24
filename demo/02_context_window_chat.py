from common import approx_tokens, simple_chat_reply


def main() -> None:
    messages = [
        {"role": "system", "content": "Du er en hjaelpsom kursusassistent for et GenAI-kursus."},
        {"role": "user", "content": "Vi planlaegger dag 1. Hvad skal deltagerne laere om prompts?"},
        {"role": "assistant", "content": "Fokus pa roller, constraints, few-shot og outputformat."},
        {
            "role": "user",
            "content": (
                "Godt. Tilfoj ogsa memory-styring, token budgeting, truncation, "
                "og hvordan man undgar over-kontekst i samtaler."
            ),
        },
        {
            "role": "assistant",
            "content": "Det giver mening. Vi kan vise et kontrolleret kontekstvindue i Python.",
        },
        {
            "role": "user",
            "content": (
                "Lav et kort forslag til demo-flow og husk at svaret skal vaere konkret "
                "og have 3 punkter."
            ),
        },
    ]

    print("=== Input-samtale ===")
    for msg in messages:
        print(f"{msg['role']:>9}: ({approx_tokens(msg['content'])} tok) {msg['content']}")

    result = simple_chat_reply(messages, max_context_tokens=45)

    print("\n=== Beholdte beskeder inden for token-budget ===")
    for msg in result["kept_messages"]:
        print(f"{msg['role']:>9}: {msg['content']}")
    print(f"Samlet token-estimat: {result['token_count']}")

    print("\n=== Mock model-svar ===")
    print(result["reply"])


if __name__ == "__main__":
    main()

