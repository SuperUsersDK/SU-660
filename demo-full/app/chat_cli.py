from __future__ import annotations

try:
    from app.rag_chain import agent_answer
except ModuleNotFoundError:
    from rag_chain import agent_answer


def main() -> None:
    print("RAG demo. Skriv 'exit' for at stoppe.")
    while True:
        try:
            question = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.strip().lower() in {"exit", "quit"}:
            break
        if not question.strip():
            continue

        answer = agent_answer(question)
        print("\n" + answer)


if __name__ == "__main__":
    main()

