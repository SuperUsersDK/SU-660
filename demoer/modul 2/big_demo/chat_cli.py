from __future__ import annotations

from rag_chain import agent_answer, format_retrieval_score_lines


def main() -> None:
    print("RAG demo for markdown, kode og PDF. Skriv 'exit' for at stoppe.")
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

        try:
            score_lines = format_retrieval_score_lines(question)
        except Exception as exc:
            score_lines = [f"(kunne ikke hente retrieval scores: {exc})"]

        print("\nRetrieved chunks fra knowledge base:")
        for line in score_lines:
            print(line)

        answer = agent_answer(question)
        print("\n" + answer)


if __name__ == "__main__":
    main()

