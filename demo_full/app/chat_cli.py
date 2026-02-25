from __future__ import annotations

try:
    from app.rag_chain import agent_answer, format_retrieval_score_lines
except ModuleNotFoundError:
    from rag_chain import agent_answer, format_retrieval_score_lines


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

        try:
            score_lines = format_retrieval_score_lines(question)
        except Exception as exc:
            score_lines = [f"(kunne ikke hente retrieval scores: {exc})"]

        print("\nRetrieved chunks:")
        for line in score_lines:
            print(line)

        answer = agent_answer(question)
        print("\n" + answer)


if __name__ == "__main__":
    main()

