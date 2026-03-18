from __future__ import annotations

from rag_chain import format_retrieval_score_lines, rag_answer
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import YELLOW, colorize


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

        answer = rag_answer(question)
        print("\n" + colorize(answer, YELLOW))


if __name__ == "__main__":
    main()

