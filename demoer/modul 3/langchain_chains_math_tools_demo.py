from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, MAGENTA, RED, YELLOW, clear_screen, colorize, print_label, print_section


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def get_chat_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def print_status(label: str, message: str, color: str = CYAN) -> None:
    print(colorize(f"{label:<10}", BOLD, color) + " " + colorize(message, DIM))


@tool
def add_numbers(a: float, b: float) -> str:
    """Add exactly two numbers. Only perform direct addition and nothing else."""
    print_status("TOOL add", f"Lægger sammen: {a} + {b}", GREEN)
    result = a + b
    return f"[add] {a} + {b} = {result}"


@tool
def divide_numbers(a: float, b: float) -> str:
    """Divide the first number by the second. Only perform direct division. Reject division by zero."""
    print_status("TOOL div", f"Dividerer: {a} / {b}", MAGENTA)
    if b == 0:
        return "[divide] Error: division med 0 er ikke tilladt."
    result = a / b
    return f"[divide] {a} / {b} = {result}"


SYSTEM_PROMPT = """
Du er et lille LangChain chain-demo med præcis to tools: add_numbers og divide_numbers.

Regler:
- Du må aldrig selv regne noget ud i hovedet.
- Hver eneste beregning skal ske via et tool-kald.
- add_numbers må kun bruges til addition.
- divide_numbers må kun bruges til division.
- Du må ikke simulere multiplikation, subtraktion, potenser, procentregning eller andre operationer via kreative mellemregninger.
- Hvis brugeren beder om noget, der ikke kan løses med direkte addition og/eller direkte division, så afvis høfligt.
- Hvis et tool returnerer en fejl, så gengiv fejlen tydeligt.
- Svar kort på dansk.
""".strip()

EXAMPLE_QUESTIONS = [
    "Hvad er 9 plus 13?",
    "Hvad er 81 divideret med 9?",
    "Læg 20 og 4 sammen og divider derefter resultatet med 2.",
    "Hvad er 12 divideret med 0?",
    "Hvad er 6 gange 7?",
]


def build_chain() -> tuple[Any, Any, Any, list[BaseTool]]:
    tools = [add_numbers, divide_numbers]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=get_chat_model(), temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    chain = prompt | llm_with_tools
    return prompt, chain, llm_with_tools, tools


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content) if content is not None else str(message)


def run_question(question: str, prompt: Any, chain: Any, llm_with_tools: Any, tools: list[BaseTool]) -> str:
    tool_map = {tool.name: tool for tool in tools}
    messages = prompt.invoke({"question": question}).to_messages()
    response = chain.invoke({"question": question})
    messages.append(response)

    for _ in range(8):
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return _message_text(response)

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            selected_tool = tool_map.get(tool_name)
            if selected_tool is None:
                tool_result = f"[tool-error] Ukendt tool: {tool_name}"
            else:
                tool_result = selected_tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                )
            )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return "Kæden stoppede efter for mange tool-kald."


def main() -> None:
    clear_screen()
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY mangler i repo-roden .env")

    prompt, chain, llm_with_tools, tools = build_chain()

    print_section("LangChain Chain Demo")
    print_label("WORKFLOW", CYAN)
    print(colorize("User", DIM))
    print(colorize("↓", DIM))
    print(colorize("Prompt | ChatOpenAI.bind_tools([...])", DIM))
    print(colorize("↓", DIM))
    print(colorize("Tool calls", DIM))
    print(colorize("↓", DIM))
    print(colorize("Kort svar", DIM))

    print_label("TOOLS", BLUE)
    print(colorize("- add_numbers(a, b) -> kun addition", BLUE))
    print(colorize("- divide_numbers(a, b) -> kun division", BLUE))

    print_label("EKSEMPLER", YELLOW)
    for question in EXAMPLE_QUESTIONS:
        print(colorize(f"- {question}", YELLOW))

    for question in EXAMPLE_QUESTIONS:
        print_section("Ny Kørsel")
        print_label("SPØRGSMÅL", YELLOW)
        print(colorize(question, YELLOW))
        answer = run_question(question, prompt, chain, llm_with_tools, tools)
        print_label("SVAR", GREEN if "Error" not in answer else RED)
        print(colorize(answer, DIM))

    print_label("INTERAKTIV", CYAN)
    print(colorize("Skriv et spørgsmål og tryk Enter. Tom linje afslutter.", DIM))
    while True:
        question = input(colorize("\n> ", BOLD, CYAN)).strip()
        if not question:
            break
        answer = run_question(question, prompt, chain, llm_with_tools, tools)
        print_label("SVAR", GREEN if "Error" not in answer else RED)
        print(colorize(answer, DIM))


if __name__ == "__main__":
    main()
