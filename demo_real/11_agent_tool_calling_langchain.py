from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from demo_real.common_real import get_models, load_env
except ModuleNotFoundError:
    from common_real import get_models, load_env


@tool
def fx_rate(base: str, quote: str) -> str:
    """Return a demo FX rate for simple currency conversion."""
    rates = {
        ("USD", "DKK"): 6.92,
        ("EUR", "DKK"): 7.46,
        ("DKK", "EUR"): 0.134,
    }
    rate = rates.get((base.upper(), quote.upper()))
    if rate is None:
        return f"No rate available for {base}->{quote}"
    return f"1 {base.upper()} = {rate} {quote.upper()}"


@tool
def incident_status(ticket_id: str) -> str:
    """Return a mock incident status by ticket id."""
    statuses = {
        "INC-1001": "Investigating elevated latency in retrieval service",
        "INC-1002": "Resolved - webhook timeout issue",
        "INC-1003": "Monitoring - vector DB CPU spike",
    }
    return statuses.get(ticket_id.upper(), "Unknown ticket id")


@tool
def current_utc_time() -> str:
    """Return current UTC time for timeline explanations."""
    return datetime.now(timezone.utc).isoformat()


def build_agent() -> Any:
    chat_model, _ = get_models()
    llm = ChatOpenAI(model=chat_model, temperature=0)
    tools = [fx_rate, incident_status, current_utc_time]

    return create_agent(
        llm,
        tools=tools,
        system_prompt=(
            "Du er en senior AI-udviklerassistent. Brug tools når opgaven kræver faktiske opslag eller beregninger. "
            "Svar kort, præcist og på dansk."
        ),
    )


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


def main() -> None:
    load_env()
    agent = build_agent()

    question = (
        "Vi laver en demo til kurset. Fortæl status på ticket INC-1002, "
        "hvad 100 USD er i DKK, og giv til sidst en kort note med nuværende UTC-tid. "
        "Svar i 3 bullets."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    final_text = _message_text(messages[-1]) if messages else str(result)
    print("\n=== Agent final response ===")
    print(final_text)


if __name__ == "__main__":
    main()
