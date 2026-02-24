from __future__ import annotations

from datetime import datetime, timezone

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from demo_real.common_real import get_models, load_env


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


def build_agent() -> AgentExecutor:
    chat_model, _ = get_models()
    llm = ChatOpenAI(model=chat_model, temperature=0)
    tools = [fx_rate, incident_status, current_utc_time]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Du er en senior AI-udviklerassistent. Brug tools når opgaven kræver faktiske opslag eller beregninger. "
                "Svar kort, præcist og på dansk.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main() -> None:
    load_env()
    executor = build_agent()

    question = (
        "Vi laver en demo til kurset. Fortæl status på ticket INC-1002, "
        "hvad 100 USD er i DKK, og giv til sidst en kort note med nuværende UTC-tid. "
        "Svar i 3 bullets."
    )

    result = executor.invoke({"input": question})
    print("\n=== Agent final response ===")
    print(result["output"])


if __name__ == "__main__":
    main()

