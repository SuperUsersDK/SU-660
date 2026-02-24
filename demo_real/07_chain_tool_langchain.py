from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from demo_real.common_real import get_models, load_env


@tool
def fx_rate(base: str, quote: str) -> str:
    """Returnerer en simpel demo-valutakurs."""
    rates = {("USD", "DKK"): 6.92, ("EUR", "DKK"): 7.46}
    rate = rates.get((base.upper(), quote.upper()), 1.0)
    return f"1 {base.upper()} = {rate} {quote.upper()}"


def main() -> None:
    load_env()
    chat_model, _ = get_models()
    llm = ChatOpenAI(model=chat_model, temperature=0)

    tool_result = fx_rate.invoke({"base": "USD", "quote": "DKK"})
    prompt = ChatPromptTemplate.from_template(
        "Bruger sporg: {question}\nTool-resultat: {tool_result}\nSvar kort pa dansk i 2 punkter."
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "question": "Hvad er 100 USD i DKK, og hvordan skal vi praesentere usikkerhed i en demo?",
            "tool_result": tool_result,
        }
    )

    print("Tool output:")
    print(tool_result)
    print("\nModel response:")
    print(response.content)


if __name__ == "__main__":
    main()

