from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

try:
    from demo_real.common_real import get_models, load_env
except ModuleNotFoundError:
    from common_real import get_models, load_env


class Answer(BaseModel):
    explanation: str = Field(description="Kort forklaring af begrebet.")
    confidence: float = Field(description="Model confidence fra 0.0 til 1.0.")


def main() -> None:
    load_env()
    chat_model, _ = get_models()

    model = ChatOpenAI(model=chat_model, temperature=0)
    parser = PydanticOutputParser(pydantic_object=Answer)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Du er en teknisk underviser. Svar kort, korrekt og kun i det format der eftersporges.",
            ),
            (
                "human",
                "Forklar konceptet '{concept}' kort for en softwarestuderende.\n"
                "Sæt confidence mellem 0 og 1.\n\n"
                "{format_instructions}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    result = chain.invoke({"concept": "CAP theorem"})

    print("Explanation:")
    print(result.explanation)
    print("\nConfidence:")
    print(result.confidence)


if __name__ == "__main__":
    main()
