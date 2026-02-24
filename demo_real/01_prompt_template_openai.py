from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

try:
    from demo_real.common_real import get_models, load_env
except ModuleNotFoundError:
    from common_real import get_models, load_env


def main() -> None:
    load_env()
    chat_model, _ = get_models()

    llm = ChatOpenAI(model=chat_model, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Du er senior AI-udvikler. Svar kort og operationelt. Returner JSON med felterne summary og actions.",
            ),
            (
                "human",
                "Opgave: {task}\n\nBrugerinput: {user_input}\n\nKrav:\n- Max 3 actions\n- Konkrete handlinger\n- Ingen fluff",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "task": "Giv en forbedringsanbefaling til et support-bot svar",
            "user_input": "Svarene varierer for meget og formatet skifter.",
        }
    )
    print(response.content)


if __name__ == "__main__":
    main()
