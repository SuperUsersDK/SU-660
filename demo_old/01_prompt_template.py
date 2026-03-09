from common import mock_llm, render_prompt


def main() -> None:
    user_input = "Skriv en kort anbefaling til forbedring af et support-bot svar."

    simple_prompt = f"Svar pa dette: {user_input}"
    robust_prompt = render_prompt(
        system_role=(
            "Du er senior AI-udvikler. Svar kort og operationelt. "
            "Returner JSON med felterne summary og actions."
        ),
        task=(
            "Analyser brugerens behov og giv en forbedringsanbefaling. "
            "Brug konkrete handlinger og undga generiske formuleringer."
        ),
        user_input=user_input,
        examples=[
            (
                "Brugeren far ustabile svar",
                '{"summary":"Stabiliser prompten","actions":["Fast outputformat","Few-shot eksempler"]}',
            )
        ],
    )

    print("=== Simpel prompt ===")
    print(simple_prompt)
    print("\n=== Simpelt svar ===")
    print(mock_llm(simple_prompt))

    print("\n=== Robust prompt-template ===")
    print(robust_prompt)
    print("\n=== Robust svar ===")
    print(mock_llm(robust_prompt))


if __name__ == "__main__":
    main()

