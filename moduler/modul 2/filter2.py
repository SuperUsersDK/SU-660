import re


def is_useful_text(text: str | None, min_length: int = 80) -> bool:
    if text is None:
        return False

    cleaned = text.strip()
    if not cleaned:
        return False

    if min_length < 0:
        raise ValueError("min_length skal vaere 0 eller stoerre")

    if len(cleaned) < min_length:
        return False

    # For lidt alfabetisk indhold = sandsynligvis støj
    letters = sum(c.isalpha() for c in cleaned)
    ratio = letters / max(len(cleaned), 1)
    if ratio < 0.5:
        return False

    # Frasortér meget typisk navigations-/støjtekst
    noise_patterns = [
        r"cookie",
        r"accept all",
        r"privacy policy",
        r"home\s*\|\s*about\s*\|\s*contact",
    ]

    lowered = cleaned.lower()
    return not any(re.search(pattern, lowered) for pattern in noise_patterns)


def main() -> None:
    examples = [
        "OK",
        "Home | About | Contact",
        None,
        "Embeddings repraesenterer tekst som numeriske vektorer, som bruges til similarity search i et RAG-system.",
    ]

    for text in examples:
        print(is_useful_text(text), "->", repr(text))


if __name__ == "__main__":
    main()
