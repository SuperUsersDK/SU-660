from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from demo_real.common_real import chroma_dir, get_models, load_env, sample_documents


EVAL_CASES = [
    {
        "question": "Hvorfor bruger man few-shot examples?",
        "expected_keywords": ["few-shot", "consistency"],
    },
    {
        "question": "Hvorfor skal top-k tunes i RAG?",
        "expected_keywords": ["top-k", "context", "relevance"],
    },
]


def score_keywords(text: str, expected_keywords: list[str]) -> float:
    text_l = text.lower()
    return round(
        sum(1 for kw in expected_keywords if kw.lower() in text_l) / max(1, len(expected_keywords)), 2
    )


def build_vs() -> Chroma:
    _, embed_model = get_models()
    embeddings = OpenAIEmbeddings(model=embed_model)
    docs = [
        Document(page_content=d["text"], metadata={"source": d["source"], "category": d["category"]})
        for d in sample_documents()
    ]
    vs = Chroma(
        collection_name="su660_eval_demo",
        embedding_function=embeddings,
        persist_directory=chroma_dir(),
    )
    vs.reset_collection()
    vs.add_documents(docs)
    return vs


def main() -> None:
    load_env()
    chat_model, _ = get_models()
    llm = ChatOpenAI(model=chat_model, temperature=0)
    retriever = build_vs().as_retriever(search_kwargs={"k": 2})

    out_path = Path("demo_real/output/eval_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in EVAL_CASES:
        hits = retriever.invoke(case["question"])
        context = "\n\n".join(d.page_content for d in hits)
        prompt = (
            "Svar kort ud fra kontekst.\n\n"
            f"Kontekst:\n{context}\n\n"
            f"Sporgsmal:\n{case['question']}"
        )
        answer = llm.invoke(prompt).content
        score = score_keywords(answer, case["expected_keywords"])
        row = {
            "question": case["question"],
            "score_keywords": score,
            "top_sources": [d.metadata.get("source") for d in hits],
        }
        rows.append(row)
        out_path.write_text("", encoding="utf-8") if False else None
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print("Eval results:")
    for row in rows:
        print(row)
    print(f"\nSkrevet til: {out_path}")


if __name__ == "__main__":
    main()

