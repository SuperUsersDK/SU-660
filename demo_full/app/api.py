from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from app.rag_chain import agent_answer, retrieval_hits_summary
except ModuleNotFoundError:
    from rag_chain import agent_answer, retrieval_hits_summary


app = FastAPI(title="PDF RAG Demo API")


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query) -> dict[str, Any]:
    retrieval_hits: list[dict[str, Any]]
    retrieval_error: str | None = None
    try:
        retrieval_hits = retrieval_hits_summary(query.question)
    except Exception as exc:
        retrieval_hits = []
        retrieval_error = str(exc)

    response: dict[str, Any] = {
        "answer": agent_answer(query.question),
        "retrieval_hits": retrieval_hits,
    }
    if retrieval_error:
        response["retrieval_error"] = retrieval_error
    return response

