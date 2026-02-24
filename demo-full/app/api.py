from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

try:
    from app.rag_chain import agent_answer
except ModuleNotFoundError:
    from rag_chain import agent_answer


app = FastAPI(title="PDF RAG Demo API")


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query) -> dict[str, str]:
    return {"answer": agent_answer(query.question)}

