from __future__ import annotations

from functools import lru_cache
import sys
from typing import Any
import warnings

if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from settings import SETTINGS
from tools import calculator, doc_stats, search_in_context


def build_vs() -> QdrantVectorStore:
    client = QdrantClient(
        url=SETTINGS.qdrant_url,
        api_key=SETTINGS.qdrant_api_key or None,
    )
    embeddings = OpenAIEmbeddings(model=SETTINGS.embedding_model)
    return QdrantVectorStore(
        client=client,
        collection_name=SETTINGS.collection_name,
        embedding=embeddings,
    )


def _doc_citation(doc: Document) -> str:
    src = doc.metadata.get("source_file", "unknown")
    page = doc.metadata.get("page")
    if page is None:
        return f"[source={src}]"
    return f"[source={src} page={page}]"


def build_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        parts.append(f"{_doc_citation(doc)}\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieval_hits_with_scores(question: str, k: int = 8) -> list[tuple[Document, float]]:
    vs = build_vs()
    return vs.similarity_search_with_score(question, k=k)


def rag_answer(question: str, k: int = 8) -> str:
    vs = build_vs()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant for technical documentation, markdown notes, code files, and PDFs. "
                "Answer ONLY using the retrieved context. "
                "Cite sources as [source=...] and include page when available. "
                "If the context is insufficient, say so clearly.",
            ),
            ("human", "Question: {q}\n\nContext:\n{ctx}"),
        ]
    )

    model = ChatOpenAI(model=SETTINGS.chat_model, temperature=0)
    response = (prompt | model).invoke({"q": question, "ctx": context})
    return response.content


def build_retrieval_tool(k: int = 8):
    vs = build_vs()
    retriever = vs.as_retriever(search_kwargs={"k": k})

    @tool
    def retrieve(query: str) -> str:
        """Retrieve relevant chunks from the vector database across markdown, code, and PDF files."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant chunks found."

        parts: list[str] = []
        sources: list[str] = []
        for doc in docs:
            src = doc.metadata.get("source_file", "unknown")
            sources.append(str(src))
            parts.append(f"{_doc_citation(doc)}\n{doc.page_content}")

        unique_sources = sorted(set(sources))
        header = f"SOURCES={unique_sources}"
        return header + "\n\n" + "\n\n".join(parts)

    return retrieve


@lru_cache(maxsize=1)
def build_agent() -> Any:
    llm = ChatOpenAI(model=SETTINGS.chat_model, temperature=0)
    retrieval_tool = build_retrieval_tool()
    tools = [retrieval_tool, calculator, doc_stats, search_in_context]

    return create_agent(
        llm,
        tools=tools,
        system_prompt=(
            "You are a RAG assistant with tools for technical documents, markdown files, code files, and PDFs. "
            "Always call retrieve() first for factual questions about the ingested material. "
            "Use tools when needed, for example calculator, doc_stats, and search_in_context. "
            "When answering, cite sources like [source=...] and include page when available. "
            "If retrieved context is insufficient, say what is missing instead of guessing."
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


def agent_answer(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    if not isinstance(result, dict):
        return str(result)
    messages = result.get("messages", [])
    return _message_text(messages[-1]) if messages else str(result)


def format_retrieval_score_lines(question: str, k: int = 8) -> list[str]:
    hits = retrieval_hits_with_scores(question, k=k)
    lines: list[str] = []
    for i, (doc, score) in enumerate(hits, start=1):
        src = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        page_part = f" page={page}" if page is not None else ""
        lines.append(f"{i}. score={score:.4f} source={src}{page_part}")
    return lines


def retrieval_hits_summary(question: str, k: int = 8) -> list[dict[str, Any]]:
    hits = retrieval_hits_with_scores(question, k=k)
    out: list[dict[str, Any]] = []
    for i, (doc, score) in enumerate(hits, start=1):
        out.append(
            {
                "rank": i,
                "score": score,
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )
    return out
